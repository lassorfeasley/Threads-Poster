"""Route tests for the clip engine in the dashboard (app/web/main.py): the
202 + polling contract of "Suggest", the status endpoint, card previews and
their suppression when a card doesn't offer what was rendered, the footage-reuse
rule (a suggestion may share footage with other clips, never their opening or
the whole clip), the revise box's routes and its in-progress state, the archive
hand-off, and the startup/shutdown wiring.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
      .venv/bin/python -m pytest -q tests/test_clip_engine_routes.py

Importing app.web.main initialises whatever database DATABASE_URL names and
starts background threads, so every test here is skipped unless that is a
throwaway in-memory SQLite. Routes then read a StaticPool engine patched in as
``main.session_scope`` (TestClient runs them on another thread); the job
runner itself is mocked.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import spend
from app.models import (STATUS_APPROVED, STATUS_ARCHIVED, STATUS_FAILED, Base, Candidate,
                        Channel, ClipProposal, Cut, utcnow)


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


def make_engine():
    """In-memory SQLite shared across threads, with real transactions."""
    db = create_engine("sqlite://", future=True, poolclass=StaticPool,
                       connect_args={"check_same_thread": False})

    @event.listens_for(db, "connect")
    def _no_implicit_transactions(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(db, "begin")
    def _begin(connection):
        connection.exec_driver_sql("BEGIN")

    Base.metadata.create_all(db)
    return db


def job_status(state: str = "queued", **fields) -> dict:
    status = {"candidate_id": 1, "job_id": "job123", "boot_id": "boot0001", "reason": "reroll",
              "setup": "gemini", "window": None, "topic": "", "state": state, "stage": "",
              "outcome": "", "error": "", "fallback": "", "attempts": 0, "new": [],
              "cost_usd": 0.0, "notes": [], "queued_at": None, "started_at": None,
              "finished_at": None, "result_path": "", "queue_position": 0, "message": ""}
    status.update(fields)
    return status


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class RouteTestBase(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.clip_engine import jobs, sidecars
        from app.web import main

        self.main, self.jobs, self.sidecars = main, jobs, sidecars
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.patch(mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.dir / "engine")}))
        self.patch(mock.patch.object(spend, "_LEDGER", self.dir / "llm_spend.json"))

        self.db = make_engine()
        self.addCleanup(self.db.dispose)
        self.Session = sessionmaker(self.db, future=True, expire_on_commit=False)

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            session = self.Session()
            try:
                yield session
                if not read_only:
                    session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        self.patch(mock.patch.object(main, "session_scope", scope))
        self.patch(mock.patch.object(main, "SessionLocal", self.Session))
        self.setup_answer = ("gemini", "")
        self.patch(mock.patch("app.clip_engine.effective_setup",
                              side_effect=lambda settings, requested=None: self.setup_answer))
        self.enqueue = self.patch(mock.patch.object(jobs, "enqueue"))
        self.status = self.patch(mock.patch.object(jobs, "status", return_value=job_status("idle")))
        self.sidecar_meta: dict[int, dict] = {}
        self.preview_files: dict[int, Path] = {}
        self.patch(mock.patch.object(sidecars, "load",
                                     side_effect=lambda pid: self.sidecar_meta.get(pid)))
        self.patch(mock.patch.object(sidecars, "preview_path",
                                     side_effect=lambda pid: self.preview_files.get(pid)))
        self.client = TestClient(main.app)

    def patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def add_candidate(self, video_id: str = "vid1", *, transcript: bool = True,
                      status: str = STATUS_ARCHIVED, **fields) -> int:
        with self.Session() as s:
            channel = s.execute(select(Channel)).scalars().first()
            if channel is None:
                channel = Channel(call_sign="TEST", url="https://www.youtube.com/@test")
                s.add(channel)
                s.flush()
            tpath = ""
            if transcript:
                tpath = str(self.dir / f"{video_id}.json")
                Path(tpath).write_text(json.dumps(
                    [{"start": float(t), "end": float(t + 5), "text": f"line {t}"}
                     for t in range(0, 300, 5)]))
            c = Candidate(video_id=video_id, channel_pk=channel.id, title=f"Title {video_id}",
                          url=f"https://youtu.be/{video_id}", status=status,
                          archived_at=utcnow(), duration_seconds=300, transcript_path=tpath,
                          **fields)
            s.add(c)
            s.commit()
            return c.id

    def add_proposal(self, cid: int, segments: list[dict], **fields) -> int:
        with self.Session() as s:
            row = ClipProposal(candidate_pk=cid, proposed_segments=json.dumps(segments),
                               story=fields.pop("story", "Bill Nye takes his glasses off"),
                               why="A famous face, a charged line", **fields)
            s.add(row)
            s.commit()
            return row.id

    def add_cut(self, cid: int, segments: list[dict], title: str = "") -> int:
        with self.Session() as s:
            cut = Cut(candidate_pk=cid, trim_segments=json.dumps(segments), clip_title=title)
            s.add(cut)
            s.commit()
            return cut.id

    def engine_sidecar(self, pid: int, segments: list[dict], *, critique=None,
                       with_file: bool = True, **meta) -> None:
        self.sidecar_meta[pid] = {
            "proposal_id": pid, "setup": "gemini", "archetype": "voice", "lens": "person",
            "hook": "Glasses off", "score": 7.4, "segments": segments,
            "topic_doubt": "", "critique": critique if critique is not None else {
                "stops_scroll": True, "stops_scroll_muted": True,
                "stop_reason": "Nye pulls his glasses off in the first second",
                "about": "A scientist losing patience", "bored_at": 75.5,
                "fix": "End on the stare", "error": ""},
            **meta}
        if with_file:
            path = self.dir / f"{pid}.mp4"
            path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\0" * 2048)
            self.preview_files[pid] = path


# ---- POST /video/{id}/suggest-clips --------------------------------------------------------

class SuggestRouteTest(RouteTestBase):
    def test_engine_setup_queues_and_answers_202(self):
        cid = self.add_candidate()
        self.enqueue.return_value = {"started": True, "conflict": False, **job_status("queued")}
        with mock.patch.object(self.main, "_source_ready",
                               side_effect=AssertionError("the request must not restore the source")):
            r = self.client.post(f"/video/{cid}/suggest-clips")
        self.assertEqual(r.status_code, 202)
        body = r.json()
        self.assertEqual(body["started"], True)
        self.assertEqual(body["engine"], "gemini")
        self.assertEqual(body["job"]["state"], "queued")
        self.assertNotIn("started", body["job"])
        self.assertNotIn("conflict", body["job"])
        args, kwargs = self.enqueue.call_args
        self.assertEqual(args, (cid,))
        self.assertEqual(kwargs["window"], None)
        self.assertEqual(kwargs["topic"], "")
        self.assertEqual(kwargs["reason"], "reroll")
        self.assertIs(kwargs["on_done"], self.main._after_background_work)

    def test_a_section_pass_carries_its_window_and_chapter(self):
        chapters = [{"start": 0, "end": 100, "topic": "Intro", "summary": "the anchor"},
                    {"start": 100, "end": 300, "topic": "Fire", "summary": "the planet burns"}]
        cid = self.add_candidate(chapters=json.dumps(chapters))
        self.enqueue.return_value = {"started": True, "conflict": False, **job_status("queued")}
        r = self.client.post(f"/video/{cid}/suggest-clips?start=120&end=180")
        self.assertEqual(r.status_code, 202)
        kwargs = self.enqueue.call_args.kwargs
        self.assertEqual(kwargs["window"], {"start": 120.0, "end": 180.0})
        self.assertEqual(kwargs["reason"], "section")
        self.assertEqual(kwargs["topic"], "Fire — the planet burns")

    def test_joining_a_running_job_is_still_202(self):
        cid = self.add_candidate()
        self.enqueue.return_value = {"started": False, "conflict": False,
                                     **job_status("running", stage="Drafting clips")}
        r = self.client.post(f"/video/{cid}/suggest-clips")
        self.assertEqual(r.status_code, 202)
        self.assertFalse(r.json()["started"])
        self.assertEqual(r.json()["job"]["stage"], "Drafting clips")

    def test_another_scope_already_running_is_409(self):
        cid = self.add_candidate()
        self.enqueue.return_value = {"started": False, "conflict": True, **job_status("running")}
        r = self.client.post(f"/video/{cid}/suggest-clips?start=10&end=40")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["error"], "Already suggesting clips for this video")
        self.assertEqual(r.json()["job"]["state"], "running")

    def test_validation_happens_before_queueing(self):
        no_transcript = self.add_candidate("vid2", transcript=False)
        cases = [(9999, 404), (no_transcript, 409)]
        for cid, code in cases:
            with self.subTest(cid=cid):
                r = self.client.post(f"/video/{cid}/suggest-clips")
                self.assertEqual(r.status_code, code)
                self.assertIn("error", r.json())
        self.enqueue.assert_not_called()

    def test_a_queueing_failure_is_500(self):
        cid = self.add_candidate()
        self.enqueue.side_effect = RuntimeError("disk full")
        r = self.client.post(f"/video/{cid}/suggest-clips")
        self.assertEqual(r.status_code, 500)
        self.assertIn("disk full", r.json()["error"])

    def test_transcript_setup_keeps_the_synchronous_path(self):
        cid = self.add_candidate()
        self.setup_answer = ("transcript", "GEMINI_API_KEY is not set")
        pending = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        with mock.patch.object(self.main, "_source_ready", return_value=None), \
                mock.patch.object(self.main.clip_proposals, "propose", return_value=[pending]) as propose:
            r = self.client.post(f"/video/{cid}/suggest-clips")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["new"], 1)
        self.assertEqual([c["id"] for c in body["clips"]], [pending])
        propose.assert_called_once()
        self.enqueue.assert_not_called()

    def test_an_unreadable_engine_setup_keeps_the_synchronous_path(self):
        cid = self.add_candidate()
        with mock.patch("app.clip_engine.effective_setup", side_effect=RuntimeError("bad yaml")), \
                mock.patch.object(self.main, "_source_ready", return_value=None), \
                mock.patch.object(self.main.clip_proposals, "propose", return_value=[]):
            r = self.client.post(f"/video/{cid}/suggest-clips")
        self.assertEqual(r.status_code, 200)
        self.enqueue.assert_not_called()


# ---- GET /video/{id}/suggest-status --------------------------------------------------------

class SuggestStatusTest(RouteTestBase):
    def test_cards_ride_along_only_when_the_run_is_over(self):
        cid = self.add_candidate(multi_clip_potential=True)
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        cases = [
            ("queued", False), ("running", False),
            ("done", True), ("failed", True), ("interrupted", True), ("idle", True),
            ("skipped", True),
        ]
        for state, terminal in cases:
            with self.subTest(state=state):
                self.status.return_value = job_status(state, new=[pid] if state == "done" else [])
                r = self.client.get(f"/video/{cid}/suggest-status")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.headers["cache-control"], "no-store")
                body = r.json()
                self.assertEqual(body["job"]["state"], state)
                self.assertEqual("clips" in body, terminal)
                if terminal:
                    self.assertEqual([c["id"] for c in body["clips"]], [pid])
                    self.assertTrue(body["multi_clip"])

    def test_unknown_video_once_terminal_is_404(self):
        r = self.client.get("/video/4242/suggest-status")
        self.assertEqual(r.status_code, 404)

    def test_a_broken_status_read_ends_the_polling(self):
        cid = self.add_candidate()
        self.status.side_effect = RuntimeError("records unreadable")
        r = self.client.get(f"/video/{cid}/suggest-status")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job"]["state"], "failed")
        self.assertIn("clips", r.json())


# ---- previews --------------------------------------------------------------------------

class PreviewTest(RouteTestBase):
    def test_preview_route(self):
        cid = self.add_candidate()
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        r = self.client.get(f"/clip-proposal/{pid}/preview")
        self.assertEqual(r.status_code, 404)
        self.engine_sidecar(pid, [{"start": 10.0, "end": 30.0}])
        r = self.client.get(f"/clip-proposal/{pid}/preview")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "video/mp4")
        self.assertEqual(r.content, self.preview_files[pid].read_bytes())
        ranged = self.client.get(f"/clip-proposal/{pid}/preview", headers={"Range": "bytes=0-99"})
        self.assertEqual(ranged.status_code, 206)
        self.assertEqual(len(ranged.content), 100)
        self.preview_files[pid].unlink()
        self.assertEqual(self.client.get(f"/clip-proposal/{pid}/preview").status_code, 404)

    def test_a_broken_sidecar_lookup_is_404(self):
        with mock.patch.object(self.sidecars, "preview_path", side_effect=OSError("io")):
            self.assertEqual(self.client.get("/clip-proposal/5/preview").status_code, 404)

    def payload(self, pid: int, others: list[dict] | None = None) -> dict:
        with self.Session() as s:
            row = s.get(ClipProposal, pid)
            return self.main._proposal_payload(row, others or [])

    def test_payload_offers_the_preview_only_while_it_shows_what_the_card_offers(self):
        cid = self.add_candidate()
        segments = [{"start": 10.0, "end": 24.0}, {"start": 40.0, "end": 52.0}]
        cases = [
            # (description, sidecar segments, file on disk, other clips, preview expected)
            ("as rendered", segments, True, [], True),
            ("rounding only", [{"start": 10.01, "end": 24.0}, {"start": 40.0, "end": 51.98}],
             True, [], True),
            # Footage another clip also uses is no longer cut out: the card
            # still offers exactly the render.
            ("sharing footage with another clip", segments, True,
             [{"title": "Other", "segments": [{"start": 30.0, "end": 45.0}]}], True),
            ("render missing", segments, False, [], False),
            ("different cut", [{"start": 10.0, "end": 24.0}], True, [], False),
            ("different order", list(reversed(segments)), True, [], False),
        ]
        for label, rendered, with_file, others, expected in cases:
            with self.subTest(label):
                pid = self.add_proposal(cid, segments)
                self.engine_sidecar(pid, rendered, with_file=with_file)
                p = self.payload(pid, others)
                self.assertEqual(p["preview"], f"/clip-proposal/{pid}/preview" if expected else "")
                self.assertEqual(p["engine"]["archetype"], "voice")
                self.assertIn("confidence", p)

    def test_other_clips_flag_a_shared_opening_and_hide_a_duplicate(self):
        cid = self.add_candidate()
        segments = [{"start": 10.0, "end": 24.0}, {"start": 40.0, "end": 52.0}]
        cases = [
            # (label, other clips, offered?, same_opening)
            ("no other clips", [], True, ""),
            ("shares footage further in", [{"title": "Loom", "segments": [
                {"start": 40.0, "end": 60.0}]}], True, ""),
            ("opens on the same shot", [{"title": "Loom", "segments": [
                {"start": 11.0, "end": 15.0}, {"start": 100.0, "end": 130.0}]}], True, "Loom"),
            ("opens just after: windows touch, no overlap", [{"title": "Loom", "segments": [
                {"start": 13.0, "end": 20.0}, {"start": 100.0, "end": 130.0}]}], True, ""),
            ("the same clip again (IoU > 0.6)", [{"title": "Loom", "segments": [
                {"start": 12.0, "end": 24.0}, {"start": 40.0, "end": 52.0}]}], False, ""),
        ]
        for label, others, offered, same_opening in cases:
            with self.subTest(label):
                pid = self.add_proposal(cid, segments)
                self.engine_sidecar(pid, segments)
                p = self.payload(pid, others)
                if not offered:
                    self.assertIsNone(p)
                    continue
                self.assertEqual(p["segments"], segments)        # never cut down
                self.assertFalse(p["trimmed"])
                self.assertEqual(p["same_opening"], same_opening)
                # The flag doesn't touch the preview: the card is still the render.
                self.assertEqual(p["preview"], f"/clip-proposal/{pid}/preview")

    def test_a_cold_open_keeps_its_order_and_its_preview(self):
        cid = self.add_candidate()
        cold_open = [{"start": 50.0, "end": 54.0}, {"start": 10.0, "end": 24.0}]
        pid = self.add_proposal(cid, cold_open)
        self.engine_sidecar(pid, cold_open)
        p = self.payload(pid)
        self.assertEqual(p["segments"], cold_open)
        self.assertEqual(p["at"], 50.0)
        self.assertEqual(p["preview"], f"/clip-proposal/{pid}/preview")

    def test_engine_card_fields(self):
        cid = self.add_candidate()
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        self.engine_sidecar(pid, [{"start": 10.0, "end": 30.0}], topic_doubt="a film, not the news")
        engine = self.payload(pid)["engine"]
        self.assertEqual(engine, {
            "setup": "gemini", "archetype": "voice", "lens": "person", "hook": "Glasses off",
            "stop_reason": "Nye pulls his glasses off in the first second",
            "stops_scroll": True, "stops_scroll_muted": True,
            "about": "A scientist losing patience", "bored_at": 75.5, "fix": "End on the stare",
            "score": 7.4, "reviewed": True, "topic_doubt": "a film, not the news",
            # Not asked (an older sidecar): no warnings.
            "language_doubt": "", "premise_clear": None, "speech_understood": None,
            "monologue": None, "follows_advice": None, "advice_note": ""})

    def test_round_two_tests_and_advice_reach_the_card(self):
        cid = self.add_candidate()
        critique = {"stops_scroll": True, "stop_reason": "floods", "error": "",
                    "premise_clear": False, "speech_understood": False, "monologue": True,
                    "follows_advice": False, "advice_note": "opens on the anchor anyway"}
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        self.engine_sidecar(pid, [{"start": 10.0, "end": 30.0}], critique=critique,
                            language_doubt="mostly Hindi, no voiceover")
        engine = self.payload(pid)["engine"]
        self.assertEqual({k: engine[k] for k in ("language_doubt", "premise_clear",
                                                 "speech_understood", "monologue",
                                                 "follows_advice", "advice_note")},
                         {"language_doubt": "mostly Hindi, no voiceover", "premise_clear": False,
                          "speech_understood": False, "monologue": True, "follows_advice": False,
                          "advice_note": "opens on the anchor anyway"})
        # An outage answers nothing, whatever its zeroed fields say.
        self.engine_sidecar(pid, [{"start": 10.0, "end": 30.0}],
                            critique=dict(critique, error="timeout"))
        engine = self.payload(pid)["engine"]
        self.assertEqual([engine[k] for k in ("premise_clear", "speech_understood", "monologue",
                                              "follows_advice")], [None] * 4)

    def test_an_outage_reads_as_not_reviewed(self):
        cid = self.add_candidate()
        cases = [
            ({"error": "timeout", "stops_scroll": False, "score": 0}, False),
            ({}, False),
            ({"stops_scroll": False, "stop_reason": "an anchor at a desk", "error": ""}, True),
        ]
        for critique, reviewed in cases:
            with self.subTest(critique=critique):
                pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
                self.engine_sidecar(pid, [{"start": 10.0, "end": 30.0}], critique=critique)
                self.assertEqual(self.payload(pid)["engine"]["reviewed"], reviewed)

    def test_no_sidecar_or_a_broken_one_means_a_plain_card(self):
        cid = self.add_candidate()
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        p = self.payload(pid)
        self.assertEqual((p["engine"], p["preview"]), ({}, ""))
        with mock.patch.object(self.sidecars, "load", side_effect=ValueError("half-written")):
            p = self.payload(pid)
        self.assertEqual((p["engine"], p["preview"]), ({}, ""))
        self.sidecar_meta[pid] = {"critique": "not a dict", "bored_at": "soon", "score": True,
                                  "segments": "nope"}
        p = self.payload(pid)
        self.assertFalse(p["engine"]["reviewed"])
        self.assertIsNone(p["engine"]["score"])
        self.assertEqual(p["preview"], "")


# ---- the pages ---------------------------------------------------------------------------

class PageTest(RouteTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.cid = self.add_candidate(local_video_path=str(self.dir / "vid1.mp4"))
        self.pid = self.add_proposal(self.cid, [{"start": 10.0, "end": 30.0}], confidence=0.9)
        self.engine_sidecar(self.pid, [{"start": 10.0, "end": 30.0}])
        self.cut_id = self.add_cut(self.cid, [])

    def get(self, url: str) -> str:
        r = self.client.get(url)
        self.assertEqual(r.status_code, 200, r.text[:2000])
        return r.text

    def test_video_page_cards(self):
        html = self.get(f"/video/{self.cid}")
        self.assertIn(f'<video src="/clip-proposal/{self.pid}/preview"', html)
        self.assertIn("Stops the scroll", html)
        self.assertIn("Gets slow at 1:15", html)
        self.assertIn('class="suggest-archetype"', html)
        self.assertNotIn("suggest-conf", html)       # confidence is no longer shown
        self.assertIn("let engineJob = null", html)
        self.assertIn('id="suggest-status"', html)

    def test_video_page_card_without_a_preview_keeps_the_player_button(self):
        self.preview_files.clear()
        html = self.get(f"/video/{self.cid}")
        self.assertNotIn("<video src=\"/clip-proposal/", html)
        self.assertIn(f"previewSuggestion({self.pid})", html)

    def test_unreviewed_card(self):
        self.engine_sidecar(self.pid, [{"start": 10.0, "end": 30.0}],
                            critique={"error": "quota", "stops_scroll": False})
        html = self.get(f"/video/{self.cid}")
        self.assertIn("Not reviewed (critic unavailable)", html)
        self.assertNotIn("stop the scroll", html.lower().replace("stops the scroll: ", ""))

    def test_pages_opened_mid_run_carry_the_job(self):
        self.status.return_value = job_status("running", stage="Watching the drafts")
        for url in (f"/video/{self.cid}", f"/cut/{self.cut_id}"):
            with self.subTest(url=url):
                html = self.get(url)
                job = json.loads(re.search(r"let engineJob = (\{.*?\});", html).group(1))
                self.assertEqual(job["stage"], "Watching the drafts")
                self.assertEqual(job["boot_id"], "boot0001")

    def test_a_finished_run_is_not_carried(self):
        self.status.return_value = job_status("done", new=[self.pid])
        self.assertIn("let engineJob = null", self.get(f"/cut/{self.cut_id}"))

    def page_value(self, html: str, pattern: str):
        return json.loads(re.search(pattern, html).group(1))

    @unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
    def test_footage_another_clip_uses_is_offered_whole_on_both_pages(self):
        # Another clip already holds the proposal's last five seconds: reuse is fine.
        self.add_cut(self.cid, [{"start": 25.0, "end": 30.0}], title="The loom")
        html = self.get(f"/cut/{self.cut_id}")
        clip = self.page_value(html, r"let suggestedClips = (\[.*?\]);\n")[0]
        self.assertEqual((clip["trimmed"], clip["preview"], clip["same_opening"]),
                         (False, f"/clip-proposal/{self.pid}/preview", ""))
        self.assertEqual(clip["segments"], [{"start": 10.0, "end": 30.0}])
        card = run_proposal_cards([clip], self.page_value(html, r"const otherSegments = (\[.*?\]);\n"),
                                  self.page_value(html, r"let segments = (\[.*?\]);\n"))[0]
        self.assertEqual(card["shown"]["segments"], [{"start": 10, "end": 30}])
        self.assertIn("Stops the scroll", card["html"])
        self.assertTrue(card["video"])
        self.assertNotIn("Shortened", card["html"])
        self.assertNotIn("same shot", card["html"])
        video_page = self.get(f"/video/{self.cid}")
        self.assertIn("Stops the scroll", video_page)
        self.assertNotIn("Shortened", video_page)

    @unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
    def test_a_shared_opening_is_flagged_on_both_pages(self):
        self.add_cut(self.cid, [{"start": 11.0, "end": 16.0}, {"start": 200.0, "end": 230.0}],
                     title="The <loom>")
        html = self.get(f"/cut/{self.cut_id}")
        clip = self.page_value(html, r"let suggestedClips = (\[.*?\]);\n")[0]
        self.assertEqual(clip["same_opening"], "The <loom>")
        card = run_proposal_cards([clip], self.page_value(html, r"const otherSegments = (\[.*?\]);\n"),
                                  self.page_value(html, r"let segments = (\[.*?\]);\n"))[0]
        self.assertEqual(card["shown"]["conflict"], "same_opening")
        self.assertIn("Opens on the same shot as “The &lt;loom&gt;”", card["html"])
        self.assertIn("Stops the scroll", card["html"])     # still the clip as rendered
        video_page = self.get(f"/video/{self.cid}")
        self.assertIn("Opens on the same shot as “The &lt;loom&gt;”", video_page)

    def test_a_duplicate_of_an_existing_clip_is_hidden_on_both_pages(self):
        self.add_cut(self.cid, [{"start": 9.0, "end": 29.0}], title="Already made")
        html = self.get(f"/cut/{self.cut_id}")
        self.assertEqual(self.page_value(html, r"let suggestedClips = (\[.*?\]);\n"), [])
        self.assertNotIn(f'data-id="{self.pid}"', self.get(f"/video/{self.cid}"))
        with self.Session() as s:
            from app import clip_proposals

            self.assertEqual(clip_proposals.pending_summary(s, [self.cid]), {})

    def test_cut_page_renders(self):
        html = self.get(f"/cut/{self.cut_id}")
        self.assertIn("let suggestedClips = ", html)
        payload = json.loads(re.search(r"let suggestedClips = (\[.*?\]);\n", html).group(1))
        self.assertEqual(payload[0]["preview"], f"/clip-proposal/{self.pid}/preview")
        self.assertEqual(payload[0]["engine"]["archetype"], "voice")

    @unittest.skipUnless(shutil.which("node"), "node is needed to syntax-check page scripts")
    def test_page_scripts_parse(self):
        self.status.return_value = job_status("running", stage="Ranking")
        for url in (f"/video/{self.cid}", f"/cut/{self.cut_id}"):
            html = self.get(url)
            scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
            # The page's own script, with the polling code, is among those checked.
            self.assertTrue(any("function handleEngineStatus" in s for s in scripts), url)
            for i, script in enumerate(scripts):
                with self.subTest(url=url, script=i):
                    path = self.dir / f"page-{i}.js"
                    path.write_text(script)
                    result = subprocess.run(["node", "--check", str(path)],
                                            capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])


# ---- the footage-reuse rule in the trim editor's routes ---------------------------------

class ReviseRoutesTest(RouteTestBase):
    """The revise box and "Better opening shot": the model sees the video's
    other clips (title, words heard, segments, opening) and may reuse their
    footage; nothing tells it "already used"; only another clip's opening is
    off limits to the opening pull-back."""

    def setUp(self) -> None:
        super().setUp()
        self.cid = self.add_candidate(local_video_path=str(self.dir / "vid1.mp4"))
        self.loom = self.add_cut(self.cid, [{"start": 100.0, "end": 115.0},
                                            {"start": 40.0, "end": 50.0}], title="The loom")
        self.cut_id = self.add_cut(self.cid, [{"start": 60.0, "end": 80.0}], title="The dyer")

    def revise(self, answer: dict, current: list[dict], instruction: str = "use the loom setup"):
        calls = []

        def fake(model, title, instruction, current, transcript, **kw):
            calls.append(dict(kw, instruction=instruction, current=current))
            return answer

        with mock.patch.object(self.main, "revise_clip", side_effect=fake):
            r = self.client.post(f"/cut/{self.cut_id}/revise",
                                 data={"instruction": instruction,
                                       "segments_json": json.dumps(current)})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json(), calls

    def test_the_model_sees_the_other_clips_not_used_ranges(self):
        current = [{"start": 60.0, "end": 80.0}]
        out, (call,) = self.revise({"segments": [{"start": 60.0, "end": 80.0},
                                                 {"start": 100.0, "end": 108.0}],
                                    "note": "Added the loom setup.", "changed": True}, current)
        self.assertNotIn("used_ranges", call)
        (other,) = call["other_clips"]                     # this cut is not "another clip"
        self.assertEqual(other["title"], "The loom")
        self.assertEqual(other["segments"], [{"start": 100.0, "end": 115.0},
                                             {"start": 40.0, "end": 50.0}])
        self.assertEqual(other["opening"], [100.0, 103.0])
        self.assertIn("line", other["words_heard"])       # the transcript's words, in play order
        # Reusing the loom's footage further in is fine: no warning, nothing refused.
        self.assertEqual((out["note"], out["conflict"], out["changed"]),
                         ("Added the loom setup.", "", True))

    def test_a_revision_onto_another_clips_opening_is_called_out(self):
        cases = [
            ({"segments": [{"start": 101.0, "end": 125.0}], "note": "Opened on the loom.",
              "changed": True}, "Heads up: it opens on the same shot as “The loom”."),
            ({"segments": [{"start": 100.0, "end": 115.0}, {"start": 40.0, "end": 50.0}],
              "note": "Rebuilt.", "changed": True},
             "Heads up: this is now nearly the same clip as “The loom”."),
        ]
        for answer, warning in cases:
            with self.subTest(warning):
                out, _ = self.revise(answer, [{"start": 60.0, "end": 80.0}])
                self.assertEqual(out["conflict"], warning)
                self.assertEqual(out["note"], f"{answer['note']} {warning}")
                self.assertEqual(out["segments"], answer["segments"])   # still the operator's call

    def test_the_prompt_says_reuse_is_fine(self):
        from app import llm

        seen = {}

        def fake_chat(model, system, user, max_tokens=0):
            seen.update(system=system, payload=json.loads(user))
            return {"segments": [{"start": 60.0, "end": 80.0}], "note": "ok"}

        with mock.patch.object(llm, "_json_chat", side_effect=fake_chat):
            r = self.client.post(f"/cut/{self.cut_id}/revise",
                                 data={"instruction": "x",
                                       "segments_json": json.dumps([{"start": 60, "end": 80}])})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("Reusing their footage is fine and expected", seen["system"])
        self.assertIn("DIFFERENT OPENING", seen["system"])
        self.assertIn("DIFFERENT STORY", seen["system"])
        self.assertNotIn("already used", seen["system"])
        self.assertNotIn("used_ranges", seen["payload"])
        self.assertEqual([c["title"] for c in seen["payload"]["other_clips"]], ["The loom"])
        self.assertEqual(seen["payload"]["other_clips"][0]["opening"], [100.0, 103.0])

    def test_better_opening_may_reach_back_over_other_clips_but_not_their_opening(self):
        from app import clip_proposals

        Path(self.dir / "vid1.mp4").write_bytes(b"\0" * 16)
        seen = {}

        def fake_refine(candidate, segments, settings, blocked, story=""):
            seen.update(blocked=blocked, story=story)
            return segments, "Kept your opening."

        current = [{"start": 60.0, "end": 80.0}, {"start": 20.0, "end": 30.0}]
        with mock.patch.object(clip_proposals, "refine_opening_detail", side_effect=fake_refine):
            r = self.client.post(f"/cut/{self.cut_id}/refine-opening",
                                 data={"segments_json": json.dumps(current)})
        self.assertEqual(r.status_code, 200, r.text)
        # The loom's OPENING window (100-103) and this clip's own later
        # segment; never the loom's other footage (40-50, 103-115).
        self.assertEqual(seen["blocked"], [{"start": 20.0, "end": 30.0},
                                           {"start": 100.0, "end": 103.0}])
        self.assertEqual(seen["story"], "The dyer")
        self.assertEqual(r.json()["conflict"], "")

    def test_better_opening_says_when_the_opening_clashes(self):
        from app import clip_proposals

        Path(self.dir / "vid1.mp4").write_bytes(b"\0" * 16)
        current = [{"start": 101.0, "end": 125.0}]
        with mock.patch.object(clip_proposals, "refine_opening_detail",
                               return_value=(current, "Kept your opening.")):
            r = self.client.post(f"/cut/{self.cut_id}/refine-opening",
                                 data={"segments_json": json.dumps(current)})
        self.assertEqual(r.json()["note"],
                         "Kept your opening. Heads up: it opens on the same shot as “The loom”.")


class MakeCutTest(RouteTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.cid = self.add_candidate()
        self.add_cut(self.cid, [{"start": 100.0, "end": 115.0}], title="The loom")

    def make(self, segments: list[dict]):
        pid = self.add_proposal(self.cid, segments)
        r = self.client.post(f"/clip-proposal/{pid}/make-cut", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        with self.Session() as s:
            row = s.get(ClipProposal, pid)
            cuts = s.execute(select(Cut).where(Cut.candidate_pk == self.cid)
                             .order_by(Cut.id)).scalars().all()
            return r.headers["location"], row.verdict, [json.loads(c.trim_segments) for c in cuts]

    def test_shared_footage_is_loaded_whole(self):
        where, verdict, cuts = self.make([{"start": 60.0, "end": 80.0},
                                          {"start": 105.0, "end": 112.0}])
        self.assertIn("?step=trim", where)
        self.assertEqual(verdict, "accepted")
        self.assertEqual(cuts[-1], [{"start": 60.0, "end": 80.0}, {"start": 105.0, "end": 112.0}])

    def test_a_shared_opening_is_made_with_a_reminder(self):
        where, verdict, cuts = self.make([{"start": 101.0, "end": 130.0}])
        self.assertEqual(verdict, "accepted")
        self.assertEqual(len(cuts), 2)
        self.assertIn("same+shot", where.replace("%20", "+"))

    def test_the_same_clip_again_is_refused(self):
        where, verdict, cuts = self.make([{"start": 100.0, "end": 116.0}])
        self.assertEqual(verdict, "pending")
        self.assertEqual(len(cuts), 1)
        self.assertIn(f"/video/{self.cid}", where)


class StrikingFramesTest(RouteTestBase):
    def test_only_other_clips_openings_are_off_limits(self):
        from app import vision

        video = self.dir / "vid1.mp4"
        video.write_bytes(b"\0" * 16)
        cid = self.add_candidate(local_video_path=str(video))
        self.add_cut(cid, [{"start": 100.0, "end": 115.0}, {"start": 40.0, "end": 50.0}])
        seen = {}

        def fake_find(c, settings, *, start, end, used, topic=""):
            seen["used"] = used
            return []

        with mock.patch.object(vision, "find_striking_frames", side_effect=fake_find), \
                mock.patch.object(self.main, "clip_duration", return_value=300.0):
            r = self.client.post(f"/video/{cid}/striking-frames")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(seen["used"], [{"start": 100.0, "end": 103.0}])


# The revise box while a request runs: cut.html's own functions in node, with
# a fake DOM, clock and fetch. Nothing is posted anywhere.
REVISE_HARNESS = r"""
let now = 1000000;
Date.now = () => now;
const timers = new Map();
let nextTimer = 1;
global.setInterval = (fn, ms) => { const id = nextTimer++; timers.set(id, fn); return id; };
global.clearInterval = id => { timers.delete(id); };
function tick(seconds) { for (let i = 0; i < seconds; i++) { now += 1000; [...timers.values()].forEach(fn => fn()); } }

function classList() {
  const set = new Set();
  return { add: c => set.add(c), remove: c => set.delete(c), contains: c => set.has(c),
           toggle: (c, on) => { if (on === undefined) on = !set.has(c); if (on) set.add(c); else set.delete(c); return on; },
           toArray: () => [...set] };
}
function el(id) {
  return { id, hidden: false, disabled: false, value: '', textContent: '', attrs: {},
           classList: classList(), offsetWidth: 0, scrolled: 0,
           setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
           scrollIntoView() { this.scrolled++; } };
}
const ids = {};
['revise-box', 'revise-input', 'revise-send', 'revise-status', 'revise-status-text',
 'revise-elapsed', 'revise-note', 'chip-opening'].forEach(id => { ids[id] = el(id); });
ids['revise-status'].hidden = true;
ids['revise-note'].hidden = true;
const chips = [ids['chip-opening']];
const body = el('body');
const document = { getElementById: id => ids[id] || null, querySelectorAll: () => chips, body };

let segments = [{ start: 10, end: 30 }];
let selectedSeg = null, undoStack = [], renders = 0, pending = { start: null, end: null };
function renderSegments() { renders++; }
function renderPending() {}
function pushUndo() { undoStack.push(JSON.stringify(segments)); }
function txFmt(t) { return String(t); }
const player = { currentTime: 0 };

let resolveFetch;
const posted = [];
global.fetch = (url, opts) => { posted.push(url); return new Promise(res => { resolveFetch = res; }); };
function answer(status, body) { resolveFetch({ ok: status < 400, status, json: async () => body }); }

%(functions)s
%(guarded)s

function snapshot() {
  return {
    inputDisabled: ids['revise-input'].disabled, inputValue: ids['revise-input'].value,
    inputBusy: ids['revise-input'].attrs['aria-busy'],
    sendDisabled: ids['revise-send'].disabled, sendLoading: ids['revise-send'].classList.contains('is-loading'),
    chipDisabled: ids['chip-opening'].disabled, chipBusy: ids['chip-opening'].classList.contains('is-busy'),
    boxBusy: ids['revise-box'].classList.contains('is-busy'),
    bodyRevising: body.classList.contains('is-revising'),
    statusHidden: ids['revise-status'].hidden, statusText: ids['revise-status-text'].textContent,
    elapsed: ids['revise-elapsed'].textContent, nudged: ids['revise-status'].classList.contains('is-nudged'),
    noteHidden: ids['revise-note'].hidden, note: ids['revise-note'].textContent,
    noteError: ids['revise-note'].classList.contains('is-error'),
    timers: timers.size, segments: JSON.parse(JSON.stringify(segments)), scrolled: ids['revise-box'].scrolled,
  };
}

(async () => {
  const out = {};
  const mode = process.argv[2];
  ids['revise-input'].value = 'cut the anchor intro';
  ids['revise-note'].hidden = false;
  ids['revise-note'].textContent = 'An older answer';
  if (mode === 'opening') refineOpening(); else submitRevise({ preventDefault() {} });
  out.start = snapshot();
  tick(12);
  out.running = snapshot();
  // Every edit path is refused while it runs.
  removeSegment(0); addSegment(); undoEdit(); setStart(); useAsOpener(50); setSegEdge(0, 'start', 12);
  out.blocked = snapshot();
  // A second request can't start on top of the first.
  submitRevise({ preventDefault() {} }); refineOpening();
  out.posted = posted.length;
  if (mode === 'error') answer(500, { error: 'model timed out' });
  else answer(200, { changed: true, note: 'Cut the intro.', segments: [{ start: 14, end: 30 }] });
  await new Promise(r => setTimeout(r, 0)); await new Promise(r => setTimeout(r, 0));
  tick(3);
  out.done = snapshot();
  removeSegment(0);
  out.after = snapshot();
  process.stdout.write(JSON.stringify(out));
})();
"""


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class ReviseLoadingStateJsTest(unittest.TestCase):
    """While a revise or "Better opening shot" request runs, the row says so
    unmistakably and the timeline takes no edits; when it ends the note says
    what happened and the counter stops."""

    @classmethod
    def setUpClass(cls) -> None:
        template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                    / "cut.html").read_text().replace("{{ cut.id }}", "7")

        def piece(start: str, end: str) -> str:
            at = template.index(start)
            return template[at:template.index(end, at)]

        cls.template = template
        cls.functions = piece("let revising = false;", "function selectSegment(i)")
        cls.guarded = "\n".join([
            piece("function undoEdit()", "function seek(t)"),
            piece("function setStart()", "// ---- Scrollable transcript"),
            piece("function addSegment()", "// ---- Drag-to-reorder"),
            piece("function useAsOpener(t)", "function hideFrameStrip()"),
            piece("function setSegEdge(i, edge, t)", "  const s = segments[i];") + "}\n",
        ])

    def run_mode(self, mode: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "revise.js"
            path.write_text(REVISE_HARNESS % {"functions": self.functions, "guarded": self.guarded})
            result = subprocess.run(["node", str(path), mode], capture_output=True, text=True,
                                    timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return json.loads(result.stdout)

    def test_a_typed_revision(self):
        out = self.run_mode("revise")
        start, running, blocked = out["start"], out["running"], out["blocked"]
        # In progress: the instruction stays readable in a disabled field...
        self.assertEqual((start["inputDisabled"], start["inputValue"], start["inputBusy"]),
                         (True, "cut the anchor intro", "true"))
        # ...with a spinner line and a seconds count right under it...
        self.assertEqual((start["statusHidden"], start["statusText"], start["elapsed"]),
                         (False, "Revising your clip…", "0s"))
        self.assertEqual(running["elapsed"], "12s")
        # ...chips and send off, the timeline marked busy, the old note gone.
        self.assertTrue(start["sendDisabled"] and start["sendLoading"] and start["chipDisabled"])
        self.assertTrue(start["boxBusy"] and start["bodyRevising"])
        self.assertTrue(start["noteHidden"])
        self.assertEqual(start["scrolled"], 1)
        # No edits while it runs, and the status line nudges to say why.
        self.assertEqual(blocked["segments"], [{"start": 10, "end": 30}])
        self.assertTrue(blocked["nudged"])
        self.assertEqual(out["posted"], 1)
        # Done: the model's note, the counter stopped, everything unlocked.
        done = out["done"]
        self.assertEqual((done["noteHidden"], done["note"], done["noteError"]),
                         (False, "Cut the intro. — ⌘Z undoes it", False))
        self.assertEqual((done["statusHidden"], done["timers"], done["elapsed"]), (True, 0, "12s"))
        self.assertFalse(done["inputDisabled"] or done["sendDisabled"] or done["chipDisabled"]
                         or done["sendLoading"] or done["boxBusy"] or done["bodyRevising"])
        self.assertEqual(done["segments"], [{"start": 14, "end": 30}])
        self.assertEqual(done["inputValue"], "")          # a change landed: ready for the next
        self.assertEqual(out["after"]["segments"], [])    # edits work again

    def test_a_failed_revision_says_so_and_keeps_the_instruction(self):
        out = self.run_mode("error")
        done = out["done"]
        self.assertEqual((done["note"], done["noteError"], done["noteHidden"]),
                         ("Couldn’t revise the clip: model timed out", True, False))
        self.assertEqual(done["inputValue"], "cut the anchor intro")
        self.assertEqual((done["statusHidden"], done["timers"]), (True, 0))
        self.assertFalse(done["inputDisabled"] or done["bodyRevising"])
        self.assertEqual(done["segments"], [{"start": 10, "end": 30}])

    def test_better_opening_shot(self):
        out = self.run_mode("opening")
        start = out["start"]
        self.assertEqual(start["statusText"], "Looking at the footage for a better opening…")
        self.assertTrue(start["chipBusy"] and start["inputDisabled"] and start["bodyRevising"])
        self.assertEqual(start["inputValue"], "cut the anchor intro")
        self.assertEqual(out["running"]["elapsed"], "12s")
        self.assertEqual(out["blocked"]["segments"], [{"start": 10, "end": 30}])
        self.assertFalse(out["done"]["chipBusy"])
        self.assertEqual(out["done"]["note"], "Cut the intro. — ⌘Z undoes it")

    def test_the_markup_and_styles_exist(self):
        for needle in ('id="revise-status"', 'class="revise-spinner"', 'id="revise-elapsed"',
                       'role="status"'):
            self.assertIn(needle, self.template)
        css = (Path(__file__).resolve().parents[1] / "app" / "web" / "static"
               / "style.css").read_text()
        for needle in (".revise-status[hidden]", ".revise-spinner", "body.is-revising #segments",
                       ".revise-box.is-busy .revise-input:disabled"):
            self.assertIn(needle, css)
        # Every pointer path that can trim checks the lock.
        self.assertIn("const h = revising ? null : handleAt(e);", self.template)
        self.assertIn("const edge = revising ? null : edgeAt(ev.clientX);", self.template)
        self.assertIn("if (editsLocked()) return;   // no reorder or trim", self.template)


# cut.html's own card functions, run in node with just enough of a DOM: for
# each suggestion, what the browser offers (visibleProposal) and the card it
# draws (proposalCard), against the page's claims.
CARD_HARNESS = """
let otherSegments = [], segments = [], suggestedClips = [], chapters = [], view = null;
let hoveredProposal = null;
const proposalVideos = new Map();
const proposalFeedback = new Map();
const NOTE_TAGS = [['wrong_language', 'Wrong language'], ['other', 'Something <else>']];
function chapterAt() { return null; }
function element(tag) {
  return { tag, innerHTML: '', attrs: {}, classList: { add() {}, remove() {} },
           setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
           addEventListener() {}, appendChild(child) { return child; },
           querySelector: () => element('div'), querySelectorAll: () => [] };
}
const document = { createElement: element };
%s
const [clips, other, mine, open] = JSON.parse(process.argv[2]);
otherSegments = other; segments = mine; suggestedClips = clips;
(open || []).forEach(st => proposalFeedback.set(st.id, st));
process.stdout.write(JSON.stringify(clips.map(clip => {
  const shown = visibleProposal(clip);
  // As on the page (proposalsInView): a suggestion offering nothing gets no card.
  if (!shown.segments.length) return { shown, html: '', video: false };
  const card = proposalCard(clip, shown);
  return { shown, html: card.innerHTML, video: proposalVideos.has(clip.id) };
})));
"""


def run_proposal_cards(clips: list[dict], other_segments: list[dict],
                       segments: list[dict], feedback: list[dict] | None = None) -> list[dict]:
    template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                / "cut.html").read_text()

    def piece(start: str, end: str) -> str:
        return template[template.index(start):template.index(end)]

    script = "\n".join([
        piece("function txFmt", "function currentTxIndex"),
        piece("function escapeHtml", "function highlightQuery"),
        piece("const PROPOSAL_OPENING_SECONDS", "// The suggestions worth showing right now."),
        piece("// Its own <video>, reused across re-renders",
              "// Nothing waiting and nothing cut yet"),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cards.js"
        path.write_text(CARD_HARNESS % script)
        result = subprocess.run(["node", str(path),
                                 json.dumps([clips, other_segments, segments, feedback or []])],
                                capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-2000:])
    return json.loads(result.stdout)


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class ProposalCardJsTest(unittest.TestCase):
    """A card offers the suggestion as proposed (footage shared with other
    clips is fine), plays the engine's render in the card, warns when it opens
    on another clip's opening shot, and isn't drawn at all when it is the same
    clip as one already made."""

    ENGINE = {"setup": "gemini", "archetype": "voice", "reviewed": True, "stops_scroll": True,
              "stop_reason": "Nye pulls his glasses off in the first second", "bored_at": 75.5,
              "about": "A scientist losing patience", "fix": "End on the stare"}

    def clip(self, cid: int, segments: list[dict], *, trimmed: bool = False, preview: bool,
             **engine) -> dict:
        return {"id": cid, "story": "Glasses off", "why": "", "segments": segments,
                "seconds": sum(s["end"] - s["start"] for s in segments), "trimmed": trimmed,
                "preview": f"/clip-proposal/{cid}/preview" if preview else "",
                "thumb": "", "flag": "", "opening": "", "closing": "", "same_opening": "",
                "engine": dict(self.ENGINE, **engine)}

    def test_cards(self):
        whole = [{"start": 10, "end": 30}]
        loom = lambda *spans: [{"start": a, "end": b, "cut_id": 4, "title": "The loom"}
                               for a, b in spans]
        cases = [
            # (label, other clips' segments, this clip's segments,
            #  offered segments, warning, own player)
            ("nothing else on the video", [], [], whole, "", True),
            ("another clip uses part of its footage", loom((25, 30)), [], whole, "", True),
            ("this clip uses part of its footage", [], [{"start": 25, "end": 30}],
             whole, "", True),
            ("another clip opens on its opening shot", loom((11, 14), (100, 130)), [], whole,
             "Opens on the same shot as “The loom”", True),
            ("the clip open here opens on it", [], [{"start": 9, "end": 12},
                                                    {"start": 100, "end": 140}],
             whole, "", True),
            ("another clip is the same clip", loom((10, 28)), [], [], "", False),
            ("the clip open here is the same clip", [], [{"start": 11, "end": 30}], [], "", False),
        ]
        for label, other, mine, offered, warning, player in cases:
            with self.subTest(label):
                card = run_proposal_cards([self.clip(1, whole, preview=True)], other, mine)[0]
                self.assertEqual(card["shown"]["segments"], offered)
                self.assertFalse(card["shown"]["trimmed"])
                self.assertEqual(re.findall(r'<div class="proposal-warn is-opening"[^>]*>(.*?) —',
                                            card["html"]), [warning] if warning else [])
                self.assertEqual("Stops the scroll: Nye pulls his glasses off" in card["html"],
                                 bool(offered))
                self.assertNotIn("Shortened", card["html"])
                self.assertEqual(card["video"], player)

    def test_whats_wrong_button_and_panel(self):
        clip = self.clip(6, [{"start": 10, "end": 30}], trimmed=False, preview=False)
        closed = run_proposal_cards([clip], [], [])[0]["html"]
        self.assertIn('data-act="feedback"', closed)
        self.assertIn('aria-expanded="false"', closed)
        self.assertNotIn("proposal-feedback", closed)
        state = {"id": 6, "tags": ["other"], "text": 'too "dark" <b>', "busy": False}
        opened = run_proposal_cards([clip], [], [], [state])[0]["html"]
        self.assertIn('aria-expanded="true"', opened)
        self.assertIn('class="proposal-feedback"', opened)
        self.assertIn("Re-suggest with this", opened)
        self.assertIn(">Dismiss</button>", opened)
        # Labels and the operator's text are escaped; the chosen chip shows as pressed.
        self.assertIn("Something &lt;else&gt;", opened)
        self.assertIn('value="too &quot;dark&quot; &lt;b&gt;"', opened)
        self.assertRegex(opened, r'data-tag="other"\s+aria-pressed="true"')
        self.assertRegex(opened, r'data-tag="wrong_language"\s+aria-pressed="false"')
        busy = run_proposal_cards([clip], [], [], [dict(state, busy=True)])[0]["html"]
        self.assertIn("Sending…", busy)
        self.assertIn(" disabled", busy)

    def test_round_two_warnings(self):
        cases = [
            ({}, []),
            ({"speech_understood": False}, ["Language: other-language speech"]),
            ({"language_doubt": "mostly <Hindi>"},
             ["Language: other-language speech — mostly &lt;Hindi&gt;"]),
            ({"premise_clear": False, "monologue": True},
             ["Premise unclear in the first seconds", "Mostly a reporter talking to camera"]),
            ({"premise_clear": None, "monologue": False, "follows_advice": True}, []),
            ({"follows_advice": False, "advice_note": "opens on the anchor"},
             ["Didn’t follow your advice: opens on the anchor"]),
        ]
        for engine, lines in cases:
            with self.subTest(engine=engine):
                clip = self.clip(7, [{"start": 10, "end": 30}], trimmed=False, preview=False,
                                 **engine)
                html = run_proposal_cards([clip], [], [])[0]["html"]
                self.assertEqual(re.findall(r'<div class="proposal-warn">(.*?)</div>', html),
                                 lines)

    def test_a_card_nobody_reviewed_says_so(self):
        clip = self.clip(5, [{"start": 10, "end": 25}], preview=False, reviewed=False)
        card = run_proposal_cards([clip], [{"start": 25, "end": 30}], [])[0]
        self.assertIn("Not reviewed (critic unavailable)", card["html"])
        self.assertNotIn("Stops the scroll", card["html"])


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class VisibleProposalJsTest(unittest.TestCase):
    """cut.html's visibleProposal / proposalConflict must mirror
    clip_proposals.visible_segments / clip_conflict: nothing is subtracted,
    play order is kept, an opening window overlapping another clip's is
    "same_opening", and IoU above 0.6 is "duplicate" (hidden). Runs the
    template's own functions in node, against the Python on the same cases."""

    LOOM = "The loom"
    CASES = [
        # (label, clip segments, other clips [(title, segments)], kind, conflicting title)
        ("cold open kept first, nothing else on the video",
         [{"start": 50, "end": 54}, {"start": 10, "end": 24}], [], "", ""),
        ("footage shared further in is fine",
         [{"start": 50, "end": 58}, {"start": 10, "end": 24}],
         [(LOOM, [{"start": 0, "end": 5}, {"start": 9, "end": 25}])], "", ""),
        ("opens inside another clip's opening window",
         [{"start": 11, "end": 30}], [(LOOM, [{"start": 10, "end": 12}, {"start": 80, "end": 99}])],
         "same_opening", LOOM),
        ("opens just before it: the windows overlap",
         [{"start": 8, "end": 30}], [(LOOM, [{"start": 10, "end": 12}, {"start": 80, "end": 99}])],
         "same_opening", LOOM),
        ("windows that only touch don't clash",
         [{"start": 13, "end": 30}], [(LOOM, [{"start": 10, "end": 12}, {"start": 80, "end": 99}])],
         "", ""),
        ("only the first PLAYED segment is the opening",
         [{"start": 50, "end": 54}, {"start": 10, "end": 24}],
         [(LOOM, [{"start": 200, "end": 210}, {"start": 10, "end": 13}])], "", ""),
        ("the same clip again is a duplicate, whatever its opening",
         [{"start": 10, "end": 30}], [(LOOM, [{"start": 12, "end": 30}])], "duplicate", LOOM),
        ("IoU exactly 0.6 is not a duplicate",
         [{"start": 0, "end": 10}], [("B", [{"start": 4, "end": 10}])], "", ""),
        ("a duplicate outranks a same-opening clash with another clip",
         [{"start": 10, "end": 30}],
         [("A", [{"start": 10, "end": 12}, {"start": 90, "end": 120}]),
          ("B", [{"start": 40, "end": 41}, {"start": 10, "end": 30}])], "duplicate", "B"),
    ]

    def test_python_agrees_on_every_case(self):
        """The server's visible_segments + clip_conflict, on the same cases."""
        from app import clip_proposals

        for label, clip, others, kind, title in self.CASES:
            with self.subTest(label):
                got, trimmed = clip_proposals.visible_segments(clip, [{"start": 0, "end": 999}])
                self.assertEqual(got, [{"start": float(s["start"]), "end": float(s["end"])}
                                       for s in clip])
                self.assertFalse(trimmed)
                found, other = clip_proposals.clip_conflict(
                    clip, [{"title": t, "segments": segs} for t, segs in others])
                self.assertEqual(found, kind)
                self.assertEqual((other or {}).get("title", ""), title)

    def test_the_page_agrees(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                    / "cut.html").read_text()
        start = template.index("const PROPOSAL_OPENING_SECONDS")
        end = template.index("// The suggestions worth showing right now.")
        functions = template[start:end]
        self.assertIn("function visibleProposal", functions)
        self.assertNotIn("PROPOSAL_MIN_FRAGMENT", template)       # nothing is subtracted
        harness = functions + """
let otherSegments = [], segments = [];
const cases = JSON.parse(process.argv[2]);
const out = cases.map(([label, clip, others]) => {
  otherSegments = [];
  others.forEach(([title, segs], i) => segs.forEach(s =>
    otherSegments.push({ start: s.start, end: s.end, cut_id: i + 1, title })));
  return visibleProposal({ segments: clip });
});
process.stdout.write(JSON.stringify(out));
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visible.js"
            path.write_text(harness)
            result = subprocess.run(["node", str(path), json.dumps(self.CASES)],
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        for (label, clip, _others, kind, title), got in zip(self.CASES, json.loads(result.stdout)):
            with self.subTest(label):
                self.assertEqual(got["conflict"], kind)
                self.assertEqual(got["conflictWith"], title)
                self.assertFalse(got["trimmed"])
                # A duplicate offers nothing (hidden); anything else is offered whole.
                self.assertEqual(got["segments"], [] if kind == "duplicate" else clip)


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class PollingJsTest(unittest.TestCase):
    """cut.html's handleEngineStatus, run in node against stubbed page helpers:
    when polling continues, and what the operator is told when a run ends."""

    HARNESS = """
let engineJob = null, engineBoot = null, engineScope = null, enginePollTimer = null;
let suggestedClips = [], reasonAsk = null, undoStack = [];
let calls;
function renderClipProposals() { calls.render++; }
function syncSuggestButton() { calls.sync++; }
function scheduleEnginePoll() { calls.poll++; }
function forgetProposalVideos() {}
function engineActive() {
  return !!engineJob && (engineJob.state === 'queued' || engineJob.state === 'running');
}
function engineStatusText(job) { return 'line'; }
const document = { querySelector: () => null, getElementById: () => null };
const window = { toast: m => calls.toasts.push(m) };
function alert(m) { calls.alerts.push(m); }
%s
const out = JSON.parse(process.argv[2]).map(c => {
  calls = { render: 0, sync: 0, poll: 0, alerts: [], toasts: [] };
  engineJob = c.before_job; engineBoot = c.boot; engineScope = c.scope;
  suggestedClips = [{ id: 1 }];
  undoStack = [JSON.stringify({ segments: [], pending: {}, suggestedClips: [{ id: 1 }] })];
  handleEngineStatus(c.d);
  return { polling: calls.poll > 0, alerts: calls.alerts, toasts: calls.toasts,
           clips: suggestedClips.map(x => x.id),
           undo: JSON.parse(undoStack[0]).suggestedClips.map(x => x.id),
           active: engineActive() };
});
process.stdout.write(JSON.stringify(out));
"""

    def run_cases(self, cases):
        template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                    / "cut.html").read_text()
        start = template.index("function handleEngineStatus")
        end = template.index("// One player at a time")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poll.js"
            path.write_text(self.HARNESS % template[start:end])
            result = subprocess.run(["node", str(path), json.dumps(cases)],
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return json.loads(result.stdout)

    def test_terminal_states(self):
        running = {"state": "running", "boot_id": "b1", "stage": "Ranking", "new": []}
        interrupted_text = "The suggestion run was interrupted — press Suggest again."
        cases = [
            # (label, boot at start, scope, status response, polling, alerts, toasts, clips)
            ("still running", "b1", None, {"job": running}, True, [], [], [1]),
            ("restart, run resumed under the new boot", "b1", None,
             {"job": dict(running, boot_id="b2")}, False, [interrupted_text], [], [1]),
            ("finished with picks", "b1", None,
             {"job": {"state": "done", "boot_id": "b1", "new": [5, 6], "message": ""},
              "clips": [{"id": 5}, {"id": 6}]}, False, [], [], [5, 6]),
            ("finished before a restart is still a result", "b1", None,
             {"job": {"state": "done", "boot_id": "b0", "new": [5], "message": ""},
              "clips": [{"id": 5}]}, False, [], [], [5]),
            ("found nothing, whole video", "b1", None,
             {"job": {"state": "done", "boot_id": "b1", "new": [], "message": ""}, "clips": []},
             False, ["The model found nothing worth clipping here."], [], []),
            ("found nothing in a chapter", "b1", {"view": {"a": 1, "b": 2}, "chapter": {"topic": "Fire"}},
             {"job": {"state": "done", "boot_id": "b1", "new": [], "message": ""}, "clips": []},
             False, ['Nothing worth clipping in "Fire".'], [], []),
            ("off topic", "b1", None,
             {"job": {"state": "done", "boot_id": "b1", "new": [], "outcome": "empty_topic",
                      "message": "The video model watched every draft and found nothing on topic."},
              "clips": [{"id": 1}]},
             False, ["The video model watched every draft and found nothing on topic."], [], [1]),
            ("fallback with picks is a toast, not an alert", "b1", None,
             {"job": {"state": "done", "boot_id": "b1", "new": [7],
                      "message": "These came from the transcript suggester."},
              "clips": [{"id": 7}]}, False, [], ["These came from the transcript suggester."], [7]),
            ("interrupted record", "b1", None,
             {"job": {"state": "interrupted", "boot_id": "b0", "new": [],
                      "message": interrupted_text}, "clips": [{"id": 1}]},
             False, [interrupted_text], [], [1]),
            ("record gone", "b1", None,
             {"job": {"state": "idle", "boot_id": "b1", "new": []}, "clips": [{"id": 1}]},
             False, [interrupted_text], [], [1]),
        ]
        payload = [{"boot": boot, "scope": scope, "before_job": running, "d": d}
                   for _label, boot, scope, d, *_ in cases]
        for case, got in zip(cases, self.run_cases(payload)):
            label, _boot, _scope, _d, polling, alerts, toasts, clips = case
            with self.subTest(label):
                self.assertEqual(got["polling"], polling)
                self.assertEqual(got["active"], polling)
                self.assertEqual(got["alerts"], alerts)
                self.assertEqual(got["toasts"], toasts)
                self.assertEqual(got["clips"], clips)
                # Undo can never bring back what the run superseded.
                self.assertEqual(got["undo"], clips)


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class VideoPollingJsTest(unittest.TestCase):
    """video.html's end of a suggestion run, run in node. Its alert() is
    dialogs.js's <dialog>, which does NOT block: the reason a run brought
    nothing must stay on screen, so only fresh suggestions may reload the page
    (an empty run replaced nothing, so there is nothing to reload for)."""

    HARNESS = """
let engineJob = null, engineBoot = null, enginePollTimer = null;
const suggestions = [{ id: 1 }];
let calls, respond, line, btn;
function scheduleEnginePoll() { calls.poll++; }
function element() {
  const classes = new Set();
  return { hidden: false, disabled: false, textContent: '',
           classList: { add: c => classes.add(c), remove: c => classes.delete(c),
                        toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)),
                        contains: c => classes.has(c) } };
}
const document = {
  getElementById: id => ({ 'suggest-status': line, 'suggest-video-btn': btn }[id] || null),
};
// As dialogs.js does it: alert() opens the styled <dialog> and returns at
// once. Nobody answers it here, so it is still up when the test looks.
const window = {
  alertDialog: m => { calls.dialogs.push(m); return new Promise(() => {}); },
};
function alert(m) { window.alertDialog(String(m)); }
const location = {
  reload() { calls.left.push('reload'); },
  set href(url) { calls.left.push(url); },
};
async function fetch() {
  const [status, body] = respond;
  return { status, ok: status < 400, json: async () => body };
}
%s
(async () => {
  const out = [];
  for (const c of JSON.parse(process.argv[2])) {
    calls = { poll: 0, dialogs: [], left: [] };
    line = element(); btn = element();
    engineJob = c.before_job; engineBoot = c.boot;
    if (c.post) { respond = c.post; await suggestVideoClips(btn); }
    else handleEngineStatus(c.d);
    await new Promise(resolve => setTimeout(resolve, 20));   // anything deferred runs now
    out.push({ polling: calls.poll > 0, dialogs: calls.dialogs, left: calls.left,
               ready: !btn.disabled && !btn.classList.contains('is-busy')
                      && btn.textContent === 'Re-suggest' && line.hidden,
               button: { disabled: btn.disabled, busy: btn.classList.contains('is-busy'),
                         text: btn.textContent } });
  }
  process.stdout.write(JSON.stringify(out));
})();
"""

    def run_cases(self, cases):
        template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                    / "video.html").read_text()
        # The Suggest button, the status line and the end of a run; not the
        # real poll timer.
        pieces = [template[template.index("async function suggestVideoClips"):
                           template.index("function scheduleEnginePoll")],
                  template[template.index("function handleEngineStatus"):
                           template.index("if (engineJob && (engineJob.state === 'queued'")]]
        script = "\n".join(pieces).replace("{{ c.id }}", "42")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_poll.js"
            path.write_text(self.HARNESS % script)
            result = subprocess.run(["node", str(path), json.dumps(cases)],
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return json.loads(result.stdout)

    def test_a_run_that_brings_nothing_keeps_its_dialog_on_screen(self):
        running = {"state": "running", "boot_id": "b1", "stage": "Ranking", "new": []}
        interrupted = "The suggestion run was interrupted — press Suggest again."
        off_topic = "The video model watched every draft and found nothing on topic."
        stood_in = ("The transcript suggester stood in and found nothing (Gemini prepaid credits "
                    "are depleted — add credit at https://ai.studio/projects).")

        def ended(state="done", **fields):
            return {"job": dict({"state": state, "boot_id": "b1", "new": [], "message": ""},
                                **fields), "clips": [{"id": 1}]}

        cases = [
            # (label, status answer or POST answer, dialogs, where the page went, polling)
            ("still running", {"d": {"job": running}}, [], [], True),
            ("off topic", {"d": ended(outcome="empty_topic", message=off_topic)},
             [off_topic], [], False),
            ("failed", {"d": ended("failed", outcome="failed",
                                   message="Suggestion failed: 402 credits depleted")},
             ["Suggestion failed: 402 credits depleted"], [], False),
            ("over budget", {"d": ended("skipped", outcome="budget",
                                        message="No suggestions: daily cap reached.")},
             ["No suggestions: daily cap reached."], [], False),
            ("interrupted record", {"d": ended("interrupted", boot_id="b0", message=interrupted)},
             [interrupted], [], False),
            ("record gone", {"d": ended("idle")}, [interrupted], [], False),
            ("restart, run resumed under the new boot",
             {"d": {"job": dict(running, boot_id="b2")}}, [interrupted], [], False),
            ("the transcript suggester stood in and found nothing",
             {"d": ended(outcome="legacy", message=stood_in)}, [stood_in], [], False),
            ("found nothing", {"d": ended(outcome="empty")},
             ["The model found nothing worth clipping here."], [], False),
            ("found nothing in a stretch", {"d": ended(window={"start": 60, "end": 120})},
             ["Nothing worth clipping in this stretch."], [], False),
            # Fresh suggestions reload the page; their note rides along as its toast.
            ("picks", {"d": ended(new=[5, 6])}, [], ["reload"], False),
            ("picks with a note",
             {"d": ended(outcome="legacy", new=[7],
                         message="These came from the transcript suggester.")},
             [], ["/video/42?msg=These%20came%20from%20the%20transcript%20suggester."], False),
            ("picks from before a restart are still a result",
             {"d": ended(boot_id="b0", new=[5])}, [], ["reload"], False),
            # Through the Suggest button.
            ("transcript pass found nothing",
             {"post": [200, {"clips": [{"id": 1}], "new": 0, "multi_clip": False}]},
             ["The model found nothing worth clipping here."], [], False),
            ("transcript pass found clips",
             {"post": [200, {"clips": [{"id": 3}], "new": 1, "multi_clip": False}]},
             [], ["reload"], False),
            ("refused while the server shuts down",
             {"post": [202, {"started": False, "job": ended("interrupted", boot_id="b9",
                                                             message=interrupted)["job"]}]},
             [interrupted], [], False),
            ("queued", {"post": [202, {"started": True, "job": dict(running, state="queued")}]},
             [], [], True),
            ("busy with another stretch",
             {"post": [409, {"error": "Already suggesting clips for this video", "job": running}]},
             ["Already suggesting clips for this video"], [], True),
        ]
        payload = [dict(case, before_job=None if "post" in case else running,
                        boot=None if "post" in case else "b1")
                   for _label, case, *_ in cases]
        for (label, _case, dialogs, left, polling), got in zip(cases, self.run_cases(payload)):
            with self.subTest(label):
                self.assertEqual(got["dialogs"], dialogs)
                self.assertEqual(got["left"], left)
                self.assertEqual(got["polling"], polling)
                if not left and not polling:
                    # The page stays, ready for another go.
                    self.assertTrue(got["ready"])


    def test_suggest_stays_live_while_an_unattended_run_waits(self):
        """An archive run (or one a restart re-queued) still waiting behind
        other videos: the button stays pressable, since pressing it moves the
        run to the front as the operator's (jobs._promote). Once it is the
        operator's, or running, the button waits like any run."""
        waiting = {"state": "queued", "boot_id": "b1", "stage": "Queued", "new": [],
                   "reason": "archive", "queue_position": 3}
        cases = [
            # (label, case, button disabled, button text, polling)
            ("an archive run waiting", {"d": {"job": waiting}}, False, "Suggest now", True),
            ("a resumed run waiting", {"d": {"job": dict(waiting, reason="resume")}},
             False, "Suggest now", True),
            ("pressed: moved up as the operator's",
             {"post": [202, {"started": False, "job": dict(waiting, reason="reroll",
                                                           queue_position=1)}]},
             True, "Suggesting…", True),
            ("the archive run under way",
             {"d": {"job": dict(waiting, state="running", stage="Ranking")}},
             True, "Suggesting…", True),
            ("a section pass waiting", {"d": {"job": dict(waiting, reason="section")}},
             True, "Suggesting…", True),
        ]
        payload = [dict(case, before_job=None if "post" in case else waiting,
                        boot=None if "post" in case else "b1")
                   for _label, case, *_ in cases]
        for (label, _case, disabled, text, polling), got in zip(cases, self.run_cases(payload)):
            with self.subTest(label):
                self.assertEqual(got["dialogs"], [])
                self.assertEqual(got["polling"], polling)
                self.assertEqual(got["button"], {"disabled": disabled, "busy": disabled,
                                                 "text": text})


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class CutSuggestButtonJsTest(unittest.TestCase):
    """cut.html's Suggest button while a run is queued or running."""

    HARNESS = """
let engineJob = null;
const btn = { disabled: false, cls: new Set(),
              classList: { toggle(c, on) { on ? btn.cls.add(c) : btn.cls.delete(c); } } };
const document = { getElementById: id => (id === 'suggest-here' ? btn : null) };
%s
process.stdout.write(JSON.stringify(JSON.parse(process.argv[2]).map(job => {
  engineJob = job;
  syncSuggestButton();
  return { disabled: btn.disabled, busy: btn.cls.has('is-busy') };
})));
"""

    def test_a_waiting_unattended_run_leaves_suggest_pressable(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                    / "cut.html").read_text()
        script = template[template.index("function engineActive"):
                          template.index("function scheduleEnginePoll")]
        cases = [
            (None, False),
            ({"state": "queued", "reason": "archive"}, False),
            ({"state": "queued", "reason": "resume"}, False),
            ({"state": "running", "reason": "archive"}, True),
            ({"state": "queued", "reason": "reroll"}, True),
            ({"state": "queued", "reason": "section"}, True),
            ({"state": "done", "reason": "archive"}, False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "button.js"
            path.write_text(self.HARNESS % script)
            result = subprocess.run(["node", str(path), json.dumps([c for c, _ in cases])],
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        for (job, busy), got in zip(cases, json.loads(result.stdout)):
            with self.subTest(job=job):
                self.assertEqual(got, {"disabled": busy, "busy": busy})


# ---- operator advice and notes (round 2) ---------------------------------------------------

class NotesRouteBase(RouteTestBase):
    def setUp(self) -> None:
        super().setUp()
        from app import config
        from app.clip_engine import notes

        self.notes, self.config = notes, config
        conf = self.dir / "config"
        conf.mkdir()
        (conf / "brand.yaml").write_text("topic: climate change\n")
        self.patch(mock.patch.object(config, "CONFIG_DIR", conf))
        self.propose_soon = self.patch(mock.patch.object(notes, "propose_rules_soon"))
        self.enqueue.return_value = {"started": True, "conflict": False, **job_status("queued")}

    def proposal_row(self, pid: int) -> ClipProposal:
        with self.Session() as s:
            return s.get(ClipProposal, pid)


class SuggestAdviceTest(NotesRouteBase):
    def test_advice_from_the_form_or_the_query(self):
        cid = self.add_candidate()
        for how in ("form", "query"):
            with self.subTest(how=how):
                if how == "form":
                    r = self.client.post(f"/video/{cid}/suggest-clips",
                                         data={"advice": "  open on the floods;  no interviews "})
                else:
                    r = self.client.post(f"/video/{cid}/suggest-clips?advice=open%20on%20the%20floods"
                                         "%3B%20no%20interviews")
                self.assertEqual(r.status_code, 202)
                self.assertEqual(self.enqueue.call_args.kwargs["advice"],
                                 "open on the floods; no interviews")
        steer = self.notes.for_candidate(cid)
        self.assertEqual([(n["kind"], n["text"], n["video_id"]) for n in steer],
                         [("steer", "open on the floods; no interviews", "vid1")] * 2)
        self.assertEqual(self.propose_soon.call_count, 2)

    def test_no_advice_no_note(self):
        cid = self.add_candidate()
        self.client.post(f"/video/{cid}/suggest-clips")
        self.assertEqual(self.enqueue.call_args.kwargs["advice"], "")
        self.assertEqual(self.notes.for_candidate(cid), [])
        self.propose_soon.assert_not_called()

    def test_a_refused_request_keeps_no_note(self):
        no_transcript = self.add_candidate("vid2", transcript=False)
        r = self.client.post(f"/video/{no_transcript}/suggest-clips", data={"advice": "x"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.notes.for_candidate(no_transcript), [])

    def test_a_busy_video_says_the_advice_waits(self):
        cid = self.add_candidate()
        self.enqueue.return_value = {"started": False, "conflict": True, **job_status("running")}
        r = self.client.post(f"/video/{cid}/suggest-clips", data={"advice": "no interviews"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("advice", r.json()["error"])

    def test_the_transcript_suggester_keeps_the_advice_as_a_note(self):
        cid = self.add_candidate()
        self.setup_answer = ("transcript", "")
        pending = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        with mock.patch.object(self.main, "_source_ready", return_value=None), \
                mock.patch.object(self.main.clip_proposals, "propose", return_value=[pending]):
            r = self.client.post(f"/video/{cid}/suggest-clips", data={"advice": "no interviews"})
            plain = self.client.post(f"/video/{cid}/suggest-clips")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["advice_ignored"])
        self.assertNotIn("advice_ignored", plain.json())
        self.assertEqual([n["kind"] for n in self.notes.for_candidate(cid)], ["steer"])
        self.enqueue.assert_not_called()


class FeedbackRouteTest(NotesRouteBase):
    def post(self, pid: int, **data):
        return self.client.post(f"/clip-proposal/{pid}/feedback", data=data)

    def test_whats_wrong_dismisses_notes_and_re_suggests_with_advice(self):
        cid = self.add_candidate()
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        r = self.post(pid, tags="other,wrong_language,monologue,bogus",
                      text="open on the floods")
        self.assertEqual(r.status_code, 202, r.text)
        body = r.json()
        self.assertEqual(body["dismissed"], pid)
        self.assertEqual(body["engine"], "gemini")
        row = self.proposal_row(pid)
        self.assertEqual((row.verdict, row.dismiss_reason), ("dismissed", "not_clipworthy"))
        [note] = self.notes.for_candidate(cid)
        self.assertEqual(body["noted"], note["id"])
        self.assertEqual((note["kind"], note["proposal_id"], note["video_id"], note["tags"],
                          note["text"]),
                         ("card", pid, "vid1", ["other", "wrong_language", "monologue"],
                          "open on the floods"))
        args, kwargs = self.enqueue.call_args
        self.assertEqual(args, (cid,))
        self.assertEqual((kwargs["window"], kwargs["reason"], kwargs["topic"]),
                         (None, "reroll", ""))
        self.assertEqual(kwargs["advice"], self.notes.advice_from(
            ["other", "wrong_language", "monologue"], "open on the floods"))
        self.assertIs(kwargs["on_done"], self.main._after_background_work)
        self.propose_soon.assert_called_once_with()

    def test_dismiss_reason_mapping(self):
        cid = self.add_candidate()
        cases = [("weak_opening", "weak_visuals"), ("other,too_long", "too_long"),
                 ("off_topic", "off_topic"), ("bad_boundaries", "bad_boundaries"),
                 ("no_premise,off_topic", "not_clipworthy"), ("other", "")]
        for tags, reason in cases:
            with self.subTest(tags=tags):
                pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
                self.assertEqual(self.post(pid, tags=tags, resuggest="0").status_code, 200)
                row = self.proposal_row(pid)
                self.assertEqual((row.verdict, row.dismiss_reason), ("dismissed", reason))

    def test_a_section_pass_re_suggests_its_own_stretch(self):
        chapters = [{"start": 0, "end": 100, "topic": "Intro", "summary": "the anchor"},
                    {"start": 100, "end": 300, "topic": "Fire", "summary": "the planet burns"}]
        cid = self.add_candidate(chapters=json.dumps(chapters))
        pid = self.add_proposal(cid, [{"start": 130.0, "end": 150.0}])
        self.engine_sidecar(pid, [{"start": 130.0, "end": 150.0}],
                            window={"start": 120.0, "end": 180.0})
        self.assertEqual(self.post(pid, tags="boring").status_code, 202)
        kwargs = self.enqueue.call_args.kwargs
        self.assertEqual((kwargs["window"], kwargs["reason"], kwargs["topic"]),
                         ({"start": 120.0, "end": 180.0}, "section", "Fire — the planet burns"))
        # A sidecar with no (or a broken) window means the whole video.
        for window in (None, {"start": "x"}, {"start": 50, "end": 10}):
            with self.subTest(window=window):
                other = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
                self.engine_sidecar(other, [{"start": 10.0, "end": 30.0}], window=window)
                self.post(other, tags="boring")
                self.assertIsNone(self.enqueue.call_args.kwargs["window"])

    def test_just_dismiss_keeps_the_note_and_stops(self):
        cid = self.add_candidate()
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        r = self.post(pid, text="nobody says where this is", resuggest="0")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["ok"], True)
        self.assertEqual(self.proposal_row(pid).verdict, "dismissed")
        self.assertEqual(len(self.notes.for_candidate(cid)), 1)
        self.enqueue.assert_not_called()

    def test_saying_nothing_is_refused_and_changes_nothing(self):
        cid = self.add_candidate()
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        r = self.post(pid, tags="bogus", text="   ")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.proposal_row(pid).verdict, "pending")
        self.assertEqual(self.notes.for_candidate(cid), [])
        self.enqueue.assert_not_called()

    def test_unknown_and_already_used(self):
        cid = self.add_candidate()
        self.assertEqual(self.post(4242, tags="boring").status_code, 404)
        used = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}], verdict="accepted")
        self.assertEqual(self.post(used, tags="boring").status_code, 409)
        self.assertEqual(self.notes.for_candidate(cid), [])
        # Already dismissed (the ✕ first, then "What's wrong?"): still taken.
        gone = self.add_proposal(cid, [{"start": 40.0, "end": 60.0}], verdict="dismissed")
        self.assertEqual(self.post(gone, tags="too_long").status_code, 202)
        self.assertEqual(self.proposal_row(gone).dismiss_reason, "too_long")

    def test_a_busy_video_still_keeps_the_note(self):
        cid = self.add_candidate()
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        self.enqueue.return_value = {"started": False, "conflict": True, **job_status("running")}
        r = self.post(pid, tags="boring")
        self.assertEqual(r.status_code, 409)
        body = r.json()
        self.assertEqual(body["job"]["state"], "running")
        self.assertTrue(body["noted"])
        self.assertEqual(body["dismissed"], pid)
        self.assertEqual(self.proposal_row(pid).verdict, "dismissed")

    def test_the_transcript_suggester_also_runs_through_the_queue(self):
        cid = self.add_candidate()
        self.setup_answer = ("transcript", "")
        pid = self.add_proposal(cid, [{"start": 10.0, "end": 30.0}])
        r = self.post(pid, tags="boring")
        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.json()["engine"], "transcript")
        self.enqueue.assert_called_once()


class RuleProposalRoutesTest(NotesRouteBase):
    def setUp(self) -> None:
        super().setUp()
        self.notes.add_proposals([
            {"text": "never open on a reporter talking to camera", "evidence": ["n1", "n2", "n3"],
             "count": 3, "summary": "From 3 of your notes on 2 videos"},
            {"text": "Open on the footage", "evidence": [101], "count": 12,
             "when": "news packages"},
        ], source="notes")
        self.notes.add_proposals([{"text": "Cut on complete sentences", "count": 9}],
                                 source="handbook")

    def test_list(self):
        body = self.client.get("/clip-engine/rule-proposals").json()
        self.assertEqual([(p["idx"], p["source"]) for p in body["proposals"]],
                         [(0, "notes"), (1, "notes"), (2, "handbook")])
        self.assertEqual(body["proposals"][0]["markdown"],
                         "**Rule:** Never open on a reporter talking to camera.\n**When:** Always"
                         "\n**Why:** From 3 of your notes on 2 videos.")
        self.client.post("/clip-engine/rule-proposals/2/dismiss")
        self.assertEqual(len(self.client.get("/clip-engine/rule-proposals").json()["proposals"]), 2)
        everything = self.client.get("/clip-engine/rule-proposals?status=all").json()
        self.assertEqual([p["status"] for p in everything["proposals"]],
                         ["pending", "pending", "dismissed"])

    def test_add_makes_an_ordinary_clips_rule(self):
        r = self.client.post("/clip-engine/rule-proposals/1/add")
        self.assertEqual(r.status_code, 200)
        rule = r.json()["rule"]
        self.assertEqual(rule["text"], "When news packages: Open on the footage.")
        self.assertEqual((rule["applies_to"], rule["source"], rule["when"]),
                         ("clips", "notes", "News packages"))
        self.assertEqual(rule["evidence"], "From 12 of your notes (when: News packages)")
        self.assertEqual(rule["original"], "Open on the footage")
        self.assertEqual(self.config.load_brand_rules(), [rule])
        self.assertIn("Open on the footage", self.config.render_brand_rules("content"))
        self.assertEqual(self.config.render_brand_rules("captions"), "")
        self.assertEqual(self.notes.proposals()[1]["status"], "added")
        # Decided: gone.
        self.assertEqual(self.client.post("/clip-engine/rule-proposals/1/add").status_code, 404)
        self.assertEqual(self.client.post("/clip-engine/rule-proposals/9/add").status_code, 404)

    def test_handbook_proposals_keep_their_source(self):
        rule = self.client.post("/clip-engine/rule-proposals/2/add").json()["rule"]
        self.assertEqual(rule["source"], "handbook")
        self.assertEqual(rule["evidence"], "Seen in 9 of your past edits")

    def test_a_reworded_rule_is_formatted_again(self):
        with mock.patch.object(self.notes, "format_rule",
                               wraps=self.notes.format_rule) as fmt:
            rule = self.client.post("/clip-engine/rule-proposals/0/add",
                                    data={"text": "never open on an anchor"}).json()["rule"]
        fmt.assert_called_once()
        self.assertEqual(rule["text"], "Never open on an anchor.")
        self.assertEqual(rule["original"], "never open on an anchor")

    def test_a_rule_already_there_still_marks_it_added(self):
        self.config.add_brand_rule("Never open on a reporter talking to camera.", "everything")
        r = self.client.post("/clip-engine/rule-proposals/0/add")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["rule"])
        self.assertEqual(self.notes.proposals()[0]["status"], "added")

    def test_dismiss(self):
        self.assertEqual(self.client.post("/clip-engine/rule-proposals/0/dismiss").json(),
                         {"ok": True})
        self.assertEqual(self.client.post("/clip-engine/rule-proposals/0/dismiss").status_code, 404)
        self.assertEqual(self.config.load_brand_rules(), [])

    def test_writes_touch_no_cached_page(self):
        for path in ("/clip-engine/rule-proposals/{idx}/add",
                     "/clip-engine/rule-proposals/{idx}/dismiss"):
            self.assertEqual(self.main._WRITE_SCOPE[path], ())

    def test_brand_page_lists_them_and_badges_rules(self):
        self.notes.add_proposals([{"text": "Never trust </script><b>this</b>"}], source="notes")
        self.config.add_brand_rule("A learned rule", "clips", source="handbook",
                                   evidence="Seen in 5 of your past edits")
        r = self.client.get("/brand")
        self.assertEqual(r.status_code, 200)
        html = r.text
        self.assertIn("Suggested rules", html)
        self.assertNotIn('id="rule-proposals" hidden', html)
        proposals = json.loads(re.search(r"let ruleProposals = (\[.*?\]);\n", html).group(1))
        self.assertEqual([p["source"] for p in proposals], ["notes", "notes", "handbook", "notes"])
        self.assertNotIn("</script><b>", html)          # tojson escapes it
        rules = json.loads(re.search(r"let rules = (\[.*?\]);\n", html).group(1))
        self.assertEqual(rules[0]["source"], "handbook")
        self.assertIn("['clips', 'Clips'", html)               # the scope groups
        self.assertIn("Learned from your edits", html)
        if shutil.which("node"):
            for i, script in enumerate(re.findall(r"<script>(.*?)</script>", html, flags=re.S)):
                path = self.dir / f"brand-{i}.js"
                path.write_text(script)
                done = subprocess.run(["node", "--check", str(path)], capture_output=True,
                                      text=True, timeout=60)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])

    def test_brand_page_without_proposals(self):
        self.notes.util.atomic_write_json(self.notes.proposals_path(), {"proposals": []})
        html = self.client.get("/brand").text
        self.assertIn('id="rule-proposals" hidden', html)
        with mock.patch.object(self.notes, "pending_proposals", side_effect=OSError("x")):
            self.assertEqual(self.client.get("/brand").status_code, 200)


class BrandRuleFormattingTest(NotesRouteBase):
    def fake_format(self):
        def fmt(text, **kw):
            return dict(self.notes._tidy_rule(text.replace("teh", "the")), formatted=True)

        return mock.patch.object(self.notes, "format_rule", side_effect=fmt)

    def test_saving_formats_new_and_edited_rules_only(self):
        formatted = {"text": "When floods: Open on the water.", "rule": "Open on the water.",
                     "when": "Floods", "markdown": "**Rule:** Open on the water.\n**When:** Floods",
                     "original": "open on teh water for floods", "source": "notes",
                     "applies_to": "clips"}
        edited = dict(formatted, text="Open on the water, always")
        rules = [formatted, edited, {"text": "write as we", "applies_to": "captions"}]
        with self.fake_format() as fmt:
            r = self.client.post("/brand", data={"topic": "climate change",
                                                 "rules_json": json.dumps(rules)},
                                 follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual([c.args[0] for c in fmt.call_args_list],
                         ["Open on the water, always", "write as we"])
        saved = self.config.load_brand_rules()
        self.assertEqual(saved[0], {**formatted, "enabled": True})        # untouched
        self.assertEqual((saved[1]["text"], saved[1]["source"], saved[1]["original"]),
                         ("Open on the water, always.", "notes", "Open on the water, always"))
        self.assertEqual((saved[2]["text"], saved[2]["applies_to"]), ("Write as we.", "captions"))

    def test_the_caption_editors_add_goes_through_the_formatter(self):
        with self.fake_format():
            r = self.client.post("/brand/rules/add", data={"text": "no emojis in teh caption"})
        rule = r.json()["rule"]
        self.assertEqual(rule["text"], "No emojis in the caption.")
        self.assertEqual(rule["original"], "no emojis in the caption")   # (the fake fixed it)
        self.assertEqual(rule["applies_to"], "captions")
        with self.fake_format():
            again = self.client.post("/brand/rules/add", data={"text": "No emojis in the caption"})
        self.assertEqual(again.status_code, 409)


class AdvicePageTest(NotesRouteBase):
    def setUp(self) -> None:
        super().setUp()
        self.cid = self.add_candidate(local_video_path=str(self.dir / "vid1.mp4"))
        self.pid = self.add_proposal(self.cid, [{"start": 10.0, "end": 30.0}])
        self.engine_sidecar(self.pid, [{"start": 10.0, "end": 30.0}], critique={
            "stops_scroll": True, "stop_reason": "floods", "error": "", "premise_clear": False,
            "speech_understood": True, "monologue": False, "follows_advice": False,
            "advice_note": "opens on <the anchor>"}, language_doubt="mostly Hindi")
        self.cut_id = self.add_cut(self.cid, [])

    def test_video_page(self):
        html = self.client.get(f"/video/{self.cid}").text
        self.assertIn('id="suggest-advice"', html)
        self.assertIn("Advice for this pass, e.g. open on the disaster footage; no interviews",
                      html)
        self.assertIn(f"toggleSuggestFeedback({self.pid}, this)", html)
        self.assertIn(f'id="suggest-feedback-{self.pid}"', html)
        for slug, label in self.notes.tag_labels():
            self.assertIn(f'data-tag="{slug}"', html)
        self.assertIn("Re-suggest with this", html)
        self.assertIn("Language: other-language speech — mostly Hindi", html)
        self.assertIn("Premise unclear in the first seconds", html)
        self.assertIn("Didn’t follow your advice: opens on &lt;the anchor&gt;", html)
        self.assertNotIn("Mostly a reporter talking to camera", html)

    def test_cut_page(self):
        html = self.client.get(f"/cut/{self.cut_id}").text
        self.assertIn('id="suggest-advice"', html)
        tags = json.loads(re.search(r"const NOTE_TAGS = (\[.*?\]);\n", html).group(1))
        self.assertEqual(tags, [list(t) for t in self.notes.tag_labels()])
        clip = json.loads(re.search(r"let suggestedClips = (\[.*?\]);\n", html).group(1))[0]
        self.assertEqual(clip["engine"]["follows_advice"], False)

    def test_pages_render_when_the_notes_module_breaks(self):
        with mock.patch.object(self.notes, "tag_labels", side_effect=RuntimeError("x")):
            html = self.client.get(f"/cut/{self.cut_id}").text
        self.assertIn("const NOTE_TAGS = [];", html)
        with mock.patch.object(self.notes, "tag_labels", side_effect=RuntimeError("x")):
            self.assertEqual(self.client.get(f"/video/{self.cid}").status_code, 200)


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class SuggestWithAdviceJsTest(unittest.TestCase):
    """The Suggest buttons send the advice box's text, and only when there is some."""

    def run_video(self, advice):
        template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                    / "video.html").read_text()
        script = template[template.index("async function suggestVideoClips"):
                          template.index("function scheduleEnginePoll")]
        harness = """
let engineJob = null, engineBoot = null; const suggestions = [];
let sent = null; const box = %s;
const document = { getElementById: id => (id === 'suggest-advice' ? box : null) };
function alert() {} function handleEngineStatus() {}
const location = { reload() {} };
async function fetch(url, opts) { sent = String(opts.body); return { status: 202, ok: true,
  json: async () => ({ started: true, job: { state: 'queued', boot_id: 'b' } }) }; }
%s
(async () => {
  const btn = { disabled: false, textContent: '', classList: { add() {}, remove() {}, toggle() {} } };
  await suggestVideoClips(btn);
  process.stdout.write(JSON.stringify({ sent, left: box ? box.value : null }));
})();
""" % (json.dumps({"value": advice}) if advice is not None else "null",
       script.replace("{{ c.id }}", "42"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "advice.js"
            path.write_text(harness)
            result = subprocess.run(["node", str(path)], capture_output=True, text=True,
                                    timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return json.loads(result.stdout)

    def test_video_page(self):
        self.assertEqual(self.run_video("  no interviews "),
                         {"sent": "advice=no+interviews", "left": ""})
        self.assertEqual(self.run_video("   "), {"sent": "", "left": ""})
        self.assertEqual(self.run_video(None), {"sent": "", "left": None})


# ---- the archive hand-off and process wiring --------------------------------------------------

class ArchiveThreadTest(RouteTestBase):
    def fake_archive(self, *, to_status=STATUS_ARCHIVED, gate=None, entered=None, calls=None):
        def archive(session, candidate, with_suggestions=True):
            if calls is not None:
                calls.append(candidate.id)
            if entered is not None:
                entered.set()
            if gate is not None:
                gate.wait(5)
            if candidate.status == STATUS_ARCHIVED:
                return
            candidate.status = to_status
            candidate.archived_at = utcnow()
        return archive

    def test_after_the_commit_the_engine_is_queued(self):
        cid = self.add_candidate(status=STATUS_APPROVED)
        seen = {}

        def enqueue(candidate_id, **kwargs):
            with self.Session() as s:     # what another session sees right now
                seen["status"] = s.get(Candidate, candidate_id).status
            seen.update(kwargs)
            return {"started": True}

        self.enqueue.side_effect = enqueue
        with mock.patch.object(self.main, "archive_candidate", self.fake_archive()):
            self.main._scrape_in_thread(cid)
        self.assertEqual(seen["status"], STATUS_ARCHIVED)          # committed first
        self.assertEqual(seen["reason"], "archive")
        self.assertIs(seen["on_done"], self.main._after_background_work)

    def test_when_nothing_is_queued(self):
        cases = ["transcript setup", "pending suggestions", "already archived",
                 "archive failed", "no transcript"]
        for i, case in enumerate(cases):
            with self.subTest(case):
                self.enqueue.reset_mock()
                self.setup_answer = ("transcript", "") if case == "transcript setup" else ("gemini", "")
                status = STATUS_ARCHIVED if case == "already archived" else STATUS_APPROVED
                cid = self.add_candidate(f"skip{i}", status=status,
                                         transcript=case != "no transcript")
                if case == "pending suggestions":
                    self.add_proposal(cid, [{"start": 1.0, "end": 9.0}])
                to_status = STATUS_FAILED if case == "archive failed" else STATUS_ARCHIVED
                with mock.patch.object(self.main, "archive_candidate",
                                       self.fake_archive(to_status=to_status)):
                    self.main._scrape_in_thread(cid)
                self.enqueue.assert_not_called()

    def test_an_engine_problem_is_not_an_archive_failure(self):
        cid = self.add_candidate(status=STATUS_APPROVED)
        self.enqueue.side_effect = RuntimeError("engine import broke")
        with mock.patch.object(self.main, "archive_candidate", self.fake_archive()), \
                self.assertLogs("web", level="ERROR") as logs:
            self.main._scrape_in_thread(cid)
        self.assertTrue(any("Couldn't queue clip suggestions" in line for line in logs.output))
        self.assertFalse(any("Background scrape failed" in line for line in logs.output))
        with self.Session() as s:
            self.assertEqual(s.get(Candidate, cid).status, STATUS_ARCHIVED)

    def test_one_archive_per_video_at_a_time(self):
        cid = self.add_candidate(status=STATUS_APPROVED)
        entered, gate, calls = threading.Event(), threading.Event(), []
        fake = self.fake_archive(gate=gate, entered=entered, calls=calls)
        with mock.patch.object(self.main, "archive_candidate", fake):
            first = threading.Thread(target=self.main._scrape_in_thread, args=(cid,), daemon=True)
            first.start()
            self.assertTrue(entered.wait(5))
            self.main._scrape_in_thread(cid)          # a Retry click mid-archive
            self.assertEqual(calls, [cid])
            gate.set()
            first.join(5)
            self.assertNotIn(cid, self.main._scrape_inflight)
            self.main._scrape_in_thread(cid)          # once it's done, another may run
        self.assertEqual(calls, [cid, cid])


class WiringTest(RouteTestBase):
    def test_the_spend_chart_shows_the_engine_beside_the_shared_budget(self):
        """Engine spend is kept out of the shared budget's figures, so the
        chart gets it separately — and a broken ledger never breaks the page."""
        from app.config import Settings

        spend.record("gemini-3.8-flash", 1_000_000, 0, bucket="engine")   # $1.50
        spend.record("claude-haiku-4-5", 1_000_000, 0)                     # $1.00, shared
        cases = [({"clip_engine": {"daily_usd": 7.5}}, 7.5), ({"clip_engine": {"daily_usd": ""}}, 5.0),
                 ({}, 5.0)]
        for raw, cap in cases:
            with self.subTest(raw=raw), \
                    mock.patch.object(self.main, "load_settings", return_value=Settings(raw=raw)):
                view = self.main._engine_spend_view()
                self.assertEqual(view["engine_spend_cap"], cap)
                self.assertAlmostEqual(view["engine_spend_today"], 1.5)
                self.assertEqual(len(view["engine_spend_recent"]), 30)
                self.assertAlmostEqual(view["engine_spend_recent"][0]["spend_usd"], 1.5)
        self.assertAlmostEqual(spend.recent(1)[0]["spend_usd"], 1.0)   # the shared bars, untouched
        with mock.patch.object(spend, "recent", side_effect=OSError("disk")):
            self.assertEqual(self.main._engine_spend_view(),
                             {"engine_spend_recent": [], "engine_spend_today": 0.0,
                              "engine_spend_cap": 0.0})

    def test_shutdown_stops_the_engine(self):
        self.assertIn(self.main._stop_clip_engine, self.main.app.router.on_shutdown)
        with mock.patch.object(self.jobs, "shutdown") as shutdown:
            self.main._stop_clip_engine()
        shutdown.assert_called_once_with()
        with mock.patch.object(self.jobs, "shutdown", side_effect=RuntimeError("x")):
            self.main._stop_clip_engine()             # never raises

    def test_startup_resumes_interrupted_runs(self):
        with mock.patch.object(self.jobs, "resume_recent", return_value=2) as resume:
            self.main._resume_clip_engine_runs()
        resume.assert_called_once_with(on_done=self.main._after_background_work)
        with mock.patch.object(self.jobs, "resume_recent", side_effect=RuntimeError("x")):
            self.main._resume_clip_engine_runs()       # never raises

    def test_background_work_starts_with_the_server_not_on_import(self):
        """Scripts import main for its loaders, smoke_pages.py against the
        live database: importing must not queue paid engine runs or restart
        downloads in a process about to exit. The server's startup does."""
        from fastapi.testclient import TestClient

        self.assertIn(self.main._start_background_work, self.main.app.router.on_startup)
        names = ("_resume_stalled_scrapes", "_resume_clip_engine_runs",
                 "_retrain_triage_model_in_background")
        with contextlib.ExitStack() as stack:
            work = {n: stack.enter_context(mock.patch.object(self.main, n)) for n in names}
            stack.enter_context(mock.patch.object(self.jobs, "shutdown"))
            work["_resume_stalled_scrapes"].side_effect = RuntimeError("database away")
            with self.assertLogs("web", "ERROR"), TestClient(self.main.app):
                pass
            for name, fn in work.items():
                fn.assert_called_once_with()           # one failing step stops no other
        # A fresh interpreter importing the module starts none of it.
        probe = ("from unittest import mock\n"
                 "from app.clip_engine import jobs\n"
                 "with mock.patch.object(jobs, 'resume_recent') as resume:\n"
                 "    import app.web.main\n"
                 "print('resumed' if resume.called else 'quiet')\n")
        env = dict(os.environ, DATABASE_URL="sqlite://", SCHEDULER_EMBEDDED="false",
                   GEMINI_API_KEY="", ANTHROPIC_API_KEY="", SUPABASE_SERVICE_KEY="",
                   CLIP_ENGINE_DIR=str(self.dir / "engine"), CLIPBENCH_DIR=str(self.dir / "bench"))
        root = Path(__file__).resolve().parents[1]
        done = subprocess.run([sys.executable, "-c", probe], cwd=root, env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        self.assertEqual(done.stdout.strip().splitlines()[-1], "quiet")

    def test_templates_are_shared_with_routers_and_write_scopes_are_real(self):
        self.assertIs(self.main.app.state.templates, self.main.templates)
        # FastAPI 0.139 lists an included router as one lazy entry in app.routes;
        # iter_route_contexts is the flattened view requests are matched against.
        try:
            from fastapi.routing import iter_route_contexts
        except ImportError:   # older FastAPI copies included routes in directly
            iter_route_contexts = list
        routes = list(iter_route_contexts(self.main.app.routes))
        known = {getattr(r, "path_format", None) for r in routes}
        self.assertEqual([p for p in self.main._WRITE_SCOPE if p not in known], [])
        bench_posts = [r.path_format for r in routes
                       if (getattr(r, "path_format", "") or "").startswith("/clip-bench")
                       and "POST" in (getattr(r, "methods", None) or ())]
        for path in bench_posts:
            self.assertEqual(self.main._WRITE_SCOPE.get(path), ())


if __name__ == "__main__":
    unittest.main()
