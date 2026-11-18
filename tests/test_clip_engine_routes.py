"""Route tests for the clip engine in the dashboard (app/web/main.py): the
202 + polling contract of "Suggest", the status endpoint, card previews and
their suppression once a suggestion is trimmed, the archive hand-off, and the
startup/shutdown wiring.

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

    def add_cut(self, cid: int, segments: list[dict]) -> int:
        with self.Session() as s:
            cut = Cut(candidate_pk=cid, trim_segments=json.dumps(segments))
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
        self.assertIs(kwargs["on_done"], self.main.pagecache.invalidate)

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

    def payload(self, pid: int, used: list[dict] | None = None) -> dict:
        with self.Session() as s:
            row = s.get(ClipProposal, pid)
            return self.main._proposal_payload(row, used or [])

    def test_payload_offers_the_preview_only_while_it_shows_what_the_card_offers(self):
        cid = self.add_candidate()
        segments = [{"start": 10.0, "end": 24.0}, {"start": 40.0, "end": 52.0}]
        cases = [
            # (description, sidecar segments, file on disk, claimed ranges, preview expected)
            ("as rendered", segments, True, [], True),
            ("rounding only", [{"start": 10.01, "end": 24.0}, {"start": 40.0, "end": 51.98}],
             True, [], True),
            ("trimmed around another clip", segments, True, [{"start": 40.0, "end": 45.0}], False),
            ("render missing", segments, False, [], False),
            ("different cut", [{"start": 10.0, "end": 24.0}], True, [], False),
            ("different order", list(reversed(segments)), True, [], False),
        ]
        for label, rendered, with_file, used, expected in cases:
            with self.subTest(label):
                pid = self.add_proposal(cid, segments)
                self.engine_sidecar(pid, rendered, with_file=with_file)
                p = self.payload(pid, used)
                self.assertEqual(p["preview"], f"/clip-proposal/{pid}/preview" if expected else "")
                self.assertEqual(p["engine"]["archetype"], "voice")
                self.assertIn("confidence", p)

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
            "score": 7.4, "reviewed": True, "topic_doubt": "a film, not the news"})

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

    @unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
    def test_a_suggestion_the_server_trimmed_loses_its_verdict_on_both_pages(self):
        # Another clip already holds the proposal's last five seconds.
        self.add_cut(self.cid, [{"start": 25.0, "end": 30.0}])
        html = self.get(f"/cut/{self.cut_id}")

        def page_value(pattern: str):
            return json.loads(re.search(pattern, html).group(1))

        clip = page_value(r"let suggestedClips = (\[.*?\]);\n")[0]
        self.assertEqual((clip["trimmed"], clip["preview"]), (True, ""))
        self.assertEqual(clip["segments"], [{"start": 10.0, "end": 25.0}])
        self.assertTrue(clip["engine"]["reviewed"])
        card = run_proposal_cards([clip], page_value(r"const otherSegments = (\[.*?\]);\n"),
                                  page_value(r"let segments = (\[.*?\]);\n"))[0]
        # The page loaded with the claim already taken out, so the browser's
        # own pass over the same claims finds nothing to trim...
        self.assertFalse(card["shown"]["trimmed"])
        # ...and the card still knows the verdict was about the longer render.
        self.assertNotIn("proposal-critic", card["html"])
        self.assertNotIn("Gets slow at", card["html"])
        self.assertIn("Shortened around footage your other clips already use.", card["html"])
        self.assertFalse(card["video"])
        video_page = self.get(f"/video/{self.cid}")
        self.assertNotIn("Stops the scroll", video_page)
        self.assertIn("Shortened around footage your other clips already use.", video_page)

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


# cut.html's own card functions, run in node with just enough of a DOM: for
# each suggestion, what the browser offers (visibleProposal) and the card it
# draws (proposalCard), against the page's claims.
CARD_HARNESS = """
let otherSegments = [], segments = [], suggestedClips = [], chapters = [], view = null;
let hoveredProposal = null;
const proposalVideos = new Map();
function chapterAt() { return null; }
function element(tag) {
  return { tag, innerHTML: '', attrs: {}, classList: { add() {}, remove() {} },
           setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
           addEventListener() {}, appendChild(child) { return child; },
           querySelector: () => element('div'), querySelectorAll: () => [] };
}
const document = { createElement: element };
%s
const [clips, other, mine] = JSON.parse(process.argv[2]);
otherSegments = other; segments = mine; suggestedClips = clips;
process.stdout.write(JSON.stringify(clips.map(clip => {
  const shown = visibleProposal(clip);
  const card = proposalCard(clip, shown);
  return { shown, html: card.innerHTML, video: proposalVideos.has(clip.id) };
})));
"""


def run_proposal_cards(clips: list[dict], other_segments: list[dict],
                       segments: list[dict]) -> list[dict]:
    template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                / "cut.html").read_text()

    def piece(start: str, end: str) -> str:
        return template[template.index(start):template.index(end)]

    script = "\n".join([
        piece("function txFmt", "function currentTxIndex"),
        piece("function escapeHtml", "function highlightQuery"),
        piece("const PROPOSAL_MIN_FRAGMENT", "// The suggestions worth showing right now."),
        piece("// Its own <video>, reused across re-renders",
              "// Nothing waiting and nothing cut yet"),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cards.js"
        path.write_text(CARD_HARNESS % script)
        result = subprocess.run(["node", str(path), json.dumps([clips, other_segments, segments])],
                                capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-2000:])
    return json.loads(result.stdout)


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class ProposalCardJsTest(unittest.TestCase):
    """The critic's verdict is about the clip as rendered: a card that offers
    less (trimmed here as you edit, or already by the server) drops it and says
    it was shortened, and plays in the editor instead of its own player."""

    ENGINE = {"setup": "gemini", "archetype": "voice", "reviewed": True, "stops_scroll": True,
              "stop_reason": "Nye pulls his glasses off in the first second", "bored_at": 75.5,
              "about": "A scientist losing patience", "fix": "End on the stare"}

    def clip(self, cid: int, segments: list[dict], *, trimmed: bool, preview: bool,
             **engine) -> dict:
        return {"id": cid, "story": "Glasses off", "why": "", "segments": segments,
                "seconds": sum(s["end"] - s["start"] for s in segments), "trimmed": trimmed,
                "preview": f"/clip-proposal/{cid}/preview" if preview else "",
                "thumb": "", "flag": "", "opening": "", "closing": "",
                "engine": dict(self.ENGINE, **engine)}

    def test_cards(self):
        whole = [{"start": 10, "end": 30}]
        cases = [
            # (label, clip, other clips' segments, this clip's segments,
            #  verdict shown, shortened note, own player)
            ("as rendered", self.clip(1, whole, trimmed=False, preview=True), [], [],
             True, False, True),
            ("trimmed by the server before the page loaded",
             self.clip(2, [{"start": 10, "end": 25}], trimmed=True, preview=False),
             [{"start": 25, "end": 30}], [], False, True, False),
            ("the server's trim came from this clip's own saved segments",
             self.clip(3, [{"start": 10, "end": 25}], trimmed=True, preview=False),
             [], [{"start": 25, "end": 30}], False, True, False),
            ("trimmed here, by a segment just marked",
             self.clip(4, whole, trimmed=False, preview=True), [], [{"start": 25, "end": 30}],
             False, True, False),
        ]
        for label, clip, other, mine, verdict, note, player in cases:
            with self.subTest(label):
                card = run_proposal_cards([clip], other, mine)[0]
                self.assertEqual("Stops the scroll: Nye pulls his glasses off" in card["html"],
                                 verdict)
                self.assertEqual("Gets slow at 1:15" in card["html"], verdict)
                self.assertEqual("Shortened around footage" in card["html"], note)
                self.assertEqual(card["video"], player)

    def test_a_card_nobody_reviewed_says_so_even_when_trimmed(self):
        clip = self.clip(5, [{"start": 10, "end": 25}], trimmed=True, preview=False,
                         reviewed=False)
        card = run_proposal_cards([clip], [{"start": 25, "end": 30}], [])[0]
        self.assertIn("Not reviewed (critic unavailable)", card["html"])
        self.assertNotIn("Stops the scroll", card["html"])


@unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
class VisibleProposalJsTest(unittest.TestCase):
    """cut.html's visibleProposal must mirror clip_proposals.subtract_play_order:
    claimed time is cut out of each segment WITHOUT re-sorting the clip, so a
    cold open stays first. Runs the template's own functions in node."""

    CASES = [
        # (label, clip segments, other clips' segments, expected segments, trimmed)
        ("cold open kept first",
         [{"start": 50, "end": 54}, {"start": 10, "end": 24}], [],
         [{"start": 50, "end": 54}, {"start": 10, "end": 24}], False),
        ("a split segment keeps its slot and its pieces in time order",
         [{"start": 50, "end": 54}, {"start": 10, "end": 24}], [{"start": 15, "end": 17}],
         [{"start": 50, "end": 54}, {"start": 10, "end": 15}, {"start": 17, "end": 24}], True),
        ("a claimed segment drops out, the rest keep their order",
         [{"start": 50, "end": 58}, {"start": 10, "end": 24}, {"start": 30, "end": 40}],
         [{"start": 9, "end": 25}],
         [{"start": 50, "end": 58}, {"start": 30, "end": 40}], True),
        ("slivers under 1.5 s go",
         [{"start": 10, "end": 30}], [{"start": 11, "end": 29}], [], True),
        ("under 5 s left is no clip",
         [{"start": 10, "end": 20}], [{"start": 14, "end": 30}], [], True),
        ("claims touching only the edges change nothing",
         [{"start": 40, "end": 50}, {"start": 10, "end": 20}],
         [{"start": 0, "end": 10}, {"start": 50, "end": 60}],
         [{"start": 40, "end": 50}, {"start": 10, "end": 20}], False),
        # The engine keeps deliberate 1-1.5 s shots: only a piece a claim CUT
        # is a sliver, so an untouched proposal keeps its preview.
        ("an untouched short segment stays",
         [{"start": 50, "end": 51.2}, {"start": 10, "end": 24}], [{"start": 30, "end": 40}],
         [{"start": 50, "end": 51.2}, {"start": 10, "end": 24}], False),
        ("a short segment a claim cuts into goes",
         [{"start": 50, "end": 51.2}, {"start": 10, "end": 24}], [{"start": 50.5, "end": 60}],
         [{"start": 10, "end": 24}], True),
    ]

    def test_python_agrees_on_every_case(self):
        """The server's visible_segments, on the same cases: the two must not drift."""
        from app import clip_proposals

        for label, clip, claimed, expected, trimmed in self.CASES:
            with self.subTest(label):
                got, was_trimmed = clip_proposals.visible_segments(clip, claimed)
                self.assertEqual([{"start": s["start"], "end": s["end"]} for s in got],
                                 [{"start": float(s["start"]), "end": float(s["end"])}
                                  for s in expected])
                self.assertEqual(was_trimmed, trimmed)

    def test_order_preserving_subtraction(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "web" / "templates"
                    / "cut.html").read_text()
        start = template.index("const PROPOSAL_MIN_FRAGMENT")
        end = template.index("// The suggestions worth showing right now.")
        functions = template[start:end]
        self.assertIn("function visibleProposal", functions)
        harness = functions + """
let otherSegments = [], segments = [];
const cases = JSON.parse(process.argv[2]);
const out = cases.map(([label, clip, claimed]) => {
  otherSegments = claimed;
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
        for (label, _clip, _claimed, expected, trimmed), got in zip(self.CASES,
                                                                    json.loads(result.stdout)):
            with self.subTest(label):
                self.assertEqual(got["segments"], expected)
                self.assertEqual(got["trimmed"], trimmed)


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
        self.assertIs(seen["on_done"], self.main.pagecache.invalidate)

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
        resume.assert_called_once_with(on_done=self.main.pagecache.invalidate)
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
