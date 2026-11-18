"""Tests for the clip engine's background runner (app/clip_engine/jobs.py) and
the archive-time hand-off in app/scrape.py.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
      .venv/bin/python -m pytest -q tests/test_clip_engine_jobs.py

Everything the runner talks to is mocked — the engine, the transcript
suggester, media, offload, taste, clip_context and the spend ledger — so no
network, ffmpeg or model call ever happens. The database is an in-memory
SQLite shared across threads (StaticPool), patched in as ``jobs.session_scope``.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: no flock, no liveness locks
    fcntl = None

from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import clip_context, clip_proposals, offload, scrape, spend
from app.clip_engine import engine as engine_mod
from app.clip_engine import jobs, legacy, media, sidecars, taste, util
from app.config import Settings
from app.models import STATUS_ARCHIVED, STATUS_REJECTED, Base, Candidate, Channel, ClipProposal, Cut, utcnow

ROOT = Path(__file__).resolve().parents[1]
REAL_BUCKET_SPEND = spend.bucket_spend   # before any test mocks it


def make_engine():
    """In-memory SQLite shared across threads, with real transactions.

    pysqlite's default lets a released SAVEPOINT commit the work under it,
    which would hide a missing rollback; explicit BEGIN makes SAVEPOINT and
    rollback behave as they do on Postgres.
    """
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


def make_settings(**clip_engine) -> Settings:
    block = {"setup": "gemini", "workers": 1, "daily_usd": 5.0, "max_source_seconds": 3600,
             "job_timeout_seconds": 1800, "min_free_gb": 10, "taste_hits": 3, "taste_flops": 2,
             "taste_lookback_days": 180}
    block.update(clip_engine)
    return Settings(raw={"clip_engine": block, "clips": {"section_max_clips": 2}})


class FakeMeter:
    """Stands in for engine.CostMeter (owned by another module)."""

    def __init__(self, bucket: str = "engine") -> None:
        self.bucket = bucket
        self.total = 0.0
        self.by_stage: dict = {}
        self.tokens: dict = {}

    def add(self, stage, usage) -> None:
        self.total += float((usage or {}).get("cost_usd") or 0.0)

    def add_cached(self, stage, cost) -> None:
        self.by_stage[f"{stage}_cached"] = cost


def make_draft(i: int, segments: list[dict], *, score=7.0, story_key=None, render=True,
               critique=None, **extra) -> dict:
    draft = {
        "id": f"d{i}", "moment_id": f"m{i}", "archetype": "voice", "lens": "person",
        "story": f"Story {i}", "story_key": story_key or f"story-{i}", "hook": f"Hook {i}",
        "payoff": f"Payoff {i}", "segments": segments, "why": f"Why {i}", "risk": "",
        "draft_caption": f"Caption {i}", "challenges_rules": "", "origin": "draft",
        "parent": None, "editor_rank": i, "rank": i, "score": score, "credits": "",
        "critique": critique if critique is not None else {
            "stops_scroll": True, "stop_reason": "Nye's face at frame one", "score": 7,
            "error": "", "topic": "central"},
        "render": f"/tmp/renders/d{i}.mp4" if render else "",
    }
    draft.update(extra)
    return draft


def make_result(top, drafts=None, *, outcome="proposals", notes=None, cost=0.42):
    return engine_mod.EngineResult(
        setup="gemini", top=list(top), drafts=list(drafts if drafts is not None else top),
        moments=[], shotlog=None, cost_usd=cost, by_stage={"draft": cost}, tokens={},
        timings={}, notes=list(notes or []), orders={}, model="engine:gemini:gemini-3.8-flash",
        policy="0123456789ab", outcome=outcome)


class JobsTestBase(unittest.TestCase):
    def setUp(self) -> None:
        jobs._reset()
        self.addCleanup(jobs._reset)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.engine_dir = self.dir / "engine"
        self.patch(mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.engine_dir)}))
        self.patch(mock.patch.object(spend, "_LEDGER", self.dir / "llm_spend.json"))

        self.db = make_engine()
        self.addCleanup(self.db.dispose)
        self.Session = sessionmaker(self.db, future=True, expire_on_commit=False)
        self.scopes: list[tuple[bool, object]] = []

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            session = self.Session()
            self.scopes.append((read_only, session))
            try:
                yield session
                if not read_only:
                    session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        self.patch(mock.patch.object(jobs, "session_scope", scope))
        self.settings = make_settings()
        self.patch(mock.patch.object(jobs, "load_settings", lambda: self.settings))
        self.patch(mock.patch.object(jobs, "brand", lambda: {"name": "Renewables.org"}))
        self.setup_answer = ("gemini", "")
        self.patch(mock.patch.object(jobs, "effective_setup",
                                     lambda settings, requested=None: self.setup_answer))
        self.patch(mock.patch.object(jobs, "_RETRY_DELAY", 0.0))

        self.run_engine = self.patch(mock.patch.object(engine_mod, "run"))
        self.estimate = self.patch(mock.patch.object(engine_mod, "estimate_usd", return_value=0.5))
        self.patch(mock.patch.object(engine_mod, "CostMeter", FakeMeter))
        self.ensure_local = self.patch(mock.patch.object(
            offload, "ensure_local", side_effect=lambda path, key: Path(path)))
        self.link = self.patch(mock.patch.object(
            media, "link_inflight",
            side_effect=lambda source, key: self.engine_dir / "inflight" / f"{key}.mp4"))
        self.release = self.patch(mock.patch.object(media, "release_inflight"))
        self.disk = self.patch(mock.patch.object(media, "disk_free_gb", return_value=100.0))
        self.drop = self.patch(mock.patch.object(media, "drop"))
        self.media_prune = self.patch(mock.patch.object(media, "prune", return_value=0))
        self.probe = self.patch(mock.patch.object(media, "probe_duration", return_value=120.0))
        self.save_sidecar = self.patch(mock.patch.object(sidecars, "save"))
        self.sidecar_prune = self.patch(mock.patch.object(sidecars, "prune", return_value=0))
        self.legacy_suggest = self.patch(mock.patch.object(legacy, "suggest"))
        self.legacy_write = self.patch(mock.patch.object(legacy, "write"))
        self.context_build = self.patch(mock.patch.object(
            clip_context, "build", return_value={"shipped_examples": ["a hit"]}))
        self.select_examples = self.patch(mock.patch.object(
            taste, "select_examples", return_value=[{"post_pk": 7, "label": "hit"}]))
        self.bucket_spend = self.patch(mock.patch.object(
            spend, "bucket_spend", create=True, return_value=0.0))
        self.within_budget = self.patch(mock.patch.object(spend, "within_budget", return_value=True))

    def patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    # ---- fixtures ----

    def add_candidate(self, video_id: str = "vid1", *, status: str = STATUS_ARCHIVED,
                      transcript: bool = True, words: bool = True, archived_at=None,
                      duration: int | None = 120, **fields) -> int:
        with self.Session() as s:
            channel = s.execute(select(Channel)).scalars().first()
            if channel is None:
                channel = Channel(call_sign="TEST", url="https://www.youtube.com/@test")
                s.add(channel)
                s.flush()
            tpath = wpath = ""
            if transcript:
                tpath = str(self.dir / f"{video_id}.json")
                Path(tpath).write_text(json.dumps([{"start": 0.0, "end": 5.0, "text": "hello"},
                                                   {"start": 5.0, "end": 9.0, "text": "world"}]))
            if words:
                wpath = str(self.dir / f"{video_id}_words.json")
                Path(wpath).write_text(json.dumps([{"word": "hello", "start": 0.2, "end": 0.6},
                                                   {"word": "world", "start": 5.1, "end": 5.5}]))
            source = self.dir / f"{video_id}.mp4"
            source.write_bytes(b"\0" * 32)
            c = Candidate(video_id=video_id, channel_pk=channel.id, title=f"Title {video_id}",
                          url=f"https://youtu.be/{video_id}", status=status,
                          archived_at=archived_at or utcnow(), duration_seconds=duration,
                          transcript_path=tpath, word_transcript_path=wpath,
                          local_video_path=str(source), source_object_key=f"src/{video_id}.mp4",
                          **fields)
            s.add(c)
            s.commit()
            return c.id

    def add_proposal(self, cid: int, segments, *, verdict="pending", reason="") -> int:
        with self.Session() as s:
            row = ClipProposal(candidate_pk=cid, proposed_segments=json.dumps(segments),
                               verdict=verdict, dismiss_reason=reason, story="old")
            s.add(row)
            s.commit()
            return row.id

    def add_cut(self, cid: int, trim_segments: str) -> None:
        with self.Session() as s:
            s.add(Cut(candidate_pk=cid, trim_segments=trim_segments))
            s.commit()

    def proposals(self, cid: int) -> list[ClipProposal]:
        with self.Session() as s:
            return list(s.execute(select(ClipProposal).where(ClipProposal.candidate_pk == cid)
                                  .order_by(ClipProposal.id)).scalars().all())

    def candidate(self, cid: int) -> Candidate:
        with self.Session() as s:
            return s.get(Candidate, cid)

    def run_job(self, cid: int, **kwargs) -> dict:
        """Queue one job with no worker thread, then run it on this thread."""
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            queued = jobs.enqueue(cid, **kwargs)
        self.assertTrue(queued["started"], queued)
        _priority, _seq, job = jobs._queue.get_nowait()
        jobs._run(job)
        return jobs.status(cid)

    def legacy_returns(self, clips=None, *, ids=(501,), cost=0.05) -> None:
        clips = clips if clips is not None else [
            {"segments": [{"start": 1.0, "end": 20.0}], "story": "From the transcript"}]
        self.legacy_suggest.return_value = {"clips": clips, "model": "claude-sonnet-5",
                                            "policy": "legacypolicy", "cost_usd": cost}
        self.legacy_write.return_value = list(ids)


# ---- the queue -----------------------------------------------------------------

class QueueTest(JobsTestBase):
    def test_operator_runs_jump_ahead_of_unattended_work(self):
        started, release = threading.Event(), threading.Event()
        order: list[int] = []

        def fake_execute(job):
            order.append(job.candidate_id)
            if len(order) == 1:
                started.set()
                release.wait(5)
            return {"state": "done", "outcome": "proposals", "new": []}

        with mock.patch.object(jobs, "_execute", side_effect=fake_execute):
            jobs.enqueue(1, reason="archive")
            self.assertTrue(started.wait(5), "the first job never started")
            jobs.enqueue(2, reason="resume")
            jobs.enqueue(3, reason="archive")
            jobs.enqueue(4, reason="reroll")
            jobs.enqueue(5, reason="section", window={"start": 0, "end": 30})
            positions = {cid: jobs.status(cid)["queue_position"] for cid in (2, 3, 4, 5)}
            self.assertEqual(jobs.status(1)["state"], "running")
            self.assertEqual(jobs.status(1)["queue_position"], 0)
            release.set()
            deadline = time.monotonic() + 5
            while (len(order) < 5 or jobs.status(2)["state"] != "done") and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual(order, [1, 4, 5, 3, 2])
        # Behind the running job, then by priority and arrival.
        self.assertEqual(positions, {4: 1, 5: 2, 3: 3, 2: 4})

    def test_suggest_on_a_waiting_unattended_run_moves_it_up(self):
        """Approve a batch in triage, open the last video, press Suggest: its
        archive run (or a restart's resume run) moves to the front instead of
        waiting behind every earlier one."""
        started, release = threading.Event(), threading.Event()
        order: list[tuple[int, str]] = []

        def fake_execute(job):
            order.append((job.candidate_id, job.reason))
            if len(order) == 1:
                started.set()
                release.wait(5)
            return {"state": "done", "outcome": "proposals", "new": []}

        with mock.patch.object(jobs, "_execute", side_effect=fake_execute):
            jobs.enqueue(1, reason="archive")
            self.assertTrue(started.wait(5), "the first job never started")
            jobs.enqueue(2, reason="archive")
            jobs.enqueue(3, reason="resume")
            queued = jobs.enqueue(4, reason="archive")
            self.assertEqual(queued["queue_position"], 2)   # behind the running one and 2
            joined = jobs.enqueue(4, reason="reroll")
            resumed = jobs.enqueue(3, reason="reroll")
            positions = {cid: jobs.status(cid)["queue_position"] for cid in (2, 3, 4)}
            release.set()
            deadline = time.monotonic() + 5
            while ((len(order) < 4 or any(jobs.status(c)["state"] != "done" for c in (2, 3, 4)))
                   and time.monotonic() < deadline):
                time.sleep(0.01)
        self.assertEqual((joined["started"], joined["conflict"]), (False, False))
        self.assertEqual((joined["state"], joined["reason"]), ("queued", "reroll"))
        self.assertNotEqual(joined["job_id"], queued["job_id"])
        self.assertEqual(resumed["reason"], "reroll")
        self.assertEqual(positions, {4: 1, 3: 2, 2: 3})
        # Each video ran once, the operator's first; the old queue slots were no-ops.
        self.assertEqual(order, [(1, "archive"), (4, "reroll"), (3, "reroll"), (2, "archive")])

    def test_a_moved_up_run_is_the_operators_not_budget_gated(self):
        ids = [self.add_candidate(f"v{i}") for i in range(3)]
        done = mock.Mock()
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            first = [jobs.enqueue(cid, reason="archive", on_done=done) for cid in ids]
            joined = jobs.enqueue(ids[2], reason="reroll")
        record = jobs._read_record(ids[2])
        self.assertEqual((record["job_id"], record["reason"]), (joined["job_id"], "reroll"))
        self.assertNotEqual(joined["job_id"], first[2]["job_id"])
        # The engine's day is spent ($4.90 + $0.50 > $5.00) and so is the LLM's.
        self.bucket_spend.return_value = 4.9
        self.within_budget.return_value = False
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        order = []
        while not jobs._queue.empty():
            _p, _s, job = jobs._queue.get_nowait()
            order.append((job.candidate_id, job.reason))
            jobs._run(job)
        self.assertEqual(order, [(ids[2], "reroll"), (ids[0], "archive"), (ids[1], "archive"),
                                 (ids[2], "archive")])
        self.run_engine.assert_called_once()
        self.assertEqual(self.run_engine.call_args.kwargs["video_id"], "v2")
        st = jobs.status(ids[2])
        self.assertEqual((st["state"], st["outcome"], st["reason"]),
                         ("done", "proposals", "reroll"))
        self.assertEqual(st["attempts"], 1)
        for cid in ids[:2]:
            self.assertEqual(jobs.status(cid)["outcome"], "budget")
        self.assertEqual(done.call_count, 3)   # the callback came along, and ran once per video

    def test_a_run_already_under_way_is_joined_as_it_is(self):
        started, release = threading.Event(), threading.Event()

        def fake_execute(job):
            started.set()
            release.wait(5)
            return {"state": "done", "outcome": "proposals", "new": []}

        with mock.patch.object(jobs, "_execute", side_effect=fake_execute):
            first = jobs.enqueue(1, reason="archive")
            self.assertTrue(started.wait(5), "the job never started")
            joined = jobs.enqueue(1, reason="reroll")
            release.set()
            deadline = time.monotonic() + 5
            while jobs.status(1)["state"] != "done" and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual((joined["started"], joined["conflict"]), (False, False))
        self.assertEqual((joined["job_id"], joined["reason"], joined["state"]),
                         (first["job_id"], "archive", "running"))
        self.assertTrue(jobs._queue.empty())

    def test_joining_never_moves_a_run_down(self):
        cases = [("reroll", "archive"), ("reroll", "resume"), ("archive", "resume"),
                 ("archive", "archive"), ("reroll", "reroll")]
        for i, (first_reason, second_reason) in enumerate(cases):
            with self.subTest(first=first_reason, second=second_reason):
                cid = 50 + i
                with mock.patch.object(jobs, "_ensure_workers", lambda: None):
                    first = jobs.enqueue(cid, reason=first_reason)
                    joined = jobs.enqueue(cid, reason=second_reason)
                self.assertEqual((joined["job_id"], joined["reason"]),
                                 (first["job_id"], first_reason))
        self.assertEqual(jobs._queue.qsize(), len(cases))

    def test_same_scope_joins_the_run_different_scope_conflicts(self):
        whole, chapter = None, {"start": 60.0, "end": 120.0}
        cases = [
            # (first window, second window, conflict)
            (whole, whole, False),
            (chapter, {"start": 60.004, "end": 120.0}, False),   # same view, rounding
            (whole, chapter, True),
            (chapter, whole, True),
            (chapter, {"start": 90.0, "end": 150.0}, True),
        ]
        for i, (first, second, conflict) in enumerate(cases):
            with self.subTest(first=first, second=second):
                cid = 10 + i
                with mock.patch.object(jobs, "_ensure_workers", lambda: None):
                    a = jobs.enqueue(cid, window=first, reason="section" if first else "reroll")
                    b = jobs.enqueue(cid, window=second, reason="section" if second else "reroll")
                self.assertTrue(a["started"])
                self.assertFalse(b["started"])
                self.assertEqual(b["conflict"], conflict)
                self.assertEqual(b["job_id"], a["job_id"])
                self.assertEqual(b["state"], "queued")

    def test_a_finished_run_makes_way_for_a_new_one(self):
        cid = self.add_candidate()
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        first = self.run_job(cid)
        self.assertEqual(first["state"], "done")
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            again = jobs.enqueue(cid)
        self.assertTrue(again["started"])
        self.assertNotEqual(again["job_id"], first["job_id"])

    def test_bad_arguments_are_refused(self):
        with self.assertRaises(ValueError):
            jobs.enqueue(1, reason="whenever")
        for window in ({"start": 30, "end": 10}, {"start": "x", "end": 4}, {"end": 5},
                       {"start": float("nan"), "end": 5}):
            with self.subTest(window=window), self.assertRaises(ValueError):
                jobs.enqueue(1, window=window, reason="section")

    def test_shutdown_drops_the_queue_and_refuses_new_work(self):
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            jobs.enqueue(1, reason="archive")
            jobs.enqueue(2, reason="archive")
        jobs.shutdown()
        self.assertTrue(jobs._queue.empty())
        self.assertTrue(jobs._cancel.is_set())
        refused = jobs.enqueue(3)
        self.assertFalse(refused["started"])
        self.assertEqual(refused["state"], "interrupted")
        # The dropped jobs' records still say queued: the next boot reports them
        # interrupted and may resume them.
        self.assertEqual(jobs._read_record(1)["state"], "queued")


# ---- status and records ----------------------------------------------------------

class StatusTest(JobsTestBase):
    def test_idle_without_a_record(self):
        st = jobs.status(42)
        self.assertEqual(st["state"], "idle")
        self.assertEqual(st["new"], [])
        self.assertEqual(st["queue_position"], 0)
        self.assertEqual(st["boot_id"], jobs.BOOT_ID)
        self.assertEqual(st["message"], "")

    def test_a_run_from_an_earlier_boot_reads_as_interrupted(self):
        cases = [
            # (recorded state, recorded boot, reported state)
            ("running", "oldboot1", "interrupted"),
            ("queued", "oldboot1", "interrupted"),
            ("done", "oldboot1", "done"),
            ("failed", "oldboot1", "failed"),
            ("running", jobs.BOOT_ID, "running"),
        ]
        for i, (state, boot, expected) in enumerate(cases):
            with self.subTest(state=state, boot=boot):
                cid = 100 + i
                util.atomic_write_json(jobs._record_path(cid), {
                    "candidate_id": cid, "job_id": "abc", "boot_id": boot, "state": state,
                    "stage": "Drafting clips", "attempts": 1, "new": [], "notes": [],
                    "error": "", "outcome": ""})
                st = jobs.status(cid)
                self.assertEqual(st["state"], expected)
                self.assertEqual(st["boot_id"], boot)
                if expected == "interrupted":
                    self.assertEqual(st["error"], jobs.INTERRUPTED_ERROR)
                    self.assertIn("press Suggest again", st["message"])
                    self.assertEqual(st["stage"], "")

    def test_this_boots_entry_wins_over_the_record(self):
        util.atomic_write_json(jobs._record_path(5), {"candidate_id": 5, "job_id": "old",
                                                      "boot_id": "oldboot1", "state": "running"})
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            queued = jobs.enqueue(5)
        st = jobs.status(5)
        self.assertEqual(st["state"], "queued")
        self.assertEqual(st["job_id"], queued["job_id"])

    def test_record_follows_the_run(self):
        cid = self.add_candidate()
        stages: list[str] = []

        def fake_run(**kwargs):
            kwargs["progress"]("Drafting clips")
            stages.append(jobs.status(cid)["stage"])
            stages.append(json.loads(jobs._record_path(cid).read_text())["stage"])
            return make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])

        self.run_engine.side_effect = fake_run
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            jobs.enqueue(cid)
        record = json.loads(jobs._record_path(cid).read_text())
        self.assertEqual(record["state"], "queued")
        self.assertEqual(record["boot_id"], jobs.BOOT_ID)
        _p, _s, job = jobs._queue.get_nowait()
        jobs._run(job)
        self.assertEqual(stages, ["Drafting clips", "Drafting clips"])
        record = json.loads(jobs._record_path(cid).read_text())
        self.assertEqual(record["state"], "done")
        self.assertEqual(record["outcome"], "proposals")
        self.assertEqual(record["attempts"], 1)
        self.assertEqual(record["setup"], "gemini")
        self.assertTrue(record["finished_at"])
        self.assertEqual(sorted(record), sorted(jobs._FIELDS))


# ---- a run, start to finish -----------------------------------------------------------

class EngineRunTest(JobsTestBase):
    def test_engine_receives_plain_inputs_and_a_pinned_source(self):
        cid = self.add_candidate(duration=300)
        self.add_cut(cid, json.dumps([{"start": 100.0, "end": 110.0}]))
        self.add_proposal(cid, [{"start": 40.0, "end": 60.0}], verdict="dismissed", reason="bad_boundaries")
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        self.run_job(cid, window={"start": 0, "end": 200}, topic="Fire — the planet", reason="section")
        kwargs = self.run_engine.call_args.kwargs
        self.assertEqual(kwargs["setup"], "gemini")
        self.assertEqual(kwargs["video_id"], "vid1")
        self.assertEqual(kwargs["duration"], 300)
        self.assertEqual(kwargs["source"], self.engine_dir / "inflight" / "vid1.mp4")
        self.assertEqual(kwargs["window"], {"start": 0.0, "end": 200.0})
        self.assertEqual(kwargs["section_topic"], "Fire — the planet")
        self.assertEqual([w["word"] for w in kwargs["words"]], ["hello", "world"])
        self.assertEqual(len(kwargs["transcript"]), 2)
        self.assertEqual(kwargs["used"], [{"start": 100.0, "end": 110.0}])
        self.assertEqual(kwargs["rejected"], [{"segments": [{"start": 40.0, "end": 60.0}],
                                               "reason": "bad_boundaries"}])
        self.assertEqual(kwargs["context"], {"shipped_examples": ["a hit"]})
        self.assertEqual(kwargs["examples"], [{"post_pk": 7, "label": "hit"}])
        self.assertEqual(kwargs["brand"], {"name": "Renewables.org"})
        self.assertIsNone(kwargs["keep"])
        self.assertEqual(kwargs["meter"].bucket, "engine")
        self.assertIs(kwargs["cancel"], jobs._cancel)
        self.assertGreater(kwargs["deadline"], time.monotonic() + 1700)
        self.link.assert_called_once_with(Path(self.dir / "vid1.mp4"), "vid1")
        self.release.assert_called_once_with(self.engine_dir / "inflight" / "vid1.mp4")
        self.ensure_local.assert_called_once_with(str(self.dir / "vid1.mp4"), "src/vid1.mp4")

    def test_top_picks_become_pending_suggestions_in_play_order(self):
        cid = self.add_candidate()
        old = self.add_proposal(cid, [{"start": 70.0, "end": 90.0}])
        cold_open = [{"start": 50.0, "end": 53.5, "lead": "picture"},
                     {"start": 10.0, "end": 24.0, "lead": "word"}]
        top = [make_draft(1, cold_open, score=7.4, story_key="nye"),
               make_draft(2, [{"start": 30.0, "end": 45.0}], score=6.0, story_key="fonda")]
        loser = make_draft(3, [{"start": 60.0, "end": 75.0}], score=2.0)
        self.run_engine.return_value = make_result(top, drafts=top + [loser], notes=["rank ok"])
        with mock.patch.object(clip_proposals, "log_run", wraps=clip_proposals.log_run) as log_run:
            st = self.run_job(cid)
        self.assertEqual((st["state"], st["outcome"]), ("done", "proposals"))
        self.assertEqual(len(st["new"]), 2)
        clips = log_run.call_args.args[2]
        # Play order kept, the engine-only "lead" stripped.
        self.assertEqual(clips[0]["segments"], [{"start": 50.0, "end": 53.5},
                                                {"start": 10.0, "end": 24.0}])
        self.assertEqual(clips[0]["confidence"], 0.74)
        self.assertEqual(clips[0]["story"], "Story 1")
        self.assertEqual(log_run.call_args.kwargs["model"], "engine:gemini:gemini-3.8-flash")
        self.assertEqual(log_run.call_args.kwargs["policy"], "0123456789ab")
        self.assertIsNone(log_run.call_args.kwargs["window"])
        rows = {r.id: r for r in self.proposals(cid)}
        self.assertEqual(rows[old].verdict, ClipProposal.VERDICT_SUPERSEDED)
        # Stored as played: the cold open stays first.
        self.assertEqual(json.loads(rows[st["new"][0]].proposed_segments),
                         [{"start": 50.0, "end": 53.5}, {"start": 10.0, "end": 24.0}])
        for pid in st["new"]:
            self.assertEqual(rows[pid].verdict, ClipProposal.VERDICT_PENDING)
            self.assertEqual(rows[pid].model, "engine:gemini:gemini-3.8-flash")
            self.assertEqual(rows[pid].policy_version, "0123456789ab")
        # One sidecar per suggestion, with what its preview actually shows.
        saved = {c.args[0]: c.kwargs for c in self.save_sidecar.call_args_list}
        self.assertEqual(set(saved), set(st["new"]))
        first = saved[st["new"][0]]
        self.assertEqual(first["meta"]["segments"], [{"start": 50.0, "end": 53.5},
                                                     {"start": 10.0, "end": 24.0}])
        self.assertEqual(first["meta"]["critique"]["stop_reason"], "Nye's face at frame one")
        self.assertEqual(first["meta"]["setup"], "gemini")
        self.assertEqual(first["meta"]["archetype"], "voice")
        self.assertEqual(first["meta"]["notes"], ["rank ok"])
        self.assertEqual(first["render"], Path("/tmp/renders/d1.mp4"))
        # Only the draft nobody will see is deleted.
        self.drop.assert_called_once_with([Path("/tmp/renders/d3.mp4")])
        self.assertTrue(Path(st["result_path"]).is_file())
        saved_result = json.loads(Path(st["result_path"]).read_text())
        self.assertEqual(saved_result["model"], "engine:gemini:gemini-3.8-flash")
        self.assertEqual(saved_result["job"]["candidate_id"], cid)
        self.assertAlmostEqual(st["cost_usd"], 0.42)

    def test_whole_video_side_effects(self):
        cases = [
            # (story keys in top, hand-set marker, expected potential, expected auto)
            (["nye", "fonda"], None, True, True),
            (["nye", "nye"], None, False, False),
            (["nye", "fonda"], (False, False), True, True),
            (["nye"], (True, False), True, False),        # the operator's own marker stands
        ]
        for i, (keys, marker, potential, auto) in enumerate(cases):
            with self.subTest(keys=keys, marker=marker):
                cid = self.add_candidate(f"side{i}")
                if marker is not None:
                    with self.Session() as s:
                        c = s.get(Candidate, cid)
                        c.multi_clip_potential, c.multi_clip_auto = marker
                        s.commit()
                top = [make_draft(j + 1, [{"start": 10.0 * j + 1, "end": 10.0 * j + 9}],
                                  story_key=key) for j, key in enumerate(keys)]
                self.run_engine.return_value = make_result(top)
                self.run_job(cid)
                c = self.candidate(cid)
                self.assertEqual(c.multi_clip_potential, potential)
                self.assertEqual(c.multi_clip_auto, auto)
                self.assertEqual(c.draft_caption, "Caption 1")

    def test_a_section_pass_leaves_the_video_level_fields_alone(self):
        cid = self.add_candidate(draft_caption="keep me")
        self.add_proposal(cid, [{"start": 400.0, "end": 420.0}])   # outside the window: stays
        top = [make_draft(1, [{"start": 10.0, "end": 20.0}], story_key="a"),
               make_draft(2, [{"start": 30.0, "end": 40.0}], story_key="b")]
        self.run_engine.return_value = make_result(top)
        st = self.run_job(cid, window={"start": 0, "end": 60}, reason="section")
        c = self.candidate(cid)
        self.assertEqual(c.draft_caption, "keep me")
        self.assertFalse(c.multi_clip_potential)
        verdicts = sorted(r.verdict for r in self.proposals(cid))
        self.assertEqual(verdicts, ["pending"] * 3)
        self.assertEqual(len(st["new"]), 2)

    def test_confidence_is_the_critics_score_or_nothing(self):
        cases = [
            (7.4, {}, 0.74),
            (12.0, {}, 1.0),
            (-3.0, {}, 0.0),
            (-1e9, {}, None),                     # the critic's failure score
            (6.0, {"error": "timeout"}, None),    # nobody watched it
            (None, {}, None),
            ("n/a", {}, None),
        ]
        for score, critique, expected in cases:
            with self.subTest(score=score, critique=critique):
                self.assertEqual(jobs._confidence({"score": score, "critique": critique}), expected)

    def test_result_is_on_disk_before_the_database_write(self):
        cid = self.add_candidate()
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        error = OperationalError("INSERT", {}, Exception("server closed the connection"))
        with mock.patch.object(clip_proposals, "log_run", side_effect=error) as log_run:
            st = self.run_job(cid)
        self.assertEqual(log_run.call_count, 2)          # retried once in a fresh session
        self.assertEqual((st["state"], st["outcome"]), ("failed", "failed"))
        self.assertIn("server closed", st["error"])
        saved = json.loads(Path(st["result_path"]).read_text())
        self.assertEqual(saved["top"][0]["id"], "d1")
        self.assertEqual(self.proposals(cid), [])

    def test_one_dropped_connection_is_retried(self):
        cid = self.add_candidate()
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        real = clip_proposals.log_run
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OperationalError("INSERT", {}, Exception("connection reset"))
            return real(*args, **kwargs)

        with mock.patch.object(clip_proposals, "log_run", side_effect=flaky):
            st = self.run_job(cid)
        self.assertEqual(st["outcome"], "proposals")
        self.assertEqual(len(self.proposals(cid)), 1)

    def test_rows_map_to_their_sidecars_when_a_clip_cleans_away(self):
        cid = self.add_candidate()
        top = [make_draft(1, [{"start": 5.0, "end": 5.0}]),        # nothing left: no row
               make_draft(2, [{"start": 30.0, "end": 44.0}])]
        ids = jobs.write_result(cid, make_result(top))
        self.assertEqual(len(ids), 1)
        meta = self.save_sidecar.call_args.kwargs["meta"]
        self.assertEqual(self.save_sidecar.call_args.args[0], ids[0])
        self.assertEqual(meta["draft_id"], "d2")

    def test_write_result_with_nothing_writes_nothing(self):
        cid = self.add_candidate()
        pending = self.add_proposal(cid, [{"start": 1.0, "end": 9.0}])
        with mock.patch.object(clip_proposals, "log_run") as log_run:
            self.assertEqual(jobs.write_result(cid, make_result([])), [])
        log_run.assert_not_called()
        self.assertEqual(self.proposals(cid)[0].id, pending)
        self.assertEqual(self.proposals(cid)[0].verdict, "pending")

    def test_on_done_runs_whatever_happens(self):
        cid = self.add_candidate()
        for side_effect in (None, RuntimeError("boom")):
            with self.subTest(side_effect=side_effect):
                done = mock.Mock()
                self.run_engine.reset_mock()
                self.run_engine.side_effect = side_effect
                self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 9.0}])])
                self.legacy_suggest.side_effect = RuntimeError("legacy down too")
                self.run_job(cid, on_done=done)
                done.assert_called_once_with()
        self.legacy_suggest.side_effect = None

    def test_caches_are_pruned_at_most_once_a_day(self):
        cid = self.add_candidate()
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 9.0}])])
        self.run_job(cid)
        self.run_job(cid)
        self.assertEqual(self.media_prune.call_count, 1)
        self.assertEqual(self.sidecar_prune.call_count, 1)
        marker = self.engine_dir / "runs" / ".pruned"
        old = time.time() - 2 * 86400
        os.utime(marker, (old, old))
        self.run_job(cid)
        self.assertEqual(self.media_prune.call_count, 2)


# ---- empty results and fallbacks ---------------------------------------------------------

class FallbackTest(JobsTestBase):
    def test_off_topic_verdict_writes_nothing_and_keeps_pending_suggestions(self):
        cid = self.add_candidate()
        pending = self.add_proposal(cid, [{"start": 1.0, "end": 12.0}])
        drafts = [make_draft(1, [{"start": 2.0, "end": 20.0}], dropped_reason="off topic")]
        self.run_engine.return_value = make_result([], drafts=drafts, outcome="empty_topic",
                                                   notes=["every draft judged off topic"])
        st = self.run_job(cid)
        self.assertEqual((st["state"], st["outcome"]), ("done", "empty_topic"))
        self.assertEqual(st["error"], "found nothing on topic")
        self.assertIn("nothing on topic", st["message"])
        self.assertIn("every draft judged off topic", st["notes"])
        self.legacy_suggest.assert_not_called()
        rows = self.proposals(cid)
        self.assertEqual([(r.id, r.verdict) for r in rows], [(pending, "pending")])
        self.drop.assert_called_once_with([Path("/tmp/renders/d1.mp4")])

    def test_an_editor_with_nothing_to_draft_writes_nothing(self):
        cid = self.add_candidate()
        pending = self.add_proposal(cid, [{"start": 1.0, "end": 12.0}])
        self.run_engine.return_value = make_result([], drafts=[], outcome="empty_drafts", cost=0.2,
                                                   notes=["The editor found nothing worth drafting."])
        st = self.run_job(cid)
        self.assertEqual((st["state"], st["outcome"]), ("done", "empty"))
        self.assertEqual(st["error"], "found nothing worth drafting")
        self.assertEqual(st["message"], "")      # the page words "found nothing" per scope
        self.assertAlmostEqual(st["cost_usd"], 0.2)
        self.legacy_suggest.assert_not_called()
        self.assertEqual([(r.id, r.verdict) for r in self.proposals(cid)], [(pending, "pending")])

    def test_an_open_gemini_breaker_goes_straight_to_the_transcript_suggester(self):
        """Quota or credit gone: no source restore, no proxies, no engine run
        for a setup that needs Gemini — the reason travels to the page."""
        from app.clip_engine import gemini

        cases = [("gemini", True), ("hybrid", True), ("claude", False)]
        for i, (setup, blocked) in enumerate(cases):
            with self.subTest(setup=setup), \
                    mock.patch.object(gemini, "breaker_reason",
                                      return_value=gemini._CREDITS_DEPLETED):
                cid = self.add_candidate(f"breaker{i}")
                self.setup_answer = (setup, "")
                self.run_engine.reset_mock()
                self.ensure_local.reset_mock()
                self.run_engine.return_value = make_result(
                    [make_draft(1, [{"start": 2.0, "end": 20.0}])])
                self.legacy_returns(ids=[800 + i])
                st = self.run_job(cid)
                if blocked:
                    self.assertEqual((st["state"], st["outcome"]), ("done", "legacy"))
                    self.assertEqual(st["fallback"], gemini._CREDITS_DEPLETED)
                    self.assertIn("add credit at https://ai.studio/projects", st["message"])
                    self.run_engine.assert_not_called()
                    self.ensure_local.assert_not_called()
                else:
                    self.assertEqual((st["state"], st["outcome"]), ("done", "proposals"))
                    self.run_engine.assert_called_once()

    def test_a_run_you_start_tries_gemini_again_after_a_402(self):
        """Credit added after a 402: Suggest goes straight back to Gemini
        instead of waiting out the breaker's 30 minutes (or a restart). An
        archive run still waits, and a day's quota (429) holds for everyone."""
        from app.clip_engine import gemini

        for patcher in (mock.patch.object(gemini, "_breaker_until", 0.0),
                        mock.patch.object(gemini, "_breaker_reason", ""),
                        mock.patch.object(gemini, "_breaker_why", gemini._DAILY_QUOTA),
                        mock.patch.object(gemini, "_breaker_status", 429)):
            self.patch(patcher)
        top = [make_draft(1, [{"start": 2.0, "end": 20.0}])]
        cases = [
            # (HTTP status that opened the breaker, reason, outcome, fallback starts with)
            (402, "archive", "legacy", gemini._CREDITS_DEPLETED),
            (402, "reroll", "proposals", ""),
            (402, "section", "proposals", ""),
            (429, "reroll", "legacy", gemini._DAILY_QUOTA),
        ]
        for i, (http, reason, outcome, fallback) in enumerate(cases):
            with self.subTest(http=http, reason=reason):
                why = gemini._CREDITS_DEPLETED if http == 402 else gemini._DAILY_QUOTA
                with self.assertLogs("clip_engine.gemini", "WARNING"):
                    gemini._open_breaker(why, f"HTTP {http}", http)
                cid = self.add_candidate(f"credit{i}")
                self.run_engine.reset_mock()
                self.run_engine.return_value = make_result(top)
                self.legacy_returns(ids=[900 + i])
                window = {"start": 0, "end": 60} if reason == "section" else None
                st = self.run_job(cid, reason=reason, window=window)
                self.assertEqual(st["outcome"], outcome, st)
                self.assertEqual(self.run_engine.called, outcome == "proposals")
                if fallback:
                    self.assertTrue(st["fallback"].startswith(fallback), st["fallback"])
                    self.assertTrue(gemini.breaker_open())
                else:
                    self.assertFalse(gemini.breaker_open())
        # What an archive run's page says while the credit breaker is open.
        with self.assertLogs("clip_engine.gemini", "WARNING"):
            gemini._open_breaker(gemini._CREDITS_DEPLETED, "HTTP 402", 402)
        self.assertIn("at once when you press Suggest", jobs._provider_blocked("gemini"))

    def test_a_fallback_that_finds_nothing_still_says_why(self):
        """The engine's failure is the news (here: Gemini credit ran out), so
        the page must hear it even when the transcript suggester found nothing."""
        cid = self.add_candidate()
        pending = self.add_proposal(cid, [{"start": 1.0, "end": 12.0}])
        self.run_engine.side_effect = engine_mod.EngineError(
            "the gemini perceiver failed: GeminiQuota: Gemini prepaid credits are depleted "
            "— add credit at https://ai.studio/projects (perceive vid: HTTP 402)")
        self.legacy_returns(clips=[], ids=[])
        st = self.run_job(cid)
        self.assertEqual((st["state"], st["outcome"], st["new"]), ("done", "legacy", []))
        self.assertTrue(st["message"].startswith("The transcript suggester stood in and found "
                                                 "nothing"), st["message"])
        self.assertIn("add credit at https://ai.studio/projects", st["message"])
        self.legacy_write.assert_not_called()
        self.assertEqual([(r.id, r.verdict) for r in self.proposals(cid)], [(pending, "pending")])

    def test_proposals_without_picks_still_leave_suggestions(self):
        cid = self.add_candidate()
        self.run_engine.return_value = make_result([], drafts=[], outcome="proposals", cost=0.2)
        self.legacy_returns(ids=[901, 902], cost=0.05)
        st = self.run_job(cid)
        self.assertEqual((st["state"], st["outcome"]), ("done", "legacy"))
        self.assertEqual(st["new"], [901, 902])
        self.assertIn("no picks", st["fallback"])
        self.assertAlmostEqual(st["cost_usd"], 0.25)
        self.assertIn("transcript suggester", st["message"])

    def test_engine_failures_fall_back(self):
        cases = [
            # Mechanical empties are errors on the engine's side of the contract.
            (engine_mod.EngineError("no draft survived snapping and filtering (length: 9)"),
             "no draft survived snapping"),
            # Drafts the editor wrote but its cleaning dropped, every one.
            (engine_mod.EngineError("drafting failed: DraftsDropped: every one of the editor's "
                                    "10 drafts was dropped (10 too long)"), "10 too long"),
            (engine_mod.EngineError("no draft could be rendered"), "no draft could be rendered"),
            (util.Cancelled("ran out of time"), "ran out of time"),
            (RuntimeError("GeminiQuota: breaker open"), "GeminiQuota"),
        ]
        for i, (exc, expected) in enumerate(cases):
            with self.subTest(exc=exc):
                cid = self.add_candidate(f"fail{i}")
                self.run_engine.side_effect = exc
                self.legacy_returns(ids=[700 + i])
                self.context_build.reset_mock()
                st = self.run_job(cid)
                self.assertEqual((st["state"], st["outcome"]), ("done", "legacy"))
                self.assertIn(expected, st["fallback"])
                self.assertEqual(st["new"], [700 + i])
                kwargs = self.legacy_suggest.call_args.kwargs
                self.assertEqual(kwargs["inputs"]["candidate"]["id"], cid)
                self.assertEqual(len(kwargs["inputs"]["transcript"]), 2)
                self.assertEqual(len(kwargs["inputs"]["words"]), 2)
                # The suggester sees the account context it always has (legacy
                # calibration), built in its own short session.
                self.assertEqual(self.context_build.call_args_list[-1].kwargs, {})
                write = self.legacy_write.call_args
                self.assertEqual(write.args[1].id, cid)
                self.assertEqual(write.kwargs["model"], "claude-sonnet-5")
                self.assertEqual(write.kwargs["policy"], "legacypolicy")
                self.release.assert_called()

    def test_a_failed_run_keeps_its_notes_saves_its_pass_and_drops_its_renders(self):
        notes = ["d3 duplicates d1.", "2 of 10 drafts couldn't be rendered (ffmpeg exploded)."]
        renders = ["/tmp/renders/d1.mp4", "/tmp/renders/d2.mp4"]
        cases = [
            # (reason, LLM budget left, state, outcome, the record points at)
            ("reroll", True, "done", "legacy", "-legacy.json"),
            ("archive", False, "skipped", "budget", "-error.json"),
        ]
        for i, (reason, llm_ok, state, outcome, points_at) in enumerate(cases):
            with self.subTest(reason=reason, llm_ok=llm_ok):
                cid = self.add_candidate(f"stopped{i}")
                self.drop.reset_mock()
                self.within_budget.return_value = llm_ok
                self.run_engine.side_effect = engine_mod.EngineError(
                    "stopped at $4.10, past the hard cap (clip_engine.hard_cap_usd $4.00)",
                    notes=notes, renders=renders)
                self.legacy_returns(ids=[42])
                st = self.run_job(cid, reason=reason)
                self.assertEqual((st["state"], st["outcome"]), (state, outcome))
                for note in notes:
                    self.assertIn(note, st["notes"])
                self.drop.assert_called_once_with([Path(p) for p in renders])
                [saved] = jobs._runs_dir().glob(f"{cid}-*-error.json")
                failed = json.loads(saved.read_text())
                self.assertEqual(failed["kind"], "engine-error")
                self.assertIn("past the hard cap", failed["error"])
                self.assertEqual(failed["notes"], notes)
                self.assertEqual(failed["job_id"], st["job_id"])
                self.assertTrue(st["result_path"].endswith(points_at), st["result_path"])

    def test_a_run_out_of_time_drops_its_renders_a_shutdown_keeps_them(self):
        """A run stopped for the deadline is over: its renders go. One stopped
        because the dashboard is going down resumes on the next start, which
        renders the same cuts — those stay in the cache for it."""
        for shutdown in (False, True):
            with self.subTest(shutdown=shutdown):
                cid = self.add_candidate(f"late{int(shutdown)}")
                self.drop.reset_mock()

                def stopped(**kwargs):
                    if shutdown:
                        jobs._cancel.set()
                    exc = util.Cancelled("cancelled" if shutdown else "ran out of time")
                    exc.notes, exc.renders = ["d1 duplicates d2."], ["/tmp/renders/d2.mp4"]
                    raise exc

                self.run_engine.side_effect = stopped
                self.legacy_returns(ids=[7])
                st = self.run_job(cid)
                if shutdown:
                    self.assertEqual(st["state"], "interrupted")
                    self.drop.assert_not_called()
                    jobs._cancel.clear()
                else:
                    self.assertEqual((st["outcome"], st["fallback"]),
                                     ("legacy", "the video model ran out of time (1800 s)"))
                    self.assertIn("d1 duplicates d2.", st["notes"])
                    self.drop.assert_called_once_with([Path("/tmp/renders/d2.mp4")])

    def test_shutdown_mid_run_is_interrupted_not_a_fallback(self):
        cid = self.add_candidate()

        def stopped(**kwargs):
            jobs._cancel.set()
            raise util.Cancelled("cancelled")

        self.run_engine.side_effect = stopped
        st = self.run_job(cid)
        self.assertEqual(st["state"], "interrupted")
        self.assertEqual(st["error"], jobs.INTERRUPTED_ERROR)
        self.legacy_suggest.assert_not_called()

    def test_a_typo_in_a_number_setting_reads_as_its_default(self):
        """A hand edit like ``30 min`` must not fail every run outright with
        no fallback, nor leave the source pinned."""
        top = [make_draft(1, [{"start": 2.0, "end": 20.0}])]
        cases = [("job_timeout_seconds", "30 min", "reroll"), ("min_free_gb", "ten", "reroll"),
                 ("daily_usd", "$5", "archive"), ("max_source_seconds", "1h", "archive")]
        for i, (key, value, reason) in enumerate(cases):
            with self.subTest(key=key, value=value):
                cid = self.add_candidate(f"typo{i}")
                self.settings = make_settings(**{key: value})
                for m in (self.run_engine, self.link, self.release):
                    m.reset_mock()
                self.run_engine.return_value = make_result(top)
                st = self.run_job(cid, reason=reason)
                self.assertEqual((st["state"], st["outcome"]), ("done", "proposals"), st)
                self.run_engine.assert_called_once()
                self.assertEqual((self.link.call_count, self.release.call_count), (1, 1))
                if key == "job_timeout_seconds":
                    self.assertGreater(self.run_engine.call_args.kwargs["deadline"],
                                       time.monotonic() + 1700)

    def test_both_suggesters_failing_is_a_failed_run(self):
        cid = self.add_candidate()
        self.run_engine.side_effect = engine_mod.EngineError("perceiver failed")
        self.legacy_suggest.side_effect = RuntimeError("anthropic 529")
        st = self.run_job(cid)
        self.assertEqual((st["state"], st["outcome"]), ("failed", "failed"))
        self.assertIn("perceiver failed", st["error"])
        self.assertIn("anthropic 529", st["error"])
        self.assertIn("Suggestion failed", st["message"])

    def test_no_transcript(self):
        cid = self.add_candidate(transcript=False)
        st = self.run_job(cid)
        self.assertEqual((st["state"], st["outcome"], st["error"]), ("done", "failed", "no transcript"))
        self.run_engine.assert_not_called()
        self.legacy_suggest.assert_not_called()

    def test_missing_source_and_low_disk_fall_back_without_the_engine(self):
        cid = self.add_candidate()
        self.legacy_returns(ids=[11])
        self.ensure_local.side_effect = offload.SourceUnavailable("its cloud copy is gone")
        st = self.run_job(cid)
        self.assertIn("source video is unavailable", st["fallback"])
        self.assertEqual(st["outcome"], "legacy")
        self.ensure_local.side_effect = lambda path, key: Path(path)
        self.disk.return_value = 3.0
        st = self.run_job(cid)
        self.assertIn("3.0 GB of disk free", st["fallback"])
        self.run_engine.assert_not_called()

    def test_an_engine_module_that_fails_to_import_still_leaves_suggestions(self):
        import app.clip_engine as package

        cid = self.add_candidate()
        self.legacy_returns(ids=[44])
        saved = package.engine
        del package.engine
        try:
            with mock.patch.dict(sys.modules, {"app.clip_engine.engine": None}):
                st = self.run_job(cid)
        finally:
            package.engine = saved
        self.assertEqual((st["outcome"], st["new"]), ("legacy", [44]))
        self.assertIn("failed to load", st["fallback"])

    def test_transcript_setup_runs_the_transcript_suggester(self):
        cid = self.add_candidate()
        self.setup_answer = ("transcript", "GEMINI_API_KEY is not set")
        self.legacy_returns(ids=[5])
        st = self.run_job(cid)
        self.assertEqual(st["outcome"], "legacy")
        self.assertEqual(st["fallback"], "GEMINI_API_KEY is not set")
        self.assertEqual(st["setup"], "transcript")
        self.run_engine.assert_not_called()

    def test_unattended_run_for_a_video_that_left_the_workflow_is_skipped(self):
        cid = self.add_candidate(status=STATUS_REJECTED)
        st = self.run_job(cid, reason="archive")
        self.assertEqual(st["state"], "skipped")
        self.run_engine.assert_not_called()

    def test_budget_gates_only_unattended_runs(self):
        top = [make_draft(1, [{"start": 2.0, "end": 20.0}])]
        cases = [
            # (reason, duration, engine spend today, LLM budget left, outcome)
            ("archive", 120, 0.0, True, "proposals"),
            ("archive", 120, 4.8, True, "legacy"),       # 4.8 + 0.5 estimate > 5.0
            ("archive", 120, 4.8, False, "budget"),
            ("resume", 7200, 0.0, True, "legacy"),       # longer than max_source_seconds
            ("resume", None, 4.8, True, "legacy"),       # length probed from the file
            ("reroll", 120, 4.8, False, "proposals"),    # a person asked: not refused
            ("section", 7200, 9.0, False, "proposals"),
        ]
        for i, (reason, duration, spent, llm_ok, outcome) in enumerate(cases):
            with self.subTest(reason=reason, duration=duration, spent=spent, llm_ok=llm_ok):
                cid = self.add_candidate(f"budget{i}", duration=duration)
                self.bucket_spend.return_value = spent
                self.within_budget.return_value = llm_ok
                self.run_engine.reset_mock()
                self.run_engine.return_value = make_result(top)
                self.legacy_returns(ids=[800 + i])
                window = {"start": 0, "end": 100} if reason == "section" else None
                st = self.run_job(cid, reason=reason, window=window)
                self.assertEqual(st["outcome"], outcome)
                self.assertEqual(self.run_engine.called, outcome == "proposals")
                if outcome == "budget":
                    self.assertEqual(st["state"], "skipped")
                    self.assertIn("daily budget", st["error"])
                    self.assertEqual(st["new"], [])

    def test_an_unreadable_ledger_fails_closed(self):
        cid = self.add_candidate()
        self.bucket_spend.side_effect = OSError("ledger locked")
        self.legacy_returns(ids=[3])
        st = self.run_job(cid, reason="archive")
        self.assertEqual(st["outcome"], "legacy")
        self.assertIn("couldn't be checked", st["fallback"])
        self.run_engine.assert_not_called()

    def test_a_damaged_ledger_fails_closed(self):
        """The ledger's own reader takes a file it can't parse for "$0 spent"."""
        self.bucket_spend.side_effect = REAL_BUCKET_SPEND
        ledger = self.dir / "llm_spend.json"
        day = {spend._today(): {"spend_usd": 0.1, "calls": 1, "by_model": {}, "buckets": {
            "engine": {"spend_usd": 4.9, "calls": 9, "by_model": {}}}}}
        self.legacy_returns(ids=[3])
        cases = [
            ("intact, the day spent", json.dumps(day), "daily budget is spent"),
            ("cut short by a hand edit", json.dumps(day)[:-7], "couldn't be checked"),
            ("not text at all", b"\xff\xfe\x00garbage", "couldn't be checked"),
        ]
        for i, (label, content, why) in enumerate(cases):
            with self.subTest(label):
                cid = self.add_candidate(f"ledger{i}")
                if isinstance(content, bytes):
                    ledger.write_bytes(content)
                else:
                    ledger.write_text(content)
                self.run_engine.reset_mock()
                st = self.run_job(cid, reason="archive")
                self.assertEqual(st["outcome"], "legacy")
                self.assertIn(why, st["fallback"])
                self.run_engine.assert_not_called()

    def test_runs_in_progress_hold_their_estimate(self):
        self.bucket_spend.return_value = 4.2
        self.estimate.return_value = 0.7
        settings = make_settings()
        self.assertEqual(jobs._budget_refusal("gemini", settings, 300.0, job_id="a"), "")
        refused = jobs._budget_refusal("gemini", settings, 300.0, job_id="b")
        self.assertIn("daily budget is spent", refused)
        self.assertIn("plus $0.70 held for runs in progress", refused)
        # A run asking again (its length was probed late) isn't counted twice.
        self.assertEqual(jobs._budget_refusal("gemini", settings, 300.0, job_id="a"), "")
        jobs._release_budget("a")
        self.assertEqual(jobs._budget_refusal("gemini", settings, 300.0, job_id="b"), "")

    def test_parallel_workers_cannot_all_pass_on_the_same_figure(self):
        self.settings = make_settings(workers=2)
        first, second = self.add_candidate("par1"), self.add_candidate("par2")
        self.bucket_spend.return_value = 4.2      # room for one more ~$0.70 run, not two
        self.estimate.return_value = 0.7
        self.legacy_returns(ids=[9])
        in_engine, release = threading.Event(), threading.Event()

        def slow_run(**kwargs):
            in_engine.set()
            release.wait(5)
            return make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])

        self.run_engine.side_effect = slow_run
        try:
            jobs.enqueue(first, reason="archive")
            self.assertTrue(in_engine.wait(5), "the first run never reached the engine")
            jobs.enqueue(second, reason="archive")
            deadline = time.monotonic() + 5
            while jobs.status(second)["state"] != "done" and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while jobs.status(first)["state"] != "done" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(jobs.status(first)["outcome"], "proposals")
        second_status = jobs.status(second)
        self.assertEqual(second_status["outcome"], "legacy")
        self.assertIn("held for runs in progress", second_status["fallback"])
        self.assertEqual(self.run_engine.call_count, 1)
        while jobs._reserved and time.monotonic() < deadline:
            time.sleep(0.01)                      # released as each run ends
        self.assertEqual(jobs._reserved, {})


# ---- inputs ----------------------------------------------------------------------

class PrepareInputsTest(JobsTestBase):
    def test_plain_values_from_a_writer_session_that_is_rolled_back(self):
        archived = dt.datetime(2026, 11, 10, 12, 0)
        cid = self.add_candidate(archived_at=archived,
                                 striking_frames=json.dumps([{"t": 12.5, "why": "flames"}]))
        self.add_cut(cid, json.dumps([{"start": 10.0, "end": 20.0}]))
        self.add_proposal(cid, [{"start": 30.0, "end": 40.0}], verdict="dismissed", reason="off_topic")
        self.add_proposal(cid, [{"start": 50.0, "end": 60.0}], verdict="pending")
        self.add_proposal(cid, [{"start": 70.0, "end": 80.0}], verdict="superseded")

        def build(session, candidate, settings, **kwargs):
            candidate.title = "scribbled on"     # must never reach the database
            return {"recently_posted": ["x"]}

        self.context_build.side_effect = build
        inputs = jobs.prepare_inputs(cid)
        self.assertEqual(inputs["candidate"]["video_id"], "vid1")
        self.assertEqual(inputs["candidate"]["archived_at"].tzinfo, dt.timezone.utc)
        self.assertEqual(inputs["used"], [{"start": 10.0, "end": 20.0}])
        self.assertEqual(inputs["rejected"], [{"segments": [{"start": 30.0, "end": 40.0}],
                                               "reason": "off_topic"}])
        self.assertEqual(inputs["striking"], [{"t": 12.5, "why": "flames"}])
        self.assertEqual(inputs["context"], {"recently_posted": ["x"]})
        self.assertEqual(inputs["notes"], [])
        # A writer session (SAVEPOINTs work there), never the AUTOCOMMIT one.
        self.assertEqual([read_only for read_only, _ in self.scopes], [False])
        build_session = self.context_build.call_args.args[0]
        self.assertIs(build_session, self.scopes[0][1])
        self.assertEqual(self.context_build.call_args.kwargs,
                         {"as_of": None, "replay": False, "calibration": "engine"})
        self.assertEqual(self.candidate(cid).title, "Title vid1")

    def test_replay_leaves_out_this_videos_own_verdicts(self):
        cid = self.add_candidate(striking_frames=json.dumps([{"t": 1.0, "why": "x"}]))
        self.add_cut(cid, json.dumps([{"start": 10.0, "end": 20.0}]))
        self.add_proposal(cid, [{"start": 30.0, "end": 40.0}], verdict="dismissed")
        as_of = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
        inputs = jobs.prepare_inputs(cid, as_of=as_of, replay=True)
        self.assertEqual((inputs["used"], inputs["rejected"], inputs["striking"]), ([], [], []))
        self.assertEqual(self.context_build.call_args.kwargs,
                         {"as_of": as_of, "replay": True, "calibration": "none"})
        kwargs = self.select_examples.call_args.kwargs
        self.assertEqual(kwargs["as_of"], as_of)
        self.assertEqual(kwargs["seed"], "vid1")
        self.assertEqual(kwargs["exclude_candidate_pk"], cid)
        self.assertEqual((kwargs["hits"], kwargs["flops"], kwargs["lookback_days"]), (3, 2, 180))

    def test_each_piece_fails_on_its_own(self):
        cid = self.add_candidate()
        self.add_cut(cid, json.dumps([{"start": 10.0, "end": 20.0}]))
        self.context_build.side_effect = RuntimeError("savepoint refused")
        self.select_examples.side_effect = ValueError("no views")
        inputs = jobs.prepare_inputs(cid)
        self.assertEqual(inputs["context"], {})
        self.assertEqual(inputs["examples"], [])
        self.assertEqual(inputs["used"], [{"start": 10.0, "end": 20.0}])
        self.assertEqual(len(inputs["notes"]), 2)
        self.assertIn("account context unavailable", inputs["notes"][0])
        self.assertIn("taste examples unavailable", inputs["notes"][1])

    def test_unknown_video(self):
        inputs = jobs.prepare_inputs(999)
        self.assertIsNone(inputs["candidate"])
        self.assertEqual(inputs["notes"], ["video not found"])


# ---- resuming after a restart ------------------------------------------------------------

class ResumeTest(JobsTestBase):
    def record(self, cid: int, **fields) -> None:
        util.atomic_write_json(jobs._record_path(cid), {"candidate_id": cid, "job_id": "old",
                                                        "attempts": 1, **fields})

    def test_selection(self):
        hours_ago = lambda h: utcnow() - dt.timedelta(hours=h)   # noqa: E731
        fresh = self.add_candidate("fresh", archived_at=hours_ago(2))
        has_pending = self.add_candidate("pending", archived_at=hours_ago(2))
        self.add_proposal(has_pending, [{"start": 1.0, "end": 9.0}])
        # Suggested before the engine ran here (no record) and dealt with since.
        dismissed_only = self.add_candidate("dismissed", archived_at=hours_ago(2))
        self.add_proposal(dismissed_only, [{"start": 1.0, "end": 9.0}], verdict="dismissed")
        superseded_only = self.add_candidate("superseded", archived_at=hours_ago(2))
        self.add_proposal(superseded_only, [{"start": 1.0, "end": 9.0}], verdict="superseded")
        accepted_cleared = self.add_candidate("accepted", archived_at=hours_ago(2))
        self.add_proposal(accepted_cleared, [{"start": 1.0, "end": 9.0}], verdict="accepted")
        self.add_cut(accepted_cleared, "[]")
        # ...but a run an earlier boot left unfinished is resumed whatever
        # the video's old suggestions say.
        killed_after_triage = self.add_candidate("killedtriaged", archived_at=hours_ago(2))
        self.add_proposal(killed_after_triage, [{"start": 1.0, "end": 9.0}], verdict="dismissed")
        self.record(killed_after_triage, boot_id="oldboot1", state="queued", attempts=0)
        empty_cut = self.add_candidate("emptycut", archived_at=hours_ago(2))
        self.add_cut(empty_cut, "")
        self.add_cut(empty_cut, "[]")
        clipped = self.add_candidate("clipped", archived_at=hours_ago(2))
        self.add_cut(clipped, json.dumps([{"start": 1.0, "end": 9.0}]))
        old = self.add_candidate("old", archived_at=hours_ago(30))
        no_transcript = self.add_candidate("notx", archived_at=hours_ago(2), transcript=False)
        not_archived = self.add_candidate("approved", archived_at=hours_ago(2), status="approved")
        killed = self.add_candidate("killed", archived_at=hours_ago(2))
        self.record(killed, boot_id="oldboot1", state="running", attempts=1)
        killed_twice = self.add_candidate("killed2", archived_at=hours_ago(2))
        self.record(killed_twice, boot_id="oldboot1", state="running", attempts=2)
        stopped = self.add_candidate("stopped", archived_at=hours_ago(2))
        self.record(stopped, boot_id="oldboot1", state="interrupted", attempts=1)
        found_nothing = self.add_candidate("nothing", archived_at=hours_ago(2))
        self.record(found_nothing, boot_id="oldboot1", state="done", outcome="empty_topic")
        over_budget = self.add_candidate("budget", archived_at=hours_ago(2))
        self.record(over_budget, boot_id="oldboot1", state="skipped", outcome="budget")
        this_boot = self.add_candidate("thisboot", archived_at=hours_ago(2))
        self.record(this_boot, boot_id=jobs.BOOT_ID, state="running")

        done = mock.Mock()
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            count = jobs.resume_recent(hours=24, on_done=done)
        queued = {job.candidate_id: job for _p, _s, job in list(jobs._queue.queue)}
        self.assertEqual(set(queued), {fresh, empty_cut, killed, stopped, killed_after_triage})
        self.assertEqual(count, 5)
        self.assertTrue(all(job.reason == "resume" and job.window is None
                            and job.on_done is done for job in queued.values()))
        # The attempt count carries over from the run a restart killed.
        self.assertEqual(jobs.status(killed)["attempts"], 1)
        self.assertEqual(jobs.status(fresh)["attempts"], 0)
        for skipped in (has_pending, dismissed_only, superseded_only, accepted_cleared, clipped,
                        old, no_transcript, not_archived, killed_twice, found_nothing,
                        over_budget, this_boot):
            self.assertNotIn(skipped, queued)

    def test_nothing_when_the_transcript_suggester_is_in_charge(self):
        self.add_candidate()
        self.setup_answer = ("transcript", "GEMINI_API_KEY is not set")
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            self.assertEqual(jobs.resume_recent(), 0)
        self.assertTrue(jobs._queue.empty())

    def test_a_resumed_run_counts_its_attempt(self):
        cid = self.add_candidate()
        self.record(cid, boot_id="oldboot1", state="running", attempts=1)
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 9.0}])])
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            jobs.resume_recent()
        _p, _s, job = jobs._queue.get_nowait()
        jobs._run(job)
        self.assertEqual(jobs.status(cid)["attempts"], 2)


# ---- records another process owns --------------------------------------------------------

# Another process with this engine dir (the dashboard, while a script that
# imports it runs): it queues a run for one video, then holds on until told.
_OTHER_PROCESS = """
import os, sys
assert os.environ["DATABASE_URL"] == "sqlite://"
from app.clip_engine import jobs
jobs._ensure_workers = lambda: None      # queue the run, never start it
jobs.enqueue(int(sys.argv[1]), reason="archive")
print("queued", flush=True)
sys.stdin.read()
"""


@unittest.skipIf(fcntl is None, "liveness locks need flock")
class LivenessTest(JobsTestBase):
    """A queued or running record counts as a dead boot's only once its
    process is gone. Any process that imports the dashboard sees its records;
    one that took a live dashboard's runs for interrupted would report them
    so and run them a second time."""

    def other_process(self, cid: int) -> subprocess.Popen:
        env = dict(os.environ)    # CLIP_ENGINE_DIR is this test's
        env.update({"DATABASE_URL": "sqlite://", "SCHEDULER_EMBEDDED": "false",
                    "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "", "SUPABASE_SERVICE_KEY": "",
                    "CLIPBENCH_DIR": str(self.dir / "bench")})
        proc = subprocess.Popen([sys.executable, "-c", _OTHER_PROCESS, str(cid)], cwd=ROOT,
                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        self.addCleanup(proc.communicate, timeout=60)
        self.addCleanup(proc.kill)
        line = proc.stdout.readline().strip()
        if line != "queued":
            proc.kill()
            self.fail(f"the other process didn't start: {proc.stderr.read()[-3000:]}")
        return proc

    def resume(self) -> set[int]:
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            jobs.resume_recent()
        return {job.candidate_id for _p, _s, job in list(jobs._queue.queue)}

    def test_a_live_process_keeps_its_runs_until_it_is_gone(self):
        cid = self.add_candidate()
        proc = self.other_process(cid)
        record = jobs._read_record(cid)
        self.assertNotEqual(record["boot_id"], jobs.BOOT_ID)
        self.assertEqual(jobs.status(cid)["state"], "queued")
        self.assertFalse(jobs._resumable(record))
        self.assertEqual(self.resume(), set())
        proc.kill()               # however it ends — kill -9 included — its lock goes
        proc.communicate(timeout=60)
        self.assertEqual(jobs.status(cid)["state"], "interrupted")
        self.assertTrue(jobs._resumable(record))
        self.assertEqual(self.resume(), {cid})

    def test_this_process_holds_its_lock_from_its_first_record(self):
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            jobs.enqueue(1, reason="archive")
        lock = jobs._runs_dir() / f".boot-{jobs.BOOT_ID}.lock"
        fd = os.open(lock, os.O_RDONLY)
        try:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        self.assertEqual(lock.read_text().strip(), str(os.getpid()))

    def test_records_without_a_lock_read_as_an_earlier_boots(self):
        """Written before there were locks (or where flock doesn't work)."""
        for boot in ("oldboot1", "../../etc", ""):
            with self.subTest(boot=boot):
                self.assertFalse(jobs._owner_alive({"boot_id": boot, "state": "running"}))
        self.assertTrue(jobs._owner_alive({"boot_id": jobs.BOOT_ID}))

    def test_locks_of_processes_that_are_gone_are_pruned(self):
        runs = jobs._runs_dir()
        runs.mkdir(parents=True, exist_ok=True)
        old = time.time() - 2 * 3600

        def lock(boot: str, *, age: float | None = None) -> Path:
            path = runs / f".boot-{boot}.lock"
            path.write_text("1\n")
            if age is not None:
                os.utime(path, (age, age))
            return path

        dead = lock("deadbeef", age=old)
        fresh = lock("newbeef1")              # its owner may be about to take it
        alive = lock("livebeef", age=old)
        fd = os.open(alive, os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            jobs.enqueue(1, reason="archive")     # takes this process's own lock
        own = runs / f".boot-{jobs.BOOT_ID}.lock"
        os.utime(own, (old, old))
        self.assertEqual(jobs._prune_boot_locks(), 1)
        self.assertFalse(dead.exists())
        for kept in (fresh, alive, own):
            self.assertTrue(kept.exists(), kept.name)


# ---- the archive hand-off (app/scrape.py) ---------------------------------------------------

class ArchiveHandOffTest(JobsTestBase):
    def archive(self, engine_setup: str) -> mock.Mock:
        with self.Session() as s:
            channel = Channel(call_sign="UP", url="upload://local")
            s.add(channel)
            s.flush()
            upload = self.dir / "upload.mp4"
            upload.write_bytes(b"\0" * 16)
            c = Candidate(video_id=f"up-{engine_setup}", channel_pk=channel.id, title="Upload",
                          url=f"upload://up-{engine_setup}", status="approved",
                          local_video_path=str(upload))
            s.add(c)
            s.flush()
            segments = [{"start": 0.0, "end": 4.0, "text": "hello"}]
            words = [{"word": "hello", "start": 0.1, "end": 0.5}]
            with mock.patch.object(scrape, "load_settings", return_value=make_settings()), \
                    mock.patch.object(scrape, "_paths_for", return_value=(self.dir, self.dir)), \
                    mock.patch.object(scrape, "_has_audio_stream", return_value=True), \
                    mock.patch.object(scrape, "transcribe_local", return_value=(segments, words)), \
                    mock.patch("app.clip_engine.effective_setup",
                               return_value=(engine_setup, "")), \
                    mock.patch.object(scrape.clip_proposals, "propose") as propose:
                scrape.archive_candidate(s, c)
            self.assertEqual(c.status, STATUS_ARCHIVED)
            self.assertTrue(c.transcript_path)
            return propose

    def test_the_engine_suggests_after_archiving_instead_of_inline(self):
        self.archive("gemini").assert_not_called()

    def test_the_transcript_suggester_still_runs_inline(self):
        self.archive("transcript").assert_called_once()

    def test_an_unreadable_engine_setup_keeps_the_inline_suggester(self):
        with mock.patch("app.clip_engine.effective_setup", side_effect=RuntimeError("bad yaml")):
            self.assertFalse(scrape._clip_engine_suggests(make_settings()))



# ---- operator advice (round 2) ----------------------------------------------------------------

class AdviceTest(JobsTestBase):
    def setUp(self) -> None:
        super().setUp()
        from app.clip_engine import notes

        self.tidy = self.patch(mock.patch.object(
            notes, "tidy_advice", side_effect=lambda text, **kw: "Tidied: " + text))

    def test_advice_is_kept_and_reaches_the_engine_tidied_with_the_original(self):
        cid = self.add_candidate()
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        st = self.run_job(cid, advice="  open on teh floods;\n no interviews ")
        self.assertEqual(st["advice"], "open on teh floods; no interviews")
        self.assertEqual(jobs._read_record(cid)["advice"], "open on teh floods; no interviews")
        self.assertEqual(self.run_engine.call_args.kwargs["advice"],
                         "Tidied: open on teh floods; no interviews\n"
                         "(Operator's words: open on teh floods; no interviews)")
        self.tidy.assert_called_once()
        self.assertEqual(st["outcome"], "proposals")
        saved = json.loads(Path(st["result_path"]).read_text())
        self.assertEqual(saved["job"]["advice"], "open on teh floods; no interviews")
        self.assertEqual(saved["job"]["advice_tidied"], "Tidied: open on teh floods; no interviews")

    def test_no_advice_no_tidy(self):
        cid = self.add_candidate()
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        st = self.run_job(cid)
        self.assertEqual(st["advice"], "")
        self.assertEqual(self.run_engine.call_args.kwargs["advice"], "")
        self.tidy.assert_not_called()

    def test_a_tidy_that_breaks_still_passes_the_words(self):
        cid = self.add_candidate()
        self.tidy.side_effect = RuntimeError("boom")
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        self.run_job(cid, advice="no interviews")
        self.assertEqual(self.run_engine.call_args.kwargs["advice"], "no interviews")

    def test_advice_is_capped(self):
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            st = jobs.enqueue(1, advice="x" * 5000)
        self.assertEqual(len(st["advice"]), jobs.MAX_ADVICE)

    def test_the_transcript_suggester_says_it_ignored_the_advice(self):
        cid = self.add_candidate()
        self.setup_answer = ("transcript", "GEMINI_API_KEY is not set")
        self.legacy_returns(ids=[5])
        st = self.run_job(cid, advice="no interviews")
        self.assertEqual(st["outcome"], "legacy")
        self.assertIn(jobs.ADVICE_IGNORED_NOTE, st["notes"])
        self.assertTrue(st["message"].endswith("Your advice wasn't used."), st["message"])
        self.run_engine.assert_not_called()
        # Nothing found: still said.
        self.legacy_returns(clips=[], ids=[])
        st = self.run_job(cid, advice="no interviews")
        self.assertIn("Your advice wasn't used.", st["message"])
        # A fallback without advice says nothing about it.
        self.legacy_returns(ids=[6])
        st = self.run_job(cid)
        self.assertNotIn(jobs.ADVICE_IGNORED_NOTE, st["notes"])
        self.assertNotIn("advice", st["message"])

    def test_an_engine_failure_falls_back_and_notes_the_advice_too(self):
        cid = self.add_candidate()
        self.run_engine.side_effect = RuntimeError("Gemini 500")
        self.legacy_returns(ids=[5])
        st = self.run_job(cid, advice="no interviews")
        self.assertEqual(st["outcome"], "legacy")
        self.assertIn(jobs.ADVICE_IGNORED_NOTE, st["notes"])

    def test_different_advice_conflicts_like_a_different_window(self):
        cases = [
            # (first reason, first advice, second advice, conflict, advice afterwards)
            ("reroll", "", "no interviews", True, ""),
            ("reroll", "no interviews", "open on fire", True, "no interviews"),
            ("reroll", "no interviews", "  no   interviews ", False, "no interviews"),
            ("reroll", "no interviews", "", False, "no interviews"),   # no advice joins
            ("reroll", "", "", False, ""),
        ]
        for i, (reason, first, second, conflict, after) in enumerate(cases):
            with self.subTest(first=first, second=second):
                cid = 100 + i
                with mock.patch.object(jobs, "_ensure_workers", lambda: None):
                    a = jobs.enqueue(cid, reason=reason, advice=first)
                    b = jobs.enqueue(cid, reason="reroll", advice=second)
                self.assertTrue(a["started"])
                self.assertFalse(b["started"])
                self.assertEqual(b["conflict"], conflict)
                self.assertEqual(b["job_id"], a["job_id"])
                self.assertEqual(jobs.status(cid)["advice"], after)

    def test_a_waiting_unattended_run_takes_the_operators_advice(self):
        cid = self.add_candidate()
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            archived = jobs.enqueue(cid, reason="archive")
            pressed = jobs.enqueue(cid, reason="reroll", advice="no interviews")
        self.assertEqual((pressed["started"], pressed["conflict"]), (False, False))
        self.assertNotEqual(pressed["job_id"], archived["job_id"])
        self.assertEqual((pressed["reason"], pressed["advice"]), ("reroll", "no interviews"))
        self.assertEqual(jobs._read_record(cid)["advice"], "no interviews")
        self.run_engine.return_value = make_result([make_draft(1, [{"start": 2.0, "end": 20.0}])])
        while not jobs._queue.empty():
            _p, _s, job = jobs._queue.get_nowait()
            jobs._run(job)
        self.run_engine.assert_called_once()
        self.assertIn("(Operator's words: no interviews)",
                      self.run_engine.call_args.kwargs["advice"])

    def test_a_running_unattended_run_with_advice_is_a_conflict(self):
        started, release = threading.Event(), threading.Event()

        def fake_execute(job):
            started.set()
            release.wait(5)
            return {"state": "done", "outcome": "proposals", "new": []}

        with mock.patch.object(jobs, "_execute", side_effect=fake_execute):
            first = jobs.enqueue(1, reason="archive")
            self.assertTrue(started.wait(5))
            pressed = jobs.enqueue(1, reason="reroll", advice="no interviews")
            release.set()
            deadline = time.monotonic() + 5
            while jobs.status(1)["state"] != "done" and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertTrue(pressed["conflict"])
        self.assertEqual(pressed["job_id"], first["job_id"])

    def test_unattended_advice_never_outranks(self):
        """An archive run asked again (unattended) with advice can't promote
        itself: only an operator's press can."""
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            jobs.enqueue(5, reason="archive")
            again = jobs.enqueue(5, reason="archive", advice="x")
        self.assertTrue(again["conflict"])

    def test_sidecar_records_the_window_and_language_doubt(self):
        cid = self.add_candidate(duration=300)
        draft = make_draft(1, [{"start": 2.0, "end": 20.0}], language_doubt="mostly Hindi")
        self.run_engine.return_value = make_result([draft])
        self.run_job(cid, window={"start": 0, "end": 200}, reason="section")
        meta = self.save_sidecar.call_args.kwargs["meta"]
        self.assertEqual(meta["window"], {"start": 0.0, "end": 200.0})
        self.assertEqual(meta["language_doubt"], "mostly Hindi")
        self.run_engine.return_value = make_result([make_draft(2, [{"start": 2.0, "end": 20.0}])])
        self.run_job(cid)
        meta = self.save_sidecar.call_args.kwargs["meta"]
        self.assertIsNone(meta["window"])
        self.assertEqual(meta["language_doubt"], "")


if __name__ == "__main__":
    unittest.main()
