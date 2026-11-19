"""Route and pipeline tests for the inbox (app/web/inbox.py, and the drafting
pipeline ``_inbox_draft_item`` in app/web/main.py).

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
      .venv/bin/python -m pytest -q tests/test_inbox_routes.py

Skipped unless DATABASE_URL is a throwaway in-memory SQLite (importing
app.web.main initialises that database). Rendering, Whisper and model calls
are stubbed: their own tests cover them; these cover the wiring.
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import inbox
from app.caption_check import Verdict
from app.models import (STATUS_ARCHIVED, Base, Candidate, Channel, ClipProposal, Cut, InboxItem,
                        ThreadsPost)


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


def make_engine():
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


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class InboxRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.web import auto_triage as at_page
        from app.web import inbox as inbox_page
        from app.web import main

        self.main = main
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

        for target in (main, inbox_page, at_page):
            self.patch(mock.patch.object(target, "session_scope", scope))
        self.patch(mock.patch.object(main, "SessionLocal", self.Session))
        self.patch(mock.patch.object(main, "_after_background_work"))
        self.patch(mock.patch("app.clip_engine.sidecars.load", return_value=None))
        self.tmp = Path(tempfile.mkdtemp())
        with self.Session() as s:
            ch = Channel(call_sign="KXYZ", url="https://example.com/c", market="Test")
            s.add(ch)
            s.flush()
            c = Candidate(video_id="ibr0001", channel_pk=ch.id, title="Flood video",
                          url="https://example.com/v", status=STATUS_ARCHIVED)
            s.add(c)
            s.flush()
            row = ClipProposal(candidate_pk=c.id, proposed_segments='[{"start": 5, "end": 30}]',
                               confidence=0.9, story="Rescue on the river")
            s.add(row)
            s.commit()
            self.candidate_pk, self.proposal_pk = c.id, row.id
        self.client = TestClient(main.app)

    def patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def drafting_item(self, proposal_pk=None) -> int:
        with self.Session() as s:
            item = InboxItem(candidate_pk=self.candidate_pk,
                             proposal_pk=proposal_pk or self.proposal_pk, score=0.9)
            s.add(item)
            s.commit()
            return item.id

    def fake_export(self, cut_id: int) -> None:
        out = self.tmp / f"cut{cut_id}.mp4"
        out.write_bytes(b"clip")
        with self.Session() as s:
            cut = s.get(Cut, cut_id)
            cut.trimmed_clip_path, cut.export_status = str(out), ""
            s.commit()

    def fake_subs(self, cut_id: int, position: str = "", force: bool = False) -> str:
        out = self.tmp / f"cut{cut_id}_subs.mp4"
        out.write_bytes(b"subs")
        with self.Session() as s:
            s.get(Cut, cut_id).subtitled_clip_path = str(out)
            s.commit()
        return str(out)

    def make_ready(self) -> tuple[int, int]:
        item_id = self.drafting_item()
        drafts = iter([{"caption": "first draft", "proposal_id": 11},
                       {"caption": "second draft", "proposal_id": 12}])
        verdicts = iter([Verdict(ok=False, reason="overclaims"), Verdict(ok=True)])
        with mock.patch.object(self.main, "_export_cut_in_thread", side_effect=self.fake_export), \
                mock.patch.object(self.main, "_burn_subtitles", side_effect=self.fake_subs), \
                mock.patch.object(self.main, "_draft_caption_for_cut",
                                  side_effect=lambda *a: next(drafts)), \
                mock.patch("app.caption_check.run", side_effect=lambda *a, **k: next(verdicts)):
            self.main._inbox_draft_item(item_id)
        with self.Session() as s:
            return item_id, s.get(InboxItem, item_id).cut_pk

    def test_drafting_makes_a_ready_clip_with_a_checked_caption(self):
        item_id, cut_id = self.make_ready()
        with self.Session() as s:
            item = s.get(InboxItem, item_id)
            self.assertEqual((item.state, item.caption, item.caption_proposal_pk, item.caption_ok),
                             (InboxItem.STATE_READY, "second draft", 12, True))
            self.assertTrue(item.use_subtitles)
            self.assertEqual(item.attempts, 1)
            row = s.get(ClipProposal, self.proposal_pk)
            self.assertEqual((row.verdict, row.cut_pk), ("accepted", cut_id))
            cut = s.get(Cut, cut_id)
            self.assertEqual(json.loads(cut.trim_segments), [{"start": 5, "end": 30}])
            # The AI caption is a draft, never the cut's caption until approved.
            self.assertEqual(cut.draft_caption or "", "")
            self.assertEqual(s.query(ThreadsPost).count(), 0)

    def test_a_duplicate_of_an_existing_clip_fails_without_retry(self):
        with self.Session() as s:
            s.add(Cut(candidate_pk=self.candidate_pk, trim_segments='[{"start": 5, "end": 30}]'))
            s.commit()
        item_id = self.drafting_item()
        with mock.patch.object(self.main, "_export_cut_in_thread") as export:
            self.main._inbox_draft_item(item_id)
        export.assert_not_called()
        with self.Session() as s:
            item = s.get(InboxItem, item_id)
            self.assertEqual(item.state, InboxItem.STATE_FAILED)
            self.assertIn("Same clip", item.error)

    def test_a_render_failure_is_retried_later(self):
        item_id = self.drafting_item()

        def broken(cut_id):
            with self.Session() as s:
                cut = s.get(Cut, cut_id)
                cut.export_status, cut.export_error = "failed", "ffmpeg exploded"
                s.commit()

        with mock.patch.object(self.main, "_export_cut_in_thread", side_effect=broken):
            self.main._inbox_draft_item(item_id)
        with self.Session() as s:
            item = s.get(InboxItem, item_id)
            self.assertEqual((item.state, item.error), (InboxItem.STATE_DRAFTING, "ffmpeg exploded"))

    def test_the_page_shows_ready_clips(self):
        self.make_ready()
        r = self.client.get("/inbox")
        self.assertEqual(r.status_code, 200)
        self.assertIn("second draft", r.text)
        self.assertIn("Approve", r.text)

    def test_approve_queues_with_the_edited_caption(self):
        item_id, cut_id = self.make_ready()
        made = {}

        def fake_queue(session, candidate, path, caption, cut=None):
            post = ThreadsPost(candidate_pk=candidate.id, cut_pk=cut.id, caption=caption,
                               status="queued", clip_local_path=path)
            session.add(post)
            session.flush()
            made.update(path=path, caption=caption)
            return post

        with mock.patch("app.publishing.queue_clip", side_effect=fake_queue):
            r = self.client.post(f"/inbox/{item_id}/approve", data={"caption": "Edited by me"},
                                 headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["queued"])
        self.assertEqual(made["caption"], "Edited by me")
        self.assertTrue(made["path"].endswith("_subs.mp4"))
        with self.Session() as s:
            item = s.get(InboxItem, item_id)
            self.assertEqual(item.state, InboxItem.STATE_QUEUED)
            self.assertEqual(s.get(Cut, cut_id).draft_caption, "Edited by me")

    def test_reject_drops_the_render_and_keeps_a_note(self):
        item_id, cut_id = self.make_ready()
        with mock.patch.object(self.main, "_save_clip_note") as note, \
                mock.patch.object(self.main, "_delete_if_unreferenced") as delete:
            r = self.client.post(f"/inbox/{item_id}/reject", data={"tags": "boring", "note": "slow"},
                                 headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 200, r.text)
        note.assert_called_once()
        self.assertEqual(note.call_args.kwargs["tags"], "boring")
        self.assertEqual(note.call_args.kwargs["proposal_id"], self.proposal_pk)
        self.assertEqual(len(delete.call_args.args[1]), 2)   # the plain and the subtitled render
        with self.Session() as s:
            self.assertEqual(s.get(InboxItem, item_id).state, InboxItem.STATE_REJECTED)
            self.assertEqual(s.get(Cut, cut_id).trimmed_clip_path, "")
            self.assertEqual(s.get(ClipProposal, self.proposal_pk).verdict, "dismissed")
        # Twice is refused.
        r = self.client.post(f"/inbox/{item_id}/reject", headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 409)

    def test_sharing_a_jobs_files_never_blocks_the_job(self):
        import threading

        release = threading.Event()
        calls = []

        def slow_push():
            calls.append(1)
            release.wait(5)

        with mock.patch.object(self.main.datasync, "push_now", side_effect=slow_push), \
                mock.patch.object(self.main.pagecache, "invalidate"):
            self.main._push_soon()
            self.main._push_soon()     # one already waiting: no second thread
            for _ in range(50):
                if calls:
                    break
                threading.Event().wait(0.01)
            self.assertEqual(len(calls), 1)
            release.set()
            for _ in range(100):
                if not self.main._push_pending:
                    break
                threading.Event().wait(0.01)
            self.assertFalse(self.main._push_pending)

    def test_settings_switch(self):
        r = self.client.post("/inbox/settings",
                             data={"enabled": "on", "per_day": "5", "min_score": "0.7",
                                   "per_video": "2", "source": "auto", "subtitles": "on"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("The%20inbox%20is%20on", r.headers["location"])
        with self.Session() as s:
            self.assertTrue(inbox.load_policy(s).enabled)
        page = self.client.get("/auto-triage")
        self.assertIn('id="inbox"', page.text)


if __name__ == "__main__":
    unittest.main()
