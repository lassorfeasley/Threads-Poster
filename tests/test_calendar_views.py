"""The calendar for an admin and for a customer (app/web/access.py): a
customer sees what's scheduled and can move a queued post, but post pages and
placement tooling are admin-only.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= .venv/bin/python -m pytest -q tests/test_calendar_views.py

The calendar opens sessions inside sessions, so the database is a temporary
file (one connection per session), not the single shared in-memory one.
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Candidate, Channel, ThreadsPost, utcnow


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class CalendarViewsTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app import posting_schedule, rerun_review, scheduler
        from app.web import main

        path = Path(tempfile.mkdtemp()) / "cal.db"
        engine = create_engine(f"sqlite:///{path}", future=True)
        Base.metadata.create_all(engine)
        self.addCleanup(engine.dispose)
        Session = sessionmaker(engine, future=True, expire_on_commit=False)

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            session = Session()
            try:
                yield session
                if not read_only:
                    session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        for target in (main, scheduler, posting_schedule, rerun_review):
            patcher = mock.patch.object(target, "session_scope", scope)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch("app.db.session_scope", scope)
        patcher.start()
        self.addCleanup(patcher.stop)
        posting_schedule.invalidate()
        self.addCleanup(posting_schedule.invalidate)
        with Session() as s:
            ch = Channel(call_sign="KXYZ", url="https://example.com/c")
            s.add(ch)
            s.flush()
            c = Candidate(video_id="cal1", channel_pk=ch.id, title="t", url="u")
            s.add(c)
            s.flush()
            post = ThreadsPost(candidate_pk=c.id, caption="Out already", status="published",
                               published_at=utcnow(), permalink="https://www.threads.com/@x/post/abc")
            s.add(post)
            s.commit()
            self.post_id = post.id
        self.client = TestClient(main.app)

    def test_admins_and_customers(self):
        for path in ("/calendar", "/calendar?view=week"):
            with self.subTest(path=path):
                admin = self.client.get(path)
                self.assertEqual(admin.status_code, 200, admin.text[:500])
                self.assertIn(f'href="/post/{self.post_id}"', admin.text)
                self.client.cookies.set("view_as", "customer")
                try:
                    page = self.client.get(path)
                finally:
                    self.client.cookies.delete("view_as")
                self.assertEqual(page.status_code, 200, page.text[:500])
                self.assertNotIn('href="/post/', page.text)     # post pages are admin pages
                self.assertIn('href="https://www.threads.com/@x/post/abc"', page.text)
                self.assertNotIn('class="cal-score-chip"', page.text)


if __name__ == "__main__":
    unittest.main()
