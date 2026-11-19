"""Route tests for the Posting schedule page, the plan's save and the time
zone's (app/web/main.py: GET /schedule, POST /schedule/plan, /schedule/timezone).

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
      .venv/bin/python -m pytest -q tests/test_schedule_routes.py

Skipped unless DATABASE_URL is a throwaway in-memory SQLite.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from unittest import mock

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import posting_plan, posting_schedule
from app.models import Base, Candidate, Channel, PostingSchedule, ThreadsPost


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class ScheduleRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.web import main

        self.main = main
        db = create_engine("sqlite://", future=True, poolclass=StaticPool,
                           connect_args={"check_same_thread": False})

        @event.listens_for(db, "connect")
        def _no_implicit(dbapi_connection, _record):
            dbapi_connection.isolation_level = None

        @event.listens_for(db, "begin")
        def _begin(connection):
            connection.exec_driver_sql("BEGIN")

        Base.metadata.create_all(db)
        self.addCleanup(db.dispose)
        self.Session = sessionmaker(db, future=True, expire_on_commit=False)

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

        for patcher in (mock.patch.object(main, "session_scope", scope),
                        mock.patch.object(posting_schedule, "session_scope", scope),
                        mock.patch("app.db.session_scope", scope),
                        mock.patch.object(main, "recycle_overview", return_value={})):
            patcher.start()
            self.addCleanup(patcher.stop)
        from app import config

        self.config = config
        config._timezone_cache = None
        self.addCleanup(setattr, config, "_timezone_cache", None)
        posting_schedule.invalidate()
        self.addCleanup(posting_schedule.invalidate)
        with self.Session() as s:
            ch = Channel(call_sign="KXYZ", url="https://example.com/c")
            s.add(ch)
            s.flush()
            c = Candidate(video_id="sr1", channel_pk=ch.id, title="t", url="u")
            s.add(c)
            s.flush()
            for _ in range(75):
                s.add(ThreadsPost(candidate_pk=c.id, caption="c", status="published"))
            s.commit()
        self.client = TestClient(main.app)

    def test_page_shows_time_zone_choice_and_rerun_progress(self):
        r = self.client.get("/schedule")
        self.assertEqual(r.status_code, 200, r.text[:500])
        for text in ("Your time zone", "During waking hours", "Around the clock",
                     "custom schedule", "Posts in your archive", "45 more posts until your 5th daily post"):
            self.assertIn(text, r.text)

    def test_choosing_a_mode_writes_the_plan_from_tomorrow(self):
        r = self.client.post("/schedule/plan", data={"mode": "waking"}, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("4%20posts%20a%20day", r.headers["location"])
        with self.Session() as s:
            self.assertEqual(posting_plan.load_mode(s), "waking")
            row = s.execute(select(PostingSchedule)).scalar_one()
            self.assertEqual(json.loads(row.windows), ["07:00", "12:00", "17:00", "22:00"])
            self.assertEqual(json.loads(row.rerun_slots), ["07:00"])
        page = self.client.get("/schedule")
        self.assertIn('value="waking" checked', page.text)
        self.assertNotIn("custom schedule", page.text)

    def test_choosing_a_time_zone_stays_closed_until_the_rollout(self):
        page = self.client.get("/schedule")
        self.assertIn("ask your account manager", page.text)
        self.assertNotIn('id="tz-select"', page.text)
        r = self.client.post("/schedule/timezone", data={"tz": "Europe/Berlin"}, follow_redirects=False)
        self.assertIn("be%20changed%20here%20yet", r.headers["location"])

    def test_a_saved_zone_shows_and_counts_while_choosing_is_closed(self):
        from app.models import AppToken

        with self.Session() as s:
            s.add(AppToken(name=self.config.TIMEZONE_TOKEN, value="Europe/Berlin"))
            s.commit()
        page = self.client.get("/schedule")
        self.assertIn("<strong>Berlin</strong>", page.text)
        self.assertNotIn('id="tz-select"', page.text)

    def test_choosing_a_time_zone_once_open(self):
        patcher = mock.patch.object(self.config, "TIMEZONE_CHOICE_OPEN", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        page = self.client.get("/schedule")
        self.assertIn('id="tz-select"', page.text)
        self.assertIn('<option value="America/New_York" selected>', page.text)
        if shutil.which("node"):
            import re
            import subprocess
            import tempfile

            for script in re.findall(r"<script>(.*?)</script>", page.text, flags=re.S):
                with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
                    f.write(script)
                done = subprocess.run(["node", "--check", f.name], capture_output=True, text=True,
                                      timeout=60)
                self.assertEqual(done.returncode, 0, done.stderr[-1500:])
        r = self.client.post("/schedule/timezone", data={"tz": "Europe/Berlin"}, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("timed%20for%20Berlin", r.headers["location"])
        page = self.client.get("/schedule")
        self.assertIn("<strong>Berlin</strong>", page.text)
        self.assertIn('<option value="Europe/Berlin" selected>', page.text)
        r = self.client.post("/schedule/timezone", data={"tz": "Mars/Olympus"}, follow_redirects=False)
        self.assertIn("isn%27t%20a%20time%20zone", r.headers["location"].replace("'", "%27"))

    def test_an_unknown_mode_is_refused(self):
        r = self.client.post("/schedule/plan", data={"mode": "sometimes"}, follow_redirects=False)
        self.assertIn("mode%20must%20be", r.headers["location"])


if __name__ == "__main__":
    unittest.main()
