"""Route tests for the Automatic triage page (app/web/auto_triage.py) and the
main.py routes it leans on: the switch, Run now, undo, spot-check verdicts,
approving and restoring over an automatic rejection, and the video page's
badge.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
      .venv/bin/python -m pytest -q tests/test_auto_triage_routes.py

Importing app.web.main initialises whatever database DATABASE_URL names and
starts background threads, so every test here is skipped unless that is a
throwaway in-memory SQLite. Routes read a StaticPool engine patched in as
``session_scope``; downloads and retraining are stubbed out.
"""
from __future__ import annotations

import contextlib
import json
import unittest
from unittest import mock

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import auto_triage, triage_model
from app.models import (STATUS_APPROVED, STATUS_NEW, STATUS_REJECTED, AutoTriageAction,
                        AutoTriageRun, Base, Candidate, Channel, TriageDecision, TriageModel,
                        utcnow)


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
class AutoTriageRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.web import auto_triage as page
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

        for target in (main, page):
            self.patch(mock.patch.object(target, "session_scope", scope))
        self.patch(mock.patch.object(main, "SessionLocal", self.Session))
        self.downloads = self.patch(mock.patch.object(main, "_in_background"))
        self.patch(mock.patch.object(main, "_retrain_triage_model_in_background"))
        self.patch(mock.patch.object(page, "_retrain_in_background"))
        triage_model._cache = None
        self.addCleanup(setattr, triage_model, "_cache", None)

        with self.Session() as s:
            channel = Channel(call_sign="KXYZ", url="https://example.com/c", market="Test")
            model = TriageModel(feature_version=triage_model.FEATURE_VERSION,
                                n_decisions=1000, n_approved=100,
                                backtest=json.dumps({"n": 100, "approved": 10, "auc": 0.8,
                                                     "handoff": []}),
                                params=json.dumps({"idf": {}, "weights": {}}))
            s.add_all([channel, model])
            s.commit()
            self.channel_pk, self.model_pk = channel.id, model.id
        self._n = 0
        self.client = TestClient(main.app)

    def patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def video(self, prob: float, status: str = STATUS_NEW) -> int:
        self._n += 1
        with self.Session() as s:
            c = Candidate(video_id=f"rt{self._n:04d}", channel_pk=self.channel_pk,
                          title=f"Route video {self._n}", url="https://example.com/v",
                          status=status, approve_prob=prob, approve_model_pk=self.model_pk)
            s.add(c)
            s.commit()
            return c.id

    def status(self, pk: int) -> str:
        with self.Session() as s:
            return s.get(Candidate, pk).status

    def spot_check(self, prob: float = 0.01) -> tuple[int, int]:
        """A rejected video drawn for a spot-check: (candidate id, action id)."""
        pk = self.video(prob, STATUS_REJECTED)
        with self.Session() as s:
            run = AutoTriageRun(stage=auto_triage.STAGE, trigger="test")
            s.add(run)
            s.flush()
            action = AutoTriageAction(run_pk=run.id, candidate_pk=pk, video_id=f"rt{self._n:04d}",
                                      stage=auto_triage.STAGE, approve_prob=prob,
                                      model_pk=self.model_pk, reject_below=0.02,
                                      audit=True, audit_rate=0.1)
            s.add(action)
            s.commit()
            return pk, action.id

    def switch_on(self):
        r = self.client.post("/auto-triage/settings",
                             data={"enabled": "on", "reject_below": "0.02", "audit_rate": "0.1"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("Automatic%20triage%20is%20on", r.headers["location"])

    def test_page_renders_while_off(self):
        self.video(0.01)
        r = self.client.get("/auto-triage")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Automation", r.text)
        self.assertIn("Your past decisions", r.text)
        self.assertIn("Brand fit", r.text)
        self.assertIn(">Off<", r.text)

    def test_switching_to_positioning_fit(self):
        r = self.client.post("/auto-triage/settings",
                             data={"enabled": "on", "signal": "fit", "fit_below": "0.5",
                                   "reject_below": "0.02", "audit_rate": "0.1"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("positioning%20fit%20under%200.5", r.headers["location"])
        with self.Session() as s:
            policy = auto_triage.load_policy(s)
            self.assertEqual((policy.signal, policy.cutoff), ("fit", 0.5))
        page = self.client.get("/auto-triage")
        self.assertEqual(page.status_code, 200)
        self.assertIn('value="fit" checked', page.text)

    def test_switch_on_run_now_then_undo(self):
        low = [self.video(0.01) for _ in range(3)]
        high = [self.video(0.3) for _ in range(2)]
        self.switch_on()
        page = self.client.get("/auto-triage")
        self.assertIn("would skip 3", page.text)

        r = self.client.post("/auto-triage/run", headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual((body["outcome"], body["rejected"], body["considered"]), ("done", 3, 5))
        self.assertEqual({self.status(pk) for pk in low}, {STATUS_REJECTED})
        self.assertEqual({self.status(pk) for pk in high}, {STATUS_NEW})

        page = self.client.get("/auto-triage")
        self.assertIn("Route video 1", page.text)
        self.assertIn("Undo", page.text)

        r = self.client.post(f"/auto-triage/runs/{body['run_pk']}/undo", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual({self.status(pk) for pk in low}, {STATUS_NEW})

    def test_settings_reject_a_cutoff_that_is_not_offered(self):
        r = self.client.post("/auto-triage/settings",
                             data={"enabled": "on", "reject_below": "0.3", "audit_rate": "0.1"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("cut-off%20must%20be", r.headers["location"])
        with self.Session() as s:
            self.assertEqual(auto_triage.load_policy(s).state, "off")

    def test_agreeing_with_a_spot_check_logs_a_flagged_reject(self):
        pk, action_id = self.spot_check()
        r = self.client.post(f"/auto-triage/spot-checks/{action_id}/agree")
        self.assertEqual(r.status_code, 200, r.text)
        with self.Session() as s:
            decision = s.execute(select(TriageDecision)).scalar_one()
            self.assertEqual((decision.candidate_pk, decision.action, decision.audit),
                             (pk, "reject", True))
            self.assertEqual(s.get(AutoTriageAction, action_id).audit_verdict,
                             AutoTriageAction.AUDIT_AGREE)
        self.assertEqual(self.status(pk), STATUS_REJECTED)
        # A second click is harmless.
        r = self.client.post(f"/auto-triage/spot-checks/{action_id}/agree")
        self.assertTrue(r.json().get("already"))

    def test_approving_a_spot_check_goes_through_the_approve_gate(self):
        pk, action_id = self.spot_check()
        r = self.client.post(f"/video/{pk}/approve", headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.status(pk), STATUS_APPROVED)
        self.downloads.assert_called_once()
        with self.Session() as s:
            action = s.get(AutoTriageAction, action_id)
            self.assertEqual((action.audit_verdict, action.undone, action.undone_reason),
                             (AutoTriageAction.AUDIT_APPROVE, True, "approved"))
            self.assertTrue(s.execute(select(TriageDecision)).scalar_one().audit)
        # It's no longer a spot-check to answer.
        r = self.client.post(f"/auto-triage/spot-checks/{action_id}/agree")
        self.assertEqual(r.status_code, 409)

    def test_back_to_queue_restores_and_ends_the_rejection(self):
        pk, action_id = self.spot_check()
        r = self.client.post(f"/video/{pk}/unreject", headers={"Accept": "application/json"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], STATUS_NEW)
        self.assertEqual(self.status(pk), STATUS_NEW)
        with self.Session() as s:
            action = s.get(AutoTriageAction, action_id)
            self.assertEqual((action.undone, action.undone_reason), (True, "restored"))

    def test_automatic_approval_section_and_switch(self):
        from app import auto_select

        page = self.client.get("/auto-triage")
        self.assertIn("Approve promising videos", page.text)
        r = self.client.post("/auto-select/settings",
                             data={"enabled": "on", "per_day": "5", "min_prob": "0.05",
                                   "fit_floor": "0.2", "explore": "0.1", "fresh_days": "2"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("Automatic%20approval%20is%20on", r.headers["location"])
        with self.Session() as s:
            policy = auto_select.load_policy(s)
            self.assertEqual((policy.enabled, policy.per_day), (True, 5))
        r = self.client.post("/auto-select/settings", data={"per_day": "11"},
                             follow_redirects=False)
        self.assertIn("per%20day%20must", r.headers["location"])

    def test_home_worker_downloads_approved_videos_one_at_a_time(self):
        first = self.video(0.5, STATUS_APPROVED)
        second = self.video(0.5, STATUS_APPROVED)
        busy = self.video(0.5, STATUS_APPROVED)
        self.video(0.5, STATUS_NEW)
        order = []
        self.main._scrape_inflight.add(busy)
        self.addCleanup(self.main._scrape_inflight.discard, busy)
        with mock.patch.object(self.main, "_scrape_in_thread", side_effect=order.append), \
                mock.patch.object(self.main, "_after_background_work"), \
                mock.patch.object(self.main, "_inbox_tick", return_value={"made": 1, "queued": 0}):
            out = self.main._home_worker_tick()
        self.assertEqual(order, [first, second])
        self.assertEqual(out, {"approved": 0, "downloads": 2, "inbox_made": 1, "inbox_queued": 0})

    def test_video_page_names_an_automatic_rejection(self):
        pk, _ = self.spot_check(prob=0.012)
        r = self.client.get(f"/video/{pk}")
        self.assertEqual(r.status_code, 200)
        self.assertIn("rejected automatically · 1.2%", r.text)
        manual = self.video(0.2, STATUS_REJECTED)
        r = self.client.get(f"/video/{manual}")
        self.assertNotIn("rejected automatically", r.text)


if __name__ == "__main__":
    unittest.main()
