"""Tests for automatic approval (app/auto_select.py).

Run with:  python -m unittest tests.test_auto_select -v

In-memory SQLite. The model is a stored TriageModel row; each video's approve
chance and fit are set directly, marked current, so the pool holds exactly
what the test chose. The positioning fingerprint is pinned by mocking the
profile.
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import auto_select, llm, positioning, triage_model
from app.models import (STATUS_APPROVED, STATUS_NEW, STATUS_REJECTED, AutoTriageAction,
                        AutoTriageRun, Base, Candidate, Channel, ClipProposal, TriageDecision,
                        TriageModel)

NY = ZoneInfo("America/New_York")
# 13:30 in New York: 14 hours of the day count toward the pace.
NOW = dt.datetime(2026, 11, 19, 18, 30, tzinfo=dt.timezone.utc)


class _Settings:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class _Base(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        triage_model._cache = None
        patcher = mock.patch.object(llm, "fit_profile", return_value={"topic": "climate"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = _Settings(**{"matching.score_threshold": 0.5})
        self.channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.own = Channel(call_sign="OURS", url="https://example.com/o", first_party=True)
        self.session.add_all([self.channel, self.own])
        self.model = TriageModel(feature_version=triage_model.FEATURE_VERSION, n_decisions=1000,
                                 n_approved=100, backtest=json.dumps({"auc": 0.8}),
                                 params=json.dumps({"idf": {}, "weights": {}}))
        self.session.add(self.model)
        self.session.flush()
        self.basis = positioning.basis(self.settings)
        self._n = 0

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def video(self, prob=0.3, fit=None, relevance=0.8, status=STATUS_NEW, age_days=0.5,
              channel=None, model_pk=-1) -> Candidate:
        self._n += 1
        c = Candidate(video_id=f"sel{self._n:04d}", channel_pk=(channel or self.channel).id,
                      title=f"video {self._n}", url="https://example.com/v", status=status,
                      relevance_score=relevance, approve_prob=prob,
                      approve_model_pk=self.model.id if model_pk == -1 else model_pk,
                      fit_score=fit, fit_basis=self.basis if fit is not None else "",
                      published_at=NOW - dt.timedelta(days=age_days))
        self.session.add(c)
        self.session.flush()
        return c

    def switch_on(self, **kw):
        kw.setdefault("explore", 0.0)
        return auto_select.set_policy(self.session, enabled=True, **kw)

    def do_run(self, now=NOW):
        return auto_select.run(self.session, self.settings, trigger="test", now=now, tz=NY)

    def status(self, c):
        self.session.refresh(c)
        return c.status


class PaceTest(_Base):
    def test_the_day_is_counted_in_the_schedule_time_zone(self):
        start, share = auto_select.day_window(NOW, NY)
        self.assertEqual(start, dt.datetime(2026, 11, 19, 5, 0, tzinfo=dt.timezone.utc))
        self.assertAlmostEqual(share, 14 / 24)

    def test_the_quota_opens_through_the_day(self):
        policy = auto_select.Policy(per_day=10)
        self.assertEqual(auto_select.allowed_now(policy, 0, 1 / 24), 1)
        self.assertEqual(auto_select.allowed_now(policy, 3, 14 / 24), 3)
        self.assertEqual(auto_select.allowed_now(policy, 10, 1.0), 0)
        self.assertEqual(auto_select.allowed_now(policy, 12, 1.0), 0)


class PoolTest(_Base):
    def test_ranks_by_approve_chance_within_the_floors(self):
        policy = self.switch_on(min_prob=0.05, fit_floor=0.2)
        best = self.video(prob=0.6)
        second = self.video(prob=0.4, fit=0.9)
        self.video(prob=0.03)                          # under the chance floor
        self.video(prob=0.9, fit=0.1)                  # current fit under the floor
        self.video(prob=0.9, relevance=0.2)            # hidden from the dashboard
        self.video(prob=0.9, age_days=3)               # older than two days
        self.video(prob=0.9, channel=self.own)         # first-party
        self.video(prob=0.9, status=STATUS_REJECTED)   # not waiting
        self.video(prob=0.9, model_pk=None)            # not scored by this model
        decided = self.video(prob=0.9)
        self.session.add(TriageDecision(candidate_pk=decided.id, action="reject", undone=True))
        touched = self.video(prob=0.9)
        run = AutoTriageRun(stage="metadata", trigger="t")
        self.session.add(run)
        self.session.flush()
        self.session.add(AutoTriageAction(run_pk=run.id, candidate_pk=touched.id, stage="metadata",
                                          undone=True, undone_reason="restored"))
        self.session.flush()
        signal, rows = auto_select.pool(self.session, self.settings, policy, NOW)
        self.assertEqual(signal, "approval")
        self.assertEqual([r[0] for r in rows], [best.id, second.id])

    def test_without_a_fit_model_it_ranks_by_positioning_fit(self):
        self.model.n_decisions = 10   # not fit to act
        self.session.flush()
        policy = self.switch_on(fit_floor=0.3)
        a = self.video(prob=None, fit=0.8, model_pk=None)
        b = self.video(prob=None, fit=0.5, model_pk=None)
        self.video(prob=None, fit=0.2, model_pk=None)
        self.video(prob=None, fit=None, model_pk=None)
        signal, rows = auto_select.pool(self.session, self.settings, policy, NOW)
        self.assertEqual(signal, "fit")
        self.assertEqual([r[0] for r in rows], [a.id, b.id])


class ChooseTest(unittest.TestCase):
    rows = [(i, f"v{i}", 1 - i / 10, None, None, None, "") for i in range(10)]

    def test_no_exploring_takes_the_top(self):
        picks = auto_select.choose(self.rows, 3, auto_select.Policy(explore=0.0),
                                   dt.date(2026, 11, 19), 0)
        self.assertEqual([(r[0], e) for r, e in picks], [(0, False), (1, False), (2, False)])

    def test_exploring_picks_from_below_the_top_and_is_repeatable(self):
        policy = auto_select.Policy(explore=1.0)
        day = dt.date(2026, 11, 19)
        first = auto_select.choose(self.rows, 3, policy, day, 0)
        self.assertEqual(first, auto_select.choose(self.rows, 3, policy, day, 0))
        self.assertTrue(all(e for _, e in first))
        self.assertTrue(all(r[0] >= 1 for r, _ in first))
        self.assertEqual(len({r[0] for r, _ in first}), 3)

    def test_a_short_pool_gives_what_it_has(self):
        picks = auto_select.choose(self.rows[:2], 5, auto_select.Policy(explore=0.0),
                                   dt.date(2026, 11, 19), 0)
        self.assertEqual(len(picks), 2)


class RunTest(_Base):
    def test_off_does_nothing(self):
        c = self.video(prob=0.9)
        self.assertEqual(self.do_run()["approved"], 0)
        self.assertEqual(self.status(c), STATUS_NEW)

    def test_approves_what_the_pace_allows_and_logs_it_apart_from_decisions(self):
        self.switch_on(per_day=5)   # 14/24 of 5 -> 3 by now
        videos = [self.video(prob=0.9 - i / 100) for i in range(6)]
        out = self.do_run()
        self.assertEqual((out["approved"], out["allowed"], out["pool"]), (3, 3, 6))
        self.assertEqual([self.status(c) for c in videos],
                         [STATUS_APPROVED] * 3 + [STATUS_NEW] * 3)
        self.assertIsNotNone(videos[0].approved_at)
        calls = self.session.execute(select(AutoTriageAction)).scalars().all()
        self.assertEqual({(a.stage, a.action, a.signal) for a in calls},
                         {("select", "approve", "approval")})
        self.assertEqual(self.session.get(AutoTriageRun, out["run_pk"]).approved, 3)
        self.assertEqual(self.session.query(TriageDecision).count(), 0)
        # The pace is met: a second run at the same moment approves nothing.
        self.assertEqual(self.do_run()["approved"], 0)
        # Later in the day, more opens up.
        evening = NOW + dt.timedelta(hours=9)
        self.assertEqual(self.do_run(now=evening)["approved"], 2)

    def test_never_fills_the_quota_under_the_floors(self):
        self.switch_on(per_day=10, min_prob=0.10)
        self.video(prob=0.5)
        self.video(prob=0.05)
        self.assertEqual(self.do_run()["approved"], 1)

    def test_a_video_decided_meanwhile_is_left_alone(self):
        self.switch_on(per_day=5)
        c = self.video(prob=0.9)
        policy = auto_select.load_policy(self.session)
        signal, rows = auto_select.pool(self.session, self.settings, policy, NOW)
        c.status = STATUS_REJECTED
        self.session.flush()
        with mock.patch.object(auto_select, "pool", return_value=(signal, rows)):
            out = self.do_run()
        self.assertEqual(out["approved"], 0)
        self.assertEqual(self.status(c), STATUS_REJECTED)

    def test_recent_picks_show_what_came_of_them(self):
        self.switch_on(per_day=5)
        c = self.video(prob=0.9)
        self.do_run()
        self.session.add(ClipProposal(candidate_pk=c.id))
        self.session.add(ClipProposal(candidate_pk=c.id))
        self.session.flush()
        (row,) = auto_select.recent_picks(self.session)
        self.assertEqual((row["c"].id, row["drafted"], row["cuts"], row["posted"]), (c.id, 2, 0, 0))

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            auto_select.set_policy(self.session, per_day=11)
        with self.assertRaises(ValueError):
            auto_select.set_policy(self.session, min_prob=0.5)
        policy = auto_select.set_policy(self.session, per_day="5", fresh_days="3", explore="0.2")
        self.assertEqual((policy.per_day, policy.fresh_days, policy.explore), (5, 3, 0.2))


if __name__ == "__main__":
    unittest.main()
