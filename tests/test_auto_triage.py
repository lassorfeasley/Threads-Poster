"""Tests for automatic triage (app/auto_triage.py).

Run with:  python -m unittest tests.test_auto_triage -v

Runs against a throwaway in-memory SQLite database: every function under test
takes the session it's given. The model is a stored TriageModel row with
empty weights (it scores everything 0.5), and the tests set each waiting
video's score directly, marked as written by that model, so a pass judges
exactly the numbers the test chose.
"""
from __future__ import annotations

import datetime as dt
import json
import unittest

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app import auto_triage, triage_model
from app.models import (STATUS_APPROVED, STATUS_NEW, STATUS_REJECTED, AutoTriageAction,
                        AutoTriageRun, Base, Candidate, Channel, TriageDecision, TriageModel,
                        utcnow)


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
        self.channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.own = Channel(call_sign="OURS", url="https://example.com/ours", first_party=True)
        self.session.add_all([self.channel, self.own])
        self.session.flush()
        self.model = self.add_model()
        self.settings = _Settings()
        self.now = utcnow()
        self._n = 0

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def add_model(self, n_decisions: int = 1000, auc: float = 0.8) -> TriageModel:
        row = TriageModel(feature_version=triage_model.FEATURE_VERSION,
                          n_decisions=n_decisions, n_approved=100,
                          backtest=json.dumps({"n": 100, "approved": 10, "auc": auc,
                                               "handoff": []}),
                          params=json.dumps({"idf": {}, "weights": {}}))
        self.session.add(row)
        self.session.flush()
        triage_model._cache = None
        return row

    def video(self, prob: float | None, status: str = STATUS_NEW, channel=None,
              model_pk: int | None = -1) -> Candidate:
        self._n += 1
        c = Candidate(video_id=f"vid{self._n:05d}", channel_pk=(channel or self.channel).id,
                      title=f"video {self._n}", description="", url="https://example.com/v",
                      status=status, approve_prob=prob,
                      approve_model_pk=self.model.id if model_pk == -1 else model_pk)
        self.session.add(c)
        self.session.flush()
        return c

    def switch_on(self, cutoff: float = 0.02, audit_rate: float = 0.1):
        return auto_triage.set_policy(self.session, enabled=True, reject_below=cutoff,
                                      audit_rate=audit_rate, now=self.now)

    def do_pass(self, **kw) -> dict:
        return auto_triage.run_pass(self.session, self.settings, trigger="test",
                                    now=kw.pop("now", self.now), **kw)

    def calls(self) -> list[AutoTriageAction]:
        return self.session.execute(select(AutoTriageAction)).scalars().all()

    def status(self, c: Candidate) -> str:
        self.session.refresh(c)
        return c.status


class PolicyTest(_Base):
    def test_off_by_default_and_a_pass_does_nothing(self):
        self.assertEqual(auto_triage.load_policy(self.session).state, "off")
        c = self.video(0.001)
        out = self.do_pass()
        self.assertEqual(out["outcome"], "off")
        self.assertEqual(self.status(c), STATUS_NEW)
        self.assertEqual(self.session.query(AutoTriageRun).count(), 0)

    def test_switching_on_clears_a_pause_and_starts_a_fresh_window(self):
        policy = self.switch_on()
        self.assertEqual(policy.state, "on")
        self.assertEqual(policy.watch_since_at, self.now)
        auto_triage._pause(self.session, policy, "test pause", self.now)
        self.assertEqual(auto_triage.load_policy(self.session).state, "paused")
        auto_triage.set_policy(self.session, enabled=False)
        later = self.now + dt.timedelta(hours=1)
        policy = auto_triage.set_policy(self.session, enabled=True, now=later)
        self.assertEqual((policy.state, policy.paused_reason), ("on", ""))
        self.assertEqual(policy.watch_since_at, later)

    def test_resume_clears_the_pause_and_restarts_the_window(self):
        policy = self.switch_on()
        auto_triage._pause(self.session, policy, "test pause", self.now)
        later = self.now + dt.timedelta(days=1)
        policy = auto_triage.resume(self.session, now=later)
        self.assertEqual(policy.state, "on")
        self.assertEqual(policy.watch_since_at, later)

    def test_only_the_offered_cutoffs_and_shares_are_accepted(self):
        with self.assertRaises(ValueError):
            auto_triage.set_policy(self.session, reject_below=0.25)
        with self.assertRaises(ValueError):
            auto_triage.set_policy(self.session, audit_rate=0.5)
        # Form posts send strings.
        policy = auto_triage.set_policy(self.session, reject_below="0.01", audit_rate="0.2")
        self.assertEqual((policy.reject_below, policy.audit_rate), (0.01, 0.2))

    def test_a_new_cutoff_restarts_the_window_a_new_share_does_not(self):
        self.switch_on(cutoff=0.02)
        later = self.now + dt.timedelta(days=2)
        policy = auto_triage.set_policy(self.session, audit_rate=0.2, now=later)
        self.assertEqual(policy.watch_since_at, self.now)
        policy = auto_triage.set_policy(self.session, reject_below=0.01, now=later)
        self.assertEqual(policy.watch_since_at, later)

    def test_unreadable_stored_settings_read_as_off(self):
        from app.models import AppToken

        self.session.add(AppToken(name=auto_triage.POLICY_TOKEN, value="{not json"))
        self.session.flush()
        self.assertEqual(auto_triage.load_policy(self.session).state, "off")


class PassTest(_Base):
    def test_rejects_only_untouched_waiting_videos_under_the_cutoff(self):
        self.switch_on(cutoff=0.02)
        target = self.video(0.01)
        above = self.video(0.03)
        decided_once = self.video(0.005)   # decided, then the decision was undone
        self.session.add(TriageDecision(candidate_pk=decided_once.id, action="reject",
                                        undone=True))
        first_party = self.video(0.001, channel=self.own)
        approved = self.video(0.002, status=STATUS_APPROVED)
        stale = self.video(0.001, model_pk=None)   # never scored by this model
        self.session.flush()

        out = self.do_pass()

        self.assertEqual(out["outcome"], "done")
        self.assertEqual(out["rejected"], 1)
        # Judged: the target, the one above the cut-off, and the stale video
        # once the pass scored it (0.5 with this model's empty weights).
        self.assertEqual(out["considered"], 3)
        self.assertEqual(self.status(target), STATUS_REJECTED)
        for c in (above, decided_once, first_party, stale):
            self.assertEqual(self.status(c), STATUS_NEW)
        self.assertEqual(self.status(approved), STATUS_APPROVED)
        self.assertEqual(stale.approve_prob, 0.5)

        (call,) = self.calls()
        self.assertEqual((call.candidate_pk, call.approve_prob, call.model_pk, call.reject_below),
                         (target.id, 0.01, self.model.id, 0.02))
        run = self.session.get(AutoTriageRun, out["run_pk"])
        self.assertEqual((run.trigger, run.considered, run.rejected, run.outcome),
                         ("test", 3, 1, "done"))

    def test_a_video_gets_one_automatic_call_ever(self):
        self.switch_on()
        c = self.video(0.01)
        self.do_pass()
        auto_triage.record_operator_restore(self.session, c)
        c.status = STATUS_NEW
        self.session.flush()
        out = self.do_pass()
        self.assertEqual(out["rejected"], 0)
        self.assertEqual(self.status(c), STATUS_NEW)

    def test_waits_while_the_model_is_not_fit_to_act(self):
        self.switch_on()
        c = self.video(0.001)
        self.add_model(n_decisions=100)
        c.approve_model_pk = self.session.execute(select(func.max(TriageModel.id))).scalar_one()
        self.session.flush()
        out = self.do_pass()
        self.assertEqual(out["outcome"], AutoTriageRun.OUTCOME_WAITING)
        self.assertIn("needs 500", out["note"])
        self.assertEqual(self.status(c), STATUS_NEW)
        self.add_model(auc=0.6)
        out = self.do_pass()
        self.assertIn("grades 0.60", out["note"])

    def test_waits_without_any_model(self):
        self.session.delete(self.model)
        self.session.flush()
        triage_model._cache = None
        self.switch_on()
        out = self.do_pass()
        self.assertEqual(out["outcome"], AutoTriageRun.OUTCOME_WAITING)
        self.assertIn("No approve predictor", out["note"])

    def test_the_spot_check_sample_is_fixed_by_video_id(self):
        self.settings = _Settings(**{"auto_triage.max_reject_share": 1.0})
        self.switch_on(cutoff=0.02, audit_rate=0.2)
        videos = [self.video(0.001) for _ in range(60)]
        out = self.do_pass()
        expected = {c.video_id for c in videos if auto_triage._audit_draw(c.video_id) < 0.2}
        self.assertEqual(out["audited"], len(expected))
        self.assertEqual({a.video_id for a in self.calls() if a.audit}, expected)
        self.assertTrue(0 < len(expected) < 60)

    def test_a_big_swing_pauses_instead_of_rejecting(self):
        self.switch_on()
        videos = [self.video(0.001) for _ in range(25)]
        out = self.do_pass()
        self.assertEqual(out["outcome"], "paused")
        self.assertIn("25 of the 25", out["note"])
        self.assertTrue(all(self.status(c) == STATUS_NEW for c in videos))
        policy = auto_triage.load_policy(self.session)
        self.assertEqual(policy.state, "paused")
        # While paused, a pass neither acts nor logs another run.
        runs = self.session.query(AutoTriageRun).count()
        self.assertEqual(self.do_pass()["outcome"], "paused")
        self.assertEqual(self.session.query(AutoTriageRun).count(), runs)

    def test_a_small_queue_is_not_judged_by_share(self):
        self.switch_on()
        videos = [self.video(0.001) for _ in range(5)]
        out = self.do_pass()
        self.assertEqual(out["rejected"], 5)
        self.assertTrue(all(self.status(c) == STATUS_REJECTED for c in videos))

    def test_the_claim_skips_videos_someone_else_moved(self):
        a, b = self.video(0.01), self.video(0.01)
        b.status = STATUS_APPROVED
        self.session.flush()
        moved = auto_triage._set_status(self.session, [a.id, b.id], STATUS_NEW, STATUS_REJECTED)
        self.assertEqual(moved, {a.id})
        # The in-memory copy follows the database.
        self.assertEqual(a.status, STATUS_REJECTED)
        self.assertEqual(self.status(b), STATUS_APPROVED)

    def test_automatic_calls_are_never_training_data(self):
        self.switch_on()
        for _ in range(3):
            self.video(0.001)
        self.assertEqual(self.do_pass()["rejected"], 3)
        self.assertEqual(self.session.query(TriageDecision).count(), 0)
        self.assertEqual(triage_model._load_examples(self.session), [])

    def test_preview_counts_each_cutoff_without_writing(self):
        for p in (0.001, 0.012, 0.025, 0.2):
            self.video(p)
        self.video(0.001, model_pk=None)
        out = auto_triage.preview(self.session, self.settings)
        self.assertEqual(out["waiting"], 4)
        self.assertEqual(out["unscored"], 1)
        self.assertEqual(out["below"][0.005], 1)
        self.assertEqual(out["below"][0.02], 2)
        self.assertEqual(out["below"][0.03], 3)
        self.assertEqual(self.calls(), [])


class OperatorTest(_Base):
    def rejected_by_pass(self, audit: bool = False) -> tuple[Candidate, AutoTriageAction]:
        c = self.video(0.01, status=STATUS_REJECTED)
        run = AutoTriageRun(stage=auto_triage.STAGE, trigger="test", started_at=self.now)
        self.session.add(run)
        self.session.flush()
        action = AutoTriageAction(run_pk=run.id, candidate_pk=c.id, video_id=c.video_id,
                                  stage=auto_triage.STAGE, approve_prob=0.01,
                                  model_pk=self.model.id, reject_below=0.02,
                                  audit=audit, audit_rate=0.1, created_at=self.now)
        self.session.add(action)
        self.session.flush()
        return c, action

    def decide(self, c: Candidate, action: str) -> TriageDecision:
        if action == "approve":
            c.status = STATUS_APPROVED
        d = auto_triage.record_operator_decision(self.session, c, action, self.settings,
                                                 now=self.now)
        self.session.flush()
        return d

    def test_approving_overrides_it_and_answers_its_spot_check(self):
        c, action = self.rejected_by_pass(audit=True)
        d = self.decide(c, "approve")
        self.assertTrue(d.audit)
        self.assertEqual((d.action, d.predicted_prob), ("approve", 0.01))
        self.assertEqual(action.audit_verdict, AutoTriageAction.AUDIT_APPROVE)
        self.assertTrue(action.undone)
        self.assertEqual(action.undone_reason, "approved")
        self.assertIsNone(auto_triage.standing_call(self.session, c.id))

    def test_rejecting_a_spot_check_records_agreement_and_it_stands(self):
        c, action = self.rejected_by_pass(audit=True)
        d = self.decide(c, "reject")
        self.assertTrue(d.audit)
        self.assertEqual(action.audit_verdict, AutoTriageAction.AUDIT_AGREE)
        self.assertFalse(action.undone)
        self.assertEqual(auto_triage.standing_call(self.session, c.id), action)

    def test_a_video_outside_the_sample_is_not_a_spot_check(self):
        c, action = self.rejected_by_pass(audit=False)
        d = self.decide(c, "approve")
        self.assertFalse(d.audit)
        self.assertEqual(action.audit_verdict, "")
        self.assertTrue(action.undone)

    def test_an_ordinary_decision_is_logged_as_before(self):
        c = self.video(0.3)
        d = self.decide(c, "reject")
        self.assertFalse(d.audit)
        self.assertEqual((d.candidate_pk, d.predicted_prob, d.model_pk),
                         (c.id, 0.3, self.model.id))

    def test_undoing_a_verdict_withdraws_it(self):
        c, action = self.rejected_by_pass(audit=True)
        d = self.decide(c, "approve")
        d.undone = True
        auto_triage.record_operator_undo(self.session, d)
        self.assertEqual(action.audit_verdict, "")
        self.assertEqual(action.undone_reason, "restored")
        self.assertEqual(auto_triage.spot_check_summary(self.session)["answered"], 0)

    def test_restoring_ends_the_rejection(self):
        c, action = self.rejected_by_pass()
        auto_triage.record_operator_restore(self.session, c, now=self.now)
        self.assertTrue(action.undone)
        self.assertEqual(action.undone_reason, "restored")


class UndoTest(_Base):
    def test_undo_puts_back_only_what_still_stands(self):
        self.switch_on()
        kept, approved, agreed = self.video(0.01), self.video(0.01), self.video(0.01)
        out = self.do_pass()
        self.assertEqual(out["rejected"], 3)
        approved.status = STATUS_APPROVED
        auto_triage.record_operator_decision(self.session, approved, "approve", self.settings)
        call = next(a for a in self.calls() if a.candidate_pk == agreed.id)
        call.audit = True
        auto_triage.record_operator_decision(self.session, agreed, "reject", self.settings)
        self.session.flush()

        n = auto_triage.undo_run(self.session, out["run_pk"], now=self.now)

        self.assertEqual(n, 1)
        self.assertEqual(self.status(kept), STATUS_NEW)
        self.assertEqual(self.status(approved), STATUS_APPROVED)
        self.assertEqual(self.status(agreed), STATUS_REJECTED)
        kept_call = next(a for a in self.calls() if a.candidate_pk == kept.id)
        self.assertEqual((kept_call.undone, kept_call.undone_reason), (True, "pass undone"))
        self.assertIsNotNone(self.session.get(AutoTriageRun, out["run_pk"]).undone_at)
        # And it won't reject them again.
        self.assertEqual(self.do_pass()["rejected"], 0)
        self.assertEqual(self.status(kept), STATUS_NEW)

    def test_undo_of_a_missing_pass_says_so(self):
        with self.assertRaises(LookupError):
            auto_triage.undo_run(self.session, 999)


class SpotCheckTest(_Base):
    def answered(self, verdict: str, rate: float = 0.1, at: dt.datetime | None = None):
        c = self.video(0.01, status=STATUS_REJECTED)
        run = AutoTriageRun(stage=auto_triage.STAGE, trigger="test", started_at=self.now)
        self.session.add(run)
        self.session.flush()
        self.session.add(AutoTriageAction(
            run_pk=run.id, candidate_pk=c.id, video_id=c.video_id, stage=auto_triage.STAGE,
            approve_prob=0.01, audit=True, audit_rate=rate, audit_verdict=verdict,
            audited_at=at or self.now, created_at=self.now))
        self.session.flush()

    def test_each_verdict_counts_by_the_share_it_was_drawn_at(self):
        self.answered(AutoTriageAction.AUDIT_APPROVE, rate=0.05)   # stands for 20
        self.answered(AutoTriageAction.AUDIT_AGREE, rate=0.2)      # stands for 5
        out = auto_triage.spot_check_summary(self.session)
        self.assertEqual((out["answered"], out["approved"]), (2, 1))
        self.assertAlmostEqual(out["rate"], 0.8)
        self.assertLessEqual(out["low"], out["rate"])
        self.assertGreaterEqual(out["high"], out["rate"])

    def test_one_early_approval_does_not_pause_it(self):
        policy = self.switch_on()
        for i in range(19):
            self.answered(AutoTriageAction.AUDIT_AGREE)
        self.answered(AutoTriageAction.AUDIT_APPROVE)
        self.assertEqual(auto_triage.check_spot_checks(self.session, self.settings, policy,
                                                       now=self.now), "")
        self.assertEqual(auto_triage.load_policy(self.session).state, "on")

    def test_clearly_too_many_approvals_pause_it(self):
        policy = self.switch_on()
        for i in range(17):
            self.answered(AutoTriageAction.AUDIT_AGREE)
        for i in range(3):
            self.answered(AutoTriageAction.AUDIT_APPROVE)
        reason = auto_triage.check_spot_checks(self.session, self.settings, policy, now=self.now)
        self.assertIn("approved 3 of 20", reason)
        self.assertEqual(auto_triage.load_policy(self.session).state, "paused")
        # A pass then rejects nothing.
        self.video(0.001)
        self.assertEqual(self.do_pass()["outcome"], "paused")

    def test_a_verdict_can_pause_it_on_the_spot(self):
        self.switch_on()
        for i in range(17):
            self.answered(AutoTriageAction.AUDIT_AGREE)
        for i in range(2):
            self.answered(AutoTriageAction.AUDIT_APPROVE)
        c = self.video(0.01, status=STATUS_REJECTED)
        run = AutoTriageRun(stage=auto_triage.STAGE, trigger="test", started_at=self.now)
        self.session.add(run)
        self.session.flush()
        self.session.add(AutoTriageAction(run_pk=run.id, candidate_pk=c.id, video_id=c.video_id,
                                          stage=auto_triage.STAGE, approve_prob=0.01,
                                          audit=True, audit_rate=0.1, created_at=self.now))
        self.session.flush()
        c.status = STATUS_APPROVED
        auto_triage.record_operator_decision(self.session, c, "approve", self.settings,
                                             now=self.now)
        self.assertEqual(auto_triage.load_policy(self.session).state, "paused")

    def test_only_answers_since_the_window_opened_count(self):
        before = self.now - dt.timedelta(days=3)
        for i in range(10):
            self.answered(AutoTriageAction.AUDIT_APPROVE, at=before)
        policy = self.switch_on()
        for i in range(20):
            self.answered(AutoTriageAction.AUDIT_AGREE)
        self.assertEqual(auto_triage.check_spot_checks(self.session, self.settings, policy,
                                                       now=self.now), "")
        self.assertEqual(auto_triage.spot_check_summary(self.session, policy.watch_since_at)
                         ["answered"], 20)

    def test_pending_counts_only_rejections_still_standing(self):
        self.switch_on()
        c = self.video(0.01, status=STATUS_REJECTED)
        run = AutoTriageRun(stage=auto_triage.STAGE, trigger="test", started_at=self.now)
        self.session.add(run)
        self.session.flush()
        self.session.add(AutoTriageAction(run_pk=run.id, candidate_pk=c.id, video_id=c.video_id,
                                          stage=auto_triage.STAGE, audit=True, audit_rate=0.1,
                                          created_at=self.now))
        self.session.flush()
        self.assertEqual(auto_triage.spot_check_summary(self.session)["pending"], 1)
        auto_triage.record_operator_restore(self.session, c)
        self.session.flush()
        self.assertEqual(auto_triage.spot_check_summary(self.session)["pending"], 0)


class ReadingTest(_Base):
    def test_summary_note(self):
        self.assertEqual(auto_triage.summary_note(None), "")
        self.assertEqual(auto_triage.summary_note({"outcome": "off"}), "")
        self.assertEqual(auto_triage.summary_note({"outcome": "done", "rejected": 0}), "")
        self.assertEqual(
            auto_triage.summary_note({"outcome": "done", "rejected": 12, "audited": 1}),
            " — 12 rejected automatically, 1 set aside for a spot-check")
        self.assertIn("paused", auto_triage.summary_note({"outcome": "paused"}))

    def test_pct(self):
        self.assertEqual(auto_triage.pct(0.02), "2%")
        self.assertEqual(auto_triage.pct(0.015), "1.5%")
        self.assertEqual(auto_triage.pct(0.005), "0.5%")

    def test_cutoff_table_reads_the_rolling_grade(self):
        rolling = triage_model.evaluate([True, False, False, True], [0.001, 0.004, 0.012, 0.5],
                                        thresholds=triage_model.CUTOFF_CHOICES,
                                        posted=[True, False, False, False])
        self.model.backtest = json.dumps({"auc": 0.8, "n": 4, "handoff": [], "rolling": rolling})
        self.session.flush()
        self.video(0.004)
        rows = {r["cutoff"]: r for r in auto_triage.cutoff_table(self.session, self.settings)}
        self.assertEqual(rows[0.005]["approvals_lost"], 1)
        self.assertEqual(rows[0.005]["posted_lost"], 1)
        self.assertEqual(rows[0.02]["history_below"], 3)
        self.assertEqual(rows[0.005]["waiting_now"], 1)
        self.assertEqual(rows[0.005]["approvals"], 2)


if __name__ == "__main__":
    unittest.main()
