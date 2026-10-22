"""Tests for the approve predictor (app/triage_model.py).

Run with:  python -m unittest tests.test_triage_model -v

Runs against a throwaway in-memory SQLite database, not the app's engine:
``train`` and ``score_new_candidates`` take the session they're given.
"""
from __future__ import annotations

import datetime as dt
import json
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import triage_model
from app.models import (STATUS_NEW, STATUS_REJECTED, Base, Candidate, Channel,
                        TriageDecision, TriageModel, utcnow)


class _Settings:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


SETTINGS = _Settings(**{"triage_model.min_decisions_each": 5})


class EvaluateTest(unittest.TestCase):
    def test_auc_perfect_and_inverted(self):
        self.assertEqual(triage_model._auc([True, False], [0.9, 0.1]), 1.0)
        self.assertEqual(triage_model._auc([True, False], [0.1, 0.9]), 0.0)

    def test_auc_ties_count_half(self):
        self.assertEqual(triage_model._auc([True, False], [0.5, 0.5]), 0.5)

    def test_auc_undefined_without_both_outcomes(self):
        self.assertIsNone(triage_model._auc([False, False], [0.1, 0.2]))

    def test_buckets_and_handoff(self):
        labels = [False, False, True, True]
        probs = [0.01, 0.03, 0.12, 1.0]
        out = triage_model.evaluate(labels, probs)
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["approved"], 2)
        first, last = out["buckets"][0], out["buckets"][-1]
        self.assertEqual((first["n"], first["approved"]), (2, 0))
        # A probability of exactly 1.0 lands in the top bucket, not nowhere.
        self.assertEqual((last["n"], last["approved"]), (1, 1))
        under_5 = next(h for h in out["handoff"] if h["threshold"] == 0.05)
        self.assertEqual((under_5["below"], under_5["approvals_lost"]), (2, 0))


class TrainAndScoreTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        self.channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.session.add(self.channel)
        self.session.flush()
        self._n = 0
        triage_model._cache = None

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def candidate(self, title: str, status: str = STATUS_NEW) -> Candidate:
        self._n += 1
        c = Candidate(video_id=f"v{self._n}", channel_pk=self.channel.id, title=title,
                      description="", url="https://example.com/v", status=status,
                      relevance_score=0.7)
        self.session.add(c)
        self.session.flush()
        return c

    def decide(self, title: str, action: str, undone: bool = False) -> None:
        c = self.candidate(title, STATUS_REJECTED)
        self.session.add(TriageDecision(
            candidate_pk=c.id, action=action, undone=undone,
            decided_at=utcnow() - dt.timedelta(minutes=1000 - self._n)))

    def seed(self, n_each: int = 12) -> None:
        for i in range(n_each):
            self.decide(f"wildfire evacuation footage {i}", "approve")
            self.decide(f"weather forecast sunny weekend {i}", "reject")
        self.session.flush()

    def test_waits_for_enough_of_each_outcome(self):
        for i in range(10):
            self.decide(f"forecast {i}", "reject")
        self.decide("wildfire", "approve")
        self.session.flush()
        self.assertIsNone(triage_model.train(self.session, SETTINGS))

    def test_learns_the_split_and_scores_the_queue(self):
        self.seed()
        row = triage_model.train(self.session, SETTINGS)
        self.assertIsNotNone(row)
        self.assertEqual(row.n_decisions, 24)
        self.assertEqual(json.loads(row.backtest)["n"], 6)

        fire = self.candidate("wildfire evacuation footage tonight")
        sun = self.candidate("weather forecast sunny weekend ahead")
        done = self.candidate("wildfire evacuation footage", STATUS_REJECTED)
        n = triage_model.score_new_candidates(self.session)
        self.assertEqual(n, 2)
        self.session.expire_all()
        self.assertGreater(fire.approve_prob, sun.approve_prob)
        self.assertEqual(fire.approve_model_pk, row.id)
        self.assertIsNone(done.approve_prob)
        # Already scored by this model: the monitor's pass skips them.
        self.assertEqual(triage_model.score_new_candidates(self.session, only_missing=True), 0)

    def test_undone_decisions_are_not_training_data(self):
        self.seed()
        self.decide("undone thing", "approve", undone=True)
        self.session.flush()
        self.assertEqual(triage_model.train(self.session, SETTINGS).n_decisions, 24)

    def test_old_models_drop_their_params(self):
        self.seed()
        rows = [triage_model.train(self.session, SETTINGS)
                for _ in range(triage_model.KEEP_PARAMS + 1)]
        self.session.flush()
        self.session.expire_all()
        self.assertEqual(rows[0].params, "")
        self.assertTrue(all(r.params for r in rows[1:]))
        self.assertEqual(triage_model.load_predictor(self.session).model_pk, rows[-1].id)
        self.assertEqual(self.session.query(TriageModel).count(), len(rows))


if __name__ == "__main__":
    unittest.main()
