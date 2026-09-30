"""Tests for the rerun review queue and decisions (app/rerun_review.py).

Run with:  python -m unittest tests.test_rerun_review -v

Runs against a throwaway in-memory SQLite database, not the app's engine:
every function under test takes the session it's given.
"""
from __future__ import annotations

import datetime as dt
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import rerun_review
from app.models import (Base, Candidate, Channel, Cut, MetricSnapshot, RerunReview,
                        ThreadsPost, utcnow)


class RerunReviewTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        self.channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.session.add(self.channel)
        self.session.flush()
        self._n = 0

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def post(self, *, views: int, shelf: str = "timely", days_ago: int = 60,
             override: str = "", caption: str = "a caption",
             clip: str = "clips/x.mp4") -> ThreadsPost:
        self._n += 1
        cand = Candidate(video_id=f"v{self._n}", channel_pk=self.channel.id,
                         title=f"Video {self._n}", url="https://example.com/v",
                         category="news", shelf_life=shelf)
        self.session.add(cand)
        self.session.flush()
        cut = Cut(candidate_pk=cand.id)
        self.session.add(cut)
        self.session.flush()
        published = utcnow() - dt.timedelta(days=days_ago)
        p = ThreadsPost(candidate_pk=cand.id, cut_pk=cut.id, status="published",
                        published_at=published, caption=caption,
                        clip_object_path=clip, shelf_life=override)
        self.session.add(p)
        self.session.flush()
        self.session.add(MetricSnapshot(post_pk=p.id, views=views,
                                        captured_at=published + dt.timedelta(hours=48)))
        self.session.flush()
        return p

    def baseline(self, n: int = 10, views: int = 100):
        """Evergreen filler that sets the account baseline without being reviewable."""
        return [self.post(views=views, shelf="evergreen") for _ in range(n)]

    def pending_ids(self) -> list[int]:
        return [f["post"].id for f in rerun_review.review_queue(self.session)["pending"]]

    def test_needs_a_baseline(self):
        self.post(views=5000)
        queue = rerun_review.review_queue(self.session)
        self.assertEqual(queue["pending"], [])
        self.assertIsNone(queue["threshold"])

    def test_selects_proven_old_non_evergreen_posts(self):
        self.baseline()
        hit = self.post(views=900)
        breaking = self.post(views=800, shelf="breaking")
        self.post(views=10)                        # below the baseline median
        self.post(views=900, days_ago=5)           # its moment hasn't passed
        self.post(views=900, shelf="evergreen")    # can already rerun
        self.post(views=900, override="evergreen")  # operator already flipped it
        self.post(views=900, caption="")           # nothing to clone
        self.post(views=900, clip="")              # no clip to clone
        self.assertEqual(self.pending_ids(), [hit.id, breaking.id])

    def test_model_verdict_orders_the_list(self):
        self.baseline()
        dated = self.post(views=2000)
        as_is = self.post(views=300)
        unjudged = self.post(views=1000)
        self.session.add_all([RerunReview(post_pk=dated.id, verdict="dated"),
                              RerunReview(post_pk=as_is.id, verdict="evergreen")])
        self.session.flush()
        self.assertEqual(self.pending_ids(), [as_is.id, unjudged.id, dated.id])

    def test_evergreen_decision_writes_the_override_and_undo_restores_it(self):
        self.baseline()
        p = self.post(views=900, override="breaking")
        self.assertTrue(rerun_review.decide(self.session, p.id, "evergreen"))
        self.assertEqual(p.shelf_life, "evergreen")
        queue = rerun_review.review_queue(self.session)
        self.assertEqual(queue["pending"], [])
        self.assertEqual([f["post"].id for f in queue["decided"]], [p.id])

        self.assertTrue(rerun_review.undo(self.session, p.id))
        self.assertEqual(p.shelf_life, "breaking")
        self.assertEqual(self.pending_ids(), [p.id])

    def test_keep_and_new_caption_leave_shelf_life_alone(self):
        self.baseline()
        kept = self.post(views=900)
        waiting = self.post(views=900)
        rerun_review.decide(self.session, kept.id, "keep")
        rerun_review.decide(self.session, waiting.id, "new_caption")
        self.assertEqual((kept.shelf_life, waiting.shelf_life), ("", ""))
        self.assertEqual(self.pending_ids(), [])

    def test_switching_away_from_evergreen_restores_the_override(self):
        self.baseline()
        p = self.post(views=900)
        rerun_review.decide(self.session, p.id, "evergreen")
        rerun_review.decide(self.session, p.id, "keep")
        self.assertEqual(p.shelf_life, "")

    def test_accept_applies_only_rerun_as_is_verdicts(self):
        self.baseline()
        as_is = self.post(views=900)
        dated = self.post(views=900)
        self.session.add_all([RerunReview(post_pk=as_is.id, verdict="evergreen"),
                              RerunReview(post_pk=dated.id, verdict="dated")])
        self.session.flush()
        self.assertEqual(rerun_review.accept_evergreen_verdicts(self.session), 1)
        self.assertEqual((as_is.shelf_life, dated.shelf_life), ("evergreen", ""))

    def test_unknown_decision_is_rejected(self):
        self.baseline()
        p = self.post(views=900)
        self.assertFalse(rerun_review.decide(self.session, p.id, "maybe"))


if __name__ == "__main__":
    unittest.main()
