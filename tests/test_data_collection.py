"""Tests for the data kept for later: follower history, scheduler decision
records, long-tail metrics, and the operator's clip labels + review reminder.

Run with:  python -m unittest tests.test_data_collection -v
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import unittest
from unittest import mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import analytics, labels, scheduler, threads_api
from app.models import (AccountSnapshot, Base, Candidate, Channel, ContentLabel, Cut,
                        MetricSnapshot, PlacementLog, ThreadsPost, utcnow)
from app.placement import (PlacementContext, PlacementDecision, PlacementSettings, PostFacts,
                           assign_posts_to_windows)


class FakePost:
    def __init__(self, post_id: int):
        self.id = post_id
        self.pinned_window_key = ""


class PlacementAlternativesTest(unittest.TestCase):
    def test_runners_up_are_recorded_without_changing_the_pick(self):
        day = dt.date(2026, 8, 16)
        facts = {i: PostFacts(post_id=i, queued_date=day - dt.timedelta(days=i))
                 for i in (1, 2, 3)}
        posts = [FakePost(i) for i in (1, 2, 3)]
        key = f"{day.isoformat()}#0"

        plain = assign_posts_to_windows(
            posts, [key], ctx=PlacementContext(settings=PlacementSettings(), facts=facts))
        ctx = PlacementContext(settings=PlacementSettings(), facts=facts)
        logged = assign_posts_to_windows(posts, [key], ctx=ctx)

        self.assertEqual([p.id for p in plain], [p.id for p in logged])
        decision = ctx.decisions[key]
        others = [pid for pid, _score, _parts in decision.alternatives]
        self.assertEqual(sorted(others), sorted({1, 2, 3} - {decision.post_id}))
        scores = [score for _pid, score, _parts in decision.alternatives]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLessEqual(max(scores), decision.score)


class FollowerCountParseTest(unittest.TestCase):
    def test_total_value_shape(self):
        reply = {"data": [{"name": "followers_count", "total_value": {"value": 4321}}]}
        with mock.patch.object(threads_api, "_api", return_value=reply):
            self.assertEqual(threads_api.fetch_followers_count(), 4321)

    def test_values_shape(self):
        reply = {"data": [{"name": "followers_count", "values": [{"value": 10}, {"value": 12}]}]}
        with mock.patch.object(threads_api, "_api", return_value=reply):
            self.assertEqual(threads_api.fetch_followers_count(), 12)

    def test_api_failure_is_none(self):
        with mock.patch.object(threads_api, "_api", side_effect=threads_api.ThreadsError("nope")):
            self.assertIsNone(threads_api.fetch_followers_count())


class DbCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        self.channel = Channel(call_sign="KTST", url="https://example.com/c")
        self.session.add(self.channel)
        self.session.flush()
        self._n = 0

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def published(self, days_ago: float, views: int | None = None) -> ThreadsPost:
        self._n += 1
        cand = Candidate(video_id=f"v{self._n}", channel_pk=self.channel.id,
                         title="t", url="u")
        self.session.add(cand)
        self.session.flush()
        cut = Cut(candidate_pk=cand.id)
        self.session.add(cut)
        self.session.flush()
        post = ThreadsPost(candidate_pk=cand.id, cut_pk=cut.id, status="published",
                           threads_media_id=f"m{self._n}",
                           published_at=utcnow() - dt.timedelta(days=days_ago))
        self.session.add(post)
        self.session.flush()
        if views is not None:
            self.session.add(MetricSnapshot(post_pk=post.id, views=views))
            self.session.flush()
        return post

    def snap(self, post, days_ago: float):
        self.session.add(MetricSnapshot(post_pk=post.id, views=1,
                                        captured_at=utcnow() - dt.timedelta(days=days_ago)))
        self.session.flush()


class AccountSnapshotTest(DbCase):
    def test_daily_and_skips_failures(self):
        with mock.patch.object(threads_api, "fetch_followers_count", return_value=500):
            self.assertTrue(analytics.snapshot_account(self.session))
            self.assertFalse(analytics.snapshot_account(self.session))   # too soon
        self.assertEqual(analytics.followers_at(self.session), 500)
        self.session.execute(AccountSnapshot.__table__.update().values(
            captured_at=utcnow() - dt.timedelta(hours=30)))
        with mock.patch.object(threads_api, "fetch_followers_count", return_value=None):
            self.assertFalse(analytics.snapshot_account(self.session))
        self.assertEqual(self.session.query(AccountSnapshot).count(), 1)


class LongTailMetricsTest(DbCase):
    def test_old_posts_are_sampled_weekly_until_a_year(self):
        young = self.published(10)
        self.snap(young, 1)                 # 24h ago: due (12h cadence)
        tail_recent = self.published(90)
        self.snap(tail_recent, 3)           # tail, sampled 3 days ago: not due
        tail_due = self.published(90)
        self.snap(tail_due, 8)              # tail, 8 days ago: due
        ancient = self.published(400)
        self.snap(ancient, 30)              # past a year: never

        settings = mock.Mock()
        settings.get = lambda key, default=None: default
        taken = []
        with mock.patch.object(analytics, "load_settings", return_value=settings), \
             mock.patch.object(analytics, "_add_snapshot",
                               side_effect=lambda s, p: taken.append(p.id) or True):
            analytics.snapshot_metrics(self.session)
        self.assertEqual(sorted(taken), sorted([young.id, tail_due.id]))


class PlacementLogTest(DbCase):
    def setUp(self):
        super().setUp()

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            yield self.session
            self.session.flush()

        p = mock.patch.object(scheduler, "session_scope", scope)
        p.start()
        self.addCleanup(p.stop)

    def test_decision_and_context_are_kept(self):
        recent = self.published(0.5, views=1200)
        self.session.add(AccountSnapshot(followers_count=900))
        self.session.flush()
        decision = PlacementDecision(post_id=recent.id, relax_step=0, score=3.5,
                                     parts={"urgency": 2.0},
                                     alternatives=((77, 2.1, {"urgency": 1.0}),))
        scheduler._log_placement("2026-11-09#0", recent.id, "publish", decision)
        row = self.session.execute(select(PlacementLog)).scalar_one()
        self.assertEqual(row.score, 3.5)
        self.assertEqual(json.loads(row.alternatives)[0]["post_id"], 77)
        ctx = json.loads(row.context)
        self.assertEqual(ctx["followers"], 900)
        self.assertEqual(ctx["recent_posts"][0]["views"], 1200)

    def test_logging_failure_never_raises(self):
        with mock.patch.object(scheduler, "_decision_context", side_effect=RuntimeError("x")):
            scheduler._log_placement("2026-11-09#0", 1, "publish", None)   # no exception


class LabelsTest(DbCase):
    def test_set_overwrite_clear(self):
        post = self.published(5)
        self.assertTrue(labels.set_label(self.session, post.cut_pk, "yes"))
        self.assertTrue(labels.set_label(self.session, post.cut_pk, "no"))
        self.session.flush()
        self.assertEqual(labels.counts(self.session), {"yes": 0, "no": 1, "total": 1})
        self.assertTrue(labels.set_label(self.session, post.cut_pk, ""))
        self.session.flush()
        self.assertEqual(labels.counts(self.session)["total"], 0)
        self.assertFalse(labels.set_label(self.session, post.cut_pk, "maybe"))
        self.assertFalse(labels.set_label(self.session, 99999, "yes"))

    def test_queue_puts_unlabelled_first_and_one_per_clip(self):
        older = self.published(10, views=50)
        newer = self.published(1, views=900)
        rerun = ThreadsPost(candidate_pk=older.candidate_pk, cut_pk=older.cut_pk,
                            status="published", repost_of_post_pk=older.id,
                            published_at=utcnow())
        self.session.add(rerun)
        labels.set_label(self.session, newer.cut_pk, "yes")
        self.session.flush()
        items = labels.queue(self.session)
        self.assertEqual([i["cut_id"] for i in items], [older.cut_pk, newer.cut_pk])
        self.assertEqual(items[0]["post_id"], older.id)   # the original, not the rerun
        self.assertEqual(items[1]["value"], "yes")

    def test_review_reminder_window(self):
        settings = mock.Mock()
        settings.get = lambda key, default=None: "2027-05-09"
        before, on = dt.date(2027, 5, 8), dt.date(2027, 5, 9)
        self.assertFalse(labels.review_due(self.session, settings, today=before))
        self.assertTrue(labels.review_due(self.session, settings, today=on))
        labels.mark_review_done(self.session)
        self.session.flush()
        self.assertFalse(labels.review_due(self.session, settings, today=on))


if __name__ == "__main__":
    unittest.main()
