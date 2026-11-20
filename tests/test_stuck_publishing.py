"""Tests for rescuing posts stuck in ``publishing`` (app/scheduler.py:
recover_stuck_publishing) when several machines publish for one account.

Run with:  python -m unittest tests.test_stuck_publishing -v

No network; an in-memory SQLite database.
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker

from app import scheduler
from app.models import (Base, Candidate, Channel, FacebookPost, InstagramPost, LinkedInPost,
                        ThreadsPost, XPost, utcnow)

NOW = dt.datetime(2026, 11, 19, 22, 0, tzinfo=dt.timezone.utc)


class RecoverStuckPublishingTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(engine, future=True, expire_on_commit=False)
        settings = mock.patch.object(scheduler, "load_settings", return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        with self.Session() as s:
            ch = Channel(call_sign="KXYZ", url="https://example.com/c")
            s.add(ch)
            s.flush()
            c = Candidate(video_id="sp1", channel_pk=ch.id, title="t", url="u")
            s.add(c)
            s.commit()
            self.candidate_pk = c.id

    def post(self, *, touched: dt.datetime | None, **kw) -> int:
        with self.Session() as s:
            p = ThreadsPost(candidate_pk=self.candidate_pk, caption="c", status="publishing",
                            pinned_window_key="2026-11-19#4", **kw)
            s.add(p)
            s.flush()
            # Set the stamp directly: a plain write would re-stamp it as now.
            s.execute(update(ThreadsPost).where(ThreadsPost.id == p.id)
                      .values(updated_at=touched))
            s.commit()
            return p.id

    def recover(self, **kw) -> int:
        with self.Session() as s:
            n = scheduler.recover_stuck_publishing(s, now=NOW, **kw)
            s.commit()
            return n

    def status(self, model, pk) -> str:
        with self.Session() as s:
            return s.get(model, pk).status

    def test_a_post_another_machine_is_publishing_is_left_alone(self):
        # Touched a minute ago: some machine is mid-publish. Neither the tick
        # nor a freshly started process may fail it.
        pk = self.post(touched=NOW - dt.timedelta(minutes=1))
        for only_inactive in (True, False):
            with self.subTest(only_inactive=only_inactive):
                self.assertEqual(self.recover(only_inactive=only_inactive), 0)
                self.assertEqual(self.status(ThreadsPost, pk), "publishing")

    def test_a_post_untouched_past_the_cutoff_is_recovered(self):
        pk = self.post(touched=NOW - dt.timedelta(minutes=46))
        self.assertEqual(self.recover(), 1)
        with self.Session() as s:
            p = s.get(ThreadsPost, pk)
            self.assertEqual((p.status, p.pinned_window_key), ("failed", ""))
            self.assertIn("interrupted", p.error)

    def test_a_row_from_before_the_stamp_existed_is_recovered(self):
        pk = self.post(touched=None)
        self.assertEqual(self.recover(), 1)
        self.assertEqual(self.status(ThreadsPost, pk), "failed")

    def test_this_process_never_fails_its_own_publish(self):
        pk = self.post(touched=NOW - dt.timedelta(hours=3))
        with mock.patch.object(scheduler, "is_publish_active", return_value=True):
            self.assertEqual(self.recover(only_inactive=True), 0)
        self.assertEqual(self.status(ThreadsPost, pk), "publishing")

    def test_paired_posts_follow_the_same_rule(self):
        # One paired row per Threads post, so one parent per age.
        parents = {label: self.post(touched=None) for label in ("fresh", "old")}
        with self.Session() as s:
            for pk in parents.values():
                s.get(ThreadsPost, pk).status = "published"
            s.commit()
        fresh, old = NOW - dt.timedelta(minutes=2), NOW - dt.timedelta(hours=1)
        rows = {}
        with self.Session() as s:
            for model in (InstagramPost, LinkedInPost, XPost, FacebookPost):
                for label, touched in (("fresh", fresh), ("old", old)):
                    row = model(threads_post_pk=parents[label], status="publishing")
                    s.add(row)
                    s.flush()
                    s.execute(update(model).where(model.id == row.id).values(updated_at=touched))
                    rows[(model, label)] = row.id
            s.commit()
        self.assertEqual(self.recover(), 4)
        for (model, label), pk in rows.items():
            with self.subTest(model=model.__name__, label=label):
                self.assertEqual(self.status(model, pk),
                                 "publishing" if label == "fresh" else "failed")

    def test_every_write_stamps_the_row(self):
        # The mechanism the rule relies on: the ORM and Core updates alike.
        pk = self.post(touched=NOW - dt.timedelta(days=2))
        with self.Session() as s:
            s.get(ThreadsPost, pk).error = "still going"
            s.commit()
        with self.Session() as s:
            stamp = s.get(ThreadsPost, pk).updated_at.replace(tzinfo=dt.timezone.utc)
        self.assertGreater(stamp, utcnow() - dt.timedelta(minutes=1))
        with self.Session() as s:
            s.execute(update(ThreadsPost).where(ThreadsPost.id == pk).values(updated_at=None))
            s.execute(update(ThreadsPost).where(ThreadsPost.id == pk).values(status="publishing"))
            s.commit()
            self.assertIsNotNone(s.get(ThreadsPost, pk).updated_at)

    def test_the_cutoff_outlasts_the_processing_waits(self):
        self.assertEqual(scheduler.stuck_publishing_cutoff(NOW), NOW - dt.timedelta(minutes=45))
        slow = {"threads.publish_poll_timeout_seconds": 1800,
                "instagram.publish_poll_timeout_seconds": 1200}
        with mock.patch.object(scheduler, "load_settings", return_value=slow):
            self.assertEqual(scheduler.stuck_publishing_cutoff(NOW),
                             NOW - dt.timedelta(minutes=30 + 20 + 15))


if __name__ == "__main__":
    unittest.main()
