"""The long-tail metrics snapshot, run by the scheduler tick.

``analytics.snapshot_metrics`` gains a call cap (most overdue first) and the
tick runs it at most every 6 hours (``scheduler.run_metrics_snapshot``, last
run kept in AppToken "metrics_snapshot_at"). The Threads API is mocked
(``analytics.fetch_insights`` / ``threads_api``); the database is in-memory
SQLite behind a patched ``scheduler.session_scope``.

Run with the suite's §0 prefix:  python -m pytest tests/test_metrics_capture.py -q
"""
from __future__ import annotations

import contextlib
import datetime as dt
import unittest
from unittest import mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import analytics, scheduler, threads_api
from app.config import Settings
from app.models import (AppToken, Base, Candidate, Channel, Cut, MetricSnapshot, ThreadsPost,
                        utcnow)


class DbCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True,
                                    connect_args={"check_same_thread": False},
                                    poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        self.addCleanup(self.engine.dispose)
        with self.Session() as s:
            ch = Channel(call_sign="KTST", url="https://example.com/c")
            s.add(ch)
            s.commit()
            self.channel_pk = ch.id
        self._n = 0
        self.settings = Settings(raw={})
        self.start(mock.patch.object(analytics, "load_settings", lambda: self.settings))
        self.start(mock.patch.object(scheduler, "load_settings", lambda: self.settings))

    def start(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def published(self, days_ago: float, *, last_snap_days_ago: float | None = None,
                  media_id: str | None = None) -> int:
        self._n += 1
        with self.Session() as s:
            cand = Candidate(video_id=f"v{self._n}", channel_pk=self.channel_pk, title="t", url="u")
            s.add(cand)
            s.flush()
            cut = Cut(candidate_pk=cand.id)
            s.add(cut)
            s.flush()
            post = ThreadsPost(candidate_pk=cand.id, cut_pk=cut.id, status="published",
                               threads_media_id=f"m{self._n}" if media_id is None else media_id,
                               published_at=utcnow() - dt.timedelta(days=days_ago))
            s.add(post)
            s.flush()
            if last_snap_days_ago is not None:
                s.add(MetricSnapshot(post_pk=post.id, views=1,
                                     captured_at=utcnow() - dt.timedelta(days=last_snap_days_ago)))
            s.commit()
            return post.id

    def snapshots(self, post_id: int) -> int:
        with self.Session() as s:
            return len(s.execute(select(MetricSnapshot).where(MetricSnapshot.post_pk == post_id))
                       .scalars().all())


class SnapshotMetricsTest(DbCase):
    def test_limit_takes_the_most_overdue_first(self):
        never = self.published(3)                           # never sampled: first
        week_old = self.published(30, last_snap_days_ago=7)
        day_old = self.published(30, last_snap_days_ago=1)
        fresh = self.published(30, last_snap_days_ago=0.1)  # not due
        tail = self.published(100, last_snap_days_ago=8)    # weekly tail: due 1 day ago
        calls = []
        with mock.patch.object(analytics, "fetch_insights",
                               side_effect=lambda mid: calls.append(mid) or {"views": 5}):
            with self.Session() as s:
                taken = analytics.snapshot_metrics(s, limit=3)
                s.commit()
        self.assertEqual(taken, 3)
        self.assertEqual(calls, [f"m{never}", f"m{week_old}", f"m{tail}"])
        self.assertEqual(self.snapshots(never), 1)
        self.assertEqual(self.snapshots(day_old), 1)        # left for next time
        self.assertEqual(self.snapshots(fresh), 1)

    def test_no_limit_takes_everything_due(self):
        ids = [self.published(2) for _ in range(4)]
        with mock.patch.object(analytics, "fetch_insights", return_value={"views": 1}) as fetch:
            with self.Session() as s:
                self.assertEqual(analytics.snapshot_metrics(s), 4)
        self.assertEqual(fetch.call_count, len(ids))

    def test_failures_count_as_calls_and_dont_stop_the_run(self):
        self.published(2)
        self.published(2)
        self.published(2, media_id="")       # nothing to ask Threads about: no call
        answers = iter([RuntimeError("boom"), {}, {"views": 3}])

        def fetch(mid):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        with mock.patch.object(analytics, "fetch_insights", side_effect=fetch) as fetcher:
            with self.Session() as s:
                self.assertEqual(analytics.snapshot_metrics(s, limit=200), 0)
        self.assertEqual(fetcher.call_count, 2)

    def test_old_posts_past_a_year_are_never_due(self):
        self.published(400)
        with mock.patch.object(analytics, "fetch_insights") as fetch:
            with self.Session() as s:
                self.assertEqual(analytics.snapshot_metrics(s, limit=5), 0)
        fetch.assert_not_called()


class SchedulerCadenceTest(DbCase):
    def setUp(self):
        super().setUp()

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            session = self.Session()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        self.start(mock.patch.object(scheduler, "session_scope", scope))
        self.authed = self.start(mock.patch.object(threads_api, "is_authenticated",
                                                   return_value=True))
        self.snapshot = self.start(mock.patch.object(scheduler, "snapshot_metrics", return_value=7))
        self.now = dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.timezone.utc)

    def stamp(self) -> str | None:
        with self.Session() as s:
            row = s.get(AppToken, scheduler.METRICS_SNAPSHOT_TOKEN)
            return row.value if row else None

    def test_runs_at_most_every_six_hours(self):
        self.assertEqual(scheduler.run_metrics_snapshot(self.now), 7)
        self.assertEqual(self.snapshot.call_args.kwargs, {"limit": 200})
        self.assertEqual(self.stamp(), self.now.isoformat())
        self.assertEqual(scheduler.run_metrics_snapshot(self.now + dt.timedelta(hours=5, minutes=59)), 0)
        self.assertEqual(self.snapshot.call_count, 1)
        self.assertEqual(scheduler.run_metrics_snapshot(self.now + dt.timedelta(hours=6)), 7)
        self.assertEqual(self.snapshot.call_count, 2)

    def test_settings_change_the_cadence_and_the_cap(self):
        self.settings = Settings(raw={"analytics": {"auto_snapshot_hours": 1,
                                                     "auto_snapshot_max_calls": 50}})
        scheduler.run_metrics_snapshot(self.now)
        self.assertEqual(self.snapshot.call_args.kwargs, {"limit": 50})
        scheduler.run_metrics_snapshot(self.now + dt.timedelta(hours=1))
        self.assertEqual(self.snapshot.call_count, 2)

    def test_zero_hours_turns_it_off(self):
        self.settings = Settings(raw={"analytics": {"auto_snapshot_hours": 0}})
        self.assertEqual(scheduler.run_metrics_snapshot(self.now), 0)
        self.snapshot.assert_not_called()
        self.assertIsNone(self.stamp())

    def test_not_connected_does_nothing(self):
        self.authed.return_value = False
        self.assertEqual(scheduler.run_metrics_snapshot(self.now), 0)
        self.snapshot.assert_not_called()
        self.assertIsNone(self.stamp())

    def test_a_failed_run_never_raises_and_waits_its_turn(self):
        self.snapshot.side_effect = RuntimeError("Threads is down")
        self.assertEqual(scheduler.run_metrics_snapshot(self.now), 0)
        self.assertEqual(self.stamp(), self.now.isoformat())     # claimed before it ran
        self.assertEqual(scheduler.run_metrics_snapshot(self.now + dt.timedelta(minutes=1)), 0)
        self.assertEqual(self.snapshot.call_count, 1)             # no retry every tick

    def test_a_garbled_stamp_is_treated_as_never(self):
        with self.Session() as s:
            s.add(AppToken(name=scheduler.METRICS_SNAPSHOT_TOKEN, value="not a date"))
            s.commit()
        self.assertEqual(scheduler.run_metrics_snapshot(self.now), 7)

    def test_end_to_end_with_a_mocked_threads_api(self):
        self.snapshot.side_effect = lambda session, limit: analytics.snapshot_metrics(session,
                                                                                    limit=limit)
        ids = [self.published(2) for _ in range(3)]
        self.settings = Settings(raw={"analytics": {"auto_snapshot_max_calls": 2}})
        with mock.patch.object(analytics, "fetch_insights",
                               return_value={"views": 10, "likes": 1}) as fetch:
            self.assertEqual(scheduler.run_metrics_snapshot(self.now), 2)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(sum(self.snapshots(i) for i in ids), 2)

    def test_the_tick_runs_it(self):
        names = ["recover_stuck_publishing", "annotate_queued_posts", "repair_clip_uploads",
                 "run_metrics_poll", "check_facebook_videos", "run_window_tick",
                 "run_first_reply_sweep", "recycle_overview"]
        for name in names:
            self.start(mock.patch.object(scheduler, name, return_value=0))
        run = self.start(mock.patch.object(scheduler, "run_metrics_snapshot", return_value=0))
        scheduler.run_tick()
        run.assert_called_once_with()

    def test_the_scheduler_stays_free_of_the_clip_engine(self):
        source = scheduler.__file__
        with open(source) as handle:
            text = handle.read()
        self.assertNotIn("clip_engine", text.split("def run_metrics_snapshot", 1)[1]
                         .split("\ndef ", 1)[0])


if __name__ == "__main__":
    unittest.main()
