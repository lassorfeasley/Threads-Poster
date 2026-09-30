"""Tests for the posting schedule (app/posting_schedule.py) and the scheduler's
per-day window lookups that read it.

Run with:  python -m unittest tests.test_posting_schedule -v

Saving runs against a throwaway in-memory SQLite database; the scheduler
lookups patch ``windows_for_date`` so nothing reads the app's database.
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import posting_schedule as ps
from app import scheduler
from app.models import Base, PostingSchedule, ThreadsPost

TODAY = dt.date(2026, 10, 1)
D = lambda n: TODAY + dt.timedelta(days=n)  # noqa: E731


class ValidationTest(unittest.TestCase):
    def test_normalize_sorts_pads_and_drops_blanks(self):
        self.assertEqual(ps.normalize(["14:00", "", "2:05", "22:00"]),
                         ["02:05", "14:00", "22:00"])

    def test_normalize_rejects_non_times(self):
        for bad in ("noon", "25:00", "10:75"):
            with self.assertRaises(ValueError):
                ps.normalize([bad])

    def test_every_four_hours_round_the_clock_is_valid(self):
        self.assertEqual(ps.problems(["02:00", "06:00", "10:00", "14:00", "18:00", "22:00"], 90), [])

    def test_gap_across_midnight_counts(self):
        # 23:30 to tomorrow's 00:30 is an hour — under a 90-minute floor.
        self.assertTrue(ps.problems(["00:30", "12:00", "23:30"], 90))

    def test_limits(self):
        self.assertTrue(ps.problems([], 90))
        self.assertTrue(ps.problems(["10:00", "10:00"], 90))
        self.assertTrue(ps.problems([f"{h:02d}:00" for h in range(0, 24, 2)], 90))  # 12 > max


class LookupTest(unittest.TestCase):
    def test_latest_row_on_or_before_the_day_wins(self):
        rows = [(D(1), ["10:00"], []), (D(5), ["08:00", "20:00"], [])]
        with mock.patch.object(ps, "yaml_windows", return_value=["09:00", "19:00", "21:00"]):
            self.assertEqual(ps._windows_in(rows, D(0)), ["09:00", "19:00", "21:00"])
        self.assertEqual(ps._windows_in(rows, D(1)), ["10:00"])
        self.assertEqual(ps._windows_in(rows, D(4)), ["10:00"])
        self.assertEqual(ps._windows_in(rows, D(9)), ["08:00", "20:00"])

    def test_overflow_index_follows_each_days_window_count(self):
        counts = {D(0): ["10:00", "13:00", "19:00"],
                  D(1): ["02:00", "06:00", "10:00", "14:00", "18:00", "22:00"]}
        with mock.patch.object(scheduler, "windows_for_date", side_effect=lambda d: counts[d]):
            self.assertTrue(scheduler._is_overflow_key(f"{D(0)}#3"))
            self.assertFalse(scheduler._is_overflow_key(f"{D(1)}#3"))
            self.assertEqual(scheduler._overflow_key(D(1)), f"{D(1)}#6")


class SaveTest(unittest.TestCase):
    OLD = ["10:00", "13:00", "19:00"]
    NEW = ["02:00", "06:00", "10:00", "14:00", "18:00", "22:00"]

    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        patches = [mock.patch.object(ps, "yaml_windows", return_value=list(self.OLD)),
                   mock.patch.object(ps, "_overflow_minutes", return_value=21 * 60)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def pinned(self, key: str) -> ThreadsPost:
        p = ThreadsPost(status="queued", pinned_window_key=key)
        self.session.add(p)
        self.session.flush()
        return p

    def test_must_start_tomorrow_or_later(self):
        with self.assertRaises(ValueError):
            ps.save(self.session, TODAY, self.NEW, today=TODAY)

    def test_pins_move_to_the_nearest_new_time(self):
        before = self.pinned(f"{D(1)}#1")    # 13:00 -> 14:00 (#3)
        evening = self.pinned(f"{D(2)}#2")   # 19:00 -> 18:00 (#4)
        overflow = self.pinned(f"{D(2)}#3")  # 21:00 overflow -> 22:00 (#5)
        untouched = self.pinned(f"{D(0)}#2")  # today keeps its schedule
        result = ps.save(self.session, D(1), self.NEW, today=TODAY)
        self.assertEqual(before.pinned_window_key, f"{D(1)}#3")
        self.assertEqual(evening.pinned_window_key, f"{D(2)}#4")
        self.assertEqual(overflow.pinned_window_key, f"{D(2)}#5")
        self.assertEqual(untouched.pinned_window_key, f"{D(0)}#2")
        self.assertEqual(result, {"moved": 3, "unpinned": 0})

    def test_fewer_windows_unpins_what_does_not_fit(self):
        a = self.pinned(f"{D(1)}#0")
        b = self.pinned(f"{D(1)}#1")
        result = ps.save(self.session, D(1), ["12:00"], today=TODAY)
        self.assertEqual(result, {"moved": 1, "unpinned": 1})
        self.assertEqual(sorted([a.pinned_window_key, b.pinned_window_key]), ["", f"{D(1)}#0"])

    def test_saving_replaces_later_changes_and_cancel_moves_pins_back(self):
        ps.save(self.session, D(5), ["12:00"], today=TODAY)
        ps.save(self.session, D(2), self.NEW, today=TODAY)
        rows = self.session.query(PostingSchedule).all()
        self.assertEqual([r.effective_from for r in rows], [D(2)])

        p = self.pinned(f"{D(3)}#3")  # 14:00 on the new schedule
        self.assertEqual(ps.cancel(self.session, D(2), today=TODAY), {"moved": 1, "unpinned": 0})
        self.assertEqual(p.pinned_window_key, f"{D(3)}#1")  # back to 13:00
        self.assertIsNone(ps.cancel(self.session, D(2), today=TODAY))

    def test_a_started_change_cannot_be_cancelled(self):
        ps.save(self.session, D(1), self.NEW, today=TODAY)
        self.assertIsNone(ps.cancel(self.session, D(1), today=D(1)))

    def test_rerun_slots_are_stored_by_time_and_must_be_windows(self):
        ps.save(self.session, D(1), self.NEW, today=TODAY, reruns=["02:00", "22:00", "23:00"])
        rows = ps._load_rows(self.session)
        self.assertEqual(rows[0][2], ["02:00", "22:00"])
        with mock.patch.object(ps, "_rows", return_value=rows):
            self.assertEqual(ps.rerun_indices_for_date(D(1)), frozenset({0, 5}))
            self.assertEqual(ps.rerun_indices_for_date(TODAY), frozenset())
            self.assertTrue(ps.is_rerun_slot(f"{D(3)}#5"))
            self.assertFalse(ps.is_rerun_slot(f"{D(3)}#4"))

    def test_rerun_slots_alone_can_change_today(self):
        # Same times as today, only the rerun flags differ: nothing renumbers.
        ps.save(self.session, TODAY, self.OLD, today=TODAY, reruns=["19:00"])
        self.assertEqual(ps._load_rows(self.session)[0][2], ["19:00"])


if __name__ == "__main__":
    unittest.main()
