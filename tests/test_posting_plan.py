"""Tests for the posting plan (app/posting_plan.py): three new clips a day,
plus a rerun for every 60 archive posts, up to six a day.

Run with:  python -m unittest tests.test_posting_plan -v

The rule and the times are pure. ``ensure`` runs against an in-memory SQLite
database and writes ordinary posting schedules (app/posting_schedule.py).
"""
from __future__ import annotations

import datetime as dt
import json
import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import posting_plan, posting_schedule
from app.models import Base, Candidate, Channel, PostingSchedule, ThreadsPost

TODAY = dt.date(2026, 11, 19)
TOMORROW = TODAY + dt.timedelta(days=1)


class RuleTest(unittest.TestCase):
    def test_a_rerun_for_every_sixty_posts_up_to_three(self):
        cases = {0: 3, 59: 3, 60: 4, 119: 4, 120: 5, 179: 5, 180: 6, 407: 6, 10_000: 6}
        for archive, per_day in cases.items():
            with self.subTest(archive=archive):
                self.assertEqual(posting_plan.per_day(archive), per_day)
                self.assertEqual(posting_plan.reruns_for(archive), per_day - 3)

    def test_progress_toward_the_next_rerun(self):
        p = posting_plan.progress(37)
        self.assertEqual((p["per_day"], p["next_at"], p["to_go"], p["maxed"]), (3, 60, 23, False))
        self.assertAlmostEqual(p["share"], 37 / 60)
        p = posting_plan.progress(130)
        self.assertEqual((p["per_day"], p["next_at"], p["to_go"]), (5, 180, 50))
        self.assertAlmostEqual(p["share"], 10 / 60)
        p = posting_plan.progress(407)
        self.assertEqual((p["per_day"], p["maxed"], p["next_at"], p["to_go"]), (6, True, None, None))

    def test_tiers(self):
        self.assertEqual([(t["posts"], t["per_day"]) for t in posting_plan.tiers()],
                         [(0, 3), (60, 4), (120, 5), (180, 6)])


class TimesTest(unittest.TestCase):
    def test_waking_hours_run_from_seven_to_ten(self):
        self.assertEqual(posting_plan.times(3, "waking"), ["07:00", "14:30", "22:00"])
        self.assertEqual(posting_plan.times(4, "waking"), ["07:00", "12:00", "17:00", "22:00"])
        self.assertEqual(posting_plan.times(5, "waking"), ["07:00", "10:45", "14:30", "18:15", "22:00"])
        self.assertEqual(posting_plan.times(6, "waking"),
                         ["07:00", "10:00", "13:00", "16:00", "19:00", "22:00"])

    def test_around_the_clock_is_evenly_spaced(self):
        self.assertEqual(posting_plan.times(3, "always"), ["05:00", "13:00", "21:00"])
        self.assertEqual(posting_plan.times(4, "always"), ["01:00", "07:00", "13:00", "19:00"])
        self.assertEqual(posting_plan.times(6, "always"),
                         ["01:00", "05:00", "09:00", "13:00", "17:00", "21:00"])
        five = posting_plan.times(5, "always")
        self.assertEqual(len(set(five)), 5)
        self.assertIn("13:00", five)

    def test_every_layout_respects_the_spacing_floor(self):
        for mode in posting_plan.MODES:
            for n in range(3, 7):
                with self.subTest(mode=mode, n=n):
                    self.assertEqual(posting_schedule.problems(posting_plan.times(n, mode), 90), [])

    def test_reruns_take_the_quietest_times(self):
        self.assertEqual(posting_plan.plan(407, "always")["reruns"], ["01:00", "05:00", "09:00"])
        self.assertEqual(posting_plan.plan(407, "waking")["reruns"], ["07:00", "10:00", "22:00"])
        self.assertEqual(posting_plan.plan(60, "waking")["reruns"], ["07:00"])
        self.assertEqual(posting_plan.plan(10, "waking")["reruns"], [])


class EnsureTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        posting_schedule.invalidate()
        ch = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.session.add(ch)
        self.session.flush()
        self.candidate = Candidate(video_id="pp1", channel_pk=ch.id, title="t", url="u")
        self.session.add(self.candidate)
        self.session.flush()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()
        posting_schedule.invalidate()

    def posts(self, n: int, rerun_of: int | None = None) -> None:
        for _ in range(n):
            self.session.add(ThreadsPost(candidate_pk=self.candidate.id, caption="c",
                                         status="published", repost_of_post_pk=rerun_of))
        self.session.flush()

    def schedules(self) -> list[tuple[dt.date, list[str], list[str]]]:
        rows = self.session.execute(select(PostingSchedule).order_by(PostingSchedule.effective_from)).scalars()
        return [(r.effective_from, json.loads(r.windows), json.loads(r.rerun_slots)) for r in rows]

    def test_hands_off_until_a_mode_is_chosen(self):
        self.posts(200)
        self.assertIsNone(posting_plan.load_mode(self.session))
        self.assertIsNone(posting_plan.ensure(self.session, today=TODAY))
        self.assertEqual(self.schedules(), [])

    def test_choosing_a_mode_writes_tomorrows_schedule_once(self):
        self.posts(10)
        posting_plan.save_mode(self.session, "waking")
        out = posting_plan.ensure(self.session, today=TODAY)
        self.assertEqual(out["effective_from"], TOMORROW)
        self.assertEqual(self.schedules(), [(TOMORROW, ["07:00", "14:30", "22:00"], [])])
        self.assertIsNone(posting_plan.ensure(self.session, today=TODAY))

    def test_the_archive_reaching_sixty_adds_a_rerun(self):
        posting_plan.save_mode(self.session, "always")
        self.posts(59)
        self.posts(30, rerun_of=1)          # reruns don't count toward the archive
        posting_plan.ensure(self.session, today=TODAY)
        self.assertEqual(self.schedules()[-1][1], ["05:00", "13:00", "21:00"])
        self.posts(1)
        later = TODAY + dt.timedelta(days=3)
        out = posting_plan.ensure(self.session, today=later)
        self.assertEqual(out["per_day"], 4)
        last = self.schedules()[-1]
        self.assertEqual(last, (later + dt.timedelta(days=1),
                                ["01:00", "07:00", "13:00", "19:00"], ["01:00"]))

    def test_switching_modes_replaces_tomorrows_schedule(self):
        self.posts(200)
        posting_plan.save_mode(self.session, "always")
        posting_plan.ensure(self.session, today=TODAY)
        posting_plan.save_mode(self.session, "waking")
        posting_plan.ensure(self.session, today=TODAY)
        self.assertEqual(self.schedules(), [(TOMORROW, ["07:00", "10:00", "13:00", "16:00", "19:00", "22:00"],
                                             ["07:00", "10:00", "22:00"])])

    def test_only_offered_modes(self):
        with self.assertRaises(ValueError):
            posting_plan.save_mode(self.session, "sometimes")


if __name__ == "__main__":
    unittest.main()
