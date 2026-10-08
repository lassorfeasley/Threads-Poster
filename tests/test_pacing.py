"""Tests for paced placement (app/placement.py) and its scheduler inputs.

Run with:  python -m unittest tests.test_pacing -v

Same fakes as test_placement: the engine reads only ``id`` and
``pinned_window_key`` off posts, and everything else comes from the context.
"""
from __future__ import annotations

import datetime as dt
import unittest

from app import scheduler
from app.placement import PlacementSettings, assign_posts_to_windows
from tests.test_placement import DAY0, FakePost, _key, facts, ids, make_ctx

SLOTS = 6


def day_keys(days: int, slots: int = SLOTS) -> list[str]:
    return [_key(d, i) for d in range(days) for i in range(slots)]


def paced_ctx(post_facts, pace: float, *, days: int = 4, slots: int = SLOTS, **kw):
    settings = PlacementSettings(pace_per_day=pace, **kw)
    ctx = make_ctx(post_facts, settings=settings)
    ctx.pace_windows = {DAY0 + dt.timedelta(days=d): tuple(range(slots))
                        for d in range(days)}
    return ctx


def per_day(assignment, slots: int = SLOTS) -> list[int]:
    return [sum(p is not None for p in assignment[i:i + slots])
            for i in range(0, len(assignment), slots)]


class TestDuePattern(unittest.TestCase):
    def due(self, pace: float, days: int = 4) -> list[bool]:
        ctx = paced_ctx({}, pace, days=days)
        return [ctx.pace_due(k) for k in day_keys(days)]

    def test_whole_rate_is_exact_per_day(self):
        pattern = self.due(3)
        self.assertEqual(per_day([True if d else None for d in pattern]), [3, 3, 3, 3])

    def test_whole_rate_is_spread_through_the_day(self):
        day = self.due(3, days=1)
        # Never two due windows back to back at half the slots.
        self.assertFalse(any(a and b for a, b in zip(day, day[1:])))

    def test_fractional_rate_carries_across_days(self):
        pattern = self.due(4.5)
        counts = per_day([True if d else None for d in pattern])
        self.assertEqual(sum(counts), 18)
        self.assertTrue(set(counts) <= {4, 5})

    def test_rate_above_slots_fills_every_window(self):
        self.assertTrue(all(self.due(9)))

    def test_rerun_slots_and_unknown_windows_are_never_due(self):
        ctx = paced_ctx({}, 6)
        ctx.pace_windows[DAY0] = (0, 1, 2, 4, 5)
        self.assertFalse(ctx.pace_due(_key(0, 3)))
        self.assertFalse(ctx.pace_due(_key(0, 6)))  # overflow window

    def test_open_keys_are_always_due(self):
        ctx = paced_ctx({}, 0.0)
        ctx.pace_open = frozenset({_key(0, 2)})
        self.assertTrue(ctx.pace_due(_key(0, 2)))


class TestPacedWalk(unittest.TestCase):
    def test_holds_new_clips_to_the_pace(self):
        posts = [FakePost(i) for i in range(1, 21)]
        ctx = paced_ctx({i: facts(i, cand=i) for i in range(1, 21)}, 3)
        out = assign_posts_to_windows(posts, day_keys(4), ctx=ctx)
        self.assertEqual(per_day(out), [3, 3, 3, 3])

    def test_without_pace_every_window_fills(self):
        posts = [FakePost(i) for i in range(1, 21)]
        ctx = make_ctx({i: facts(i, cand=i) for i in range(1, 21)})
        out = assign_posts_to_windows(posts, day_keys(2), ctx=ctx)
        self.assertEqual(per_day(out), [6, 6])

    def test_short_queue_airs_soon_then_leaves_the_rest_to_reruns(self):
        posts = [FakePost(i) for i in range(1, 5)]
        ctx = paced_ctx({i: facts(i, cand=i) for i in range(1, 5)}, 3, days=3)
        out = assign_posts_to_windows(posts, day_keys(3), ctx=ctx)
        self.assertEqual(per_day(out), [3, 1, 0])

    def test_newly_cut_timely_clip_goes_ahead_of_evergreen_backlog(self):
        # Evergreen clips have waited two days; a timely clip cut today still
        # takes the first due window.
        posts = [FakePost(i) for i in (1, 2, 3)]
        f = {i: facts(i, cand=i, queued_days_ago=2) for i in (1, 2)}
        f[3] = facts(3, cand=3, half_life=7.0)
        ctx = paced_ctx(f, 3, days=1)
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertEqual([p for p in ids(out) if p][0], 3)

    def test_blocked_due_window_carries_to_the_next(self):
        # Only clip is a sibling of something aired yesterday: blocked by the
        # 2-day floor today, it airs at the first window it can.
        posts = [FakePost(1)]
        ctx = paced_ctx({1: facts(1, cand=7)}, 3, days=3,
                        source_floor_days=2)
        ctx.candidate_air_dates = {7: [DAY0 - dt.timedelta(days=1)]}
        keys = day_keys(3)
        out = assign_posts_to_windows(posts, keys, ctx=ctx)
        placed = [k for k, p in zip(keys, out) if p is not None]
        self.assertEqual(placed, [_key(1, 0)])

    def test_urgent_clip_jumps_the_pace(self):
        # Pace 0.0001: nothing is ever due, yet a clip expiring tomorrow airs.
        posts = [FakePost(1), FakePost(2)]
        f = {1: facts(1, cand=1, half_life=1.0, content_days_ago=2),
             2: facts(2, cand=2)}
        ctx = paced_ctx(f, 0.0001, days=1)
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertEqual([p for p in ids(out) if p], [1])

    def test_re_airs_ignore_the_pace(self):
        posts = [FakePost(1)]
        ctx = paced_ctx({1: facts(1, cand=1, repost=True)}, 0.0001, days=1)
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertEqual(ids(out)[0], 1)

    def test_new_pins_spend_the_pace(self):
        # Two pinned new clips on day 0 use up most of a pace of 3.
        posts = [FakePost(1, pin=_key(0, 0)), FakePost(2, pin=_key(0, 1))] + \
                [FakePost(i) for i in range(3, 10)]
        ctx = paced_ctx({i: facts(i, cand=i) for i in range(1, 10)}, 3, days=1)
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertLessEqual(sum(p is not None for p in out), 4)

    def test_open_key_lifts_the_hold(self):
        posts = [FakePost(1)]
        ctx = paced_ctx({1: facts(1, cand=1)}, 0.0001, days=1)
        ctx.pace_open = frozenset({_key(0, 4)})
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertEqual(ids(out)[4], 1)


class TestSiblingSpacing(unittest.TestCase):
    def test_floor_is_hard(self):
        # Four clips of one video, plenty of pace: never within 2 days.
        posts = [FakePost(i) for i in range(1, 5)]
        ctx = paced_ctx({i: facts(i, cand=9) for i in range(1, 5)}, 6, days=8,
                        source_floor_days=2)
        keys = day_keys(8)
        out = assign_posts_to_windows(posts, keys, ctx=ctx)
        days = [scheduler.window_key_date(k) for k, p in zip(keys, out) if p]
        self.assertEqual(len(days), 4)
        self.assertTrue(all((b - a).days >= 2 for a, b in zip(days, days[1:])))

    def test_siblings_interleave_with_other_videos(self):
        # Video 9's three clips win every tie on id, so only the same-video
        # penalty can keep them from airing back to back.
        siblings = (1, 2, 3)
        posts = [FakePost(i) for i in (*siblings, 10, 11, 12, 13, 14, 15)]
        f = {i: facts(i, cand=9) for i in siblings}
        f.update({i: facts(i, cand=i) for i in range(10, 16)})
        ctx = paced_ctx(f, 2, days=5, slots=2, source_floor_days=1)
        out = assign_posts_to_windows(posts, day_keys(5, slots=2), ctx=ctx)
        order = [p for p in ids(out) if p]
        self.assertEqual(len(order), 9)
        sibling_runs = sum(1 for a, b in zip(order, order[1:])
                           if a in siblings and b in siblings)
        self.assertEqual(sibling_runs, 0)

    def test_fair_gap_spreads_a_video_over_the_queue(self):
        ctx = paced_ctx({}, 3, source_floor_days=2, same_source_days=10)
        ctx.unplaced_paced, ctx.unplaced_by_candidate = 24, {9: 2, 8: 8}
        self.assertEqual(ctx.sibling_gap_days(9), 4.0)   # 8 days of queue / 2 clips
        self.assertEqual(ctx.sibling_gap_days(8), 2.0)   # floor
        ctx.unplaced_paced = 90
        self.assertEqual(ctx.sibling_gap_days(9), 10.0)  # capped at same_source_days

    def test_penalty_shows_in_the_score_parts(self):
        ctx = paced_ctx({1: facts(1, cand=9)}, 3)
        ctx.unplaced_paced, ctx.unplaced_by_candidate = 60, {9: 1}
        ctx.candidate_air_dates = {9: [DAY0 - dt.timedelta(days=5)]}
        _, parts = ctx.score(1, DAY0)
        self.assertAlmostEqual(parts["sibling_penalty"], 3.0)


class TestDailyFloor(unittest.TestCase):
    def test_floor_beats_sibling_spacing(self):
        # One video, six clips: spacing alone would air one every other day.
        posts = [FakePost(i) for i in range(1, 7)]
        ctx = paced_ctx({i: facts(i, cand=9) for i in range(1, 7)}, 3, days=3,
                        source_floor_days=2, pace_min_per_day=3)
        out = assign_posts_to_windows(posts, day_keys(3), ctx=ctx)
        self.assertEqual(per_day(out), [3, 3, 0])

    def test_floor_counts_what_already_aired_today(self):
        posts = [FakePost(i) for i in range(1, 7)]
        ctx = paced_ctx({i: facts(i, cand=9) for i in range(1, 7)}, 3, days=2,
                        source_floor_days=2, pace_min_per_day=3)
        ctx.pace_done = {DAY0: 2}
        out = assign_posts_to_windows(posts, day_keys(2), ctx=ctx)
        self.assertEqual(per_day(out)[0], 1)

    def test_floor_stops_when_the_queue_runs_out(self):
        posts = [FakePost(1), FakePost(2)]
        ctx = paced_ctx({1: facts(1, cand=1), 2: facts(2, cand=2)}, 3, days=2,
                        pace_min_per_day=3)
        out = assign_posts_to_windows(posts, day_keys(2), ctx=ctx)
        self.assertEqual(per_day(out), [2, 0])

    def test_without_floor_spacing_holds(self):
        posts = [FakePost(i) for i in range(1, 7)]
        ctx = paced_ctx({i: facts(i, cand=9) for i in range(1, 7)}, 3, days=3,
                        source_floor_days=2)
        out = assign_posts_to_windows(posts, day_keys(3), ctx=ctx)
        self.assertEqual(per_day(out), [1, 0, 1])


class TestDailyCeiling(unittest.TestCase):
    def test_ceiling_holds_even_with_credit_and_windows_left(self):
        posts = [FakePost(i) for i in range(1, 13)]
        ctx = paced_ctx({i: facts(i, cand=i) for i in range(1, 13)}, 6, days=2,
                        pace_max_per_day=4)
        out = assign_posts_to_windows(posts, day_keys(2), ctx=ctx)
        self.assertEqual(per_day(out), [4, 4])

    def test_ceiling_beats_urgency(self):
        posts = [FakePost(i) for i in range(1, 7)]
        f = {i: facts(i, cand=i, half_life=1.0, content_days_ago=2)
             for i in range(1, 7)}
        ctx = paced_ctx(f, 3, days=1, pace_max_per_day=3)
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertEqual(per_day(out), [3])

    def test_ceiling_counts_what_already_aired_today(self):
        posts = [FakePost(i) for i in range(1, 7)]
        ctx = paced_ctx({i: facts(i, cand=i) for i in range(1, 7)}, 4, days=1,
                        pace_max_per_day=4)
        ctx.pace_done = {DAY0: 3}
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertEqual(per_day(out), [1])

    def test_reopened_window_is_exempt(self):
        posts = [FakePost(i) for i in range(1, 7)]
        ctx = paced_ctx({i: facts(i, cand=i) for i in range(1, 7)}, 4, days=1,
                        pace_max_per_day=4)
        ctx.pace_done = {DAY0: 4}
        ctx.pace_open = frozenset({day_keys(1)[-1]})
        out = assign_posts_to_windows(posts, day_keys(1), ctx=ctx)
        self.assertEqual(per_day(out), [1])

    def test_share_ceiling_scales_with_windows(self):
        posts = [FakePost(i) for i in range(1, 13)]
        ctx = paced_ctx({i: facts(i, cand=i) for i in range(1, 13)}, 6, days=2,
                        pace_max_share=0.5)
        out = assign_posts_to_windows(posts, day_keys(2), ctx=ctx)
        self.assertEqual(per_day(out), [3, 3])

    def test_tighter_ceiling_wins(self):
        ctx = paced_ctx({}, 6, pace_max_share=0.5, pace_max_per_day=2)
        self.assertEqual(ctx.day_cap(DAY0), 2)
        ctx = paced_ctx({}, 6, pace_max_share=0.5, pace_max_per_day=4)
        self.assertEqual(ctx.day_cap(DAY0), 3)


class TestPaceRate(unittest.TestCase):
    CFG = {"fixed": None, "spread_days": 7, "min_per_day": 3}

    def test_capped_at_max_per_day(self):
        self.assertEqual(scheduler.pace_per_day({**self.CFG, "max_per_day": 4}, 37), 4.0)
        self.assertEqual(scheduler.pace_per_day(
            {**self.CFG, "max_per_day": 4, "fixed": 6}, 0), 4)

    def test_capped_at_share_of_windows(self):
        cfg = {**self.CFG, "max_per_day": 4, "max_share": 0.5}
        self.assertEqual(scheduler.pace_per_day(cfg, 46, windows=6), 3.0)
        self.assertEqual(scheduler.pace_per_day(cfg, 46, windows=10), 4.0)

    def test_deep_queue_spreads_over_a_week(self):
        self.assertEqual(scheduler.pace_per_day(self.CFG, 35), 5.0)

    def test_short_queue_airs_at_the_floor(self):
        self.assertEqual(scheduler.pace_per_day(self.CFG, 6), 3.0)

    def test_rounded_to_quarters(self):
        self.assertEqual(scheduler.pace_per_day(self.CFG, 33), 4.75)

    def test_fixed(self):
        self.assertEqual(scheduler.pace_per_day({**self.CFG, "fixed": 3.5}, 99), 3.5)


class TestTimelinessClock(unittest.TestCase):
    CUT = dt.date(2026, 10, 1)

    def test_old_video_starts_one_half_life_old(self):
        clock = scheduler._timeliness_clock(dt.date(2018, 11, 5), self.CUT, 7.0)
        self.assertEqual(clock, self.CUT - dt.timedelta(days=7))

    def test_fresh_news_keeps_its_own_date(self):
        aired = self.CUT - dt.timedelta(days=2)
        self.assertEqual(scheduler._timeliness_clock(aired, self.CUT, 7.0), aired)

    def test_evergreen_is_untouched(self):
        old = dt.date(2018, 11, 5)
        self.assertEqual(scheduler._timeliness_clock(old, self.CUT, None), old)


if __name__ == "__main__":
    unittest.main()
