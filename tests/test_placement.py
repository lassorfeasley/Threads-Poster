"""Golden tests for the pure placement engine (app/placement.py).

Run with:  python -m unittest tests.test_placement -v

The engine reads only ``id`` and ``pinned_window_key`` off post objects, so a
two-field fake drives everything; per-post facts live in the context. One test
per gate, one per relaxation step, and one asserting that ``ctx=None``
reproduces the original FIFO behavior exactly — that last one is the contract
that lets scored mode ship dark.
"""
from __future__ import annotations

import datetime as dt
import unittest

from app.placement import (
    PlacementContext,
    PlacementSettings,
    PostFacts,
    assign_posts_to_windows,
)

DAY0 = dt.date(2026, 8, 16)


def _key(day_offset: int, index: int = 0) -> str:
    return f"{(DAY0 + dt.timedelta(days=day_offset)).isoformat()}#{index}"


class FakePost:
    def __init__(self, post_id: int, pin: str = ""):
        self.id = post_id
        self.pinned_window_key = pin

    def __repr__(self):  # readable assertion failures
        return f"P{self.id}"


def make_ctx(facts: dict[int, PostFacts], *, settings: PlacementSettings | None = None,
             **history) -> PlacementContext:
    return PlacementContext(settings=settings or PlacementSettings(),
                            facts=facts, **history)


def facts(post_id: int, *, cand: int | None = None, chan: int | None = None,
          tags: set[str] | None = None, half_life: float | None = None,
          content_days_ago: int = 0, queued_days_ago: int = 0,
          repost: bool = False, exempt: bool = False) -> PostFacts:
    """Facts shorthand; day offsets are relative to DAY0 (the first window)."""
    return PostFacts(
        post_id=post_id, candidate_pk=cand, channel_pk=chan,
        facets=frozenset(tags or ()), half_life_days=half_life,
        content_date=DAY0 - dt.timedelta(days=content_days_ago),
        queued_date=DAY0 - dt.timedelta(days=queued_days_ago),
        is_repost=repost, channel_exempt=exempt,
    )


def ids(assignment) -> list[int | None]:
    return [p.id if p is not None else None for p in assignment]


class TestFifoPassthrough(unittest.TestCase):
    """ctx=None must reproduce the original behavior byte for byte."""

    def test_plain_fifo(self):
        posts = [FakePost(i) for i in (1, 2, 3)]
        keys = [_key(0), _key(0, 1), _key(1)]
        self.assertEqual(ids(assign_posts_to_windows(posts, keys)), [1, 2, 3])

    def test_pin_claims_window_leaving_earlier_open(self):
        # A single post pinned to the last window leaves earlier windows empty.
        posts = [FakePost(1, pin=_key(1))]
        keys = [_key(0), _key(0, 1), _key(1)]
        self.assertEqual(ids(assign_posts_to_windows(posts, keys)), [None, None, 1])

    def test_pins_first_then_fifo_into_gaps(self):
        posts = [FakePost(1), FakePost(2, pin=_key(0, 1)), FakePost(3)]
        keys = [_key(0), _key(0, 1), _key(1)]
        self.assertEqual(ids(assign_posts_to_windows(posts, keys)), [1, 2, 3])

    def test_first_pin_wins_on_conflict(self):
        posts = [FakePost(1, pin=_key(0)), FakePost(2, pin=_key(0))]
        keys = [_key(0), _key(0, 1)]
        # Post 2's pin loses; it falls back into FIFO flow.
        self.assertEqual(ids(assign_posts_to_windows(posts, keys)), [1, 2])

    def test_stale_pin_ignored(self):
        posts = [FakePost(1, pin="2020-01-01#0"), FakePost(2)]
        keys = [_key(0), _key(0, 1)]
        self.assertEqual(ids(assign_posts_to_windows(posts, keys)), [1, 2])


class TestGates(unittest.TestCase):
    def test_same_source_gate_defers_sibling_clip(self):
        # Two clips from the same video: the second must wait out the gate,
        # so the unrelated post takes the second window.
        posts = [FakePost(1), FakePost(2), FakePost(3)]
        ctx = make_ctx({
            1: facts(1, cand=10),
            2: facts(2, cand=10),
            3: facts(3, cand=11),
        })
        keys = [_key(0), _key(0, 1)]
        out = assign_posts_to_windows(posts, keys, ctx=ctx)
        self.assertEqual(ids(out), [1, 3])

    def test_same_source_gate_respects_history(self):
        # The candidate aired 3 days ago (gate 10) — its clip must lose to a
        # clean one even though it queued first.
        posts = [FakePost(1), FakePost(2)]
        ctx = make_ctx(
            {1: facts(1, cand=10), 2: facts(2, cand=11)},
            candidate_air_dates={10: [DAY0 - dt.timedelta(days=3)]},
        )
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [2])

    def test_same_channel_gate(self):
        # Channel 5 aired yesterday (gate 3 days): its post waits, channel 6 runs.
        posts = [FakePost(1), FakePost(2)]
        ctx = make_ctx(
            {1: facts(1, cand=10, chan=5), 2: facts(2, cand=11, chan=6)},
            channel_air_dates={5: [DAY0 - dt.timedelta(days=1)]},
        )
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [2])

    def test_synthetic_channel_exempt_from_channel_gate(self):
        posts = [FakePost(1)]
        ctx = make_ctx(
            {1: facts(1, cand=10, chan=5, exempt=True)},
            channel_air_dates={5: [DAY0 - dt.timedelta(days=1)]},
        )
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [1])
        self.assertEqual(ctx.decisions[_key(0)].relax_step, 0)

    def test_variety_gate_blocks_facet_overlap(self):
        # The feed just showed {news}; another pure-news post is over the
        # overlap threshold, so the nature post runs first.
        posts = [FakePost(1), FakePost(2)]
        ctx = make_ctx(
            {1: facts(1, cand=10, tags={"news"}), 2: facts(2, cand=11, tags={"nature"})},
            facet_trail=[frozenset({"news"})],
        )
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [2])

    def test_variety_gate_is_similarity_not_equality(self):
        # Half-alike (Jaccard 1/3 with default threshold 0.34) squeaks under
        # the gate; identical (1.0) does not.
        posts = [FakePost(1), FakePost(2)]
        ctx = make_ctx(
            {
                1: facts(1, cand=10, tags={"archival", "nature_documentary"}),
                2: facts(2, cand=11, tags={"archival", "found_footage"}),
            },
            facet_trail=[frozenset({"archival", "nature_documentary"})],
        )
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        # Post 1 is identical to the trail (blocked); post 2 overlaps 1/3.
        self.assertEqual(ids(out), [2])

    def test_expired_post_never_placed(self):
        # Breaking (half-life 1d) content from 10 days ago: expired, window
        # stays empty even though the queue is non-empty. Expiry never relaxes.
        posts = [FakePost(1)]
        ctx = make_ctx({1: facts(1, cand=10, half_life=1.0, content_days_ago=10)})
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [None])
        self.assertNotIn(_key(0), ctx.decisions)


class TestScoring(unittest.TestCase):
    def test_fresh_breaking_jumps_the_line(self):
        # An evergreen post queued 3 days ago vs breaking news queued today:
        # urgency (10) beats patience (3).
        posts = [FakePost(1), FakePost(2)]
        ctx = make_ctx({
            1: facts(1, cand=10, queued_days_ago=3),
            2: facts(2, cand=11, half_life=1.0),
        })
        out = assign_posts_to_windows(posts, [_key(0), _key(0, 1)], ctx=ctx)
        self.assertEqual(ids(out), [2, 1])

    def test_patience_eventually_beats_decayed_news(self):
        # Breaking news from 12 days ago (urgency ~0.02) loses to an evergreen
        # post that has waited 5 days (patience 5). Nothing starves.
        posts = [FakePost(1), FakePost(2)]
        ctx = make_ctx({
            1: facts(1, cand=10, half_life=7.0, content_days_ago=12),
            2: facts(2, cand=11, queued_days_ago=5),
        })
        # timely at 7d half-life, 12 days old: urgency 10*0.5^(12/7) ≈ 3.05;
        # patience 5 wins.
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [2])

    def test_repost_penalty_breaks_tie_toward_original(self):
        posts = [FakePost(1), FakePost(2)]
        ctx = make_ctx({
            1: facts(1, cand=10, repost=True),
            2: facts(2, cand=11),
        })
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [2])

    def test_exact_tie_breaks_on_lower_id(self):
        posts = [FakePost(2), FakePost(1)]  # queue order deliberately reversed
        ctx = make_ctx({1: facts(1, cand=10), 2: facts(2, cand=11)})
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [1])

    def test_scores_are_rounded(self):
        ctx = make_ctx({1: facts(1, cand=10, half_life=7.0, content_days_ago=5)})
        score, _parts = ctx.score(1, DAY0)
        self.assertEqual(score, round(score, 6))


class TestRelaxationLadder(unittest.TestCase):
    """One test per step: a single post that fails everything up to step N."""

    def _place_single(self, ctx):
        out = assign_posts_to_windows([FakePost(1)], [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [1], "ladder must fill the window")
        return ctx.decisions[_key(0)].relax_step

    def test_step0_no_relaxation(self):
        ctx = make_ctx({1: facts(1, cand=10, chan=5, tags={"news"})})
        self.assertEqual(self._place_single(ctx), 0)

    def test_step1_variety_relaxed(self):
        ctx = make_ctx(
            {1: facts(1, cand=10, tags={"news"})},
            facet_trail=[frozenset({"news"})],
        )
        self.assertEqual(self._place_single(ctx), 1)

    def test_step2_channel_relaxed(self):
        ctx = make_ctx(
            {1: facts(1, cand=10, chan=5)},
            channel_air_dates={5: [DAY0 - dt.timedelta(days=1)]},
        )
        self.assertEqual(self._place_single(ctx), 2)

    def test_step3_source_gap_halved(self):
        # Candidate aired 6 days ago: fails the 10-day gate, passes at 5.
        ctx = make_ctx(
            {1: facts(1, cand=10)},
            candidate_air_dates={10: [DAY0 - dt.timedelta(days=6)]},
        )
        self.assertEqual(self._place_single(ctx), 3)

    def test_step4_all_spacing_dropped(self):
        # Candidate aired yesterday: only the final step lets it through.
        ctx = make_ctx(
            {1: facts(1, cand=10)},
            candidate_air_dates={10: [DAY0 - dt.timedelta(days=1)]},
        )
        self.assertEqual(self._place_single(ctx), 4)


class TestPins(unittest.TestCase):
    def test_pin_bypasses_gates(self):
        # An expired, gate-violating post still airs where it was pinned.
        posts = [FakePost(1, pin=_key(0))]
        ctx = make_ctx(
            {1: facts(1, cand=10, half_life=1.0, content_days_ago=30)},
            candidate_air_dates={10: [DAY0 - dt.timedelta(days=1)]},
        )
        out = assign_posts_to_windows(posts, [_key(0)], ctx=ctx)
        self.assertEqual(ids(out), [1])
        self.assertTrue(ctx.decisions[_key(0)].pinned)

    def test_pin_shapes_gates_in_both_directions(self):
        # Post 2 is pinned to day 2; its sibling clip (same candidate) must not
        # be auto-placed on day 0 or 1 — the unrelated post 3 runs instead.
        posts = [FakePost(1), FakePost(2, pin=_key(2)), FakePost(3)]
        ctx = make_ctx({
            1: facts(1, cand=10),
            2: facts(2, cand=10),
            3: facts(3, cand=11),
        })
        keys = [_key(0), _key(1), _key(2)]
        out = assign_posts_to_windows(posts, keys, ctx=ctx)
        self.assertEqual(out[2].id, 2, "pin must hold")
        # Post 1 (sibling of the pin) is kept away from days 0-1 by the
        # absolute-distance source gate; post 3 fills one window and the other
        # is filled by post 1 only through the ladder — assert it never lands
        # clean next to its sibling.
        placed_1 = [i for i, p in enumerate(out) if p is not None and p.id == 1]
        if placed_1:
            step = ctx.decisions[keys[placed_1[0]]].relax_step
            self.assertGreaterEqual(step, 3, "sibling near a pin only via ladder")

    def test_variety_trail_includes_pinned_posts(self):
        # A pinned news post at window 0 should make a second news post lose
        # window 1 to a nature post on variety.
        posts = [FakePost(1, pin=_key(0)), FakePost(2), FakePost(3)]
        ctx = make_ctx({
            1: facts(1, cand=10, tags={"news"}),
            2: facts(2, cand=11, tags={"news"}),
            3: facts(3, cand=12, tags={"nature"}),
        })
        out = assign_posts_to_windows(posts, [_key(0), _key(0, 1)], ctx=ctx)
        self.assertEqual(ids(out), [1, 3])


class TestDecisions(unittest.TestCase):
    def test_score_breakdown_recorded(self):
        ctx = make_ctx({1: facts(1, cand=10, queued_days_ago=2)})
        assign_posts_to_windows([FakePost(1)], [_key(0)], ctx=ctx)
        d = ctx.decisions[_key(0)]
        self.assertEqual(d.post_id, 1)
        self.assertEqual(d.parts["patience"], 2.0)
        self.assertEqual(d.parts["urgency"], 0.0)


if __name__ == "__main__":
    unittest.main()
