"""Pure placement engine: maps queued posts onto posting windows.

Two modes, selected by whether a :class:`PlacementContext` is supplied:

- ``ctx=None`` — the original FIFO behavior, byte-for-byte: pins claim their
  window, everything else fills the remaining slots in ``created_at`` order.
- ``ctx=PlacementContext`` — scored placement. Windows are walked in
  chronological order; each takes the highest-scoring queued post that passes
  every gate (same-source spacing, same-channel spacing, facet variety),
  relaxing gates softest-first rather than ever leaving a window empty while
  eligible posts exist. Expired timely posts are never placed.

Everything here is pure — no I/O, no ORM queries, no settings reads. The
context carries resolved settings plus history, so the same inputs always
produce the same plan. That matters beyond testability: three schedulers
(dashboard thread, GitHub Actions cron, Fly worker) each compute the head for
a due window independently and share nothing but the database, so any
divergence between their answers means nondeterministic content. Hence the
rules at the bottom of this docstring:

- No randomness anywhere in scoring.
- Every comparison ends in a stable ``id`` tiebreak.
- Scores are rounded to 6 decimals before comparing: ``0.5 ** x`` routes
  through platform libm and can differ by an ULP between an ARM laptop and an
  x86 CI runner; addition and multiplication are bit-identical under IEEE 754
  but ``pow`` is not.

The engine reads only ``id`` and ``pinned_window_key`` off the post objects;
everything else comes from the per-post :class:`PostFacts` in the context, so
tests can drive it with trivial fakes.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

# Shelf-life vocabulary. Empty string means "unset" and resolves through the
# candidate -> category-default -> evergreen chain in the context builder.
SHELF_BREAKING = "breaking"
SHELF_TIMELY = "timely"
SHELF_EVERGREEN = "evergreen"
SHELF_LIVES = (SHELF_BREAKING, SHELF_TIMELY, SHELF_EVERGREEN)

# Gate names, relaxed softest-first when nothing passes. ``source_half``
# halves the same-source gap rather than dropping it, so step 3 still keeps
# sibling clips of one video apart — just less far apart. Expiry is never
# relaxed: a stale news clip should surface for the operator, not air late.
RELAXATION_LADDER: tuple[frozenset[str], ...] = (
    frozenset(),
    frozenset({"variety"}),
    frozenset({"variety", "channel"}),
    frozenset({"variety", "channel", "source_half"}),
    frozenset({"variety", "channel", "source"}),
)


@dataclass(frozen=True)
class PlacementSettings:
    """The ``scheduler.placement`` block, resolved once per plan build.

    ``load_settings()`` re-parses YAML on every call, and placement runs the
    queue against a 60-day window horizon — settings must never be read
    inside the walk.
    """

    same_source_days: int = 10
    same_channel_days: int = 3
    max_facet_overlap: float = 0.34
    lookback_windows: int = 3
    # Whether the ladder may relax the same-source gates (its last two steps).
    # False when a rerun fallback exists: better to leave the window to a
    # library re-air than to pack sibling clips of one video back-to-back.
    relax_source_gates: bool = True
    urgency_max: float = 10.0
    patience_per_day: float = 1.0
    variety_penalty_max: float = 4.0
    repost_penalty: float = 0.5
    # half-life (days) per shelf life; evergreen never decays or expires.
    half_life_breaking: float = 1.0
    half_life_timely: float = 7.0
    expire_after_half_lives: float = 3.0

    def half_life_days(self, shelf_life: str) -> float | None:
        """Decay half-life for a resolved shelf life; None = evergreen."""
        if shelf_life == SHELF_BREAKING:
            return self.half_life_breaking
        if shelf_life == SHELF_TIMELY:
            return self.half_life_timely
        return None


@dataclass(frozen=True)
class PostFacts:
    """Everything placement needs to know about one queued post.

    Precomputed by the context builder so the walk never touches the ORM.
    ``half_life_days`` is the fully resolved shelf life (post override ->
    candidate tag -> category default -> evergreen); None means evergreen.
    ``channel_exempt`` marks synthetic channels (operator uploads, pasted
    URLs) that aren't editorial sources — spacing "the Uploads channel" would
    ration the operator's own hand-picked material for no reason.
    """

    post_id: int
    candidate_pk: int | None = None
    channel_pk: int | None = None
    facets: frozenset[str] = frozenset()
    half_life_days: float | None = None
    content_date: dt.date | None = None
    queued_date: dt.date | None = None
    is_repost: bool = False
    channel_exempt: bool = False


@dataclass
class PlacementDecision:
    """Why a window got the post it got — feeds the calendar's markers."""

    post_id: int
    relax_step: int | None  # None = pinned (gates bypassed)
    score: float | None = None
    parts: dict[str, float] = field(default_factory=dict)
    pinned: bool = False


@dataclass
class PlacementContext:
    """Resolved settings + history, plus the mutable plan-so-far state.

    History (aired dates, recent facet trail) is what the database said when
    the context was built; the plan-so-far state accumulates as the walk
    places posts, so a clip placed at Tuesday's window correctly blocks its
    sibling from Wednesday's. Build a fresh context per plan — the walk
    mutates it.
    """

    settings: PlacementSettings
    facts: dict[int, PostFacts] = field(default_factory=dict)
    # History + plan-so-far, merged: every known air date per candidate/channel.
    # Future dates are legal (a pin later this week blocks siblings before it).
    candidate_air_dates: dict[int, list[dt.date]] = field(default_factory=dict)
    channel_air_dates: dict[int, list[dt.date]] = field(default_factory=dict)
    # Facet sets of recently published posts (oldest -> newest); placements
    # append here so variety sees the plan as it grows.
    facet_trail: list[frozenset[str]] = field(default_factory=list)
    # window_key -> decision, for the calendar's relaxation markers.
    decisions: dict[str, PlacementDecision] = field(default_factory=dict)

    def facts_for(self, post_id: int) -> PostFacts:
        return self.facts.get(post_id) or PostFacts(post_id=post_id)

    # -- gates ---------------------------------------------------------------

    def expired(self, post_id: int, day: dt.date) -> bool:
        f = self.facts_for(post_id)
        if f.half_life_days is None or f.content_date is None:
            return False
        age = (day - f.content_date).days
        return age > f.half_life_days * self.settings.expire_after_half_lives

    def gates_pass(self, post_id: int, day: dt.date,
                   relaxed: frozenset[str]) -> bool:
        f = self.facts_for(post_id)
        s = self.settings

        if "source" not in relaxed and f.candidate_pk is not None:
            gap = s.same_source_days
            if "source_half" in relaxed:
                gap = (gap + 1) // 2
            if gap > 0 and self._within(
                    self.candidate_air_dates.get(f.candidate_pk, ()), day, gap):
                return False

        if ("channel" not in relaxed and f.channel_pk is not None
                and not f.channel_exempt):
            if s.same_channel_days > 0 and self._within(
                    self.channel_air_dates.get(f.channel_pk, ()),
                    day, s.same_channel_days):
                return False

        if "variety" not in relaxed and f.facets:
            if self._max_recent_overlap(f.facets) > s.max_facet_overlap:
                return False
        return True

    @staticmethod
    def _within(dates, day: dt.date, gap_days: int) -> bool:
        """Whether any known air date is closer than ``gap_days`` to ``day``.

        Absolute distance, so a pin later this week blocks a sibling clip
        placed just before it, not only just after.
        """
        return any(abs((day - d).days) < gap_days for d in dates)

    def _max_recent_overlap(self, facets: frozenset[str]) -> float:
        trail = self.facet_trail[-self.settings.lookback_windows:]
        return max((_jaccard(facets, prev) for prev in trail), default=0.0)

    # -- score ---------------------------------------------------------------

    def score(self, post_id: int, day: dt.date) -> tuple[float, dict[str, float]]:
        f = self.facts_for(post_id)
        s = self.settings

        urgency = 0.0
        if f.half_life_days is not None and f.content_date is not None:
            age = max(0, (day - f.content_date).days)
            urgency = s.urgency_max * 0.5 ** (age / f.half_life_days)

        # The anti-starvation term: grows linearly while urgency decays
        # exponentially, so fresh news jumps the line and everything
        # eventually gets out.
        patience = 0.0
        if f.queued_date is not None:
            patience = s.patience_per_day * max(0, (day - f.queued_date).days)

        variety_penalty = 0.0
        if f.facets:
            trail = self.facet_trail[-self.settings.lookback_windows:]
            variety_penalty = s.variety_penalty_max * sum(
                _jaccard(f.facets, prev) * 0.5 ** j
                for j, prev in enumerate(reversed(trail))
            )

        repost_penalty = s.repost_penalty if f.is_repost else 0.0

        parts = {
            "urgency": round(urgency, 6),
            "patience": round(patience, 6),
            "variety_penalty": round(variety_penalty, 6),
            "repost_penalty": round(repost_penalty, 6),
        }
        # Rounded so a libm ULP difference between runners can't flip an order.
        total = round(urgency + patience - variety_penalty - repost_penalty, 6)
        return total, parts

    # -- plan-so-far bookkeeping ----------------------------------------------

    def note_air_dates(self, post_id: int, day: dt.date) -> None:
        """Record that this post occupies a window on ``day`` (pin or placement),
        so source/channel gates elsewhere in the plan see it."""
        f = self.facts_for(post_id)
        if f.candidate_pk is not None:
            self.candidate_air_dates.setdefault(f.candidate_pk, []).append(day)
        if f.channel_pk is not None and not f.channel_exempt:
            self.channel_air_dates.setdefault(f.channel_pk, []).append(day)

    def record(self, post_id: int, window_key: str, *,
               relax_step: int | None, score: float | None = None,
               parts: dict[str, float] | None = None,
               pinned: bool = False) -> None:
        self.facet_trail.append(self.facts_for(post_id).facets)
        self.decisions[window_key] = PlacementDecision(
            post_id=post_id, relax_step=relax_step, score=score,
            parts=parts or {}, pinned=pinned,
        )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def window_key_date(key: str) -> dt.date | None:
    """``YYYY-MM-DD#N`` -> date. Day granularity is all the gates need, which
    is why placement never touches real timestamps."""
    day_s, sep, _ = (key or "").partition("#")
    if not sep:
        return None
    try:
        return dt.date.fromisoformat(day_s)
    except ValueError:
        return None


def _fifo_assign(posts: list, window_keys: list[str]) -> list:
    """The original behavior, unchanged: pins first, then FIFO into gaps."""
    assignment: list = [None] * len(window_keys)
    key_index = {k: i for i, k in enumerate(window_keys)}
    placed: set[int] = set()

    # Pins that target a known upcoming window (first pin wins on conflict).
    for p in posts:
        pin = (p.pinned_window_key or "").strip()
        if not pin or pin not in key_index:
            continue
        i = key_index[pin]
        if assignment[i] is None:
            assignment[i] = p
            placed.add(p.id)

    remaining = [p for p in posts if p.id not in placed]
    ri = 0
    for i in range(len(assignment)):
        if assignment[i] is None and ri < len(remaining):
            assignment[i] = remaining[ri]
            ri += 1
    return assignment


def choose_for_window(ctx: PlacementContext, remaining: list,
                      window_key: str) -> tuple[object | None, int | None, float | None, dict]:
    """Best eligible post for one window: ``(post, relax_step, score, parts)``.

    Walks the relaxation ladder until a pool survives the gates, then takes
    the highest score (ties break on lower id). With ``relax_source_gates``
    the last ladder step drops every relaxable gate, so ``(None, ...)`` means
    the queue is empty or everything left has expired. Without it the ladder
    stops before the source steps, and ``None`` can also mean "only sibling
    clips remain" — the window is left for the rerun fallback instead.
    """
    day = window_key_date(window_key)
    if day is None:
        return None, None, None, {}
    live = [p for p in remaining if not ctx.expired(p.id, day)]
    ladder = (RELAXATION_LADDER if ctx.settings.relax_source_gates
              else RELAXATION_LADDER[:3])
    for step, relaxed in enumerate(ladder):
        pool = [p for p in live if ctx.gates_pass(p.id, day, relaxed)]
        if not pool:
            continue
        best = None
        best_key: tuple[float, int] | None = None
        best_score, best_parts = None, {}
        for p in pool:
            score, parts = ctx.score(p.id, day)
            key = (score, -p.id)
            if best_key is None or key > best_key:
                best, best_key, best_score, best_parts = p, key, score, parts
        return best, step, best_score, best_parts
    return None, None, None, {}


def assign_posts_to_windows(posts: list, window_keys: list[str], *,
                            ctx: PlacementContext | None = None) -> list:
    """Map queued posts onto window keys.

    Returns a list parallel to ``window_keys``. Earlier windows may stay empty
    when a post is pinned to a later slot (both modes) or when everything left
    in the queue has expired (scored mode).

    Pins are absolute in both modes: an operator drag bypasses every gate and
    every score. In scored mode a pin still shapes the rest of the plan — its
    air dates feed the source/channel gates (in both directions, so a sibling
    clip is kept away from a pin later in the week too) and its facets join
    the variety trail as the walk passes it.
    """
    if ctx is None:
        return _fifo_assign(posts, window_keys)

    assignment: list = [None] * len(window_keys)
    key_index = {k: i for i, k in enumerate(window_keys)}
    placed: set[int] = set()

    for p in posts:
        pin = (p.pinned_window_key or "").strip()
        if not pin or pin not in key_index:
            continue
        i = key_index[pin]
        if assignment[i] is None:
            assignment[i] = p
            placed.add(p.id)
            day = window_key_date(pin)
            if day is not None:
                ctx.note_air_dates(p.id, day)

    remaining = [p for p in posts if p.id not in placed]
    for i, key in enumerate(window_keys):
        if assignment[i] is not None:
            ctx.record(assignment[i].id, key, relax_step=None, pinned=True)
            continue
        pick, step, score, parts = choose_for_window(ctx, remaining, key)
        if pick is None:
            continue
        assignment[i] = pick
        remaining.remove(pick)
        day = window_key_date(key)
        if day is not None:
            ctx.note_air_dates(pick.id, day)
        ctx.record(pick.id, key, relax_step=step, score=score, parts=parts)
    return assignment
