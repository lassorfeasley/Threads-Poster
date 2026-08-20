"""Window scheduler for the Threads post queue.

Publishes the ``queued`` list at fixed daily windows (US Eastern by default):
one post per window, gated by active hours and the spacing floor. Pinned posts
claim their window. The rest fill remaining slots either FIFO (the default) or
via the scored placement engine (``scheduler.placement.mode: scored`` — see
app/placement.py), which spaces sibling clips of one source video, keeps
content facets varied, runs timely material before it expires, and relaxes its
own gates rather than leaving a window empty. Either way, a window is never
given up because of how an earlier post is performing.

Also drives the frequent metrics poller that feeds analytics, the queue-time
footage annotation pass that gives placement its facets, two staging rotations
(branded promos, evergreen-winner reposts), and the just-in-time filler that
re-airs a quiet evergreen post when a due window finds the queue empty.
"""
from __future__ import annotations

import bisect
import datetime as dt
import logging
import math
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import selectinload

from . import threads_api
from .analytics import poll_recent_metrics
from .categories import category_by_slug, default_shelf_life, is_first_party
from .config import load_settings, scheduler_timezone
from .db import session_scope
from .models import Candidate, Channel, Cut, SchedulerState, ThreadsPost, utcnow
from .placement import (
    SHELF_EVERGREEN,
    PlacementContext,
    PlacementSettings,
    PostFacts,
    assign_posts_to_windows,
    window_key_date,
)
from .publishing import (
    clear_publishing,
    is_publish_active,
    mark_publishing,
    publish_paired_reel,
    publish_post,
    record_post,
)

log = logging.getLogger("scheduler")

STATUS_QUEUED = "queued"
STATUS_PUBLISHING = "publishing"

# Reserved category whose cuts feed the promo rotation (app/categories.py).
PROMO_CATEGORY = "promos"

# How many promo cycles ahead the rotation will look for a free window. It only
# reaches past the first one when nearer windows are already claimed by pins —
# and pins land on promo windows often, because dragging a card pins both it and
# the card it displaced. One cycle of lookahead would strand the rotation behind
# a single dragged post until the window passed.
PROMO_LOOKAHEAD_CYCLES = 4

# Pins can target windows up to this many days out (pin_post_to_window's
# horizon). Every stale-pin check must scan at least this far ahead, or valid
# pins get wiped as "stale" — never derive the stale-check set from a shorter
# UI view range.
PIN_HORIZON_DAYS = 60

_INTERRUPTED_PUBLISH_MSG = (
    "Publishing was interrupted before it finished — the app restarted or "
    "crashed mid-publish. Check Threads to see if this clip went out; if it "
    "didn't, retry from here."
)

_thread: threading.Thread | None = None


def recover_stuck_publishing(session, *, only_inactive: bool = True) -> int:
    """Rescue posts orphaned in the ``publishing`` status.

    A manual publish (or scheduler tick) flips a post to ``publishing`` and then
    uploads to Threads in a background thread. If that process is killed first
    (dashboard ``--reload``, a crash, or a machine shutting down), the post is
    stranded: it's no longer ``queued`` so the scheduler won't retry it, and it
    isn't ``published`` so the calendar drops it — the post silently disappears.

    Flip those back to ``failed`` with an explanatory error so they resurface in
    the Posts list and can be retried. ``only_inactive`` skips posts this process
    is actively publishing right now (so a live in-flight publish is never
    clobbered); pass False only at startup, when nothing can be in flight yet.
    """
    rows = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == STATUS_PUBLISHING)
    ).scalars().all()
    recovered = 0
    for p in rows:
        if only_inactive and is_publish_active(p.id):
            continue
        p.status = "failed"
        p.error = _INTERRUPTED_PUBLISH_MSG
        p.pinned_window_key = ""
        recovered += 1
        log.warning("Recovered post %s stuck in 'publishing' -> 'failed'", p.id)
    # Paired Instagram reels publish inline during the Threads publish, so a
    # crash strands them the same way. A reel is in-flight exactly when its
    # paired Threads post is actively publishing in this process.
    from .models import InstagramPost

    ig_rows = session.execute(
        select(InstagramPost).where(InstagramPost.status == STATUS_PUBLISHING)
    ).scalars().all()
    for ig in ig_rows:
        if only_inactive and ig.threads_post_pk and is_publish_active(ig.threads_post_pk):
            continue
        ig.status = "failed"
        ig.error = _INTERRUPTED_PUBLISH_MSG
        recovered += 1
        log.warning("Recovered Instagram reel %s stuck in 'publishing' -> 'failed'", ig.id)
    return recovered


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.strip().split(":")
    return int(hour), int(minute)


def _tz() -> ZoneInfo:
    return scheduler_timezone()


def _windows_for_day(day: dt.date, tz: ZoneInfo) -> list[dt.datetime]:
    """Return today's posting windows as aware UTC datetimes."""
    settings = load_settings()
    windows = settings.get("scheduler.windows") or ["10:00", "14:30", "19:00"]
    out: list[dt.datetime] = []
    for raw in windows:
        h, m = _parse_hhmm(str(raw))
        local = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=tz)
        out.append(local.astimezone(dt.timezone.utc))
    return out


def _window_key(day: dt.date, index: int) -> str:
    return f"{day.isoformat()}#{index}"


def _get_state(session) -> SchedulerState:
    state = session.get(SchedulerState, 1)
    if state is None:
        state = SchedulerState(id=1)
        session.add(state)
        session.flush()
    return state


def spacing_allows_publish(session, now: dt.datetime | None = None) -> tuple[bool, int]:
    """Whether a publish is allowed under the spacing floor.

    Returns ``(ok, minutes_remaining)``. ``minutes_remaining`` is 0 when ok.
    """
    state = _get_state(session)
    now = now or utcnow()
    settings = load_settings()
    floor_min = int(settings.get("scheduler.spacing_floor_minutes", 90))
    floor = dt.timedelta(minutes=floor_min)
    last = state.last_publish_at
    if last is None:
        # Fall back to the most recent published post's timestamp.
        last_post = session.execute(
            select(ThreadsPost).where(
                ThreadsPost.status == "published",
                ThreadsPost.published_at.is_not(None),
            ).order_by(ThreadsPost.published_at.desc()).limit(1)
        ).scalar_one_or_none()
        last = last_post.published_at if last_post else None
    if last is None:
        return True, 0
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    remaining = floor - (now - last)
    if remaining.total_seconds() <= 0:
        return True, 0
    return False, max(1, int(remaining.total_seconds() // 60) + 1)


def _spacing_ok(state: SchedulerState, now: dt.datetime) -> bool:
    settings = load_settings()
    floor = dt.timedelta(minutes=int(settings.get("scheduler.spacing_floor_minutes", 90)))
    last = state.last_publish_at
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    return now - last >= floor


def _queue_regular(session) -> list[ThreadsPost]:
    """Queued posts in FIFO (created_at) order."""
    return list(session.execute(
        select(ThreadsPost)
        .where(ThreadsPost.status == STATUS_QUEUED)
        .order_by(ThreadsPost.created_at.asc())
    ).scalars().all())


def _clear_stale_pins(posts: list[ThreadsPost], upcoming_keys: set[str]) -> None:
    """Drop pins that point at windows that are no longer upcoming."""
    for p in posts:
        pin = (p.pinned_window_key or "").strip()
        if pin and pin not in upcoming_keys:
            p.pinned_window_key = ""


# The assignment function itself lives in app/placement.py (pure, no I/O).
# Everything below builds its context: settings resolved once, history read
# from the database, per-post facts precomputed.

# Synthetic channels that own operator uploads and pasted URLs (see
# app/web/main.py and app/scrape.py). Not editorial sources: spacing "the
# Uploads channel" would ration the operator's own hand-picked material.
# String literals rather than imports so the headless scheduler image never
# needs the scrape module's dependencies.
_SYNTHETIC_CHANNEL_URLS = ("upload://local", "youtube://pasted")

# Shelf life set on staged re-airs: a repost of a timely clip is evergreen AS
# a repost — its moment already happened; it re-airs on proven performance.
_REPOST_SHELF_LIFE = "evergreen"


def _placement_settings(settings) -> PlacementSettings:
    """Resolve the ``scheduler.placement`` block once (load_settings re-parses
    YAML on every call — never read settings inside the placement walk)."""

    def g(key: str, default):
        return settings.get(f"scheduler.placement.{key}", default)

    return PlacementSettings(
        same_source_days=int(g("gates.same_source_days", 10)),
        same_channel_days=int(g("gates.same_channel_days", 3)),
        max_facet_overlap=float(g("gates.max_facet_overlap", 0.34)),
        lookback_windows=int(g("gates.lookback_windows", 3)),
        urgency_max=float(g("weights.urgency_max", 10.0)),
        patience_per_day=float(g("weights.patience_per_day", 1.0)),
        variety_penalty_max=float(g("weights.variety_penalty_max", 4.0)),
        repost_penalty=float(g("weights.repost_penalty", 0.5)),
        half_life_breaking=float(g("half_lives.breaking", 1.0)),
        half_life_timely=float(g("half_lives.timely", 7.0)),
        expire_after_half_lives=float(g("expire_after_half_lives", 3.0)),
        # With filler on, an "empty" window becomes a library re-air, so the
        # ladder must never trade same-source spacing for a fill — leaving the
        # slot to a rerun beats airing sibling clips back-to-back.
        relax_source_gates=not bool(g("filler.enabled", False)),
    )


def _as_utc_date(when: dt.datetime | None) -> dt.date | None:
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone(dt.timezone.utc).date()


def _format_tag_set(post: ThreadsPost | None,
                    candidate: Candidate | None) -> frozenset[str]:
    """Format tags for a post: the ground-truth ``format_tags`` annotated from
    the trimmed clip at queue time, falling back to the candidate's storyboard
    prediction."""
    raw = ""
    if post is not None:
        raw = (post.format_tags or "").strip()
    if not raw and candidate is not None:
        raw = (candidate.format_tags or "").strip()
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def resolve_facets(post: ThreadsPost | None, candidate: Candidate | None,
                   facet_mode: str) -> frozenset[str]:
    """The facet label set variety runs on.

    ``union`` mode combines the category with the format tags. The category is
    a floor every post has, so tagged and untagged posts still overlap
    meaningfully during a tagging backlog (two same-category posts score 0.5,
    not 0.0), while the format tags grade the comparison: same category but
    different production form is 1/3 — under the 0.34 gate — where the
    category alone would be a hard 1.0 block.

    ``format`` mode prefers the format tags alone, falling back to the
    category so an untagged post still gets category-level variety rather
    than none. ``category`` mode is the one-hot degenerate case.
    """
    cat = ((candidate.category if candidate else "") or "").strip()
    cat_set = frozenset({cat}) if cat else frozenset()
    if facet_mode == "union":
        return cat_set | _format_tag_set(post, candidate)
    if facet_mode == "format":
        tags = _format_tag_set(post, candidate)
        if tags:
            return tags
    return cat_set


def resolve_shelf_life(post: ThreadsPost | None, candidate: Candidate | None) -> str:
    """Post override -> candidate's LLM tag -> category default -> evergreen."""
    shelf = ((post.shelf_life if post else "") or "").strip().lower()
    if not shelf and candidate is not None:
        shelf = (candidate.shelf_life or "").strip().lower()
    if not shelf:
        shelf = default_shelf_life(candidate.category if candidate else "")
    return shelf or "evergreen"


def shelf_life_outlook(post: ThreadsPost | None, candidate: Candidate | None) -> dict:
    """How the resolved shelf life plays out for a post today, for the UI.

    Returns the resolved value, every layer of the override chain (so the
    panel can say "your override — the AI said timely"), today's urgency
    contribution, and the expiry date (None = never expires). Uses the same
    math as ``PlacementContext.score`` / ``expired`` so the panel shows what
    placement will actually do.
    """
    resolved = resolve_shelf_life(post, candidate)
    override = ((post.shelf_life if post else "") or "").strip().lower()
    ai_tag = ((candidate.shelf_life if candidate else "") or "").strip().lower()
    default = default_shelf_life(candidate.category if candidate else "") or "evergreen"
    source = "override" if override else ("ai" if ai_tag else "default")

    ps = _placement_settings(load_settings())
    half = ps.half_life_days(resolved)
    content_date = (_as_utc_date(candidate.published_at) if candidate else None) \
        or (_as_utc_date(post.created_at) if post else None)
    urgency = 0.0
    expires_on = None
    expired = False
    if half is not None and content_date is not None:
        today = utcnow().date()
        age = max(0, (today - content_date).days)
        urgency = round(ps.urgency_max * 0.5 ** (age / half), 2)
        expires_on = content_date + dt.timedelta(days=math.ceil(half * ps.expire_after_half_lives))
        expired = (today - content_date).days > half * ps.expire_after_half_lives
    return {
        "shelf_life": resolved, "source": source,
        "override": override, "ai_tag": ai_tag, "default": default,
        "urgency": urgency, "urgency_max": ps.urgency_max,
        "content_date": content_date, "expires_on": expires_on, "expired": expired,
    }


def build_placement_context(session, posts: list[ThreadsPost],
                            *, force: bool = False) -> PlacementContext | None:
    """History + per-post facts for the scored placement engine.

    Returns None when ``scheduler.placement.mode`` is not ``scored`` — the
    caller then gets the original FIFO behavior. ``force`` builds one anyway
    (the replay simulation and the calendar's scored preview both need to run
    scored placement while the live mode is still fifo). Build a fresh context
    per plan; the engine mutates it as it places posts.
    """
    settings = load_settings()
    if not force and str(settings.get("scheduler.placement.mode", "fifo")).lower() != "scored":
        return None
    ps = _placement_settings(settings)
    facet_mode = str(settings.get("scheduler.placement.facet", "category")).lower()
    now = utcnow()

    cand_ids = {p.candidate_pk for p in posts if p.candidate_pk}
    candidates: dict[int, Candidate] = {}
    if cand_ids:
        candidates = {
            c.id: c for c in session.execute(
                select(Candidate).where(Candidate.id.in_(cand_ids))
            ).scalars().all()
        }
    exempt_channels = set(session.execute(
        select(Channel.id).where(Channel.url.in_(_SYNTHETIC_CHANNEL_URLS))
    ).scalars().all())

    facts: dict[int, PostFacts] = {}
    for p in posts:
        c = candidates.get(p.candidate_pk) if p.candidate_pk else None
        queued_date = _as_utc_date(p.created_at) or now.date()
        content_date = _as_utc_date(c.published_at if c else None) or queued_date
        facts[p.id] = PostFacts(
            post_id=p.id,
            candidate_pk=p.candidate_pk,
            channel_pk=c.channel_pk if c else None,
            facets=resolve_facets(p, c, facet_mode),
            half_life_days=ps.half_life_days(resolve_shelf_life(p, c)),
            content_date=content_date,
            queued_date=queued_date,
            is_repost=p.repost_of_post_pk is not None,
            channel_exempt=(c.channel_pk in exempt_channels) if c else False,
        )

    # Recent publish history feeds the source/channel gates. The horizon is
    # the widest gate; anything older can't block a placement anyway.
    horizon = max(ps.same_source_days, ps.same_channel_days)
    since = now - dt.timedelta(days=horizon + 1)
    candidate_air: dict[int, list[dt.date]] = {}
    channel_air: dict[int, list[dt.date]] = {}
    if horizon > 0:
        rows = session.execute(
            select(ThreadsPost.candidate_pk, Candidate.channel_pk,
                   ThreadsPost.published_at)
            .join(Candidate, ThreadsPost.candidate_pk == Candidate.id)
            .where(ThreadsPost.status == "published",
                   ThreadsPost.published_at.is_not(None),
                   ThreadsPost.published_at >= since)
        ).all()
        for cand_pk, chan_pk, when in rows:
            day = _as_utc_date(when)
            if day is None:
                continue
            if cand_pk is not None:
                candidate_air.setdefault(cand_pk, []).append(day)
            if chan_pk is not None and chan_pk not in exempt_channels:
                channel_air.setdefault(chan_pk, []).append(day)

    # The most recent published posts seed the variety trail, so the plan's
    # first window is judged against what the feed actually just showed.
    recent = session.execute(
        select(ThreadsPost)
        .options(selectinload(ThreadsPost.candidate))
        .where(ThreadsPost.status == "published",
               ThreadsPost.published_at.is_not(None))
        .order_by(ThreadsPost.published_at.desc())
        .limit(max(1, ps.lookback_windows))
    ).scalars().all()
    trail = [resolve_facets(p, p.candidate, facet_mode) for p in reversed(recent)]

    return PlacementContext(
        settings=ps,
        facts=facts,
        candidate_air_dates=candidate_air,
        channel_air_dates=channel_air,
        facet_trail=trail,
    )


def _assign_with_mode(
    session,
    posts: list[ThreadsPost],
    keys: list[str],
) -> tuple[list[ThreadsPost | None], PlacementContext | None]:
    """One entry point for every caller (tick head, pin swap, calendar plan),
    so the plan the operator sees and the post the tick publishes can never
    come from different modes."""
    ctx = build_placement_context(session, posts)
    return assign_posts_to_windows(posts, keys, ctx=ctx), ctx


def _queue_head_for_window(session, window_key: str) -> ThreadsPost | None:
    """Post that should publish at ``window_key`` (pin-aware)."""
    tz = _tz()
    now = utcnow()
    state = _get_state(session)
    day = now.astimezone(tz).date()
    # Look far enough ahead that every legal pin is visible (pins can be
    # created up to PIN_HORIZON_DAYS out); a shorter horizon here would wipe
    # farther-out pins as "stale".
    upcoming = _upcoming_window_slots(
        day, day + dt.timedelta(days=PIN_HORIZON_DAYS),
        now=now, last_window_key=state.last_window_key or "",
    )
    keys = [k for k, _, _ in upcoming]
    # The due window has always already fired by the time the tick evaluates it,
    # so _upcoming_window_slots excludes it (win <= now). Re-attach it at the
    # front: pins targeting it must survive _clear_stale_pins, and the
    # assignment must match the calendar plan as it stood before the window
    # fired. (Previously this fell back to the raw FIFO head, ignoring pins.)
    if window_key not in keys:
        keys.insert(0, window_key)
    posts = _queue_regular(session)
    _clear_stale_pins(posts, set(keys))
    assignment, _ = _assign_with_mode(session, posts, keys)
    return assignment[keys.index(window_key)]


def pin_post_to_window(session, post_id: int, window_key: str) -> str:
    """Pin a queued post to an upcoming window. Returns a short status message.

    If another post already occupies that window (via pin or FIFO projection),
    swap pins so the dragged post lands on the target and the other moves to
    the dragged post's previous projected window.
    """
    window_key = (window_key or "").strip()
    if not window_key or "#" not in window_key:
        raise ValueError("Invalid window key")

    tz = _tz()
    now = utcnow()
    state = _get_state(session)
    day = now.astimezone(tz).date()
    upcoming = _upcoming_window_slots(
        day, day + dt.timedelta(days=PIN_HORIZON_DAYS),
        now=now, last_window_key=state.last_window_key or "",
    )
    keys = [k for k, _, _ in upcoming]
    if window_key not in keys:
        raise ValueError("That window is no longer available")

    post = session.get(ThreadsPost, post_id)
    if post is None or post.status != STATUS_QUEUED:
        raise ValueError("Only a queued post can be pinned")

    posts = _queue_regular(session)
    _clear_stale_pins(posts, set(keys))
    assignment, _ = _assign_with_mode(session, posts, keys)

    # Where is the dragged post currently projected?
    from_key = ""
    for k, p in zip(keys, assignment):
        if p is not None and p.id == post_id:
            from_key = k
            break

    occupant = None
    target_i = keys.index(window_key)
    if assignment[target_i] is not None and assignment[target_i].id != post_id:
        occupant = session.get(ThreadsPost, assignment[target_i].id)

    post.pinned_window_key = window_key
    if occupant is not None:
        # Swap: send the occupant to where the dragged post came from (or clear).
        occupant.pinned_window_key = from_key if from_key and from_key != window_key else ""

    session.flush()
    return f"Moved to {window_key}"


# --- Promo rotation ---------------------------------------------------------
#
# Branded content is metered onto its own cadence rather than flowing through
# the FIFO queue. Every ``every_days`` days one window belongs to a promo, and
# the tick stages that promo ahead of time as an ordinary queued post pinned to
# the window. Two things fall out of staging rather than special-casing the
# assignment: the promo can never drift into an organic window (it is pinned
# before it is ever a queue candidate), and it appears on the calendar as a
# normal card the operator can edit, drag, or remove.
#
# The pool recycles instead of draining — a clip re-enters rotation after it
# airs — so promo production is decoupled from promo pacing.


def _parse_window_key(key: str) -> tuple[dt.date, int] | None:
    """``YYYY-MM-DD#N`` -> ``(date, index)``, or None when unparseable.

    Promo bookkeeping compares keys across days, so it compares parsed tuples
    rather than raw strings: ``...#10`` sorts before ``...#2`` as text. (The
    same-day checks elsewhere in this module are safe because they compare
    within one date prefix.)
    """
    day_s, sep, idx_s = (key or "").partition("#")
    if not sep:
        return None
    try:
        return dt.date.fromisoformat(day_s), int(idx_s)
    except ValueError:
        return None


def _promo_config() -> dict | None:
    """Promo rotation settings, or None when it's off or misconfigured."""
    settings = load_settings()
    if not settings.get("scheduler.promos.enabled", False):
        return None
    every = int(settings.get("scheduler.promos.every_days", 3) or 0)
    if every < 1:
        return None
    raw_anchor = str(settings.get("scheduler.promos.anchor", "") or "").strip()
    try:
        anchor = dt.date.fromisoformat(raw_anchor)
    except ValueError:
        log.warning("Invalid scheduler.promos.anchor %r; promo rotation disabled",
                    raw_anchor)
        return None
    return {
        "every_days": every,
        "anchor": anchor,
        "window_index": settings.get("scheduler.promos.window_index", "middle"),
        "min_days_between_repeats": float(
            settings.get("scheduler.promos.min_days_between_repeats", 0) or 0),
    }


def _promo_window_index(raw, window_count: int) -> int | None:
    """Resolve the configured promo window to a 0-based index into the day."""
    if window_count <= 0:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "middle":
        return window_count // 2
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        log.warning("Invalid scheduler.promos.window_index %r; promo rotation disabled",
                    raw)
        return None
    return idx if 0 <= idx < window_count else None


def _is_promo_window(day: dt.date, index: int, cfg: dict, window_count: int) -> bool:
    """Whether this window belongs to the promo cadence.

    Phase comes from the configured anchor date, not from stored state, so both
    schedulers (dashboard thread and the Actions cron) agree without sharing
    anything, and the answer survives restarts and config reloads.
    """
    promo_index = _promo_window_index(cfg["window_index"], window_count)
    if promo_index is None or index != promo_index:
        return False
    return (day - cfg["anchor"]).days % cfg["every_days"] == 0


def _upcoming_promo_slots(cfg: dict, now: dt.datetime,
                          last_promo_key: str) -> list[tuple[str, dt.datetime]]:
    """Upcoming windows the rotation may claim, earliest first.

    Windows at or before ``last_promo_key`` are spent: that cycle was already
    offered a promo, whether or not the staged post survived.
    """
    tz = _tz()
    today = now.astimezone(tz).date()
    last = _parse_window_key(last_promo_key)
    # Several cycles, so a claimed window has somewhere to step to. Staging
    # stays close to the calendar anyway: the caller takes the FIRST free slot,
    # and ``_promo_pending`` stops a second promo being staged behind it.
    horizon = today + dt.timedelta(days=cfg["every_days"] * PROMO_LOOKAHEAD_CYCLES)
    out: list[tuple[str, dt.datetime]] = []
    d = today
    while d <= horizon:
        windows = _windows_for_day(d, tz)
        for i, win in enumerate(windows):
            if win <= now or not _is_promo_window(d, i, cfg, len(windows)):
                continue
            if last is not None and (d, i) <= last:
                continue
            out.append((_window_key(d, i), win))
        d += dt.timedelta(days=1)
    return out


# SQL predicate for first-party (brand-owned) content: the channel-level
# provenance flag, with the legacy reserved ``promos`` category kept as a
# per-video back-compat path until existing rows are backfilled. Mirrors
# ``categories.is_first_party`` — keep the two in sync.
def _first_party_clause():
    return or_(Candidate.category == PROMO_CATEGORY,
               Channel.first_party.is_(True))


def _promo_pending(session) -> bool:
    """Whether a promo is already on its way out.

    Only one promo is ever in flight. On a promo day the current cycle and the
    next one are both inside the search horizon, so without this the tick would
    stage the following cycle the moment the current one was claimed — draining
    the rotation days ahead of the calendar.

    Deliberately counts *any* queued promo, staged or hand-queued: either way a
    piece of branded content is already scheduled, and a second would double up.
    """
    row = session.execute(
        select(ThreadsPost.id)
        .join(Candidate, ThreadsPost.candidate_pk == Candidate.id)
        .join(Channel, Candidate.channel_pk == Channel.id)
        .where(
            ThreadsPost.status.in_((STATUS_QUEUED, STATUS_PUBLISHING)),
            _first_party_clause(),
        ).limit(1)
    ).first()
    return row is not None


def _claimed_window_keys(session) -> set[str]:
    """Windows an operator pin already holds.

    Pins are not only deliberate claims: ``pin_post_to_window`` swaps, so
    dragging any card pins both it and the card it displaced. With a promo
    window every few days those byproduct pins land on promo slots constantly,
    so the rotation steps over a claimed slot to the next one rather than
    treating it as a refusal and stalling there.
    """
    return {
        key for key in session.execute(
            select(ThreadsPost.pinned_window_key).where(
                ThreadsPost.status.in_((STATUS_QUEUED, STATUS_PUBLISHING)),
                ThreadsPost.pinned_window_key != "",
            )
        ).scalars().all() if key
    }


def _promo_rotation(session) -> list[tuple[Cut, dt.datetime | None]]:
    """Promo cuts eligible to air, least-recently-aired first.

    A cut joins the rotation once the operator has queued it by hand — a post
    of its own reaching ``queued`` or ``published`` is that opt-in, so nothing
    airs unattended that the operator hasn't already sent out deliberately.

    A cut drops out while it has a post already queued or publishing: that
    airing is booked, and staging a second would double-book the same clip.
    """
    cuts = session.execute(
        select(Cut)
        .join(Candidate, Cut.candidate_pk == Candidate.id)
        .join(Channel, Candidate.channel_pk == Channel.id)
        .options(selectinload(Cut.candidate))
        .where(_first_party_clause(), Cut.trimmed_clip_path != "")
    ).scalars().all()
    if not cuts:
        return []

    last_aired: dict[int, dt.datetime] = dict(session.execute(
        select(ThreadsPost.cut_pk, func.max(ThreadsPost.published_at))
        .where(ThreadsPost.status == "published",
               ThreadsPost.cut_pk.is_not(None),
               ThreadsPost.published_at.is_not(None))
        .group_by(ThreadsPost.cut_pk)
    ).all())
    booked = set(session.execute(
        select(ThreadsPost.cut_pk).where(
            ThreadsPost.cut_pk.is_not(None),
            ThreadsPost.status.in_((STATUS_QUEUED, STATUS_PUBLISHING)),
        )
    ).scalars().all())
    opted_in = booked | set(last_aired)

    def aired_at(cut: Cut) -> dt.datetime | None:
        when = last_aired.get(cut.id)
        if when is not None and when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        return when

    eligible = [c for c in cuts if c.id in opted_in and c.id not in booked]
    # Never-aired cuts sort first; ids break ties so the order is stable across
    # processes (both schedulers must pick the same head).
    epoch = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    eligible.sort(key=lambda c: (aired_at(c) or epoch, c.id))
    return [(c, aired_at(c)) for c in eligible]


def _promo_due(session, cfg: dict, now: dt.datetime) -> Cut | None:
    """The promo to air next, or None when the rotation should sit this one out.

    With a small pool the head comes back around quickly, and the caption is
    reused verbatim — ``min_days_between_repeats`` is what stops that from
    reading as a loop. Skipping simply leaves the window to organic content.
    """
    rotation = _promo_rotation(session)
    if not rotation:
        return None
    cut, last_aired = rotation[0]
    gap_days = cfg["min_days_between_repeats"]
    if last_aired is not None and gap_days > 0:
        if (now - last_aired).total_seconds() < gap_days * 86400:
            return None
    return cut


def _stage_promo(session, cut: Cut, window_key: str) -> ThreadsPost | None:
    """Queue a fresh airing of ``cut``, pinned to ``window_key``.

    Normally this clones the cut's last airing: caption and clip paths are
    copied rather than rebuilt, so the clip is already in cloud storage and a
    headless runner holding none of the operator's local files can still stage
    and publish it — and reusing the caption is what the rotation is for.

    With no usable prior airing it falls back to the cut's own draft caption
    and exported file, which needs that file on this machine to upload. That
    path is best-effort: a headless runner without the file simply leaves the
    cycle for whichever scheduler does have it.
    """
    prior = session.execute(
        select(ThreadsPost)
        .where(ThreadsPost.cut_pk == cut.id, ThreadsPost.status == "published")
        .order_by(ThreadsPost.published_at.desc().nullslast())
        .limit(1)
    ).scalar_one_or_none()

    if (prior is not None and (prior.caption or "").strip()
            and (prior.clip_object_path or prior.clip_local_path)):
        post = ThreadsPost(
            candidate_pk=cut.candidate_pk,
            cut_pk=cut.id,
            caption=prior.caption,
            clip_local_path=prior.clip_local_path,
            clip_object_path=prior.clip_object_path,
            clip_length_seconds=prior.clip_length_seconds,
            attribution_text=prior.attribution_text,
            attribution_skipped=prior.attribution_skipped,
            status=STATUS_QUEUED,
            pinned_window_key=window_key,
        )
        session.add(post)
        session.flush()
        return post

    caption = (cut.draft_caption or "").strip()
    if not caption:
        return None
    clip_path = cut.trimmed_clip_path
    if cut.use_subtitles and cut.subtitled_clip_path:
        if Path(cut.subtitled_clip_path).expanduser().exists():
            clip_path = cut.subtitled_clip_path
    if not clip_path:
        return None
    try:
        post = record_post(session, cut.candidate, clip_path, caption,
                           status=STATUS_QUEUED, cut=cut)
    except FileNotFoundError:
        log.info("Promo cut %s has no prior airing and its export isn't on this "
                 "machine; leaving the cycle for another scheduler", cut.id)
        return None
    post.pinned_window_key = window_key
    session.flush()
    return post


def ensure_promo_staged() -> str | None:
    """Stage the next promo onto its window if the rotation has one ready.

    Returns a short action string when a post was staged, else None. Safe to
    call every tick: once a promo is pinned to the upcoming slot this costs one
    indexed lookup.
    """
    cfg = _promo_config()
    if cfg is None:
        return None
    now = utcnow()
    with session_scope() as session:
        state = _get_state(session)
        if _promo_pending(session):
            return None
        slots = _upcoming_promo_slots(cfg, now, state.last_promo_window_key or "")
        if not slots:
            return None
        cut = _promo_due(session, cfg, now)
        if cut is None:
            # Nothing eligible yet. Leave the cycle unconsumed so a promo that
            # becomes eligible before the window fires can still take it; once
            # the window passes, the slot drops out of the horizon by itself.
            return None
        claimed = _claimed_window_keys(session)
        for window_key, _win in slots:
            if window_key in claimed:
                continue
            post = _stage_promo(session, cut, window_key)
            if post is None:
                return None
            # Consume the cycle even though the post could still be cancelled:
            # cancelling deletes the row, and without this marker the next tick
            # would mint it straight back and the operator could never decline
            # one. Skipped-over slots stay unconsumed, so freeing one up before
            # it fires still lets a promo land there.
            state.last_promo_window_key = window_key
            state.updated_at = utcnow()
            log.info("Staged promo cut %s as post %s for window %s",
                     cut.id, post.id, window_key)
            return f"promo_staged:{window_key}:post={post.id}"
        return None


# --- Repost rotation ----------------------------------------------------------
#
# The promo machinery, generalized: every ``every_days`` days one window may
# re-air a PROVEN ORGANIC post — published 3-6 months ago and a top performer
# at the same age. Same staging model as promos (an ordinary queued post,
# pinned to its window, cloned from the prior airing so a headless runner
# needs no local files), so the operator can edit, drag, or cancel it. The two
# rotations differ only in their pool and cadence config.
#
# Performance is judged age-normalized (views at a fixed age via the metric
# snapshots), never lifetime totals — raw lifetime views would just re-elect
# the oldest posts over and over.

# Below this many published posts with metrics, a percentile is noise — the
# rotation sits out until the account has a baseline.
_REPOST_MIN_BASELINE = 10


def _repost_config() -> dict | None:
    """Repost rotation settings, or None when it's off or misconfigured.

    Shares the promo cadence helpers (``_is_promo_window`` and
    ``_upcoming_promo_slots`` only read anchor / every_days / window_index
    from the dict they're given).
    """
    settings = load_settings()
    if not settings.get("scheduler.placement.reposts.enabled", False):
        return None

    def g(key: str, default):
        return settings.get(f"scheduler.placement.reposts.{key}", default)

    every = int(g("every_days", 7) or 0)
    if every < 1:
        return None
    raw_anchor = str(g("anchor", "2026-01-01") or "").strip()
    try:
        anchor = dt.date.fromisoformat(raw_anchor)
    except ValueError:
        log.warning("Invalid scheduler.placement.reposts.anchor %r; "
                    "repost rotation disabled", raw_anchor)
        return None
    return {
        "every_days": every,
        "anchor": anchor,
        "window_index": g("window_index", 0),
        "min_age_days": float(g("min_age_days", 90) or 0),
        "min_days_between_repeats": float(g("min_days_between_repeats", 90) or 0),
        # 0 = no lifetime cap; the filler rotation's growing quiet period is
        # what keeps a much-aired clip from dominating instead.
        "max_airings_per_cut": int(g("max_airings_per_cut", 0) or 0),
        "percentile": float(g("percentile", 90) or 0),
    }


def _repost_pending(session) -> bool:
    """Whether a staged re-air is already on its way out (one in flight, ever —
    same rationale as ``_promo_pending``). Staged re-airs — rotation reposts
    and just-in-time filler alike — are the only posts that carry
    ``repost_of_post_pk``, so the flag is the marker for both."""
    row = session.execute(
        select(ThreadsPost.id).where(
            ThreadsPost.status.in_((STATUS_QUEUED, STATUS_PUBLISHING)),
            ThreadsPost.repost_of_post_pk.is_not(None),
        ).limit(1)
    ).first()
    return row is not None


def _repost_pool(session, cfg: dict, now: dt.datetime) -> list[tuple[Cut, ThreadsPost]]:
    """``(cut, prior_airing)`` pairs eligible to re-air, least-recently-aired
    first (ids break ties so all runners pick the same head).

    A cut qualifies when it resolves evergreen (old timely/breaking coverage
    must not re-air as a "proven winner" — its moment already happened, the
    same rule the filler rotation applies), its latest airing is at least
    ``min_age_days`` old — there is no age ceiling; archival evergreen is the
    whole point of re-airing — its views at the fixed comparison age clear
    the account's ``percentile``, and it isn't first-party (branded content
    has its own rotation) or already booked. ``max_airings_per_cut`` caps
    lifetime airings when non-zero; 0 means uncapped.
    """
    from .analytics import metrics_at_age_bulk

    posts = session.execute(
        select(ThreadsPost)
        .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
                 selectinload(ThreadsPost.cut))
        .where(ThreadsPost.status == "published",
               ThreadsPost.published_at.is_not(None))
    ).scalars().all()
    if len(posts) < _REPOST_MIN_BASELINE:
        return []

    settings = load_settings()
    age_hours = int(settings.get("learning.metric_age_hours", 48))
    views_at_age = metrics_at_age_bulk(session, posts, "views", age_hours)
    values = sorted(views_at_age.values())
    if len(values) < _REPOST_MIN_BASELINE:
        return []
    # Deterministic percentile: floor index into the sorted values. Every
    # runner computes the same threshold from the same snapshots.
    idx = min(len(values) - 1, int(cfg["percentile"] * (len(values) - 1) // 100))
    threshold = values[idx]

    by_cut: dict[int, list[ThreadsPost]] = {}
    for p in posts:
        if p.cut_pk is not None:
            by_cut.setdefault(p.cut_pk, []).append(p)

    booked = set(session.execute(
        select(ThreadsPost.cut_pk).where(
            ThreadsPost.cut_pk.is_not(None),
            ThreadsPost.status.in_((STATUS_QUEUED, STATUS_PUBLISHING)),
        )
    ).scalars().all())

    min_gap = max(cfg["min_age_days"], cfg["min_days_between_repeats"])
    eligible: list[tuple[dt.datetime, Cut, ThreadsPost]] = []
    for cut_pk, airings in by_cut.items():
        if cut_pk in booked:
            continue
        if cfg["max_airings_per_cut"] > 0 and len(airings) >= cfg["max_airings_per_cut"]:
            continue
        airings.sort(key=lambda p: (p.published_at, p.id))
        last = airings[-1]
        last_at = last.published_at
        if last_at.tzinfo is None:
            last_at = last_at.replace(tzinfo=dt.timezone.utc)
        age_days = (now - last_at).total_seconds() / 86400
        if age_days < min_gap:
            continue
        if last.cut is None or is_first_party(last.candidate):
            continue
        # Evergreen only. This supersedes the old max_age_days ceiling: an
        # evergreen clip never ages out of the pool, and non-evergreen never
        # enters it at any age.
        if resolve_shelf_life(last, last.candidate) != SHELF_EVERGREEN:
            continue
        best = max((views_at_age.get(p.id, 0) for p in airings), default=0)
        if best < threshold or best <= 0:
            continue
        # Cloning needs a usable prior airing (caption + a clip reachable from
        # any runner); without one the cut just isn't eligible yet.
        if not (last.caption or "").strip():
            continue
        if not (last.clip_object_path or last.clip_local_path):
            continue
        eligible.append((last_at, last.cut, last))

    eligible.sort(key=lambda t: (t[0], t[1].id))
    return [(cut, prior) for _, cut, prior in eligible]


def _stage_repost(session, cut: Cut, prior: ThreadsPost, window_key: str) -> ThreadsPost:
    """Queue a re-air of ``prior``, pinned to ``window_key``.

    Clones the airing the way ``_stage_promo`` does — caption and clip paths
    copied, so the clip is already in cloud storage and a headless runner can
    publish it. Declares itself with ``repost_of_post_pk`` and ships as
    evergreen: the content's moment already happened, it re-airs on proven
    performance. Facet annotations are copied too (same file — re-annotating
    would spend an LLM call to learn the same answer).
    """
    post = ThreadsPost(
        candidate_pk=cut.candidate_pk,
        cut_pk=cut.id,
        caption=prior.caption,
        clip_local_path=prior.clip_local_path,
        clip_object_path=prior.clip_object_path,
        clip_length_seconds=prior.clip_length_seconds,
        attribution_text=prior.attribution_text,
        attribution_skipped=prior.attribution_skipped,
        status=STATUS_QUEUED,
        pinned_window_key=window_key,
        repost_of_post_pk=prior.id,
        shelf_life=_REPOST_SHELF_LIFE,
        footage_traits=prior.footage_traits,
        format_tags=prior.format_tags,
        footage_rationale=prior.footage_rationale,
        footage_scored_at=prior.footage_scored_at,
    )
    session.add(post)
    session.flush()
    return post


def ensure_repost_staged() -> str | None:
    """Stage the next proven re-air onto its window if one is due.

    Mirrors ``ensure_promo_staged``: safe to call every tick, consumes its
    cycle via ``last_repost_window_key`` so a cancelled staged repost is a
    decision, not a re-mint loop.
    """
    cfg = _repost_config()
    if cfg is None:
        return None
    now = utcnow()
    with session_scope() as session:
        state = _get_state(session)
        if _repost_pending(session):
            return None
        slots = _upcoming_promo_slots(cfg, now, state.last_repost_window_key or "")
        if not slots:
            return None
        pool = _repost_pool(session, cfg, now)
        if not pool:
            return None
        cut, prior = pool[0]
        claimed = _claimed_window_keys(session)
        for window_key, _win in slots:
            if window_key in claimed:
                continue
            post = _stage_repost(session, cut, prior, window_key)
            state.last_repost_window_key = window_key
            state.updated_at = utcnow()
            log.info("Staged repost of post %s (cut %s) as post %s for window %s",
                     prior.id, cut.id, post.id, window_key)
            return f"repost_staged:{window_key}:post={post.id}"
        return None


def _filler_config() -> dict | None:
    """Just-in-time filler settings, or None when the feature is off.

    Filler is the "never go silent" floor: when a window comes due and the
    queue is empty, re-air the least-recently-aired evergreen post instead of
    skipping the slot. It needs no cadence config — it runs exactly when a
    window would otherwise be empty, so real queued content always wins by
    construction.
    """
    settings = load_settings()
    if not settings.get("scheduler.placement.filler.enabled", False):
        return None
    quiet = float(settings.get("scheduler.placement.filler.min_quiet_days", 45) or 0)
    growth = float(settings.get("scheduler.placement.filler.airing_growth", 0.5) or 0)
    return {"min_quiet_days": max(1.0, quiet), "airing_growth": max(0.0, growth)}


def _filler_rotation(session, cfg: dict) -> tuple[list[dict], dict[int, dt.date]]:
    """The rerun library: per-cut facts for filler picking, plus each
    candidate's latest air date (for same-source spacing).

    Each fact is ``{cut, prior, last_aired, required_quiet_days,
    candidate_pk}`` for a cut that may re-air as filler once it has been
    quiet long enough. Deliberately looser than ``_repost_pool`` — filler is
    the floor under an empty calendar, not a best-of selection — so there is
    no performance bar, no lifetime airing cap, and no maximum age. What
    remains:

    - evergreen only, by the resolved shelf life of the last airing — old
      *news* re-airing as filler would be wrong, not just weak
    - not first-party (promos have their own rotation), not already booked,
      and clone-able (caption + a clip path reachable from any runner)

    Performance sets the cadence, prior airings stretch it:
    ``required_quiet_days = min_quiet_days x (1 + airing_growth x (airings-1))
    / (1 + rank)`` where rank is the cut's percentile (0..1) of views at the
    learning comparison age. A top clip re-airs after roughly half the quiet
    period of a median one; a dud waits the full period; and every re-air
    pushes the next one further out, so recycling never hard-stops but a
    much-aired clip cannot dominate the rotation. The rank is computed from
    stored metric snapshots, so every runner derives the same cadence.
    """
    from .analytics import metrics_at_age_bulk

    posts = session.execute(
        select(ThreadsPost)
        .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
                 selectinload(ThreadsPost.cut))
        .where(ThreadsPost.status == "published",
               ThreadsPost.published_at.is_not(None))
    ).scalars().all()

    def _aware(ts: dt.datetime) -> dt.datetime:
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=dt.timezone.utc)

    by_cut: dict[int, list[tuple[dt.datetime, ThreadsPost]]] = {}
    latest_by_candidate: dict[int, dt.date] = {}
    for p in posts:
        at = _aware(p.published_at)
        if p.cut_pk is not None:
            by_cut.setdefault(p.cut_pk, []).append((at, p))
        if p.candidate_pk is not None:
            prev = latest_by_candidate.get(p.candidate_pk)
            if prev is None or at.date() > prev:
                latest_by_candidate[p.candidate_pk] = at.date()

    booked = set(session.execute(
        select(ThreadsPost.cut_pk).where(
            ThreadsPost.cut_pk.is_not(None),
            ThreadsPost.status.in_((STATUS_QUEUED, STATUS_PUBLISHING)),
        )
    ).scalars().all())

    settings = load_settings()
    age_hours = int(settings.get("learning.metric_age_hours", 48))
    views_at_age = metrics_at_age_bulk(session, posts, "views", age_hours)
    ranked = sorted(views_at_age.values())

    def _rank(value: float) -> float:
        """Percentile rank 0..1 via bisect over every published post's views —
        deterministic, and a missing metric ranks 0 (full quiet period)."""
        if len(ranked) < 2:
            return 0.0
        return bisect.bisect_left(ranked, value) / (len(ranked) - 1)

    facts: list[dict] = []
    for cut_pk, airings in by_cut.items():
        if cut_pk in booked:
            continue
        airings.sort(key=lambda t: (t[0], t[1].id))
        last_at, last = airings[-1]
        if last.cut is None or is_first_party(last.candidate):
            continue
        if resolve_shelf_life(last, last.candidate) != SHELF_EVERGREEN:
            continue
        if not (last.caption or "").strip():
            continue
        if not (last.clip_object_path or last.clip_local_path):
            continue
        best = max((views_at_age.get(p.id, 0) for _, p in airings), default=0)
        rank = min(1.0, max(0.0, _rank(best)))
        # Each prior airing stretches the quiet period: the decay curve that
        # replaced the old hard lifetime cap on re-airs.
        growth = 1.0 + cfg["airing_growth"] * max(0, len(airings) - 1)
        facts.append({
            "cut": last.cut,
            "prior": last,
            "last_aired": last_at.date(),
            "airings": len(airings),
            "required_quiet_days": cfg["min_quiet_days"] * growth / (1.0 + rank),
            "candidate_pk": last.candidate_pk,
            # Performance receipt for the UI: what set this cut's cadence.
            # base_quiet_days is the wait before the performance division —
            # the gap between it and required_quiet_days IS the rank's effect.
            "rank": rank,
            "views_at_age": best,
            "base_quiet_days": cfg["min_quiet_days"] * growth,
            "metric_age_hours": age_hours,
        })
    # Stable order so ties in the picker resolve identically on every runner.
    facts.sort(key=lambda f: f["cut"].id)
    return facts, latest_by_candidate


def _pick_filler(facts: list[dict], day: dt.date,
                 cand_last: dict[int, dt.date],
                 source_gap_days: float) -> dict | None:
    """Most-overdue eligible rerun for ``day``, or None.

    Eligible = quiet at least its own ``required_quiet_days`` (hits cycle
    faster) and no sibling clip of the same video within
    ``source_gap_days``. Overdue ratio (days quiet / required) decides, cut
    id breaks ties, so all runners and the calendar projection agree.
    """
    best = None
    best_key: tuple[float, int] | None = None
    for f in facts:
        quiet = (day - f["last_aired"]).days
        if quiet < f["required_quiet_days"]:
            continue
        cand_pk = f["candidate_pk"]
        if cand_pk is not None:
            seen = cand_last.get(cand_pk)
            if seen is not None and abs((day - seen).days) < source_gap_days:
                continue
        key = (quiet / f["required_quiet_days"], -f["cut"].id)
        if best_key is None or key > best_key:
            best, best_key = f, key
    return best


# The rerun outlook is expensive to compute: _filler_rotation loads every
# published post AND ranks performance from each one's full metric-snapshot
# series (polled every 15 minutes, so tens of thousands of rows on a mature
# account). The UI asks for it on every published post page and every
# library rebuild, where that cost is pure latency. Its inputs only change
# on the publish/metrics cadence — minutes — so UI reads serve a short-lived
# module cache of plain values (no ORM objects). The staging path
# (_stage_filler_jit) never reads this cache: it recomputes fresh facts
# inside its own transaction.
_recycle_ui_cache: tuple[float, dict[int, dict]] | None = None
_RECYCLE_UI_TTL_S = 300.0


def invalidate_recycle_overview() -> None:
    """Drop the cached rerun outlook. Call after writing an input the
    operator expects to see reflected immediately (e.g. a shelf-life
    override); everything else just waits out the TTL."""
    global _recycle_ui_cache
    _recycle_ui_cache = None


def recycle_overview(session) -> dict[int, dict]:
    """Filler-rotation outlook per cut, for the UI: ``{cut_pk: fact}``.

    Each fact is ``{airings, last_aired, required_quiet_days, next_eligible,
    overdue}`` for a cut currently in the rerun library. Cuts the rotation
    excludes (booked, first-party, not evergreen, no caption/clip) are absent.
    Empty when filler is off. Shares ``_filler_rotation`` with the staging
    path so the page shows what the scheduler would do, but serves a cached
    copy for up to ``_RECYCLE_UI_TTL_S`` seconds (see note above).
    """
    global _recycle_ui_cache
    cached = _recycle_ui_cache
    if cached is not None and time.monotonic() - cached[0] < _RECYCLE_UI_TTL_S:
        return cached[1]
    cfg = _filler_config()
    if cfg is None:
        out: dict[int, dict] = {}
        _recycle_ui_cache = (time.monotonic(), out)
        return out
    facts, _ = _filler_rotation(session, cfg)
    today = utcnow().date()
    # The repost rotation's performance bar, when that feature is on — lets
    # the page say a cut also qualifies for proactive "proven winner" re-airs
    # rather than only filler. Rank comparison mirrors _repost_pool's
    # value-threshold check closely enough for display.
    repost_cfg = _repost_config()
    repost_pct = float(repost_cfg["percentile"]) if repost_cfg else None
    out = {}
    for f in facts:
        required = float(f["required_quiet_days"])
        next_at = f["last_aired"] + dt.timedelta(days=math.ceil(required))
        rank = float(f["rank"])
        out[f["cut"].id] = {
            "airings": f["airings"],
            "last_aired": f["last_aired"],
            "required_quiet_days": required,
            "next_eligible": next_at,
            "overdue": today >= next_at,
            "quiet_days": (today - f["last_aired"]).days,
            "rank": rank,
            "views_at_age": f["views_at_age"],
            "base_quiet_days": float(f["base_quiet_days"]),
            "metric_age_hours": f["metric_age_hours"],
            "airing_growth": cfg["airing_growth"],
            "repost_percentile": repost_pct,
            "proven_winner": (repost_pct is not None
                              and rank >= repost_pct / 100.0
                              and f["views_at_age"] > 0),
        }
    _recycle_ui_cache = (time.monotonic(), out)
    return out


def recycle_status(session, post: ThreadsPost) -> dict | None:
    """Recycling outlook for one published post's cut, or None when n/a.

    ``{"state": "eligible", ...recycle_overview fact}`` when the cut is in
    the rerun library; ``{"state": "ineligible", "reason", "airings",
    "last_aired"}`` when the rotation excludes it; ``{"state": "off"}`` when
    filler is disabled. The reason mirrors ``_filler_rotation``'s checks so
    the page explains the scheduler instead of guessing at it.
    """
    if post.status != "published" or post.cut_pk is None:
        return None
    if _filler_config() is None:
        return {"state": "off"}
    info = recycle_overview(session).get(post.cut_pk)
    if info is not None:
        return {"state": "eligible", **info}

    airings = session.execute(
        select(ThreadsPost)
        .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
                 selectinload(ThreadsPost.cut))
        .where(ThreadsPost.cut_pk == post.cut_pk,
               ThreadsPost.status == "published",
               ThreadsPost.published_at.is_not(None))
    ).scalars().all()
    airings.sort(key=lambda p: (p.published_at, p.id))
    last = airings[-1] if airings else post
    last_at = _as_utc_date(last.published_at)

    booked = session.execute(
        select(ThreadsPost.id).where(
            ThreadsPost.cut_pk == post.cut_pk,
            ThreadsPost.status.in_((STATUS_QUEUED, STATUS_PUBLISHING)),
        ).limit(1)
    ).first() is not None

    if booked:
        reason = "a re-air of this clip is already queued"
    elif last.cut is None:
        reason = "the original cut no longer exists"
    elif is_first_party(last.candidate):
        reason = "first-party content rotates through promos, not reruns"
    elif resolve_shelf_life(last, last.candidate) != SHELF_EVERGREEN:
        reason = ("only evergreen clips re-air — this one resolves "
                  f"{resolve_shelf_life(last, last.candidate)}")
    elif not (last.caption or "").strip():
        reason = "the last airing has no caption to clone"
    elif not (last.clip_object_path or last.clip_local_path):
        reason = "no clip file reachable from the schedulers"
    else:
        reason = "not in the rerun library yet"
    return {"state": "ineligible", "reason": reason,
            "airings": len(airings), "last_aired": last_at}


def _stage_filler_jit(session, state: SchedulerState, window_key: str,
                      now: dt.datetime) -> int | None:
    """Stage a filler re-air for a due window whose queue came up empty.

    Returns the staged post's id, or None when filler is off, nothing
    qualifies, or another runner beat us to the window.

    Unlike the promo/repost rotations (which stage into *future* windows and
    tolerate benign races), this stages into a window being consumed *right
    now* while several runners may be ticking on it. Two runners each staging
    their own row would each pass the atomic post-claim — a double post. So
    the window itself is the lock: a compare-and-set on
    ``SchedulerState.last_window_key`` (from the value this runner read to
    this window's key) admits exactly one runner; the losers see rowcount 0
    and back off. If the winner's publish then fails, the window is spent —
    the same cost as the empty window we were about to record anyway, and the
    failed post surfaces in notifications.
    """
    cfg = _filler_config()
    if cfg is None:
        return None
    # One re-air in flight at a time, shared with the repost rotation — the
    # marker is the same ``repost_of_post_pk`` flag.
    if _repost_pending(session):
        return None
    day = window_key_date(window_key)
    if day is None:
        return None
    facts, cand_last = _filler_rotation(session, cfg)
    source_gap = float(_placement_settings(load_settings()).same_source_days)
    pick = _pick_filler(facts, day, cand_last, source_gap)
    if pick is None:
        return None

    won = session.execute(
        update(SchedulerState)
        .where(SchedulerState.id == state.id,
               SchedulerState.last_window_key == (state.last_window_key or ""))
        .values(last_window_key=window_key,
                last_action=f"filler_staged:{window_key}",
                updated_at=utcnow())
    ).rowcount
    if won != 1:
        return None
    session.expire(state)

    cut, prior = pick["cut"], pick["prior"]
    post = _stage_repost(session, cut, prior, window_key)
    log.info("Staged filler re-air of post %s (cut %s) as post %s for empty window %s",
             prior.id, cut.id, post.id, window_key)
    return post.id


def _claim_and_publish(post_id: int, window_key: str, state_action: str) -> bool:
    """Claim ``window_key`` for the post, publish it, record ``last_publish_at``."""
    with session_scope() as session:
        # Atomic claim: only one scheduler (local dashboard vs headless runner)
        # can win when both tick at the same time.
        claimed = session.execute(
            update(ThreadsPost)
            .where(ThreadsPost.id == post_id, ThreadsPost.status == STATUS_QUEUED)
            # ``published_window_key`` is written here, inside the claim
            # transaction, so it can never disagree with ``last_window_key``
            # below — a separate write after the publish could be lost.
            .values(status=STATUS_PUBLISHING, error="", pinned_window_key="",
                    published_window_key=window_key)
        ).rowcount
        if claimed != 1:
            return False
        # Consume the window in the SAME transaction that claims the post, so the
        # two can never disagree: either this window is spent and a post owns it,
        # or neither happened and the next tick retries cleanly.
        #
        # Marking the window spent *before* the claim (as this used to) meant any
        # failure in between silently cost a whole slot: the window was gone, yet
        # the post was still ``queued``, so it just slid to the next window and
        # nothing anywhere recorded a skip. A laptop waking from sleep hits this
        # every time — the first post-wake tick runs the claim on a pooled
        # connection whose TCP state died during sleep.
        state = _get_state(session)
        state.last_window_key = window_key
        state.updated_at = utcnow()

    settings = load_settings()
    retries = max(0, int(settings.get("scheduler.publish_retries", 1)))
    retry_delay = max(0, int(settings.get("scheduler.publish_retry_delay_seconds", 30)))

    ok = False
    mark_publishing(post_id)
    try:
        # Auto-retry once (by default) before giving up: a first-of-day publish
        # can fail on a transient hiccup (token refresh, Meta processing) that
        # succeeds on a second attempt, sparing the operator a silent failure.
        for attempt in range(retries + 1):
            with session_scope() as session:
                post = session.get(ThreadsPost, post_id)
                if post is None:
                    return False
                try:
                    publish_post(session, post)
                    ok = True
                except Exception as exc:
                    log.warning(
                        "Queue post %s publish attempt %d/%d failed: %s",
                        post_id, attempt + 1, retries + 1, exc,
                    )
            if ok:
                break
            if attempt < retries:
                time.sleep(retry_delay)
        if not ok:
            log.warning("Queue post %s failed after %d attempt(s)", post_id, retries + 1)
        if ok:
            # After the publish transaction above has committed (see
            # publish_paired_reel), and still inside the mark_publishing window
            # so crash recovery can tell an in-flight reel from a stranded one.
            publish_paired_reel(post_id)
    finally:
        clear_publishing(post_id)

    with session_scope() as session:
        state = _get_state(session)
        if ok:
            state.last_publish_at = utcnow()
            state.last_action = state_action
        else:
            state.last_action = f"publish_failed:{post_id}"
        state.updated_at = utcnow()
    return ok


def _within_active_hours(now_local: dt.datetime) -> bool:
    settings = load_settings()
    start_h, start_m = _parse_hhmm(settings.get("scheduler.active_hours_start", "08:00"))
    end_h, end_m = _parse_hhmm(settings.get("scheduler.active_hours_end", "22:00"))
    mins = now_local.hour * 60 + now_local.minute
    return (start_h * 60 + start_m) <= mins < (end_h * 60 + end_m)


def _earliest_due_window(
    day: dt.date,
    windows: list[dt.datetime],
    now: dt.datetime,
    last_window_key: str,
) -> int | None:
    """Index of the earliest window that has fired and is not yet processed."""
    for i, win in enumerate(windows):
        if now < win:
            break
        key = _window_key(day, i)
        if last_window_key and last_window_key.startswith(day.isoformat()) and last_window_key >= key:
            continue
        return i
    return None


def run_window_tick() -> str | None:
    """Evaluate due posting windows. Returns the action taken (or None)."""
    settings = load_settings()
    if not settings.get("scheduler.enabled", True):
        return None

    # Before the auth and active-hours gates: staging only writes a queued row,
    # and a promo that shows on the calendar early is a promo the operator can
    # still edit or decline. A failure here must never block publishing.
    try:
        staged = ensure_promo_staged()
        if staged:
            log.info("Promo rotation: %s", staged)
    except Exception:
        log.exception("Promo staging failed")

    # Same contract for the evergreen-winners repost rotation.
    try:
        staged = ensure_repost_staged()
        if staged:
            log.info("Repost rotation: %s", staged)
    except Exception:
        log.exception("Repost staging failed")

    if not threads_api.is_authenticated():
        return None

    tz = _tz()
    now = utcnow()
    now_local = now.astimezone(tz)
    if not _within_active_hours(now_local):
        return None

    day = now_local.date()
    windows = _windows_for_day(day, tz)

    with session_scope() as session:
        state = _get_state(session)
        due_index = _earliest_due_window(day, windows, now, state.last_window_key or "")
        if due_index is None:
            return None

        key = _window_key(day, due_index)

        if not _spacing_ok(state, now):
            state.last_window_key = key
            state.last_action = f"spacing_block:{key}"
            state.updated_at = utcnow()
            return f"spacing_block:{key}"

        head = _queue_head_for_window(session, key)
        if head is not None:
            post_id = head.id
            verb = "publish"
        else:
            # Empty queue: re-air an evergreen post rather than going silent
            # (scheduler.placement.filler). Real queued content always wins —
            # this runs only when the plan left this window with nothing.
            post_id = _stage_filler_jit(session, state, key, now)
            verb = "filler_publish"
            if post_id is None:
                state.last_window_key = key
                state.last_action = f"empty:{key}"
                state.updated_at = utcnow()
                return f"empty:{key}"

    # ``_claim_and_publish`` marks the window spent as part of claiming the post.
    # Nothing is recorded here, so a queue-publish failure below leaves the
    # window due and the next tick picks it up again instead of dropping the
    # slot. (A filler publish already consumed the window when it staged; if
    # it fails the slot is spent, same as the empty window it replaced.)
    action = f"{verb}:{key}:post={post_id}"
    if _claim_and_publish(post_id, key, action):
        log.info("Published queue post %s at window %s", post_id, key)
        return action

    with session_scope() as session:
        state = _get_state(session)
        state.last_action = f"publish_failed:{key}:post={post_id}"
        state.updated_at = utcnow()
    return f"publish_failed:{key}:post={post_id}"


def annotate_queued_posts(limit: int = 2) -> int:
    """Ground-truth facet tagging for queued posts, at queue time.

    Footage annotation used to run after publishing — too late for the
    placement gates to use it. Running it here, on the trimmed clip that will
    actually ship, gives the scheduler its facets before anything is placed
    while preserving the learning loop's ground-truth semantics (same file).
    The ``footage_scored_at`` guard keeps the total at one call per post, so
    moving the annotation forward doesn't change LLM spend; the per-tick
    ``limit`` just keeps any single tick short. Headless runners without the
    operator's files skip silently — the dashboard's tick picks those up.

    A queued post whose clip file changes after annotation (captions toggled,
    which swaps in the subtitled variant) is re-annotated: the requeue path
    clears ``footage_scored_at`` when it swaps the file.
    """
    settings = load_settings()
    if not settings.get("vision.enabled", True):
        return 0
    from .db import active_traits_by_facet
    from .vision import annotate_post_footage

    done = 0
    with session_scope() as session:
        vocab = active_traits_by_facet(session)
        posts = session.execute(
            select(ThreadsPost).where(
                ThreadsPost.status == STATUS_QUEUED,
                ThreadsPost.footage_scored_at.is_(None),
                ThreadsPost.clip_local_path != "",
            ).order_by(ThreadsPost.created_at.asc())
        ).scalars().all()
        for post in posts:
            if done >= limit:
                break
            if not Path(post.clip_local_path).expanduser().exists():
                continue  # not this machine's file — another runner has it
            if annotate_post_footage(post, settings, vocab["subject"],
                                     format_traits=vocab["format"]):
                done += 1
    return done


def expired_queued_posts(session) -> list[ThreadsPost]:
    """Queued posts whose shelf life has run out (scored mode only).

    The placement engine never places an expired post, so without this they
    would just sink silently. Surfaced in the notifications list instead, for
    the operator to publish by hand, re-shelve, or delete;
    ``attention_dismissed_at`` acknowledges one without deleting it.
    """
    posts = _queue_regular(session)
    if not posts:
        return []
    ctx = build_placement_context(session, posts)
    if ctx is None:
        return []  # FIFO mode has no expiry
    today = utcnow().astimezone(_tz()).date()
    return [p for p in posts
            if ctx.expired(p.id, today) and p.attention_dismissed_at is None]


def run_metrics_poll() -> int:
    """Poll recent post insights when the poll interval has elapsed."""
    if not threads_api.is_authenticated():
        return 0
    settings = load_settings()
    interval = dt.timedelta(
        minutes=int(settings.get("scheduler.metrics_poll_interval_minutes", 15))
    )
    with session_scope() as session:
        state = _get_state(session)
        now = utcnow()
        last = state.last_metrics_poll_at
        if last is not None:
            if last.tzinfo is None:
                last = last.replace(tzinfo=dt.timezone.utc)
            if now - last < interval:
                return 0
        n = poll_recent_metrics(session)
        state.last_metrics_poll_at = now
        state.updated_at = now
        return n


def scheduler_status(session) -> dict:
    """Snapshot of scheduler state for the Posts UI panel."""
    settings = load_settings()
    tz = _tz()
    now = utcnow()
    now_local = now.astimezone(tz)
    day = now_local.date()
    windows = _windows_for_day(day, tz)
    state = _get_state(session)

    due_index = _earliest_due_window(day, windows, now, state.last_window_key or "")
    next_window_local = None
    next_window_key = None
    due_now = False
    if due_index is not None:
        next_window_local = windows[due_index].astimezone(tz)
        next_window_key = _window_key(day, due_index)
        due_now = True
    else:
        for i, win in enumerate(windows):
            if now < win:
                next_window_local = win.astimezone(tz)
                next_window_key = _window_key(day, i)
                break
        if next_window_local is None:
            tomorrow = day + dt.timedelta(days=1)
            tw = _windows_for_day(tomorrow, tz)
            if tw:
                next_window_local = tw[0].astimezone(tz)
                next_window_key = _window_key(tomorrow, 0)

    queue_count = session.execute(
        select(func.count(ThreadsPost.id)).where(ThreadsPost.status == STATUS_QUEUED)
    ).scalar_one()

    return {
        "enabled": bool(settings.get("scheduler.enabled", True)),
        "timezone": str(tz),
        "windows": settings.get("scheduler.windows") or [],
        "next_window_local": next_window_local,
        "next_window_key": next_window_key,
        "due_now": due_now,
        "last_window_key": state.last_window_key or "",
        "last_publish_at": state.last_publish_at,
        "last_action": state.last_action or "",
        "spacing_floor_minutes": int(settings.get("scheduler.spacing_floor_minutes", 90)),
        "queue_count": queue_count,
        "within_active_hours": _within_active_hours(now_local),
    }


def _upcoming_window_slots(
    start_day: dt.date,
    end_day: dt.date,
    *,
    now: dt.datetime | None = None,
    last_window_key: str = "",
) -> list[tuple[str, dt.datetime, int]]:
    """Return ``(window_key, utc_dt, index)`` for upcoming (not-yet-processed) windows."""
    tz = _tz()
    now = now or utcnow()
    slots: list[tuple[str, dt.datetime, int]] = []
    d = start_day
    while d <= end_day:
        for i, win in enumerate(_windows_for_day(d, tz)):
            key = _window_key(d, i)
            already = (
                last_window_key
                and last_window_key.startswith(d.isoformat())
                and last_window_key >= key
            )
            if win <= now or already:
                continue
            slots.append((key, win, i))
        d += dt.timedelta(days=1)
    return slots


def _score_detail(decision) -> str:
    """One-line score breakdown for a placed slot's hover tooltip, so 'why is
    this post here' is answerable from the calendar."""
    if decision is None:
        return ""
    if decision.pinned:
        return "Pinned by hand — bypasses scoring"
    p = decision.parts or {}
    if decision.score is None or not p:
        return ""
    text = (f"score {decision.score:g} = urgency {p.get('urgency', 0):g}"
            f" + patience {p.get('patience', 0):g}"
            f" − variety {p.get('variety_penalty', 0):g}"
            f" − repost {p.get('repost_penalty', 0):g}")
    if decision.relax_step:
        text += f" (relaxed gates: step {decision.relax_step})"
    return text


def _post_display_title(p) -> str:
    """Best label for a post: its cut's short calendar name (sized to fit the
    calendar's window slots), else the full clip title, else the source video
    title, else — for cut-less posts like imported Threads history — the
    post's own short calendar name condensed from its caption. When the post's
    source video has a programming category, its emoji leads the label so the
    calendar/queue shows the channel mix — including how promos are spaced —
    at a glance."""
    title = ""
    if p.cut and (p.cut.calendar_name or "").strip():
        title = p.cut.calendar_name
    elif p.cut and (p.cut.clip_title or "").strip():
        title = p.cut.clip_title
    elif p.candidate and p.candidate.title:
        title = p.candidate.title
    else:
        title = p.calendar_name or ""
    cat = category_by_slug(p.candidate.category if p.candidate else "")
    if cat and cat["emoji"]:
        return f"{cat['emoji']} {title}".strip()
    return title


def window_time_labels(day: dt.date | None = None) -> list[str]:
    """Posting windows as operator-local labels, in window order.

    The calendar times every card in the operator's zone, so its empty slots
    have to be labelled the same way — falling back to the raw scheduler-zone
    strings from settings made a vacant slot read as though it were the window
    the post directly beneath it had published into.
    """
    tz = _tz()
    day = day or utcnow().astimezone(tz).date()
    return [w.astimezone().strftime("%-I:%M %p") for w in _windows_for_day(day, tz)]


def _assign_published_slots(
    day_posts: list[ThreadsPost],
    day_windows: list[dt.datetime],
    taken: set[int],
) -> dict[int, int]:
    """Map one day's published posts onto that day's posting-window slots.

    Returns ``{post_id: window_index}``. A post lands in the window it ran
    closest to, so a day whose 10:00 window failed shows an empty first slot
    rather than sliding the 13:00 post up into it. Two constraints hold: list
    order is preserved (an earlier post never occupies a later post's slot), and
    slots already held by an upcoming open/queued entry are off limits.

    Nothing records which window a post was published from — manual publishes
    belong to no window at all — so placement is inferred from publish time.
    Days that ran more posts than there are windows leave the worst-fitting
    ones unmapped for the caller to surface as an overflow count.
    """
    free = [i for i in range(len(day_windows)) if i not in taken]
    if not day_posts or not free:
        return {}

    times = []
    for p in day_posts:
        when = p.published_at
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        times.append(when)

    # Order-preserving assignment: fill as many slots as possible, then minimise
    # total distance between publish time and window time. State is
    # ``(-matched, cost)`` so plain tuple comparison prefers more matches first.
    k, n = len(times), len(free)
    dp: list[list[tuple[int, float]]] = [[(0, 0.0)] * (n + 1) for _ in range(k + 1)]
    back: list[list[str]] = [[""] * (n + 1) for _ in range(k + 1)]
    for i in range(k + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            best: tuple[int, float] | None = None
            move = ""
            if i > 0 and (best is None or dp[i - 1][j] < best):
                best, move = dp[i - 1][j], "skip_post"
            if j > 0 and (best is None or dp[i][j - 1] < best):
                best, move = dp[i][j - 1], "skip_slot"
            if i > 0 and j > 0:
                matched, cost = dp[i - 1][j - 1]
                gap = abs((times[i - 1] - day_windows[free[j - 1]]).total_seconds())
                cand = (matched - 1, cost + gap)
                if best is None or cand < best:
                    best, move = cand, "match"
            dp[i][j] = best or (0, 0.0)
            back[i][j] = move

    out: dict[int, int] = {}
    i, j = k, n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "match":
            out[day_posts[i - 1].id] = free[j - 1]
            i -= 1
            j -= 1
        elif move == "skip_post":
            i -= 1
        else:
            j -= 1
    return out


def build_window_plan(
    session,
    start_local: dt.datetime,
    end_local: dt.datetime,
    *,
    horizon_days: int | None = None,
) -> list[dict]:
    """Build a linear plan of posting windows with queue assignments + open placeholders.

    Each entry is a slot dict for the calendar/queue UI:
      kind: open | queued | published
      window_key, sort (operator-local), time, day, caption, post_id, …

    Upcoming windows always appear (empty = ``open``). Published posts in range
    are attached when their publish time falls near a window; otherwise they are
    listed as standalone published entries.
    """
    tz = _tz()
    now = utcnow()
    state = _get_state(session)
    # Normalize range bounds to aware datetimes in the operator-local zone
    # (calendar passes naive local midnights).
    if start_local.tzinfo is None:
        start_local = start_local.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
    if end_local.tzinfo is None:
        end_local = end_local.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
    start_day = start_local.astimezone(tz).date()
    end_day = (end_local - dt.timedelta(seconds=1)).astimezone(tz).date()
    if horizon_days is not None:
        horizon_end = now.astimezone(tz).date() + dt.timedelta(days=horizon_days)
        if horizon_end < end_day:
            end_day = horizon_end

    queued = session.execute(
        select(ThreadsPost)
        .options(
            selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
            selectinload(ThreadsPost.cut),
        )
        .where(ThreadsPost.status == STATUS_QUEUED)
        .order_by(ThreadsPost.created_at.asc())
    ).scalars().all()
    regular = list(queued)

    start_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)
    published = session.execute(
        select(ThreadsPost)
        .options(
            selectinload(ThreadsPost.candidate),
            selectinload(ThreadsPost.cut),
        )
        .where(
            ThreadsPost.status == "published",
            ThreadsPost.published_at.is_not(None),
            ThreadsPost.published_at >= start_utc,
            ThreadsPost.published_at < end_utc,
        ).order_by(ThreadsPost.published_at.asc())
    ).scalars().all()

    plan: list[dict] = []

    upcoming = _upcoming_window_slots(
        max(start_day, now.astimezone(tz).date()),
        end_day,
        now=now,
        last_window_key=state.last_window_key or "",
    )
    # Only slots that fall inside the requested local range.
    visible = []
    for key, win_utc, idx in upcoming:
        local = win_utc.astimezone()
        if local < start_local or local >= end_local:
            continue
        visible.append((key, win_utc, idx, local))

    # Stale-check and assign over the full pin horizon, NOT the requested view
    # range: a narrow view (e.g. one week) must neither wipe pins that target
    # windows outside it nor FIFO-fill visible slots with posts that are
    # actually pinned/projected beyond it.
    today = now.astimezone(tz).date()
    full_upcoming = _upcoming_window_slots(
        today, today + dt.timedelta(days=PIN_HORIZON_DAYS),
        now=now, last_window_key=state.last_window_key or "",
    )
    full_keys = [k for k, _, _ in full_upcoming]
    _clear_stale_pins(regular, set(full_keys))
    assignment, ctx = _assign_with_mode(session, regular, full_keys)
    post_by_key = dict(zip(full_keys, assignment))
    decisions = ctx.decisions if ctx is not None else {}

    # Project the rerun program into windows the queue leaves open, walking
    # the FULL horizon (not just the visible range) so the projection is
    # stable across views. This is display only — nothing is staged until a
    # window actually comes due (the just-in-time filler in run_window_tick
    # uses the same rotation and picker, so what airs is what was projected,
    # modulo whatever reality changes in between). Fresh content displaces
    # reruns by construction: real posts claim windows first, reruns only
    # ever take the leftovers.
    rerun_by_key: dict[str, dict] = {}
    fcfg = _filler_config()
    if fcfg is not None:
        facts, cand_last = _filler_rotation(session, fcfg)
        source_gap = float(_placement_settings(load_settings()).same_source_days)
        for key, _win_utc, _idx in full_upcoming:
            day_of = window_key_date(key)
            if day_of is None:
                continue
            assigned = post_by_key.get(key)
            if assigned is not None:
                # A real post occupies this window; its source video blocks
                # sibling reruns around it just like a real airing would.
                if assigned.candidate_pk is not None:
                    prev = cand_last.get(assigned.candidate_pk)
                    if prev is None or day_of > prev:
                        cand_last[assigned.candidate_pk] = day_of
                continue
            pick = _pick_filler(facts, day_of, cand_last, source_gap)
            if pick is None:
                continue
            rerun_by_key[key] = pick
            pick["last_aired"] = day_of
            if pick["candidate_pk"] is not None:
                cand_last[pick["candidate_pk"]] = day_of

    for key, _win_utc, idx, local in visible:
        post = post_by_key.get(key)
        rerun = rerun_by_key.get(key) if post is None else None
        if post is None and rerun is not None:
            prior = rerun["prior"]
            plan.append({
                "kind": "rerun",
                "window_key": key,
                "window_index": idx,
                "sort": local,
                "time": local.strftime("%-I:%M %p"),
                "day": local.day,
                "date_label": local.strftime("%a %-d"),
                "caption": (prior.caption or "").strip(),
                "status": "rerun",
                # Links to the ORIGINAL airing: no row exists for the rerun
                # until its window comes due and the filler stages it.
                "post_id": prior.id,
                "video_id": prior.candidate.id if prior.candidate else None,
                "channel": (prior.candidate.channel.call_sign
                            if prior.candidate and prior.candidate.channel else ""),
                "thumbnail": (prior.candidate.thumbnail_url if prior.candidate else ""),
                "title": _post_display_title(prior),
                "permalink": "",
                "projected": True,
                "empty": False,
                "pinned": False,
            })
        elif post is None:
            plan.append({
                "kind": "open",
                "window_key": key,
                "window_index": idx,
                "sort": local,
                "time": local.strftime("%-I:%M %p"),
                "day": local.day,
                "date_label": local.strftime("%a %-d"),
                "caption": "",
                "status": "open",
                "post_id": None,
                "video_id": None,
                "channel": "",
                "thumbnail": "",
                "title": "",
                "permalink": "",
                "projected": True,
                "empty": True,
                "pinned": False,
            })
        else:
            decision = decisions.get(key)
            plan.append({
                "kind": "queued",
                "window_key": key,
                "window_index": idx,
                "sort": local,
                "time": local.strftime("%-I:%M %p"),
                "day": local.day,
                "date_label": local.strftime("%a %-d"),
                "caption": (post.caption or "").strip(),
                "status": "queued",
                "post_id": post.id,
                "video_id": post.candidate.id if post.candidate else None,
                "channel": (post.candidate.channel.call_sign
                            if post.candidate and post.candidate.channel else ""),
                "thumbnail": (post.candidate.thumbnail_url if post.candidate else ""),
                "title": _post_display_title(post),
                "permalink": post.permalink,
                "projected": True,
                "empty": False,
                "pinned": bool((post.pinned_window_key or "").strip()),
                # Scored-mode explainability: which relaxation step filled the
                # slot (0 = clean; >=1 means gates were loosened, a sign the
                # queue is shallow or concentrated) and the score breakdown —
                # flattened for tooltips, structured for the calendar popover.
                "relax_step": (decision.relax_step
                               if decision is not None and not decision.pinned
                               else None),
                "score_detail": _score_detail(decision),
                "score": (decision.score
                          if decision is not None and not decision.pinned
                          else None),
                "score_parts": (dict(decision.parts)
                                if decision is not None and not decision.pinned
                                and decision.parts else None),
            })

    # Published posts in range (for calendar history). The month grid lays each
    # day out as its fixed posting windows, so a post needs a window index to be
    # visible there at all. Matching only near-exact window times meant manual
    # publishes and posts that ran long matched nothing and vanished from the
    # calendar despite having gone out. Instead, each day's posts fill that
    # day's remaining slots in chronological order, so a cell always reads
    # top-to-bottom in the order things actually published. Slots already held
    # by an upcoming open/queued entry are never taken; posts beyond the day's
    # window count get no slot and are surfaced as a "+N" badge by the template.
    # Slots are keyed by the OPERATOR-LOCAL date, because that is what the grid
    # buckets cells by (``sort``/``day``). Keying by the scheduler zone instead
    # would misfile any post published after midnight ET but before midnight
    # locally: it renders on the previous local day yet would claim a slot on
    # the next one, pushing that day's real posts out of their cell.
    claimed: dict[dt.date, set[int]] = {}
    for e in plan:
        if e["window_index"] is not None:
            claimed.setdefault(e["sort"].date(), set()).add(e["window_index"])

    by_day: dict[dt.date, list[ThreadsPost]] = {}
    for p in published:
        when = p.published_at
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        by_day.setdefault(when.astimezone().date(), []).append(p)

    slot_for_post: dict[int, int] = {}
    for pub_day, day_posts in by_day.items():
        slot_for_post.update(_assign_published_slots(
            day_posts, _windows_for_day(pub_day, tz), claimed.get(pub_day, set())
        ))

    for p in published:
        when = p.published_at
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        local = when.astimezone()
        matched_idx = slot_for_post.get(p.id)
        matched_key = (
            _window_key(local.date(), matched_idx) if matched_idx is not None else ""
        )
        plan.append({
            "kind": "published",
            "window_key": matched_key,
            "window_index": matched_idx,
            "sort": local,
            "time": local.strftime("%-I:%M %p"),
            "day": local.day,
            "date_label": local.strftime("%a %-d"),
            "caption": (p.caption or "").strip(),
            "status": "published",
            "post_id": p.id,
            "video_id": p.candidate.id if p.candidate else None,
            "channel": (p.candidate.channel.call_sign
                        if p.candidate and p.candidate.channel else ""),
            "thumbnail": (p.candidate.thumbnail_url if p.candidate else ""),
            "title": _post_display_title(p),
            "permalink": p.permalink,
            "projected": False,
            "empty": False,
            "pinned": False,
        })

    plan.extend(_missed_window_slots(plan, state, now, start_local, end_local))

    plan.sort(key=lambda e: e["sort"])
    return plan


def _missed_window_slots(
    plan: list[dict],
    state: SchedulerState,
    now: dt.datetime,
    start_local: dt.datetime,
    end_local: dt.datetime,
) -> list[dict]:
    """Entries for today's windows that the scheduler spent without publishing.

    A window that came and went with nothing live behind it leaves no post for
    the calendar to draw, so the grid rendered it as an anonymous vacant slot —
    identical to a day that simply had nothing queued. That made a lost slot
    (empty queue, spacing block, or a publish that failed outright) invisible.

    Only today: ``SchedulerState`` keeps a single ``last_window_key``, so there
    is no record of which windows older days actually spent. A window that has
    fired but is not spent yet is left alone — it is still due, and the next
    tick will publish into it.
    """
    tz = _tz()
    today_local = now.astimezone().date()
    filled = {
        e["window_index"] for e in plan
        if e["window_index"] is not None and e["sort"].date() == today_local
    }
    scheduler_day = now.astimezone(tz).date()
    last_key = state.last_window_key or ""
    out: list[dict] = []
    for idx, win_utc in enumerate(_windows_for_day(scheduler_day, tz)):
        key = _window_key(scheduler_day, idx)
        spent = last_key.startswith(scheduler_day.isoformat()) and last_key >= key
        if not spent or win_utc > now or idx in filled:
            continue
        local = win_utc.astimezone()
        if local.date() != today_local or local < start_local or local >= end_local:
            continue
        out.append({
            "kind": "missed",
            "window_key": key,
            "window_index": idx,
            "sort": local,
            "time": local.strftime("%-I:%M %p"),
            "day": local.day,
            "date_label": local.strftime("%a %-d"),
            "caption": "",
            "status": "missed",
            "post_id": None,
            "video_id": None,
            "channel": "",
            "thumbnail": "",
            "title": "",
            "permalink": "",
            "projected": False,
            "empty": True,
            "pinned": False,
        })
    return out


def projected_window_slots(
    session,
    start_local: dt.datetime,
    end_local: dt.datetime,
) -> list[dict]:
    """Backward-compatible alias: non-empty projected/queued/open slots for a range."""
    return [
        e for e in build_window_plan(session, start_local, end_local)
        if e["kind"] in ("queued", "open")
    ]


def projected_slot_for_post(session, post_id: int, horizon_days: int = 60) -> dict | None:
    """The projected publishing slot for one queued post, using the same plan the
    calendar shows. Returns the slot dict (with ``sort`` = local publish datetime,
    ``time``/``date_label`` labels) or ``None`` if the post isn't a queued post
    that lands within ``horizon_days``."""
    now_local = utcnow().astimezone()
    start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(days=horizon_days + 1)
    for entry in build_window_plan(session, start, end, horizon_days=horizon_days):
        if entry.get("post_id") == post_id and entry["kind"] == "queued":
            return entry
    return None


def run_tick() -> None:
    """One scheduler loop iteration: recover → annotate → metrics → window."""
    try:
        with session_scope() as session:
            n = recover_stuck_publishing(session, only_inactive=True)
        if n:
            log.info("Recovered %d post(s) stuck in 'publishing'", n)
    except Exception:
        log.exception("Stuck-publish recovery failed")

    # Queue-time facet annotation, before the window evaluation so a post
    # queued minutes ago can still be placed on its facets this same tick.
    try:
        n = annotate_queued_posts()
        if n:
            log.info("Annotated %d queued post(s) with footage facets", n)
    except Exception:
        log.exception("Queued-post annotation failed")

    try:
        n = run_metrics_poll()
        if n:
            log.info("Metrics poll took %d snapshot(s)", n)
    except Exception:
        log.exception("Metrics poll failed")

    try:
        action = run_window_tick()
        if action:
            log.info("Window tick: %s", action)
    except Exception:
        log.exception("Window tick failed")

    # Keep the rerun-outlook cache warm off the request path. The rotation's
    # metric scan (every published post's snapshot series) takes seconds on a
    # remote database; recomputing it here means no page view — and no
    # pagecache rebuild, which every write triggers — ever pays that cost.
    try:
        cached = _recycle_ui_cache
        if cached is None or time.monotonic() - cached[0] >= _RECYCLE_UI_TTL_S:
            with session_scope() as session:
                recycle_overview(session)
    except Exception:
        log.exception("Rerun-outlook warmup failed")


def start_scheduler_thread(interval_seconds: int = 60) -> None:
    """Start the background adaptive-scheduler loop. Idempotent."""
    global _thread
    if _thread and _thread.is_alive():
        return

    def _loop() -> None:
        log.info("Adaptive scheduler started (checking every %ss)", interval_seconds)
        # Nothing can be mid-publish in a freshly started process, so any post
        # sitting in 'publishing' is an orphan from a previous run — recover it.
        try:
            with session_scope() as session:
                n = recover_stuck_publishing(session, only_inactive=False)
            if n:
                log.info("Startup: recovered %d post(s) stuck in 'publishing'", n)
        except Exception:
            log.exception("Startup stuck-publish recovery failed")
        while True:
            try:
                run_tick()
            except Exception:
                log.exception("Scheduler tick failed; will retry")
            time.sleep(interval_seconds)

    _thread = threading.Thread(target=_loop, daemon=True, name="threads-scheduler")
    _thread.start()
