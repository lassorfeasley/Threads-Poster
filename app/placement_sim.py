"""Replay simulation for scored placement — tune weights here, not in prod.

Replays real history through the scored placement engine: every post published
in the window is put back in the queue at its actual ``created_at`` and the
engine fills the period's posting windows chronologically, each window seeing
only the posts that had been queued by then. Because the engine is pure, the
replay exercises exactly the code that would run live.

The report is what the plan needs answered before flipping
``scheduler.placement.mode`` to ``scored``:

- facet distribution per week (did variety actually improve?)
- days between same-source clips, replayed vs. what actually happened
- how often gates had to relax (a run of relaxed fills means the queue is
  too shallow for the configured gates)
- worst-case queue wait (is patience strong enough that nothing starves?)
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .models import Candidate, ThreadsPost, utcnow
from .placement import choose_for_window, window_key_date
from .scheduler import (
    _tz,
    _window_key,
    _windows_for_day,
    build_placement_context,
)


def _gaps_by_candidate(placements: list[tuple[dt.date, ThreadsPost]]) -> list[int]:
    """Day gaps between consecutive airings of the same source video."""
    days_by_cand: dict[int, list[dt.date]] = defaultdict(list)
    for day, post in placements:
        if post.candidate_pk:
            days_by_cand[post.candidate_pk].append(day)
    gaps: list[int] = []
    for days in days_by_cand.values():
        days.sort()
        gaps.extend((b - a).days for a, b in zip(days, days[1:]))
    return gaps


def run_replay(session, days: int = 90) -> dict:
    """Replay the last ``days`` of publish history through scored placement."""
    now = utcnow()
    tz = _tz()
    start = now - dt.timedelta(days=days)

    posts = session.execute(
        select(ThreadsPost)
        .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel))
        .where(ThreadsPost.status == "published",
               ThreadsPost.published_at.is_not(None),
               ThreadsPost.published_at >= start)
        .order_by(ThreadsPost.published_at.asc())
    ).scalars().all()
    if not posts:
        return {"posts": 0}

    # The context is built as if every post were queued today; history that
    # predates the replay window is deliberately left out so the replay is
    # self-contained (its own placements build the air-date history).
    ctx = build_placement_context(session, posts, force=True)
    ctx.candidate_air_dates.clear()
    ctx.channel_air_dates.clear()
    ctx.facet_trail.clear()

    def queued_at(p: ThreadsPost) -> dt.datetime:
        when = p.created_at or p.published_at
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        return when

    # Walk every posting window in the replay period, offering each the posts
    # that had been queued by that moment.
    remaining = sorted(posts, key=lambda p: (queued_at(p), p.id))
    placements: list[tuple[dt.date, ThreadsPost]] = []
    step_counts: Counter[int] = Counter()
    waits_days: list[float] = []
    facets_by_week: dict[str, Counter] = defaultdict(Counter)
    empty_windows = 0

    day = start.astimezone(tz).date()
    end_day = now.astimezone(tz).date()
    while day <= end_day:
        for idx, win_utc in enumerate(_windows_for_day(day, tz)):
            if win_utc < start or win_utc > now:
                continue
            key = _window_key(day, idx)
            available = [p for p in remaining if queued_at(p) <= win_utc]
            pick, step, score, parts = choose_for_window(ctx, available, key)
            if pick is None:
                empty_windows += 1
                continue
            remaining.remove(pick)
            win_day = window_key_date(key)
            ctx.note_air_dates(pick.id, win_day)
            ctx.record(pick.id, key, relax_step=step, score=score, parts=parts)
            placements.append((win_day, pick))
            step_counts[step] += 1
            waits_days.append((win_utc - queued_at(pick)).total_seconds() / 86400)
            week = f"{win_day.isocalendar().year}-W{win_day.isocalendar().week:02d}"
            for f in sorted(ctx.facts_for(pick.id).facets) or ["(untagged)"]:
                facets_by_week[week][f] += 1
        day += dt.timedelta(days=1)

    # What actually happened, for contrast.
    actual: list[tuple[dt.date, ThreadsPost]] = []
    for p in posts:
        when = p.published_at
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        actual.append((when.astimezone(tz).date(), p))

    gate = ctx.settings.same_source_days
    sim_gaps = _gaps_by_candidate(placements)
    actual_gaps = _gaps_by_candidate(actual)
    return {
        "posts": len(posts),
        "placed": len(placements),
        "unplaced": len(remaining),
        "empty_windows": empty_windows,
        "relaxation_steps": dict(sorted(step_counts.items())),
        "same_source_gate_days": gate,
        "sim_same_source_gaps": sorted(sim_gaps),
        "sim_same_source_violations": sum(1 for g in sim_gaps if g < gate),
        "actual_same_source_gaps": sorted(actual_gaps),
        "actual_same_source_violations": sum(1 for g in actual_gaps if g < gate),
        "worst_wait_days": round(max(waits_days), 1) if waits_days else 0.0,
        "mean_wait_days": round(sum(waits_days) / len(waits_days), 1) if waits_days else 0.0,
        "facets_by_week": {w: dict(c.most_common()) for w, c in sorted(facets_by_week.items())},
    }


def format_report(r: dict) -> str:
    if not r.get("posts"):
        return "No published posts in the replay window — nothing to simulate."
    lines = [
        f"Replayed {r['posts']} published post(s) through scored placement.",
        f"  placed {r['placed']}, unplaced at end {r['unplaced']}, "
        f"windows left empty {r['empty_windows']}",
        "",
        f"Relaxation steps used (0 = all gates held): {r['relaxation_steps'] or '{}'}",
        "  A run of step>=2 fills means the queue is too shallow or too",
        "  concentrated for the configured gates.",
        "",
        f"Same-source spacing (gate: {r['same_source_gate_days']} days apart):",
        f"  replayed:  {r['sim_same_source_violations']} violation(s), "
        f"gaps {r['sim_same_source_gaps'] or '—'}",
        f"  actual:    {r['actual_same_source_violations']} violation(s), "
        f"gaps {r['actual_same_source_gaps'] or '—'}",
        "",
        f"Queue wait: worst {r['worst_wait_days']}d, mean {r['mean_wait_days']}d",
        "",
        "Facet mix per week (replayed):",
    ]
    for week, counts in r["facets_by_week"].items():
        mix = ", ".join(f"{k} x{v}" for k, v in counts.items())
        lines.append(f"  {week}: {mix}")
    return "\n".join(lines)
