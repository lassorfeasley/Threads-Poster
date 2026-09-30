"""Posting schedule: how many times a day to post, and at what times.

Set from the Posting schedule page and stored in the database, so every runner
(dashboard, Fly worker, Actions cron) reads the same windows — settings.yaml is
baked into each deploy and can't be changed from the app. Each saved schedule
applies from its ``effective_from`` day onward; with none in effect,
``scheduler.windows`` in settings.yaml still applies.

Each schedule can also reserve some of its times for reruns: organic
placement leaves those windows alone and the scheduler stages a rerun into
them ahead of time (``scheduler.ensure_rerun_slots_staged``).

Changes to the times always start on a future day. Window keys are ``YYYY-MM-DD#index``, and
the scheduler marks windows spent by comparing keys against a high-water mark,
so renumbering a day that is already under way could re-fire or skip windows.
Queued posts pinned to affected days are moved to the nearest new window.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time

from sqlalchemy import select

from .config import load_settings
from .db import session_scope
from .models import PostingSchedule, ThreadsPost

log = logging.getLogger("posting_schedule")

DEFAULT_WINDOWS = ["10:00", "14:30", "19:00"]

# Window keys are compared as strings within a day, and the overflow window
# takes the index one past the last regular window. Ten or more windows would
# put "#10" before "#9".
MAX_WINDOWS = 9

# _windows_for_day runs hundreds of times per calendar build and on every
# scheduler tick, so the rows are cached briefly. Saving from this process
# invalidates at once; other runners pick a change up within the TTL, and a
# change never applies before tomorrow anyway.
_CACHE_TTL_S = 60.0
_cache_lock = threading.Lock()
# Each row: (effective_from, windows, rerun slot times).
Row = tuple[dt.date, list[str], list[str]]
_cache: tuple[float, list[Row]] | None = None


def invalidate() -> None:
    global _cache
    with _cache_lock:
        _cache = None


def yaml_windows() -> list[str]:
    return [str(w) for w in (load_settings().get("scheduler.windows") or DEFAULT_WINDOWS)]


def _load_rows(session) -> list[Row]:
    rows = session.execute(
        select(PostingSchedule).order_by(PostingSchedule.effective_from.asc())
    ).scalars().all()
    out = []
    for r in rows:
        try:
            windows = [str(w) for w in json.loads(r.windows or "[]")]
            reruns = [str(w) for w in json.loads(r.rerun_slots or "[]")]
        except ValueError:
            log.warning("Posting schedule %s has unreadable windows; skipped", r.id)
            continue
        if windows:
            out.append((r.effective_from, windows, [w for w in reruns if w in windows]))
    return out


def _rows() -> list[Row]:
    global _cache
    with _cache_lock:
        cached = _cache
    if cached is not None and time.monotonic() - cached[0] < _CACHE_TTL_S:
        return cached[1]
    try:
        with session_scope(read_only=True) as session:
            rows = _load_rows(session)
    except Exception:
        # The table may not exist yet (a runner on older schema) — the YAML
        # windows are the right answer then, not a crashed tick.
        log.warning("Couldn't read posting schedules; using settings.yaml windows",
                    exc_info=True)
        rows = []
    with _cache_lock:
        _cache = (time.monotonic(), rows)
    return rows


def _entry_in(rows: list[Row], day: dt.date) -> tuple[list[str], list[str]]:
    current = None
    for effective, windows, reruns in rows:
        if effective <= day:
            current = (windows, reruns)
        else:
            break
    return current if current is not None else (yaml_windows(), [])


def _windows_in(rows: list[Row], day: dt.date) -> list[str]:
    return _entry_in(rows, day)[0]


def windows_for_date(day: dt.date) -> list[str]:
    """``HH:MM`` windows (scheduler timezone) that apply on ``day``."""
    return _windows_in(_rows(), day)


def rerun_indices_for_date(day: dt.date) -> frozenset[int]:
    """Indices of ``day``'s windows reserved for reruns."""
    windows, reruns = _entry_in(_rows(), day)
    return frozenset(i for i, w in enumerate(windows) if w in reruns)


def is_rerun_slot(key: str) -> bool:
    day_s, sep, idx_s = (key or "").partition("#")
    try:
        day, idx = dt.date.fromisoformat(day_s), int(idx_s)
    except ValueError:
        return False
    return bool(sep) and idx in rerun_indices_for_date(day)


def overview(today: dt.date) -> dict:
    """What the page shows: today's windows and where they come from, plus
    every saved change still ahead."""
    rows = _rows()
    in_effect = [r for r in rows if r[0] <= today]
    windows, reruns = _entry_in(rows, today)
    return {
        "today": today,
        "windows": windows,
        "reruns": reruns,
        "source": "saved" if in_effect else "settings",
        "since": in_effect[-1][0] if in_effect else None,
        "upcoming": [{"effective_from": d, "windows": w, "reruns": r}
                     for d, w, r in rows if d > today],
    }


# --- Validation ---------------------------------------------------------------

def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def normalize(raw: list[str]) -> list[str]:
    """``HH:MM`` strings, chronological, blanks dropped. Raises ValueError on
    anything that isn't a time of day."""
    out = []
    for value in raw:
        value = (value or "").strip()
        if not value:
            continue
        try:
            h, m = value.split(":")
            h, m = int(h), int(m)
        except ValueError:
            raise ValueError(f"'{value}' isn't a time — use HH:MM") from None
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"'{value}' isn't a time of day")
        out.append(f"{h:02d}:{m:02d}")
    return sorted(out, key=_minutes)


def problems(windows: list[str], spacing_floor_minutes: int) -> list[str]:
    """Reasons a schedule can't be saved; empty when it can."""
    errs = []
    if not windows:
        errs.append("Add at least one posting time.")
        return errs
    if len(windows) > MAX_WINDOWS:
        errs.append(f"At most {MAX_WINDOWS} posts a day.")
    if len(set(windows)) != len(windows):
        errs.append("Two posting times are the same.")
    mins = [_minutes(w) for w in windows]
    gaps = [b - a for a, b in zip(mins, mins[1:])]
    if len(mins) > 1:
        gaps.append(mins[0] + 1440 - mins[-1])  # last window to tomorrow's first
    if any(0 < g < spacing_floor_minutes for g in gaps):
        errs.append(f"Posts must be at least {spacing_floor_minutes} minutes apart "
                    "(scheduler.spacing_floor_minutes) — a closer window would "
                    "always be skipped.")
    return errs


def notes(windows: list[str], reruns: list[str] | None = None) -> list[str]:
    """Side effects worth knowing before saving; none block the save."""
    settings = load_settings()
    out = []
    if reruns and not settings.get("scheduler.placement.filler.enabled", False):
        out.append("Rerun slots pick from the filler rotation, which is off "
                   "(scheduler.placement.filler.enabled) — they'll post new "
                   "clips until it's turned on.")
    if settings.get("scheduler.placement.overflow.enabled", False):
        overflow = str(settings.get("scheduler.placement.overflow.time", "21:00"))
        try:
            if windows and _minutes(overflow) <= _minutes(windows[-1]):
                out.append(f"The extra overflow window ({overflow}) has to come after "
                           f"the last post of the day, so it won't open on this "
                           "schedule. Move scheduler.placement.overflow.time later, "
                           "or turn it off.")
        except ValueError:
            pass
    n = len(windows)
    for label, key in (("Promos", "scheduler.promos"),
                       ("Weekly re-airs", "scheduler.placement.reposts")):
        if not settings.get(f"{key}.enabled", False) or not n:
            continue
        raw = settings.get(f"{key}.window_index", "middle" if "promos" in key else 0)
        if isinstance(raw, str) and raw.strip().lower() == "middle":
            idx = n // 2
        else:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                continue
        if 0 <= idx < n:
            out.append(f"{label} will use the {windows[idx]} slot "
                       f"({key}.window_index).")
        else:
            out.append(f"{label} are set to a slot this schedule doesn't have "
                       f"({key}.window_index = {raw}), so they'll stop.")
    return out


# --- Saving -------------------------------------------------------------------

def _overflow_minutes() -> int | None:
    settings = load_settings()
    if not settings.get("scheduler.placement.overflow.enabled", False):
        return None
    try:
        return _minutes(str(settings.get("scheduler.placement.overflow.time", "21:00")))
    except ValueError:
        return None


def _remap_pins(session, from_day: dt.date, old_rows, new_rows) -> dict:
    """Move queued posts pinned on ``from_day`` or later to the nearest window
    of the new schedule. A post whose day has no free window left is unpinned
    and goes back to normal placement."""
    moved = unpinned = 0
    posts = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == "queued",
                                  ThreadsPost.pinned_window_key != "")
    ).scalars().all()
    by_day: dict[dt.date, list[tuple[int, ThreadsPost]]] = {}
    overflow = _overflow_minutes()
    for p in posts:
        day_s, sep, idx_s = p.pinned_window_key.partition("#")
        try:
            day, idx = dt.date.fromisoformat(day_s), int(idx_s)
        except ValueError:
            continue
        if not sep or day < from_day:
            continue
        old = _windows_in(old_rows, day)
        if _windows_in(new_rows, day) == old:
            continue
        if idx < len(old):
            at = _minutes(old[idx])
        elif idx == len(old) and overflow is not None:
            at = overflow
        else:
            p.pinned_window_key = ""
            unpinned += 1
            continue
        by_day.setdefault(day, []).append((at, p))

    for day, pins in by_day.items():
        new = [_minutes(w) for w in _windows_in(new_rows, day)]
        taken: set[int] = set()
        for at, p in sorted(pins, key=lambda t: (t[0], t[1].id)):
            free = [i for i in range(len(new)) if i not in taken]
            if not free:
                p.pinned_window_key = ""
                unpinned += 1
                continue
            best = min(free, key=lambda i: (abs(new[i] - at), i))
            taken.add(best)
            key = f"{day.isoformat()}#{best}"
            if key != p.pinned_window_key or new[best] != at:
                p.pinned_window_key = key
                moved += 1
    return {"moved": moved, "unpinned": unpinned}


def save(session, effective_from: dt.date, windows: list[str], *, today: dt.date,
         reruns: list[str] | None = None) -> dict:
    """Schedule ``windows`` from ``effective_from`` on, with ``reruns`` (a
    subset of ``windows``) reserved for reruns. Replaces any saved change on or
    after that day. Returns ``{"moved", "unpinned"}`` for the pins it had to
    shift.

    Today is allowed only when the times match today's: switching a slot
    between new clips and reruns renumbers nothing."""
    old_rows = _load_rows(session)
    if effective_from < today or (
            effective_from == today and windows != _windows_in(old_rows, today)):
        raise ValueError("A change to the posting times can start tomorrow at the earliest.")
    for r in session.execute(
        select(PostingSchedule).where(PostingSchedule.effective_from >= effective_from)
    ).scalars().all():
        session.delete(r)
    session.flush()
    session.add(PostingSchedule(
        effective_from=effective_from, windows=json.dumps(windows),
        rerun_slots=json.dumps([w for w in (reruns or []) if w in windows])))
    session.flush()
    result = _remap_pins(session, effective_from, old_rows, _load_rows(session))
    invalidate()
    return result


def cancel(session, effective_from: dt.date, *, today: dt.date) -> dict | None:
    """Drop a saved change that hasn't started yet; None when there's no such
    change (or it already took effect — those stay, as history)."""
    if effective_from <= today:
        return None
    row = session.execute(
        select(PostingSchedule).where(PostingSchedule.effective_from == effective_from)
    ).scalar_one_or_none()
    if row is None:
        return None
    old_rows = _load_rows(session)
    session.delete(row)
    session.flush()
    result = _remap_pins(session, effective_from, old_rows, _load_rows(session))
    invalidate()
    return result
