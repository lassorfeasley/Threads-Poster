"""The posting plan: one product for every account.

Three new clips a day, plus one rerun for every 60 posts in the archive — four
posts a day from 60 posts, five from 120, and six (the most) from 180. The
archive counts published posts that weren't themselves reruns.

An account chooses only *when* its posts go out:

- ``waking`` — evenly from 7:00 AM to 10:00 PM in the account's time zone, for
  an audience that mostly lives near it;
- ``always`` — evenly around the clock, for an audience spread across the
  world.

Reruns take the quietest times — the ones nearest the small hours (3:30 AM),
when the fewest people are scrolling — and new clips the busier ones.

The plan writes ordinary posting schedules (app/posting_schedule.py), so the
scheduler on every runner reads it exactly as before. A workspace follows the
plan only once a mode has been chosen; until then its saved schedule is left
alone and the page shows it as custom. ``ensure`` keeps the schedule in step
as the archive grows: the home worker runs it every round, and a change starts
tomorrow, never part-way through today.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from sqlalchemy import func, select

from . import posting_schedule
from .config import WORKSPACE
from .models import AppToken, ThreadsPost, utcnow

log = logging.getLogger("posting_plan")

NEW_PER_DAY = 3
POSTS_PER_RERUN = 60
MAX_RERUNS = 3
MODES = ("waking", "always")

WAKING_FIRST = 7 * 60        # 7:00 AM
WAKING_LAST = 22 * 60        # 10:00 PM
ALWAYS_ANCHOR = 13 * 60      # around the clock, one post lands at 1:00 PM
QUIETEST = 3 * 60 + 30       # reruns go nearest 3:30 AM
ROUND_TO = 5                 # minutes

TOKEN = f"posting_plan:{WORKSPACE}"
if len(TOKEN) > 40:  # AppToken.name is String(40)
    raise ValueError(f"workspace slug too long for the posting plan: {TOKEN!r}")


# --- The rule ------------------------------------------------------------------------

def reruns_for(archive: int) -> int:
    return max(0, min(MAX_RERUNS, int(archive) // POSTS_PER_RERUN))


def per_day(archive: int) -> int:
    return NEW_PER_DAY + reruns_for(archive)


def tiers() -> list[dict]:
    """Each step of the plan: how many archive posts unlock it, and what it gives."""
    return [{"posts": r * POSTS_PER_RERUN, "per_day": NEW_PER_DAY + r, "reruns": r}
            for r in range(MAX_RERUNS + 1)]


def progress(archive: int) -> dict:
    """Where an account stands on the way to its next rerun."""
    reruns = reruns_for(archive)
    maxed = reruns >= MAX_RERUNS
    next_at = None if maxed else (reruns + 1) * POSTS_PER_RERUN
    prev_at = reruns * POSTS_PER_RERUN
    return {
        "archive": int(archive),
        "reruns": reruns,
        "per_day": NEW_PER_DAY + reruns,
        "maxed": maxed,
        "next_at": next_at,
        "to_go": None if maxed else next_at - int(archive),
        # Share of the way from the last step to the next.
        "share": 1.0 if maxed else (int(archive) - prev_at) / POSTS_PER_RERUN,
    }


# --- The times ------------------------------------------------------------------------

def _hhmm(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _rounded(minutes: float) -> int:
    return int(ROUND_TO * round(minutes / ROUND_TO))


def times(n: int, mode: str) -> list[str]:
    """``n`` posting times (``HH:MM``, chronological) for ``mode``."""
    if n <= 0:
        return []
    if mode == "waking":
        if n == 1:
            slots = [_rounded((WAKING_FIRST + WAKING_LAST) / 2)]
        else:
            step = (WAKING_LAST - WAKING_FIRST) / (n - 1)
            slots = [_rounded(WAKING_FIRST + i * step) for i in range(n)]
    else:
        step = 24 * 60 / n
        slots = [_rounded(ALWAYS_ANCHOR + i * step) % (24 * 60) for i in range(n)]
    return sorted(_hhmm(m) for m in slots)


def _from_quietest(hhmm: str) -> int:
    h, m = hhmm.split(":")
    diff = abs(int(h) * 60 + int(m) - QUIETEST)
    return min(diff, 24 * 60 - diff)


def rerun_times(windows: list[str], count: int) -> list[str]:
    """The ``count`` quietest times: those nearest the small hours."""
    if count <= 0:
        return []
    by_quiet = sorted(windows, key=lambda w: (_from_quietest(w), w))
    return sorted(by_quiet[:count])


def plan(archive: int, mode: str) -> dict:
    n = per_day(archive)
    windows = times(n, mode)
    return {"mode": mode, "per_day": n, "windows": windows,
            "reruns": rerun_times(windows, reruns_for(archive))}


# --- State ------------------------------------------------------------------------------

def archive_size(session) -> int:
    """Published posts that weren't themselves reruns."""
    return int(session.execute(
        select(func.count(ThreadsPost.id)).where(
            ThreadsPost.status == "published", ThreadsPost.repost_of_post_pk.is_(None))
    ).scalar_one() or 0)


def load_mode(session) -> str | None:
    """The chosen mode, or None while the workspace keeps a custom schedule."""
    row = session.get(AppToken, TOKEN)
    if row is None or not row.value:
        return None
    try:
        mode = json.loads(row.value).get("mode")
    except (ValueError, AttributeError):
        return None
    return mode if mode in MODES else None


def save_mode(session, mode: str, now: dt.datetime | None = None) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    now = now or utcnow()
    row = session.get(AppToken, TOKEN)
    if row is None:
        row = AppToken(name=TOKEN)
        session.add(row)
    row.value = json.dumps({"mode": mode, "updated_at": now.isoformat()})
    row.updated_at = now


def ensure(session, *, today: dt.date) -> dict | None:
    """Bring the schedule from tomorrow on in line with the plan. Returns what
    it saved, or None when there was nothing to do (no mode chosen, or the
    schedule already matches)."""
    mode = load_mode(session)
    if mode is None:
        return None
    want = plan(archive_size(session), mode)
    tomorrow = today + dt.timedelta(days=1)
    rows = posting_schedule._load_rows(session)
    windows, reruns = posting_schedule._entry_in(rows, tomorrow)
    later = [r for r in rows if r[0] > tomorrow]
    if windows == want["windows"] and sorted(reruns) == want["reruns"] and not later:
        return None
    result = posting_schedule.save(session, tomorrow, want["windows"], today=today,
                                   reruns=want["reruns"])
    log.info("Posting plan (%s): %d a day from %s at %s (reruns at %s)", mode, want["per_day"],
             tomorrow, ", ".join(want["windows"]), ", ".join(want["reruns"]) or "none")
    return {**want, "effective_from": tomorrow, **result}
