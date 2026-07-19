"""Adaptive window scheduler for the Threads post queue.

Publishes the FIFO ``queued`` list at fixed daily windows (US Eastern by
default). At each window, if the most recently published post is still "hot"
(likes gained over the trailing hour exceed a threshold), the queue head is
deferred to the next window — unless a guardrail forces publish (max
deferrals) or skip (last window of the day). Breaking-news posts bypass
windows and publish as soon as the spacing floor allows.

Also drives the frequent metrics poller that feeds the hotness check.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from zoneinfo import ZoneInfo

from sqlalchemy import select

from . import threads_api
from .analytics import is_last_post_hot, poll_recent_metrics
from .config import load_settings
from .db import session_scope
from .models import SchedulerState, ThreadsPost, utcnow
from .publishing import publish_post

log = logging.getLogger("scheduler")

STATUS_QUEUED = "queued"
STATUS_PUBLISHING = "publishing"

_thread: threading.Thread | None = None


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.strip().split(":")
    return int(hour), int(minute)


def _tz() -> ZoneInfo:
    settings = load_settings()
    return ZoneInfo(settings.get("scheduler.timezone", "America/New_York"))


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
    """Non-breaking queued posts in FIFO (created_at) order."""
    return list(session.execute(
        select(ThreadsPost)
        .where(ThreadsPost.status == STATUS_QUEUED, ThreadsPost.is_breaking.is_(False))
        .order_by(ThreadsPost.created_at.asc())
    ).scalars().all())


def _clear_stale_pins(posts: list[ThreadsPost], upcoming_keys: set[str]) -> None:
    """Drop pins that point at windows that are no longer upcoming."""
    for p in posts:
        pin = (p.pinned_window_key or "").strip()
        if pin and pin not in upcoming_keys:
            p.pinned_window_key = ""


def assign_posts_to_windows(
    posts: list[ThreadsPost],
    window_keys: list[str],
) -> list[ThreadsPost | None]:
    """Map queued posts onto window keys: pins first, then FIFO into gaps.

    Returns a list parallel to ``window_keys``. Earlier windows may stay empty
    when a post is pinned to a later slot.
    """
    assignment: list[ThreadsPost | None] = [None] * len(window_keys)
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


def _queue_head_for_window(session, window_key: str) -> ThreadsPost | None:
    """Post that should publish at ``window_key`` (pin-aware)."""
    tz = _tz()
    now = utcnow()
    state = _get_state(session)
    day = now.astimezone(tz).date()
    # Look far enough ahead that reserved pins are visible.
    upcoming = _upcoming_window_slots(
        day, day + dt.timedelta(days=21),
        now=now, last_window_key=state.last_window_key or "",
    )
    keys = [k for k, _, _ in upcoming]
    posts = _queue_regular(session)
    _clear_stale_pins(posts, set(keys))
    if window_key not in keys:
        # Window should still be in upcoming when called from a due tick; fall back.
        pinned = [p for p in posts if (p.pinned_window_key or "") == window_key]
        if pinned:
            return pinned[0]
        return posts[0] if posts else None
    assignment = assign_posts_to_windows(posts, keys)
    return assignment[keys.index(window_key)]


def _breaking_heads(session) -> list[ThreadsPost]:
    return list(session.execute(
        select(ThreadsPost)
        .where(ThreadsPost.status == STATUS_QUEUED, ThreadsPost.is_breaking.is_(True))
        .order_by(ThreadsPost.created_at.asc())
    ).scalars().all())


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
        day, day + dt.timedelta(days=60),
        now=now, last_window_key=state.last_window_key or "",
    )
    keys = [k for k, _, _ in upcoming]
    if window_key not in keys:
        raise ValueError("That window is no longer available")

    post = session.get(ThreadsPost, post_id)
    if post is None or post.status != STATUS_QUEUED or post.is_breaking:
        raise ValueError("Only a non-breaking queued post can be pinned")

    posts = _queue_regular(session)
    _clear_stale_pins(posts, set(keys))
    assignment = assign_posts_to_windows(posts, keys)

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


def _claim_and_publish(post_id: int, state_action: str) -> bool:
    """Flip post to publishing, publish it, update SchedulerState.last_publish_at."""
    with session_scope() as session:
        post = session.get(ThreadsPost, post_id)
        if post is None or post.status != STATUS_QUEUED:
            return False
        post.status = STATUS_PUBLISHING
        post.error = ""
        post.pinned_window_key = ""

    ok = False
    with session_scope() as session:
        post = session.get(ThreadsPost, post_id)
        if post is None:
            return False
        try:
            publish_post(session, post)
            ok = True
        except Exception as exc:
            log.warning("Queue post %s failed: %s", post_id, exc)

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


def run_breaking_posts() -> int:
    """Publish queued breaking posts immediately (spacing floor only)."""
    if not threads_api.is_authenticated():
        return 0

    published = 0
    with session_scope() as session:
        state = _get_state(session)
        now = utcnow()
        if not _spacing_ok(state, now):
            return 0
        heads = _breaking_heads(session)
        ids = [p.id for p in heads]

    for pid in ids:
        with session_scope() as session:
            state = _get_state(session)
            if not _spacing_ok(state, utcnow()):
                break
        if _claim_and_publish(pid, f"breaking:{pid}"):
            published += 1
            log.info("Published breaking post %s", pid)
        else:
            break  # spacing or failure — wait for next tick
    return published


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
    if not threads_api.is_authenticated():
        return None

    tz = _tz()
    now = utcnow()
    now_local = now.astimezone(tz)
    if not _within_active_hours(now_local):
        return None

    day = now_local.date()
    windows = _windows_for_day(day, tz)
    max_deferrals = int(settings.get("scheduler.max_deferrals", 2))

    with session_scope() as session:
        state = _get_state(session)
        due_index = _earliest_due_window(day, windows, now, state.last_window_key or "")
        if due_index is None:
            return None

        key = _window_key(day, due_index)
        has_later_window = due_index < len(windows) - 1

        head = _queue_head_for_window(session, key)
        if head is None:
            state.last_window_key = key
            state.last_action = f"empty:{key}"
            state.updated_at = utcnow()
            return f"empty:{key}"

        if not _spacing_ok(state, now):
            state.last_window_key = key
            state.last_action = f"spacing_block:{key}"
            state.updated_at = utcnow()
            return f"spacing_block:{key}"

        hot, delta = is_last_post_hot(session)
        state.last_hot_check_at = now
        state.last_hot_result = hot
        state.last_hot_likes_delta = delta

        if hot and head.defer_count < max_deferrals:
            if has_later_window:
                head.defer_count = int(head.defer_count or 0) + 1
                head.last_deferred_at = now
                state.last_window_key = key
                state.last_action = f"defer:{key}:post={head.id}:n={head.defer_count}"
                state.updated_at = utcnow()
                log.info(
                    "Deferred post %s at window %s (hot delta=%s, defer=%s)",
                    head.id, key, delta, head.defer_count,
                )
                return state.last_action
            # Last window of the day: skip rather than post into overnight.
            state.last_window_key = key
            state.last_action = f"skip_day:{key}:post={head.id}"
            state.updated_at = utcnow()
            log.info(
                "Skipped post %s at last window %s (still hot, delta=%s); resumes next day",
                head.id, key, delta,
            )
            return state.last_action

        # Publish (not hot, or max deferrals reached).
        post_id = head.id
        force = bool(hot and head.defer_count >= max_deferrals)
        state.last_window_key = key
        state.updated_at = utcnow()
        session.flush()

    action = f"{'force_' if force else ''}publish:{key}:post={post_id}"
    if _claim_and_publish(post_id, action):
        with session_scope() as session:
            post = session.get(ThreadsPost, post_id)
            if post is not None:
                post.defer_count = 0
                post.is_breaking = False
        log.info("Published queue post %s at window %s", post_id, key)
        return action

    with session_scope() as session:
        state = _get_state(session)
        state.last_action = f"publish_failed:{key}:post={post_id}"
        state.updated_at = utcnow()
    return f"publish_failed:{key}:post={post_id}"


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

    queue_rows = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == STATUS_QUEUED)
    ).scalars().all()
    hot, delta = is_last_post_hot(session)

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
        "last_hot_check_at": state.last_hot_check_at,
        "last_hot_result": state.last_hot_result,
        "last_hot_likes_delta": state.last_hot_likes_delta,
        "current_hot": hot,
        "current_likes_delta": delta,
        "hot_threshold": int(settings.get("scheduler.hot.threshold", 100)),
        "hot_window_minutes": int(settings.get("scheduler.hot.window_minutes", 60)),
        "spacing_floor_minutes": int(settings.get("scheduler.spacing_floor_minutes", 90)),
        "max_deferrals": int(settings.get("scheduler.max_deferrals", 2)),
        "queue_count": len(queue_rows),
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


def build_window_plan(
    session,
    start_local: dt.datetime,
    end_local: dt.datetime,
    *,
    horizon_days: int | None = None,
) -> list[dict]:
    """Build a linear plan of posting windows with queue assignments + open placeholders.

    Each entry is a slot dict for the calendar/queue UI:
      kind: open | queued | published | breaking
      window_key, sort (operator-local), time, day, caption, post_id, …

    Upcoming windows always appear (empty = ``open``). Published posts in range
    are attached when their publish time falls near a window; otherwise they are
    listed as standalone published entries. Breaking queued posts prepend as ASAP.
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
        .where(ThreadsPost.status == STATUS_QUEUED)
        .order_by(ThreadsPost.created_at.asc())
    ).scalars().all()
    breaking = [p for p in queued if p.is_breaking]
    regular = [p for p in queued if not p.is_breaking]

    # Touch relationships while session is open.
    for p in queued:
        if p.candidate is not None:
            _ = p.candidate.id
            if p.candidate.channel is not None:
                _ = p.candidate.channel.call_sign

    start_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)
    published = session.execute(
        select(ThreadsPost).where(
            ThreadsPost.status == "published",
            ThreadsPost.published_at.is_not(None),
            ThreadsPost.published_at >= start_utc,
            ThreadsPost.published_at < end_utc,
        ).order_by(ThreadsPost.published_at.asc())
    ).scalars().all()
    for p in published:
        if p.candidate is not None:
            _ = p.candidate.id

    plan: list[dict] = []

    for p in breaking:
        plan.append({
            "kind": "breaking",
            "window_key": "",
            "window_index": None,
            "sort": now.astimezone(),  # top of the live queue
            "time": "ASAP",
            "day": now.astimezone().day,
            "date_label": "Breaking",
            "caption": (p.caption or "").strip(),
            "status": "queued",
            "post_id": p.id,
            "video_id": p.candidate.id if p.candidate else None,
            "channel": (p.candidate.channel.call_sign
                        if p.candidate and p.candidate.channel else ""),
            "permalink": p.permalink,
            "projected": True,
            "is_breaking": True,
            "defer_count": int(p.defer_count or 0),
            "empty": False,
            "pinned": False,
        })

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

    keys = [k for k, _, _, _ in visible]
    _clear_stale_pins(regular, set(k for k, _, _ in upcoming))
    assignment = assign_posts_to_windows(regular, keys)

    for (key, _win_utc, idx, local), post in zip(visible, assignment):
        if post is None:
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
                "permalink": "",
                "projected": True,
                "is_breaking": False,
                "defer_count": 0,
                "empty": True,
                "pinned": False,
            })
        else:
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
                "permalink": post.permalink,
                "projected": True,
                "is_breaking": False,
                "defer_count": int(post.defer_count or 0),
                "empty": False,
                "pinned": bool((post.pinned_window_key or "").strip()),
            })

    # Published posts in range (for calendar history).
    half = dt.timedelta(minutes=45)
    for p in published:
        when = p.published_at
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        local = when.astimezone()
        # Prefer matching a configured window label when close.
        day = local.astimezone(tz).date()
        matched_key = ""
        matched_idx = None
        for i, win in enumerate(_windows_for_day(day, tz)):
            if abs((win - when).total_seconds()) <= half.total_seconds():
                matched_key = _window_key(day, i)
                matched_idx = i
                # Display at the window time for alignment with placeholders.
                local = win.astimezone()
                break
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
            "permalink": p.permalink,
            "projected": False,
            "is_breaking": False,
            "defer_count": 0,
            "empty": False,
            "pinned": False,
        })

    # Breaking stays at the top of the linear queue; everything else by time.
    breaking_rows = [e for e in plan if e["kind"] == "breaking"]
    rest = [e for e in plan if e["kind"] != "breaking"]
    rest.sort(key=lambda e: e["sort"])
    return breaking_rows + rest


def projected_window_slots(
    session,
    start_local: dt.datetime,
    end_local: dt.datetime,
) -> list[dict]:
    """Backward-compatible alias: non-empty projected/queued/open slots for a range."""
    return [
        e for e in build_window_plan(session, start_local, end_local)
        if e["kind"] in ("queued", "open", "breaking")
    ]


def run_tick() -> None:
    """One scheduler loop iteration: metrics → breaking → window."""
    try:
        n = run_metrics_poll()
        if n:
            log.info("Metrics poll took %d snapshot(s)", n)
    except Exception:
        log.exception("Metrics poll failed")

    try:
        n = run_breaking_posts()
        if n:
            log.info("Published %d breaking post(s)", n)
    except Exception:
        log.exception("Breaking publish failed")

    try:
        action = run_window_tick()
        if action:
            log.info("Window tick: %s", action)
    except Exception:
        log.exception("Window tick failed")


def start_scheduler_thread(interval_seconds: int = 60) -> None:
    """Start the background adaptive-scheduler loop. Idempotent."""
    global _thread
    if _thread and _thread.is_alive():
        return

    def _loop() -> None:
        log.info("Adaptive scheduler started (checking every %ss)", interval_seconds)
        while True:
            try:
                run_tick()
            except Exception:
                log.exception("Scheduler tick failed; will retry")
            time.sleep(interval_seconds)

    _thread = threading.Thread(target=_loop, daemon=True, name="threads-scheduler")
    _thread.start()
