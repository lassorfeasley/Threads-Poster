"""Window scheduler for the Threads post queue.

Publishes the FIFO ``queued`` list at fixed daily windows (US Eastern by
default): one post per window, gated only by active hours and the spacing
floor. Pinned posts claim their window; the rest fill remaining slots in order.
A window is never given up because of how an earlier post is performing.

Also drives the frequent metrics poller that feeds analytics.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from . import threads_api
from .analytics import poll_recent_metrics
from .categories import category_by_slug
from .config import load_settings
from .db import session_scope
from .models import Candidate, SchedulerState, ThreadsPost, utcnow
from .publishing import (
    clear_publishing,
    is_publish_active,
    mark_publishing,
    publish_post,
)

log = logging.getLogger("scheduler")

STATUS_QUEUED = "queued"
STATUS_PUBLISHING = "publishing"

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
    return recovered


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
    assignment = assign_posts_to_windows(posts, keys)
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
        # Atomic claim: only one scheduler (local dashboard vs headless runner)
        # can win when both tick at the same time.
        claimed = session.execute(
            update(ThreadsPost)
            .where(ThreadsPost.id == post_id, ThreadsPost.status == STATUS_QUEUED)
            .values(status=STATUS_PUBLISHING, error="", pinned_window_key="")
        ).rowcount
        if claimed != 1:
            return False

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

        post_id = head.id
        state.last_window_key = key
        state.updated_at = utcnow()
        session.flush()

    action = f"publish:{key}:post={post_id}"
    if _claim_and_publish(post_id, action):
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


def _post_display_title(p) -> str:
    """Best label for a post: its cut's short calendar name (sized to fit the
    calendar's window slots), else the full clip title, else the source video
    title, else — for cut-less posts like imported Threads history — the
    post's own short calendar name condensed from its caption. When the post's
    source video has a programming category, its emoji (📰/🌿/📼) leads the
    label so the calendar/queue shows the channel mix at a glance."""
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
    post_by_key = dict(zip(full_keys, assign_posts_to_windows(regular, full_keys)))

    for key, _win_utc, idx, local in visible:
        post = post_by_key.get(key)
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
                "thumbnail": "",
                "title": "",
                "permalink": "",
                "projected": True,
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
            "thumbnail": (post.candidate.thumbnail_url if post.candidate else ""),
            "title": _post_display_title(post),
                "permalink": post.permalink,
                "projected": True,
                "empty": False,
                "pinned": bool((post.pinned_window_key or "").strip()),
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

    plan.sort(key=lambda e: e["sort"])
    return plan


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
    """One scheduler loop iteration: recover → metrics → window."""
    try:
        with session_scope() as session:
            n = recover_stuck_publishing(session, only_inactive=True)
        if n:
            log.info("Recovered %d post(s) stuck in 'publishing'", n)
    except Exception:
        log.exception("Stuck-publish recovery failed")

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
