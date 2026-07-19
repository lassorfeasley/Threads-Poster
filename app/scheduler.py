"""Background publisher for scheduled Threads posts.

A scheduled post is a ``ThreadsPost`` with status ``scheduled`` and a future
``scheduled_at``. When its time arrives this module publishes it. It only ever
touches posts the operator explicitly scheduled — nothing is auto-generated.
"""
from __future__ import annotations

import logging
import threading
import time

from sqlalchemy import select

from . import threads_api
from .db import session_scope
from .models import ThreadsPost
from .models import utcnow
from .publishing import publish_post

log = logging.getLogger("scheduler")

STATUS_SCHEDULED = "scheduled"
STATUS_PUBLISHING = "publishing"

_thread: threading.Thread | None = None


def run_due_posts() -> int:
    """Publish every scheduled post whose time has come. Returns count published.

    Due posts are claimed (status -> publishing) in a short committed step, then
    each is published in its own session so a slow upload doesn't hold a
    transaction open or get re-picked on the next tick.
    """
    if not threads_api.is_authenticated():
        return 0

    with session_scope() as session:
        now = utcnow()
        due_ids = session.execute(
            select(ThreadsPost.id).where(
                ThreadsPost.status == STATUS_SCHEDULED,
                ThreadsPost.scheduled_at.is_not(None),
                ThreadsPost.scheduled_at <= now,
            ).order_by(ThreadsPost.scheduled_at)
        ).scalars().all()
        for pid in due_ids:
            post = session.get(ThreadsPost, pid)
            if post:
                post.status = STATUS_PUBLISHING

    published = 0
    for pid in due_ids:
        with session_scope() as session:
            post = session.get(ThreadsPost, pid)
            if post is None:
                continue
            try:
                publish_post(session, post)
                published += 1
            except Exception as exc:  # publish_post already recorded failure
                log.warning("Scheduled post %s failed: %s", pid, exc)
    return published


def start_scheduler_thread(interval_seconds: int = 60) -> None:
    """Start the background loop that publishes due scheduled posts. Idempotent."""
    global _thread
    if _thread and _thread.is_alive():
        return

    def _loop() -> None:
        log.info("Scheduler started (checking every %ss)", interval_seconds)
        while True:
            try:
                n = run_due_posts()
                if n:
                    log.info("Scheduler published %d due post(s)", n)
            except Exception:
                log.exception("Scheduler tick failed; will retry")
            time.sleep(interval_seconds)

    _thread = threading.Thread(target=_loop, daemon=True, name="threads-scheduler")
    _thread.start()
