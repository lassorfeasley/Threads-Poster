"""Pull comments on the operator's own posts and post manual replies.

Hard scope limits enforced in code:
  * Comments are only ever read from posts this tool published (ThreadsPost
    rows), i.e. the operator's own posts. There is no search or outreach
    surface at all; `engagement.allow_other_users_posts` is checked and, since
    no other-post code path exists, enabling it still does nothing beyond
    logging a warning. It exists to document the posture.
  * Replies are composed on the post page; posting happens only via
    post_approved_reply(), and pacing caps are enforced there.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import func, select

from .config import load_settings
from .models import ThreadsComment, ThreadsPost, utcnow
from .threads_api import fetch_replies, publish_text_reply

log = logging.getLogger("engagement")


class PacingLimitError(RuntimeError):
    """Raised when posting a reply would exceed the hourly/daily caps."""


def _parse_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00").replace("+0000", "+00:00"))
    except ValueError:
        return None


def sync_comments(session) -> dict:
    """Pull new comments on the operator's own published posts."""
    settings = load_settings()
    if settings.get("engagement.allow_other_users_posts", False):
        log.warning(
            "engagement.allow_other_users_posts is enabled in config, but this build "
            "intentionally has no code path for other users' posts. Ignoring."
        )

    posts = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == "published")
    ).scalars().all()

    new_comments = 0

    for post in posts:
        try:
            replies = fetch_replies(post.threads_media_id)
        except Exception as exc:
            log.warning("Could not fetch replies for post %s: %s", post.threads_media_id, exc)
            continue

        for r in replies:
            comment_id = r.get("id", "")
            if not comment_id:
                continue
            exists = session.execute(
                select(ThreadsComment.id).where(ThreadsComment.comment_id == comment_id)
            ).scalar_one_or_none()
            if exists is not None:
                continue

            session.add(ThreadsComment(
                post_pk=post.id,
                comment_id=comment_id,
                username=r.get("username", ""),
                text=r.get("text", "") or "",
                commented_at=_parse_ts(r.get("timestamp")),
                reply_status="pending",
            ))
            new_comments += 1

    log.info("Comment sync: %d new", new_comments)
    return {"new_comments": new_comments}


def check_pacing(session) -> None:
    """Raise PacingLimitError if another reply would exceed hourly/daily caps."""
    settings = load_settings()
    now = utcnow()
    hour_ago = now - dt.timedelta(hours=1)
    day_ago = now - dt.timedelta(days=1)

    per_hour = session.execute(
        select(func.count(ThreadsComment.id)).where(
            ThreadsComment.reply_status == "posted", ThreadsComment.replied_at >= hour_ago
        )
    ).scalar_one()
    per_day = session.execute(
        select(func.count(ThreadsComment.id)).where(
            ThreadsComment.reply_status == "posted", ThreadsComment.replied_at >= day_ago
        )
    ).scalar_one()

    max_hour = settings.get("engagement.max_replies_per_hour", 4)
    max_day = settings.get("engagement.max_replies_per_day", 12)
    if per_hour >= max_hour:
        raise PacingLimitError(f"Hourly reply cap reached ({per_hour}/{max_hour}). Try again later.")
    if per_day >= max_day:
        raise PacingLimitError(f"Daily reply cap reached ({per_day}/{max_day}). Try again tomorrow.")


def post_approved_reply(session, comment: ThreadsComment, final_text: str,
                        *, gif_id: str | None = None) -> None:
    """Post an operator-composed reply (text and/or GIF). Enforces pacing caps."""
    check_pacing(session)
    text = (final_text or "").strip()
    gif_id = (gif_id or "").strip() or None
    result = publish_text_reply(text, comment.comment_id, gif_id=gif_id)
    comment.reply_status = "posted"
    if text and gif_id:
        comment.reply_text_posted = text
    elif gif_id:
        comment.reply_text_posted = "[GIF]"
    else:
        comment.reply_text_posted = text
    comment.reply_id = result["media_id"]
    comment.replied_at = utcnow()
    session.flush()
    log.info("Posted reply to comment %s", comment.comment_id)
