"""Import the authenticated account's existing Threads posts (made outside this
tool) so the analytics and engagement flows can cover them too.

Imported posts get ``source='threads'`` and no linked candidate/clip. Posts the
app already knows about (published locally) are left untouched.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select

from .config import load_settings
from .llm import caption_attributes
from .models import ThreadsPost
from .threads_api import fetch_user_posts

log = logging.getLogger("history")


def _parse_ts(ts: str | None) -> dt.datetime | None:
    """Threads timestamps look like ``2024-08-01T12:34:56+0000``."""
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("+0000", "+00:00"))
    except ValueError:
        return None


def import_history(session, limit: int = 200, tag_captions: bool = False) -> dict:
    """Upsert the account's own Threads posts into ThreadsPost.

    Returns {"imported": int, "skipped": int, "fetched": int}. ``tag_captions``
    runs the (slower, paid) LLM tone/question/CTA tagging on each new post.
    """
    settings = load_settings()
    fetched = fetch_user_posts(limit=limit)
    imported = skipped = 0

    for p in fetched:
        media_id = str(p.get("id") or "")
        if not media_id:
            continue
        existing = session.execute(
            select(ThreadsPost).where(ThreadsPost.threads_media_id == media_id)
        ).scalar_one_or_none()
        if existing is not None:
            if not existing.permalink and p.get("permalink"):
                existing.permalink = p["permalink"]
            skipped += 1
            continue

        caption = p.get("text") or ""
        published = _parse_ts(p.get("timestamp"))
        post = ThreadsPost(
            candidate_pk=None,
            threads_media_id=media_id,
            permalink=p.get("permalink", ""),
            caption=caption,
            status="published",
            source="threads",
            published_at=published,
            caption_length=len(caption),
            caption_hashtag_count=caption.count("#"),
        )
        if published is not None:
            local = published.astimezone()
            post.post_day_of_week = local.strftime("%a")
            post.post_hour_local = local.hour
        if tag_captions and caption.strip():
            try:
                attrs = caption_attributes(settings.get("matching.model", "claude-haiku-4-5"), caption)
                post.caption_tone = attrs["tone"]
                post.caption_has_question = attrs["has_question"]
                post.caption_has_cta = attrs["has_cta"]
                post.caption_hashtag_count = attrs["hashtag_count"]
            except Exception as exc:
                log.warning("Caption tagging failed for %s: %s", media_id, exc)
        session.add(post)
        imported += 1

    session.flush()
    log.info("Imported %d Threads posts (%d already known, %d fetched)", imported, skipped, len(fetched))
    return {"imported": imported, "skipped": skipped, "fetched": len(fetched)}
