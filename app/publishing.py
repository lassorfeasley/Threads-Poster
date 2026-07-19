"""Part 2 publishing flow: operator-provided trimmed clip + caption -> Threads.

Every post here is created only on explicit operator action (post now or
schedule). ``schedule_clip`` records a post to publish later; the scheduler
module (app/scheduler.py) calls ``publish_post`` once its time arrives.
"""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
from pathlib import Path

from .config import load_settings
from .llm import caption_attributes
from .models import Candidate, ThreadsPost, utcnow
from .storage_supabase import upload_trimmed_clip
from .threads_api import publish_video

log = logging.getLogger("publishing")


def _clip_duration_seconds(path: Path) -> int | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return int(float(out.stdout.strip()))
    except Exception:
        return None


def _object_key(clip: Path) -> str:
    now = utcnow()
    return f"{now.strftime('%Y/%m')}/{clip.stem}_{now.strftime('%Y%m%dT%H%M%S')}.mp4"


def record_post(session, candidate: Candidate | None, clip_path: str, caption: str,
                *, status: str, scheduled_at: dt.datetime | None = None) -> ThreadsPost:
    """Create a ThreadsPost row without contacting Threads. Used for both the
    immediate-post and schedule paths; publishing happens in ``publish_post``."""
    clip = Path(clip_path).expanduser()
    if not clip.exists():
        raise FileNotFoundError(f"Clip not found: {clip}")
    post = ThreadsPost(
        candidate_pk=candidate.id if candidate else None,
        caption=caption,
        clip_local_path=str(clip),
        clip_object_path=_object_key(clip),
        status=status,
        scheduled_at=scheduled_at,
    )
    session.add(post)
    session.flush()
    return post


def _apply_post_attributes(post: ThreadsPost) -> None:
    """Fill analytics attributes at publish time (day/hour reflect actual post)."""
    settings = load_settings()
    caption = post.caption
    local_time = dt.datetime.now()
    post.caption_length = len(caption)
    post.caption_hashtag_count = caption.count("#")
    post.post_day_of_week = local_time.strftime("%a")
    post.post_hour_local = local_time.hour
    if post.clip_local_path:
        post.clip_length_seconds = _clip_duration_seconds(Path(post.clip_local_path))
    try:
        attrs = caption_attributes(settings.get("matching.model", "claude-haiku-4-5"), caption)
        post.caption_tone = attrs["tone"]
        post.caption_has_question = attrs["has_question"]
        post.caption_has_cta = attrs["has_cta"]
        post.caption_hashtag_count = attrs["hashtag_count"]
    except Exception as exc:
        log.warning("Caption attribute tagging failed: %s", exc)


def publish_post(session, post: ThreadsPost) -> ThreadsPost:
    """Upload the post's clip to Supabase and publish it to Threads. Works for a
    fresh draft or a due scheduled post. Sets status=failed + error on failure."""
    clip = Path(post.clip_local_path).expanduser()
    if not clip.exists():
        post.status = "failed"
        post.error = f"Clip missing: {clip}"
        session.flush()
        raise FileNotFoundError(f"Clip not found: {clip}")
    if not post.clip_object_path:
        post.clip_object_path = _object_key(clip)

    try:
        signed_url = upload_trimmed_clip(clip, post.clip_object_path)
        result = publish_video(signed_url, post.caption)
        post.threads_media_id = result["media_id"]
        post.permalink = result["permalink"]
        post.status = "published"
        post.published_at = utcnow()
        post.error = ""
    except Exception as exc:
        post.status = "failed"
        post.error = str(exc)[:1000]
        session.flush()
        raise

    _apply_post_attributes(post)
    session.flush()
    log.info("Published Threads post %s (%s)", post.threads_media_id, post.permalink)
    return post


def publish_clip(session, candidate: Candidate | None, clip_path: str, caption: str) -> ThreadsPost:
    """Immediate publish: record the post, then post it to Threads right away."""
    post = record_post(session, candidate, clip_path, caption, status="draft")
    return publish_post(session, post)


def schedule_clip(session, candidate: Candidate | None, clip_path: str, caption: str,
                  scheduled_at: dt.datetime) -> ThreadsPost:
    """Record a post to be published later by the scheduler. Nothing is sent now."""
    return record_post(session, candidate, clip_path, caption,
                        status="scheduled", scheduled_at=scheduled_at)
