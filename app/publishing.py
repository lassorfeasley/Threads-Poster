"""Part 2 publishing flow: operator-provided trimmed clip + caption -> Threads.

Called only from the dashboard on explicit operator confirmation. Nothing here
runs on a schedule.
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


def publish_clip(session, candidate: Candidate | None, clip_path: str, caption: str) -> ThreadsPost:
    """Upload the operator's trimmed clip to Supabase, post to Threads, record everything."""
    settings = load_settings()
    clip = Path(clip_path).expanduser()
    if not clip.exists():
        raise FileNotFoundError(f"Clip not found: {clip}")

    now = utcnow()
    object_key = f"{now.strftime('%Y/%m')}/{clip.stem}_{now.strftime('%Y%m%dT%H%M%S')}.mp4"

    post = ThreadsPost(
        candidate_pk=candidate.id if candidate else None,
        caption=caption,
        clip_local_path=str(clip),
        clip_object_path=object_key,
        status="draft",
    )
    session.add(post)
    session.flush()

    try:
        signed_url = upload_trimmed_clip(clip, object_key)
        result = publish_video(signed_url, caption)
        post.threads_media_id = result["media_id"]
        post.permalink = result["permalink"]
        post.status = "published"
        post.published_at = utcnow()
    except Exception as exc:
        post.status = "failed"
        post.error = str(exc)[:1000]
        session.flush()
        raise

    # Analytics attributes.
    local_time = dt.datetime.now()
    post.caption_length = len(caption)
    post.caption_hashtag_count = caption.count("#")
    post.post_day_of_week = local_time.strftime("%a")
    post.post_hour_local = local_time.hour
    post.clip_length_seconds = _clip_duration_seconds(clip)
    try:
        attrs = caption_attributes(settings.get("matching.model", "claude-haiku-4-5"), caption)
        post.caption_tone = attrs["tone"]
        post.caption_has_question = attrs["has_question"]
        post.caption_has_cta = attrs["has_cta"]
        post.caption_hashtag_count = attrs["hashtag_count"]
    except Exception as exc:
        log.warning("Caption attribute tagging failed: %s", exc)

    session.flush()
    log.info("Published Threads post %s (%s)", post.threads_media_id, post.permalink)
    return post
