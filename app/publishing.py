"""Part 2 publishing flow: operator-provided trimmed clip + caption -> Threads.

Every post here is created only on explicit operator action (post now or
add to queue). ``queue_clip`` records a post for the adaptive window scheduler;
``app/scheduler.py`` calls ``publish_post`` when a window fires.
"""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
import threading
from pathlib import Path

from .config import load_first_reply, load_settings
from .llm import caption_attributes
from .models import Candidate, Cut, ThreadsPost, utcnow
from .storage_supabase import signed_clip_url, upload_trimmed_clip
from .threads_api import publish_text_reply, publish_video

log = logging.getLogger("publishing")

# Post IDs currently being published *in this process* (manual publish thread or
# scheduler tick). Used so recovery can tell a genuinely in-flight publish apart
# from one orphaned in the ``publishing`` status by a restart/crash. Single
# process only, which is exactly where both publish paths run.
_ACTIVE_PUBLISHES: set[int] = set()
_ACTIVE_LOCK = threading.Lock()


def mark_publishing(post_id: int) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PUBLISHES.add(post_id)


def clear_publishing(post_id: int) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PUBLISHES.discard(post_id)


def is_publish_active(post_id: int) -> bool:
    with _ACTIVE_LOCK:
        return post_id in _ACTIVE_PUBLISHES


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
                *, status: str, cut: Cut | None = None) -> ThreadsPost:
    """Create a ThreadsPost row without contacting Threads. Used for immediate
    post, draft, and queue paths; publishing happens in ``publish_post``."""
    clip = Path(clip_path).expanduser()
    if not clip.exists():
        raise FileNotFoundError(f"Clip not found: {clip}")
    if candidate is None and cut is not None:
        candidate = cut.candidate
    # Freeze the LLM draft as it stood, so the diff against the operator's final
    # caption survives as a voice signal (see app/voice.py). Prefer the cut's
    # own draft, falling back to the video-level seed.
    draft = ""
    if cut is not None:
        draft = cut.draft_caption or ""
    if not draft and candidate is not None:
        draft = candidate.draft_caption or ""
    post = ThreadsPost(
        candidate_pk=candidate.id if candidate else None,
        cut_pk=cut.id if cut else None,
        caption=caption,
        suggested_caption=draft,
        clip_local_path=str(clip),
        clip_object_path=_object_key(clip),
        status=status,
    )
    session.add(post)
    session.flush()
    # Upload now, while the file is guaranteed to be on this machine, so a
    # headless scheduler (GitHub Actions / cron) can publish later without this
    # disk. Best-effort: publish_post re-uploads from local when it can.
    try:
        upload_trimmed_clip(clip, post.clip_object_path)
    except Exception as exc:
        log.warning("Queue-time clip upload failed (will retry at publish): %s", exc)
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


def _persist_publish_result(post_id: int, media_id: str, permalink: str,
                            when: dt.datetime) -> None:
    """Record a successful Threads publish in its own session/transaction.

    ``publish_video`` can take minutes, and by the time it returns the
    caller's session may be sitting on a connection that died in the meantime
    (laptop sleep, Supabase idle close). The clip is already live at that
    point, so losing this write strands a published post as ``failed`` —
    write it through a fresh connection, with one retry, instead of trusting
    the caller's."""
    from sqlalchemy import update
    from sqlalchemy.exc import OperationalError

    from .db import session_scope

    last_exc: Exception | None = None
    for _ in range(2):
        try:
            with session_scope() as s:
                s.execute(
                    update(ThreadsPost)
                    .where(ThreadsPost.id == post_id)
                    .values(threads_media_id=media_id, permalink=permalink,
                            status="published", published_at=when, error="")
                )
            return
        except OperationalError as exc:
            last_exc = exc
    raise last_exc


def publish_post(session, post: ThreadsPost) -> ThreadsPost:
    """Publish the post's clip to Threads. Uses the local file when it exists
    (re-uploading to Supabase for the freshest copy); otherwise falls back to
    the copy uploaded at queue time, so a headless runner can publish without
    this machine's disk. Sets status=failed + error on failure."""
    if post.threads_media_id:
        # Already live on Threads — a retry after the *record* of a publish
        # failed to write. Publishing again would duplicate the post.
        post.status = "published"
        post.published_at = post.published_at or utcnow()
        post.error = ""
        session.flush()
        return post

    clip = Path(post.clip_local_path).expanduser()
    have_local = clip.exists()
    if not have_local and not post.clip_object_path:
        post.status = "failed"
        post.error = f"Clip missing: {clip} (and no uploaded copy)"
        session.flush()
        raise FileNotFoundError(f"Clip not found: {clip}")
    if not post.clip_object_path:
        post.clip_object_path = _object_key(clip)

    try:
        if have_local:
            signed_url = upload_trimmed_clip(clip, post.clip_object_path)
        else:
            signed_url = signed_clip_url(post.clip_object_path)
        result = publish_video(signed_url, post.caption)
        published_at = utcnow()
        # The clip is live on Threads from here on: persist that fact before
        # anything else (annotation, the caller's own session) gets a chance
        # to fail and roll it back.
        _persist_publish_result(post.id, result["media_id"], result["permalink"],
                                published_at)
        post.threads_media_id = result["media_id"]
        post.permalink = result["permalink"]
        post.status = "published"
        post.published_at = published_at
        post.error = ""
    except Exception as exc:
        post.status = "failed"
        post.error = str(exc)[:1000]
        session.flush()
        raise

    _apply_post_attributes(post)
    _annotate_footage(session, post)
    session.flush()
    log.info("Published Threads post %s (%s)", post.threads_media_id, post.permalink)
    maybe_post_first_reply(session, post)
    return post


def _annotate_footage(session, post: ThreadsPost) -> None:
    """Ground-truth footage trait tagging from the posted clip. Best-effort —
    a tagging hiccup must never undo a successful publish (the backfill
    command / dashboard button can retry later)."""
    try:
        from .db import active_traits
        from .vision import annotate_post_footage

        annotate_post_footage(post, load_settings(), active_traits(session))
    except Exception as exc:
        log.warning("Footage annotation failed for post %s: %s", post.id, exc)


def maybe_post_first_reply(session, post: ThreadsPost, *, force: bool = False) -> bool:
    """Post the configured first reply under a published post.

    Returns True if a reply was posted. Skips when disabled / empty / already
    posted (unless ``force``). Never raises — stores ``first_reply_error`` instead
    so a reply hiccup cannot undo a successful publish.
    """
    if post.status != "published" or not post.threads_media_id:
        return False
    if post.first_reply_id and not force:
        return False

    cfg = load_first_reply()
    text = (cfg.get("text") or "").strip()
    if not force and (not cfg.get("enabled") or not text):
        return False
    if force and not text:
        post.first_reply_error = "First reply text is empty — set it under Replies settings"
        session.flush()
        return False

    try:
        result = publish_text_reply(text, post.threads_media_id)
        post.first_reply_id = result["media_id"]
        post.first_reply_text = text
        post.first_reply_error = ""
        post.first_reply_at = utcnow()
        session.flush()
        log.info("Posted first reply %s under %s", post.first_reply_id, post.threads_media_id)
        return True
    except Exception as exc:
        post.first_reply_error = str(exc)[:1000]
        session.flush()
        log.warning("First reply failed for post %s: %s", post.id, exc)
        return False


def publish_clip(session, candidate: Candidate | None, clip_path: str, caption: str,
                 *, cut: Cut | None = None) -> ThreadsPost:
    """Immediate publish: record the post, then post it to Threads right away."""
    post = record_post(session, candidate, clip_path, caption, status="draft", cut=cut)
    return publish_post(session, post)


def queue_clip(session, candidate: Candidate | None, clip_path: str, caption: str,
               *, cut: Cut | None = None) -> ThreadsPost:
    """Add a post to the adaptive FIFO queue. The window scheduler publishes it."""
    return record_post(session, candidate, clip_path, caption, status="queued", cut=cut)
