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

from sqlalchemy import select

from .config import load_first_reply, load_settings, scheduler_timezone
from .instagram_api import publish_reel
from .llm import caption_attributes, suggest_attribution
from .models import Candidate, Cut, InstagramPost, ThreadsPost, utcnow
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
                *, status: str, cut: Cut | None = None,
                attribution: str = "") -> ThreadsPost:
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
        # Measure now, while the file is certainly on this machine. Doing it at
        # publish time meant any headless publisher (CI, always-on container)
        # ffprobed a path that only exists on the operator's disk and silently
        # recorded no duration at all.
        clip_length_seconds=_clip_duration_seconds(clip),
    )
    session.add(post)
    session.flush()
    # Attribution first-comment: an operator-reviewed text (from the cut page's
    # first-reply module) wins outright. Otherwise draft one now (not at publish
    # time), so it can be previewed/edited on the post page before it goes out —
    # and so a headless scheduler can publish without an LLM call. Best-effort:
    # a drafting hiccup must never block queueing (Suggest on the post page retries).
    if attribution.strip():
        post.attribution_text = attribution.strip()
    elif candidate is not None and load_first_reply().get("attribution_enabled"):
        try:
            post.attribution_text = generate_attribution(candidate)
        except Exception as exc:
            log.warning("Attribution draft failed for post %s: %s", post.id, exc)
    # Upload now, while the file is guaranteed to be on this machine, so a
    # headless scheduler (GitHub Actions / cron) can publish later without this
    # disk. Best-effort: publish_post re-uploads from local when it can.
    try:
        upload_trimmed_clip(clip, post.clip_object_path)
    except Exception as exc:
        log.warning("Queue-time clip upload failed (will retry at publish): %s", exc)
    return post


def generate_attribution(candidate: Candidate) -> str:
    """LLM-drafted courtesy line crediting the source station/publisher (and the
    program/journalists when the video's own metadata establishes them). DRAFT
    ONLY — the operator reviews it on the post page before it publishes."""
    channel = candidate.channel
    settings = load_settings()
    return suggest_attribution(
        settings.get("engagement.draft_model", "claude-sonnet-5"),
        channel={
            "call_sign": channel.call_sign if channel else "",
            "network": channel.network if channel else "",
            "market": channel.market if channel else "",
            "region": channel.region if channel else "",
            "country": channel.country if channel else "",
            "channel_title": channel.channel_title if channel else "",
        },
        video_title=candidate.title or "",
        description=candidate.description or "",
        transcript_excerpt=candidate.transcript_text or "",
    )


def post_time_attributes(when: dt.datetime) -> tuple[str, int]:
    """``(weekday, hour)`` for a publish time, in the scheduler's posting zone.

    Pinned to ``scheduler.timezone`` rather than read off the publishing
    machine's clock, because these two fields drive the analytics that tune the
    posting windows: an hour has to mean the same thing whether the post went
    out from the operator's laptop, a CI runner, or an always-on container.
    Reading the local clock silently broke that — posts published from GitHub
    Actions recorded the UTC hour, filing a 10:00 ET post under hour 17.

    Bonus: in the scheduler zone an hour now lines up with the window that
    produced it, so "hour 10" is literally the 10:00 window.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    local = when.astimezone(scheduler_timezone())
    return local.strftime("%a"), local.hour


def _apply_post_attributes(post: ThreadsPost) -> None:
    """Fill analytics attributes at publish time (day/hour reflect actual post)."""
    settings = load_settings()
    caption = post.caption
    post.caption_length = len(caption)
    post.caption_hashtag_count = caption.count("#")
    post.post_day_of_week, post.post_hour_local = post_time_attributes(
        post.published_at or utcnow()
    )
    # Normally measured at queue time, while the clip is still on the operator's
    # disk. This covers posts queued before that and any path that skipped it —
    # and does nothing on a headless publisher, where the file isn't present.
    if post.clip_length_seconds is None and post.clip_local_path:
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


# --- Instagram Reels (paired posts) ------------------------------------------

def record_instagram_post(session, cut: Cut | None, threads_post: ThreadsPost,
                          clip_path: str, caption: str) -> InstagramPost:
    """Create (or refresh) the queued reel paired with ``threads_post``.

    The reel posts the cut's VERTICAL composite while the Threads post keeps
    its 16:9 clip, so the reel owns its own Supabase object. Uploaded now,
    while the file is certainly on this machine, so a headless runner can
    publish later (same rationale as ``record_post``). Reuses an existing
    not-yet-published row for the cut instead of stacking duplicates.
    """
    clip = Path(clip_path).expanduser()
    if not clip.exists():
        raise FileNotFoundError(f"Vertical clip not found: {clip}")
    existing = session.execute(
        select(InstagramPost).where(
            InstagramPost.cut_pk == (cut.id if cut else None),
            InstagramPost.status.in_(["queued", "draft", "failed"]),
        ).order_by(InstagramPost.created_at.desc())
    ).scalars().all() if cut else []
    if existing:
        ig = existing[0]
        for extra in existing[1:]:
            session.delete(extra)
    else:
        ig = InstagramPost(cut_pk=cut.id if cut else None)
        session.add(ig)
    ig.threads_post_pk = threads_post.id
    ig.caption = caption
    if ig.clip_local_path != str(clip) or not ig.clip_object_path:
        ig.clip_local_path = str(clip)
        ig.clip_object_path = _object_key(clip)
        try:
            upload_trimmed_clip(clip, ig.clip_object_path)
        except Exception as exc:
            log.warning("Queue-time reel upload failed (will retry at publish): %s", exc)
    ig.status = "queued"
    ig.error = ""
    session.flush()
    return ig


def _write_ig_row(ig_id: int, **values) -> None:
    """Update an InstagramPost through a fresh connection/transaction.

    Same rationale as ``_persist_publish_result``: the container poll can take
    minutes and outlive the caller's connection. It also keeps the caller's
    session free of pending writes during the publish — on SQLite an
    uncommitted write in the caller would hold the database's write lock and
    deadlock these fresh-connection updates against it."""
    from sqlalchemy import update
    from sqlalchemy.exc import OperationalError

    from .db import session_scope

    last_exc: Exception | None = None
    for _ in range(2):
        try:
            with session_scope() as s:
                s.execute(
                    update(InstagramPost).where(InstagramPost.id == ig_id).values(**values)
                )
            return
        except OperationalError as exc:
            last_exc = exc
    raise last_exc


def publish_instagram_post(session, ig: InstagramPost) -> InstagramPost:
    """Publish the paired reel to Instagram. Prefers the local composite
    (re-uploading for the freshest copy); otherwise signs the copy uploaded at
    queue time so a headless runner works. Sets status=failed + error on
    failure and re-raises."""
    if ig.ig_media_id:
        # Already live — a retry after only the record failed to write.
        ig.status = "published"
        ig.published_at = ig.published_at or utcnow()
        ig.error = ""
        session.flush()
        return ig

    clip = Path(ig.clip_local_path or "").expanduser()
    have_local = bool(ig.clip_local_path) and clip.exists()
    if not have_local and not ig.clip_object_path:
        ig.status = "failed"
        ig.error = f"Clip missing: {clip} (and no uploaded copy)"
        session.flush()
        raise FileNotFoundError(f"Clip not found: {clip}")

    if not ig.clip_object_path:
        ig.clip_object_path = _object_key(clip)
    # Claim + all publish-state writes go through fresh connections (see
    # _write_ig_row) so the caller's session carries no pending writes while
    # Meta processes the video.
    _write_ig_row(ig.id, status="publishing", error="",
                  clip_object_path=ig.clip_object_path)
    try:
        if have_local:
            signed_url = upload_trimmed_clip(clip, ig.clip_object_path)
        else:
            signed_url = signed_clip_url(ig.clip_object_path)
        result = publish_reel(signed_url, ig.caption)
        published_at = utcnow()
        # Live on Instagram from here: persist before anything else can fail.
        _write_ig_row(ig.id, ig_media_id=result["media_id"],
                      permalink=result["permalink"], status="published",
                      published_at=published_at, error="")
        ig.ig_media_id = result["media_id"]
        ig.permalink = result["permalink"]
        ig.status = "published"
        ig.published_at = published_at
        ig.error = ""
    except Exception as exc:
        try:
            _write_ig_row(ig.id, status="failed", error=str(exc)[:1000])
        except Exception:
            log.warning("Could not record reel failure for %s", ig.id)
        ig.status = "failed"
        ig.error = str(exc)[:1000]
        raise
    log.info("Published Instagram reel %s (%s)", ig.ig_media_id, ig.permalink)
    return ig


def publish_paired_reel(post_id: int) -> InstagramPost | None:
    """Fire the reel queued alongside a just-published Threads post.

    Called by every publish path (scheduler window, Post now, retry) AFTER the
    Threads publish transaction commits — it must not run inside that
    transaction, because reel state is persisted through fresh connections
    (``_write_ig_row``) which on SQLite would deadlock against the caller's
    uncommitted writes. Best-effort by design: the Threads result is already
    on disk, and a reel failure only marks the ``InstagramPost`` row failed —
    it never unwinds the Threads publish. Returns a detached snapshot of the
    reel row (or None when no reel is paired)."""
    from .db import session_scope

    with session_scope() as session:
        ig = session.execute(
            select(InstagramPost).where(
                InstagramPost.threads_post_pk == post_id,
                InstagramPost.status.in_(["queued", "failed", "publishing"]),
            ).order_by(InstagramPost.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        if ig is None:
            return None
        try:
            publish_instagram_post(session, ig)
        except Exception as exc:
            log.warning("Paired Instagram reel failed for post %s: %s", post_id, exc)
        session.expunge(ig)
    return ig


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


def _draft_attribution_now(session, post: ThreadsPost) -> tuple[str, str]:
    """``(attribution, error)`` for a post that reached publish without one.

    Attribution is normally drafted at queue time so the operator can review it,
    but a post can still arrive here empty: it was queued before attribution
    comments existed, or its queue-time draft raised (best-effort, so queueing
    went ahead anyway). Drafting now means such a post still credits its source
    instead of publishing bare. A cleared attribution never reaches this — see
    ``ThreadsPost.attribution_skipped``.
    """
    if post.candidate is None:
        return "", ("No first comment: there's no source video on record to credit, "
                    "and no static reply text is enabled under Replies settings.")
    try:
        text = generate_attribution(post.candidate)
    except Exception as exc:
        log.warning("Publish-time attribution draft failed for post %s: %s", post.id, exc)
        return "", f"Could not draft an attribution comment: {exc}"[:1000]
    if not text:
        return "", "The model returned an empty attribution comment."
    post.attribution_text = text
    session.flush()
    return text, ""


def _no_first_comment_reason(post: ThreadsPost, cfg: dict) -> str:
    """Plain-language reason a published post carries no first comment, stored on
    the post so it shows up on the post page instead of looking untouched."""
    if post.attribution_skipped:
        return ("No first comment: the attribution was cleared for this post, and "
                "no static reply text is enabled under Replies settings.")
    if not cfg.get("attribution_enabled") and not cfg.get("enabled"):
        return ("No first comment: attribution comments and the static reply are "
                "both switched off under Replies settings.")
    return ("No first comment: this post has no attribution text, and no static "
            "reply text is enabled under Replies settings.")


def maybe_post_first_reply(session, post: ThreadsPost, *, force: bool = False) -> bool:
    """Post the first reply under a published post: the post's own attribution
    comment when it has one, else the static configured text. A post that arrives
    with no attribution gets one drafted here rather than publishing bare.

    Returns True if a reply was posted. Skips when disabled / already posted
    (unless ``force``), recording why on the post either way. Never raises —
    stores ``first_reply_error`` so a reply hiccup cannot undo a publish.
    """
    if post.status != "published" or not post.threads_media_id:
        return False
    if post.first_reply_id and not force:
        return False

    cfg = load_first_reply()
    attribution = (post.attribution_text or "").strip()
    draft_error = ""
    if not attribution and cfg.get("attribution_enabled") and not post.attribution_skipped:
        attribution, draft_error = _draft_attribution_now(session, post)
    static_text = (cfg.get("text") or "").strip()
    text = ""
    if attribution and (cfg.get("attribution_enabled") or force):
        text = attribution
    elif static_text and (cfg.get("enabled") or force):
        text = static_text
    if not text:
        # Record why, always — not just on ``force``. Publishing with no first
        # comment used to write nothing anywhere, so four posts in a row went out
        # bare and the post page showed a clean slate for each of them.
        post.first_reply_error = draft_error or _no_first_comment_reason(post, cfg)
        session.flush()
        log.warning("Post %s published without a first comment: %s",
                    post.id, post.first_reply_error)
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
                 *, cut: Cut | None = None, attribution: str = "") -> ThreadsPost:
    """Immediate publish: record the post, then post it to Threads right away."""
    post = record_post(session, candidate, clip_path, caption, status="draft", cut=cut,
                       attribution=attribution)
    return publish_post(session, post)


def queue_clip(session, candidate: Candidate | None, clip_path: str, caption: str,
               *, cut: Cut | None = None, attribution: str = "") -> ThreadsPost:
    """Add a post to the adaptive FIFO queue. The window scheduler publishes it."""
    return record_post(session, candidate, clip_path, caption, status="queued", cut=cut,
                       attribution=attribution)
