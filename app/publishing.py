"""Part 2 publishing flow: operator-provided trimmed clip + caption -> Threads.

Every post here is created only on explicit operator action (post now or
add to queue). ``queue_clip`` records a post for the adaptive window scheduler;
``app/scheduler.py`` calls ``publish_post`` when a window fires.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import or_, select, update

import requests

from . import facebook_api, linkedin_api, x_api
from .categories import is_first_party
from .config import WATCH_URL_PLACEHOLDER, load_first_reply, load_settings, scheduler_timezone
from .draft_proposals import KIND_HOOK, attach_to_post as attach_draft_proposal
from .instagram_api import publish_reel
from .llm import caption_attributes
from .models import (AppToken, Candidate, Cut, FacebookPost, InstagramPost, LinkedInPost,
                     ThreadsPost, XPost, utcnow)
from .outro import with_outro
from .storage_supabase import signed_clip_url, upload_trimmed_clip
from .threads_api import ThreadsError, publish_text_reply, publish_video
from .watch_pages import display_url, ensure_watch_page

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


# LinkedInPost IDs being published in this process, for the same recovery
# question. Tracked apart from Threads posts because the mirror also runs from
# paths (Post now, notification retry) that never mark the Threads post.
_ACTIVE_LINKEDIN: set[int] = set()


def is_linkedin_publish_active(li_id: int) -> bool:
    with _ACTIVE_LOCK:
        return li_id in _ACTIVE_LINKEDIN


_ACTIVE_X: set[int] = set()


def is_x_publish_active(x_id: int) -> bool:
    with _ACTIVE_LOCK:
        return x_id in _ACTIVE_X


_ACTIVE_FACEBOOK: set[int] = set()


def is_facebook_publish_active(fb_id: int) -> bool:
    with _ACTIVE_LOCK:
        return fb_id in _ACTIVE_FACEBOOK


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


# Queue-time upload retry schedule (seconds between attempts). Short and
# synchronous — this runs inside a dashboard request — but enough to ride out
# the transient Supabase hiccups that otherwise strand a queued post/reel
# with no storage copy, which a headless runner then can't publish at all
# ("Object not found" at signing time).
_UPLOAD_RETRY_DELAYS = (2, 5)


def _upload_clip_with_retries(clip: Path, object_key: str) -> str:
    """``upload_trimmed_clip`` with a couple of quick retries. Raises the last
    error when every attempt fails."""
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_UPLOAD_RETRY_DELAYS, None)):
        try:
            return upload_trimmed_clip(clip, object_key)
        except Exception as exc:
            last_exc = exc
            log.warning("Clip upload attempt %d/%d failed for %s: %s",
                        attempt + 1, len(_UPLOAD_RETRY_DELAYS) + 1, object_key, exc)
            if delay is not None:
                time.sleep(delay)
    raise last_exc


def _storage_object_missing(exc: Exception) -> bool:
    """Whether an exception is Supabase's 404 for a missing storage object —
    the signature of a clip whose queue-time upload never happened. Retrying
    on a machine without the local file can never fix this."""
    text = str(exc)
    return "Object not found" in text or "not_found" in text


def record_post(session, candidate: Candidate | None, clip_path: str, caption: str,
                *, status: str, cut: Cut | None = None) -> ThreadsPost:
    """Create a ThreadsPost row without contacting Threads. Used for immediate
    post, draft, and queue paths; publishing happens in ``publish_post``."""
    clip = Path(clip_path).expanduser()
    if not clip.exists():
        raise FileNotFoundError(f"Clip not found: {clip}")
    if candidate is None and cut is not None:
        candidate = cut.candidate
    post = ThreadsPost(
        candidate_pk=candidate.id if candidate else None,
        cut_pk=cut.id if cut else None,
        caption=caption,
        clip_local_path=str(clip),
        clip_object_path=_object_key(clip),
        status=status,
        # Measure now, while the file is certainly on this machine. Doing it at
        # publish time meant any headless publisher (CI, always-on container)
        # ffprobed a path that only exists on the operator's disk and silently
        # recorded no duration at all. Measured with the outro, which the
        # upload below appends (rendered once here, cached for the upload).
        clip_length_seconds=_clip_duration_seconds(with_outro(clip)),
    )
    # Tags set on the clip (vision on export, optionally corrected by hand)
    # describe this exact file, so they carry over now and spare the post the
    # annotation pass entirely.
    from .vision import seed_post_tags_from_cut

    seed_post_tags_from_cut(post, cut)
    session.add(post)
    session.flush()
    # Freeze the LLM draft, so the diff against the operator's final caption
    # survives as a voice signal (see app/voice.py). It comes from the caption
    # ledger, NOT from ``Cut.draft_caption``: that field holds whatever the
    # operator last typed, so reading it here recorded every hand-written
    # caption as a draft the model had nailed — the exact inverse of the truth.
    try:
        post.suggested_caption = attach_draft_proposal(
            session, cut.id if cut else None, caption, post_pk=post.id)
    except Exception:
        log.exception("Caption proposal resolution failed for post %s", post.id)
    # Upload now, while the file is guaranteed to be on this machine, so a
    # headless scheduler (GitHub Actions / cron) can publish later without this
    # disk. Retried here, and again by the scheduler's repair sweep while the
    # post waits (``clip_uploaded_at`` stays NULL until a copy is confirmed) —
    # a publish on a machine WITHOUT the file can only sign what's already up.
    try:
        _upload_clip_with_retries(clip, post.clip_object_path)
        post.clip_uploaded_at = utcnow()
    except Exception as exc:
        log.warning("Queue-time clip upload failed (repair sweep will retry "
                    "while the post is queued): %s", exc)
    return post


def _clip_transcript_text(cut: Cut | None) -> str:
    """What is said in the clip itself, or "" when that can't be pinned down.

    Prefers the trimmed clip's saved Whisper word stream; otherwise slices the
    source transcript to the cut's trim windows. Never transcribes: this runs
    on the record path and on the headless scheduler. Never returns the whole
    source transcript either: a news segment often covers several stories, and
    a drafter handed all of them will happily caption the wrong one."""
    if cut is None:
        return ""
    if cut.clip_transcript_path:
        try:
            # subtitles pulls in Pillow, which the headless scheduler image
            # doesn't install; there the sidecar isn't on disk anyway, so fall
            # through to slicing the stored transcript.
            from .subtitles import load_clip_words, words_to_plain
            words = load_clip_words(cut.clip_transcript_path)
        except Exception:
            words = None
        if words:
            text = words_to_plain(words).strip()
            if text:
                return text
    cand = cut.candidate
    if cand is None:
        return ""
    return _slice_transcript(cand.transcript_text or "", cut.trim_segments or "")


_TRANSCRIPT_LINE = re.compile(r"^\[(?:(\d+):)?(\d+):(\d{2})\]\s*(.*)$")


def _slice_transcript(transcript: str, trim_segments: str) -> str:
    """Lines of a ``[mm:ss] text`` transcript that overlap the trim windows, in
    window order. A line runs until the next line's timestamp."""
    try:
        windows = [(float(w["start"]), float(w["end"]))
                   for w in json.loads(trim_segments or "[]")]
    except (ValueError, TypeError, KeyError):
        return ""
    lines: list[tuple[float, str]] = []
    for raw in transcript.splitlines():
        m = _TRANSCRIPT_LINE.match(raw.strip())
        if m:
            h, mnt, sec, text = m.groups()
            lines.append((int(h or 0) * 3600 + int(mnt) * 60 + int(sec), text.strip()))
    if not windows or not lines:
        return ""
    out: list[str] = []
    for ws, we in windows:
        for i, (start, text) in enumerate(lines):
            end = lines[i + 1][0] if i + 1 < len(lines) else start + 5
            if end > ws and start < we and text:
                out.append(text)
    return " ".join(out).strip()


def first_comment_suppressed(candidate: Candidate | None) -> bool:
    """Whether this post gets no first reply because of where it came from.

    True for the brand's own promos: on those the clip is the ad, so the call
    to action belongs in the caption, visible the moment the post lands.
    """
    return is_first_party(candidate)


def first_reply_text(cfg: dict, watch_url: str) -> str:
    """The first reply for a post: the watch text when its page is up, else
    the fallback text."""
    if watch_url and cfg.get("watch_text"):
        return cfg["watch_text"].replace(WATCH_URL_PLACEHOLDER, display_url(watch_url))
    return cfg.get("fallback_text", "")


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
            try:
                signed_url = signed_clip_url(post.clip_object_path)
            except Exception as exc:
                if _storage_object_missing(exc):
                    raise RuntimeError(
                        "The clip is not in storage (its queue-time upload never "
                        "completed) and this machine doesn't have the local file "
                        "to re-upload. Retry from the dashboard on the machine "
                        f"that exported the clip. ({exc})"
                    ) from exc
                raise
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
    # A no-op when the reply is gated on a delay or on organic engagement; the
    # scheduler's sweep posts it once the post has earned it.
    maybe_post_first_reply(session, post)
    return post


# --- Instagram Reels (paired posts) ------------------------------------------

def record_instagram_post(session, cut: Cut | None, threads_post: ThreadsPost | None,
                          clip_path: str, caption: str) -> InstagramPost:
    """Create (or refresh) the queued reel paired with ``threads_post``.

    The reel posts the cut's VERTICAL composite while the Threads post keeps
    its 16:9 clip, so the reel owns its own Supabase object. Uploaded now,
    while the file is certainly on this machine, so a headless runner can
    publish later (same rationale as ``record_post``). Reuses an existing
    not-yet-published row for the cut instead of stacking duplicates.

    ``threads_post`` is None for a reel shipped on its own (Instagram-only from
    the clip's Post step); it then has no scheduler window of its own and the
    caller publishes it directly. An existing row's pairing is left alone so
    reusing it doesn't orphan a Threads post that's still waiting to go out.
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
    if threads_post is not None:
        ig.threads_post_pk = threads_post.id
    ig.caption = caption
    if ig.clip_local_path != str(clip) or not ig.clip_object_path:
        ig.clip_local_path = str(clip)
        ig.clip_object_path = _object_key(clip)
        ig.clip_uploaded_at = None
        try:
            _upload_clip_with_retries(clip, ig.clip_object_path)
            ig.clip_uploaded_at = utcnow()
        except Exception as exc:
            # Not fatal here — but a headless publish will 404 signing an
            # object that never made it up, so the scheduler's repair sweep
            # keeps retrying from this machine while the reel is queued.
            log.warning("Queue-time reel upload failed (repair sweep will "
                        "retry while the reel is queued): %s", exc)
    ig.status = "queued"
    ig.error = ""
    session.flush()
    # Resolve the drafted hook against the one actually burned into the reel.
    # The hook has no accept/dismiss card — the draft lands straight in
    # ``Cut.hook_text`` and the operator types over it — so this diff is the
    # only place that rewrite is ever recorded.
    if cut is not None:
        try:
            attach_draft_proposal(session, cut.id, cut.hook_text or "",
                                  kind=KIND_HOOK, ig_post_pk=ig.id)
        except Exception:
            log.exception("Hook proposal resolution failed for reel %s", ig.id)
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
            try:
                signed_url = signed_clip_url(ig.clip_object_path)
            except Exception as exc:
                if _storage_object_missing(exc):
                    raise RuntimeError(
                        "The reel's video is not in storage (its queue-time "
                        "upload never completed) and this machine doesn't have "
                        "the local composite to re-upload. Retry from the "
                        "dashboard on the machine that exported the clip. "
                        f"({exc})"
                    ) from exc
                raise
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
    reel row (or None when no reel is paired).

    Transient failures retry in place, on the same knobs as the Threads
    window publish (``scheduler.publish_retries`` / ``…_retry_delay_seconds``)
    — the reel used to get exactly one attempt while its Threads twin got
    two, so a first-of-day token or Meta-processing hiccup routinely failed
    only the reel. A missing storage object on a machine without the local
    composite is permanent (nothing to upload, nothing to sign) and is not
    retried; the error says to retry from the machine that has the file.
    """
    from .db import session_scope

    settings = load_settings()
    retries = max(0, int(settings.get("scheduler.publish_retries", 1)))
    retry_delay = max(0, int(settings.get("scheduler.publish_retry_delay_seconds", 30)))

    snapshot: InstagramPost | None = None
    for attempt in range(retries + 1):
        retryable = False
        with session_scope() as session:
            ig = session.execute(
                select(InstagramPost).where(
                    InstagramPost.threads_post_pk == post_id,
                    InstagramPost.status.in_(["queued", "failed", "publishing"]),
                ).order_by(InstagramPost.created_at.desc()).limit(1)
            ).scalar_one_or_none()
            if ig is None:
                return snapshot
            try:
                publish_instagram_post(session, ig)
                session.expunge(ig)
                return ig
            except Exception as exc:
                log.warning("Paired Instagram reel attempt %d/%d failed for post %s: %s",
                            attempt + 1, retries + 1, post_id, exc)
                have_local = bool(ig.clip_local_path) and \
                    Path(ig.clip_local_path).expanduser().exists()
                retryable = (not isinstance(exc, FileNotFoundError)
                             and not (_storage_object_missing(exc) and not have_local))
                session.expunge(ig)
                snapshot = ig
        if not retryable or attempt >= retries:
            break
        time.sleep(retry_delay)
    return snapshot


def publish_reel_now(ig_id: int) -> InstagramPost | None:
    """Fire a recorded reel immediately, with no Threads post involved.

    Same out-of-transaction contract as ``publish_paired_reel``: the caller's
    recording transaction must have committed, because reel state is persisted
    through fresh connections (``_write_ig_row``). Returns a detached snapshot
    of the reel row (failures come back as ``status='failed'`` with ``error``
    set rather than raising)."""
    from .db import session_scope

    with session_scope() as session:
        ig = session.get(InstagramPost, ig_id)
        if ig is None:
            return None
        try:
            publish_instagram_post(session, ig)
        except Exception as exc:
            log.warning("Instagram-only reel %s failed: %s", ig_id, exc)
        session.expunge(ig)
    return ig


# --- LinkedIn company-page mirror ---------------------------------------------

def linkedin_mirror_enabled() -> bool:
    """Whether published Threads posts should mirror to LinkedIn right now."""
    return (bool(load_settings().get("linkedin.enabled", True))
            and linkedin_api.is_authenticated())


def _write_li_row(li_id: int, **values) -> None:
    """Update a LinkedInPost through a fresh connection (see ``_write_ig_row``)."""
    from sqlalchemy.exc import OperationalError

    from .db import session_scope

    last_exc: Exception | None = None
    for _ in range(2):
        try:
            with session_scope() as s:
                s.execute(update(LinkedInPost).where(LinkedInPost.id == li_id)
                          .values(**values))
            return
        except OperationalError as exc:
            last_exc = exc
    raise last_exc


def _claim_linkedin_mirror(post_id: int) -> tuple[int, str, str, str] | None:
    """Take ownership of mirroring ``post_id``: insert its row, or flip a
    failed one back to ``publishing``. Returns ``(li_id, caption, local clip,
    storage key)`` for the winner and None for everyone else — including a
    second runner whose insert hit the one-row-per-post constraint."""
    from sqlalchemy.exc import IntegrityError

    from .db import session_scope

    try:
        with session_scope() as session:
            post = session.get(ThreadsPost, post_id)
            if post is None or post.status != "published" or not post.threads_media_id:
                return None
            li = session.execute(
                select(LinkedInPost).where(LinkedInPost.threads_post_pk == post_id)
            ).scalar_one_or_none()
            if li is None:
                li = LinkedInPost(threads_post_pk=post_id, caption=post.caption or "",
                                  status="publishing")
                session.add(li)
                session.flush()
            else:
                won = session.execute(
                    update(LinkedInPost)
                    .where(LinkedInPost.id == li.id, LinkedInPost.status == "failed")
                    .values(status="publishing", error="", caption=post.caption or "")
                ).rowcount
                if won != 1:
                    return None
            return (li.id, post.caption or "", post.clip_local_path or "",
                    post.clip_object_path or "")
    except IntegrityError:
        return None


@contextmanager
def _mirror_clip(local_path: str, object_path: str):
    """The exact file the Threads post shipped: the local copy (with the outro
    the upload added) when this machine has it, else the storage copy streamed
    to a temp file (deleted on exit) — never held in memory, which the small
    Fly worker can't afford."""
    clip = Path(local_path).expanduser() if local_path else None
    if clip is not None and clip.exists():
        yield with_outro(clip)
        return
    if not object_path:
        raise FileNotFoundError(f"Clip not found: {local_path} (and no uploaded copy)")
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        with requests.get(signed_clip_url(object_path), stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
        tmp.flush()
        yield Path(tmp.name)


def _linkedin_retryable(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError) or _storage_object_missing(exc):
        return False
    status = getattr(exc, "status", None)
    # 4xx is LinkedIn rejecting the request (bad token, missing permission,
    # invalid video) and will fail identically again; 429 is just throttling.
    return status is None or status >= 500 or status == 429


def _linkedin_snapshot(post_id: int) -> LinkedInPost | None:
    from .db import session_scope

    with session_scope(read_only=True) as session:
        li = session.execute(
            select(LinkedInPost).where(LinkedInPost.threads_post_pk == post_id)
        ).scalar_one_or_none()
        if li is not None:
            session.expunge(li)
        return li


def publish_linkedin_mirror(post_id: int, *, force: bool = False) -> LinkedInPost | None:
    """Mirror a just-published Threads post to the LinkedIn company page.

    Called by every publish path AFTER the Threads publish commits, with the
    same contract as ``publish_paired_reel``: best-effort, never unwinds the
    Threads post, state written through fresh connections. ``force`` is the
    operator retrying by hand, which ignores ``linkedin.enabled``. Returns a
    detached snapshot of the row (None when nothing was mirrored).
    """
    if not force and not linkedin_mirror_enabled():
        return None
    claim = _claim_linkedin_mirror(post_id)
    if claim is None:
        return _linkedin_snapshot(post_id)
    li_id, caption, local_path, object_path = claim

    settings = load_settings()
    retries = max(0, int(settings.get("scheduler.publish_retries", 1)))
    retry_delay = max(0, int(settings.get("scheduler.publish_retry_delay_seconds", 30)))
    with _ACTIVE_LOCK:
        _ACTIVE_LINKEDIN.add(li_id)
    try:
        result = None
        for attempt in range(retries + 1):
            try:
                with _mirror_clip(local_path, object_path) as video:
                    result = linkedin_api.publish_video(video, caption)
                break
            except Exception as exc:
                log.warning("LinkedIn mirror attempt %d/%d failed for post %s: %s",
                            attempt + 1, retries + 1, post_id, exc)
                if attempt >= retries or not _linkedin_retryable(exc):
                    _write_li_row(li_id, status="failed", error=str(exc)[:1000])
                    break
                time.sleep(retry_delay)
        if result is not None:
            # Outside the retry loop: the post is live, so failing to record
            # that must never send a second copy.
            _write_li_row(li_id, post_urn=result["post_urn"],
                          video_urn=result["video_urn"],
                          permalink=result["permalink"], status="published",
                          published_at=utcnow(), error="")
            log.info("Mirrored post %s to LinkedIn (%s)", post_id, result["permalink"])
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_LINKEDIN.discard(li_id)
    return _linkedin_snapshot(post_id)


# --- X mirror ------------------------------------------------------------------

def x_mirror_enabled() -> bool:
    """Whether published Threads posts should mirror to X right now."""
    return bool(load_settings().get("x.enabled", True)) and x_api.is_authenticated()


def x_skip_reason(caption: str) -> str:
    """Why a caption won't go to X, or "" when it will."""
    if x_api.contains_link(caption):
        return "Not sent to X: the caption contains a link, which X bills at $0.20 a post."
    if x_api.weighted_length(caption) > x_api.MAX_WEIGHTED_LENGTH:
        return (f"Not sent to X: the caption is over X's "
                f"{x_api.MAX_WEIGHTED_LENGTH}-character limit.")
    return ""


def _write_x_row(x_id: int, **values) -> None:
    """Update an XPost through a fresh connection (see ``_write_ig_row``)."""
    from sqlalchemy.exc import OperationalError

    from .db import session_scope

    last_exc: Exception | None = None
    for _ in range(2):
        try:
            with session_scope() as s:
                s.execute(update(XPost).where(XPost.id == x_id).values(**values))
            return
        except OperationalError as exc:
            last_exc = exc
    raise last_exc


def _claim_x_mirror(post_id: int) -> tuple[int, str, str, str] | None:
    """``_claim_linkedin_mirror`` for X. A caption that can't go to X is
    recorded as ``skipped`` here and nobody wins the claim."""
    from sqlalchemy.exc import IntegrityError

    from .db import session_scope

    try:
        with session_scope() as session:
            post = session.get(ThreadsPost, post_id)
            if post is None or post.status != "published" or not post.threads_media_id:
                return None
            caption = post.caption or ""
            skip = x_skip_reason(caption)
            row = session.execute(
                select(XPost).where(XPost.threads_post_pk == post_id)
            ).scalar_one_or_none()
            if row is None:
                row = XPost(threads_post_pk=post_id, caption=caption,
                            status="skipped" if skip else "publishing", error=skip)
                session.add(row)
                session.flush()
                if skip:
                    return None
            else:
                won = session.execute(
                    update(XPost)
                    .where(XPost.id == row.id, XPost.status == "failed")
                    .values(status="skipped" if skip else "publishing",
                            error=skip, caption=caption)
                ).rowcount
                if won != 1 or skip:
                    return None
            return (row.id, caption, post.clip_local_path or "", post.clip_object_path or "")
    except IntegrityError:
        return None


def _x_retryable(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError) or _storage_object_missing(exc):
        return False
    status = getattr(exc, "status", None)
    return status is None or status >= 500 or status == 429


def _x_snapshot(post_id: int) -> XPost | None:
    from .db import session_scope

    with session_scope(read_only=True) as session:
        row = session.execute(
            select(XPost).where(XPost.threads_post_pk == post_id)
        ).scalar_one_or_none()
        if row is not None:
            session.expunge(row)
        return row


def publish_x_mirror(post_id: int, *, force: bool = False) -> XPost | None:
    """Mirror a just-published Threads post to X: the caption and clip only —
    never the first comment. Same contract as ``publish_linkedin_mirror``."""
    if not force and not x_mirror_enabled():
        return None
    claim = _claim_x_mirror(post_id)
    if claim is None:
        return _x_snapshot(post_id)
    x_id, caption, local_path, object_path = claim

    settings = load_settings()
    retries = max(0, int(settings.get("scheduler.publish_retries", 1)))
    retry_delay = max(0, int(settings.get("scheduler.publish_retry_delay_seconds", 30)))
    with _ACTIVE_LOCK:
        _ACTIVE_X.add(x_id)
    try:
        result = None
        for attempt in range(retries + 1):
            try:
                with _mirror_clip(local_path, object_path) as video:
                    result = x_api.publish_video(video, caption)
                break
            except x_api.XDuplicateError as exc:
                _write_x_row(x_id, status="skipped",
                             error=f"Not sent to X: X refused it as a duplicate of an "
                                   f"earlier post ({str(exc)[:300]})")
                break
            except Exception as exc:
                log.warning("X mirror attempt %d/%d failed for post %s: %s",
                            attempt + 1, retries + 1, post_id, exc)
                if attempt >= retries or not _x_retryable(exc):
                    _write_x_row(x_id, status="failed", error=str(exc)[:1000])
                    break
                time.sleep(retry_delay)
        if result is not None:
            _write_x_row(x_id, tweet_id=result["tweet_id"], media_id=result["media_id"],
                         permalink=result["permalink"], status="published",
                         published_at=utcnow(), error="")
            log.info("Mirrored post %s to X (%s)", post_id, result["permalink"])
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_X.discard(x_id)
    return _x_snapshot(post_id)


# --- Facebook Page mirror -------------------------------------------------------

# AppToken row holding the copyright pause: {"reason", "at", "threads_post_pk"}.
# In the database rather than settings.yaml so a headless runner can set it
# and the dashboard sees it.
FACEBOOK_PAUSE_TOKEN = "facebook_pause"


def facebook_pause_info() -> dict | None:
    """Why the Facebook mirror paused itself, or None when it hasn't."""
    from .db import session_scope

    with session_scope(read_only=True) as session:
        row = session.get(AppToken, FACEBOOK_PAUSE_TOKEN)
        if row is None or not row.value:
            return None
        try:
            return json.loads(row.value)
        except ValueError:
            return {"reason": row.value}


def pause_facebook_mirror(reason: str, threads_post_pk: int | None = None) -> None:
    from .db import session_scope

    payload = json.dumps({"reason": reason, "at": utcnow().isoformat(),
                          "threads_post_pk": threads_post_pk})
    with session_scope() as session:
        row = session.get(AppToken, FACEBOOK_PAUSE_TOKEN)
        if row is None:
            session.add(AppToken(name=FACEBOOK_PAUSE_TOKEN, value=payload, updated_at=utcnow()))
        elif not row.value:  # keep the first reason if it's already paused
            row.value = payload
            row.updated_at = utcnow()
    log.warning("Facebook mirror paused: %s", reason)


def resume_facebook_mirror() -> None:
    from .db import session_scope

    with session_scope() as session:
        row = session.get(AppToken, FACEBOOK_PAUSE_TOKEN)
        if row is not None:
            session.delete(row)


def facebook_mirror_enabled() -> bool:
    """Whether published Threads posts should mirror to Facebook right now."""
    return (bool(load_settings().get("facebook.enabled", True))
            and facebook_api.is_authenticated()
            and facebook_pause_info() is None)


def _write_fb_row(fb_id: int, **values) -> None:
    """Update a FacebookPost through a fresh connection (see ``_write_ig_row``)."""
    from sqlalchemy.exc import OperationalError

    from .db import session_scope

    last_exc: Exception | None = None
    for _ in range(2):
        try:
            with session_scope() as s:
                s.execute(update(FacebookPost).where(FacebookPost.id == fb_id)
                          .values(**values))
            return
        except OperationalError as exc:
            last_exc = exc
    raise last_exc


def _claim_facebook_mirror(post_id: int) -> tuple[int, str, str, str] | None:
    """``_claim_linkedin_mirror`` for Facebook."""
    from sqlalchemy.exc import IntegrityError

    from .db import session_scope

    try:
        with session_scope() as session:
            post = session.get(ThreadsPost, post_id)
            if post is None or post.status != "published" or not post.threads_media_id:
                return None
            caption = post.caption or ""
            row = session.execute(
                select(FacebookPost).where(FacebookPost.threads_post_pk == post_id)
            ).scalar_one_or_none()
            if row is None:
                row = FacebookPost(threads_post_pk=post_id, caption=caption,
                                   status="publishing")
                session.add(row)
                session.flush()
            else:
                won = session.execute(
                    update(FacebookPost)
                    .where(FacebookPost.id == row.id, FacebookPost.status == "failed")
                    .values(status="publishing", error="", caption=caption,
                            attention_dismissed_at=None)
                ).rowcount
                if won != 1:
                    return None
            return (row.id, caption, post.clip_local_path or "", post.clip_object_path or "")
    except IntegrityError:
        return None


def _facebook_retryable(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError) or _storage_object_missing(exc):
        return False
    if isinstance(exc, facebook_api.FacebookError) and exc.is_copyright:
        return False
    status = getattr(exc, "status", None)
    return status is None or status >= 500 or status == 429


def _facebook_snapshot(post_id: int) -> FacebookPost | None:
    from .db import session_scope

    with session_scope(read_only=True) as session:
        row = session.execute(
            select(FacebookPost).where(FacebookPost.threads_post_pk == post_id)
        ).scalar_one_or_none()
        if row is not None:
            session.expunge(row)
        return row


def publish_facebook_mirror(post_id: int, *, force: bool = False) -> FacebookPost | None:
    """Mirror a just-published Threads post to the Facebook Page: caption and
    clip. The first comment follows from ``check_facebook_videos`` once the
    video is ready. Same contract as
    ``publish_linkedin_mirror``; ``force`` (the operator retrying by hand) also
    goes through a copyright pause.

    An upload Facebook refuses on copyright grounds pauses the mirror, so one
    match can't turn into a run of strikes against the Page."""
    if not force and not facebook_mirror_enabled():
        return None
    claim = _claim_facebook_mirror(post_id)
    if claim is None:
        return _facebook_snapshot(post_id)
    fb_id, caption, local_path, object_path = claim

    settings = load_settings()
    retries = max(0, int(settings.get("scheduler.publish_retries", 1)))
    retry_delay = max(0, int(settings.get("scheduler.publish_retry_delay_seconds", 30)))
    with _ACTIVE_LOCK:
        _ACTIVE_FACEBOOK.add(fb_id)
    try:
        result = None
        for attempt in range(retries + 1):
            try:
                with _mirror_clip(local_path, object_path) as video:
                    result = facebook_api.publish_video(video, caption)
                break
            except Exception as exc:
                log.warning("Facebook mirror attempt %d/%d failed for post %s: %s",
                            attempt + 1, retries + 1, post_id, exc)
                if attempt >= retries or not _facebook_retryable(exc):
                    _write_fb_row(fb_id, status="failed", error=str(exc)[:1000])
                    if isinstance(exc, facebook_api.FacebookError) and exc.is_copyright:
                        pause_facebook_mirror(
                            f"Facebook refused a clip on copyright grounds: {str(exc)[:300]}",
                            post_id)
                    break
                time.sleep(retry_delay)
        if result is not None:
            _write_fb_row(fb_id, video_id=result["video_id"], permalink=result["permalink"],
                          status="published", published_at=utcnow(), error="")
            log.info("Mirrored post %s to Facebook (%s)", post_id, result["permalink"])
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_FACEBOOK.discard(fb_id)
    return _facebook_snapshot(post_id)


# How long after the Facebook upload a first comment is still worth posting,
# and when an unfinished claim on one can be retaken.
_FB_COMMENT_HORIZON = dt.timedelta(days=7)
_FB_COMMENT_CLAIM_STALE = dt.timedelta(minutes=15)


def post_facebook_comment(fb_id: int, *, force: bool = False) -> bool:
    """Post the Threads post's first reply as a Page comment on its Facebook
    video. Only text Threads actually posted goes out, so every Threads gate
    (delay, traction floor, promo suppression) carries over. ``force`` is the
    operator retrying by hand: it clears a stored error and takes over a claim.
    Returns True when a comment went out; failures land in ``comment_error``.
    """
    from .db import session_scope

    now = utcnow()
    with session_scope() as session:
        row = session.get(FacebookPost, fb_id)
        post = session.get(ThreadsPost, row.threads_post_pk) if row else None
        text = (post.first_reply_text or "").strip() if post and post.first_reply_id else ""
        if row is None or row.status != "published" or not row.video_id or not text \
                or row.comment_id or (row.comment_error and not force):
            return False
        where = [FacebookPost.id == fb_id, FacebookPost.comment_id == ""]
        if not force:
            where.append(or_(FacebookPost.comment_claimed_at.is_(None),
                             FacebookPost.comment_claimed_at < now - _FB_COMMENT_CLAIM_STALE))
        won = session.execute(
            update(FacebookPost).where(*where)
            .values(comment_claimed_at=now, comment_error="")
            .execution_options(synchronize_session=False)
        ).rowcount
        video_id = row.video_id
    if won != 1:
        return False
    try:
        comment_id = facebook_api.publish_comment(video_id, text)
    except Exception as exc:
        log.warning("Facebook first comment failed for row %s: %s", fb_id, exc)
        _write_fb_row(fb_id, comment_error=str(exc)[:1000])
        return False
    _write_fb_row(fb_id, comment_id=comment_id or "posted", comment_error="")
    log.info("Posted Facebook first comment %s on video %s", comment_id, video_id)
    return True


def check_facebook_videos(now: dt.datetime | None = None) -> int:
    """Read mirrored Facebook videos back, catching what only shows up after
    the upload — processing failures, copyright matches, takedowns — and
    posting the first comment once the video is ready.

    A video is read when its copyright watch is due (every
    ``facebook.copyright_check_minutes`` for ``facebook.copyright_watch_hours``
    after it went out — rights claims often land hours later) or when its
    Threads post has a first reply the Page hasn't commented yet. A match or a
    vanished video flags the post and pauses the mirror. Returns the number of
    posts whose state changed (comments included)."""
    from .db import session_scope

    if not facebook_api.is_authenticated():
        return 0
    settings = load_settings()
    now = now or utcnow()
    watch = dt.timedelta(hours=float(settings.get("facebook.copyright_watch_hours", 72)))
    every = dt.timedelta(minutes=float(settings.get("facebook.copyright_check_minutes", 60)))
    copyright_due = (FacebookPost.published_at >= now - watch) & or_(
        FacebookPost.checked_at.is_(None), FacebookPost.checked_at <= now - every)
    comment_due = ((FacebookPost.comment_id == "") & (FacebookPost.comment_error == "")
                   & (ThreadsPost.first_reply_id != "")
                   & (FacebookPost.published_at >= now - _FB_COMMENT_HORIZON)
                   & or_(FacebookPost.comment_claimed_at.is_(None),
                         FacebookPost.comment_claimed_at < now - _FB_COMMENT_CLAIM_STALE))
    with session_scope(read_only=True) as session:
        due = session.execute(
            select(FacebookPost.id, FacebookPost.video_id, FacebookPost.threads_post_pk,
                   comment_due)
            .join(ThreadsPost, ThreadsPost.id == FacebookPost.threads_post_pk)
            .where(FacebookPost.status == "published", FacebookPost.video_id != "",
                   or_(copyright_due, comment_due))
        ).all()
    changed = 0
    for fb_id, video_id, post_pk, wants_comment in due:
        try:
            info = facebook_api.video_status(video_id)
        except Exception as exc:
            log.warning("Facebook status check failed for video %s: %s", video_id, exc)
            continue
        values: dict = {"checked_at": now}
        if info.get("permalink"):
            values["permalink"] = info["permalink"]
        if info["copyright_match"] or info["state"] == "missing":
            reason = ("Facebook's copyright check matched this video against someone's "
                      "content." if info["copyright_match"] else
                      "This video is no longer on the Page — deleted by hand, or taken "
                      "down by a rights claim.")
            values.update(status="flagged", error=reason)
            pause_facebook_mirror(reason, post_pk)
            changed += 1
        elif info["state"] == "error":
            values.update(status="failed",
                          error=("Facebook couldn't process the video, so it never went "
                                 "out" + (f": {info['detail']}" if info.get("detail") else ".")))
            changed += 1
        _write_fb_row(fb_id, **values)
        if wants_comment and info["state"] == "ready" and not info["copyright_match"]:
            changed += post_facebook_comment(fb_id)
    return changed


def _annotate_footage(session, post: ThreadsPost) -> None:
    """Ground-truth footage trait tagging from the posted clip. Normally the
    scheduler tick has already done this at queue time (the placement variety
    gate needs the format facet before placing anything); the
    ``footage_scored_at`` guard makes this publish-time call a no-op then.
    Best-effort — a tagging hiccup must never undo a successful publish (the
    backfill command / dashboard button can retry later)."""
    try:
        from .db import active_traits_by_facet
        from .vision import annotate_post_footage

        vocab = active_traits_by_facet(session)
        annotate_post_footage(post, load_settings(), vocab["subject"],
                              format_traits=vocab["format"])
    except Exception as exc:
        log.warning("Footage annotation failed for post %s: %s", post.id, exc)


# A claim this old is treated as abandoned and can be retaken. Normally a claim
# lives and dies with the transaction that set it; this is the safety valve for
# one that somehow outlives it, so a comment is never blocked forever by a
# claim no process is acting on.
FIRST_REPLY_CLAIM_STALE = dt.timedelta(minutes=10)


def _claim_first_reply(session, post: ThreadsPost, *, force: bool = False) -> bool:
    """Take exclusive ownership of posting this post's first comment.

    A deferred reply is decided by whichever scheduler happens to tick — and
    both the dashboard thread and the Actions cron sweep the same due posts —
    so the decision has to be a conditional UPDATE, not a read followed by a
    write. Losers see rowcount 0 and leave the post alone. ``force`` is the
    operator asking by hand, which takes over a claim of any age.
    """
    now = utcnow()
    where = [ThreadsPost.id == post.id, ThreadsPost.first_reply_id == ""]
    if not force:
        where.append(or_(
            ThreadsPost.first_reply_claimed_at.is_(None),
            ThreadsPost.first_reply_claimed_at < now - FIRST_REPLY_CLAIM_STALE,
        ))
    # synchronize_session=False because the staleness test compares timestamps:
    # SQLAlchemy's default tries to re-evaluate the criteria in Python against
    # the identity map, and SQLite hands back naive datetimes where ``now`` is
    # aware — which raises rather than claiming anything. The row is expired
    # below instead, so the in-memory copy still refreshes.
    won = session.execute(
        update(ThreadsPost).where(*where).values(first_reply_claimed_at=now)
        .execution_options(synchronize_session=False)
    ).rowcount
    if won == 1:
        session.expire(post, ["first_reply_claimed_at"])
    return won == 1


def _publish_first_reply(text: str, reply_to_id: str, link: str) -> dict:
    """Post the first reply, pinning the preview card to ``link`` when given.

    ``link_attachment`` on a reply is the one part of this Meta hasn't been
    seen to accept yet. If it rejects the parameter (code 100, raised while
    creating the container, so nothing was published), post again without it.
    The URL is still in the text, so Threads can build the card from that.
    """
    if not link:
        return publish_text_reply(text, reply_to_id)
    try:
        return publish_text_reply(text, reply_to_id, link_attachment=link)
    except ThreadsError as exc:
        if exc.code != 100:
            raise
        log.warning("Threads rejected link_attachment on a reply (%s); posting without it", exc)
        return publish_text_reply(text, reply_to_id)


def maybe_post_first_reply(session, post: ThreadsPost, *, force: bool = False) -> bool:
    """Post the first reply under a published post.

    The watch text, linking the post's full-segment page, when that page can be
    published and loads; otherwise the fallback text. ``force`` is the operator
    posting by hand from the post page, which overrides the on/off switch and
    the brand-promo rule.

    Returns True if a reply was posted. Never raises: a post that gets no reply
    records why in ``first_reply_error`` (which also drops it out of the
    scheduler's sweep), so a reply hiccup can't undo a publish.
    """
    if post.status != "published" or not post.threads_media_id:
        return False
    if post.first_reply_id and not force:
        return False

    cfg = load_first_reply()
    if not force and first_comment_suppressed(post.candidate):
        post.first_reply_error = (
            "No first reply: this is the brand's own promo, so the call to "
            "action rides in the caption instead.")
        post.first_reply_skipped = True
        session.flush()
        log.info("First reply suppressed for first-party post %s", post.id)
        return False
    if not force and not cfg["enabled"]:
        post.first_reply_error = "No first reply: first replies are switched off on the First reply page."
        session.flush()
        return False

    if not _claim_first_reply(session, post, force=force):
        log.debug("First reply for post %s already claimed elsewhere", post.id)
        return False

    # After the claim, so two schedulers never both publish a page for the
    # same reply. No watch text means no page to link, so don't publish one.
    watch_url = ensure_watch_page(post) if cfg["watch_text"] else ""
    text = first_reply_text(cfg, watch_url)
    if not text:
        post.first_reply_error = "No first reply: there's no fallback text on the First reply page."
        session.flush()
        log.warning("Post %s published without a first reply: %s", post.id, post.first_reply_error)
        return False
    link = watch_url if watch_url and display_url(watch_url) in text else ""

    try:
        result = _publish_first_reply(text, post.threads_media_id, link)
        post.first_reply_id = result["media_id"]
        post.first_reply_text = text
        post.first_reply_error = ""
        # Both paths below clear the skip flag: whatever the scheduler decided
        # earlier, an operator posting by hand has overruled it, and the row
        # must not keep claiming the post was passed over.
        post.first_reply_skipped = False
        post.first_reply_at = utcnow()
        session.flush()
        log.info("Posted first reply %s under %s", post.first_reply_id, post.threads_media_id)
        return True
    except Exception as exc:
        post.first_reply_error = str(exc)[:1000]
        post.first_reply_skipped = False
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
