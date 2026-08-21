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
from .draft_proposals import KIND_HOOK, attach_to_post as attach_draft_proposal
from .instagram_api import publish_reel
from .llm import caption_attributes, suggest_attribution, suggest_first_reply
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
        # recorded no duration at all.
        clip_length_seconds=_clip_duration_seconds(clip),
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
    # First comment: only ever text the caller already has in hand — the
    # operator's own, or a call-to-action draft the ship path produced via
    # ``draft_first_reply_for_cut`` before opening this session. Nothing is
    # drafted here, because a model call inside this transaction would hold a
    # pooled connection for the seconds it takes. An empty field means this post
    # gets no first comment beyond whatever static fallback is configured.
    if attribution.strip():
        post.attribution_text = attribution.strip()
    # Upload now, while the file is guaranteed to be on this machine, so a
    # headless scheduler (GitHub Actions / cron) can publish later without this
    # disk. Best-effort: publish_post re-uploads from local when it can.
    try:
        upload_trimmed_clip(clip, post.clip_object_path)
    except Exception as exc:
        log.warning("Queue-time clip upload failed (will retry at publish): %s", exc)
    return post


def generate_attribution(candidate: Candidate) -> str:
    """LLM-drafted formal citation crediting the source station/publisher (and
    the program/journalists when the video's own material establishes them).
    Only runs when the operator clicks Suggest — never automatically. Gets the
    FULL source video context (title, description, publish date, complete
    transcript), not just the clipped segments. Returns "" when the data can't
    support a credible citation."""
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
        transcript=candidate.transcript_text or "",
        published_at=(candidate.published_at.strftime("%B %-d, %Y")
                      if candidate.published_at else ""),
        video_url=candidate.url or "",
    )


def _clip_transcript_text(cut: Cut | None) -> str:
    """Plain text of the trimmed clip's Whisper word stream, when one was already
    saved. Never transcribes: this runs on the record path, where a Whisper pass
    would be far too slow, so a missing sidecar just falls back to the source
    video's transcript."""
    if cut is None or not cut.clip_transcript_path:
        return ""
    from .subtitles import load_clip_words, words_to_plain
    try:
        words = load_clip_words(cut.clip_transcript_path)
    except Exception:
        return ""
    return words_to_plain(words).strip() if words else ""


def _recent_first_replies(session, limit: int = 12) -> list[str]:
    """The most recent first comments, newest first, fed to the drafting prompt so
    a fresh invitation doesn't echo the ones around it.

    Covers posted replies AND the text still sitting in unpublished posts' boxes:
    a queue drafted in one sitting publishes days apart, so looking only at what
    has already gone out would let every post in that batch open the same way.
    """
    rows = session.execute(
        select(ThreadsPost.first_reply_text, ThreadsPost.attribution_text)
        .where((ThreadsPost.first_reply_text != "") | (ThreadsPost.attribution_text != ""))
        .order_by(ThreadsPost.id.desc())
        .limit(limit)
    ).all()
    seen: list[str] = []
    for posted, pending in rows:
        text = (posted or "").strip() or (pending or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def first_reply_context(session, candidate: Candidate | None, cut: Cut | None,
                        caption: str = "") -> dict | None:
    """Snapshot everything the call-to-action prompt needs as plain values.

    Split from the model call on purpose: the draft takes several seconds, and
    doing it mid-session pinned a pooled database connection for the whole of
    it. Returns None when invitation mode is off or has no brief.
    """
    cfg = load_first_reply()
    instruction = (cfg.get("instruction") or "").strip()
    if cfg.get("mode") != "invitation" or not instruction:
        return None
    transcript = _clip_transcript_text(cut)
    if not transcript and candidate is not None:
        transcript = (candidate.transcript_text or "").strip()
    return {
        "instruction": instruction,
        "fallback": (cfg.get("text") or "").strip(),
        "video_title": candidate.title if candidate else "",
        "description": candidate.description if candidate else "",
        "transcript": transcript,
        "caption": caption or "",
        "recent_replies": _recent_first_replies(session),
    }


def draft_first_reply(context: dict | None) -> str:
    """Run the call-to-action draft for a ``first_reply_context`` snapshot.

    Holds no database connection: callers gather the context in a short session,
    let it close, then call this. It is only ever a DRAFT — the text lands in the
    editable first-reply box and the operator can rewrite or clear it before the
    post publishes.
    """
    if not context:
        return ""
    settings = load_settings()
    text = suggest_first_reply(
        settings.get("engagement.draft_model", "claude-sonnet-5"),
        context["instruction"],
        video_title=context.get("video_title", ""),
        description=context.get("description", ""),
        transcript=context.get("transcript", ""),
        caption=context.get("caption", ""),
        recent_replies=context.get("recent_replies") or [],
    )
    # The whole point of invitation mode is that the call to action rides under
    # every post, so a clip the model declines to write a custom opener for
    # falls back to the configured static pitch rather than to nothing.
    return text or context.get("fallback", "")


def draft_first_reply_for_cut(cut_id: int, caption: str = "") -> str:
    """Call-to-action draft for a cut, from outside any caller's transaction.

    Used by the ship paths (post now / queue / save draft) before they open the
    session that records the post, so the model call never runs with a
    connection checked out. Returns "" when invitation mode is off.
    """
    from .db import session_scope
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return ""
        context = first_reply_context(session, cut.candidate, cut, caption)
    return draft_first_reply(context)


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
        try:
            upload_trimmed_clip(clip, ig.clip_object_path)
        except Exception as exc:
            log.warning("Queue-time reel upload failed (will retry at publish): %s", exc)
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


def _no_first_comment_reason(post: ThreadsPost, cfg: dict) -> str:
    """Plain-language reason a published post carries no first comment, stored on
    the post so it shows up on the post page instead of looking untouched."""
    kind = "call to action" if cfg.get("mode") == "invitation" else "attribution"
    if (post.attribution_text or "").strip() and not cfg.get("attribution_enabled"):
        return (f"No first comment: this post has {kind} text, but first comments "
                "are switched off under Replies settings.")
    if post.attribution_skipped:
        return (f"No first comment: the {kind} was cleared for this post, and "
                "no static reply text is enabled under Replies settings.")
    return (f"No first comment: no {kind} was set on this post, and no static "
            "reply text is enabled under Replies settings.")


def maybe_post_first_reply(session, post: ThreadsPost, *, force: bool = False) -> bool:
    """Post the first reply under a published post: the post's own attribution
    comment when the operator set one, else the static configured text. Never
    drafts anything itself — a post whose attribution field was left empty
    simply publishes without an attribution comment.

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
        post.first_reply_error = _no_first_comment_reason(post, cfg)
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
