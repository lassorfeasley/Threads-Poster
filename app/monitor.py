"""Discovery pipeline: poll channels -> keyword filter -> LLM score -> store candidates.

Nothing here downloads video. This is the only step that talks to the
YouTube Data API, which is sanctioned and key-authenticated.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select

from . import youtube
from .config import load_keywords, load_settings
from .db import active_traits, session_scope, sync_channels_from_config, sync_traits_from_config
from .llm import score_relevance
from .matching import find_keyword_matches
from .models import Candidate, Channel, utcnow
from .ranking import load_trait_weights, trait_guidance_text
from .vision import apply_visual_score

log = logging.getLogger("monitor")


def ensure_channel_resolved(channel: Channel) -> bool:
    """Resolve URL -> canonical channel id + uploads playlist, once, and cache in DB."""
    if channel.channel_id and channel.uploads_playlist_id:
        return True
    try:
        info = youtube.resolve_channel(channel.url)
    except youtube.YouTubeAPIError as exc:
        channel.last_error = f"resolve failed: {exc}"
        log.warning("Could not resolve %s (%s): %s", channel.call_sign, channel.url, exc)
        return False
    channel.channel_id = info["channel_id"]
    channel.uploads_playlist_id = info["uploads_playlist_id"]
    channel.channel_title = info["title"]
    channel.last_error = ""
    return True


def poll_channel(session, channel: Channel, keywords: list[str], settings,
                 lookback_days: int | None = None, vision_state: dict | None = None,
                 visual_guidance: str = "", desirable: list[str] | None = None,
                 undesirable: list[str] | None = None) -> int:
    """Check one channel for new uploads; store keyword-matching candidates.

    `lookback_days` overrides the normal since-last-check watermark to scan
    further back (backfill). Duplicates are impossible either way — video ids
    are unique in the store. Returns count stored. `vision_state` carries the
    running per-pass count so the vision-scoring cap spans all channels.
    """
    if not ensure_channel_resolved(channel):
        return 0

    max_results = settings.get("monitor.max_results_per_channel", 25)
    if lookback_days is not None:
        since = utcnow() - dt.timedelta(days=lookback_days)
        # Backfills page deeper than a routine poll.
        max_results = max(max_results, min(50 * lookback_days, 200))
    else:
        # Default window: calendar days "today + yesterday" (configurable). Scans
        # the whole window every run rather than only since the last check; video
        # ids are unique so re-listed items are skipped before any LLM scoring.
        window_days = settings.get("monitor.default_lookback_days", 2)
        today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        since = today - dt.timedelta(days=max(window_days, 1) - 1)
    if since.tzinfo is None:
        since = since.replace(tzinfo=dt.timezone.utc)

    try:
        uploads = youtube.list_recent_uploads(
            channel.uploads_playlist_id,
            since=since,
            max_results=max_results,
        )
    except youtube.YouTubeAPIError as exc:
        channel.last_error = f"poll failed: {exc}"
        log.warning("Poll failed for %s: %s", channel.call_sign, exc)
        return 0

    channel.last_checked_at = utcnow()
    channel.last_error = ""
    stored = 0

    for up in uploads:
        # Advance last-seen watermark regardless of keyword match.
        if channel.last_seen_published_at is None or up.published_at > channel.last_seen_published_at.replace(
            tzinfo=dt.timezone.utc
        ):
            channel.last_seen_published_at = up.published_at

        matched = find_keyword_matches(f"{up.title}\n{up.description}", keywords)
        if not matched:
            continue

        # Dedupe: video ids are globally unique, so shared-newsroom channels and
        # re-polls can never create a duplicate candidate.
        exists = session.execute(
            select(Candidate.id).where(Candidate.video_id == up.video_id)
        ).scalar_one_or_none()
        if exists is not None:
            continue

        # Portrait/vertical uploads (Shorts) don't fit the clip workflow.
        if settings.get("monitor.exclude_portrait", True) and youtube.is_short(up.video_id):
            log.info("Skipping Short: [%s] %s", channel.call_sign, up.title)
            continue

        candidate = Candidate(
            video_id=up.video_id,
            channel_pk=channel.id,
            title=up.title,
            description=up.description,
            url=up.url,
            thumbnail_url=up.thumbnail_url,
            published_at=up.published_at,
            duration_seconds=up.duration_seconds,
            matched_keywords=",".join(matched),
        )

        try:
            result = score_relevance(
                settings.get("matching.model", "claude-haiku-4-5"), up.title, up.description, matched
            )
            candidate.relevance_score = result["score"]
            candidate.relevance_rationale = result["rationale"]
            candidate.climate_topic = ",".join(result["topics"])  # multi-topic CSV
        except Exception as exc:  # scoring failure shouldn't lose the candidate
            log.warning("LLM scoring failed for %s: %s", up.video_id, exc)
            candidate.relevance_rationale = f"(scoring failed: {exc})"

        # Vision scoring: gated by relevance, a per-pass cap, and the daily
        # budget (all enforced inside apply_visual_score). Metadata-only.
        if settings.get("vision.enabled", True) and settings.get("vision.score_at_monitor", True):
            try:
                apply_visual_score(candidate, settings, run_state=vision_state,
                                   learned_guidance=visual_guidance,
                                   desirable=desirable, undesirable=undesirable)
            except Exception as exc:  # never lose a candidate over vision scoring
                log.warning("Vision scoring failed for %s: %s", up.video_id, exc)

        session.add(candidate)
        stored += 1
        log.info("Candidate: [%s] %s (score=%s, visual=%s)", channel.call_sign, up.title,
                 candidate.relevance_score, candidate.visual_score)

    return stored


def run_monitor_once(lookback_days: int | None = None) -> dict:
    """One full pass over all enabled channels. `lookback_days` scans that far
    back regardless of last-seen state (backfill). Returns summary counts."""
    settings = load_settings()
    keywords = load_keywords()
    total_stored = 0
    checked = 0
    # Shared across all channels so the per-pass vision cap is a global limit.
    vision_state = {"scored": 0}
    with session_scope() as session:
        sync_channels_from_config(session)
        sync_traits_from_config(session)
        # Feed learned trait performance back into the vision prompt as a soft prior.
        visual_guidance = trait_guidance_text(load_trait_weights(session), settings)
        desirable, undesirable = active_traits(session)
        channels = session.execute(select(Channel).where(Channel.enabled)).scalars().all()
        for channel in channels:
            total_stored += poll_channel(session, channel, keywords, settings, lookback_days,
                                         vision_state=vision_state, visual_guidance=visual_guidance,
                                         desirable=desirable, undesirable=undesirable)
            checked += 1
            # Commit per channel: keeps write transactions short (no long lock
            # for other sessions) and preserves progress if a pass is interrupted.
            session.commit()
    log.info("Monitor pass done: %d channels checked, %d new candidates, %d vision-scored",
             checked, total_stored, vision_state["scored"])
    return {"channels_checked": checked, "candidates_stored": total_stored,
            "vision_scored": vision_state["scored"]}
