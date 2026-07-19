"""Vision scoring: judge how engaging a candidate's footage looks from YouTube's
storyboard stills, before anything is downloaded.

Pipeline: storyboard sprite sheets (metadata-only fetch, already cached by
``storyboard.get_storyboard``) -> download a few sheet JPEGs -> Claude vision ->
a 0-1 ``visual_score`` plus detected popularity traits. All gated by relevance,
a per-run cap, and the daily spend budget so cost stays bounded.
"""
from __future__ import annotations

import logging

import requests

from . import llm, spend
from .config import Settings
from .models import Candidate, utcnow
from .storyboard import get_storyboard

log = logging.getLogger("vision")

DEFAULT_TRAITS = [
    "action", "people_doing_things", "fire", "flood", "storm_damage",
    "destruction", "rescue_or_emergency_response", "crowd", "dramatic_weather",
    "aerial_or_sweeping_shot",
]
DEFAULT_UNDESIRABLE_TRAITS = [
    "talking_head_or_anchor_at_desk", "static_graphic_or_slideshow",
    "chart_or_data_screen", "text_heavy_lower_thirds",
]


def _clean(values) -> list[str]:
    return [str(x).strip() for x in (values or []) if str(x).strip()]


def _settings_traits(settings: Settings) -> tuple[list[str], list[str]]:
    """Fallback trait lists from config (used when the DB lists aren't passed)."""
    desirable = _clean(settings.get("vision.desirable_traits") or settings.get("vision.traits")) \
        or DEFAULT_TRAITS
    undesirable = _clean(settings.get("vision.undesirable_traits")) or DEFAULT_UNDESIRABLE_TRAITS
    return desirable, undesirable


def storyboard_images(video_id: str, max_sheets: int = 4) -> list[bytes]:
    """Download up to ``max_sheets`` storyboard sprite sheets (evenly spaced
    across the clip) as JPEG bytes. Each sheet is a contact-sheet of stills."""
    sb = get_storyboard(video_id)
    if not sb.get("available"):
        return []
    frags = [f for f in sb.get("fragments", []) if f.get("url")]
    if not frags:
        return []
    if len(frags) > max_sheets:
        # Evenly sample sheets across the timeline for broad coverage.
        idxs = [round(k * (len(frags) - 1) / (max_sheets - 1)) for k in range(max_sheets)] \
            if max_sheets > 1 else [len(frags) // 2]
        frags = [frags[i] for i in sorted(set(idxs))]
    images: list[bytes] = []
    for frag in frags:
        try:
            resp = requests.get(frag["url"], timeout=20)
            if resp.ok and resp.content:
                images.append(resp.content)
        except requests.RequestException as exc:
            log.info("Storyboard sheet fetch failed for %s: %s", video_id, exc)
    return images


def should_score(candidate: Candidate, settings: Settings, run_state: dict | None,
                 force: bool) -> tuple[bool, str]:
    """Gate a candidate for vision scoring. Returns (ok, reason_if_not)."""
    if not settings.get("vision.enabled", True):
        return False, "vision disabled"
    if candidate.visual_score is not None and not force:
        return False, "already scored"
    if not force:
        min_rel = settings.get("vision.min_relevance", 0.5)
        if candidate.relevance_score is not None and candidate.relevance_score < min_rel:
            return False, "below min_relevance"
        cap = settings.get("vision.max_per_run", 40)
        if run_state is not None and run_state.get("scored", 0) >= cap:
            return False, "max_per_run reached"
    # Budget guard applies even to manual/forced scoring — cost is real either way.
    if not spend.within_budget():
        return False, "daily budget reached"
    return True, ""


def apply_visual_score(candidate: Candidate, settings: Settings,
                       run_state: dict | None = None, force: bool = False,
                       learned_guidance: str = "",
                       desirable: list[str] | None = None,
                       undesirable: list[str] | None = None) -> dict | None:
    """Score one candidate's visuals and write the result onto it (caller
    commits). ``desirable``/``undesirable`` come from the trait database; if not
    passed they fall back to config. Returns the score dict, or None if
    skipped/unavailable."""
    ok, reason = should_score(candidate, settings, run_state, force)
    if not ok:
        log.info("Skipping visual score for %s: %s", candidate.video_id, reason)
        return None

    images = storyboard_images(candidate.video_id, settings.get("vision.max_sheets", 4))
    if not images:
        log.info("No storyboard images for %s; cannot vision-score", candidate.video_id)
        return None

    if desirable is None or undesirable is None:
        cfg_desirable, cfg_undesirable = _settings_traits(settings)
        desirable = desirable if desirable is not None else cfg_desirable
        undesirable = undesirable if undesirable is not None else cfg_undesirable

    try:
        result = llm.score_visuals(
            settings.get("vision.model", "claude-haiku-4-5"),
            images, desirable, undesirable, title=candidate.title,
            learned_guidance=learned_guidance,
        )
    except Exception as exc:
        log.warning("Visual scoring failed for %s: %s", candidate.video_id, exc)
        return None

    candidate.visual_score = result["visual_score"]
    candidate.visual_traits = ",".join(result["traits"])
    candidate.visual_rationale = result["why"]
    candidate.visual_scored_at = utcnow()
    if run_state is not None:
        run_state["scored"] = run_state.get("scored", 0) + 1
    log.info("Visual score for %s: %.2f [%s]", candidate.video_id,
             result["visual_score"], candidate.visual_traits)
    return result
