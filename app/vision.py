"""Footage trait tagging — observation only, no good/bad scores.

Two entry points:

- ``tag_candidate_storyboard`` — optional pre-download tags from YouTube
  storyboard stills (metadata-only). Neutral labels for triage visibility;
  not used for ranking judgment.
- ``annotate_post_footage`` — ground truth from the actual posted clip file
  (ffmpeg contact sheet). This is what the learning loop trains on.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import requests

from . import llm, spend
from .config import Settings
from .models import Candidate, ThreadsPost, utcnow
from .storyboard import get_storyboard

log = logging.getLogger("vision")

DEFAULT_TRAITS = [
    "action", "people_doing_things", "fire", "flood", "storm_damage",
    "destruction", "rescue_or_emergency_response", "crowd", "dramatic_weather",
    "aerial_or_sweeping_shot", "talking_head_or_anchor_at_desk",
    "static_graphic_or_slideshow", "chart_or_data_screen",
    "text_heavy_lower_thirds", "press_conference_podium",
    "low_motion_studio_segment",
]


def _clean(values) -> list[str]:
    return [str(x).strip() for x in (values or []) if str(x).strip()]


def vocabulary_from_settings(settings: Settings) -> list[str]:
    """Flat trait vocabulary from config (fallback when DB list isn't passed)."""
    traits = _clean(settings.get("vision.traits"))
    if traits:
        return traits
    # Legacy seed keys still accepted so old settings keep working.
    return _clean(settings.get("vision.desirable_traits")) \
        + _clean(settings.get("vision.undesirable_traits")) \
        or list(DEFAULT_TRAITS)


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


def should_tag(candidate: Candidate, settings: Settings, run_state: dict | None,
               force: bool) -> tuple[bool, str]:
    """Gate a candidate for storyboard tagging. Returns (ok, reason_if_not)."""
    if not settings.get("vision.enabled", True):
        return False, "vision disabled"
    if candidate.visual_traits and not force:
        return False, "already tagged"
    # Legacy: previously gated on visual_score; treat any prior score as tagged.
    if candidate.visual_score is not None and not force and not candidate.visual_traits:
        return False, "already scored"
    if not force:
        min_rel = settings.get("vision.min_relevance", 0.5)
        if candidate.relevance_score is not None and candidate.relevance_score < min_rel:
            return False, "below min_relevance"
        cap = settings.get("vision.max_per_run", 40)
        if run_state is not None and run_state.get("scored", 0) >= cap:
            return False, "max_per_run reached"
    if not spend.within_budget():
        return False, "daily budget reached"
    return True, ""


def tag_candidate_storyboard(candidate: Candidate, settings: Settings,
                             run_state: dict | None = None, force: bool = False,
                             traits: list[str] | None = None) -> dict | None:
    """Tag a candidate from YouTube storyboard stills (caller commits). Neutral
    labels only — does not write a visual score. Returns the tag dict, or None
    if skipped/unavailable."""
    ok, reason = should_tag(candidate, settings, run_state, force)
    if not ok:
        log.info("Skipping storyboard tag for %s: %s", candidate.video_id, reason)
        return None

    images = storyboard_images(candidate.video_id, settings.get("vision.max_sheets", 4))
    if not images:
        log.info("No storyboard images for %s; cannot tag", candidate.video_id)
        return None

    vocab = traits if traits is not None else vocabulary_from_settings(settings)
    try:
        result = llm.tag_footage(
            settings.get("vision.model", "claude-haiku-4-5"),
            images, vocab, title=candidate.title,
        )
    except Exception as exc:
        log.warning("Storyboard tagging failed for %s: %s", candidate.video_id, exc)
        return None

    candidate.visual_traits = ",".join(result["traits"])
    candidate.visual_rationale = result["why"]
    candidate.visual_scored_at = utcnow()
    # Stop writing judgment scores; clear any stale value on re-tag.
    candidate.visual_score = None
    if run_state is not None:
        run_state["scored"] = run_state.get("scored", 0) + 1
    log.info("Storyboard tags for %s: [%s]", candidate.video_id, candidate.visual_traits)
    return result


# Back-compat alias used by older call sites / CLI.
apply_visual_score = tag_candidate_storyboard


def _clip_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def clip_contact_sheet(path: str | Path, frames: int = 12, tile: str = "4x3",
                       width: int = 320) -> bytes | None:
    """One JPEG contact sheet of ``frames`` evenly spaced stills from a local
    video file. Returns None when ffmpeg/ffprobe fail."""
    clip = Path(path).expanduser()
    if not clip.exists():
        return None
    duration = _clip_duration(clip)
    if not duration or duration <= 0:
        return None
    vf = f"fps={frames}/{duration:.3f},scale={width}:-2,tile={tile}"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(clip), "-vf", vf, "-frames:v", "1",
             "-q:v", "5", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0 or not proc.stdout:
            log.info("Contact sheet failed for %s: %s", clip,
                     proc.stderr.decode(errors="replace")[-300:])
            return None
        return proc.stdout
    except Exception as exc:
        log.info("Contact sheet failed for %s: %s", clip, exc)
        return None


def annotate_post_footage(post: ThreadsPost, settings: Settings,
                          traits: list[str], force: bool = False) -> dict | None:
    """Tag a published post from its posted clip file (caller commits). This is
    the ground-truth signal for learning — traits of what actually shipped,
    paired later with performance. Budget-guarded; returns None when skipped."""
    if post.footage_scored_at is not None and not force:
        return None
    if not post.clip_local_path:
        return None
    if not spend.within_budget():
        log.info("Skipping footage annotation for post %s: daily budget reached", post.id)
        return None

    sheet = clip_contact_sheet(post.clip_local_path,
                               settings.get("vision.post_frames", 12))
    if sheet is None:
        return None
    try:
        result = llm.tag_footage(
            settings.get("vision.model", "claude-haiku-4-5"),
            [sheet], traits, title=post.caption[:200],
        )
    except Exception as exc:
        log.warning("Footage annotation failed for post %s: %s", post.id, exc)
        return None

    post.footage_traits = ",".join(result["traits"])
    post.footage_rationale = result["why"]
    post.footage_score = None  # judgment scores retired
    post.footage_scored_at = utcnow()
    log.info("Footage traits for post %s: [%s]", post.id, post.footage_traits)
    return result
