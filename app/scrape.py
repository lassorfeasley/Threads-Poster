"""Post-approval scrape: transcript from YouTube captions + full-segment
download via yt-dlp. ONLY runs for operator-approved candidates.

Designed to run on a residential IP. Downloads are sequential with randomized
politeness delays; there is deliberately no parallelism here.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

from .config import ROOT, load_settings
from .llm import suggest_highlight
from .models import STATUS_ARCHIVED, STATUS_FAILED, Candidate, utcnow

log = logging.getLogger("scrape")

_last_ytdlp_call: float = 0.0


def _politeness_delay(settings) -> None:
    """Sleep a randomized interval since the last yt-dlp/YouTube-page operation."""
    global _last_ytdlp_call
    lo = settings.get("scrape.delay_min_seconds", 8)
    hi = settings.get("scrape.delay_max_seconds", 25)
    wait = random.uniform(lo, hi)
    elapsed = time.time() - _last_ytdlp_call
    if elapsed < wait:
        time.sleep(wait - elapsed)
    _last_ytdlp_call = time.time()


def _safe_name(text: str, limit: int = 80) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in text)
    return "_".join(keep.split())[:limit]


def _paths_for(candidate: Candidate, settings) -> tuple[Path, Path]:
    """(video_dir, transcript_dir) organized by channel/date."""
    channel = _safe_name(candidate.channel.call_sign or "unknown").replace("/", "-")
    date = (candidate.published_at or utcnow()).strftime("%Y-%m-%d")
    video_dir = ROOT / settings.get("storage.download_dir", "data/videos") / channel / date
    transcript_dir = ROOT / settings.get("storage.transcript_dir", "data/transcripts") / channel / date
    video_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, transcript_dir


# --- Transcripts -------------------------------------------------------------

def fetch_captions(video_id: str) -> list[dict] | None:
    """Try YouTube captions/auto-captions. Returns [{start, end, text}] or None."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi().list(video_id)
        # find_transcript prefers manually created captions over auto-generated.
        transcript = transcript_list.find_transcript(["en", "en-US"])
        segments = transcript.fetch()
        return [
            {"start": s.start, "end": s.start + s.duration, "text": s.text}
            for s in segments
        ]
    except Exception as exc:
        log.info("Captions unavailable for %s: %s", video_id, exc)
        return None


def _write_transcript(segments: list[dict], transcript_dir: Path, video_id: str) -> tuple[Path, str]:
    """Write timestamped JSON + a readable .txt. Returns (json_path, plain_text)."""
    json_path = transcript_dir / f"{video_id}.json"
    json_path.write_text(json.dumps(segments, indent=1))
    lines = [f"[{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}] {s['text']}" for s in segments]
    plain = "\n".join(lines)
    (transcript_dir / f"{video_id}.txt").write_text(plain)
    return json_path, plain


# --- Download ----------------------------------------------------------------

def download_video(candidate: Candidate, video_dir: Path, settings) -> Path:
    """Download the full segment with yt-dlp. Idempotent: skips if file exists."""
    import yt_dlp

    existing = list(video_dir.glob(f"{candidate.video_id}.*"))
    media = [p for p in existing if p.suffix in (".mp4", ".mkv", ".webm", ".mov")]
    if media:
        log.info("Already downloaded: %s", media[0].name)
        return media[0]

    _politeness_delay(settings)
    opts = {
        "format": settings.get("scrape.ytdlp_format", "bv*[height<=1080]+ba/b[height<=1080]"),
        "outtmpl": str(video_dir / f"{candidate.video_id}.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        # Gentle: single connection, no fragment parallelism.
        "concurrent_fragment_downloads": 1,
    }
    # YouTube occasionally serves transient 403s on freshly extracted media
    # URLs; a re-extraction after a polite pause usually succeeds.
    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([candidate.url])
            break
        except Exception as exc:
            transient = "403" in str(exc) or "Forbidden" in str(exc)
            if transient and attempt < 2:
                wait = random.uniform(20, 40)
                log.info("Transient 403 for %s; retrying in %.0fs (attempt %d/3)",
                         candidate.video_id, wait, attempt + 2)
                time.sleep(wait)
            else:
                raise

    media = [p for p in video_dir.glob(f"{candidate.video_id}.*") if p.suffix in (".mp4", ".mkv", ".webm", ".mov")]
    if not media:
        raise RuntimeError("yt-dlp reported success but no media file found")
    return media[0]


# --- Orchestration -----------------------------------------------------------

def archive_candidate(session, candidate: Candidate, with_highlight: bool = True) -> None:
    """Full post-approval pipeline for one approved candidate.

    Idempotent: already-archived candidates return immediately; a re-run after
    partial failure resumes (existing files are reused, not re-downloaded).
    """
    if candidate.status == STATUS_ARCHIVED:
        return

    settings = load_settings()
    video_dir, transcript_dir = _paths_for(candidate, settings)

    try:
        # 1. Transcript from YouTube captions (no media download needed).
        segments = fetch_captions(candidate.video_id)
        method = "captions" if segments else ""

        # 2. Full segment download.
        video_path = download_video(candidate, video_dir, settings)
        candidate.local_video_path = str(video_path)

        if segments:
            json_path, plain = _write_transcript(segments, transcript_dir, candidate.video_id)
            candidate.transcript_path = str(json_path)
            candidate.transcript_text = plain
        else:
            log.warning("No captions for %s; archiving without a transcript", candidate.video_id)
        candidate.transcription_method = method

        # 3. Optional LLM highlight suggestion + draft caption (clearly drafts).
        if with_highlight and segments:
            try:
                hl = suggest_highlight(
                    settings.get("matching.model", "claude-haiku-4-5"), candidate.title, segments
                )
                if hl["end"] > hl["start"]:
                    candidate.suggested_highlight = (
                        f"{_fmt(hl['start'])}-{_fmt(hl['end'])}: {hl['why']}"
                    )
                    candidate.draft_caption = hl["draft_caption"]
            except Exception as exc:
                log.warning("Highlight suggestion failed for %s: %s", candidate.video_id, exc)

        candidate.status = STATUS_ARCHIVED
        candidate.archived_at = utcnow()
        candidate.scrape_error = ""
        log.info("Archived %s -> %s (%s)", candidate.video_id, video_path.name, method)
    except Exception as exc:
        candidate.status = STATUS_FAILED
        candidate.scrape_error = str(exc)[:1000]
        log.error("Scrape failed for %s: %s", candidate.video_id, exc)
    finally:
        session.flush()


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
