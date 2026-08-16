"""Post-approval scrape: transcript from YouTube captions + full-segment
download via yt-dlp. ONLY runs for operator-approved candidates.

Designed to run on a residential IP. Downloads are sequential with randomized
politeness delays; there is deliberately no parallelism here.
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path

from . import clip_proposals, spend
from .config import ROOT, storage_dir, load_settings
from .llm import suggest_title_from_transcript
from .models import STATUS_ARCHIVED, STATUS_FAILED, Candidate, utcnow

log = logging.getLogger("scrape")

# The last-call timestamp is shared by ALL workspace processes on this machine:
# every workspace downloads from the same residential IP, so the politeness
# interval must be global to the machine, not per process — otherwise two
# dashboards double the request rate YouTube sees.
_THROTTLE_FILE = ROOT / "workspaces" / ".shared" / "ytdlp_last"


def _politeness_delay(settings) -> None:
    """Sleep a randomized interval since the last yt-dlp/YouTube-page operation
    by ANY workspace process. The flock is held through the sleep on purpose:
    a second process arriving mid-wait queues behind the first and then times
    its own delay from the updated stamp, keeping downloads sequential
    machine-wide."""
    import fcntl

    lo = settings.get("scrape.delay_min_seconds", 8)
    hi = settings.get("scrape.delay_max_seconds", 25)
    _THROTTLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_THROTTLE_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            try:
                last = float(f.read().strip() or 0.0)
            except ValueError:
                last = 0.0
            wait = random.uniform(lo, hi)
            elapsed = time.time() - last
            if elapsed < wait:
                time.sleep(wait - elapsed)
            f.seek(0)
            f.truncate()
            f.write(str(time.time()))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _safe_name(text: str, limit: int = 80) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in text)
    return "_".join(keep.split())[:limit]


def _paths_for(candidate: Candidate, settings) -> tuple[Path, Path]:
    """(video_dir, transcript_dir) organized by channel/date."""
    channel = _safe_name(candidate.channel.call_sign or "unknown").replace("/", "-")
    date = (candidate.published_at or utcnow()).strftime("%Y-%m-%d")
    video_dir = storage_dir(settings, "storage.download_dir", "data/videos") / channel / date
    transcript_dir = storage_dir(settings, "storage.transcript_dir", "data/transcripts") / channel / date
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


_whisper_model = None


def _get_whisper_model(settings):
    """Lazily load the faster-whisper model (CPU, int8) once per process."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        size = settings.get("upload.whisper_model", "base")
        log.info("Loading faster-whisper model %r (first run downloads it)...", size)
        _whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_local(media_path: str | Path, settings) -> tuple[list[dict] | None, list[dict] | None]:
    """Transcribe a local media file with faster-whisper.

    Returns ``(segments, words)``: segment-level ``[{start, end, text}]`` and
    the word-level ``[{word, start, end}]`` stream from the same pass. Word
    timestamps come free with the transcription (no second decode), and they
    are what clip suggestions and burned-in captions cut against.
    Both are None when transcription fails or hears nothing.
    """
    try:
        model = _get_whisper_model(settings)
        # language pinned to English — see transcribe_words in subtitles.py:
        # auto-detect can misread noisy audio as Welsh and derail the output.
        segments, _info = model.transcribe(str(media_path), language="en",
                                           word_timestamps=True, vad_filter=True)
        out: list[dict] = []
        words: list[dict] = []
        for s in segments:
            out.append({"start": float(s.start), "end": float(s.end),
                        "text": (s.text or "").strip()})
            for w in s.words or []:
                text = (w.word or "").strip()
                if text:
                    words.append({"word": text, "start": float(w.start),
                                  "end": float(w.end)})
        return (out or None), (words or None)
    except Exception as exc:
        log.warning("Local transcription failed for %s: %s", media_path, exc)
        return None, None


def _write_transcript(segments: list[dict], transcript_dir: Path, video_id: str) -> tuple[Path, str]:
    """Write timestamped JSON + a readable .txt. Returns (json_path, plain_text)."""
    json_path = transcript_dir / f"{video_id}.json"
    json_path.write_text(json.dumps(segments, indent=1))
    lines = [f"[{int(s['start'] // 60):02d}:{int(s['start'] % 60):02d}] {s['text']}" for s in segments]
    plain = "\n".join(lines)
    (transcript_dir / f"{video_id}.txt").write_text(plain)
    return json_path, plain


def _write_word_transcript(words: list[dict], transcript_dir: Path, video_id: str) -> Path:
    """Persist the full-video Whisper word stream (``[{word, start, end}]``).

    Same shape ``subtitles.py`` uses for clip-level word streams, so exports
    can slice this file by trim windows instead of re-running Whisper."""
    path = transcript_dir / f"{video_id}_words.json"
    path.write_text(json.dumps(words, indent=1))
    return path


def load_word_transcript(candidate) -> list[dict]:
    """The video's Whisper word stream, or [] when it has none."""
    path = getattr(candidate, "word_transcript_path", "") or ""
    if not path:
        return []
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [w for w in data if isinstance(w, dict) and (w.get("word") or "").strip()]


# --- Metadata ----------------------------------------------------------------

def fetch_video_metadata(url: str) -> dict:
    """Fetch title/duration/uploader for a YouTube URL without downloading.

    Uses yt-dlp's metadata extraction (no API key required). Returns a dict with
    video_id, title, duration_seconds, uploader, upload_date (YYYYMMDD or None),
    and webpage_url. Raises whatever yt-dlp raises on a bad/unavailable URL.
    """
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "video_id": info.get("id"),
        "title": info.get("title") or "",
        "duration_seconds": int(info["duration"]) if info.get("duration") else None,
        "uploader": info.get("uploader") or info.get("channel") or "",
        "upload_date": info.get("upload_date"),
        "webpage_url": info.get("webpage_url") or url,
    }


# --- Download ----------------------------------------------------------------

_MEDIA_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")
# yt-dlp sidecars / in-progress names left behind when a download is killed
# (laptop sleep, kill -9, network drop) before the final merge.
_PARTIAL_SUFFIXES = (".part", ".ytdl", ".temp", ".tmp")


def _has_stream(path: Path, codec_type: str) -> bool:
    """True when ffprobe sees at least one stream of ``codec_type`` (``audio`` /
    ``video``) on ``path``."""
    import subprocess

    sel = {"audio": "a", "video": "v"}.get(codec_type)
    if not sel:
        return False
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", sel,
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return codec_type in (out.stdout or "")
    except Exception:
        return False


def _has_audio_stream(path: Path) -> bool:
    return _has_stream(path, "audio")


def _has_video_stream(path: Path) -> bool:
    return _has_stream(path, "video")


def _is_format_fragment(path: Path) -> bool:
    """yt-dlp names single-format intermediates ``id.f399.mp4`` / ``id.f140.m4a``.
    Those are never the finished merge — treating them as done is what made an
    interrupted download look permanently "ready" with no waveform."""
    return bool(re.search(r"\.f\d+$", path.stem))


def _is_usable_media(path: Path) -> bool:
    """Finished download: real video+audio, not a yt-dlp format fragment."""
    if not path.exists() or path.suffix.lower() not in _MEDIA_SUFFIXES:
        return False
    if _is_format_fragment(path):
        return False
    return _has_video_stream(path) and _has_audio_stream(path)


def _candidate_artifacts(video_dir: Path, video_id: str) -> list[Path]:
    """Every on-disk artifact for a video id (media, audio sidecars, partials)."""
    return sorted(p for p in video_dir.glob(f"{video_id}.*") if p.is_file())


def _candidate_media_files(video_dir: Path, video_id: str) -> list[Path]:
    return [p for p in _candidate_artifacts(video_dir, video_id)
            if p.suffix.lower() in _MEDIA_SUFFIXES]


def _pick_usable_media(paths: list[Path]) -> Path | None:
    usable = [p for p in paths if _is_usable_media(p)]
    if not usable:
        return None
    # Prefer mp4 (our merge_output_format) when several somehow exist.
    usable.sort(key=lambda p: (0 if p.suffix.lower() == ".mp4" else 1, p.name))
    return usable[0]


def _purge_download_artifacts(video_dir: Path, video_id: str, *, keep: Path | None = None) -> None:
    """Wipe incomplete yt-dlp leftovers for ``video_id``.

    Safe to call before a download, after a failed attempt, or when a "finished"
    run produced only fragments. Keeps ``keep`` (the verified merged file).
    """
    for p in _candidate_artifacts(video_dir, video_id):
        if keep is not None and p.resolve() == keep.resolve():
            continue
        # Always drop partials / format fragments / audio-only sidecars.
        drop = (
            p.suffix.lower() in _PARTIAL_SUFFIXES
            or _is_format_fragment(p)
            or p.suffix.lower() in (".m4a", ".aac", ".opus")
            or (p.suffix.lower() in _MEDIA_SUFFIXES and not _is_usable_media(p))
        )
        if not drop:
            continue
        try:
            p.unlink()
            log.warning("Removed incomplete download artifact: %s", p.name)
        except OSError as exc:
            log.warning("Could not remove %s: %s", p.name, exc)


def download_video(candidate: Candidate, video_dir: Path, settings) -> Path:
    """Download the full segment with yt-dlp. Idempotent only when a *usable*
    audio+video merge already exists. Interrupted downloads (laptop sleep,
    killed process, failed audio merge) leave video-only ``.fNNN`` fragments —
    those are purged and the download is retried rather than archived as ready.
    """
    import yt_dlp

    existing = _candidate_media_files(video_dir, candidate.video_id)
    usable = _pick_usable_media(existing)
    if usable is not None:
        # Drop any leftover fragments sitting next to a good merge.
        _purge_download_artifacts(video_dir, candidate.video_id, keep=usable)
        log.info("Already downloaded: %s", usable.name)
        return usable
    # Nothing usable — clear partials so yt-dlp starts clean (a resume of a
    # half-written fragment after sleep is how we used to get stuck).
    _purge_download_artifacts(video_dir, candidate.video_id)

    _politeness_delay(settings)
    opts = {
        "format": settings.get("scrape.ytdlp_format", "bv*[height<=1080]+ba/b[height<=1080]"),
        "outtmpl": str(video_dir / f"{candidate.video_id}.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        # Don't resume a half-written fragment from an interrupted prior run —
        # those are what looked "done" with no audio. Fresh download each try.
        "continuedl": False,
        "overwrites": True,
        # Gentle: single connection, no fragment parallelism.
        "concurrent_fragment_downloads": 1,
    }
    # YouTube occasionally serves transient 403s on freshly extracted media
    # URLs; a re-extraction after a polite pause usually succeeds. Also retry
    # when the process is interrupted mid-merge (laptop sleep → empty/partial).
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([candidate.url])
            usable = _pick_usable_media(
                _candidate_media_files(video_dir, candidate.video_id)
            )
            if usable is not None:
                _purge_download_artifacts(video_dir, candidate.video_id, keep=usable)
                return usable
            # yt-dlp exited 0 but left only fragments (typical after a sleep
            # during the audio fetch / ffmpeg merge).
            _purge_download_artifacts(video_dir, candidate.video_id)
            last_exc = RuntimeError(
                "download finished without a merged audio+video file "
                "(interrupted mid-merge?); will retry"
            )
            log.warning("%s for %s (attempt %d/3)",
                        last_exc, candidate.video_id, attempt + 1)
        except Exception as exc:
            last_exc = exc
            _purge_download_artifacts(video_dir, candidate.video_id)
            if attempt < 2:
                wait = random.uniform(20, 40) if "403" in str(exc) or "Forbidden" in str(exc) else random.uniform(3, 8)
                log.info("Download failed for %s (%s); retrying in %.0fs (attempt %d/3)",
                         candidate.video_id, exc, wait, attempt + 2)
                time.sleep(wait)
                continue
            raise

        if attempt < 2:
            wait = random.uniform(3, 8)
            log.info("Retrying download for %s in %.0fs (attempt %d/3)",
                     candidate.video_id, wait, attempt + 2)
            time.sleep(wait)

    raise RuntimeError(
        f"yt-dlp could not produce a merged audio+video file for {candidate.video_id}"
        + (f": {last_exc}" if last_exc else "")
    )


# --- Orchestration -----------------------------------------------------------

# The synthetic channel that owns every pasted-URL clip (see the web layer's
# _get_or_create_pasted_channel).
PASTED_CHANNEL_URL = "youtube://pasted"


def _wants_transcript_title(candidate: Candidate) -> bool:
    """Whether this clip's title should be rewritten from its transcript.

    Only clips the operator brought in by pasting a URL. Discovered clips keep
    their real YouTube titles: those are the review context on the dashboard and
    they feed relevance matching, so rewriting them would be destructive. A
    pasted clip's title is nothing but a label, and the source headline can be
    wrong about its own footage, so the spoken words are the better source.
    """
    channel = candidate.channel
    return bool(channel and channel.url == PASTED_CHANNEL_URL)


def archive_candidate(session, candidate: Candidate, with_suggestions: bool = True) -> None:
    """Full post-approval pipeline for one approved candidate.

    Idempotent for a *successful* archive. A re-run after failure (or after
    Retry) resumes: incomplete downloads are purged and re-fetched rather than
    reused, so an interrupted laptop sleep can't leave a video permanently
    stuck as "ready" with no waveform.
    """
    if candidate.status == STATUS_ARCHIVED:
        return

    settings = load_settings()
    video_dir, transcript_dir = _paths_for(candidate, settings)
    is_upload = (candidate.url or "").startswith("upload://")

    try:
        if is_upload:
            # Operator-uploaded file: already on disk, nothing to download.
            # Transcribe locally (no YouTube captions exist for it).
            video_path = Path(candidate.local_video_path)
            if not video_path.exists():
                raise RuntimeError(f"Uploaded file missing: {video_path}")
            if not _has_audio_stream(video_path):
                raise RuntimeError(
                    f"Uploaded file has no audio track (needed for waveform/"
                    f"transcript): {video_path.name}"
                )
            segments, words = transcribe_local(video_path, settings)
            method = "whisper" if segments else ""
        else:
            # 1. Transcript from YouTube captions (no media download needed).
            segments = fetch_captions(candidate.video_id)
            method = "captions" if segments else ""
            # 2. Full segment download (refuses to return a video-only partial).
            video_path = download_video(candidate, video_dir, settings)
            if not _is_usable_media(video_path):
                raise RuntimeError(
                    "Download did not produce a usable audio+video file — "
                    "often caused by interrupting the download (e.g. closing "
                    "the laptop). Hit Retry and leave it running until it finishes."
                )
            # 3. Whisper always runs on the downloaded file: the word-level
            # timestamps it produces are what clip suggestions and burned-in
            # captions cut against, and YouTube captions can't provide them.
            # When captions exist they still win the segment-level transcript
            # (better punctuation); Whisper failing must NOT fail the archive.
            w_segments, words = transcribe_local(video_path, settings)
            if segments and words:
                method = "captions+whisper"
            elif not segments:
                log.info("No captions for %s; using local Whisper transcript",
                         candidate.video_id)
                segments = w_segments
                method = "whisper" if segments else ""

        candidate.local_video_path = str(video_path)

        if segments:
            json_path, plain = _write_transcript(segments, transcript_dir, candidate.video_id)
            candidate.transcript_path = str(json_path)
            candidate.transcript_text = plain
        else:
            log.warning("No transcript for %s (captions and Whisper both empty); "
                        "archiving without one", candidate.video_id)
        candidate.transcription_method = method
        if words:
            word_path = _write_word_transcript(words, transcript_dir, candidate.video_id)
            candidate.word_transcript_path = str(word_path)

        # 4. Retitle pasted clips from what is actually said. Runs before the
        # clip pass so the caption draft is written against the corrected
        # title. Best-effort: any failure leaves the ingest-time title standing.
        if segments and _wants_transcript_title(candidate):
            try:
                title = suggest_title_from_transcript(
                    settings.get("matching.model", "claude-haiku-4-5"),
                    candidate.transcript_text or "",
                    source_title=candidate.description or "",
                )
                if title:
                    log.info("Retitled %s from transcript: %r -> %r",
                             candidate.video_id, candidate.title, title)
                    candidate.title = title[:300]
            except Exception as exc:
                log.warning("Transcript title failed for %s: %s", candidate.video_id, exc)

        # 5. Clip suggestions: which clips this video holds and where they run
        # (clearly drafts). Budget-gated because this runs unattended over a
        # whole monitor pass; the operator's on-demand re-roll is not.
        if with_suggestions and segments:
            if not spend.within_budget():
                log.info("Skipping clip suggestions for %s: daily LLM budget reached",
                         candidate.video_id)
            else:
                try:
                    clip_proposals.propose(session, candidate, settings, segments)
                except Exception as exc:
                    log.warning("Clip suggestion failed for %s: %s", candidate.video_id, exc)

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
