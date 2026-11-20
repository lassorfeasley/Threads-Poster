"""Trim/supercut export: cut one or more segments from the downloaded source
video and join them into a single social-ready mp4 using ffmpeg.

Each segment is re-encoded (frame-accurate cuts, uniform params), then joined
with the concat demuxer, and the result's audio is levelled to one loudness
(``level_audio``). Output goes to data/clips/.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from .config import DATA_DIR

log = logging.getLogger("clipper")

CLIPS_DIR = DATA_DIR / "clips"
THUMBS_DIR = DATA_DIR / "thumbs"

ENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-c:a", "aac", "-b:a", "128k",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
]


class ClipExportError(RuntimeError):
    pass


# Every exported clip is levelled to one loudness, so a quiet interview and a
# loud protest play at the same volume in the feed. -14 LUFS is where the big
# feeds normalise to; -1.5 dBTP keeps peaks clear of clipping after AAC.
TARGET_LUFS = -14.0
TARGET_PEAK_DBTP = -1.5
TARGET_LRA = 11.0
# Quieter than this is silence (or room tone): leave it alone, don't pump it up.
_SILENCE_LUFS = -50.0


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise ClipExportError(f"ffmpeg failed: {proc.stderr[-500:]}")


def compress_to_fit(source_path: str | Path, max_bytes: int, *, audio_kbps: int = 128) -> Path:
    """Re-encode ``source_path`` so the result fits under ``max_bytes``.

    Threads only accepts video by URL, and the Supabase upload rejects anything
    over the project's per-file limit (413 Payload too large) — which silently
    fails the publish. We compute a target video bitrate from the clip duration
    and the size budget so the output lands just under the ceiling in one pass.
    Returns a temp file path (caller is responsible for cleanup); raises
    ClipExportError if the clip can't be measured.
    """
    source = Path(source_path)
    if not source.exists():
        raise ClipExportError(f"Clip to compress not found: {source}")
    duration = clip_duration(source)
    if not duration or duration <= 0:
        raise ClipExportError(f"Could not read duration of {source} to compress it")

    # 0.92 leaves headroom for container/muxing overhead so we stay under the cap.
    total_kbps = (max_bytes * 8 / 1000.0) / duration * 0.92
    video_kbps = int(total_kbps - audio_kbps)
    if video_kbps < 200:
        # Clip is long enough that fitting the budget needs an aggressive bitrate;
        # floor it so the video stays watchable and downscale to 720p to help.
        video_kbps = 200
    scale = ["-vf", "scale='min(1280,iw)':-2"] if video_kbps <= 1500 else []

    fd, tmp_name = tempfile.mkstemp(prefix="compressed_", suffix=".mp4")
    os.close(fd)
    dest = Path(tmp_name)
    _run_ffmpeg([
        "-i", str(source),
        *scale,
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", f"{video_kbps}k",
        "-maxrate", f"{int(video_kbps * 1.45)}k", "-bufsize", f"{video_kbps * 2}k",
        "-c:a", "aac", "-b:a", f"{audio_kbps}k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dest),
    ])
    log.info(
        "Compressed %s (%.1fMB) -> %.1fMB at ~%dkbps to fit %.1fMB cap",
        source.name, source.stat().st_size / 1e6, dest.stat().st_size / 1e6,
        video_kbps, max_bytes / 1e6,
    )
    return dest


def _loudness(path: Path) -> dict | None:
    """EBU R128 measurements of ``path``'s audio (loudnorm's first pass), or
    None when it has no audio to measure."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0?", "-vn",
         "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK_DBTP}:LRA={TARGET_LRA}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600)
    text = proc.stderr
    start, end = text.rfind("{"), text.rfind("}")
    if proc.returncode != 0 or start < 0 or end < start:
        return None
    try:
        stats = json.loads(text[start:end + 1])
        float(stats["input_i"])
    except (ValueError, KeyError):
        return None
    return stats


def level_audio(path: str | Path) -> bool:
    """Bring ``path``'s audio to TARGET_LUFS in place (two-pass loudnorm: one
    measured, even gain, so the dynamics are kept). The video is copied, not
    re-encoded. Returns whether it changed anything. Never raises: a clip
    whose audio can't be levelled keeps the audio it had."""
    path = Path(path)
    try:
        stats = _loudness(path)
        if stats is None:
            return False
        measured = float(stats["input_i"])
        if measured < _SILENCE_LUFS or abs(measured - TARGET_LUFS) < 0.5:
            return False
        tmp = path.with_name(f".{path.stem}.level{path.suffix}")
        _run_ffmpeg([
            "-i", str(path), "-map", "0:v:0?", "-map", "0:a:0",
            "-c:v", "copy",
            "-af", (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK_DBTP}:LRA={TARGET_LRA}"
                    f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
                    f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
                    f":offset={stats['target_offset']}:linear=true"),
            "-ar", "48000", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(tmp),
        ])
        tmp.replace(path)
        log.info("Levelled %s from %.1f to %.0f LUFS", path.name, measured, TARGET_LUFS)
        return True
    except Exception as exc:
        log.warning("Couldn't level the audio of %s (%s); keeping it as it was", path.name, exc)
        return False


def export_supercut(source_path: str | Path, segments: list[dict], output_name: str) -> Path:
    """Cut `segments` ([{start, end}, ...] seconds, in order) from source and
    export one joined mp4. Returns the output path."""
    source = Path(source_path)
    if not source.exists():
        raise ClipExportError(f"Source video not found: {source}")
    cleaned = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end - start < 0.5:
            raise ClipExportError(f"Segment {start:.1f}-{end:.1f}s is too short")
        cleaned.append((start, end))
    if not cleaned:
        raise ClipExportError("No segments provided")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    output = CLIPS_DIR / f"{output_name}.mp4"

    if len(cleaned) == 1:
        start, end = cleaned[0]
        _run_ffmpeg(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source), *ENCODE_ARGS, str(output)])
        level_audio(output)
        log.info("Exported single-segment clip %s", output.name)
        return output

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        part_paths = []
        for i, (start, end) in enumerate(cleaned):
            part = tmpdir / f"part{i}.mp4"
            _run_ffmpeg(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source), *ENCODE_ARGS, str(part)])
            part_paths.append(part)
        concat_list = tmpdir / "list.txt"
        concat_list.write_text("".join(f"file '{p}'\n" for p in part_paths))
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output)])

    level_audio(output)   # the whole clip at once, so its segments sit at one level
    log.info("Exported %d-segment supercut %s", len(cleaned), output.name)
    return output


WAVEFORM_DIR = DATA_DIR / "waveforms"


def get_waveform(source_path: str | Path, video_id: str, buckets: int = 1200) -> dict:
    """Audio peak envelope for the trim UI: `buckets` normalized 0-1 values.

    Decodes mono 8kHz PCM via ffmpeg and takes the max amplitude per bucket.
    Cached to data/waveforms/<video_id>.json (computed once per video).
    """
    import array
    import json as json_mod

    WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)
    cache = WAVEFORM_DIR / f"{video_id}.json"
    if cache.exists():
        return json_mod.loads(cache.read_text())

    source = Path(source_path)
    if not source.exists():
        raise ClipExportError(f"Source video not found: {source}")

    proc = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(source), "-ac", "1", "-ar", "8000",
         "-f", "s16le", "-"],
        capture_output=True, timeout=300,
    )
    if proc.returncode != 0 or not proc.stdout:
        # Distinguish "no audio track" (common after a failed video+audio merge)
        # from a genuine decode failure — the trim UI's "waveform unavailable"
        # is much clearer with the real reason.
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(source)],
            capture_output=True, text=True, timeout=30,
        )
        if "audio" not in (probe.stdout or ""):
            raise ClipExportError("source has no audio track")
        raise ClipExportError("ffmpeg could not decode audio for waveform")

    samples = array.array("h")
    samples.frombytes(proc.stdout[: len(proc.stdout) // 2 * 2])
    n = len(samples)
    duration = clip_duration(source) or (n / 8000.0)

    per_bucket = max(1, n // buckets)
    peaks = []
    for i in range(0, n, per_bucket):
        chunk = samples[i:i + per_bucket]
        peaks.append(max(abs(s) for s in chunk) if len(chunk) else 0)
        if len(peaks) >= buckets:
            break
    top = max(peaks) or 1
    result = {"peaks": [round(p / top, 3) for p in peaks], "duration": duration}
    cache.write_text(json_mod.dumps(result))
    return result


def clip_duration(path: str | Path) -> float | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(proc.stdout.strip())
    except Exception:
        return None


def extract_still(source_path: str | Path, dest_path: str | Path,
                  at_seconds: float | None = None) -> Path:
    """Grab a single representative frame from ``source_path`` as a JPEG.

    Seeks a little way in by default, since a first frame is often black or
    mid-fade. Raises ClipExportError if no frame can be extracted.
    """
    source = Path(source_path)
    if not source.exists():
        raise ClipExportError(f"Clip for still not found: {source}")
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if at_seconds is None:
        dur = clip_duration(source) or 0.0
        # A short way in avoids fade-ins/blank first frames; clamp for tiny clips.
        at_seconds = min(1.0, dur / 2) if dur else 0.0

    _run_ffmpeg([
        "-ss", f"{max(0.0, at_seconds):.3f}", "-i", str(source),
        "-frames:v", "1", "-q:v", "3",
        "-vf", "scale='min(640,iw)':-2",
        str(dest),
    ])
    if not dest.exists() or dest.stat().st_size == 0:
        raise ClipExportError(f"Still extraction produced no image for {source}")
    return dest


def cached_still(source_path: str | Path, key: str,
                 at_seconds: float | None = None) -> Path:
    """Poster frame for ``source_path``, extracted once and reused thereafter.

    ``at_seconds`` picks the moment (default: a short way in); callers asking
    for a specific moment must fold it into ``key``.

    Keyed rather than written next to the source because clips live in several
    trees (per-channel, uploads, pasted URLs) and some of those are pruned.
    A source newer than its cached frame wins, so replacing a video file — a
    re-download, a re-upload over the same id — refreshes the still.
    """
    source = Path(source_path)
    if not source.exists():
        raise ClipExportError(f"Clip for still not found: {source}")
    dest = THUMBS_DIR / f"{key}.jpg"
    try:
        if dest.stat().st_size > 0 and dest.stat().st_mtime >= source.stat().st_mtime:
            return dest
    except OSError:
        pass
    # Extract to a private path and rename into place: two requests for the same
    # clip can race here, and a half-written JPEG would be cached as a valid one.
    tmp = dest.with_name(f"{key}.{uuid4().hex[:8]}.tmp.jpg")
    try:
        extract_still(source, tmp, at_seconds=at_seconds)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
