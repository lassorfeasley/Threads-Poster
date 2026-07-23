"""Trim/supercut export: cut one or more segments from the downloaded source
video and join them into a single social-ready mp4 using ffmpeg.

Each segment is re-encoded (frame-accurate cuts, uniform params), then joined
with the concat demuxer. Output goes to data/clips/.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from .config import ROOT

log = logging.getLogger("clipper")

CLIPS_DIR = ROOT / "data" / "clips"

ENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-c:a", "aac", "-b:a", "128k",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
]


class ClipExportError(RuntimeError):
    pass


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

    log.info("Exported %d-segment supercut %s", len(cleaned), output.name)
    return output


WAVEFORM_DIR = ROOT / "data" / "waveforms"


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
