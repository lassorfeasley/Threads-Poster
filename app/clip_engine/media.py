"""ffmpeg work for the clip engine: proxies, draft renders, shot maps, frames.

Kept apart from ``clipper`` (the export path) because the engine's needs are
different and its failure modes are worse:

- It runs unattended in the dashboard, many times per video. Every ffmpeg and
  ffprobe call here has a timeout scaled to the media it reads, and ANY
  failure — a non-zero exit, a timeout, a missing binary — comes back as
  ``MediaError``. ``clipper._run_ffmpeg`` lets ``TimeoutExpired`` escape, so
  one stuck render would sink a whole engine run instead of one draft.
- Renders keep PLAY order (a cold open plays its payoff first) and are made in
  ONE ffmpeg process — an input-side seek per segment feeding a concat filter
  — so sound and picture stay exactly aligned at every join. Each segment's
  audio gets a 30 ms fade in and out; hard audio cuts click, and a critic
  listening for "dead air" hears the click as a glitch.
- Proxies, renders and shot maps are cached by the source file's identity
  (path, size, mtime). The dashboard and a ``run.py clipbench`` process share
  these directories, so every write goes to a tmp name unique to the process
  and thread and is renamed into place, cache hits refresh the mtime (``prune``
  is age-based), and an error or an empty result is never cached — a cached
  failure would quietly disable snapping for that video forever.
- Shot maps are the engine's exact clock: scdet scene-change scores for every
  frame, and 50 ms RMS windows for pauses. A pause is measured RELATIVE to the
  source's own median loudness, because broadcast audio has no fixed silence
  floor (music beds, room tone).

Times are seconds on the SOURCE timeline unless a function says otherwise
(``clip_frames`` returns clip seconds).
"""
from __future__ import annotations

import contextlib
import glob
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

from . import engine_dir, util

log = logging.getLogger("clip_engine.media")

VERSION = 1

# ffprobe reads a header, not the media: a fixed limit is enough.
_PROBE_TIMEOUT = 60.0
# Renders refuse segments shorter than this (a few frames isn't a shot).
_MIN_SEGMENT = 0.3
# Frame sampling runs at most this many seconds of source per ffmpeg call, so
# a long source never rides on a single call's timeout.
_FRAME_CHUNK_SECONDS = 300.0
# Pauses: 50 ms RMS windows (800 samples at 16 kHz) quieter than the source's
# median by this many dB, lasting at least this long.
_RMS_RATE = 16000
_RMS_WINDOW_SAMPLES = 800
_QUIET_BELOW_MEDIAN_DB = 12.0
_QUIET_MIN_SECONDS = 0.15
# No engine run holds a source pin (or writes a tmp file) for a day; anything
# older is a leftover of a crash.
_STALE_SECONDS = 86400.0

_ENCODE_VIDEO = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
_FASTSTART = ["-movflags", "+faststart"]

# The fps filter emits, for each output slot, the LAST input frame whose
# rounded time is <= the slot. ``start_time=0`` pins slot 0 to the start of
# the (seeked) input, so a first frame landing a few ms after the seek point
# doesn't shift the whole segment a frame late. Renders round DOWN: each
# segment comes out within a frame of its length, so joins don't drift. Stills
# round UP: the frame labelled t shows what is on screen AT t (rounding to
# nearest would show up to half an interval later — a mislabelled opening).
_RENDER_FPS = "fps=25:start_time=0:round=down"


class MediaError(RuntimeError):
    """An ffmpeg/ffprobe call failed, timed out or produced nothing usable."""


# ---- running ffmpeg --------------------------------------------------------

def _run(cmd: list[str], *, timeout: float, what: str) -> subprocess.CompletedProcess:
    """Run one ffmpeg/ffprobe command; every way it can fail is a MediaError."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise MediaError(f"{what} timed out after {timeout:.0f}s") from None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise MediaError(f"{what} could not run: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", errors="replace").strip()[-400:]
        raise MediaError(f"{what} failed (exit {proc.returncode}): {tail}")
    return proc


def _ffmpeg(args: list[str], *, timeout: float, what: str) -> subprocess.CompletedProcess:
    return _run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args],
                timeout=timeout, what=what)


def timeout_for(seconds: float | None) -> float:
    """How long an ffmpeg call over ``seconds`` of media may take: 120 s plus
    half real time (decoding 1080p AV1 runs at roughly 50x real time here, so
    this only trips on a genuinely stuck process). 600 s when unknown."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return 600.0
    if not math.isfinite(value) or value <= 0:
        return 600.0
    return 120.0 + 0.5 * value


# ---- probing ---------------------------------------------------------------

def _existing(path) -> Path:
    if path is None or str(path) == "":
        raise MediaError("no media file given")
    p = Path(path).expanduser()
    if not p.is_file():
        raise MediaError(f"media file not found: {p}")
    return p


def _probe(path: Path) -> dict:
    """``{"duration": float | None, "audio": bool, "video": bool}``."""
    proc = _run(["ffprobe", "-v", "error", "-show_entries",
                 "format=duration:stream=codec_type,duration", "-of", "json", str(path)],
                timeout=_PROBE_TIMEOUT, what=f"ffprobe {path.name}")
    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "{}")
    except ValueError as exc:
        raise MediaError(f"ffprobe {path.name}: unreadable output") from exc
    streams = [s for s in data.get("streams") or [] if isinstance(s, dict)]

    def seconds(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) and out > 0 else None

    duration = seconds((data.get("format") or {}).get("duration"))
    if duration is None:
        # Some containers only carry per-stream durations.
        duration = max((d for d in (seconds(s.get("duration")) for s in streams) if d),
                       default=None)
    kinds = {s.get("codec_type") for s in streams}
    return {"duration": duration, "audio": "audio" in kinds, "video": "video" in kinds}


def probe_duration(path) -> float | None:
    """The file's duration in seconds, or None when it can't be read."""
    try:
        return _probe(_existing(path))["duration"]
    except MediaError as exc:
        log.info("Couldn't read the duration of %s: %s", path, exc)
        return None


def has_audio(path) -> bool:
    """Whether the file has an audio stream. Raises MediaError when the file
    can't be probed at all (a broken file is not the same as a silent one)."""
    return _probe(_existing(path))["audio"]


def file_key(path) -> str:
    """Cache key for a file's current contents: sha1(abs path, size, mtime).
    A re-download or restore changes the mtime, so stale caches can't match."""
    p = Path(os.path.abspath(os.path.expanduser(str(path))))
    try:
        st = p.stat()
    except OSError as exc:
        raise MediaError(f"media file not found: {p}") from exc
    return util.sha1(str(p), st.st_size, st.st_mtime_ns, n=16)


# ---- cache plumbing --------------------------------------------------------

_build_locks: dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """One lock per cache file, so two threads asking for the same render
    wait for one encode instead of running two."""
    key = str(path)
    with _build_locks_guard:
        if len(_build_locks) > 512:
            for stale in [k for k, lock in _build_locks.items() if not lock.locked()]:
                del _build_locks[stale]
        return _build_locks.setdefault(key, threading.Lock())


def _tmp_for(dest: Path) -> Path:
    """A sibling tmp name unique to this process and thread, keeping the
    extension so tools that sniff it still work."""
    return dest.with_name(f".{dest.stem}.{os.getpid()}.{threading.get_ident()}.tmp{dest.suffix}")


def _reuse(dest: Path) -> bool:
    """A cached file is reusable when it exists and isn't empty; reuse refreshes
    its mtime so ``prune`` keeps what's still being used."""
    try:
        if dest.stat().st_size > 0:
            util.touch(dest)
            return True
    except OSError:
        pass
    return False


def _encode(args: list[str], dest: Path, *, timeout: float, what: str) -> Path:
    """Run ffmpeg writing to a private tmp file, then rename it over ``dest``.
    Nothing reaches ``dest`` unless ffmpeg succeeded and wrote something."""
    dest = Path(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaError(f"{what}: can't create {dest.parent}: {exc}") from exc
    tmp = _tmp_for(dest)
    try:
        _ffmpeg([*args, str(tmp)], timeout=timeout, what=what)
        try:
            if tmp.stat().st_size <= 0:
                raise MediaError(f"{what} produced an empty file")
            os.replace(tmp, dest)
        except OSError as exc:
            raise MediaError(f"{what} produced no output: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return dest


def _even(value, what: str = "height") -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise MediaError(f"bad {what}: {value!r}") from None
    if n < 2:
        raise MediaError(f"bad {what}: {value!r}")
    return n - n % 2


def _num(value, digits: int | None = None) -> str:
    if digits is not None:
        return f"{float(value):.{digits}f}"
    return f"{float(value):g}"


# Width follows the DISPLAY aspect ratio (dar), so an anamorphic source isn't
# squashed once ``setsar=1`` makes its pixels square.
def _scale_to_height(height: int) -> str:
    return f"scale=w='trunc({height}*dar/2)*2':h={height},setsar=1"


def _scale_down_to_height(height: int) -> str:
    """Like ``_scale_to_height`` but never upscales a smaller source."""
    return (f"scale=w='trunc(oh*dar/2)*2':h='trunc(min({height},ih)/2)*2',setsar=1")


def _scale_still(width: int) -> str:
    """Stills ``width`` wide (never upscaled), full-range for the JPEG encoder."""
    return (f"scale=w='trunc(min({width},iw)/2)*2':h='trunc(ow/dar/2)*2':out_range=full,"
            "setsar=1,format=yuvj420p")


# ---- proxies and renders ---------------------------------------------------

def perception_proxy(source, *, height: int = 360, fps: int = 10,
                     start: float | None = None, end: float | None = None) -> Path:
    """A small, fast-to-upload copy of the source for a video model to watch.

    With ``start``/``end`` the proxy covers ``[start, end)`` and its clock
    starts at 0 there (input-side seek); callers add ``start`` back to any time
    read off it. Cached under ``engine_dir()/proxies``.
    """
    src = _existing(source)
    h = _even(height)
    try:
        rate = float(fps)
    except (TypeError, ValueError):
        raise MediaError(f"bad fps: {fps!r}") from None
    if not math.isfinite(rate) or rate <= 0:
        raise MediaError(f"bad fps: {fps!r}")
    lo = 0.0 if start is None else max(0.0, float(start))
    hi = None if end is None else float(end)
    if hi is not None and hi - lo <= 0:
        raise MediaError(f"empty proxy window {lo:.2f}-{hi:.2f}")
    name = (f"src-{file_key(src)}-{h}-{_num(rate)}-{_num(lo, 2)}-"
            f"{'end' if hi is None else _num(hi, 2)}-v{VERSION}.mp4")
    dest = engine_dir() / "proxies" / name
    with _lock_for(dest):
        if _reuse(dest):
            return dest
        span = (hi - lo) if hi is not None else probe_duration(src)
        seek = ["-ss", f"{lo:.3f}"] if lo > 0 else []
        if hi is not None:
            seek += ["-to", f"{hi:.3f}"]
        started = time.monotonic()
        _encode([*seek, "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                 "-vf", f"{_scale_down_to_height(h)},fps={_num(rate)}:start_time=0:round=down",
                 *_ENCODE_VIDEO, "-crf", "30",
                 "-c:a", "aac", "-b:a", "64k", "-ac", "1",
                 *_FASTSTART, "-f", "mp4"],
                dest, timeout=timeout_for(span), what=f"proxy of {src.name}")
        log.info("Proxy %s (%s) in %.1fs", dest.name, src.name, time.monotonic() - started)
    return dest


def _render_spans(segments, ndigits: int | None = None) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for seg in segments or []:
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            raise MediaError(f"bad segment {seg!r}") from None
        if not (math.isfinite(start) and math.isfinite(end)):
            raise MediaError(f"bad segment {seg!r}")
        start = max(0.0, start)
        if ndigits is not None:
            start, end = round(start, ndigits), round(end, ndigits)
        if end - start < _MIN_SEGMENT:
            raise MediaError(f"segment {start:.2f}-{end:.2f}s is shorter than {_MIN_SEGMENT}s")
        spans.append((start, end))
    if not spans:
        raise MediaError("no segments to render")
    return spans


def render_segments(source, segments, dest, *, height: int = 360, crf: int = 28,
                    fade: float = 0.03) -> Path:
    """Render ``segments`` IN LIST ORDER (play order) into one mp4 at ``dest``.

    One ffmpeg process: each segment is its own input-seeked input, scaled to
    ``height`` at 25 fps, its audio resampled to 48 kHz stereo with a short
    fade at both ends, all joined by a concat filter. A source without audio
    gets silent audio, so every render has the same stream layout. Segments
    running past the end of the source are clamped to it.
    """
    src = _existing(source)
    spans = _render_spans(segments)
    h = _even(height)
    info = _probe(src)
    if not info["video"]:
        raise MediaError(f"{src.name} has no video stream")
    if info["duration"]:
        clamped = []
        for start, end in spans:
            end = min(end, info["duration"])
            if end - start < _MIN_SEGMENT:
                raise MediaError(f"segment {start:.2f}-{end:.2f}s runs past the end of "
                                 f"{src.name} ({info['duration']:.2f}s)")
            clamped.append((start, end))
        spans = clamped
    fade = max(0.0, float(fade or 0.0))

    inputs: list[str] = []
    chains: list[str] = []
    for k, (start, end) in enumerate(spans):
        length = end - start
        inputs += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src)]
        chains.append(f"[{k}:v:0]{_scale_to_height(h)},{_RENDER_FPS}[v{k}]")
        if info["audio"]:
            audio = "aresample=48000,aformat=channel_layouts=stereo"
            f = min(fade, length / 2)
            if f > 0:
                audio += (f",afade=t=in:d={f:.3f}"
                          f",afade=t=out:st={length - f:.3f}:d={f:.3f}")
            chains.append(f"[{k}:a:0]{audio}[a{k}]")
        else:
            chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={length:.3f}[a{k}]")
    joins = "".join(f"[v{k}][a{k}]" for k in range(len(spans)))
    graph = ";".join([*chains, f"{joins}concat=n={len(spans)}:v=1:a=1[v][a]"])
    total = sum(end - start for start, end in spans)
    started = time.monotonic()
    out = _encode([*inputs, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                   *_ENCODE_VIDEO, "-crf", str(int(crf)),
                   "-c:a", "aac", "-b:a", "96k", *_FASTSTART, "-f", "mp4"],
                  Path(dest), timeout=timeout_for(total),
                  what=f"render of {len(spans)} segment(s) from {src.name}")
    log.info("Rendered %s: %d segment(s), %.1fs of clip in %.1fs", Path(dest).name,
             len(spans), total, time.monotonic() - started)
    return out


def _safe_name(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text or "")).strip("-")[:60]
    return cleaned or fallback


def render_cached(source, segments, *, height: int = 360, kind: str = "draft") -> Path:
    """``render_segments`` into ``engine_dir()/renders``, keyed by the source,
    the segments (2 dp, play order), the height and VERSION — the same draft
    asked for twice (a revision that changed nothing, a re-run over a cached
    shot log) renders once."""
    src = _existing(source)
    spans = _render_spans(segments, ndigits=2)
    h = _even(height)
    segs = [{"start": start, "end": end} for start, end in spans]
    digest = util.sha1(file_key(src), json.dumps(segs), h, VERSION, n=20)
    dest = engine_dir() / "renders" / f"{_safe_name(kind, 'draft')}-{digest}.mp4"
    with _lock_for(dest):
        if _reuse(dest):
            return dest
        render_segments(src, segs, dest, height=h)
    return dest


def trim_tail(path, seconds: float, dest, *, height: int = 360) -> Path:
    """Re-encode ``path`` at the render format (``height``, 25 fps) without its
    last ``seconds`` — e.g. a published clip's appended outro — into ``dest``."""
    src = _existing(path)
    h = _even(height)
    try:
        cut = max(0.0, float(seconds or 0.0))
    except (TypeError, ValueError):
        raise MediaError(f"bad trim: {seconds!r}") from None
    info = _probe(src)
    if not info["video"]:
        raise MediaError(f"{src.name} has no video stream")
    keep = info["duration"]
    if cut > 0:
        if not keep:
            raise MediaError(f"can't trim the tail of {src.name}: its duration is unknown")
        keep -= cut
        if keep < _MIN_SEGMENT:
            raise MediaError(f"trimming {cut:.2f}s leaves nothing of {src.name}")
    length = ["-t", f"{keep:.3f}"] if cut > 0 else []
    return _encode([*length, "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", f"{_scale_to_height(h)},{_RENDER_FPS}",
                    *_ENCODE_VIDEO, "-crf", "28", "-c:a", "aac", "-b:a", "96k",
                    *_FASTSTART, "-f", "mp4"],
                   Path(dest), timeout=timeout_for(keep), what=f"trim of {src.name}")


# ---- shot maps -------------------------------------------------------------

_PTS_TIME = re.compile(r"pts_time:\s*(\S+)")


def _metadata_values(text: str, key: str) -> list[tuple[float, float]]:
    """``[(pts_time, value)]`` from the metadata/ametadata filters' print
    output: a ``frame:… pts_time:T`` line followed by ``key=value`` lines."""
    out: list[tuple[float, float]] = []
    t: float | None = None
    prefix = key + "="
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            match = _PTS_TIME.search(line)
            try:
                t = float(match.group(1)) if match else None
            except ValueError:
                t = None
        elif t is not None and line.startswith(prefix):
            try:
                out.append((t, float(line[len(prefix):])))
            except ValueError:
                continue
    return out


def _scene_scores(src: Path, threshold: float, timeout: float) -> list[tuple[float, float]]:
    """``[(t, score)]`` for every decoded frame (scdet's 0-100 change score)."""
    proc = _ffmpeg(["-i", str(src), "-an", "-map", "0:v:0", "-vf",
                    f"scale=320:-2,scdet=t={threshold:g},"
                    "metadata=mode=print:key=lavfi.scd.score:file=-", "-f", "null", "-"],
                   timeout=timeout, what=f"scene detection on {src.name}")
    return _metadata_values(proc.stdout.decode("utf-8", errors="replace"), "lavfi.scd.score")


def _rms_windows(src: Path, timeout: float) -> list[tuple[float, float]]:
    """``[(t, dB)]`` RMS level of each 50 ms window of the first audio stream.

    Mixed to mono by the resampler rather than ``pan=…c0+c1``: a 5.1 source
    keeps its dialogue (centre channel) and a mono one needs no special case.
    The pause threshold is relative to the median, so the mix gain is moot.
    """
    proc = _ffmpeg(["-i", str(src), "-vn", "-map", "0:a:0", "-af",
                    f"aresample={_RMS_RATE},aformat=channel_layouts=mono,"
                    f"asetnsamples=n={_RMS_WINDOW_SAMPLES}:p=0,"
                    "astats=metadata=1:reset=1:measure_overall=RMS_level:measure_perchannel=none,"
                    "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
                    "-f", "null", "-"],
                   timeout=timeout, what=f"loudness scan of {src.name}")
    return _metadata_values(proc.stdout.decode("utf-8", errors="replace"),
                            "lavfi.astats.Overall.RMS_level")


def _quiet_runs(windows: list[tuple[float, float]], *, window: float = 0.05,
               below_median_db: float = _QUIET_BELOW_MEDIAN_DB,
               min_seconds: float = _QUIET_MIN_SECONDS,
               duration: float | None = None) -> list[list[float]]:
    """Runs of consecutive RMS windows quieter than the median by
    ``below_median_db`` (a silent window, -inf, always counts) lasting at
    least ``min_seconds``, as ``[[start, end]]``."""
    levels = [db for _, db in windows if math.isfinite(db)]
    threshold = (statistics.median(levels) - below_median_db) if levels else math.inf
    runs: list[list[float]] = []
    run_start: float | None = None
    run_end = 0.0

    def close() -> None:
        if run_start is not None and run_end - run_start >= min_seconds - 1e-6:
            end = run_end if duration is None else min(run_end, duration)
            runs.append([round(run_start, 3), round(end, 3)])

    for t, db in sorted(windows):
        quiet = not math.isfinite(db) or db < threshold
        if quiet and run_start is not None and t > run_end + window / 2:
            close()             # a hole in the timestamps ends the run
            run_start = None
        if quiet:
            if run_start is None:
                run_start = t
            run_end = t + window
        else:
            close()
            run_start = None
    close()
    return runs


def _shot_map_error(duration: float | None, error: str) -> dict:
    return {"version": VERSION, "duration": round(duration or 0.0, 3), "cuts": [],
            "quiet": [], "error": error}


def _build_shot_map(src: Path, threshold: float) -> dict:
    info = _probe(src)
    duration = info["duration"]
    timeout = timeout_for(duration)
    errors: list[str] = []
    cuts: list[dict] = []
    try:
        scores = _scene_scores(src, threshold, timeout)
        if not scores:
            raise MediaError("no video frames decoded")
        cuts = [{"t": round(t, 3), "score": round(score, 2)}
                for t, score in sorted(scores) if score >= threshold and t > 0]
        if not duration:
            duration = max(t for t, _ in scores)
    except MediaError as exc:
        errors.append(f"cuts: {exc}")
    quiet: list[list[float]] = []
    if info["audio"]:
        try:
            windows = _rms_windows(src, timeout)
            if not windows:
                raise MediaError("no audio decoded")
            quiet = _quiet_runs(windows, duration=duration)
        except MediaError as exc:
            errors.append(f"quiet: {exc}")
    return {"version": VERSION, "duration": round(duration or 0.0, 3), "cuts": cuts,
            "quiet": quiet, "error": "; ".join(errors)}


def _usable_map(data, threshold: float) -> bool:
    """A cached map is reused only when it is complete and was made with the
    same threshold; anything else (old version, corrupt file) is rebuilt."""
    if not (isinstance(data, dict) and data.get("version") == VERSION
            and not data.get("error") and isinstance(data.get("cuts"), list)
            and isinstance(data.get("quiet"), list)):
        return False
    try:
        return float(data.get("threshold", threshold)) == threshold
    except (TypeError, ValueError):
        return False


def shot_map(source, *, scene_threshold: float = 8.0) -> dict:
    """Shot changes and pauses in the source:
    ``{"version", "duration", "cuts": [{"t", "score"}], "quiet": [[s, e]], "error"}``.

    Every frame's scdet score is read and cuts scoring ``scene_threshold`` or
    more are kept with their score (news b-roll cuts every few seconds, so the
    snapper decides how strong a cut must be). ``quiet`` are runs of 50 ms
    windows more than 12 dB under the median loudness, ≥ 0.15 s long.

    Never raises: a failure comes back with ``error`` set (and whatever part
    did work), and is not cached, so the next run tries again.
    """
    try:
        threshold = float(scene_threshold)
        src = _existing(source)
        key = file_key(src)
    except (MediaError, TypeError, ValueError) as exc:
        return _shot_map_error(None, str(exc))
    suffix = "" if threshold == 8.0 else f"-t{threshold:g}"
    path = engine_dir() / "shotmaps" / f"{key}{suffix}-v{VERSION}.json"
    try:
        with _lock_for(path):
            cached = util.read_json(path)
            if _usable_map(cached, threshold):
                util.touch(path)
                return cached
            started = time.monotonic()
            result = _build_shot_map(src, threshold)
            result["threshold"] = threshold
            if result["error"]:
                log.warning("Shot map for %s incomplete (not cached): %s",
                            src.name, result["error"])
                return result
            try:
                util.atomic_write_json(path, result)
            except OSError as exc:
                log.warning("Couldn't cache the shot map for %s: %s", src.name, exc)
            log.info("Shot map for %s: %d cuts, %d pauses in %.1fs", src.name,
                     len(result["cuts"]), len(result["quiet"]), time.monotonic() - started)
            return result
    except Exception as exc:  # noqa: BLE001 - the contract is "never raises"
        log.warning("Shot map for %s failed: %s", source, exc)
        return _shot_map_error(None, str(exc) or exc.__class__.__name__)


# ---- frames ----------------------------------------------------------------

def _frames_chunk(src: Path, start: float, length: float, rate: Fraction,
                  width: int) -> list[tuple[float, bytes]]:
    """Frames every ``1/rate`` s from ``[start, start + length)`` as
    ``(source time, jpeg)``."""
    with tempfile.TemporaryDirectory(prefix="clip-engine-frames-") as tmp:
        _ffmpeg(["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
                 "-map", "0:v:0", "-vf", f"fps={rate.numerator}/{rate.denominator}"
                 f":start_time=0:round=up,{_scale_still(width)}", "-q:v", "3",
                 str(Path(tmp) / "f_%06d.jpg")],
                timeout=timeout_for(length), what=f"frames from {src.name}")
        period = 1 / rate
        frames: list[tuple[float, bytes]] = []
        for image in sorted(Path(tmp).glob("f_*.jpg")):
            try:
                index = int(image.stem.split("_")[1]) - 1
            except (IndexError, ValueError):
                continue
            offset = float(index * period)
            if offset >= length - 1e-6:
                continue    # the fps filter can round one frame past the end
            data = image.read_bytes()
            if data:
                frames.append((round(start + offset, 2), data))
        return frames


def sample_frames(path, *, start: float, end: float, interval: float,
                  width: int) -> list[tuple[float, bytes]]:
    """Stills every ``interval`` seconds from ``[start, end)`` as
    ``(source seconds, jpeg)``, ``width`` px wide.

    The span is read in chunks of at most 300 s, each with its own timeout,
    on one grid so timestamps continue across chunks. Raises MediaError when
    a chunk fails, or when a non-empty span yields no frames at all — a model
    given zero frames would describe nothing and look like it had.
    """
    src = _existing(path)
    try:
        step = float(interval)
        lo, hi = max(0.0, float(start)), float(end)
    except (TypeError, ValueError):
        raise MediaError(f"bad frame span {start!r}-{end!r} / {interval!r}") from None
    if not (math.isfinite(step) and step > 0 and math.isfinite(lo)):
        raise MediaError(f"bad frame interval {interval!r}")
    w = _even(width, "width")
    if not hi > lo:
        return []
    duration = probe_duration(src)
    if duration:
        hi = min(hi, duration)
    # The fps filter takes an exact rational rate; frame k is labelled
    # start + k * period with the same rational, so labels never drift.
    period_q = Fraction(step).limit_denominator(1000)
    if period_q <= 0:
        raise MediaError(f"bad frame interval {interval!r}")
    rate = 1 / period_q
    period = float(period_q)
    per_chunk = max(1, int(_FRAME_CHUNK_SECONDS // period))
    frames: list[tuple[float, bytes]] = []
    chunk = 0
    while True:
        chunk_start = lo + chunk * per_chunk * period
        if chunk_start >= hi - 1e-6:
            break
        chunk_end = min(hi, chunk_start + per_chunk * period)
        got = _frames_chunk(src, chunk_start, chunk_end - chunk_start, rate, w)
        if not got:
            log.info("No frames from %s %.1f-%.1fs", src.name, chunk_start, chunk_end)
        frames.extend(got)
        chunk += 1
    if not frames:
        raise MediaError(f"no frames from {src.name} between {lo:.1f}s and {float(end):.1f}s")
    return frames


def clip_frames(path, *, interval: float, width: int,
                max_frames: int) -> list[tuple[float, bytes]]:
    """Stills of a whole rendered clip, ``(clip seconds, jpeg)``: every
    ``interval`` seconds, spread wider when that would exceed ``max_frames``."""
    try:
        cap = int(max_frames)
    except (TypeError, ValueError):
        raise MediaError(f"bad max_frames: {max_frames!r}") from None
    if cap <= 0:
        return []
    src = _existing(path)
    duration = probe_duration(src)
    if not duration:
        raise MediaError(f"can't read the duration of {src.name}")
    step = max(float(interval), duration / cap)
    return sample_frames(src, start=0.0, end=duration, interval=step, width=width)[:cap]


# ---- in-flight source pins -------------------------------------------------
#
# A run reads its source for minutes. Meanwhile the offload sweep may delete
# it, or a re-archive may overwrite it. A hard link under engine_dir()/inflight
# keeps those bytes alive (and their path stable, so cache keys hold) for the
# whole run. The dashboard and a bench process can pin the same video at once,
# so every pin is registered with a marker file naming its process, and the
# link goes only when the last live holder releases it.

_held: dict[str, int] = {}
_held_guard = threading.Lock()


def _inflight_dir() -> Path:
    path = engine_dir() / "inflight"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def _inflight_lock(directory: Path):
    """Serialize pin bookkeeping across this process's threads and, through
    ``flock``, across processes sharing the directory."""
    with _held_guard:
        with open(directory / ".lock", "a+b") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass   # no cross-process lock here; the thread lock still holds
            yield


def _marker(link: Path, pid: int | None = None) -> Path:
    return link.with_name(f".{link.name}.{os.getpid() if pid is None else pid}.hold")


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return True   # os.kill(pid, 0) would terminate it on Windows
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True   # exists, owned by someone else
    return True


def _live_holders(link: Path, now: float) -> int:
    """Holders of ``link`` still running; stale markers are cleared. This
    process knows its own pins exactly; another process's marker counts while
    that process is alive and the marker is under a day old (no run lasts a
    day, so an older one survived a crash, or its pid was reused)."""
    live = 0
    for marker in link.parent.glob(f".{glob.escape(link.name)}.*.hold"):
        try:
            pid = int(marker.name[len(link.name) + 2:-len(".hold")])
        except ValueError:
            continue
        if pid == os.getpid():
            alive = _held.get(str(link), 0) > 0
        else:
            try:
                fresh = now - marker.stat().st_mtime < _STALE_SECONDS
            except OSError:
                continue
            alive = fresh and _pid_alive(pid)
        if alive:
            live += 1
        else:
            with contextlib.suppress(OSError):
                marker.unlink()
    return live


def _same_file(link: Path, st: os.stat_result) -> bool:
    try:
        other = link.stat()
    except OSError:
        return False
    if (other.st_dev, other.st_ino) == (st.st_dev, st.st_ino):
        return True
    return other.st_size == st.st_size and other.st_mtime_ns == st.st_mtime_ns


def _hold(link: Path) -> None:
    count = _held.get(str(link), 0)
    if count == 0:
        _marker(link).write_text(util.utcnow_iso())
    _held[str(link)] = count + 1


def link_inflight(source, key: str) -> Path:
    """Pin ``source`` for the length of a run: a hard link (a copy when the
    filesystem can't link) at ``engine_dir()/inflight/<key>-<identity><ext>``.
    Run ffmpeg on the returned path and ``release_inflight`` it when done."""
    src = _existing(source)
    try:
        st = src.stat()
        directory = _inflight_dir()
        identity = util.sha1(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, n=8)
        link = directory / f"{_safe_name(key, 'source')}-{identity}{src.suffix.lower() or '.mp4'}"
        with _inflight_lock(directory):
            if _same_file(link, st):
                _hold(link)
                return link
        # Made outside the lock: a copy of a multi-GB source takes a while.
        tmp = _tmp_for(link)
        try:
            try:
                os.link(src, tmp)
                how = "linked"
            except OSError:
                shutil.copy2(src, tmp)   # keeps the mtime, so cache keys still match
                how = "copied"
            with _inflight_lock(directory):
                if not _same_file(link, st):
                    os.replace(tmp, link)
                _hold(link)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
    except OSError as exc:
        raise MediaError(f"couldn't pin {src.name} for this run: {exc}") from exc
    log.info("Pinned %s as %s (%s)", src.name, link.name, how)
    return link


def release_inflight(path) -> None:
    """Drop this run's pin; the link goes once no live run holds it. Never
    raises, and ignores anything outside the inflight directory."""
    if not path:
        return
    try:
        link = Path(path)
        directory = _inflight_dir()
        if link.parent.resolve() != directory.resolve():
            log.warning("Not releasing %s: it isn't an in-flight pin", link)
            return
        with _inflight_lock(directory):
            count = _held.get(str(link), 0)
            if count > 1:
                _held[str(link)] = count - 1
                return
            _held.pop(str(link), None)
            with contextlib.suppress(OSError):
                _marker(link).unlink()
            if not _live_holders(link, time.time()):
                with contextlib.suppress(FileNotFoundError):
                    link.unlink()
    except Exception as exc:  # noqa: BLE001 - best effort by contract
        log.warning("Couldn't release %s: %s", path, exc)


def _prune_inflight(now: float) -> int:
    removed = 0
    directory = _inflight_dir()
    with _inflight_lock(directory):
        for entry in directory.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                if not entry.is_file() or _live_holders(entry, now):
                    continue
                # A hard link shares the source's mtime (often months old);
                # its ctime moved when the link was made.
                if now - entry.stat().st_ctime > _STALE_SECONDS:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
        for leftover in directory.glob(".*.tmp*"):
            with contextlib.suppress(OSError):
                if now - leftover.stat().st_mtime > _STALE_SECONDS:
                    leftover.unlink()
                    removed += 1
        # Markers whose pin is already gone (a crash between unlinks).
        for marker in directory.glob(".*.hold"):
            name = marker.name[1:].rsplit(".", 2)[0]
            if not (directory / name).exists():
                with contextlib.suppress(OSError):
                    marker.unlink()
    return removed


# ---- housekeeping ----------------------------------------------------------

def drop(paths) -> None:
    """Best-effort delete of files (a run's unchosen renders). Accepts one
    path or many; skips blanks, missing files and directories. Never raises."""
    if paths is None:
        return
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    try:
        items = list(paths)
    except TypeError:
        return
    for item in items:
        if not item:
            continue
        try:
            path = Path(item)
            if path.is_file() or path.is_symlink():
                path.unlink()
        except Exception as exc:  # noqa: BLE001 - best effort by contract
            log.info("Couldn't delete %s: %s", item, exc)


def prune(max_age_days: int = 14) -> int:
    """Delete cached proxies, renders, shot maps and taste clips not used in
    ``max_age_days`` (reuse refreshes a file's mtime), abandoned tmp files,
    and in-flight pins no live run holds that are over a day old. Returns how
    many files went. Never raises."""
    now = time.time()
    try:
        days = max(0.0, float(max_age_days))
    except (TypeError, ValueError):
        days = 14.0
    cutoff = now - days * 86400.0
    removed = 0
    try:
        root = engine_dir()
        for sub in ("proxies", "renders", "shotmaps", "taste"):
            directory = root / sub
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                try:
                    if not entry.is_file():
                        continue
                    mtime = entry.stat().st_mtime
                    abandoned = (entry.name.startswith(".") and ".tmp" in entry.name
                                 and now - mtime > _STALE_SECONDS)
                    if mtime < cutoff or abandoned:
                        entry.unlink()
                        removed += 1
                except OSError:
                    continue
        removed += _prune_inflight(now)
    except Exception as exc:  # noqa: BLE001 - housekeeping must never break a run
        log.warning("Pruning the clip engine cache stopped early: %s", exc)
    if removed:
        log.info("Pruned %d clip engine file(s)", removed)
    return removed


def disk_free_gb(path) -> float:
    """Free space, in GiB, on the volume holding ``path`` (or its nearest
    existing parent). 0.0 when it can't be measured, so a guard built on it
    errs toward stopping rather than filling the disk."""
    p = Path(os.path.abspath(os.path.expanduser(str(path))))
    for candidate in (p, *p.parents):
        try:
            return shutil.disk_usage(candidate).free / float(1 << 30)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            log.warning("Couldn't measure free disk at %s: %s", candidate, exc)
            return 0.0
    return 0.0
