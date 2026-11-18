"""The shot log: a time-coded account of what a source video shows and says.

An editor that only reads a transcript can't see the picture, so it opens
clips on anchors and b-roll and misses the moment that carries a story. The
shot log is what a video model saw and heard, beat by beat — the shot type,
what is literally on screen (with the time of every cut inside the beat), who
speaks and whether their face is on camera, the line, the sound, and how
strong the beat is — so the editors that can't watch the video (the hybrid
and claude setups, and sources too long to attach) still cut from the
picture, and so every judge can be told who is on screen (``credits_for``).

Logging costs a dozen model calls for a long source, so:

- Whole-video logs are cached per (video, perceiver, model, prompt version,
  perception settings), and a section pass reuses the whole-video log
  filtered to its window. The settings are part of the key so a log made at
  one frame rate, resolution, window size or effort is never reused for a run
  that asks for another (a bench arm at 2 fps would otherwise measure the
  1 fps log). A reused log's stored cost is reported as ``perceive_cached``
  so the bench sees what a run really cost.
- Long sources are logged in overlapping windows. Each finished window is
  cached the moment it completes, so a restart, or one window that fails,
  never pays for the others again.
- A window whose answer runs past the output limit (fast-cut news yields
  hundreds of beats) is split in half and retried, two levels deep, instead
  of failing the whole run — unless the answer is far longer than any real
  log of that stretch, which is a model repeating itself: splitting would
  only pay for more of it.
- The caller's ``check`` (cancel, deadline, the run's cost cap) runs before
  every window and every split, so a run can't spend far past its cap
  inside this one stage.
- A window is complete only when its beats cover it: an answer that stops
  early is kept for this run but never cached, so the hole isn't permanent.
- Nothing incomplete or empty is ever cached as the whole-video log; a log
  with a failed window is still returned (flagged ``complete: False``) so one
  bad window doesn't throw the run away.

Times in a log are float seconds on the SOURCE clock, whatever clock the model
answered in: MM:SS positions within a window proxy (whose clock starts at 0),
or seconds from the frame labels.

Every beat also says what language its words are in (``speech_language``:
english / other / none) and whether an English voiceover is heard over it
(``english_voiceover``). This audience only follows English speech — a
non-English interview with burned-in subtitles is a clip the operator never
posts — so the editor needs to see it coming and the engine can drop a draft
the critic couldn't follow when the log agrees its speech is foreign
(version 3; older logs are rebuilt because the version is in the cache key).
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Callable

from . import claude, engine_dir, gemini, media, model_for, opt, prompts, util

log = logging.getLogger("clip_engine.shotlog")

VERSION = 3

FACES = ("none", "small", "close")
HIGHLIGHT_KINDS = ("visual", "line", "sound", "person")
# How a person's name is known. "none" = described, not named (a face alone
# never names anyone), so they never reach the credits.
EVIDENCE = ("title", "description", "on_screen", "said", "none")

# prompt_hash() fingerprints the prompt TEMPLATE: cached logs should go stale
# when the instructions change, not when the brand's wording does.
_HASH_BRAND = {"name": "{name}", "topic": "{topic}"}

_PART_OVERLAP = 10.0            # seconds neighbouring windows share
_MAX_SPLIT_LEVEL = 2            # a truncated window is halved at most twice...
_MIN_SPLIT_SECONDS = 20.0       # ...and never into halves shorter than this
# A real log runs well under 150 output tokens per second of video, even for
# fast-cut news with thinking on; a truncated answer at twice that is a model
# repeating itself, and its halves would only loop again.
_LOOP_TOKENS_PER_SECOND = 300.0
_MIN_COVERAGE = 0.85            # share of its span a window's beats must cover
_LONG_SOURCE_SECONDS = 1200.0   # past this Gemini samples at long_video_fps
_PART_WORKERS = 3               # windows logged at once
_CUT_LEAD = 0.5                 # a cut this close before a beat starts belongs to it
_CREDIT_PROMINENCE = ("famous", "public_figure", "expert")
_GEMINI_MAX_OUTPUT = 65536      # output tokens (thinking included) for one window
_CLAUDE_MAX_TOKENS = 32000      # ...for one frames chunk
_CHUNK_SECONDS = 300.0          # a frames chunk spans at most this, however sparse
_STRETCH_CHARS = 16000          # transcript text sent with one frames chunk

_SHOT_ALIASES = {
    "aerial_shot": "aerial", "drone": "aerial", "wide": "wide_scene", "establishing": "wide_scene",
    "wide_shot": "wide_scene", "close_up": "action_closeup", "closeup": "action_closeup",
    "broll": "b_roll", "interview_shot": "interview", "talking": "talking_head",
    "anchor": "anchor_desk", "presenter": "anchor_desk", "studio": "anchor_desk",
    "podium": "press_podium", "press_conference": "press_podium", "hearing": "press_podium",
    "protest": "crowd", "rally": "crowd", "chart": "graphic", "map": "graphic",
    "animation": "graphic", "archive": "archival", "slate": "text_card", "title_card": "text_card",
    "logo": "text_card",
}
_LANGUAGE_ALIASES = {"en": "english", "eng": "english", "english_language": "english",
                     "no_speech": "none", "silence": "none", "silent": "none", "n/a": "none",
                     "na": "none", "no": "none", "nobody": "none", "music": "none",
                     "natural_sound": "none"}
_KIND_ALIASES = {"image": "visual", "picture": "visual", "shot": "visual", "quote": "line",
                 "speech": "line", "audio": "sound", "people": "person", "face": "person"}

_TIME = {"type": "string",
         "description": "MM:SS or MM:SS.s (seconds such as \"75.5\" when logging from frames)"}

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "format": {"type": "string"},
        "structure": {"type": "string"},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "prominence": {"type": "string", "enum": list(prompts.PROMINENCE)},
                    "evidence": {"type": "string", "enum": list(EVIDENCE),
                                 "description": "how the name is known: title, description, "
                                                "on_screen, said, or none (not named)"},
                },
                "required": ["name", "role", "prominence", "evidence"],
            },
        },
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": _TIME,
                    "end": _TIME,
                    "shot": {"type": "string", "enum": list(prompts.SHOT_TYPES)},
                    "on_screen": {"type": "string"},
                    "face": {"type": "string", "enum": list(FACES)},
                    "speaker": {"type": "string"},
                    "line": {"type": "string"},
                    "sound": {"type": "string"},
                    "speech_language": {"type": "string", "enum": list(prompts.SPEECH_LANGUAGES),
                                        "description": "english, other (any other language), "
                                                       "or none (nobody speaks)"},
                    "english_voiceover": {"type": "boolean",
                                          "description": "an English voiceover or narration is "
                                                         "heard over this beat (subtitles don't "
                                                         "count)"},
                    "energy": {"type": "integer", "minimum": 1, "maximum": 5},
                    "visual": {"type": "integer", "minimum": 1, "maximum": 5},
                    "topic": {"type": "integer", "minimum": 0, "maximum": 2},
                    "note": {"type": "string"},
                },
                "required": ["start", "end", "shot", "on_screen", "face", "speaker", "line",
                             "sound", "speech_language", "english_voiceover", "energy",
                             "visual", "topic", "note"],
            },
        },
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": _TIME,
                    "end": _TIME,
                    "kind": {"type": "string", "enum": list(HIGHLIGHT_KINDS)},
                    "what": {"type": "string"},
                },
                "required": ["start", "end", "kind", "what"],
            },
        },
    },
    "required": ["summary", "format", "structure", "people", "beats", "highlights"],
}


class ShotLogError(RuntimeError):
    """No usable log could be produced (unknown duration, every part failed)."""


# ---- small coercions ----------------------------------------------------------------

def _mode(perceiver: str) -> str:
    return "video" if perceiver == "gemini" else "frames"


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:limit].rstrip() if len(text) > limit else text


def _trim(value: Any, limit: int) -> str:
    """Like ``_text`` but marks the cut, for text a model will read."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _int(value: Any, lo: int, hi: int, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(lo, min(hi, number))


def _key(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())


def _choice(value: Any, allowed: tuple, default: str, aliases: dict | None = None) -> str:
    key = _key(value)
    if key in allowed:
        return key
    if aliases and key in aliases:
        return aliases[key]
    return default


def _evidence(value: Any) -> str:
    key = _key(value)
    if key in EVIDENCE:
        return key
    if "title" in key:
        return "title"
    if "descr" in key:
        return "description"
    if any(w in key for w in ("screen", "lower", "caption", "shown", "graphic", "chyron")):
        return "on_screen"
    if any(w in key for w in ("said", "spoke", "aloud", "introduc", "named")):
        return "said"
    return "none"


def _language(value: Any, *, line: str = "") -> str:
    """A beat's speech language: "english", "other" or "none". A named
    language other than English ("Hindi", "mixed") is "other". A missing
    answer is never read as "other" (that drops drafts): "english" when the
    beat has a line, else "none"."""
    key = _key(value)
    if key in prompts.SPEECH_LANGUAGES:
        return key
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    if not key or key in ("unknown", "null"):
        return "english" if line else "none"
    if key.startswith("english") or key.startswith("en_"):
        return "english"
    return "other"


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in ("true", "yes", "y", "1")


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _float_setting(settings, key: str, default: float) -> float:
    try:
        return float(opt(settings, key, default))
    except (TypeError, ValueError):
        return float(default)


def _int_setting(settings, key: str, default: int) -> int:
    try:
        return int(float(opt(settings, key, default)))
    except (TypeError, ValueError):
        return int(default)


def _bounds(window: Any) -> tuple[float, float] | None:
    """``(start, end)`` from a ``{"start", "end"}`` dict or a pair; None when
    absent or empty."""
    if not window:
        return None
    try:
        if isinstance(window, dict):
            start, end = float(window["start"]), float(window["end"])
        else:
            start, end = float(window[0]), float(window[1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "x"


def _num_tag(value: float) -> str:
    return f"{round(float(value), 1):.1f}".rstrip("0").rstrip(".")


def _fmt_seconds(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") + "s"


def _bucket(meter) -> str | None:
    """The meter's spend bucket; None (no meter) lets the provider use the
    process default, else "engine" — never the shared LLM budget."""
    return getattr(meter, "bucket", None) or None


def _add_usage(meter, stage: str, usage: dict | None) -> None:
    if meter is not None and usage:
        meter.add(stage, usage)


def _add_cached(meter, log_or_part: dict) -> None:
    if meter is not None:
        meter.add_cached("perceive", float(log_or_part.get("cost_usd") or 0.0))


# ---- times ------------------------------------------------------------------------------

_CLOCK = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_ts(value) -> float | None:
    """Seconds from whatever a model wrote: 75.5, "75.5", "75.5s", "t=75.5s",
    "01:15", "01:15.5", "1:02:03". None for anything else (and for negative,
    infinite or boolean values)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    text = str(value).strip().lower().replace(",", ".")
    if text.startswith("t="):
        text = text[2:].strip()
    for suffix in ("seconds", "second", "secs", "sec", "s"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    if not text:
        return None
    match = _CLOCK.match(text)
    if match:
        hours, minutes, seconds = match.groups()
        number = int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        try:
            number = float(text)
        except ValueError:
            return None
    return number if math.isfinite(number) and number >= 0 else None


_CLOCK_TOKEN = r"(?:\d{1,2}:)?\d{1,3}:\d{2}(?:[.,]\d+)?"
# A shot list entry in on_screen: "00:12.5 aerial of the town; 00:15.0 a man ..."
_SHOT_TIME = re.compile(rf"(^|;\s*)({_CLOCK_TOKEN})(?=\s)")
_ANY_CLOCK = re.compile(rf"(?<![\d:.])({_CLOCK_TOKEN})(?![\d:])")


def _absolute_shot_times(text: str, offset: float) -> str:
    """Rewrite the MM:SS shot times inside an on_screen shot list as source
    seconds, so the editor reads one clock everywhere."""
    def repl(match: re.Match) -> str:
        seconds = parse_ts(match.group(2))
        if seconds is None:
            return match.group(0)
        return f"{match.group(1)}{_fmt_seconds(seconds + offset)}"
    return _SHOT_TIME.sub(repl, text)


def _absolute_clock_text(text: str, offset: float) -> str:
    def repl(match: re.Match) -> str:
        seconds = parse_ts(match.group(1))
        return match.group(0) if seconds is None else _fmt_seconds(seconds + offset)
    return _ANY_CLOCK.sub(repl, text)


def _span(start: Any, end: Any, offset: float, horizon: float | None, *,
          default_len: float | None = None, min_len: float = 0.05) -> tuple[float, float] | None:
    s = parse_ts(start)
    if s is None:
        return None
    e = parse_ts(end)
    if e is None or e <= s:
        if default_len is None:
            return None
        e = s + default_len
    s, e = max(0.0, s + offset), e + offset
    if horizon is not None:
        s, e = min(s, horizon), min(e, horizon)
    if e - s < min_len:
        return None
    return round(s, 2), round(e, 2)


# ---- normalizing and merging ---------------------------------------------------------------

def _better_person(a: dict, b: dict) -> dict:
    """The more prominent of two entries for one name, gaps filled from the other."""
    rank = prompts.PROMINENCE.index
    best, other = (a, b) if rank(a["prominence"]) <= rank(b["prominence"]) else (b, a)
    out = dict(best)
    if not out.get("role"):
        out["role"] = other.get("role", "")
    if out.get("evidence", "none") == "none" and other.get("evidence", "none") != "none":
        out["evidence"] = other["evidence"]
    return out


def _merge_people(items: list) -> list[dict]:
    """Unique people by lowercase name, keeping the most prominent entry."""
    by_name: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), 80)
        if not name:
            continue
        entry = {
            "name": name,
            "role": _text(item.get("role"), 120),
            "prominence": _choice(item.get("prominence"), prompts.PROMINENCE, "unknown"),
            "evidence": _evidence(item.get("evidence")),
        }
        key = name.lower()
        by_name[key] = _better_person(by_name[key], entry) if key in by_name else entry
    return list(by_name.values())


def prompt_hash(mode: str) -> str:
    """sha1 of the perceiver prompt template for ``mode`` ("video"/"frames"), 8 hex."""
    return _prompt_hash(str(mode))


@functools.lru_cache(maxsize=8)
def _prompt_hash(mode: str) -> str:
    text = prompts.perceiver_system(_HASH_BRAND, mode=mode)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def perception_settings(perceiver: str, settings=None, seconds: float | None = None) -> dict:
    """What decides how a perceiver sees a source and how hard it thinks.

    Gemini: the frame rate it samples (``long_video_fps`` past 1200 s, else
    ``perception_fps``), the media resolution, the thinking level, the window
    size and the output cap. Claude: the frame spacing, the frame budget, the
    frame width, frames per call, the effort and output cap, and the longest
    chunk and transcript a call gets (the spacing actually used follows from
    these and the source's length). Shot logs are cached per these values
    (``cache_path``) and record them. ``settings`` None means the defaults;
    ``seconds`` is the source's length (unknown counts as short)."""
    s = settings if settings is not None else {}
    if perceiver == "gemini":
        if (_positive(seconds) or 0.0) > _LONG_SOURCE_SECONDS:
            fps = _float_setting(s, "clip_engine.gemini.long_video_fps", 0.5)
        else:
            fps = _float_setting(s, "clip_engine.gemini.perception_fps", 1.0)
        return {
            "fps": fps,
            "media_resolution": str(opt(s, "clip_engine.gemini.media_resolution",
                                        "MEDIA_RESOLUTION_LOW")).strip(),
            "thinking": str(opt(s, "clip_engine.gemini.perceiver_thinking", "LOW")).strip(),
            "window_seconds": _float_setting(s, "clip_engine.perception_window_seconds", 300.0),
            "max_output_tokens": _GEMINI_MAX_OUTPUT,
        }
    return {
        "frame_interval": max(0.25, _float_setting(s, "clip_engine.claude.frame_interval", 2.0)),
        "max_total_frames": max(1, _int_setting(s, "clip_engine.claude.max_total_frames", 600)),
        "frame_width": _int_setting(s, "clip_engine.claude.frame_width", 448),
        "max_frames_per_call": max(1, _int_setting(s, "clip_engine.claude.max_frames_per_call", 150)),
        "effort": str(opt(s, "clip_engine.claude.perceiver_effort", "low")).strip().lower(),
        "max_tokens": _CLAUDE_MAX_TOKENS,
        "chunk_seconds": _CHUNK_SECONDS,
        "transcript_chars": _STRETCH_CHARS,
    }


def _settings_tag(knobs: dict) -> str:
    text = json.dumps(knobs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def normalize_log(raw, *, duration, perceiver, model, cost, offset=0.0) -> dict:
    """A model's answer as a clean ShotLog (spec §3).

    ``offset`` is added to every time (a window proxy's clock starts at 0);
    times are clamped to ``[0, duration]``; beats need start < end and come
    back sorted; scores are clamped (energy/visual 1-5, topic 0-2); unknown
    shot types become "other". A log with no beats is marked incomplete so
    it is never cached."""
    raw = raw if isinstance(raw, dict) else {}
    horizon = _positive(duration)
    offset = float(offset or 0.0)
    clock = perceiver != "claude"   # Gemini answers in MM:SS; frame logs in seconds
    beats: list[dict] = []
    for item in _list(raw.get("beats")):
        if not isinstance(item, dict):
            continue
        span = _span(item.get("start"), item.get("end"), offset, horizon)
        if span is None:
            continue
        on_screen = _text(item.get("on_screen"), 1000)
        if clock:
            on_screen = _absolute_shot_times(on_screen, offset)
        line = _text(item.get("line"), 500)
        beats.append({
            "start": span[0], "end": span[1],
            "shot": _choice(item.get("shot"), prompts.SHOT_TYPES, "other", _SHOT_ALIASES),
            "on_screen": on_screen,
            "face": _choice(item.get("face"), FACES, "none"),
            "speaker": _text(item.get("speaker"), 120),
            "line": line,
            "sound": _text(item.get("sound"), 200),
            "speech_language": _language(item.get("speech_language"), line=line),
            "english_voiceover": _flag(item.get("english_voiceover")),
            "energy": _int(item.get("energy"), 1, 5, 3),
            "visual": _int(item.get("visual"), 1, 5, 3),
            "topic": _int(item.get("topic"), 0, 2, 1),
            "note": _text(item.get("note"), 200),
        })
    beats.sort(key=lambda b: (b["start"], b["end"]))

    highlights: list[dict] = []
    for item in _list(raw.get("highlights")):
        if not isinstance(item, dict):
            continue
        span = _span(item.get("start"), item.get("end"), offset, horizon, default_len=2.0)
        if span is None:
            continue
        highlights.append({
            "start": span[0], "end": span[1],
            "kind": _choice(item.get("kind"), HIGHLIGHT_KINDS, "visual", _KIND_ALIASES),
            "what": _text(item.get("what"), 300),
        })
    highlights.sort(key=lambda h: (h["start"], h["end"]))

    structure = _text(raw.get("structure"), 1500)
    if clock:
        structure = _absolute_clock_text(structure, offset)
    return {
        "version": VERSION,
        "perceiver": perceiver,
        "model": model,
        "prompt_hash": prompt_hash(_mode(perceiver)),
        "source_seconds": round(horizon, 2) if horizon else None,
        "created_at": util.utcnow_iso(),
        "cost_usd": round(float(cost or 0.0), 6),
        "complete": bool(beats),
        "summary": _text(raw.get("summary"), 600),
        "format": _text(raw.get("format"), 160),
        "structure": structure,
        "people": _merge_people(_list(raw.get("people"))),
        "beats": beats,
        "highlights": highlights,
    }


def _part_span(offset: float, part: dict) -> tuple[float, float]:
    span = part.get("span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            return float(span[0]), float(span[1])
        except (TypeError, ValueError):
            pass
    ends = [b["end"] for b in part.get("beats") or []]
    return offset, max(ends) if ends else offset


def _cut_point(prev: tuple[float, float], nxt: tuple[float, float]) -> float:
    """Where one part hands over to the next: the middle of their overlap."""
    if nxt[0] < prev[1]:
        return (nxt[0] + prev[1]) / 2
    return nxt[0]


def merge_windows(parts: list[tuple[float, dict]]) -> dict:
    """One log from per-window logs (already on the source clock).

    In each overlap the earlier part's beats run to the overlap's midpoint and
    the later part's take over from there (beats straddling it are clipped).
    People are unioned by lowercase name (most prominent kept); structures are
    joined with "; "; summary and format come from the first part; the log is
    complete only when every part was."""
    ordered = sorted(((float(off), lg) for off, lg in parts or [] if isinstance(lg, dict)),
                     key=lambda p: p[0])
    if not ordered:
        return {"version": VERSION, "perceiver": "", "model": "", "prompt_hash": "",
                "source_seconds": None, "created_at": util.utcnow_iso(), "cost_usd": 0.0,
                "complete": False, "summary": "", "format": "", "structure": "",
                "people": [], "beats": [], "highlights": []}
    spans = [_part_span(off, lg) for off, lg in ordered]
    beats: list[dict] = []
    highlights: list[dict] = []
    people: list[dict] = []
    for i, (_off, part) in enumerate(ordered):
        lo = -math.inf if i == 0 else _cut_point(spans[i - 1], spans[i])
        hi = math.inf if i == len(ordered) - 1 else _cut_point(spans[i], spans[i + 1])
        for beat in part.get("beats") or []:
            start, end = max(beat["start"], lo), min(beat["end"], hi)
            if end - start >= 0.05:
                beats.append({**beat, "start": round(start, 2), "end": round(end, 2)})
        for item in part.get("highlights") or []:
            if lo <= item["start"] < hi:
                highlights.append(dict(item))
        people.extend(part.get("people") or [])
    first = ordered[0][1]
    seconds = [lg.get("source_seconds") for _, lg in ordered if lg.get("source_seconds")]
    beats.sort(key=lambda b: (b["start"], b["end"]))
    highlights.sort(key=lambda h: (h["start"], h["end"]))
    return {
        "version": VERSION,
        "perceiver": first.get("perceiver", ""),
        "model": first.get("model", ""),
        "prompt_hash": first.get("prompt_hash", ""),
        "source_seconds": max(seconds) if seconds else None,
        "created_at": util.utcnow_iso(),
        "cost_usd": round(sum(float(lg.get("cost_usd") or 0.0) for _, lg in ordered), 6),
        "complete": all(bool(lg.get("complete")) for _, lg in ordered),
        "summary": first.get("summary") or next(
            (lg.get("summary") for _, lg in ordered if lg.get("summary")), ""),
        "format": first.get("format") or next(
            (lg.get("format") for _, lg in ordered if lg.get("format")), ""),
        "structure": "; ".join(lg["structure"] for _, lg in ordered if lg.get("structure")),
        "people": _merge_people(people),
        "beats": beats,
        "highlights": highlights,
    }


def filter_to_window(log, window) -> dict:
    """The log restricted to beats and highlights overlapping ``window``.

    People stay when a beat inside the window names them, or when no beat
    anywhere does (where they appear is unknown, so they may be here). Someone
    named only by beats outside the window is left out: with those beats gone,
    ``credits_for`` would list them as "on screen at some point" in every clip
    of the section."""
    bounds = _bounds(window)
    if not log or bounds is None:
        return log
    start, end = bounds
    out = dict(log)
    every_beat = log.get("beats") or []
    out["beats"] = [dict(b) for b in every_beat if b["end"] > start and b["start"] < end]
    out["highlights"] = [dict(h) for h in log.get("highlights") or []
                         if h["end"] > start and h["start"] < end]
    out["people"] = [dict(p) for p in log.get("people") or []
                     if any(_names(b, p) for b in out["beats"])
                     or not any(_names(b, p) for b in every_beat)]
    out["window"] = {"start": start, "end": end}
    return out


# ---- the editor's view -------------------------------------------------------------------

def _cut_times(cuts, min_shot_score: float) -> list[float]:
    times: list[float] = []
    for cut in cuts or []:
        try:
            if isinstance(cut, dict):
                t, score = float(cut["t"]), float(cut.get("score", min_shot_score))
            else:
                t, score = float(cut), float(min_shot_score)
        except (KeyError, TypeError, ValueError):
            continue
        if score >= min_shot_score and math.isfinite(t):
            times.append(t)
    return sorted(times)


def _join(a: str, b: str, sep: str) -> str:
    if not a:
        return b
    if not b or b == a:
        return a
    return f"{a}{sep}{b}"


_LANGUAGE_RANK = {"other": 2, "english": 1, "none": 0}


def _merged_language(a: dict, b: dict) -> tuple[str | None, bool | None]:
    """Speech language and voiceover for two merged beats (None when neither
    beat says). Foreign speech wins (it is the thing the editor must see),
    and it counts as voiced over only when every foreign part was."""
    parts = [x for x in (a, b) if x.get("speech_language")]
    if not parts:
        return None, None
    language = max((x["speech_language"] for x in parts), key=lambda v: _LANGUAGE_RANK.get(v, 0))
    if language == "other":
        voiceover = all(bool(x.get("english_voiceover")) for x in parts
                        if x["speech_language"] == "other")
    else:
        voiceover = any(bool(x.get("english_voiceover")) for x in parts)
    return language, voiceover


def _merge_beats(a: dict, b: dict) -> dict:
    dur_a, dur_b = a["end"] - a["start"], b["end"] - b["start"]
    if a["shot"] == b["shot"]:
        shot = a["shot"]
    else:
        shot = a["shot"] if (a["visual"], dur_a) >= (b["visual"], dur_b) else b["shot"]
    speaker = a["speaker"] if a["speaker"] == b["speaker"] else _join(a["speaker"], b["speaker"], " / ")
    language, voiceover = _merged_language(a, b)
    merged = {
        "start": a["start"], "end": max(a["end"], b["end"]), "shot": shot,
        "on_screen": _join(a["on_screen"], b["on_screen"], "; "),
        "face": max(a["face"], b["face"], key=FACES.index),
        "speaker": speaker,
        "line": _join(a["line"], b["line"], " … "),
        "sound": _join(a["sound"], b["sound"], "; "),
        "energy": max(a["energy"], b["energy"]),
        "visual": max(a["visual"], b["visual"]),
        "topic": max(a["topic"], b["topic"]),
        "note": _join(a["note"], b["note"], "; "),
        "cuts": sorted(set(a["cuts"]) | set(b["cuts"])),
    }
    if language is not None:
        merged["speech_language"] = language
        merged["english_voiceover"] = voiceover
    return merged


def _fit_beats(beats: list[dict], cap: int) -> list[dict]:
    """At most ``cap`` beats: merge neighbours that share a shot type first
    (shortest pairs first), then any neighbours. Coarsens instead of dropping
    the end of a long video."""
    beats = list(beats)
    if cap <= 0:
        return beats
    for same_shot_only in (True, False):
        while len(beats) > cap:
            pairs = sorted(
                (beats[i + 1]["end"] - beats[i]["start"], i) for i in range(len(beats) - 1)
                if not same_shot_only or beats[i]["shot"] == beats[i + 1]["shot"])
            if not pairs:
                break
            need, taken, merge_at = len(beats) - cap, set(), set()
            for _, i in pairs:
                if len(merge_at) >= need:
                    break
                if i in taken or i + 1 in taken:
                    continue
                taken.update((i, i + 1))
                merge_at.add(i)
            out, i = [], 0
            while i < len(beats):
                if i in merge_at:
                    out.append(_merge_beats(beats[i], beats[i + 1]))
                    i += 2
                else:
                    out.append(beats[i])
                    i += 1
            beats = out
    return beats


def compact_for_prompt(log, *, window=None, max_beats=400, cuts=None, min_shot_score=20.0) -> dict:
    """The log as an editor reads it: times to 0.1 s, long strings trimmed
    (on_screen 240, line 200, sound 80, note 60), and each beat's exact shot
    changes from the shot map (``cuts``: times with score >= min_shot_score)
    so a picture-led segment can start exactly on a cut. More than
    ``max_beats`` beats are merged with their neighbours.

    ``speech_language`` is kept on every beat that has one;
    ``english_voiceover`` only where it matters (foreign speech, or a
    voiceover that is there), which saves tokens on long logs."""
    log = log or {}
    bounds = _bounds(window)
    times = _cut_times(cuts, float(min_shot_score))
    beats: list[dict] = []
    for b in log.get("beats") or []:
        if bounds and not (b["end"] > bounds[0] and b["start"] < bounds[1]):
            continue
        beats.append({
            "start": round(b["start"], 1), "end": round(b["end"], 1),
            "shot": b.get("shot", "other"),
            "on_screen": _trim(b.get("on_screen"), 240),
            "face": b.get("face", "none") if b.get("face") in FACES else "none",
            "speaker": _trim(b.get("speaker"), 120),
            "line": _trim(b.get("line"), 200),
            "sound": _trim(b.get("sound"), 80),
            "energy": b.get("energy", 3), "visual": b.get("visual", 3), "topic": b.get("topic", 1),
            "note": _trim(b.get("note"), 60),
            "cuts": [round(t, 2) for t in times if b["start"] - _CUT_LEAD <= t < b["end"]],
        })
        if b.get("speech_language") in prompts.SPEECH_LANGUAGES:
            beats[-1]["speech_language"] = b["speech_language"]
            beats[-1]["english_voiceover"] = bool(b.get("english_voiceover"))
    fitted = _fit_beats(beats, int(max_beats))
    for beat in fitted:
        beat["on_screen"] = _trim(beat["on_screen"], 240)
        beat["line"] = _trim(beat["line"], 200)
        beat["sound"] = _trim(beat["sound"], 80)
        beat["note"] = _trim(beat["note"], 60)
        for key in ("line", "sound", "note", "cuts"):
            if not beat[key]:
                del beat[key]   # absent = nothing there; saves tokens on long logs
        if ("english_voiceover" in beat and not beat["english_voiceover"]
                and beat.get("speech_language") != "other"):
            del beat["english_voiceover"]
    highlights = [
        {"start": round(h["start"], 1), "end": round(h["end"], 1), "kind": h.get("kind", "visual"),
         "what": _trim(h.get("what"), 200)}
        for h in log.get("highlights") or []
        if not bounds or (h["end"] > bounds[0] and h["start"] < bounds[1])
    ]
    out = {
        "summary": log.get("summary", ""),
        "format": log.get("format", ""),
        "structure": _trim(log.get("structure"), 800),
        "people": [{"name": p.get("name", ""), "role": p.get("role", ""),
                    "prominence": p.get("prominence", "unknown"),
                    "evidence": p.get("evidence", "none")}
                   for p in log.get("people") or []],
        "beats": fitted,
        "highlights": highlights,
    }
    if log and not log.get("complete", True):
        out["warning"] = ("Parts of this video could not be logged; where beats are missing, "
                          "rely on the transcript.")
    return out


# ---- credits ------------------------------------------------------------------------------

_EVIDENCE_PHRASES = {
    "title": "named in the title",
    "description": "named in the video description",
    "on_screen": "name shown on screen",
    "said": "named aloud",
}


def _who(person: dict) -> str:
    text = person.get("name", "")
    if person.get("role"):
        text += f", {person['role']}"
    phrase = _EVIDENCE_PHRASES.get(person.get("evidence", ""))
    return f"{text} ({phrase})" if phrase else text


def _word_in(phrase: str, text: str, *, case: bool) -> bool:
    """``phrase`` as whole words in ``text`` ("Lee" is not in "fleeing" or
    "Leeds"); ``case`` makes the match case-sensitive."""
    if not phrase or not text:
        return False
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text, 0 if case else re.IGNORECASE) is not None


def _names(beat: dict, person: dict) -> bool:
    """Does this beat's speaker or on-screen description name the person?

    Whole words only. The speaker field may use the full name or the surname
    alone, in any case. on_screen (free description) must use the full name
    (as capitalised when it is one word) or the capitalised surname, so "a
    burning bush" never credits a Mr Bush or a "Bush", and "protesting" never
    credits Sting."""
    name = " ".join(str(person.get("name") or "").split())
    if not name:
        return False
    tokens = name.split()
    surname = tokens[-1] if len(tokens) > 1 and len(tokens[-1]) >= 3 else ""
    speaker = " ".join(str(beat.get("speaker") or "").split())
    on_screen = " ".join(str(beat.get("on_screen") or "").split())
    if _word_in(name, speaker, case=False) or _word_in(name, on_screen, case=len(tokens) == 1):
        return True
    if surname:
        return _word_in(surname, speaker, case=False) or _word_in(surname, on_screen, case=True)
    return False


def _segment_spans(segments) -> list[tuple[float, float]]:
    spans = []
    for seg in segments or []:
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            spans.append((start, end))
    return spans


def _merge_spans(spans: list[tuple[float, float]], gap: float = 1.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _fmt_span(span: tuple[float, float]) -> str:
    start = int(math.floor(span[0] + 1e-6))
    end = max(start + 1, int(math.ceil(span[1] - 1e-6)))
    return f"{start}-{end} s"


def credits_for(log, segments) -> str:
    """Who a viewer would recognise in a clip, on the CLIP clock.

    People the log ranks famous, public_figure or expert, and whose name is
    known (title, description, on screen or said), are mapped to clip time
    through the beats that name them: "0-12 s: Bill Nye, science educator
    (named in the title)". People no beat names are listed as "On screen at
    some point: …"; people only named outside the clip are left out. ""
    when nobody qualifies. Judges can't name a face themselves, so without
    this a famous person reads as "a man talks to camera"."""
    log = log or {}
    people = [p for p in log.get("people") or []
              if p.get("prominence") in _CREDIT_PROMINENCE
              and p.get("evidence") in _EVIDENCE_PHRASES]
    if not people:
        return ""
    segs = _segment_spans(segments)
    beats = log.get("beats") or []
    timed: list[tuple[float, str]] = []
    untimed: list[str] = []
    for person in people:
        named = [b for b in beats if _names(b, person)]
        if not named:
            untimed.append(_who(person))
            continue
        spans, offset = [], 0.0
        for seg_start, seg_end in segs:
            for beat in named:
                lo, hi = max(beat["start"], seg_start), min(beat["end"], seg_end)
                if hi - lo > 0.05:
                    spans.append((offset + lo - seg_start, offset + hi - seg_start))
            offset += seg_end - seg_start
        if spans:
            merged = _merge_spans(spans)
            timed.append((merged[0][0], f"{', '.join(_fmt_span(s) for s in merged)}: {_who(person)}"))
    parts = [text for _, text in sorted(timed, key=lambda t: t[0])]
    if untimed:
        parts.append("On screen at some point: " + ", ".join(untimed))
    return "; ".join(parts)


# ---- caching ------------------------------------------------------------------------------

def _cache_file(video_id, perceiver, model, knobs: dict, *,
                part: tuple[float, float] | None = None) -> Path:
    name = (f"{_safe(video_id)}__{_safe(perceiver)}__{_safe(model)}__v{VERSION}__"
            f"{prompt_hash(_mode(perceiver))}__s{_settings_tag(knobs)}")
    if part is not None:
        name += f"__p{_num_tag(part[0])}-{_num_tag(part[1])}"
    return engine_dir() / "shotlogs" / f"{name}.json"


def cache_path(video_id, perceiver, model, *, part: tuple[float, float] | None = None,
               settings=None, seconds: float | None = None) -> Path:
    """engine_dir()/shotlogs/<video_id>__<perceiver>__<model>__v2__<prompt_hash>__s<settings>[__p<start>-<end>].json

    ``<settings>`` fingerprints ``perception_settings(perceiver, settings,
    seconds)``: pass the run's settings and the source's length to find the
    log a run with them would use (``settings`` None = the defaults)."""
    return _cache_file(video_id, perceiver, model,
                       perception_settings(perceiver, settings, seconds), part=part)


def _usable(data: Any) -> bool:
    return (isinstance(data, dict) and data.get("version") == VERSION
            and data.get("complete") is True and bool(data.get("beats")))


def _read_usable(path: Path) -> dict | None:
    data = util.read_json(path, None)
    if not _usable(data):
        return None
    util.touch(path)
    return data


def load_cached(video_id, perceiver, model, *, settings=None,
                seconds: float | None = None) -> dict | None:
    """The cached whole-video log made with these perception settings (see
    ``cache_path``), only when it is complete and non-empty."""
    return _read_usable(cache_path(video_id, perceiver, model, settings=settings,
                                   seconds=seconds))


# ---- building -----------------------------------------------------------------------------

def _spans(lo: float, hi: float, size: float, overlap: float) -> list[tuple[float, float]]:
    """Windows of at most ``size`` seconds covering [lo, hi], neighbours
    sharing ``overlap``, all the same length (no tiny tail window)."""
    total = hi - lo
    size = max(size, overlap + 30.0)
    if total <= size + 1e-6:
        return [(round(lo, 1), round(hi, 1))]
    count = math.ceil((total - overlap) / (size - overlap))
    length = (total + (count - 1) * overlap) / count
    spans = []
    for i in range(count):
        start = lo + i * (length - overlap)
        end = hi if i == count - 1 else start + length
        spans.append((round(start, 1), round(end, 1)))
    return spans


def _chunks(lo: float, hi: float, interval: float, per_call: int,
            max_seconds: float | None = None) -> list[tuple[float, float]]:
    """Back-to-back stretches of at most ``per_call`` frames at ``interval``
    and at most ``max_seconds`` long. Without the length cap a sparsely
    sampled hour would go in quarters: more speech than one call's transcript
    holds, and more beats than one answer."""
    frames = max(1, math.ceil((hi - lo) / interval - 1e-9))
    count = math.ceil(frames / per_call)
    if max_seconds and max_seconds > 0:
        count = max(count, math.ceil((hi - lo) / max_seconds - 1e-9))
    count = max(1, min(count, frames))
    length = math.ceil(frames / count) * interval
    spans = []
    for i in range(count):
        start = lo + i * length
        end = hi if i == count - 1 else min(hi, start + length)
        if end - start > 1e-6:
            spans.append((round(start, 2), round(end, 2)))
    return spans


def _halves(start: float, end: float) -> list[tuple[float, float]]:
    mid = (start + end) / 2
    half_overlap = min(_PART_OVERLAP, (end - start) / 4) / 2
    return [(start, round(mid + half_overlap, 1)), (round(mid - half_overlap, 1), end)]


def _speech_entries(words, transcript) -> list[dict]:
    """Clause-sized speech lines: from the word stream when there is one
    (exact edges), else the transcript entries."""
    if words:
        from .. import llm

        clauses = llm._words_to_clauses(list(words))
        if clauses:
            return clauses
    entries = []
    for item in transcript or []:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})
    return entries


def _stretch(entries: list[dict], start: float, end: float) -> str:
    """The speech in [start, end] as "[t] text" lines, every line kept: past
    ``_STRETCH_CHARS`` the longest lines are shortened alike rather than the
    end of the stretch dropped (a chunk with no words at its end gets beats
    with no speaker or line, and so no credits)."""
    lines = [f"[{e['start']:.1f}] {e['text']}" for e in entries
             if e["end"] > start and e["start"] < end]
    if sum(len(line) + 1 for line in lines) > _STRETCH_CHARS:
        share = max(24, _STRETCH_CHARS // len(lines) - 1)
        lines = [_trim(line, share) for line in lines]
    return "\n".join(lines)


def _coverage(beats: list[dict], start: float, stop: float, *, slack: float = 0.0) -> float:
    """The share of [start, stop] the beats cover (0-1). The last ``slack``
    seconds needn't be: no frame was sampled after the final one."""
    need = (stop - start) - min(max(0.0, slack), (stop - start) / 4)
    if need <= 0:
        return 1.0
    covered, reach = 0.0, start
    for beat in sorted(beats, key=lambda b: b["start"]):
        lo, hi = max(beat["start"], reach), min(beat["end"], stop)
        if hi > lo:
            covered += hi - lo
            reach = hi
    return min(1.0, covered / need)


def _looping(usage: Any, start: float, end: float) -> bool:
    """Did an answer that hit the output limit run far past what any real log
    of [start, end] needs? Then the model was repeating itself, not logging
    a dense stretch."""
    try:
        tokens = float((usage or {}).get("output_tokens") or 0)
    except (AttributeError, TypeError, ValueError):
        return False
    return tokens >= _LOOP_TOKENS_PER_SECOND * max(1.0, end - start)


def _answered_in_source_time(raw: Any, start: float, end: float) -> bool:
    """Did the model give SOURCE times for a window (despite being told to
    count from the window's 00:00)? Times past the window's own length can't
    be window positions; if they all sit inside the source span, they are
    source times and must not be offset again."""
    starts = [parse_ts(b.get("start")) for b in _list((raw or {}).get("beats"))
              if isinstance(b, dict)]
    starts = [s for s in starts if s is not None]
    if not starts or start <= 0:
        return False
    length = end - start
    return max(starts) > length + 5.0 and min(starts) >= start - 5.0


class _Runner:
    """Logs a span part by part: cached parts are reused, finished parts are
    cached at once, truncated parts are split, failed parts become an
    incomplete placeholder (only media failures, an open quota breaker and
    the caller's ``check`` stop the whole log). Subclasses supply ``spans``
    and ``call``."""

    perceiver = ""
    split_errors: tuple = ()
    fatal_errors: tuple = ()

    def __init__(self, *, video_id, source, knobs, brand, title, description, seconds,
                 meter, force, model, check=None, cancel=None) -> None:
        self.video_id, self.source, self.knobs = video_id, source, dict(knobs)
        self.brand, self.title, self.description = brand or {}, title or "", description or ""
        self.seconds, self.meter, self.force, self.model = seconds, meter, force, model
        self.check, self.cancel = check, cancel
        self._lock = threading.Lock()

    # -- provider-specific --
    def spans(self, lo: float, hi: float) -> list[tuple[float, float]]:  # pragma: no cover
        raise NotImplementedError

    def call(self, start: float, end: float, index: int, count: int) -> tuple[dict, dict, float]:
        raise NotImplementedError  # pragma: no cover

    def step(self) -> float:
        """Seconds between sampled frames: a log may stop this short of a span's end."""
        return 0.0

    def done_with(self, start: float, end: float) -> None:
        """This span's work is finished (cached, or split into halves)."""

    def finish(self) -> None:
        """Best-effort cleanup once every part has run."""

    # -- shared --
    def run(self, lo: float, hi: float) -> list[tuple[float, dict]]:
        spans = self.spans(lo, hi)
        count = len(spans)
        try:
            results = util.parallel_map(
                lambda job: self.part(job[1][0], job[1][1], index=job[0] + 1, count=count, level=0),
                list(enumerate(spans)), workers=min(_PART_WORKERS, count), cancel=self.cancel,
                name="clip-engine-perceive")
        finally:
            self.finish()
        parts: list[tuple[float, dict]] = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            parts.extend(result)
        if not any(part.get("beats") for _, part in parts):
            errors = [part.get("error") for _, part in parts if part.get("error")]
            raise ShotLogError("the video could not be logged" + (f": {errors[0]}" if errors else ""))
        return parts

    def _failed(self, start: float, end: float, error: str) -> dict:
        return {"version": VERSION, "perceiver": self.perceiver, "model": self.model,
                "prompt_hash": prompt_hash(_mode(self.perceiver)), "source_seconds": self.seconds,
                "created_at": util.utcnow_iso(), "cost_usd": 0.0, "complete": False,
                "summary": "", "format": "", "structure": "", "people": [], "beats": [],
                "highlights": [], "span": [start, end], "error": error[:300]}

    def _path(self, start: float, end: float) -> Path:
        return _cache_file(self.video_id, self.perceiver, self.model, self.knobs,
                           part=(start, end))

    def _was_split(self, start: float, end: float) -> bool:
        return any(self._path(*half).is_file() for half in _halves(start, end))

    def _split(self, start: float, end: float, *, index: int, count: int, level: int):
        out: list[tuple[float, dict]] = []
        for half in _halves(start, end):
            out.extend(self.part(half[0], half[1], index=index, count=count, level=level + 1))
        return out

    def part(self, start: float, end: float, *, index: int, count: int, level: int):
        # Before anything is spent on this span — a window, or a half of one
        # that was split: a cancelled run, a passed deadline or a run past its
        # cost cap stops here. Raised outside the try below, so it ends the
        # log instead of becoming a failed placeholder.
        if self.check is not None:
            self.check()
        path = self._path(start, end)
        if not self.force:
            cached = _read_usable(path)
            if cached is not None:
                _add_cached(self.meter, cached)
                return [(start, cached)]
            # An earlier run already split this span: go straight to its halves
            # rather than paying for another truncated answer.
            if level < _MAX_SPLIT_LEVEL and self._was_split(start, end):
                return self._split(start, end, index=index, count=count, level=level)
        try:
            raw, usage, offset = self.call(start, end, index, count)
        except self.fatal_errors:
            raise
        except self.split_errors as exc:
            spent = getattr(exc, "usage", None)
            _add_usage(self.meter, "perceive", spent)
            if _looping(spent, start, end):
                log.warning("shot log %s %.1f-%.1fs: %s output tokens for %.0f s of video; the "
                            "model is repeating itself, so it isn't split", self.video_id, start,
                            end, (spent or {}).get("output_tokens"), end - start)
                return [(start, self._failed(start, end, f"answer too long (looping): {exc}"))]
            if level < _MAX_SPLIT_LEVEL and end - start >= 2 * _MIN_SPLIT_SECONDS:
                log.info("shot log %s %.1f-%.1fs ran past the output limit; splitting it",
                         self.video_id, start, end)
                self.done_with(start, end)
                return self._split(start, end, index=index, count=count, level=level)
            log.warning("shot log %s %.1f-%.1fs still too long after splitting: %s",
                        self.video_id, start, end, exc)
            return [(start, self._failed(start, end, f"answer too long: {exc}"))]
        except Exception as exc:  # noqa: BLE001 - one bad window must not sink the log
            _add_usage(self.meter, "perceive", getattr(exc, "usage", None))
            log.warning("shot log %s %.1f-%.1fs failed: %s", self.video_id, start, end, exc)
            return [(start, self._failed(start, end, str(exc) or type(exc).__name__))]
        _add_usage(self.meter, "perceive", usage)
        part = normalize_log(raw, duration=self.seconds, perceiver=self.perceiver,
                             model=self.model, cost=(usage or {}).get("cost_usd", 0.0),
                             offset=offset)
        part["span"] = [start, end]
        part["perceiver_settings"] = dict(self.knobs)
        if part["complete"]:
            # An answer that stops early (finishReason STOP after a few
            # minutes) is used for this run but not cached: cached, the hole
            # would be reused by every later run, unflagged.
            stop = min(end, self.seconds)
            covered = _coverage(part["beats"], start, stop, slack=self.step())
            if covered < _MIN_COVERAGE:
                part["complete"] = False
                part["error"] = f"the model logged only {covered:.0%} of {start:g}-{stop:g}s"
                log.warning("shot log %s: %s; not caching it", self.video_id, part["error"])
        if part["complete"]:
            util.atomic_write_json(path, part)
            self.done_with(start, end)
        elif not part.get("error"):
            part["error"] = "the model logged no beats"
        return [(start, part)]


class _GeminiRunner(_Runner):
    perceiver = "gemini"
    split_errors = (gemini.GeminiTruncated,)
    fatal_errors = (gemini.GeminiQuota, media.MediaError)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        k = self.knobs      # perception_settings("gemini", ...): the fps for this length
        self.fps, self.thinking = k["fps"], k["thinking"]
        self.media_resolution, self.window = k["media_resolution"], k["window_seconds"]
        self.max_output_tokens = k["max_output_tokens"]
        self.system = prompts.perceiver_system(self.brand, mode="video")
        self._window_uploads: dict[tuple[float, float], tuple[list[str], Path]] = {}
        self._finished: list[tuple[list[str], Path]] = []

    def spans(self, lo, hi):
        return _spans(lo, hi, self.window, _PART_OVERLAP)

    def step(self):
        return 1.0 / self.fps if self.fps > 0 else 0.0

    def _whole(self, start: float, end: float) -> bool:
        return start <= 0.05 and end >= self.seconds - 0.05

    def call(self, start, end, index, count):
        whole = self._whole(start, end)
        if whole:
            # The whole-source proxy is shared with the editor, which watches it.
            proxy = media.perception_proxy(self.source)
        else:
            proxy = media.perception_proxy(self.source, start=start, end=end)
        video = gemini.video_parts([{"path": proxy, "fps": self.fps, "upload": True}])
        if not whole:
            uris = [p.get("fileData", {}).get("fileUri", "") for p in video if isinstance(p, dict)]
            with self._lock:
                self._window_uploads[(start, end)] = ([u for u in uris if u], Path(proxy))
        user = prompts.perceiver_user(
            title=self.title, description=self.description, duration=self.seconds,
            part=None if whole else (index, count), span=None if whole else (start, end))
        data, usage = gemini.generate_json(
            self.model, system=self.system, parts=list(video) + [gemini.text_part(user)],
            schema=SCHEMA, thinking=self.thinking, media_resolution=self.media_resolution,
            max_output_tokens=self.max_output_tokens,
            label=f"perceive {self.video_id} {start:g}-{end:g}s", bucket=_bucket(self.meter))
        offset = 0.0 if whole else start
        if not whole and _answered_in_source_time(data, start, end):
            log.warning("shot log %s %.1f-%.1fs came back in source time; not offsetting it",
                        self.video_id, start, end)
            offset = 0.0
        return data, usage, offset

    def done_with(self, start, end):
        with self._lock:
            entry = self._window_uploads.pop((start, end), None)
            if entry is not None:
                self._finished.append(entry)

    def finish(self):
        """Delete finished window proxies' uploads and the proxies themselves.

        Dropping the local proxy also matters for correctness: a re-made proxy
        gets a new mtime, so the upload cache can't hand back a deleted file."""
        with self._lock:
            finished, self._finished = self._finished, []
        for uris, proxy in finished:
            for uri in uris:
                try:
                    gemini.delete_file(uri)
                except Exception as exc:  # noqa: BLE001 - files expire on their own
                    log.debug("could not delete %s: %s", uri, exc)
            try:
                media.drop([proxy])
            except Exception as exc:  # noqa: BLE001
                log.debug("could not drop %s: %s", proxy, exc)


class _ClaudeRunner(_Runner):
    perceiver = "claude"
    split_errors = (claude.ClaudeTruncated,)
    fatal_errors = (media.MediaError,)

    def __init__(self, *, words, transcript, **kwargs) -> None:
        super().__init__(**kwargs)
        k = self.knobs      # perception_settings("claude", ...)
        self.frame_interval, self.max_total = k["frame_interval"], k["max_total_frames"]
        self.width, self.per_call = k["frame_width"], k["max_frames_per_call"]
        self.effort, self.max_tokens = k["effort"], k["max_tokens"]
        self.entries = _speech_entries(words, transcript)
        self.interval = self.frame_interval
        self.system = prompts.perceiver_system(self.brand, mode="frames", frame_interval=self.interval)

    def spans(self, lo, hi):
        # Long sources are sampled more sparsely to stay under max_total_frames,
        # but still go in chunks of at most chunk_seconds.
        self.interval = max(self.frame_interval, (hi - lo) / self.max_total)
        self.system = prompts.perceiver_system(self.brand, mode="frames", frame_interval=self.interval)
        return _chunks(lo, hi, self.interval, self.per_call, self.knobs["chunk_seconds"])

    def step(self):
        return self.interval

    def call(self, start, end, index, count):
        frames = media.sample_frames(self.source, start=start, end=end, interval=self.interval,
                                     width=self.width)
        if not frames:
            raise media.MediaError(f"no frames between {start:g}s and {end:g}s")
        text = (prompts.perceiver_user(title=self.title, description=self.description,
                                       duration=self.seconds)
                + f"\nThese frames cover {start:.1f}s to {end:.1f}s of the video."
                + "\nTranscript for this stretch:\n"
                + (_stretch(self.entries, start, end) or "(no speech)"))
        content = [claude.text_block(text)] + list(claude.frames_blocks(frames))
        data, usage = claude.json_call(
            self.model, system=self.system, content=content, schema=SCHEMA, effort=self.effort,
            max_tokens=self.max_tokens, label=f"perceive {self.video_id} {start:g}-{end:g}s",
            bucket=_bucket(self.meter))
        return data, usage, 0.0   # frame labels are already on the source clock


def _source_seconds(duration, source) -> float:
    seconds = _positive(duration)
    if seconds is None and source:
        seconds = _positive(media.probe_duration(source))
    if seconds is None:
        raise ShotLogError("the source's duration is unknown")
    return float(seconds)


def build(*, video_id, source, perceiver, settings, brand, title, description, duration,
          words, transcript, window=None, meter=None, force=False,
          check: Callable[[], None] | None = None,
          cancel: threading.Event | None = None) -> dict:
    """The shot log for ``source`` (or just ``window`` of it).

    A cached whole-video log made with the same perception settings is
    reused (filtered to ``window`` when given) and its stored cost reported
    as ``perceive_cached``. Otherwise the span is logged part by part; a
    window-only log is built only when no whole-video log exists, and only a
    complete whole-video log is cached. ``check`` runs before every part and
    every split (the engine's cancel / deadline / cost-cap checkpoint): what it
    raises ends the log and propagates. ``cancel`` stops parts not yet
    started. Raises ``ShotLogError`` (or a media / quota error) when nothing
    could be logged."""
    if perceiver not in ("gemini", "claude"):
        raise ShotLogError(f"unknown perceiver {perceiver!r}")
    model = model_for(settings, perceiver, "perceiver")
    bounds = _bounds(window)
    seconds = _source_seconds(duration, source)
    knobs = perception_settings(perceiver, settings, seconds)
    whole_path = _cache_file(video_id, perceiver, model, knobs)
    if not force:
        whole = _read_usable(whole_path)
        if whole is not None:
            _add_cached(meter, whole)
            log.info("shot log for %s: reusing the %s/%s log (%d beats)", video_id, perceiver,
                     model, len(whole["beats"]))
            return filter_to_window(whole, window) if bounds else whole
    lo, hi = (0.0, seconds) if bounds is None else (max(0.0, bounds[0]), min(seconds, bounds[1]))
    if hi - lo < 1.0:
        raise ShotLogError(f"nothing to log between {lo:g}s and {hi:g}s")
    common = dict(video_id=video_id, source=source, knobs=knobs, brand=brand, title=title,
                  description=description, seconds=seconds, meter=meter, force=force, model=model,
                  check=check, cancel=cancel)
    if perceiver == "gemini":
        runner: _Runner = _GeminiRunner(**common)
    else:
        runner = _ClaudeRunner(words=words, transcript=transcript, **common)
    merged = merge_windows(runner.run(lo, hi))
    merged["source_seconds"] = round(seconds, 2)
    merged["perceiver_settings"] = dict(knobs)
    if bounds is not None:
        merged["window"] = {"start": lo, "end": hi}
    elif merged["complete"] and merged["beats"]:
        util.atomic_write_json(whole_path, merged)
    else:
        log.warning("shot log for %s is incomplete; the whole-video log is not cached", video_id)
    log.info("shot log for %s (%s/%s): %d beats, %d people, complete=%s", video_id, perceiver,
             model, len(merged["beats"]), len(merged["people"]), merged["complete"])
    return merged
