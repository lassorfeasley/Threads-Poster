"""Watching drafts cold, ranking them head-to-head, and checking revisions.

Three judgments, each built against a known way model judges go wrong:

- ``cold_watch`` is one stranger meeting one rendered draft in a feed, muted
  first: it knows nothing about the account except who is on screen (the
  Credits line — models won't name a face, so without it a famous person's
  charged line reads as "a man talks to camera"). Gemini also sees the first
  three seconds at a high frame rate, because the opening decides the
  scroll. It never raises: an outage yields a "not reviewed" critique that
  ``cold_score`` sorts last, so a provider failure never reads as a verdict.
- ``rank`` compares drafts side by side against this account's own hits and
  flops. Pairwise and listwise judges favour positions and letters, so it
  runs twice — in the given order and reversed, with letters assigned by
  position in each pass — and merges the two with a Borda count (ties
  broken by the cold score, not by which pass came first).
- ``prefer`` decides whether a revision beats its original. A single
  forced-choice call between near-identical cuts is close to a coin flip, so
  it asks in both presentation orders and accepts the revision only when it
  wins both; "same" or any failure keeps the original.

Every critique also answers three tests that came out of the first live
run: ``premise_clear`` (within ~5 s you know what is happening and where),
``speech_understood`` (every word is English or carried by an English
voiceover; subtitles don't count) and ``monologue`` (mostly one reporter or
anchor talking to camera). ``cold_score`` charges for each. When the
operator gave advice for the pass, the critic also says whether the clip
``follows_advice`` (with an ``advice_note``), and the ranking and revision
judges read the advice too.

Critique times are CLIP seconds (from the start of the rendered draft).
"""
from __future__ import annotations

import copy
import logging
import math
import re
from pathlib import Path
from typing import Any

from . import claude, gemini, media, model_for, opt, prompts, util

log = logging.getLogger("clip_engine.critic")

MAX_VIDEOS_PER_REQUEST = 10      # Gemini's documented limit
_OPENING_SECONDS = 3.0
_COLD_MAX_FRAMES = 60
_RANK_EXAMPLE_INTERVAL, _RANK_EXAMPLE_FRAMES = 3.0, 10
_RANK_DRAFT_INTERVAL, _RANK_DRAFT_FRAMES = 2.0, 20
_WORDS_CHARS = 4000
_STRINGS = {"first_impression": 400, "muted_reason": 300, "stop_reason": 300, "about": 300,
            "dangling": 200, "topic_note": 200, "person": 200, "fix": 300}
_BOOLS = ("stops_scroll_muted", "stops_scroll", "person_at_start", "makes_sense", "ends_well")
# The three tests, with the answer that costs nothing when a critique leaves
# one out (an older critique, a sloppy answer): a missing answer is not a
# verdict.
_TESTS = {"premise_clear": True, "speech_understood": True, "monologue": False}
_ADVICE_NOTE_CHARS = 300

# cold_score's charges for the three tests and for ignoring the advice.
PENALTY_SPEECH = 3.0
PENALTY_PREMISE = 2.0
PENALTY_MONOLOGUE = 2.0
PENALTY_ADVICE = 3.0

CRITIQUE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "first_impression": {"type": "string"},
        "stops_scroll_muted": {"type": "boolean"},
        "muted_reason": {"type": "string"},
        "stops_scroll": {"type": "boolean"},
        "stop_reason": {"type": "string"},
        "person_at_start": {"type": "boolean"},
        "about": {"type": "string"},
        "makes_sense": {"type": "boolean"},
        "dangling": {"type": "string"},
        "premise_clear": {"type": "boolean",
                          "description": "within about 5 s you know what is happening and where"},
        "speech_understood": {"type": "boolean",
                              "description": "every spoken word is English or carried by an "
                                             "English voiceover; subtitles don't count"},
        "monologue": {"type": "boolean",
                      "description": "mostly one reporter or anchor talking to camera"},
        "bored_at": {"type": ["number", "null"],
                     "description": "seconds from the clip's start; null if you'd watch to the end"},
        "dead_air": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
                "required": ["start", "end"],
            },
        },
        "ends_well": {"type": "boolean"},
        "topic": {"type": "string", "enum": list(prompts.TOPIC_LABELS)},
        "topic_note": {"type": "string"},
        "person": {"type": "string"},
        "visual": {"type": "integer", "minimum": 1, "maximum": 5},
        "audio": {"type": "integer", "minimum": 1, "maximum": 5},
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "fix": {"type": "string"},
    },
    "required": ["first_impression", "stops_scroll_muted", "muted_reason", "stops_scroll",
                 "stop_reason", "person_at_start", "about", "makes_sense", "dangling",
                 "premise_clear", "speech_understood", "monologue",
                 "bored_at", "dead_air", "ends_well", "topic", "topic_note", "person",
                 "visual", "audio", "score", "fix"],
}


def critique_schema(*, advice: bool = False) -> dict:
    """CRITIQUE_SCHEMA, plus ``follows_advice`` and ``advice_note`` (right
    after the three tests, before the verdict fields) when the operator gave
    advice for this pass."""
    if not advice:
        return CRITIQUE_SCHEMA
    properties: dict = {}
    for key, value in CRITIQUE_SCHEMA["properties"].items():
        properties[key] = copy.deepcopy(value)
        if key == "monologue":
            properties["follows_advice"] = {
                "type": "boolean", "description": "the clip does what the operator asked"}
            properties["advice_note"] = {"type": "string"}
    return {"type": "object", "properties": properties, "required": list(properties)}

# Notes come first: a judge that has written down what it saw in every draft
# ranks more consistently than one that ranks first and justifies after.
RANK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "note": {"type": "string"}},
                "required": ["id", "note"],
            },
        },
        "ranking": {"type": "array", "items": {"type": "string"}},
        "best_reason": {"type": "string"},
    },
    "required": ["notes", "ranking", "best_reason"],
}

PREFER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["1", "2", "same"]},
        "reason": {"type": "string"},
    },
    "required": ["choice", "reason"],
}


# ---- small helpers ------------------------------------------------------------------------

def _text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip() if len(text) > limit else text


def _bucket(meter) -> str | None:
    """The meter's spend bucket. Without a meter, None lets the provider use
    the process default (``spend.set_bucket``: "bench" in the bench CLI), else
    "engine" — never the shared LLM budget."""
    return getattr(meter, "bucket", None) or None


def _add_usage(meter, stage: str, usage: dict | None) -> None:
    if meter is not None and usage:
        meter.add(stage, usage)


def _metered(meter, stage: str, call):
    try:
        data, usage = call()
    except Exception as exc:
        _add_usage(meter, stage, getattr(exc, "usage", None))
        raise
    _add_usage(meter, stage, usage)
    return data if isinstance(data, dict) else {}


def _float(settings, key: str, default: float) -> float:
    try:
        return float(opt(settings, key, default))
    except (TypeError, ValueError):
        return float(default)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        from .shotlog import parse_ts

        return parse_ts(value)
    return number if math.isfinite(number) else None


def _int(value: Any, lo: int, hi: int) -> int | None:
    number = _number(value)
    if number is None:
        return None
    return max(lo, min(hi, int(round(number))))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in ("true", "yes", "y", "1")


def _fmt_secs(value: float) -> str:
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def words_lines(clip_words, limit: int = _WORDS_CHARS) -> str:
    """Words heard as "[1.2] text" lines on the clip clock. Accepts a word
    stream ({word, start, end}) or transcript entries ({start, end, text})."""
    items = [w for w in clip_words or [] if isinstance(w, dict)]
    if not items:
        return ""
    if any("word" in w for w in items):
        from .. import llm

        entries = llm._words_to_clauses(items)
    else:
        entries = items
    lines, used = [], 0
    for entry in entries:
        try:
            line = f"[{float(entry['start']):.1f}] {str(entry.get('text', '')).strip()}"
        except (KeyError, TypeError, ValueError):
            continue
        used += len(line) + 1
        if used > limit:
            lines.append("[…]")
            break
        lines.append(line)
    return "\n".join(lines)


# ---- critiques ----------------------------------------------------------------------------

def failure_critique(error: str, *, usage: dict | None = None) -> dict:
    """The critique of a draft nobody could watch: score 0, every boolean
    false, topic "central" (an outage must not read as off-topic)."""
    out = {
        "first_impression": "", "stops_scroll_muted": False, "muted_reason": "",
        "stops_scroll": False, "stop_reason": "", "person_at_start": False, "about": "",
        "makes_sense": False, "dangling": "", "premise_clear": False,
        "speech_understood": False, "monologue": False, "bored_at": None, "dead_air": [],
        "ends_well": False, "topic": "central", "topic_note": "", "person": "",
        "visual": 0, "audio": 0, "score": 0, "fix": "", "error": _text(error, 300) or "failed",
    }
    if usage:
        out["usage"] = dict(usage)
    return out


def normalize_critique(raw, *, clip_seconds, advice: bool = False) -> dict:
    """A model's critique, coerced: ints clamped (visual/audio 1-5, score
    1-10), ``bored_at`` None or within the clip, ``dead_air`` as clip-time
    {start, end} spans, topic one of prompts.TOPIC_LABELS (else "central").
    The three tests default to their no-charge answer when missing. With
    ``advice``, ``follows_advice`` (default True when missing) and
    ``advice_note`` are kept too. An answer with no usable score is a
    failure, not a verdict."""
    if not isinstance(raw, dict):
        return failure_critique("the critic returned no critique")
    score = _int(raw.get("score"), 1, 10)
    if score is None:
        return failure_critique("the critic returned no score")
    seconds = max(0.0, _number(clip_seconds) or 0.0)
    bored = _number(raw.get("bored_at"))
    if bored is not None:
        bored = max(0.0, bored)
        bored = None if seconds and bored >= seconds else round(bored, 2)
    dead_air = []
    for item in raw.get("dead_air") if isinstance(raw.get("dead_air"), list) else []:
        if isinstance(item, dict):
            start, end = _number(item.get("start")), _number(item.get("end"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = _number(item[0]), _number(item[1])
        else:
            continue
        if start is None or end is None:
            continue
        start = max(0.0, start)
        if seconds:
            end = min(seconds, end)
        if end > start:
            dead_air.append({"start": round(start, 2), "end": round(end, 2)})
    topic = str(raw.get("topic") or "").strip().lower()
    out: dict[str, Any] = {key: _text(raw.get(key), limit) for key, limit in _STRINGS.items()}
    out.update({key: _bool(raw.get(key)) for key in _BOOLS})
    out.update({key: _bool(raw[key]) if raw.get(key) is not None else default
                for key, default in _TESTS.items()})
    if advice:
        out["follows_advice"] = (_bool(raw["follows_advice"])
                                 if raw.get("follows_advice") is not None else True)
        out["advice_note"] = _text(raw.get("advice_note"), _ADVICE_NOTE_CHARS)
    out.update({
        "bored_at": bored,
        "dead_air": dead_air,
        "topic": topic if topic in prompts.TOPIC_LABELS else "central",
        "visual": _int(raw.get("visual"), 1, 5) or 3,
        "audio": _int(raw.get("audio"), 1, 5) or 3,
        "score": score,
        "error": "",
    })
    order = list(critique_schema(advice=advice)["properties"]) + ["error"]
    return {key: out[key] for key in order}


def _cold_gemini(settings, brand, render, seconds, credits, meter, label, advice="") -> dict:
    opening_end = min(_OPENING_SECONDS, seconds) if seconds > 0 else _OPENING_SECONDS
    parts = list(gemini.video_parts([
        {"path": render, "fps": _float(settings, "clip_engine.gemini.critic_fps", 2.0)},
        {"path": render, "fps": _float(settings, "clip_engine.gemini.opening_fps", 8.0),
         "start": 0.0, "end": opening_end},
    ]))
    parts.append(gemini.text_part(
        f"Clip length: {_fmt_secs(seconds)} s. The second video is the same clip's first "
        f"{_fmt_secs(opening_end)} seconds again, at a higher frame rate.\n"
        f"Credits: {credits or 'none'}"))
    return _metered(meter, "cold", lambda: gemini.generate_json(
        model_for(settings, "gemini", "critic"),
        system=prompts.cold_system(brand, mode="video", advice=advice),
        parts=parts, schema=critique_schema(advice=bool(advice)),
        thinking=opt(settings, "clip_engine.gemini.critic_thinking", "LOW"),
        media_resolution=opt(settings, "clip_engine.gemini.critic_media_resolution",
                             "MEDIA_RESOLUTION_LOW"),
        max_output_tokens=16384, label=label, bucket=_bucket(meter)))


def _cold_claude(settings, brand, render, seconds, clip_words, credits, meter, label,
                 advice="") -> dict:
    interval = _float(settings, "clip_engine.claude.critic_frame_interval", 1.0)
    frames = media.clip_frames(render, interval=interval,
                               width=int(_float(settings, "clip_engine.claude.frame_width", 448)),
                               max_frames=_COLD_MAX_FRAMES)
    if not frames:
        raise media.MediaError(f"no frames from {render}")
    effective = max(interval, seconds / _COLD_MAX_FRAMES) if seconds > 0 else interval
    text = (f"Clip length: {_fmt_secs(seconds)} s.\nCredits: {credits or 'none'}\n"
            f"Words heard (clip seconds):\n{words_lines(clip_words) or '(none)'}")
    content = [claude.text_block(text)] + list(claude.frames_blocks(frames, clock="clip"))
    return _metered(meter, "cold", lambda: claude.json_call(
        model_for(settings, "claude", "critic"),
        system=prompts.cold_system(brand, mode="frames", frame_interval=effective, advice=advice),
        content=content, schema=critique_schema(advice=bool(advice)),
        effort=opt(settings, "clip_engine.claude.critic_effort", "medium"), max_tokens=16000,
        label=label, bucket=_bucket(meter)))


def cold_watch(*, role, settings, brand, render, clip_seconds, clip_words, credits, meter,
               advice: str = "") -> dict:
    """One cold critique of one rendered draft (spec §3 Critique, clip time).

    With ``advice`` (the operator's steer for this pass) the critic also
    answers ``follows_advice`` / ``advice_note``. Never raises: any failure
    returns ``failure_critique`` carrying the error's usage, which is also
    charged to ``meter`` under "cold"."""
    seconds = max(0.0, _number(clip_seconds) or 0.0)
    label = f"cold {Path(str(render)).name}"
    advice = prompts.clean_advice(advice)
    try:
        if role == "gemini":
            raw = _cold_gemini(settings, brand or {}, render, seconds, credits, meter, label,
                               advice)
        elif role == "claude":
            raw = _cold_claude(settings, brand or {}, render, seconds, clip_words, credits, meter,
                               label, advice)
        else:
            raise ValueError(f"unknown critic role {role!r}")
    except Exception as exc:  # noqa: BLE001 - an outage is "not reviewed", never a verdict
        usage = getattr(exc, "usage", None)
        log.warning("cold watch of %s failed: %s", render, exc)
        return failure_critique(f"{type(exc).__name__}: {exc}", usage=usage)
    return normalize_critique(raw, clip_seconds=seconds, advice=bool(advice))


def _answer(critique: dict, key: str) -> bool | None:
    """A test's answer, or None when the critique doesn't give one."""
    value = critique.get(key)
    return None if value is None else _bool(value)


def cold_score(critique, *, clip_seconds, archetype="") -> float:
    """One number per critique, for ordering drafts (spec §4.11).

    The critic's 1-10 score, less 2 when the topic is incidental/none, 2 when
    it doesn't make sense, 1 when it ends badly, 1.5 when the opening stops
    nobody (0.5 when it stops a viewer only muted or only with sound), up to
    2 for an early swipe; the picture counts (+-0.25 a point) only where the
    footage IS the hook (scene, revelation), the sound for every clip. Then
    the three tests: 3 when the speech isn't understood, 2 when the premise
    isn't clear in ~5 s, 2 for a reporter/anchor monologue; and 3 when the
    clip doesn't follow the operator's advice. A test the critique doesn't
    answer costs nothing. A failed critique scores -1e9 so it sorts after
    every reviewed draft."""
    critique = critique or {}
    if critique.get("error"):
        return -1e9
    s = float(_number(critique.get("score")) or 0.0)
    if critique.get("topic") in ("incidental", "none"):
        s -= 2
    if not critique.get("makes_sense"):
        s -= 2
    if not critique.get("ends_well"):
        s -= 1
    hook = 0.5 * bool(critique.get("stops_scroll_muted")) + 0.5 * bool(critique.get("stops_scroll"))
    if hook == 0:
        s -= 1.5
    elif hook == 0.5:
        s -= 0.5
    bored = _number(critique.get("bored_at"))
    if bored is not None:
        seconds = max(_number(clip_seconds) or 0.0, 1.0)
        s -= 2 * max(0.0, 1 - bored / seconds)
    if archetype in ("scene", "revelation"):
        s += 0.25 * ((_number(critique.get("visual")) or 3) - 3)
    s += 0.2 * ((_number(critique.get("audio")) or 3) - 3)
    if _answer(critique, "speech_understood") is False:
        s -= PENALTY_SPEECH
    if _answer(critique, "premise_clear") is False:
        s -= PENALTY_PREMISE
    if _answer(critique, "monologue") is True:
        s -= PENALTY_MONOLOGUE
    if _answer(critique, "follows_advice") is False:
        s -= PENALTY_ADVICE
    return round(s, 3)


# ---- ranking ------------------------------------------------------------------------------

def borda(rankings, *, tiebreak: dict[str, float] | None = None) -> list[str]:
    """Merge rankings (best first): each id gets n - position points per
    ranking (n = all ids seen). Ties go to the higher ``tiebreak`` score, then
    to the first ranking's order."""
    lists = [[str(x) for x in r] for r in rankings or [] if r]
    ids: list[str] = []
    for ranking in lists:
        for x in ranking:
            if x not in ids:
                ids.append(x)
    n = len(ids)
    points = dict.fromkeys(ids, 0)
    for ranking in lists:
        position = 0
        for x in dict.fromkeys(ranking):
            points[x] += n - position
            position += 1
    first = {x: i for i, x in enumerate(ids)}
    breaks = tiebreak or {}

    def tie(x: str) -> float:
        value = _number(breaks.get(x))
        return value if value is not None else -math.inf

    return sorted(ids, key=lambda x: (-points[x], -tie(x), first[x]))


def _by_id(value: Any, drafts: list[dict], key: str) -> dict[str, Any]:
    """``value`` as {draft id: item}: accepts a dict keyed by id (what
    engine.run passes) or a list aligned with ``drafts``; falls back to each
    draft's own ``key``."""
    out: dict[str, Any] = {}
    for i, d in enumerate(drafts):
        did = str(d.get("id"))
        item = None
        if isinstance(value, dict):
            item = value.get(did)
        elif isinstance(value, (list, tuple)) and i < len(value):
            item = value[i]
        if item is None:
            item = d.get(key)
        out[did] = item
    return out


def _example_pairs(examples, example_files) -> list[tuple[dict, Path]]:
    pairs = []
    for i, example in enumerate(examples or []):
        if not isinstance(example, dict):
            continue
        path = None
        if isinstance(example_files, dict):
            path = example_files.get(example.get("post_pk"))
            if path is None:
                path = example_files.get(str(example.get("post_pk")))
        elif isinstance(example_files, (list, tuple)) and i < len(example_files):
            path = example_files[i]
        if path and Path(path).is_file():
            pairs.append((example, Path(path)))
    return pairs


def _letter(value: Any, allowed: list[str]) -> str | None:
    text = re.sub(r"(?i)\bdraft\b", " ", str(value or "")).strip().upper()
    match = re.match(r"^\W*([A-Z])\b", text)
    letter = match.group(1) if match else None
    return letter if letter in allowed else None


def _ranked_ids(data: dict, order: list[dict]) -> list[str]:
    """A pass's answer as draft ids: letters mapped back by this pass's
    positions, unknown or repeated letters ignored (``rank`` appends any
    draft the answer left out)."""
    letters = _letters(len(order))
    by_letter = {letter: str(d["id"]) for letter, d in zip(letters, order)}
    ranked: list[str] = []
    for item in data.get("ranking") if isinstance(data.get("ranking"), list) else []:
        letter = _letter(item, letters)
        if letter and by_letter[letter] not in ranked:
            ranked.append(by_letter[letter])
    return ranked


def _rank_schema(letters: list[str]) -> dict:
    schema = copy.deepcopy(RANK_SCHEMA)
    schema["properties"]["notes"]["items"]["properties"]["id"]["enum"] = list(letters)
    schema["properties"]["ranking"]["items"]["enum"] = list(letters)
    return schema


def _example_label(i: int, example: dict) -> str:
    from .taste import performance_line

    return f"Example {i} — {example.get('title') or 'untitled'}: {performance_line(example)}"


def _rank_gemini(settings, brand, playbook_text, drafts, renders, credits, examples, meter,
                 advice=""):
    """Builds the request parts once (examples first, uploaded; then drafts)
    and returns ``(run, drafts shown)``; ``run`` does one pass in a given
    order. Examples are optional: if they can't be attached, the drafts are
    ranked without them."""
    fps = _float(settings, "clip_engine.gemini.critic_fps", 2.0)
    draft_items = [{"path": renders[d["id"]], "fps": fps} for d in drafts]
    try:
        videos = list(gemini.video_parts([{"path": path, "upload": True} for _, path in examples]
                                         + draft_items))
    except Exception as exc:
        if not examples:
            raise
        log.warning("taste examples couldn't be attached (%s); ranking without them", exc)
        examples = []
        videos = list(gemini.video_parts(draft_items))
    example_parts = []
    for i, (example, _path) in enumerate(examples):
        example_parts += [gemini.text_part(_example_label(i + 1, example)), videos[i]]
    draft_video = {str(d["id"]): videos[len(examples) + j] for j, d in enumerate(drafts)}
    model = model_for(settings, "gemini", "judge")

    def run(order: list[dict], label: str) -> list[str]:
        letters = _letters(len(order))
        parts = list(example_parts)
        for letter, d in zip(letters, order):
            parts += [gemini.text_part(f"Draft {letter} — Credits: {credits.get(d['id']) or 'none'}"),
                      draft_video[str(d["id"])]]
        parts.append(gemini.text_part(f"Rank all {len(order)} drafts."))
        system = prompts.rank_system(brand, playbook_text or "", k=len(order),
                                     has_examples=bool(examples), mode="video", advice=advice)
        data = _metered(meter, "rank", lambda: gemini.generate_json(
            model, system=system, parts=parts, schema=_rank_schema(letters),
            thinking=opt(settings, "clip_engine.gemini.judge_thinking", "MEDIUM"),
            media_resolution=opt(settings, "clip_engine.gemini.critic_media_resolution",
                                 "MEDIA_RESOLUTION_LOW"),
            max_output_tokens=16384, label=label, bucket=_bucket(meter)))
        return _ranked_ids(data, order)

    return run, list(drafts)


def _frames_or_none(path, *, interval: float, width: int, max_frames: int):
    try:
        return media.clip_frames(path, interval=interval, width=width, max_frames=max_frames) or None
    except Exception as exc:  # noqa: BLE001 - one unreadable file only drops that item
        log.warning("no frames from %s: %s", path, exc)
        return None


def _rank_claude(settings, brand, playbook_text, drafts, renders, credits, examples, meter,
                 advice=""):
    """Extracts every file's frames once and returns ``(run, drafts shown)``.
    An example without frames is left out; so is a draft (it is appended
    after the ranked ones by ``rank``)."""
    width = int(_float(settings, "clip_engine.claude.frame_width", 448))
    example_blocks = []
    shown_examples = 0
    for example, path in examples:
        frames = _frames_or_none(path, interval=_RANK_EXAMPLE_INTERVAL, width=width,
                                 max_frames=_RANK_EXAMPLE_FRAMES)
        if frames:
            shown_examples += 1
            example_blocks += ([claude.text_block(_example_label(shown_examples, example))]
                               + list(claude.frames_blocks(frames, clock="clip")))
    draft_frames = {}
    for d in drafts:
        frames = _frames_or_none(renders[d["id"]], interval=_RANK_DRAFT_INTERVAL, width=width,
                                 max_frames=_RANK_DRAFT_FRAMES)
        if frames:
            draft_frames[str(d["id"])] = list(claude.frames_blocks(frames, clock="clip"))
    shown = [d for d in drafts if str(d["id"]) in draft_frames]
    model = model_for(settings, "claude", "judge")

    def run(order: list[dict], label: str) -> list[str]:
        letters = _letters(len(order))
        content = list(example_blocks)
        for letter, d in zip(letters, order):
            # engine.run attaches each reviewed draft's words on the clip clock.
            words = words_lines(d.get("clip_words")) or "(none)"
            content.append(claude.text_block(
                f"Draft {letter} — Credits: {credits.get(d['id']) or 'none'}\n"
                f"Words heard (clip seconds):\n{words}"))
            content += draft_frames[str(d["id"])]
        content.append(claude.text_block(f"Rank all {len(order)} drafts."))
        system = prompts.rank_system(brand, playbook_text or "", k=len(order),
                                     has_examples=bool(example_blocks), mode="frames",
                                     frame_interval=_RANK_DRAFT_INTERVAL, advice=advice)
        data = _metered(meter, "rank", lambda: claude.json_call(
            model, system=system, content=content, schema=_rank_schema(letters),
            effort=opt(settings, "clip_engine.claude.judge_effort", "medium"), max_tokens=16000,
            label=label, bucket=_bucket(meter)))
        return _ranked_ids(data, order)

    return run, shown


def rank(*, role, settings, brand, playbook_text, drafts, renders, credits, scores,
         examples, example_files, meter, advice: str = "") -> list[str]:
    """Draft ids, best first, from two head-to-head passes (given order, then
    reversed; letters by position in each pass; examples first in both)
    merged by ``borda`` with the cold ``scores`` as tie-break. ``advice``
    (the operator's steer for this pass) goes into the judge's system prompt.

    Gemini sends at most 10 videos per request (examples are dropped first).
    Drafts that can't be shown (no render) and drafts a pass left out are
    appended in input order. Never raises: when both passes fail it logs a
    WARNING and returns the input order."""
    drafts = [d for d in drafts or [] if isinstance(d, dict) and d.get("id") is not None]
    ids = [str(d["id"]) for d in drafts]
    renders_by = _by_id(renders, drafts, "render")
    credits_by = _by_id(credits, drafts, "credits")
    scores_by = _by_id(scores, drafts, "score")
    shown = [d for d in drafts if renders_by.get(str(d["id"]))
             and Path(str(renders_by[str(d["id"])])).is_file()]
    if len(shown) < 2:
        return ids
    pairs = _example_pairs(examples, example_files)
    if role == "gemini":
        shown = shown[:MAX_VIDEOS_PER_REQUEST]
        pairs = pairs[:max(0, MAX_VIDEOS_PER_REQUEST - len(shown))]
    renders_by = {str(d["id"]): Path(str(renders_by[str(d["id"])])) for d in shown}
    credits_by = {k: str(v or "") for k, v in credits_by.items()}
    try:
        if role == "gemini":
            run, shown = _rank_gemini(settings, brand or {}, playbook_text, shown, renders_by,
                                      credits_by, pairs, meter, prompts.clean_advice(advice))
        elif role == "claude":
            run, shown = _rank_claude(settings, brand or {}, playbook_text, shown, renders_by,
                                      credits_by, pairs, meter, prompts.clean_advice(advice))
        else:
            raise ValueError(f"unknown judge role {role!r}")
    except Exception as exc:  # noqa: BLE001 - ranking is advisory; fall back to the input order
        log.warning("ranking failed before it started (%s); keeping the input order", exc)
        return ids
    if len(shown) < 2:
        return ids
    passes = [list(shown), list(reversed(shown))]
    results = util.parallel_map(lambda job: run(job[1], f"rank pass {job[0] + 1}"),
                                list(enumerate(passes)), workers=2, name="clip-engine-rank")
    rankings = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            log.warning("ranking pass %d failed: %s", i + 1, result)
            continue
        # Missing drafts go after the ranked ones, in input order.
        rankings.append(result + [str(d["id"]) for d in shown if str(d["id"]) not in result])
    if not rankings:
        log.warning("ranking failed in both passes; keeping the input order")
        return ids
    merged = borda(rankings, tiebreak={k: v for k, v in scores_by.items() if v is not None})
    return merged + [i for i in ids if i not in merged]


# ---- revision check -----------------------------------------------------------------------

def _pair_credits(credits: Any) -> tuple[str, str]:
    """One credits line per version. A single string serves both; a pair
    (a, b) or a dict {"a", "b"} (what engine.run passes) gives each its own,
    since a revision can put the same person at a different clip time."""
    if isinstance(credits, dict):
        return str(credits.get("a") or ""), str(credits.get("b") or "")
    if isinstance(credits, (list, tuple)) and len(credits) == 2:
        return str(credits[0] or ""), str(credits[1] or "")
    return str(credits or ""), str(credits or "")


def _choice(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("1", "one", "version 1", "v1"):
        return "1"
    if text in ("2", "two", "version 2", "v2"):
        return "2"
    return "same"


def prefer(*, role, settings, brand, a, b, render_a, render_b, credits, intent, meter,
           advice: str = "") -> str:
    """"b" when the revision ``b`` beats the original ``a`` in BOTH
    presentation orders; "a" otherwise (any "same", disagreement or failure).
    ``advice`` (the operator's steer for this pass) goes into the judge's
    system prompt."""
    credit_a, credit_b = _pair_credits(credits)
    versions = {"a": (a or {}, render_a, credit_a), "b": (b or {}, render_b, credit_b)}
    advice = prompts.clean_advice(advice)
    try:
        if role == "gemini":
            ask = _prefer_gemini(settings, brand or {}, versions, intent, meter, advice)
        elif role == "claude":
            ask = _prefer_claude(settings, brand or {}, versions, intent, meter, advice)
        else:
            raise ValueError(f"unknown judge role {role!r}")
    except Exception as exc:  # noqa: BLE001 - no verdict keeps the original
        log.warning("revision check could not run (%s); keeping the original", exc)
        return "a"
    orders = [("a", "b"), ("b", "a")]
    results = util.parallel_map(lambda order: ask(order), orders, workers=2,
                                name="clip-engine-prefer")
    choices = []
    for order, result in zip(orders, results):
        if isinstance(result, BaseException):
            log.warning("revision check (%s first) failed: %s; keeping the original",
                        order[0], result)
            return "a"
        choices.append(result)
    # Order (a, b): the revision is version 2. Order (b, a): it is version 1.
    verdict = "b" if choices == ["2", "1"] else "a"
    log.info("revision check: %s / %s -> keep %s", choices[0], choices[1],
             "the revision" if verdict == "b" else "the original")
    return verdict


def _prefer_gemini(settings, brand, versions, intent, meter, advice=""):
    fps = _float(settings, "clip_engine.gemini.critic_fps", 2.0)
    videos = list(gemini.video_parts([{"path": versions[k][1], "fps": fps} for k in ("a", "b")]))
    video = {"a": videos[0], "b": videos[1]}
    system = prompts.prefer_system(brand, mode="video", intent=intent or "", advice=advice)
    model = model_for(settings, "gemini", "judge")

    def ask(order: tuple[str, str]) -> str:
        parts = []
        for n, key in enumerate(order, start=1):
            parts += [gemini.text_part(f"Version {n} — Credits: {versions[key][2] or 'none'}"),
                      video[key]]
        parts.append(gemini.text_part("Which version would you post?"))
        data = _metered(meter, "prefer", lambda: gemini.generate_json(
            model, system=system, parts=parts, schema=PREFER_SCHEMA,
            thinking=opt(settings, "clip_engine.gemini.judge_thinking", "MEDIUM"),
            media_resolution=opt(settings, "clip_engine.gemini.critic_media_resolution",
                                 "MEDIA_RESOLUTION_LOW"),
            max_output_tokens=8192, label=f"prefer {order[0]}-first", bucket=_bucket(meter)))
        return _choice(data.get("choice"))

    return ask


def _prefer_claude(settings, brand, versions, intent, meter, advice=""):
    interval = _float(settings, "clip_engine.claude.critic_frame_interval", 1.0)
    width = int(_float(settings, "clip_engine.claude.frame_width", 448))
    blocks = {}
    effective = interval
    for key in ("a", "b"):
        frames = media.clip_frames(versions[key][1], interval=interval, width=width,
                                   max_frames=_COLD_MAX_FRAMES)
        if not frames:
            raise media.MediaError(f"no frames from {versions[key][1]}")
        blocks[key] = list(claude.frames_blocks(frames, clock="clip"))
        seconds = sum(max(0.0, float(s["end"]) - float(s["start"]))
                      for s in versions[key][0].get("segments") or [])
        effective = max(effective, seconds / _COLD_MAX_FRAMES if seconds else interval)
    system = prompts.prefer_system(brand, mode="frames", frame_interval=effective,
                                   intent=intent or "", advice=advice)
    model = model_for(settings, "claude", "judge")

    def ask(order: tuple[str, str]) -> str:
        content = []
        for n, key in enumerate(order, start=1):
            draft, _render, credit = versions[key]
            words = words_lines(draft.get("clip_words")) or "(none)"
            content.append(claude.text_block(
                f"Version {n} — Credits: {credit or 'none'}\nWords heard (clip seconds):\n{words}"))
            content += blocks[key]
        content.append(claude.text_block("Which version would you post?"))
        data = _metered(meter, "prefer", lambda: claude.json_call(
            model, system=system, content=content, schema=PREFER_SCHEMA,
            effort=opt(settings, "clip_engine.claude.judge_effort", "medium"), max_tokens=8000,
            label=f"prefer {order[0]}-first", bucket=_bucket(meter)))
        return _choice(data.get("choice"))

    return ask
