"""Drafting and revising candidate clips.

The editor is asked for real alternatives (different moments, and the same
moment cut different ways: single take vs tightened, straight vs cold open)
because a critic watches every draft cold afterwards and only the strongest
reach the operator. It first lists the distinct moments worth posting, so a
miss can later be told apart as "never found" versus "found but cut badly".

What the editor reads is a JSON payload: the shot log (compact, with the
exact shot changes inside every beat so a picture-led segment can start on a
cut), word-accurate transcript entries, the account's history and the ranges
other clips already use. A Gemini editor also watches the WHOLE source proxy
when it is short enough (never a window proxy: its clock would start at 0 and
every time would land in the wrong place); a Claude editor gets keyframes of
the strongest shots instead. The payload is always the last part of a Gemini
request — after the video, as Google recommends, and never dropped.

Segments come back in PLAY order (a cold open plays a later moment first) and
are kept that way: ``clean_drafts`` clamps and de-overlaps without sorting and
removes used footage piece by piece.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import re
from typing import Any

from . import claude, gemini, media, model_for, opt, prompts, util
from . import shotlog as _shotlog

log = logging.getLogger("clip_engine.editor")

LEADS = ("word", "picture")
MOMENT_KINDS = ("person", "picture", "line", "sound")

_MAX_ENTRIES = 600              # transcript entries in one payload
_DESCRIPTION_CHARS = 2000
_SECTION_MARGIN = 30.0          # transcript context around a section
_REVISE_MARGIN = 60.0           # shot log / transcript around a draft being revised
_LONG_SOURCE_SECONDS = 1200.0
_MIN_SEGMENT = 1.0              # the snapper's own minimum
_KEYFRAME_SPACING = 1.5
_KEYFRAME_AFTER_CUT = 0.15      # land on the new shot, not the transition frame
_STRINGS = {"story": 200, "hook": 300, "payoff": 300, "why": 300, "risk": 300,
            "draft_caption": 280, "challenges_rules": 300}

_SEGMENT = {
    "type": "object",
    "properties": {
        "start": {"type": "number", "description": "seconds on the source clock"},
        "end": {"type": "number", "description": "seconds on the source clock"},
        "lead": {"type": "string", "enum": list(LEADS)},
    },
    "required": ["start", "end", "lead"],
}

DRAFT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "kind": {"type": "string", "enum": list(MOMENT_KINDS)},
                    "why": {"type": "string"},
                },
                "required": ["id", "start", "end", "kind", "why"],
            },
        },
        "drafts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "moment_id": {"type": "string"},
                    "archetype": {"type": "string"},
                    "lens": {"type": "string", "enum": list(prompts.LENSES)},
                    "story": {"type": "string"},
                    "story_key": {"type": "string"},
                    "hook": {"type": "string"},
                    "payoff": {"type": "string"},
                    "segments": {"type": "array", "items": _SEGMENT},
                    "why": {"type": "string"},
                    "risk": {"type": "string"},
                    "draft_caption": {"type": "string"},
                    "challenges_rules": {"type": "string"},
                },
                "required": ["id", "moment_id", "archetype", "lens", "story", "story_key", "hook",
                             "payoff", "segments", "why", "risk", "draft_caption",
                             "challenges_rules"],
            },
        },
    },
    "required": ["moments", "drafts"],
}

REVISE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "change": {"type": "string",
                   "description": "what you changed and why, or why the draft stays as it is"},
        "archetype": {"type": "string"},
        "segments": {"type": "array", "items": _SEGMENT},
    },
    "required": ["change", "archetype", "segments"],
}


# ---- small helpers ------------------------------------------------------------------------

def _text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip() if len(text) > limit else text


def _slug(value: Any, limit: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")[:limit].strip("-")


def _bucket(meter) -> str | None:
    """The meter's spend bucket; None (no meter) lets the provider use the
    process default, else "engine" — never the shared LLM budget."""
    return getattr(meter, "bucket", None) or None


def _add_usage(meter, stage: str, usage: dict | None) -> None:
    if meter is not None and usage:
        meter.add(stage, usage)


def _metered(meter, stage: str, call):
    """Run a provider call, charging its usage to ``meter`` whether it
    succeeds or fails (a billed failure still costs money)."""
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


def _window(window: Any) -> dict | None:
    bounds = _shotlog._bounds(window)
    return None if bounds is None else {"start": bounds[0], "end": bounds[1]}


def _limits(window: dict | None, duration: float | None) -> tuple[float, float | None]:
    """(floor, horizon) every segment must fall inside."""
    if window:
        return window["start"], window["end"]
    seconds = _shotlog._positive(duration)
    return 0.0, seconds


def _time(value: Any) -> float | None:
    """A model's time as seconds. Numbers pass through even when negative (a
    start a beat before 0:00 is clamped to the floor later, not discarded);
    strings go through ``shotlog.parse_ts`` ("01:15.5", "75.5s")."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return _shotlog.parse_ts(value)


def _seconds(segments: list[dict]) -> float:
    return sum(s["end"] - s["start"] for s in segments)


def _cuts(shot_map: Any) -> list:
    return list((shot_map or {}).get("cuts") or []) if isinstance(shot_map, dict) else []


def _min_shot_score(settings) -> float:
    return _float(settings, "clip_engine.snap.min_shot_score", 20.0)


_ARCHETYPE_LINE = re.compile(r"^- ([a-z0-9_]+) — ", re.MULTILINE)


def _archetype_keys(playbook_text: str) -> list[str]:
    """The archetype keys in a rendered playbook (``playbook.render`` lists
    each as "- <key> — <name>: ..." before the flops section). Falls back to
    the playbook on disk."""
    archetype_section = (playbook_text or "").split("\nWHAT FLOPS HERE:", 1)[0]
    keys = list(dict.fromkeys(_ARCHETYPE_LINE.findall(archetype_section)))
    if keys:
        return keys
    from . import playbook

    return playbook.keys(playbook.load())


def _schema_for(base: dict, archetypes: list[str], *, max_drafts: int | None = None) -> dict:
    """``base`` with the archetype enum filled in (and a cap on drafts):
    a constrained field is one the model can't get wrong."""
    schema = copy.deepcopy(base)
    allowed = list(dict.fromkeys([*archetypes, "other"]))
    if "drafts" in schema["properties"]:
        drafts = schema["properties"]["drafts"]
        drafts["items"]["properties"]["archetype"]["enum"] = allowed
        if max_drafts:
            drafts["maxItems"] = int(max_drafts)
    else:
        schema["properties"]["archetype"]["enum"] = allowed
    return schema


def _entries_near(entries, lo: float, hi: float) -> list[dict]:
    out = []
    for e in entries or []:
        try:
            if float(e["end"]) > lo and float(e["start"]) < hi:
                out.append(e)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


# ---- transcript entries -------------------------------------------------------------------

def entries_for(words, transcript) -> tuple[list[dict], bool]:
    """The transcript the editor cuts against, and whether it is word-accurate.

    With a Whisper word stream: clause-sized entries that start and end on
    word boundaries (coarsened, never truncated, to 600). Without one:
    the transcript's own entries (rolling captions, approximate times)."""
    from .. import llm

    if words:
        clauses = llm._words_to_clauses(list(words))
        if clauses:
            return llm._fit_entries(clauses, _MAX_ENTRIES), True
    entries = []
    for item in transcript or []:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if text and end >= start:
            entries.append({"start": round(start, 1), "end": round(end, 1), "text": text})
    return llm._fit_entries(entries, _MAX_ENTRIES), False


# ---- keyframes ----------------------------------------------------------------------------

def _keyframe_times(shotlog: dict, shot_map: Any, *, min_score: float, limit: int) -> list[float]:
    """When to grab stills: the first shot change inside every beat with
    visual >= 4 or a famous/public-figure speaker (strongest first), then the
    highlights; spaced at least 1.5 s apart, at most ``limit``."""
    stars = [p for p in shotlog.get("people") or []
             if p.get("prominence") in ("famous", "public_figure")]
    cuts = _shotlog._cut_times(_cuts(shot_map), min_score)
    picks: list[tuple[int, float, float]] = []   # (priority, -visual, t)
    for beat in shotlog.get("beats") or []:
        starred = any(_shotlog._names(beat, p) for p in stars)
        if beat.get("visual", 0) < 4 and not starred:
            continue
        inside = [t for t in cuts if beat["start"] <= t < beat["end"]]
        if inside:
            t = inside[0] + _KEYFRAME_AFTER_CUT
        else:
            t = beat["start"] + min(0.5, (beat["end"] - beat["start"]) / 2)
        picks.append((0, -(beat.get("visual", 0) + (2 if starred else 0)), t))
    for item in shotlog.get("highlights") or []:
        picks.append((1, 0, item["start"] + min(0.5, (item["end"] - item["start"]) / 2)))
    chosen: list[float] = []
    for _, _, t in sorted(picks, key=lambda p: (p[0], p[1], p[2])):
        if len(chosen) >= limit:
            break
        if all(abs(t - c) >= _KEYFRAME_SPACING for c in chosen):
            chosen.append(round(t, 2))
    return sorted(chosen)


def keyframes(source, shotlog, shot_map, *, settings, limit) -> list[tuple[float, bytes]]:
    """Stills of the strongest shots for an editor that can't watch the video
    (one per notable beat, then highlights), at ``frame_width``. Best effort:
    a still that can't be grabbed is skipped; never raises."""
    if not source or not shotlog or int(limit or 0) <= 0:
        return []
    try:
        times = _keyframe_times(shotlog, shot_map, min_score=_min_shot_score(settings),
                                limit=int(limit))
    except Exception as exc:  # noqa: BLE001 - keyframes are an optional extra
        log.warning("keyframe selection failed: %s", exc)
        return []
    width = int(_float(settings, "clip_engine.claude.frame_width", 448))

    def grab(t: float):
        frames = media.sample_frames(source, start=t, end=t + 0.5, interval=1.0, width=width)
        return frames[0] if frames else None

    results = util.parallel_map(grab, times, workers=4, name="clip-engine-keyframes")
    out = [r for r in results if isinstance(r, tuple) and len(r) == 2]
    if len(out) < len(times):
        log.info("keyframes: %d of %d stills grabbed", len(out), len(times))
    return sorted(out, key=lambda f: f[0])


# ---- cleaning -----------------------------------------------------------------------------

def _clean_segments(raw, *, floor: float, horizon: float | None, used, max_segments: int) -> list[dict]:
    """Model segments -> clean play-order segments with a lead."""
    from .. import clip_proposals

    segments = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        start, end = _time(item.get("start")), _time(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        lead = str(item.get("lead") or "").strip().lower()
        segments.append({"start": start, "end": end, "lead": lead if lead in LEADS else "word"})
    segments = clip_proposals.clean_play_order(segments, min_len=_MIN_SEGMENT, floor=floor,
                                               horizon=horizon)
    segments = clip_proposals.subtract_play_order(segments, list(used or []))
    out = [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
            "lead": s.get("lead") if s.get("lead") in LEADS else "word"} for s in segments]
    cap = max(1, int(max_segments or 1))
    if len(out) > cap:
        # Keep the opening (the hook) and the last segment (the payoff); the
        # middle is what a tighter cut gives up first.
        out = out[:cap - 1] + out[-1:] if cap > 1 else out[:1]
    return out


def _within_length(segments: list[dict], min_s: float, max_s: float) -> bool:
    seconds = _seconds(segments)
    return 0.8 * float(min_s) - 1e-6 <= seconds <= 1.3 * float(max_s) + 1e-6


def clean_drafts(raw, *, floor, horizon, min_s, max_s, max_segments, used, archetypes) -> list[dict]:
    """The editor's drafts, cleaned, in the editor's own order.

    Segments keep PLAY order (clamped to [floor, horizon], overlap with an
    earlier segment removed, used footage subtracted), each with a lead
    ("word" by default), at most ``max_segments``; drafts outside
    0.8·min_s..1.3·max_s are dropped. Ids are d1, d2, ... and ``editor_rank``
    is the position. Unknown archetypes become "other", unknown lenses
    "line"."""
    allowed = {str(a).strip().lower() for a in archetypes or [] if str(a).strip()}
    out: list[dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        segments = _clean_segments(item.get("segments"), floor=float(floor or 0.0),
                                   horizon=horizon, used=used, max_segments=max_segments)
        if not segments or not _within_length(segments, min_s, max_s):
            continue
        rank = len(out)
        archetype = str(item.get("archetype") or "").strip().lower()
        lens = str(item.get("lens") or "").strip().lower()
        story = _text(item.get("story"), _STRINGS["story"])
        draft = {
            "id": f"d{rank + 1}",
            "moment_id": _text(item.get("moment_id"), 20),
            "archetype": archetype if archetype in allowed else "other",
            "lens": lens if lens in prompts.LENSES else "line",
            "story": story,
            "story_key": _slug(item.get("story_key")) or _slug(story, 40) or f"story-{rank + 1}",
            "hook": _text(item.get("hook"), _STRINGS["hook"]),
            "payoff": _text(item.get("payoff"), _STRINGS["payoff"]),
            "segments": segments,
            "why": _text(item.get("why"), _STRINGS["why"]),
            "risk": _text(item.get("risk"), _STRINGS["risk"]),
            "draft_caption": _text(item.get("draft_caption"), _STRINGS["draft_caption"]),
            "challenges_rules": _text(item.get("challenges_rules"), _STRINGS["challenges_rules"]),
            "origin": "draft",
            "parent": None,
            "editor_rank": rank,
        }
        out.append(draft)
    return out


def _clean_moments(raw, *, floor: float, horizon: float | None) -> list[dict]:
    """The editor's moments: times clamped, ids unique (the model's own ids
    are kept when usable, since drafts refer to them)."""
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        start, end = _time(item.get("start")), _time(item.get("end"))
        if start is None or end is None:
            continue
        start = max(float(floor or 0.0), start)
        if horizon is not None:
            end = min(float(horizon), end)
        if end <= start:
            continue
        mid = _text(item.get("id"), 20)
        if not mid or mid in seen:
            mid = f"m{len(out) + 1}"
            while mid in seen:
                mid += "x"
        seen.add(mid)
        kind = str(item.get("kind") or "").strip().lower()
        out.append({"id": mid, "start": round(start, 2), "end": round(end, 2),
                    "kind": kind if kind in MOMENT_KINDS else "line",
                    "why": _text(item.get("why"), 300)})
    return out


def clip_map(segments) -> list[dict]:
    """Where each segment sits in CLIP time and in SOURCE time, in play order,
    so a reviser can translate the critic's clip-clock notes."""
    out, clock = [], 0.0
    for seg in segments or []:
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        entry = {"clip_start": round(clock, 2), "clip_end": round(clock + end - start, 2),
                 "source_start": round(start, 2), "source_end": round(end, 2)}
        if seg.get("lead") in LEADS:
            entry["lead"] = seg["lead"]
        out.append(entry)
        clock += end - start
    return out


# ---- drafting -----------------------------------------------------------------------------

def _payload(*, title, description, duration, window, shotlog, shot_map, entries, context,
             used, settings) -> dict:
    payload: dict[str, Any] = {
        "title": title or "",
        "description": (description or "")[:_DESCRIPTION_CHARS],
        "duration": round(float(duration), 1) if _shotlog._positive(duration) else None,
    }
    if window:
        payload["section"] = window
    payload["shot_log"] = _shotlog.compact_for_prompt(
        shotlog, window=window, cuts=_cuts(shot_map), min_shot_score=_min_shot_score(settings))
    if window:
        entries = _entries_near(entries, window["start"] - _SECTION_MARGIN,
                                window["end"] + _SECTION_MARGIN)
    payload["transcript"] = list(entries or [])
    if context:
        payload["account_context"] = context
    payload["used_ranges"] = list(used or [])
    return payload


def _gemini_fps(settings, duration) -> float:
    seconds = _shotlog._positive(duration) or 0.0
    if seconds <= _LONG_SOURCE_SECONDS:
        return 1.0
    return _float(settings, "clip_engine.gemini.long_video_fps", 0.5)


def _gemini_parts(source_video, fps: float, payload_text: str) -> list[dict]:
    """The video first (when there is one), the payload text ALWAYS last.
    Spelled out on purpose: ``[video] if v else [] + [text]`` would drop the
    payload whenever a video is attached."""
    video = gemini.video_parts([{"path": source_video, "fps": fps}]) if source_video else []
    return list(video) + [gemini.text_part(payload_text)]


def draft(*, role, settings, brand, playbook_text, rules, title, description, duration,
          shotlog, shot_map, entries, has_words, context, used, window, section_topic, n,
          min_s, max_s, max_segments, source_video, keyframe_frames, meter) -> tuple[list[dict], list[dict]]:
    """Draft ``n`` candidate clips. Returns ``(drafts, moments)``.

    ``role`` is the editor's provider ("gemini" or "claude"). Provider errors
    propagate (with their usage charged to ``meter`` under "draft")."""
    if role not in ("gemini", "claude"):
        raise ValueError(f"unknown editor role {role!r}")
    section = _window(window)
    floor, horizon = _limits(section, duration)
    archetypes = _archetype_keys(playbook_text)
    payload_text = _dumps(_payload(title=title, description=description, duration=duration,
                                   window=section, shotlog=shotlog, shot_map=shot_map,
                                   entries=entries, context=context, used=used,
                                   settings=settings))
    schema = _schema_for(DRAFT_SCHEMA, archetypes, max_drafts=n)
    common = dict(n=int(n), min_s=min_s, max_s=max_s, max_segments=int(max_segments),
                  section=section, section_topic=section_topic or "", has_words=bool(has_words))
    label = f"draft {(title or '')[:40]}"
    if role == "gemini":
        system = prompts.editor_system(brand or {}, playbook_text or "", rules or "",
                                       sees_video=bool(source_video), **common)
        parts = _gemini_parts(source_video, _gemini_fps(settings, duration), payload_text)
        data = _metered(meter, "draft", lambda: gemini.generate_json(
            model_for(settings, "gemini", "editor"), system=system, parts=parts, schema=schema,
            thinking=opt(settings, "clip_engine.gemini.editor_thinking", "HIGH"),
            media_resolution=opt(settings, "clip_engine.gemini.media_resolution",
                                 "MEDIA_RESOLUTION_LOW"),
            max_output_tokens=32768, label=label, bucket=_bucket(meter)))
    else:
        system = prompts.editor_system(brand or {}, playbook_text or "", rules or "",
                                       sees_video=False, has_keyframes=bool(keyframe_frames),
                                       **common)
        content = [claude.text_block(payload_text)]
        if keyframe_frames:
            content += [claude.text_block("Keyframes:")] + list(claude.frames_blocks(keyframe_frames))
        data = _metered(meter, "draft", lambda: claude.json_call(
            model_for(settings, "claude", "editor"), system=system, content=content, schema=schema,
            effort=opt(settings, "clip_engine.claude.editor_effort", "high"), max_tokens=32000,
            label=label, bucket=_bucket(meter)))
    drafts = clean_drafts(data.get("drafts"), floor=floor, horizon=horizon, min_s=min_s,
                          max_s=max_s, max_segments=max_segments, used=used, archetypes=archetypes)
    moments = _clean_moments(data.get("moments"), floor=floor, horizon=horizon)
    raw_count = len(data.get("drafts") or []) if isinstance(data.get("drafts"), list) else 0
    log.info("editor (%s) drafted %d clips (%d kept) from %d moments", role, raw_count,
             len(drafts), len(moments))
    return drafts, moments


# ---- revising -----------------------------------------------------------------------------

_DRAFT_FIELDS = ("id", "moment_id", "archetype", "lens", "story", "story_key", "hook", "payoff",
                 "why", "risk", "draft_caption", "challenges_rules", "origin", "parent",
                 "editor_rank")


def _same_cut(a: list[dict], b: list[dict], tolerance: float = 0.05) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x["start"]) - float(y["start"])) <= tolerance
               and abs(float(x["end"]) - float(y["end"])) <= tolerance for x, y in zip(a, b))


def _critique_for_prompt(critique: Any) -> dict:
    if not isinstance(critique, dict):
        return {}
    return {k: v for k, v in critique.items() if k not in ("error", "usage")}


def revise(*, role, settings, brand, playbook_text, rules, draft, critique, shotlog,
           shot_map, entries, has_words, used, window, min_s, max_s, max_segments,
           source_video, meter) -> dict | None:
    """One revision of ``draft`` from the critic's notes.

    Returns a new Draft (id ``<id>r``, origin "revision", parent = the
    original's id, archetype from the answer or the parent) or None when the
    answer leaves the cut unchanged or isn't a valid clip. Provider errors
    propagate (usage charged to ``meter`` under "revise")."""
    if role not in ("gemini", "claude"):
        raise ValueError(f"unknown editor role {role!r}")
    original = [dict(s) for s in draft.get("segments") or []]
    if not original:
        return None
    section = _window(window)
    duration = (shotlog or {}).get("source_seconds")   # None = no known horizon
    floor, horizon = _limits(section, duration)
    span_lo = min(float(s["start"]) for s in original) - _REVISE_MARGIN
    span_hi = max(float(s["end"]) for s in original) + _REVISE_MARGIN
    near = {"start": max(floor, span_lo),
            "end": min(horizon, span_hi) if horizon is not None else span_hi}
    archetypes = _archetype_keys(playbook_text)
    payload = {
        "draft": {**{k: draft.get(k) for k in ("story", "archetype", "lens", "hook", "payoff",
                                               "why", "risk")},
                  "segments": original, "clip_map": clip_map(original)},
        "critique": _critique_for_prompt(critique),
        "shot_log": _shotlog.compact_for_prompt(shotlog, window=near, cuts=_cuts(shot_map),
                                                min_shot_score=_min_shot_score(settings)),
        "transcript": _entries_near(entries, near["start"], near["end"]),
        "used_ranges": list(used or []),
    }
    if section:
        payload["section"] = section
    payload_text = _dumps(payload)
    schema = _schema_for(REVISE_SCHEMA, archetypes)
    label = f"revise {draft.get('id', '')}"
    common = dict(min_s=min_s, max_s=max_s, max_segments=int(max_segments),
                  has_words=bool(has_words))
    if role == "gemini":
        system = prompts.revise_system(brand or {}, playbook_text or "", rules or "",
                                       sees_video=bool(source_video), **common)
        # The WHOLE source proxy only (source clock), never a window proxy.
        parts = _gemini_parts(source_video, _gemini_fps(settings, duration), payload_text)
        data = _metered(meter, "revise", lambda: gemini.generate_json(
            model_for(settings, "gemini", "editor"), system=system, parts=parts, schema=schema,
            thinking=opt(settings, "clip_engine.gemini.editor_thinking", "HIGH"),
            media_resolution=opt(settings, "clip_engine.gemini.media_resolution",
                                 "MEDIA_RESOLUTION_LOW"),
            max_output_tokens=16384, label=label, bucket=_bucket(meter)))
    else:
        system = prompts.revise_system(brand or {}, playbook_text or "", rules or "",
                                       sees_video=False, **common)
        data = _metered(meter, "revise", lambda: claude.json_call(
            model_for(settings, "claude", "editor"), system=system,
            content=[claude.text_block(payload_text)], schema=schema,
            effort=opt(settings, "clip_engine.claude.editor_effort", "high"), max_tokens=16000,
            label=label, bucket=_bucket(meter)))
    segments = _clean_segments(data.get("segments"), floor=floor, horizon=horizon, used=used,
                               max_segments=max_segments)
    change = _text(data.get("change"), 300)
    if not segments or not _within_length(segments, min_s, max_s):
        log.info("revision of %s rejected (no valid cut): %s", draft.get("id"), change)
        return None
    if _same_cut(segments, original):
        log.info("revision of %s left the cut unchanged: %s", draft.get("id"), change)
        return None
    archetype = str(data.get("archetype") or "").strip().lower()
    revised = {k: draft.get(k) for k in _DRAFT_FIELDS}
    revised.update({
        "id": f"{draft.get('id', 'd')}r",
        "archetype": archetype if archetype in archetypes else (draft.get("archetype") or "other"),
        "segments": segments,
        "origin": "revision",
        "parent": draft.get("id"),
        "change": change,
    })
    return revised
