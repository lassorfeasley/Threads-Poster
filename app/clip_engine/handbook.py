"""House rules learned from the operator's past edits.

The playbook says what kinds of clip work for this account; it can't say how
THIS operator cuts. He has made ~430 edits, and they encode choices no prompt
spells out: he never opens on an anchor, never keeps a non-English interview
without an English voiceover, opens on the disaster picture and then names
the place. The first live run of the engine broke exactly those habits. So
this module reads his past edits and writes them down as rules the editor,
critic and ranker can follow:

1. ``examples`` — every exported cut (an exported file or a published post)
   whose source and transcript are still available, weighted by how much it
   says about his taste: published 1.0, exported but never posted 0.5, and a
   cut that started from an accepted transcript-suggester proposal 0.3 (its
   boundaries are partly the suggester's). A deterministic 20% of VIDEOS
   (by sha1 of the video id) is held out and never analysed, so ``validate``
   can test the rules on edits they were not learned from.
2. ``analyse`` — one Gemini call per cut, text only: the video's shot log
   (cached; built once per video), the transcript, his segments in play order
   and the stretches he skipped, with their beats. It DESCRIBES the choices
   with timestamps (what the clip opens on, what was kept and dropped, the
   structure, which languages were heard) and never guesses at motives. Each
   description is cached per cut and prompt.
3. ``synthesize`` — one Claude call over the compact descriptions writes
   imperative rules with their support and exceptions. Rules with support >= 5
   that hold at least 70% of the time are "rules"; the rest "tendencies".
4. ``validate`` — replays held-out videos through the engine with the
   current rules and with the current rules plus the candidates, and reports
   moment-hit@1/@3 for both.

Rules learned here are NOT a separate rule store. ``propose`` pushes the
candidates into the same proposal queue as rules drawn from the operator's
notes (``notes.add_proposals(source="handbook")``); on the Brand & audience
page he adds or dismisses them like any other suggestion, and an added one
is an ordinary brand rule the engine already reads. Nothing changes the
engine's behaviour until he does. ``render_active`` stays for compatibility
and returns "".

config/clip_handbook.yaml is the analysis REPORT (when it was built, how many
edits, the holdout, every candidate with its support, exceptions and example
cuts, the validation) shown on the Clip handbook page. Caches live under
``engine_dir()/handbook``. Reading the database happens only in ``examples``
(a writer session, rolled back there).
"""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import yaml

from . import engine_dir, model_for, opt, util

log = logging.getLogger("clip_engine.handbook")

VERSION = 1

WEIGHT_PUBLISHED = 1.0
WEIGHT_EXPORTED = 0.5
WEIGHT_SEEDED = 0.3
HOLDOUT_BUCKETS = 5            # 1 in 5 videos (20%) is held out
RULE_MIN_SUPPORT = 5
RULE_MIN_SHARE = 0.7
MIN_EDITS = 10                 # fewer analysed edits than this: nothing to generalise from
TEXT_LIMIT = 200
SLIVER_S = 0.3
MAX_EXAMPLE_IDS = 8
SKIP_CONTEXT_S = 90.0          # skipped stretches shown this far around the kept material
MAX_SKIPPED_BEATS = 12
OPENING_S = 60.0              # the start of the source is always shown, skipped or not
PROPOSE_MIN_SUPPORT = 3        # a candidate seen fewer times than this isn't proposed
HEADER_TITLE = "HOUSE RULES LEARNED FROM YOUR PAST EDITS:"

KINDS = ("anchor_intro", "studio_toss", "reporter_standup", "non_english_interview",
         "interview", "b_roll", "aerial", "action", "graphic", "sign_off", "number",
         "witness", "famous_person", "natural_sound", "other")
ENDING_KINDS = ("line_lands", "reaction", "action", "number", "sign_off", "natural_sound",
                "mid_sentence", "other")
STRENGTHS = ("rule", "tendency")

_KIND_FIELD = {"type": "string", "enum": list(KINDS)}

EDIT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "opening": {
            "type": "object",
            "properties": {
                "kind": _KIND_FIELD,
                "what": {"type": "string",
                         "description": "what the first ~3 s of the clip show and say, with source times"},
                "beat_shot": {"type": "string",
                              "description": "the shot type of the shot-log beat the clip opens on"},
                "speaker": {"type": "string",
                            "description": "who is speaking at the open; empty when nobody"},
                "why_visible": {"type": "string",
                                "description": "what a viewer can see or hear in that first moment "
                                               "(observable; not a guess at the editor's motive)"},
            },
            "required": ["kind", "what", "beat_shot", "speaker", "why_visible"],
        },
        "kept": {"type": "array", "items": {
            "type": "object",
            "properties": {"what": {"type": "string"}, "kind": _KIND_FIELD},
            "required": ["what", "kind"]}},
        "dropped": {"type": "array", "items": {
            "type": "object",
            "properties": {"what": {"type": "string"}, "kind": _KIND_FIELD},
            "required": ["what", "kind"]}},
        "structure": {
            "type": "object",
            "properties": {
                "cold_open": {"type": "boolean"},
                "segments": {"type": "integer"},
                "seconds": {"type": "number"},
                "ending_kind": {"type": "string", "enum": list(ENDING_KINDS)},
            },
            "required": ["cold_open", "segments", "seconds", "ending_kind"],
        },
        "language_choices": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["opening", "kept", "dropped", "structure", "language_choices", "notes"],
}

RULES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rules": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "short lowercase slug"},
                "text": {"type": "string",
                         "description": "one imperative instruction an editor can follow, "
                                        "<= 200 chars, WITHOUT the condition"},
                "when": {"type": "string",
                         "description": "when it applies, in plain words (a channel, a source "
                                        "format, an archetype), or 'Always'"},
                "support": {"type": "integer",
                            "description": "how many records follow it"},
                "exceptions": {"type": "integer",
                               "description": "records where the condition held and the edit did otherwise"},
                "examples": {"type": "array", "items": {"type": "integer"},
                             "description": "ids of up to 5 supporting records"},
                "strength": {"type": "string", "enum": list(STRENGTHS)},
            },
            "required": ["id", "text", "when", "support", "exceptions", "examples", "strength"],
        }},
        "summary": {"type": "string"},
    },
    "required": ["rules", "summary"],
}


class HandbookError(ValueError):
    """A handbook operation that can't go ahead: too few edits, nothing to
    validate, a model answer that doesn't hold together. Says what to do."""


class LowDisk(HandbookError):
    """Free disk fell below ``clip_engine.min_free_gb``: stop the batch."""


# ---- prompts -------------------------------------------------------------------------

def edit_system(brand: dict | None = None) -> str:
    from . import prompts

    return (
        "You are studying how one human editor cut a clip for a social account, so his habits "
        "can be written down. " + prompts.account_lines(brand or {}) + "\n\n"
        "You get the source video's shot log (time-coded beats: shot type, what is on screen, "
        "who speaks, the line, energy/visual/topic scores), its transcript, the editor's "
        "segments in PLAY order (source seconds, and when each plays in the clip), and the "
        "stretches of the source he skipped near the clip and at its very start, with their "
        "beats.\n\n"
        "Describe what he CHOSE, with source timestamps. Never guess at why; never invent a "
        "motive. Only say what the shot log and transcript show.\n"
        f"- kinds come from this list: {', '.join(KINDS)}. non_english_interview = speech in "
        "another language with no English voiceover; famous_person = a named, widely known "
        "person; number = a statistic or figure lands; witness = an ordinary person's "
        "first-hand account.\n"
        "- opening: what plays in the clip's first ~3 seconds — kind, what (concrete, with the "
        "source time), beat_shot (the shot type of the beat it opens on), speaker (who speaks "
        "then, or \"\"), why_visible (what a viewer sees or hears in that first moment).\n"
        "- kept: each distinct piece of the clip, in play order, with times.\n"
        "- dropped: notable material he left out — near the clip and at the start of the "
        "source (anchor intros, studio tosses, reporter standups, other-language speech, "
        "graphics, sign-offs) — with times. Leave out filler nobody would consider.\n"
        "- structure: cold_open (does a later moment play before an earlier one), segments, "
        f"seconds, ending_kind (one of {', '.join(ENDING_KINDS)}).\n"
        "- language_choices: which languages are heard in kept vs dropped material and "
        "whether an English voiceover carried any other language; \"\" if everything is "
        "English.\n"
        "- notes: anything else observable about the edit, at most two sentences."
    )


def synth_system(brand: dict | None = None) -> str:
    from . import prompts

    return (
        "You are writing the house rules of one human editor from descriptions of his past "
        "edits, so an editing model can cut the way he does. "
        + prompts.account_lines(brand or {}) + "\n\n"
        "Each record describes one clip he cut: what it opens on, what he kept and dropped "
        "(with kinds), its structure, the languages heard, the channel and format of the "
        "source, whether it was published and how it performed (percentile among posts from "
        "the same weeks), and a weight: published clips 1.0, exported but never posted 0.5, "
        "clips that started from an automatic suggestion 0.3. Weigh records by it.\n\n"
        "Write rules an editor can follow on a new video:\n"
        f"- text: ONE imperative instruction, concrete, at most {TEXT_LIMIT} characters, "
        "WITHOUT its condition (\"Open on the aerial b-roll, not the anchor\"; not \"make it "
        "engaging\"). It must read on its own once the condition is put in front of it.\n"
        "- when: the condition in plain words — a channel (\"France 24 packages\"), a source "
        "format, an archetype, a situation (\"an interview in another language\") — or "
        "\"Always\".\n"
        "- support: how many records follow the rule; exceptions: how many records where the "
        "condition held and he did otherwise. Count from the records honestly; never round up.\n"
        "- examples: the ids of up to 5 supporting records.\n"
        f"- strength: \"rule\" when support >= {RULE_MIN_SUPPORT} and it holds at least "
        f"{int(RULE_MIN_SHARE * 100)}% of the time, else \"tendency\".\n"
        "Prefer rules about openings, what gets cut, language, length and structure. State "
        "the pattern, never a motive. Don't restate one rule in two ways. summary: two to "
        "four sentences on his style."
    )


def prompt_hash() -> str:
    """What an analysis depends on besides its inputs: the prompt and schema."""
    return util.sha1(VERSION, edit_system({}), json.dumps(EDIT_SCHEMA, sort_keys=True), n=10)


# ---- small helpers -------------------------------------------------------------------

def _text(value: Any, limit: int) -> str:
    flat = " ".join(str(value or "").split())
    return flat if len(flat) <= limit else flat[:limit].rstrip()


def _int(value: Any, default: int = 0) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0, int(round(number)))


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.isoformat()
    return str(value)


def _parse_time(value) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _is_file(path) -> bool:
    try:
        return bool(path) and Path(path).is_file()
    except OSError:
        return False


def segments_of(raw) -> list[dict]:
    """Cut.trim_segments (JSON text or a list) in PLAY order; invalid entries
    and slivers shorter than 0.3 s dropped."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except (TypeError, ValueError):
            return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end - start >= SLIVER_S:
            out.append({"start": round(start, 2), "end": round(end, 2)})
    return out


def is_holdout(video_id: str) -> bool:
    """Deterministic 20% of videos, by sha1 of the video id."""
    return int(util.sha1("handbook-holdout", video_id, n=8), 16) % HOLDOUT_BUCKETS == 0


def weight_for(*, published: bool, seeded: bool) -> float:
    """Published 1.0, exported-unpublished 0.5; a cut that started from an
    accepted transcript-suggester proposal says less about the operator's own
    taste (its boundaries are partly the suggester's), so 0.3 either way."""
    if seeded:
        return WEIGHT_SEEDED
    return WEIGHT_PUBLISHED if published else WEIGHT_EXPORTED


# ---- 1. examples -------------------------------------------------------------------

def examples(session, *, limit: int | None = None, holdout: bool = False) -> list[dict]:
    """The operator's exported cuts that can be studied, as plain dicts, best
    evidence first (weight, then percentile, then cut id).

    Exported = an exported file (``trimmed_clip_path``) or a published post.
    Studied only when the source can be had (on disk, or in storage with a
    local path to restore to) and its transcript file is on this machine; the
    brand's own footage is left out. ``holdout=True`` returns ONLY the
    held-out videos' cuts (``is_holdout``); False only the rest. Read-only:
    the session is rolled back before returning."""
    from sqlalchemy import select

    from ..categories import is_first_party
    from ..models import Candidate, Channel, ClipProposal, Cut, ThreadsPost

    try:
        rows = session.execute(
            select(Cut.id, Cut.candidate_pk, Cut.trim_segments, Cut.trimmed_clip_path,
                   Cut.clip_title, Cut.format_tags, Cut.created_at,
                   Candidate.video_id, Candidate.title, Candidate.category,
                   Candidate.duration_seconds, Candidate.local_video_path,
                   Candidate.source_object_key, Candidate.transcript_path,
                   Candidate.word_transcript_path, Candidate.archived_at,
                   Candidate.format_tags.label("candidate_format_tags"),
                   Channel.call_sign, Channel.first_party)
            .join(Candidate, Cut.candidate_pk == Candidate.id)
            .outerjoin(Channel, Candidate.channel_pk == Channel.id)
            .order_by(Cut.id)
        ).all()
        cut_ids = [r.id for r in rows]
        published: dict[int, list] = {}
        seeded: set[int] = set()
        for start in range(0, len(cut_ids), 500):
            chunk = cut_ids[start:start + 500]
            for p in session.execute(
                    select(ThreadsPost.id, ThreadsPost.cut_pk, ThreadsPost.format_tags)
                    .where(ThreadsPost.cut_pk.in_(chunk), ThreadsPost.status == "published")
                    .order_by(ThreadsPost.published_at, ThreadsPost.id)).all():
                published.setdefault(p.cut_pk, []).append(p)
            seeded |= set(session.execute(
                select(ClipProposal.cut_pk)
                .where(ClipProposal.cut_pk.in_(chunk),
                       ClipProposal.verdict == ClipProposal.VERDICT_ACCEPTED)).scalars().all())
        scores = _performance(session)
    finally:
        session.rollback()

    out = []
    for r in rows:
        is_published = r.id in published
        if not ((r.trimmed_clip_path or "").strip() or is_published):
            continue
        source = SimpleNamespace(channel=SimpleNamespace(first_party=bool(r.first_party)),
                                 category=r.category or "")
        if is_first_party(source):
            continue
        if not _is_file(r.transcript_path):
            continue
        local, key = r.local_video_path or "", r.source_object_key or ""
        if not (_is_file(local) or (local and key)):
            continue
        segments = segments_of(r.trim_segments)
        if not segments or is_holdout(r.video_id) != bool(holdout):
            continue
        score = scores.get(r.id) or {}
        post_tags = next((p.format_tags for p in published.get(r.id, []) if p.format_tags), "")
        out.append({
            "cut_pk": r.id, "candidate_pk": r.candidate_pk, "video_id": r.video_id,
            "title": (r.clip_title or "").strip(), "source_title": r.title or "",
            "channel": r.call_sign or "",
            "format": r.format_tags or post_tags or r.candidate_format_tags or "",
            "published": is_published, "seeded": r.id in seeded,
            "weight": weight_for(published=is_published, seeded=r.id in seeded),
            "views": score.get("views"), "percentile": score.get("percentile"),
            "segments": segments,
            "seconds": round(sum(s["end"] - s["start"] for s in segments), 2),
            "duration": float(r.duration_seconds) if r.duration_seconds else None,
            "local_video_path": local, "source_object_key": key,
            "transcript_path": r.transcript_path or "",
            "word_transcript_path": r.word_transcript_path or "",
            "archived_at": _iso(r.archived_at), "created_at": _iso(r.created_at),
            "holdout": bool(holdout),
        })
    out.sort(key=lambda e: (-e["weight"],
                            -(e["percentile"] if e["percentile"] is not None else -1.0),
                            e["cut_pk"]))
    return out[:int(limit)] if limit else out


def _performance(session) -> dict[int, dict]:
    """``cut_pk -> {views, percentile}`` for first airings measured at 6 h
    (taste's fixed-age, same-weeks percentile). Best effort: {} on failure."""
    try:
        from .taste import scored_first_airings

        with session.begin_nested():
            posts = scored_first_airings(session, as_of=dt.datetime.now(dt.timezone.utc),
                                         lookback_days=3650, min_age_hours=24)
    except Exception as exc:  # noqa: BLE001 - performance is context, not a requirement
        log.warning("handbook: post performance unavailable (%s)", exc)
        return {}
    return {p["cut_pk"]: {"views": p["views"], "percentile": p["percentile"]} for p in posts}


# ---- sources -----------------------------------------------------------------------

@contextlib.contextmanager
def source_for(fields: dict, *, min_free_gb: float, key: str):
    """The source video for the length of the block: restored from storage
    when it was offloaded (only while free disk stays above ``min_free_gb``;
    else ``LowDisk``), hard-linked under inflight/ while in use, and deleted
    again afterwards when this block restored it (it stays in the bucket)."""
    from .. import offload
    from . import media

    local = fields.get("local_video_path") or ""
    path = Path(local) if local else None
    existed = bool(path) and path.is_file() and path.stat().st_size > 0
    if not existed:
        where = path.parent if path else engine_dir()
        while not where.exists() and where.parent != where:
            where = where.parent
        try:
            free = float(media.disk_free_gb(where))
        except Exception:  # noqa: BLE001 - unknown free space doesn't block a restore
            free = None
        if free is not None and free < min_free_gb:
            raise LowDisk(f"only {free:.1f} GB free at {where} "
                          f"(clip_engine.min_free_gb is {min_free_gb:g})")
    source = Path(offload.ensure_local(local, fields.get("source_object_key") or ""))
    stamp = ()
    if not existed:
        with contextlib.suppress(OSError):
            st = source.stat()
            stamp = (st.st_ino, st.st_size)
    try:
        link = media.link_inflight(source, key)
    except Exception as exc:  # noqa: BLE001 - the link is a safety net
        log.info("handbook: no in-flight link for %s (%s); using it in place", source, exc)
        link = source
    try:
        yield link
    finally:
        if link != source:
            media.release_inflight(link)
        if not existed:
            try:
                st = source.stat()
                if not stamp or (st.st_ino, st.st_size) == stamp:
                    source.unlink()
                    log.info("handbook: removed restored source %s", source)
            except OSError as exc:
                log.info("handbook: left restored source %s in place: %s", source, exc)


def _texts(example: dict) -> tuple[list[dict], list[dict]]:
    from .. import clip_proposals

    ns = SimpleNamespace(transcript_path=example.get("transcript_path") or "",
                         word_transcript_path=example.get("word_transcript_path") or "")
    transcript = clip_proposals.load_transcript(ns)
    try:
        words = clip_proposals.load_words(ns)
    except Exception:  # noqa: BLE001 - words only sharpen the times
        words = []
    return transcript, words


# ---- 2. analyse ----------------------------------------------------------------------

def edit_path(cut_pk) -> Path:
    return engine_dir() / "handbook" / "edits" / f"{int(cut_pk)}.json"


def _cache_key(example: dict, model: str) -> str:
    return util.sha1(prompt_hash(), model, json.dumps(example.get("segments") or []), n=16)


def _merged(segments: list[dict]) -> list[tuple[float, float]]:
    spans = sorted((s["start"], s["end"]) for s in segments)
    out: list[list[float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def play_map(segments: list[dict]) -> list[dict]:
    """The operator's segments in play order with when each plays in the clip."""
    out, clock = [], 0.0
    for seg in segments:
        length = seg["end"] - seg["start"]
        out.append({"source_start": round(seg["start"], 2), "source_end": round(seg["end"], 2),
                    "clip_start": round(clock, 2), "clip_end": round(clock + length, 2)})
        clock += length
    return out


def skipped_stretches(segments: list[dict], beats: list[dict], *, duration: float | None,
                      context: float = SKIP_CONTEXT_S) -> list[dict]:
    """The source he did NOT use, near the clip (``context`` seconds either
    side and between its pieces) and at the start of the source, each with
    brief beats from the shot log."""
    kept = _merged(segments)
    if not kept:
        return []
    horizon = float(duration) if duration else max(
        [kept[-1][1]] + [float(b.get("end") or 0.0) for b in beats])
    lo, hi = max(0.0, kept[0][0] - context), min(horizon, kept[-1][1] + context)
    gaps: list[tuple[float, float]] = []
    # The source's opening minute (where anchor intros and studio tosses live)
    # and the run-up to the clip; one stretch when they touch.
    if lo <= OPENING_S:
        gaps.append((0.0, kept[0][0]))
    else:
        gaps += [(0.0, OPENING_S), (lo, kept[0][0])]
    for (_, b), (a, _) in zip(kept, kept[1:]):
        if a - b >= 0.5:
            gaps.append((b, a))
    if hi - kept[-1][1] >= 0.5:
        gaps.append((kept[-1][1], hi))
    out = []
    for a, b in gaps:
        if b - a < 0.5:
            continue
        inside = [{"t": round(float(x.get("start") or 0.0), 1), "shot": x.get("shot", "other"),
                   "speaker": _text(x.get("speaker"), 60), "line": _text(x.get("line"), 100),
                   "on_screen": _text(x.get("on_screen"), 80)}
                  for x in beats
                  if float(x.get("end") or 0.0) > a and float(x.get("start") or 0.0) < b]
        out.append({"start": round(a, 1), "end": round(b, 1), "seconds": round(b - a, 1),
                    "beats": inside[:MAX_SKIPPED_BEATS],
                    "more_beats": max(0, len(inside) - MAX_SKIPPED_BEATS)})
    return out


def edit_payload(example: dict, shot_log: dict, entries: list[dict]) -> dict:
    """What the analyst reads (text only)."""
    from . import shotlog

    segments = example["segments"]
    return {
        "source_title": example.get("source_title") or "",
        "clip_title": example.get("title") or "",
        "channel": example.get("channel") or "",
        "source_seconds": example.get("duration") or shot_log.get("source_seconds"),
        "shot_log": shotlog.compact_for_prompt(shot_log),
        "transcript": entries,
        "editor_segments_play_order": play_map(segments),
        "clip_seconds": round(sum(s["end"] - s["start"] for s in segments), 2),
        "skipped_stretches": skipped_stretches(segments, shot_log.get("beats") or [],
                                               duration=example.get("duration")
                                               or shot_log.get("source_seconds")),
    }


def _kind(value) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in KINDS else "other"


def clean_edit(raw: Any) -> dict:
    """A model's EDIT_SCHEMA answer coerced to the vocabulary and limits."""
    raw = raw if isinstance(raw, dict) else {}
    opening = raw.get("opening") if isinstance(raw.get("opening"), dict) else {}
    structure = raw.get("structure") if isinstance(raw.get("structure"), dict) else {}

    def items(value) -> list[dict]:
        return [{"what": _text(i.get("what"), 240), "kind": _kind(i.get("kind"))}
                for i in (value if isinstance(value, list) else [])[:20]
                if isinstance(i, dict) and str(i.get("what") or "").strip()]

    try:
        seconds = round(float(structure.get("seconds") or 0.0), 1)
    except (TypeError, ValueError):
        seconds = 0.0
    ending = str(structure.get("ending_kind") or "").strip().lower()
    return {
        "opening": {"kind": _kind(opening.get("kind")), "what": _text(opening.get("what"), 300),
                    "beat_shot": _text(opening.get("beat_shot"), 40),
                    "speaker": _text(opening.get("speaker"), 120),
                    "why_visible": _text(opening.get("why_visible"), 300)},
        "kept": items(raw.get("kept")),
        "dropped": items(raw.get("dropped")),
        "structure": {"cold_open": bool(structure.get("cold_open")),
                      "segments": _int(structure.get("segments")),
                      "seconds": seconds if math.isfinite(seconds) else 0.0,
                      "ending_kind": ending if ending in ENDING_KINDS else "other"},
        "language_choices": _text(raw.get("language_choices"), 300),
        "notes": _text(raw.get("notes"), 400),
    }


def _record(example: dict, edit: dict, *, model: str, cost: float, key: str) -> dict:
    keep = ("cut_pk", "candidate_pk", "video_id", "title", "channel", "format", "published",
            "seeded", "weight", "views", "percentile", "seconds", "segments")
    record = {k: example.get(k) for k in keep}
    record.update({"edit": edit, "model": model, "prompt_hash": prompt_hash(), "key": key,
                   "cost_usd": round(float(cost or 0.0), 6), "created_at": util.utcnow_iso()})
    return record


def load_edit(example: dict, settings) -> dict | None:
    """The cached analysis of this cut when its prompt, model and segments
    still match; None otherwise."""
    model = model_for(settings, "gemini", "editor")
    data = util.read_json(edit_path(example["cut_pk"]), None)
    if isinstance(data, dict) and data.get("key") == _cache_key(example, model) \
            and isinstance(data.get("edit"), dict):
        return data
    return None


def analyse(example: dict, settings, meter=None) -> dict:
    """Describe one past edit: shot log (cached per video; built with the
    Gemini perceiver when missing) then ONE Gemini call, text only. Returns
    the edit record (the example's facts + ``edit``) and caches it under
    ``engine_dir()/handbook/edits/<cut_pk>.json``. The source is only
    restored when no shot log is cached, and deleted again if it was."""
    from . import brand as load_brand
    from . import editor, gemini, shotlog

    cached = load_edit(example, settings)
    if cached is not None:
        if meter is not None and hasattr(meter, "add_cached"):
            meter.add_cached("handbook_analyse", cached.get("cost_usd") or 0.0)
        return cached

    brand = load_brand()
    transcript, words = _texts(example)
    if not transcript and not words:
        raise HandbookError(f"cut {example['cut_pk']}: no transcript on this machine")
    perceiver_model = model_for(settings, "gemini", "perceiver")
    shot_log = shotlog.load_cached(example["video_id"], "gemini", perceiver_model,
                                   settings=settings, seconds=example.get("duration"))
    if shot_log is None:
        min_free = float(opt(settings, "clip_engine.min_free_gb", 10))
        with source_for(example, min_free_gb=min_free,
                        key=f"handbook-{example['video_id']}") as source:
            shot_log = shotlog.build(
                video_id=example["video_id"], source=source, perceiver="gemini",
                settings=settings, brand=brand, title=example.get("source_title") or "",
                description="", duration=example.get("duration"), words=words,
                transcript=transcript, window=None, meter=meter)
    elif meter is not None and hasattr(meter, "add_cached"):
        meter.add_cached("perceive", shot_log.get("cost_usd") or 0.0)

    entries, _ = editor.entries_for(words, transcript)
    payload = edit_payload(example, shot_log, entries)
    model = model_for(settings, "gemini", "editor")
    bucket = getattr(meter, "bucket", None) or None
    try:
        data, usage = gemini.generate_json(
            model, system=edit_system(brand),
            parts=[gemini.text_part(json.dumps(payload, ensure_ascii=False))],
            schema=EDIT_SCHEMA, thinking="MEDIUM", media_resolution=None,
            label=f"handbook-analyse {example['video_id']}#{example['cut_pk']}", bucket=bucket)
    except Exception as exc:
        if meter is not None and getattr(exc, "usage", None):
            meter.add("handbook_analyse", exc.usage)
        raise
    if meter is not None:
        meter.add("handbook_analyse", usage)
    record = _record(example, clean_edit(data), model=model,
                     cost=(usage or {}).get("cost_usd", 0.0), key=_cache_key(example, model))
    try:
        util.atomic_write_json(edit_path(example["cut_pk"]), record)
    except OSError as exc:  # a cache that can't be written must not lose the answer
        log.warning("handbook: couldn't cache the analysis of cut %s: %s", example["cut_pk"], exc)
    return record


# ---- 3. synthesize ------------------------------------------------------------------

def compact_edit(record: dict) -> dict:
    """One edit as the synthesizer reads it."""
    edit = record.get("edit") or {}
    return {
        "id": record.get("cut_pk"), "channel": record.get("channel") or "",
        "format": record.get("format") or "", "published": bool(record.get("published")),
        "percentile": record.get("percentile"), "weight": record.get("weight"),
        "seconds": record.get("seconds"), "opening": edit.get("opening"),
        "kept": edit.get("kept"), "dropped": edit.get("dropped"),
        "structure": edit.get("structure"), "language_choices": edit.get("language_choices"),
        "notes": edit.get("notes"),
    }


def strength_for(support: int, exceptions: int) -> str:
    total = support + exceptions
    if support >= RULE_MIN_SUPPORT and total and support / total >= RULE_MIN_SHARE:
        return "rule"
    return "tendency"


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")[:40]


def _when(value: Any) -> str:
    text = _text(value, 120).rstrip(".")
    return "Always" if not text or text.lower() in ("always", "any", "all") else text


def why_for(support: int, exceptions: int, when: str) -> str:
    """The evidence in plain words, from the counts (not the model's prose)."""
    total = support + exceptions
    where = "" if when == "Always" else " where this applied"
    return f"{support} of your {total} past edits{where} did this"


def clean_rule(raw: Any, *, known_ids: set | None = None,
               max_support: int | None = None) -> dict | None:
    """One candidate rule coerced: text trimmed to 200 chars (no trailing
    condition — that is ``when``), counts as whole numbers (support capped at
    the number of records it could come from), strength recomputed from the
    counts (the model's own label isn't trusted), examples kept only when
    they name a known record. None when it has no text or no support."""
    if not isinstance(raw, dict):
        return None
    text = _text(raw.get("text"), TEXT_LIMIT)
    support = _int(raw.get("support"))
    if max_support is not None:
        support = min(support, max_support)
    if not text or support <= 0:
        return None
    exceptions = _int(raw.get("exceptions"))
    examples_: list[int] = []
    for value in raw.get("examples") if isinstance(raw.get("examples"), list) else []:
        number = _int(value, -1)
        if number >= 0 and (known_ids is None or number in known_ids) and number not in examples_:
            examples_.append(number)
    when = _when(raw.get("when"))
    return {"id": _slug(raw.get("id")), "text": text, "when": when,
            "support": support, "exceptions": exceptions,
            "examples": examples_[:MAX_EXAMPLE_IDS],
            "strength": strength_for(support, exceptions),
            "why": why_for(support, exceptions, when)}


def _unique_ids(rules: list[dict]) -> list[dict]:
    seen: set[str] = set()
    for i, rule in enumerate(rules, 1):
        base = rule.get("id") or f"r{i}"
        rid, n = base, 2
        while rid in seen:
            rid, n = f"{base}-{n}", n + 1
        rule["id"] = rid
        seen.add(rid)
    return rules


def _order(rules: list[dict]) -> list[dict]:
    """Rules before tendencies, then by support (stable otherwise)."""
    return sorted(rules, key=lambda r: (r.get("strength") != "rule", -_int(r.get("support"))))


def synthesize(edits: list[dict], settings, meter=None, *, model: str | None = None) -> dict:
    """ONE Claude call over the compact edit records -> the handbook REPORT
    (``candidates`` with support, exceptions and example cuts). Nothing is
    proposed or saved here (see ``propose`` and ``save``)."""
    from . import brand as load_brand
    from . import claude

    records = [e for e in edits or [] if isinstance(e, dict) and isinstance(e.get("edit"), dict)]
    if len(records) < MIN_EDITS:
        raise HandbookError(f"only {len(records)} analysed edits; need at least {MIN_EDITS}")
    payload = {"records": [compact_edit(r) for r in records]}
    model_id = model or model_for(settings, "claude", "editor")
    try:
        data, usage = claude.json_call(
            model_id, system=synth_system(load_brand()),
            content=[claude.text_block(json.dumps(payload, ensure_ascii=False))],
            schema=RULES_SCHEMA, effort="high", max_tokens=32000, label="handbook-synthesize",
            bucket=getattr(meter, "bucket", None) or None)
    except Exception as exc:
        if meter is not None and getattr(exc, "usage", None):
            meter.add("handbook_synthesize", exc.usage)
        raise
    if meter is not None:
        meter.add("handbook_synthesize", usage)
    known = {r.get("cut_pk") for r in records}
    candidates = [rule for rule in (clean_rule(raw, known_ids=known, max_support=len(records))
                                    for raw in (data or {}).get("rules") or [])
                  if rule is not None]
    if not candidates:
        raise HandbookError("the synthesis came back without any usable rule")
    return {
        "version": VERSION, "generated_at": util.utcnow_iso(), "model": model_id,
        "examples_used": len(records),
        "summary": _text((data or {}).get("summary"), 1200),
        "validation": None,
        "candidates": _order(_unique_ids(candidates)),
    }


# ---- proposing: the shared rule queue --------------------------------------------------
#
# Rules learned here are not kept apart from the operator's own: they go into
# the same proposal queue as rules drawn from his notes (notes.add_proposals),
# and once he adds one on the Brand & audience page it is an ordinary brand
# rule, which the engine reads like every other (llm._operator_rules).

def proposal_text(candidate: dict) -> str:
    """The rule as it is proposed: the imperative, with "Usually" in front of
    a tendency ("Usually open on ...")."""
    text = str(candidate.get("text") or "").strip()
    if candidate.get("strength") == "rule" or not text:
        return text
    if text.lower().startswith("usually"):
        return text
    return "Usually " + text[0].lower() + text[1:]


def evidence_for(candidate: dict) -> str:
    total = _int(candidate.get("support")) + _int(candidate.get("exceptions"))
    cuts = ", ".join(str(c) for c in (candidate.get("examples") or [])[:5])
    line = f"{_int(candidate.get('support'))} of {total} edits"
    return line + (f"; examples cut {cuts}" if cuts else "")


def proposal_item(candidate: dict) -> dict:
    """What ``notes.add_proposals`` takes for one candidate."""
    why = candidate.get("why") or why_for(_int(candidate.get("support")),
                                          _int(candidate.get("exceptions")),
                                          _when(candidate.get("when")))
    return {"text": proposal_text(candidate), "when": _when(candidate.get("when")),
            "why": why, "summary": why, "evidence": evidence_for(candidate),
            "count": _int(candidate.get("support")),
            "examples": [f"cut {c}" for c in (candidate.get("examples") or [])[:3]]}


def rule_line(candidate: dict) -> str:
    """The rule as the engine would read it once added, before the shared
    formatter has seen it: "When <condition>: <rule>" (the shape
    ``notes.rule_text`` gives formatted rules), or just the rule when it
    always applies. A candidate the queue formatted carries ``rule_text``."""
    if str(candidate.get("rule_text") or "").strip():
        return str(candidate["rule_text"]).strip()
    text = proposal_text(candidate)
    when = _when(candidate.get("when"))
    if not text or when == "Always":
        return text
    first, _, rest = when.partition(" ")
    if first.lower() in _LOWER_FIRST:   # "When the source...", but "When France 24..."
        when = " ".join(x for x in (first.lower(), rest) if x)
    return f"When {when}: {text}"


_LOWER_FIRST = frozenset({"a", "an", "the", "any", "every", "this", "there", "it", "its",
                          "they", "you", "he", "she", "most", "some", "one", "another"})


def _keys_of(item: dict) -> set[str]:
    """Every text a queued proposal or a brand rule can be recognised by: what
    the engine reads, the formatted sentence, and what was originally given."""
    return {k for k in (_key(item.get("text")), _key(item.get("rule")),
                        _key(item.get("original"))) if k}


def _queue_entries() -> list[dict]:
    try:
        from . import notes

        return [p for p in notes.proposals() or [] if isinstance(p, dict)]
    except Exception as exc:  # noqa: BLE001 - presentation only
        log.info("handbook: proposal queue unreadable (%s)", exc)
        return []


def _brand_rules() -> list[dict]:
    try:
        from ..config import load_brand_rules

        return [r for r in load_brand_rules() or [] if isinstance(r, dict)]
    except Exception as exc:  # noqa: BLE001 - presentation only
        log.info("handbook: brand rules unreadable (%s)", exc)
        return []


def _remember_format(candidate: dict, entry: dict) -> None:
    """Copy the shared formatter's output onto the candidate (the report
    stores the markdown and the text the engine will read)."""
    if str(entry.get("markdown") or "").strip():
        candidate["formatted"] = str(entry["markdown"])
    if str(entry.get("text") or "").strip():
        candidate["rule_text"] = str(entry["text"])


def propose(report: dict, *, settings=None) -> list[dict]:
    """Push the report's candidates into the shared proposal queue
    (``notes.add_proposals(source="handbook")``, which runs every item through
    the shared rule formatter) — rules and tendencies with support >=
    ``PROPOSE_MIN_SUPPORT``. Marks each candidate in ``report``: ``proposed``
    (queued by this call), and ``formatted`` / ``rule_text`` from the queue
    entry (new, or the one already there). The queue skips what it already
    holds and what is already a brand rule, so a rebuild never re-asks what
    the operator turned down. Returns what was queued; never raises."""
    from . import notes

    candidates = [c for c in report.get("candidates") or [] if isinstance(c, dict)]
    chosen = [c for c in candidates if _int(c.get("support")) >= PROPOSE_MIN_SUPPORT]
    for candidate in candidates:
        candidate["proposed"] = False
    try:
        queued = notes.add_proposals([proposal_item(c) for c in chosen], source="handbook",
                                     settings=settings) or []
    except Exception as exc:  # noqa: BLE001 - the report is still worth saving
        log.warning("handbook: couldn't queue the rule proposals: %s", exc)
        queued = []
    queued = [q for q in queued if isinstance(q, dict)]
    known = _queue_entries() + _brand_rules()
    for candidate in candidates:
        key = _key(proposal_text(candidate))
        mine = next((q for q in queued if key in _keys_of(q)), None)
        if mine is not None:
            candidate["proposed"] = True
            _remember_format(candidate, mine)
            continue
        earlier = next((q for q in known if key in _keys_of(q)), None)
        if earlier is not None:
            _remember_format(candidate, earlier)
    report["proposed"] = len(queued)
    return queued


def _key(text: Any) -> str:
    return " ".join(str(text or "").lower().split()).rstrip(".")


def statuses(report: dict | None) -> list[dict]:
    """The report's candidates with where each stands now: "added" (a brand
    rule), "pending" / "dismissed" (in the shared proposal queue), or "not
    proposed" (too little support, or the queue was unreachable). Matched by
    text: the engine's text, the formatted sentence or the text as given.
    Never raises."""
    candidates = [dict(c) for c in (report or {}).get("candidates") or [] if isinstance(c, dict)]
    queue = _queue_entries()
    rules = set().union(*[_keys_of(r) for r in _brand_rules()]) if candidates else set()
    for c in candidates:
        keys = {k for k in (_key(proposal_text(c)), _key(c.get("rule_text"))) if k}
        if keys & rules:
            c["status"] = "added"
            continue
        entry = next((q for q in queue if keys & _keys_of(q)), None)
        if entry is not None and entry.get("status") == "added":
            c["status"] = "added"
        elif entry is not None:
            c["status"] = str(entry.get("status") or "pending")
        else:
            c["status"] = "not proposed"
    return candidates


# ---- storage: the analysis report --------------------------------------------------------

_lock = threading.RLock()
_META_KEYS = ("version", "generated_at", "model", "examples_used", "holdout", "holdout_videos",
              "failed", "cost_usd", "stopped", "summary", "proposed", "validation")
_CANDIDATE_KEYS = ("id", "text", "when", "why", "support", "exceptions", "examples", "strength",
                   "formatted", "rule_text", "proposed")


def path() -> Path:
    """config/clip_handbook.yaml in the active workspace."""
    from ..config import CONFIG_DIR

    return CONFIG_DIR / "clip_handbook.yaml"


def normalize(data: Any) -> dict:
    """A report dict (from a file, possibly hand-edited) made consistent."""
    data = data if isinstance(data, dict) else {}
    out = {k: copy.deepcopy(data[k]) for k in _META_KEYS if k in data}
    candidates = []
    for raw in data.get("candidates") or []:
        candidate = clean_rule(raw) if isinstance(raw, dict) else None
        if candidate is None:
            continue
        for extra in ("formatted", "rule_text"):
            if str(raw.get(extra) or "").strip():
                candidate[extra] = str(raw[extra])
        if "proposed" in raw:
            candidate["proposed"] = bool(raw["proposed"])
        candidates.append({k: candidate[k] for k in _CANDIDATE_KEYS if k in candidate})
    out["candidates"] = _order(_unique_ids(candidates))
    return out


def load() -> dict | None:
    """The report, normalized; None when there is none or it can't be read
    (a broken file is logged, never raised)."""
    from ..shared_docs import read_text

    file = path()
    try:
        text = read_text("handbook", file)
        if text is None:
            return None
        return normalize(yaml.safe_load(text) or {})
    except Exception as exc:  # noqa: BLE001
        log.warning("%s is not a usable handbook report (%s); ignoring it", file, exc)
        return None


def save(report: dict) -> Path:
    """Write the report atomically, with a header saying what it is."""
    data = normalize(report)
    stamp = str(data.get("generated_at") or util.utcnow_iso())[:10]
    header = (
        f"# Clip handbook report — what your past edits have in common, generated {stamp}.\n"
        "# A report, not a rule store: its candidate rules wait with your other suggested\n"
        "# rules on the Brand & audience page, and only the ones you add there reach the\n"
        "# clip engine. Rebuild with: python run.py clip-engine handbook build\n")
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    with _lock:
        from ..shared_docs import write_text

        target = path()
        write_text("handbook", header + body, target)
    log.info("handbook report saved to %s (%d candidates)", target, len(data["candidates"]))
    return target


def backup() -> Path | None:
    """Copy the current report under engine_dir()/handbook/backups/ (before a
    rebuild replaces it). None when there is nothing to back up."""
    file = path()
    if not file.is_file():
        return None
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return util.atomic_copy(file, engine_dir() / "handbook" / "backups" / f"clip_handbook-{stamp}.yaml")


def set_validation(summary: dict) -> None:
    """Store a validation summary in the current report."""
    with _lock:
        report = load()
        if report is None:
            raise HandbookError("there is no handbook report yet "
                                "(python run.py clip-engine handbook build)")
        report["validation"] = summary
        save(report)


def render(report: dict | None) -> str:
    """The candidates as plain text (for the CLI): rules first, tendencies
    phrased with "usually", each with its condition and evidence."""
    candidates = (report or {}).get("candidates") or []
    if not candidates:
        return ""
    lines = [HEADER_TITLE]
    for c in _order([dict(x) for x in candidates]):
        when = _when(c.get("when"))
        condition = "" if when == "Always" else f" (when: {when})"
        lines.append(f"- {proposal_text(c)}{condition} — {evidence_for(c)}")
    return "\n".join(lines)


def render_active() -> str:
    """Kept for compatibility: always "". Rules learned from past edits reach
    the engine only once the operator adds them as brand rules (the engine
    reads those through ``llm._operator_rules``), so there is no second,
    handbook-only rule text to append."""
    return ""


def fingerprint(texts: list[str]) -> str:
    return util.sha1(*texts, n=12)


# ---- build ------------------------------------------------------------------------------

@contextlib.contextmanager
def _db():
    """A WRITER session that is always rolled back (see bench._db for why
    never the read-only one)."""
    from ..db import session_scope

    with session_scope() as session:
        try:
            yield session
        finally:
            session.rollback()


def build(settings, *, limit: int | None = None, max_usd: float = 40.0, meter=None,
          say: Callable[[str], Any] = log.info) -> dict:
    """Analyse the non-holdout exported cuts (best evidence first; ``limit``
    of them) until ``max_usd`` is spent or the disk runs low, then synthesize.
    Cached analyses cost nothing and are reused. Returns the report; neither
    proposes nor saves (the caller does both)."""
    from . import engine

    meter = meter or engine.CostMeter("bench")
    with _db() as session:
        train = examples(session, limit=limit)
    with _db() as session:
        held = examples(session, holdout=True)
    say(f"{len(train)} past edits to study; {len(held)} edits on "
        f"{len({e['video_id'] for e in held})} held-out videos kept back for validation")
    edits, failed, stopped = [], 0, ""
    for i, example in enumerate(train, 1):
        if meter.total >= max_usd:
            stopped = f"stopped at the ${max_usd:g} cap after {i - 1} of {len(train)} edits"
            say(stopped)
            break
        try:
            record = analyse(example, settings, meter)
        except LowDisk as exc:
            stopped = f"stopped: {exc}"
            say(stopped)
            break
        except Exception as exc:  # noqa: BLE001 - one bad edit must not stop the batch
            failed += 1
            say(f"[{i}/{len(train)}] cut {example['cut_pk']} ({example['video_id']}): "
                f"failed — {type(exc).__name__}: {exc}")
            continue
        edits.append(record)
        opening = (record.get("edit") or {}).get("opening") or {}
        say(f"[{i}/{len(train)}] cut {example['cut_pk']} ({example['video_id']}): opens on "
            f"{opening.get('kind', '?')} · ${meter.total:.2f} so far")
    report = synthesize(edits, settings, meter)
    report.update({
        "holdout": len(held), "holdout_videos": len({e["video_id"] for e in held}),
        "failed": failed, "stopped": stopped, "cost_usd": round(float(meter.total), 4),
    })
    return report


# ---- 4. validate ------------------------------------------------------------------------

def _hits(props: list[list[dict]], targets: list[list[dict]]) -> dict:
    from .bench import HIT_IOU, MOMENT_OVERLAP, iou, overlap_share

    def moment(p):
        return any(overlap_share(p, t) >= MOMENT_OVERLAP for t in targets)

    def hit(p):
        return any(iou(p, t) >= HIT_IOU for t in targets)

    top3 = props[:3]
    return {"moment1": bool(props) and moment(props[0]), "moment3": any(moment(p) for p in top3),
            "hit1": bool(props) and hit(props[0]), "hit3": any(hit(p) for p in top3)}


def _rate(rows: list[dict], key: str) -> float | None:
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 3) if rows else None


def summarize_validation(rows: dict[str, list[dict]]) -> dict:
    """Per arm: n, moment-hit@1/@3, any-hit@1/@3, errors, empty — on the
    videos EVERY arm finished (errors and empty answers count as misses)."""
    sets = [{r["video_id"] for r in arm_rows} for arm_rows in rows.values()]
    common = set.intersection(*sets) if sets else set()
    out = {}
    for arm, arm_rows in rows.items():
        kept = [r for r in arm_rows if r["video_id"] in common]
        out[arm] = {"n": len(kept), "moment_hit1": _rate(kept, "moment1"),
                    "moment_hit3": _rate(kept, "moment3"), "hit1": _rate(kept, "hit1"),
                    "hit3": _rate(kept, "hit3"),
                    "errors": sum(1 for r in kept if r.get("error")),
                    "empty": sum(1 for r in kept if not r.get("error") and not r.get("n_props"))}
    return out


def holdout_videos(holdout_examples: list[dict], *, limit: int | None) -> list[dict]:
    """Held-out examples grouped per video (targets = every exported cut of
    it), in a fixed order (sha1 of the video id)."""
    videos: dict[str, dict] = {}
    for ex in holdout_examples:
        video = videos.setdefault(ex["video_id"], {**ex, "targets": []})
        video["targets"].append(ex["segments"])
    ordered = sorted(videos.values(), key=lambda v: util.sha1("handbook-validate", v["video_id"]))
    return ordered[:int(limit)] if limit else ordered


def validation_dir(fp: str) -> Path:
    return engine_dir() / "handbook" / "validation" / fp


def candidates_to_test(report: dict) -> list[str]:
    """The candidates validate adds on top of the current rules: every one not
    already a brand rule and not dismissed, in the text the engine would read
    once added (``rule_line``)."""
    out = []
    for c in statuses(report):
        if c["status"] in ("added", "dismissed"):
            continue
        out.append(rule_line(c))
    return out


def validate(report: dict, settings, *, limit: int = 15, max_usd: float | None = None,
             setup: str = "gemini", say: Callable[[str], Any] = log.info) -> dict:
    """Replay held-out videos through the engine twice — with the CURRENT
    rules ("current") and with the current rules plus the report's candidate
    rules ("candidates", injected through the settings override
    ``clip_engine.extra_rules``) — and report moment-hit@1/@3 (and any-hit)
    for both against the operator's own cuts.

    Results are cached per candidate-set fingerprint, arm and video, so a
    rerun only pays for what's missing; ``max_usd`` stops it between videos.
    Replays are as of each video's archive day (``prepare_inputs(replay=True)``)
    with the playbook's example titles stripped, as the bench does. Returns the
    summary (the caller stores it)."""
    # SPEC-GAP: relies on Agent A's engine appending
    # opt(settings, "clip_engine.extra_rules", []) (a list of rule strings) to its
    # rules text; without it both arms run the same rules and tie.
    from . import brand as load_brand
    from . import playbook as playbooks
    from .bench import settings_for

    extra = candidates_to_test(normalize(report))
    if not extra:
        raise HandbookError("no candidate rules left to validate (every one is already a "
                            "brand rule or was dismissed)")
    fp = fingerprint(extra)
    max_usd = float(max_usd if max_usd is not None
                    else opt(settings, "clip_engine.bench.max_usd", 40))
    with _db() as session:
        videos = holdout_videos(examples(session, holdout=True), limit=limit)
    if not videos:
        raise HandbookError("no held-out videos with a source and a transcript to replay")
    arms = {
        "current": settings_for({"overrides": {"clip_engine.extra_rules": []}}, settings),
        "candidates": settings_for({"overrides": {"clip_engine.extra_rules": list(extra)}},
                                   settings),
    }
    brand = load_brand()
    playbook = playbooks.strip_examples(playbooks.load())
    min_free = float(opt(settings, "clip_engine.min_free_gb", 10))
    timeout = float(opt(settings, "clip_engine.job_timeout_seconds", 1800))
    rows: dict[str, list[dict]] = {arm: [] for arm in arms}
    spent, stopped = 0.0, ""
    say(f"Validating {len(extra)} candidate rules on {len(videos)} held-out videos "
        f"(set {fp}, setup {setup})")
    for n, video in enumerate(videos, 1):
        vid = video["video_id"]
        records = {arm: util.read_json(validation_dir(fp) / arm / f"{vid}.json", None)
                   for arm in arms}
        pending = [arm for arm, rec in records.items()
                   if not (isinstance(rec, dict) and not rec.get("error"))]
        if pending and spent >= max_usd:
            stopped = f"stopped at the ${max_usd:g} cap"
            say(stopped)
            break
        if pending:
            try:
                spent += _replay(video, pending, arms, records, fp=fp, setup=setup, brand=brand,
                                 playbook=playbook, min_free=min_free, timeout=timeout, say=say)
            except LowDisk as exc:
                stopped = f"stopped: {exc}"
                say(stopped)
                break
            except Exception as exc:  # noqa: BLE001 - this video is skipped, not the run
                say(f"[{n}/{len(videos)}] {vid}: skipped — {type(exc).__name__}: {exc}")
                continue
        line = []
        if not all(isinstance(rec, dict) for rec in records.values()):
            continue
        for arm, rec in records.items():
            row = {"video_id": vid, "error": rec.get("error") or "",
                   "n_props": len(rec.get("top") or [])}
            row.update(_hits(rec.get("top") or [], video["targets"]))
            rows[arm].append(row)
            line.append(f"{arm}: {'moment' if row['moment1'] else 'miss'}"
                        + (f" ({row['error'][:60]})" if row["error"] else ""))
        say(f"[{n}/{len(videos)}] {vid}: " + " · ".join(line) + f" · ${spent:.2f} spent")
    summary = summarize_validation(rows)
    return {"at": util.utcnow_iso(), "setup": setup, "fingerprint": fp,
            "rules_tested": len(extra), "rules": extra,
            "videos": min((v["n"] for v in summary.values()), default=0),
            "current": summary.get("current", {}), "candidates": summary.get("candidates", {}),
            "cost_usd": round(spent, 4), "stopped": stopped}


def _replay(video: dict, pending: list[str], arms: dict, records: dict, *, fp: str, setup: str,
            brand: dict, playbook: dict, min_free: float, timeout: float, say) -> float:
    """Run the pending arms on one held-out video; writes each record and
    puts it in ``records``. Returns what was spent."""
    from . import engine, jobs

    vid = video["video_id"]
    inputs = jobs.prepare_inputs(video["candidate_pk"], as_of=_parse_time(video.get("archived_at")),
                                 replay=True) or {}
    cand = inputs.get("candidate") or {}
    transcript, words = _texts(video)
    if not transcript:
        raise HandbookError("no transcript")
    fields = dict(video)
    for k in ("local_video_path", "source_object_key"):
        if cand.get(k):
            fields[k] = cand[k]
    spent = 0.0
    with source_for(fields, min_free_gb=min_free, key=f"handbook-{vid}") as source:
        for arm in pending:
            meter = engine.CostMeter("bench")
            started = time.monotonic()
            try:
                result = engine.run(
                    setup=setup, settings=arms[arm], brand=brand, playbook=playbook,
                    video_id=vid, title=cand.get("title") or video.get("source_title") or "",
                    description=cand.get("description") or "", duration=video.get("duration"),
                    source=source, words=words, transcript=transcript,
                    context=inputs.get("context") or {}, rejected=[],
                    examples=inputs.get("examples") or [], used=[], window=None, meter=meter,
                    progress=lambda stage, a=arm: say(f"    {vid} {a}: {stage}"),
                    deadline=time.monotonic() + timeout)
                top = [segments_of(item.get("segments")) for item in result.top or []]
                record = {"top": [t for t in top if t], "error": "", "outcome": result.outcome}
            except Exception as exc:  # noqa: BLE001 - recorded as a miss
                record = {"top": [], "error": f"{type(exc).__name__}: {exc}"[:400],
                          "outcome": "error"}
            record.update({"video_id": vid, "arm": arm, "fingerprint": fp,
                           "cost_usd": round(float(meter.total), 6),
                           "seconds": round(time.monotonic() - started, 1),
                           "created_at": util.utcnow_iso()})
            spent += record["cost_usd"]
            util.atomic_write_json(validation_dir(fp) / arm / f"{vid}.json", record)
            records[arm] = record
    return spent
