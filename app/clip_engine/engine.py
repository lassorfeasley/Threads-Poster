"""The clip engine's pipeline: log, draft, snap, render, watch, rank, revise,
choose. Spec §4.12.

``run`` is orchestration over plain values. It never touches the database:
``jobs`` copies the inputs out of a short session first and writes the
proposals in another one afterwards, so a run that takes minutes never holds a
transaction (or a pooled Postgres connection) open, and the bench can replay
it on frozen inputs. Every model call happens inside a collaborator
(``shotlog``, ``editor``, ``critic``) that records its spend on the shared
``CostMeter``; this module decides what to do with what they return.

The selection rules encode what the design review found:

- Duplicates are only TRUE duplicates: same material, same play order, same
  opening. The editor is asked for cold-open and tightened variants of one
  great moment, and an order-blind overlap test deletes exactly those.
- A moment the operator already dismissed on this video stays dismissed,
  unless the complaint was the boundaries and the boundaries moved.
- The topic gate needs two witnesses. A draft is dropped only when the cold
  critic AND the shot log both say its material is off topic; one
  low-effort boolean would delete the account's second-biggest kind of post
  (an allegory that never names the topic).
- The language gate works the same way: a draft goes when the critic
  couldn't understand its speech AND the shot log says most of its material
  is speech in another language with no English voiceover. This audience
  never gets those clips (burned-in subtitles aren't enough). But it never
  empties a run on its own: if every draft left would go, they all stay,
  scored down, and the operator decides.
- Only the best cut of each story reaches the head-to-head ranking, so five
  variants of one moment can't crowd out a second story.
- A revision replaces its original only when it breaks nothing the original
  got right, the cold critic scores it at least as high, and the judge
  prefers it in BOTH presentation orders. Pairwise judges are close to a coin
  flip on near-identical cuts, so the original keeps the tie.
- The operator's ``advice`` for a pass reaches every call that chooses
  (draft, revise, cold watch, rank, prefer) and is recorded in the notes.
  Rules learned from the operator's past edits become ordinary brand rules
  (read with the rest through ``llm._operator_rules``); anything
  ``handbook.render_active`` still returns is appended to the playbook text.
  ``clip_engine.extra_rules`` (a list of strings) adds rules to the
  operator's rules block for one run, so the handbook's validation can
  replay a run with candidate rules before the operator adopts them.
- Mechanical failures (nothing could be rendered, nothing survived snapping)
  raise ``EngineError`` and the caller falls back to the transcript
  suggester. An editorial "nothing here" (no drafts, or only off-topic ones)
  returns an empty result instead, which must never supersede the operator's
  pending suggestions.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .. import clip_proposals, llm
from . import ROLES, critic, editor, legacy, media, model_for, opt, prompts, shotlog, snap, taste, util
from . import playbook as playbooks

log = logging.getLogger("clip_engine.engine")


class EngineError(RuntimeError):
    """The run can't produce proposals worth trusting (the perceiver or the
    editor failed, nothing could be rendered, the hard cost cap was hit). The
    caller falls back to the transcript suggester. ``notes`` carries what the
    run had noted before it stopped, so the fallback can say why; ``renders``
    the draft renders it had made (paths), which nobody will be shown."""

    def __init__(self, message: str = "", *, notes: list[str] | None = None,
                 renders: list[str] | None = None) -> None:
        super().__init__(message)
        self.notes = list(notes or [])
        self.renders = [str(p) for p in renders or []]


# ---- accounting -----------------------------------------------------------------

def _money(value: Any) -> float:
    """A cost or token count as a finite, non-negative float (0.0 when it
    isn't one): accounting must never be the thing that breaks a run."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or number < 0:
        return 0.0
    return number


class CostMeter:
    """What one run spent, by stage and by model.

    Thread-safe: the render and critic stages fan out over daemon threads that
    all add to one meter. ``bucket`` is the spend-ledger bucket the providers
    record under ("engine" in the dashboard, "bench" in the replay benchmark),
    which keeps engine spend out of ``llm.daily_budget_usd``.

    ``add_cached`` records what a reused artifact (a cached shot log) cost when
    it was made. It shows in ``by_stage`` as ``<stage>_cached`` and counts
    toward ``total_with_cached`` (what the run would cost from scratch), but
    never toward ``total``, the money actually spent now. The caller owns the
    meter, so a run that raises still leaves its partial spend readable.
    """

    def __init__(self, bucket: str = "engine") -> None:
        self.bucket = bucket
        self.total = 0.0
        self.calls = 0
        self.by_stage: dict[str, float] = {}
        self.tokens: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def add(self, stage: str, usage: dict | None) -> None:
        """Count one model call (``{"model","input_tokens","output_tokens",
        "cost_usd"}``): its cost into ``total`` and ``by_stage[stage]``, its
        tokens under its model. Malformed usage counts as zero; never raises."""
        if not isinstance(usage, dict):
            return
        cost = _money(usage.get("cost_usd"))
        model = str(usage.get("model") or "").strip()
        tokens_in = int(_money(usage.get("input_tokens")))
        tokens_out = int(_money(usage.get("output_tokens")))
        with self._lock:
            self.total += cost
            self.calls += 1
            self.by_stage[stage] = self.by_stage.get(stage, 0.0) + cost
            if model:
                row = self.tokens.setdefault(model, {"input": 0, "output": 0})
                row["input"] += tokens_in
                row["output"] += tokens_out

    def add_cached(self, stage: str, cost: float) -> None:
        """Record the stored cost of a reused artifact under
        ``by_stage[f"{stage}_cached"]``; not part of ``total``."""
        cost = _money(cost)
        with self._lock:
            key = f"{stage}_cached"
            self.by_stage[key] = self.by_stage.get(key, 0.0) + cost

    @property
    def total_with_cached(self) -> float:
        """``total`` plus the stored cost of everything reused from a cache.
        Budget decisions inside a run use this, so a run behaves the same
        whether or not an earlier run (or bench arm) left a shot log behind."""
        with self._lock:
            return self.total + sum(v for k, v in self.by_stage.items() if k.endswith("_cached"))

    def snapshot(self) -> dict:
        """A rounded copy for the result: ``cost_usd``, ``by_stage``,
        ``tokens``, ``calls``."""
        with self._lock:
            return {
                "cost_usd": round(self.total, 6),
                "by_stage": {k: round(v, 6) for k, v in self.by_stage.items()},
                "tokens": {m: dict(t) for m, t in self.tokens.items()},
                "calls": self.calls,
            }


@dataclass
class EngineResult:
    """What one run produced. Spec §3.

    ``top`` holds at most ``keep`` drafts, each with its ``critique``,
    ``rank`` (1-based), ``score``, ``render`` (a path string) and
    ``credits``. ``drafts`` holds every draft the editor wrote plus any
    revision, each with ``dropped_reason`` ("" while it was in the running).
    ``orders`` are id lists for the bench: "editor" (the editor's own order,
    after the mechanical filters), "cold" (by cold-critic score, after the
    topic gate), "rank" (after the head-to-head ranking) and "final" (the top
    picks, after revision and the overlap skip).
    """
    setup: str
    top: list[dict] = field(default_factory=list)
    drafts: list[dict] = field(default_factory=list)
    moments: list[dict] = field(default_factory=list)
    shotlog: dict | None = None
    cost_usd: float = 0.0
    by_stage: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    orders: dict = field(default_factory=dict)
    model: str = ""
    policy: str = ""
    outcome: str = "proposals"

    def to_dict(self) -> dict:
        """A JSON-safe copy (paths become strings), for the run record."""
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return value


# ---- models and estimates -------------------------------------------------------

def models_for(setup: str, settings) -> dict:
    """Role -> model id for one setup: perceiver, editor, critic, judge.

    Engine setups resolve every role through ``model_for`` (a blank setting
    falls back to the provider-wide model, then the code default; the judge
    falls back to the critic). ``transcript`` has a single model, the legacy
    suggester's (``clips.model``), reported as its editor."""
    if setup == "transcript":
        return {"perceiver": "", "editor": legacy.suggester_model(settings),
                "critic": "", "judge": ""}
    roles = ROLES.get(setup)
    if roles is None:
        raise ValueError(f"unknown clip engine setup {setup!r}")
    return {
        "perceiver": model_for(settings, roles["perceiver"], "perceiver"),
        "editor": model_for(settings, roles["editor"], "editor"),
        "critic": model_for(settings, roles["critic"], "critic"),
        "judge": model_for(settings, roles["critic"], "judge"),
    }


_ESTIMATE_BASE = {"gemini": 0.60, "hybrid": 1.40, "claude": 1.40}
_ESTIMATE_PER_MINUTE = {"gemini": 0.03, "hybrid": 0.02, "claude": 0.06}
# An unknown source length is priced as this many minutes so the budget guard
# errs high (most sources run 2-20 minutes).
_UNKNOWN_MINUTES = 20.0


def _rate(settings, key: str, defaults: dict, setup: str) -> float:
    """One setup's rate from a ``{setup: usd}`` setting; the code default
    when the setting, or that setup's entry, is missing or not a price."""
    table = opt(settings, key, None)
    if isinstance(table, dict) and setup in table:
        try:
            number = float(table[setup])
        except (TypeError, ValueError):
            number = -1.0
        if math.isfinite(number) and number >= 0:
            return number
        log.warning("Setting %s.%s=%r isn't a price; using %s", key, setup,
                    table[setup], defaults.get(setup, 0.0))
    return float(defaults.get(setup, 0.0))


def estimate_usd(setup: str, settings, duration: float | None) -> float:
    """What one whole run is expected to cost: ``estimate_usd_base[setup] +
    estimate_usd_per_minute[setup] * minutes``. The budget gate reserves this
    before an unattended run starts. ``transcript`` costs nothing here (the
    legacy suggester is budgeted under ``llm.daily_budget_usd``)."""
    base = _rate(settings, "clip_engine.estimate_usd_base", _ESTIMATE_BASE, setup)
    per_minute = _rate(settings, "clip_engine.estimate_usd_per_minute",
                       _ESTIMATE_PER_MINUTE, setup)
    seconds = _money(duration)
    minutes = seconds / 60.0 if seconds > 0 else _UNKNOWN_MINUTES
    return round(base + per_minute * minutes, 4)


# ---- segment helpers ---------------------------------------------------------------

def seconds_of(segments: list[dict]) -> float:
    """Played length of a clip: the sum of its segments (play order makes no
    difference to the length)."""
    total = 0.0
    for s in segments or []:
        try:
            total += max(0.0, float(s["end"]) - float(s["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    return round(total, 2)


def plain_segments(segments: list[dict]) -> list[dict]:
    """``[{start, end}]`` at 2 dp, play order kept: what a render or the
    ledger needs (the ``lead`` hint only matters to the snapper), and a stable
    render cache key."""
    out = []
    for s in segments or []:
        try:
            out.append({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2)})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _source_order(segments: list[dict]) -> tuple[int, ...]:
    """The permutation that sorts a clip's segments by source time: equal for
    two clips that play their material in the same order."""
    return tuple(sorted(range(len(segments)), key=lambda i: (float(segments[i]["start"]), i)))


def same_cut(a: list[dict], b: list[dict]) -> bool:
    """True duplicates only: the same number of segments, played in the same
    order, opening within 0.5 s of each other, covering the same material
    (IoU > 0.95). A cold-open reordering of the same material, or a tightened
    take, is a different cut the critic should see."""
    if not a or not b or len(a) != len(b):
        return False
    try:
        if abs(float(a[0]["start"]) - float(b[0]["start"])) > 0.5:
            return False
        if _source_order(a) != _source_order(b):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return clip_proposals.iou(a, b) > 0.95


def clean_other_clips(raw: Any) -> list[dict]:
    """The video's other clips as plain dicts — ``{title, words_heard,
    segments (play order, {start, end})}`` — dropping entries with no usable
    segment. Whatever else rides along is left out."""
    out: list[dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        segments = plain_segments(_segments_of(item))
        segments = [s for s in segments if s["end"] > s["start"]]
        if not segments:
            continue
        out.append({"title": " ".join(str(item.get("title") or "").split())[:200]
                    or "another clip",
                    "words_heard": " ".join(str(item.get("words_heard") or "").split())[:600],
                    "segments": segments})
    return out


def _segments_of(item: Any) -> list[dict]:
    raw = item.get("segments") if isinstance(item, dict) else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except ValueError:
            return []
    return [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []


def repeats_rejection(segments: list[dict], rejected: list[dict] | None) -> dict | None:
    """The dismissed proposal (``{segments, reason}``) this clip repeats, or
    None. A clip repeats one when they share at least half their material
    (IoU >= 0.5). A dismissal for bad boundaries only rules out the same
    boundaries: once an outer edge (earliest start or latest end) moved by
    2 s or more, the moment may come back."""
    for item in rejected or ():
        theirs = _segments_of(item)
        if not theirs or clip_proposals.iou(segments, theirs) < 0.5:
            continue
        if str(item.get("reason") or "") == "bad_boundaries":
            start_delta, end_delta = clip_proposals.span_deltas(theirs, segments)
            if max(abs(start_delta or 0.0), abs(end_delta or 0.0)) >= 2.0:
                continue
        return item
    return None


ON_TOPIC = ("central", "allegory")
OFF_TOPIC = ("incidental", "none")


def material_topic(shot_log: dict | None, segments: list[dict]) -> float | None:
    """The shot log's 0-2 ``topic`` for the material a clip uses: the mean
    over the beats it overlaps, weighted by how much of each it uses (so a
    sliver of a neighboring beat can't swing it). None when no beat
    overlaps, i.e. the log has no opinion."""
    weighted = covered = 0.0
    for beat in (shot_log or {}).get("beats") or []:
        try:
            b_start, b_end = float(beat["start"]), float(beat["end"])
            topic = float(beat["topic"])
        except (KeyError, TypeError, ValueError):
            continue
        for s in segments or []:
            try:
                overlap = min(b_end, float(s["end"])) - max(b_start, float(s["start"]))
            except (KeyError, TypeError, ValueError):
                continue
            if overlap > 0:
                weighted += overlap * topic
                covered += overlap
    return round(weighted / covered, 3) if covered > 0 else None


LANGUAGE_SHARE = 0.5   # "mostly": at least half the clip's material


def foreign_speech_share(shot_log: dict | None, segments: list[dict]) -> float | None:
    """How much of a clip's material (0-1, overlap-weighted) the shot log
    marks as speech in another language with no English voiceover. None
    when no beat it overlaps says what language it is in (an older log, or
    no log): no opinion, so never a reason to drop."""
    foreign = covered = 0.0
    for beat in (shot_log or {}).get("beats") or []:
        language = beat.get("speech_language") if isinstance(beat, dict) else None
        if language not in ("english", "other", "none"):
            continue
        try:
            b_start, b_end = float(beat["start"]), float(beat["end"])
        except (KeyError, TypeError, ValueError):
            continue
        unheard = language == "other" and not beat.get("english_voiceover")
        for s in segments or []:
            try:
                overlap = min(b_end, float(s["end"])) - max(b_start, float(s["start"]))
            except (KeyError, TypeError, ValueError):
                continue
            if overlap > 0:
                covered += overlap
                if unheard:
                    foreign += overlap
    return round(foreign / covered, 3) if covered > 0 else None


def language_verdict(critique: dict | None, share: float | None) -> tuple[str, str]:
    """The language gate for one reviewed draft: ``("keep" | "drop", why)``.

    Two witnesses, like the topic gate: the critic says the speech isn't
    understood AND the shot log puts at least half the material on speech in
    another language without an English voiceover. A critique that errored,
    or a log with no language data, never drops anything."""
    if not critique or critique.get("error"):
        return "keep", ""
    if critique.get("speech_understood", True) is not False:
        return "keep", ""
    if share is None or share < LANGUAGE_SHARE:
        return "keep", ""
    return "drop", (f"critic: speech not understood; shot log: {share:.0%} in another "
                    "language with no English voiceover")


def topic_verdict(critique: dict | None, material: float | None) -> tuple[str, str]:
    """The topic gate for one reviewed draft: ``("keep" | "doubt" | "drop",
    why)``.

    Dropped only with two witnesses: the critic calls the topic incidental or
    none AND the shot log rates the material below 1 (of 2). When only the
    critic doubts it, the draft stays (its cold score already carries the
    penalty) with a note for the card. A critique that errored is kept as
    "not reviewed": an outage says nothing about the clip."""
    if not critique or critique.get("error"):
        return "keep", ""
    label = str(critique.get("topic") or "central").strip().lower()
    if label not in OFF_TOPIC:
        return "keep", ""
    note = " ".join(str(critique.get("topic_note") or "").split())
    if material is not None and material < 1.0:
        why = f"critic: {label}; shot log: {material:.1f}/2"
        return "drop", f"{why} ({note})" if note else why
    return "doubt", note or f"The critic called the topic {label}."


# ---- ordering -------------------------------------------------------------------

UNREVIEWED_SCORE = -1e9   # critic.cold_score's score for a critique that errored


def order_key(draft: dict) -> tuple:
    """Best first: cold score, then the editor's own rank, then id."""
    score = draft.get("score")
    score = float(score) if isinstance(score, (int, float)) else float("-inf")
    rank = draft.get("editor_rank")
    rank = rank if isinstance(rank, int) else 1_000_000
    return (-score, rank, str(draft.get("id", "")))


def _story(draft: dict) -> str:
    key = " ".join(str(draft.get("story_key") or "").lower().split())
    return key or f"#{draft.get('id')}"


def _sensible(draft: dict) -> bool:
    critique = draft.get("critique") or {}
    if critique.get("error"):
        return True   # not reviewed: the judge watches it instead
    return bool(critique.get("makes_sense", True))


def compare_set(drafts: list[dict], *, size: int) -> list[dict]:
    """The drafts that go to the head-to-head ranking, best first.

    Hard gate first (the critic could follow it, or never reviewed it), then
    the best draft of each story by cold score (editor order breaks ties).
    When there are fewer stories than ``size``, each story's runner-up fills
    the remaining places — so variants compete only when there is no second
    story to rank."""
    if size <= 0:
        return []
    groups: dict[str, list[dict]] = {}
    for d in sorted((d for d in drafts if _sensible(d)), key=order_key):
        groups.setdefault(_story(d), []).append(d)
    chosen = sorted((g[0] for g in groups.values()), key=order_key)[:size]
    if len(chosen) < size:
        runners = sorted((g[1] for g in groups.values() if len(g) > 1), key=order_key)
        chosen += runners[: size - len(chosen)]
    return sorted(chosen, key=order_key)


def merge_ranking(ranked_ids: list, group: list[dict], others: list[dict]) -> list[dict]:
    """The full order after ranking: the judge's order for the compare set
    (ids it invented are ignored; ones it forgot follow in score order), then
    everything that wasn't compared, by score."""
    by_id = {d["id"]: d for d in group}
    out: list[dict] = []
    seen: set[str] = set()
    for raw in ranked_ids or []:
        did = str(raw)
        if did in by_id and did not in seen:
            out.append(by_id[did])
            seen.add(did)
    for d in sorted(group, key=order_key):
        if d["id"] not in seen:
            out.append(d)
            seen.add(d["id"])
    out.extend(d for d in sorted(others, key=order_key) if d["id"] not in seen)
    return out


_HOLDS = (("makes_sense", "no longer makes sense"), ("ends_well", "no longer ends well"),
          ("premise_clear", "lost the premise"),
          ("speech_understood", "has speech the audience can't follow"),
          ("follows_advice", "no longer follows your advice"))


def new_failure(before: dict, after: dict) -> str:
    """What a revision broke that the original got right ("" if nothing):
    following with no context, ending on the payoff, the premise, speech the
    audience understands, not being a monologue, the operator's advice,
    staying on topic. A test a critique doesn't answer counts as passed."""
    for key, broke in _HOLDS:
        if before.get(key, True) and not after.get(key, True):
            return broke
    if not before.get("monologue", False) and after.get("monologue", False):
        return "became a reporter monologue"
    was = str(before.get("topic") or "central").lower()
    now = str(after.get("topic") or "central").lower()
    if was in ON_TOPIC and now not in ON_TOPIC:
        return "drifted off topic"
    return ""


def _score(draft: dict) -> float:
    score = draft.get("score")
    return float(score) if isinstance(score, (int, float)) else UNREVIEWED_SCORE


def revision_blocker(original: dict, revision: dict) -> str:
    """Why a reviewed revision can't replace its original, before the judge
    is asked ("" when it may): it must have been reviewed, break nothing the
    original got right, and score at least as high cold."""
    after = revision.get("critique") or {}
    if after.get("error"):
        return "the critic couldn't review it"
    broke = new_failure(original.get("critique") or {}, after)
    if broke:
        return f"it {broke}"
    if _score(revision) < _score(original):
        return f"it scored lower cold ({_score(revision):.2f} < {_score(original):.2f})"
    return ""


def pick_top(ordered: list[dict], *, keep: int,
             max_overlap: float = 0.6) -> tuple[list[dict], list[tuple[dict, dict, float]]]:
    """The first ``keep`` drafts of the order, skipping any that shares more
    than ``max_overlap`` (IoU) with one already picked: the operator should
    get choices, not one clip three times. Returns ``(chosen, skipped)`` with
    ``(draft, the pick it clashed with, iou)`` per skip."""
    chosen: list[dict] = []
    skipped: list[tuple[dict, dict, float]] = []
    for d in ordered:
        if len(chosen) >= keep:
            break
        clash = None
        for c in chosen:
            overlap = clip_proposals.iou(d["segments"], c["segments"])
            if overlap > max_overlap:
                clash = (d, c, overlap)
                break
        if clash:
            skipped.append(clash)
        else:
            chosen.append(d)
    return chosen, skipped


def transcript_clip_entries(transcript: list[dict], segments: list[dict]) -> list[dict]:
    """Transcript entries mapped onto the clip's clock, in play order, as
    ``[{start, end, text}]`` (a line that spans a cut appears in each
    segment it reaches). The fallback for videos without a Whisper word
    stream, so the critic still knows what is said and roughly when; the
    entry shape (not ``word``) tells the critic these are caption lines, not
    words to regroup."""
    out: list[dict] = []
    offset = 0.0
    for s in segments or []:
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b <= a:
            continue
        for entry in transcript or []:
            try:
                e_start, e_end = float(entry["start"]), float(entry["end"])
                text = " ".join(str(entry.get("text", "")).split())
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            if not text or e_end <= a or e_start >= b:
                continue
            out.append({"start": round(offset + max(0.0, e_start - a), 3),
                        "end": round(offset + min(b - a, e_end - a), 3), "text": text})
        offset += b - a
    return out


def failed_critique(error: BaseException | str) -> dict:
    """The §3 failure critique: error set, score 0, booleans false, topic
    "central" (an outage must not read as an off-topic verdict), plus the
    failed call's usage when the error carries it."""
    out = {"first_impression": "", "stops_scroll_muted": False, "muted_reason": "",
           "stops_scroll": False, "stop_reason": "", "person_at_start": False,
           "about": "", "makes_sense": False, "dangling": "", "premise_clear": False,
           "speech_understood": False, "monologue": False, "bored_at": None,
           "dead_air": [], "ends_well": False, "topic": "central", "topic_note": "",
           "person": "", "visual": 0, "audio": 0, "score": 0, "fix": "",
           "error": _brief(error) or "the critic failed"}
    usage = getattr(error, "usage", None)
    if isinstance(usage, dict):
        out["usage"] = usage
    return out


def clean_shot_map(raw: Any, duration: float) -> dict:
    """The shot map in the shape the snapper and editor expect, whatever
    came back: cuts sorted by time, quiet spans as ``[start, end]``."""
    raw = raw if isinstance(raw, dict) else {"error": "no shot map"}
    cuts = []
    for c in raw.get("cuts") or []:
        try:
            cuts.append({"t": float(c["t"]), "score": float(c.get("score", 0.0))})
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    quiet = []
    for q in raw.get("quiet") or []:
        try:
            a, b = float(q[0]), float(q[1])
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if b > a:
            quiet.append([a, b])
    return {"version": raw.get("version", 1),
            "duration": _money(raw.get("duration")) or float(duration or 0.0),
            "cuts": sorted(cuts, key=lambda c: c["t"]), "quiet": quiet,
            "error": str(raw.get("error") or "")}


def _brief(error: BaseException | str | None, limit: int = 240) -> str:
    if error is None:
        return ""
    if isinstance(error, BaseException):
        text = str(error).strip()
        text = f"{type(error).__name__}: {text}" if text else type(error).__name__
    else:
        text = str(error)
    return " ".join(text.split())[:limit]


def _usage_of(error: BaseException) -> dict | None:
    """``.usage`` on the error or on what it was raised from (a collaborator
    may wrap a provider error)."""
    seen = 0
    while error is not None and seen < 4:
        usage = getattr(error, "usage", None)
        if isinstance(usage, dict):
            return usage
        error = error.__cause__ or error.__context__
        seen += 1
    return None


# ---- settings ----------------------------------------------------------------------

_SNAP_DEFAULTS = {"word_window": 0.6, "picture_window": 1.0, "flash_guard": 0.4,
                  "start_pad": 0.08, "end_pad": 0.25, "min_shot_score": 20.0}


def _count(value: Any) -> int:
    return int(float(value))


def _number(settings, key: str, default, *, cast: Callable = float, low=None):
    raw = opt(settings, key, default)
    try:
        value = cast(raw)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(raw)
    except (TypeError, ValueError):
        log.warning("Setting %s=%r isn't a number; using %r", key, raw, default)
        value = cast(default)
    if low is not None and value < low:
        value = cast(low)
    return value


# Mirrors the "content" block ``llm._operator_rules`` writes, for runs whose
# only rules come from ``clip_engine.extra_rules``.
_RULES_PREAMBLE = ("\n\nTHE OPERATOR'S RULES — hard constraints that outrank everything "
                   "else here:\n")
_RULES_CODA = ("Material that challenges these rules or the account's premise is "
               "never credible. It can still be worth using — as something to "
               "respond to — so judge that on its merits rather than discarding it.")
_EXTRA_RULE_CHARS = 300


def extra_rules(settings) -> list[str]:
    """``clip_engine.extra_rules``: extra rule texts for this run (a list of
    strings, or one string), tidied; [] when unset or unusable."""
    raw = opt(settings, "clip_engine.extra_rules", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        text = " ".join(str(item or "").split())[:_EXTRA_RULE_CHARS].strip()
        text = text.lstrip("-• ").strip()
        if text and text not in out:
            out.append(text)
    return out


def with_extra_rules(rules: str, extra: list[str]) -> str:
    """The operator's rules block with ``extra`` added as more bullets: after
    the existing bullets (before the block's closing note) when there is a
    block, else a new block with the same preamble ``llm._operator_rules``
    uses."""
    bullets = "\n".join(f"- {text}" for text in extra or [])
    if not bullets:
        return rules or ""
    rules = rules or ""
    if not rules.strip() or ("THROUGHLINE'S RULES" in rules and "THE OPERATOR'S RULES" not in rules):
        # No account block yet (at most Throughline's universal rules, which
        # stay as they are): the extra rules start the account's block.
        return rules + _RULES_PREAMBLE + bullets + "\n" + _RULES_CODA
    marker = "\nMaterial that challenges"
    at = rules.rfind(marker)
    if at >= 0:
        return rules[:at].rstrip("\n") + "\n" + bullets + rules[at:]
    return rules.rstrip("\n") + "\n" + bullets


def _policy_number(value: float):
    """Whole numbers hash as ints, the way the YAML writes them, so the
    policy fingerprint doesn't change with how a value was parsed."""
    return int(value) if float(value).is_integer() else value


@dataclass(frozen=True)
class _Config:
    drafts: int
    section_drafts: int
    keep: int
    section_keep: int
    compare_top: int
    revise_top: int
    min_s: float
    max_s: float
    max_segments: int
    max_usd_per_video: float
    hard_cap_usd: float
    proxy_height: int
    editor_video_max_seconds: float
    editor_keyframes: int
    snap: dict

    @classmethod
    def read(cls, settings) -> "_Config":
        min_s = _number(settings, "clip_engine.min_seconds", 12, low=1.0)
        max_s = max(min_s, _number(settings, "clip_engine.max_seconds", 45, low=1.0))
        return cls(
            drafts=_number(settings, "clip_engine.drafts", 10, cast=_count, low=1),
            section_drafts=_number(settings, "clip_engine.section_drafts", 6, cast=_count, low=1),
            keep=_number(settings, "clip_engine.keep", 3, cast=_count, low=1),
            section_keep=_number(settings, "clips.section_max_clips", 2, cast=_count, low=1),
            compare_top=_number(settings, "clip_engine.compare_top", 5, cast=_count, low=0),
            revise_top=_number(settings, "clip_engine.revise_top", 1, cast=_count, low=0),
            min_s=min_s,
            max_s=max_s,
            max_segments=_number(settings, "clip_engine.max_segments", 6, cast=_count, low=1),
            max_usd_per_video=_number(settings, "clip_engine.max_usd_per_video", 1.50, low=0.0),
            hard_cap_usd=_number(settings, "clip_engine.hard_cap_usd", 4.00, low=0.0),
            proxy_height=_number(settings, "clip_engine.proxy_height", 360, cast=_count, low=90),
            editor_video_max_seconds=_number(settings, "clip_engine.editor_video_max_seconds",
                                             3600, low=0.0),
            editor_keyframes=_number(settings, "clip_engine.claude.editor_keyframes", 40,
                                     cast=_count, low=0),
            snap={k: _number(settings, f"clip_engine.snap.{k}", v) for k, v in _SNAP_DEFAULTS.items()},
        )


# ---- one run ------------------------------------------------------------------------

_DRAFT_FIELDS = ("id", "moment_id", "archetype", "lens", "story", "story_key", "hook",
                 "payoff", "segments", "why", "risk", "draft_caption", "challenges_rules",
                 "origin", "parent", "editor_rank")

DROPPED_LENGTH = "length"
DROPPED_DUPLICATE = "duplicate"
DROPPED_REJECTED = "rejected before"
DROPPED_RENDER = "render failed"
DROPPED_OFF_TOPIC = "off topic"
DROPPED_LANGUAGE = "language"              # speech the audience can't follow
DROPPED_SAME_OPENING = "same opening"      # opens on another clip's opening shot
DROPPED_SAME_CLIP = "same clip"            # IoU > 0.6 with a clip already made
DROPPED_REVISED = "revised"                # replaced in place by its adopted revision
DROPPED_REVISION = "revision not used"     # a revision that lost to its original


def _bookkeeping(segments: list[dict]) -> dict:
    return {"segments": segments, "editor_segments": [dict(s) for s in segments],
            "dropped_reason": "", "critique": None, "score": None, "render": "",
            "credits": "", "seconds": seconds_of(segments), "topic_doubt": "",
            "language_doubt": ""}


def adopt_drafts(raw: list) -> list[dict]:
    """The editor's drafts with the engine's bookkeeping attached (and ids
    made unique, defensively). ``editor_segments`` keeps the boundaries the
    editor asked for, before snapping, so the bench can tell the two apart."""
    out: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        d = dict(item)
        did = str(d.get("id") or f"d{index + 1}")
        if did in seen:
            did = f"{did}-{index + 1}"
        seen.add(did)
        segments = [dict(s) for s in (d.get("segments") or []) if isinstance(s, dict)]
        rank = d.get("editor_rank")
        d.update(_bookkeeping(segments))
        d.update({"id": did, "editor_rank": rank if isinstance(rank, int) else index,
                  "origin": d.get("origin") or "draft", "parent": d.get("parent")})
        out.append(d)
    return out


def adopt_revision(original: dict, proposal: dict) -> dict:
    """A revision as a draft of its own: the editor's fields over the
    original's (so a sparse answer still has a story and archetype), linked
    to its parent, sharing the parent's editor rank."""
    revision = {k: original[k] for k in _DRAFT_FIELDS if k in original}
    revision.update({k: v for k, v in proposal.items() if v is not None})
    segments = [dict(s) for s in (revision.get("segments") or []) if isinstance(s, dict)]
    revision.update(_bookkeeping(segments))
    revision.update({"id": str(proposal.get("id") or f"{original['id']}r"),
                     "origin": "revision", "parent": original["id"],
                     "editor_rank": original.get("editor_rank"),
                     "dropped_reason": DROPPED_REVISION})
    return revision


class _Run:
    """One engine run: its inputs, what each stage produced, and the
    bookkeeping (notes, timings, the meter, the cancel/deadline/cost checks).
    Stages are methods so ``execute`` reads as the pipeline it is."""

    def __init__(self, *, setup, settings, brand, playbook, video_id, title, description,
                 duration, source, words, transcript, context, rejected, examples, used,
                 window, section_topic, keep, meter, progress, cancel, deadline,
                 advice="", other_clips=None) -> None:
        self.setup = setup
        self.roles = ROLES[setup]
        self.settings = settings
        self.cfg = _Config.read(settings)
        self.brand = dict(brand or {})
        self.playbook = playbook
        self.video_id = video_id
        self.title = title or ""
        self.description = description or ""
        self.duration = duration
        self.source = Path(source)
        self.words = list(words or [])
        self.transcript = list(transcript or [])
        self.context = context or {}
        self.rejected = list(rejected or [])
        self.examples = [e for e in (examples or []) if isinstance(e, dict)]
        # Kept for old callers, never used: clips may share footage.
        self.used = list(used or [])
        # The clips already made from this video. Their footage is fair game;
        # their opening shots and their stories are not.
        self.other_clips = clean_other_clips(other_clips)
        self.opening_blocks = clip_proposals.opening_windows(self.other_clips)
        self.window = window
        self.section_topic = section_topic or ""
        self.advice = prompts.clean_advice(advice)
        self.keep = keep
        self.meter = meter if meter is not None else CostMeter("engine")
        self.progress = progress
        self.cancel = cancel
        self.deadline = deadline

        self.notes: list[str] = []
        self.timings: dict[str, float] = {}
        self.orders: dict[str, list[str]] = {"editor": [], "cold": [], "rank": [], "final": []}
        self.log: dict | None = None
        self.shot_map: dict = clean_shot_map({}, 0.0)
        self.drafts: list[dict] = []
        self.moments: list[dict] = []
        self.order: list[dict] = []
        self.top: list[dict] = []
        self.source_video: Path | None = None
        self.entries: list[dict] = []
        self.has_words = bool(self.words)
        self.playbook_text = ""
        self.rules = ""
        self.floor = 0.0
        self.horizon = 0.0
        self.model = ""
        self.policy = ""

    # -- bookkeeping ------------------------------------------------------------

    def note(self, text: str) -> None:
        self.notes.append(text)
        log.info("%s [%s]: %s", self.video_id, self.setup, text)

    def say(self, label: str) -> None:
        log.info("%s [%s]: %s", self.video_id, self.setup, label)
        if self.progress is None:
            return
        try:
            self.progress(label)
        except Exception as exc:  # a broken status line must not stop the run
            log.debug("progress callback failed: %s", exc)

    def checkpoint(self) -> None:
        """Between stages and provider batches: stop on cancel or deadline
        (``util.Cancelled``) and on the hard cost cap (``EngineError``)."""
        util.check(self.cancel, self.deadline)
        if self.meter.total > self.cfg.hard_cap_usd:
            raise EngineError(f"stopped at ${self.meter.total:.2f}, past the hard cap "
                              f"(clip_engine.hard_cap_usd ${self.cfg.hard_cap_usd:.2f})")

    @contextlib.contextmanager
    def stage(self, key: str, label: str = "") -> Iterator[None]:
        if label:
            self.say(label)
        started = time.monotonic()
        try:
            yield
        finally:
            self.timings[key] = round(self.timings.get(key, 0.0) + time.monotonic() - started, 2)
        self.checkpoint()

    def meter_failure(self, stage: str, error: BaseException, before: float) -> None:
        """Count what a failed call cost. Providers record spend before
        raising and attach it as ``usage``; it goes on the meter here unless
        the collaborator already counted it (the meter moved during the
        call), so it is never counted twice."""
        usage = _usage_of(error)
        if usage is not None and self.meter.total == before:
            self.meter.add(stage, usage)

    def live(self) -> list[dict]:
        return [d for d in self.drafts if not d["dropped_reason"]]

    def rendered(self) -> list[str]:
        """Every render this run made so far (drafts and revisions)."""
        return [str(d["render"]) for d in self.drafts if d.get("render")]

    def fits(self, seconds: float) -> bool:
        return seconds > 0 and 0.8 * self.cfg.min_s <= seconds <= 1.3 * self.cfg.max_s

    # -- the pipeline ---------------------------------------------------------------

    def execute(self) -> EngineResult:
        started = time.monotonic()
        self.checkpoint()
        self.prepare()
        with self.stage("perceive", "Logging the video"):
            self.perceive()
        with self.stage("draft", "Drafting clips"):
            self.draft()
        if not self.drafts:
            self.note("The editor found nothing worth drafting.")
            return self.result("empty_drafts", started)
        with self.stage("snap"):
            self.snap_all()
            self.filter_all()
        examples = self.start_examples()
        with self.stage("render", "Rendering drafts"):
            self.render_all()
        with self.stage("watch", "Watching the drafts"):
            self.watch_all()
            self.topic_gate()
            self.language_gate()
        if not self.live():
            return self.result("empty_topic", started)
        with self.stage("rank", "Ranking"):
            self.rank(examples)
        if self.cfg.revise_top > 0:
            with self.stage("revise", "Revising"):
                self.revise()
        self.choose()
        return self.result("proposals", started)

    def prepare(self) -> None:
        """Resolve what the stages share: the length, the section bounds, the
        playbook and rules, how many to keep, and the ledger labels."""
        if self.duration is None or not _money(self.duration) > 0:
            try:
                self.duration = media.probe_duration(self.source)
            except Exception as exc:
                log.warning("Couldn't probe %s: %s", self.source, exc)
                self.duration = None
        if self.duration is None or not _money(self.duration) > 0:
            raise EngineError(f"couldn't read the length of {self.source.name}")
        self.duration = float(self.duration)

        if self.window:
            try:
                start = max(0.0, float(self.window["start"]))
                end = min(self.duration, float(self.window["end"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise EngineError(f"bad section {self.window!r}") from exc
            if end - start < 1.0:
                raise EngineError(f"the section {start:.0f}-{end:.0f} s is empty")
            self.window = {"start": start, "end": end}
            self.floor, self.horizon = start, end
        else:
            self.window = None
            self.floor, self.horizon = 0.0, self.duration

        if self.keep is None:
            self.keep = self.cfg.section_keep if self.window else self.cfg.keep
        self.keep = max(1, int(self.keep))
        if self.advice:
            self.note(f"advice: {self.advice}")
        if self.playbook is None:
            self.playbook = playbooks.load()
        self.playbook_text = playbooks.render(self.playbook)
        house = self.house_rules()
        if house:
            # After a blank line: editor._archetype_keys reads archetypes only
            # up to the first one, so a house rule is never taken for one.
            self.playbook_text = self.playbook_text.rstrip() + "\n\n" + house
            self.note("Used the house rules learned from your past edits.")
        self.rules = llm._operator_rules("content")
        extra = extra_rules(self.settings)
        if extra:
            self.rules = with_extra_rules(self.rules, extra)
            self.note(f"Extra rules for this run: {len(extra)}.")
        self.model = f"engine:{self.setup}:{models_for(self.setup, self.settings)['editor']}"[:60]
        self.policy = clip_proposals.policy_version(
            model=self.model, max_clips=self.keep,
            max_segments_per_clip=self.cfg.max_segments,
            min_seconds=_policy_number(self.cfg.min_s),
            max_seconds=_policy_number(self.cfg.max_s),
            scope="section" if self.window else "video",
            prompt=f"{clip_proposals.ENGINE_PROMPT_VERSION}:{self.setup}")

    def house_rules(self) -> str:
        """Whatever ``handbook.render_active`` returns (approved house rules
        from the operator's past edits), or "" when there are none or the
        handbook can't be read — a broken handbook must never stop a run."""
        try:
            from . import handbook

            text = handbook.render_active()
        except Exception as exc:  # noqa: BLE001 - optional extra; the run goes on
            log.warning("%s: the handbook couldn't be read (%s); running without it",
                        self.video_id, exc)
            return ""
        return str(text or "").strip()

    # 1. perceive ---------------------------------------------------------------------

    def perceive(self) -> None:
        """The shot log (cached per video by ``shotlog``) while ffmpeg maps
        the shot changes and pauses on a daemon thread alongside it.

        Logging a long source is a dozen or more window calls, the costliest
        stretch of a run and the one that grows with the source's length, so
        the checkpoint runs inside it too — before every window and every
        split (``check``) — not only once the whole log is back: a four-hour
        hearing stops at the hard cap or the deadline instead of paying for
        every window first. Windows logged before the stop stay cached, so a
        later run picks up where this one stopped."""
        source = self.source
        wait_map = util.run_in_thread(lambda: media.shot_map(source),
                                      name="clip-engine-shotmap")
        perceiver = self.roles["perceiver"]
        before = self.meter.total
        try:
            found = shotlog.build(
                video_id=self.video_id, source=source, perceiver=perceiver,
                settings=self.settings, brand=self.brand, title=self.title,
                description=self.description, duration=self.duration, words=self.words,
                transcript=self.transcript, window=self.window, meter=self.meter,
                check=self.checkpoint, cancel=self.cancel)
        except (util.Cancelled, EngineError):
            # Canceled, out of time, or past the hard cap (the checkpoint's
            # own errors): not a perceiver failure, so not reworded as one.
            raise
        except Exception as exc:
            self.meter_failure("perceive", exc, before)
            raise EngineError(f"the {perceiver} perceiver failed: {_brief(exc)}") from exc
        if not isinstance(found, dict) or not found.get("beats"):
            raise EngineError(f"the {perceiver} perceiver logged nothing")
        if found.get("complete") is False:
            self.note("Part of the video couldn't be logged; the editor worked from the rest.")
        self.log = found
        try:
            raw_map = wait_map()
        except Exception as exc:
            raw_map = {"error": _brief(exc)}
        self.shot_map = clean_shot_map(raw_map, self.duration)
        if self.shot_map["error"]:
            self.note(f"No shot map ({self.shot_map['error']}): boundaries snap to words only.")

    # 2. draft ------------------------------------------------------------------------

    def draft(self) -> None:
        role = self.roles["editor"]
        keyframe_frames: list = []
        if role == "gemini":
            self.source_video = self.whole_source_proxy()
        else:
            keyframe_frames = self.keyframes()
        before = self.meter.total
        try:
            self.entries, self.has_words = editor.entries_for(self.words, self.transcript)
            drafts, moments = editor.draft(
                role=role, settings=self.settings, brand=self.brand,
                playbook_text=self.playbook_text, rules=self.rules, title=self.title,
                description=self.description, duration=self.duration, shotlog=self.log,
                shot_map=self.shot_map, entries=self.entries, has_words=self.has_words,
                context=self.context, other_clips=self.other_clips, window=self.window,
                section_topic=self.section_topic,
                n=self.cfg.section_drafts if self.window else self.cfg.drafts,
                min_s=self.cfg.min_s, max_s=self.cfg.max_s,
                max_segments=self.cfg.max_segments, source_video=self.source_video,
                keyframe_frames=keyframe_frames, meter=self.meter, advice=self.advice)
        except util.Cancelled:
            raise
        except Exception as exc:
            self.meter_failure("draft", exc, before)
            raise EngineError(f"drafting failed: {_brief(exc)}") from exc
        self.moments = [m for m in (moments or []) if isinstance(m, dict)]
        self.drafts = adopt_drafts(drafts or [])

    def whole_source_proxy(self) -> Path | None:
        """The video a Gemini editor watches: a proxy of the WHOLE source on
        the source clock (never a section proxy, whose 00:00 is the section
        start), or nothing past ``editor_video_max_seconds``."""
        limit = self.cfg.editor_video_max_seconds
        if self.duration > limit:
            self.note(f"The editor drafted from the shot log alone: the source runs "
                      f"{self.duration / 60:.0f} min, past editor_video_max_seconds "
                      f"({limit / 60:.0f} min).")
            return None
        try:
            return media.perception_proxy(self.source)
        except Exception as exc:
            self.note(f"The editor couldn't watch the video ({_brief(exc)}); "
                      "it drafted from the shot log.")
            return None

    def keyframes(self) -> list:
        """Stills of the strongest shots for a Claude editor, which can't
        watch video."""
        try:
            return list(editor.keyframes(self.source, self.log, self.shot_map,
                                         settings=self.settings,
                                         limit=self.cfg.editor_keyframes) or [])
        except Exception as exc:
            self.note(f"No keyframes for the editor ({_brief(exc)}).")
            return []

    # 3-4. snap and filter ---------------------------------------------------------------

    def snap(self, segments: list[dict], *, label: str) -> list[dict]:
        if not segments:
            return []
        try:
            out = snap.snap_segments(
                segments, words=self.words, cuts=self.shot_map["cuts"],
                quiet=self.shot_map["quiet"], floor=self.floor, horizon=self.horizon,
                blocked=self.opening_blocks, **self.cfg.snap)
        except Exception as exc:
            log.warning("Snapping %s failed: %s", label, exc)
            self.note(f"{label}: snapping failed ({_brief(exc)}); kept the editor's boundaries.")
            return [dict(s) for s in segments]
        return [dict(s) for s in (out or [])]

    def snap_all(self) -> None:
        """Every boundary onto a word edge, shot change or pause; then the
        length bounds again (0.8 x min .. 1.3 x max), since snapping moves
        them."""
        for d in self.live():
            d["segments"] = self.snap(d["segments"], label=d["id"])
            d["seconds"] = seconds_of(d["segments"])
            if not self.fits(d["seconds"]):
                d["dropped_reason"] = DROPPED_LENGTH
        short = [d for d in self.drafts if d["dropped_reason"] == DROPPED_LENGTH]
        if short:
            self.note(f"Outside {0.8 * self.cfg.min_s:.1f}-{1.3 * self.cfg.max_s:.1f} s after "
                      "snapping: " + ", ".join(f"{d['id']} ({d['seconds']:.1f} s)" for d in short))

    def conflict(self, segments: list[dict]) -> tuple[str, dict | None]:
        """``clip_proposals.clip_conflict`` against this video's other clips,
        with the shot map's cuts when there is one (two openings inside the
        same shot are the same opening shot)."""
        if not self.other_clips:
            return "", None
        cuts = [] if self.shot_map.get("error") else self.shot_map.get("cuts") or []
        return clip_proposals.clip_conflict(
            segments, self.other_clips, cuts=cuts,
            min_shot_score=float(self.cfg.snap.get("min_shot_score", 20.0)))

    def drop_conflict(self, d: dict) -> bool:
        """Drop a draft that opens on another clip's opening shot or is
        another clip again. Recycled footage further in is fine."""
        kind, other = self.conflict(d["segments"])
        if not kind:
            return False
        title = (other or {}).get("title") or "another clip"
        if kind == clip_proposals.CONFLICT_DUPLICATE:
            d["dropped_reason"] = DROPPED_SAME_CLIP
            self.note(f"{d['id']} is the same clip as “{title}”, already made.")
        else:
            d["dropped_reason"] = DROPPED_SAME_OPENING
            self.note(f"{d['id']} opens on the same shot as “{title}”.")
        d["conflicts_with"] = title
        return True

    def filter_all(self) -> None:
        """Drop what the operator already dismissed here, drafts that open on
        another clip's opening shot or repeat a clip already made, then true
        duplicates (the editor's earlier draft wins). Nothing left at all is
        a mechanical failure: the caller falls back."""
        kept: list[dict] = []
        for d in self.live():
            dismissed = repeats_rejection(d["segments"], self.rejected)
            if dismissed is not None:
                d["dropped_reason"] = DROPPED_REJECTED
                reason = dismissed.get("reason") or "no reason given"
                self.note(f"{d['id']} repeats a clip you dismissed here ({reason}).")
                continue
            if self.drop_conflict(d):
                continue
            twin = next((k for k in kept if same_cut(k["segments"], d["segments"])), None)
            if twin is not None:
                d["dropped_reason"] = DROPPED_DUPLICATE
                d["duplicate_of"] = twin["id"]
                self.note(f"{d['id']} duplicates {twin['id']}.")
                continue
            kept.append(d)
        if not kept:
            raise EngineError(f"no draft survived snapping and filtering ({self.drop_summary()})")
        self.orders["editor"] = [d["id"] for d in sorted(kept, key=lambda d: (d["editor_rank"], d["id"]))]

    def drop_summary(self) -> str:
        counts: dict[str, int] = {}
        for d in self.drafts:
            if d["dropped_reason"]:
                counts[d["dropped_reason"]] = counts.get(d["dropped_reason"], 0) + 1
        return ", ".join(f"{n} {why}" for why, n in counts.items()) or "no drafts"

    # 5. render ---------------------------------------------------------------------------

    def start_examples(self) -> Callable[[], Any] | None:
        """Fetch the taste examples' files (download + trim, cached) on a
        daemon thread while the drafts render and get watched."""
        if not self.examples:
            return None
        examples, settings, height = list(self.examples), self.settings, self.cfg.proxy_height

        def resolve() -> list[tuple[dict, Path]]:
            pairs = []
            for example in examples:
                try:
                    path = taste.example_file(example, settings, height=height)
                except Exception as exc:
                    log.warning("Taste example %s unavailable: %s", example.get("post_pk"), exc)
                    continue
                if path:
                    pairs.append((example, Path(path)))
            return pairs

        return util.run_in_thread(resolve, name="clip-engine-taste")

    def render_all(self) -> None:
        """A 360p render of every draft, three at a time. A failed render
        drops that draft; drafts existed but none rendered is a mechanical
        failure (``EngineError``), not "nothing to suggest"."""
        drafts = self.live()
        source, height = self.source, self.cfg.proxy_height
        results = util.parallel_map(
            lambda d: media.render_cached(source, plain_segments(d["segments"]), height=height),
            drafts, workers=3, cancel=self.cancel, name="clip-engine-render")
        for d, out in zip(drafts, results):
            if out and not isinstance(out, BaseException):
                # Noted before the check below, so a run stopped here still
                # hands its renders back for cleaning up (``run``).
                d["render"] = str(out)
        util.check(self.cancel, self.deadline)
        failed = []
        for d, out in zip(drafts, results):
            if isinstance(out, BaseException) or not out:
                d["dropped_reason"] = DROPPED_RENDER
                d["render_error"] = _brief(out) if out else "no file"
                failed.append(d)
        if failed:
            log.warning("%s: %d of %d renders failed: %s", self.video_id, len(failed),
                        len(drafts), failed[0]["render_error"])
            self.note(f"{len(failed)} of {len(drafts)} drafts couldn't be rendered "
                      f"({failed[0]['render_error']}).")
        if len(failed) == len(drafts):
            raise EngineError(f"no draft could be rendered: {failed[0]['render_error']}")

    # 6-7. watch, topic gate ----------------------------------------------------------------

    def credits(self, segments: list[dict]) -> str:
        try:
            return str(shotlog.credits_for(self.log, segments) or "")
        except Exception as exc:
            log.warning("credits_for failed: %s", exc)
            return ""

    def clip_words(self, segments: list[dict]) -> list[dict]:
        """What the critic hears, on the clip's clock: the Whisper words, or
        transcript lines for videos that only have captions."""
        if self.words:
            from ..subtitles import slice_source_words

            return slice_source_words(self.words, segments)
        return transcript_clip_entries(self.transcript, segments)

    def watch(self, d: dict) -> dict:
        # Kept on the draft for the judges after the cold critic: a Claude
        # judge (rank, prefer) sees frames only, so the words are its ears.
        # Stripped from the result (``result``): they are bulky and derived.
        d["clip_words"] = self.clip_words(d["segments"])
        return critic.cold_watch(
            role=self.roles["critic"], settings=self.settings, brand=self.brand,
            render=Path(d["render"]), clip_seconds=d["seconds"],
            clip_words=d["clip_words"], credits=d["credits"], meter=self.meter,
            advice=self.advice)

    def cold_score(self, d: dict) -> float:
        critique = d["critique"]
        try:
            return float(critic.cold_score(critique, clip_seconds=d["seconds"],
                                           archetype=str(d.get("archetype") or "")))
        except Exception as exc:
            log.warning("cold_score failed for %s: %s", d["id"], exc)
            return UNREVIEWED_SCORE if critique.get("error") else _money(critique.get("score"))

    def review(self, d: dict, critique: Any) -> None:
        if not isinstance(critique, dict):
            critique = failed_critique(critique if isinstance(critique, BaseException)
                                       else "the critic returned nothing")
        d["critique"] = critique
        d["score"] = self.cold_score(d)

    def watch_all(self) -> None:
        """A stranger's cold read of every render, four at a time. A critique
        that failed keeps its draft as "not reviewed"."""
        drafts = self.live()
        for d in drafts:
            d["credits"] = self.credits(d["segments"])
        critiques = util.parallel_map(self.watch, drafts, workers=4, cancel=self.cancel,
                                      name="clip-engine-critic")
        util.check(self.cancel, self.deadline)
        for d, critique in zip(drafts, critiques):
            self.review(d, critique)
        unreviewed = [d for d in drafts if d["critique"].get("error")]
        if unreviewed:
            self.note(f"{len(unreviewed)} of {len(drafts)} drafts weren't reviewed (critic "
                      f"unavailable: {unreviewed[0]['critique']['error']}).")

    def topic_gate(self) -> None:
        dropped = []
        for d in self.live():
            verdict, why = topic_verdict(d["critique"], material_topic(self.log, d["segments"]))
            if verdict == "drop":
                d["dropped_reason"] = DROPPED_OFF_TOPIC
                d["topic_doubt"] = why
                dropped.append(d)
            elif verdict == "doubt":
                d["topic_doubt"] = why
        if dropped:
            self.note("Off topic: " + "; ".join(f"{d['id']} ({d['topic_doubt']})" for d in dropped))

    def language_gate(self) -> None:
        """Drop drafts whose speech the audience can't follow (critic and
        shot log agree; see ``language_verdict``) — unless that would drop
        every draft left, in which case they all stay with their cold-score
        penalty and a note, so a run never comes back empty on this alone."""
        live = self.live()
        flagged = []
        for d in live:
            verdict, why = language_verdict(d["critique"],
                                            foreign_speech_share(self.log, d["segments"]))
            if verdict == "drop":
                d["language_doubt"] = why
                flagged.append(d)
        if flagged and len(flagged) == len(live):
            self.note("Every draft left has speech the audience can't follow (not English, no "
                      "English voiceover); kept them, scored down: "
                      + ", ".join(d["id"] for d in flagged) + ".")
        elif flagged:
            for d in flagged:
                d["dropped_reason"] = DROPPED_LANGUAGE
            self.note("Not understood (another language, no English voiceover): "
                      + "; ".join(f"{d['id']} ({d['language_doubt']})" for d in flagged))
        self.orders["cold"] = [d["id"] for d in sorted(self.live(), key=order_key)]

    # 8. rank -----------------------------------------------------------------------------

    def examples_ready(self, wait: Callable[[], Any] | None) -> tuple[list[dict], list[Path]]:
        if wait is None:
            return [], []
        try:
            pairs = wait() or []
        except Exception as exc:
            self.note(f"No taste examples ({_brief(exc)}).")
            return [], []
        if len(pairs) < len(self.examples):
            self.note(f"{len(self.examples) - len(pairs)} of {len(self.examples)} taste "
                      "examples had no file; ranked without them.")
        return [e for e, _ in pairs], [p for _, p in pairs]

    def rank(self, examples_wait: Callable[[], Any] | None) -> None:
        """Head-to-head over the compare set, against this account's own hits
        and flops; then everything else by cold score."""
        live = self.live()
        group = compare_set(live, size=self.cfg.compare_top)
        for d in group:
            d["compared"] = True
        ids = {d["id"] for d in group}
        others = [d for d in live if d["id"] not in ids]
        ranked = [d["id"] for d in group]
        if len(group) >= 2:
            examples, files = self.examples_ready(examples_wait)
            before = self.meter.total
            try:
                # renders/credits/scores are keyed by draft id (``scores``
                # doubles as borda's ``tiebreak``); example_files is a list
                # aligned with ``examples``. Each draft also carries its own
                # "render", "credits", "score" and "clip_words" (the words a
                # frames-only Claude judge reads).
                ranked = list(critic.rank(
                    role=self.roles["critic"], settings=self.settings, brand=self.brand,
                    playbook_text=self.playbook_text, drafts=group,
                    renders={d["id"]: Path(d["render"]) for d in group},
                    credits={d["id"]: d["credits"] for d in group},
                    scores={d["id"]: d["score"] for d in group},
                    examples=examples, example_files=files, meter=self.meter,
                    advice=self.advice) or [])
            except util.Cancelled:
                raise
            except Exception as exc:
                self.meter_failure("rank", exc, before)
                log.warning("%s: ranking failed: %s", self.video_id, exc)
                self.note(f"Ranking failed ({_brief(exc)}); ordered by cold score.")
                ranked = [d["id"] for d in group]
        self.order = merge_ranking(ranked, group, others)
        self.orders["rank"] = [d["id"] for d in self.order]

    # 9. revise ---------------------------------------------------------------------------

    def intent(self, d: dict) -> str:
        try:
            return str(playbooks.open_rule(self.playbook, str(d.get("archetype") or "")) or "")
        except Exception:
            return ""

    def revise(self) -> None:
        """One revision round for the best ``revise_top`` drafts, while the
        video's cost (cached perception included) is under
        ``max_usd_per_video``. Any failure keeps the original."""
        for original in list(self.order[: self.cfg.revise_top]):
            spent = self.meter.total_with_cached
            if spent >= self.cfg.max_usd_per_video:
                self.note(f"Revision skipped: this video has cost ${spent:.2f} "
                          f"(max_usd_per_video ${self.cfg.max_usd_per_video:.2f}).")
                break
            if (original.get("critique") or {}).get("error"):
                self.note(f"{original['id']} wasn't revised: the critic never reviewed it.")
                continue
            try:
                self.revise_one(original)
            except (util.Cancelled, EngineError):
                raise
            except Exception as exc:
                log.warning("%s: revising %s failed: %s", self.video_id, original["id"], exc)
                self.note(f"Revising {original['id']} failed ({_brief(exc)}); kept the original.")
            self.checkpoint()

    def revise_one(self, original: dict) -> None:
        before = self.meter.total
        try:
            proposal = editor.revise(
                role=self.roles["editor"], settings=self.settings, brand=self.brand,
                playbook_text=self.playbook_text, rules=self.rules, draft=original,
                critique=original["critique"], shotlog=self.log, shot_map=self.shot_map,
                entries=self.entries, has_words=self.has_words, other_clips=self.other_clips,
                window=self.window, min_s=self.cfg.min_s, max_s=self.cfg.max_s,
                max_segments=self.cfg.max_segments, source_video=self.source_video,
                meter=self.meter, advice=self.advice)
        except Exception as exc:
            self.meter_failure("revise", exc, before)
            raise
        if not isinstance(proposal, dict) or not proposal.get("segments"):
            self.note(f"{original['id']}: the editor kept its cut.")
            return
        revision = adopt_revision(original, proposal)
        if any(d["id"] == revision["id"] for d in self.drafts):
            # Ids key the bench's order analysis; never let a revision shadow a draft.
            revision["id"] = f"{original['id']}r{len(self.drafts) + 1}"
        self.drafts.append(revision)
        try:
            blocker = self.judge_revision(original, revision)
        except (util.Cancelled, EngineError):
            raise
        except Exception as exc:
            log.warning("%s: revision %s failed: %s", self.video_id, revision["id"], exc)
            blocker = f"it failed ({_brief(exc)})"
        if blocker:
            revision["revision_note"] = blocker
            self.note(f"Revision of {original['id']} not used: {blocker}.")
            return
        index = next(i for i, d in enumerate(self.order) if d is original)
        self.order[index] = revision
        original["dropped_reason"] = DROPPED_REVISED
        original["revised_to"] = revision["id"]
        revision["dropped_reason"] = ""
        revision["adopted"] = True
        change = " ".join(str(revision.get("change") or "").split())
        self.note(f"Revision of {original['id']} adopted" + (f": {change}" if change else "") + ".")

    def judge_revision(self, original: dict, revision: dict) -> str:
        """Snap, render and cold-watch the revision, then ask the judge — in
        both presentation orders — only if nothing cheaper already rules it
        out. Returns why it can't replace the original, or ""."""
        revision["segments"] = self.snap(revision["segments"], label=revision["id"])
        revision["seconds"] = seconds_of(revision["segments"])
        if not self.fits(revision["seconds"]):
            return f"it ran {revision['seconds']:.1f} s after snapping"
        if plain_segments(revision["segments"]) == plain_segments(original["segments"]):
            return "it came out the same as the original after snapping"
        if repeats_rejection(revision["segments"], self.rejected) is not None:
            return "it repeats a clip you dismissed"
        kind, other = self.conflict(revision["segments"])
        if kind:
            title = (other or {}).get("title") or "another clip"
            return (f"it is the same clip as “{title}”" if kind == clip_proposals.CONFLICT_DUPLICATE
                    else f"it opens on the same shot as “{title}”")
        revision["render"] = str(media.render_cached(
            self.source, plain_segments(revision["segments"]), height=self.cfg.proxy_height))
        revision["credits"] = self.credits(revision["segments"])
        self.review(revision, self.watch(revision))
        blocker = revision_blocker(original, revision)
        if blocker:
            return blocker
        self.checkpoint()
        before = self.meter.total
        try:
            choice = critic.prefer(
                role=self.roles["critic"], settings=self.settings, brand=self.brand,
                a=original, b=revision, render_a=Path(original["render"]),
                render_b=Path(revision["render"]),
                # One credits line per version (critic.prefer takes {"a","b"}):
                # the two cuts can put the same person at different clip times.
                credits={"a": original.get("credits", ""), "b": revision.get("credits", "")},
                intent=self.intent(original), meter=self.meter, advice=self.advice)
        except util.Cancelled:
            raise
        except Exception as exc:
            self.meter_failure("prefer", exc, before)
            return f"the judge failed ({_brief(exc)})"
        if str(choice or "").strip().lower() != "b":
            return "the judge kept the original"
        return ""

    # 10. choose --------------------------------------------------------------------------

    def choose(self) -> None:
        chosen, skipped = pick_top(self.order, keep=self.keep)
        for d, clash, overlap in skipped:
            self.note(f"{d['id']} skipped: it overlaps {clash['id']} (IoU {overlap:.2f}).")
        self.top = [dict(d, rank=i + 1) for i, d in enumerate(chosen)]
        self.orders["final"] = [d["id"] for d in chosen]

    def result(self, outcome: str, started: float) -> EngineResult:
        self.timings["total"] = round(time.monotonic() - started, 2)
        for d in (*self.drafts, *self.top):
            d.pop("clip_words", None)   # judge input only; see ``watch``
        spent = self.meter.snapshot()
        log.info("%s [%s]: %s, %d drafts, top %s, $%.3f, %.0f s", self.video_id, self.setup,
                 outcome, len(self.drafts), self.orders["final"], spent["cost_usd"],
                 self.timings["total"])
        return EngineResult(
            setup=self.setup, top=self.top, drafts=self.drafts, moments=self.moments,
            shotlog=self.log, cost_usd=spent["cost_usd"], by_stage=spent["by_stage"],
            tokens=spent["tokens"], timings=dict(self.timings), notes=list(self.notes),
            orders={k: list(v) for k, v in self.orders.items()}, model=self.model,
            policy=self.policy, outcome=outcome)


def run(*, setup: str, settings, brand: dict, playbook: dict | None = None, video_id: str,
        title: str, description: str, duration: float | None, source, words: list[dict],
        transcript: list[dict], context: dict, rejected: list[dict], examples: list[dict],
        used: list[dict] | None = None, window: dict | None = None, section_topic: str = "",
        keep: int | None = None, meter: CostMeter | None = None,
        progress: Callable[[str], Any] | None = None,
        cancel: threading.Event | None = None, deadline: float | None = None,
        advice: str = "", other_clips: list[dict] | None = None) -> EngineResult:
    """Suggest clips for one video (or one section of it) by watching it.

    Pure: no database. ``source`` must be a local file for the whole run (jobs
    hardlinks it into ``engine_dir()/inflight``). ``words``/``transcript`` are
    on the source clock; ``other_clips`` the clips already made from this
    video (``[{title, words_heard, segments, opening?}]``, segments in play
    order): their footage may be reused, but a draft may not open on another
    clip's opening shot (snapping keeps first segments out of those windows;
    a draft that still opens there, or within the same shot-map shot, is
    dropped as "same opening") nor repeat one (IoU > 0.6: "same clip"), and
    the editor is told to tell a different story. ``used`` is accepted for
    old callers and ignored. ``rejected`` the dismissed proposals
    on this video (``[{segments, reason}]``); ``examples`` this account's taste
    examples (``taste.select_examples``). ``window`` (``{"start", "end"}``)
    scopes the run to one section. ``keep`` defaults to ``clip_engine.keep``
    (``clips.section_max_clips`` for a section). Pass ``meter`` to keep a
    partial cost when the run raises. ``progress(label)`` hears each stage;
    ``cancel`` / ``deadline`` (monotonic) and the hard cost cap are checked
    between stages and provider batches, and inside perception before every
    window the shot log pays for. ``advice`` is the operator's plain-words
    steer for this pass: every choosing call reads it (above the playbook),
    the cold critic checks each draft against it, and it is recorded in
    ``notes`` as "advice: …". Rejected proposals still filter as usual.

    Raises ``EngineError`` for anything the caller should answer with the
    transcript suggester, and ``util.Cancelled`` when stopped. Either carries
    what the run had when it stopped: ``notes`` (the best clue to why) and
    ``renders``, the draft renders it made — files nobody will be shown, for
    the caller to delete or keep (a retry reuses them from the render cache).
    An empty result (``outcome`` "empty_drafts" or "empty_topic") is an
    answer, not an error.
    """
    if setup not in ROLES:
        raise EngineError(f"setup {setup!r} has no engine pipeline")
    job = _Run(setup=setup, settings=settings, brand=brand, playbook=playbook,
               video_id=video_id, title=title, description=description, duration=duration,
               source=source, words=words, transcript=transcript, context=context,
               rejected=rejected, examples=examples, used=used, window=window,
               section_topic=section_topic, keep=keep, meter=meter, progress=progress,
               cancel=cancel, deadline=deadline, advice=advice, other_clips=other_clips)
    try:
        return job.execute()
    except (EngineError, util.Cancelled) as exc:
        if not getattr(exc, "notes", None):
            exc.notes = list(job.notes)
        exc.renders = job.rendered()
        if isinstance(exc, EngineError):
            log.warning("%s [%s]: engine stopped: %s", video_id, setup, exc)
        raise
