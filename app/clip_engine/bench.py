"""Replay benchmark: run the clip engine's setups over past videos and judge
them against what this account actually shipped.

A suggester is only better if it finds what the operator would have cut and
cuts it at least as well. So the bench replays real, already-handled videos
through every arm, each as of the moment it was archived, and scores the
picks two ways:

- Against the operator's own exported cuts (``evaluate``, files only). An
  honest comparison needs guards: every arm scored on the same videos; errors
  and empty answers counted as misses; per-video hit rates shown beside a
  chance baseline (on a two-minute source a random window often "hits");
  cuts the operator started from a legacy suggestion split out, because they
  favor the transcript arm; and the archive-time legacy runs kept in their
  own "anchored" table, because the operator saw those before cutting.
- Blind, by the operator (``prepare_pairs`` -> ``next_item``/``judge`` ->
  ``reveal``). The primary test puts the engine's top pick against the
  transcript suggester's, with sides assigned by hash, so neither clip is
  the operator's own. A secondary "ceiling" test pits the engine against the
  operator's best cut, and never-cut videos are judged on their own.
  Results stay sealed until ``reveal``, so hit rates can't steer the
  judging, and answers lock once they're out.

Everything a run decides is frozen to files under ``bench_dir()/<name>``:
which videos, which cuts count as ground truth, which taste examples each
video's critic saw, and a fingerprint of the prompts, settings, playbook,
brand rules and code. A resume that would mix versions refuses unless asked
to start a new version under new arm labels. ``touched.json`` remembers every
video a run or a hand try has looked at, so the test seed draws a holdout
nobody has tuned on.

The database is only ever READ, through writer sessions that are rolled back
(``clip_context.build`` and ``taste`` need SAVEPOINTs, which the autocommit
read-only engine silently breaks on Postgres). Nothing here holds a session
across a model call, a download or an ffmpeg run.
"""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import errno
import hashlib
import importlib
import json
import logging
import math
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from . import ROLES, SETUPS, bench_dir, engine_dir, min_free_gb, model_for, opt, util
from . import brand as engine_brand

_log = logging.getLogger("clip_engine.bench")

# Bump when the meaning of a stored result changes (part of the fingerprint).
BENCH_VERSION = 1

HIT_IOU = 0.5                 # a proposal "is" the operator's cut at this IoU
MOMENT_OVERLAP = 0.5          # ...or found the same moment: overlap ≥ half the shorter clip
OPENING_TOLERANCE_S = 1.5     # first played segment starts this close to the operator's
CLUSTER_IOU = 0.5             # re-cuts of one moment are one target
AUTO_TIE_IOU = 0.9            # two sides this alike are scored a tie, not shown
SLIVER_S = 0.3                # hand-made slivers shorter than this are dropped (ffmpeg can't render them)
CEILING_BAR = 0.40            # engine vs your own cut: win-or-tie share that would justify switching
Z95 = 1.959963984540054       # two-sided 95%
Z90 = 1.6448536269514722      # two-sided 90% = one-sided 95%
CHANCE_STEPS = 400

ORDER_NAMES = ("editor", "cold", "rank", "final")
PAIR_CHOICES = ("a", "b", "tie", "neither")
SOLO_CHOICES = ("yes", "no")
REASONS = ("weak_opening", "boring", "off_topic", "bad_boundaries", "no_sense")

LENGTH_BUCKETS = ("<120s", "120-300s", "300-1200s", ">1200s")
LONG_BUCKET = ">1200s"
SOURCE_KINDS = ("channel", "pasted", "upload")

RUBRIC_NOTE = ("the rubric and default playbook were written from these same posts; "
               "confirm on videos archived after 2026-11-17")
LEDGER_LABEL = "shown to you (anchored); not comparable to replay arms"
DEV_BANNER = "DEV RUN — not the benchmark"

# ---- arm specs ---------------------------------------------------------------
# Bare override keys an arm may carry, per settings block (§2.1). Anything else
# must be written as a dotted key, and dotted keys are checked against the same
# lists (``_check_path``), so a typo fails loudly instead of silently running
# the default under a label that claims otherwise.

_TOP_KEYS = frozenset({
    "drafts", "section_drafts", "keep", "compare_top", "revise_top", "min_seconds",
    "max_seconds", "max_segments", "taste_hits", "taste_flops", "taste_lookback_days",
    "max_usd_per_video", "hard_cap_usd", "proxy_height", "perception_window_seconds",
    "editor_video_max_seconds", "job_timeout_seconds",
})
_SNAP_KEYS = frozenset({"word_window", "picture_window", "flash_guard", "start_pad",
                        "end_pad", "min_shot_score"})
_PROVIDER_KEYS = {
    "gemini": frozenset({
        "model", "perceiver_model", "editor_model", "critic_model", "judge_model",
        "media_resolution", "critic_media_resolution", "perception_fps", "long_video_fps",
        "critic_fps", "opening_fps", "perceiver_thinking", "editor_thinking",
        "critic_thinking", "judge_thinking", "inline_request_mb",
    }),
    "claude": frozenset({
        "model", "perceiver_model", "editor_model", "critic_model", "judge_model",
        "perceiver_effort", "editor_effort", "critic_effort", "judge_effort",
        "frame_interval", "max_total_frames", "frame_width", "critic_frame_interval",
        "max_frames_per_call", "editor_keyframes",
    }),
}
_ROLE_PREFIXES = ("perceiver", "editor", "critic", "judge")
# Dotted keys starting with one of these live under clip_engine.
_ENGINE_BLOCKS = ("gemini", "claude", "snap", "bench")

# Keys no single arm can change, and why. The bench freezes taste examples
# per video when it creates the run (an arm can only drop some of them,
# through taste_hits/taste_flops), and the Gemini client reads its upload
# limit from settings.yaml itself.
_NOT_PER_ARM = {
    "clip_engine.taste_lookback_days": "taste examples are frozen per video when the run is "
                                       "created; an arm can only show fewer of them "
                                       "(taste_hits, taste_flops)",
    "clip_engine.gemini.inline_request_mb": "the Gemini client reads it from settings.yaml, "
                                            "not per arm",
}
# What the perceiver of each provider reads to log a video. Shot logs are
# cached, and an arm that changed one of these would silently reuse a log made
# without it (one its baseline arm, a live run or a try left) unless the cache
# key covers it. So each is refused unless ``shotlog.cache_path`` is seen to
# file logs apart by it (``_cached_apart``); a different perceiver_model is
# always part of the key and is fine.
_PERCEPTION_PATHS = {
    "gemini": frozenset({"clip_engine.gemini.perception_fps", "clip_engine.gemini.long_video_fps",
                         "clip_engine.gemini.media_resolution",
                         "clip_engine.gemini.perceiver_thinking",
                         "clip_engine.perception_window_seconds"}),
    "claude": frozenset({"clip_engine.claude.frame_interval", "clip_engine.claude.max_total_frames",
                         "clip_engine.claude.frame_width", "clip_engine.claude.max_frames_per_call",
                         "clip_engine.claude.perceiver_effort"}),
}
# Top-level keys only a Gemini model in one role reads; elsewhere they'd
# change nothing.
_GEMINI_ROLE_KEYS = {"clip_engine.perception_window_seconds": "perceiver",
                     "clip_engine.editor_video_max_seconds": "editor"}
# clips.* keys an arm may change. Both kinds of arm get the two that shape the
# replay context (the bench applies them per arm, and can only take away);
# the transcript arm also reads what the transcript suggester reads per pass.
_CONTEXT_KEYS = frozenset({"learn_from_history", "context_examples"})
_TRANSCRIPT_CLIPS_KEYS = _CONTEXT_KEYS | {
    "model", "max_clips", "max_segments", "min_seconds", "max_seconds", "opening_hold_seconds",
    "opening_vision", "opening_model", "opening_lead_seconds", "opening_frame_interval",
}
# Overrides whose values the bench itself reads, so they must be well typed.
_COUNT_PATHS = ("clip_engine.taste_hits", "clip_engine.taste_flops", "clips.context_examples")

# Settings that change how a run is scheduled or budgeted, not what it
# produces; left out of the fingerprint so tuning them never blocks a resume.
_OPERATIONAL_KEYS = ("setup", "bench", "workers", "daily_usd", "estimate_usd_base",
                     "estimate_usd_per_minute", "max_source_seconds", "min_free_gb",
                     "job_timeout_seconds")

# Source files whose behavior the results depend on.
_CODE_FILES = (
    "clip_engine/__init__.py", "clip_engine/util.py", "clip_engine/engine.py",
    "clip_engine/editor.py", "clip_engine/critic.py", "clip_engine/shotlog.py",
    "clip_engine/snap.py", "clip_engine/media.py", "clip_engine/taste.py",
    "clip_engine/playbook.py", "clip_engine/gemini.py", "clip_engine/claude.py",
    "clip_engine/legacy.py", "clip_engine/jobs.py", "clip_proposals.py",
    "clip_context.py", "llm.py", "subtitles.py",
)
_BRAND_PROMPT_FIELDS = ("name", "topic", "mission", "audience", "voice_notes", "source_kind",
                        "relevance_rules", "false_positives", "strong_openings",
                        "weak_openings", "clip_guidance")

# Gemini Flash is recorded at its 2027 list price on purpose (the spend guard
# errs high); Google bills half that until the end of 2026.
_INTRO_PRICING = {"gemini-3.8-flash": ("2026-12-31", 0.5),
                  "gemini-3.6-flash": ("2026-12-31", 0.5)}

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class BenchError(ValueError):
    """A bench operation that can't go ahead as asked: a bad arm spec, an
    unknown run, a version mismatch on resume, judging after the reveal. The
    message says what to do instead."""


class _StopRun(Exception):
    """Stop the whole run (not just one video): the disk is too full."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


# ---- small helpers -------------------------------------------------------------

def _now() -> str:
    return util.utcnow_iso()


def _iso(value) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return (value.astimezone(dt.timezone.utc).replace(microsecond=0)
                .isoformat().replace("+00:00", "Z"))
    return str(value)


def _parse_time(value) -> dt.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        moment = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _json_default(value):
    if isinstance(value, (dt.datetime, dt.date)):
        return _iso(value) if isinstance(value, dt.datetime) else value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _jsonable(value):
    """A plain-JSON deep copy (datetimes as ISO strings, paths as str)."""
    return json.loads(json.dumps(value, default=_json_default))


def _json_sha(value, n: int = 12) -> str:
    text = json.dumps(value, sort_keys=True, default=_json_default)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [v for v in value if v not in (None, "")]


def _is_file(path) -> bool:
    try:
        return bool(path) and Path(path).is_file()
    except OSError:
        return False


def _safe_file(video_id) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(video_id)) or "_"


def _chunks(seq: list, n: int = 500) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _mean(values: list) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


@contextlib.contextmanager
def _db():
    """A WRITER session that is always rolled back.

    Never ``session_scope(read_only=True)``: that is the AUTOCOMMIT engine, on
    which Postgres refuses the SAVEPOINTs ``clip_context.build`` and ``taste``
    rely on — the context would silently come back empty and the bench would
    compare arms on a system that never runs in production. Rolling back also
    guarantees a replay can't write a production row.
    """
    from ..db import session_scope

    with session_scope() as session:
        try:
            yield session
        finally:
            session.rollback()


@contextlib.contextmanager
def _metered():
    """``spend.metered()`` when the spend module has it, else a null meter."""
    from .. import spend

    factory = getattr(spend, "metered", None)
    if factory is None:
        yield SimpleNamespace(total=0.0, calls=0)
        return
    with factory() as meter:
        yield meter


@contextlib.contextmanager
def _bench_bucket():
    """Route this process's un-bucketed spend (the transcript arm's LLM calls)
    to the "bench" bucket so it stays out of the dashboard's daily budget, and
    put back whatever was set before (the CLI sets "bench" for its whole
    process; nothing else should be left holding it)."""
    from .. import spend

    setter = getattr(spend, "set_bucket", None)
    if setter is None:
        _log.warning("spend.set_bucket is unavailable; bench spend lands in the default bucket")
        yield
        return
    getter = getattr(spend, "current_bucket", None)
    try:
        previous = getter() if getter is not None else None
        setter("bench")
    except Exception as exc:  # pragma: no cover - spend accounting must never stop a run
        _log.warning("Could not set the spend bucket: %s", exc)
        yield
        return
    try:
        yield
    finally:
        try:
            setter(previous)
        except Exception as exc:  # pragma: no cover
            _log.warning("Could not restore the spend bucket: %s", exc)


@contextlib.contextmanager
def _file_lock(path: Path):
    """Cross-process lock for small read-modify-write files (touched.json,
    a run's manifest)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def _run_lock(name: str):
    """Held for the whole of ``run``: a second process (or thread) running the
    same run would redo every arm on the same videos, overwrite the first's
    results, and check the cap against only its own spend. It refuses at once
    instead. The lock file sits beside the run's directory, not in it, so a
    run that fails to be created leaves nothing behind."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield
        return
    path = bench_dir() / f".{name}.run.lock"
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                raise
            raise BenchError(f"run {name} is already running in another process; let it "
                             "finish (or stop it) before running it again") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ---- interval math (mirrors clip_proposals.normalize/iou; kept local so the
# file-only evaluation never imports the LLM stack) --------------------------------

def _segments(raw) -> list[dict]:
    """Coerce segments (a list or its JSON text) to ``[{start, end}]`` in PLAY
    order, dropping invalid or empty ones. Extra keys (``lead``) are dropped."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except (ValueError, TypeError):
            return []
    out = []
    for seg in raw if isinstance(raw, list) else []:
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            out.append({"start": round(start, 2), "end": round(end, 2)})
    return out


def _drop_slivers(segments: list[dict], min_len: float = SLIVER_S) -> list[dict]:
    """Play order kept (deliberate reuse too); only too-short pieces go."""
    return [s for s in _segments(segments) if s["end"] - s["start"] >= min_len]


def _norm(segments: list[dict]) -> list[dict]:
    spans = sorted(_segments(segments), key=lambda s: s["start"])
    merged: list[dict] = []
    for s in spans:
        if merged and s["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(dict(s))
    return merged


def _dur(segments: list[dict]) -> float:
    return sum(s["end"] - s["start"] for s in _norm(segments))


def _overlap_norm(a: list[dict], b: list[dict]) -> float:
    total = 0.0
    for x in a:
        for y in b:
            total += max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    return total


def iou(a: list[dict], b: list[dict]) -> float:
    """Temporal IoU of two clips on the source timeline (order-blind)."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    inter = _overlap_norm(na, nb)
    union = _dur(na) + _dur(nb) - inter
    return inter / union if union > 0 else 0.0


def overlap_share(a: list[dict], b: list[dict]) -> float:
    """Overlap as a share of the SHORTER clip: "found the same moment", even
    when one cut is much tighter than the other."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    shorter = min(_dur(na), _dur(nb))
    return _overlap_norm(na, nb) / shorter if shorter > 0 else 0.0


def chance_hit(length: float, targets: list[list[dict]], duration: float | None,
               *, steps: int = CHANCE_STEPS) -> float:
    """Probability that ONE random window of ``length`` seconds, placed
    uniformly in the source, scores an IoU hit on any target. On a short
    source a random window often clears 0.5, so hit rates mean little without
    this beside them."""
    norm_targets = [(_norm(t), _dur(t)) for t in targets if _norm(t)]
    if length <= 0 or not norm_targets:
        return 0.0
    if not duration or duration <= 0:
        duration = max(t[-1]["end"] for t, _ in norm_targets)
    length = min(length, duration)

    def window_hits(start: float) -> bool:
        end = start + length
        for spans, tdur in norm_targets:
            inter = sum(max(0.0, min(end, s["end"]) - max(start, s["start"])) for s in spans)
            union = length + tdur - inter
            if union > 0 and inter / union >= HIT_IOU:
                return True
        return False

    room = duration - length
    if room <= 1e-9:
        return 1.0 if window_hits(0.0) else 0.0
    hits = sum(1 for k in range(steps) if window_hits(room * (k + 0.5) / steps))
    return hits / steps


# ---- arms ----------------------------------------------------------------------

def _scalar(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            value = cast(text)
        except ValueError:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            break
        return value
    return text


def _override_path(setup: str, key: str) -> str:
    if "." in key:
        head = key.split(".", 1)[0]
        return f"clip_engine.{key}" if head in _ENGINE_BLOCKS else key
    if setup == "transcript":
        raise BenchError(f"{key!r}: the transcript arm takes dotted keys only "
                         "(e.g. transcript+clips.model=claude-sonnet-5)")
    if key in _TOP_KEYS:
        return f"clip_engine.{key}"
    if key in _SNAP_KEYS:
        return f"clip_engine.snap.{key}"
    roles = ROLES[setup]
    prefix = key.split("_", 1)[0]
    if prefix in _ROLE_PREFIXES:
        provider = roles["critic" if prefix == "judge" else prefix]
        if key in _PROVIDER_KEYS[provider]:
            return f"clip_engine.{provider}.{key}"
        raise BenchError(f"{key!r} doesn't apply to {provider}, which plays the {prefix} "
                         f"in the {setup} setup")
    owners = sorted(p for p in set(roles.values()) if key in _PROVIDER_KEYS[p])
    if len(owners) == 1:
        return f"clip_engine.{owners[0]}.{key}"
    if owners:
        raise BenchError(f"{key!r} is ambiguous in the {setup} setup; write "
                         f"{owners[0]}.{key} or {owners[1]}.{key}")
    raise BenchError(f"unknown setting {key!r} for the {setup} setup; use a dotted key "
                     f"such as clip_engine.gemini.{key}")


def _check_path(setup: str, key: str, path: str) -> None:
    """Refuse an override this arm wouldn't actually run with: a typo, a
    setting the setup never reads, or one the bench can't apply per arm.
    Otherwise the label would claim a configuration that never ran."""
    parts = path.split(".")
    if parts[0] == "clips":
        allowed = _TRANSCRIPT_CLIPS_KEYS if setup == "transcript" else _CONTEXT_KEYS
        if len(parts) != 2 or parts[1] not in allowed:
            raise BenchError(f"{key!r} isn't a clips setting the {setup} arm reads (it reads "
                             f"clips.{', clips.'.join(sorted(allowed))})")
        return
    if parts[0] != "clip_engine":
        raise BenchError(f"{key!r}: an arm can change clip_engine.* and clips.* settings only")
    if setup == "transcript":
        raise BenchError(f"{key!r}: the transcript arm reads clips.* settings, not clip_engine.*")
    block = parts[1] if len(parts) == 3 else ""
    if block == "bench":
        raise BenchError(f"{key!r}: clip_engine.bench.* configures the run, not an arm")
    if not ((len(parts) == 2 and parts[1] in _TOP_KEYS)
            or (block == "snap" and parts[2] in _SNAP_KEYS)
            or (block in _PROVIDER_KEYS and parts[2] in _PROVIDER_KEYS[block])):
        raise BenchError(f"unknown setting {key!r} (as {path}) for the {setup} setup")
    if path in _NOT_PER_ARM:
        raise BenchError(f"{key!r} can't change per arm: {_NOT_PER_ARM[path]}")
    if block in _PROVIDER_KEYS and block not in ROLES[setup].values():
        raise BenchError(f"{key!r} doesn't apply: no {block} model plays a role in the "
                         f"{setup} setup")
    role = _GEMINI_ROLE_KEYS.get(path)
    if role and ROLES[setup][role] != "gemini":
        raise BenchError(f"{key!r} doesn't apply: only a Gemini {role} reads it, and "
                         f"{ROLES[setup][role]} is the {role} in the {setup} setup")
    perceiver = ROLES[setup]["perceiver"]
    if path in _PERCEPTION_PATHS[perceiver] and not _cached_apart(perceiver, path):
        raise BenchError(
            f"{key!r} shapes the {perceiver} shot log, but shot logs are not cached per "
            f"{parts[-1]}, so this arm would reuse a log made without it. It can't vary per "
            "arm until the shot-log cache key covers it (a different perceiver_model is fine).")


def _cached_apart(perceiver: str, path: str) -> bool:
    """Whether ``shotlog.cache_path`` files shot logs made with different
    values of ``path`` apart, for a short or a long source. False when it
    can't tell (an older ``cache_path`` takes no settings): refusing the
    arm is the safe answer."""
    from . import shotlog

    def where(value, seconds):
        return shotlog.cache_path("probe", perceiver, "model", seconds=seconds,
                                  settings=settings_for({"overrides": {path: value}}, {}))

    pairs = ((1, 2), ("LOW", "HIGH"), ("low", "high"),
             ("MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM"))
    try:
        return any(where(a, seconds) != where(b, seconds)
                   for a, b in pairs for seconds in (None, 3600.0))
    except Exception:
        return False


def _check_value(key: str, path: str, value) -> None:
    """Values the bench reads itself (taste counts, the replay context, the
    per-arm deadline) must be well typed: a bad one would otherwise fall
    back to the default under a label that claims otherwise."""
    if path in _COUNT_PATHS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchError(f"{key} must be a whole number of at least 0, not {value!r}")
    elif path == "clips.learn_from_history":
        if not isinstance(value, bool):
            raise BenchError(f"{key} must be true or false, not {value!r}")
    elif path == "clip_engine.job_timeout_seconds":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise BenchError(f"{key} must be a number of seconds above 0, not {value!r}")


def parse_arm(spec: str) -> dict:
    """``"gemini+editor_thinking=HIGH+critic_media_resolution=MEDIA_RESOLUTION_MEDIUM"``
    -> ``{"label": spec, "setup": "gemini", "overrides": {"clip_engine.gemini.editor_thinking":
    "HIGH", "clip_engine.gemini.critic_media_resolution": "MEDIA_RESOLUTION_MEDIUM"}}``.

    A bare key with a role prefix (``editor_``, ``critic_``, ``perceiver_``,
    ``judge_``) goes under the provider playing that role in this setup
    (the judge is the critic's provider); other bare keys go to the one
    settings block that has them. Dotted keys are used as-is, except that
    ``gemini.``/``claude.``/``snap.`` are short for ``clip_engine.…``.
    Values are typed (ints, floats, true/false); everything else is a string.

    Every key is checked (``_check_path``): typos, settings the setup never
    reads, perception settings the shot-log cache isn't keyed by, and keys
    the bench fixes per run raise ``BenchError``, so a label never claims a
    configuration that didn't run. ``run`` also refuses asking for more taste
    examples or context than the run holds (an arm can only take away).
    """
    text = str(spec or "").strip()
    if not text:
        raise BenchError("empty arm spec")
    if "/" in text or "\\" in text or text.startswith(".") or "@" in text:
        raise BenchError(f"arm {text!r} can't contain '/', '\\' or '@' or start with '.'")
    parts = [p.strip() for p in text.split("+")]
    setup = parts[0].lower()
    if setup not in SETUPS:
        raise BenchError(f"unknown setup {parts[0]!r} in arm {text!r} "
                         f"(one of {', '.join(SETUPS)})")
    overrides: dict[str, Any] = {}
    for raw in parts[1:]:
        key, sep, value = raw.partition("=")
        key = key.strip()
        if not sep or not key:
            raise BenchError(f"override {raw!r} in arm {text!r} must look like key=value")
        path = _override_path(setup, key)
        if path in overrides:
            raise BenchError(f"{key} is set twice in arm {text!r}")
        _check_path(setup, key, path)
        overrides[path] = _scalar(value.strip())
        _check_value(key, path, overrides[path])
    return {"label": "+".join([setup] + parts[1:]), "setup": setup, "overrides": overrides}


def settings_for(arm: dict, settings):
    """A deep copy of ``settings`` with the arm's overrides applied (the shared
    settings object is never mutated)."""
    from ..config import Settings

    raw = copy.deepcopy(getattr(settings, "raw", settings) or {})
    if not isinstance(raw, dict):
        raise BenchError("settings have no mapping to copy")
    for path, value in (arm.get("overrides") or {}).items():
        parts = path.split(".")
        node = raw
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                raise BenchError(f"can't set {path}: {part} is not a mapping in settings.yaml")
            node = child
        node[parts[-1]] = value
    return Settings(raw=raw)


def _display_label(label: str, setup: str) -> str:
    """The transcript arm runs today's prompt (v3, single-take wording, Opus
    opening pass) — not the suggester behind the historical figures — so it's
    labeled with its prompt version."""
    if setup != "transcript" or not label.startswith("transcript"):
        return label
    try:
        from .. import clip_proposals

        version = str(getattr(clip_proposals, "PROMPT_VERSION", "") or "")
    except Exception:  # pragma: no cover - display only
        version = ""
    return label.replace("transcript", f"transcript-{version}", 1) if version else label


# ---- fingerprint -----------------------------------------------------------------

def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def _git_sha(root: Path) -> str:
    """HEAD's commit, for provenance (read-only git; never modifies state)."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _role_models(setup: str, settings) -> dict:
    roles = ROLES[setup]
    return {role: model_for(settings, roles["critic" if role == "judge" else role], role)
            for role in ("perceiver", "editor", "critic", "judge")}


def _playbook_fingerprint(playbook: dict) -> str:
    try:
        from . import playbook as playbook_mod

        value = playbook_mod.fingerprint(playbook)
        if value:
            return str(value)
    except Exception:
        pass
    return _json_sha(playbook)


def fingerprint(settings, playbook) -> dict:
    """What determines a result, hashed per component, plus ``hash`` over all
    of them. ``git`` is recorded for provenance but left out of ``hash``: a
    commit that touches nothing below is no reason to refuse a resume, and an
    uncommitted edit that does is caught by the code hash anyway."""
    app_dir = Path(__file__).resolve().parent.parent
    engine_block = copy.deepcopy(settings.get("clip_engine", {}) or {})
    if isinstance(engine_block, dict):
        for key in _OPERATIONAL_KEYS:
            engine_block.pop(key, None)
    try:
        from ..config import load_brand, load_brand_rules

        brand_fields = load_brand() or {}
        brand_view = {"fields": {k: brand_fields.get(k, "") for k in _BRAND_PROMPT_FIELDS},
                      "rules": load_brand_rules()}
    except Exception as exc:  # pragma: no cover - brand.yaml unreadable
        brand_view = {"error": str(exc)}
    versions: dict[str, Any] = {"bench": BENCH_VERSION}
    for module in ("shotlog", "media"):
        try:
            versions[module] = getattr(importlib.import_module(f".{module}", __package__),
                                       "VERSION", None)
        except Exception:
            versions[module] = None
    try:
        from .. import clip_proposals

        versions["prompt"] = getattr(clip_proposals, "PROMPT_VERSION", None)
        versions["engine_prompt"] = getattr(clip_proposals, "ENGINE_PROMPT_VERSION", None)
    except Exception:
        versions["prompt"] = versions["engine_prompt"] = None
    models = {setup: _role_models(setup, settings) for setup in ROLES}
    models["transcript"] = {
        "model": opt(settings, "clips.model", opt(settings, "matching.model", "claude-haiku-4-5")),
        "opening_model": opt(settings, "clips.opening_model",
                             opt(settings, "vision.model", "claude-haiku-4-5")),
    }
    fp = {
        "git": _git_sha(app_dir.parent),
        "prompts": _file_sha(app_dir / "clip_engine" / "prompts.py"),
        "code": _json_sha([_file_sha(app_dir / rel) for rel in _CODE_FILES]),
        "playbook": _playbook_fingerprint(playbook),
        "settings": _json_sha(engine_block),
        "clips_settings": _json_sha(settings.get("clips", {}) or {}),
        "brand": _json_sha(brand_view),
        "models": models,
        "versions": versions,
    }
    fp["hash"] = _json_sha({k: v for k, v in fp.items() if k != "git"}, n=16)
    return fp


def _fingerprint_diff(old: dict, new: dict) -> list[str]:
    keys = sorted((set(old) | set(new)) - {"hash", "git"})
    return [k for k in keys if old.get(k) != new.get(k)]


# ---- touched videos -------------------------------------------------------------

_touch_lock = threading.Lock()


def _touched_path() -> Path:
    return bench_dir() / "touched.json"


def _touched_map(data) -> dict[str, list]:
    if isinstance(data, dict) and "videos" in data:
        data = data["videos"]
    if isinstance(data, dict):
        return {str(k): (v if isinstance(v, list) else [v]) for k, v in data.items() if k}
    if isinstance(data, list):
        out: dict[str, list] = {}
        for item in data:
            vid = item.get("video_id") if isinstance(item, dict) else item
            if vid:
                out.setdefault(str(vid), []).append(item if isinstance(item, dict) else {})
        return out
    return {}


def touched_video_ids() -> set[str]:
    """Every video a bench run or a ``clip-engine try`` has looked at: the test
    seed never draws these, so the verdict is measured on videos nobody tuned on."""
    return set(_touched_map(util.read_json(_touched_path(), None)))


def touch_videos(video_ids: Iterable, *, source: str, seed: str = "") -> None:
    """Record videos as touched (bench runs, hand-picked runs, ``clip-engine
    try``). Never raises."""
    ids = [str(v) for v in video_ids if v]
    if not ids:
        return
    path = _touched_path()
    try:
        with _touch_lock, _file_lock(path.with_name(path.name + ".lock")):
            videos = _touched_map(util.read_json(path, None))
            stamp = {"by": source, "seed": seed, "at": _now()}
            for vid in ids:
                videos.setdefault(vid, []).append(stamp)
            util.atomic_write_json(path, {"version": 1, "videos": videos})
    except Exception as exc:  # pragma: no cover - bookkeeping must not stop a run
        _log.warning("Could not update %s: %s", path, exc)


# ---- selection ------------------------------------------------------------------

def _split_ids(ids) -> tuple[list[int], list[str]]:
    pks, vids = [], []
    for raw in _as_list(ids):
        text = str(raw).strip()
        if isinstance(raw, int) or text.isdigit():
            pks.append(int(text))
        elif text:
            vids.append(text)
    return pks, vids


def _source_kind(url: str, channel_url: str) -> str:
    if (url or "").startswith("upload://"):
        return "upload"
    if (channel_url or "").startswith("youtube://pasted"):
        return "pasted"
    return "channel"


def _length_bucket(seconds) -> str:
    if seconds is None:
        return "unknown"
    s = float(seconds)
    if s < 120:
        return "<120s"
    if s < 300:
        return "120-300s"
    if s <= 1200:
        return "300-1200s"
    return ">1200s"


def _transcript_end(path: str) -> float | None:
    data = util.read_json(Path(path), None) if path else None
    ends = []
    for entry in data if isinstance(data, list) else []:
        try:
            ends.append(float(entry["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(ends) if ends else None


def _candidates(session, *, ids=None, status: str | None = None,
                without_cuts: bool = False) -> list[dict]:
    """Candidate rows copied to plain dicts (the session is rolled back after)."""
    from sqlalchemy import exists, or_, select
    from sqlalchemy.orm import defer, selectinload

    from ..categories import is_first_party
    from ..models import Candidate, Cut

    query = select(Candidate).options(selectinload(Candidate.channel),
                                      defer(Candidate.transcript_text),
                                      defer(Candidate.description))
    if status:
        query = query.where(Candidate.status == status)
    if ids is not None:
        pks, vids = _split_ids(ids)
        conditions = []
        if pks:
            conditions.append(Candidate.id.in_(pks))
        if vids:
            conditions.append(Candidate.video_id.in_(vids))
        if not conditions:
            return []
        query = query.where(or_(*conditions))
    if without_cuts:
        query = query.where(~exists().where(Cut.candidate_pk == Candidate.id))
    out = []
    for c in session.execute(query.order_by(Candidate.id)).scalars().all():
        channel = c.channel
        out.append({
            "candidate_pk": c.id, "video_id": c.video_id, "title": c.title or "",
            "url": c.url or "", "status": c.status or "",
            "first_party": bool(is_first_party(c)),
            "channel": getattr(channel, "call_sign", "") or "",
            "channel_url": getattr(channel, "url", "") or "",
            "duration_seconds": c.duration_seconds,
            "archived_at": _iso(c.archived_at), "approved_at": _iso(c.approved_at),
            "created_at": _iso(c.created_at),
            "local_video_path": c.local_video_path or "",
            "source_object_key": c.source_object_key or "",
            "transcript_path": c.transcript_path or "",
            "word_transcript_path": c.word_transcript_path or "",
        })
    return out


def _ineligible(c: dict, *, require_words: bool) -> str:
    """'' when the candidate can be replayed, else why not."""
    if c["status"] != "archived":
        return "not archived"
    if c["first_party"]:
        return "first-party footage"
    if not _is_file(c["transcript_path"]):
        return "no transcript file on this machine"
    if require_words and not _is_file(c["word_transcript_path"]):
        return "no word-stream file (run retranscribe --missing-words, or allow no words)"
    if not (_is_file(c["local_video_path"]) or c["source_object_key"]):
        return "source video is neither on disk nor in storage"
    return ""


def _cluster(cuts: list[dict]) -> list[dict]:
    """Re-cuts of one moment become one target. Cuts are visited best-first
    (published, then most viewed, then oldest) and join the first target whose
    representative they overlap with IoU ≥ 0.5, so the representative is the
    published, most-viewed cut and distinct moments never chain together."""
    order = sorted(cuts, key=lambda c: (not c["published"],
                                        -(c["views"] if c["views"] is not None else -1),
                                        c["created_at"] or "", c["cut_pk"]))
    clusters: list[dict] = []
    for cut in order:
        for cluster in clusters:
            if iou(cluster["rep"]["segments"], cut["segments"]) >= CLUSTER_IOU:
                cluster["members"].append(cut)
                break
        else:
            clusters.append({"rep": cut, "members": [cut]})
    targets = []
    for cluster in clusters:
        rep = cluster["rep"]
        targets.append({
            "cut_pk": rep["cut_pk"], "segments": rep["segments"],
            "published": rep["published"], "views": rep["views"],
            # A moment counts as anchored to the transcript suggester when any
            # cut of it started from an accepted suggestion.
            "seeded": any(m["seeded"] for m in cluster["members"]),
            "title": rep["title"], "members": [m["cut_pk"] for m in cluster["members"]],
        })
    targets.sort(key=lambda t: (t["segments"][0]["start"], t["cut_pk"]))
    return targets


def _exported_targets(session, candidate_pks: list[int], *,
                      before: dt.datetime | None = None) -> dict[int, list[dict]]:
    """``candidate_pk -> targets``: the operator's EXPORTED cuts (an exported
    file or a published post), clustered. Unexported cuts are left out: one may
    be an abandoned legacy proposal loaded into the editor, which would score
    the transcript suggester against its own output."""
    from sqlalchemy import select

    from .. import analytics
    from ..models import ClipProposal, Cut, ThreadsPost

    before = before or dt.datetime.now(dt.timezone.utc)
    rows = []
    for chunk in _chunks(list(candidate_pks)):
        rows += session.execute(
            select(Cut.id, Cut.candidate_pk, Cut.trim_segments, Cut.trimmed_clip_path,
                   Cut.created_at, Cut.clip_title)
            .where(Cut.candidate_pk.in_(chunk))
        ).all()
    rows = [r for r in rows if r.created_at is None or _parse_time(r.created_at) < before]
    cut_ids = [r.id for r in rows]
    posts = []
    seeded: set[int] = set()
    for chunk in _chunks(cut_ids):
        posts += session.execute(
            select(ThreadsPost.id, ThreadsPost.cut_pk)
            .where(ThreadsPost.cut_pk.in_(chunk), ThreadsPost.status == "published")
        ).all()
        seeded |= set(session.execute(
            select(ClipProposal.cut_pk)
            .where(ClipProposal.cut_pk.in_(chunk),
                   ClipProposal.verdict == ClipProposal.VERDICT_ACCEPTED)
        ).scalars().all())
    latest = analytics.latest_metrics_bulk(session, [p.id for p in posts]) if posts else {}
    published: set[int] = set()
    views: dict[int, int] = {}
    for p in posts:
        published.add(p.cut_pk)
        value = (latest.get(p.id) or {}).get("views")
        if value is not None:
            views[p.cut_pk] = max(views.get(p.cut_pk, value), value)
    by_candidate: dict[int, list[dict]] = {}
    for r in rows:
        if not ((r.trimmed_clip_path or "").strip() or r.id in published):
            continue
        segments = _drop_slivers(_segments(r.trim_segments))
        if not segments:
            continue
        by_candidate.setdefault(r.candidate_pk, []).append({
            "cut_pk": r.id, "segments": segments, "published": r.id in published,
            "views": views.get(r.id), "seeded": r.id in seeded,
            "created_at": _iso(r.created_at), "title": r.clip_title or "",
        })
    return {pk: _cluster(cuts) for pk, cuts in by_candidate.items()}


def _video_entry(c: dict, targets: list[dict], *, negative: bool = False) -> dict:
    duration = c["duration_seconds"] or _transcript_end(c["transcript_path"])
    kind = _source_kind(c["url"], c["channel_url"])
    bucket = _length_bucket(duration)
    # The archive-time pass ran at archive time; older rows without the stamp
    # fall back to approval (archiving starts right after), then discovery.
    as_of, as_of_source = c["archived_at"], "archived_at"
    if not as_of:
        as_of, as_of_source = c["approved_at"], "approved_at"
    if not as_of:
        as_of, as_of_source = c["created_at"], "created_at"
    return {
        "candidate_pk": c["candidate_pk"], "video_id": c["video_id"], "title": c["title"],
        "channel": c["channel"], "kind": kind,
        "duration": round(float(duration), 2) if duration else None,
        "length_bucket": bucket, "stratum": f"{bucket}|{kind}",
        "archived_at": c["archived_at"], "as_of": as_of, "as_of_source": as_of_source,
        "has_words": _is_file(c["word_transcript_path"]),
        "local_video_path": c["local_video_path"], "source_object_key": c["source_object_key"],
        "transcript_path": c["transcript_path"], "word_transcript_path": c["word_transcript_path"],
        "targets": targets, "negative": negative,
    }


def _order_hash(seed: str, video_id: str) -> str:
    return _hex(f"{seed}{video_id}")


def _stratified(videos: list[dict], *, limit: int, seed: str) -> list[dict]:
    """Choose ``limit`` videos across strata (source length × source kind).

    Every non-empty stratum gets one seat while seats last; the rest go by
    highest averages on stratum size, with sources over 20 minutes weighted ×2
    (that's where the Gemini editor stops watching and sampling halves, and a
    proportional draw would leave only a handful). Inside a stratum the order
    is sha1(seed + video_id). The final list is shuffled the same way, so a
    run cut short by its budget is still a spread, not one stratum.
    """
    cells: dict[str, list[dict]] = {}
    for video in videos:
        cells.setdefault(video["stratum"], []).append(video)
    for members in cells.values():
        members.sort(key=lambda v: _order_hash(seed, v["video_id"]))
    weight = {k: len(v) * (2 if k.startswith(LONG_BUCKET) else 1) for k, v in cells.items()}
    keys = sorted(cells, key=lambda k: (-weight[k], k))
    alloc = {k: 0 for k in keys}
    remaining = max(0, min(int(limit), len(videos)))
    for key in keys:
        if remaining <= 0:
            break
        alloc[key] = 1
        remaining -= 1
    while remaining > 0:
        open_keys = [k for k in keys if alloc[k] < len(cells[k])]
        if not open_keys:
            break
        best = max(open_keys, key=lambda k: (weight[k] / (alloc[k] + 1), -keys.index(k)))
        alloc[best] += 1
        remaining -= 1
    chosen = [v for k in keys for v in cells[k][:alloc[k]]]
    chosen.sort(key=lambda v: _order_hash(f"{seed}|run|", v["video_id"]))
    return chosen


def select_videos(session, *, limit, seed: str = "test", require_words: bool = True,
                  candidate_ids=None, exclude_video_ids=()) -> list[dict]:
    """Past videos to replay, with their ground-truth targets (see the module
    docstring). Reads only; the caller's writer session is rolled back.

    Sampled (no ``candidate_ids``): archived, not first-party, transcript (and
    words when ``require_words``) on this machine, source on disk or in
    storage, at least one exported cut; stratified (``_stratified``);
    ``exclude_video_ids`` skipped. Hand-picked: exactly those candidates (pks
    or video ids) that qualify, in the order given; one with no exported cut
    is kept as a never-cut video (``negative``) and judged on its own.
    """
    from ..models import STATUS_ARCHIVED

    hand_picked = bool(_as_list(candidate_ids))
    excluded = {str(v) for v in (exclude_video_ids or ())}
    pool = (_candidates(session, ids=candidate_ids) if hand_picked
            else _candidates(session, status=STATUS_ARCHIVED))
    eligible = []
    for c in pool:
        if c["video_id"] in excluded and not hand_picked:
            continue
        why = _ineligible(c, require_words=require_words)
        if why:
            if hand_picked:
                _log.warning("Bench: skipping %s (%s)", c["video_id"], why)
            continue
        eligible.append(c)
    targets = _exported_targets(session, [c["candidate_pk"] for c in eligible])
    videos = []
    for c in eligible:
        found = targets.get(c["candidate_pk"], [])
        if found:
            videos.append(_video_entry(c, found))
        elif hand_picked:
            _log.info("Bench: %s has no exported cut; it will be judged on its own", c["video_id"])
            videos.append(_video_entry(c, [], negative=True))
    if hand_picked:
        pks, vids = _split_ids(candidate_ids)
        rank = {("pk", p): i for i, p in enumerate(pks)}
        rank.update({("vid", v): len(pks) + i for i, v in enumerate(vids)})
        videos.sort(key=lambda v: min(rank.get(("pk", v["candidate_pk"]), 10 ** 9),
                                      rank.get(("vid", v["video_id"]), 10 ** 9)))
        return videos
    return _stratified(videos, limit=int(limit), seed=str(seed))


def select_negatives(session, *, count: int, seed: str = "test", require_words: bool = True,
                     exclude_video_ids=()) -> list[dict]:
    """Archived videos the operator NEVER cut, to measure false positives: a
    top pick on a video with nothing worth posting is judged on its own."""
    from ..models import STATUS_ARCHIVED

    if count <= 0:
        return []
    excluded = {str(v) for v in (exclude_video_ids or ())}
    pool = [c for c in _candidates(session, status=STATUS_ARCHIVED, without_cuts=True)
            if c["video_id"] not in excluded
            and not _ineligible(c, require_words=require_words)]
    pool.sort(key=lambda c: _order_hash(f"{seed}|negative|", c["video_id"]))
    return [_video_entry(c, [], negative=True) for c in pool[:int(count)]]


def _taste_counts(settings) -> dict:
    """How many taste examples a run freezes per video, and from how far back."""
    return {"hits": int(opt(settings, "clip_engine.taste_hits", 3)),
            "flops": int(opt(settings, "clip_engine.taste_flops", 2)),
            "lookback_days": int(opt(settings, "clip_engine.taste_lookback_days", 180))}


def _is_flop(example) -> bool:
    return isinstance(example, dict) and example.get("label") == "flop"


def _late_flops(session, examples: list, as_of: dt.datetime | None) -> set:
    """``cut_pk``s of the flops among ``examples`` whose "not gripping" label
    was set (or last changed: a relabel overwrites the row) after ``as_of``,
    or that can't be dated (None stands for a flop without a cut). A replay
    must not tell the critic about a verdict the operator hadn't given yet,
    and ``taste.select_examples`` dates posts, not labels."""
    from sqlalchemy import select

    from ..models import ContentLabel

    cuts = [e.get("cut_pk") for e in examples if _is_flop(e)]
    if as_of is None or not cuts:
        return set()
    stamps = {}
    known = [pk for pk in cuts if pk is not None]
    if known:
        for cut_pk, created, updated in session.execute(
                select(ContentLabel.cut_pk, ContentLabel.created_at, ContentLabel.updated_at)
                .where(ContentLabel.label == ContentLabel.LABEL_GRIPPING,
                       ContentLabel.cut_pk.in_(known))).all():
            stamps[cut_pk] = _parse_time(updated) or _parse_time(created)
    return {pk for pk in cuts if stamps.get(pk) is None or stamps[pk] > as_of}


def _freeze_examples(session, videos: list[dict], settings, notes: list[str]) -> None:
    """Resolve each video's taste examples once, at creation, so every arm and
    every resume shows the critic the same hits and flops. Flops whose label
    came after the video's archive time are left out (see ``_late_flops``)."""
    from . import taste

    counts = _taste_counts(settings)
    failures = trimmed = 0
    for video in videos:
        as_of = _parse_time(video["as_of"])
        try:
            with session.begin_nested():
                examples = taste.select_examples(
                    session, as_of=as_of, exclude_candidate_pk=video["candidate_pk"],
                    hits=counts["hits"], flops=counts["flops"],
                    lookback_days=counts["lookback_days"], seed=video["video_id"]) or []
        except Exception as exc:
            failures += 1
            _log.warning("Taste examples for %s unavailable: %s", video["video_id"], exc)
            examples = []
        if any(_is_flop(e) for e in examples):
            try:
                with session.begin_nested():
                    late = _late_flops(session, examples, as_of)
            except Exception as exc:
                _log.warning("Couldn't date the flop labels for %s; leaving its flops out: %s",
                             video["video_id"], exc)
                late = {e.get("cut_pk") for e in examples if _is_flop(e)}
            if late:
                trimmed += 1
                examples = [e for e in examples if not (_is_flop(e) and e.get("cut_pk") in late)]
        video["taste_examples"] = _jsonable(examples)
        video["taste_post_pks"] = [e.get("post_pk") for e in examples if isinstance(e, dict)]
    if failures:
        notes.append(f"taste examples unavailable for {failures} video(s); "
                     "their critics rank without examples")
    if trimmed:
        notes.append(f"taste flops labeled 'not gripping' after the video was archived were "
                     f"left out on {trimmed} video(s): a replay only sees labels made by then")


# ---- run -----------------------------------------------------------------------

def _validate_name(name: str) -> str:
    text = str(name or "").strip()
    if not _NAME_RE.match(text):
        raise BenchError(f"run name {name!r} must be letters, digits, '.', '_' or '-' "
                         "(up to 64, starting with a letter or digit)")
    return text


def _run_dir(name: str) -> Path:
    return bench_dir() / _validate_name(name)


def run_manifest(name: str) -> dict | None:
    """Run ``name``'s manifest as it is on disk, or None when there's no such
    run (yet). Raises ``BenchError`` only for an invalid name."""
    data = util.read_json(_run_dir(name) / "manifest.json", None)
    return data if isinstance(data, dict) else None


def _load_run(name: str) -> tuple[Path, dict, list[dict]]:
    run_dir = _run_dir(name)
    manifest = util.read_json(run_dir / "manifest.json", None)
    if not isinstance(manifest, dict):
        raise BenchError(f"no bench run named {name!r} (looked in {run_dir})")
    videos = util.read_json(run_dir / "videos.json", None)
    if not isinstance(videos, list):
        raise BenchError(f"run {name!r} has no readable videos.json")
    return run_dir, manifest, videos


_manifest_thread_lock = threading.Lock()


@contextlib.contextmanager
def _manifest_lock(run_dir: Path):
    """Held around every read-modify-write of a run's manifest. A run (the
    CLI), the judging page's reveal (the dashboard) and ``prepare_pairs`` are
    separate processes: an unlocked whole-file write from one would undo
    another's, e.g. a run finishing after a reveal would reopen judging.
    Not re-entrant: never nest it."""
    with _manifest_thread_lock, _file_lock(run_dir / ".manifest.lock"):
        yield


@contextlib.contextmanager
def _locked_manifest(run_dir: Path, fallback: dict | None = None):
    """The manifest as it is on disk now, under ``_manifest_lock``; changes
    are saved only by ``_write_manifest`` inside the block. ``fallback`` (a
    copy of it) stands in when the file can't be read."""
    with _manifest_lock(run_dir):
        fresh = util.read_json(run_dir / "manifest.json", None)
        if not isinstance(fresh, dict):
            if fallback is None:
                raise BenchError(f"run {run_dir.name} has no readable manifest.json")
            fresh = copy.deepcopy(fallback)
        yield fresh


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    """Save the whole manifest. Callers hold ``_manifest_lock`` and write
    what they read inside it (``_locked_manifest``), never an older copy."""
    manifest["updated_at"] = _now()
    util.atomic_write_json(run_dir / "manifest.json", manifest)


def _result_path(run_dir: Path, label: str, video_id: str) -> Path:
    return run_dir / label / f"{_safe_file(video_id)}.json"


def _load_result(run_dir: Path, label: str, video_id: str) -> dict | None:
    data = util.read_json(_result_path(run_dir, label, video_id), None)
    return data if isinstance(data, dict) else None


def _result_cost(result: dict | None) -> float:
    if not result:
        return 0.0
    return float(result.get("cost_usd") or 0.0) + float(result.get("prior_cost_usd") or 0.0)


def _run_spent(run_dir: Path) -> float:
    """What every result in the run cost, failed attempts included."""
    total = 0.0
    for path in run_dir.glob("*/*.json"):
        if path.parent.name == "renders":
            continue
        total += _result_cost(util.read_json(path, None))
    return round(total, 6)


def _pending_arms(run_dir: Path, video: dict, labels: list[str], keep_errors: bool) -> list[str]:
    out = []
    for label in labels:
        result = _load_result(run_dir, label, video["video_id"])
        if result is None or (result.get("error") and not keep_errors):
            out.append(label)
    return out


def _arm_entry(arm: dict, label: str) -> dict:
    return {"spec": arm["label"], "setup": arm["setup"], "overrides": arm["overrides"],
            "display": _display_label(label, arm["setup"]), "added_at": _now()}


def _playbook_provenance(raw: dict, bench: dict, videos: list[dict]) -> tuple[dict, list[str]]:
    """Where the engine arms' playbook came from, and a warning when a
    generated one was built from posts made after some benched video was
    archived (its archetypes may then describe the very clips being replayed)."""
    raw = raw if isinstance(raw, dict) else {}
    generated = raw.get("generated_at")
    info = {"fingerprint": _playbook_fingerprint(bench),
            "source": "generated" if generated else "default",
            "generated_at": _iso(generated), "as_of": _iso(raw.get("as_of")),
            "excluded": raw.get("excluded"), "examples": "stripped"}
    warnings: list[str] = []
    if generated:
        starts = [t for t in (_parse_time(v.get("as_of")) for v in videos) if t is not None]
        earliest = min(starts) if starts else None
        built = _parse_time(raw.get("as_of") or generated)
        if earliest is None or built is None or built > earliest:
            warnings.append(
                "config/clip_playbook.yaml was generated from posts up to "
                f"{_iso(built) or 'an unknown date'}, after the earliest benched video was "
                f"archived ({_iso(earliest) or 'unknown'}): its archetypes may describe the "
                "clips being replayed, which flatters the engine arms. Regenerate it as of "
                "an earlier date, or move it aside to bench the default playbook.")
    return info, warnings


def _whole(value, default: int) -> int:
    """A count from settings: a whole number ≥ 0, else ``default``."""
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _check_inputs(arm: dict, settings, taste: dict) -> None:
    """Refuse an arm that asks for more than the run holds. Taste examples
    are frozen per video and the replay context is built once per video, so
    an arm can only show fewer of them (``_arm_examples``, ``_arm_context``)."""
    over = arm.get("overrides") or {}
    label = arm.get("label", "")
    for key in ("hits", "flops"):
        wanted, frozen = over.get(f"clip_engine.taste_{key}"), taste.get(key)
        if wanted is not None and frozen is not None and wanted > int(frozen):
            raise BenchError(f"arm {label!r} asks for {wanted} taste {key}, but the run froze "
                             f"{frozen} per video; an arm can only show fewer")
    if (over.get("clips.learn_from_history") is True
            and not settings.get("clips.learn_from_history", True)):
        raise BenchError(f"arm {label!r} turns clips.learn_from_history on, but the replay "
                         "context is built with it off; an arm can only drop the context")
    wanted = over.get("clips.context_examples")
    base = _whole(settings.get("clips.context_examples", 6), 6)
    if wanted is not None and wanted > base:
        raise BenchError(f"arm {label!r} asks for {wanted} shipped examples, but the replay "
                         f"context holds {base} (clips.context_examples); an arm can only "
                         "show fewer")


def _resolve_arms(arms, settings, taste: dict) -> list[dict]:
    specs = _as_list(arms) or _as_list(opt(settings, "clip_engine.bench.arms",
                                           ["gemini", "transcript"]))
    parsed, seen = [], set()
    for spec in specs:
        arm = parse_arm(spec)
        _check_inputs(arm, settings, taste)
        if arm["label"] not in seen:
            seen.add(arm["label"])
            parsed.append(arm)
    if not parsed:
        raise BenchError("no arms to run")
    return parsed


def _recheck_arm(label: str, specs: dict, settings, taste: dict) -> None:
    """An arm already in the run, checked again before it runs more videos:
    one this version of the bench would refuse must not keep adding results
    under a label that claims a configuration it doesn't run."""
    spec = (specs.get(label) or {}).get("spec") or label.split("@", 1)[0]
    try:
        _check_inputs(parse_arm(spec), settings, taste)
    except BenchError as exc:
        raise BenchError(f"arm {label} can't run as labeled: {exc}. Start a new run "
                         "without it.") from None


def _create_run(run_dir: Path, *, name, arms, limit, max_usd, seed, candidate_ids,
                require_words, negatives, settings, fp, raw_playbook, bench_playbook,
                say) -> tuple[dict, list[dict]]:
    taste = _taste_counts(settings)
    parsed = _resolve_arms(arms, settings, taste)
    hand_picked = bool(_as_list(candidate_ids))
    limit = int(limit if limit is not None else opt(settings, "clip_engine.bench.limit", 60))
    max_usd = _positive_usd(max_usd if max_usd is not None
                            else opt(settings, "clip_engine.bench.max_usd", 40))
    if limit <= 0:
        raise BenchError(f"limit must be at least 1, not {limit}")
    exclude = touched_video_ids() if (seed == "test" and not hand_picked) else set()
    notes: list[str] = []
    with _db() as session:
        videos = select_videos(session, limit=limit, seed=seed, require_words=require_words,
                               candidate_ids=candidate_ids, exclude_video_ids=exclude)
        if negatives and not hand_picked:
            videos += select_negatives(session, count=int(negatives), seed=seed,
                                       require_words=require_words,
                                       exclude_video_ids=exclude | {v["video_id"] for v in videos})
        _freeze_examples(session, videos, settings, notes)
    if not videos:
        raise BenchError(
            "no eligible videos: replay needs archived, non-first-party videos with a "
            "transcript (and words, unless allowed without) on this machine, a source "
            "on disk or in storage, and an exported cut"
            + ("" if hand_picked or seed != "test" else "; videos already touched are excluded"))
    provenance, warnings = _playbook_provenance(raw_playbook, bench_playbook, videos)
    for warning in warnings:
        _log.warning(warning)
        say(f"WARNING: {warning}")
    labels = [a["label"] for a in parsed]
    manifest = {
        "name": name, "bench_version": BENCH_VERSION, "seed": seed,
        "mode": "hand-picked" if hand_picked else "sampled",
        "arms": labels, "arm_specs": {a["label"]: _arm_entry(a, a["label"]) for a in parsed},
        "retired_arms": [], "created_at": _now(), "fingerprint": fp,
        "fingerprint_history": [], "version_tag": "",
        "limit": limit, "max_usd": max_usd, "require_words": bool(require_words),
        "negatives": sum(1 for v in videos if v.get("negative")),
        "candidate_ids": [str(c) for c in _as_list(candidate_ids)],
        "excluded_touched": len(exclude),
        "context": ("replay: account context as of each video's archive time; this "
                    "video's own dismissals and clips left out; calibration omitted"),
        "taste": taste,
        "playbook": provenance, "warnings": warnings, "notes": notes,
        "stopped": None, "revealed_at": None, "invocations": [], "restored": [],
        "skipped": {},
    }
    util.atomic_write_json(run_dir / "videos.json", videos)
    with _manifest_lock(run_dir):
        _write_manifest(run_dir, manifest)   # last: the run exists only once it's whole
    touch_videos([v["video_id"] for v in videos], source=f"bench:{name}", seed=seed)
    say(f"Created bench run {name}: {len(videos)} videos "
        f"({manifest['negatives']} never cut), arms {', '.join(labels)}, seed {seed}.")
    return manifest, videos


def _resume_run(run_dir: Path, manifest: dict, *, videos, arms, max_usd, seed, limit,
                candidate_ids, require_words, negatives, new_version, settings, fp,
                say) -> list[str]:
    """Bring an existing run's manifest up to date; returns the arm labels to
    run in this invocation. ``arms``/``max_usd`` of None keep the run's own."""
    if max_usd is not None:
        manifest["max_usd"] = _positive_usd(max_usd)   # a resume may change the cap, never remove it
    old = manifest.get("fingerprint") or {}
    specs = manifest.setdefault("arm_specs", {})
    taste = manifest.get("taste") or _taste_counts(settings)
    if old.get("hash") != fp["hash"]:
        changed = ", ".join(_fingerprint_diff(old, fp)) or "unknown"
        if not new_version:
            raise BenchError(
                f"run {manifest.get('name')} was made with a different version (changed: "
                f"{changed}). Pass --new-version to start every arm over under new labels "
                "(the cap counts what the run already spent), or start a new run.")
        tag = "@" + fp["hash"][:8]
        requested = ([a["label"] for a in _resolve_arms(arms, settings, taste)] if _as_list(arms)
                     else [specs.get(label, {}).get("spec", label.split("@", 1)[0])
                           for label in manifest.get("arms", [])])
        parsed = []
        for spec in requested:
            arm = parse_arm(spec)
            _check_inputs(arm, settings, taste)
            parsed.append(arm)
        manifest.setdefault("fingerprint_history", []).append(
            {"fingerprint": old, "until": _now(), "arms": list(manifest.get("arms", []))})
        manifest.setdefault("retired_arms", []).extend(manifest.get("arms", []))
        manifest["fingerprint"] = fp
        manifest["version_tag"] = tag
        manifest["arms"] = []
        manifest["stopped"] = None
        for arm in parsed:
            label = arm["label"] + tag
            manifest["arms"].append(label)
            specs[label] = _arm_entry(arm, label)
        say(f"New version {tag[1:]} (changed: {changed}); arms now {', '.join(manifest['arms'])}.")
        _warn_new_version(run_dir, manifest, videos, settings, say)
        return list(manifest["arms"])
    if new_version:
        say("Fingerprint unchanged; --new-version has nothing to do.")
    tag = manifest.get("version_tag", "") or ""
    if not _as_list(arms):
        to_run = list(manifest.get("arms", []))
        for label in to_run:
            _recheck_arm(label, specs, settings, taste)
    else:
        to_run = []
        for arm in _resolve_arms(arms, settings, taste):
            label = arm["label"] + tag
            if label not in manifest.setdefault("arms", []):
                manifest["arms"].append(label)
                specs[label] = _arm_entry(arm, label)
                say(f"Added arm {label} to run {manifest.get('name')}.")
            to_run.append(label)
    frozen = {"seed": seed, "limit": limit, "require_words": bool(require_words)}
    for key, value in frozen.items():
        if value is not None and key in manifest and manifest[key] != value:
            say(f"Note: run {manifest.get('name')} is frozen with {key}={manifest[key]!r}; "
                f"ignoring {value!r}.")
    if _as_list(candidate_ids) and [str(c) for c in _as_list(candidate_ids)] != manifest.get("candidate_ids"):
        say("Note: the run's videos are frozen; ignoring --candidate.")
    if negatives and int(negatives) != int(manifest.get("negatives") or 0):
        say("Note: the run's videos are frozen; ignoring --negatives.")
    return to_run


def _warn_new_version(run_dir: Path, manifest: dict, videos: list[dict], settings, say) -> None:
    """A new version starts over: every arm runs again on every frozen video
    under its new label, judgments of the old labels can't carry over (item
    ids include the label), and the cap still counts what the run spent on
    the old version."""
    spent = _run_spent(run_dir)
    cap = float(manifest.get("max_usd") or 0.0)
    estimate = sum(_estimate_video(v, list(manifest.get("arms") or []),
                                   manifest.get("arm_specs") or {}, settings) for v in videos)
    say(f"WARNING: a new version re-runs every arm on all {len(videos)} videos, and its pairs "
        "must be prepared and judged again: judgments of the old version don't carry over.")
    line = (f"WARNING: the run has already spent ${spent:.2f} of its ${cap:.2f} cap (the cap "
            f"counts every version); the new version is estimated at about ${estimate:.2f}")
    if spent >= cap:
        line += ". Nothing will run until you raise --max-usd."
    elif spent + estimate > cap:
        line += f". Raise --max-usd to about ${spent + estimate:.2f} to finish it."
    else:
        line += "."
    say(line)


def _positive_usd(value) -> float:
    """A spending cap. Zero or less is refused rather than read as "no cap"."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        raise BenchError(f"max_usd must be a number of dollars, not {value!r}") from None
    if not math.isfinite(amount) or amount <= 0:
        raise BenchError(f"max_usd must be more than $0, not {value!r}")
    return amount


class _LoopState:
    def __init__(self, spent: float) -> None:
        self.lock = threading.Lock()
        self.spent = spent
        self.reserved = 0.0
        self.new_cost = 0.0
        self.stop: dict | None = None
        self.ran = 0


def _existing_dir(path: Path) -> Path:
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _ensure_source(local: str, key: str, min_free_gb: float) -> tuple[Path, bool, tuple]:
    """``(path, restored_by_us, (inode, size))``. Raises ``_StopRun`` when the
    disk is too full to continue, ``offload.SourceUnavailable`` when this one
    video's source can't be had."""
    from .. import offload
    from . import media

    path = Path(local) if local else None
    existed = bool(path) and path.is_file() and path.stat().st_size > 0
    if not existed:
        # A bulk replay restores dozens of offloaded sources; stop before one
        # fills the disk the dashboard and its exports also live on.
        where = _existing_dir(path.parent if path else engine_dir())
        try:
            free = float(media.disk_free_gb(where))
        except Exception:
            free = None
        if free is not None and free < min_free_gb:
            raise _StopRun("low_disk", f"only {free:.1f} GB free at {where} "
                                       f"(clip_engine.min_free_gb is {min_free_gb:g})")
    source = Path(offload.ensure_local(local, key))
    stamp: tuple = ()
    if not existed:
        try:
            st = source.stat()
            stamp = (st.st_ino, st.st_size)
        except OSError:
            stamp = ()
    return source, not existed, stamp


def _release_restored(source: Path, stamp: tuple, manifest: dict, lock: threading.Lock,
                      video_id: str) -> None:
    """Delete a source this run brought back from storage (it stays in the
    bucket; the dashboard restores it on demand), unless it changed since."""
    try:
        st = source.stat()
        if stamp and (st.st_ino, st.st_size) != stamp:
            return
        source.unlink()
    except OSError as exc:
        _log.info("Left restored source %s in place: %s", source, exc)
        return
    with lock:
        manifest.setdefault("restored", []).append(
            {"video_id": video_id, "path": str(source), "removed_at": _now()})


def _thawed(example: dict) -> dict:
    """A frozen (JSON) example with its timestamp a datetime again, as
    ``taste.select_examples`` hands them out."""
    out = dict(example)
    if isinstance(out.get("published_at"), str):
        out["published_at"] = _parse_time(out["published_at"])
    return out


def _frozen_examples(video: dict, fresh) -> list[dict]:
    """The taste examples frozen for this video: fresh copies from
    ``prepare_inputs`` where they match, the frozen copy otherwise, in the
    frozen order — so every arm and every resume shows the same examples."""
    if "taste_post_pks" not in video:
        return list(fresh or [])
    wanted = [pk for pk in video.get("taste_post_pks") or [] if pk is not None]
    by_pk = {e.get("post_pk"): e for e in (fresh or []) if isinstance(e, dict)}
    frozen = {e.get("post_pk"): e for e in video.get("taste_examples") or []
              if isinstance(e, dict)}
    return [by_pk[pk] if pk in by_pk else _thawed(frozen[pk])
            for pk in wanted if pk in by_pk or pk in frozen]


def _arm_examples(examples: list, arm_settings) -> list:
    """The frozen examples one engine arm's critic sees: the first
    ``taste_hits`` hits and the first ``taste_flops`` flops (an example not
    labeled "flop" counts as a hit, as the ranker reads it). An arm can only
    show fewer than the run froze; ``_check_inputs`` refuses asking for more."""
    limits = {"hit": _whole(opt(arm_settings, "clip_engine.taste_hits", 3), 3),
              "flop": _whole(opt(arm_settings, "clip_engine.taste_flops", 2), 2)}
    seen = {"hit": 0, "flop": 0}
    out = []
    for example in examples or []:
        kind = "flop" if _is_flop(example) else "hit"
        if seen[kind] < limits[kind]:
            seen[kind] += 1
            out.append(example)
    return out


def _arm_context(context: dict, arm_settings) -> dict:
    """The replay context as ``clip_context.build`` would have built it with
    this arm's settings: none without ``clips.learn_from_history``, and at most
    ``clips.context_examples`` shipped examples (best first, as built)."""
    if not arm_settings.get("clips.learn_from_history", True):
        return {}
    out = dict(context or {})
    shipped = out.get("shipped_examples")
    if isinstance(shipped, list):
        limit = _whole(arm_settings.get("clips.context_examples", 6), len(shipped))
        out["shipped_examples"] = shipped[:limit]
    return {key: value for key, value in out.items() if value}


def _arm_timeout(arm_settings, default: float) -> float:
    """Seconds an engine arm may take on one video (``job_timeout_seconds``)."""
    try:
        seconds = float(opt(arm_settings, "clip_engine.job_timeout_seconds", default))
    except (TypeError, ValueError):
        return default
    return seconds if math.isfinite(seconds) and seconds > 0 else default


def _top_items(top) -> list[dict]:
    out = []
    for item in top or []:
        get = item.get if isinstance(item, dict) else (lambda k, d=None: getattr(item, k, d))
        segments = _segments(get("segments"))
        if not segments:
            continue
        out.append({
            "id": get("id"), "segments": segments, "story": str(get("story", "") or ""),
            "story_key": str(get("story_key", "") or ""),
            "archetype": str(get("archetype", "") or ""), "lens": str(get("lens", "") or ""),
            "hook": str(get("hook", "") or ""), "score": get("score"), "rank": get("rank"),
            "critique": _jsonable(get("critique")), "credits": str(get("credits", "") or ""),
            "origin": get("origin"),
        })
    return out


def _engine_record(result, meter) -> dict:
    get = lambda key, default=None: getattr(result, key, default)  # noqa: E731
    cost = get("cost_usd")
    return {
        "policy": str(get("policy", "") or ""), "model": str(get("model", "") or ""),
        "top": _top_items(get("top")),
        "drafts": _jsonable(list(get("drafts") or [])),
        "moments": _jsonable(list(get("moments") or [])),
        "orders": _jsonable(dict(get("orders") or {})),
        "cost_usd": round(float(cost if cost is not None else getattr(meter, "total", 0.0)), 6),
        "by_stage": _jsonable(dict(get("by_stage") or getattr(meter, "by_stage", {}) or {})),
        "tokens": _jsonable(dict(get("tokens") or getattr(meter, "tokens", {}) or {})),
        "timings": _jsonable(dict(get("timings") or {})),
        "notes": [str(n) for n in get("notes") or []],
        "outcome": str(get("outcome", "") or "proposals"), "error": "",
    }


def _error_record(exc: BaseException, *, cost: float = 0.0, by_stage=None, tokens=None) -> dict:
    # EngineError carries what the run noted before it stopped: the best clue
    # to why an arm keeps failing.
    notes = [str(n) for n in getattr(exc, "notes", None) or []]
    return {"policy": "", "model": "", "top": [], "drafts": [], "moments": [], "orders": {},
            "cost_usd": round(float(cost or 0.0), 6), "by_stage": _jsonable(dict(by_stage or {})),
            "tokens": _jsonable(dict(tokens or {})), "timings": {}, "notes": notes,
            "outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}


def legacy_arm(*, video: dict, source, settings, inputs: dict) -> dict:
    """The transcript arm: today's transcript suggester (``legacy.suggest``),
    replayed with no other clips and no striking-frame scan (both came later
    than the archive-time pass) and the same replay context the engine arms
    get. Labeled with its prompt version ("transcript-v3"): this is the
    suggester that would remain as the fallback, not the one behind the
    historical numbers. Raises on failure; the caller records the error."""
    from .. import clip_proposals
    from . import legacy

    started = time.monotonic()
    replay = dict(inputs)
    replay["used"] = []
    replay["other_clips"] = []   # the archive moment: no clip had been made yet
    replay["striking"] = []
    replay["rejected"] = []
    # The opening pass samples frames from the video: point it at the
    # in-flight link the bench holds (legacy.suggest reads inputs["source"],
    # falling back to the candidate's own path).
    replay["source"] = str(source)
    replay["candidate"] = {**(inputs.get("candidate") or {}), "local_video_path": str(source)}
    with _metered() as meter:
        out = legacy.suggest(settings=settings, inputs=replay, window=None, topic="",
                             context=inputs.get("context") or {}) or {}
    top = []
    for clip in out.get("clips") or []:
        segments = _segments(clip.get("segments"))
        if not segments:
            continue
        top.append({"segments": segments, "story": str(clip.get("story", "") or ""),
                    "archetype": "", "score": None, "critique": None, "story_key": "",
                    "why": str(clip.get("why", "") or ""), "confidence": clip.get("confidence")})
    version = str(getattr(clip_proposals, "PROMPT_VERSION", "") or "")
    cost = out.get("cost_usd")
    cost = float(cost if cost is not None else getattr(meter, "total", 0.0) or 0.0)
    return {
        "arm": "transcript", "label": f"transcript-{version}" if version else "transcript",
        "video_id": video.get("video_id"), "candidate_pk": video.get("candidate_pk"),
        "policy": str(out.get("policy", "") or ""), "model": str(out.get("model", "") or ""),
        "prompt_version": version, "top": top, "drafts": [], "moments": [], "orders": {},
        "cost_usd": round(cost, 6), "by_stage": {"legacy": round(cost, 6)}, "tokens": {},
        "timings": {}, "seconds": round(time.monotonic() - started, 1),
        "notes": [str(n) for n in out.get("notes") or []],
        "outcome": "proposals" if top else "empty", "error": "",
    }


def _run_arm(label: str, arm: dict, video: dict, *, inputs: dict, source: Path,
             duration, transcript, words, examples, settings, playbook, brand, timeout,
             cancel, say, fp_hash: str, previous: dict | None) -> dict:
    from . import engine

    started = time.monotonic()
    arm_settings = settings_for({"overrides": arm.get("overrides") or {}}, settings)
    # The inputs were read once per video with the run's settings; this arm's
    # own settings decide how much of them it gets (they can only take away).
    context = _arm_context(inputs.get("context") or {}, arm_settings)
    if arm["setup"] == "transcript":
        examples = []   # the transcript suggester takes no taste examples
        with _metered() as meter:
            try:
                record = legacy_arm(video=video, source=source, settings=arm_settings,
                                    inputs={**inputs, "context": context,
                                            "transcript": transcript, "words": words})
            except Exception as exc:
                record = _error_record(exc, cost=getattr(meter, "total", 0.0))
    else:
        examples = _arm_examples(examples, arm_settings)
        meter = engine.CostMeter("bench")
        try:
            result = engine.run(
                setup=arm["setup"], settings=arm_settings, brand=brand, playbook=playbook,
                video_id=video["video_id"],
                title=(inputs.get("candidate") or {}).get("title") or video.get("title", ""),
                description=(inputs.get("candidate") or {}).get("description") or "",
                duration=duration, source=source, words=words, transcript=transcript,
                context=context, rejected=[], examples=examples, used=[],
                other_clips=[], window=None, meter=meter,
                progress=lambda stage: say(f"      {label}: {stage}"),
                cancel=cancel, deadline=time.monotonic() + _arm_timeout(arm_settings, timeout))
            record = _engine_record(result, meter)
        except Exception as exc:
            record = _error_record(exc, cost=getattr(meter, "total", 0.0),
                                   by_stage=getattr(meter, "by_stage", {}),
                                   tokens=getattr(meter, "tokens", {}))
    prior = 0.0
    attempts = 1
    if previous:
        attempts = int(previous.get("attempts") or 1) + 1
        prior = _result_cost(previous)
    record.update({
        "arm": label, "spec": arm.get("spec", label), "setup": arm["setup"],
        "overrides": arm.get("overrides") or {}, "video_id": video["video_id"],
        "candidate_pk": video["candidate_pk"], "fingerprint": fp_hash,
        "as_of": video.get("as_of"), "source_seconds": duration,
        "examples": [e.get("post_pk") for e in examples if isinstance(e, dict)],
        "context_keys": sorted(context.keys()),
        "input_notes": [str(n) for n in inputs.get("notes") or []],
        "seconds": round(time.monotonic() - started, 1), "created_at": _now(),
        "attempts": attempts, "prior_cost_usd": round(prior, 6),
    })
    return record


def _bench_video(run_dir: Path, manifest: dict, video: dict, pending: list[str], *, specs,
                 settings, playbook, brand, min_free_gb, timeout, cancel, say, charge,
                 prefix, manifest_lock) -> None:
    """Run every pending arm on one video, handing each result's cost to
    ``charge`` as soon as it's known. A video that can't be set up (inputs,
    transcript, source) is recorded in the manifest's ``skipped`` and retried
    on the next resume."""
    from .. import clip_proposals
    from . import jobs, media

    vid = video["video_id"]

    def skip(reason: str) -> None:
        say(f"{prefix} {vid}: skipped — {reason}")
        with manifest_lock:
            manifest.setdefault("skipped", {})[vid] = {"reason": reason[:300], "at": _now()}

    try:
        inputs = jobs.prepare_inputs(video["candidate_pk"], as_of=_parse_time(video.get("as_of")),
                                     replay=True) or {}
    except Exception as exc:
        return skip(f"inputs: {type(exc).__name__}: {exc}")
    cand = inputs.get("candidate")
    if not cand:
        # Gone from the database: its replay context can't be rebuilt.
        why = "; ".join(str(n) for n in inputs.get("notes") or []) or "video not found"
        return skip(f"inputs: {why}")
    transcript = clip_proposals.load_transcript(SimpleNamespace(
        transcript_path=cand.get("transcript_path") or video.get("transcript_path") or ""))
    words = clip_proposals.load_words(SimpleNamespace(
        word_transcript_path=cand.get("word_transcript_path") or video.get("word_transcript_path") or ""))
    if not transcript:
        return skip("no transcript")
    if not inputs.get("context"):
        _log.warning("Replay context for %s came back empty — if this account has history, "
                     "check that prepare_inputs used a writer session", vid)
    examples = _frozen_examples(video, inputs.get("examples"))
    local = cand.get("local_video_path") or video.get("local_video_path") or ""
    key = cand.get("source_object_key") or video.get("source_object_key") or ""
    try:
        source, restored, stamp = _ensure_source(local, key, min_free_gb)
    except _StopRun:
        raise
    except Exception as exc:
        return skip(f"source: {type(exc).__name__}: {exc}")
    try:
        link = media.link_inflight(source, f"bench-{_safe_file(vid)}")
    except Exception as exc:
        _log.info("No in-flight link for %s (%s); using the source in place", vid, exc)
        link = source
    try:
        try:
            duration = media.probe_duration(link)
        except Exception:
            duration = None
        duration = duration or video.get("duration")
        for label in pending:
            if cancel.is_set():
                break   # the run is stopping; the remaining arms run on resume
            previous = _load_result(run_dir, label, vid)
            record = _run_arm(label, specs[label], video, inputs=inputs, source=link,
                              duration=duration, transcript=transcript, words=words,
                              examples=examples, settings=settings, playbook=playbook,
                              brand=brand, timeout=timeout, cancel=cancel, say=say,
                              fp_hash=(manifest.get("fingerprint") or {}).get("hash", ""),
                              previous=previous)
            charge(float(record.get("cost_usd") or 0.0))   # spent, even if the write fails
            util.atomic_write_json(_result_path(run_dir, label, vid), record)
            outcome = record["error"] or f"{record.get('outcome')} · {len(record['top'])} clip(s)"
            say(f"{prefix} {vid} · {label}: {outcome} · ${record['cost_usd']:.2f} · "
                f"{record['seconds']:.0f}s")
        with manifest_lock:
            manifest.setdefault("skipped", {}).pop(vid, None)
    finally:
        if link != source:
            try:
                media.release_inflight(link)
            except Exception as exc:  # pragma: no cover - best effort
                _log.info("Could not release %s: %s", link, exc)
        if restored:
            _release_restored(source, stamp, manifest, manifest_lock, vid)


def _estimate_video(video: dict, labels: list[str], specs: dict, settings) -> float:
    from . import engine

    total = 0.0
    for label in labels:
        setup = specs[label]["setup"]
        if setup == "transcript":
            total += 0.10
            continue
        try:
            total += float(engine.estimate_usd(setup, settings, video.get("duration")))
        except Exception:
            total += 1.50
    return total


def _bucket_total(dates: list[str]) -> float | None:
    from .. import spend

    reader = getattr(spend, "bucket_spend", None)
    if reader is None:
        return None
    try:
        return round(sum(float(reader("bench", d) or 0.0) for d in dates), 6)
    except Exception:
        return None


def _dates_between(start: str, end: str) -> list[str]:
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    return [(first + dt.timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


def _write_ledger(run_dir: Path, videos: list[dict], settings, say) -> None:
    """``ledger.json``: per benched video, the transcript suggester's
    archive-time run — the earliest video-scope run that predates the first
    cut. The operator SAW these before cutting, so they're reported apart
    ("anchored"), never as an arm."""
    from sqlalchemy import func, select

    from .. import clip_proposals
    from ..models import ClipProposal, Cut

    by_pk = {v["candidate_pk"]: v for v in videos if not v.get("negative")}
    if not by_pk:
        return
    try:
        with _db() as session:
            first_cut = {}
            rows = []
            for chunk in _chunks(list(by_pk)):
                first_cut.update({pk: _parse_time(at) for pk, at in session.execute(
                    select(Cut.candidate_pk, func.min(Cut.created_at))
                    .where(Cut.candidate_pk.in_(chunk)).group_by(Cut.candidate_pk)).all()})
                rows += [dict(r._mapping) for r in session.execute(
                    select(ClipProposal.id, ClipProposal.candidate_pk, ClipProposal.run_id,
                           ClipProposal.clip_index, ClipProposal.proposed_segments,
                           ClipProposal.story, ClipProposal.verdict, ClipProposal.model,
                           ClipProposal.policy_version, ClipProposal.created_at)
                    .where(ClipProposal.candidate_pk.in_(chunk))).all()]
    except Exception as exc:
        _log.warning("Ledger unavailable: %s", exc)
        say(f"Ledger (anchored history) unavailable: {exc}")
        return
    section = _section_policies(rows, settings, clip_proposals)
    runs: dict[tuple, list[dict]] = {}
    for row in rows:
        if str(row.get("model") or "").startswith("engine:"):
            continue
        created = _parse_time(row.get("created_at"))
        run_key = row.get("run_id") or f"t{int(created.timestamp() // 10) if created else row['id']}"
        runs.setdefault((row["candidate_pk"], run_key), []).append({**row, "created": created})
    chosen: dict[int, tuple] = {}
    for (pk, run_key), members in runs.items():
        start = min((m["created"] for m in members if m["created"]), default=None)
        first_id = min(m["id"] for m in members)
        cut_at = first_cut.get(pk)
        if cut_at is not None and (start is None or start >= cut_at):
            continue
        if members[0].get("policy_version") in section:
            continue
        rank = (start or dt.datetime.max.replace(tzinfo=dt.timezone.utc), first_id)
        if pk not in chosen or rank < chosen[pk][0]:
            chosen[pk] = (rank, run_key, members)
    out: dict[str, dict] = {}
    by_policy: dict[str, int] = {}
    for pk, (rank, run_key, members) in chosen.items():
        members.sort(key=lambda m: (m.get("clip_index") or 0, m["id"]))
        policy = members[0].get("policy_version") or ""
        by_policy[policy] = by_policy.get(policy, 0) + 1
        out[by_pk[pk]["video_id"]] = {
            "candidate_pk": pk, "run_id": run_key, "created_at": _iso(rank[0]),
            "policy_version": policy, "model": members[0].get("model") or "",
            "clips": [{"segments": _segments(m.get("proposed_segments")),
                       "story": m.get("story") or "", "clip_index": m.get("clip_index"),
                       "verdict": m.get("verdict")} for m in members],
        }
    util.atomic_write_json(run_dir / "ledger.json", {
        "label": LEDGER_LABEL, "created_at": _now(), "runs": out, "by_policy": by_policy})


def _section_policies(rows: list[dict], settings, clip_proposals) -> set[str]:
    """Policy fingerprints a SECTION pass would have had with today's clip
    settings and any prompt version — best effort, to keep chapter passes out
    of the archive-time ledger."""
    try:
        out = set()
        models = {str(r.get("model") or "") for r in rows}
        for model in models:
            for prompt in ("", "v2", "v3"):
                out.add(clip_proposals.policy_version(
                    model=model, max_clips=settings.get("clips.section_max_clips", 2),
                    max_segments_per_clip=settings.get("clips.max_segments", 4),
                    min_seconds=settings.get("clips.min_seconds", 15),
                    max_seconds=settings.get("clips.max_seconds", 40),
                    scope="section", prompt=prompt))
        return out
    except Exception:
        return set()


def run(name: str, *, arms=None, limit=None, max_usd=None, seed: str = "test",
        workers: int = 1, candidate_ids=None, require_words: bool = True, negatives: int = 0,
        new_version: bool = False, keep_errors: bool = False, settings=None,
        log: Callable[[str], Any] | None = print) -> dict:
    """Create (or resume) bench run ``name`` and run every arm on every frozen
    video: videos outside, arms inside, so the budget check after each video
    leaves every arm compared on the same videos. Returns a summary.

    ``arms``/``limit``/``max_usd`` default to ``clip_engine.bench.*`` for a
    new run. A resume reuses the frozen selection, and None keeps the run's
    own arms and cap (given arms are added, a given cap replaces it);
    existing results are kept (errors are retried unless ``keep_errors``); a
    changed fingerprint refuses unless ``new_version``. Stops starting new
    videos once the run's results cost ``max_usd`` (failed attempts
    included), or when the disk gets too full. One process at a time runs a
    given run; a second refuses (``_run_lock``).
    """
    from ..config import load_settings
    from . import playbook as playbook_mod

    say = log or (lambda *_: None)
    name = _validate_name(name)
    workers = int(workers or 1)
    if workers < 1:
        raise BenchError(f"workers must be at least 1, not {workers}")
    settings = settings or load_settings()
    run_dir = bench_dir() / name
    with _run_lock(name), _bench_bucket():
        raw_playbook = playbook_mod.load()
        bench_playbook = playbook_mod.strip_examples(raw_playbook)
        fp = fingerprint(settings, bench_playbook)
        if (run_dir / "manifest.json").exists():
            run_dir, manifest, videos = _load_run(name)
            with _locked_manifest(run_dir, manifest) as fresh:
                labels = _resume_run(run_dir, fresh, videos=videos, arms=arms, max_usd=max_usd,
                                     seed=seed, limit=limit, candidate_ids=candidate_ids,
                                     require_words=require_words, negatives=negatives,
                                     new_version=new_version, settings=settings, fp=fp,
                                     say=say)
                _write_manifest(run_dir, fresh)
            manifest = fresh
            if manifest.get("revealed_at"):
                say(f"Note: run {name} was revealed at {manifest['revealed_at']}; "
                    "new results won't be judged.")
            say(f"Resuming bench run {name}: arms {', '.join(labels)} · {len(videos)} videos · "
                f"stops once results pass ${float(manifest.get('max_usd') or 0):.2f}.")
        else:
            manifest, videos = _create_run(
                run_dir, name=name, arms=arms, limit=limit, max_usd=max_usd, seed=seed,
                candidate_ids=candidate_ids, require_words=require_words,
                negatives=negatives, settings=settings, fp=fp, raw_playbook=raw_playbook,
                bench_playbook=bench_playbook, say=say)
            labels = list(manifest["arms"])
        return _execute(run_dir, manifest, videos, labels, settings=settings,
                        playbook=bench_playbook, workers=workers, keep_errors=keep_errors,
                        say=say)


def _execute(run_dir: Path, manifest: dict, videos: list[dict], labels: list[str], *,
             settings, playbook, workers: int, keep_errors: bool, say) -> dict:
    specs = manifest.get("arm_specs") or {}
    labels = [l for l in labels if l in specs]
    max_usd = float(manifest.get("max_usd") or 0.0)
    min_free = min_free_gb(settings)
    timeout = float(opt(settings, "clip_engine.job_timeout_seconds", 1800))
    brand = engine_brand()
    state = _LoopState(_run_spent(run_dir))
    manifest_lock = threading.Lock()
    cancel = threading.Event()
    started_day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    invocation = {"started_at": _now(), "arms": list(labels),
                  "bucket_before": _bucket_total([started_day])}
    total = len(videos)
    # Sources this invocation restores and deletes are appended after these.
    restored_from = len(manifest.setdefault("restored", []))

    def stop(reason: str, message: str) -> None:
        if state.stop is None:
            state.stop = {"reason": reason, "message": message, "at": _now(),
                          "spent": round(state.spent, 4), "max_usd": max_usd}
            say(f"Stopping: {message}")

    def charge(cost: float) -> None:
        """Count an arm's cost the moment its result is on disk, so the cap
        (and the other workers' checks) see it even if the video then fails."""
        with state.lock:
            state.spent += cost
            state.new_cost += cost

    def one(item: tuple[int, dict]) -> None:
        index, video = item
        pending = _pending_arms(run_dir, video, labels, keep_errors)
        if not pending:
            return
        with state.lock:
            if state.stop is not None or cancel.is_set():
                return
            if max_usd and state.spent + state.reserved >= max_usd:
                stop("max_usd", f"the run's results cost ${state.spent:.2f}, at or over "
                                f"its ${max_usd:.2f} cap")
                return
            estimate = (_estimate_video(video, pending, specs, settings)
                        if workers > 1 else 0.0)
            state.reserved += estimate
        try:
            _bench_video(run_dir, manifest, video, pending, specs=specs, settings=settings,
                         playbook=playbook, brand=brand, min_free_gb=min_free,
                         timeout=timeout, cancel=cancel, say=say, charge=charge,
                         prefix=f"[{index}/{total}]", manifest_lock=manifest_lock)
        except _StopRun as exc:
            with state.lock:
                stop(exc.reason, str(exc))
            cancel.set()
        except Exception as exc:
            # A bug or a full disk on one video shouldn't sink the run: note it
            # against the video (retried on resume) and carry on.
            _log.exception("Bench video %s failed", video["video_id"])
            with manifest_lock:
                manifest.setdefault("skipped", {})[video["video_id"]] = {
                    "reason": f"{type(exc).__name__}: {exc}"[:300], "at": _now()}
            say(f"[{index}/{total}] {video['video_id']}: failed — {type(exc).__name__}: {exc}")
        finally:
            with state.lock:
                state.reserved -= estimate
                state.ran += 1
                if max_usd and state.spent >= max_usd and state.stop is None:
                    stop("max_usd", f"the run's results cost ${state.spent:.2f}, at or over "
                                    f"its ${max_usd:.2f} cap")

    items = list(enumerate(videos, start=1))
    interrupted = False
    try:
        if workers > 1:
            util.parallel_map(one, items, workers=int(workers), cancel=cancel, name="clip-bench")
        else:
            for item in items:
                if state.stop is not None:
                    break
                one(item)
    except KeyboardInterrupt:
        interrupted = True
        cancel.set()
        with state.lock:
            stop("interrupted", "interrupted; run the same command again to resume")
    finally:
        final = _finish(run_dir, manifest, videos, labels, state=state, invocation=invocation,
                        started_day=started_day, settings=settings, say=say,
                        manifest_lock=manifest_lock, restored_from=restored_from)
    summary = _summary(run_dir, final, videos, state)
    if interrupted:
        raise KeyboardInterrupt
    return summary


def _finish(run_dir: Path, manifest: dict, videos: list[dict], labels: list[str], *, state,
            invocation: dict, started_day: str, settings, say, manifest_lock,
            restored_from: int) -> dict:
    """Record the invocation and return the manifest as saved.

    Only what the run owns is written, onto the manifest as it is on disk now
    (``_locked_manifest``): a reveal from the judging page, or the sources
    ``prepare_pairs`` restored, may have landed while it ran, and writing
    back the copy loaded at the start would undo them — reopening judging
    after the results were seen."""
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    after = _bucket_total(_dates_between(started_day, today))
    before = invocation.pop("bucket_before", None)
    invocation.update({
        "finished_at": _now(), "results_cost": round(state.new_cost, 6),
        "ledger_delta": (round(after - before, 6)
                         if after is not None and before is not None else None),
        "videos_run": state.ran,
    })
    stopped = None   # a resume clears an earlier stop's banner; it returns if this one stopped
    if state.stop is not None:
        done = sum(1 for v in videos if all(
            _load_result(run_dir, l, v["video_id"]) is not None for l in manifest.get("arms", [])))
        stopped = {**state.stop, "videos_done": done, "videos_total": len(videos)}
    with manifest_lock:   # workers still in flight after a Ctrl-C may be adding to these
        skipped = dict(manifest.get("skipped") or {})
        restored = list((manifest.get("restored") or [])[restored_from:])
    with _locked_manifest(run_dir, manifest) as fresh:
        fresh.setdefault("invocations", []).append(invocation)
        fresh["stopped"] = stopped
        fresh["skipped"] = skipped
        fresh.setdefault("restored", []).extend(restored)
        fresh["finished_at"] = _now()
        _write_manifest(run_dir, fresh)
    if state.stop is None or state.stop.get("reason") != "interrupted":
        _write_ledger(run_dir, videos, settings, say)
    try:
        from . import media

        media.prune()
    except Exception as exc:  # pragma: no cover - housekeeping only
        _log.info("media.prune skipped: %s", exc)
    return fresh


def _summary(run_dir: Path, manifest: dict, videos: list[dict], state) -> dict:
    complete, errors = {}, {}
    for label in manifest.get("arms", []):
        results = [_load_result(run_dir, label, v["video_id"]) for v in videos]
        complete[label] = sum(1 for r in results if r is not None and not r.get("error"))
        errors[label] = sum(1 for r in results if r is not None and r.get("error"))
    return {"name": manifest.get("name"), "run_dir": str(run_dir),
            "arms": list(manifest.get("arms", [])), "videos": len(videos),
            "complete": complete, "errors": errors, "ran": state.ran,
            "skipped": dict(manifest.get("skipped") or {}),
            "cost_usd": round(state.spent, 4), "cost_this_run": round(state.new_cost, 4),
            "max_usd": manifest.get("max_usd"), "stopped": manifest.get("stopped")}


# ---- evaluate --------------------------------------------------------------------

def _top_segments(result: dict | None) -> list[list[dict]]:
    if not result or result.get("error"):
        return []
    out = []
    for item in result.get("top") or []:
        segments = _segments(item.get("segments") if isinstance(item, dict) else None)
        if segments:
            out.append(segments)
    return out


def _opening_match(proposal: list[dict], targets: list[list[dict]]) -> bool:
    """Does the proposal open where the operator opened the matched cut?
    Raw play order: a cold open that starts on the payoff only matches a cut
    that starts there too."""
    best = max(targets, key=lambda t: iou(proposal, t))
    return abs(proposal[0]["start"] - best[0]["start"]) <= OPENING_TOLERANCE_S


def _score_props(props: list[list[dict]], targets: list[list[dict]], duration,
                 *, openings: bool = True) -> dict:
    def hit(p):
        return any(iou(p, t) >= HIT_IOU for t in targets)

    def moment(p):
        return any(overlap_share(p, t) >= MOMENT_OVERLAP for t in targets)

    top3 = props[:3]
    hit1 = bool(props) and hit(props[0])
    first_hit = next((p for p in top3 if hit(p)), None)
    return {
        "hit1": hit1, "hit3": first_hit is not None,
        "moment1": bool(props) and moment(props[0]),
        "moment3": any(moment(p) for p in top3),
        "recall3": (sum(1 for t in targets if any(iou(p, t) >= HIT_IOU for p in top3))
                    / len(targets)) if targets else None,
        "open1": (_opening_match(props[0], targets) if hit1 else None) if openings else None,
        "open3": (_opening_match(first_hit, targets) if first_hit is not None else None)
        if openings else None,
        "chance1": chance_hit(_dur(props[0]), targets, duration) if props else 0.0,
    }


def _cost_now(result: dict) -> float:
    """The result's cost at today's prices (Gemini Flash's introductory rate)."""
    cost = float(result.get("cost_usd") or 0.0)
    tokens = result.get("tokens") or {}
    if not isinstance(tokens, dict) or not tokens:
        return cost
    from .. import spend

    estimate = getattr(spend, "estimate_cost", None)
    if estimate is None:
        return cost
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    saving = 0.0
    for model, used in tokens.items():
        intro = _INTRO_PRICING.get(model)
        if not intro or today > intro[0] or not isinstance(used, dict):
            continue
        try:
            saving += float(estimate(model, int(used.get("input", 0) or 0),
                                     int(used.get("output", 0) or 0))) * (1 - intro[1])
        except Exception:
            continue
    return max(0.0, cost - saving)


def _video_row(video: dict, result: dict | None, targets: list[list[dict]], duration,
               *, openings: bool = True, costs: bool = True) -> dict:
    props = _top_segments(result)
    error = bool(result and result.get("error"))
    row = _score_props(props, targets, duration, openings=openings)
    row.update({
        "video_id": video["video_id"], "length_bucket": video.get("length_bucket", "unknown"),
        "kind": video.get("kind", "channel"), "error": error,
        "empty": not error and not props, "n_props": len(props),
        "prop_seconds": [round(_dur(p), 2) for p in props],
        "cost": None, "cost_full": None, "cost_now": None, "seconds": None,
    })
    if costs and result is not None:
        cost = float(result.get("cost_usd") or 0.0)
        cached = float((result.get("by_stage") or {}).get("perceive_cached") or 0.0)
        row.update({"cost": cost, "cost_full": cost + cached, "cost_now": _cost_now(result),
                    "seconds": result.get("seconds")})
    return row


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    hit1 = _rate(rows, "hit1")
    chance1 = _mean([r["chance1"] for r in rows])
    seconds = [s for r in rows for s in r["prop_seconds"]]
    agg = {
        "n": n, "errors": sum(1 for r in rows if r["error"]),
        "error_rate": _rate(rows, "error"), "empty_rate": _rate(rows, "empty"),
        "hit1": hit1, "hit3": _rate(rows, "hit3"),
        "moment1": _rate(rows, "moment1"), "moment3": _rate(rows, "moment3"),
        "recall3": _mean([r["recall3"] for r in rows]),
        "open1": _rate(rows, "open1"), "open1_n": sum(1 for r in rows if r["open1"] is not None),
        "open3": _rate(rows, "open3"), "open3_n": sum(1 for r in rows if r["open3"] is not None),
        "chance1": chance1,
        "lift1": (hit1 - chance1) if hit1 is not None and chance1 is not None else None,
        "lift_x1": (hit1 / chance1) if hit1 is not None and chance1 else None,
        "props_per_video": _mean([r["n_props"] for r in rows]),
        "median_seconds": statistics.median(seconds) if seconds else None,
        "cost_per_video": _mean([r["cost"] for r in rows]),
        "cost_full_per_video": _mean([r["cost_full"] for r in rows]),
        "cost_now_per_video": _mean([r["cost_now"] for r in rows]),
        "seconds_per_video": _mean([r["seconds"] for r in rows]),
    }
    agg["strata"] = {
        dim: {key: {"n": len(sub), "hit1": _rate(sub, "hit1"), "hit3": _rate(sub, "hit3")}
              for key, sub in sorted(_group(rows, dim).items())}
        for dim in ("length_bucket", "kind")
    }
    return agg


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row.get(key)), []).append(row)
    return out


def _view_targets(video: dict, view: str) -> list[list[dict]]:
    targets = video.get("targets") or []
    if view == "unseeded":
        targets = [t for t in targets if not t.get("seeded")]
    return [t["segments"] for t in targets if _segments(t.get("segments"))]


def _order_props(result: dict | None, order: str) -> list[list[dict]]:
    if not result or result.get("error"):
        return []
    segs: dict[str, list[dict]] = {}
    for item in list(result.get("drafts") or []) + list(result.get("top") or []):
        if isinstance(item, dict) and item.get("id") is not None:
            found = _segments(item.get("segments"))
            if found:
                segs.setdefault(str(item["id"]), found)
    ids = (result.get("orders") or {}).get(order) or []
    return [segs[str(i)] for i in ids if str(i) in segs]


def _source_duration(video: dict, results: Iterable[dict | None]) -> float | None:
    for result in results:
        if result and result.get("source_seconds"):
            try:
                return float(result["source_seconds"])
            except (TypeError, ValueError):
                continue
    return video.get("duration")


def _arm_roles(manifest: dict) -> tuple[list[str], list[str], str | None]:
    specs = manifest.get("arm_specs") or {}
    labels = [l for l in manifest.get("arms") or [] if l in specs]
    engine_labels = [l for l in labels if specs[l].get("setup") != "transcript"]
    transcript = next((l for l in labels if specs[l].get("setup") == "transcript"), None)
    return labels, engine_labels, transcript


def evaluate(name: str) -> dict:
    """Score every arm against the operator's exported cuts — files only.

    Arms are compared on the COMMON set: positive videos where every arm has
    a result (errors and empty tops count as misses). Each metric is given
    for all targets and for unseeded targets only (the headline: a cut started
    from a legacy suggestion favors the transcript arm). See the module
    docstring for what each number means.
    """
    run_dir, manifest, videos = _load_run(name)
    labels, engine_labels, transcript = _arm_roles(manifest)
    specs = manifest.get("arm_specs") or {}
    positives = [v for v in videos if not v.get("negative")]
    results = {label: {v["video_id"]: _load_result(run_dir, label, v["video_id"])
                       for v in videos} for label in labels}
    common = [v for v in positives
              if labels and all(results[l][v["video_id"]] is not None for l in labels)]
    ledger = (util.read_json(run_dir / "ledger.json", {}) or {}).get("runs") or {}
    durations = {v["video_id"]: _source_duration(v, [results[l][v["video_id"]] for l in labels])
                 for v in common}

    views: dict[str, dict] = {}
    for view in ("unseeded", "all"):
        scope = [v for v in common if _view_targets(v, view)]
        arms_out, fallback, orders, ledger_rows = {}, {}, {}, []
        for label in labels:
            rows = [_video_row(v, results[label][v["video_id"]], _view_targets(v, view),
                               durations[v["video_id"]]) for v in scope]
            arms_out[label] = _aggregate(rows)
        for label in engine_labels:
            orders[label] = {}
            for order in ORDER_NAMES:
                rows = [_score_props(_order_props(results[label][v["video_id"]], order),
                                     _view_targets(v, view), durations[v["video_id"]],
                                     openings=False) for v in scope]
                orders[label][order] = {"n": len(rows), "hit1": _rate(rows, "hit1"),
                                        "hit3": _rate(rows, "hit3")}
            if transcript:
                rows = []
                for v in scope:
                    result = results[label][v["video_id"]]
                    if result is None or result.get("error") or result.get("outcome") == "empty_drafts":
                        result = results[transcript][v["video_id"]]
                    rows.append(_video_row(v, result, _view_targets(v, view),
                                           durations[v["video_id"]]))
                fallback[f"{label}+fallback"] = _aggregate(rows)
        for v in scope:
            entry = ledger.get(v["video_id"])
            if entry:
                props = [c["segments"] for c in entry.get("clips") or [] if _segments(c.get("segments"))]
                ledger_rows.append(_video_row(v, {"top": [{"segments": p} for p in props]},
                                              _view_targets(v, view), durations[v["video_id"]],
                                              openings=False, costs=False))
        target_seconds = [_dur(t) for v in scope for t in _view_targets(v, view)]
        views[view] = {
            "n_videos": len(scope), "arms": arms_out, "fallback": fallback, "orders": orders,
            "ledger": _aggregate(ledger_rows) if ledger_rows else None,
            "ledger_by_policy": _group_count(ledger, [v["video_id"] for v in scope]),
            "operator_median_seconds": statistics.median(target_seconds) if target_seconds else None,
            "operator_targets_per_video": _mean([len(_view_targets(v, view)) for v in scope]),
        }

    invocations = manifest.get("invocations") or []
    deltas = [i.get("ledger_delta") for i in invocations if i.get("ledger_delta") is not None]
    return {
        "name": manifest.get("name"), "seed": manifest.get("seed"), "mode": manifest.get("mode"),
        "dev": _is_dev(manifest), "arms": labels,
        "display": {l: specs[l].get("display", l) for l in labels},
        "engine_arms": engine_labels, "transcript_arm": transcript,
        "videos": len(positives), "negatives": len(videos) - len(positives),
        "common": len(common),
        "attempted": {l: sum(1 for v in positives if results[l][v["video_id"]] is not None)
                      for l in labels},
        "views": views,
        "spent": {l: round(sum(_result_cost(r) for r in results[l].values()), 4) for l in labels},
        "reconciliation": {
            "results_cost": round(sum(float(i.get("results_cost") or 0) for i in invocations), 4),
            "ledger_delta": round(sum(deltas), 4) if deltas else None,
            "invocations": len(invocations),
        },
        "stopped": manifest.get("stopped"), "warnings": list(manifest.get("warnings") or []),
        "fingerprint": (manifest.get("fingerprint") or {}).get("hash", ""),
    }


def _group_count(ledger: dict, video_ids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for vid in video_ids:
        entry = ledger.get(vid)
        if entry:
            policy = entry.get("policy_version") or ""
            out[policy] = out.get(policy, 0) + 1
    return out


def _is_dev(manifest: dict) -> bool:
    return manifest.get("seed") != "test" or manifest.get("mode") != "sampled"


# ---- pairs -------------------------------------------------------------------------

def _reference_target(video: dict) -> dict | None:
    """Your reference cut: exported, published first, most viewed."""
    targets = [t for t in video.get("targets") or [] if _segments(t.get("segments"))]
    if not targets:
        return None
    return sorted(targets, key=lambda t: (not t.get("published"),
                                          -(t["views"] if t.get("views") is not None else -1)))[0]


def _item_id(name: str, kind: str, video_id: str, arm: str) -> str:
    return _hex(f"{name}|{kind}|{video_id}|{arm}")[:12]


def _arm_on_b(name: str, item_id: str) -> bool:
    """Side assignment: the arm's clip goes on B when sha1(name + item_id) is odd."""
    return int(_hex(name + item_id), 16) % 2 == 1


def _make_item(name: str, kind: str, video: dict, arm: str, first: tuple,
               second: tuple | None, sources: list) -> dict:
    item_id = _item_id(name, kind, video["video_id"], arm)
    if second is None:
        placed = [first]
    else:
        placed = [second, first] if _arm_on_b(name, item_id) else [first, second]
    sides = {}
    for side, (source, ref, segments) in zip("ab", placed):
        kept = _drop_slivers(segments)
        sides[side] = {"source": source, "ref": ref, "segments": kept,
                       "dropped_slivers": len(_segments(segments)) - len(kept),
                       "render": f"renders/{item_id}-{side}.mp4"}
    item = {"item_id": item_id, "kind": kind, "video_id": video["video_id"], "arm": arm,
            "sides": sides, "auto": None, "iou": None,
            "source_hash": _json_sha(sources, n=16), "order_key": _hex(name + item_id),
            "render_error": ""}
    if second is not None:
        item["iou"] = round(iou(first[2], second[2]), 4)
        if item["iou"] >= AUTO_TIE_IOU:
            item["auto"] = "tie"
    return item


def _pair_items(name: str, manifest: dict, videos: list[dict], run_dir: Path) -> tuple[list[dict], dict]:
    labels, engine_labels, transcript = _arm_roles(manifest)
    results = {label: {v["video_id"]: _load_result(run_dir, label, v["video_id"])
                       for v in videos} for label in labels}
    positives = [v for v in videos if not v.get("negative")]
    common = [v for v in positives
              if labels and all(results[l][v["video_id"]] is not None for l in labels)]
    items: list[dict] = []
    no_top1: dict[str, list[str]] = {label: [] for label in engine_labels}
    for v in common:
        vid = v["video_id"]
        reference = _reference_target(v)
        for label in engine_labels:
            mine = results[label][vid]
            top = _top_segments(mine)
            if transcript:
                theirs = results[transcript][vid]
                other = _top_segments(theirs)
                if top and other:
                    items.append(_make_item(name, "arms", v, label, ("arm", label, top[0]),
                                            ("transcript", transcript, other[0]),
                                            [mine, theirs]))
            if reference is not None:
                if top:
                    items.append(_make_item(name, "ceiling", v, label, ("arm", label, top[0]),
                                            ("reference", reference["cut_pk"],
                                             reference["segments"]), [mine, reference]))
                else:
                    no_top1[label].append(vid)
    if engine_labels:
        primary = engine_labels[0]
        for v in videos:
            if not v.get("negative"):
                continue
            mine = results[primary][v["video_id"]]
            top = _top_segments(mine)
            if top:
                items.append(_make_item(name, "solo", v, primary, ("arm", primary, top[0]),
                                        None, [mine]))
    return items, no_top1


def _source_fields(videos: list[dict]) -> dict[str, tuple[str, str]]:
    """Fresh ``video_id -> (local path, object key)`` (a source may have been
    offloaded since the run was frozen); the frozen values when the DB can't
    be read."""
    out = {v["video_id"]: (v.get("local_video_path") or "", v.get("source_object_key") or "")
           for v in videos}
    try:
        from sqlalchemy import select

        from ..models import Candidate

        pks = [v["candidate_pk"] for v in videos]
        with _db() as session:
            for chunk in _chunks(pks):
                for vid, local, key in session.execute(
                        select(Candidate.video_id, Candidate.local_video_path,
                               Candidate.source_object_key).where(Candidate.id.in_(chunk))).all():
                    if vid in out:
                        out[vid] = (local or out[vid][0], key or out[vid][1])
    except Exception as exc:
        _log.info("Using frozen source paths: %s", exc)
    return out


def prepare_pairs(name: str, *, height: int = 540) -> int:
    """Build the blind judging queue and render every side.

    Kinds: "arms" (each engine arm's top pick vs the transcript arm's, on
    common videos where both have one — the PRIMARY test), "ceiling" (engine
    top pick vs your reference cut — secondary), "solo" (the first engine
    arm's top pick on never-cut videos). Sides with IoU ≥ 0.9 are scored a tie
    without being shown. Both sides render through the same pipeline at the
    same height. Idempotent: unchanged items keep their renders and
    judgments; an item whose results changed is re-rendered and its old
    judgments no longer count. Returns how many items are waiting to be judged.
    """
    run_dir, manifest, videos = _load_run(name)
    if manifest.get("revealed_at"):
        raise BenchError(f"run {name} was revealed at {manifest['revealed_at']}; judging is "
                         "closed. Start a new run to judge again.")
    height = int(height)
    items, no_top1 = _pair_items(manifest.get("name", name), manifest, videos, run_dir)
    previous = {it["item_id"]: it for it in _pairs(run_dir).get("items") or []}
    (run_dir / "renders").mkdir(parents=True, exist_ok=True)
    stale = 0
    needed: dict[str, list[dict]] = {}
    for item in items:
        item["height"] = height
        if item["auto"]:
            continue
        prev = previous.get(item["item_id"])
        files = [run_dir / s["render"] for s in item["sides"].values()]
        if (prev and prev.get("source_hash") == item["source_hash"]
                and prev.get("height") == height and not prev.get("render_error")
                and all(f.is_file() for f in files)):
            continue
        if prev and prev.get("source_hash") != item["source_hash"]:
            stale += 1
        for f in files:
            f.unlink(missing_ok=True)
        needed.setdefault(item["video_id"], []).append(item)

    restored: list[dict] = []
    if needed:
        _render_needed(run_dir, needed, [v for v in videos if v["video_id"] in needed],
                       height=height, restored=restored)
    failures = sum(1 for it in items if it["render_error"])
    judgeable = sum(1 for it in items if not it["auto"] and not it["render_error"])
    kinds: dict[str, int] = {}
    for item in items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    pairs = {
        "name": manifest.get("name", name), "updated_at": _now(), "height": height,
        "items": items, "no_top1": no_top1, "render_failures": failures, "stale_items": stale,
        "judgeable": judgeable, "auto_ties": sum(1 for it in items if it["auto"]),
        "kinds": kinds, "slivers_dropped": sum(s["dropped_slivers"] for it in items
                                               for s in it["sides"].values()),
    }
    # Under the manifest lock (a run or the judging page may write it from
    # another process meanwhile), so a reveal lands either before the new
    # queue — which is then never written — or after all of it.
    with _judge_lock, _locked_manifest(run_dir, manifest) as fresh:
        if restored:
            fresh.setdefault("restored", []).extend(restored)
            _write_manifest(run_dir, fresh)
        if fresh.get("revealed_at"):
            raise BenchError(f"run {name} was revealed at {fresh['revealed_at']} while its "
                             "pairs were being prepared; judging is closed, so the judging "
                             "queue was left as it was.")
        util.atomic_write_json(run_dir / "pairs.json", pairs)
    return judgeable


def _render_needed(run_dir: Path, needed: dict[str, list[dict]], videos: list[dict], *,
                   height: int, restored: list[dict]) -> None:
    """Render the sides of ``needed`` items, one source video at a time (each
    restored from storage only while its sides render), a few sides at once.
    A failure marks its item (``render_error``) instead of dropping it, so the
    report can count what couldn't be shown."""
    from . import media

    min_free = 10.0
    try:
        from ..config import load_settings

        min_free = min_free_gb(load_settings())
    except Exception:
        pass
    sources = _source_fields(videos)
    lock = threading.Lock()
    stopped = ""
    for vid, todo in needed.items():
        if stopped:
            for item in todo:
                item["render_error"] = stopped
            continue
        local, key = sources.get(vid, ("", ""))
        try:
            source, was_restored, stamp = _ensure_source(local, key, min_free)
        except _StopRun as exc:
            stopped = f"not rendered: {exc}"
            for item in todo:
                item["render_error"] = stopped
            continue
        except Exception as exc:
            for item in todo:
                item["render_error"] = f"source: {type(exc).__name__}: {exc}"[:300]
            continue
        try:
            link = media.link_inflight(source, f"bench-pairs-{_safe_file(vid)}")
        except Exception as exc:
            _log.info("No in-flight link for %s (%s); rendering from the source", vid, exc)
            link = source
        try:
            work = [(item, side) for item in todo for side in item["sides"].values()]

            def render(job):
                item, side = job
                if not side["segments"]:
                    raise media.MediaError("nothing left to render after dropping slivers")
                return media.render_segments(link, side["segments"], run_dir / side["render"],
                                             height=height)

            for (item, _side), outcome in zip(work, util.parallel_map(
                    render, work, workers=3, name="clip-bench-render")):
                if isinstance(outcome, BaseException) and not item["render_error"]:
                    item["render_error"] = f"{type(outcome).__name__}: {outcome}"[:300]
            for item in todo:
                if item["render_error"]:
                    for side in item["sides"].values():
                        (run_dir / side["render"]).unlink(missing_ok=True)
        finally:
            if link != source:
                try:
                    media.release_inflight(link)
                except Exception as exc:  # pragma: no cover - best effort
                    _log.info("Could not release %s: %s", link, exc)
            if was_restored:
                holder: dict = {}
                _release_restored(source, stamp, holder, lock, vid)
                restored.extend(holder.get("restored", []))


# ---- judging ------------------------------------------------------------------------

_judge_lock = threading.Lock()


def _pairs(run_dir: Path) -> dict:
    data = util.read_json(run_dir / "pairs.json", {}) or {}
    return data if isinstance(data, dict) else {}


def _judgeable(items: list[dict]) -> list[dict]:
    return [it for it in items if not it.get("auto") and not it.get("render_error")]


def _judgments(run_dir: Path) -> list[dict]:
    path = run_dir / "judgments.jsonl"
    out = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("item_id"):
            out.append(entry)
    return out


def _latest(run_dir: Path, items: list[dict], revealed_at: str | None = None) -> dict[str, dict]:
    """``item_id -> judgment``: the latest one given for the item's CURRENT
    results (and before the reveal)."""
    current = {it["item_id"]: it.get("source_hash") for it in items}
    cutoff = _parse_time(revealed_at)
    out: dict[str, dict] = {}
    for entry in _judgments(run_dir):
        item_id = entry["item_id"]
        if item_id not in current or entry.get("source_hash") != current[item_id]:
            continue
        if cutoff is not None and (_parse_time(entry.get("at")) or cutoff) > cutoff:
            continue
        out[item_id] = entry
    return out


def _judging_order(items: list[dict]) -> list[dict]:
    """sha1(name + item_id) order (interleaves kinds and arms), with a video's
    items spaced apart so the same footage isn't judged twice in a row."""
    pending = sorted(items, key=lambda it: it.get("order_key") or it["item_id"])
    distinct = len({it["video_id"] for it in items})
    gap = max(0, min(3, distinct - 1))
    recent: list[str] = []
    out = []
    while pending:
        pick = pending[0]
        if gap:
            pick = next((it for it in pending if it["video_id"] not in recent[-gap:]), pending[0])
        pending.remove(pick)
        out.append(pick)
        recent.append(pick["video_id"])
    return out


def _progress(items: list[dict], latest: dict) -> dict:
    judgeable = _judgeable(items)
    judged = sum(1 for it in judgeable if it["item_id"] in latest)
    return {"judged": judged, "total": len(judgeable), "remaining": len(judgeable) - judged}


def _public_item(name: str, item: dict, progress: dict) -> dict:
    """What the judging page may see: no arm, no kind beyond pair/solo (the
    page must not say which pair holds your own cut), no video id."""
    solo = item["kind"] == "solo"
    sides = ["a"] if solo else ["a", "b"]
    return {"item_id": item["item_id"], "kind": "solo" if solo else "pair",
            "mode": "solo" if solo else "pair", "sides": sides,
            "choices": list(SOLO_CHOICES if solo else PAIR_CHOICES), "reasons": list(REASONS),
            "media": {s: f"/clip-bench/{name}/media/{item['item_id']}/{s}" for s in sides},
            "progress": progress}


def next_item(name: str) -> dict | None:
    """The next item to judge (see ``_public_item`` for what it reveals), or
    None when everything is judged or the results were revealed."""
    run_dir, manifest, _ = _load_run(name)
    if manifest.get("revealed_at"):
        return None
    items = _pairs(run_dir).get("items") or []
    latest = _latest(run_dir, items)
    progress = _progress(items, latest)
    for item in _judging_order(_judgeable(items)):
        if item["item_id"] not in latest:
            return _public_item(manifest.get("name", name), item, progress)
    return None


def media_path(name: str, item_id: str, side: str) -> Path | None:
    """The rendered file for one side of a judgeable item, or None."""
    try:
        run_dir, _, _ = _load_run(name)
    except BenchError:
        return None
    if side not in ("a", "b"):
        return None
    for item in _judgeable(_pairs(run_dir).get("items") or []):
        if item["item_id"] == item_id and side in item["sides"]:
            path = run_dir / item["sides"][side]["render"]
            return path if path.is_file() else None
    return None


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def judge(name: str, item_id: str, choice: str, *, recognised=False, reason: str = "") -> dict:
    """Record one blind judgment. Pairs take a|b|tie|neither, solo items
    yes|no; ``reason`` (optional, for non-wins) is one of ``REASONS``. The
    latest answer for an item wins until ``reveal``; after it, judging is
    closed and this raises."""
    run_dir, manifest, _ = _load_run(name)
    choice = str(choice or "").strip().lower()
    reason = str(reason or "").strip().lower()
    # The manifest lock too: a reveal from another process (`clipbench reveal`
    # while the page judges) lands either before this check or after the
    # judgment is on file.
    with _judge_lock, _locked_manifest(run_dir, manifest) as manifest:
        if manifest.get("revealed_at"):
            raise BenchError(f"results for {name} were revealed at {manifest['revealed_at']}; "
                             "judging is closed")
        items = _pairs(run_dir).get("items") or []
        item = next((it for it in _judgeable(items) if it["item_id"] == item_id), None)
        if item is None:
            raise BenchError(f"no item {item_id!r} to judge in run {name}")
        allowed = SOLO_CHOICES if item["kind"] == "solo" else PAIR_CHOICES
        if choice not in allowed:
            raise BenchError(f"choice must be one of {', '.join(allowed)}, not {choice!r}")
        if reason and reason not in REASONS:
            raise BenchError(f"reason must be one of {', '.join(REASONS)} or empty, not {reason!r}")
        entry = {"item_id": item_id, "choice": choice, "recognised": _truthy(recognised),
                 "reason": reason, "at": _now(), "source_hash": item.get("source_hash"),
                 "kind": item["kind"], "arm": item["arm"], "video_id": item["video_id"],
                 "sides": {s: v["source"] for s, v in item["sides"].items()}}
        path = run_dir / "judgments.jsonl"
        with open(path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
    latest = _latest(run_dir, items)
    return {"ok": True, "item_id": item_id, "choice": choice,
            "progress": _progress(items, latest), "next": next_item(name)}


def reveal(name: str) -> dict:
    """Unseal the results. Judging closes for good: answers given after seeing
    the numbers would no longer be blind. Every other manifest writer merges
    into the file under the same lock, so nothing can undo this."""
    run_dir, manifest, _ = _load_run(name)
    with _judge_lock, _locked_manifest(run_dir, manifest) as manifest:
        if not manifest.get("revealed_at"):
            manifest["revealed_at"] = _now()
            _write_manifest(run_dir, manifest)
    items = _pairs(run_dir).get("items") or []
    progress = _progress(items, _latest(run_dir, items, manifest["revealed_at"]))
    return {"name": manifest.get("name", name), "revealed_at": manifest["revealed_at"],
            "progress": progress}


# ---- statistics ----------------------------------------------------------------------

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (binomial, p = 0.5) on non-tied judgments."""
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def ceiling_verdict(k: int, n: int, bar: float = CEILING_BAR) -> str:
    """"pass" when the one-sided 95% lower bound (the low end of the two-sided
    90% interval) clears ``bar``; "fail" only when the whole two-sided 95%
    interval — the one the report prints beside the verdict — lies below it;
    "inconclusive" otherwise (that interval straddles ``bar``)."""
    if n <= 0:
        return "no data"
    if wilson(k, n, Z90)[0] >= bar:
        return "pass"
    if wilson(k, n, Z95)[1] < bar:
        return "fail"
    return "inconclusive"


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (average ranks for ties); None below 3 pairs
    or when either side is constant."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


# ---- results / report ----------------------------------------------------------------------

def _winner(item: dict, choice: str) -> str | None:
    """Which source won a pair judgment ("arm", "transcript", "reference")."""
    if choice in ("a", "b"):
        return item["sides"][choice]["source"]
    return None


def _losers(kind: str, label: str, transcript: str | None, item: dict, choice: str) -> list[str]:
    """Arm labels a judgment's optional reason counts against: the losing
    arm's; both arms' on a tie or "neither" (the reason is about both); none
    when your own reference cut lost."""
    if kind == "solo":
        return [label] if choice == "no" else []
    won = _winner(item, choice)
    if won is None:
        return [label] + ([transcript] if kind == "arms" and transcript else [])
    if won == "arm":
        return [transcript] if kind == "arms" and transcript else []
    return [label]


def _ceiling_stats(rows: list[tuple[dict, dict | None]], no_top1: int) -> dict:
    judged = [(it, j) for it, j in rows if j is not None]
    auto = sum(1 for it, _ in rows if it.get("auto"))
    wins = sum(1 for it, j in judged if _winner(it, j["choice"]) == "arm")
    losses = sum(1 for it, j in judged if _winner(it, j["choice"]) == "reference")
    ties = sum(1 for _, j in judged if j["choice"] == "tie")
    neither = sum(1 for _, j in judged if j["choice"] == "neither")
    n = len(judged) + auto + no_top1
    k = wins + ties + auto
    low, high = wilson(k, n)
    return {"judged": len(judged), "auto_ties": auto, "no_top1": no_top1, "wins": wins,
            "ties": ties, "losses": losses, "neither": neither, "n": n,
            "win_or_tie": (k / n) if n else None, "ci": [low, high],
            "lower_1s": wilson(k, n, Z90)[0] if n else None,
            "verdict": ceiling_verdict(k, n)}


def results(name: str) -> dict:
    """The blind judgments, tallied — only after ``reveal``; before it, just
    progress. See the module docstring for the decision rules."""
    run_dir, manifest, _ = _load_run(name)
    pairs = _pairs(run_dir)
    items = pairs.get("items") or []
    revealed_at = manifest.get("revealed_at")
    latest = _latest(run_dir, items, revealed_at)
    out: dict[str, Any] = {"name": manifest.get("name", name), "revealed": bool(revealed_at),
                           "revealed_at": revealed_at, "progress": _progress(items, latest),
                           "prepared": bool(pairs)}
    if not revealed_at:
        return out
    labels, engine_labels, transcript = _arm_roles(manifest)
    by_kind: dict[str, dict[str, list]] = {"arms": {}, "ceiling": {}, "solo": {}}
    for item in items:
        if item.get("render_error"):
            continue
        by_kind.setdefault(item["kind"], {}).setdefault(item["arm"], []).append(
            (item, latest.get(item["item_id"])))

    arms_out = {}
    for label, rows in by_kind["arms"].items():
        judged = [(it, j) for it, j in rows if j is not None]
        wins = sum(1 for it, j in judged if _winner(it, j["choice"]) == "arm")
        losses = sum(1 for it, j in judged if _winner(it, j["choice"]) == "transcript")
        decided = wins + losses
        low, high = wilson(wins, decided)
        arms_out[label] = {
            "vs": transcript, "judged": len(judged),
            "auto_ties": sum(1 for it, _ in rows if it.get("auto")),
            "wins": wins, "losses": losses,
            "ties": sum(1 for _, j in judged if j["choice"] == "tie"),
            "neither": sum(1 for _, j in judged if j["choice"] == "neither"),
            "pending": sum(1 for it, j in rows if j is None and not it.get("auto")),
            "win_rate": (wins / decided) if decided else None, "ci": [low, high],
            "sign_p": sign_test(wins, losses),
        }

    ceiling_out = {}
    no_top1 = pairs.get("no_top1") or {}
    for label in engine_labels:
        rows = by_kind["ceiling"].get(label, [])
        missing = len(no_top1.get(label) or [])
        if not rows and not missing:
            continue
        unrecognised = [(it, j) for it, j in rows if not (j and j.get("recognised"))]
        ceiling_out[label] = {
            "overall": _ceiling_stats(rows, missing),
            "unrecognised": _ceiling_stats(unrecognised, missing),
            "recognised": sum(1 for _, j in rows if j and j.get("recognised")),
            "pending": sum(1 for it, j in rows if j is None and not it.get("auto")),
        }

    solo_out = {}
    for label, rows in by_kind["solo"].items():
        judged = [j for _, j in rows if j is not None]
        yes = sum(1 for j in judged if j["choice"] == "yes")
        low, high = wilson(yes, len(judged))
        solo_out[label] = {"judged": len(judged), "yes": yes, "no": len(judged) - yes,
                           "share_yes": (yes / len(judged)) if judged else None,
                           "ci": [low, high],
                           "pending": sum(1 for _, j in rows if j is None)}

    reasons: dict[str, dict[str, dict[str, int]]] = {}
    for kind, per_arm in by_kind.items():
        for label, rows in per_arm.items():
            for item, judgment in rows:
                if not judgment or not judgment.get("reason"):
                    continue
                for loser in _losers(kind, label, transcript, item, judgment["choice"]):
                    bucket = reasons.setdefault(loser, {}).setdefault(kind, {})
                    bucket[judgment["reason"]] = bucket.get(judgment["reason"], 0) + 1

    out.update({"arms": arms_out, "ceiling": ceiling_out, "solo": solo_out,
                "loss_reasons": reasons, "render_failures": int(pairs.get("render_failures") or 0),
                "display": {l: (manifest.get("arm_specs") or {}).get(l, {}).get("display", l)
                            for l in labels}})
    return out


def _pct(value, digits: int = 0) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def _usd(value) -> str:
    return "—" if value is None else f"${value:.2f}"


def _num(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths)))

    return [line(headers), "  ".join("-" * w for w in widths)] + [line(r) for r in rows]


def _banners(manifest: dict, evaluation: dict | None = None) -> list[str]:
    out = []
    if _is_dev(manifest):
        why = ("hand-picked videos" if manifest.get("mode") != "sampled"
               else f"seed {manifest.get('seed')!r}")
        out.append(f"!! {DEV_BANNER} ({why}). Decide on a sampled run with seed 'test'.")
    stopped = manifest.get("stopped")
    if stopped:
        if stopped.get("reason") == "max_usd":
            out.append(f"!! Stopped at the ${float(stopped.get('max_usd') or 0):.2f} cap after "
                       f"{stopped.get('videos_done', '?')} of {stopped.get('videos_total', '?')} "
                       "videos; arms are compared only on videos every arm finished.")
        else:
            out.append(f"!! Stopped early ({stopped.get('reason')}): {stopped.get('message', '')}")
    for warning in manifest.get("warnings") or []:
        out.append(f"!! {warning}")
    if evaluation is not None:
        missing = {l: evaluation["videos"] - n for l, n in evaluation["attempted"].items()
                   if n < evaluation["videos"]}
        if missing:
            out.append("!! Not every video has a result for every arm (" + ", ".join(
                f"{evaluation['display'].get(l, l)}: {m} missing" for l, m in missing.items())
                + "); the tables use the common set only.")
    out.append(f"Note: {RUBRIC_NOTE}.")
    return out


def _hits_table(view: dict, display: dict) -> list[str]:
    headers = ["arm", "n", "err", "empty", "hit@1", "hit@3", "mom@1", "mom@3", "rec@3",
               "open@1", "chance@1", "lift@1", "clips/vid", "med s", "$/vid", "s/vid"]
    rows = []
    entries = [(display.get(l, l), agg) for l, agg in view["arms"].items()]
    entries += [(f"{display.get(l.rsplit('+fallback', 1)[0], l.rsplit('+fallback', 1)[0])} + fallback", agg)
                for l, agg in view["fallback"].items()]
    for label, agg in entries:
        rows.append([label, agg["n"], _pct(agg["error_rate"]), _pct(agg["empty_rate"]),
                     _pct(agg["hit1"]), _pct(agg["hit3"]), _pct(agg["moment1"]),
                     _pct(agg["moment3"]), _pct(agg["recall3"]),
                     f"{_pct(agg['open1'])} ({agg['open1_n']})", _pct(agg["chance1"], 1),
                     ("—" if agg["lift1"] is None else f"{100 * agg['lift1']:+.0f} pts"),
                     _num(agg["props_per_video"]), _num(agg["median_seconds"], 0),
                     _usd(agg["cost_per_video"]), _num(agg["seconds_per_video"], 0)])
    lines = _table(headers, rows)
    if view.get("operator_median_seconds") is not None:
        lines.append(f"Your cuts: median {view['operator_median_seconds']:.0f} s, "
                     f"{_num(view.get('operator_targets_per_video'))} per video.")
    return lines


def report(name: str) -> str:
    """``evaluate`` + ``results`` as aligned text tables (n on every row).
    Until ``reveal`` it shows only progress: seeing hit rates first would
    color the blind judging."""
    run_dir, manifest, videos = _load_run(name)
    labels, engine_labels, transcript = _arm_roles(manifest)
    specs = manifest.get("arm_specs") or {}
    display = {l: specs[l].get("display", l) for l in labels}
    fp = (manifest.get("fingerprint") or {}).get("hash", "")
    lines = [f"Clip bench — {manifest.get('name', name)}",
             f"seed {manifest.get('seed')} · {manifest.get('mode')} · created "
             f"{manifest.get('created_at', '')[:10]} · version {fp[:8]} · arms: "
             + ", ".join(display[l] for l in labels),
             f"{sum(1 for v in videos if not v.get('negative'))} videos with cuts, "
             f"{sum(1 for v in videos if v.get('negative'))} never cut."]
    res = results(name)
    if not res["revealed"]:
        lines += [""] + _banners(manifest)
        lines += ["", "Results stay hidden until you reveal them (clipbench reveal, or the "
                      "Reveal button on the bench page): seeing hit rates first would color "
                      "your judging. Reveal closes judging.", "", "Progress"]
        rows = []
        for label in labels:
            found = [_load_result(run_dir, label, v["video_id"]) for v in videos]
            rows.append([display[label], sum(1 for r in found if r is not None),
                         sum(1 for r in found if r and r.get("error")),
                         _usd(sum(_result_cost(r) for r in found))])
        lines += _table(["arm", "results", "errors", "spent"], rows)
        p = res["progress"]
        lines.append(f"Judging: {p['judged']} of {p['total']} judged"
                     + ("" if res["prepared"] else " (pairs not prepared yet)") + ".")
        return "\n".join(lines)

    ev = evaluate(name)
    lines += [""] + _banners(manifest, ev)
    for view, title in (("unseeded", "Hits vs your exported cuts — unseeded targets (headline)"),
                        ("all", "Hits — all targets (incl. cuts started from a suggestion)")):
        block = ev["views"][view]
        lines += ["", f"{title}: {block['n_videos']} videos"]
        lines += _hits_table(block, display)
    head = ev["views"]["unseeded"]
    if engine_labels:
        lines += ["", "Engine orders (unseeded): would an earlier step's order have found your cut?"]
        rows = [[display[l], o, head["orders"][l][o]["n"], _pct(head["orders"][l][o]["hit1"]),
                 _pct(head["orders"][l][o]["hit3"])] for l in engine_labels for o in ORDER_NAMES]
        lines += _table(["arm", "order", "n", "hit@1", "hit@3"], rows)
    for dim, title in (("length_bucket", "By source length"), ("kind", "By source kind")):
        rows = []
        for label in labels:
            for key, cell in head["arms"][label]["strata"][dim].items():
                rows.append([display[label], key, cell["n"], _pct(cell["hit1"]), _pct(cell["hit3"])])
        if rows:
            lines += ["", f"{title} (unseeded)"] + _table(["arm", "stratum", "n", "hit@1", "hit@3"], rows)
    ledger = head.get("ledger")
    lines += ["", f"Anchored history — the transcript suggester's archive-time runs: {LEDGER_LABEL}"]
    if ledger:
        lines += _table(["", "n", "hit@1", "hit@3", "mom@1", "rec@3", "chance@1", "clips/vid"],
                        [["ledger", ledger["n"], _pct(ledger["hit1"]), _pct(ledger["hit3"]),
                          _pct(ledger["moment1"]), _pct(ledger["recall3"]),
                          _pct(ledger["chance1"], 1), _num(ledger["props_per_video"])]])
        if head.get("ledger_by_policy"):
            lines.append("By policy: " + ", ".join(f"{p or '(none)'} × {n}"
                                                   for p, n in head["ledger_by_policy"].items()))
    else:
        lines.append("No archive-time runs found for these videos.")
    lines += ["", "Cost per video (common set; Gemini Flash is recorded at its 2027 price)"]
    rows = [[display[l], _usd(ev["views"]["all"]["arms"][l]["cost_per_video"]),
             _usd(ev["views"]["all"]["arms"][l]["cost_now_per_video"]),
             _usd(ev["views"]["all"]["arms"][l]["cost_full_per_video"]),
             _usd(ev["spent"][l])] for l in labels]
    lines += _table(["arm", "2027 prices", "today", "incl. cached log", "spent (all tries)"], rows)
    rec = ev["reconciliation"]
    lines.append(f"Reconciliation: results ${rec['results_cost']:.2f} vs the bench spend bucket "
                 + ("(unavailable)" if rec["ledger_delta"] is None else f"${rec['ledger_delta']:.2f}")
                 + f" over {rec['invocations']} invocation(s).")

    lines += ["", f"Blind judging: {res['progress']['judged']} of {res['progress']['total']} judged"
              + ("" if res["prepared"] else " (no pairs were prepared before the reveal)")
              + (f"; {res['render_failures']} item(s) failed to render" if res["render_failures"] else "")]
    if res["arms"]:
        lines += ["", f"Engine vs {display.get(transcript, transcript)} (PRIMARY: neither clip is yours)"]
        rows = [[display.get(l, l), a["judged"], a["wins"], a["losses"], a["ties"], a["auto_ties"],
                 a["neither"], _pct(a["win_rate"]), f"{_pct(a['ci'][0])}–{_pct(a['ci'][1])}",
                 f"{a['sign_p']:.3f}"] for l, a in res["arms"].items()]
        lines += _table(["arm", "judged", "W", "L", "tie", "auto-tie", "neither", "win rate",
                         "95% CI", "sign p"], rows)
        lines.append("Win rate = engine wins among decided pairs (ties and 'neither' excluded).")
    if res["ceiling"]:
        lines += ["", "Engine vs your own cut (secondary ceiling; bar: win-or-tie ≥ 40%)"]
        rows = []
        for label, c in res["ceiling"].items():
            for variant in ("overall", "unrecognised"):
                s = c[variant]
                rows.append([display.get(label, label), variant, s["n"], s["wins"],
                             s["ties"] + s["auto_ties"], s["losses"], s["neither"], s["no_top1"],
                             _pct(s["win_or_tie"]), f"{_pct(s['ci'][0])}–{_pct(s['ci'][1])}",
                             _pct(s.get("lower_1s")), s["verdict"]])
        lines += _table(["arm", "items", "n", "W", "T", "L", "neither", "no top-1", "win-or-tie",
                         "95% CI", "1-sided low", "verdict"], rows)
        lines.append("Win-or-tie = (wins + ties) / (judged + auto-ties + videos with no top pick); "
                     "pass needs the one-sided 95% lower bound ≥ 40%, fail needs the whole "
                     "95% CI below 40%; anything between is inconclusive.")
    if res["solo"]:
        lines += ["", "Never-cut videos: would you post the top pick?"]
        lines += _table(["arm", "judged", "yes", "share", "95% CI"],
                        [[display.get(l, l), s["judged"], s["yes"], _pct(s["share_yes"]),
                          f"{_pct(s['ci'][0])}–{_pct(s['ci'][1])}"] for l, s in res["solo"].items()])
    if res["loss_reasons"]:
        lines += ["", "Why clips lost (optional reason on non-wins)"]
        rows = [[display.get(l, l), kind] + [counts.get(r, 0) for r in REASONS]
                for l, kinds in res["loss_reasons"].items() for kind, counts in kinds.items()]
        lines += _table(["arm", "kind"] + list(REASONS), rows)
    if transcript:
        lines += ["", f"{display.get(transcript, transcript)} is today's transcript suggester, "
                      "replayed; it is not the suggester behind the historical figures."]
    return "\n".join(lines)


# ---- critic check ---------------------------------------------------------------------------

def _local_percentiles(posts: list, views: dict[int, int], window_days: int = 30) -> dict[int, float]:
    """``post_pk -> percentile`` of each post's fixed-age views among posts
    published within ±``window_days`` (follower growth makes a pool-wide
    percentile meaningless)."""
    dated = [(p.id, _parse_time(p.published_at), views[p.id]) for p in posts
             if p.id in views and p.published_at is not None]
    out = {}
    span = dt.timedelta(days=window_days)
    for pk, when, value in dated:
        pool = [v for other, t, v in dated if other != pk and abs(t - when) <= span]
        if not pool:
            continue
        below = sum(1 for v in pool if v < value)
        ties = sum(1 for v in pool if v == value)
        out[pk] = (below + 0.5 * ties) / len(pool)
    return out


def _critic_items(session, *, limit: int) -> list[dict]:
    from sqlalchemy import select
    from sqlalchemy.orm import defer, selectinload

    from .. import analytics
    from ..categories import is_first_party
    from ..models import Candidate, ContentLabel, Cut, ThreadsPost

    rows = session.execute(
        select(ThreadsPost.id, ThreadsPost.cut_pk, ThreadsPost.published_at,
               ThreadsPost.clip_object_path, ThreadsPost.clip_local_path, ThreadsPost.caption)
        .where(ThreadsPost.status == "published", ThreadsPost.cut_pk.is_not(None),
               ThreadsPost.repost_of_post_pk.is_(None), ThreadsPost.published_at.is_not(None))
        .order_by(ThreadsPost.published_at.asc(), ThreadsPost.id.asc())
    ).all()
    first: dict[int, Any] = {}
    for row in rows:
        first.setdefault(row.cut_pk, row)
    cuts = {}
    for chunk in _chunks(list(first)):
        cuts.update({c.id: c for c in session.execute(
            select(Cut.id, Cut.candidate_pk, Cut.trim_segments, Cut.clip_title)
            .where(Cut.id.in_(chunk))).all()})
    cand_pks = sorted({c.candidate_pk for c in cuts.values()})
    candidates = {}
    for chunk in _chunks(cand_pks):
        candidates.update({c.id: c for c in session.execute(
            select(Candidate).options(selectinload(Candidate.channel),
                                      defer(Candidate.transcript_text),
                                      defer(Candidate.description))
            .where(Candidate.id.in_(chunk))).scalars().all()})
    firsts = [p for p in first.values() if p.cut_pk in cuts
              and not is_first_party(candidates.get(cuts[p.cut_pk].candidate_pk))]
    labels = dict(session.execute(
        select(ContentLabel.cut_pk, ContentLabel.value)
        .where(ContentLabel.label == "gripping", ContentLabel.value.in_(("yes", "no")))).all())
    views = analytics.metrics_at_age_bulk(session, firsts, "views", 6)
    percentiles = _local_percentiles(firsts, views)
    newest = sorted(firsts, key=lambda p: (_parse_time(p.published_at), p.id), reverse=True)
    chosen = [p for p in newest if p.cut_pk in labels][:limit]
    if len(chosen) < 20:
        chosen += [p for p in newest if p.cut_pk not in labels][:limit]
    items = []
    for p in chosen:
        cut = cuts[p.cut_pk]
        cand = candidates.get(cut.candidate_pk)
        segments = _segments(cut.trim_segments)
        if not segments:
            continue
        items.append({
            "post_pk": p.id, "cut_pk": p.cut_pk, "candidate_pk": cut.candidate_pk,
            "video_id": getattr(cand, "video_id", "") or "", "title": cut.clip_title or "",
            "caption": p.caption or "", "label": labels.get(p.cut_pk),
            "segments": segments, "seconds": round(sum(s["end"] - s["start"] for s in segments), 2),
            "views": views.get(p.id), "percentile": percentiles.get(p.id),
            "published_at": _iso(p.published_at), "clip_object_path": p.clip_object_path or "",
            "clip_local_path": p.clip_local_path or "",
            "word_transcript_path": getattr(cand, "word_transcript_path", "") or "",
        })
    return items


def _agreement(rows: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for label in ("yes", "no"):
        sub = [r for r in rows if r.get("label") == label and not r.get("error")]
        want = label == "yes"
        stops = sum(1 for r in sub if bool(r.get("stops_scroll")) == want)
        score6 = sum(1 for r in sub if (float(r.get("score") or 0) >= 6) == want)
        out[label] = {"n": len(sub), "stops_scroll_agree": stops, "score6_agree": score6,
                      "stops_scroll_rate": stops / len(sub) if sub else None,
                      "score6_rate": score6 / len(sub) if sub else None}
    yes, no = out["yes"], out["no"]
    if yes["n"] < 5 or no["n"] < 5:
        out["gate"] = "too few labels"
    elif (yes["stops_scroll_rate"] or 0) >= 0.8 and (no["stops_scroll_rate"] or 0) >= 0.8:
        out["gate"] = "pass"
    else:
        out["gate"] = "fail"
    return out


def critic_check(name_or_none, *, roles=("gemini",), limit: int = 40, settings=None,
                 log: Callable[[str], Any] | None = print) -> dict:
    """Does the critic agree with you? ``critic.cold_watch`` over your
    published first airings that carry a "gripping" label (plus, while there
    are fewer than 20 labels, the most recent posts), reporting how often
    ``stops_scroll`` and score ≥ 6 match your yes/no labels — separately, with
    counts — and the Spearman correlation of score with local-percentile views
    at 6 h. Until a critic clears ~80% on both, treat its scores as a gate,
    not a ranking. Writes ``bench_dir()/critic_check-<date>.json`` (and into
    the run's directory when a run name is given)."""
    from ..config import load_settings

    say = log or (lambda *_: None)
    settings = settings or load_settings()
    roles = [str(r).strip().lower() for r in _as_list(roles)]
    if not roles:
        raise BenchError("no critic roles to check (gemini and/or claude)")
    for role in roles:
        if role not in ("gemini", "claude"):
            raise BenchError(f"critic role must be gemini or claude, not {role!r}")
    run_dir = None
    if name_or_none:
        run_dir, _, _ = _load_run(str(name_or_none))
    with _bench_bucket():
        return _critic_check(run_dir, name_or_none, roles=roles, limit=int(limit),
                             settings=settings, say=say)


def _critic_check(run_dir: Path | None, name_or_none, *, roles: list[str], limit: int,
                  settings, say) -> dict:
    from . import critic, engine, media, taste

    with _db() as session:
        items = _critic_items(session, limit=limit)
    labelled = sum(1 for it in items if it["label"])
    say(f"Critic check: {len(items)} posts ({labelled} labeled), roles {', '.join(roles)}.")
    b = engine_brand()

    def prepare(item: dict):
        example = {**item, "label": "hit", "transcript": ""}
        path = taste.example_file(example, settings, height=360)
        if path is None:
            return None
        try:
            seconds = media.probe_duration(path) or item["seconds"]
        except Exception:
            seconds = item["seconds"]
        return Path(path), float(seconds)

    files = util.parallel_map(prepare, items, workers=3, name="clip-bench-files")
    word_cache: dict[str, list] = {}

    def clip_words(item: dict) -> list:
        from .. import clip_proposals, subtitles

        path = item.get("word_transcript_path") or ""
        if path not in word_cache:
            word_cache[path] = clip_proposals.load_words(
                SimpleNamespace(word_transcript_path=path)) if path else []
        return subtitles.slice_source_words(word_cache[path], item["segments"])

    out_roles = {}
    for role in roles:
        meter = engine.CostMeter("bench")

        def watch(pair):
            item, prepared = pair
            if prepared is None or isinstance(prepared, BaseException):
                return {"error": "no clip file"}
            path, seconds = prepared
            return critic.cold_watch(role=role, settings=settings, brand=b, render=path,
                                     clip_seconds=seconds, clip_words=clip_words(item),
                                     credits="", meter=meter)

        critiques = util.parallel_map(watch, list(zip(items, files)), workers=4,
                                      name=f"clip-bench-critic-{role}")
        rows = []
        for item, critique in zip(items, critiques):
            if isinstance(critique, BaseException):
                critique = {"error": f"{type(critique).__name__}: {critique}"}
            critique = critique if isinstance(critique, dict) else {"error": "no answer"}
            rows.append({"post_pk": item["post_pk"], "cut_pk": item["cut_pk"],
                         "label": item["label"], "percentile": item["percentile"],
                         "views": item["views"], "score": critique.get("score"),
                         "stops_scroll": critique.get("stops_scroll"),
                         "stops_scroll_muted": critique.get("stops_scroll_muted"),
                         "error": str(critique.get("error") or "")})
        scored = [r for r in rows if not r["error"] and r["percentile"] is not None
                  and r["score"] is not None]
        rho = spearman([float(r["score"]) for r in scored], [r["percentile"] for r in scored])
        out_roles[role] = {
            "n": len(rows), "errors": sum(1 for r in rows if r["error"]),
            "agreement": _agreement(rows), "spearman": rho, "n_spearman": len(scored),
            "cost_usd": round(float(getattr(meter, "total", 0.0)), 4), "items": rows,
        }
        agree = out_roles[role]["agreement"]
        say(f"  {role}: yes {_pct(agree['yes']['stops_scroll_rate'])} of {agree['yes']['n']}, "
            f"no {_pct(agree['no']['stops_scroll_rate'])} of {agree['no']['n']} "
            f"({agree['gate']}); Spearman {_num(rho, 2)} on {len(scored)}; "
            f"${out_roles[role]['cost_usd']:.2f}")
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    report_data = {"date": today, "created_at": _now(), "roles": out_roles,
                   "posts": len(items), "labelled": labelled,
                   "run": str(name_or_none) if name_or_none else None,
                   "missing_files": sum(1 for f in files if f is None or isinstance(f, BaseException))}
    path = bench_dir() / f"critic_check-{today}.json"
    util.atomic_write_json(path, report_data)
    if run_dir is not None:
        util.atomic_write_json(run_dir / "critic_check.json", report_data)
    report_data["path"] = str(path)
    return report_data


# ---- listing ------------------------------------------------------------------------------

def list_runs() -> list[dict]:
    """Every run under ``bench_dir()``, newest first. Never raises."""
    out = []
    try:
        dirs = sorted(p for p in bench_dir().iterdir() if p.is_dir())
    except OSError:
        return []
    for run_dir in dirs:
        try:
            manifest = util.read_json(run_dir / "manifest.json", None)
            if not isinstance(manifest, dict):
                continue
            videos = util.read_json(run_dir / "videos.json", []) or []
            specs = manifest.get("arm_specs") or {}
            arms = list(manifest.get("arms") or [])
            items = _pairs(run_dir).get("items") or []
            progress = _progress(items, _latest(run_dir, items, manifest.get("revealed_at")))
            out.append({
                "name": manifest.get("name", run_dir.name), "seed": manifest.get("seed"),
                "mode": manifest.get("mode"), "dev": _is_dev(manifest),
                "arms": [specs.get(a, {}).get("display", a) for a in arms], "arm_labels": arms,
                "created_at": manifest.get("created_at"), "videos": len(videos),
                "results": {a: sum(1 for _ in (run_dir / a).glob("*.json")) for a in arms},
                "judged": progress["judged"], "total": progress["total"],
                "revealed": bool(manifest.get("revealed_at")),
                "revealed_at": manifest.get("revealed_at"), "stopped": manifest.get("stopped"),
            })
        except Exception as exc:  # pragma: no cover - one broken run must not hide the rest
            _log.info("Skipping unreadable bench run %s: %s", run_dir.name, exc)
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return out
