"""LLM spend ledger + daily budget guardrails.

Every model call — Anthropic and Gemini — records its token usage here; cost is
estimated from per-model prices (``llm.pricing`` in settings over
``DEFAULT_PRICING``). The ledger is a small JSON file at ``data/llm_spend.json``
keyed by UTC date — no DB migration, safe to delete.

The point is a hard cap: unattended features check ``within_budget()`` before
spending, so a freak high-volume day can never blow past
``llm.daily_budget_usd``.

Buckets. The clip engine spends far more per run ($0.4-1.5 a video) than
anything else here, so its spend is kept OUT of that shared budget: engine calls
pass ``bucket="engine"`` and land under ``day["buckets"]["engine"]``, which
``today_spend()``/``within_budget()``/``recent()`` never count. The engine has
its own cap (``clip_engine.daily_usd`` against ``bucket_spend("engine")``).
Without the split, a few re-rolls would switch off caption checks, rerun
captions and footage scoring for the rest of the UTC day. The bucket is passed
explicitly because the dashboard is one process with many threads;
``set_bucket()`` sets a process-wide default and is only for single-purpose CLI
processes (the bench records everything it spends under "bench").

Concurrency. The dashboard and a ``run.py`` CLI can record at the same moment,
so the read-modify-write holds an exclusive ``flock`` on ``<ledger>.lock`` (plus
a thread lock), and the tmp file name is unique per process and thread — with a
shared tmp name one writer renames the other's half-written file away.

``metered()`` totals what the calling thread spends inside a block (the legacy
clip suggester reports a run's cost that way) without re-reading the ledger.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

try:  # POSIX. Elsewhere the thread lock alone guards the ledger.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from .config import DATA_DIR, load_settings

log = logging.getLogger("spend")

_LEDGER = DATA_DIR / "llm_spend.json"
_lock = threading.Lock()

# USD per 1,000,000 tokens (output includes thinking). Only used to estimate
# spend for the budget guards; a row in settings ``llm.pricing`` overrides the
# row here. Gemini 3.8/3.6 Flash are priced at their 2027 rate on purpose — the
# guard errs high (the real price is half until 2026-12-31). claude-opus-5 and
# claude-opus-4-8 are the server-side fallback targets for Opus 5.5 refusals.
DEFAULT_PRICING = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "gemini-3.8-flash": {"input": 1.50, "output": 7.50},
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-4-hydrogen": {"input": 1.50, "output": 7.50},
    "gemini-4-carbon": {"input": 4.00, "output": 20.00},
    "gemini-4-argon": {"input": 4.00, "output": 20.00},
    "gemini-3.1-pro-preview": {"input": 4.00, "output": 18.00},
    "default": {"input": 3.0, "output": 15.0},
}

# The process-wide default bucket (CLI processes only; see the module docstring).
_default_bucket: str | None = None
# Per-thread stack of open ``metered()`` blocks.
_local = threading.local()
# Models already warned about as unpriced, so the log says it once per process.
_unpriced_warned: set[str] = set()


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _date_key(date: Any = None) -> str:
    """A ledger day key from None (today, UTC), a date/datetime, or a string."""
    if date is None:
        return _today()
    if isinstance(date, dt.datetime):
        if date.tzinfo is not None:
            date = date.astimezone(dt.timezone.utc)
        return date.strftime("%Y-%m-%d")
    if isinstance(date, dt.date):
        return date.isoformat()
    return str(date).strip()[:10]


def _count(value: Any) -> int:
    """A token count as a non-negative int; anything unusable counts as 0."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


# --- pricing ----------------------------------------------------------------

def _pricing_table() -> dict:
    """``DEFAULT_PRICING`` with the settings rows laid over it. A settings row
    replaces the code row key by key, so a row that only sets ``input`` keeps
    the code's ``output``. Malformed rows are ignored."""
    table = {name: dict(row) for name, row in DEFAULT_PRICING.items()}
    try:
        override = load_settings().get("llm.pricing", {}) or {}
    except Exception as exc:  # a broken settings.yaml must not break a call
        log.warning("Could not read llm.pricing: %s", exc)
        override = {}
    if isinstance(override, dict):
        for name, row in override.items():
            if isinstance(row, dict):
                table[str(name)] = {**table.get(str(name), {}), **row}
    return table


def _pricing_for(model: str) -> dict:
    """The price row for ``model``: an exact match; else the longest row whose
    id prefixes it at a dash (a dated snapshot id such as
    ``claude-opus-5-5-20261001`` is priced as ``claude-opus-5-5``); else
    ``default``, with a one-time warning so a missing row gets noticed."""
    table = _pricing_table()
    if model in table:
        return table[model]
    best = ""
    for name in table:
        if name != "default" and model.startswith(name + "-") and len(name) > len(best):
            best = name
    if best:
        return table[best]
    if model not in _unpriced_warned:
        _unpriced_warned.add(model)
        log.warning("No price for model %r; using the default row. Add it to llm.pricing.", model)
    return table.get("default") or DEFAULT_PRICING["default"]


def _rate(row: dict, key: str) -> float:
    """One price from a row, falling back to the default row's price when the
    value is missing, malformed or negative."""
    try:
        value = float(row.get(key))
        if value >= 0:
            return value
    except (TypeError, ValueError):
        pass
    return float(DEFAULT_PRICING["default"][key])


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _pricing_for(model)
    return (_count(input_tokens) / 1_000_000) * _rate(p, "input") + \
           (_count(output_tokens) / 1_000_000) * _rate(p, "output")


# --- locking + file I/O ---------------------------------------------------------

@contextlib.contextmanager
def file_lock(path: Path, *, timeout: float = 10.0) -> Iterator[bool]:
    """Hold an exclusive cross-process advisory lock (POSIX ``flock``) on
    ``path`` for the block; yields whether the lock is held.

    Shared by this ledger and the clip engine's Gemini upload cache — both are
    read-modify-written by the dashboard and CLI processes at once. Gives up
    after ``timeout`` seconds (or when flock is unavailable) and yields False:
    callers then proceed under their thread lock alone, because a rare lost
    update beats blocking a model call behind a stuck process."""
    if fcntl is None:  # pragma: no cover - Windows
        yield False
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        log.warning("Could not open lock file %s: %s", path, exc)
        yield False
        return
    held = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    log.warning("Timed out waiting for %s; continuing without it", path)
                    break
                time.sleep(0.01)
            except OSError as exc:
                log.warning("Could not lock %s: %s", path, exc)
                break
        yield held
    finally:
        if held:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def _ledger_lock() -> Iterator[None]:
    ledger = _LEDGER
    with _lock, file_lock(ledger.with_name(ledger.name + ".lock")):
        yield


def _read() -> dict:
    """The ledger for reading; {} when missing or unreadable (writes are
    atomic, so a reader never sees a half-written file)."""
    try:
        data = json.loads(_LEDGER.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_for_update() -> dict:
    """The ledger for a read-modify-write (caller holds the lock). A corrupt
    file is moved aside rather than silently replaced by today's tally."""
    ledger = _LEDGER
    try:
        text = ledger.read_text()
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    backup = ledger.with_name(f"{ledger.name}.corrupt-{int(time.time())}")
    with contextlib.suppress(OSError):
        os.replace(ledger, backup)
    log.warning("Spend ledger %s was unreadable; moved it to %s and started fresh",
                ledger, backup.name)
    return {}


def _write(data: dict) -> None:
    ledger = _LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger.with_name(f".{ledger.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, ledger)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _empty_tally() -> dict:
    return {"spend_usd": 0.0, "calls": 0, "by_model": {}}


def _tally(target: dict, model: str, cost: float, input_tokens: int, output_tokens: int) -> None:
    """Add one call to a day's (or a bucket's) totals, repairing any field a
    hand edit or an older version left in the wrong shape."""
    target["spend_usd"] = round(_num(target.get("spend_usd")) + cost, 6)
    target["calls"] = _count(target.get("calls")) + 1
    by_model = target.get("by_model")
    if not isinstance(by_model, dict):
        by_model = target["by_model"] = {}
    m = by_model.get(model)
    if not isinstance(m, dict):
        m = by_model[model] = {"spend_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
    m["spend_usd"] = round(_num(m.get("spend_usd")) + cost, 6)
    m["calls"] = _count(m.get("calls")) + 1
    m["input_tokens"] = _count(m.get("input_tokens")) + input_tokens
    m["output_tokens"] = _count(m.get("output_tokens")) + output_tokens


# --- buckets + meters -------------------------------------------------------------

def _clean_bucket(name: Any) -> str | None:
    text = str(name).strip() if name is not None else ""
    return text or None


def set_bucket(name: str | None) -> None:
    """Make ``name`` the bucket for every ``record()`` in this PROCESS that
    doesn't pass one (None/"" clears it). For single-purpose CLI processes
    only — in the dashboard it would capture every thread's spend."""
    global _default_bucket
    _default_bucket = _clean_bucket(name)


def current_bucket() -> str | None:
    """The process default set by ``set_bucket`` (None in the dashboard)."""
    return _default_bucket


class Meter:
    """What the calling thread recorded while a ``metered()`` block was open."""

    def __init__(self) -> None:
        self.total = 0.0
        self.calls = 0
        self.by_model: dict[str, float] = {}

    def _add(self, model: str, cost: float) -> None:
        self.total += cost
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0.0) + cost

    def __repr__(self) -> str:
        return f"Meter(total={self.total:.6f}, calls={self.calls})"


def _meters() -> list[Meter]:
    stack = getattr(_local, "meters", None)
    if stack is None:
        stack = _local.meters = []
    return stack


@contextlib.contextmanager
def metered() -> Iterator[Meter]:
    """``with spend.metered() as m: ...`` — ``m.total``/``m.calls`` count every
    ``record()`` made on THIS thread inside the block (bucketed or not).
    Blocks nest: each open meter gets the cost. Work handed to other threads
    is not counted (the engine's own ``CostMeter`` covers its pools)."""
    meter = Meter()
    stack = _meters()
    stack.append(meter)
    try:
        yield meter
    finally:
        with contextlib.suppress(ValueError):
            stack.remove(meter)


# --- recording ----------------------------------------------------------------------

def record(model: str, input_tokens: int, output_tokens: int, *,
           bucket: str | None = None) -> float:
    """Add one call's usage to today's ledger and return its estimated cost.

    ``bucket`` (or else the process default from ``set_bucket``) files the
    spend under ``day["buckets"][bucket]`` instead of the day's main total,
    which is what ``llm.daily_budget_usd`` caps. Every ``metered()`` block
    open on this thread gets the cost. Never raises — spend tracking must not
    break an LLM call."""
    model = str(model or "").strip() or "unknown"
    tokens_in, tokens_out = _count(input_tokens), _count(output_tokens)
    try:
        cost = estimate_cost(model, tokens_in, tokens_out)
    except Exception as exc:  # pragma: no cover - pricing is defensive already
        log.warning("Could not price %s: %s", model, exc)
        cost = 0.0
    for meter in list(_meters()):
        meter._add(model, cost)
    name = _clean_bucket(bucket) or _default_bucket
    try:
        with _ledger_lock():
            data = _read_for_update()
            today = _today()
            day = data.get(today)
            if not isinstance(day, dict):
                day = data[today] = _empty_tally()
            target = day
            if name:
                buckets = day.get("buckets")
                if not isinstance(buckets, dict):
                    buckets = day["buckets"] = {}
                target = buckets.get(name)
                if not isinstance(target, dict):
                    target = buckets[name] = _empty_tally()
            _tally(target, model, cost, tokens_in, tokens_out)
            _write(data)
    except Exception as exc:  # best-effort accounting
        log.warning("Could not record spend: %s", exc)
    return cost


def bucket_spend(name: str, date: Any = None) -> float:
    """USD recorded under bucket ``name`` on ``date`` (UTC day; default today).
    0.0 when nothing was recorded or the ledger is unreadable."""
    day = _read().get(_date_key(date))
    if not isinstance(day, dict):
        return 0.0
    buckets = day.get("buckets")
    entry = buckets.get(str(name)) if isinstance(buckets, dict) else None
    return _num(entry.get("spend_usd")) if isinstance(entry, dict) else 0.0


# --- the shared daily budget (un-bucketed spend only) ------------------------------------

def today_spend() -> float:
    day = _read().get(_today())
    return _num(day.get("spend_usd")) if isinstance(day, dict) else 0.0


def daily_budget() -> float:
    value = load_settings().get("llm.daily_budget_usd", 3.0)
    try:
        return float(value)
    except (TypeError, ValueError):  # blank or malformed YAML value
        return 3.0


def within_budget(headroom: float = 0.0) -> bool:
    """True if today's spend (plus an optional expected ``headroom``) is under
    the configured daily budget."""
    return today_spend() + headroom < daily_budget()


def remaining_today() -> float:
    return max(0.0, daily_budget() - today_spend())


def recent(days: int = 7, *, bucket: str | None = None) -> list[dict]:
    """Newest-first per-day summaries for the UI: the shared budget's spend,
    or with ``bucket`` that bucket's (which the shared figures never include,
    so the clip engine's spend is shown beside them rather than in them)."""
    data = _read()
    name = _clean_bucket(bucket)
    out = []
    for i in range(days):
        d = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        tally = data.get(d)
        if name and isinstance(tally, dict):
            buckets = tally.get("buckets")
            tally = buckets.get(name) if isinstance(buckets, dict) else None
        tally = tally if isinstance(tally, dict) else {}
        out.append({"date": d, "spend_usd": round(_num(tally.get("spend_usd")), 4),
                    "calls": _count(tally.get("calls"))})
    return out
