"""Lightweight LLM spend ledger + daily budget guardrail.

Every Anthropic call records its token usage here; cost is estimated from the
per-model prices in settings (``llm.pricing``). The ledger is a small JSON file
at ``data/llm_spend.json`` keyed by UTC date — no DB migration, safe to delete.

The point is a hard cap: vision scoring checks ``within_budget()`` before
spending, so a freak high-volume day can never blow past ``llm.daily_budget_usd``.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from pathlib import Path

from .config import ROOT, load_settings

log = logging.getLogger("spend")

_LEDGER = ROOT / "data" / "llm_spend.json"
_lock = threading.Lock()

# USD per 1,000,000 tokens. Only used to estimate spend for the budget guard;
# overridable via settings ``llm.pricing``. Kept conservative when unknown.
DEFAULT_PRICING = {
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-5": {"input": 3.0, "output": 15.0},
    "default": {"input": 3.0, "output": 15.0},
}


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _pricing_for(model: str) -> dict:
    table = dict(DEFAULT_PRICING)
    override = load_settings().get("llm.pricing", {}) or {}
    if isinstance(override, dict):
        table.update(override)
    return table.get(model) or table.get("default") or DEFAULT_PRICING["default"]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _pricing_for(model)
    return (input_tokens / 1_000_000) * float(p.get("input", 0)) + \
           (output_tokens / 1_000_000) * float(p.get("output", 0))


def _read() -> dict:
    if not _LEDGER.exists():
        return {}
    try:
        return json.loads(_LEDGER.read_text())
    except Exception:
        return {}


def _write(data: dict) -> None:
    _LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = _LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(_LEDGER)


def record(model: str, input_tokens: int, output_tokens: int) -> float:
    """Add one call's usage to today's tally. Returns the estimated cost added.
    Never raises — spend tracking must not break an LLM call."""
    cost = estimate_cost(model, input_tokens, output_tokens)
    try:
        with _lock:
            data = _read()
            day = data.setdefault(_today(), {"spend_usd": 0.0, "calls": 0, "by_model": {}})
            day["spend_usd"] = round(day.get("spend_usd", 0.0) + cost, 6)
            day["calls"] = day.get("calls", 0) + 1
            m = day["by_model"].setdefault(model, {"spend_usd": 0.0, "calls": 0,
                                                    "input_tokens": 0, "output_tokens": 0})
            m["spend_usd"] = round(m["spend_usd"] + cost, 6)
            m["calls"] += 1
            m["input_tokens"] += int(input_tokens)
            m["output_tokens"] += int(output_tokens)
            _write(data)
    except Exception as exc:  # pragma: no cover - best-effort accounting
        log.warning("Could not record spend: %s", exc)
    return cost


def today_spend() -> float:
    return float(_read().get(_today(), {}).get("spend_usd", 0.0))


def daily_budget() -> float:
    return float(load_settings().get("llm.daily_budget_usd", 3.0))


def within_budget(headroom: float = 0.0) -> bool:
    """True if today's spend (plus an optional expected ``headroom``) is under
    the configured daily budget."""
    return today_spend() + headroom < daily_budget()


def remaining_today() -> float:
    return max(0.0, daily_budget() - today_spend())


def recent(days: int = 7) -> list[dict]:
    """Newest-first per-day summaries for the UI."""
    data = _read()
    out = []
    for i in range(days):
        d = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        day = data.get(d)
        out.append({"date": d, "spend_usd": round(float(day.get("spend_usd", 0.0)), 4) if day else 0.0,
                    "calls": day.get("calls", 0) if day else 0})
    return out
