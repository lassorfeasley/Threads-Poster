"""Throughline's universal rules: the house method every account follows.

Rules (prose, with a scope) and a few hard settings (clip length) that apply
to every account and win over the account's own rules and settings. The
defaults ship in app/universal_rules.yaml; the operator edits the live copy
in Throughline Admin, which keeps it in the platform's Storage and serves it
to every account's workers at ``UNIVERSAL_RULES_URL`` (with
``UNIVERSAL_RULES_TOKEN``).

Workers re-read it every ten minutes and keep the last good copy on disk, so
a sleeping or unreachable admin site never leaves an account without rules:
the order is live copy, then last good copy, then the shipped defaults.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("universal_rules")

DEFAULTS_FILE = Path(__file__).with_name("universal_rules.yaml")
SCOPES = ("videos", "clips", "captions", "everything")
SCOPE_LABELS = {"videos": "Videos and moments", "clips": "Moments and cuts",
                "captions": "Captions", "everything": "Everything"}
# Which rule scopes reach each kind of prompt (the kinds llm._operator_rules takes).
_REACHES = {"screening": ("everything", "videos"),
            "content": ("everything", "videos", "clips"),
            "captions": ("everything", "captions")}
TTL = 600.0
MIN_SECONDS_FLOOR, MAX_SECONDS_CEILING = 5, 180
MAX_RULE_CHARS = 600

_cache: tuple[float, dict] | None = None
_lock = threading.Lock()


# --- Shape ------------------------------------------------------------------------

def clean(data: Any) -> dict:
    """A checked copy: known scopes, short text, sane lengths. Raises
    ValueError with a message for the admin page."""
    if not isinstance(data, dict):
        raise ValueError("Rules must be a mapping")
    length = data.get("clip_length") or {}
    lo, hi = int(length.get("min_seconds", 20)), int(length.get("max_seconds", 45))
    if not (MIN_SECONDS_FLOOR <= lo < hi <= MAX_SECONDS_CEILING):
        raise ValueError(f"Clip length must run from at least {MIN_SECONDS_FLOOR}s to at most "
                         f"{MAX_SECONDS_CEILING}s, shortest first")
    rules, seen = [], set()
    for i, r in enumerate(data.get("rules") or []):
        if not isinstance(r, dict):
            continue
        text = " ".join(str(r.get("text") or "").split())
        if not text:
            continue
        if len(text) > MAX_RULE_CHARS:
            raise ValueError(f"Keep each rule under {MAX_RULE_CHARS} characters")
        scope = str(r.get("applies_to") or "everything").lower()
        rid = re.sub(r"[^a-z0-9-]+", "-", str(r.get("id") or text[:30]).lower()).strip("-") or f"rule-{i}"
        while rid in seen:
            rid += "-2"
        seen.add(rid)
        rules.append({"id": rid, "text": text, "enabled": bool(r.get("enabled", True)),
                      "applies_to": scope if scope in SCOPES else "everything"})
    out = {"clip_length": {"min_seconds": lo, "max_seconds": hi}, "rules": rules}
    if data.get("updated_at"):
        out["updated_at"] = str(data["updated_at"])
    return out


def defaults() -> dict:
    return clean(yaml.safe_load(DEFAULTS_FILE.read_text()) or {})


# --- Reading ----------------------------------------------------------------------

def _last_good_path() -> Path:
    from .config import DATA_DIR

    return DATA_DIR / "universal_rules.json"


def _fetch() -> dict | None:
    url = os.environ.get("UNIVERSAL_RULES_URL", "").strip()
    token = os.environ.get("UNIVERSAL_RULES_TOKEN", "").strip()
    if not url or not token:
        return None
    import httpx

    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=8)
        r.raise_for_status()
        data = clean(r.json())
    except Exception as exc:
        log.warning("Universal rules unavailable from %s: %s", url, exc)
        return None
    try:
        path = _last_good_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except OSError:
        pass
    return data


def load() -> dict:
    """The rules in force: live copy, else last good copy, else defaults.
    Never raises."""
    global _cache
    now = time.monotonic()
    with _lock:
        if _cache is not None and now - _cache[0] < TTL:
            return _cache[1]
    data = _fetch()
    if data is None:
        try:
            data = clean(json.loads(_last_good_path().read_text()))
        except Exception:
            data = None
    if data is None:
        try:
            data = defaults()
        except Exception:
            log.exception("Universal rules defaults unreadable")
            data = {"clip_length": {"min_seconds": 20, "max_seconds": 45}, "rules": []}
    with _lock:
        _cache = (now, data)
    return data


def forget() -> None:
    global _cache
    with _lock:
        _cache = None


def render(kind: str, data: dict | None = None) -> str:
    """Enabled rules reaching ``kind`` ("screening", "content", "captions")
    as a bullet list, or ""."""
    data = data or load()
    scopes = _REACHES.get(kind, ("everything",))
    return "\n".join(f"- {r['text']}" for r in data["rules"]
                     if r["enabled"] and r["applies_to"] in scopes)


def clip_length(data: dict | None = None) -> tuple[int, int]:
    length = (data or load())["clip_length"]
    return int(length["min_seconds"]), int(length["max_seconds"])


def version(data: dict | None = None) -> str:
    """Changes whenever the rules do (for caches built on them)."""
    return json.dumps(data or load(), sort_keys=True)
