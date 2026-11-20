"""Small workspace documents every machine must agree on, kept in the database.

The brand, keywords and caption rules, the clip handbook and playbook, and the
clip engine's notes and rule proposals are files under the workspace. That's
fine while one machine runs everything. Once the hosted dashboard, the home
worker and the operator's Mac share a workspace, each would edit its own copy:
a rule added in the browser would never reach the engine on the worker.

With SHARED_WORKSPACE on, each document lives in the shared database
(``AppToken`` row ``doc:<workspace>:<key>``, no schema change) and its file is a
mirror, kept so the repo still shows the history. The database copy wins
whenever it exists; the file is the seed until the first save. Edits made to a
file by hand are NOT pushed automatically (file times aren't trustworthy:
``git checkout`` touches them and could silently undo a browser edit); run
``python run.py config push`` to send them, ``config status`` to compare.

With SHARED_WORKSPACE off, these functions read and write the file only,
exactly as before.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

from .config import WORKSPACE, env

log = logging.getLogger("shared_docs")

# Short keys: AppToken.name is String(40).
DOC_FILES = {
    "brand": ("config", "brand.yaml"),
    "keywords": ("config", "keywords.yaml"),
    "captions": ("config", "caption_style.yaml"),
    "handbook": ("config", "clip_handbook.yaml"),
    "playbook": ("config", "clip_playbook.yaml"),
    "notes": ("engine", "notes.jsonl"),
    "proposals": ("engine", "rule_proposals.json"),
    "onboarding": ("engine", "onboarding.json"),
    "obsources": ("engine", "onboarding_sources.json"),
}

_CACHE_TTL = 30.0
_cache: dict[str, tuple[float, str | None]] = {}
_cache_lock = threading.Lock()


def enabled() -> bool:
    return env("SHARED_WORKSPACE").strip().lower() in ("1", "true", "on", "yes")


def row_name(key: str) -> str:
    name = f"doc:{WORKSPACE}:{key}"
    if len(name) > 40:
        raise ValueError(f"workspace slug too long to share documents: {name!r}")
    return name


def default_path(key: str) -> Path:
    where, filename = DOC_FILES[key]
    if where == "config":
        from .config import CONFIG_DIR

        return CONFIG_DIR / filename
    from .clip_engine import engine_dir

    return engine_dir() / filename


# --- database ----------------------------------------------------------------------

def _db_get(key: str) -> str | None:
    """The database copy (cached briefly), or None when there is none. Raises
    when the database can't be read."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL:
            return hit[1]
    from .db import session_scope
    from .models import AppToken

    with session_scope(read_only=True) as session:
        row = session.get(AppToken, row_name(key))
        value = row.value if row is not None else None
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def _forget(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)


def _db_update(key: str, fn: Callable[[str | None], str | None],
               seed: Callable[[], str | None]) -> str | None:
    """Read-modify-write the row so two machines appending never lose an entry.
    ``fn`` gets the current text (the file's, ``seed``, when the row doesn't
    exist yet) and returns the new text, or None to leave it alone.

    A row lock can't guard a row that doesn't exist, so the first write is an
    insert; when another writer created the row first, the insert fails on
    the primary key and the locked update below takes over."""
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from .db import session_scope
    from .models import AppToken, utcnow

    name = row_name(key)
    with session_scope(read_only=True) as session:
        exists = session.get(AppToken, name) is not None
    if not exists:
        current = seed()
        new = fn(current)
        if new is None:
            return current
        try:
            with session_scope() as session:
                session.add(AppToken(name=name, value=new, updated_at=utcnow()))
            _forget(key)
            return new
        except IntegrityError:
            pass  # created meanwhile: update it under the lock instead

    with session_scope() as session:
        row = session.execute(
            select(AppToken).where(AppToken.name == name).with_for_update()
        ).scalar_one()
        new = fn(row.value)
        if new is None:
            return row.value
        row.value = new
        row.updated_at = utcnow()
    _forget(key)
    return new


# --- the API -----------------------------------------------------------------------

def _read_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_text(key: str, path: Path | None = None) -> str | None:
    """The document's text, or None when it doesn't exist anywhere. A database
    that can't be reached falls back to the file rather than failing the page."""
    path = path or default_path(key)
    if enabled():
        try:
            value = _db_get(key)
            if value is not None:
                return value
        except Exception as exc:
            log.warning("Could not read %s from the database (using %s): %s", key, path, exc)
    return _read_file(path)


def write_text(key: str, text: str, path: Path | None = None) -> None:
    """Save the document. With sharing on, the database write comes first and
    raises on failure: a save that only reached this machine's file would look
    successful while every other machine kept the old version."""
    path = path or default_path(key)
    if enabled():
        _db_update(key, lambda _current: text, seed=lambda: None)
    _write_file(path, text)


def update_text(key: str, fn: Callable[[str | None], str | None],
                path: Path | None = None) -> str | None:
    """Atomic read-modify-write across machines. Returns the resulting text."""
    path = path or default_path(key)
    if enabled():
        new = _db_update(key, fn, seed=lambda: _read_file(path))
        if new is not None:
            _write_file(path, new)
        return new
    current = _read_file(path)
    new = fn(current)
    if new is None:
        return current
    _write_file(path, new)
    return new


# --- operator tools ----------------------------------------------------------------

def status() -> list[dict]:
    """Per document: where the text comes from, and whether this machine's file
    matches the database copy."""
    out = []
    for key in DOC_FILES:
        path = default_path(key)
        file_text = _read_file(path)
        _forget(key)
        db_text = _db_get(key)
        out.append({"key": key, "file": str(path), "in_db": db_text is not None,
                    "on_disk": file_text is not None,
                    "same": db_text is None or file_text == db_text})
    return out


def push(key: str) -> bool:
    """Send this machine's file to the database (overwriting the shared copy).
    False when there's no file to send."""
    text = _read_file(default_path(key))
    if text is None:
        return False
    _db_update(key, lambda _current: text, seed=lambda: None)
    return True


def pull(key: str) -> bool:
    """Overwrite this machine's file with the database copy. False when there's
    no database copy."""
    _forget(key)
    text = _db_get(key)
    if text is None:
        return False
    _write_file(default_path(key), text)
    return True


__all__ = ["DOC_FILES", "enabled", "read_text", "write_text", "update_text",
           "status", "push", "pull", "default_path"]
