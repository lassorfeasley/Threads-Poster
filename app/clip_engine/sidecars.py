"""Per-proposal sidecar files: the preview render and the critic's notes.
Spec §4.14.

A suggestion card wants two things the ledger row doesn't hold: a playable
preview of exactly what was proposed, and what the cold critic said about it.
Neither belongs in the database (no schema change; a render is megabytes and
only exists on the machine that made it), so each proposal gets a pair of
files: ``engine_dir()/proposals/<id>.json`` and ``<id>.mp4``.

They are read on every /cut and /video render and on every 3 s status poll,
and written right after the proposals commit, so:

- every reader tolerates a missing, half-written or corrupt file (``None``),
- every write is atomic (tmp + ``os.replace``), so a poll never sees half a
  file,
- nothing here raises: a broken sidecar costs a card its preview, never the
  page.

The JSON records the exact mp4 it was saved with (name and size), and
``preview_path`` only hands out an mp4 that matches it. A proposal id reused
by a different database (a dev SQLite file, a reset) must never play someone
else's footage, so saving without a render also deletes any old mp4 under
that id.

The preview is a hardlink to the engine's render when the filesystem allows
it (renders are written once and replaced, never edited in place), so keeping
it costs no extra disk on a volume that is already 90% full.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import engine_dir, util

log = logging.getLogger("clip_engine.sidecars")

# Leftover tmp files from a crashed write are swept after this long.
_TMP_GRACE_SECONDS = 86400


def directory() -> Path:
    """``engine_dir()/proposals``, created on first use."""
    path = engine_dir() / "proposals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paths(proposal_id: Any) -> tuple[Path, Path] | None:
    """``(<id>.json, <id>.mp4)``, or None for anything that isn't a positive
    integer id (which also keeps a crafted id from leaving the directory)."""
    try:
        pid = int(proposal_id)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or str(pid) != str(proposal_id).strip():
        return None
    base = directory()
    return base / f"{pid}.json", base / f"{pid}.mp4"


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("Could not remove %s: %s", path, exc)


def _place(src: Path, dest: Path) -> None:
    """Put ``src`` at ``dest`` atomically: a hardlink when the filesystem
    allows it, else a copy."""
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.link.tmp")
    try:
        _unlink(tmp)
        os.link(src, tmp)
        os.replace(tmp, dest)
        return
    except OSError:
        _unlink(tmp)
    util.atomic_copy(src, dest)


def save(proposal_id: int, *, meta: dict, render: Path | str | None) -> None:
    """Write the sidecar for one proposal: the preview mp4 (when ``render``
    is a real file) and ``meta`` as JSON with ``proposal_id``, ``created_at``
    (unless given) and ``preview_file``. Never raises."""
    try:
        paths = _paths(proposal_id)
        if paths is None:
            log.warning("Not saving a sidecar for proposal id %r", proposal_id)
            return
        json_path, mp4_path = paths
        data = dict(meta or {})
        data["proposal_id"] = int(proposal_id)
        data.setdefault("created_at", util.utcnow_iso())
        preview = None
        src = Path(render) if render else None
        if src is not None and src.is_file() and src.stat().st_size > 0:
            try:
                _place(src, mp4_path)
                preview = {"name": mp4_path.name, "bytes": mp4_path.stat().st_size}
            except OSError as exc:
                log.warning("Could not keep the preview for proposal %s: %s", proposal_id, exc)
        elif render:
            log.info("No preview for proposal %s: %s is missing or empty", proposal_id, render)
        if preview is None:
            _unlink(mp4_path)
        data["preview_file"] = preview
        util.atomic_write_json(json_path, data)
    except Exception as exc:
        log.warning("Could not save the sidecar for proposal %s: %s", proposal_id, exc)


def _fetch(path: Path) -> bool:
    """On disk, or brought back from the shared data mirror (app/datasync.py)."""
    from ..datasync import ensure_file

    return ensure_file(path)


def load(proposal_id: int) -> dict | None:
    """The sidecar's JSON, or None (no sidecar, or it can't be read). Never
    raises."""
    try:
        paths = _paths(proposal_id)
        if paths is None:
            return None
        _fetch(paths[0])
        data = util.read_json(paths[0])
        return data if isinstance(data, dict) else None
    except Exception as exc:
        log.warning("Could not read the sidecar for proposal %s: %s", proposal_id, exc)
        return None


def preview_path(proposal_id: int) -> Path | None:
    """The preview mp4, only when the sidecar names it and the file on disk
    is the one it was saved with. Never raises."""
    try:
        paths = _paths(proposal_id)
        if paths is None:
            return None
        meta = load(proposal_id)
        info = meta.get("preview_file") if meta else None
        if not isinstance(info, dict) or info.get("name") != paths[1].name:
            return None
        if not _fetch(paths[1]):
            return None
        size = paths[1].stat().st_size
        if size <= 0 or int(info.get("bytes", size)) != size:
            return None
        return paths[1]
    except Exception as exc:
        log.warning("Could not check the preview for proposal %s: %s", proposal_id, exc)
        return None


def prune(max_age_days: int = 45) -> int:
    """Delete sidecars untouched for ``max_age_days`` (a pair goes together,
    by its newest file) and stale tmp files. Returns the number of files
    removed. Never raises."""
    removed = 0
    try:
        base = directory()
        now = time.time()
        cutoff = now - max(0.0, float(max_age_days)) * 86400
        groups: dict[str, list[tuple[Path, float]]] = {}
        for path in base.iterdir():
            try:
                if not path.is_file():
                    continue
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                if mtime < now - _TMP_GRACE_SECONDS:
                    _unlink(path)
                    removed += 0 if path.exists() else 1
                continue
            if path.suffix in (".json", ".mp4"):
                groups.setdefault(path.stem, []).append((path, mtime))
        for files in groups.values():
            if max(m for _, m in files) >= cutoff:
                continue
            for path, _ in files:
                _unlink(path)
                removed += 0 if path.exists() else 1
    except Exception as exc:
        log.warning("Pruning proposal sidecars failed: %s", exc)
    if removed:
        log.info("Pruned %d proposal sidecar file(s)", removed)
    return removed
