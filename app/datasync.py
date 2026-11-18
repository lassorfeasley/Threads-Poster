"""Share the workspace's working files between machines through Storage.

The operator's Mac, the hosted dashboard and the home worker each have their
own disk but one database. Database rows name files (transcripts, clips,
thumbnails, clip-engine previews) that only the machine that made them has.
This module closes that gap with two halves:

- ``push_sweep`` uploads every new or changed file under the mirrored folders
  to the Storage bucket at ``<workspace>/<path under data/>``. A background
  thread (``start_mirror_thread``) runs it every minute; ``python run.py
  datasync push`` runs it once (the first run backfills).
- ``ensure_file`` brings a missing file back from Storage before a caller
  reads it. Callers keep their ordinary "is it on disk" checks.

Clip keys match the clip archive's (``offload.clip_object_key``), so a clip
mirrored here is already "archived" when the offload sweep reaches it. Source
videos are not mirrored (they have their own keys and restore path, see
offload.py), and nor are caches any machine can rebuild (waveforms, frames,
outro renders, clip-engine proxies).

Off unless DATA_MIRROR is set (``1``/``true``/``on``), so a machine that
doesn't share its disk behaves exactly as before.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, WORKSPACE, env

log = logging.getLogger("datasync")

# Folders under data/ whose files other machines need. Order matters only for
# the first backfill: small, most-needed files first.
MIRRORED = ("transcripts", "thumbs", "brand", "clip_engine/proposals", "clips")

# A file still being written must not be uploaded half-done: wait until it has
# sat unchanged this long.
SETTLE_SECONDS = 30.0
# Above this, upload over the resumable endpoint (and its 6 MB chunks).
_SIMPLE_UPLOAD_MAX = 6 * 1024 * 1024
_MISS_TTL = 300.0
_MANIFEST = ".datasync.json"


def enabled() -> bool:
    return env("DATA_MIRROR").strip().lower() in ("1", "true", "on", "yes")


def _bucket() -> str:
    from .offload import _bucket as bucket

    return bucket()


def _data_dir() -> Path:
    return DATA_DIR.resolve()


def relative(path: str | Path) -> str | None:
    """``path`` relative to data/, when it's inside a mirrored folder."""
    try:
        rel = Path(path).expanduser().resolve().relative_to(_data_dir()).as_posix()
    except (ValueError, OSError):
        return None
    if any(rel.startswith(f"{folder}/") for folder in MIRRORED):
        name = rel.rsplit("/", 1)[-1]
        if not name.startswith(".") and not name.endswith((".part", ".lock", ".tmp")):
            return rel
    return None


def key_for(path: str | Path) -> str | None:
    rel = relative(path)
    return f"{WORKSPACE}/{rel}" if rel else None


# --- pulling ---------------------------------------------------------------------

_misses: dict[str, float] = {}
_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def ensure_file(path: str | Path | None) -> bool:
    """True when ``path`` is on disk, fetching it from Storage first if it's a
    mirrored file this machine doesn't have. Never raises; a file that can't
    be fetched leaves the caller's own missing-file handling in charge. A miss
    is remembered for a few minutes so a page full of absent thumbnails doesn't
    ask Storage again on every request."""
    if not path:
        return False
    p = Path(path).expanduser()
    if p.is_file():
        return True
    if not enabled():
        return False
    key = key_for(p)
    if key is None:
        return False
    now = time.monotonic()
    if now - _misses.get(key, -1e9) < _MISS_TTL:
        return False
    try:
        with _lock_for(key):
            if p.is_file():
                return True
            from .offload import remote_size
            from .storage_supabase import download_to_file

            size = remote_size(key)
            if size is None:
                _misses[key] = now
                return False
            p.parent.mkdir(parents=True, exist_ok=True)
            download_to_file(_bucket(), key, p, expected_size=size)
            # Pulled files are already in Storage: don't push them back.
            _Manifest.load().note(relative(p) or "", p).save()
            log.info("Fetched %s from storage", key)
            return True
    except Exception as exc:
        log.warning("Could not fetch %s: %s", key, exc)
        _misses[key] = now
        return False


def ensure_text(path: str | Path | None) -> str | None:
    """The file's text after ``ensure_file``, or None when it isn't available."""
    if not ensure_file(path):
        return None
    try:
        return Path(path).read_text()  # type: ignore[arg-type]
    except OSError:
        return None


# --- pushing ---------------------------------------------------------------------

class _Manifest:
    """What this machine has already uploaded (or pulled): path -> [size,
    mtime_ns]. Local to the machine; losing it only costs a re-upload."""

    _guard = threading.Lock()

    def __init__(self, entries: dict[str, list[int]]):
        self.entries = entries

    @classmethod
    def path(cls) -> Path:
        return DATA_DIR / _MANIFEST

    @classmethod
    def load(cls) -> "_Manifest":
        try:
            return cls(json.loads(cls.path().read_text()))
        except (OSError, ValueError):
            return cls({})

    def current(self, rel: str, st: os.stat_result) -> bool:
        return self.entries.get(rel) == [st.st_size, st.st_mtime_ns]

    def note(self, rel: str, p: Path) -> "_Manifest":
        if rel:
            st = p.stat()
            self.entries[rel] = [st.st_size, st.st_mtime_ns]
        return self

    def save(self) -> None:
        with self._guard:
            # Merge with what's on disk: a pull and a sweep may both write.
            merged = {**_Manifest.load().entries, **self.entries}
            tmp = self.path().with_name(f"{_MANIFEST}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(merged, separators=(",", ":")))
            os.replace(tmp, self.path())
            self.entries = merged


@dataclass
class PushResult:
    uploaded: int = 0
    uploaded_bytes: int = 0
    failed: int = 0
    pending: list[tuple[str, int]] = field(default_factory=list)


def _candidates(manifest: _Manifest, now: float):
    root = DATA_DIR
    for folder in MIRRORED:
        base = root / folder
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            rel = relative(p)
            if rel is None or not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if manifest.current(rel, st) or now - st.st_mtime < SETTLE_SECONDS:
                continue
            yield rel, p, st


def upload(p: Path, key: str) -> None:
    from .storage_supabase import ensure_bucket, upload_file_resumable

    bucket = _bucket()
    ensure_bucket(bucket)
    content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    if p.stat().st_size > _SIMPLE_UPLOAD_MAX:
        upload_file_resumable(p, bucket, key, content_type=content_type)
        return
    import httpx

    from .storage_supabase import _auth_headers, _object_url

    r = httpx.post(_object_url("object", bucket, key), content=p.read_bytes(), timeout=120,
                   headers={**_auth_headers(), "x-upsert": "true",
                            "Content-Type": content_type})
    if r.status_code >= 300:
        raise RuntimeError(f"upload refused ({r.status_code}): {r.text[:200]}")


def push_sweep(*, dry_run: bool = False, limit: int | None = None) -> PushResult:
    """Upload new or changed mirrored files. Per-file failures are counted and
    retried on the next sweep."""
    manifest = _Manifest.load()
    result = PushResult()
    now = time.time()
    done = 0
    for rel, p, st in _candidates(manifest, now):
        if limit is not None and done >= limit:
            break
        done += 1
        result.pending.append((rel, st.st_size))
        if dry_run:
            continue
        try:
            upload(p, f"{WORKSPACE}/{rel}")
        except Exception as exc:
            result.failed += 1
            log.warning("Could not upload %s: %s", rel, exc)
            continue
        manifest.entries[rel] = [st.st_size, st.st_mtime_ns]
        _misses.pop(f"{WORKSPACE}/{rel}", None)
        result.uploaded += 1
        result.uploaded_bytes += st.st_size
        if result.uploaded % 25 == 0:
            manifest.save()  # a long backfill keeps its progress
    if not dry_run:
        manifest.save()
    return result


_thread: threading.Thread | None = None
_sweep_lock = threading.Lock()


def run_push_once(**kwargs) -> PushResult | None:
    """One sweep, unless another is already running in this process."""
    if not _sweep_lock.acquire(blocking=False):
        return None
    try:
        return push_sweep(**kwargs)
    finally:
        _sweep_lock.release()


def start_mirror_thread(interval: float = 60.0) -> None:
    """Background push for a long-running process. Idempotent; a no-op unless
    DATA_MIRROR is on."""
    global _thread
    if not enabled() or (_thread and _thread.is_alive()):
        return

    def _loop() -> None:
        time.sleep(30)  # let the process finish booting first
        while True:
            try:
                res = run_push_once()
                if res and (res.uploaded or res.failed):
                    log.info("Data mirror: %d uploaded (%.1f MB), %d failed",
                             res.uploaded, res.uploaded_bytes / 1e6, res.failed)
            except Exception:
                log.exception("Data mirror sweep failed")
            time.sleep(interval)

    _thread = threading.Thread(target=_loop, daemon=True, name="data-mirror")
    _thread.start()


__all__ = ["MIRRORED", "enabled", "relative", "key_for", "ensure_file", "ensure_text",
           "push_sweep", "run_push_once", "start_mirror_thread", "PushResult"]
