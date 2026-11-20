"""Small shared helpers for the clip engine: daemon-thread fan-out, atomic
file writes, hashing.

Daemon threads, not ``ThreadPoolExecutor``: executor workers are non-daemon,
and CPython joins them at exit, so a queue of engine work would hold a
``--reload`` restart (or Ctrl-C) hostage until every job finished. Every other
long-lived thread in this app is a daemon for the same reason.

Atomic writes use a tmp name unique to the process and thread, then
``os.replace``: the dashboard and a ``run.py clipbench`` process share these
directories, and a fixed tmp name lets one writer rename the other's file away.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Iterable


class Cancelled(RuntimeError):
    """Raised by ``check`` when a run was canceled or ran out of time."""


def check(cancel: threading.Event | None = None, deadline: float | None = None) -> None:
    """Raise ``Cancelled`` when ``cancel`` is set or the monotonic ``deadline``
    has passed. Called between stages and between items."""
    import time

    if cancel is not None and cancel.is_set():
        raise Cancelled("cancelled")
    if deadline is not None and time.monotonic() > deadline:
        raise Cancelled("ran out of time")


def parallel_map(fn: Callable[[Any], Any], items: Iterable[Any], *, workers: int = 4,
                 cancel: threading.Event | None = None,
                 name: str = "clip-engine") -> list[Any]:
    """``[fn(x) for x in items]`` on up to ``workers`` daemon threads, in order.

    An exception from ``fn`` is returned in that item's slot (not raised), so
    one failed render or critique never sinks its siblings — callers decide
    what a failure means. Items not started before ``cancel`` is set come back
    as ``Cancelled`` instances.
    """
    items = list(items)
    results: list[Any] = [None] * len(items)
    if not items:
        return results
    todo: queue.Queue = queue.Queue()
    for i, item in enumerate(items):
        todo.put((i, item))

    def worker() -> None:
        while True:
            try:
                i, item = todo.get_nowait()
            except queue.Empty:
                return
            if cancel is not None and cancel.is_set():
                results[i] = Cancelled("cancelled")
                continue
            try:
                results[i] = fn(item)
            except BaseException as exc:  # noqa: BLE001 - handed back to the caller
                results[i] = exc

    threads = [threading.Thread(target=worker, daemon=True, name=f"{name}-{n}")
               for n in range(max(1, min(workers, len(items))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def run_in_thread(fn: Callable[[], Any], *, name: str = "clip-engine-bg") -> Callable[[], Any]:
    """Start ``fn`` on a daemon thread now; returns ``wait()`` which joins it
    and returns its result, re-raising its exception."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True, name=name)
    t.start()

    def wait() -> Any:
        t.join()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    return wait


def _tmp_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def atomic_write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path, data: Any) -> Path:
    return atomic_write_text(path, json.dumps(data, indent=1, sort_keys=False, default=str))


def atomic_copy(src: Path, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(dest)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    return dest


def read_json(path: Path, default: Any = None) -> Any:
    """The parsed file, or ``default`` on any error (missing, half-written, corrupt)."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def sha1(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def touch(path: Path) -> None:
    """Refresh a cached file's mtime so age-based pruning keeps live caches."""
    try:
        os.utime(path, None)
    except OSError:
        pass
