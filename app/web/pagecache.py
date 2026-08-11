"""Warm read-through cache for the pages that read more than they change.

The database is remote and this is a single-operator tool: the whole working
set is a few MB, but every read of it crosses a slow link, and the list pages
re-read the same rows on every visit. Keeping the built result here turns a
page render into a dict lookup, and a background thread refreshes the entries
somebody is actually using so the copy is rarely cold.

What belongs here: whole-page datasets assembled from detached ORM rows by a
GET handler. Entries are shared across requests and threads, so a cached
dataset is strictly read-only — never hand one to a session or mutate it.

Staleness is bounded from both ends: every write calls ``invalidate()`` (a
middleware catches the HTTP ones, ``_in_background`` the worker threads), and
``TTL_SECONDS`` backstops anything that changed the database from outside this
process.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("pagecache")

# Correctness comes from invalidate(), not from expiry: every write this app
# makes goes through it. The TTL is the backstop for changes made outside the
# web process (a CLI run against the same database), so it can be generous.
TTL_SECONDS = 600.0
# Past this, nobody is using the page and it isn't worth keeping warm on a link
# the downloader and monitor also need.
ACTIVE_WINDOW_SECONDS = 3600.0
REFRESH_INTERVAL_SECONDS = 20.0
# After a write, wait for the operator to stop writing before rebuilding — a
# run of approvals should cost one rebuild, not one per click.
SETTLE_SECONDS = 3.0


@dataclass
class _Entry:
    loader: Callable[[], Any]
    background: bool = True    # may the refresher build this on spec?
    volatile: bool = True      # does any write make this wrong?
    lock: threading.Lock = field(default_factory=threading.Lock)
    value: Any = None
    loaded_at: float = 0.0     # 0 = never loaded, or invalidated by a write
    last_read: float = 0.0


_entries: dict[str, _Entry] = {}
_thread: threading.Thread | None = None
_wake = threading.Event()
# Bumped by every write, so a rebuild that started before it can tell that what
# it just read is already behind.
_version = 0


def register(name: str, loader: Callable[[], Any], *, background: bool = True,
             volatile: bool = True) -> None:
    """Declare a dataset and how to (re)build it.

    ``background=False`` for a dataset too expensive to build on the chance
    somebody wants it: it's built on demand instead, which still saves a page
    load every time but never costs the downloader and monitor a slow link
    they also need.

    ``volatile=False`` for a dataset most writes have no bearing on. It then
    lives by its TTL, and whatever does change it calls ``drop`` — worth the
    extra thought only where rebuilding is expensive.
    """
    _entries[name] = _Entry(loader=loader, background=background, volatile=volatile)


def _is_fresh(entry: _Entry, now: float) -> bool:
    return bool(entry.loaded_at) and (now - entry.loaded_at) < TTL_SECONDS


def read(name: str) -> Any:
    """The dataset, rebuilding it first if the copy is missing or stale."""
    entry = _entries[name]
    entry.last_read = time.monotonic()
    if _is_fresh(entry, entry.last_read):
        return entry.value
    return _rebuild(name, entry)


def peek(name: str) -> Any:
    """The dataset if it's already built, else None — for a page that would
    rather draw a placeholder than wait for it."""
    entry = _entries[name]
    now = time.monotonic()
    if not _is_fresh(entry, now):
        return None
    entry.last_read = now
    return entry.value


def read_or_last(name: str, fallback: Any = None) -> Any:
    """Like ``read``, but a failed rebuild falls back to the last known value.

    For the things drawn on every page, where one unlucky database moment must
    not take down every render with it.
    """
    try:
        return read(name)
    except Exception:
        log.warning("Serving the last known %s: rebuild failed", name, exc_info=True)
        value = _entries[name].value
        return fallback if value is None else value


def is_warm(name: str) -> bool:
    """Would reading this dataset be free right now?"""
    entry = _entries.get(name)
    return bool(entry) and _is_fresh(entry, time.monotonic())


def _rebuild(name: str, entry: _Entry) -> Any:
    # One build at a time per dataset: two tabs asking at once should wait on
    # a single slow read, not start two of them.
    with entry.lock:
        now = time.monotonic()
        if _is_fresh(entry, now):
            return entry.value
        started, started_version = time.perf_counter(), _version
        entry.value = entry.loader()
        # If a write landed while this was loading, the rows it read may already
        # be out of date. Still hand the value to the caller who's waiting on it,
        # but leave the entry stale so it's rebuilt before anyone else sees it.
        entry.loaded_at = time.monotonic() if _version == started_version else 0.0
        log.debug("Rebuilt %s in %.0f ms", name, (time.perf_counter() - started) * 1000)
        return entry.value


def drop(name: str) -> None:
    """Force one dataset to be rebuilt before it's served again."""
    global _version
    _version += 1
    _entries[name].loaded_at = 0.0
    _wake.set()


def invalidate() -> None:
    """Something was written: every dataset has to be rebuilt before it's served.

    Deliberately blunt. Working out which pages a given write touches would be
    a second source of truth to keep in sync, and the refresher makes the cost
    of over-invalidating small — except where it doesn't, which is what
    ``volatile=False`` is for.
    """
    global _version
    _version += 1
    for entry in _entries.values():
        if entry.volatile:
            entry.loaded_at = 0.0
    # Start rebuilding now rather than at the next tick: the operator is mid-flow
    # and about to look at one of these pages.
    _wake.set()


def start_refresher() -> None:
    """Rebuild in-use datasets off the request path. Idempotent."""
    global _thread
    if _thread and _thread.is_alive():
        return

    def _loop() -> None:
        while True:
            now = time.monotonic()
            for name, entry in list(_entries.items()):
                # A dataset nobody has asked for yet is still worth building,
                # so the first visit after a restart isn't the slow one. After
                # that, only keep warm what's in use.
                idle = (entry.value is not None
                        and now - entry.last_read > ACTIVE_WINDOW_SECONDS)
                if not entry.background or idle or _is_fresh(entry, now):
                    continue
                try:
                    _rebuild(name, entry)
                except Exception:     # a bad read must not kill the refresher
                    log.exception("Could not refresh %s", name)
            _wake.wait(REFRESH_INTERVAL_SECONDS)
            while _wake.is_set():     # woken by a write; let the burst finish
                _wake.clear()
                time.sleep(SETTLE_SECONDS)

    _thread = threading.Thread(target=_loop, daemon=True, name="pagecache-refresher")
    _thread.start()
