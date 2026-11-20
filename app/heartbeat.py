"""Heartbeats: each long-running worker says "alive at ..." in the database.

The admin site (app/admin) shows whether the scheduler and the home Mac's
worker are running. Without a heartbeat, a quiet night and a dead worker
look the same. Each beat is one ``AppToken`` row per role and machine,
overwritten each time, carrying the machine name, the code version, and
that machine's LLM spend today (the spend ledger is a file on each machine,
so this is how it reaches the admin site).

Beats are throttled to one a minute per role. They never raise: a worker
must never stop because it couldn't report that it's alive.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time

log = logging.getLogger("heartbeat")

PREFIX = "hb:"
THROTTLE_SECONDS = 60.0
_last: dict[str, float] = {}
_lock = threading.Lock()


def _host() -> str:
    return (os.environ.get("FLY_MACHINE_ID") or socket.gethostname() or "?")[:20]


def row_name(role: str, host: str | None = None) -> str:
    return f"{PREFIX}{role[:12]}:{(host or _host())}"[:40]


def beat(role: str, **extra) -> None:
    """Record that ``role`` (e.g. "scheduler", "home-worker") is alive on
    this machine."""
    now = time.monotonic()
    with _lock:
        if now - _last.get(role, -1e9) < THROTTLE_SECONDS:
            return
        _last[role] = now
    try:
        from . import spend
        from .db import session_scope
        from .models import AppToken, utcnow

        payload = {"role": role, "host": _host(), "at": utcnow().isoformat(),
                   "version": (os.environ.get("FLY_IMAGE_REF") or os.environ.get("GIT_SHA") or "")[-40:],
                   "region": os.environ.get("FLY_REGION", "home"),
                   "spend_today": round(spend.today_spend(), 4), **extra}
        with session_scope() as session:
            name = row_name(role)
            row = session.get(AppToken, name)
            if row is None:
                row = AppToken(name=name)
                session.add(row)
            row.value = json.dumps(payload)
            row.updated_at = utcnow()
    except Exception as exc:
        log.debug("Heartbeat for %s not recorded: %s", role, exc)
