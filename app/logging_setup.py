"""Shared logging setup: console + rotating file under ``data/logs``.

The dashboard runs uvicorn with ``--reload``, so the app — and the scheduler
thread started alongside it — lives in a spawned worker process that never
executes ``run.py``. That worker's root logger had no handlers at all, which
meant every scheduler ``log.info`` (window ticks, publishes, recoveries) was
discarded and exceptions fell through to logging's bare last-resort stderr
handler. Nothing survived on disk, so a posting window that came and went
without publishing left no evidence to diagnose afterwards.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
LOG_FILE = LOG_DIR / "app.log"

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Send root-logger output to the console and a rotating file. Idempotent.

    Safe to call from every entry point (``run.py`` and the uvicorn worker);
    repeat calls and a read-only ``data/`` are both no-ops rather than errors.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)

    if not any(not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)

    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # A few days of ticks is plenty to explain a missed window, and the cap
        # keeps an always-on dashboard from growing the log without bound.
        handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    except OSError as exc:
        root.warning("File logging disabled (%s): %s", LOG_FILE, exc)
