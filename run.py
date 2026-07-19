#!/usr/bin/env python3
"""Entry point for the Climate Clip Monitor.

Usage:
  python run.py dashboard          # local web UI at http://127.0.0.1:8321
  python run.py monitor            # one discovery pass over all channels
  python run.py monitor --loop     # poll forever at the configured interval
  python run.py metrics            # snapshot Threads metrics for published posts
  python run.py comments           # sync + classify comments on own posts
  python run.py digest             # print the analytics digest to stdout
  python run.py cleanup            # apply the retention setting (never automatic)
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("run")


def cmd_dashboard(args) -> None:
    import uvicorn

    uvicorn.run("app.web.main:app", host="127.0.0.1", port=args.port, log_level="info")


def cmd_monitor(args) -> None:
    from app.config import load_settings
    from app.db import init_db
    from app.monitor import run_monitor_once

    init_db()
    if not args.loop:
        run_monitor_once(args.lookback)
        return
    interval_min = load_settings().get("monitor.poll_interval_minutes", 240)
    log.info("Monitoring every %d minutes. Ctrl-C to stop.", interval_min)
    while True:
        try:
            run_monitor_once()
        except Exception:
            log.exception("Monitor pass failed; will retry next interval")
        time.sleep(interval_min * 60)


def cmd_metrics(_args) -> None:
    from app.analytics import snapshot_metrics
    from app.db import init_db, session_scope

    init_db()
    with session_scope() as session:
        n = snapshot_metrics(session)
    print(f"Snapshots taken: {n}")


def cmd_comments(_args) -> None:
    from app.db import init_db, session_scope
    from app.engagement import sync_comments

    init_db()
    with session_scope() as session:
        result = sync_comments(session)
    print(f"New comments: {result['new_comments']}, drafts: {result['drafts']}")


def cmd_digest(_args) -> None:
    from app.analytics import generate_report
    from app.db import init_db, session_scope

    init_db()
    with session_scope() as session:
        report = generate_report(session)
    print(report["digest"] or "(no published posts yet)")


def cmd_cleanup(_args) -> None:
    """Prune full segments older than the retention setting. Only ever runs
    when the operator invokes this command; nothing auto-deletes."""
    from pathlib import Path

    from app.config import ROOT, load_settings

    settings = load_settings()
    retention = settings.get("storage.retention", "keep")
    if retention == "keep":
        print("storage.retention is 'keep' — nothing to prune. Set it to a number of days to enable.")
        return
    cutoff = time.time() - int(retention) * 86400
    root = ROOT / settings.get("storage.download_dir", "data/videos")
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            print(f"Removing {path}")
            path.unlink()
            removed += 1
    print(f"Removed {removed} files older than {retention} days.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dashboard", help="run the local web dashboard")
    p.add_argument("--port", type=int, default=8321)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("monitor", help="run channel discovery")
    p.add_argument("--loop", action="store_true", help="poll forever at the configured interval")
    p.add_argument("--lookback", type=int, default=None, metavar="DAYS",
                   help="scan this many days back instead of since-last-check (backfill)")
    p.set_defaults(func=cmd_monitor)

    sub.add_parser("metrics", help="snapshot Threads post metrics").set_defaults(func=cmd_metrics)
    sub.add_parser("comments", help="sync and classify comments on own posts").set_defaults(func=cmd_comments)
    sub.add_parser("digest", help="print the analytics digest").set_defaults(func=cmd_digest)
    sub.add_parser("cleanup", help="apply retention setting to downloaded segments").set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
