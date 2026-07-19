#!/usr/bin/env python3
"""Entry point for the Climate Clip Monitor.

Usage:
  python run.py dashboard          # local web UI at http://127.0.0.1:8321
  python run.py monitor            # one discovery pass over all channels
  python run.py monitor --loop     # poll forever at the configured interval
  python run.py score-visuals      # backfill vision scores for unscored candidates
  python run.py metrics            # snapshot Threads metrics for published posts
  python run.py comments           # sync + classify comments on own posts
  python run.py digest             # print the analytics digest to stdout
  python run.py cleanup            # apply the retention setting (never automatic)
  python run.py scheduler          # publish any scheduled posts that are due
  python run.py scheduler --loop   # keep publishing scheduled posts as they come due
  python run.py migrate-db         # copy local SQLite data into DATABASE_URL (Supabase)
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

    # reload watches the source tree so newly added routes/templates are picked
    # up without a manual restart. Disable with --no-reload for a stable run.
    uvicorn.run("app.web.main:app", host="127.0.0.1", port=args.port,
                log_level="info", reload=args.reload)


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


def cmd_score_visuals(args) -> None:
    """Backfill vision scores for stored candidates that don't have one yet
    (respects vision.min_relevance and the daily budget). Handy after enabling
    vision on an existing DB, or to score without waiting for a monitor pass."""
    from sqlalchemy import select

    from app import spend
    from app.config import load_settings
    from app.db import active_traits, init_db, session_scope, sync_traits_from_config
    from app.models import Candidate
    from app.ranking import load_trait_weights, trait_guidance_text
    from app.vision import apply_visual_score

    init_db()
    settings = load_settings()
    min_rel = settings.get("vision.min_relevance", 0.5)
    scored = skipped = 0
    with session_scope() as session:
        sync_traits_from_config(session)
        guidance = trait_guidance_text(load_trait_weights(session), settings)
        desirable, undesirable = active_traits(session)
        query = select(Candidate).where(
            Candidate.visual_score.is_(None),
            (Candidate.relevance_score.is_(None)) | (Candidate.relevance_score >= min_rel),
        ).order_by(Candidate.relevance_score.desc().nullslast())
        if args.limit:
            query = query.limit(args.limit)
        for c in session.execute(query).scalars().all():
            if not spend.within_budget():
                log.info("Daily budget reached (${:.2f}); stopping.", spend.today_spend())
                break
            result = apply_visual_score(c, settings, force=True, learned_guidance=guidance,
                                        desirable=desirable, undesirable=undesirable)
            if result is None:
                skipped += 1
            else:
                scored += 1
                session.commit()  # persist incrementally
    print(f"Vision-scored {scored} candidates, skipped {skipped}. "
          f"Spent ${spend.today_spend():.2f} of ${spend.daily_budget():.2f} today.")


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


def cmd_scheduler(args) -> None:
    """Publish due scheduled posts. Runs one pass, or loops when --loop is set.
    The dashboard runs this automatically; use this for headless operation."""
    from app.db import init_db
    from app.scheduler import run_due_posts

    init_db()
    if not args.loop:
        n = run_due_posts()
        print(f"Published {n} due post(s)")
        return
    log.info("Scheduler loop started (every %ds). Ctrl-C to stop.", args.interval)
    while True:
        try:
            n = run_due_posts()
            if n:
                log.info("Published %d due post(s)", n)
        except Exception:
            log.exception("Scheduler pass failed; will retry")
        time.sleep(args.interval)


def cmd_migrate_db(_args) -> None:
    """Copy local SQLite data into the DATABASE_URL target (Supabase Postgres)."""
    from app.migrate import migrate_sqlite_to_target

    counts = migrate_sqlite_to_target()
    print("Migration complete:")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")


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
    p.add_argument("--no-reload", dest="reload", action="store_false",
                   help="disable auto-reload on source changes")
    p.set_defaults(func=cmd_dashboard, reload=True)

    p = sub.add_parser("monitor", help="run channel discovery")
    p.add_argument("--loop", action="store_true", help="poll forever at the configured interval")
    p.add_argument("--lookback", type=int, default=None, metavar="DAYS",
                   help="scan this many days back instead of since-last-check (backfill)")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("scheduler", help="publish due scheduled posts")
    p.add_argument("--loop", action="store_true", help="keep running and publish posts as they come due")
    p.add_argument("--interval", type=int, default=60, help="seconds between checks in --loop mode")
    p.set_defaults(func=cmd_scheduler)

    sub.add_parser("migrate-db", help="copy local SQLite data into DATABASE_URL (e.g. Supabase)").set_defaults(func=cmd_migrate_db)
    p = sub.add_parser("score-visuals", help="backfill vision scores for unscored candidates")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="score at most N candidates this run")
    p.set_defaults(func=cmd_score_visuals)

    sub.add_parser("metrics", help="snapshot Threads post metrics").set_defaults(func=cmd_metrics)
    sub.add_parser("comments", help="sync and classify comments on own posts").set_defaults(func=cmd_comments)
    sub.add_parser("digest", help="print the analytics digest").set_defaults(func=cmd_digest)
    sub.add_parser("cleanup", help="apply retention setting to downloaded segments").set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
