#!/usr/bin/env python3
"""Entry point for the clip monitor & poster.

Every command runs against ONE workspace (an isolated environment under
workspaces/<slug>/ with its own config, data, database, and Meta account).
Select it with --workspace or the WORKSPACE env var; the default is "climate".
`dashboard` with no --workspace supervises ALL enabled workspaces at once.

Usage:
  python run.py dashboard          # supervisor: one dashboard per enabled workspace
  python run.py dashboard --workspace climate   # a single workspace's dashboard
  python run.py workspace list     # show the workspace registry
  python run.py workspace create <slug>         # scaffold a new workspace
  python run.py monitor            # one discovery pass over all channels
  python run.py monitor --loop     # poll forever at the configured interval
  python run.py score-visuals      # backfill vision scores for unscored candidates
  python run.py annotate-posts     # backfill footage traits for published posts
  python run.py backfill-post-times # restate post weekday/hour in the scheduler timezone
  python run.py backfill-calendar-names  # short calendar labels for old titled cuts
  python run.py backfill-categories # auto-tag programming categories for untagged videos
  python run.py metrics            # snapshot Threads metrics for published posts
  python run.py comments           # sync comments on own posts
  python run.py digest             # write a fresh analytics digest (one LLM call)
  python run.py cleanup            # apply the retention setting (never automatic)
  python run.py scheduler          # one adaptive-scheduler tick (windows + metrics)
  python run.py scheduler --loop   # keep the adaptive scheduler running
  python run.py migrate-db         # copy local SQLite data into DATABASE_URL (Supabase)
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# IMPORTANT: no ``app.*`` import may happen at module scope. Importing
# ``app.config`` resolves WORKSPACE from the environment, so the --workspace
# flag must be applied to os.environ first (see main()). Logging setup is
# deferred for the same reason: the log file lives in the workspace data tree.
log = logging.getLogger("run")

_REPO_ROOT = Path(__file__).resolve().parent
_REGISTRY_FILE = _REPO_ROOT / "workspaces.yaml"


# --- Workspace registry (parsed locally: must work on a fresh clone and in the
# --- supervisor, neither of which should import app.config) -------------------

def _load_registry() -> list[dict]:
    """Entries from workspaces.yaml as [{slug, label, port, accent, enabled}]."""
    import yaml

    if not _REGISTRY_FILE.exists():
        return []
    data = yaml.safe_load(_REGISTRY_FILE.read_text()) or {}
    out = []
    for item in data.get("workspaces", []) or []:
        if isinstance(item, dict) and str(item.get("slug") or "").strip():
            out.append({
                "slug": str(item["slug"]).strip(),
                "label": str(item.get("label") or item["slug"]),
                "port": int(item.get("port") or 8321),
                "accent": str(item.get("accent") or ""),
                "enabled": bool(item.get("enabled", True)),
            })
    return out


def cmd_dashboard(args) -> None:
    if args.workspace:
        _run_single_dashboard(args)
    else:
        _run_supervisor(args)


def _run_single_dashboard(args) -> None:
    import uvicorn

    from app.config import ROOT, WORKSPACE, current_workspace

    port = args.port or current_workspace()["port"]
    log.info("Dashboard for workspace %r on http://127.0.0.1:%d", WORKSPACE, port)
    # reload watches only ``app/`` so newly added routes/templates are picked
    # up without a manual restart — never ``data/`` (multi-GB downloads) or
    # ``.venv``. Disable with --no-reload for a stable run.
    uvicorn.run(
        "app.web.main:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
        reload=args.reload,
        reload_dirs=[str(ROOT / "app")] if args.reload else None,
    )


def _run_supervisor(args) -> None:
    """Spawn one dashboard process per enabled workspace and stream their
    output with a [slug] prefix. Ctrl-C stops all of them."""
    entries = [w for w in _load_registry()
               if w["enabled"] and (_REPO_ROOT / "workspaces" / w["slug"]).is_dir()]
    if not entries:
        raise SystemExit(
            "No enabled workspaces found in workspaces.yaml. Create one with "
            "`python run.py workspace create <slug>` or run a single dashboard "
            "with --workspace <slug>."
        )

    procs: list[tuple[str, subprocess.Popen]] = []

    def _pump(slug: str, proc: subprocess.Popen) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(f"[{slug}] {line}")
            sys.stdout.flush()

    for ws in entries:
        cmd = [sys.executable, str(_REPO_ROOT / "run.py"), "dashboard",
               "--workspace", ws["slug"], "--port", str(ws["port"])]
        if not args.reload:
            cmd.append("--no-reload")
        env = dict(os.environ, WORKSPACE=ws["slug"])
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        threading.Thread(target=_pump, args=(ws["slug"], proc), daemon=True).start()
        procs.append((ws["slug"], proc))
        print(f"[supervisor] {ws['label']} ({ws['slug']}) -> http://127.0.0.1:{ws['port']}")

    try:
        while True:
            for slug, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[supervisor] workspace {slug!r} exited with code {code}; "
                          "stopping the rest.")
                    raise KeyboardInterrupt
            time.sleep(1)
    except KeyboardInterrupt:
        for _slug, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for _slug, proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# --- Workspace scaffolding ----------------------------------------------------

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Files reset to a blank slate on create (content is per-topic); settings.yaml
# is copied from the source workspace so tuning carries over.
_BLANK_KEYWORDS = "# Keyword list for the first-pass filter (edit via the Keywords page).\n\nkeywords: []\n"
_BLANK_CHANNELS = "# Monitored YouTube channels (edit via the Channels page).\n\nchannels: []\n"
_BLANK_CAPTION_STYLE = "# Operator style rules for AI-drafted captions (edit via the Style guide page).\n\nrules: []\n"
_BLANK_FIRST_REPLY = "# Auto first-reply settings (edit via the Replies page).\n\nenabled: false\nattribution_enabled: true\ntext: ''\n"

_ENV_TEMPLATE = """\
# Workspace-specific secrets for the '{slug}' workspace. The shared root .env
# holds only cross-workspace keys (ANTHROPIC_API_KEY, YOUTUBE_API_KEY, ...).
# Fill these in before connecting accounts; blank keys fall back to nothing.

# Postgres for this workspace (its own Supabase project). Leave unset to use
# local SQLite at workspaces/{slug}/data/app.db (no headless publishing).
#DATABASE_URL=

# Supabase Storage for trimmed clips (same project as DATABASE_URL).
#SUPABASE_URL=
#SUPABASE_SERVICE_KEY=

# This workspace's own Meta app (do NOT reuse another workspace's app).
#THREADS_APP_ID=
#THREADS_APP_SECRET=
#THREADS_REDIRECT_URI=https://localhost/threads-callback
#INSTAGRAM_APP_ID=
#INSTAGRAM_APP_SECRET=
#INSTAGRAM_REDIRECT_URI=https://localhost/instagram-callback
"""

_REGISTRY_HEADER = """\
# Workspace registry: every account/topic this checkout can run. Each entry is
# a fully isolated environment under workspaces/<slug>/ with its own config/,
# data/, database, and Meta credentials, served by its own process on `port`.
#
# Add a new workspace with: python run.py workspace create <slug>

workspaces:
"""


def _registry_entry_text(ws: dict) -> str:
    return (f"  - slug: {ws['slug']}\n"
            f"    label: {ws['label']}\n"
            f"    port: {ws['port']}\n"
            f"    accent: \"{ws['accent']}\"\n"
            f"    enabled: {'true' if ws['enabled'] else 'false'}\n")


def cmd_workspace_list(_args) -> None:
    entries = _load_registry()
    if not entries:
        print("No workspaces registered. Create one with: python run.py workspace create <slug>")
        return
    for ws in entries:
        exists = (_REPO_ROOT / "workspaces" / ws["slug"]).is_dir()
        state = "enabled" if ws["enabled"] else "disabled"
        note = "" if exists else "  (directory missing!)"
        print(f"{ws['slug']:<16} {ws['label']:<24} port {ws['port']}  {state}{note}")


def cmd_workspace_create(args) -> None:
    """Scaffold a new isolated workspace: config seeded from an existing one,
    blank per-topic files, a .env skeleton, and a registry entry. Idempotent
    guards: refuses to overwrite an existing slug."""
    slug = args.slug.strip().lower()
    if not _SLUG_RE.match(slug):
        raise SystemExit(f"Invalid slug {slug!r}: use lowercase letters, digits, - and _")
    ws_dir = _REPO_ROOT / "workspaces" / slug
    if ws_dir.exists():
        raise SystemExit(f"Workspace directory already exists: {ws_dir}")
    entries = _load_registry()
    if any(w["slug"] == slug for w in entries):
        raise SystemExit(f"Workspace {slug!r} is already registered in workspaces.yaml")

    # Seed settings from an existing workspace so tuning carries over.
    source = args.copy_settings_from or (entries[0]["slug"] if entries else None)
    source_settings = (_REPO_ROOT / "workspaces" / source / "config" / "settings.yaml"
                       if source else None)

    label = args.label or slug.replace("_", " ").replace("-", " ").title()
    port = args.port or (max((w["port"] for w in entries), default=8320) + 1)
    accent = args.accent or "#4A6B8A"

    config_dir = ws_dir / "config"
    config_dir.mkdir(parents=True)
    (ws_dir / "data").mkdir()

    if source_settings and source_settings.exists():
        (config_dir / "settings.yaml").write_text(source_settings.read_text())
        print(f"settings.yaml copied from workspace {source!r} — review the topic-specific")
        print("sections (categories.options, vision.traits, subtitles colors) for the new topic.")
    else:
        print("No source workspace for settings.yaml — the app will run on built-in defaults")
        print("until you create config/settings.yaml.")

    (config_dir / "keywords.yaml").write_text(_BLANK_KEYWORDS)
    (config_dir / "channels.yaml").write_text(_BLANK_CHANNELS)
    (config_dir / "caption_style.yaml").write_text(_BLANK_CAPTION_STYLE)
    (config_dir / "first_reply.yaml").write_text(_BLANK_FIRST_REPLY)
    (config_dir / "brand.yaml").write_text(
        f"# Who this workspace is (edit via the Brand & audience page).\n\n"
        f"name: {label}\nmission: ''\naudience: ''\nvoice_notes: ''\ntopic: ''\n"
        f"app_name: {label}\nlogo_file: ''\n"
    )
    (ws_dir / ".env").write_text(_ENV_TEMPLATE.format(slug=slug))

    entry = {"slug": slug, "label": label, "port": port, "accent": accent,
             "enabled": bool(args.enabled)}
    if _REGISTRY_FILE.exists():
        text = _REGISTRY_FILE.read_text()
        if "workspaces:" not in text:
            text += "\n" + _REGISTRY_HEADER
        if not text.endswith("\n"):
            text += "\n"
        _REGISTRY_FILE.write_text(text + _registry_entry_text(entry))
    else:
        _REGISTRY_FILE.write_text(_REGISTRY_HEADER + _registry_entry_text(entry))

    print(f"\nWorkspace {slug!r} created at {ws_dir} (port {port}, "
          f"{'enabled' if entry['enabled'] else 'disabled'}).")
    print("Next steps:")
    print(f"  1. Fill in workspaces/{slug}/.env (Meta app, Supabase project)")
    print(f"  2. python run.py dashboard --workspace {slug}   # configure + OAuth")
    print("  3. Edit Brand & audience, Keywords, and Channels in the dashboard")
    if not entry["enabled"]:
        print(f"  4. Set enabled: true for {slug!r} in workspaces.yaml to include it in the supervisor")


def _monitor_pass_with_record(lookback: int | None) -> None:
    """One monitor pass, recorded in the MonitorRun table so the dashboard's
    "last refreshed" state also reflects headless (cron / GitHub Actions) runs."""
    from app.db import session_scope
    from app.models import MonitorRun, utcnow
    from app.monitor import run_monitor_once

    scope = f"last {lookback} days" if lookback else "since last check"
    with session_scope() as session:
        run = MonitorRun(status=MonitorRun.STATUS_RUNNING, scope=scope, lookback_days=lookback)
        session.add(run)
        session.flush()
        run_id = run.id
    try:
        result = run_monitor_once(lookback)
    except Exception as exc:
        with session_scope() as session:
            run = session.get(MonitorRun, run_id)
            if run is not None:
                run.status = MonitorRun.STATUS_FAILED
                run.error = str(exc)
                run.result = f"Monitor pass failed: {exc}"
                run.finished_at = utcnow()
        raise
    with session_scope() as session:
        run = session.get(MonitorRun, run_id)
        if run is not None:
            run.status = MonitorRun.STATUS_DONE
            run.channels_checked = result["channels_checked"]
            run.candidates_stored = result["candidates_stored"]
            run.vision_scored = result.get("vision_scored", 0)
            run.result = (
                f"{result['channels_checked']} channels checked, "
                f"{result['candidates_stored']} new candidates, "
                f"{result.get('vision_scored', 0)} vision-scored"
            )
            run.finished_at = utcnow()


def cmd_monitor(args) -> None:
    from app.config import load_settings
    from app.db import init_db

    init_db()
    if not args.loop:
        _monitor_pass_with_record(args.lookback)
        return
    interval_min = load_settings().get("monitor.poll_interval_minutes", 360)
    log.info("Monitoring every %d minutes. Ctrl-C to stop.", interval_min)
    while True:
        try:
            _monitor_pass_with_record(None)
        except Exception:
            log.exception("Monitor pass failed; will retry next interval")
        time.sleep(interval_min * 60)


def cmd_score_visuals(args) -> None:
    """Backfill storyboard trait tags for candidates that don't have them yet
    (respects vision.min_relevance and the daily budget). Neutral labels only."""
    from sqlalchemy import select

    from app import spend
    from app.config import load_settings
    from app.db import active_traits, init_db, session_scope, sync_traits_from_config
    from app.models import Candidate
    from app.vision import tag_candidate_storyboard

    init_db()
    settings = load_settings()
    min_rel = settings.get("vision.min_relevance", 0.5)
    tagged = skipped = 0
    with session_scope() as session:
        sync_traits_from_config(session)
        traits = active_traits(session)
        query = select(Candidate).where(
            (Candidate.visual_traits == "") | (Candidate.visual_traits.is_(None)),
            (Candidate.relevance_score.is_(None)) | (Candidate.relevance_score >= min_rel),
        ).order_by(Candidate.relevance_score.desc().nullslast())
        if args.limit:
            query = query.limit(args.limit)
        for c in session.execute(query).scalars().all():
            if not spend.within_budget():
                log.info("Daily budget reached ($%.2f); stopping.", spend.today_spend())
                break
            result = tag_candidate_storyboard(c, settings, force=True, traits=traits)
            if result is None:
                skipped += 1
            else:
                tagged += 1
                session.commit()
    print(f"Tagged {tagged} candidates, skipped {skipped}. "
          f"Spent ${spend.today_spend():.2f} of ${spend.daily_budget():.2f} today.")


def cmd_annotate_posts(args) -> None:
    """Backfill ground-truth footage traits for published posts whose clip files
    are still on local disk (extract frames -> tag -> store on the post),
    then recompute trait verdicts. Budget-guarded like all vision work."""
    from sqlalchemy import select

    from app import spend
    from app.analytics import learn_trait_weights
    from app.config import load_settings
    from app.db import active_traits, init_db, session_scope
    from app.models import ThreadsPost, TraitWeight
    from app.vision import annotate_post_footage

    init_db()
    settings = load_settings()
    annotated = skipped = 0
    with session_scope() as session:
        traits = active_traits(session)
        query = select(ThreadsPost).where(
            ThreadsPost.status == "published",
            ThreadsPost.clip_local_path != "",
        ).order_by(ThreadsPost.published_at.desc())
        if not args.force:
            query = query.where(ThreadsPost.footage_scored_at.is_(None))
        if args.limit:
            query = query.limit(args.limit)
        for post in session.execute(query).scalars().all():
            if not spend.within_budget():
                log.info("Daily budget reached ($%.2f); stopping.", spend.today_spend())
                break
            result = annotate_post_footage(post, settings, traits, force=args.force)
            if result is None:
                skipped += 1
            else:
                annotated += 1
                session.commit()
        verdicts = learn_trait_weights(session)
    active_n = sum(1 for v in verdicts if v["status"] == TraitWeight.STATUS_ACTIVE)
    print(f"Annotated {annotated} post(s), skipped {skipped} (missing clip/budget). "
          f"Verdicts: {len(verdicts)} trait(s) seen, {active_n} active. "
          f"Spent ${spend.today_spend():.2f} of ${spend.daily_budget():.2f} today.")


def cmd_backfill_post_times(args) -> None:
    """Recompute each published post's weekday/hour in the scheduler timezone.

    These two fields used to be read off the publishing machine's clock, so rows
    written by a CI runner or any non-local publisher landed in UTC while the
    operator's own posts landed in their laptop's zone — a silent mix feeding the
    analytics that tune the posting windows. ``published_at`` is stored as UTC,
    so every row can be restated exactly. Free (no LLM calls) and idempotent.

    Also fills ``clip_length_seconds`` for posts that published headlessly and
    never got measured, where the clip file is still on local disk.
    """
    from sqlalchemy import select

    from app.db import init_db, session_scope
    from app.models import ThreadsPost
    from app.publishing import _clip_duration_seconds, post_time_attributes

    init_db()
    retimed = relengthed = 0
    with session_scope() as session:
        posts = session.execute(
            select(ThreadsPost).where(
                ThreadsPost.status == "published",
                ThreadsPost.published_at.is_not(None),
            ).order_by(ThreadsPost.published_at.asc())
        ).scalars().all()
        for post in posts:
            dow, hour = post_time_attributes(post.published_at)
            if (post.post_day_of_week, post.post_hour_local) != (dow, hour):
                if args.dry_run:
                    log.info("post %s: %s h%s -> %s h%s", post.id,
                             post.post_day_of_week or "-", post.post_hour_local, dow, hour)
                else:
                    post.post_day_of_week, post.post_hour_local = dow, hour
                retimed += 1
            if post.clip_length_seconds is None and post.clip_local_path:
                if Path(post.clip_local_path).exists():
                    seconds = _clip_duration_seconds(Path(post.clip_local_path))
                    if seconds is not None:
                        if not args.dry_run:
                            post.clip_length_seconds = seconds
                        relengthed += 1
        if args.dry_run:
            session.rollback()

    if args.dry_run:
        print(f"Would restate {retimed} of {len(posts)} published post(s) into the "
              f"scheduler timezone and fill {relengthed} missing clip length(s).")
    else:
        print(f"Restated {retimed} of {len(posts)} published post(s) into the scheduler "
              f"timezone; filled {relengthed} missing clip length(s).")


def cmd_backfill_calendar_names(args) -> None:
    """Backfill the short 2-5 word calendar label for everything the calendar
    can display, in three passes (budget-guarded, like other LLM backfills):

    1. Cuts with a ``clip_title`` but no ``calendar_name`` yet (the common
       case: titled before the field existed) -> just condense the title.
    2. Cuts with no ``clip_title`` at all (from before auto-titling existed,
       or a titling attempt that failed) -> generate a title from the video's
       own transcript, then condense that into a calendar name.
    3. Cut-less posts (Threads history imported from outside the app — no
       clip/title concept, just a caption) -> condense the caption directly
       into ``ThreadsPost.calendar_name``.
    """
    from sqlalchemy import select

    from app import spend
    from app.config import load_settings
    from app.db import init_db, session_scope
    from app.llm import suggest_calendar_name, suggest_title
    from app.models import Cut, ThreadsPost

    init_db()
    settings = load_settings()
    model = settings.get("engagement.draft_model", "claude-sonnet-5")
    named = skipped = 0

    def _budget_ok() -> bool:
        if spend.within_budget():
            return True
        log.info("Daily budget reached ($%.2f); stopping.", spend.today_spend())
        return False

    with session_scope() as session:
        # Pass 1: already-titled cuts, just missing the short name.
        query = select(Cut).where(Cut.clip_title != "")
        if not args.force:
            query = query.where(Cut.calendar_name == "")
        query = query.order_by(Cut.id.desc())
        if args.limit:
            query = query.limit(args.limit)
        for cut in session.execute(query).scalars().all():
            if not _budget_ok():
                break
            try:
                name = suggest_calendar_name(model, cut.clip_title, cut.draft_caption or None)
            except Exception as exc:
                log.warning("Calendar name failed for cut %s: %s", cut.id, exc)
                skipped += 1
                continue
            if name:
                cut.calendar_name = name
                named += 1
                session.commit()
            else:
                skipped += 1

        # Pass 2: cuts that were never auto-titled — title, then name.
        query = select(Cut).where(Cut.clip_title == "")
        if args.limit:
            query = query.limit(args.limit)
        for cut in session.execute(query).scalars().all():
            if not _budget_ok():
                break
            c = cut.candidate
            if c is None:
                skipped += 1
                continue
            try:
                title = suggest_title(model, c.title, c.transcript_text[:3000], cut.draft_caption or None)
                if not title:
                    skipped += 1
                    continue
                name = suggest_calendar_name(model, title, cut.draft_caption or None)
            except Exception as exc:
                log.warning("Title/calendar name failed for cut %s: %s", cut.id, exc)
                skipped += 1
                continue
            cut.clip_title = title
            if name:
                cut.calendar_name = name
                named += 1
            session.commit()

        # Pass 3: cut-less posts (imported Threads history) — name from caption.
        query = select(ThreadsPost).where(ThreadsPost.cut_pk.is_(None), ThreadsPost.caption != "")
        if not args.force:
            query = query.where(ThreadsPost.calendar_name == "")
        query = query.order_by(ThreadsPost.id.desc())
        if args.limit:
            query = query.limit(args.limit)
        for post in session.execute(query).scalars().all():
            if not _budget_ok():
                break
            try:
                name = suggest_calendar_name(model, post.caption)
            except Exception as exc:
                log.warning("Calendar name failed for post %s: %s", post.id, exc)
                skipped += 1
                continue
            if name:
                post.calendar_name = name
                named += 1
                session.commit()
            else:
                skipped += 1

    print(f"Named {named} item(s), skipped {skipped}. "
          f"Spent ${spend.today_spend():.2f} of ${spend.daily_budget():.2f} today.")


def cmd_backfill_categories(args) -> None:
    """Auto-tag the programming category for candidates that don't have one
    yet, newest first. Budget-guarded like the other LLM backfills. Rejected
    candidates are skipped unless --all. Videos already in a reserved category
    (promos) are left alone even under --force — see app/categories.py."""
    from sqlalchemy import select

    from app import spend
    from app.categories import auto_tag_candidate
    from app.config import load_settings
    from app.db import init_db, session_scope
    from app.models import Candidate

    init_db()
    settings = load_settings()
    tagged = skipped = 0
    with session_scope() as session:
        query = select(Candidate)
        if not getattr(args, "all", False):
            query = query.where(Candidate.status != "rejected")
        if not args.force:
            query = query.where((Candidate.category == "") | (Candidate.category.is_(None)))
        query = query.order_by(Candidate.id.desc())
        if args.limit:
            query = query.limit(args.limit)
        for c in session.execute(query).scalars().all():
            if not spend.within_budget():
                log.info("Daily budget reached ($%.2f); stopping.", spend.today_spend())
                break
            try:
                result = auto_tag_candidate(c, settings)
            except Exception as exc:
                log.warning("Category tagging failed for candidate %s: %s", c.id, exc)
                skipped += 1
                continue
            if result is None:
                skipped += 1
            else:
                tagged += 1
                session.commit()
    print(f"Tagged {tagged} video(s), skipped {skipped}. "
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
    print(f"New comments: {result['new_comments']}")


def cmd_digest(_args) -> None:
    from app.analytics import write_and_store_digest
    from app.db import init_db, session_scope

    init_db()
    with session_scope() as session:
        result = write_and_store_digest(session)
    print(result["text"] or "(no published posts yet)")


def cmd_scheduler(args) -> None:
    """Run the adaptive window scheduler. One tick, or loop when --loop is set.
    The dashboard runs this automatically; use this for headless operation."""
    from app.config import database_url, env
    from app.db import init_db
    from app.scheduler import run_tick, start_scheduler_thread

    # Say which database this tick is talking to. With no DATABASE_URL the app
    # falls back to a local SQLite file, so a headless runner missing that secret
    # would tick happily forever against an empty queue and publish nothing —
    # exactly the kind of silent no-op that's invisible from outside.
    backend = database_url().split("://", 1)[0]
    if backend.startswith("sqlite"):
        log.warning("Scheduler is on local SQLite%s — a headless runner almost "
                    "certainly wants the shared Postgres via DATABASE_URL.",
                    "" if env("DATABASE_URL") else " (DATABASE_URL is unset)")
    else:
        log.info("Scheduler database backend: %s", backend)

    init_db()
    if not args.loop:
        run_tick()
        print("Scheduler tick complete")
        return
    start_scheduler_thread(interval_seconds=args.interval)
    log.info("Scheduler loop started (every %ds). Ctrl-C to stop.", args.interval)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


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

    from app.config import load_settings, storage_dir

    settings = load_settings()
    retention = settings.get("storage.retention", "keep")
    if retention == "keep":
        print("storage.retention is 'keep' — nothing to prune. Set it to a number of days to enable.")
        return
    cutoff = time.time() - int(retention) * 86400
    root = storage_dir(settings, "storage.download_dir", "data/videos")
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

    # Shared by every command that runs against one workspace. Must be applied
    # to os.environ before any app.* import (see the bottom of main()).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", default=None, metavar="SLUG",
                        help="workspace to run against (default: WORKSPACE env var, then 'climate')")

    p = sub.add_parser("dashboard", parents=[common],
                       help="run the web dashboard(s); with no --workspace, supervises all enabled workspaces")
    p.add_argument("--port", type=int, default=None,
                   help="override the workspace's registry port (single-workspace mode only)")
    p.add_argument("--no-reload", dest="reload", action="store_false",
                   help="disable auto-reload on source changes")
    p.set_defaults(func=cmd_dashboard, reload=True)

    p = sub.add_parser("workspace", help="manage the workspace registry")
    wsub = p.add_subparsers(dest="ws_command", required=True)
    wp = wsub.add_parser("list", help="show registered workspaces")
    wp.set_defaults(func=cmd_workspace_list)
    wp = wsub.add_parser("create", help="scaffold a new isolated workspace")
    wp.add_argument("slug", help="directory name under workspaces/ (lowercase)")
    wp.add_argument("--label", default=None, help="display name (default: title-cased slug)")
    wp.add_argument("--port", type=int, default=None, help="dashboard port (default: next free)")
    wp.add_argument("--accent", default=None, metavar="HEX", help="sidebar accent color")
    wp.add_argument("--copy-settings-from", default=None, metavar="SLUG",
                    help="workspace whose settings.yaml to copy (default: first registered)")
    wp.add_argument("--enabled", action="store_true",
                    help="include in the supervisor immediately (default: disabled until configured)")
    wp.set_defaults(func=cmd_workspace_create)

    p = sub.add_parser("monitor", parents=[common], help="run channel discovery")
    p.add_argument("--loop", action="store_true", help="poll forever at the configured interval")
    p.add_argument("--lookback", type=int, default=None, metavar="DAYS",
                   help="scan this many days back instead of since-last-check (backfill)")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("scheduler", parents=[common], help="run the adaptive posting scheduler")
    p.add_argument("--loop", action="store_true", help="keep running window checks + metrics polls")
    p.add_argument("--interval", type=int, default=60, help="seconds between checks in --loop mode")
    p.set_defaults(func=cmd_scheduler)

    sub.add_parser("migrate-db", parents=[common],
                   help="copy local SQLite data into DATABASE_URL (e.g. Supabase)").set_defaults(func=cmd_migrate_db)
    p = sub.add_parser("score-visuals", parents=[common],
                       help="backfill vision scores for unscored candidates")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="score at most N candidates this run")
    p.set_defaults(func=cmd_score_visuals)

    p = sub.add_parser("annotate-posts", parents=[common],
                       help="backfill footage traits for published posts (from posted clip files)")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="annotate at most N posts this run")
    p.add_argument("--force", action="store_true",
                   help="re-annotate posts that already have footage traits")
    p.set_defaults(func=cmd_annotate_posts)

    p = sub.add_parser("backfill-post-times", parents=[common],
                       help="restate published posts' weekday/hour in the scheduler timezone")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change without writing")
    p.set_defaults(func=cmd_backfill_post_times)

    p = sub.add_parser("backfill-calendar-names", parents=[common],
                       help="generate short 2-5 word calendar labels for cuts titled before that field existed")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="name at most N cuts this run")
    p.add_argument("--force", action="store_true",
                   help="regenerate cuts that already have a calendar_name")
    p.set_defaults(func=cmd_backfill_calendar_names)

    p = sub.add_parser("backfill-categories", parents=[common],
                       help="auto-tag programming categories for untagged videos")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="tag at most N videos this run")
    p.add_argument("--force", action="store_true",
                   help="re-tag videos that already have a category (never promos)")
    p.add_argument("--all", action="store_true",
                   help="include rejected candidates")
    p.set_defaults(func=cmd_backfill_categories)

    sub.add_parser("metrics", parents=[common],
                   help="snapshot Threads post metrics").set_defaults(func=cmd_metrics)
    sub.add_parser("comments", parents=[common],
                   help="sync comments on own posts").set_defaults(func=cmd_comments)
    sub.add_parser("digest", parents=[common],
                   help="write a fresh analytics digest").set_defaults(func=cmd_digest)
    sub.add_parser("cleanup", parents=[common],
                   help="apply retention setting to downloaded segments").set_defaults(func=cmd_cleanup)

    args = parser.parse_args()

    # Apply the workspace choice BEFORE any app.* import resolves it.
    if getattr(args, "workspace", None):
        os.environ["WORKSPACE"] = args.workspace

    # The registry commands and the supervisor must work without any existing
    # workspace (fresh clone), so they never import app.*; everything else gets
    # the workspace's file logging.
    if args.command == "workspace" or (
        args.command == "dashboard" and not getattr(args, "workspace", None)
    ):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")
    else:
        from app.logging_setup import setup_logging

        setup_logging(logging.INFO)

    args.func(args)


if __name__ == "__main__":
    main()
