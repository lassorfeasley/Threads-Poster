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
  python run.py triage-train       # retrain the approve predictor, print its backtest
  python run.py metrics            # snapshot Threads metrics for published posts
  python run.py comments           # sync comments on own posts
  python run.py digest             # write a fresh analytics digest (one LLM call)
  python run.py cleanup            # apply the retention setting (never automatic)
  python run.py offload --dry-run  # videos/clips that would move to Supabase or be deleted
  python run.py offload            # move finished videos and clips to Supabase, free the disk
  python run.py scheduler          # one adaptive-scheduler tick (windows + metrics)
  python run.py scheduler --loop   # keep the adaptive scheduler running
  python run.py migrate-db         # copy local SQLite data into DATABASE_URL (Supabase)

Clip engine (docs/clip-engine.md):
  python run.py clip-engine try <id>        # run the engine on one video, print every draft (writes nothing)
  python run.py clip-engine try <id> --log  # ...and save its top picks as suggestions
  python run.py clip-engine shotlog <id>    # build (or show the cached) shot log of a video
  python run.py clip-engine playbook        # print the clip playbook (--refresh rebuilds it from your posts)
  python run.py clipbench run --name N      # replay past videos through each setup (resumable)
  python run.py clipbench pairs --name N    # render the blind A/B pairs for Analytics -> Clip bench
  python run.py clipbench report --name N   # a run's report (results only after the reveal)
  python run.py clipbench reveal --name N   # unlock a run's results; closes its judging
  python run.py clipbench critic-check      # does the critic agree with your "gripping?" labels?
  python run.py clipbench list              # bench runs and their judging progress
  python run.py retranscribe --missing-words  # add Whisper word streams to videos that lack them
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import logging
import math
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
_BLANK_FIRST_REPLY = "# First reply settings (edit on the First reply page).\n\nenabled: false\nwatch_text: ''\nfallback_text: ''\n"

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
    from app.monitor import failed_channels_note, run_monitor_once

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
                f"{failed_channels_note(result)}"
            )
            run.finished_at = utcnow()
    failed = result.get("channels_failed") or []
    if failed and os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning title=Monitor::{len(failed)} channels failing: {', '.join(failed)}")


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
    from app.db import active_traits_by_facet, init_db, session_scope, sync_traits_from_config
    from app.models import Candidate
    from app.vision import tag_candidate_storyboard

    init_db()
    settings = load_settings()
    min_rel = settings.get("vision.min_relevance", 0.5)
    tagged = skipped = 0
    with session_scope() as session:
        sync_traits_from_config(session)
        vocab = active_traits_by_facet(session)
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
            result = tag_candidate_storyboard(c, settings, force=True,
                                              traits=vocab["subject"],
                                              format_traits=vocab["format"])
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
    from app.db import active_traits_by_facet, init_db, session_scope
    from app.models import ThreadsPost, TraitWeight
    from app.vision import annotate_post_footage

    init_db()
    settings = load_settings()
    annotated = skipped = 0
    with session_scope() as session:
        vocab = active_traits_by_facet(session)
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
            result = annotate_post_footage(post, settings, vocab["subject"],
                                           force=args.force,
                                           format_traits=vocab["format"])
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


def cmd_triage_train(_args) -> None:
    """Retrain the approve predictor now, rescore the triage queue, and print
    how the new model backtested against your most recent decisions."""
    from app import triage_model
    from app.db import init_db, session_scope

    init_db()
    if not triage_model.maybe_retrain(force=True):
        print("Not trained: too few approvals/rejections yet (see triage_model.min_decisions_each).")
        return
    with session_scope(read_only=True) as session:
        card = triage_model.scorecard(session)
    m, bt = card["model"], card["model"]["backtest"]
    print(f"Model #{m['id']}: {m['n_decisions']} decisions ({m['n_approved']} approved)")
    print(f"Backtest on the newest {bt['n']} ({bt['approved']} approved): AUC {bt['auc']:.3f}")
    print("Predicted range -> share you actually approved:")
    for b in bt["buckets"]:
        rate = f"{b['approved'] / b['n']:.0%}" if b["n"] else "-"
        print(f"  {b['lo']:>4.0%}-{b['hi']:<4.0%}  {rate:>4} of {b['n']}")
    print("Had it auto-rejected everything under:")
    for h in bt["handoff"]:
        print(f"  {h['threshold']:.0%}: {h['below']} of {bt['n']} videos, "
              f"{h['approvals_lost']} of {bt['approved']} approvals lost")


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


def cmd_placement_sim(args) -> None:
    """Replay recent publish history through the scored placement engine and
    report facet mix, same-source spacing, relaxation frequency, and queue
    wait. Read-only: tune scheduler.placement weights against this, then flip
    mode to 'scored'."""
    from app.db import init_db, session_scope
    from app.placement_sim import format_report, run_replay

    init_db()
    with session_scope() as session:
        report = run_replay(session, days=args.days)
    print(format_report(report))


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


def cmd_offload(args) -> None:
    """Upload source videos that are out of the workflow, and the clips of cuts
    whose posts have all published, to Supabase and delete the local copies;
    delete rejected videos' downloads, unreferenced clip files, and outro
    renders nothing pending needs. The dashboard does the same hourly when
    storage.offload_sources is on."""
    from app.db import init_db
    from app.offload import free_bytes, run_sweep_once

    init_db()
    res = run_sweep_once(dry_run=args.dry_run, limit=args.limit)
    if res is None:
        print("A sweep is already running in this process.")
        return
    labels = {"offload": "upload + delete", "delete": "delete (rejected)",
              "archive": "archive clip", "delete-orphan": "delete orphan clip",
              "delete-outro": "delete outro render"}
    for verb, size, path in res.planned:
        print(f"{'would ' if args.dry_run else ''}{labels[verb]:20s} {size / 1e6:9.1f} MB  {path}")
    if args.dry_run:
        print()
        for verb, label in labels.items():
            hits = [size for v, size, _ in res.planned if v == verb]
            if hits:
                print(f"Would {label}: {len(hits)} files, {sum(hits) / 1e9:.1f} GB")
        print(f"Keeping {res.kept} files on disk.")
    else:
        print(f"\nUploaded {res.offloaded} files ({res.offloaded_bytes / 1e9:.1f} GB) and "
              f"deleted their local copies; deleted {res.deleted} more "
              f"({res.deleted_bytes / 1e9:.1f} GB); {res.failed} failed (see log). "
              f"Kept {res.kept} on disk.")
    print(f"Free disk space: {free_bytes() / 1e9:.1f} GB")


# --- Clip engine and its replay bench (docs/clip-engine.md) -------------------
#
# How the engine is tried, inspected and benchmarked outside the dashboard. The
# rules of the dashboard's job runner apply here too: no DB session is held
# across a model call, a download, ffmpeg or Whisper (read into plain values,
# roll back, do the slow work, write in a fresh short session); sources are
# hard-linked under the engine's inflight/ directory while ffmpeg reads them;
# and `clipbench` books its spend to the "bench" bucket, apart from the
# dashboard's daily budget and the engine's own cap.

_ENGINE_SETUPS = ("gemini", "hybrid", "claude")
_CRITIC_ROLES = ("gemini", "claude")
# Run names become a directory under data/clipbench and a URL path segment.
# Keep in step with bench.py and app/web/clip_bench.py (a test checks).
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_CANDIDATE_FIELDS = ("id", "video_id", "title", "description", "duration_seconds",
                     "local_video_path", "source_object_key", "transcript_path",
                     "word_transcript_path", "transcription_method", "status",
                     "archived_at")


def _say(message: str) -> None:
    print(message, flush=True)


# argparse types. They run before --workspace is applied, so no app.* imports.

def _parse_time(text: str) -> float:
    """Seconds from "95", "95.5", "1:35" or "1:01:35"."""
    parts = str(text).strip().split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        values = []
    ok = (bool(values) and len(values) <= 3
          and all(math.isfinite(v) and v >= 0 for v in values)
          and all(v < 60 for v in values[1:])
          and all(v == int(v) for v in values[:-1]))
    if not ok:
        raise argparse.ArgumentTypeError(
            f"not a time: {text!r} (use seconds, m:ss or h:mm:ss)")
    seconds = 0.0
    for v in values:
        seconds = seconds * 60 + v
    return seconds


def _run_name(text: str) -> str:
    name = str(text).strip()
    if not _RUN_NAME_RE.match(name):
        raise argparse.ArgumentTypeError(
            f"bad run name {text!r}: start with a letter or digit, then letters, "
            "digits, '.', '_' or '-' (64 at most)")
    return name


def _csv(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _critic_roles(text: str) -> list[str]:
    roles = _csv(text)
    if not roles or any(r not in _CRITIC_ROLES for r in roles):
        raise argparse.ArgumentTypeError(
            f"a comma list of {', '.join(_CRITIC_ROLES)}, got {text!r}")
    return list(dict.fromkeys(roles))


def _int_at_least(minimum: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value
    return parse


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0, got {text!r}")
    return value


# Reading a candidate: short sessions, plain values out.

def _resolve_candidate_pk(session, ident: str) -> int:
    """A candidate's pk from its pk or its YouTube video id (the newest
    archived row when one video id was ingested more than once)."""
    from sqlalchemy import select

    from app.models import STATUS_ARCHIVED, Candidate

    ident = str(ident).strip()
    if ident.isdigit():
        pk = session.execute(select(Candidate.id).where(Candidate.id == int(ident))).scalar()
        if pk is not None:
            return int(pk)
    rows = session.execute(select(Candidate.id, Candidate.status)
                           .where(Candidate.video_id == ident)
                           .order_by(Candidate.id.desc())).all()
    if not rows:
        raise SystemExit(f"No video with id or YouTube id {ident!r}.")
    archived = [r for r in rows if r.status == STATUS_ARCHIVED]
    return int((archived or rows)[0].id)


def _candidate_fields(session, pk: int) -> dict | None:
    from sqlalchemy import select

    from app.models import Candidate

    row = session.execute(select(*[getattr(Candidate, f) for f in _CANDIDATE_FIELDS])
                          .where(Candidate.id == pk)).first()
    return dict(row._mapping) if row is not None else None


def _read_candidate(ident: str) -> dict:
    """One candidate's plain fields, read in a short session that is rolled
    back and closed before anything slow happens."""
    from app.db import session_scope

    with session_scope() as session:
        fields = _candidate_fields(session, _resolve_candidate_pk(session, ident))
        session.rollback()
    if fields is None:
        raise SystemExit(f"No video with id or YouTube id {ident!r}.")
    return fields


def _resolve_candidate_pks(idents: list[str]) -> list[int]:
    from app.db import session_scope

    with session_scope() as session:
        pks = [_resolve_candidate_pk(session, ident) for ident in idents]
        session.rollback()
    return list(dict.fromkeys(pks))


def _load_texts(fields: dict) -> tuple[list[dict], list[dict]]:
    """(transcript entries, Whisper words) from the local JSON files."""
    from types import SimpleNamespace

    from app import clip_proposals

    ns = SimpleNamespace(transcript_path=fields.get("transcript_path") or "",
                         word_transcript_path=fields.get("word_transcript_path") or "")
    return clip_proposals.load_transcript(ns), clip_proposals.load_words(ns)


def _duration(fields: dict) -> float | None:
    try:
        value = float(fields.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _inflight_key(video_id: str) -> str:
    """The CLI's inflight link name for a video. Stable, so consecutive CLI
    runs share the render/proxy caches (they key on the linked path), and
    distinct from the dashboard's (it links under the bare video id), so
    neither process's release removes a link the other is still reading."""
    return f"{video_id}-cli"


@contextlib.contextmanager
def _inflight_source(fields: dict, key: str):
    """The source video, restored from storage when it was offloaded, and
    hard-linked under the engine's inflight/ directory while it is in use, so
    an offload sweep or a re-archive can't pull it out from under ffmpeg."""
    from app.clip_engine import media
    from app.offload import SourceUnavailable, ensure_local

    try:
        source = ensure_local(fields.get("local_video_path") or "",
                              fields.get("source_object_key") or "")
    except SourceUnavailable as exc:
        raise SystemExit(f"The source video is unavailable: {exc}") from None
    except Exception as exc:  # storage/network trouble while restoring it
        raise SystemExit(f"Couldn't bring the source video back: "
                         f"{type(exc).__name__}: {exc}") from None
    try:
        linked = media.link_inflight(Path(source), key)
    except Exception as exc:  # the link is a safety net; the run can go on without it
        log.warning("Could not link %s under inflight/ (%s); using it in place", source, exc)
        yield Path(source)
        return
    try:
        yield linked
    finally:
        media.release_inflight(linked)


def _require_provider(provider: str) -> None:
    if provider == "gemini":
        from app.clip_engine import gemini

        ok, key = gemini.is_configured(), "GEMINI_API_KEY"
    else:
        from app.clip_engine import claude

        ok, key = claude.is_configured(), "ANTHROPIC_API_KEY"
    if not ok:
        raise SystemExit(f"{key} is not set (see .env.example).")


def _engine_setup(settings, requested: str | None) -> str:
    """The engine setup to run, or exit saying why it can't run."""
    from app import clip_engine

    setup, reason = clip_engine.effective_setup(settings, requested)
    if setup != "transcript":
        return setup
    if not requested and not reason:
        raise SystemExit("clip_engine.setup is 'transcript' (the original suggester). "
                         "Pass --setup gemini|hybrid|claude to try the engine.")
    wanted = requested or clip_engine.configured_setup(settings)
    raise SystemExit(f"Can't run the {wanted} setup: {reason}.")


def _mark_touched(video_ids, *, by: str) -> None:
    """Add videos to the bench's touched.json. A video you've seen engine
    output for is tuning data now, and the test-seed bench must never select
    it — a go/no-go measured on the videos the prompts were tuned against
    says nothing. Best effort: a failure warns but never stops the command."""
    # bench.touch_videos is touched.json's one writer, so the file has one
    # owner and one format.
    ids = list(dict.fromkeys(str(v) for v in video_ids if v))
    if not ids:
        return
    try:
        from app.clip_engine import bench

        bench.touch_videos(ids, source=by)
    except Exception as exc:
        log.warning("Could not add %s to the bench's touched.json: %s", ", ".join(ids), exc)
        print(f"Warning: couldn't record {', '.join(ids)} as touched ({exc}); "
              "keep it out of test-seed bench runs by hand.")


# Text output for the engine commands (pure; tested directly).

def _trim(text, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:max(0, limit - 1)] + "…"


def _clock(seconds, decimals: int = 1) -> str:
    """75.3 -> "1:15.3"; 3723 -> "1:02:03.0"; "?" for anything that isn't a time."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(value):
        return "?"
    t = round(max(0.0, value), decimals)
    hours, rem = divmod(t, 3600)
    minutes, secs = divmod(rem, 60)
    width = 3 + decimals if decimals else 2
    sec = f"{secs:0{width}.{decimals}f}"
    return f"{int(hours)}:{int(minutes):02d}:{sec}" if hours else f"{int(minutes)}:{sec}"


def _seg_seconds(segments) -> float:
    total = 0.0
    for seg in segments or []:
        try:
            total += max(0.0, float(seg["end"]) - float(seg["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    return total


def _fmt_segments(segments) -> str:
    """Segments in PLAY order (a cold open shows as a later stretch first)."""
    parts = []
    for seg in segments or []:
        try:
            parts.append(f"{_clock(seg['start'])}-{_clock(seg['end'])}")
        except (KeyError, TypeError):
            continue
    return " -> ".join(parts) or "-"


def _score(value) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return "error" if v <= -1e8 else f"{v:.2f}"


def _critic_line(critique, *, with_fix: bool = True) -> str:
    """The cold critic's verdict in one line, the way the suggestion cards say it."""
    if not isinstance(critique, dict) or not critique:
        return "not watched"
    if critique.get("error"):
        return f"not reviewed ({_trim(critique['error'], 80)})"
    verdict = "Stops the scroll" if critique.get("stops_scroll") else "Wouldn't stop the scroll"
    text = f"{verdict}: {critique.get('stop_reason') or '-'}"
    bored = critique.get("bored_at")
    if isinstance(bored, (int, float)) and not isinstance(bored, bool):
        text += f" · slow at {_clock(bored, decimals=0)}"
    if critique.get("topic") in ("incidental", "none"):
        text += f" · topic: {critique['topic']}"
    if with_fix and critique.get("fix"):
        text += f" · fix: {critique['fix']}"
    return text


def _cost_lines(total, by_stage) -> list[str]:
    stages = {}
    for key, value in (by_stage or {}).items():
        try:
            stages[str(key)] = float(value or 0)
        except (TypeError, ValueError):
            continue
    charged = {k: v for k, v in stages.items() if not k.endswith("_cached")}
    cached = {k: v for k, v in stages.items() if k.endswith("_cached")}
    line = f"Cost: ${float(total or 0):.4f}"
    if charged:
        line += " — " + " · ".join(f"{k} ${v:.4f}" for k, v in charged.items())
    lines = [line]
    for key, value in cached.items():
        lines.append(f"  (+ {key[:-len('_cached')]} ${value:.4f} paid by an earlier run: "
                     "its stored result was reused, not charged again)")
    return lines


def format_try_report(result) -> str:
    """An EngineResult as text: every draft in the editor's order with its
    critic verdict and why it was dropped; the top picks; cost, tokens,
    timings, the four orderings (editor / cold / rank / final) and notes."""
    top = [t for t in (getattr(result, "top", None) or []) if isinstance(t, dict)]
    drafts = [d for d in (getattr(result, "drafts", None) or []) if isinstance(d, dict)]
    picked = {str(t.get("id")): i for i, t in enumerate(top, 1)}
    out = [f"Setup {result.setup} · {result.model or '-'} · outcome {result.outcome}", ""]

    out.append(f"Drafts ({len(drafts)}, in the editor's order):")
    if drafts:
        out.append(f"  {'id':<6} {'archetype':<11} {'lens':<8} {'secs':>5} {'score':>6}  "
                   f"{'result':<22} critic")
    for d in drafts:
        did = str(d.get("id") or "")
        if did in picked:
            status = f"pick #{picked[did]}"
        elif d.get("dropped_reason"):
            status = f"dropped: {d['dropped_reason']}"
        else:
            status = "not picked"
        out.append(f"  {did:<6.6} {str(d.get('archetype') or ''):<11.11} "
                   f"{str(d.get('lens') or ''):<8.8} {_seg_seconds(d.get('segments')):>5.1f} "
                   f"{_score(d.get('score')):>6}  {status:<22.22} "
                   f"{_trim(_critic_line(d.get('critique')), 110)}")

    out.append("")
    if not top:
        out.append(f"No top picks (outcome: {result.outcome}).")
    else:
        out.append("Top picks:")
    for i, t in enumerate(top, 1):
        origin = (f" (revision of {t['parent']})"
                  if t.get("origin") == "revision" and t.get("parent") else "")
        out.append(f"  {i}. {t.get('id')}{origin} · {t.get('archetype') or '-'}/"
                   f"{t.get('lens') or '-'} · {_seg_seconds(t.get('segments')):.1f} s · "
                   f"score {_score(t.get('score'))}")
        out.append(f"     plays: {_fmt_segments(t.get('segments'))}")
        for key in ("story", "hook", "payoff", "why", "risk"):
            if t.get(key):
                out.append(f"     {key}: {_trim(t[key], 160)}")
        critique = t.get("critique") if isinstance(t.get("critique"), dict) else {}
        out.append(f"     critic: {_trim(_critic_line(critique, with_fix=False), 160)}")
        if critique.get("fix") and not critique.get("error"):
            out.append(f"     fix: {_trim(critique['fix'], 160)}")
        if t.get("credits"):
            out.append(f"     credits: {_trim(t['credits'], 160)}")
        if t.get("render"):
            out.append(f"     render: {t['render']}")

    out.append("")
    out.extend(_cost_lines(getattr(result, "cost_usd", 0.0), getattr(result, "by_stage", {})))
    tokens = getattr(result, "tokens", None) or {}
    if tokens:
        out.append("Tokens: " + " · ".join(
            f"{model} {int(t.get('input', 0) or 0):,} in / {int(t.get('output', 0) or 0):,} out"
            for model, t in tokens.items() if isinstance(t, dict)))
    timings = {k: v for k, v in (getattr(result, "timings", None) or {}).items()
               if isinstance(v, (int, float))}
    if timings:
        out.append("Timings: " + " · ".join(f"{k} {float(v):.1f} s" for k, v in timings.items()))
    orders = getattr(result, "orders", None) or {}
    for key in ("editor", "cold", "rank", "final"):
        if orders.get(key):
            out.append(f"Order by {key}: {' '.join(str(x) for x in orders[key])}")
    notes = getattr(result, "notes", None) or []
    if notes:
        out.append("Notes:")
        out.extend(f"  - {_trim(n, 300)}" for n in notes)
    return "\n".join(out)


def format_shotlog_report(shot_log: dict, *, video_id: str = "") -> str:
    """A shot log as text: the summary, who is on screen, every beat, highlights."""
    beats = [b for b in (shot_log.get("beats") or []) if isinstance(b, dict)]
    highlights = [h for h in (shot_log.get("highlights") or []) if isinstance(h, dict)]
    people = [p for p in (shot_log.get("people") or []) if isinstance(p, dict)]
    head = (f"Shot log {video_id} · {shot_log.get('perceiver') or '?'}/"
            f"{shot_log.get('model') or '?'} · {_clock(shot_log.get('source_seconds') or 0)} · "
            f"{len(beats)} beats · {len(highlights)} highlights · "
            f"${float(shot_log.get('cost_usd') or 0):.4f}")
    if not shot_log.get("complete", True):
        head += " · INCOMPLETE"
    out = [head.replace("Shot log  ·", "Shot log ·")]
    for label, key in (("Summary", "summary"), ("Format", "format"), ("Structure", "structure")):
        if shot_log.get(key):
            out.append(f"{label}: {shot_log[key]}")
    if people:
        out.append("People:")
        for p in people:
            out.append(f"  {p.get('name') or '?'} — {p.get('role') or '?'} "
                       f"({p.get('prominence') or '?'}; {p.get('evidence') or 'no evidence'})")
    if beats:
        out.append("Beats:")
    for b in beats:
        pieces = []
        if b.get("line"):
            pieces.append(f"{b.get('speaker') or '?'}: \"{b['line']}\"")
        if b.get("on_screen"):
            pieces.append(f"on screen: {b['on_screen']}")
        if b.get("sound"):
            pieces.append(f"sound: {b['sound']}")
        if b.get("note"):
            pieces.append(str(b["note"]))
        out.append(f"  {_clock(b.get('start'))}-{_clock(b.get('end'))}  "
                   f"{str(b.get('shot') or ''):<14.14} face {str(b.get('face') or '-'):<5.5} "
                   f"E{b.get('energy', '-')} V{b.get('visual', '-')} T{b.get('topic', '-')}  "
                   f"{_trim(' | '.join(pieces), 160)}")
    if highlights:
        out.append("Highlights:")
    for h in highlights:
        out.append(f"  {_clock(h.get('start'))}-{_clock(h.get('end'))}  "
                   f"{str(h.get('kind') or ''):<7.7} {_trim(h.get('what') or '', 160)}")
    return "\n".join(out)


def _print_summary(data, indent: str = "  ", depth: int = 0) -> None:
    """A dict from bench.py, printed as key: value lines (nested two deep;
    lists shown by length) — its shape belongs to bench.py, not this file."""
    if not isinstance(data, dict):
        if data is not None:
            print(f"{indent}{data}")
        return
    for key, value in data.items():
        if isinstance(value, dict) and depth < 2:
            print(f"{indent * (depth + 1)}{key}:")
            _print_summary(value, indent, depth + 1)
        elif isinstance(value, (list, tuple)):
            shown = ", ".join(str(v) for v in value) if all(
                isinstance(v, (str, int, float)) for v in value) else f"{len(value)} item(s)"
            print(f"{indent * (depth + 1)}{key}: {_trim(shown, 200)}")
        else:
            print(f"{indent * (depth + 1)}{key}: {_trim(value, 200)}")


def _save_try(result, video_id: str) -> Path | None:
    """The full EngineResult as JSON under the engine's tries/ directory: the
    printed tables truncate, this keeps every draft, critique and ordering."""
    import dataclasses

    from app.clip_engine import engine_dir, util

    try:
        data = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else dict(vars(result))
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return util.atomic_write_json(
            engine_dir() / "tries" / f"{video_id}-{result.setup}-{stamp}.json", data)
    except Exception as exc:
        log.warning("Could not save the engine result: %s", exc)
        return None


def cmd_clip_engine_try(args) -> None:
    """Run the clip engine on one archived video and print everything it did.

    Dry by default — nothing is written to the database. ``--log`` writes the
    top picks as the video's pending suggestions, exactly like a Re-suggest in
    the dashboard (the previous pending ones are superseded). The video is
    added to the bench's touched.json either way."""
    from app import clip_engine, spend
    from app.clip_engine import engine, jobs
    from app.config import load_settings
    from app.db import init_db, session_scope

    settings = load_settings()
    setup = _engine_setup(settings, args.setup)
    window = None
    if args.section:
        start, end = args.section
        if end <= start:
            raise SystemExit(f"--section END must be after START ({_clock(start)} >= {_clock(end)}).")
        window = {"start": start, "end": end}

    init_db()
    with session_scope() as session:
        pk = _resolve_candidate_pk(session, args.id)
        session.rollback()
    inputs = jobs.prepare_inputs(pk, window=window)
    cand = dict(inputs.get("candidate") or {})
    if not cand:
        raise SystemExit(f"Video {pk} not found.")
    if cand.get("status") != "archived":
        raise SystemExit(f"Video {pk} isn't archived (status {cand.get('status')!r}); "
                         "the engine needs the downloaded video and its transcript.")
    transcript, words = _load_texts(cand)
    if not transcript:
        raise SystemExit(f"Video {pk} has no transcript on this machine "
                         f"({cand.get('transcript_path') or 'no transcript_path'}).")
    for note in inputs.get("notes") or []:
        print(f"note: {note}")
    _mark_touched([cand.get("video_id")], by="clip-engine try")

    duration = _duration(cand)
    print(f"Video {pk} ({cand.get('video_id')}): {_trim(cand.get('title'), 90)}")
    detail = [f"setup {setup}", f"{len(words):,} words" if words else "no word stream (rougher cuts)"]
    if duration:
        detail.append(f"{_clock(duration, 0)} long")
        try:
            detail.append(f"estimate ~${engine.estimate_usd(setup, settings, duration):.2f}")
        except Exception:
            pass
    if window:
        detail.append(f"section {_clock(window['start'])}-{_clock(window['end'])}")
    try:
        cap = float(clip_engine.opt(settings, "clip_engine.daily_usd", 5.0))
        detail.append(f"engine spend today ${spend.bucket_spend('engine'):.2f} of ${cap:.2f}")
    except Exception:
        pass
    print(" · ".join(detail))

    meter = engine.CostMeter("engine")
    started = time.monotonic()
    timeout = float(clip_engine.opt(settings, "clip_engine.job_timeout_seconds", 1800))

    def progress(label: str) -> None:
        _say(f"[{time.monotonic() - started:6.1f}s] {label}")

    with _inflight_source(cand, key=_inflight_key(cand.get("video_id") or str(pk))) as source:
        try:
            result = engine.run(
                setup=setup, settings=settings, brand=clip_engine.brand(), playbook=None,
                video_id=cand.get("video_id") or "", title=cand.get("title") or "",
                description=cand.get("description") or "", duration=duration,
                source=source, words=words, transcript=transcript,
                context=inputs.get("context") or {}, rejected=inputs.get("rejected") or [],
                examples=inputs.get("examples") or [], used=inputs.get("used") or [],
                window=window, section_topic="", keep=None, meter=meter, progress=progress,
                cancel=None, deadline=started + timeout)
        except KeyboardInterrupt:
            print("\nInterrupted. Spent so far:")
            print("\n".join(_cost_lines(meter.total, meter.by_stage)))
            raise SystemExit(130) from None
        except Exception as exc:
            print(f"\nThe engine run failed: {type(exc).__name__}: {exc}")
            print("\n".join(_cost_lines(meter.total, meter.by_stage)))
            raise SystemExit(1) from None

    print()
    print(format_try_report(result))
    saved = _save_try(result, cand.get("video_id") or str(pk))
    if saved:
        print(f"\nFull result: {saved}")
    if not args.log:
        print("Dry run: nothing written. Add --log to save the top picks as suggestions.")
    elif not result.top:
        print(f"Nothing to log (outcome: {result.outcome}); the video's pending "
              "suggestions are unchanged.")
    else:
        ids = jobs.write_result(pk, result, window=window)
        print(f"Logged {len(ids)} suggestion(s) {list(ids)} — see /video/{pk}")


def cmd_clip_engine_shotlog(args) -> None:
    """Build a video's shot log (or show the cached one) and print it."""
    from app import clip_engine
    from app.clip_engine import engine, shotlog
    from app.config import load_settings
    from app.db import init_db

    init_db()
    settings = load_settings()
    perceiver = args.perceiver or clip_engine.ROLES.get(
        clip_engine.configured_setup(settings), {}).get("perceiver", "gemini")
    model = clip_engine.model_for(settings, perceiver, "perceiver")
    cand = _read_candidate(args.id)
    video_id = cand.get("video_id") or ""
    # Shot logs are cached per perception settings, and which settings apply
    # depends on the source's length (long sources sample at long_video_fps).
    seconds = _duration(cand)

    cached = None if args.force else shotlog.load_cached(video_id, perceiver, model,
                                                          settings=settings, seconds=seconds)
    if cached is not None:
        print(format_shotlog_report(cached, video_id=video_id))
        path = shotlog.cache_path(video_id, perceiver, model, settings=settings, seconds=seconds)
        print(f"\nCached (no new spend): {path}")
        return

    _require_provider(perceiver)
    transcript, words = _load_texts(cand)
    if not transcript and not words:
        print("note: no transcript on this machine; the log will rest on the picture and sound alone")
    meter = engine.CostMeter("engine")
    print(f"Logging {video_id} with {perceiver} ({model})…", flush=True)
    with _inflight_source(cand, key=_inflight_key(video_id or str(cand.get("id")))) as source:
        try:
            shot_log = shotlog.build(
                video_id=video_id, source=source, perceiver=perceiver, settings=settings,
                brand=clip_engine.brand(), title=cand.get("title") or "",
                description=cand.get("description") or "", duration=seconds,
                words=words, transcript=transcript, window=None, meter=meter,
                force=args.force)
        except Exception as exc:
            print(f"Logging failed: {type(exc).__name__}: {exc}")
            print("\n".join(_cost_lines(meter.total, meter.by_stage)))
            raise SystemExit(1) from None
    print(format_shotlog_report(shot_log, video_id=video_id))
    print()
    print("\n".join(_cost_lines(meter.total, meter.by_stage)))
    # The length build() worked with (it probes the file when the row has none).
    logged = shot_log.get("source_seconds") or seconds
    print(f"Cached at {shotlog.cache_path(video_id, perceiver, model, settings=settings, seconds=logged)}")


def _backup_playbook(path: Path) -> Path | None:
    """Copy a playbook about to be overwritten to the engine's playbooks/
    directory (outside the git-tracked config/), so a refresh never loses
    hand edits."""
    if not path.is_file():
        return None
    from app.clip_engine import engine_dir, util

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return util.atomic_copy(path, engine_dir() / "playbooks" / f"clip_playbook-{stamp}.yaml")


def cmd_clip_engine_playbook(args) -> None:
    """Print the playbook the engine runs with, or rebuild it from this
    account's own first airings (``--refresh``: one Claude call)."""
    from app.clip_engine import playbook

    if args.model and not args.refresh:
        raise SystemExit("--model only applies with --refresh.")
    if not args.refresh:
        pb = playbook.load()
        path = playbook.path()
        where = str(path) if path.is_file() else "the built-in default (no config/clip_playbook.yaml)"
        print(f"Playbook: {where} · fingerprint {playbook.fingerprint(pb)}\n")
        print(playbook.render(pb))
        return

    from app.clip_engine import engine
    from app.config import load_settings
    from app.db import init_db, session_scope

    _require_provider("claude")
    init_db()
    settings = load_settings()
    meter = engine.CostMeter("engine")
    print("Reading this account's first airings and asking Claude for a playbook "
          "(a minute or two)…", flush=True)
    try:
        with session_scope() as session:
            try:
                pb = playbook.generate(session, settings, model=args.model, meter=meter)
            finally:
                session.rollback()
    except Exception as exc:
        raise SystemExit(f"Playbook generation failed ({type(exc).__name__}: {exc}); "
                         "the current playbook is unchanged.") from None
    backup = _backup_playbook(playbook.path())
    path = playbook.save(pb)
    print(f"Saved {path} (fingerprint {playbook.fingerprint(pb)}, cost ${meter.total:.4f}).")
    if backup:
        print(f"The previous playbook is at {backup}.")
    print("Bench runs strip the example titles, but the rules were learned from these "
          "posts: judge the engine on videos archived after today (docs/clip-engine.md).\n")
    print(playbook.render(pb))


def _bench():
    """bench.py, with this process's spend booked to the "bench" bucket — kept
    out of the dashboard's daily budget and the engine's own cap."""
    from app import spend
    from app.clip_engine import bench

    spend.set_bucket("bench")
    return bench


def _settings_list(value) -> list[str]:
    if isinstance(value, str):
        return _csv(value)
    return [str(v).strip() for v in (value or []) if str(v).strip()]


def _bench_run_view(bench, name: str) -> dict:
    from app.web.clip_bench import run_view

    for entry in bench.list_runs() or []:
        if isinstance(entry, dict) and str(entry.get("name") or "") == name:
            return run_view(entry)
    raise SystemExit(f"No bench run named {name!r} (see: python run.py clipbench list).")


def cmd_clipbench_run(args) -> None:
    """Replay past videos through each arm and save every result (resumable:
    rerun the same command to continue; finished results are kept).

    Flags left out are passed on as None: a new run takes them from
    clip_engine.bench.*, a resumed one keeps its own arms and cap (bench.py
    decides both). Filling them in here would add the settings' arms to an
    existing run and replace the cap it was given."""
    from app import clip_engine
    from app.config import load_settings
    from app.db import init_db

    bench = _bench()
    settings = load_settings()
    existing = bench.run_manifest(args.name)
    arms = args.arms or ([] if existing else _settings_list(
        clip_engine.opt(settings, "clip_engine.bench.arms", ["gemini", "transcript"])))
    if not arms and not existing:
        raise SystemExit("No arms to run (--arms gemini,transcript).")
    for spec in arms:
        try:
            bench.parse_arm(spec)
        except ValueError as exc:
            raise SystemExit(f"Bad arm {spec!r}: {exc}") from None

    init_db()
    candidate_ids = _resolve_candidate_pks(args.candidate) if args.candidate else None
    if existing:
        print(f"Resuming bench run {args.name!r} (its own videos, arms and cap, unless "
              "--arms or --max-usd say otherwise)…", flush=True)
    else:
        limit = args.limit if args.limit is not None else int(
            clip_engine.opt(settings, "clip_engine.bench.limit", 60))
        max_usd = args.max_usd if args.max_usd is not None else float(
            clip_engine.opt(settings, "clip_engine.bench.max_usd", 40))
        picked = (f"{len(candidate_ids)} hand-picked videos" if candidate_ids
                  else f"up to {limit} videos")
        print(f"Bench run {args.name!r}: arms {', '.join(arms)} · {picked} · seed {args.seed} · "
              f"stops once results pass ${max_usd:.2f}", flush=True)
    try:
        summary = bench.run(
            args.name, arms=args.arms, limit=args.limit, max_usd=args.max_usd, seed=args.seed,
            workers=args.workers, candidate_ids=candidate_ids,
            require_words=not args.allow_no_words, negatives=args.negatives,
            new_version=args.new_version, keep_errors=args.keep_errors,
            settings=settings, log=_say)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished results are kept: run the same command again to resume.")
        raise SystemExit(130) from None
    except ValueError as exc:
        raise SystemExit(f"The bench refused: {exc}") from None
    print("\nDone:")
    _print_summary(summary)
    print(f"\nNext: python run.py clipbench pairs --name {args.name}")


def cmd_clipbench_pairs(args) -> None:
    """Render the blind A/B pairs of a run for the judging page."""
    from app import clip_engine
    from app.config import current_workspace, load_settings

    bench = _bench()
    settings = load_settings()
    height = int(clip_engine.opt(settings, "clip_engine.bench.judge_height", 540))
    try:
        n = bench.prepare_pairs(args.name, height=height)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Couldn't prepare pairs for {args.name!r}: {exc}") from None
    port = current_workspace()["port"]
    print(f"{n} item(s) ready to judge: http://127.0.0.1:{port}/clip-bench?run={args.name} "
          "(Analytics -> Clip bench).")


def cmd_clipbench_report(args) -> None:
    bench = _bench()
    try:
        print(bench.report(args.name))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"No report for {args.name!r}: {exc}") from None


def cmd_clipbench_reveal(args) -> None:
    """Unlock a run's results. Irreversible: judging closes for the run."""
    bench = _bench()
    run = _bench_run_view(bench, args.name)
    if not run["revealed"]:
        if not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit("Revealing closes judging for this run for good; "
                                 "pass --yes to confirm.")
            answer = input(f"{run['judged']} of {run['total']} judged. Reveal {args.name!r} "
                           "and close its judging? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Not revealed.")
                return
        try:
            bench.reveal(args.name)
        except ValueError as exc:
            raise SystemExit(f"The reveal was refused: {exc}") from None
    else:
        print(f"{args.name!r} was already revealed.\n")
    print(bench.report(args.name))


def cmd_clipbench_critic_check(args) -> None:
    """How well does the critic agree with your "gripping?" labels? Runs the
    cold critic over labelled posted clips (about $0.01 a clip on Gemini)."""
    from app.config import load_settings
    from app.db import init_db

    bench = _bench()
    roles = tuple(args.roles or ("gemini",))
    for role in roles:
        _require_provider(role)
    init_db()
    settings = load_settings()
    result = bench.critic_check(args.name, roles=roles, limit=args.limit,
                                settings=settings, log=_say)
    print("\nCritic check:")
    _print_summary(result)


def cmd_clipbench_list(_args) -> None:
    from app.web.clip_bench import run_view

    bench = _bench()
    runs = [run_view(r) for r in (bench.list_runs() or []) if isinstance(r, dict)]
    if not runs:
        print("No bench runs yet. Start one: "
              "python run.py clipbench run --name dev-1 --seed dev --limit 10")
        return
    print(f"{'run':<24} {'seed':<6} {'arms':<34} {'videos':>6} {'judged':>9}  status")
    for r in runs:
        if r["revealed"]:
            status = "revealed"
        elif r["total"] and r["judged"] >= r["total"]:
            status = "ready to reveal"
        else:
            status = "judging" if r["total"] else "needs pairs"
        seed = "picked" if r["mode"] == "hand-picked" else (r["seed"] or "-")
        judged = f"{r['judged']}/{r['total']}"
        print(f"{r['name']:<24} {seed:<6} {_trim(', '.join(r['arms']), 34):<34} "
              f"{r['videos']:>6} {judged:>9}  {status}")


def _is_file(path) -> bool:
    return bool(path) and Path(str(path)).is_file()


def _free_gb_near(path: Path | None) -> float:
    """Free space on the disk that would hold ``path``. media.disk_free_gb
    measures the nearest existing parent (the file and its folder may not
    exist yet) and answers 0.0 when it can't tell, so the guard stops."""
    from app.clip_engine import media
    from app.config import DATA_DIR

    return float(media.disk_free_gb(Path(path) if path else DATA_DIR))


def _missing_word_streams(session) -> list[dict]:
    """Archived videos whose transcript file is on disk but whose Whisper word
    stream isn't. Videos with an exported clip come first (they are the bench's
    pool, and the account's celebrity hits predate word streams), then newest."""
    from sqlalchemy import select

    from app.models import STATUS_ARCHIVED, Candidate, Cut, ThreadsPost

    rows = session.execute(
        select(Candidate.id, Candidate.video_id, Candidate.title, Candidate.transcript_path,
               Candidate.word_transcript_path, Candidate.local_video_path,
               Candidate.source_object_key, Candidate.archived_at)
        .where(Candidate.status == STATUS_ARCHIVED, Candidate.transcript_path != "")).all()
    exported = set(session.execute(
        select(Cut.candidate_pk).where(Cut.trimmed_clip_path != "")).scalars())
    exported |= set(session.execute(
        select(ThreadsPost.candidate_pk).where(ThreadsPost.status == "published",
                                               ThreadsPost.candidate_pk.is_not(None))).scalars())
    out = []
    for r in rows:
        if not _is_file(r.transcript_path) or _is_file(r.word_transcript_path):
            continue
        local = Path(r.local_video_path) if r.local_video_path else None
        out.append({
            "id": int(r.id), "video_id": r.video_id, "title": r.title or "",
            "exported": r.id in exported, "archived_at": r.archived_at,
            "source": ("local" if local is not None and local.is_file()
                       else "storage" if r.source_object_key else "none"),
        })

    def order(item: dict):
        stamp = item["archived_at"].timestamp() if item["archived_at"] else 0.0
        return (not item["exported"], -stamp, -item["id"])

    out.sort(key=order)
    return out


def _add_word_stream(pk: int, settings, *, min_free_gb: float) -> tuple[str, str]:
    """Whisper one video and record its word stream. Returns ``(outcome,
    detail)`` with outcome added | no_words | failed | skipped | stop.

    Words only: the transcript itself (transcript_path, transcript_text) is
    never touched; word_transcript_path is set and transcription_method
    "captions" becomes "captions+whisper". A source this restored from
    storage is deleted again afterwards."""
    from app import scrape
    from app.db import session_scope
    from app.models import Candidate
    from app.offload import SourceUnavailable, ensure_local

    with session_scope() as session:
        c = session.get(Candidate, pk)
        fields = None if c is None else {
            "video_id": c.video_id, "transcript_path": c.transcript_path or "",
            "word_transcript_path": c.word_transcript_path or "",
            "local": c.local_video_path or "", "key": c.source_object_key or ""}
        session.rollback()
    if fields is None or not _is_file(fields["transcript_path"]):
        return "skipped", "its transcript is gone"
    if _is_file(fields["word_transcript_path"]):
        return "skipped", "it has a word stream now"

    local = Path(fields["local"]) if fields["local"] else None
    on_disk = local is not None and local.is_file() and local.stat().st_size > 0
    if on_disk:
        source = local
    else:
        if not (fields["local"] and fields["key"]):
            return "skipped", "no source video on disk or in storage"
        free = _free_gb_near(local)
        if free < min_free_gb:
            return "stop", f"only {free:.1f} GB free (clip_engine.min_free_gb is {min_free_gb:g})"
        try:
            source = ensure_local(fields["local"], fields["key"])
        except SourceUnavailable as exc:
            return "skipped", str(exc)
        except Exception as exc:  # storage/network trouble: this one fails, the batch goes on
            log.warning("Restoring %s failed: %s", fields["video_id"], exc)
            return "failed", f"couldn't restore the source: {type(exc).__name__}: {exc}"
    try:
        _segments, words, tier = scrape.transcribe_word_stream(source, settings)
    except Exception as exc:
        log.warning("Whisper failed for %s: %s", fields["video_id"], exc)
        return "failed", f"Whisper failed: {exc}"
    finally:
        if not on_disk:
            try:
                Path(source).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not delete restored source %s: %s", source, exc)
    if not words:
        return "no_words", "Whisper heard no words"

    try:
        transcript_dir = Path(fields["transcript_path"]).parent
        transcript_dir.mkdir(parents=True, exist_ok=True)
        word_path = scrape._write_word_transcript(words, transcript_dir, fields["video_id"])
        with session_scope() as session:
            c = session.get(Candidate, pk)
            if c is None:
                return "failed", "the video was deleted meanwhile"
            c.word_transcript_path = str(word_path)
            if (c.transcription_method or "") == "captions":
                c.transcription_method = "captions+whisper"
    except Exception as exc:
        log.warning("Saving the word stream of %s failed: %s", fields["video_id"], exc)
        return "failed", f"couldn't save the word stream: {type(exc).__name__}: {exc}"
    return "added", f"{len(words):,} words ({tier})"


def cmd_retranscribe(args) -> None:
    """Add Whisper word streams to archived videos that have a transcript but
    no word file — mostly videos archived before word streams existed, which
    the bench otherwise has to leave out.

    WORDS ONLY. ``scrape.retranscribe_candidate`` is not used: it re-fetches
    YouTube captions and rewrites the transcript, and captions that have since
    vanished would quietly be replaced by Whisper segments. Here the existing
    transcript is never touched. No DB session is open during a download or a
    Whisper run, sources restored from storage are deleted again after use,
    and the batch stops when free disk drops below clip_engine.min_free_gb."""
    from app import clip_engine
    from app.config import load_settings
    from app.db import init_db, session_scope

    init_db()
    settings = load_settings()
    min_free_gb = float(clip_engine.opt(settings, "clip_engine.min_free_gb", 10))
    with session_scope() as session:
        todo = _missing_word_streams(session)
        session.rollback()
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Every archived video with a transcript on this machine has a word stream.")
        return
    if args.dry_run:
        for item in todo:
            mark = " (has an exported clip)" if item["exported"] else ""
            print(f"  {item['id']:>6} {item['video_id']:<12} source: {item['source']:<7} "
                  f"{_trim(item['title'], 60)}{mark}")
        print(f"Would add word streams to {len(todo)} video(s). Nothing changed.")
        return

    counts = {"added": 0, "no_words": 0, "failed": 0, "skipped": 0}
    stopped = ""
    interrupted = False
    try:
        for i, item in enumerate(todo, 1):
            started = time.monotonic()
            _say(f"[{i}/{len(todo)}] {item['video_id']} {_trim(item['title'], 60)}")
            outcome, detail = _add_word_stream(item["id"], settings, min_free_gb=min_free_gb)
            if outcome == "stop":
                stopped = f"{detail}. Free some space and run it again."
                break
            counts[outcome] += 1
            _say(f"    {outcome.replace('_', ' ')}: {detail} ({time.monotonic() - started:.0f} s)")
    except KeyboardInterrupt:
        interrupted = True
        stopped = "interrupted. Finished videos are saved; run it again to continue."
    print(f"\nAdded word streams to {counts['added']} video(s); {counts['no_words']} heard no "
          f"words; {counts['failed']} failed; {counts['skipped']} skipped.")
    if stopped:
        print(f"Stopped early: {stopped}")
    if interrupted:
        raise SystemExit(130)


def _add_clip_engine_parsers(sub, common) -> None:
    """``clip-engine``, ``clipbench`` and ``retranscribe``. ``parents=[common]``
    sits on the leaf parsers only: on a group AND its leaf, the leaf's default
    would overwrite a --workspace given at group level with None."""
    p = sub.add_parser("clip-engine",
                       help="try the clip engine on one video; inspect shot logs and the playbook")
    esub = p.add_subparsers(dest="engine_command", required=True)

    ep = esub.add_parser("try", parents=[common],
                         help="run the engine on one archived video and print every draft "
                              "(writes nothing unless --log)")
    ep.add_argument("id", help="candidate id or YouTube video id")
    ep.add_argument("--setup", choices=_ENGINE_SETUPS, default=None,
                    help="engine setup (default: clip_engine.setup)")
    ep.add_argument("--section", nargs=2, type=_parse_time, metavar=("START", "END"), default=None,
                    help="suggest inside this stretch only, like 'Suggest a clip here' "
                         "(seconds or m:ss)")
    ep.add_argument("--log", action="store_true",
                    help="save the top picks as the video's pending suggestions, like Re-suggest")
    ep.set_defaults(func=cmd_clip_engine_try)

    ep = esub.add_parser("shotlog", parents=[common],
                         help="build (or show the cached) shot log of a video")
    ep.add_argument("id", help="candidate id or YouTube video id")
    ep.add_argument("--perceiver", choices=_CRITIC_ROLES, default=None,
                    help="who logs the video (default: the configured setup's perceiver)")
    ep.add_argument("--force", action="store_true",
                    help="log it again even when a cached log exists (paid again)")
    ep.set_defaults(func=cmd_clip_engine_shotlog)

    ep = esub.add_parser("playbook", parents=[common],
                         help="print the clip playbook; --refresh rebuilds it from your posts")
    ep.add_argument("--refresh", action="store_true",
                    help="regenerate config/clip_playbook.yaml with Claude (one call; the old "
                         "file is backed up)")
    ep.add_argument("--model", default=None, metavar="M",
                    help="model for --refresh (default: the Claude editor model)")
    ep.set_defaults(func=cmd_clip_engine_playbook)

    p = sub.add_parser("clipbench", help="replay benchmark for the clip engine (docs/clip-engine.md)")
    bsub = p.add_subparsers(dest="bench_command", required=True)

    bp = bsub.add_parser("run", parents=[common],
                         help="replay past videos through each arm (resumable)")
    bp.add_argument("--name", required=True, type=_run_name, metavar="N",
                    help="run name (a folder under data/clipbench)")
    bp.add_argument("--arms", type=_csv, default=None, metavar="A,B",
                    help="comma list of arms, e.g. gemini,transcript or "
                         "gemini+editor_thinking=HIGH (default: clip_engine.bench.arms for a "
                         "new run, the run's own arms on a resume; given on a resume, only "
                         "these run, added to the run if new)")
    bp.add_argument("--limit", type=_int_at_least(1), default=None, metavar="N",
                    help="videos to sample (default: clip_engine.bench.limit; fixed once the "
                         "run exists)")
    bp.add_argument("--max-usd", type=_positive_float, default=None, metavar="USD",
                    help="stop after the video whose results pass this spend "
                         "(default: clip_engine.bench.max_usd for a new run; a resumed run "
                         "keeps its own cap unless this replaces it)")
    bp.add_argument("--seed", choices=("test", "dev"), default="test",
                    help="'dev' while tuning; 'test' (default) for the decision: it only draws "
                         "videos no try or earlier run has touched")
    bp.add_argument("--workers", type=_int_at_least(1), default=1, metavar="N",
                    help="videos replayed at once (default 1)")
    bp.add_argument("--candidate", action="extend", nargs="+", default=None, metavar="ID",
                    help="hand-pick videos (candidate or YouTube ids) instead of sampling; "
                         "a hand-picked run is never the benchmark")
    bp.add_argument("--allow-no-words", action="store_true",
                    help="also sample videos without a Whisper word stream")
    bp.add_argument("--negatives", type=_int_at_least(0), default=0, metavar="K",
                    help="add K archived videos you never cut, judged 'would you post this?'")
    bp.add_argument("--new-version", action="store_true",
                    help="after prompts or settings changed: start every arm over on all the "
                         "run's videos under new labels. The cap counts what the run already "
                         "spent (raise --max-usd by about a full run's cost), and the pairs "
                         "must be prepared and judged again")
    bp.add_argument("--keep-errors", action="store_true",
                    help="don't retry videos whose earlier attempt errored")
    bp.set_defaults(func=cmd_clipbench_run)

    bp = bsub.add_parser("pairs", parents=[common],
                         help="render a run's blind A/B pairs for the judging page")
    bp.add_argument("--name", required=True, type=_run_name, metavar="N")
    bp.set_defaults(func=cmd_clipbench_pairs)

    bp = bsub.add_parser("report", parents=[common],
                         help="print a run's report (results only after the reveal)")
    bp.add_argument("--name", required=True, type=_run_name, metavar="N")
    bp.set_defaults(func=cmd_clipbench_report)

    bp = bsub.add_parser("reveal", parents=[common],
                         help="unlock a run's results and close its judging (irreversible)")
    bp.add_argument("--name", required=True, type=_run_name, metavar="N")
    bp.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    bp.set_defaults(func=cmd_clipbench_reveal)

    bp = bsub.add_parser("critic-check", parents=[common],
                         help="run the critic over your labelled posted clips and compare "
                              "with your 'gripping?' labels")
    bp.add_argument("--roles", type=_critic_roles, default=None, metavar="R,R",
                    help="critics to check: gemini, claude (default: gemini)")
    bp.add_argument("--limit", type=_int_at_least(1), default=40, metavar="N",
                    help="clips to check (default 40)")
    bp.add_argument("--name", type=_run_name, default=None, metavar="N",
                    help="file the check under this bench run")
    bp.set_defaults(func=cmd_clipbench_critic_check)

    bp = bsub.add_parser("list", parents=[common], help="bench runs and their judging progress")
    bp.set_defaults(func=cmd_clipbench_list)

    p = sub.add_parser("retranscribe", parents=[common],
                       help="add Whisper word streams to archived videos that lack them "
                            "(transcripts untouched)")
    p.add_argument("--missing-words", action="store_true", required=True,
                   help="videos with a transcript but no word-stream file (the only mode)")
    p.add_argument("--limit", type=_int_at_least(1), default=None, metavar="N",
                   help="process at most N videos")
    p.add_argument("--dry-run", action="store_true", help="list them, change nothing")
    p.set_defaults(func=cmd_retranscribe)


def build_parser() -> argparse.ArgumentParser:
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

    p = sub.add_parser("placement-sim", parents=[common],
                       help="replay publish history through scored placement "
                            "(read-only; tune weights before flipping the mode)")
    p.add_argument("--days", type=int, default=90,
                   help="how far back to replay (default 90)")
    p.set_defaults(func=cmd_placement_sim)
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

    sub.add_parser("triage-train", parents=[common],
                   help="retrain the approve predictor and print its backtest"
                   ).set_defaults(func=cmd_triage_train)
    sub.add_parser("metrics", parents=[common],
                   help="snapshot Threads post metrics").set_defaults(func=cmd_metrics)
    sub.add_parser("comments", parents=[common],
                   help="sync comments on own posts").set_defaults(func=cmd_comments)
    sub.add_parser("digest", parents=[common],
                   help="write a fresh analytics digest").set_defaults(func=cmd_digest)
    sub.add_parser("cleanup", parents=[common],
                   help="apply retention setting to downloaded segments").set_defaults(func=cmd_cleanup)
    p = sub.add_parser("offload", parents=[common],
                       help="move finished source videos and clips to Supabase storage")
    p.add_argument("--dry-run", action="store_true", help="list what would happen, change nothing")
    p.add_argument("--limit", type=int, default=None, help="act on at most this many files")
    p.set_defaults(func=cmd_offload)

    _add_clip_engine_parsers(sub, common)
    return parser


def main() -> None:
    parser = build_parser()
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
