"""Local review dashboard (FastAPI + Jinja). Single-operator, localhost only.

Run: python run.py dashboard   (serves http://127.0.0.1:8321)

Workflow per video: Review → download/transcribe → Trim → Post. The /video/{id}
screen is a profile (player + transcript, clips in the rail); trimming happens
on /cut/{id}.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import selectinload

from .. import spend, threads_api, youtube
from ..analytics import generate_report, snapshot_metrics
from ..categories import category_by_slug, category_options
from ..clipper import ClipExportError, cached_still, clip_duration, export_supercut, get_waveform
from ..config import (
    load_caption_rules, load_first_reply, load_keywords, load_settings,
    render_caption_guide, save_caption_rules, save_first_reply, save_keywords,
)
from ..db import (
    SessionLocal,
    active_traits,
    init_db,
    invalidate_traits_cache,
    session_scope,
    sync_channels_from_config,
    sync_traits_from_config,
)
from ..engagement import PacingLimitError, post_approved_reply, sync_comments
from ..giphy import is_configured as giphy_configured
from ..history import import_history
from ..llm import (
    suggest_calendar_name,
    suggest_channel_fields,
    suggest_post_caption,
    suggest_short_title,
    suggest_title,
)
from ..logging_setup import setup_logging
from ..models import (
    STATUS_APPROVED,
    STATUS_ARCHIVED,
    STATUS_NEW,
    STATUS_REJECTED,
    Candidate,
    Channel,
    Cut,
    MetricSnapshot,
    MonitorRun,
    SchedulerState,
    ThreadsComment,
    ThreadsPost,
    Trait,
    TraitWeight,
    TriageDecision,
    utcnow,
)
from ..monitor import run_monitor_once
from ..publishing import (
    clear_publishing,
    generate_attribution,
    mark_publishing,
    maybe_post_first_reply,
    publish_clip,
    publish_post,
    queue_clip,
    record_post,
)
from ..ranking import load_trait_weights, order_expr, sort_candidates
from ..scheduler import (
    build_window_plan,
    pin_post_to_window,
    projected_slot_for_post,
    scheduler_status,
    spacing_allows_publish,
    start_scheduler_thread,
    window_time_labels,
)
from ..scrape import PASTED_CHANNEL_URL, archive_candidate, fetch_video_metadata
from ..vision import annotate_post_footage, tag_candidate_storyboard
from ..voice import voice_context
from ..youtube import YouTubeAPIError, parse_video_url

log = logging.getLogger("web")

app = FastAPI(title="Climate Clip Monitor")
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
# Cache-bust static assets whenever style.css changes so browsers pick up edits.
# Evaluated lazily on every render (via __str__) so CSS-only edits bust the cache
# immediately, without needing an app restart.
class _StaticVersion:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __str__(self) -> str:
        try:
            return str(int(self._path.stat().st_mtime))
        except OSError:
            return "0"


templates.env.globals["static_v"] = _StaticVersion(_STATIC_DIR / "style.css")
# Programming-category vocabulary, available to every template (badge/picker
# rendering). Both are cheap: backed by a short in-process cache.
templates.env.globals["category_options"] = category_options
templates.env.globals["category_by_slug"] = category_by_slug


def _thumb_for(candidate) -> str:
    """Poster image for a candidate, or "" when there is nothing to show.

    Prefers the source thumbnail (a remote YouTube still, free to serve) and
    falls back to a frame cut out of the local file, which is all uploads and
    pasted URLs have. Callers pass possibly-None candidates — a post can hang
    off a cut whose candidate was pruned — so None is a normal input here.
    """
    if candidate is None:
        return ""
    if candidate.thumbnail_url:
        return candidate.thumbnail_url
    return f"/media/thumb/{candidate.id}" if candidate.local_video_path else ""


templates.env.globals["thumb_for"] = _thumb_for


def _attention_count() -> int:
    """Number of unacknowledged failed posts. Rendered on every page (the sidebar
    notification bell), so it must never raise — return 0 on any error."""
    try:
        with session_scope() as session:
            return int(session.execute(
                select(func.count()).select_from(ThreadsPost).where(
                    ThreadsPost.status == "failed",
                    ThreadsPost.attention_dismissed_at.is_(None),
                )
            ).scalar_one())
    except Exception:
        return 0


templates.env.globals["attention_count"] = _attention_count


def _nav_scheduler_status() -> dict:
    """Slim scheduler snapshot for the persistent sidebar widget. Rendered on
    every page, so it must never raise — return an empty dict on any error."""
    try:
        with session_scope() as session:
            return scheduler_status(session)
    except Exception:
        return {}


templates.env.globals["nav_scheduler"] = _nav_scheduler_status


def _threads_authenticated() -> bool:
    """Rendered on every page (sidebar nav dot), so it must never raise."""
    try:
        return threads_api.is_authenticated()
    except Exception:
        return False


templates.env.globals["threads_authenticated"] = _threads_authenticated

init_db()
with session_scope() as _s:
    sync_channels_from_config(_s)
    sync_traits_from_config(_s)
    # A monitor pass runs in an in-process thread, so any run still marked
    # "running" here was killed by a restart/crash — reconcile it to
    # "interrupted" so the dashboard doesn't show a spinner that never resolves.
    for _run in _s.execute(
        select(MonitorRun).where(MonitorRun.status == MonitorRun.STATUS_RUNNING)
    ).scalars().all():
        _run.status = MonitorRun.STATUS_INTERRUPTED
        _run.finished_at = utcnow()
        if not _run.result:
            _run.result = "Interrupted — the server restarted while the pass was running."

# Adaptive window scheduler (queue + hotness + metrics poll) while the dashboard runs.
# Configure logging first: this module is the uvicorn worker's entry point, so
# without it the scheduler's own log output has nowhere to go.
setup_logging()
start_scheduler_thread()


def _flash(url: str, msg: str) -> RedirectResponse:
    sep = "&" if "?" in url else "?"
    return RedirectResponse(f"{url}{sep}msg={msg}", status_code=303)


def _scrape_in_thread(candidate_id: int) -> None:
    session = SessionLocal()
    try:
        candidate = session.get(Candidate, candidate_id)
        if candidate:
            archive_candidate(session, candidate)
            session.commit()
    except Exception:
        session.rollback()
        log.exception("Background scrape failed for candidate %s", candidate_id)
    finally:
        session.close()


def _resume_stalled_scrapes() -> None:
    """Restart downloads left in ``approved`` after a server restart.

    Approve kicks off an in-process thread; a restart kills it and leaves the
    row looking like it's still downloading forever. Resume those here.
    """
    with session_scope() as session:
        stalled = session.execute(
            select(Candidate.id).where(Candidate.status == STATUS_APPROVED)
        ).scalars().all()
    for cid in stalled:
        log.info("Resuming stalled scrape for candidate %s", cid)
        threading.Thread(target=_scrape_in_thread, args=(cid,), daemon=True).start()


_resume_stalled_scrapes()


# --- Workflow step helpers ----------------------------------------------------

def workflow_state(session, c: Candidate, post_statuses: set[str] | None = None,
                   has_exported_cut: bool | None = None) -> dict:
    """Compute breadcrumb step states for a candidate.

    Pass ``post_statuses`` (the set of ThreadsPost.status values for this
    candidate) to skip the per-row published-post lookup, and
    ``has_exported_cut`` (whether the video has at least one cut with an
    exported clip) to skip a per-row cuts lookup — used by the dashboard/library
    when rendering many rows against a remote DB.
    """
    if post_statuses is None:
        posted = session.execute(
            select(ThreadsPost.id).where(
                ThreadsPost.candidate_pk == c.id, ThreadsPost.status == "published"
            ).limit(1)
        ).scalar_one_or_none() is not None
    else:
        posted = "published" in post_statuses

    if has_exported_cut is None:
        trimmed = session.execute(
            select(Cut.id).where(
                Cut.candidate_pk == c.id, Cut.trimmed_clip_path != ""
            ).limit(1)
        ).scalar_one_or_none() is not None
    else:
        trimmed = has_exported_cut

    reviewed = c.status not in (STATUS_NEW, STATUS_REJECTED)
    scraped = c.status == STATUS_ARCHIVED

    if not reviewed:
        current = "review"
    elif not scraped:
        current = "scrape"
    elif not trimmed:
        current = "trim"
    else:
        current = "post"

    return {
        "reviewed": reviewed, "scraped": scraped, "trimmed": trimmed, "posted": posted,
        "current": current,
    }


def _post_statuses_by_candidate(session, candidate_ids: list[int]) -> dict[int, set[str]]:
    """One query: candidate_pk -> set of ThreadsPost.status values."""
    if not candidate_ids:
        return {}
    rows = session.execute(
        select(ThreadsPost.candidate_pk, ThreadsPost.status).where(
            ThreadsPost.candidate_pk.in_(candidate_ids)
        )
    ).all()
    out: dict[int, set[str]] = {}
    for pk, status in rows:
        if pk is None:
            continue
        out.setdefault(pk, set()).add(status)
    return out


def _exported_cut_candidate_ids(session, candidate_ids: list[int]) -> set[int]:
    """Candidate ids that have at least one cut with an exported clip (one query)."""
    if not candidate_ids:
        return set()
    rows = session.execute(
        select(Cut.candidate_pk).where(
            Cut.candidate_pk.in_(candidate_ids), Cut.trimmed_clip_path != ""
        ).distinct()
    ).all()
    return {pk for (pk,) in rows if pk is not None}


# --- Dashboard -----------------------------------------------------------------

def _parse_date(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except (ValueError, TypeError):
        return None


def _relative_time_ago(when: dt.datetime | None) -> str:
    """Human-friendly relative string, e.g. "5 minutes ago" / "just now" / "never"."""
    if when is None:
        return "never"
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    secs = (dt.datetime.now(dt.timezone.utc) - when).total_seconds()
    if secs < 45:
        return "just now"
    minutes = secs / 60
    if minutes < 45:
        n = max(1, round(minutes))
        return f"{n} minute{'s' if n != 1 else ''} ago"
    hours = secs / 3600
    if hours < 24:
        n = round(hours)
        return f"{n} hour{'s' if n != 1 else ''} ago"
    days = secs / 86400
    if days < 7:
        n = max(1, round(days))
        return f"{n} day{'s' if n != 1 else ''} ago"
    weeks = round(days / 7)
    if weeks < 5:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    return when.strftime("%b %-d")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, q: str = "", channel_id: int = 0,
              keyword: list[str] = Query(default=[]),
              region: str = "", country: str = "", scope: str = "",
              status: str = "new", date_from: str = "", date_to: str = "",
              show_hidden: int = 0, msg: str = ""):
    settings = load_settings()
    threshold = settings.get("matching.score_threshold", 0.5)
    keyword = [k for k in keyword if k.strip()]
    filtering = bool(q or channel_id or keyword or region or country or scope
                     or date_from or date_to or status != "new")

    # On a bare visit (no query string), default the view to today + yesterday by
    # publish date. Any filter interaction submits a query string and is respected
    # as-is, so the operator can widen the window or clear it entirely.
    date_defaulted = not request.query_params
    if date_defaulted:
        window_days = settings.get("monitor.default_lookback_days", 2)
        today = dt.datetime.now(dt.timezone.utc).date()
        date_from = (today - dt.timedelta(days=max(window_days, 1) - 1)).isoformat()
        date_to = today.isoformat()

    with session_scope() as session:
        # Order by the blended relevance+visual ranking so the row cap keeps the
        # top-ranked candidates (not just the most relevant).
        query = (
            select(Candidate)
            .options(selectinload(Candidate.channel))
            .order_by(order_expr(settings).desc(), Candidate.published_at.desc())
        )
        if status != "all":
            query = query.where(Candidate.status == status)
        if q:
            like = f"%{q}%"
            query = query.where(
                Candidate.title.ilike(like)
                | Candidate.description.ilike(like)
                | Candidate.matched_keywords.ilike(like)
            )
        if channel_id:
            query = query.where(Candidate.channel_pk == channel_id)
        if keyword:
            # matched_keywords is a CSV list; match rows containing ANY selected keyword.
            query = query.where(or_(
                *[("," + Candidate.matched_keywords + ",").like(f"%,{k},%") for k in keyword]
            ))
        channel_filters = []
        if region:
            channel_filters.append(Channel.region == region)
        if country:
            channel_filters.append(Channel.country == country)
        if scope:
            channel_filters.append(Channel.scope == scope)
        if channel_filters:
            query = query.join(Channel, Candidate.channel_pk == Channel.id).where(*channel_filters)
        start = _parse_date(date_from)
        if start:
            query = query.where(Candidate.published_at >= start)
        end = _parse_date(date_to)
        if end:
            query = query.where(Candidate.published_at < end + dt.timedelta(days=1))
        if status == "new" and not show_hidden:
            query = query.where(
                (Candidate.relevance_score.is_(None)) | (Candidate.relevance_score >= threshold)
            )
        # Total matching the current filters, before the render cap below.
        total_matches = session.execute(
            select(func.count()).select_from(query.order_by(None).subquery())
        ).scalar_one()
        row_cap = 150
        candidates = session.execute(query.limit(row_cap)).scalars().all()
        # Re-rank by relevance, nudged by ACTIVE trait verdicts once unlocked.
        trait_weights = load_trait_weights(session)
        candidates = sort_candidates(candidates, trait_weights, settings)

        # Keyword filter chips come from the active keyword list (what we monitor
        # for), so removed/legacy terms never show up as filters.
        keywords_options = sorted(load_keywords())

        # Items mid-workflow (shown on the default view only), split into two
        # buckets: clips still awaiting a trim, and clips already trimmed but
        # never posted (a supercut was exported, yet nothing was published).
        in_progress_rows = []   # selected clips still needing a trim
        trimmed_rows = []       # trimmed clips that were never posted
        if not filtering:
            in_progress = session.execute(
                select(Candidate)
                .options(selectinload(Candidate.channel))
                .where(Candidate.status.in_([STATUS_APPROVED, STATUS_ARCHIVED, "failed"]))
                .order_by(Candidate.approved_at.desc())
                .limit(30)
            ).scalars().all()
            # One query for all post statuses instead of 2×N per-row lookups.
            ip_ids = [c.id for c in in_progress]
            statuses = _post_statuses_by_candidate(session, ip_ids)
            exported = _exported_cut_candidate_ids(session, ip_ids)
            handled_statuses = {"published", "queued", "publishing", "draft"}
            for c in in_progress:
                post_st = statuses.get(c.id, set())
                state = workflow_state(session, c, post_statuses=post_st,
                                       has_exported_cut=c.id in exported)
                # A candidate with a FAILED post stays visible so it can be
                # retried; it's already trimmed, so it belongs in "Trimmed".
                if "failed" in post_st:
                    state["post_failed"] = True
                    trimmed_rows.append((c, state))
                    continue
                # The multi-clip marker pins the video to "Selected to trim":
                # more clips are expected, so exports/handled posts don't move
                # it along until the operator toggles the marker off.
                if c.multi_clip_potential:
                    in_progress_rows.append((c, state))
                    continue
                # Hide once published or sitting in the outbound queue/drafts.
                # If the operator deletes their only draft/queue, post_st is
                # empty and the clip stays visible so they can re-post.
                if post_st & handled_statuses:
                    continue
                # "post" = supercut exported but nothing published yet.
                if state["current"] == "post":
                    trimmed_rows.append((c, state))
                else:
                    in_progress_rows.append((c, state))

        monitor_running, monitor_result, monitor_last_refreshed = _monitor_view_state(session)

    return templates.TemplateResponse(
        request, "dashboard.html",
        {"candidates": candidates, "total_matches": total_matches, "row_cap": row_cap,
         "in_progress": in_progress_rows, "trimmed": trimmed_rows, "threshold": threshold,
         "date_defaulted": date_defaulted,
         "show_hidden": show_hidden, "filtering": filtering,
         "q": q, "channel_id": channel_id, "keyword": keyword, "region": region,
         "country": country, "scope": scope, "status": status,
         "date_from": date_from, "date_to": date_to,
         "keywords_options": keywords_options,
         "monitor_running": monitor_running,
         "monitor_result": monitor_result,
         "monitor_last_refreshed": monitor_last_refreshed,
         "msg": msg, "active": "dashboard"},
    )


# Monitor passes run in a background thread. Progress is persisted to the
# MonitorRun table (durable across refreshes and restarts); this in-memory flag
# only guards against starting a second pass within the SAME process.
_monitor_running = threading.Event()


def _monitor_in_thread(run_id: int, days: int | None) -> None:
    try:
        result = run_monitor_once(days)
        summary = (
            f"{result['channels_checked']} channels checked, "
            f"{result['candidates_stored']} new candidates, "
            f"{result.get('vision_scored', 0)} vision-scored "
            f"(spent ${spend.today_spend():.2f} of ${spend.daily_budget():.2f} today)"
        )
        with session_scope() as session:
            run = session.get(MonitorRun, run_id)
            if run is not None:
                run.status = MonitorRun.STATUS_DONE
                run.channels_checked = result["channels_checked"]
                run.candidates_stored = result["candidates_stored"]
                run.vision_scored = result.get("vision_scored", 0)
                run.result = summary
                run.finished_at = utcnow()
    except Exception as exc:
        log.exception("Monitor pass failed")
        with session_scope() as session:
            run = session.get(MonitorRun, run_id)
            if run is not None:
                run.status = MonitorRun.STATUS_FAILED
                run.error = str(exc)
                run.result = f"Monitor pass failed: {exc}"
                run.finished_at = utcnow()
    finally:
        _monitor_running.clear()


def _latest_monitor_run(session) -> MonitorRun | None:
    return session.execute(
        select(MonitorRun).order_by(MonitorRun.started_at.desc()).limit(1)
    ).scalar_one_or_none()


def _monitor_view_state(session) -> tuple[bool, str, str]:
    """(running, message, last_refreshed) for the dashboard, from durable run state.

    ``running`` is true only when a pass is genuinely in flight in THIS process,
    so a restart can never leave a spinner stuck on. ``last_refreshed`` is a
    human-friendly relative string ("5 minutes ago" / "just now" / "never").
    """
    run = _latest_monitor_run(session)
    running = _monitor_running.is_set() and run is not None and run.status == MonitorRun.STATUS_RUNNING
    if run is None:
        return running, "", "never"
    when = run.finished_at or run.started_at
    last_refreshed = _relative_time_ago(when)
    stamp = when.strftime("%b %-d %H:%M") if when else ""
    scope = run.scope or "since last check"
    if run.status == MonitorRun.STATUS_DONE:
        return running, f"Last pass ({scope}, {stamp}): {run.result}", last_refreshed
    if run.status == MonitorRun.STATUS_FAILED:
        return running, run.result or "Last pass failed.", last_refreshed
    if run.status == MonitorRun.STATUS_INTERRUPTED:
        return running, (
            f"Last pass ({scope}, started {stamp}) was interrupted by a server "
            "restart. Run the monitor again to finish checking."
        ), last_refreshed
    return running, "", last_refreshed  # currently running: badge is shown instead


@app.post("/monitor/run")
def monitor_now(lookback_days: str = Form("")):
    if _monitor_running.is_set():
        return _flash("/", "A monitor pass is already running — refresh to see progress")
    days: int | None = None
    if lookback_days.strip():
        try:
            days = max(1, min(int(lookback_days), 30))
        except ValueError:
            days = None
    scope = f"last {days} days" if days else "since last check"
    with session_scope() as session:
        run = MonitorRun(status=MonitorRun.STATUS_RUNNING, scope=scope, lookback_days=days)
        session.add(run)
        session.flush()
        run_id = run.id
    _monitor_running.set()
    threading.Thread(target=_monitor_in_thread, args=(run_id, days), daemon=True).start()
    verb = f"backfilling {days} days" if days else "checking since last run"
    return _flash("/", f"Monitor started ({verb}) — running in the background, refresh for updates")


# --- Triage mode (one at a time, keyboard-driven) --------------------------------

@app.get("/triage", response_class=HTMLResponse)
def triage(request: Request, q: str = "", channel_id: int = 0,
           keyword: list[str] = Query(default=[]),
           region: str = "", country: str = "", scope: str = "",
           date_from: str = "", date_to: str = "",
           show_hidden: int = 0, msg: str = ""):
    """Focused review: new candidates one at a time with keyboard actions."""
    settings = load_settings()
    threshold = settings.get("matching.score_threshold", 0.5)
    keyword = [k for k in keyword if k.strip()]

    with session_scope() as session:
        query = (
            select(Candidate)
            .where(Candidate.status == STATUS_NEW)
            .order_by(order_expr(settings).desc(), Candidate.published_at.desc())
        )
        if q:
            like = f"%{q}%"
            query = query.where(
                Candidate.title.ilike(like)
                | Candidate.description.ilike(like)
                | Candidate.matched_keywords.ilike(like)
            )
        if channel_id:
            query = query.where(Candidate.channel_pk == channel_id)
        if keyword:
            query = query.where(or_(
                *[("," + Candidate.matched_keywords + ",").like(f"%,{k},%") for k in keyword]
            ))
        channel_filters = []
        if region:
            channel_filters.append(Channel.region == region)
        if country:
            channel_filters.append(Channel.country == country)
        if scope:
            channel_filters.append(Channel.scope == scope)
        if channel_filters:
            query = query.join(Channel, Candidate.channel_pk == Channel.id).where(*channel_filters)
        start = _parse_date(date_from)
        if start:
            query = query.where(Candidate.published_at >= start)
        end = _parse_date(date_to)
        if end:
            query = query.where(Candidate.published_at < end + dt.timedelta(days=1))
        if not show_hidden:
            query = query.where(
                (Candidate.relevance_score.is_(None)) | (Candidate.relevance_score >= threshold)
            )
        candidates = session.execute(
            query.options(selectinload(Candidate.channel)).limit(200)
        ).scalars().all()
        trait_weights = load_trait_weights(session)
        candidates = sort_candidates(candidates, trait_weights, settings)
        total_published = session.execute(
            select(func.count(ThreadsPost.id)).where(ThreadsPost.status == "published")
        ).scalar_one()
        queue = [
            {
                "id": c.id,
                "video_id": c.video_id,
                "title": c.title,
                "channel": f"{c.channel.call_sign} — {c.channel.market}",
                "published": c.published_at.strftime("%b %d, %Y %H:%M UTC") if c.published_at else "?",
                "duration": (f"{c.duration_seconds // 60}m {c.duration_seconds % 60}s"
                             if c.duration_seconds else ""),
                "score": c.relevance_score,
                "visual_traits": [t for t in (c.visual_traits or "").split(",") if t],
                "visual_rationale": c.visual_rationale,
                "keywords": [k for k in (c.matched_keywords or "").split(",") if k],
            }
            for c in candidates
        ]

    return templates.TemplateResponse(
        request, "triage.html",
        {"queue": queue, "threshold": threshold,
         "trait_stats": trait_weights,
         "learning_min_trait": settings.get("learning.min_trait_posts", 20),
         "learning_min_total": settings.get("learning.min_total_posts", 100),
         "total_published": total_published,
         "msg": msg, "active": "dashboard"},
    )


@app.get("/video/{candidate_id}/waveform")
def video_waveform(candidate_id: int):
    """Audio peak envelope of the downloaded file, for the trim editor."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.local_video_path or not Path(c.local_video_path).exists():
            return JSONResponse({"error": "no local video"}, status_code=404)
        path, vid = c.local_video_path, c.video_id
    try:
        return get_waveform(path, vid)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/video/{candidate_id}/storyboard")
def video_storyboard(candidate_id: int):
    """Filmstrip data for triage: YouTube's own scrub-preview sprite sheets
    (metadata fetch only — nothing is downloaded)."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"available": False}, status_code=404)
        video_id = c.video_id
    from ..storyboard import get_storyboard
    return get_storyboard(video_id)


@app.post("/video/{candidate_id}/score-visuals")
def video_score_visuals(candidate_id: int):
    """On-demand storyboard trait tagging for one candidate (budget-guarded).
    Neutral labels only — no visual score."""
    settings = load_settings()
    if not spend.within_budget():
        return JSONResponse(
            {"error": f"Daily LLM budget of ${spend.daily_budget():.2f} reached "
                      f"(spent ${spend.today_spend():.2f}). Try again tomorrow or raise "
                      f"llm.daily_budget_usd."},
            status_code=429,
        )
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        traits = active_traits(session)
        result = tag_candidate_storyboard(c, settings, force=True, traits=traits)
        if result is None:
            return JSONResponse(
                {"error": "No storyboard available for this video, or tagging failed."},
                status_code=502,
            )
        return {"traits": result["traits"], "why": result["why"]}


@app.post("/video/{candidate_id}/category")
def set_video_category(candidate_id: int, category: str = Form(""), next: str = Form("")):
    """Operator sets (or clears) the programming category by hand. The picker
    lives on the video, cut, and post pages; ``next`` returns to whichever one
    submitted the form."""
    dest = next if next.startswith("/") else f"/video/{candidate_id}"
    category = category.strip().lower()
    cat = category_by_slug(category)
    if category and cat is None:
        return _flash(dest, "Unknown category")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/", "Video not found")
        c.category = category
        c.category_rationale = ""  # operator's own choice needs no LLM rationale
    return _flash(dest,
                  f"Category set to {cat['emoji']} {cat['label']}" if cat else "Category cleared")


# --- Per-video workflow ----------------------------------------------------------

def _cut_state(cut: Cut, posted_cut_pks: set[int]) -> dict:
    """Per-cut status for the video page cut list."""
    exported = bool(cut.trimmed_clip_path) and Path(cut.trimmed_clip_path).exists()
    return {
        "exported": exported,
        "posted": cut.id in posted_cut_pks,
        "captioned": bool(cut.subtitled_clip_path),
    }


@app.get("/video/{candidate_id}", response_class=HTMLResponse)
def video_detail(request: Request, candidate_id: int, step: str = "", msg: str = ""):
    """Video profile: player + searchable transcript, clips/posts in the rail.

    ``step`` is accepted for old bookmarks (``?step=cuts`` etc.) but ignored —
    the page is no longer a Review → Scrape → Clips wizard.
    """
    with session_scope() as session:
        c = session.execute(
            select(Candidate)
            .options(selectinload(Candidate.channel), selectinload(Candidate.cuts))
            .where(Candidate.id == candidate_id)
        ).scalar_one_or_none()
        if c is None:
            return _flash("/", "Video not found")

        cuts = list(c.cuts)
        has_exported_cut = any(cut.trimmed_clip_path for cut in cuts)
        state = workflow_state(session, c, has_exported_cut=has_exported_cut)

        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.candidate_pk == c.id)
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
        posted_cut_pks = {p.cut_pk for p in posts
                          if p.status == "published" and p.cut_pk is not None}
        posts_by_cut: dict[int, int] = {}
        for p in posts:
            if p.cut_pk is not None:
                posts_by_cut[p.cut_pk] = posts_by_cut.get(p.cut_pk, 0) + 1

        cut_rows = [
            {"cut": cut, "state": _cut_state(cut, posted_cut_pks),
             "post_count": posts_by_cut.get(cut.id, 0)}
            for cut in cuts
        ]

        transcript_segments = []
        if c.transcript_path and Path(c.transcript_path).exists():
            try:
                transcript_segments = json.loads(Path(c.transcript_path).read_text())
            except Exception:
                transcript_segments = []

        has_local = bool(c.local_video_path and Path(c.local_video_path).exists())

    return templates.TemplateResponse(
        request, "video.html",
        {"c": c, "state": state, "step": step,
         "cut_rows": cut_rows, "posts": posts,
         "transcript_segments": transcript_segments,
         "has_local": has_local,
         "msg": msg, "active": "dashboard"},
    )


# --- Cuts (first-class trimmed clips) ---------------------------------------------

@app.post("/video/{candidate_id}/cuts")
def create_cut(candidate_id: int):
    """Start a new cut for a video and jump straight into its trim editor."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/", "Video not found")
        if c.status != STATUS_ARCHIVED:
            return _flash(f"/video/{candidate_id}", "Download the video before clipping it")
        # No caption seed: the video-level draft was written against the FULL
        # transcript. The caption is proposed after export, from the trimmed
        # clip's own transcript, and only adopted when the operator accepts it.
        cut = Cut(candidate_pk=c.id)
        session.add(cut)
        session.flush()
        cut_id = cut.id
    return _flash(f"/cut/{cut_id}?step=trim", "New clip — pick your segments")


@app.post("/video/{candidate_id}/multi-clip")
def toggle_multi_clip(candidate_id: int):
    """Flip the video's "multiple clips potential" marker (AJAX). While on, the
    video stays in the dashboard's "Selected to trim" bucket regardless of
    exports or queued/published posts."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"error": "Video not found"}, status_code=404)
        c.multi_clip_potential = not bool(c.multi_clip_potential)
        flag = c.multi_clip_potential
    return JSONResponse({"multi_clip": flag})


@app.get("/video/{candidate_id}/cut")
def open_cut(candidate_id: int):
    """Jump straight into the trim editor for a video.

    Right after a download the operator expects to land in the editor, not on an
    empty Cuts list. So: no cuts yet → create the first one and open its trim
    editor; exactly one cut → reopen it; several cuts → show the Cuts list so
    they can pick which one to work on (or add another)."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/", "Video not found")
        if c.status != STATUS_ARCHIVED:
            return _flash(f"/video/{candidate_id}", "Download the video before clipping it")
        existing = session.execute(
            select(Cut).where(Cut.candidate_pk == c.id).order_by(Cut.created_at.desc())
        ).scalars().all()
        # Several cuts, or one that's already been exported: show the Cuts list so
        # the operator deliberately picks "reopen" vs "＋ New cut". Silently
        # reopening finished work is how a second trim ends up replacing the first.
        if len(existing) > 1 or any(cu.trimmed_clip_path for cu in existing):
            return RedirectResponse(f"/video/{candidate_id}", status_code=303)
        if len(existing) == 1:
            return RedirectResponse(f"/cut/{existing[0].id}?step=trim", status_code=303)
        # No caption seed — see create_cut: captions are drafted from the
        # trimmed transcript after export and require operator acceptance.
        cut = Cut(candidate_pk=c.id)
        session.add(cut)
        session.flush()
        cut_id = cut.id
    return _flash(f"/cut/{cut_id}?step=trim", "New clip — pick your segments")


@app.get("/cut/{cut_id}", response_class=HTMLResponse)
def cut_detail(request: Request, cut_id: int, step: str = "", msg: str = ""):
    with session_scope() as session:
        cut = session.execute(
            select(Cut)
            .options(selectinload(Cut.candidate).selectinload(Candidate.channel))
            .where(Cut.id == cut_id)
        ).scalar_one_or_none()
        if cut is None:
            return _flash("/", "Clip not found")
        c = cut.candidate

        exported = bool(cut.trimmed_clip_path) and Path(cut.trimmed_clip_path).exists()
        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.cut_pk == cut.id)
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
        posted = any(p.status == "published" for p in posts)
        cut_state = {"exported": exported, "posted": posted,
                     "captioned": bool(cut.subtitled_clip_path)}
        # A not-yet-published post pins the exact clip file it was queued with,
        # so re-exporting won't change it — warn before the operator assumes it will.
        pending = next((p for p in posts if p.status in ("queued", "draft", "failed")), None)
        pending_post_status = pending.status if pending else ""
        pending_post_id = pending.id if pending else None

        active_step = step if step in ("trim", "post") else ("post" if exported else "trim")

        segments = []
        if cut.trim_segments:
            try:
                segments = json.loads(cut.trim_segments)
            except Exception:
                pass

        # Segments already claimed by the video's OTHER cuts, drawn as dashed
        # outlines on the trim waveform so a second pass doesn't re-clip the
        # same material.
        other_cut_segments = []
        siblings = session.execute(
            select(Cut).where(
                Cut.candidate_pk == c.id, Cut.id != cut.id,
                Cut.trim_segments != "",
            )
        ).scalars().all()
        for sib in siblings:
            try:
                sib_segments = json.loads(sib.trim_segments)
            except Exception:
                continue
            title = (sib.clip_title or "").strip() or f"Clip {sib.id}"
            for s in sib_segments:
                other_cut_segments.append(
                    {"start": s["start"], "end": s["end"],
                     "cut_id": sib.id, "title": title})

        transcript_segments = []
        if c.transcript_path and Path(c.transcript_path).exists():
            try:
                transcript_segments = json.loads(Path(c.transcript_path).read_text())
            except Exception:
                pass
        clip_transcript, clip_transcript_text = _load_clip_transcript(cut)

    threads_ok = threads_api.is_authenticated()
    return templates.TemplateResponse(
        request, "cut.html",
        {"cut": cut, "c": c, "state": cut_state, "step": active_step,
         "transcript_segments": transcript_segments, "saved_segments": segments,
         "other_cut_segments": other_cut_segments,
         "clip_transcript": clip_transcript,
         "clip_transcript_text": clip_transcript_text,
         "posts": posts, "threads_ok": threads_ok,
         "account_name": threads_api.account_username(),
         "pending_post_status": pending_post_status,
         "pending_post_id": pending_post_id,
         # Attribution first-comment, editable before the post even exists:
         # prefill from the pending post so requeueing round-trips cleanly.
         "attribution_text": (pending.attribution_text or "") if pending else "",
         "attribution_enabled": bool(load_first_reply().get("attribution_enabled")),
         "auth_url": "" if threads_ok else threads_api.authorize_url(),
         "subs_position": (getattr(cut, "subs_position", "") or load_settings().get("subtitles.position", "bottom")),
         "msg": msg, "active": "dashboard"},
    )


@app.post("/cut/{cut_id}/delete")
def delete_cut(cut_id: int):
    """Delete a cut and any of its not-yet-published posts. Published posts are
    detached (kept for history) rather than removed."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return _flash("/", "Clip not found")
        candidate_id = cut.candidate_pk
        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.cut_pk == cut.id)
        ).scalars().all()
        for p in posts:
            if p.status in ("queued", "draft", "failed"):
                session.delete(p)
            else:
                p.cut_pk = None  # keep published history, drop the link
        session.delete(cut)
    return _flash(f"/video/{candidate_id}", "Clip deleted")





_UPLOAD_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".webm")


def _get_or_create_upload_channel(session) -> Channel:
    """A single synthetic channel that owns all operator-uploaded clips."""
    ch = session.execute(
        select(Channel).where(Channel.url == "upload://local")
    ).scalar_one_or_none()
    if ch is None:
        ch = Channel(call_sign="Uploads", network="", market="My uploads",
                     region="", country="", scope="local",
                     url="upload://local", enabled=False)
        session.add(ch)
        session.flush()
    return ch


@app.post("/upload")
async def upload_clip(file: UploadFile = File(...), title: str = Form("")):
    """Bring an operator's own video file into the same pipeline as discovered
    clips: it lands pre-'downloaded', gets transcribed locally, then flows through
    trim -> caption -> post like everything else."""
    import uuid

    from ..config import ROOT

    filename = file.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in _UPLOAD_EXTS:
        return _flash("/", f"Unsupported file type '{ext or '?'}'. "
                           f"Use one of: {', '.join(_UPLOAD_EXTS)}")

    video_id = "up" + uuid.uuid4().hex[:16]  # unique, fits String(20)
    settings = load_settings()
    upload_dir = ROOT / settings.get("storage.download_dir", "data/videos") / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{video_id}{ext}"

    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                out.write(chunk)
    finally:
        await file.close()
    if size == 0:
        dest.unlink(missing_ok=True)
        return _flash("/", "Upload was empty")

    duration = clip_duration(dest)

    # No thumbnail_url: an upload has no remote poster image, so the cards fall
    # back to /media/thumb, which pulls a frame out of the file we just wrote.
    with session_scope() as session:
        ch = _get_or_create_upload_channel(session)
        c = Candidate(
            video_id=video_id,
            channel_pk=ch.id,
            title=(title.strip() or Path(filename).stem)[:300],
            url=f"upload://{video_id}",
            published_at=utcnow(),
            duration_seconds=int(duration) if duration else None,
            local_video_path=str(dest),
            status=STATUS_APPROVED,
            approved_at=utcnow(),
        )
        session.add(c)
        session.flush()
        cid = c.id
    threading.Thread(target=_scrape_in_thread, args=(cid,), daemon=True).start()
    return _flash(f"/video/{cid}", "Uploaded — transcribing now")


def _get_or_create_pasted_channel(session) -> Channel:
    """A single synthetic channel that owns all pasted-URL YouTube clips."""
    ch = session.execute(
        select(Channel).where(Channel.url == PASTED_CHANNEL_URL)
    ).scalar_one_or_none()
    if ch is None:
        ch = Channel(call_sign="Pasted URLs", network="", market="Pasted YouTube URLs",
                     region="", country="", scope="local",
                     url=PASTED_CHANNEL_URL, enabled=False)
        session.add(ch)
        session.flush()
    return ch


@app.post("/upload-url")
def upload_url(urls: str = Form(...)):
    """Queue one or more pasted YouTube URLs for download.

    Reuses the approve/scrape pipeline: each URL becomes an 'approved' Candidate
    with a real YouTube url, then a background thread downloads it via yt-dlp and
    pulls captions — exactly like an approved discovered clip.
    """
    import re

    parts = [p.strip() for p in re.split(r"[\s,]+", urls or "") if p.strip()]
    if not parts:
        return _flash("/", "Paste at least one YouTube URL")

    settings = load_settings()
    title_model = settings.get("matching.model", "claude-haiku-4-5")
    queued: list[int] = []
    duplicates = 0
    invalid: list[str] = []

    for raw in parts:
        try:
            video_id = parse_video_url(raw)
        except YouTubeAPIError:
            invalid.append(raw)
            continue

        canonical = f"https://www.youtube.com/watch?v={video_id}"
        title = ""
        duration = None
        published_at = utcnow()
        # Best-effort metadata; if it fails we still queue the download.
        try:
            meta = fetch_video_metadata(canonical)
            title = meta.get("title") or ""
            duration = meta.get("duration_seconds")
            upload_date = meta.get("upload_date")
            if upload_date:
                try:
                    published_at = dt.datetime.strptime(upload_date, "%Y%m%d").replace(
                        tzinfo=dt.timezone.utc
                    )
                except ValueError:
                    pass
        except Exception as exc:
            log.info("Metadata fetch failed for %s: %s", canonical, exc)

        # Give the clip a concise 2-5 word AI title instead of the raw (often
        # long/clickbait) YouTube title. Best-effort: fall back to the source
        # title, then the URL. The original title is kept in `description`.
        # This is the label shown while the download runs; once the transcript
        # lands, archive_candidate rewrites the title from what is actually said,
        # which is also all we have to go on if the video has no source title.
        source_title = title.strip()
        display_title = source_title
        if source_title:
            try:
                short = suggest_short_title(title_model, source_title)
                if short:
                    display_title = short
            except Exception as exc:
                log.info("Short-title generation failed for %s: %s", canonical, exc)

        with session_scope() as session:
            existing = session.execute(
                select(Candidate.id).where(Candidate.video_id == video_id)
            ).scalar_one_or_none()
            if existing is not None:
                duplicates += 1
                continue
            ch = _get_or_create_pasted_channel(session)
            c = Candidate(
                video_id=video_id,
                channel_pk=ch.id,
                title=(display_title or canonical)[:300],
                description=source_title[:2000],
                url=canonical,
                # Same poster URL the Data API hands back for discovered clips.
                # Metadata-only, so the card has a still while the download runs.
                thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                published_at=published_at,
                duration_seconds=duration,
                status=STATUS_APPROVED,
                approved_at=utcnow(),
            )
            session.add(c)
            session.flush()
            queued.append(c.id)

    for cid in queued:
        threading.Thread(target=_scrape_in_thread, args=(cid,), daemon=True).start()

    bits = []
    if queued:
        bits.append(f"{len(queued)} queued — downloading now")
    if duplicates:
        bits.append(f"{duplicates} already in library")
    if invalid:
        bits.append(f"{len(invalid)} not recognized")
    msg = "; ".join(bits) or "Nothing to do"

    # Single new video: jump straight to its workflow screen, like /upload.
    if len(queued) == 1 and not duplicates and not invalid:
        return _flash(f"/video/{queued[0]}", "Queued — downloading and transcribing now")
    return _flash("/", msg)


def _log_triage_decision(session, c: Candidate, action: str) -> None:
    """Record what the operator decided given the signals on screen. This is
    the training record for eventual AI-assisted triage."""
    session.add(TriageDecision(
        candidate_pk=c.id,
        video_id=c.video_id,
        action=action,
        relevance_score=c.relevance_score,
        visual_score=c.visual_score,
        visual_traits=c.visual_traits or "",
    ))


@app.post("/video/{candidate_id}/approve")
def approve(request: Request, candidate_id: int):
    """The approve gate. This is the ONLY place a download is ever triggered."""
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return (JSONResponse({"error": "not found"}, status_code=404)
                    if wants_json else _flash("/", "Video not found"))
        if c.status == STATUS_ARCHIVED:
            return (JSONResponse({"ok": True, "status": "archived"})
                    if wants_json else _flash(f"/video/{candidate_id}", "Already archived"))
        c.status = STATUS_APPROVED
        c.approved_at = utcnow()
        _log_triage_decision(session, c, "approve")
    threading.Thread(target=_scrape_in_thread, args=(candidate_id,), daemon=True).start()
    # AJAX callers transition the page in place (no full reload); others redirect.
    if wants_json:
        return JSONResponse({"ok": True, "status": "approved"})
    return _flash(f"/video/{candidate_id}", "Approved — downloading and transcribing now")


@app.post("/video/{candidate_id}/reject")
def reject(request: Request, candidate_id: int):
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c:
            c.status = STATUS_REJECTED
            _log_triage_decision(session, c, "reject")
    # AJAX callers (dashboard delete buttons) get a light JSON reply so the page
    # can drop the row in place instead of reloading and re-fetching filmstrips.
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"ok": True})
    return _flash("/", "Rejected")


@app.post("/video/{candidate_id}/reset")
def reset_to_new(candidate_id: int):
    """Undo a triage decision: return the candidate to the 'new' review state.
    Used by the triage Undo button for both approve and reject."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        c.status = STATUS_NEW
        c.approved_at = None
        last = session.execute(
            select(TriageDecision)
            .where(TriageDecision.candidate_pk == c.id, TriageDecision.undone.is_(False))
            .order_by(TriageDecision.decided_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if last is not None:
            last.undone = True
    return {"ok": True}


@app.post("/video/{candidate_id}/unreject")
def unreject(candidate_id: int):
    """Restore a rejected clip. If it was already downloaded, return it to the
    archived state (ready to trim); otherwise send it back to the review gate.
    Never re-downloads."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/", "Video not found")
        downloaded = bool(c.local_video_path and Path(c.local_video_path).exists())
        c.status = STATUS_ARCHIVED if downloaded else STATUS_NEW
        dest = STATUS_ARCHIVED if downloaded else STATUS_NEW
    return _flash(f"/video/{candidate_id}", f"Restored — status is now {dest}")


@app.post("/video/{candidate_id}/retry")
def retry(candidate_id: int):
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c:
            c.status = STATUS_APPROVED
    threading.Thread(target=_scrape_in_thread, args=(candidate_id,), daemon=True).start()
    return _flash(f"/video/{candidate_id}", "Retrying scrape")


@app.get("/video/{candidate_id}/status")
def video_status(candidate_id: int):
    """Polled by the scrape step to auto-advance when the download finishes."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {"status": c.status, "error": c.scrape_error,
                "has_transcript": bool(c.transcript_text)}


# --- Media serving (local files for the trim/post players) -----------------------

@app.get("/media/source/{candidate_id}")
def media_source(candidate_id: int):
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.local_video_path or not Path(c.local_video_path).exists():
            return JSONResponse({"error": "no local video"}, status_code=404)
        return FileResponse(c.local_video_path, media_type="video/mp4")


@app.get("/media/thumb/{candidate_id}")
def media_thumb(candidate_id: int):
    """Poster frame pulled from a candidate's downloaded file.

    Only clips discovered through the YouTube Data API arrive with a
    ``thumbnail_url``; operator uploads and pasted URLs that fail metadata
    lookup have none, and their cards rendered as empty placeholders. Extracting
    lazily (and caching) covers those without a migration, and keeps working if
    ffmpeg happens to fail at ingest time.
    """
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.local_video_path:
            return JSONResponse({"error": "no local video"}, status_code=404)
        source = c.local_video_path
    try:
        path = cached_still(source, str(candidate_id))
    except Exception as exc:
        log.info("No still for candidate %s: %s", candidate_id, exc)
        return JSONResponse({"error": "no still"}, status_code=404)
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/media/clip/{cut_id}")
def media_clip(cut_id: int):
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        return FileResponse(cut.trimmed_clip_path, media_type="video/mp4")


@app.get("/media/subtitled/{cut_id}")
def media_subtitled(cut_id: int):
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.subtitled_clip_path or not Path(cut.subtitled_clip_path).exists():
            return JSONResponse({"error": "no subtitled clip"}, status_code=404)
        return FileResponse(cut.subtitled_clip_path, media_type="video/mp4")


@app.get("/media/post/{post_id}")
def media_post_clip(post_id: int):
    """Serve the exact clip attached to a post (captioned or plain).

    The post page used to always play ``/media/clip/{candidate}`` — the plain
    trim — even when the queued/published file was the subtitled variant stored
    on ``ThreadsPost.clip_local_path``. That made burnt-in captions look missing
    on scheduled-post pages even though publish would upload the right file.
    """
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = p.clip_local_path
        if (not path or not Path(path).exists()) and p.cut:
            cut = p.cut
            if cut.use_subtitles and cut.subtitled_clip_path and Path(cut.subtitled_clip_path).exists():
                path = cut.subtitled_clip_path
            else:
                path = cut.trimmed_clip_path
        if not path or not Path(path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        return FileResponse(path, media_type="video/mp4")


@app.get("/cut/{cut_id}/download-clip")
def download_clip(cut_id: int, captioned: int = 1):
    """Serve the exported clip as a file attachment so the operator can save it
    locally and post it manually elsewhere. Defaults to the captioned version
    (matching the preview default); pass ``captioned=0`` for the original."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        use_subs = bool(captioned) and bool(cut.subtitled_clip_path) \
            and Path(cut.subtitled_clip_path).exists()
        path = cut.subtitled_clip_path if use_subs else cut.trimmed_clip_path
        vid = cut.candidate.video_id if cut.candidate else None
        suffix = "-captioned" if use_subs else ""
        return FileResponse(path, media_type="video/mp4",
                            filename=f"{vid or 'clip'}-cut{cut.id}{suffix}.mp4")


@app.get("/post/{post_id}/download-clip")
def download_post_clip(post_id: int):
    """Download the clip attached to a post (used from the Posts page)."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = p.clip_local_path
        if (not path or not Path(path).exists()) and p.cut:
            path = p.cut.trimmed_clip_path
        if not path or not Path(path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        return FileResponse(path, media_type="video/mp4",
                            filename=f"threads-post-{p.id}.mp4")


# --- Trim / export ----------------------------------------------------------------

def _delete_if_unreferenced(session, paths: list[str]) -> None:
    """Remove superseded clip files that no ThreadsPost still points at.

    Exports are versioned per run, so a pending post keeps the exact file it was
    queued with. We only reclaim the disk space when nothing references the old
    file any more."""
    for path in {p for p in paths if p}:
        referenced = session.execute(
            select(ThreadsPost.id).where(ThreadsPost.clip_local_path == path).limit(1)
        ).scalar_one_or_none()
        if referenced is not None:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not remove superseded clip %s: %s", path, exc)


@app.post("/cut/{cut_id}/export")
def export_clip(cut_id: int, segments_json: str = Form(...), as_new: str = Form("0")):
    try:
        segments = json.loads(segments_json)
        assert isinstance(segments, list) and segments
    except Exception:
        return _flash(f"/cut/{cut_id}?step=trim", "No segments to export")
    save_as_new = as_new in ("1", "true", "on", "yes")
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return _flash("/", "Clip not found")
        c = cut.candidate
        if c is None or not c.local_video_path:
            return _flash("/", "Video not found or not downloaded")
        # "Save as new clip": keep the open cut untouched and write the
        # marked segments into a fresh sibling cut on the same video.
        if save_as_new:
            target = Cut(candidate_pk=c.id)
            session.add(target)
            session.flush()
        else:
            target = cut
        try:
            # Version every export. Re-exporting a cut must never overwrite the
            # file a queued post already points at — that would silently swap the
            # video under a scheduled post.
            stamp = utcnow().strftime("%Y%m%dT%H%M%S")
            previous = ([] if save_as_new else
                        [target.trimmed_clip_path, target.subtitled_clip_path,
                         target.clip_transcript_path])
            out = export_supercut(c.local_video_path, segments,
                                  f"{c.video_id}_cut{target.id}_{stamp}")
            target.trim_segments = json.dumps(segments)
            target.trimmed_clip_path = str(out)
            target.updated_at = utcnow()
            # Any previously generated captions no longer match the new cut.
            target.subtitled_clip_path = ""
            target.clip_transcript_path = ""
            target.use_subtitles = False
            if previous:
                _delete_if_unreferenced(session, previous)
            # Auto-title the fresh clip from its own transcript, but only when the
            # operator hasn't already set one (regeneration stays available in the
            # Post step). A titling failure must never block the export.
            if not (target.clip_title or "").strip():
                try:
                    settings = load_settings()
                    model = settings.get("engagement.draft_model", "claude-sonnet-5")
                    excerpt = _transcript_excerpt(c, segments)
                    title = suggest_title(model, c.title, excerpt, target.draft_caption or None)
                    if title:
                        target.clip_title = title
                        target.calendar_name = suggest_calendar_name(
                            model, title, target.draft_caption or None)
                except Exception:
                    pass
            n = len(segments)
            # Now that the clip's own transcript exists (trim_segments is
            # saved), have the Post step draft a caption from it — shown as a
            # proposal the operator must accept, never written into the field.
            # Cuts whose caption was already edited by the operator (or legacy
            # cuts still carrying the old video-level seed) skip the auto-run
            # only when the text is genuinely theirs.
            seed = (c.draft_caption or "").strip()
            current = (target.draft_caption or "").strip()
            autocaption = "&autocaption=1" if (not current or current == seed) else ""
            # From the second exported clip onwards, a multi-clip video prompts
            # the operator (on the Post step) to consider turning the marker off.
            askmulti = ""
            if c.multi_clip_potential:
                exported_cuts = session.execute(
                    select(func.count()).select_from(Cut).where(
                        Cut.candidate_pk == c.id, Cut.trimmed_clip_path != "")
                ).scalar() or 0
                if exported_cuts >= 2:
                    askmulti = f"&askmulti={exported_cuts}"
            # autosubs=1 makes the Post step kick off caption generation
            # immediately, so the captioned variant is the default.
            verb = "Saved new clip" if save_as_new else "Saved"
            return _flash(f"/cut/{target.id}?step=post&autosubs=1{autocaption}{askmulti}",
                          f"{verb} — {n} segment{'s' if n > 1 else ''} — generating captions…")
        except ClipExportError as exc:
            return _flash(f"/cut/{cut_id}?step=trim", f"Export failed: {exc}")


@app.post("/cut/{cut_id}/subtitles")
def generate_subtitles(cut_id: int, position: str = Form("")):
    """Generate the stylized-caption variant of the exported clip (AJAX).

    Runs whisper word timestamps + the Pillow/ffmpeg burn; takes roughly
    10-60s for a typical clip, longer on the first run while the whisper
    model downloads. Persists the Whisper word stream so Suggest caption /
    Copy transcript can use the same source as the burned-in captions.
    """
    from ..subtitles import (
        SubtitleError, clip_transcript_path_for, create_subtitled_clip,
    )

    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse({"error": "Export a clip first"}, status_code=404)
        clip_path = cut.trimmed_clip_path
        previous_subs = cut.subtitled_clip_path
    # Version each render so regenerating captions never overwrites the file a
    # queued post is already pointing at.
    stamp = utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = Path(clip_path).with_name(f"{Path(clip_path).stem}_subs_{stamp}.mp4")
    try:
        out = create_subtitled_clip(clip_path, position=position or None, out_path=out_path)
    except SubtitleError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except Exception as exc:
        log.exception("Caption generation failed for cut %s", cut_id)
        return JSONResponse({"error": f"Caption generation failed: {exc}"}, status_code=500)
    transcript_path = clip_transcript_path_for(clip_path)
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is not None:
            cut.subtitled_clip_path = str(out)
            if transcript_path.exists():
                cut.clip_transcript_path = str(transcript_path)
            _delete_if_unreferenced(session, [previous_subs])
            cut.use_subtitles = True
            cut.subs_position = "top" if (position or "").lower() == "top" else "bottom"
            cut.updated_at = utcnow()
    return {"url": f"/media/subtitled/{cut_id}"}


def _chosen_clip_path(cut: Cut, use_subtitles_form: str) -> str:
    """The file the operator wants to post: captioned variant when the box is
    ticked and the file exists, otherwise the plain export. Persists the choice."""
    want = str(use_subtitles_form).lower() in ("1", "true", "on", "yes")
    cut.use_subtitles = want and bool(cut.subtitled_clip_path)
    if cut.use_subtitles and Path(cut.subtitled_clip_path).exists():
        return cut.subtitled_clip_path
    return cut.trimmed_clip_path


# --- Caption suggestion + posting -------------------------------------------------

def _excerpt_segments(all_segments: list[dict], windows: list[dict]) -> list[dict]:
    """Transcript lines overlapping the trimmed windows, in clip (window) order.

    Each returned line is tagged with ``clip_start`` — its position in seconds
    within the exported supercut — so it can seek the joined clip on playback.
    """
    out: list[dict] = []
    clip_offset = 0.0
    for window in windows:
        try:
            ws, we = float(window["start"]), float(window["end"])
        except (KeyError, TypeError, ValueError):
            continue
        for s in all_segments:
            try:
                s_start, s_end = float(s["start"]), float(s["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if s_end >= ws and s_start <= we:
                out.append({
                    "start": s_start,
                    "text": s.get("text", ""),
                    "clip_start": round(clip_offset + max(0.0, s_start - ws), 2),
                })
        clip_offset += max(0.0, we - ws)
    return out


def _transcript_excerpt(c: Candidate, segments: list[dict]) -> str:
    """Transcript text inside the trimmed windows (falls back to full text)."""
    try:
        all_segments = json.loads(Path(c.transcript_path).read_text())
    except Exception:
        return c.transcript_text[:3000]
    if not segments:
        return c.transcript_text[:3000]
    parts = [s["text"] for s in _excerpt_segments(all_segments, segments)]
    return " ".join(parts)[:3000] or c.transcript_text[:3000]


def _clip_transcript_plain(clip_transcript: list[dict]) -> str:
    """Newline-joined text of the trimmed clip's transcript lines (for copy)."""
    lines = [str(s.get("text", "")).strip() for s in clip_transcript]
    return "\n".join(line for line in lines if line)


def _load_whisper_clip_transcript(cut: Cut | None) -> tuple[list[dict], str]:
    """Load the Whisper word stream of an exported cut (burned-in caption source).

    Prefer the cut's stored sidecar; fall back to the default path next to the
    trimmed clip. Returns ([], "") when nothing has been transcribed yet.
    """
    from ..subtitles import load_clip_words, words_to_lines, words_to_plain

    if cut is None:
        return [], ""
    path = ""
    if cut.clip_transcript_path and Path(cut.clip_transcript_path).exists():
        path = cut.clip_transcript_path
    elif cut.trimmed_clip_path:
        sidecar = Path(cut.trimmed_clip_path).with_name(
            f"{Path(cut.trimmed_clip_path).stem}_transcript.json"
        )
        if sidecar.exists():
            path = str(sidecar)
    if not path:
        return [], ""
    words = load_clip_words(path)
    if not words:
        return [], ""
    lines = words_to_lines(words)
    return lines, words_to_plain(words)


def _ensure_whisper_clip_transcript(cut: Cut) -> tuple[list[dict], str]:
    """Return the clip Whisper transcript, transcribing the trim if needed.

    Used by Suggest caption when the operator asks before (or without) burning
    captions in — still the same audio source the burned-in captions use.
    """
    from ..subtitles import SubtitleError, ensure_clip_words, words_to_lines, words_to_plain

    lines, plain = _load_whisper_clip_transcript(cut)
    if plain.strip():
        return lines, plain
    if not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
        return [], ""
    try:
        words, path = ensure_clip_words(
            cut.trimmed_clip_path,
            cut.clip_transcript_path or None,
        )
    except SubtitleError:
        return [], ""
    cut.clip_transcript_path = str(path)
    lines = words_to_lines(words)
    return lines, words_to_plain(words)


def _load_clip_transcript(cut: Cut | None = None,
                          candidate: Candidate | None = None,
                          trim_segments_json: str = "") -> tuple[list[dict], str]:
    """Return (timestamped lines, plain text) for a cut's exported windows.

    Prefer the Whisper transcript of the trimmed clip (same pass as burned-in
    captions). Fall back to slicing the source video's YouTube/upload transcript
    by trim windows only when Whisper hasn't run yet.
    """
    lines, plain = _load_whisper_clip_transcript(cut)
    if plain.strip():
        return lines, plain

    if candidate is None and cut is not None:
        candidate = cut.candidate
    if not trim_segments_json and cut is not None:
        trim_segments_json = cut.trim_segments or ""
    if not candidate or not trim_segments_json:
        return [], ""
    try:
        windows = json.loads(trim_segments_json)
    except Exception:
        return [], ""
    if not windows:
        return [], ""
    all_segments: list[dict] = []
    if candidate.transcript_path and Path(candidate.transcript_path).exists():
        try:
            all_segments = json.loads(Path(candidate.transcript_path).read_text())
        except Exception:
            all_segments = []
    if not all_segments:
        return [], ""
    lines = _excerpt_segments(all_segments, windows)
    return lines, _clip_transcript_plain(lines)


@app.post("/cut/{cut_id}/suggest-caption")
def suggest_caption(cut_id: int):
    settings = load_settings()
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        c = cut.candidate
        # Draft from the Whisper transcript of the exported clip — the same
        # source burned-in captions use. Transcribe on demand if captions
        # haven't been generated yet. Never fall back to the full-video
        # YouTube captions (those describe parts that aren't in the clip).
        if not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse(
                {"error": "Export the clip first — the caption is drafted "
                          "from what is said in the trimmed clip."},
                status_code=409)
        _lines, clip_text = _ensure_whisper_clip_transcript(cut)
        if not clip_text.strip():
            return JSONResponse(
                {"error": "No speech detected in the clip — nothing to draft "
                          "a caption from."},
                status_code=409)
        excerpt = " ".join(clip_text.split())[:3000]
        seconds = clip_duration(cut.trimmed_clip_path) if cut.trimmed_clip_path else None
        # Voice matching from past captions — never let it break drafting.
        try:
            voice = voice_context(session, settings)
        except Exception as exc:
            log.warning("Voice context failed (drafting generic): %s", exc)
            voice = {"examples": [], "style_guide": ""}
        try:
            caption = suggest_post_caption(
                settings.get("engagement.draft_model", "claude-sonnet-5"),
                c.title, c.channel.call_sign, c.channel.market, excerpt, seconds,
                examples=voice["examples"], style_guide=voice["style_guide"],
                operator_guide=render_caption_guide(),
            )
            # Not persisted: the suggestion is only a proposal until the
            # operator explicitly accepts it (/cut/{id}/caption).
            return {"caption": caption, "voice_examples": len(voice["examples"]),
                    "transcript": clip_text}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/cut/{cut_id}/transcript")
def cut_transcript(cut_id: int):
    """Plain Whisper transcript of the exported clip (for Copy transcript).

    Transcribes on demand when burned-in captions haven't been generated yet,
    so the button works even with "No captions" selected.
    """
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse(
                {"error": "Export the clip first."}, status_code=409)
        _lines, clip_text = _ensure_whisper_clip_transcript(cut)
        if not clip_text.strip():
            return JSONResponse(
                {"error": "No speech detected in the clip."}, status_code=409)
        return {"transcript": clip_text}


@app.post("/cut/{cut_id}/caption")
def save_cut_caption(cut_id: int, caption: str = Form("")):
    """Persist the caption the operator accepted (or edited) on the Post step."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        cut.draft_caption = caption.strip()
        cut.updated_at = utcnow()
    return {"ok": True}


@app.post("/cut/{cut_id}/suggest-title")
def suggest_clip_title(cut_id: int):
    settings = load_settings()
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        c = cut.candidate
        segments = json.loads(cut.trim_segments) if cut.trim_segments else []
        excerpt = _transcript_excerpt(c, segments)
        model = settings.get("engagement.draft_model", "claude-sonnet-5")
        try:
            title = suggest_title(model, c.title, excerpt, cut.draft_caption or None)
            if title:
                cut.clip_title = title
                cut.calendar_name = suggest_calendar_name(model, title, cut.draft_caption or None)
            return {"title": title or cut.clip_title, "calendar_name": cut.calendar_name}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/cut/{cut_id}/post")
def post_to_threads(cut_id: int, caption: str = Form(...),
                    use_subtitles: str = Form(""), attribution: str = Form("")):
    """Operator-confirmed publish of the exported clip."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/cut/{cut_id}?step=post", "Caption is empty")
    with session_scope() as session:
        ok, wait_min = spacing_allows_publish(session)
        if not ok:
            return _flash(
                f"/cut/{cut_id}?step=post",
                f"Spacing floor: wait ~{wait_min} more minute{'s' if wait_min != 1 else ''} "
                f"before publishing another post",
            )
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path:
            return _flash(f"/cut/{cut_id}", "Export a clip first")
        try:
            post = publish_clip(session, cut.candidate,
                                _chosen_clip_path(cut, use_subtitles), caption, cut=cut,
                                attribution=attribution)
            # Keep the clip's caption in sync with what was actually posted.
            cut.draft_caption = caption
            state = session.get(SchedulerState, 1)
            if state is None:
                state = SchedulerState(id=1)
                session.add(state)
            state.last_publish_at = utcnow()
            state.last_action = f"manual_publish:post={post.id}"
            state.updated_at = utcnow()
            msg = f"Published: {post.permalink or post.threads_media_id}"
            if post.first_reply_id:
                msg += " · first reply posted"
            elif post.first_reply_error:
                msg += f" · no first reply: {post.first_reply_error[:120]}"
            return _flash(f"/cut/{cut_id}?step=post", msg)
        except Exception as exc:
            return _flash(f"/cut/{cut_id}?step=post", f"Publish failed: {exc}")


@app.post("/cut/{cut_id}/queue")
def queue_to_threads(cut_id: int, caption: str = Form(...),
                     use_subtitles: str = Form(""), attribution: str = Form("")):
    """Add the exported clip to the adaptive FIFO queue (no immediate post)."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/cut/{cut_id}?step=post", "Caption is empty")
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path:
            return _flash(f"/cut/{cut_id}", "Export a clip first")
        clip_path = _chosen_clip_path(cut, use_subtitles)
        try:
            # Reuse an existing not-yet-published post for THIS cut rather than
            # creating a duplicate queue entry.
            existing = session.execute(
                select(ThreadsPost).where(
                    ThreadsPost.cut_pk == cut.id,
                    ThreadsPost.status.in_(["queued", "draft", "failed"]),
                ).order_by(ThreadsPost.created_at.desc())
            ).scalars().all()
            if existing:
                keep = existing[0]
                keep.caption = caption
                # The cut page pre-fills this field from the pending post, so
                # whatever came back (edited or cleared) is the operator's call —
                # a clear has to survive publishing, which otherwise drafts a
                # replacement for any post it finds without one.
                keep.attribution_text = attribution.strip()
                keep.attribution_skipped = not attribution.strip()
                if keep.clip_local_path != clip_path:
                    # Captions were toggled since this post was created — point
                    # at the chosen file and refresh the cloud copy.
                    from ..publishing import _object_key
                    from ..storage_supabase import upload_trimmed_clip
                    keep.clip_local_path = clip_path
                    keep.clip_object_path = _object_key(Path(clip_path))
                    try:
                        upload_trimmed_clip(Path(clip_path), keep.clip_object_path)
                    except Exception as exc:
                        log.warning("Clip re-upload failed (will retry at publish): %s", exc)
                keep.status = "queued"
                keep.scheduled_at = None
                keep.pinned_window_key = ""
                keep.error = ""
                for extra in existing[1:]:
                    session.delete(extra)
            else:
                queue_clip(session, cut.candidate, clip_path, caption, cut=cut,
                           attribution=attribution)
            # Persist the queued caption back onto the clip so the clip reflects
            # what was scheduled, not the original generated draft. (Done after
            # record_post has frozen the AI draft as suggested_caption.)
            cut.draft_caption = caption
            updated = bool(existing)
        except Exception as exc:
            return _flash(f"/cut/{cut_id}?step=post", f"Queue failed: {exc}")
    return _flash(f"/cut/{cut_id}?step=post",
                  "Queued post updated" if updated else "Added to the posting queue")


@app.post("/cut/{cut_id}/save-draft")
def save_draft(cut_id: int, caption: str = Form(...),
               use_subtitles: str = Form(""), attribution: str = Form("")):
    """Save the exported clip + caption as a draft to publish or queue later."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/cut/{cut_id}?step=post", "Caption is empty")
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path:
            return _flash(f"/cut/{cut_id}", "Export a clip first")
        try:
            record_post(session, cut.candidate, _chosen_clip_path(cut, use_subtitles),
                        caption, status="draft", cut=cut, attribution=attribution)
            # Keep the clip's caption in sync with the saved draft.
            cut.draft_caption = caption
        except Exception as exc:
            return _flash(f"/cut/{cut_id}?step=post", f"Save failed: {exc}")
    return _flash(f"/cut/{cut_id}?step=post",
                  "Saved as draft — publish or queue it any time from Posts")


@app.post("/cut/{cut_id}/suggest-attribution")
def suggest_cut_attribution(cut_id: int):
    """(Re)draft the attribution first-comment while still on the cut page.
    Returns the suggestion only — it rides along with queue/post/draft."""
    with session_scope() as session:
        cut = session.execute(
            select(Cut)
            .options(selectinload(Cut.candidate).selectinload(Candidate.channel))
            .where(Cut.id == cut_id)
        ).scalar_one_or_none()
        if cut is None or cut.candidate is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            text = generate_attribution(cut.candidate)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
    if not text:
        return JSONResponse({"error": "The model returned an empty attribution — try again."},
                            status_code=500)
    return {"text": text}


@app.post("/post/{post_id}/cancel")
def cancel_queued_post(post_id: int, next: str = Form("/calendar")):
    """Remove a queued or draft (not-yet-published) post."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return _flash(next, "Post not found")
        if p.status not in ("queued", "draft"):
            return _flash(next, "Only queued or draft posts can be removed")
        was = p.status
        cut_id = p.cut_pk
        session.delete(p)
        # Send the operator back to the cut when deleting its only post record,
        # so they don't lose track of a trimmed export.
        if cut_id and (not next or next in ("/posts", "/calendar", "/")):
            next = f"/cut/{cut_id}?step=post"
    label = "Queued post" if was == "queued" else "Draft"
    return _flash(next, f"{label} removed — your clip is still here; queue or post again when ready.")


@app.post("/post/{post_id}/queue")
def queue_existing_post(post_id: int, caption: str = Form(""),
                        attribution: str | None = Form(None),
                        next: str = Form("/calendar")):
    """Move a draft/failed post into the adaptive queue (or update a queued one)."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status not in ("draft", "failed", "queued"):
            return _flash(next, "Only a draft, failed, or queued post can be (re)queued")
        # None = the form had no attribution field (e.g. requeue from
        # Notifications); an empty string is a deliberate clear (skip the comment).
        if attribution is not None:
            p.attribution_text = attribution.strip()
            p.attribution_skipped = not attribution.strip()
        if caption.strip():
            p.caption = caption.strip()
            # Mirror the edited caption onto the clip so re-opening the cut shows
            # what was queued, not the original generated draft. suggested_caption
            # on the post already froze the AI draft, so voice-learning is intact.
            if p.cut_pk is not None:
                cut = session.get(Cut, p.cut_pk)
                if cut is not None:
                    cut.draft_caption = caption.strip()
        p.status = "queued"
        p.scheduled_at = None
        p.error = ""
        if p.cut_pk is not None:
            dupes = session.execute(
                select(ThreadsPost).where(
                    ThreadsPost.cut_pk == p.cut_pk,
                    ThreadsPost.id != p.id,
                    ThreadsPost.status.in_(["queued", "draft", "failed"]),
                )
            ).scalars().all()
            for extra in dupes:
                session.delete(extra)
    return _flash(next, "Added to the posting queue")


@app.post("/post/{post_id}/pin-window")
def pin_window(request: Request, post_id: int, window_key: str = Form(...),
               next: str = Form("/calendar")):
    """Pin a queued post to an upcoming window (calendar drag-and-drop)."""
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        try:
            msg = pin_post_to_window(session, post_id, window_key)
        except ValueError as exc:
            if wants_json:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return _flash(next, str(exc))
    if wants_json:
        return JSONResponse({"ok": True, "message": msg, "window_key": window_key})
    return _flash(next, msg)


@app.post("/post/{post_id}/unpin")
def unpin_window(post_id: int, next: str = Form("/calendar")):
    """Clear a queued post's window pin — it goes back to filling the next
    open window FIFO, like a post that was never dragged/picked at all."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status != "queued":
            return _flash(next, "Only a queued post can be unpinned")
        p.pinned_window_key = ""
    return _flash(next, "Unpinned — back to the next open window")


def _publish_in_thread(post_id: int) -> None:
    """Publish a post in the background. publish_post sets status to
    published/failed (+ error) itself, so we just swallow the exception here."""
    try:
        with session_scope() as session:
            p = session.get(ThreadsPost, post_id)
            if p is None:
                return
            try:
                publish_post(session, p)
                state = session.get(SchedulerState, 1)
                if state is None:
                    state = SchedulerState(id=1)
                    session.add(state)
                state.last_publish_at = utcnow()
                state.last_action = f"manual_publish:post={post_id}"
                state.updated_at = utcnow()
            except Exception:
                log.exception("Background publish failed for post %s", post_id)
    finally:
        clear_publishing(post_id)


@app.post("/post/{post_id}/publish-now")
def publish_scheduled_now(request: Request, post_id: int, next: str = Form("/calendar")):
    """Publish a queued, draft, or previously failed post immediately.

    Publishing a video can take minutes (upload + Threads-side processing), so we
    kick it off in the background and return right away. The post flips to a
    'publishing' state that the UI can poll via /post/{id}/status."""
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        ok, wait_min = spacing_allows_publish(session)
        if not ok:
            msg = (
                f"Spacing floor: wait ~{wait_min} more minute{'s' if wait_min != 1 else ''} "
                f"before publishing another post"
            )
            return (JSONResponse({"error": msg}, status_code=409)
                    if wants_json else _flash(next, msg))
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status not in ("queued", "failed", "draft"):
            return (JSONResponse({"error": "Nothing to publish"}, status_code=409)
                    if wants_json else _flash(next, "Nothing to publish"))
        p.status = "publishing"
        p.error = ""
    # Register before spawning so a scheduler tick in the gap between here and
    # the thread starting doesn't mistake this for an orphaned publish.
    mark_publishing(post_id)
    threading.Thread(target=_publish_in_thread, args=(post_id,), daemon=True).start()
    if wants_json:
        return JSONResponse({"ok": True, "status": "publishing"})
    return _flash(next, "Publishing now — video can take a minute to process.")


@app.get("/post/{post_id}/status")
def post_status(post_id: int):
    """Lightweight status poll for a post that's publishing in the background."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({
            "status": p.status,
            "permalink": p.permalink or "",
            "error": (p.error or "")[:300],
        })


@app.post("/post/{post_id}/recaption")
def post_recaption(post_id: int, position: str = Form("bottom")):
    """Switch a not-yet-published post between no burned-in captions and
    captions at the top or bottom.

    ``position=none`` points the post at the plain trimmed clip (and flips
    ``use_subtitles`` off). ``top`` / ``bottom`` re-renders captions onto a
    fresh file and moves the post onto it. Renders are versioned, so this is
    an explicit opt-in — unlike a passive re-export, which deliberately leaves
    a queued post on the clip it was queued with.
    """
    from ..publishing import _object_key
    from ..storage_supabase import upload_trimmed_clip
    from ..subtitles import SubtitleError, clip_transcript_path_for, create_subtitled_clip

    raw = str(position).strip().lower()
    if raw in ("none", "off", "plain"):
        mode = "none"
    elif raw == "top":
        mode = "top"
    else:
        mode = "bottom"

    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if p.status not in ("draft", "queued", "failed"):
            return JSONResponse(
                {"error": "Only a post that hasn't published yet can be re-captioned"},
                status_code=409)
        cut = p.cut
        if cut is None or not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse({"error": "No trimmed clip to caption"}, status_code=404)
        cut_id = cut.id
        plain_path = cut.trimmed_clip_path
        superseded = [p.clip_local_path]

        if mode == "none":
            # Drop burned-in captions: post the plain trim. Leave any existing
            # subtitled file on the cut so the operator can flip back without a
            # full re-render (they'll re-render if they change position).
            cut.use_subtitles = False
            cut.updated_at = utcnow()
            out = Path(plain_path)
            p.clip_local_path = str(out)
            p.clip_object_path = _object_key(out)
            try:
                upload_trimmed_clip(out, p.clip_object_path)
            except Exception as exc:
                log.warning("Plain-clip upload failed (will retry at publish): %s", exc)
            _delete_if_unreferenced(session, [s for s in superseded if s and s != str(out)])
            return JSONResponse({"ok": True, "position": "none"})

        superseded.append(cut.subtitled_clip_path)

    stamp = utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = Path(plain_path).with_name(f"{Path(plain_path).stem}_subs_{stamp}.mp4")
    try:
        out = create_subtitled_clip(plain_path, position=mode, out_path=out_path)
    except SubtitleError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except Exception as exc:
        log.exception("Re-caption failed for post %s", post_id)
        return JSONResponse({"error": f"Caption render failed: {exc}"}, status_code=500)

    transcript_path = clip_transcript_path_for(plain_path)
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        cut = session.get(Cut, cut_id)
        if p is None or cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        cut.subtitled_clip_path = str(out)
        if transcript_path.exists():
            cut.clip_transcript_path = str(transcript_path)
        cut.subs_position = mode
        cut.use_subtitles = True
        cut.updated_at = utcnow()
        p.clip_local_path = str(out)
        p.clip_object_path = _object_key(out)
        try:
            upload_trimmed_clip(out, p.clip_object_path)
        except Exception as exc:
            log.warning("Re-caption upload failed (will retry at publish): %s", exc)
        _delete_if_unreferenced(session, superseded)
    return JSONResponse({"ok": True, "position": mode})


_POST_METRICS = ("views", "likes", "replies", "reposts", "quotes", "shares")


@app.get("/post/{post_id}", response_class=HTMLResponse)
def post_detail(request: Request, post_id: int, msg: str = ""):
    """A single post's profile: manage the queue before it publishes, and once
    it's live, see its stats and the replies it received."""
    with session_scope() as session:
        p = session.execute(
            select(ThreadsPost)
            .options(
                selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
                selectinload(ThreadsPost.cut),
            )
            .where(ThreadsPost.id == post_id)
        ).scalar_one_or_none()
        if p is None:
            return _flash("/calendar", "Post not found")

        cand = p.candidate
        cut = p.cut
        clip_path = p.clip_local_path if (p.clip_local_path and Path(p.clip_local_path).exists()) else ""
        if not clip_path and cut:
            if (cut.use_subtitles and cut.subtitled_clip_path
                    and Path(cut.subtitled_clip_path).exists()):
                clip_path = cut.subtitled_clip_path
            elif cut.trimmed_clip_path and Path(cut.trimmed_clip_path).exists():
                clip_path = cut.trimmed_clip_path
        has_clip = bool(clip_path)
        # Burnt-in captions live in *_subs.mp4; surface that on the post page
        # so the plain Threads text caption isn't confused with video subs.
        has_burned_captions = bool(clip_path and clip_path.endswith("_subs.mp4"))
        _clip_lines, clip_transcript_text = _load_clip_transcript(cut)
        snap = session.execute(
            select(MetricSnapshot).where(MetricSnapshot.post_pk == p.id)
            .order_by(MetricSnapshot.captured_at.desc()).limit(1)
        ).scalar_one_or_none()
        metrics = {m: getattr(snap, m) for m in _POST_METRICS} if snap else None
        snapshot_count = session.execute(
            select(func.count(MetricSnapshot.id)).where(MetricSnapshot.post_pk == p.id)
        ).scalar_one()
        comments = session.execute(
            select(ThreadsComment).where(ThreadsComment.post_pk == p.id)
            .order_by(ThreadsComment.created_at.desc())
        ).scalars().all()
        comment_rows = [
            {"id": c.id, "username": c.username, "text": c.text,
             "reply_status": c.reply_status,
             "reply_text": c.reply_text_posted,
             "commented_at": c.commented_at}
            for c in comments
        ]
        # Projected publishing slot (same plan the calendar shows), so a queued
        # post says exactly when it's expected to go out. Reschedule from the
        # calendar — no move/unpin controls here.
        schedule = None
        if p.status == "queued":
            slot = projected_slot_for_post(session, p.id)
            if slot is None:
                schedule = {"unknown": True}
            else:
                schedule = {"when": slot.get("sort"), "time": slot.get("time"),
                            "window_index": slot.get("window_index"),
                            "pinned": bool(slot.get("pinned"))}
        ctx = {
            "pid": p.id, "status": p.status, "caption": p.caption or "",
            "account_name": threads_api.account_username(),
            "permalink": p.permalink, "source": p.source, "error": p.error,
            "candidate_id": cand.id if cand else None,
            "category": cand.category if cand else "",
            "category_rationale": cand.category_rationale if cand else "",
            "cut_id": cut.id if cut else None,
            "channel_sign": cand.channel.call_sign if (cand and cand.channel) else "",
            "video_title": cand.title if cand else "",
            "clip_title": cut.clip_title if cut else "",
            "has_clip": has_clip,
            "has_burned_captions": has_burned_captions,
            # Caption position can be re-rendered right here until the post goes out.
            "can_recaption": bool(
                cut and cut.trimmed_clip_path and Path(cut.trimmed_clip_path).exists()
                and p.status in ("draft", "queued", "failed")
            ),
            # none | bottom | top — reflects what this post's clip actually has.
            "subs_mode": (
                ((cut.subs_position or "bottom") if cut else "bottom")
                if has_burned_captions else "none"
            ),
            "clip_transcript_text": clip_transcript_text,
            "scheduled_at": p.scheduled_at, "published_at": p.published_at,
            "created_at": p.created_at,
            "attribution_text": p.attribution_text or "",
            "attribution_enabled": load_first_reply().get("attribution_enabled", True),
            "can_suggest_attribution": bool(cand),
            "first_reply_id": p.first_reply_id or "",
            "first_reply_text": p.first_reply_text or "",
            "first_reply_error": p.first_reply_error or "",
            "first_reply_at": p.first_reply_at,
            "metrics": metrics, "metrics_captured": snap.captured_at if snap else None,
            "snapshot_count": snapshot_count,
            "comments": comment_rows,
            "schedule": schedule,
            "giphy_enabled": giphy_configured(),
        }
    return templates.TemplateResponse(
        request, "post.html", {**ctx, "msg": msg, "active": "posts"}
    )


@app.post("/post/{post_id}/refresh-stats")
def refresh_post_stats(post_id: int, next: str = Form("")):
    """Force a fresh metric snapshot for one published post."""
    dest = next or f"/post/{post_id}"
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status != "published" or not p.threads_media_id:
            return _flash(dest, "No published post to refresh")
        try:
            data = threads_api.fetch_insights(p.threads_media_id)
        except Exception as exc:
            return _flash(dest, f"Refresh failed: {exc}")
        if not data:
            return _flash(dest, "No insights returned yet")
        session.add(MetricSnapshot(
            post_pk=p.id,
            views=data.get("views"), likes=data.get("likes"),
            replies=data.get("replies"), reposts=data.get("reposts"),
            quotes=data.get("quotes"), shares=data.get("shares"),
        ))
    return _flash(dest, "Stats refreshed")


@app.post("/post/{post_id}/sync-replies")
def sync_post_replies(post_id: int, next: str = Form("")):
    """Pull the replies on this (and other) published posts."""
    dest = next or f"/post/{post_id}"
    with session_scope() as session:
        try:
            result = sync_comments(session)
            return _flash(dest, f"Synced: {result['new_comments']} new replies")
        except Exception as exc:
            return _flash(dest, f"Reply sync failed: {exc}")


def _clean_auth_code(raw: str) -> str:
    """Normalize a pasted Threads OAuth code. Accepts the bare code, a code with
    the trailing ``#_`` fragment the browser appends, or the whole redirect URL."""
    import urllib.parse

    raw = (raw or "").strip()
    if "code=" in raw:  # user pasted the full redirect URL
        parsed = urllib.parse.urlparse(raw)
        vals = urllib.parse.parse_qs(parsed.query).get("code")
        if vals:
            raw = vals[0]
    raw = raw.split("#")[0]  # drop Meta's "#_" fragment (and anything after)
    return raw.strip()


@app.post("/threads/connect")
def threads_connect(code: str = Form(...), next: str = Form("/calendar")):
    try:
        threads_api.exchange_code(_clean_auth_code(code))
        return _flash(next, "Threads connected")
    except Exception as exc:
        return _flash(next, f"Auth failed: {exc}")


@app.get("/threads-account", response_class=HTMLResponse)
def threads_account_page(request: Request, msg: str = ""):
    """Threads OAuth connection status + account-level actions (Configure area)."""
    authenticated = threads_api.is_authenticated()
    return templates.TemplateResponse(
        request, "threads_account.html",
        {"authenticated": authenticated,
         "auth_url": threads_api.authorize_url() if not authenticated else "",
         "msg": msg, "active": "threads_account"},
    )


# --- Archive -----------------------------------------------------------------

@app.get("/archive")
def archive_redirect(section: str = ""):
    """Back-compat: the Archive page is now the Library's Videos section."""
    return RedirectResponse("/library" + (f"?section={section}" if section else ""),
                            status_code=307)


@app.get("/library", response_class=HTMLResponse)
def library_page(request: Request, section: str = "videos", msg: str = ""):
    """The content library: Videos → Cuts → Posts, the three types that cascade
    into each other, shown as a toggle group. ``section`` selects the open tab."""
    if section not in ("videos", "cuts", "posts"):
        section = "videos"
    with session_scope() as session:
        # --- Videos (downloaded/archived source clips) ---
        videos = session.execute(
            select(Candidate)
            .options(selectinload(Candidate.channel))
            .where(Candidate.status == STATUS_ARCHIVED)
            .order_by(Candidate.archived_at.desc())
        ).scalars().all()
        vids = [c.id for c in videos]
        statuses = _post_statuses_by_candidate(session, vids)
        exported_ids = _exported_cut_candidate_ids(session, vids)
        cut_counts: dict[int, int] = {}
        if vids:
            for pk, n in session.execute(
                select(Cut.candidate_pk, func.count(Cut.id))
                .where(Cut.candidate_pk.in_(vids)).group_by(Cut.candidate_pk)
            ).all():
                cut_counts[pk] = n
        video_rows = [
            {"c": c,
             "state": workflow_state(session, c, post_statuses=statuses.get(c.id, set()),
                                     has_exported_cut=c.id in exported_ids),
             "cut_count": cut_counts.get(c.id, 0)}
            for c in videos
        ]

        # --- Cuts (first-class trimmed clips) ---
        cuts = session.execute(
            select(Cut)
            .options(selectinload(Cut.candidate).selectinload(Candidate.channel))
            .order_by(Cut.created_at.desc())
        ).scalars().all()
        published_cut_pks = {
            pk for (pk,) in session.execute(
                select(ThreadsPost.cut_pk).where(
                    ThreadsPost.status == "published", ThreadsPost.cut_pk.is_not(None)
                ).distinct()
            ).all()
        }
        cut_rows = [
            {"cut": cut,
             "exported": bool(cut.trimmed_clip_path),
             "posted": cut.id in published_cut_pks,
             "captioned": bool(cut.subtitled_clip_path)}
            for cut in cuts
        ]

        # --- Posts (recent, any status) ---
        posts = session.execute(
            select(ThreadsPost)
            .options(
                selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
                selectinload(ThreadsPost.cut)
                .selectinload(Cut.candidate).selectinload(Candidate.channel),
            )
            .order_by(ThreadsPost.created_at.desc()).limit(100)
        ).scalars().all()

        # --- Filter vocabularies, drawn from what's actually on the page so the
        # dropdowns never offer a choice that matches nothing. ---
        def _call_sign(candidate) -> str:
            ch = getattr(candidate, "channel", None) if candidate is not None else None
            return (ch.call_sign or "") if ch is not None else ""

        call_signs = {_call_sign(r["c"]) for r in video_rows}
        call_signs |= {_call_sign(r["cut"].candidate) for r in cut_rows}
        call_signs |= {
            _call_sign(p.cut.candidate if (p.cut and p.cut.candidate) else p.candidate)
            for p in posts
        }
        used_categories = {r["c"].category for r in video_rows if r["c"].category}
        category_choices = [
            (opt["slug"], f"{opt['emoji']} {opt['label']}".strip())
            for opt in category_options() if opt["slug"] in used_categories
        ]
        post_statuses = sorted({p.status for p in posts if p.status})

    return templates.TemplateResponse(
        request, "library.html",
        {"video_rows": video_rows, "cut_rows": cut_rows, "posts": posts,
         "section": section,
         "counts": {"videos": len(video_rows), "cuts": len(cut_rows), "posts": len(posts)},
         "channel_choices": sorted(cs for cs in call_signs if cs),
         "category_choices": category_choices,
         "post_status_choices": post_statuses,
         "msg": msg, "active": "library"},
    )


# --- Posts (history + manual publish + Threads connect) --------------------------

@app.post("/threads/import-history")
def threads_import_history(next: str = Form("/calendar")):
    """Pull the account's own existing Threads posts into the DB, then kick off
    an insights snapshot for them in the background. Reachable from the nav bar
    on any page, so it redirects back to wherever it was submitted from."""
    if not threads_api.is_authenticated():
        return _flash(next, "Connect Threads first")
    with session_scope() as session:
        try:
            result = import_history(session)
        except Exception as exc:
            return _flash(next, f"Import failed: {exc}")

    def _pull_insights():
        try:
            with session_scope() as s:
                snapshot_metrics(s)
        except Exception as exc:  # pragma: no cover - background best-effort
            logging.getLogger("history").warning("Post-import snapshot failed: %s", exc)

    threading.Thread(target=_pull_insights, daemon=True).start()
    return _flash(
        next,
        f"Imported {result['imported']} posts ({result['skipped']} already known) — "
        f"pulling insights in the background; see Analytics shortly.",
    )


@app.get("/calendar", response_class=HTMLResponse)
def calendar_page(request: Request, year: int = 0, month: int = 0, msg: str = ""):
    """Month grid of window slots + linear posting queue (local time)."""
    import calendar as _cal

    now_local = dt.datetime.now()
    y = year or now_local.year
    m = month or now_local.month
    if m < 1:
        y, m = y - 1, 12
    elif m > 12:
        y, m = y + 1, 1

    first_local = dt.datetime(y, m, 1)
    next_first_local = dt.datetime(y + 1, 1, 1) if m == 12 else dt.datetime(y, m + 1, 1)

    events: dict[int, list[dict]] = {}
    drafts_count = 0
    queue_count = 0
    linear: list[dict] = []
    status = {}
    windows_et: list[str] = []
    with session_scope() as session:
        drafts_count = session.execute(
            select(func.count()).select_from(ThreadsPost).where(ThreadsPost.status == "draft")
        ).scalar_one()
        queue_count = session.execute(
            select(func.count()).select_from(ThreadsPost).where(ThreadsPost.status == "queued")
        ).scalar_one()
        status = scheduler_status(session)
        windows_et = list(status.get("windows") or [])

        plan = build_window_plan(session, first_local, next_first_local)
        for e in plan:
            # Calendar grid: published history + upcoming filled/open windows.
            events.setdefault(e["day"], []).append(e)

        # Linear queue: upcoming windows only (not published history).
        linear = [e for e in plan if e["kind"] in ("queued", "open")]
        # Cap the linear list to the next ~21 slots so it stays scannable.
        linear = linear[:21]

    for day in events:
        events[day].sort(key=lambda e: e["sort"])

    cal = _cal.Calendar(firstweekday=6)  # Sunday-first
    weeks = cal.monthdayscalendar(y, m)
    today = now_local.day if (y == now_local.year and m == now_local.month) else 0

    prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
    next_y, next_m = (y + 1, 1) if m == 12 else (y, m + 1)

    return templates.TemplateResponse(
        request, "calendar.html",
        {"weeks": weeks, "events": events, "today": today,
         "year": y, "month": m, "month_name": _cal.month_name[m],
         "prev_y": prev_y, "prev_m": prev_m, "next_y": next_y, "next_m": next_m,
         "dow": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
         "drafts_count": drafts_count, "queue_count": queue_count,
         "linear": linear, "windows_et": windows_et,
         "windows_local": window_time_labels(),
         "msg": msg, "active": "calendar"},
    )


@app.get("/posts")
def posts_page(msg: str = ""):
    """Retired page. The queue lives on the Calendar, drafts/history in the
    Library, and Threads connection + scheduler status now live on the Calendar
    too. Kept as a redirect so old links and bookmarks still resolve."""
    return RedirectResponse("/calendar" + (f"?msg={msg}" if msg else ""), status_code=307)


@app.get("/notifications", response_class=HTMLResponse)
def notifications_page(request: Request, msg: str = ""):
    """Operator alerts — currently failed posts that dropped out of the queue and
    need a decision (retry, re-queue, or dismiss)."""
    with session_scope() as session:
        failed = session.execute(
            select(ThreadsPost)
            .options(selectinload(ThreadsPost.cut).selectinload(Cut.candidate),
                     selectinload(ThreadsPost.candidate))
            .where(ThreadsPost.status == "failed",
                   ThreadsPost.attention_dismissed_at.is_(None))
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
    return templates.TemplateResponse(
        request, "notifications.html",
        {"failed": failed, "msg": msg, "active": "notifications"},
    )


@app.post("/post/{post_id}/dismiss")
def dismiss_attention(post_id: int, next: str = Form("/notifications")):
    """Acknowledge a failed post so it leaves the notifications list (kept in history)."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return _flash("/notifications", "Post not found")
        p.attention_dismissed_at = utcnow()
    return _flash(next, "Dismissed")


# --- Engagement ----------------------------------------------------------------

@app.get("/engagement")
def engagement_page():
    """Replies are reviewed on each post page now — keep this URL as a redirect."""
    return RedirectResponse("/calendar", status_code=303)


@app.post("/engagement/sync")
def engagement_sync(next: str = Form("")):
    dest = next or "/calendar"
    with session_scope() as session:
        try:
            result = sync_comments(session)
            return _flash(dest, f"Synced: {result['new_comments']} new comments")
        except Exception as exc:
            return _flash(dest, f"Sync failed: {exc}")


@app.post("/engagement/{comment_id}/post")
def engagement_post(request: Request, comment_id: int,
                    reply_text: str = Form(""), gif_id: str = Form(""),
                    next: str = Form("")):
    wants_json = "application/json" in request.headers.get("accept", "")
    dest = next or "/calendar"
    text = reply_text.strip()
    gif = gif_id.strip()
    if not text and not gif:
        return (JSONResponse({"error": "Reply needs text or a GIF"}, status_code=400)
                if wants_json else _flash(dest, "Reply needs text or a GIF"))
    with session_scope() as session:
        comment = session.get(ThreadsComment, comment_id)
        if comment is None:
            msg = "Comment not found"
            return (JSONResponse({"error": msg}, status_code=404)
                    if wants_json else _flash(dest, msg))
        if comment.reply_status == "posted":
            msg = "Already replied to this comment"
            return (JSONResponse({"error": msg}, status_code=409)
                    if wants_json else _flash(dest, msg))
        try:
            post_approved_reply(session, comment, text, gif_id=gif or None)
            return (JSONResponse({"ok": True}) if wants_json
                    else _flash(dest, "Reply posted"))
        except PacingLimitError as exc:
            return (JSONResponse({"error": str(exc)}, status_code=429)
                    if wants_json else _flash(dest, str(exc)))
        except Exception as exc:
            return (JSONResponse({"error": f"Post failed: {exc}"}, status_code=500)
                    if wants_json else _flash(dest, f"Post failed: {exc}"))


@app.get("/giphy/search")
def giphy_search(q: str = "", limit: int = 24):
    """Proxy Giphy search/trending so the API key stays server-side."""
    from ..giphy import GiphyError, is_configured, search as giphy_search_fn
    if not is_configured():
        return JSONResponse({"error": "GIPHY_API_KEY not set"}, status_code=503)
    try:
        items = giphy_search_fn(q, limit=limit)
        return JSONResponse({"gifs": items})
    except GiphyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": f"Giphy failed: {exc}"}, status_code=500)


# --- Analytics -----------------------------------------------------------------

@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request, msg: str = ""):
    with session_scope() as session:
        report = generate_report(session)
    return templates.TemplateResponse(
        request, "analytics.html",
        {"rows": report["rows"], "slices": report["slices"], "digest": report["digest"],
         "timeseries": report["timeseries"], "summary": report["summary"],
         "spend_today": spend.today_spend(), "spend_budget": spend.daily_budget(),
         "spend_recent": spend.recent(30),
         "msg": msg, "active": "analytics"},
    )


@app.post("/analytics/snapshot")
def analytics_snapshot():
    with session_scope() as session:
        try:
            n = snapshot_metrics(session)
            return _flash("/analytics", f"Took {n} metric snapshots")
        except Exception as exc:
            return _flash("/analytics", f"Snapshot failed: {exc}")


@app.post("/post/{post_id}/first-reply")
def retry_first_reply(post_id: int, next: str = Form("")):
    """Post the configured first reply under a published post (manual / retry)."""
    dest = next or f"/post/{post_id}"
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status != "published" or not p.threads_media_id:
            return _flash(dest, "No published post to reply under")
        if p.first_reply_id:
            return _flash(dest, "First reply already posted")
        # Manual action: use current configured text even if auto-post is disabled.
        if maybe_post_first_reply(session, p, force=True):
            return _flash(dest, "First reply posted")
        err = p.first_reply_error or "First reply not posted — set text under Replies settings"
        return _flash(dest, err)


@app.post("/post/{post_id}/suggest-attribution")
def suggest_post_attribution(post_id: int):
    """(Re)draft the attribution first-comment for a post. Returns the suggestion
    only — it is saved when the operator updates the queue, so nothing changes
    until they've seen it."""
    with session_scope() as session:
        p = session.execute(
            select(ThreadsPost)
            .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel))
            .where(ThreadsPost.id == post_id)
        ).scalar_one_or_none()
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if p.candidate is None:
            return JSONResponse(
                {"error": "No source video on record for this post, so there's "
                          "nothing to attribute from."},
                status_code=409)
        try:
            text = generate_attribution(p.candidate)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
    if not text:
        return JSONResponse({"error": "The model returned an empty attribution — try again."},
                            status_code=500)
    return {"text": text}


# --- Keywords ----------------------------------------------------------------

@app.get("/keywords", response_class=HTMLResponse)
def keywords_page(request: Request, msg: str = ""):
    keywords = sorted(load_keywords())
    # How often has each keyword actually matched a stored candidate?
    with session_scope() as session:
        rows = session.execute(select(Candidate.matched_keywords)).all()
    hits: dict[str, int] = {}
    for (matched,) in rows:
        for kw in (matched or "").split(","):
            kw = kw.strip().lower()
            if kw:
                hits[kw] = hits.get(kw, 0) + 1
    return templates.TemplateResponse(
        request, "keywords.html",
        {"keywords": keywords, "hits": hits, "msg": msg, "active": "keywords"},
    )


@app.post("/keywords/add")
def keyword_add(keyword: str = Form(...)):
    kw = keyword.strip().lower()
    if not kw:
        return _flash("/keywords", "Empty keyword")
    keywords = load_keywords()
    if kw in [k.lower() for k in keywords]:
        return _flash("/keywords", f"'{kw}' is already in the list")
    save_keywords([*keywords, kw])
    return _flash("/keywords", f"Added '{kw}' — applies from the next monitor run")


@app.post("/keywords/delete")
def keyword_delete(keyword: str = Form(...)):
    kw = keyword.strip().lower()
    keywords = [k for k in load_keywords() if k.lower() != kw]
    save_keywords(keywords)
    return _flash("/keywords", f"Removed '{kw}'")


# --- First reply (under Replies) ---------------------------------------------

@app.get("/engagement/first-reply", response_class=HTMLResponse)
def first_reply_page(request: Request, msg: str = ""):
    cfg = load_first_reply()
    return templates.TemplateResponse(
        request, "first_reply.html",
        {"enabled": cfg["enabled"], "text": cfg["text"],
         "attribution_enabled": cfg["attribution_enabled"],
         "msg": msg, "active": "engagement"},
    )


@app.post("/engagement/first-reply")
def first_reply_save(enabled: str = Form(""), text: str = Form(""),
                     attribution_enabled: str = Form("")):
    text = (text or "").strip()
    on = str(enabled).lower() in ("1", "true", "on", "yes")
    attribution_on = str(attribution_enabled).lower() in ("1", "true", "on", "yes")
    if on and not text:
        return _flash("/engagement/first-reply", "Add reply text before enabling")
    if len(text) > 500:
        return _flash("/engagement/first-reply", f"Reply is {len(text)} characters — Threads limit is 500")
    save_first_reply(enabled=on, text=text, attribution_enabled=attribution_on)
    state = "enabled" if on else "disabled"
    attr_state = "on" if attribution_on else "off"
    return _flash("/engagement/first-reply",
                  f"Saved — attribution comments {attr_state}, static reply {state}")


@app.get("/first-reply")
def first_reply_redirect():
    return RedirectResponse("/engagement/first-reply", status_code=303)


# --- Style guide (caption drafting prompt) -----------------------------------

@app.get("/style-guide", response_class=HTMLResponse)
def style_guide_page(request: Request, msg: str = ""):
    from ..caption_insights import has_generated, load_suggestions

    with session_scope() as session:
        suggestions = load_suggestions(session)
        generated = has_generated(session)
    return templates.TemplateResponse(
        request, "style_guide.html",
        {"rules": load_caption_rules(), "suggestions": suggestions,
         "has_generated": generated, "msg": msg, "active": "style_guide"},
    )


@app.post("/style-guide")
def style_guide_save(rules_json: str = Form("[]")):
    try:
        raw = json.loads(rules_json or "[]")
        if not isinstance(raw, list):
            raise ValueError("rules must be a list")
    except (ValueError, TypeError) as exc:
        return _flash("/style-guide", f"Could not save rules: {exc}")
    save_caption_rules(raw)
    n = len(load_caption_rules())
    return _flash("/style-guide", f"Saved {n} style rule{'s' if n != 1 else ''}")


@app.post("/style-guide/insights/dismiss")
def style_guide_dismiss_insight(key: str = Form(...)):
    from ..caption_insights import dismiss_insight

    with session_scope() as session:
        dismiss_insight(session, key)
    return JSONResponse({"ok": True})


@app.post("/style-guide/suggestions/refresh")
def style_guide_refresh_suggestions():
    """Distill editorial rule suggestions from the operator's captions (LLM)."""
    from ..caption_insights import generate_suggestions

    try:
        with session_scope() as session:
            items = generate_suggestions(session)
        return JSONResponse({"ok": True, "items": items})
    except Exception as exc:
        log.exception("Caption rule suggestion failed")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


# --- Traits (flat footage vocabulary + post-performance learning) ---------------

def _normalize_trait(name: str) -> str:
    """Snake_case a trait name so it stays consistent with the seed + model output."""
    return "_".join(name.strip().lower().split())


@app.get("/traits", response_class=HTMLResponse)
def traits_page(request: Request, msg: str = ""):
    settings = load_settings()
    with session_scope() as session:
        traits = session.execute(select(Trait).order_by(Trait.name)).scalars().all()
        weight_rows = session.execute(
            select(TraitWeight).where(TraitWeight.metric == "views")
        ).scalars().all()
        weights = {
            w.trait: {"lift": w.lift, "n_posts": w.n_posts or 0,
                      "effective_n": w.effective_n, "status": w.status,
                      "median_metric": w.median_metric, "baseline": w.baseline}
            for w in weight_rows
        }
        # Verdict summary table (moved here from Analytics), best lift first.
        trait_weights = sorted(
            ({"trait": w.trait, "n_posts": w.n_posts or 0, "status": w.status,
              "median_metric": w.median_metric, "baseline": w.baseline, "lift": w.lift}
             for w in weight_rows),
            key=lambda d: (d["lift"] if d["lift"] is not None else -99), reverse=True)
        baseline = next((w.baseline for w in weight_rows if w.baseline is not None), None)
        post_tag_rows = session.execute(
            select(ThreadsPost.footage_traits).where(ThreadsPost.status == "published")
        ).all()
        published_total = session.execute(
            select(func.count(ThreadsPost.id)).where(ThreadsPost.status == "published")
        ).scalar_one()
        unannotated = session.execute(
            select(func.count(ThreadsPost.id)).where(
                ThreadsPost.status == "published",
                ThreadsPost.footage_scored_at.is_(None),
                ThreadsPost.clip_local_path != "",
            )
        ).scalar_one()
    post_counts: dict[str, int] = {}
    annotated_posts = 0
    for (v,) in post_tag_rows:
        tags = [t.strip() for t in (v or "").split(",") if t.strip()]
        if tags:
            annotated_posts += 1
        for t in tags:
            post_counts[t] = post_counts.get(t, 0) + 1
    return templates.TemplateResponse(
        request, "traits.html",
        {"traits": traits, "post_counts": post_counts,
         "annotated_posts": annotated_posts,
         "published_total": published_total, "unannotated": unannotated,
         "weights": weights, "baseline": baseline, "trait_weights": trait_weights,
         "min_trait_posts": settings.get("learning.min_trait_posts", 20),
         "min_total_posts": settings.get("learning.min_total_posts", 100),
         "metric_age_hours": settings.get("learning.metric_age_hours", 48),
         "backfill_running": _post_annotate_running.is_set(),
         "msg": msg, "active": "traits"},
    )


@app.get("/traits/{trait_name}", response_class=HTMLResponse)
def trait_detail(request: Request, trait_name: str):
    """Published posts carrying this ground-truth footage trait + latest metrics."""
    name = _normalize_trait(trait_name)
    with session_scope() as session:
        weight = session.execute(
            select(TraitWeight).where(TraitWeight.trait == name, TraitWeight.metric == "views")
        ).scalar_one_or_none()
        posts = session.execute(
            select(ThreadsPost)
            .options(
                selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
                selectinload(ThreadsPost.cut),
            )
            .where(
                ThreadsPost.status == "published",
                func.concat(",", func.coalesce(ThreadsPost.footage_traits, ""), ",")
                .like(f"%,{name},%"),
            )
            .order_by(ThreadsPost.published_at.desc().nullslast())
        ).scalars().all()
        rows = []
        for p in posts:
            snap = session.execute(
                select(MetricSnapshot)
                .where(MetricSnapshot.post_pk == p.id)
                .order_by(MetricSnapshot.captured_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            rows.append({
                "post": p,
                "views": snap.views if snap else None,
                "likes": snap.likes if snap else None,
            })
        weight_dict = None
        if weight is not None:
            weight_dict = {"n_posts": weight.n_posts, "lift": weight.lift,
                           "status": weight.status, "median_metric": weight.median_metric,
                           "baseline": weight.baseline}
    return templates.TemplateResponse(
        request, "trait_detail.html",
        {"trait": name, "posts": rows, "weight": weight_dict, "active": "traits"},
    )


# Background footage-trait backfill (one at a time; it's LLM + ffmpeg work).
_post_annotate_running = threading.Event()


def _annotate_posts_in_thread() -> None:
    settings = load_settings()
    try:
        with session_scope() as session:
            traits = active_traits(session)
            posts = session.execute(
                select(ThreadsPost).where(
                    ThreadsPost.status == "published",
                    ThreadsPost.footage_scored_at.is_(None),
                    ThreadsPost.clip_local_path != "",
                ).order_by(ThreadsPost.published_at.desc())
            ).scalars().all()
            done = 0
            for post in posts:
                if not spend.within_budget():
                    log.info("Footage backfill stopped: daily budget reached")
                    break
                if annotate_post_footage(post, settings, traits):
                    done += 1
                    session.commit()
            from ..analytics import learn_trait_weights
            learn_trait_weights(session)
            log.info("Footage backfill annotated %d post(s)", done)
    except Exception:
        log.exception("Footage trait backfill failed")
    finally:
        _post_annotate_running.clear()


@app.post("/traits/annotate-posts")
def traits_annotate_posts():
    """Backfill ground-truth footage traits for published posts whose clip
    files are still on disk (runs in the background, budget-guarded)."""
    if _post_annotate_running.is_set():
        return _flash("/traits", "A backfill is already running")
    _post_annotate_running.set()
    threading.Thread(target=_annotate_posts_in_thread, daemon=True).start()
    return _flash("/traits", "Backfill started — annotating published clips in the background")


@app.post("/traits/relearn")
def traits_relearn():
    """Recompute trait verdicts from the current post annotations + metrics."""
    from ..analytics import learn_trait_weights

    with session_scope() as session:
        results = learn_trait_weights(session)
    active_n = sum(1 for r in results if r["status"] == TraitWeight.STATUS_ACTIVE)
    return _flash("/traits", f"Verdicts recomputed: {len(results)} trait(s) seen, {active_n} active")


@app.post("/traits/add")
def trait_add(name: str = Form(...)):
    name = _normalize_trait(name)
    if not name:
        return _flash("/traits", "Empty trait name")
    with session_scope() as session:
        exists = session.execute(select(Trait).where(Trait.name == name)).scalar_one_or_none()
        if exists:
            return _flash("/traits", f"'{name}' already exists")
        session.add(Trait(name=name, kind=Trait.KIND_NEUTRAL, enabled=True))
    invalidate_traits_cache()
    return _flash("/traits", f"Added '{name}'")


@app.post("/traits/{trait_id}/toggle")
def trait_toggle(trait_id: int):
    with session_scope() as session:
        t = session.get(Trait, trait_id)
        if t:
            t.enabled = not t.enabled
    invalidate_traits_cache()
    return _flash("/traits", "Updated")


@app.post("/traits/{trait_id}/delete")
def trait_delete(trait_id: int):
    with session_scope() as session:
        t = session.get(Trait, trait_id)
        if t:
            session.delete(t)
    invalidate_traits_cache()
    return _flash("/traits", "Deleted")


# --- Channels ----------------------------------------------------------------

@app.get("/channels", response_class=HTMLResponse)
def channels_page(request: Request, msg: str = ""):
    with session_scope() as session:
        channels = session.execute(
            select(Channel).order_by(Channel.market, Channel.call_sign)
        ).scalars().all()
    return templates.TemplateResponse(
        request, "channels.html", {"channels": channels, "msg": msg, "active": "channels"}
    )


@app.post("/channels/add")
def channel_add(call_sign: str = Form(...), network: str = Form(""), market: str = Form(""),
                region: str = Form(""), country: str = Form(""), scope: str = Form("local"),
                url: str = Form(...)):
    scope = scope.strip().lower()
    if scope not in ("local", "national", "international"):
        scope = "local"
    with session_scope() as session:
        exists = session.execute(select(Channel).where(Channel.url == url.strip())).scalar_one_or_none()
        if exists:
            return _flash("/channels", "A channel with that URL already exists")
        session.add(Channel(call_sign=call_sign.strip(), network=network.strip(),
                            market=market.strip(), region=region.strip(),
                            country=country.strip(), scope=scope, url=url.strip()))
    return _flash("/channels", f"Added {call_sign}")


@app.post("/channels/prefill")
def channel_prefill(url: str = Form(...)):
    """Resolve a YouTube channel URL and let an agent draft the editorial fields.

    Returns JSON the add-channel dialog uses to prefill its inputs. Nothing is
    saved — the operator reviews and submits the form as usual."""
    url = url.strip()
    if not url:
        return JSONResponse({"error": "Enter a YouTube channel URL first"}, status_code=400)
    try:
        info = youtube.resolve_channel(url)
    except youtube.YouTubeAPIError as exc:
        return JSONResponse({"error": f"Could not resolve channel: {exc}"}, status_code=400)

    with session_scope() as session:
        existing = session.execute(
            select(Channel).where(
                or_(Channel.url == url, Channel.channel_id == info["channel_id"])
            )
        ).scalar_one_or_none()
        already = existing.call_sign if existing else None

    # Recent upload titles give the agent extra signal (best-effort, low quota).
    recent_titles: list[str] = []
    try:
        uploads = youtube.list_recent_uploads(
            info["uploads_playlist_id"], dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=120),
            max_results=10,
        )
        recent_titles = [u.title for u in uploads]
    except youtube.YouTubeAPIError:
        pass

    settings = load_settings()
    model = settings.get("engagement.draft_model", "claude-sonnet-5")
    try:
        fields = suggest_channel_fields(
            model, url, title=info.get("title", ""),
            description=info.get("description", ""),
            country_code=info.get("country", ""), recent_titles=recent_titles,
        )
    except Exception as exc:  # noqa: BLE001 — surface any LLM/parse error to the UI
        return JSONResponse({"error": f"Agent prefill failed: {exc}"}, status_code=500)

    return JSONResponse({
        "fields": fields,
        "channel_title": info.get("title", ""),
        "channel_id": info["channel_id"],
        "already_exists": already,
    })


@app.post("/channels/import-csv")
def channels_import_csv(csv_text: str = Form(...)):
    """Bulk-add channels from pasted CSV:
    call_sign,network,market,region,country,scope,url (or minimal: call_sign,url).
    Header rows, comments, and blanks are skipped; duplicates (by URL) are ignored."""
    import csv as csv_mod
    import io

    added, skipped, errors = 0, 0, []
    with session_scope() as session:
        existing_urls = {u for (u,) in session.execute(select(Channel.url)).all()}
        for lineno, row in enumerate(csv_mod.reader(io.StringIO(csv_text)), 1):
            cells = [c.strip() for c in row if c.strip()]
            if not cells or cells[0].startswith("#"):
                continue
            # Skip a header row.
            if lineno == 1 and cells[0].lower() in ("call_sign", "callsign", "call sign"):
                continue
            url = next((c for c in cells if "youtube.com/" in c), "")
            if not url:
                errors.append(f"line {lineno}: no YouTube URL found")
                continue
            rest = [c for c in cells if c != url]
            call_sign = rest[0] if rest else url.rstrip("/").split("/")[-1].lstrip("@")
            network = rest[1] if len(rest) > 1 else ""
            market = rest[2] if len(rest) > 2 else ""
            region = rest[3] if len(rest) > 3 else ""
            country = rest[4] if len(rest) > 4 else ""
            scope = (rest[5] if len(rest) > 5 else "local").lower()
            if scope not in ("local", "national", "international"):
                scope = "local"
            if url in existing_urls:
                skipped += 1
                continue
            session.add(Channel(call_sign=call_sign, network=network, market=market,
                                region=region, country=country, scope=scope, url=url))
            existing_urls.add(url)
            added += 1

    msg = f"Imported {added} channel{'s' if added != 1 else ''}, skipped {skipped} duplicate{'s' if skipped != 1 else ''}"
    if errors:
        msg += f" — {len(errors)} line(s) had problems: " + "; ".join(errors[:3])
    return _flash("/channels", msg)


@app.post("/channels/{channel_id}/toggle")
def channel_toggle(channel_id: int):
    with session_scope() as session:
        channel = session.get(Channel, channel_id)
        if channel:
            channel.enabled = not channel.enabled
    return _flash("/channels", "Updated")


@app.post("/channels/{channel_id}/delete")
def channel_delete(channel_id: int):
    with session_scope() as session:
        channel = session.get(Channel, channel_id)
        if channel is None:
            return _flash("/channels", "Not found")
        n = session.execute(
            select(func.count(Candidate.id)).where(Candidate.channel_pk == channel_id)
        ).scalar_one()
        if n:
            channel.enabled = False
            return _flash("/channels", f"{channel.call_sign} has {n} stored candidates; disabled instead of deleted")
        session.delete(channel)
    return _flash("/channels", "Deleted")


# --- Product roadmap (scope only; not a build queue) -------------------------

@app.get("/product-roadmap", response_class=HTMLResponse)
def product_roadmap_page(request: Request):
    """Living scope doc for productization — no features are built from this page."""
    return templates.TemplateResponse(
        request, "product_roadmap.html",
        {"msg": "", "active": "product_roadmap"},
    )
