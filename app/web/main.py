"""Local review dashboard (FastAPI + Jinja). Single-operator, localhost only.

Run: python run.py dashboard   (serves http://127.0.0.1:8321)

Workflow per video: Review -> Scrape & Transcribe -> Trim -> Post, surfaced as
breadcrumb steps on the /video/{id} screen.
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

from .. import spend, threads_api
from ..analytics import generate_report, snapshot_metrics
from ..clipper import ClipExportError, clip_duration, export_supercut, get_waveform
from ..config import load_first_reply, load_keywords, load_settings, save_first_reply, save_keywords
from ..db import (
    SessionLocal,
    active_traits,
    init_db,
    session_scope,
    sync_channels_from_config,
    sync_traits_from_config,
)
from ..engagement import PacingLimitError, post_approved_reply, redraft_comment, sync_comments
from ..history import import_history
from ..llm import suggest_post_caption, suggest_title
from ..models import (
    STATUS_APPROVED,
    STATUS_ARCHIVED,
    STATUS_NEW,
    STATUS_REJECTED,
    Candidate,
    Channel,
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
from ..publishing import maybe_post_first_reply, publish_clip, publish_post, queue_clip, record_post
from ..ranking import load_trait_weights, order_expr, sort_candidates
from ..scheduler import (
    build_window_plan,
    pin_post_to_window,
    scheduler_status,
    spacing_allows_publish,
    start_scheduler_thread,
)
from ..scrape import archive_candidate
from ..vision import annotate_post_footage, tag_candidate_storyboard
from ..voice import voice_context

log = logging.getLogger("web")

app = FastAPI(title="Climate Clip Monitor")
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
# Cache-bust static assets whenever style.css changes so browsers pick up edits.
try:
    templates.env.globals["static_v"] = str(int((_STATIC_DIR / "style.css").stat().st_mtime))
except OSError:
    templates.env.globals["static_v"] = "0"

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

def workflow_state(session, c: Candidate, post_statuses: set[str] | None = None) -> dict:
    """Compute breadcrumb step states for a candidate.

    Pass ``post_statuses`` (the set of ThreadsPost.status values for this
    candidate) to skip the per-row published-post lookup — used by the
    dashboard/archive when rendering many rows against a remote DB.
    """
    if post_statuses is None:
        posted = session.execute(
            select(ThreadsPost.id).where(
                ThreadsPost.candidate_pk == c.id, ThreadsPost.status == "published"
            ).limit(1)
        ).scalar_one_or_none() is not None
    else:
        posted = "published" in post_statuses

    reviewed = c.status not in (STATUS_NEW, STATUS_REJECTED)
    scraped = c.status == STATUS_ARCHIVED
    trimmed = bool(c.trimmed_clip_path) and Path(c.trimmed_clip_path).exists()

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

        # Filter dropdown options: only channels that actually have candidates.
        filter_channels = session.execute(
            select(Channel).join(Candidate, Candidate.channel_pk == Channel.id)
            .distinct().order_by(Channel.call_sign)
        ).scalars().all()
        # Keyword filter chips come from the active keyword list (what we monitor
        # for), so removed/legacy terms never show up as filters.
        keywords_options = sorted(load_keywords())
        regions = [
            r for (r,) in session.execute(
                select(Channel.region).distinct().order_by(Channel.region)
            ).all() if r
        ]
        countries = [
            c for (c,) in session.execute(
                select(Channel.country).distinct().order_by(Channel.country)
            ).all() if c
        ]

        # Items mid-workflow (shown on the default view only).
        in_progress_rows = []
        if not filtering:
            in_progress = session.execute(
                select(Candidate)
                .options(selectinload(Candidate.channel))
                .where(Candidate.status.in_([STATUS_APPROVED, STATUS_ARCHIVED, "failed"]))
                .order_by(Candidate.approved_at.desc())
                .limit(30)
            ).scalars().all()
            # One query for all post statuses instead of 2×N per-row lookups.
            statuses = _post_statuses_by_candidate(session, [c.id for c in in_progress])
            handled_statuses = {"published", "queued", "publishing", "draft"}
            for c in in_progress:
                post_st = statuses.get(c.id, set())
                state = workflow_state(session, c, post_statuses=post_st)
                # Drop it from "In progress" once the clip has been dealt with:
                # trimmed clip exported/saved ("ready to post"), posted,
                # queued, or saved as a draft. A candidate with a FAILED
                # post stays visible so it can be retried.
                if "failed" in post_st:
                    state["post_failed"] = True  # keep the row visible for retry
                elif post_st & handled_statuses:
                    # Hide once published or sitting in the outbound queue/drafts.
                    # If the operator deletes their only draft/queue, post_st is
                    # empty and the clip stays visible so they can re-post.
                    continue
                in_progress_rows.append((c, state))

        monitor_running, monitor_result, monitor_last_refreshed = _monitor_view_state(session)

    return templates.TemplateResponse(
        request, "dashboard.html",
        {"candidates": candidates, "total_matches": total_matches, "row_cap": row_cap,
         "in_progress": in_progress_rows, "threshold": threshold,
         "date_defaulted": date_defaulted,
         "show_hidden": show_hidden, "filtering": filtering,
         "q": q, "channel_id": channel_id, "keyword": keyword, "region": region,
         "country": country, "scope": scope, "status": status,
         "date_from": date_from, "date_to": date_to,
         "filter_channels": filter_channels, "keywords_options": keywords_options,
         "regions": regions,
         "countries": countries, "scopes": ["local", "national", "international"],
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
                "rationale": c.relevance_rationale,
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


# --- Per-video workflow ----------------------------------------------------------

@app.get("/video/{candidate_id}", response_class=HTMLResponse)
def video_detail(request: Request, candidate_id: int, step: str = "", msg: str = ""):
    with session_scope() as session:
        c = session.execute(
            select(Candidate)
            .options(selectinload(Candidate.channel))
            .where(Candidate.id == candidate_id)
        ).scalar_one_or_none()
        if c is None:
            return _flash("/", "Video not found")
        state = workflow_state(session, c)

        # Operator can revisit any unlocked step; default to the current one.
        allowed = {"review"}
        if state["reviewed"]:
            allowed.add("scrape")
        if state["scraped"]:
            allowed.update({"trim", "post"})
        active_step = step if step in allowed else state["current"]

        transcript_segments = []
        if c.transcript_path and Path(c.transcript_path).exists():
            try:
                transcript_segments = json.loads(Path(c.transcript_path).read_text())
            except Exception:
                pass

        segments = []
        if c.trim_segments:
            try:
                segments = json.loads(c.trim_segments)
            except Exception:
                pass

        # Transcript covering just the exported segments, in clip order.
        clip_transcript = _excerpt_segments(transcript_segments, segments) if segments else []

        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.candidate_pk == c.id)
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()

    threads_ok = threads_api.is_authenticated()
    return templates.TemplateResponse(
        request, "video.html",
        {"c": c, "state": state, "step": active_step,
         "transcript_segments": transcript_segments, "saved_segments": segments,
         "clip_transcript": clip_transcript,
         "posts": posts, "threads_ok": threads_ok,
         "auth_url": "" if threads_ok else threads_api.authorize_url(),
         "subs_position": load_settings().get("subtitles.position", "bottom"),
         "msg": msg, "active": "dashboard"},
    )





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


@app.get("/media/clip/{candidate_id}")
def media_clip(candidate_id: int):
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path or not Path(c.trimmed_clip_path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        return FileResponse(c.trimmed_clip_path, media_type="video/mp4")


@app.get("/media/subtitled/{candidate_id}")
def media_subtitled(candidate_id: int):
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.subtitled_clip_path or not Path(c.subtitled_clip_path).exists():
            return JSONResponse({"error": "no subtitled clip"}, status_code=404)
        return FileResponse(c.subtitled_clip_path, media_type="video/mp4")


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
        if (not path or not Path(path).exists()) and p.candidate:
            c = p.candidate
            if c.use_subtitles and c.subtitled_clip_path and Path(c.subtitled_clip_path).exists():
                path = c.subtitled_clip_path
            else:
                path = c.trimmed_clip_path
        if not path or not Path(path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        return FileResponse(path, media_type="video/mp4")


@app.get("/video/{candidate_id}/download-clip")
def download_clip(candidate_id: int):
    """Serve the exported clip as a file attachment so the operator can save it
    locally and post it manually elsewhere."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path or not Path(c.trimmed_clip_path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        return FileResponse(c.trimmed_clip_path, media_type="video/mp4",
                            filename=f"{c.video_id or f'clip-{c.id}'}.mp4")


@app.get("/post/{post_id}/download-clip")
def download_post_clip(post_id: int):
    """Download the clip attached to a post (used from the Posts page)."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = p.clip_local_path
        if (not path or not Path(path).exists()) and p.candidate:
            path = p.candidate.trimmed_clip_path
        if not path or not Path(path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        return FileResponse(path, media_type="video/mp4",
                            filename=f"threads-post-{p.id}.mp4")


# --- Trim / export ----------------------------------------------------------------

@app.post("/video/{candidate_id}/export")
def export_clip(candidate_id: int, segments_json: str = Form(...)):
    try:
        segments = json.loads(segments_json)
        assert isinstance(segments, list) and segments
    except Exception:
        return _flash(f"/video/{candidate_id}?step=trim", "No segments to export")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.local_video_path:
            return _flash("/", "Video not found or not downloaded")
        try:
            out = export_supercut(c.local_video_path, segments, f"{c.video_id}_cut")
            c.trim_segments = json.dumps(segments)
            c.trimmed_clip_path = str(out)
            # Any previously generated captions no longer match the new cut.
            c.subtitled_clip_path = ""
            c.use_subtitles = False
            # Auto-title the fresh clip from its own transcript, but only when the
            # operator hasn't already set one (regeneration stays available in the
            # Post step). A titling failure must never block the export.
            if not (c.clip_title or "").strip():
                try:
                    settings = load_settings()
                    excerpt = _transcript_excerpt(c, segments)
                    title = suggest_title(
                        settings.get("engagement.draft_model", "claude-sonnet-5"),
                        c.title, excerpt, c.draft_caption or None,
                    )
                    if title:
                        c.clip_title = title
                except Exception:
                    pass
            n = len(segments)
            # autosubs=1 makes the Post step kick off caption generation
            # immediately, so the captioned variant is the default.
            return _flash(f"/video/{candidate_id}?step=post&autosubs=1",
                          f"Exported {n} segment{'s' if n > 1 else ''} — generating captions…")
        except ClipExportError as exc:
            return _flash(f"/video/{candidate_id}?step=trim", f"Export failed: {exc}")


@app.post("/video/{candidate_id}/subtitles")
def generate_subtitles(candidate_id: int, position: str = Form("")):
    """Generate the stylized-caption variant of the exported clip (AJAX).

    Runs whisper word timestamps + the Pillow/ffmpeg burn; takes roughly
    10-60s for a typical clip, longer on the first run while the whisper
    model downloads.
    """
    from ..subtitles import SubtitleError, create_subtitled_clip

    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path or not Path(c.trimmed_clip_path).exists():
            return JSONResponse({"error": "Export a clip first"}, status_code=404)
        clip_path = c.trimmed_clip_path
    try:
        out = create_subtitled_clip(clip_path, position=position or None)
    except SubtitleError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except Exception as exc:
        log.exception("Caption generation failed for candidate %s", candidate_id)
        return JSONResponse({"error": f"Caption generation failed: {exc}"}, status_code=500)
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is not None:
            c.subtitled_clip_path = str(out)
            c.use_subtitles = True
    return {"url": f"/media/subtitled/{candidate_id}"}


def _chosen_clip_path(c: Candidate, use_subtitles_form: str) -> str:
    """The file the operator wants to post: captioned variant when the box is
    ticked and the file exists, otherwise the plain export. Persists the choice."""
    want = str(use_subtitles_form).lower() in ("1", "true", "on", "yes")
    c.use_subtitles = want and bool(c.subtitled_clip_path)
    if c.use_subtitles and Path(c.subtitled_clip_path).exists():
        return c.subtitled_clip_path
    return c.trimmed_clip_path


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


@app.post("/video/{candidate_id}/suggest-caption")
def suggest_caption(candidate_id: int):
    settings = load_settings()
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        segments = json.loads(c.trim_segments) if c.trim_segments else []
        excerpt = _transcript_excerpt(c, segments)
        seconds = clip_duration(c.trimmed_clip_path) if c.trimmed_clip_path else None
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
            )
            c.draft_caption = caption
            return {"caption": caption, "voice_examples": len(voice["examples"])}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/video/{candidate_id}/suggest-title")
def suggest_clip_title(candidate_id: int):
    settings = load_settings()
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        segments = json.loads(c.trim_segments) if c.trim_segments else []
        excerpt = _transcript_excerpt(c, segments)
        try:
            title = suggest_title(
                settings.get("engagement.draft_model", "claude-sonnet-5"),
                c.title, excerpt, c.draft_caption or None,
            )
            if title:
                c.clip_title = title
            return {"title": title or c.clip_title}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/video/{candidate_id}/post")
def post_to_threads(candidate_id: int, caption: str = Form(...),
                    clip_title: str = Form(""), use_subtitles: str = Form("")):
    """Operator-confirmed publish of the exported clip."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/video/{candidate_id}?step=post", "Caption is empty")
    with session_scope() as session:
        ok, wait_min = spacing_allows_publish(session)
        if not ok:
            return _flash(
                f"/video/{candidate_id}?step=post",
                f"Spacing floor: wait ~{wait_min} more minute{'s' if wait_min != 1 else ''} "
                f"before publishing another post",
            )
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path:
            return _flash(f"/video/{candidate_id}", "Export a clip first")
        c.clip_title = clip_title.strip()
        try:
            post = publish_clip(session, c, _chosen_clip_path(c, use_subtitles), caption)
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
                msg += f" · first reply failed: {post.first_reply_error[:120]}"
            return _flash(f"/video/{candidate_id}?step=post", msg)
        except Exception as exc:
            return _flash(f"/video/{candidate_id}?step=post", f"Publish failed: {exc}")


@app.post("/video/{candidate_id}/queue")
def queue_to_threads(candidate_id: int, caption: str = Form(...),
                     is_breaking: str = Form(""), clip_title: str = Form(""),
                     use_subtitles: str = Form("")):
    """Add the exported clip to the adaptive FIFO queue (no immediate post)."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/video/{candidate_id}?step=post", "Caption is empty")
    breaking = str(is_breaking).lower() in ("1", "true", "on", "yes")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path:
            return _flash(f"/video/{candidate_id}", "Export a clip first")
        c.clip_title = clip_title.strip()
        clip_path = _chosen_clip_path(c, use_subtitles)
        try:
            # Reuse an existing not-yet-published post for this clip rather than
            # creating a duplicate queue entry.
            existing = session.execute(
                select(ThreadsPost).where(
                    ThreadsPost.candidate_pk == c.id,
                    ThreadsPost.status.in_(["queued", "draft", "failed"]),
                ).order_by(ThreadsPost.created_at.desc())
            ).scalars().all()
            if existing:
                keep = existing[0]
                keep.caption = caption
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
                keep.is_breaking = breaking
                keep.defer_count = 0
                keep.last_deferred_at = None
                keep.pinned_window_key = ""
                keep.error = ""
                for extra in existing[1:]:
                    session.delete(extra)
            else:
                queue_clip(session, c, clip_path, caption, is_breaking=breaking)
        except Exception as exc:
            return _flash(f"/video/{candidate_id}?step=post", f"Queue failed: {exc}")
    note = " (breaking — publishes ASAP)" if breaking else ""
    return _flash(f"/video/{candidate_id}?step=post",
                  f"Added to the posting queue{note}")


@app.post("/video/{candidate_id}/save-draft")
def save_draft(candidate_id: int, caption: str = Form(...),
               clip_title: str = Form(""), use_subtitles: str = Form("")):
    """Save the exported clip + caption as a draft to publish or queue later."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/video/{candidate_id}?step=post", "Caption is empty")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path:
            return _flash(f"/video/{candidate_id}", "Export a clip first")
        c.clip_title = clip_title.strip()
        try:
            record_post(session, c, _chosen_clip_path(c, use_subtitles), caption, status="draft")
        except Exception as exc:
            return _flash(f"/video/{candidate_id}?step=post", f"Save failed: {exc}")
    return _flash(f"/video/{candidate_id}?step=post",
                  "Saved as draft — publish or queue it any time from Posts")


@app.post("/post/{post_id}/cancel")
def cancel_queued_post(post_id: int, next: str = Form("/posts")):
    """Remove a queued or draft (not-yet-published) post."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return _flash(next, "Post not found")
        if p.status not in ("queued", "draft"):
            return _flash(next, "Only queued or draft posts can be removed")
        was = p.status
        candidate_id = p.candidate_pk
        session.delete(p)
        # Send the operator back to the clip when deleting its only post record,
        # so they don't lose track of a trimmed export.
        if candidate_id and (not next or next in ("/posts", "/")):
            next = f"/video/{candidate_id}?step=post"
    label = "Queued post" if was == "queued" else "Draft"
    return _flash(next, f"{label} removed — your clip is still here; queue or post again when ready.")


@app.post("/post/{post_id}/queue")
def queue_existing_post(post_id: int, caption: str = Form(""),
                        is_breaking: str = Form(""), next: str = Form("/posts")):
    """Move a draft/failed post into the adaptive queue (or update a queued one)."""
    breaking = str(is_breaking).lower() in ("1", "true", "on", "yes")
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status not in ("draft", "failed", "queued"):
            return _flash(next, "Only a draft, failed, or queued post can be (re)queued")
        if caption.strip():
            p.caption = caption.strip()
        p.status = "queued"
        p.scheduled_at = None
        p.is_breaking = breaking
        if breaking:
            p.pinned_window_key = ""
        p.error = ""
        if p.candidate_pk is not None:
            dupes = session.execute(
                select(ThreadsPost).where(
                    ThreadsPost.candidate_pk == p.candidate_pk,
                    ThreadsPost.id != p.id,
                    ThreadsPost.status.in_(["queued", "draft", "failed"]),
                )
            ).scalars().all()
            for extra in dupes:
                session.delete(extra)
    note = " as breaking" if breaking else ""
    return _flash(next, f"Added to the posting queue{note}")


@app.post("/post/{post_id}/toggle-breaking")
def toggle_breaking(post_id: int, next: str = Form("/posts")):
    """Flip the breaking-news flag on a queued/draft post."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status not in ("queued", "draft", "failed"):
            return _flash(next, "Only a queued, draft, or failed post can be marked breaking")
        p.is_breaking = not bool(p.is_breaking)
        if p.is_breaking:
            p.pinned_window_key = ""
        if p.status in ("draft", "failed"):
            p.status = "queued"
            p.scheduled_at = None
            p.error = ""
        flag = p.is_breaking
    return _flash(next, "Marked breaking — will publish ASAP" if flag else "Cleared breaking flag")


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


def _publish_in_thread(post_id: int) -> None:
    """Publish a post in the background. publish_post sets status to
    published/failed (+ error) itself, so we just swallow the exception here."""
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


@app.post("/post/{post_id}/publish-now")
def publish_scheduled_now(request: Request, post_id: int, next: str = Form("/posts")):
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


_POST_METRICS = ("views", "likes", "replies", "reposts", "quotes", "shares")


@app.get("/post/{post_id}", response_class=HTMLResponse)
def post_detail(request: Request, post_id: int, msg: str = ""):
    """A single post's profile: manage the queue before it publishes, and once
    it's live, see its stats and the replies it received."""
    with session_scope() as session:
        p = session.execute(
            select(ThreadsPost)
            .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel))
            .where(ThreadsPost.id == post_id)
        ).scalar_one_or_none()
        if p is None:
            return _flash("/posts", "Post not found")

        cand = p.candidate
        clip_path = p.clip_local_path if (p.clip_local_path and Path(p.clip_local_path).exists()) else ""
        if not clip_path and cand:
            if (cand.use_subtitles and cand.subtitled_clip_path
                    and Path(cand.subtitled_clip_path).exists()):
                clip_path = cand.subtitled_clip_path
            elif cand.trimmed_clip_path and Path(cand.trimmed_clip_path).exists():
                clip_path = cand.trimmed_clip_path
        has_clip = bool(clip_path)
        # Burnt-in captions live in *_subs.mp4; surface that on the post page
        # so the plain Threads text caption isn't confused with video subs.
        has_burned_captions = bool(clip_path and clip_path.endswith("_subs.mp4"))
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
            {"username": c.username, "text": c.text,
             "classification": c.classification, "reply_status": c.reply_status,
             "reply_text": c.reply_text_posted,
             "commented_at": c.commented_at}
            for c in comments
        ]
        ctx = {
            "pid": p.id, "status": p.status, "caption": p.caption or "",
            "permalink": p.permalink, "source": p.source, "error": p.error,
            "candidate_id": cand.id if cand else None,
            "channel_sign": cand.channel.call_sign if (cand and cand.channel) else "",
            "video_title": cand.title if cand else "",
            "clip_title": cand.clip_title if cand else "",
            "has_clip": has_clip,
            "has_burned_captions": has_burned_captions,
            "scheduled_at": p.scheduled_at, "published_at": p.published_at,
            "created_at": p.created_at,
            "is_breaking": bool(p.is_breaking),
            "defer_count": int(p.defer_count or 0),
            "last_deferred_at": p.last_deferred_at,
            "first_reply_id": p.first_reply_id or "",
            "first_reply_text": p.first_reply_text or "",
            "first_reply_error": p.first_reply_error or "",
            "first_reply_at": p.first_reply_at,
            "metrics": metrics, "metrics_captured": snap.captured_at if snap else None,
            "snapshot_count": snapshot_count,
            "comments": comment_rows,
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
    """Pull and classify the replies on this (and other) published posts."""
    dest = next or f"/post/{post_id}"
    with session_scope() as session:
        try:
            result = sync_comments(session)
            return _flash(dest, f"Synced: {result['new_comments']} new replies, "
                                f"{result['drafts']} drafts")
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
def threads_connect(code: str = Form(...), next: str = Form("/posts")):
    try:
        threads_api.exchange_code(_clean_auth_code(code))
        return _flash(next, "Threads connected")
    except Exception as exc:
        return _flash(next, f"Auth failed: {exc}")


# --- Archive -----------------------------------------------------------------

@app.get("/archive", response_class=HTMLResponse)
def archive_page(request: Request, country: str = "", scope: str = "", msg: str = ""):
    with session_scope() as session:
        query = (
            select(Candidate)
            .options(selectinload(Candidate.channel))
            .where(Candidate.status == STATUS_ARCHIVED)
            .order_by(Candidate.archived_at.desc())
        )
        channel_filters = []
        if country:
            channel_filters.append(Channel.country == country)
        if scope:
            channel_filters.append(Channel.scope == scope)
        if channel_filters:
            query = query.join(Channel, Candidate.channel_pk == Channel.id).where(*channel_filters)
        items = session.execute(query).scalars().all()
        statuses = _post_statuses_by_candidate(session, [c.id for c in items])
        rows = [(c, workflow_state(session, c, post_statuses=statuses.get(c.id, set())))
                for c in items]

        # Filter options limited to countries/scopes that have archived items.
        archived_channel_ids = select(Candidate.channel_pk).where(Candidate.status == STATUS_ARCHIVED)
        countries = [
            c for (c,) in session.execute(
                select(Channel.country).where(Channel.id.in_(archived_channel_ids))
                .distinct().order_by(Channel.country)
            ).all() if c
        ]
    return templates.TemplateResponse(
        request, "archive.html",
        {"rows": rows, "country": country, "scope": scope,
         "countries": countries, "scopes": ["local", "national", "international"],
         "filtering": bool(country or scope), "msg": msg, "active": "archive"},
    )


# --- Posts (history + manual publish + Threads connect) --------------------------

@app.post("/threads/import-history")
def threads_import_history():
    """Pull the account's own existing Threads posts into the DB, then kick off
    an insights snapshot for them in the background."""
    if not threads_api.is_authenticated():
        return _flash("/posts", "Connect Threads first")
    with session_scope() as session:
        try:
            result = import_history(session)
        except Exception as exc:
            return _flash("/posts", f"Import failed: {exc}")

    def _pull_insights():
        try:
            with session_scope() as s:
                snapshot_metrics(s)
        except Exception as exc:  # pragma: no cover - background best-effort
            logging.getLogger("history").warning("Post-import snapshot failed: %s", exc)

    threading.Thread(target=_pull_insights, daemon=True).start()
    return _flash(
        "/posts",
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
            if e["kind"] == "breaking":
                continue  # breaking is ASAP — linear only
            events.setdefault(e["day"], []).append(e)

        # Linear queue: breaking + upcoming windows only (not published history).
        linear = [e for e in plan if e["kind"] in ("breaking", "queued", "open")]
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
         "linear": linear, "scheduler": status, "windows_et": windows_et,
         "msg": msg, "active": "calendar"},
    )


@app.get("/posts", response_class=HTMLResponse)
def posts_page(request: Request, msg: str = ""):
    post_opts = (
        selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
    )
    with session_scope() as session:
        # Two queries instead of three: active queue/drafts + recent history.
        active = session.execute(
            select(ThreadsPost).options(*post_opts)
            .where(ThreadsPost.status.in_(["queued", "publishing", "draft"]))
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
        queued = sorted(
            [p for p in active if p.status in ("queued", "publishing")],
            key=lambda p: p.created_at or utcnow(),
        )
        drafts = [p for p in active if p.status == "draft"]
        posts = session.execute(
            select(ThreadsPost).options(*post_opts)
            .where(ThreadsPost.status.notin_(["queued", "publishing", "draft"]))
            .order_by(ThreadsPost.created_at.desc()).limit(100)
        ).scalars().all()
        status = scheduler_status(session)
    authenticated = threads_api.is_authenticated()
    return templates.TemplateResponse(
        request, "posts.html",
        {"posts": posts, "queued": queued, "drafts": drafts,
         "scheduler": status,
         "authenticated": authenticated,
         "auth_url": threads_api.authorize_url() if not authenticated else "",
         "msg": msg, "active": "posts"},
    )


# --- Engagement ----------------------------------------------------------------

@app.get("/engagement", response_class=HTMLResponse)
def engagement_page(request: Request, view: str = "queue", msg: str = ""):
    with session_scope() as session:
        # Eager-load the related post in one round-trip. Without this, the template
        # lazy-loads c.post per row (N+1), which is painfully slow against remote Postgres.
        base = select(ThreadsComment).options(selectinload(ThreadsComment.post))
        if view == "filtered":
            comments = session.execute(
                base.where(ThreadsComment.reply_status.in_(["filtered", "skipped"]))
                .order_by(ThreadsComment.created_at.desc()).limit(200)
            ).scalars().all()
        elif view == "posted":
            comments = session.execute(
                base.where(ThreadsComment.reply_status == "posted")
                .order_by(ThreadsComment.replied_at.desc()).limit(200)
            ).scalars().all()
        else:
            comments = session.execute(
                base.where(
                    ThreadsComment.reply_status == "pending", ThreadsComment.eligible_for_reply
                ).order_by(ThreadsComment.created_at.desc()).limit(200)
            ).scalars().all()
    return templates.TemplateResponse(
        request, "engagement.html", {"comments": comments, "view": view, "msg": msg, "active": "engagement"}
    )


@app.post("/engagement/sync")
def engagement_sync():
    with session_scope() as session:
        try:
            result = sync_comments(session)
            return _flash("/engagement", f"Synced: {result['new_comments']} new comments, {result['drafts']} drafts")
        except Exception as exc:
            return _flash("/engagement", f"Sync failed: {exc}")


@app.post("/engagement/{comment_id}/post")
def engagement_post(request: Request, comment_id: int, reply_text: str = Form(...)):
    wants_json = "application/json" in request.headers.get("accept", "")
    text = reply_text.strip()
    if not text:
        return (JSONResponse({"error": "Reply text is empty"}, status_code=400)
                if wants_json else _flash("/engagement", "Reply text is empty"))
    with session_scope() as session:
        comment = session.get(ThreadsComment, comment_id)
        if comment is None or not comment.eligible_for_reply or comment.reply_status != "pending":
            msg = "Comment is not eligible or already handled"
            return (JSONResponse({"error": msg}, status_code=409)
                    if wants_json else _flash("/engagement", msg))
        try:
            post_approved_reply(session, comment, text)
            return (JSONResponse({"ok": True}) if wants_json
                    else _flash("/engagement", "Reply posted"))
        except PacingLimitError as exc:
            return (JSONResponse({"error": str(exc)}, status_code=429)
                    if wants_json else _flash("/engagement", str(exc)))
        except Exception as exc:
            return (JSONResponse({"error": f"Post failed: {exc}"}, status_code=500)
                    if wants_json else _flash("/engagement", f"Post failed: {exc}"))


@app.post("/engagement/{comment_id}/redraft")
def engagement_redraft(request: Request, comment_id: int):
    """Regenerate a queued comment's draft reply with the latest reply guidance."""
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        comment = session.get(ThreadsComment, comment_id)
        if comment is None:
            return (JSONResponse({"error": "Comment not found"}, status_code=404)
                    if wants_json else _flash("/engagement", "Comment not found"))
        try:
            new_draft = redraft_comment(session, comment)
            return (JSONResponse({"ok": True, "draft": new_draft or ""})
                    if wants_json else _flash("/engagement", "Draft regenerated"))
        except Exception as exc:
            return (JSONResponse({"error": f"Redraft failed: {exc}"}, status_code=500)
                    if wants_json else _flash("/engagement", f"Redraft failed: {exc}"))


@app.post("/engagement/{comment_id}/skip")
def engagement_skip(request: Request, comment_id: int):
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        comment = session.get(ThreadsComment, comment_id)
        if comment:
            comment.reply_status = "skipped"
    return (JSONResponse({"ok": True}) if wants_json else _flash("/engagement", "Skipped"))


# --- Analytics -----------------------------------------------------------------

@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request, msg: str = ""):
    settings = load_settings()
    with session_scope() as session:
        report = generate_report(session)
    return templates.TemplateResponse(
        request, "analytics.html",
        {"rows": report["rows"], "slices": report["slices"], "digest": report["digest"],
         "timeseries": report["timeseries"], "summary": report["summary"],
         "trait_weights": report.get("trait_weights", []),
         "min_trait_posts": settings.get("learning.min_trait_posts", 20),
         "min_total_posts": settings.get("learning.min_total_posts", 100),
         "spend_today": spend.today_spend(), "spend_budget": spend.daily_budget(),
         "spend_recent": spend.recent(7),
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
        {"enabled": cfg["enabled"], "text": cfg["text"], "msg": msg, "active": "engagement"},
    )


@app.post("/engagement/first-reply")
def first_reply_save(enabled: str = Form(""), text: str = Form("")):
    text = (text or "").strip()
    on = str(enabled).lower() in ("1", "true", "on", "yes")
    if on and not text:
        return _flash("/engagement/first-reply", "Add reply text before enabling")
    if len(text) > 500:
        return _flash("/engagement/first-reply", f"Reply is {len(text)} characters — Threads limit is 500")
    save_first_reply(enabled=on, text=text)
    state = "enabled" if on else "disabled"
    return _flash("/engagement/first-reply", f"Saved — auto first reply is {state}")


@app.get("/first-reply")
def first_reply_redirect():
    return RedirectResponse("/engagement/first-reply", status_code=303)


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
         "weights": weights, "baseline": baseline,
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
            .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel))
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
    return _flash("/traits", f"Added '{name}'")


@app.post("/traits/{trait_id}/toggle")
def trait_toggle(trait_id: int):
    with session_scope() as session:
        t = session.get(Trait, trait_id)
        if t:
            t.enabled = not t.enabled
    return _flash("/traits", "Updated")


@app.post("/traits/{trait_id}/delete")
def trait_delete(trait_id: int):
    with session_scope() as session:
        t = session.get(Trait, trait_id)
        if t:
            session.delete(t)
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
