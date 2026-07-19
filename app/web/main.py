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

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import and_, func, or_, select

from .. import spend, threads_api
from ..analytics import generate_report, snapshot_metrics
from ..clipper import ClipExportError, clip_duration, export_supercut, get_waveform
from ..config import load_keywords, load_settings, save_keywords
from ..db import (
    SessionLocal,
    active_traits,
    init_db,
    session_scope,
    sync_channels_from_config,
    sync_traits_from_config,
)
from ..engagement import PacingLimitError, post_approved_reply, sync_comments
from ..history import import_history
from ..llm import suggest_post_caption
from ..models import (
    STATUS_APPROVED,
    STATUS_ARCHIVED,
    STATUS_NEW,
    STATUS_REJECTED,
    Candidate,
    Channel,
    ThreadsComment,
    ThreadsPost,
    Trait,
    utcnow,
)
from ..monitor import run_monitor_once
from ..publishing import publish_clip, publish_post, record_post, schedule_clip
from ..ranking import load_trait_weights, order_expr, sort_candidates, trait_guidance_text
from ..scheduler import start_scheduler_thread
from ..scrape import archive_candidate
from ..vision import apply_visual_score

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

# Publish scheduled posts in the background while the dashboard runs.
start_scheduler_thread()


def _flash(url: str, msg: str) -> RedirectResponse:
    sep = "&" if "?" in url else "?"
    return RedirectResponse(f"{url}{sep}msg={msg}", status_code=303)


# --- Workflow step helpers ----------------------------------------------------

def workflow_state(session, c: Candidate) -> dict:
    """Compute breadcrumb step states for a candidate."""
    posted = session.execute(
        select(ThreadsPost.id).where(
            ThreadsPost.candidate_pk == c.id, ThreadsPost.status == "published"
        ).limit(1)
    ).scalar_one_or_none() is not None

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


# --- Dashboard -----------------------------------------------------------------

def _parse_date(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except (ValueError, TypeError):
        return None


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, q: str = "", channel_id: int = 0,
              topic: list[str] = Query(default=[]),
              region: str = "", country: str = "", scope: str = "",
              status: str = "new", date_from: str = "", date_to: str = "",
              show_hidden: int = 0, msg: str = ""):
    settings = load_settings()
    threshold = settings.get("matching.score_threshold", 0.5)
    topic = [t for t in topic if t.strip()]
    filtering = bool(q or channel_id or topic or region or country or scope
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
        query = select(Candidate).order_by(
            order_expr(settings).desc(), Candidate.published_at.desc()
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
        if topic:
            # climate_topic is a CSV list; match rows containing ANY selected topic.
            query = query.where(or_(
                *[("," + Candidate.climate_topic + ",").like(f"%,{t},%") for t in topic]
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
        # Re-rank the fetched page by the blended relevance+visual score (nudged
        # by learned trait weights). SQL already ordered by relevance so the page
        # is the strongest set; this reorders it to surface visually punchy clips.
        trait_weights = load_trait_weights(session)
        candidates = sort_candidates(candidates, trait_weights, settings)
        undesirable_traits = active_traits(session)[1]
        for c in candidates:
            _ = c.channel

        # Filter dropdown options: only channels/topics that actually have candidates.
        filter_channels = session.execute(
            select(Channel).join(Candidate, Candidate.channel_pk == Channel.id)
            .distinct().order_by(Channel.call_sign)
        ).scalars().all()
        raw_topic_values = session.execute(select(Candidate.climate_topic).distinct()).all()
        topics = sorted({
            t.strip() for (v,) in raw_topic_values for t in (v or "").split(",") if t.strip()
        })
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
                .where(Candidate.status.in_([STATUS_APPROVED, STATUS_ARCHIVED, "failed"]))
                .order_by(Candidate.approved_at.desc())
                .limit(30)
            ).scalars().all()
            for c in in_progress:
                state = workflow_state(session, c)
                # Drop it from "In progress" once the clip has been dealt with:
                # posted, scheduled, or saved as a draft for later. (A failed
                # post stays so it can be retried.)
                handled = session.execute(
                    select(ThreadsPost.id).where(
                        ThreadsPost.candidate_pk == c.id,
                        ThreadsPost.status.in_(["published", "scheduled", "publishing", "draft"]),
                    ).limit(1)
                ).scalar_one_or_none() is not None
                if handled:
                    continue
                in_progress_rows.append((c, state))
                _ = c.channel

    return templates.TemplateResponse(
        request, "dashboard.html",
        {"candidates": candidates, "total_matches": total_matches, "row_cap": row_cap,
         "in_progress": in_progress_rows, "threshold": threshold,
         "undesirable_traits": undesirable_traits,
         "date_defaulted": date_defaulted,
         "show_hidden": show_hidden, "filtering": filtering,
         "q": q, "channel_id": channel_id, "topic": topic, "region": region,
         "country": country, "scope": scope, "status": status,
         "date_from": date_from, "date_to": date_to,
         "filter_channels": filter_channels, "topics": topics, "regions": regions,
         "countries": countries, "scopes": ["local", "national", "international"],
         "monitor_running": _monitor_state["running"],
         "monitor_result": _monitor_state["last_result"],
         "msg": msg, "active": "dashboard"},
    )


# Monitor passes run in a background thread; state is read by the dashboard.
_monitor_state = {"running": False, "last_result": ""}


def _monitor_in_thread(days: int | None) -> None:
    scope = f"last {days} days" if days else "since last check"
    try:
        result = run_monitor_once(days)
        _monitor_state["last_result"] = (
            f"Last pass ({scope}): {result['channels_checked']} channels checked, "
            f"{result['candidates_stored']} new candidates, "
            f"{result.get('vision_scored', 0)} vision-scored "
            f"(spent ${spend.today_spend():.2f} of ${spend.daily_budget():.2f} today)"
        )
    except Exception as exc:
        log.exception("Monitor pass failed")
        _monitor_state["last_result"] = f"Monitor pass failed: {exc}"
    finally:
        _monitor_state["running"] = False


@app.post("/monitor/run")
def monitor_now(lookback_days: str = Form("")):
    if _monitor_state["running"]:
        return _flash("/", "A monitor pass is already running — refresh to see progress")
    days: int | None = None
    if lookback_days.strip():
        try:
            days = max(1, min(int(lookback_days), 30))
        except ValueError:
            days = None
    _monitor_state["running"] = True
    _monitor_state["last_result"] = ""
    threading.Thread(target=_monitor_in_thread, args=(days,), daemon=True).start()
    scope = f"backfilling {days} days" if days else "checking since last run"
    return _flash("/", f"Monitor started ({scope}) — running in the background, refresh for updates")


# --- Triage mode (one at a time, keyboard-driven) --------------------------------

@app.get("/triage", response_class=HTMLResponse)
def triage(request: Request, q: str = "", channel_id: int = 0,
           topic: list[str] = Query(default=[]),
           region: str = "", country: str = "", scope: str = "",
           date_from: str = "", date_to: str = "",
           show_hidden: int = 0, msg: str = ""):
    """Focused review: new candidates one at a time with keyboard actions."""
    settings = load_settings()
    threshold = settings.get("matching.score_threshold", 0.5)
    topic = [t for t in topic if t.strip()]

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
        if topic:
            query = query.where(or_(
                *[("," + Candidate.climate_topic + ",").like(f"%,{t},%") for t in topic]
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
        candidates = session.execute(query.limit(200)).scalars().all()
        trait_weights = load_trait_weights(session)
        candidates = sort_candidates(candidates, trait_weights, settings)
        undesirable_traits = active_traits(session)[1]
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
                "visual_score": c.visual_score,
                "visual_traits": [t for t in (c.visual_traits or "").split(",") if t],
                "visual_rationale": c.visual_rationale,
                "topics": [t for t in (c.climate_topic or "").split(",") if t],
                "keywords": c.matched_keywords,
                "rationale": c.relevance_rationale,
            }
            for c in candidates
        ]

    return templates.TemplateResponse(
        request, "triage.html",
        {"queue": queue, "threshold": threshold, "undesirable_traits": undesirable_traits,
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
    """On-demand vision scoring for one candidate (respects the daily budget).
    Used by the triage/detail 'Score visuals' button and to refresh a score."""
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
        guidance = trait_guidance_text(load_trait_weights(session), settings)
        desirable, undesirable = active_traits(session)
        result = apply_visual_score(c, settings, force=True, learned_guidance=guidance,
                                    desirable=desirable, undesirable=undesirable)
        if result is None:
            return JSONResponse(
                {"error": "No storyboard available for this video, or scoring failed."},
                status_code=502,
            )
        return {"visual_score": result["visual_score"], "traits": result["traits"],
                "why": result["why"]}


# --- Per-video workflow ----------------------------------------------------------

@app.get("/video/{candidate_id}", response_class=HTMLResponse)
def video_detail(request: Request, candidate_id: int, step: str = "", msg: str = ""):
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/", "Video not found")
        state = workflow_state(session, c)
        _ = c.channel

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

        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.candidate_pk == c.id)
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
        undesirable_traits = active_traits(session)[1]

    return templates.TemplateResponse(
        request, "video.html",
        {"c": c, "state": state, "step": active_step,
         "transcript_segments": transcript_segments, "saved_segments": segments,
         "undesirable_traits": undesirable_traits,
         "posts": posts, "threads_ok": threads_api.is_authenticated(),
         "auth_url": "" if threads_api.is_authenticated() else threads_api.authorize_url(),
         "msg": msg, "active": "dashboard"},
    )


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


@app.post("/video/{candidate_id}/approve")
def approve(candidate_id: int):
    """The approve gate. This is the ONLY place a download is ever triggered."""
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/", "Video not found")
        if c.status == STATUS_ARCHIVED:
            return _flash(f"/video/{candidate_id}", "Already archived")
        c.status = STATUS_APPROVED
        c.approved_at = utcnow()
    threading.Thread(target=_scrape_in_thread, args=(candidate_id,), daemon=True).start()
    return _flash(f"/video/{candidate_id}", "Approved — downloading and transcribing now")


@app.post("/video/{candidate_id}/reject")
def reject(candidate_id: int):
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c:
            c.status = STATUS_REJECTED
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
            n = len(segments)
            return _flash(f"/video/{candidate_id}?step=post",
                          f"Exported {n} segment{'s' if n > 1 else ''} — ready to post")
        except ClipExportError as exc:
            return _flash(f"/video/{candidate_id}?step=trim", f"Export failed: {exc}")


# --- Caption suggestion + posting -------------------------------------------------

def _transcript_excerpt(c: Candidate, segments: list[dict]) -> str:
    """Transcript text inside the trimmed windows (falls back to full text)."""
    try:
        all_segments = json.loads(Path(c.transcript_path).read_text())
    except Exception:
        return c.transcript_text[:3000]
    if not segments:
        return c.transcript_text[:3000]
    parts = []
    for window in segments:
        ws, we = float(window["start"]), float(window["end"])
        for s in all_segments:
            if s["end"] >= ws and s["start"] <= we:
                parts.append(s["text"])
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
        try:
            caption = suggest_post_caption(
                settings.get("engagement.draft_model", "claude-sonnet-5"),
                c.title, c.channel.call_sign, c.channel.market, excerpt, seconds,
            )
            c.draft_caption = caption
            return {"caption": caption}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/video/{candidate_id}/post")
def post_to_threads(candidate_id: int, caption: str = Form(...)):
    """Operator-confirmed publish of the exported clip."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/video/{candidate_id}?step=post", "Caption is empty")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path:
            return _flash(f"/video/{candidate_id}", "Export a clip first")
        try:
            post = publish_clip(session, c, c.trimmed_clip_path, caption)
            return _flash(f"/video/{candidate_id}?step=post",
                          f"Published: {post.permalink or post.threads_media_id}")
        except Exception as exc:
            return _flash(f"/video/{candidate_id}?step=post", f"Publish failed: {exc}")


def _parse_local_datetime(value: str) -> dt.datetime | None:
    """Parse an <input type=datetime-local> value (naive, in the operator's
    local timezone) into an aware UTC datetime. This is a localhost, single-
    operator app, so the browser and server share the same local timezone."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        naive = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    # astimezone() on a naive datetime interprets it as system local time.
    return naive.astimezone(dt.timezone.utc)


@app.post("/video/{candidate_id}/schedule")
def schedule_to_threads(candidate_id: int, caption: str = Form(...),
                        scheduled_at: str = Form(...)):
    """Queue the exported clip to publish at a future time (no immediate post)."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/video/{candidate_id}?step=post", "Caption is empty")
    when = _parse_local_datetime(scheduled_at)
    if when is None:
        return _flash(f"/video/{candidate_id}?step=post", "Pick a valid date and time")
    if when <= utcnow():
        return _flash(f"/video/{candidate_id}?step=post", "Scheduled time must be in the future")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path:
            return _flash(f"/video/{candidate_id}", "Export a clip first")
        try:
            schedule_clip(session, c, c.trimmed_clip_path, caption, when)
        except Exception as exc:
            return _flash(f"/video/{candidate_id}?step=post", f"Schedule failed: {exc}")
    local = when.astimezone()
    return _flash(f"/video/{candidate_id}?step=post",
                  f"Scheduled for {local.strftime('%b %-d, %-I:%M %p')}")


@app.post("/video/{candidate_id}/save-draft")
def save_draft(candidate_id: int, caption: str = Form(...)):
    """Save the exported clip + caption as a draft to publish or schedule later."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/video/{candidate_id}?step=post", "Caption is empty")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None or not c.trimmed_clip_path:
            return _flash(f"/video/{candidate_id}", "Export a clip first")
        try:
            record_post(session, c, c.trimmed_clip_path, caption, status="draft")
        except Exception as exc:
            return _flash(f"/video/{candidate_id}?step=post", f"Save failed: {exc}")
    return _flash(f"/video/{candidate_id}?step=post",
                  "Saved as draft — publish or schedule it any time from Posts")


@app.post("/post/{post_id}/cancel")
def cancel_scheduled_post(post_id: int, next: str = Form("/posts")):
    """Remove a scheduled or draft (not-yet-published) post."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return _flash(next, "Post not found")
        if p.status not in ("scheduled", "draft"):
            return _flash(next, "Only scheduled or draft posts can be removed")
        was = p.status
        session.delete(p)
    return _flash(next, f"{'Scheduled post' if was == 'scheduled' else 'Draft'} removed")


@app.post("/post/{post_id}/schedule")
def schedule_existing_post(post_id: int, scheduled_at: str = Form(...),
                           caption: str = Form(""), next: str = Form("/posts")):
    """Turn a draft (or failed) post into a scheduled one."""
    when = _parse_local_datetime(scheduled_at)
    if when is None:
        return _flash(next, "Pick a valid date and time")
    if when <= utcnow():
        return _flash(next, "Scheduled time must be in the future")
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status not in ("draft", "failed"):
            return _flash(next, "Only a draft can be scheduled")
        if caption.strip():
            p.caption = caption.strip()
        p.status = "scheduled"
        p.scheduled_at = when
        p.error = ""
    local = when.astimezone()
    return _flash(next, f"Scheduled for {local.strftime('%b %-d, %-I:%M %p')}")


@app.post("/post/{post_id}/publish-now")
def publish_scheduled_now(post_id: int, next: str = Form("/posts")):
    """Publish a scheduled, draft, or previously failed post immediately."""
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None or p.status not in ("scheduled", "failed", "draft"):
            return _flash(next, "Nothing to publish")
        try:
            publish_post(session, p)
            return _flash(next, f"Published: {p.permalink or p.threads_media_id}")
        except Exception as exc:
            return _flash(next, f"Publish failed: {exc}")


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
            select(Candidate).where(Candidate.status == STATUS_ARCHIVED)
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
        rows = [(c, workflow_state(session, c)) for c in items]
        for c in items:
            _ = c.channel

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
    """Month grid of scheduled + published posts, in the operator's local time."""
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
    start_utc = first_local.astimezone(dt.timezone.utc)
    end_utc = next_first_local.astimezone(dt.timezone.utc)

    events: dict[int, list[dict]] = {}
    drafts_count = 0
    with session_scope() as session:
        rows = session.execute(
            select(ThreadsPost).where(
                or_(
                    and_(ThreadsPost.scheduled_at >= start_utc, ThreadsPost.scheduled_at < end_utc),
                    and_(ThreadsPost.published_at >= start_utc, ThreadsPost.published_at < end_utc),
                )
            )
        ).scalars().all()
        drafts_count = session.execute(
            select(func.count()).select_from(ThreadsPost).where(ThreadsPost.status == "draft")
        ).scalar_one()
        for p in rows:
            when = p.scheduled_at if p.status in ("scheduled", "publishing") else p.published_at
            if when is None:
                continue
            local = when.astimezone()
            events.setdefault(local.day, []).append({
                "time": local.strftime("%-I:%M %p"),
                "sort": local,
                "caption": (p.caption or "").strip(),
                "status": p.status,
                "video_id": p.candidate.id if p.candidate else None,
                "permalink": p.permalink,
            })
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
         "drafts_count": drafts_count, "msg": msg, "active": "calendar"},
    )


@app.get("/posts", response_class=HTMLResponse)
def posts_page(request: Request, msg: str = ""):
    with session_scope() as session:
        scheduled = session.execute(
            select(ThreadsPost).where(ThreadsPost.status.in_(["scheduled", "publishing"]))
            .order_by(ThreadsPost.scheduled_at.asc())
        ).scalars().all()
        drafts = session.execute(
            select(ThreadsPost).where(ThreadsPost.status == "draft")
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.status.notin_(["scheduled", "publishing", "draft"]))
            .order_by(ThreadsPost.created_at.desc()).limit(100)
        ).scalars().all()
        for p in [*scheduled, *drafts, *posts]:
            _ = p.candidate
    return templates.TemplateResponse(
        request, "posts.html",
        {"posts": posts, "scheduled": scheduled, "drafts": drafts,
         "authenticated": threads_api.is_authenticated(),
         "auth_url": threads_api.authorize_url() if not threads_api.is_authenticated() else "",
         "msg": msg, "active": "posts"},
    )


# --- Engagement ----------------------------------------------------------------

@app.get("/engagement", response_class=HTMLResponse)
def engagement_page(request: Request, view: str = "queue", msg: str = ""):
    with session_scope() as session:
        if view == "filtered":
            comments = session.execute(
                select(ThreadsComment).where(ThreadsComment.reply_status.in_(["filtered", "skipped"]))
                .order_by(ThreadsComment.created_at.desc()).limit(200)
            ).scalars().all()
        elif view == "posted":
            comments = session.execute(
                select(ThreadsComment).where(ThreadsComment.reply_status == "posted")
                .order_by(ThreadsComment.replied_at.desc()).limit(200)
            ).scalars().all()
        else:
            comments = session.execute(
                select(ThreadsComment).where(
                    ThreadsComment.reply_status == "pending", ThreadsComment.eligible_for_reply
                ).order_by(ThreadsComment.created_at.desc()).limit(200)
            ).scalars().all()
        for c in comments:
            _ = c.post
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
def engagement_post(comment_id: int, reply_text: str = Form(...)):
    text = reply_text.strip()
    if not text:
        return _flash("/engagement", "Reply text is empty")
    with session_scope() as session:
        comment = session.get(ThreadsComment, comment_id)
        if comment is None or not comment.eligible_for_reply or comment.reply_status != "pending":
            return _flash("/engagement", "Comment is not eligible or already handled")
        try:
            post_approved_reply(session, comment, text)
            return _flash("/engagement", "Reply posted")
        except PacingLimitError as exc:
            return _flash("/engagement", str(exc))
        except Exception as exc:
            return _flash("/engagement", f"Post failed: {exc}")


@app.post("/engagement/{comment_id}/skip")
def engagement_skip(comment_id: int):
    with session_scope() as session:
        comment = session.get(ThreadsComment, comment_id)
        if comment:
            comment.reply_status = "skipped"
    return _flash("/engagement", "Skipped")


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
         "trait_min_posts": settings.get("ranking.trait_min_posts", 8),
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


# --- Traits (visual desirable/undesirable trait database) -----------------------

def _normalize_trait(name: str) -> str:
    """Snake_case a trait name so it stays consistent with the seed + model output."""
    return "_".join(name.strip().lower().split())


@app.get("/traits", response_class=HTMLResponse)
def traits_page(request: Request, msg: str = ""):
    settings = load_settings()
    with session_scope() as session:
        traits = session.execute(select(Trait).order_by(Trait.name)).scalars().all()
        tag_rows = session.execute(select(Candidate.visual_traits)).all()
        weights = load_trait_weights(session)
        desirable = [t for t in traits if t.kind == Trait.KIND_DESIRABLE]
        undesirable = [t for t in traits if t.kind == Trait.KIND_UNDESIRABLE]
    counts: dict[str, int] = {}
    scored_total = 0
    for (v,) in tag_rows:
        tags = [t.strip() for t in (v or "").split(",") if t.strip()]
        if tags:
            scored_total += 1
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    return templates.TemplateResponse(
        request, "traits.html",
        {"desirable": desirable, "undesirable": undesirable, "counts": counts,
         "weights": weights, "scored_total": scored_total,
         "trait_min_posts": settings.get("ranking.trait_min_posts", 8),
         "msg": msg, "active": "traits"},
    )


@app.post("/traits/add")
def trait_add(name: str = Form(...), kind: str = Form("desirable"), description: str = Form("")):
    name = _normalize_trait(name)
    if not name:
        return _flash("/traits", "Empty trait name")
    if kind not in (Trait.KIND_DESIRABLE, Trait.KIND_UNDESIRABLE):
        kind = Trait.KIND_DESIRABLE
    with session_scope() as session:
        exists = session.execute(select(Trait).where(Trait.name == name)).scalar_one_or_none()
        if exists:
            return _flash("/traits", f"'{name}' already exists")
        session.add(Trait(name=name, kind=kind, description=description.strip(), enabled=True))
    return _flash("/traits", f"Added '{name}' — applies from the next scoring run")


@app.post("/traits/{trait_id}/toggle")
def trait_toggle(trait_id: int):
    with session_scope() as session:
        t = session.get(Trait, trait_id)
        if t:
            t.enabled = not t.enabled
    return _flash("/traits", "Updated")


@app.post("/traits/{trait_id}/update")
def trait_update(trait_id: int, kind: str = Form(...), description: str = Form("")):
    with session_scope() as session:
        t = session.get(Trait, trait_id)
        if t:
            if kind in (Trait.KIND_DESIRABLE, Trait.KIND_UNDESIRABLE):
                t.kind = kind
            t.description = description.strip()
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
