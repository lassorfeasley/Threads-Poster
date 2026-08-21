"""Local review dashboard (FastAPI + Jinja). Single-operator, localhost only.

Run: python run.py dashboard   (serves http://127.0.0.1:8321)

Workflow per video: Review → download/transcribe → Trim → Post. The /video/{id}
screen is a profile (player + transcript, clips in the rail); trimming happens
on /cut/{id}.
"""
from __future__ import annotations

import bisect
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
from sqlalchemy.orm import defer, object_session, selectinload

from .. import clip_proposals, instagram_api, spend, threads_api, youtube
from ..analytics import (generate_report, latest_metrics_bulk, metrics_at_age_bulk,
                         snapshot_metrics, write_and_store_digest)
from ..categories import category_by_slug, category_options
from ..clipper import ClipExportError, cached_still, clip_duration, export_supercut, get_waveform
from ..config import (
    DATA_DIR, DEFAULT_APP_NAME, WORKSPACE, current_workspace, env, load_brand,
    load_caption_rules, load_first_reply, load_keywords, load_settings,
    load_workspaces, render_caption_guide, save_brand, save_caption_rules,
    save_first_reply, save_keywords,
)
from ..db import (
    SessionLocal,
    active_traits_by_facet,
    init_db,
    invalidate_traits_cache,
    session_scope,
    sync_channels_from_config,
    sync_traits_from_config,
)
from ..draft_proposals import KIND_CAPTION, KIND_HOOK
from ..draft_proposals import log_proposal as log_draft_proposal
from ..draft_proposals import operator_written as operator_written_drafts
from ..draft_proposals import policy_version as draft_policy_version
from ..draft_proposals import record_verdict as record_draft_verdict
from ..engagement import PacingLimitError, post_approved_reply, sync_comments
from ..giphy import is_configured as giphy_configured
from ..history import import_history
from ..llm import (
    suggest_calendar_name,
    suggest_channel_fields,
    suggest_hook_text,
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
    ClipProposal,
    Cut,
    InstagramPost,
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
    draft_first_reply,
    draft_first_reply_for_cut,
    first_reply_context,
    generate_attribution,
    mark_publishing,
    maybe_post_first_reply,
    publish_clip,
    publish_instagram_post,
    publish_paired_reel,
    publish_post,
    publish_reel_now,
    queue_clip,
    record_instagram_post,
    record_post,
)
from ..placement import SHELF_LIVES
from ..ranking import load_trait_weights, order_expr, sort_candidates
from ..scheduler import (
    build_window_plan,
    expired_queued_posts,
    invalidate_recycle_overview,
    pin_post_to_window,
    projected_slot_for_post,
    recycle_overview,
    recycle_status,
    resolve_shelf_life,
    scheduler_status,
    shelf_life_outlook,
    spacing_allows_publish,
    start_scheduler_thread,
    window_time_labels,
)
from ..scrape import PASTED_CHANNEL_URL, archive_candidate, fetch_video_metadata
from ..vision import (
    annotate_cut_footage, annotate_post_footage, suggest_subs_position,
    tag_candidate_storyboard,
)
from ..voice import voice_context
from ..youtube import YouTubeAPIError, parse_video_url
from . import pagecache

log = logging.getLogger("web")

app = FastAPI(title="Clip Monitor")
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
# The uploaded brand logo is workspace data, not code: it lives under the
# workspace's data tree so one workspace's upload can never erase another's.
_BRAND_LOGO_DIR = DATA_DIR / "brand"
_BRAND_LOGO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/brand-static", StaticFiles(directory=str(_BRAND_LOGO_DIR)), name="brand-static")
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


def _unacknowledged(model):
    return (
        select(func.count()).select_from(model)
        .where(model.status == "failed", model.attention_dismissed_at.is_(None))
        .scalar_subquery()
    )


def stranded_reel_filters() -> tuple:
    """Criteria for a reel that got left behind: its Threads post is live, but
    the reel never went out and nothing will ever try again.

    The paired publish is best-effort and fires exactly once, immediately after
    the Threads post commits. A reel that missed that moment — a scheduler build
    predating reels, a crash in the gap — keeps the ``queued`` status it was
    born with, so every screen reports it as healthy and waiting while the
    window it was waiting for is long gone.
    """
    return (
        InstagramPost.status.in_(("queued", "draft")),
        InstagramPost.attention_dismissed_at.is_(None),
        select(ThreadsPost.id).where(
            ThreadsPost.id == InstagramPost.threads_post_pk,
            ThreadsPost.status == "published",
        ).exists(),
    )


def _stranded_reels():
    return (
        select(func.count()).select_from(InstagramPost)
        .where(*stranded_reel_filters()).scalar_subquery()
    )


def _load_attention_count() -> int:
    """Unacknowledged failed posts, failed reels, reels stranded behind an
    already-published post, and queued posts whose shelf life ran out."""
    with session_scope() as session:
        count = int(session.execute(
            select(_unacknowledged(ThreadsPost) + _unacknowledged(InstagramPost)
                   + _stranded_reels())
        ).scalar_one())
        try:
            count += len(expired_queued_posts(session))
        except Exception:
            log.exception("Expired-post check failed")
        return count


pagecache.register("attention", _load_attention_count)


def _attention_count() -> int:
    """The bell's count. It renders on every page, so it's kept warm in the
    background and must never raise."""
    return pagecache.read_or_last("attention", 0)


templates.env.globals["attention_count"] = _attention_count


def _threads_authenticated() -> bool:
    """Rendered on every page (sidebar nav dot), so it must never raise."""
    try:
        return threads_api.is_authenticated()
    except Exception:
        return False


templates.env.globals["threads_authenticated"] = _threads_authenticated


def _brand_name() -> str:
    """White-label app name for the sidebar/title. Renders on every page, so it
    must never raise."""
    try:
        return load_brand().get("app_name") or DEFAULT_APP_NAME
    except Exception:
        return DEFAULT_APP_NAME


def _brand_logo() -> str:
    """URL of the uploaded white-label logo, or "" for the default mark.
    mtime query busts the browser cache when a new logo replaces the old."""
    try:
        fn = load_brand().get("logo_file") or ""
        if not fn:
            return ""
        path = _BRAND_LOGO_DIR / fn
        return f"/brand-static/{fn}?v={int(path.stat().st_mtime)}"
    except Exception:
        return ""


def _ws_current() -> dict:
    """This process's workspace registry entry. Renders on every page, so it
    must never raise."""
    try:
        return current_workspace()
    except Exception:
        return {"slug": WORKSPACE, "label": WORKSPACE, "port": 0,
                "accent": "", "enabled": True}


def _workspace_options() -> list[dict]:
    """Workspaces for the sidebar switcher: every enabled registry entry plus
    the current one, each with the absolute URL of its own dashboard process.
    Never raises."""
    try:
        out = []
        for ws in load_workspaces():
            if not ws["enabled"] and ws["slug"] != WORKSPACE:
                continue
            out.append(dict(ws, current=(ws["slug"] == WORKSPACE),
                            url=f"http://127.0.0.1:{ws['port']}/"))
        return out
    except Exception:
        return []


templates.env.globals["brand_name"] = _brand_name
templates.env.globals["brand_logo"] = _brand_logo
templates.env.globals["default_app_name"] = DEFAULT_APP_NAME
templates.env.globals["ws_current"] = _ws_current
templates.env.globals["workspace_options"] = _workspace_options

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
    # In-process export threads die with the server; leave cuts marked exporting
    # looking stuck forever. Flip them to failed so the Post step can retry.
    for _cut in _s.execute(
        select(Cut).where(Cut.export_status == "exporting")
    ).scalars().all():
        _cut.export_status = "failed"
        _cut.export_error = "Interrupted — the server restarted while saving. Try again."
        _cut.updated_at = utcnow()

# Adaptive window scheduler (queue + hotness + metrics poll) while the dashboard runs.
# Configure logging first: this module is the uvicorn worker's entry point, so
# without it the scheduler's own log output has nowhere to go.
setup_logging()
# SCHEDULER_EMBEDDED=false skips the in-process scheduler when a dedicated
# worker (the Fly app) owns publishing — the dashboard then spends nothing on
# ticks, annotation, or metric polls. Publishing still overlaps safely when
# both run (the window claim is atomic); this flag is about not paying twice,
# and about keeping a dev dashboard from publishing at all.
if env("SCHEDULER_EMBEDDED", "true").strip().lower() not in ("false", "0", "no"):
    start_scheduler_thread()
# Keeps the list pages' datasets warm, so a page render doesn't wait on the
# database (see pagecache: only pages in active use are refreshed).
pagecache.start_refresher()


# Datasets a given write can actually change, keyed by the route's path
# pattern. Only the highest-frequency operator actions are mapped — a triage
# burst or a calendar drag shouldn't force every list page to rebuild against
# a remote database. Anything not listed falls back to invalidating everything,
# so a new route is stale-safe by default; add it here only once it's verified
# which cached pages its write can touch.
_WRITE_SCOPE: dict[str, tuple[str, ...]] = {
    # Triage: candidate status + triage ledger only. Candidates appear on the
    # dashboard lists and the library's Videos section, never on the
    # calendar/queue, notifications, or the attention count.
    "/video/{candidate_id}/approve": ("dashboard", "library"),
    "/video/{candidate_id}/reject": ("dashboard", "library"),
    "/video/{candidate_id}/reset": ("dashboard", "library"),
    "/video/{candidate_id}/unreject": ("dashboard", "library"),
    "/video/{candidate_id}/retry": ("dashboard", "library"),
    # Calendar drag-and-drop: pins only reorder upcoming windows. Shelf-life
    # expiry (what attention/notifications watch) is content-age-based and
    # unaffected by which window a post is pinned to.
    "/post/{post_id}/pin-window": ("calendar",),
    "/post/{post_id}/unpin": ("calendar",),
    # One metric snapshot row: of the cached pages, only the library surfaces
    # per-post metrics (analytics lives on its own TTL, volatile=False).
    "/post/{post_id}/refresh-stats": ("library",),
    # Comments aren't part of any cached dataset.
    "/post/{post_id}/sync-replies": (),
}


@app.middleware("http")
async def _drop_cached_reads_after_writes(request: Request, call_next):
    """Any successful write invalidates the cached reads.

    Hooking the request instead of ``_flash`` covers the endpoints that answer
    JSON to an in-page action, which don't redirect — and will cover the next
    one added without anyone having to remember.
    """
    response = await call_next(request)
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and response.status_code < 400:
        route = request.scope.get("route")
        scope = _WRITE_SCOPE.get(getattr(route, "path_format", None))
        if scope is None:
            pagecache.invalidate()
        else:
            for name in scope:
                pagecache.drop(name)
    return response


def _flash(url: str, msg: str) -> RedirectResponse:
    sep = "&" if "?" in url else "?"
    return RedirectResponse(f"{url}{sep}msg={msg}", status_code=303)


def _in_background(fn, *args) -> None:
    """Run a worker off the request path, dropping the cached reads when it's
    done. These threads change what the list pages show but never touch HTTP,
    so the write-invalidating middleware can't see them finish."""
    def _run() -> None:
        try:
            fn(*args)
        finally:
            pagecache.invalidate()

    threading.Thread(target=_run, daemon=True, name=getattr(fn, "__name__", "worker")).start()


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
        _in_background(_scrape_in_thread, cid)


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


# A Candidate row carries the video's whole description and transcript (the
# biggest here is ~1 MB on its own). List pages draw a title, a channel and a
# thumbnail from it and nothing else, so leaving those columns in the SELECT is
# pure transfer time — which is what dominates a page load against a remote DB.
# ``raiseload`` makes a missed column a loud error rather than a silent
# per-row refetch. Detail pages (video, archive, triage) load normally.
_CANDIDATE_LIST_ONLY = (
    defer(Candidate.description, raiseload=True),
    defer(Candidate.transcript_text, raiseload=True),
    defer(Candidate.relevance_rationale, raiseload=True),
    defer(Candidate.category_rationale, raiseload=True),
    defer(Candidate.visual_rationale, raiseload=True),
)

# Same idea for a post: the library tile shows the caption and the status, not
# the LLM's original draft or the footage annotations.
_POST_LIST_ONLY = (
    defer(ThreadsPost.suggested_caption, raiseload=True),
    defer(ThreadsPost.footage_traits, raiseload=True),
    defer(ThreadsPost.footage_rationale, raiseload=True),
    defer(ThreadsPost.attribution_text, raiseload=True),
    defer(ThreadsPost.first_reply_text, raiseload=True),
)


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


def _digest_meta(digest: dict) -> str:
    """The line under the digest: when it was written and what from. The page
    and the write action both render it, so it reads the same either way."""
    if not digest or not digest.get("text"):
        return "Not written yet"
    line = (f"Written {_relative_time_ago(digest.get('generated_at'))} "
            f"from {digest.get('post_count', 0)} posts")
    new = digest.get("new_posts") or 0
    if new:
        line += f" · {new} post{'s' if new != 1 else ''} published since"
    return line


templates.env.globals["digest_meta"] = _digest_meta


def _default_date_window(settings) -> tuple[str, str]:
    """Today + yesterday by publish date — what a bare visit to "/" shows."""
    window_days = settings.get("monitor.default_lookback_days", 2)
    today = dt.datetime.now(dt.timezone.utc).date()
    return ((today - dt.timedelta(days=max(window_days, 1) - 1)).isoformat(),
            today.isoformat())


def _dashboard_data(settings, threshold, q="", channel_id=0, keyword=(),
                    region="", country="", scope="", status="new",
                    date_from="", date_to="", show_hidden=0, filtering=False) -> dict:
    """The dashboard's reads: matching candidates plus the in-progress buckets."""
    with session_scope() as session:
        # Order by the blended relevance+visual ranking so the row cap keeps the
        # top-ranked candidates (not just the most relevant).
        query = (
            select(Candidate)
            .options(selectinload(Candidate.channel), *_CANDIDATE_LIST_ONLY)
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
                .options(selectinload(Candidate.channel), *_CANDIDATE_LIST_ONLY)
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

    return {
        "candidates": candidates, "total_matches": total_matches, "row_cap": row_cap,
        "in_progress": in_progress_rows, "trimmed": trimmed_rows,
        "keywords_options": keywords_options,
        "monitor_running": monitor_running,
        "monitor_result": monitor_result,
        "monitor_last_refreshed": monitor_last_refreshed,
        "date_from": date_from, "date_to": date_to,
    }


def _default_dashboard_data() -> dict:
    """The bare "/" view. Every action redirects here, so it's the one worth
    keeping warm; a filtered view is a one-off and goes straight to the DB."""
    settings = load_settings()
    date_from, date_to = _default_date_window(settings)
    return _dashboard_data(settings, settings.get("matching.score_threshold", 0.5),
                           date_from=date_from, date_to=date_to)


pagecache.register("dashboard", _default_dashboard_data)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, q: str = "", channel_id: int = 0,
              keyword: list[str] = Query(default=[]),
              region: str = "", country: str = "", scope: str = "",
              status: str = "new", date_from: str = "", date_to: str = "",
              show_hidden: int = 0, msg: str = "", view: str = "candidates"):
    settings = load_settings()
    threshold = settings.get("matching.score_threshold", 0.5)
    keyword = [k for k in keyword if k.strip()]
    filtering = bool(q or channel_id or keyword or region or country or scope
                     or date_from or date_to or status != "new")
    # Which of the three lists is open. Rendering it server-side keeps a linked
    # view from flashing the default one first.
    if view not in ("candidates", "inprogress", "trimmed"):
        view = "candidates"

    # On a bare visit (no query string), default the view to today + yesterday by
    # publish date. Any filter interaction submits a query string and is respected
    # as-is, so the operator can widen the window or clear it entirely.
    # A flash redirect lands here as "/?msg=…", and "?view=" only picks a list,
    # so neither is a filter interaction: counting them as one drops the window
    # and re-queries the whole backlog after every action.
    date_defaulted = not (set(request.query_params) - {"msg", "view"})
    if date_defaulted:
        data = pagecache.read("dashboard")
        date_from, date_to = data["date_from"], data["date_to"]
    else:
        data = _dashboard_data(settings, threshold, q=q, channel_id=channel_id,
                               keyword=keyword, region=region, country=country,
                               scope=scope, status=status, date_from=date_from,
                               date_to=date_to, show_hidden=show_hidden,
                               filtering=filtering)

    return templates.TemplateResponse(
        request, "dashboard.html",
        {**data, "threshold": threshold,
         "date_defaulted": date_defaulted,
         "show_hidden": show_hidden, "filtering": filtering,
         "q": q, "channel_id": channel_id, "keyword": keyword, "region": region,
         "country": country, "scope": scope, "status": status,
         "msg": msg, "view": view, "active": "dashboard"},
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
def monitor_now(request: Request, lookback_days: str = Form("")):
    wants_json = "application/json" in request.headers.get("accept", "")
    if _monitor_running.is_set():
        msg = "A monitor pass is already running — refresh to see progress"
        return JSONResponse({"ok": True, "msg": msg}) if wants_json else _flash("/", msg)
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
    _in_background(_monitor_in_thread, run_id, days)
    verb = f"backfilling {days} days" if days else "checking since last run"
    msg = f"Monitor started ({verb}) — running in the background"
    # Starting a pass changes nothing on screen except the badge, so the
    # dashboard swaps that itself rather than reloading to be told.
    return JSONResponse({"ok": True, "msg": msg}) if wants_json else _flash("/", msg)


@app.get("/monitor/status")
def monitor_status():
    """Just enough for the dashboard to watch a pass without re-rendering itself.
    A pass runs for minutes; polling this costs one query, where reloading the
    page costs a full render against a remote DB.

    ``ready`` says whether reloading would actually be cheap: a finished pass
    invalidates the dashboard's cached rows, and waiting the extra beat for the
    refresher beats making the operator watch a cold render."""
    with session_scope() as session:
        running, result, last_refreshed = _monitor_view_state(session)
    return JSONResponse({"running": running, "result": result,
                         "last_refreshed": last_refreshed,
                         "ready": pagecache.is_warm("dashboard")})


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
                # For the hand-off cards on the queue-clear screen, which are
                # built client-side from whatever the operator approved.
                "thumb": _thumb_for(c),
                "channel": f"{c.channel.call_sign} — {c.channel.market}",
                "published": c.published_at.strftime("%b %d, %Y %H:%M UTC") if c.published_at else "?",
                "duration": (f"{c.duration_seconds // 60}m {c.duration_seconds % 60}s"
                             if c.duration_seconds else ""),
                # Card overlays read as m:ss everywhere else in the app.
                "duration_short": (f"{c.duration_seconds // 60}:{c.duration_seconds % 60:02d}"
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
        vocab = active_traits_by_facet(session)
        result = tag_candidate_storyboard(c, settings, force=True,
                                          traits=vocab["subject"],
                                          format_traits=vocab["format"])
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
        # Keep category_rationale: it's the model's original reasoning, and
        # the scheduling panel shows it as provenance ("the AI said …") even
        # after an operator override. Wiping it left overrides unexplainable.
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

        reruns = recycle_overview(session)
        cut_rows = [
            {"cut": cut, "state": _cut_state(cut, posted_cut_pks),
             "post_count": posts_by_cut.get(cut.id, 0),
             "recycle": reruns.get(cut.id)}
            for cut in cuts
        ]

        transcript_segments = []
        if c.transcript_path and Path(c.transcript_path).exists():
            try:
                transcript_segments = json.loads(Path(c.transcript_path).read_text())
            except Exception:
                transcript_segments = []

        has_local = bool(c.local_video_path and Path(c.local_video_path).exists())
        trait_vocab = active_traits_by_facet(session)

    return templates.TemplateResponse(
        request, "video.html",
        {"c": c, "state": state, "step": step,
         "cut_rows": cut_rows, "posts": posts,
         "transcript_segments": transcript_segments,
         "has_local": has_local,
         "trait_vocab": trait_vocab,
         # Same chain the scheduler uses (candidate tag -> category default ->
         # evergreen), so the badge always shows what placement will do.
         "shelf_life": resolve_shelf_life(None, c),
         "shelf_life_tagged": bool((c.shelf_life or "").strip()),
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
        # Touching the toggle by hand makes it the operator's marker, not the
        # suggester's — a later pass must not silently flip it back.
        c.multi_clip_auto = False
        flag = c.multi_clip_potential
    return JSONResponse({"multi_clip": flag})


def _proposal_payload(row, used: list[dict]) -> dict | None:
    """One pending proposal, shaped for the trim editor's panel + overlay.

    Material already claimed by another clip is cut out here rather than at
    generation time: the archive-time pass runs before any cut exists, so a
    proposal only becomes a duplicate later. Nothing offered can overlap a clip
    that already exists, whatever the model asked for.
    """
    try:
        proposed = json.loads(row.proposed_segments or "[]")
    except (ValueError, TypeError):
        return None
    segments, trimmed = clip_proposals.visible_segments(proposed, used)
    if not segments:
        return None
    return {"id": row.id, "story": row.story, "why": row.why,
            "confidence": row.confidence, "segments": segments,
            "seconds": clip_proposals.duration(segments), "trimmed": trimmed}


def _pending_proposals(session, candidate_pk: int) -> list[dict]:
    used = clip_proposals.used_ranges(session, candidate_pk)
    rows = clip_proposals.pending_for_candidate(session, candidate_pk)
    return [p for p in (_proposal_payload(r, used) for r in rows) if p]


@app.post("/video/{candidate_id}/suggest-clips")
def video_suggest_clips(candidate_id: int):
    """Re-run the clip suggester for one video (operator-initiated).

    Deliberately ungated, matching the other on-demand suggest routes: the
    budget guard exists to stop an unattended monitor pass from spending the
    day's allowance, not to refuse a person who just clicked a button.
    """
    settings = load_settings()
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return JSONResponse({"error": "Video not found"}, status_code=404)
        transcript = clip_proposals.load_transcript(c)
        if not transcript:
            return JSONResponse(
                {"error": "No transcript for this video — clip suggestions are "
                          "drafted from what is said in it."},
                status_code=409)
        try:
            clip_proposals.propose(session, c, settings, transcript)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        clips = _pending_proposals(session, candidate_id)
        multi_clip = bool(c.multi_clip_potential)
    return JSONResponse({"clips": clips, "multi_clip": multi_clip})


@app.post("/clip-proposal/{proposal_id}/accept")
def accept_clip_proposal(proposal_id: int, cut_id: int = Form(...)):
    """Record that a proposal was loaded into a cut's segments.

    Accepting is an intent, not an outcome — the boundary metrics stay empty
    until that cut is exported and the operator's edits are visible.
    """
    with session_scope() as session:
        row = clip_proposals.accept(session, proposal_id, cut_id)
        if row is None:
            return JSONResponse({"error": "Proposal already decided"}, status_code=409)
    return JSONResponse({"ok": True})


@app.post("/clip-proposal/{proposal_id}/dismiss")
def dismiss_clip_proposal(proposal_id: int):
    with session_scope() as session:
        if not clip_proposals.dismiss(session, proposal_id):
            return JSONResponse({"error": "Proposal already decided"}, status_code=409)
    return JSONResponse({"ok": True})


@app.post("/video/{candidate_id}/dismiss-clip-proposals")
def dismiss_clip_proposals(candidate_id: int):
    """Reject every open proposal on a video at once — "none of these"."""
    with session_scope() as session:
        n = clip_proposals.dismiss_pending(session, candidate_id)
    return JSONResponse({"ok": True, "dismissed": n})


@app.post("/clip-proposal/{proposal_id}/new-cut")
def clip_proposal_new_cut(proposal_id: int):
    """Open a proposed clip as its own cut — the multi-story path.

    Lands in the trim editor with the segments loaded but nothing exported, so
    the proposal still has to survive a look before it becomes a clip.
    """
    with session_scope() as session:
        row = session.get(ClipProposal, proposal_id)
        if row is None or row.verdict != ClipProposal.VERDICT_PENDING:
            return JSONResponse({"error": "Proposal already decided"}, status_code=409)
        c = session.get(Candidate, row.candidate_pk)
        if c is None or c.status != STATUS_ARCHIVED:
            return JSONResponse({"error": "Download the video before clipping it"},
                                status_code=409)
        # Seed the new cut with the unclaimed part only — the stored proposal
        # predates the clips made since, and a new cut must not re-use footage.
        try:
            proposed = json.loads(row.proposed_segments or "[]")
        except (ValueError, TypeError):
            proposed = []
        segments, _ = clip_proposals.visible_segments(
            proposed, clip_proposals.used_ranges(session, c.id))
        if not segments:
            return JSONResponse(
                {"error": "Your other clips already cover this suggestion."},
                status_code=409)
        cut = Cut(candidate_pk=c.id, trim_segments=json.dumps(segments))
        session.add(cut)
        session.flush()
        clip_proposals.accept(session, proposal_id, cut.id)
        cut_id = cut.id
    return JSONResponse({"ok": True, "redirect": f"/cut/{cut_id}?step=trim"})


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
    needs_tags = False
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
        exporting = cut.export_status == "exporting"
        export_failed = cut.export_status == "failed"
        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.cut_pk == cut.id)
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
        posted = any(p.status == "published" for p in posts)
        cut_state = {"exported": exported, "posted": posted,
                     "captioned": bool(cut.subtitled_clip_path),
                     "exporting": exporting, "export_failed": export_failed}
        # A not-yet-published post pins the exact clip file it was queued with,
        # so re-exporting won't change it — warn before the operator assumes it will.
        pending = next((p for p in posts if p.status in ("queued", "draft", "failed")), None)
        pending_post_status = pending.status if pending else ""
        pending_post_id = pending.id if pending else None

        active_step = step if step in ("trim", "post") else (
            "post" if (exported or exporting or export_failed) else "trim")

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

        # Clips the model proposed and the operator hasn't ruled on, with any
        # material this video's cuts already claim removed. What survives is
        # always fresh footage — the dashed overlay and the violet one can
        # never describe the same seconds.
        suggested_clips = _pending_proposals(session, c.id)

        transcript_segments = []
        if c.transcript_path and Path(c.transcript_path).exists():
            try:
                transcript_segments = json.loads(Path(c.transcript_path).read_text())
            except Exception:
                pass
        clip_transcript, clip_transcript_text = _load_clip_transcript(cut)

        # Pending reel for this cut. Include is on by default for new clips;
        # if a pending Threads post already exists without a reel, keep it off
        # so requeueing doesn't silently re-add Instagram.
        pending_reel = session.execute(
            select(InstagramPost).where(
                InstagramPost.cut_pk == cut.id,
                InstagramPost.status.in_(["queued", "draft", "failed"]),
            ).order_by(InstagramPost.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        include_instagram = bool(pending_reel) if pending else True

        trait_vocab = active_traits_by_facet(session)
        # What the video's storyboard pass guessed, shown only while the clip
        # itself is untagged — it's the set the scheduler falls back to.
        inherited_fmt = [t.strip() for t in (c.format_tags or "").split(",") if t.strip()]
        inherited_subj = [t.strip() for t in (c.visual_traits or "").split(",") if t.strip()]
        cut_fmt = (cut.format_tags or "").strip()
        cut_subj = (cut.footage_traits or "").strip()
        # Already-exported clips that predate export-time tagging: fill in the
        # background so the next reload shows Format/Subject without waiting
        # for a re-export.
        needs_tags = bool(
            exported and not exporting
            and cut.footage_tagged_at is None
            and not cut_fmt and not cut_subj
            and cut.id not in _cut_annotate_inflight
        )
        if needs_tags:
            _cut_annotate_inflight.add(cut.id)
            _in_background(_annotate_cut_in_thread, cut.id)
        trait_source = cut_subj or c.visual_traits or ""

    threads_ok = threads_api.is_authenticated()
    instagram_ok = instagram_api.is_authenticated()
    return templates.TemplateResponse(
        request, "cut.html",
        {"cut": cut, "c": c, "state": cut_state, "step": active_step,
         "transcript_segments": transcript_segments, "saved_segments": segments,
         "other_cut_segments": other_cut_segments,
         "suggested_clips": suggested_clips,
         "clip_transcript": clip_transcript,
         "clip_transcript_text": clip_transcript_text,
         "posts": posts, "threads_ok": threads_ok,
         "instagram_ok": instagram_ok,
         "has_vertical": bool(cut.vertical_clip_path) and Path(cut.vertical_clip_path).exists(),
         "include_instagram": include_instagram,
         "account_name": threads_api.account_username(),
         "pending_post_status": pending_post_status,
         "pending_post_id": pending_post_id,
         "trait_vocab": trait_vocab,
         "inherited_fmt": inherited_fmt,
         "inherited_subj": inherited_subj,
         # Content-level shelf life (candidate tag -> category default), shown
         # and correctable in the rail. Post-level overrides happen later, on
         # the post page.
         "shelf_life": resolve_shelf_life(None, c),
         "shelf_life_tagged": bool((c.shelf_life or "").strip()),
         "tagging": needs_tags,
         "export_error": (cut.export_error or "") if export_failed else "",
         # Attribution first-comment, editable before the post even exists:
         # prefill from the pending post so requeueing round-trips cleanly.
         "attribution_text": (pending.attribution_text or "") if pending else "",
         "attribution_enabled": bool(load_first_reply().get("attribution_enabled")),
         "first_reply_mode": load_first_reply().get("mode", "citation"),
         "auth_url": "" if threads_ok else threads_api.authorize_url(),
         "subs_position": (
             cut.subs_position if cut.use_subtitles
             else suggest_subs_position(trait_source)
             or load_settings().get("subtitles.position", "bottom")
         ),
         "suggested_subs": suggest_subs_position(trait_source),
         "msg": msg, "active": "dashboard"},
    )


@app.post("/cut/{cut_id}/delete")
def delete_cut(request: Request, cut_id: int):
    """Delete a cut and any of its not-yet-published posts. Published posts are
    detached (kept for history) rather than removed."""
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return (JSONResponse({"error": "Clip not found"}, status_code=404)
                    if wants_json else _flash("/", "Clip not found"))
        candidate_id = cut.candidate_pk
        reels = session.execute(
            select(InstagramPost).where(InstagramPost.cut_pk == cut.id)
        ).scalars().all()
        deleted_reel_ids = set()
        for ig in reels:
            if ig.status == "published":
                ig.cut_pk = None  # keep published history
            else:
                deleted_reel_ids.add(ig.id)
                session.delete(ig)
        posts = session.execute(
            select(ThreadsPost).where(ThreadsPost.cut_pk == cut.id)
        ).scalars().all()
        for p in posts:
            if p.status in ("queued", "draft", "failed"):
                # A surviving (published) reel must not reference a deleted post.
                for ig in reels:
                    if ig.threads_post_pk == p.id and ig.id not in deleted_reel_ids:
                        ig.threads_post_pk = None
                session.delete(p)
            else:
                p.cut_pk = None  # keep published history, drop the link
        session.delete(cut)
    # Callers that show the clip in a list drop the row themselves rather than
    # re-rendering the whole list to lose one row.
    if wants_json:
        return JSONResponse({"ok": True})
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

    from ..config import storage_dir

    filename = file.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in _UPLOAD_EXTS:
        return _flash("/", f"Unsupported file type '{ext or '?'}'. "
                           f"Use one of: {', '.join(_UPLOAD_EXTS)}")

    video_id = "up" + uuid.uuid4().hex[:16]  # unique, fits String(20)
    settings = load_settings()
    upload_dir = storage_dir(settings, "storage.download_dir", "data/videos") / "uploads"
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
    _in_background(_scrape_in_thread, cid)
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
        _in_background(_scrape_in_thread, cid)

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
    _in_background(_scrape_in_thread, candidate_id)
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
    _in_background(_scrape_in_thread, candidate_id)
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


@app.get("/media/cut-thumb/{cut_id}")
def media_cut_thumb(cut_id: int):
    """Poster frame pulled from a cut's exported clip, so clip cards show the
    footage that actually shipped rather than the source video's poster.
    ``cached_still`` re-extracts when the clip file is newer, so re-exports
    refresh the card on their own."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path:
            return JSONResponse({"error": "no clip"}, status_code=404)
        source = cut.trimmed_clip_path
    try:
        path = cached_still(source, f"cut-{cut_id}")
    except Exception as exc:
        log.info("No still for cut %s: %s", cut_id, exc)
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


@app.get("/media/vertical/{cut_id}")
def media_vertical(cut_id: int):
    """The cut's 1080x1920 Instagram Reels composite (hook + clip + captions)."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.vertical_clip_path or not Path(cut.vertical_clip_path).exists():
            return JSONResponse({"error": "no vertical composite"}, status_code=404)
        return FileResponse(cut.vertical_clip_path, media_type="video/mp4")


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


def _safe_download_stem(text: str, limit: int = 60) -> str:
    """Filesystem-safe stem: letters/digits/space/-/_ only, spaces collapsed."""
    keep = "".join(c if c.isalnum() or c in " -_" else " " for c in (text or ""))
    return "-".join(keep.split())[:limit].strip("-") or ""


def _cut_download_filename(cut: Cut, *, kind: str = "") -> str:
    """Human-readable attachment name from station + clip title.

    e.g. ``KXYZ-Flooding-hits-Houston-vertical.mp4`` instead of a YouTube id.
    """
    title = (cut.clip_title or cut.calendar_name or "").strip()
    if not title and cut.candidate:
        title = (cut.candidate.title or "").strip()
    base = _safe_download_stem(title) or f"clip-{cut.id}"
    station = ""
    if cut.candidate and cut.candidate.channel:
        station = _safe_download_stem(cut.candidate.channel.call_sign or "", limit=20)
    stem = f"{station}-{base}" if station else base
    if kind:
        stem = f"{stem}-{kind}"
    return f"{stem}.mp4"


@app.get("/cut/{cut_id}/download-clip")
def download_clip(cut_id: int, captioned: int = 1):
    """Serve the exported clip as a file attachment so the operator can save it
    locally and post it manually elsewhere. Defaults to the captioned version
    (matching the preview default); pass ``captioned=0`` for the original."""
    with session_scope() as session:
        cut = session.execute(
            select(Cut)
            .options(selectinload(Cut.candidate).selectinload(Candidate.channel))
            .where(Cut.id == cut_id)
        ).scalar_one_or_none()
        if cut is None or not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        use_subs = bool(captioned) and bool(cut.subtitled_clip_path) \
            and Path(cut.subtitled_clip_path).exists()
        path = cut.subtitled_clip_path if use_subs else cut.trimmed_clip_path
        kind = "captioned" if use_subs else ""
        return FileResponse(path, media_type="video/mp4",
                            filename=_cut_download_filename(cut, kind=kind))


@app.get("/cut/{cut_id}/download-vertical")
def download_vertical(cut_id: int):
    """The vertical Reels composite as an attachment (for manual posting)."""
    with session_scope() as session:
        cut = session.execute(
            select(Cut)
            .options(selectinload(Cut.candidate).selectinload(Candidate.channel))
            .where(Cut.id == cut_id)
        ).scalar_one_or_none()
        if cut is None or not cut.vertical_clip_path or not Path(cut.vertical_clip_path).exists():
            return JSONResponse({"error": "no vertical composite"}, status_code=404)
        return FileResponse(cut.vertical_clip_path, media_type="video/mp4",
                            filename=_cut_download_filename(cut, kind="vertical"))


@app.get("/post/{post_id}/download-clip")
def download_post_clip(post_id: int):
    """Download the clip attached to a post (used from the Posts page)."""
    with session_scope() as session:
        p = session.execute(
            select(ThreadsPost)
            .options(
                selectinload(ThreadsPost.cut)
                .selectinload(Cut.candidate)
                .selectinload(Candidate.channel),
            )
            .where(ThreadsPost.id == post_id)
        ).scalar_one_or_none()
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = p.clip_local_path
        if (not path or not Path(path).exists()) and p.cut:
            path = p.cut.trimmed_clip_path
        if not path or not Path(path).exists():
            return JSONResponse({"error": "no clip"}, status_code=404)
        if p.cut:
            kind = "captioned" if path and str(path).endswith("_subs.mp4") else ""
            name = _cut_download_filename(p.cut, kind=kind)
        else:
            name = f"threads-post-{p.id}.mp4"
        return FileResponse(path, media_type="video/mp4", filename=name)


# --- Trim / export ----------------------------------------------------------------

def _delete_if_unreferenced(session, paths: list[str]) -> None:
    """Remove superseded clip files that no ThreadsPost or InstagramPost still
    points at.

    Exports are versioned per run, so a pending post keeps the exact file it was
    queued with. We only reclaim the disk space when nothing references the old
    file any more."""
    for path in {p for p in paths if p}:
        referenced = session.execute(
            select(ThreadsPost.id).where(ThreadsPost.clip_local_path == path).limit(1)
        ).scalar_one_or_none()
        if referenced is None:
            referenced = session.execute(
                select(InstagramPost.id)
                .where(InstagramPost.clip_local_path == path).limit(1)
            ).scalar_one_or_none()
        if referenced is not None:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not remove superseded clip %s: %s", path, exc)


@app.post("/cut/{cut_id}/export")
def export_clip(request: Request, cut_id: int, segments_json: str = Form(...),
                as_new: str = Form("0")):
    """Persist segments and kick off the supercut in the background.

    Returns immediately so the Trim step can navigate to Post with a skeleton;
    the Post page polls ``/cut/{id}/export-status`` until the file is ready.
    """
    wants_json = "application/json" in request.headers.get("accept", "")
    try:
        segments = json.loads(segments_json)
        assert isinstance(segments, list) and segments
    except Exception:
        msg = "No segments to export"
        return (JSONResponse({"error": msg}, status_code=400) if wants_json
                else _flash(f"/cut/{cut_id}?step=trim", msg))
    save_as_new = as_new in ("1", "true", "on", "yes")
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return (JSONResponse({"error": "Clip not found"}, status_code=404) if wants_json
                    else _flash("/", "Clip not found"))
        c = cut.candidate
        if c is None or not c.local_video_path:
            return (JSONResponse({"error": "Video not found or not downloaded"}, status_code=404)
                    if wants_json else _flash("/", "Video not found or not downloaded"))
        if cut.export_status == "exporting" and not save_as_new:
            dest = f"/cut/{cut.id}?step=post"
            return (JSONResponse({"ok": True, "cut_id": cut.id, "redirect": dest,
                                  "msg": "Export already in progress"})
                    if wants_json else _flash(dest, "Export already in progress"))
        # "Save as new clip": keep the open cut untouched and write the
        # marked segments into a fresh sibling cut on the same video.
        if save_as_new:
            target = Cut(candidate_pk=c.id)
            session.add(target)
            session.flush()
        else:
            target = cut
        target.trim_segments = json.dumps(segments)
        target.updated_at = utcnow()
        # Score the model's proposal against what actually shipped. Best-effort:
        # a ledger failure must never cost the operator an export.
        try:
            clip_proposals.resolve_for_cut(session, target.id, c.id, segments,
                                           from_cut_pk=cut.id)
        except Exception as exc:
            log.warning("Clip proposal resolve failed for cut %s: %s", target.id, exc)
        # Keep existing trimmed/captioned files until the worker swaps them —
        # a failed export must not orphan an already-good clip. The Post step
        # shows a skeleton while export_status is "exporting".
        target.export_status = "exporting"
        target.export_error = ""
        target_id = target.id
        n = len(segments)

    _in_background(_export_cut_in_thread, target_id)
    verb = "Saving new clip" if save_as_new else "Saving"
    dest = f"/cut/{target_id}?step=post"
    msg = f"{verb} — {n} segment{'s' if n != 1 else ''}…"
    if wants_json:
        return JSONResponse({"ok": True, "cut_id": target_id, "redirect": dest, "msg": msg})
    return _flash(dest, msg)


def _export_cut_in_thread(cut_id: int) -> None:
    """ffmpeg supercut + optional title draft for a cut marked ``exporting``."""
    try:
        with session_scope() as session:
            cut = session.get(Cut, cut_id)
            if cut is None:
                return
            c = cut.candidate
            if c is None or not c.local_video_path:
                cut.export_status = "failed"
                cut.export_error = "Video not found or not downloaded"
                cut.updated_at = utcnow()
                return
            try:
                segments = json.loads(cut.trim_segments or "[]")
                assert isinstance(segments, list) and segments
            except Exception:
                cut.export_status = "failed"
                cut.export_error = "No segments to export"
                cut.updated_at = utcnow()
                return
            source = c.local_video_path
            video_id = c.video_id
            word_transcript_path = c.word_transcript_path or ""
            previous = [cut.trimmed_clip_path, cut.subtitled_clip_path,
                        cut.vertical_clip_path, cut.clip_transcript_path]
            need_title = not (cut.clip_title or "").strip()
            draft_caption = cut.draft_caption or ""
            video_title = c.title or ""
            candidate_pk = c.id

        stamp = utcnow().strftime("%Y%m%dT%H%M%S")
        out = export_supercut(source, segments, f"{video_id}_cut{cut_id}_{stamp}")

        # Slice the archive-time Whisper word stream to the exported windows so
        # captions / Suggest caption never re-transcribe the trim. Best-effort:
        # videos archived before word sidecars fall back to Whisper-on-the-clip
        # inside ensure_clip_words, exactly as before.
        transcript_sidecar = ""
        if word_transcript_path and Path(word_transcript_path).exists():
            try:
                from ..subtitles import (
                    load_clip_words, save_clip_transcript, slice_source_words,
                )

                clip_words = slice_source_words(
                    load_clip_words(word_transcript_path), segments)
                if clip_words:
                    transcript_sidecar = str(save_clip_transcript(out, clip_words))
            except Exception:
                log.exception("Word-stream slice failed for cut %s", cut_id)

        title = ""
        calendar = ""
        if need_title:
            try:
                settings = load_settings()
                model = settings.get("engagement.draft_model", "claude-sonnet-5")
                with session_scope() as session:
                    c = session.get(Candidate, candidate_pk)
                    excerpt = _transcript_excerpt(c, segments) if c else ""
                title = suggest_title(model, video_title, excerpt, draft_caption or None) or ""
                if title:
                    calendar = suggest_calendar_name(
                        model, title, draft_caption or None) or ""
            except Exception:
                log.exception("Auto-title failed for cut %s", cut_id)

        with session_scope() as session:
            cut = session.get(Cut, cut_id)
            if cut is None:
                return
            cut.trimmed_clip_path = str(out)
            cut.updated_at = utcnow()
            cut.subtitled_clip_path = ""
            cut.vertical_clip_path = ""
            if not (cut.hook_text or "").strip():
                cut.hook_autodrafted = False
            cut.clip_transcript_path = transcript_sidecar
            cut.use_subtitles = False
            # New file = new on-screen content; drop stale tags so the vision
            # pass below can refill Format/Subject for the clip page.
            cut.format_tags = ""
            cut.footage_traits = ""
            cut.footage_tagged_at = None
            if title and not (cut.clip_title or "").strip():
                cut.clip_title = title
                if calendar:
                    cut.calendar_name = calendar
            # Tag before clearing export_status so the Post-step reload sees
            # Format/Subject already filled (footage content, not captions).
            try:
                settings = load_settings()
                vocab = active_traits_by_facet(session)
                annotate_cut_footage(cut, settings, vocab["subject"],
                                     format_traits=vocab["format"])
            except Exception:
                log.exception("Cut footage tagging failed for cut %s", cut_id)
            _sync_cut_tags_to_draft_posts(session, cut)
            cut.export_status = ""
            cut.export_error = ""
            if previous:
                _delete_if_unreferenced(session, previous)
    except ClipExportError as exc:
        log.warning("Background export failed for cut %s: %s", cut_id, exc)
        with session_scope() as session:
            cut = session.get(Cut, cut_id)
            if cut is None:
                return
            cut.export_status = "failed"
            cut.export_error = str(exc)[:500]
            cut.updated_at = utcnow()
    except Exception as exc:
        log.exception("Background export crashed for cut %s", cut_id)
        with session_scope() as session:
            cut = session.get(Cut, cut_id)
            if cut is None:
                return
            cut.export_status = "failed"
            cut.export_error = f"Export failed: {exc}"[:500]
            cut.updated_at = utcnow()


@app.get("/cut/{cut_id}/export-status")
def cut_export_status(cut_id: int):
    """Polled by the Post-step skeleton while a background save is running."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        c = cut.candidate
        exported = bool(cut.trimmed_clip_path) and Path(cut.trimmed_clip_path).exists()
        status = cut.export_status or ("ready" if exported else "idle")
        if status == "exporting":
            return {
                "status": "exporting",
                "exported": exported,
                "error": "",
                "updated_at": cut.updated_at.isoformat() if cut.updated_at else "",
            }
        if status == "failed":
            return {
                "status": "failed",
                "exported": exported,
                "error": cut.export_error or "Export failed",
            }
        # Ready (or idle-but-exported): include the Post-step boot flags the
        # sync redirect used to carry.
        seed = (c.draft_caption or "").strip() if c else ""
        current = (cut.draft_caption or "").strip()
        autocaption = bool(exported and (not current or current == seed))
        autohook = bool(exported and not (cut.hook_text or "").strip())
        askmulti = 0
        if c and c.multi_clip_potential and exported:
            exported_cuts = session.execute(
                select(func.count()).select_from(Cut).where(
                    Cut.candidate_pk == c.id, Cut.trimmed_clip_path != "")
            ).scalar() or 0
            if exported_cuts >= 2:
                askmulti = exported_cuts
        return {
            "status": "ready" if exported else "idle",
            "exported": exported,
            "error": "",
            "autosubs": exported,
            "autocaption": autocaption,
            "autohook": autohook,
            "askmulti": askmulti,
        }


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


@app.post("/cut/{cut_id}/vertical")
def generate_vertical(cut_id: int, hook_text: str = Form("")):
    """Generate the 1080x1920 Instagram Reels composite for this cut (AJAX).

    Composes the PLAIN trimmed export (the composite renders its own captions
    below the video, so the burned-in 16:9 variant would double them). Runs
    Whisper only if the word sidecar doesn't exist yet — same source as the
    burned-in captions and Suggest caption.

    The first compose of a cut nobody has written a hook for drafts one, so the
    reel arrives with on-screen text instead of a bare video. The transcript
    that draft needs is the same one the composite's captions come from, so it
    costs one model call rather than a second Whisper pass.
    """
    from ..subtitles import clip_transcript_path_for
    from ..vertical import VerticalCompositeError, create_vertical_composite

    autodrafted = ""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
            return JSONResponse({"error": "Export a clip first"}, status_code=404)
        hook = hook_text.strip()
        if not hook and not cut.hook_autodrafted:
            try:
                autodrafted, _clip_text = _draft_hook_for_cut(cut, load_settings())
                hook = autodrafted
                cut.hook_autodrafted = True
            except ValueError as exc:
                # Silent clip: there is nothing to draft from, now or later.
                log.info("Hook auto-draft not possible for cut %s: %s", cut_id, exc)
                cut.hook_autodrafted = True
            except Exception as exc:
                # Model/network trouble — leave the flag so the next compose retries.
                log.warning("Hook auto-draft failed for cut %s: %s", cut_id, exc)
        # Persist the hook right away so it survives a failed render.
        cut.hook_text = hook
        cut.updated_at = utcnow()
        clip_path = cut.trimmed_clip_path
        transcript_path = cut.clip_transcript_path or None
        previous_vertical = cut.vertical_clip_path
    try:
        out = create_vertical_composite(clip_path, hook,
                                        transcript_path=transcript_path)
    except VerticalCompositeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except Exception as exc:
        log.exception("Vertical composite failed for cut %s", cut_id)
        return JSONResponse({"error": f"Vertical render failed: {exc}"}, status_code=500)
    sidecar = clip_transcript_path_for(clip_path)
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is not None:
            cut.vertical_clip_path = str(out)
            if sidecar.exists():
                cut.clip_transcript_path = str(sidecar)
            _delete_if_unreferenced(session, [previous_vertical])
            cut.updated_at = utcnow()
    warning = ""
    try:
        duration = clip_duration(out)
        # Outside 5-90s Meta still publishes, but as a plain video post that
        # never reaches the Reels tab — worth flagging before it's queued.
        if duration > 90:
            warning = (f"Clip is {duration:.0f}s — reels over 90s publish as a "
                       f"regular video post, not in the Reels tab.")
        elif duration < 5:
            warning = f"Clip is {duration:.1f}s — under Meta's 5s Reels minimum."
    except Exception:
        pass
    return {"url": f"/media/vertical/{cut_id}", "warning": warning,
            "hook": autodrafted}


@app.post("/cut/{cut_id}/ig-copy")
def save_ig_copy(cut_id: int, hook_text: str = Form(None)):
    """Autosave the Instagram hook text (reel caption reuses the Threads caption)."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if hook_text is not None:
            cut.hook_text = hook_text.strip()
        cut.updated_at = utcnow()
    return {"ok": True}


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
            voice = {"examples": [], "style_guide": "", "target_words": None}
        model = settings.get("engagement.draft_model", "claude-sonnet-5")
        max_chars = int(settings.get("engagement.caption_max_chars", 220))
        operator_guide = render_caption_guide()
        target_words = voice.get("target_words")
        try:
            caption = suggest_post_caption(
                model,
                c.title, c.channel.call_sign, c.channel.market, excerpt, seconds,
                examples=voice["examples"], style_guide=voice["style_guide"],
                operator_guide=operator_guide,
                max_chars=max_chars,
                target_words=target_words,
                description=c.description,
            )
            # The caption field itself is left untouched — this is a proposal
            # until the operator accepts it (/cut/{id}/caption). The DRAFT is
            # persisted here regardless, so dismissing it records a rejection
            # instead of discarding the most informative event in the loop.
            proposal_id = log_draft_proposal(
                session, cut_id, caption, kind=KIND_CAPTION,
                model=model,
                policy=draft_policy_version(
                    model=model, max_chars=max_chars, target_words=target_words,
                    style_guide=voice["style_guide"], operator_guide=operator_guide,
                    examples=len(voice["examples"]),
                ),
                max_chars=max_chars,
                target_words=target_words,
                voice_examples=len(voice["examples"]),
            )
            return {"caption": caption, "voice_examples": len(voice["examples"]),
                    "proposal_id": proposal_id, "target_words": target_words,
                    "transcript": clip_text}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/cut/{cut_id}/caption-verdict")
def caption_verdict(cut_id: int, proposal_id: int = Form(...),
                    verdict: str = Form(...)):
    """Record what the operator did with a drafted caption (used / dismissed).

    Best-effort telemetry: a failure here must never block captioning, so the
    response is always OK-shaped and the caller doesn't wait on it.
    """
    try:
        with session_scope() as session:
            ok = record_draft_verdict(session, proposal_id, verdict)
        return JSONResponse({"ok": bool(ok)})
    except Exception:
        log.exception("Caption verdict logging failed (cut %s)", cut_id)
        return JSONResponse({"ok": False})


def _draft_hook_for_cut(cut: Cut, settings) -> tuple[str, str]:
    """Draft on-video hook text from the clip's own transcript.

    Returns ``(hook, clip_transcript)``. Raises ``ValueError`` when the clip
    can't support a hook at all (not exported yet, or nothing is said in it) and
    ``RuntimeError`` when the model round trip fails — callers treat the first
    as settled and the second as worth retrying. Must run inside a session.

    Logs the draft to the ledger before returning. The hook has no accept /
    dismiss card — it lands directly in ``Cut.hook_text`` and the operator types
    over it — so without this row the rewrite would leave no trace at all.
    """
    if not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
        raise ValueError("Export the clip first — the hook is drafted from "
                         "what is said in the trimmed clip.")
    _lines, clip_text = _ensure_whisper_clip_transcript(cut)
    if not clip_text.strip():
        raise ValueError("No speech detected in the clip — nothing to draft "
                         "a hook from.")
    c = cut.candidate
    session = object_session(cut)
    model = settings.get("engagement.draft_model", "claude-sonnet-5")
    # Voice for hooks comes only from hooks the operator rewrote: they're burned
    # into the video rather than published as text, so there's no post history.
    examples: list[str] = []
    if session is not None and settings.get("voice.enabled", True):
        try:
            examples = operator_written_drafts(
                session, KIND_HOOK, limit=int(settings.get("voice.hook_examples", 6)))
        except Exception:
            log.exception("Hook voice examples failed (drafting generic)")
    hook = suggest_hook_text(
        model, c.title, c.channel.call_sign, c.channel.market,
        " ".join(clip_text.split())[:3000], examples=examples,
        description=c.description,
    )
    if not hook:
        raise RuntimeError("The model returned an empty hook — try again.")
    if session is not None:
        try:
            log_draft_proposal(
                session, cut.id, hook, kind=KIND_HOOK, model=model,
                policy=draft_policy_version(
                    model=model, max_chars=80, target_words=None,
                    style_guide="", operator_guide="", examples=len(examples)),
                max_chars=80, target_words=None, voice_examples=len(examples))
        except Exception:
            log.exception("Hook proposal logging failed for cut %s", cut.id)
    return hook, clip_text


@app.post("/cut/{cut_id}/suggest-hook")
def suggest_hook(cut_id: int):
    """Draft short on-video hook text for the Instagram vertical composite."""
    settings = load_settings()
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            hook, clip_text = _draft_hook_for_cut(cut, settings)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        # Persist so a regenerate / leave-and-return keeps the draft.
        cut.hook_text = hook
        cut.hook_autodrafted = True
        cut.updated_at = utcnow()
        return {"hook": hook, "transcript": clip_text}


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


def _wants_instagram(value: str) -> bool:
    return str(value).lower() in ("1", "true", "on", "yes")


def _publish_targets(target: str, include_instagram: str) -> tuple[bool, bool]:
    """Which platforms a Post now click ships to, as ``(threads, instagram)``.

    The Post now dialog sends an explicit ``target``; anything else (an older
    cached page, a scripted POST) falls back to the Instagram toggle, which is
    how this endpoint behaved before the dialog existed."""
    choice = str(target).strip().lower()
    if choice == "threads":
        return True, False
    if choice == "instagram":
        return False, True
    if choice == "both":
        return True, True
    return True, _wants_instagram(include_instagram)


def _publish_reel_only(cut_id: int, caption: str):
    """Ship just the Instagram reel for a cut — no Threads post is created, so
    the Threads spacing floor doesn't apply and the scheduler state is left
    untouched."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path:
            return _flash(f"/cut/{cut_id}", "Export a clip first")
        ig_error = _instagram_ready_error(cut, True)
        if ig_error:
            return _flash(f"/cut/{cut_id}?step=post", ig_error)
        try:
            # No Threads post to pair with: the reel is published directly
            # below, once this transaction has committed.
            ig = record_instagram_post(session, cut, None, cut.vertical_clip_path,
                                       caption)
            cut.draft_caption = caption
            ig_id = ig.id
        except Exception as exc:
            return _flash(f"/cut/{cut_id}?step=post", f"Reel failed: {exc}")
    ig = publish_reel_now(ig_id)
    if ig is not None and ig.status == "published":
        return _flash(f"/cut/{cut_id}?step=post",
                      f"Reel live: {ig.permalink or ig.ig_media_id}")
    detail = (ig.error or "")[:120] if ig is not None else "reel record vanished"
    return _flash(f"/cut/{cut_id}?step=post", f"Reel failed: {detail}")


def _instagram_ready_error(cut: Cut, want_ig: bool) -> str:
    """Why the reel can't ride along, or '' when it can."""
    if not want_ig:
        return ""
    if not instagram_api.is_authenticated():
        return "Connect Instagram first (Configure → Accounts)"
    if not cut.vertical_clip_path or not Path(cut.vertical_clip_path).exists():
        return "Generate the vertical composite first"
    return ""


def _drop_pending_reels(session, cut: Cut) -> None:
    """Remove not-yet-published reels for a cut (operator un-ticked Instagram)."""
    for ig in session.execute(
        select(InstagramPost).where(
            InstagramPost.cut_pk == cut.id,
            InstagramPost.status.in_(["queued", "draft", "failed"]),
        )
    ).scalars().all():
        session.delete(ig)


@app.post("/cut/{cut_id}/post")
def post_to_threads(cut_id: int, caption: str = Form(...),
                    use_subtitles: str = Form(""), attribution: str = Form(""),
                    include_instagram: str = Form(""),
                    publish_target: str = Form("")):
    """Operator-confirmed publish of the exported clip. ``publish_target`` picks
    the platforms — the Threads post, the Instagram reel, or both."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/cut/{cut_id}?step=post", "Caption is empty")
    want_threads, want_ig = _publish_targets(publish_target, include_instagram)
    if not want_threads:
        return _publish_reel_only(cut_id, caption)
    attribution = _with_first_reply_draft(cut_id, caption, attribution)
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
        ig_error = _instagram_ready_error(cut, want_ig)
        if ig_error:
            return _flash(f"/cut/{cut_id}?step=post", ig_error)
        try:
            # Same flow as publish_clip, with the reel recorded before the
            # publish so publish_paired_reel finds it afterwards.
            post = record_post(session, cut.candidate,
                               _chosen_clip_path(cut, use_subtitles), caption,
                               status="draft", cut=cut, attribution=attribution)
            if want_ig:
                # Reel caption is the Threads caption — one copy for both.
                record_instagram_post(session, cut, post, cut.vertical_clip_path,
                                      caption)
            post = publish_post(session, post)
            # Keep the clip's caption in sync with what was actually posted.
            cut.draft_caption = caption
            state = session.get(SchedulerState, 1)
            if state is None:
                state = SchedulerState(id=1)
                session.add(state)
            state.last_publish_at = utcnow()
            state.last_action = f"manual_publish:post={post.id}"
            state.updated_at = utcnow()
            post_id = post.id
            msg = f"Published: {post.permalink or post.threads_media_id}"
            if post.first_reply_id:
                msg += " · first reply posted"
            elif post.first_reply_error:
                msg += f" · no first reply: {post.first_reply_error[:120]}"
        except Exception as exc:
            return _flash(f"/cut/{cut_id}?step=post", f"Publish failed: {exc}")
    if want_ig:
        # Outside the session above: the reel publishes through its own
        # transactions once the Threads publish is committed.
        ig = publish_paired_reel(post_id)
        if ig is not None and ig.status == "published":
            msg += f" · reel live: {ig.permalink or ig.ig_media_id}"
        elif ig is not None:
            msg += f" · reel {ig.status}: {(ig.error or '')[:120]}"
    return _flash(f"/cut/{cut_id}?step=post", msg)


@app.post("/cut/{cut_id}/queue")
def queue_to_threads(cut_id: int, caption: str = Form(...),
                     use_subtitles: str = Form(""), attribution: str = Form(""),
                     include_instagram: str = Form("")):
    """Add the exported clip to the adaptive FIFO queue (no immediate post).
    With the Instagram toggle on, the same action queues the paired reel —
    queueing stays the operator-approval step for both platforms."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/cut/{cut_id}?step=post", "Caption is empty")
    want_ig = _wants_instagram(include_instagram)
    attribution = _with_first_reply_draft(cut_id, caption, attribution)
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None or not cut.trimmed_clip_path:
            return _flash(f"/cut/{cut_id}", "Export a clip first")
        ig_error = _instagram_ready_error(cut, want_ig)
        if ig_error:
            return _flash(f"/cut/{cut_id}?step=post", ig_error)
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
                # whatever came back (edited or cleared) is the operator's call.
                # Publishing never drafts a replacement — empty means no
                # attribution comment.
                keep.attribution_text = attribution.strip()
                keep.attribution_skipped = not attribution.strip()
                if keep.clip_local_path != clip_path:
                    # Captions were toggled since this post was created — point
                    # at the chosen file and refresh the cloud copy.
                    from ..publishing import _object_key
                    from ..storage_supabase import upload_trimmed_clip
                    keep.clip_local_path = clip_path
                    keep.clip_object_path = _object_key(Path(clip_path))
                    # A different file may read differently on screen (burned-in
                    # subtitles): clear the facet annotation so the scheduler's
                    # queue-time pass re-tags the file that will actually ship.
                    keep.footage_scored_at = None
                    keep.footage_traits = ""
                    keep.format_tags = ""
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
                post = keep
            else:
                post = queue_clip(session, cut.candidate, clip_path, caption, cut=cut,
                                  attribution=attribution)
            if want_ig:
                # Reel caption is the Threads caption — one copy for both.
                record_instagram_post(session, cut, post, cut.vertical_clip_path,
                                      caption)
            else:
                # Toggle round-trips: un-ticking removes the pending reel.
                _drop_pending_reels(session, cut)
            # Persist the queued caption back onto the clip so the clip reflects
            # what was scheduled. Safe to overwrite now: the AI draft lives in
            # the caption ledger, not in this field.
            cut.draft_caption = caption
            updated = bool(existing)
        except Exception as exc:
            return _flash(f"/cut/{cut_id}?step=post", f"Queue failed: {exc}")
    suffix = " (with Instagram reel)" if want_ig else ""
    return _flash(f"/cut/{cut_id}?step=post",
                  ("Queued post updated" if updated else "Added to the posting queue") + suffix)


@app.post("/cut/{cut_id}/save-draft")
def save_draft(cut_id: int, caption: str = Form(...),
               use_subtitles: str = Form(""), attribution: str = Form("")):
    """Save the exported clip + caption as a draft to publish or queue later."""
    caption = caption.strip()
    if not caption:
        return _flash(f"/cut/{cut_id}?step=post", "Caption is empty")
    attribution = _with_first_reply_draft(cut_id, caption, attribution)
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


def _first_reply_is_invitation() -> bool:
    return load_first_reply().get("mode") == "invitation"


def _with_first_reply_draft(cut_id: int, caption: str, attribution: str) -> str:
    """The operator's first-reply text, or a call-to-action draft when they left
    the box empty and invitation mode is on.

    Called BEFORE the ship path opens its session: the draft is a multi-second
    model call, and running it inside that transaction would hold a pooled
    database connection the whole time. Never raises — a draft that fails just
    leaves the box empty, exactly as it would have been.
    """
    if attribution.strip() or not _first_reply_is_invitation():
        return attribution
    try:
        return draft_first_reply_for_cut(cut_id, caption) or attribution
    except Exception:
        log.exception("First-reply draft failed for cut %s", cut_id)
        return attribution


def _no_suggestion_message(invitation: bool) -> str:
    """Why the model returned nothing, in the operator's terms."""
    if invitation:
        return ("No invitation drafted — check that the brief under Replies "
                "settings says what the call to action should be.")
    return ("No citation drafted — the source data available isn't enough for a "
            "reliable attribution.")


@app.post("/cut/{cut_id}/suggest-attribution")
def suggest_cut_attribution(cut_id: int):
    """(Re)draft the first-comment while still on the cut page — a source
    citation or a call to action, per the mode on the Replies page. Returns the
    suggestion only; it rides along with queue/post/draft."""
    invitation = load_first_reply().get("mode") == "invitation"
    with session_scope() as session:
        cut = session.execute(
            select(Cut)
            .options(selectinload(Cut.candidate).selectinload(Candidate.channel))
            .where(Cut.id == cut_id)
        ).scalar_one_or_none()
        if cut is None or cut.candidate is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            # Gather inside the session, call the model outside it: see
            # ``first_reply_context``.
            if invitation:
                pending = first_reply_context(session, cut.candidate, cut,
                                              cut.draft_caption or "")
            else:
                text = generate_attribution(cut.candidate)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
    if invitation:
        try:
            text = draft_first_reply(pending)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
    if not text:
        # The model declined rather than guess — surface that honestly instead
        # of proposing a made-up credit.
        return {"text": "", "unavailable": True,
                "message": _no_suggestion_message(invitation)}
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
        # The paired reel rides on this post's publish, so it goes with it
        # (published reels are kept for history, just detached).
        for ig in session.execute(
            select(InstagramPost).where(InstagramPost.threads_post_pk == p.id)
        ).scalars().all():
            if ig.status == "published":
                ig.threads_post_pk = None
            else:
                session.delete(ig)
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
            # Paired reel uses the same caption — keep it in sync while still
            # editable (queued/draft/failed).
            ig = session.execute(
                select(InstagramPost).where(
                    InstagramPost.threads_post_pk == p.id,
                    InstagramPost.status.in_(["queued", "draft", "failed"]),
                ).order_by(InstagramPost.created_at.desc()).limit(1)
            ).scalar_one_or_none()
            if ig is not None:
                ig.caption = caption.strip()
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
    published = False
    try:
        with session_scope() as session:
            p = session.get(ThreadsPost, post_id)
            if p is None:
                return
            try:
                publish_post(session, p)
                published = True
                state = session.get(SchedulerState, 1)
                if state is None:
                    state = SchedulerState(id=1)
                    session.add(state)
                state.last_publish_at = utcnow()
                state.last_action = f"manual_publish:post={post_id}"
                state.updated_at = utcnow()
            except Exception:
                log.exception("Background publish failed for post %s", post_id)
        if published:
            # After the publish transaction commits; see publish_paired_reel.
            publish_paired_reel(post_id)
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
    _in_background(_publish_in_thread, post_id)
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
        first_reply_cid = (p.first_reply_id or "").strip()
        comment_rows = [
            {"id": c.id, "username": c.username, "text": c.text,
             "reply_status": c.reply_status,
             "reply_text": c.reply_text_posted,
             "commented_at": c.commented_at}
            for c in comments
            if c.comment_id != first_reply_cid or not first_reply_cid
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
            # Placement facets: what the scored scheduler sees for this post.
            "format_tags": [t.strip() for t in (p.format_tags or "").split(",") if t.strip()],
            "footage_traits": [t.strip() for t in (p.footage_traits or "").split(",") if t.strip()],
            "trait_vocab": active_traits_by_facet(session),
            # Fully resolved (post -> candidate -> category default ->
            # evergreen) so the badge always shows what placement will do,
            # even for posts ingested before shelf-life tagging existed.
            "shelf_life": resolve_shelf_life(p, cand),
            "shelf_life_tagged": bool(((p.shelf_life or "").strip())
                                      or ((cand.shelf_life or "").strip() if cand else "")),
            # Full override chain + derived urgency/expiry for the panel.
            "shelf": shelf_life_outlook(p, cand),
            # Rerun outlook with its performance receipt: why this clip's
            # quiet period is what it is (rank of views at the comparison
            # age), or why the rotation excludes it. None unless published.
            "recycle": recycle_status(session, p),
            "repost_of": p.repost_of_post_pk,
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
            # Smart default from footage traits: where captions should go
            # (or whether they should be skipped entirely) before the
            # operator has made an explicit choice.
            "suggested_subs": suggest_subs_position(
                p.footage_traits or (cand.visual_traits if cand else "")),
            "clip_transcript_text": clip_transcript_text,
            "scheduled_at": p.scheduled_at, "published_at": p.published_at,
            "created_at": p.created_at,
            "attribution_text": p.attribution_text or "",
            "attribution_skipped": bool(p.attribution_skipped),
            "attribution_enabled": load_first_reply().get("attribution_enabled", True),
            "first_reply_mode": load_first_reply().get("mode", "citation"),
            # A call to action is written from the brief, not from the source
            # video, so it can be drafted even for a post with no candidate.
            "can_suggest_attribution": bool(cand) or _first_reply_is_invitation(),
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
        # Paired Instagram reel (queued alongside this post, publishes with it).
        ig = session.execute(
            select(InstagramPost).where(InstagramPost.threads_post_pk == p.id)
            .order_by(InstagramPost.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        ig_video_url = ""
        if ig is not None and ig.cut_pk and cut is not None and cut.vertical_clip_path \
                and Path(cut.vertical_clip_path).exists():
            ig_video_url = f"/media/vertical/{cut.id}"
        ctx["ig"] = ({
            "id": ig.id, "status": ig.status, "caption": ig.caption,
            "permalink": ig.permalink, "error": ig.error,
            "published_at": ig.published_at,
            "video_url": ig_video_url,
        } if ig else None)
        instagram_ok = instagram_api.is_authenticated()
        ctx["instagram_ok"] = instagram_ok
        ctx["include_instagram"] = bool(
            ig and ig.status in ("queued", "draft", "failed", "published"))
        ctx["ig_switch_editable"] = (
            p.status in ("draft", "queued", "failed")
            and (ig is None or ig.status in ("queued", "draft", "failed"))
        )
        ctx["show_ig_switch"] = (
            (p.status in ("draft", "queued", "failed") and cut is not None and has_clip)
            or ig is not None
        )
    return templates.TemplateResponse(
        request, "post.html", {**ctx, "msg": msg, "active": "posts"}
    )


@app.post("/post/{post_id}/instagram")
def set_post_instagram(post_id: int, include_instagram: str = Form(""),
                       next: str = Form("")):
    """Pair or unpair an Instagram reel with a not-yet-published post."""
    dest = next if next and next.startswith("/") else f"/post/{post_id}"
    want = _wants_instagram(include_instagram)
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return _flash(dest, "Post not found")
        ig = session.execute(
            select(InstagramPost).where(InstagramPost.threads_post_pk == p.id)
            .order_by(InstagramPost.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        if want:
            if p.status not in ("draft", "queued", "failed"):
                return _flash(dest, "Can't add a reel to a published post")
            cut = session.get(Cut, p.cut_pk) if p.cut_pk else None
            if cut is None:
                return _flash(dest, "No clip attached")
            ig_error = _instagram_ready_error(cut, True)
            if ig_error:
                return _flash(dest, ig_error)
            record_instagram_post(session, cut, p, cut.vertical_clip_path,
                                  p.caption or "")
            return _flash(dest, "Instagram reel will publish with this post")
        if ig is None:
            return _flash(dest, "No reel to remove")
        if ig.status == "published":
            return _flash(dest, "This reel is already live")
        session.delete(ig)
    return _flash(dest, "Instagram reel removed")


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


@app.post("/instagram/connect")
def instagram_connect(code: str = Form(...), next: str = Form("/connections")):
    try:
        instagram_api.exchange_code(_clean_auth_code(code))
        return _flash(next, "Instagram connected")
    except Exception as exc:
        return _flash(next, f"Instagram auth failed: {exc}")


@app.get("/connections", response_class=HTMLResponse)
def connections_page(request: Request, msg: str = ""):
    """Threads + Instagram OAuth connection status (Configure area)."""
    authenticated = threads_api.is_authenticated()
    ig_authenticated = instagram_api.is_authenticated()
    try:
        ig_auth_url = instagram_api.authorize_url() if not ig_authenticated else ""
    except Exception:  # missing .env keys shouldn't 500 the page
        ig_auth_url = ""
    return templates.TemplateResponse(
        request, "connections.html",
        {"authenticated": authenticated,
         "auth_url": threads_api.authorize_url() if not authenticated else "",
         "ig_authenticated": ig_authenticated,
         "ig_auth_url": ig_auth_url,
         "ig_username": instagram_api.account_username() if ig_authenticated else "",
         "ig_configured": bool(env("INSTAGRAM_APP_ID")),
         "msg": msg, "active": "connections"},
    )


@app.get("/threads-account")
def threads_account_redirect():
    """Back-compat: the Threads account page is now Connections."""
    return RedirectResponse("/connections", status_code=303)


# --- Archive -----------------------------------------------------------------

@app.get("/archive")
def archive_redirect(section: str = ""):
    """Back-compat: the Archive page is now the Library's Videos section."""
    return RedirectResponse("/library" + (f"?section={section}" if section else ""),
                            status_code=307)


def _library_dataset() -> dict:
    """Everything the Library page draws. The page takes no server-side filters
    (it ships the whole library and narrows it in the browser), so this is one
    cacheable dataset rather than one per view."""
    with session_scope() as session:
        # --- Videos (downloaded/archived source clips) ---
        videos = session.execute(
            select(Candidate)
            .options(selectinload(Candidate.channel), *_CANDIDATE_LIST_ONLY)
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
            .options(selectinload(Cut.candidate).options(
                selectinload(Candidate.channel), *_CANDIDATE_LIST_ONLY))
            .order_by(Cut.created_at.desc())
        ).scalars().all()
        published_cut_pks = {
            pk for (pk,) in session.execute(
                select(ThreadsPost.cut_pk).where(
                    ThreadsPost.status == "published", ThreadsPost.cut_pk.is_not(None)
                ).distinct()
            ).all()
        }
        # Rerun outlook per cut (one rotation pass for the whole page), so a
        # posted clip's card can say when it next qualifies to re-air.
        reruns = recycle_overview(session)
        cut_rows = [
            {"cut": cut,
             "exported": bool(cut.trimmed_clip_path),
             "posted": cut.id in published_cut_pks,
             "captioned": bool(cut.subtitled_clip_path),
             "recycle": reruns.get(cut.id)}
            for cut in cuts
        ]

        # --- Posts (recent, any status) ---
        posts = session.execute(
            select(ThreadsPost)
            .options(
                selectinload(ThreadsPost.candidate).options(
                    selectinload(Candidate.channel), *_CANDIDATE_LIST_ONLY),
                selectinload(ThreadsPost.cut).selectinload(Cut.candidate).options(
                    selectinload(Candidate.channel), *_CANDIDATE_LIST_ONLY),
                *_POST_LIST_ONLY,
            )
            .order_by(ThreadsPost.created_at.desc()).limit(100)
        ).scalars().all()

        # Newest counts per post, so the Posts tab can show and sort on reach.
        # `engagement` is the interaction total (views excluded — it's the
        # denominator, not an interaction); None until at least one is known.
        snaps = latest_metrics_bulk(session, [p.id for p in posts])
        post_metrics: dict[int, dict] = {}
        for p in posts:
            snap = snaps.get(p.id) or {}
            row = {m: snap.get(m) for m in ("views", "likes", "replies", "reposts")}
            interactions = [row[m] for m in ("likes", "replies", "reposts")
                            if row[m] is not None]
            row["engagement"] = sum(interactions) if interactions else None
            post_metrics[p.id] = row

        # --- Advanced-filter facets per post: resolved shelf life, rerun
        # state, performance tier, and the placement tags. Plain values only —
        # footage_traits is raiseload-deferred on the list query, so tags come
        # from one extra column fetch instead of the ORM objects. ---
        traits_by_post: dict[int, str] = dict(session.execute(
            select(ThreadsPost.id, ThreadsPost.footage_traits)
            .where(ThreadsPost.id.in_([p.id for p in posts]))
        ).all()) if posts else {}

        # Performance tier from the same views-at-fixed-age ranking the
        # scheduler's rerun cadence uses (not lifetime views, which would let
        # old posts always win). Tokens ladder so "above median" also matches
        # every top-10% post: data-tier="top10,top25,above".
        published = [p for p in posts if p.status == "published" and p.published_at]
        age_hours = int(load_settings().get("learning.metric_age_hours", 48))
        views_at_age = metrics_at_age_bulk(session, published, "views", age_hours)
        ranked_views = sorted(views_at_age.values())

        def _tier_tokens(post_id: int) -> list[str]:
            value = views_at_age.get(post_id)
            if value is None or len(ranked_views) < 2:
                return ["nodata"]
            rank = bisect.bisect_left(ranked_views, value) / (len(ranked_views) - 1)
            if rank < 0.5:
                return ["below"]
            tokens = ["above"]
            if rank >= 0.75:
                tokens.insert(0, "top25")
            if rank >= 0.9:
                tokens.insert(0, "top10")
            return tokens

        post_facets: dict[int, dict] = {}
        for p in posts:
            cand = p.cut.candidate if (p.cut and p.cut.candidate) else p.candidate
            shelf = resolve_shelf_life(p, cand)
            rerun: list[str] = []
            if p.repost_of_post_pk is not None:
                rerun.append("reair")
            if p.status == "published":
                info = reruns.get(p.cut_pk) if p.cut_pk is not None else None
                if info is not None:
                    rerun.append("ready" if info["overdue"] else "waiting")
                else:
                    rerun.append("ineligible")
            formats = [t.strip() for t in (p.format_tags or "").split(",") if t.strip()]
            subjects = [t.strip() for t in (traits_by_post.get(p.id) or "").split(",")
                        if t.strip()]
            # Extra haystack for the free-text box, so typing "evergreen" or a
            # tag narrows without touching the dropdowns.
            words = [shelf] + formats + subjects
            if "ready" in rerun:
                words.append("rerun-ready")
            if "reair" in rerun:
                words.append("re-air")
            post_facets[p.id] = {
                "shelf": shelf,
                "rerun": rerun,
                "tier": _tier_tokens(p.id) if p.status == "published" else [],
                "formats": formats,
                "subjects": subjects,
                "search_extra": " ".join(words),
            }

        # Advanced-filter vocabularies, restricted to what's on the page so no
        # option can match nothing (same rule as the category dropdown).
        _shelves = {f["shelf"] for f in post_facets.values() if f["shelf"]}
        shelf_choices = [(s, s.capitalize()) for s in ("breaking", "timely", "evergreen")
                         if s in _shelves]
        _rerun_used = {t for f in post_facets.values() for t in f["rerun"]}
        rerun_choices = [(k, lbl) for k, lbl in
                         (("ready", "Rerun-ready"),
                          ("waiting", "In rotation — waiting"),
                          ("ineligible", "Not in rotation"),
                          ("reair", "Is a re-air"))
                         if k in _rerun_used]
        _tier_used = {t for f in post_facets.values() for t in f["tier"]}
        tier_choices = [(k, lbl) for k, lbl in
                        (("top10", "Top 10%"), ("top25", "Top 25%"),
                         ("above", "Above median"), ("below", "Below median"),
                         ("nodata", "No view data yet"))
                        if k in _tier_used]
        # Raw tag as the value (rows carry it verbatim), prettified as the label.
        format_choices = [(t, t.replace("_", " ")) for t in
                          sorted({t for f in post_facets.values() for t in f["formats"]})]
        subject_choices = [(t, t.replace("_", " ")) for t in
                           sorted({t for f in post_facets.values() for t in f["subjects"]})]

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
        # Category filters all three views, so its vocabulary has to come from
        # all three: a video archived long ago may be the only thing carrying a
        # category that several posts on the page inherit.
        def _category(candidate) -> str:
            return (candidate.category or "") if candidate is not None else ""

        used_categories = {_category(r["c"]) for r in video_rows}
        used_categories |= {_category(r["cut"].candidate) for r in cut_rows}
        used_categories |= {
            _category(p.cut.candidate if (p.cut and p.cut.candidate) else p.candidate)
            for p in posts
        }
        used_categories.discard("")
        category_choices = [
            (opt["slug"], f"{opt['emoji']} {opt['label']}".strip())
            for opt in category_options() if opt["slug"] in used_categories
        ]
        post_statuses = sorted({p.status for p in posts if p.status})

    return {
        "video_rows": video_rows, "cut_rows": cut_rows, "posts": posts,
        "post_metrics": post_metrics, "post_facets": post_facets,
        "counts": {"videos": len(video_rows), "cuts": len(cut_rows), "posts": len(posts)},
        "channel_choices": sorted(cs for cs in call_signs if cs),
        "category_choices": category_choices,
        "post_status_choices": post_statuses,
        "shelf_choices": shelf_choices, "rerun_choices": rerun_choices,
        "tier_choices": tier_choices, "format_choices": format_choices,
        "subject_choices": subject_choices,
    }


pagecache.register("library", _library_dataset)


@app.get("/library", response_class=HTMLResponse)
def library_page(request: Request, section: str = "videos", msg: str = ""):
    """The content library: Videos → Cuts → Posts, the three types that cascade
    into each other, shown as a toggle group. ``section`` selects the open tab."""
    if section not in ("videos", "cuts", "posts"):
        section = "videos"
    return templates.TemplateResponse(
        request, "library.html",
        {**pagecache.read("library"),
         "section": section, "msg": msg, "active": "library"},
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
            pagecache.drop("analytics")   # new numbers: the report is now wrong
        except Exception as exc:  # pragma: no cover - background best-effort
            logging.getLogger("history").warning("Post-import snapshot failed: %s", exc)

    _in_background(_pull_insights)
    return _flash(
        next,
        f"Imported {result['imported']} posts ({result['skipped']} already known) — "
        f"pulling insights in the background; see Analytics shortly.",
    )


def _calendar_data(y: int, m: int) -> dict:
    """The reads behind one month of the calendar: slots, queue and counts."""
    first_local = dt.datetime(y, m, 1)
    next_first_local = dt.datetime(y + 1, 1, 1) if m == 12 else dt.datetime(y, m + 1, 1)

    events: dict[int, list[dict]] = {}
    drafts_count = 0
    queue_count = 0
    linear: list[dict] = []
    status = {}
    windows_et: list[str] = []
    with session_scope() as session:
        # One grouped query instead of two counts: round trips are the page's
        # whole cost on a remote database.
        status_counts = dict(session.execute(
            select(ThreadsPost.status, func.count())
            .where(ThreadsPost.status.in_(("draft", "queued")))
            .group_by(ThreadsPost.status)
        ).all())
        drafts_count = status_counts.get("draft", 0)
        queue_count = status_counts.get("queued", 0)
        status = scheduler_status(session)
        windows_et = list(status.get("windows") or [])

        plan = build_window_plan(session, first_local, next_first_local)
        # Posts that carry a paired Instagram reel get a marker on their card.
        reel_post_ids = {
            pk for (pk,) in session.execute(
                select(InstagramPost.threads_post_pk)
                .where(InstagramPost.threads_post_pk.is_not(None))
            ).all()
        }
        for e in plan:
            # Reruns link to their ORIGINAL airing; that post's reel already
            # went out and won't re-post, so no +IG marker on the rerun card.
            e["has_reel"] = bool(e["kind"] != "rerun" and e.get("post_id")
                                 and e["post_id"] in reel_post_ids)
            # Calendar grid: published history + upcoming filled/open windows.
            events.setdefault(e["day"], []).append(e)

        # Linear queue: upcoming windows only (not published history).
        linear = [e for e in plan if e["kind"] in ("queued", "open", "rerun")]
        # Cap the linear list to the next ~21 slots so it stays scannable.
        linear = linear[:21]

    for day in events:
        events[day].sort(key=lambda e: e["sort"])

    return {"events": events, "drafts_count": drafts_count, "queue_count": queue_count,
            "linear": linear, "windows_et": windows_et, "year": y, "month": m}


def _current_month_calendar_data() -> dict:
    now_local = dt.datetime.now()
    return _calendar_data(now_local.year, now_local.month)


pagecache.register("calendar", _current_month_calendar_data)


def _week_start_for(day: dt.date) -> dt.date:
    """The Sunday on or before ``day`` (the calendar is Sunday-first)."""
    return day - dt.timedelta(days=(day.weekday() + 1) % 7)


def _calendar_week_data(week_start: dt.date) -> dict:
    """One week of window slots for the week view, bucketed by ISO date.

    Separate from the month read because a week can straddle two months, and
    because the week cards show real stills: exported clips get their own
    footage via /media/cut-thumb, uploads fall back to a local frame — the
    month grid never pays for any of that.
    """
    start_local = dt.datetime(week_start.year, week_start.month, week_start.day)
    end_local = start_local + dt.timedelta(days=7)

    week_events: dict[str, list[dict]] = {}
    with session_scope() as session:
        plan = build_window_plan(session, start_local, end_local)
        reel_post_ids = {
            pk for (pk,) in session.execute(
                select(InstagramPost.threads_post_pk)
                .where(InstagramPost.threads_post_pk.is_not(None))
            ).all()
        }
        # Resolve a better still per post than candidate.thumbnail_url alone
        # (which is all build_window_plan carries): the exported clip's own
        # frame first, then the source video's local frame for uploads.
        post_ids = sorted({e["post_id"] for e in plan if e.get("post_id")})
        thumb_by_post: dict[int, str] = {}
        if post_ids:
            rows = session.execute(
                select(ThreadsPost.id, Cut.id, Cut.trimmed_clip_path,
                       Candidate.id, Candidate.local_video_path)
                .select_from(ThreadsPost)
                .outerjoin(Cut, ThreadsPost.cut_pk == Cut.id)
                .outerjoin(Candidate, ThreadsPost.candidate_pk == Candidate.id)
                .where(ThreadsPost.id.in_(post_ids))
            ).all()
            for pid, cut_id, clip_path, cand_id, local_video in rows:
                if cut_id and clip_path:
                    thumb_by_post[pid] = f"/media/cut-thumb/{cut_id}"
                elif cand_id and local_video:
                    thumb_by_post[pid] = f"/media/thumb/{cand_id}"
        for e in plan:
            e["has_reel"] = bool(e["kind"] != "rerun" and e.get("post_id")
                                 and e["post_id"] in reel_post_ids)
            pid = e.get("post_id")
            if pid and thumb_by_post.get(pid):
                e["thumbnail"] = thumb_by_post[pid]
            week_events.setdefault(e["sort"].date().isoformat(), []).append(e)

    for k in week_events:
        week_events[k].sort(key=lambda e: e["sort"])
    return {"week_events": week_events, "week_start": week_start}


def _current_week_calendar_data() -> dict:
    return _calendar_week_data(_week_start_for(dt.date.today()))


pagecache.register("calendar-week", _current_week_calendar_data)


@app.get("/calendar", response_class=HTMLResponse)
def calendar_page(request: Request, year: int = 0, month: int = 0,
                  start: str = "", msg: str = ""):
    """Week cards (default), month grid and linear queue of posting windows
    (local time). ``start`` focuses the week view on any date in that week;
    ``year``/``month`` page the month grid."""
    import calendar as _cal

    now_local = dt.datetime.now()
    y = year or now_local.year
    m = month or now_local.month
    if m < 1:
        y, m = y - 1, 12
    elif m > 12:
        y, m = y + 1, 1

    # Only this month is kept warm; paging back through history is rare enough
    # to read directly (and the cached month self-corrects after a rollover).
    data = pagecache.read("calendar")
    if (data["year"], data["month"]) != (y, m):
        data = _calendar_data(y, m)

    # Week focus: any date normalizes to its Sunday. Current week stays warm
    # in the pagecache; paging to other weeks reads live.
    try:
        focus = dt.date.fromisoformat(start) if start else now_local.date()
    except ValueError:
        focus = now_local.date()
    week_start = _week_start_for(focus)
    wdata = pagecache.read("calendar-week")
    if wdata["week_start"] != week_start:
        wdata = _calendar_week_data(week_start)
    week_days = [week_start + dt.timedelta(days=i) for i in range(7)]
    week_end = week_days[-1]
    if week_start.month == week_end.month:
        week_title = f"{week_start.strftime('%b')} {week_start.day} – {week_end.day}, {week_end.year}"
    else:
        week_title = (f"{week_start.strftime('%b')} {week_start.day} – "
                      f"{week_end.strftime('%b')} {week_end.day}, {week_end.year}")

    cal = _cal.Calendar(firstweekday=6)  # Sunday-first
    weeks = cal.monthdayscalendar(y, m)
    today = now_local.day if (y == now_local.year and m == now_local.month) else 0

    prev_y, prev_m = (y - 1, 12) if m == 1 else (y, m - 1)
    next_y, next_m = (y + 1, 1) if m == 12 else (y, m + 1)

    return templates.TemplateResponse(
        request, "calendar.html",
        {**data, **wdata, "weeks": weeks, "today": today,
         "month_name": _cal.month_name[m],
         "prev_y": prev_y, "prev_m": prev_m, "next_y": next_y, "next_m": next_m,
         "week_days": week_days, "week_title": week_title,
         "today_date": now_local.date(),
         "week_prev": (week_start - dt.timedelta(days=7)).isoformat(),
         "week_next": (week_start + dt.timedelta(days=7)).isoformat(),
         "dow": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
         "windows_local": window_time_labels(),
         "msg": msg, "active": "calendar"},
    )


@app.get("/posts")
def posts_page(msg: str = ""):
    """Retired page. The queue lives on the Calendar, drafts/history in the
    Library, and Threads connection + scheduler status now live on the Calendar
    too. Kept as a redirect so old links and bookmarks still resolve."""
    return RedirectResponse("/calendar" + (f"?msg={msg}" if msg else ""), status_code=307)


def _notifications_data() -> dict:
    """Failed posts and reels awaiting a decision, plus reels left behind."""
    with session_scope() as session:
        failed = session.execute(
            select(ThreadsPost)
            .options(selectinload(ThreadsPost.cut).selectinload(Cut.candidate),
                     selectinload(ThreadsPost.candidate))
            .where(ThreadsPost.status == "failed",
                   ThreadsPost.attention_dismissed_at.is_(None))
            .order_by(ThreadsPost.created_at.desc())
        ).scalars().all()
        ig_failed = session.execute(
            select(InstagramPost)
            .options(selectinload(InstagramPost.cut).selectinload(Cut.candidate),
                     selectinload(InstagramPost.threads_post))
            .where(InstagramPost.status == "failed",
                   InstagramPost.attention_dismissed_at.is_(None))
            .order_by(InstagramPost.created_at.desc())
        ).scalars().all()
        ig_stranded = session.execute(
            select(InstagramPost)
            .options(selectinload(InstagramPost.cut).selectinload(Cut.candidate),
                     selectinload(InstagramPost.threads_post))
            .where(*stranded_reel_filters())
            .order_by(InstagramPost.created_at.desc())
        ).scalars().all()
        # Timely posts that went stale in the queue. The scored scheduler
        # never auto-places an expired post, so without this they'd sink
        # silently; the operator decides — post now, or delete.
        try:
            expired = expired_queued_posts(session)
            for p in expired:  # eager-load for the template before detaching
                _ = (p.cut.candidate if p.cut else None), p.candidate
        except Exception:
            log.exception("Expired-post check failed")
            expired = []
    return {"failed": failed, "ig_failed": ig_failed, "ig_stranded": ig_stranded,
            "expired": expired}


pagecache.register("notifications", _notifications_data)


@app.get("/notifications", response_class=HTMLResponse)
def notifications_page(request: Request, msg: str = ""):
    """Operator alerts — currently failed posts that dropped out of the queue and
    need a decision (retry, re-queue, or dismiss)."""
    return templates.TemplateResponse(
        request, "notifications.html",
        {**pagecache.read("notifications"), "msg": msg, "active": "notifications"},
    )


@app.post("/post/{post_id}/shelf-life")
def set_post_shelf_life(post_id: int, shelf_life: str = Form(""), next: str = Form("")):
    """Operator overrides this post's shelf life, or clears the override.

    Writes the POST-level column only — ``resolve_shelf_life`` reads it ahead
    of the candidate's LLM tag, and the tag itself is never touched, so
    clearing the override ("reset to AI") falls straight back to the model's
    answer (or the category default). Shelf life drives placement urgency and
    the expiry gate, so a correction takes effect on the next plan build.
    """
    dest = next if next.startswith("/") else f"/post/{post_id}"
    shelf = (shelf_life or "").strip().lower()
    if shelf and shelf not in SHELF_LIVES:
        return _flash(dest, f"Unknown shelf life '{shelf}'")
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return _flash("/calendar", "Post not found")
        p.shelf_life = shelf
    # Shelf life gates rerun eligibility; refresh the outlook immediately so
    # the Recycling card reflects the override on the very next page load.
    invalidate_recycle_overview()
    return _flash(dest, f"Shelf life set to {shelf} (your override)" if shelf
                  else "Shelf life reset to the AI's answer")


@app.post("/video/{candidate_id}/shelf-life")
def set_candidate_shelf_life(candidate_id: int, shelf_life: str = Form(""), next: str = Form("")):
    """Operator corrects the content's shelf life at the video/clip stage.

    Writes the candidate-level tag — the same column the monitor's LLM pass
    fills — so every clip and future post from this video inherits the
    correction (posts with their own override still win). Unlike the post
    route there is no "clear": overwriting the tag IS the correction, and
    blanking it would silently fall back to the category default.
    """
    dest = next if next.startswith("/") else f"/video/{candidate_id}"
    shelf = (shelf_life or "").strip().lower()
    if shelf not in SHELF_LIVES:
        return _flash(dest, f"Unknown shelf life '{shelf}'")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/library", "Video not found")
        c.shelf_life = shelf
    # Shelf life gates rerun eligibility; refresh the outlook immediately so
    # the Recycling card reflects the correction on the very next page load.
    invalidate_recycle_overview()
    return _flash(dest, f"Shelf life set to {shelf}")


def _apply_trait_tags(session, obj, action: str, tags: list[str], facet: str,
                      subject_col: str, format_col: str) -> list[str]:
    """Add or remove trait names on ``obj``, returning what actually changed.

    Shared by the post, video and clip rails: the three stages keep their tags
    in differently-named columns but the add/remove semantics are identical, so
    the caller only has to say which column holds which facet.
    """
    changed: list[str] = []
    for t in tags:
        # Which facet a name belongs to is a property of the trait, not of
        # the box it was typed into, so the vocabulary overrules the form
        # on add. Removal trusts the form: the pill was rendered from one
        # column and that is the column it has to come out of, however it
        # got there.
        t_facet = facet
        if action == "add":
            known = session.execute(
                select(Trait).where(Trait.name == t)
            ).scalar_one_or_none()
            if known is not None:
                t_facet = known.facet
        col = format_col if t_facet == Trait.FACET_FORMAT else subject_col
        existing = [x.strip() for x in (getattr(obj, col) or "").split(",") if x.strip()]
        if action == "add":
            if t not in existing:
                existing.append(t)
                changed.append(t)
        else:
            if t in existing:
                changed.append(t)
            existing = [x for x in existing if x != t]
        setattr(obj, col, ",".join(existing))
    return changed


def _clean_tag_form(tag: list[str]) -> list[str]:
    """Normalize submitted trait names. ``tag`` repeats — the add combo posts
    every checked name in one go (remove buttons still send a single one)."""
    tags: list[str] = []
    for t in tag:
        t = t.strip().lower().replace(" ", "_")
        if t and t not in tags:
            tags.append(t)
    return tags


def _tag_flash(dest: str, action: str, changed: list[str]):
    if not changed:
        return _flash(dest, "Already tagged" if action == "add" else "Nothing to remove")
    return _flash(dest, f"{'Added' if action == 'add' else 'Removed'} {', '.join(changed)}")


@app.post("/post/{post_id}/tags")
def update_post_tags(post_id: int, action: str = Form(...),
                     tag: list[str] = Form([]), facet: str = Form("subject"),
                     next: str = Form("")):
    """Add or remove trait tags on a post.

    ``facet`` picks the column — ``format_tags`` for format, ``footage_traits``
    for subject. Each facet has its own add box, so the caller always states
    which one it meant.
    """
    dest = next if next and next.startswith("/") else f"/post/{post_id}"
    tags = _clean_tag_form(tag)
    if not tags:
        return _flash(dest, "No tag selected")
    if action not in ("add", "remove"):
        return _flash(dest, f"Unknown action '{action}'")
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return _flash("/calendar", "Post not found")
        changed = _apply_trait_tags(session, p, action, tags, facet,
                                    "footage_traits", "format_tags")
        if changed:
            # A hand-edit is ground truth and outranks the model. The
            # queue-time annotation pass claims every queued post whose
            # ``footage_scored_at`` is still null and overwrites both columns,
            # so without this stamp a draft tagged here would be silently
            # reverted the moment it was queued.
            p.footage_scored_at = p.footage_scored_at or utcnow()
    return _tag_flash(dest, action, changed)


@app.post("/video/{candidate_id}/tags")
def update_video_tags(candidate_id: int, action: str = Form(...),
                      tag: list[str] = Form([]), facet: str = Form("subject"),
                      next: str = Form("")):
    """Add or remove trait tags on a source video.

    These are the storyboard tagger's predictions about the whole video, and
    the post's own annotation supersedes them — but not before they've had two
    effects nothing revisits: the subject facet multiplies the candidate's
    ranking score in triage, and the format facet is what the scheduler's
    variety gate reads until a post is annotated. Correcting a bad guess here
    is worth doing on its own.
    """
    dest = next if next and next.startswith("/") else f"/video/{candidate_id}"
    tags = _clean_tag_form(tag)
    if not tags:
        return _flash(dest, "No tag selected")
    if action not in ("add", "remove"):
        return _flash(dest, f"Unknown action '{action}'")
    with session_scope() as session:
        c = session.get(Candidate, candidate_id)
        if c is None:
            return _flash("/", "Video not found")
        changed = _apply_trait_tags(session, c, action, tags, facet,
                                    "visual_traits", "format_tags")
    return _tag_flash(dest, action, changed)


def _sync_cut_tags_to_draft_posts(session, cut: Cut) -> None:
    """Push clip tags onto any draft/queued/failed post made from this cut."""
    for p in session.execute(
        select(ThreadsPost).where(
            ThreadsPost.cut_pk == cut.id,
            ThreadsPost.status.in_(("draft", "queued", "failed")),
        )
    ).scalars().all():
        p.footage_traits = cut.footage_traits
        p.format_tags = cut.format_tags
        if (cut.format_tags or "").strip() or (cut.footage_traits or "").strip():
            p.footage_scored_at = p.footage_scored_at or utcnow()
            p.footage_rationale = p.footage_rationale or "Tagged on the clip."


_cut_annotate_inflight: set[int] = set()


def _annotate_cut_in_thread(cut_id: int) -> None:
    """Backfill Format/Subject for an already-exported clip that has none."""
    try:
        with session_scope() as session:
            cut = session.get(Cut, cut_id)
            if cut is None:
                return
            if cut.footage_tagged_at is not None:
                return
            if (cut.format_tags or "").strip() or (cut.footage_traits or "").strip():
                cut.footage_tagged_at = utcnow()
                return
            if not cut.trimmed_clip_path or not Path(cut.trimmed_clip_path).exists():
                return
            settings = load_settings()
            vocab = active_traits_by_facet(session)
            if annotate_cut_footage(cut, settings, vocab["subject"],
                                    format_traits=vocab["format"]):
                _sync_cut_tags_to_draft_posts(session, cut)
    except Exception:
        log.exception("Background cut tagging failed for cut %s", cut_id)
    finally:
        _cut_annotate_inflight.discard(cut_id)


@app.get("/cut/{cut_id}/tags-status")
def cut_tags_status(cut_id: int):
    """Polled while background tagging runs on an already-exported clip."""
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        tagged = bool(
            cut.footage_tagged_at is not None
            or (cut.format_tags or "").strip()
            or (cut.footage_traits or "").strip()
        )
        return {"tagged": tagged, "pending": cut_id in _cut_annotate_inflight and not tagged}


@app.post("/cut/{cut_id}/tags")
def update_cut_tags(cut_id: int, action: str = Form(...),
                    tag: list[str] = Form([]), facet: str = Form("subject"),
                    next: str = Form("")):
    """Add or remove trait tags on a clip.

    The clip is the footage that actually ships, so tagging here is ground
    truth rather than a prediction: ``seed_post_tags_from_cut`` copies these
    onto the post at queue time and stamps it annotated, which stands in for
    the LLM pass entirely. Edits also flow straight through to any draft post
    already made from this clip, so the two views can't disagree.
    """
    dest = next if next and next.startswith("/") else f"/cut/{cut_id}?step=post"
    tags = _clean_tag_form(tag)
    if not tags:
        return _flash(dest, "No tag selected")
    if action not in ("add", "remove"):
        return _flash(dest, f"Unknown action '{action}'")
    with session_scope() as session:
        cut = session.get(Cut, cut_id)
        if cut is None:
            return _flash("/library", "Clip not found")
        changed = _apply_trait_tags(session, cut, action, tags, facet,
                                    "footage_traits", "format_tags")
        if changed:
            cut.footage_tagged_at = cut.footage_tagged_at or utcnow()
            _sync_cut_tags_to_draft_posts(session, cut)
    return _tag_flash(dest, action, changed)


@app.post("/post/{post_id}/dismiss")
def dismiss_attention(request: Request, post_id: int, next: str = Form("/notifications")):
    """Acknowledge a failed post so it leaves the notifications list (kept in history)."""
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        p = session.get(ThreadsPost, post_id)
        if p is None:
            return (JSONResponse({"error": "Post not found"}, status_code=404)
                    if wants_json else _flash("/notifications", "Post not found"))
        p.attention_dismissed_at = utcnow()
    return JSONResponse({"ok": True}) if wants_json else _flash(next, "Dismissed")


@app.post("/igpost/{ig_id}/retry")
def retry_instagram_post(ig_id: int, next: str = Form("/notifications")):
    """Operator-confirmed retry of a failed reel. Synchronous, like manual
    Threads publishing — Meta's processing poll can take a minute or two."""
    with session_scope() as session:
        ig = session.get(InstagramPost, ig_id)
        if ig is None:
            return _flash(next, "Reel not found")
        if ig.status not in ("failed", "queued", "draft"):
            return _flash(next, "This reel is not retryable")
        try:
            publish_instagram_post(session, ig)
            return _flash(next, f"Reel published: {ig.permalink or ig.ig_media_id}")
        except Exception as exc:
            return _flash(next, f"Reel publish failed: {exc}")


@app.post("/igpost/{ig_id}/dismiss")
def dismiss_instagram_attention(request: Request, ig_id: int,
                                next: str = Form("/notifications")):
    """Acknowledge a failed reel so it leaves the notifications list."""
    wants_json = "application/json" in request.headers.get("accept", "")
    with session_scope() as session:
        ig = session.get(InstagramPost, ig_id)
        if ig is None:
            return (JSONResponse({"error": "Reel not found"}, status_code=404)
                    if wants_json else _flash("/notifications", "Reel not found"))
        ig.attention_dismissed_at = utcnow()
    return JSONResponse({"ok": True}) if wants_json else _flash(next, "Dismissed")


@app.post("/igpost/{ig_id}/cancel")
def cancel_instagram_post(ig_id: int, next: str = Form("/notifications")):
    """Remove a not-yet-published reel (the Threads post is untouched)."""
    with session_scope() as session:
        ig = session.get(InstagramPost, ig_id)
        if ig is None:
            return _flash(next, "Reel not found")
        if ig.status == "published":
            return _flash(next, "This reel is already live")
        session.delete(ig)
    return _flash(next, "Reel removed — the Threads post is unaffected")


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

def _analytics_report() -> dict:
    """The whole report. It reads every post's metric history, which is by far
    the heaviest read in the app, and it takes no parameters."""
    with session_scope() as session:
        return generate_report(session)


# Seconds to build, and only new metric snapshots change what it says — so it
# isn't built on spec, and approving a video doesn't throw it away. The TTL
# covers the scheduler's own metric polls; taking a snapshot by hand drops it
# outright.
pagecache.register("analytics", _analytics_report, background=False, volatile=False)


def _analytics_view(report: dict) -> dict:
    return {"rows": report["rows"], "slices": report["slices"], "digest": report["digest"],
            "timeseries": report["timeseries"], "summary": report["summary"],
            "spend_today": spend.today_spend(), "spend_budget": spend.daily_budget(),
            "spend_recent": spend.recent(30)}


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page(request: Request, msg: str = ""):
    # Only wait for the report if it's already built. Otherwise the page draws
    # without it and asks for the body separately, so a cold report costs a
    # placeholder rather than a blank tab.
    report = pagecache.peek("analytics")
    return templates.TemplateResponse(
        request, "analytics.html",
        {**(_analytics_view(report) if report else {}),
         "ready": report is not None, "msg": msg, "active": "analytics"},
    )


@app.get("/analytics/body", response_class=HTMLResponse)
def analytics_body(request: Request):
    """The report on its own, as a fragment the page swaps in once it's built."""
    return templates.TemplateResponse(
        request, "components/analytics_body.html",
        _analytics_view(pagecache.read("analytics")),
    )


@app.post("/analytics/digest")
def analytics_digest():
    """Write a fresh digest — the one analytics action that spends an LLM call,
    so it happens because the operator asked, never because a page was opened.

    Synchronous: it's a deliberate click that takes about half a minute, and the
    button spins for it rather than the page inventing a progress protocol."""
    if not spend.within_budget():
        return JSONResponse(
            {"error": f"Daily LLM budget of ${spend.daily_budget():.2f} is used up"},
            status_code=429)
    try:
        with session_scope() as session:
            result = write_and_store_digest(session)
    except Exception as exc:
        log.warning("Digest generation failed: %s", exc)
        return JSONResponse({"error": f"Could not write the digest: {exc}"}, status_code=502)
    pagecache.drop("analytics")   # the report carries the digest with it
    if not result["text"]:
        return JSONResponse({"error": "No published posts to write about"}, status_code=400)
    return JSONResponse({
        "ok": True,
        "text": result["text"],
        "meta": _digest_meta(result),
    })


@app.post("/analytics/snapshot")
def analytics_snapshot():
    with session_scope() as session:
        try:
            n = snapshot_metrics(session)
            pagecache.drop("analytics")   # new numbers: the report is now wrong
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
    """(Re)draft the first-comment for a post — a source citation or a call to
    action, per the mode on the Replies page. Returns the suggestion only: it is
    saved when the operator updates the queue, so nothing changes until they've
    seen it."""
    invitation = load_first_reply().get("mode") == "invitation"
    with session_scope() as session:
        p = session.execute(
            select(ThreadsPost)
            .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel))
            .where(ThreadsPost.id == post_id)
        ).scalar_one_or_none()
        if p is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if p.candidate is None and not invitation:
            return JSONResponse(
                {"error": "No source video on record for this post, so there's "
                          "nothing to attribute from."},
                status_code=409)
        try:
            # Gather inside the session, call the model outside it: see
            # ``first_reply_context``.
            if invitation:
                pending = first_reply_context(session, p.candidate, p.cut,
                                              p.caption or "")
            else:
                text = generate_attribution(p.candidate)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
    if invitation:
        try:
            text = draft_first_reply(pending)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
    if not text:
        # The model declined rather than guess — surface that honestly instead
        # of proposing a made-up credit.
        return {"text": "", "unavailable": True,
                "message": _no_suggestion_message(invitation)}
    return {"text": text}


# --- Settings ----------------------------------------------------------------

@app.get("/settings")
def settings_index():
    """The sidebar's one door to configuration; the rail lives in settings_base.html.

    Every settings page keeps its own top-level URL, so this only has to land on
    the first one.
    """
    return RedirectResponse("/brand", status_code=307)


# --- Brand & audience ----------------------------------------------------------

_ALLOWED_LOGO_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif")


@app.get("/brand", response_class=HTMLResponse)
def brand_page(request: Request, msg: str = ""):
    """Workspace identity + white-label appearance (Configure area)."""
    return templates.TemplateResponse(
        request, "brand.html",
        {"brand": load_brand(), "default_app_name": DEFAULT_APP_NAME,
         "msg": msg, "active": "brand"},
    )


@app.post("/brand")
async def brand_save(
    name: str = Form(""), mission: str = Form(""), audience: str = Form(""),
    topic: str = Form(""), voice_notes: str = Form(""), app_name: str = Form(""),
    source_kind: str = Form(""), relevance_rules: str = Form(""),
    false_positives: str = Form(""), strong_openings: str = Form(""),
    weak_openings: str = Form(""), clip_guidance: str = Form(""),
    logo: UploadFile | None = File(None),
):
    values = {"name": name, "mission": mission, "audience": audience,
              "topic": topic, "voice_notes": voice_notes, "app_name": app_name,
              "source_kind": source_kind, "relevance_rules": relevance_rules,
              "false_positives": false_positives,
              "strong_openings": strong_openings,
              "weak_openings": weak_openings, "clip_guidance": clip_guidance}
    if logo is not None and (logo.filename or "").strip():
        ext = Path(logo.filename).suffix.lower()
        if ext not in _ALLOWED_LOGO_EXTS:
            return _flash("/brand", "Logo must be a PNG, JPG, SVG, WebP, or GIF")
        data = await logo.read()
        if len(data) > 2 * 1024 * 1024:
            return _flash("/brand", "Logo is too large — keep it under 2 MB")
        _BRAND_LOGO_DIR.mkdir(parents=True, exist_ok=True)
        # One canonical file: a new upload replaces the old, whatever its type.
        for old in _BRAND_LOGO_DIR.glob("logo.*"):
            old.unlink(missing_ok=True)
        (_BRAND_LOGO_DIR / f"logo{ext}").write_bytes(data)
        values["logo_file"] = f"logo{ext}"
    save_brand(values)
    return _flash("/brand", "Brand & audience saved")


@app.post("/brand/logo/remove")
def brand_logo_remove():
    for old in _BRAND_LOGO_DIR.glob("logo.*"):
        old.unlink(missing_ok=True)
    save_brand({"logo_file": ""})
    return _flash("/brand", "Logo removed")


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
         "mode": cfg["mode"], "instruction": cfg["instruction"],
         "msg": msg, "active": "engagement"},
    )


@app.post("/engagement/first-reply")
def first_reply_save(enabled: str = Form(""), text: str = Form(""),
                     attribution_enabled: str = Form(""), mode: str = Form("citation"),
                     instruction: str = Form("")):
    text = (text or "").strip()
    instruction = (instruction or "").strip()
    mode = (mode or "citation").strip().lower()
    on = str(enabled).lower() in ("1", "true", "on", "yes")
    attribution_on = str(attribution_enabled).lower() in ("1", "true", "on", "yes")
    if on and not text:
        return _flash("/engagement/first-reply", "Add reply text before enabling")
    if len(text) > 500:
        return _flash("/engagement/first-reply", f"Reply is {len(text)} characters — Threads limit is 500")
    if mode == "invitation" and not instruction:
        return _flash("/engagement/first-reply",
                      "Write the brief before switching to call-to-action replies")
    save_first_reply(enabled=on, text=text, attribution_enabled=attribution_on,
                     mode=mode, instruction=instruction)
    drafts = "a call to action" if mode == "invitation" else "a source citation"
    state = "enabled" if on else "disabled"
    attr_state = "on" if attribution_on else "off"
    return _flash("/engagement/first-reply",
                  f"Saved — first replies draft {drafts}, posting {attr_state}, "
                  f"static fallback {state}")


@app.get("/first-reply")
def first_reply_redirect():
    return RedirectResponse("/engagement/first-reply", status_code=303)


# --- Style guide (caption drafting prompt) -----------------------------------

@app.get("/style-guide", response_class=HTMLResponse)
def style_guide_page(request: Request, msg: str = ""):
    from ..caption_insights import has_generated, load_suggestions
    from ..draft_proposals import recent_pairs
    from ..draft_proposals import stats as draft_loop_stats
    from ..voice import collect_voice_captions, length_target

    settings = load_settings()
    with session_scope() as session:
        suggestions = load_suggestions(session)
        generated = has_generated(session)
        # Not "loop": Jinja shadows that name inside every {% for %} block.
        caption_loop = draft_loop_stats(session, KIND_CAPTION)
        hook_loop = draft_loop_stats(session, KIND_HOOK)
        pairs = recent_pairs(session, KIND_CAPTION)
        hook_pairs = recent_pairs(session, KIND_HOOK)
        # Read the target directly rather than through voice_context: that
        # would distill the style guide, and a page view must not spend an
        # LLM call.
        try:
            target_words = length_target(collect_voice_captions(session), settings)
        except Exception:
            log.exception("Caption length target failed")
            target_words = None
    return templates.TemplateResponse(
        request, "style_guide.html",
        {"rules": load_caption_rules(), "suggestions": suggestions,
         "has_generated": generated, "msg": msg, "active": "style_guide",
         "caption_loop": caption_loop, "pairs": pairs, "target_words": target_words,
         "hook_loop": hook_loop, "hook_pairs": hook_pairs,
         "max_chars": int(settings.get("engagement.caption_max_chars", 220))},
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
        # Both tag columns: subject traits live in footage_traits, format
        # traits in format_tags. One count dict works because a name belongs
        # to exactly one facet (split_tags_by_facet partitions on it).
        post_tag_rows = session.execute(
            select(ThreadsPost.footage_traits, ThreadsPost.format_tags)
            .where(ThreadsPost.status == "published")
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
    for subject_csv, format_csv in post_tag_rows:
        tags = [t.strip() for t in (subject_csv or "").split(",") if t.strip()]
        tags += [t.strip() for t in (format_csv or "").split(",") if t.strip()]
        if tags:
            annotated_posts += 1
        for t in set(tags):
            post_counts[t] = post_counts.get(t, 0) + 1
    facet_counts = {
        "subject": sum(1 for t in traits if t.facet != Trait.FACET_FORMAT),
        "format": sum(1 for t in traits if t.facet == Trait.FACET_FORMAT),
    }
    return templates.TemplateResponse(
        request, "traits.html",
        {"traits": traits, "post_counts": post_counts,
         "annotated_posts": annotated_posts, "facet_counts": facet_counts,
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
            vocab = active_traits_by_facet(session)
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
                if annotate_post_footage(post, settings, vocab["subject"],
                                         format_traits=vocab["format"]):
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
    _in_background(_annotate_posts_in_thread)
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
def trait_add(name: str = Form(...), facet: str = Form(Trait.FACET_SUBJECT)):
    name = _normalize_trait(name)
    if not name:
        return _flash("/traits", "Empty trait name")
    facet = (facet or "").strip().lower()
    if facet not in (Trait.FACET_SUBJECT, Trait.FACET_FORMAT):
        return _flash("/traits", f"Unknown facet '{facet}'")
    with session_scope() as session:
        exists = session.execute(select(Trait).where(Trait.name == name)).scalar_one_or_none()
        if exists:
            return _flash("/traits", f"'{name}' already exists")
        session.add(Trait(name=name, kind=Trait.KIND_NEUTRAL, facet=facet,
                          enabled=True))
    invalidate_traits_cache()
    label = "format" if facet == Trait.FACET_FORMAT else "subject"
    return _flash("/traits", f"Added '{name}' ({label})")


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


@app.post("/channels/{channel_id}/first-party")
def channel_first_party_toggle(channel_id: int):
    """Mark a channel as the operator's own content (provenance facet).

    First-party posts join the promo rotation pool and are excluded from
    voice/caption learning, which is trained on found footage only.
    """
    with session_scope() as session:
        channel = session.get(Channel, channel_id)
        if channel:
            channel.first_party = not bool(channel.first_party)
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
