"""The Inbox page: finished clips waiting for one click (app/inbox.py).

"Today's posts": the page a customer lands on after signing in. Each clip has
three actions: Approve (one click: queued for the next open slot), Reject (one
click, a few seconds to undo, then optional "what was wrong?" reasons that go
into the clip-engine notes like any dismissed suggestion) and Edit caption
(then Save & approve). Admins also see where each clip came from and what was
decided lately (app/web/access.py). Its settings are on the admin Automation
page (``/inbox/settings`` posts from there).

Like the other routers it renders through ``request.app.state.templates``.
The few helpers that live in main.py (deleting a rejected clip's render,
saving a clip note) are imported inside the handlers: by the time a request
arrives main.py has finished loading, so there is no import cycle.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .. import auto_select, inbox
from . import access
from ..config import load_settings, scheduler_timezone
from ..db import session_scope
from ..models import Candidate, Cut, InboxItem, utcnow

log = logging.getLogger("inbox.page")

router = APIRouter()


def _wants_json(request: Request) -> bool:
    return "application/json" in request.headers.get("accept", "")


def _reply(request: Request, ok: bool, msg: str, status: int = 200, **extra):
    if _wants_json(request):
        body = {"ok": ok, "msg": msg, **extra} if ok else {"error": msg, **extra}
        return JSONResponse(body, status_code=status)
    return RedirectResponse(f"/inbox?msg={quote(msg)}", status_code=303)


@router.get("/inbox", response_class=HTMLResponse)
def inbox_page(request: Request, msg: str = ""):
    with session_scope(read_only=True) as session:
        items = inbox.ready_items(session)
        cuts = {c.id: c for c in session.execute(
            select(Cut).options(selectinload(Cut.candidate).selectinload(Candidate.channel))
            .where(Cut.id.in_([i.cut_pk for i in items if i.cut_pk]))
        ).scalars()} if items else {}
        start, _ = auto_select.day_window(utcnow(), scheduler_timezone())
        today = inbox.counts(session, since=start)
        policy = inbox.load_policy(session)
        admin = access.is_admin(request)
        decided = _decided(session) if admin else []
    cards = []
    for item in items:
        cut = cuts.get(item.cut_pk)
        if cut is None:
            continue
        c = cut.candidate
        cards.append({
            "item": item,
            "cut": cut,
            "video_title": c.title if c else "",
            "channel": c.channel.call_sign if c and c.channel else "",
            "candidate_id": c.id if c else None,
            "src": f"/inbox/{item.id}/media",
        })
    return request.app.state.templates.TemplateResponse(
        request, "inbox.html",
        {"cards": cards, "today": today, "policy": policy, "msg": msg, "active": "inbox",
         "admin": admin, "decided": decided,
         "day_label": f"{utcnow().astimezone(scheduler_timezone()):%A, %B} "
                      f"{utcnow().astimezone(scheduler_timezone()).day}"})


def _decided(session, limit: int = 30) -> list[dict]:
    """For admins: what was approved, rejected or failed lately, and why."""
    items = inbox.recent(session, limit=limit)
    cands = {c.id: c for c in session.execute(
        select(Candidate).where(Candidate.id.in_({i.candidate_pk for i in items}))
    ).scalars()} if items else {}
    rows = []
    for it in items:
        c = cands.get(it.candidate_pk)
        rows.append({"item": it, "title": it.story or (c.title if c else ""),
                     "candidate_id": c.id if c else None,
                     "reasons": ", ".join(t.replace("_", " ") for t in (it.reject_tags or "").split(",") if t)})
    return rows


# --- The clip as it will post -------------------------------------------------------------

_render_locks: dict[str, threading.Lock] = {}
_render_locks_guard = threading.Lock()


def _plain_clip(item: InboxItem, cut: Cut) -> str:
    """The render the post will use, by path (it may not be on this machine)."""
    if item.use_subtitles and cut.subtitled_clip_path:
        return cut.subtitled_clip_path
    return cut.trimmed_clip_path or ""


def finished_clip(plain: str) -> Path | None:
    """``plain`` as it will actually post: with the workspace's label tab and
    closing card (app/outro.py), the same cached copy publishing uses. Fetched
    from Storage first on a machine without the file (the hosted dashboard).
    None when the clip can't be had here at all."""
    from .. import datasync
    from ..outro import with_outro

    if not plain or not datasync.ensure_file(plain):
        return None
    with _render_locks_guard:
        lock = _render_locks.setdefault(plain, threading.Lock())
    with lock:  # a player asks for the file several times at once; render it once
        return with_outro(plain)


@router.get("/inbox/{item_id}/media")
def inbox_media(item_id: int):
    """The clip with its tab and closing card, so what you approve is what
    posts. Falls back to the plain clip when this machine can't build it."""
    from . import main as web

    with session_scope(read_only=True) as session:
        item = session.get(InboxItem, item_id)
        cut = session.get(Cut, item.cut_pk) if item is not None and item.cut_pk else None
        plain = _plain_clip(item, cut) if cut is not None else ""
    if not plain:
        return JSONResponse({"error": "no clip"}, status_code=404)
    finished = finished_clip(plain)
    if finished is not None and finished.is_file():
        return FileResponse(str(finished), media_type="video/mp4")
    return web._clip_response(plain, "no clip")


@router.post("/inbox/{item_id}/approve")
def inbox_approve(request: Request, item_id: int, caption: str = Form("")):
    with session_scope() as session:
        item = session.get(InboxItem, item_id)
        if item is None:
            return _reply(request, False, "Clip not found", 404)
        try:
            inbox.approve(session, item, caption)
            session.flush()
            queued = inbox.queue_approved(session, item)
        except ValueError as exc:
            return _reply(request, False, str(exc), 409)
        except Exception as exc:
            log.exception("Approving inbox clip %s failed", item_id)
            return _reply(request, False, f"Couldn't queue it: {exc}", 500)
    msg = ("Queued for the next open slot" if queued
           else "Approved — the home machine queues it within a few minutes")
    return _reply(request, True, msg, queued=queued)


@router.post("/inbox/{item_id}/reject")
def inbox_reject(request: Request, item_id: int, tags: str = Form(""), note: str = Form("")):
    from . import main as web

    with session_scope() as session:
        item = session.get(InboxItem, item_id)
        if item is None:
            return _reply(request, False, "Clip not found", 404)
        try:
            inbox.reject(session, item, tags.split(","), note)
        except ValueError as exc:
            return _reply(request, False, str(exc), 409)
        cut = session.get(Cut, item.cut_pk) if item.cut_pk else None
        if cut is not None:
            web._inbox_discard_render(session, cut)
        candidate = session.get(Candidate, item.candidate_pk)
        video_id = candidate.video_id if candidate else ""
        proposal_pk, candidate_pk = item.proposal_pk, item.candidate_pk
        clean_tags, clean_note = item.reject_tags, item.reject_note
    if clean_tags or clean_note:
        web._save_clip_note(candidate_pk, video_id, kind="card", tags=clean_tags,
                            text=clean_note, proposal_id=proposal_pk)
    return _reply(request, True, "Rejected" + (" — noted for the house rules"
                                                if clean_tags or clean_note else ""))


@router.post("/inbox/{item_id}/reasons")
def inbox_reasons(request: Request, item_id: int, tags: str = Form(""), note: str = Form("")):
    """"What was wrong?" after a one-click rejection: optional, and it never
    holds the rejection up."""
    from . import main as web

    with session_scope() as session:
        item = session.get(InboxItem, item_id)
        if item is None:
            return _reply(request, False, "Clip not found", 404)
        try:
            added, clean_note = inbox.add_reasons(session, item, tags.split(","), note)
        except ValueError as exc:
            return _reply(request, False, str(exc), 409)
        candidate = session.get(Candidate, item.candidate_pk)
        video_id = candidate.video_id if candidate else ""
        proposal_pk, candidate_pk = item.proposal_pk, item.candidate_pk
    if added or clean_note:
        web._save_clip_note(candidate_pk, video_id, kind="card", tags=added,
                            text=clean_note, proposal_id=proposal_pk)
    return _reply(request, True, "Thanks — noted for the house rules")


@router.post("/inbox/settings")
def inbox_settings(enabled: str = Form(""), per_day: str = Form(""), min_score: str = Form(""),
                   per_video: str = Form(""), source: str = Form(""),
                   subtitles: str = Form("")):
    yes = lambda v: v.strip().lower() in ("1", "on", "true", "yes")  # noqa: E731
    try:
        with session_scope() as session:
            before = inbox.load_policy(session)
            policy = inbox.set_policy(session, enabled=yes(enabled), per_day=per_day or None,
                                      min_score=min_score or None, per_video=per_video or None,
                                      source=source or None, subtitles=yes(subtitles))
    except ValueError as exc:
        msg = str(exc)
    else:
        if policy.enabled and not before.enabled:
            msg = (f"The inbox is on: up to {policy.per_day} finished clips a day, made on the "
                   f"home machine from drafts the critic scored {policy.min_score:.1f} or better")
        elif before.enabled and not policy.enabled:
            msg = "The inbox is off — no more clips will be made automatically"
        else:
            msg = "Saved"
    return RedirectResponse(f"/auto-triage?msg={quote(msg)}#inbox", status_code=303)
