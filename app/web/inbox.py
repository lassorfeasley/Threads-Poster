"""The Inbox page: finished clips waiting for one click (app/inbox.py).

Approve queues the clip for the next open slot with the caption as you left
it; Reject takes the "What's wrong?" chips, which go into the clip-engine
notes like any dismissed suggestion. The page has nothing else on purpose —
it's the screen a customer gets. Its settings are on the admin Automatic
triage page (``/inbox/settings`` posts from there).

Like the other routers it renders through ``request.app.state.templates``.
The few helpers that live in main.py (deleting a rejected clip's render,
saving a clip note) are imported inside the handlers: by the time a request
arrives main.py has finished loading, so there is no import cycle.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .. import auto_select, inbox
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
            "src": f"/media/subtitled/{cut.id}" if item.use_subtitles else f"/media/clip/{cut.id}",
        })
    return request.app.state.templates.TemplateResponse(
        request, "inbox.html",
        {"cards": cards, "today": today, "policy": policy, "msg": msg, "active": "inbox"})


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
