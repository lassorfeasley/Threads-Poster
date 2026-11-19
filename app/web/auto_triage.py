"""The Automatic triage page (admin): the switch, the cut-off, the spot-checks,
every pass and every video it rejected — with undo at each level.

The logic is app/auto_triage.py; this is only the page. Like the clip bench
and handbook pages it lives in its own router and renders through
``request.app.state.templates`` (base.html's globals live on main.py's
environment). Approving and restoring a video reuse main.py's own routes
(``/video/{id}/approve``, ``/video/{id}/unreject``), so there is one approve
gate and one place every operator decision is logged.
"""
from __future__ import annotations

import json
import logging
import threading
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import selectinload

from .. import auto_select, auto_triage, inbox, triage_model
from ..config import env, load_settings
from ..db import session_scope
from ..models import STATUS_REJECTED, AutoTriageAction, Candidate
from . import pagecache

log = logging.getLogger("auto_triage.page")

router = APIRouter()

PAGE_SIZE = 50
SPOT_CHECKS_SHOWN = 24


def _back(msg: str, anchor: str = "") -> RedirectResponse:
    return RedirectResponse(f"/auto-triage?msg={quote(msg)}{anchor}", status_code=303)


def _retrain_in_background() -> None:
    """A spot-check verdict is a decision like any other: let the predictor
    catch up once enough have come in (the check is one count query)."""
    def _run() -> None:
        try:
            if triage_model.maybe_retrain():
                pagecache.invalidate()
        except Exception:
            log.exception("Triage model retrain failed")

    threading.Thread(target=_run, daemon=True, name="triage-retrain").start()


def _video_row(action: AutoTriageAction, c: Candidate) -> dict:
    duration = c.duration_seconds or 0
    return {
        "action_id": action.id,
        "run_pk": action.run_pk,
        "c": c,
        "id": c.id,
        "title": c.title,
        "url": c.url,
        "channel": c.channel.call_sign if c.channel else "",
        "published": c.published_at.strftime("%b %-d, %Y") if c.published_at else "?",
        "duration": f"{duration // 60}:{duration % 60:02d}" if duration else "",
        "prob": action.approve_prob,
        "signal": action.signal or auto_triage.SIGNAL_APPROVAL,
        "fit": action.fit_score,
        "fit_reason": action.fit_reason or "",
        "rejected_at": action.created_at,
        "audit": action.audit and not action.audit_verdict,
    }


def _model_view(session, settings) -> dict:
    predictor = triage_model.load_predictor(session)
    model, why = auto_triage.readiness(session, settings, predictor)
    if model is None:
        return {"ready": not why, "why": why, "model": None}
    grade = auto_triage.model_grade(model)
    try:
        rolling = "rolling" in json.loads(model.backtest or "{}")
    except ValueError:
        rolling = False
    return {"ready": not why, "why": why,
            "model": {"id": model.id, "n_decisions": model.n_decisions,
                      "n_approved": model.n_approved, "auc": grade.get("auc"),
                      "graded_on": grade.get("n"), "rolling": rolling,
                      "trained_at": model.trained_at}}


def _profile_view() -> dict:
    """Which parts of the Brand & audience profile fit has to go on."""
    from ..llm import fit_profile

    try:
        p = fit_profile()
    except Exception:
        log.exception("Brand profile unreadable")
        return {"filled": [], "missing": ["mission", "audience", "positioning"]}
    keys = ("mission", "audience", "positioning")
    return {"filled": [k for k in keys if p.get(k)], "missing": [k for k in keys if not p.get(k)]}


@router.get("/auto-triage", response_class=HTMLResponse)
def auto_triage_page(request: Request, run: int = 0, page: int = 1, msg: str = ""):
    settings = load_settings()
    page = max(1, page)
    with session_scope(read_only=True) as session:
        policy = auto_triage.load_policy(session)
        since = policy.watch_since_at
        spot = auto_triage.spot_check_summary(session, since)
        spot_all = auto_triage.spot_check_summary(session)
        rejected_since = auto_triage.count_standing(session, since)
        pending = [
            _video_row(a, c) for a, c in auto_triage.standing_rejections(
                session, audit_only=True, limit=SPOT_CHECKS_SHOWN)
        ]
        rejected = [
            _video_row(a, c) for a, c in auto_triage.standing_rejections(
                session, run_pk=run or None, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
        ]
        signal = policy.signal
        _, fit_why = auto_triage.readiness(session, settings, signal=auto_triage.SIGNAL_FIT)
        model_view = _model_view(session, settings)
        ready = model_view["ready"] if signal == auto_triage.SIGNAL_APPROVAL else not fit_why
        ctx = {
            "policy": policy,
            "state": policy.state,
            "signal": signal,
            "ready": ready,
            "not_ready_why": model_view["why"] if signal == auto_triage.SIGNAL_APPROVAL else fit_why,
            "model_view": model_view,
            "tables": {
                s: auto_triage.cutoff_table(session, settings, s) for s in auto_triage.SIGNALS
            },
            "preview": auto_triage.preview(session, settings, signal),
            "profile": _profile_view(),
            "select": auto_select.preview(session, settings),
            "select_recent": auto_select.recent_picks(session),
            "select_choices": {
                "per_day": auto_select.PER_DAY_CHOICES, "min_prob": auto_select.MIN_PROB_CHOICES,
                "fit_floor": auto_select.FIT_FLOOR_CHOICES, "explore": auto_select.EXPLORE_CHOICES,
                "fresh_days": auto_select.FRESH_DAYS_CHOICES},
            "engine_cap": settings.get("clip_engine.daily_usd", 5.0),
            "inbox_policy": inbox.load_policy(session),
            "inbox_counts": inbox.counts(session),
            "inbox_recent": inbox.recent(session, limit=15),
            "inbox_choices": {"per_day": inbox.PER_DAY_CHOICES,
                              "min_score": inbox.MIN_SCORE_CHOICES,
                              "per_video": inbox.PER_VIDEO_CHOICES},
            "archives_here": env("ARCHIVE_ON_THIS_MACHINE", "on").strip().lower()
            not in ("0", "off", "false", "no"),
            "spot": spot,
            "spot_all": spot_all,
            "rejected_since": rejected_since,
            "pending": pending,
            "runs": auto_triage.recent_runs(session),
            "rejected": rejected,
            "rejected_total": auto_triage.count_standing(session),
            "page": page,
            "page_size": PAGE_SIZE,
            "run_filter": run,
            "cutoff_choices": auto_triage.CUTOFF_CHOICES,
            "fit_choices": auto_triage.FIT_CUTOFF_CHOICES,
            "audit_choices": auto_triage.AUDIT_CHOICES,
            "guard": {k: auto_triage.guardrail(settings, k) for k in auto_triage.GUARDRAILS},
            "pct": auto_triage.pct,
            "cutoff_label": auto_triage.cutoff_label,
        }
    return request.app.state.templates.TemplateResponse(
        request, "auto_triage.html", {**ctx, "msg": msg, "active": "auto_triage"})


@router.post("/auto-triage/settings")
def auto_triage_settings(enabled: str = Form(""), signal: str = Form(""),
                         reject_below: str = Form(""), fit_below: str = Form(""),
                         audit_rate: str = Form("")):
    on = enabled.strip().lower() in ("1", "on", "true", "yes")
    try:
        with session_scope() as session:
            before = auto_triage.load_policy(session)
            policy = auto_triage.set_policy(session, enabled=on, signal=signal or None,
                                            reject_below=reject_below or None,
                                            fit_below=fit_below or None,
                                            audit_rate=audit_rate or None)
    except ValueError as exc:
        return _back(str(exc))
    if policy.enabled and not before.enabled:
        what = ("positioning fit" if policy.signal == auto_triage.SIGNAL_FIT
                else "approve chance")
        msg = (f"Automatic triage is on: videos with {what} under "
               f"{auto_triage.cutoff_label(policy.signal, policy.cutoff)} will be rejected at "
               f"the next monitor pass, or now with Run now")
    elif not policy.enabled and before.enabled:
        msg = "Automatic triage is off — nothing more will be rejected automatically"
    else:
        msg = "Saved"
    return _back(msg)


@router.post("/auto-triage/resume")
def auto_triage_resume():
    with session_scope() as session:
        auto_triage.resume(session)
    return _back("Resumed — it rejects again from the next pass, with a fresh spot-check count")


@router.post("/auto-triage/run")
def auto_triage_run(request: Request):
    with session_scope() as session:
        result = auto_triage.run_pass(session, load_settings(), trigger="manual")
    if result["outcome"] == "done":
        spot = (f"; {result['audited']} set aside for a spot-check"
                if result["audited"] else "")
        msg = (f"Rejected {result['rejected']} of {result['considered']} waiting videos{spot}"
               if result["rejected"] else
               f"Nothing to reject — none of the {result['considered']} waiting videos is "
               f"under the cut-off")
    elif result["outcome"] == "off":
        msg = "Automatic triage is off — switch it on first"
    else:
        msg = result["note"] or f"Nothing done ({result['outcome']})"
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse({**result, "msg": msg})
    return _back(msg)


@router.post("/auto-triage/runs/{run_id}/undo")
def auto_triage_undo(run_id: int):
    try:
        with session_scope() as session:
            n = auto_triage.undo_run(session, run_id)
    except LookupError as exc:
        return _back(str(exc))
    return _back(f"Pass #{run_id} undone: {n} videos are back in your queue, "
                 f"and automatic triage won't reject them again")


@router.post("/auto-triage/spot-checks/{action_id}/agree")
def auto_triage_agree(action_id: int):
    """The operator agrees with a spot-checked rejection: an ordinary reject
    decision (flagged as a spot-check verdict); the video stays rejected."""
    with session_scope() as session:
        action = session.get(AutoTriageAction, action_id)
        if action is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        c = session.get(Candidate, action.candidate_pk,
                        options=[selectinload(Candidate.channel)])
        if c is None or c.status != STATUS_REJECTED or action.undone:
            return JSONResponse(
                {"error": "This video is no longer rejected automatically."}, status_code=409)
        if action.audit_verdict:
            return JSONResponse({"ok": True, "already": True})
        auto_triage.record_operator_decision(session, c, "reject")
        state = auto_triage.load_policy(session).state
    _retrain_in_background()
    return JSONResponse({"ok": True, "state": state})


@router.post("/auto-select/settings")
def auto_select_settings(enabled: str = Form(""), per_day: str = Form(""),
                         min_prob: str = Form(""), fit_floor: str = Form(""),
                         explore: str = Form(""), fresh_days: str = Form("")):
    on = enabled.strip().lower() in ("1", "on", "true", "yes")
    try:
        with session_scope() as session:
            before = auto_select.load_policy(session)
            policy = auto_select.set_policy(
                session, enabled=on, per_day=per_day or None, min_prob=min_prob or None,
                fit_floor=fit_floor or None, explore=explore or None,
                fresh_days=fresh_days or None)
    except ValueError as exc:
        return _back(str(exc), "#approval")
    if policy.enabled and not before.enabled:
        msg = (f"Automatic approval is on: up to {policy.per_day} videos a day, spread over "
               f"the day — the home machine downloads them and the clip engine watches them")
    elif before.enabled and not policy.enabled:
        msg = "Automatic approval is off — nothing more will be approved automatically"
    else:
        msg = "Saved"
    return _back(msg, "#approval")


@router.post("/auto-select/run")
def auto_select_run():
    """Approve what today's pace allows now. The downloads start on the home
    machine's next round (within about ten minutes)."""
    with session_scope() as session:
        result = auto_select.run(session, load_settings(), trigger="manual")
    if result["state"] == "off":
        msg = "Automatic approval is off — switch it on first"
    elif result["approved"]:
        msg = (f"Approved {result['approved']} — {result['picked_today'] + result['approved']} "
               f"of today's quota used; downloads start within about ten minutes")
    elif not result["allowed"]:
        msg = "Today's pace is already met — more opens up as the day goes on"
    else:
        msg = "Nothing in the pool clears the floors right now"
    return _back(msg, "#approval")
