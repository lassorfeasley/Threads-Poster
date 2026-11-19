"""Positioning fit: how well a video suits THIS account, from the Brand &
audience profile alone.

The approve predictor (app/triage_model.py) learns one operator's taste from
their own decisions, so a new workspace has nothing to go on. Fit needs no
history: one cheap model call per video reads the title and description
against the profile (mission, audience, positioning, the on/off-topic notes
and the rules) and returns a 0-1 score and the deciding factor in a few words
(``llm.score_fit``). It's the first gate a customer's videos pass, and what
the queue is ordered by until an approval model exists.

What gets scored: waiting videos the dashboard shows (relevance at or above
``matching.score_threshold`` — under it they're off topic already, and
scoring them would be spend on junk). New ones at the end of every monitor
pass; older ones whenever automatic triage judges by fit; any video on
request (``run.py auto-triage grade-fit`` scores past decisions to check fit
against them).

``Candidate.fit_basis`` fingerprints the profile, the prompt version and the
model behind a score. Editing the profile makes every score stale, and stale
ones are redone a batch at a time — a stale score is never judged.
"""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select, update

from . import llm
from .config import Settings, load_settings
from .models import STATUS_NEW, Candidate, TriageDecision, utcnow
from .ranking import visible

log = logging.getLogger("positioning")

DEFAULTS = {"enabled": True, "model": "", "workers": 8, "per_pass": 200}


def setting(settings: Settings | None, key: str):
    value = (settings or load_settings()).get(f"positioning.{key}", DEFAULTS[key])
    return DEFAULTS[key] if value is None else value


def enabled(settings: Settings | None = None) -> bool:
    return bool(setting(settings, "enabled"))


def model(settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    return (str(setting(settings, "model") or "").strip()
            or settings.get("matching.model", "claude-haiku-4-5"))


def basis(settings: Settings | None = None) -> str:
    """Fingerprint of everything a fit score depends on."""
    payload = json.dumps({"v": llm.FIT_PROMPT_VERSION, "model": model(settings),
                          "profile": llm.fit_profile()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _needs_score(key: str):
    # Spelled out for NULL: ``fit_basis != key`` alone is NULL (false) there.
    return (Candidate.fit_basis.is_(None) | (Candidate.fit_basis != key)
            | Candidate.fit_score.is_(None))


def _score_rows(session, rows: list[tuple[int, str, str]], settings: Settings,
                key: str) -> dict:
    """Score ``(id, title, description)`` rows in parallel and write them back.
    A failed call leaves that video unscored (and so unjudged); it's retried
    next time."""
    name = model(settings)

    def one(row):
        pk, title, description = row
        try:
            return pk, llm.score_fit(name, title or "", description or "")
        except Exception as exc:
            log.warning("Positioning fit failed for candidate %s: %s", pk, exc)
            return pk, None

    workers = max(1, int(setting(settings, "workers")))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fit") as pool:
        results = list(pool.map(one, rows))
    now = utcnow()
    scored = 0
    for pk, out in results:
        if out is None:
            continue
        session.execute(
            update(Candidate).where(Candidate.id == pk)
            .values(fit_score=round(out["fit"], 3), fit_reason=out["reason"],
                    fit_basis=key, fit_scored_at=now)
            .execution_options(synchronize_session=False))
        scored += 1
    # Refresh any copy the caller's session holds.
    done = {pk for pk, out in results if out is not None}
    for obj in list(session.identity_map.values()):
        if isinstance(obj, Candidate) and obj.id in done:
            session.expire(obj, ["fit_score", "fit_reason", "fit_basis", "fit_scored_at"])
    return {"scored": scored, "failed": len(rows) - scored}


def score_waiting(session, settings: Settings | None = None, limit: int | None = None) -> dict:
    """Score (or re-score, after a profile edit) waiting videos the dashboard
    shows, newest first. Returns ``{scored, failed, left}``."""
    settings = settings or load_settings()
    if not enabled(settings):
        return {"scored": 0, "failed": 0, "left": 0}
    key = basis(settings)
    limit = int(setting(settings, "per_pass") if limit is None else limit)
    todo = (Candidate.status == STATUS_NEW) & visible(settings) & _needs_score(key)
    rows = session.execute(
        select(Candidate.id, Candidate.title, Candidate.description)
        .where(todo).order_by(Candidate.created_at.desc(), Candidate.id.desc()).limit(limit)
    ).all()
    out = _score_rows(session, [tuple(r) for r in rows], settings, key) if rows else \
        {"scored": 0, "failed": 0}
    out["left"] = int(session.execute(select(func.count(Candidate.id)).where(todo)).scalar_one())
    if out["scored"] or out["failed"]:
        log.info("Positioning fit: scored %d waiting videos (%d failed, %d left)",
                 out["scored"], out["failed"], out["left"])
    return out


def score_decided(session, settings: Settings | None = None, limit: int = 300) -> dict:
    """Score videos you've already decided (newest videos first), so fit can
    be graded against what you actually did. Each is a model call: ``limit``
    bounds the spend."""
    settings = settings or load_settings()
    key = basis(settings)
    decided = select(TriageDecision.candidate_pk).where(TriageDecision.undone.is_(False))
    rows = session.execute(
        select(Candidate.id, Candidate.title, Candidate.description)
        .where(Candidate.id.in_(decided), _needs_score(key))
        .order_by(Candidate.id.desc()).limit(limit)
    ).all()
    return _score_rows(session, [tuple(r) for r in rows], settings, key) if rows else \
        {"scored": 0, "failed": 0}
