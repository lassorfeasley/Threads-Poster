"""Automatic approval: approve the day's most promising videos without waiting
for you, so the clip engine can watch them and draft clips.

This is the first automatic step that acts outward: an approved video gets
downloaded (on the home machine — see ``_home_worker_tick`` in app/web/main.py)
and then watched by the clip engine, which costs money. So it is off until
switched on, capped per day, and never fills its quota with a video under the
floors.

How it picks, each run:

- **The pool.** Waiting videos the dashboard shows, published in the last
  ``fresh_days``, that you've never decided and that automatic triage never
  touched (a video you restored from an automatic rejection is yours to
  decide), not from a first-party channel.
- **The ranking.** The approve predictor's chance when its model is fit to act
  (app/auto_triage.readiness), else positioning fit. A video under
  ``min_prob`` (approve chance) or ``fit_floor`` (positioning fit, when it has
  a current score) is never picked, even to fill the quota.
- **The pace.** ``per_day`` picks spread evenly over the day in the posting
  schedule's time zone: by any hour, at most the matching share of the day's
  quota. A run picks the best of what has arrived so far.
- **Exploring.** A share of picks (``explore``) goes, at random, to a video in
  the pool below the top. That's the only way to learn where the ranking is
  wrong once you're no longer deciding; those picks are flagged.

Picks are recorded in ``auto_triage_actions`` (stage "select"), never as your
decisions, so the approve predictor doesn't learn from them. Whether a pick
was worth it shows up downstream: what the clip engine drafted from it, what
you approved in the inbox, and how the posts performed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass

from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from . import auto_triage, positioning, triage_model
from .config import WORKSPACE, Settings, load_settings, scheduler_timezone
from .models import (
    STATUS_APPROVED,
    STATUS_NEW,
    AppToken,
    AutoTriageAction,
    AutoTriageRun,
    Candidate,
    Channel,
    ClipProposal,
    Cut,
    ThreadsPost,
    TriageDecision,
    utcnow,
)
from .ranking import visible

log = logging.getLogger("auto_select")

STAGE = "select"
POLICY_TOKEN = f"auto_select:{WORKSPACE}"
if len(POLICY_TOKEN) > 40:  # AppToken.name is String(40)
    raise ValueError(f"workspace slug too long for the automatic-approval switch: {POLICY_TOKEN!r}")

PER_DAY_CHOICES = (2, 3, 5, 8, 10, 15, 20)
MIN_PROB_CHOICES = (0.02, 0.05, 0.10, 0.15, 0.20)
FIT_FLOOR_CHOICES = (0.0, 0.2, 0.3, 0.4)
EXPLORE_CHOICES = (0.0, 0.1, 0.2)
FRESH_DAYS_CHOICES = (1, 2, 3, 7)


@dataclass
class Policy:
    enabled: bool = False
    per_day: int = 10
    min_prob: float = 0.05
    fit_floor: float = 0.2
    explore: float = 0.1
    fresh_days: int = 2
    updated_at: str = ""

    @property
    def state(self) -> str:
        return "on" if self.enabled else "off"


def _choice(value, options):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return next((o for o in options if abs(o - number) < 1e-9), None)


def load_policy(session) -> Policy:
    row = session.get(AppToken, POLICY_TOKEN)
    if row is None or not row.value:
        return Policy()
    try:
        data = json.loads(row.value)
    except ValueError:
        log.warning("Unreadable automatic-approval settings; treating it as off")
        return Policy()
    if not isinstance(data, dict):
        return Policy()
    default = Policy()
    pick = lambda key, options: _choice(data.get(key), options)  # noqa: E731
    per_day = pick("per_day", PER_DAY_CHOICES)
    fresh = pick("fresh_days", FRESH_DAYS_CHOICES)
    return Policy(
        enabled=bool(data.get("enabled", False)),
        per_day=int(per_day) if per_day is not None else default.per_day,
        min_prob=pick("min_prob", MIN_PROB_CHOICES) or default.min_prob,
        fit_floor=(pick("fit_floor", FIT_FLOOR_CHOICES)
                   if pick("fit_floor", FIT_FLOOR_CHOICES) is not None else default.fit_floor),
        explore=(pick("explore", EXPLORE_CHOICES)
                 if pick("explore", EXPLORE_CHOICES) is not None else default.explore),
        fresh_days=int(fresh) if fresh is not None else default.fresh_days,
        updated_at=str(data.get("updated_at") or ""),
    )


def set_policy(session, *, enabled: bool | None = None, per_day=None, min_prob=None,
               fit_floor=None, explore=None, fresh_days=None,
               now: dt.datetime | None = None) -> Policy:
    """Change any setting. Raises ValueError for a value that isn't on offer."""
    now = now or utcnow()
    policy = load_policy(session)
    for name, value, options in (("per_day", per_day, PER_DAY_CHOICES),
                                 ("min_prob", min_prob, MIN_PROB_CHOICES),
                                 ("fit_floor", fit_floor, FIT_FLOOR_CHOICES),
                                 ("explore", explore, EXPLORE_CHOICES),
                                 ("fresh_days", fresh_days, FRESH_DAYS_CHOICES)):
        if value is None:
            continue
        chosen = _choice(value, options)
        if chosen is None:
            raise ValueError(f"{name.replace('_', ' ')} must be one of "
                             + ", ".join(str(o) for o in options))
        setattr(policy, name, int(chosen) if name in ("per_day", "fresh_days") else chosen)
    if enabled is not None:
        policy.enabled = bool(enabled)
    policy.updated_at = now.isoformat()
    row = session.get(AppToken, POLICY_TOKEN)
    if row is None:
        row = AppToken(name=POLICY_TOKEN)
        session.add(row)
    row.value = json.dumps(asdict(policy))
    row.updated_at = now
    return policy


# --- The day --------------------------------------------------------------------------

def day_window(now: dt.datetime, tz=None) -> tuple[dt.datetime, float]:
    """(start of today in UTC, share of today elapsed counting the current hour)
    in the posting schedule's time zone."""
    tz = tz or scheduler_timezone()
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    hours = (local - midnight).total_seconds() / 3600
    return midnight.astimezone(dt.timezone.utc), min(1.0, (math.floor(hours) + 1) / 24)


def picked_since(session, since: dt.datetime) -> int:
    return int(session.execute(
        select(func.count(AutoTriageAction.id))
        .where(AutoTriageAction.stage == STAGE, AutoTriageAction.created_at >= since)
    ).scalar_one() or 0)


def allowed_now(policy: Policy, picked_today: int, share_of_day: float) -> int:
    return max(0, math.ceil(policy.per_day * share_of_day) - picked_today)


# --- The pool ----------------------------------------------------------------------------

def _draw(*parts) -> float:
    digest = hashlib.sha256(("auto-select:" + ":".join(map(str, parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def ranking_signal(session, settings: Settings) -> tuple[str, object]:
    """("approval", model pk) when the approve predictor is fit to act on, else
    ("fit", profile fingerprint)."""
    predictor = triage_model.load_predictor(session)
    _, why = auto_triage.readiness(session, settings, predictor)
    if not why and predictor is not None:
        return auto_triage.SIGNAL_APPROVAL, predictor.model_pk
    return auto_triage.SIGNAL_FIT, positioning.basis(settings)


def pool(session, settings: Settings, policy: Policy, now: dt.datetime) -> tuple[str, list]:
    """The videos a run may pick from, best first, with the signal ranking them.
    Each row: (id, video_id, score, approve_prob, approve_model_pk, fit, fit_reason)."""
    signal, key = ranking_signal(session, settings)
    decided = select(TriageDecision.id).where(TriageDecision.candidate_pk == Candidate.id)
    automated = select(AutoTriageAction.id).where(AutoTriageAction.candidate_pk == Candidate.id)
    first_party = select(Channel.id).where(Channel.id == Candidate.channel_pk,
                                           Channel.first_party.is_(True))
    since = now - dt.timedelta(days=policy.fresh_days)
    fresh = func.coalesce(Candidate.published_at, Candidate.created_at) >= since
    where = [Candidate.status == STATUS_NEW, visible(settings), fresh,
             ~decided.exists(), ~automated.exists(), ~first_party.exists()]
    # coalesce: a NULL basis would make the NOT below NULL, silently dropping the row.
    fit_current = (func.coalesce(Candidate.fit_basis, "") == positioning.basis(settings)) & \
        Candidate.fit_score.is_not(None)
    if signal == auto_triage.SIGNAL_APPROVAL:
        where += [Candidate.approve_model_pk == key, Candidate.approve_prob >= policy.min_prob,
                  # A current fit under the floor rules a video out; no fit yet doesn't.
                  ~(fit_current & (Candidate.fit_score < policy.fit_floor))]
        score = Candidate.approve_prob
    else:
        where += [fit_current, Candidate.fit_score >= policy.fit_floor]
        score = Candidate.fit_score
    rows = session.execute(
        select(Candidate.id, Candidate.video_id, score, Candidate.approve_prob,
               Candidate.approve_model_pk, Candidate.fit_score, Candidate.fit_reason)
        .where(*where).order_by(score.desc(), Candidate.id.desc())
    ).all()
    return signal, [tuple(r) for r in rows]


def choose(rows: list, n: int, policy: Policy, day: dt.date, first_slot: int) -> list[tuple]:
    """Pick ``n`` rows: the best, except that each slot is, with probability
    ``policy.explore``, a random pick from below the top ``n``. The draws are
    fixed by the date and the slot number, so a re-run picks the same way.
    Returns (row, explored) pairs."""
    picks: list[tuple] = []
    remaining = list(rows)
    for slot in range(first_slot, first_slot + n):
        if not remaining:
            break
        below_top = remaining[max(1, n - len(picks)):]
        if below_top and _draw(day, slot, "explore") < policy.explore:
            row = below_top[int(_draw(day, slot, "which") * len(below_top))]
            picks.append((row, True))
        else:
            row = remaining[0]
            picks.append((row, False))
        remaining.remove(row)
    return picks


# --- A run -------------------------------------------------------------------------------

def run(session, settings: Settings | None = None, *, trigger: str,
        now: dt.datetime | None = None, tz=None) -> dict:
    """Approve what today's pace allows. Writes through ``session``; the caller
    commits. The downloads themselves start on the home machine."""
    settings = settings or load_settings()
    now = now or utcnow()
    policy = load_policy(session)
    result = {"state": policy.state, "approved": 0, "explored": 0, "allowed": 0,
              "picked_today": 0, "pool": 0, "signal": "", "run_pk": None, "ids": []}
    if not policy.enabled:
        return result
    start, share = day_window(now, tz)
    picked_today = picked_since(session, start)
    allowed = allowed_now(policy, picked_today, share)
    result.update(picked_today=picked_today, allowed=allowed)
    if not allowed:
        return result
    signal, rows = pool(session, settings, policy, now)
    result.update(signal=signal, pool=len(rows))
    if not rows:
        return result
    day = start.astimezone(tz or scheduler_timezone()).date()
    picks = choose(rows, allowed, policy, day, picked_today)
    ids = [row[0] for row, _ in picks]
    claimed = set(session.execute(
        update(Candidate)
        .where(Candidate.id.in_(ids), Candidate.status == STATUS_NEW)
        .values(status=STATUS_APPROVED, approved_at=now)
        .returning(Candidate.id)
        .execution_options(synchronize_session=False)
    ).scalars())
    for obj in list(session.identity_map.values()):
        if isinstance(obj, Candidate) and obj.id in claimed:
            session.expire(obj, ["status", "approved_at"])
    run_row = AutoTriageRun(stage=STAGE, signal=signal, trigger=trigger[:30],
                            considered=len(rows), outcome=AutoTriageRun.OUTCOME_DONE,
                            started_at=now)
    session.add(run_row)
    session.flush()
    approved = explored = 0
    for (pk, vid, _, prob, prob_model, fit, fit_reason), explore in picks:
        if pk not in claimed:
            continue
        session.add(AutoTriageAction(
            run_pk=run_row.id, candidate_pk=pk, video_id=vid, stage=STAGE, action="approve",
            signal=signal, approve_prob=prob, model_pk=prob_model, fit_score=fit,
            fit_reason=fit_reason or "", explore=explore, created_at=now))
        approved += 1
        explored += int(explore)
    run_row.approved = approved
    session.flush()
    result.update(approved=approved, explored=explored, run_pk=run_row.id,
                  ids=[pk for pk in ids if pk in claimed])
    log.info("Automatic approval (%s): approved %d (%d exploring) of %d in the pool; "
             "%d of %d today", trigger, approved, explored, len(rows),
             picked_today + approved, policy.per_day)
    return result


def preview(session, settings: Settings | None = None, now: dt.datetime | None = None,
            tz=None) -> dict:
    """What a run would do now, without writing — and what the rest of today's
    quota would go to if nothing better arrived."""
    settings = settings or load_settings()
    now = now or utcnow()
    policy = load_policy(session)
    start, share = day_window(now, tz)
    picked_today = picked_since(session, start)
    signal, rows = pool(session, settings, policy, now)
    allowed = allowed_now(policy, picked_today, share)
    day = start.astimezone(tz or scheduler_timezone()).date()
    picks = choose(rows, allowed, policy, day, picked_today)
    ids = [row[0] for row, _ in picks]
    cands = {c.id: c for c in session.execute(
        select(Candidate).options(selectinload(Candidate.channel)).where(Candidate.id.in_(ids))
    ).scalars()} if ids else {}
    return {"policy": policy, "signal": signal, "pool": len(rows), "allowed": allowed,
            "picked_today": picked_today, "share_of_day": share,
            "picks": [{"c": cands[row[0]], "score": row[2], "fit": row[5],
                       "fit_reason": row[6] or "", "explore": explore}
                      for row, explore in picks if row[0] in cands]}


def recent_picks(session, limit: int = 30) -> list[dict]:
    """Recent automatic approvals and what came of each: where the download
    stands, what the clip engine drafted, and what got cut or posted."""
    rows = session.execute(
        select(AutoTriageAction, Candidate)
        .join(Candidate, Candidate.id == AutoTriageAction.candidate_pk)
        .options(selectinload(Candidate.channel))
        .where(AutoTriageAction.stage == STAGE)
        .order_by(AutoTriageAction.id.desc()).limit(limit)
    ).all()
    ids = [c.id for _, c in rows]
    def count(col) -> dict:
        if not ids:
            return {}
        return dict(session.execute(
            select(col, func.count()).where(col.in_(ids)).group_by(col)).all())

    drafted = count(ClipProposal.candidate_pk)
    cut = count(Cut.candidate_pk)
    posted = dict(session.execute(
        select(ThreadsPost.candidate_pk, func.count()).where(
            ThreadsPost.candidate_pk.in_(ids), ThreadsPost.status == "published")
        .group_by(ThreadsPost.candidate_pk)).all()) if ids else {}
    return [{"action": a, "c": c, "drafted": drafted.get(c.id, 0), "cuts": cut.get(c.id, 0),
             "posted": posted.get(c.id, 0)} for a, c in rows]


def summary_note(result: dict | None) -> str:
    if not result or not result.get("approved"):
        return ""
    return f" — {result['approved']} approved automatically"
