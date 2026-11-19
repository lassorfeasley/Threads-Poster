"""Automatic triage, stage 1: reject the waiting videos that are surest not to
be worth your time, before they reach your queue.

It judges by one of two signals, chosen on the Automatic triage page:

- **approval** — the approve predictor's chance (app/triage_model.py),
  learned from your own approve/reject history. Strong once there are hundreds
  of decisions; nothing to go on in a new workspace.
- **fit** — positioning fit (app/positioning.py): how well the video suits the
  account, judged from the Brand & audience profile. Works from the first
  video, so it's the gate a new customer's workspace starts on.

A pass rejects every judged video under the chosen signal's cut-off. Nothing
is downloaded or posted, and every rejection can be undone, one video at a
time or a whole pass at once.

What it judges: waiting videos the dashboard shows — relevance at or above
``matching.score_threshold``. Under it they're already treated as off topic,
and judging them would fill the spot-checks with junk nobody looks at.

What it never touches: a video you have decided (or decided and then undid),
a video it already rejected once — even if you restored it since — and videos
from first-party channels. It judges only scores that are current: the
approve chance from the newest model, the fit from the current profile.

Where it runs: at the end of every monitor pass (the dashboard's and the one
on GitHub Actions), from "Run now" on the Automatic triage page, and from
``run.py auto-triage run``. The switch, signal, cut-offs and spot-check share
live in the shared database (an ``AppToken`` row), so every runner obeys the
same switch the moment it's flipped. The guardrails are in settings.yaml
(``auto_triage:``).

Keeping it honest:

- **Spot-checks.** A fixed share of its rejections (drawn at random, by
  video id) is set aside for you to judge. Only these verdicts measure what
  it throws away: a video rescued while browsing the rejected list was picked
  by eye, so it says nothing about the rate. It pauses itself when the
  spot-checks answered since it was last switched on, resumed or re-tuned show
  you'd approve clearly more than ``pause_above_rate`` of what it rejects
  (the lower end of a one-sided 95% interval has to clear it, so one early
  approval can't trip it).
- **A sudden swing.** A pass that would reject more than ``max_reject_share``
  of the waiting videos pauses instead; that's what a broken or badly shifted
  signal looks like.
- **A weak model.** Judging by approval, it waits while the model has learned
  from too few decisions or grades below ``min_backtest_auc``.
- **Not training data.** Its calls live in ``auto_triage_actions``, apart
  from ``triage_decisions``, so the model never learns from itself. Your
  spot-check verdicts are ordinary decisions (flagged ``audit``) and are.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass

from sqlalchemy import case, func, or_, select, update
from sqlalchemy.orm import selectinload

from . import positioning, triage_model
from .config import WORKSPACE, Settings, load_settings
from .models import (
    STATUS_NEW,
    STATUS_REJECTED,
    AppToken,
    AutoTriageAction,
    AutoTriageRun,
    Candidate,
    Channel,
    TriageDecision,
    TriageModel,
    utcnow,
)
from .ranking import visible

log = logging.getLogger("auto_triage")

STAGE = "metadata"
POLICY_TOKEN = f"auto_triage:{WORKSPACE}"
if len(POLICY_TOKEN) > 40:  # AppToken.name is String(40)
    raise ValueError(f"workspace slug too long for the automatic-triage switch: {POLICY_TOKEN!r}")

SIGNAL_APPROVAL = "approval"
SIGNAL_FIT = "fit"
SIGNALS = (SIGNAL_APPROVAL, SIGNAL_FIT)

CUTOFF_CHOICES = triage_model.CUTOFF_CHOICES          # approve chance
FIT_CUTOFF_CHOICES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)   # positioning fit
AUDIT_CHOICES = (0.05, 0.10, 0.20)
DEFAULT_CUTOFF = 0.02
DEFAULT_FIT_CUTOFF = 0.4
DEFAULT_AUDIT_RATE = 0.10

# settings.yaml ``auto_triage.*`` defaults.
GUARDRAILS = {
    "min_model_decisions": 500,
    "min_backtest_auc": 0.65,
    "max_reject_share": 0.8,
    "max_reject_share_min_considered": 20,
    "pause_above_rate": 0.05,
    "pause_min_audits": 20,
}

# One-sided 95%: the pause needs the LOWER end of this interval above the
# allowed rate; the page shows the 90% two-sided range it implies.
Z = 1.645
CHUNK = 500


def guardrail(settings: Settings | None, key: str):
    value = (settings or load_settings()).get(f"auto_triage.{key}", GUARDRAILS[key])
    try:
        return type(GUARDRAILS[key])(value)
    except (TypeError, ValueError):
        return GUARDRAILS[key]


def choices(signal: str) -> tuple[float, ...]:
    return FIT_CUTOFF_CHOICES if signal == SIGNAL_FIT else CUTOFF_CHOICES


def _score_column(signal: str):
    return Candidate.fit_score if signal == SIGNAL_FIT else Candidate.approve_prob


# --- The switch ---------------------------------------------------------------------

@dataclass
class Policy:
    enabled: bool = False
    signal: str = SIGNAL_APPROVAL
    reject_below: float = DEFAULT_CUTOFF      # cut-off on the approve chance
    fit_below: float = DEFAULT_FIT_CUTOFF     # cut-off on positioning fit
    audit_rate: float = DEFAULT_AUDIT_RATE
    # Set when a safety check stopped it; cleared by resuming or switching on.
    paused_reason: str = ""
    paused_at: str = ""
    # Spot-checks answered after this moment count toward the pause check:
    # older ones judged a different setting, or led to a pause already seen.
    watch_since: str = ""
    updated_at: str = ""

    @property
    def state(self) -> str:
        if not self.enabled:
            return "off"
        return "paused" if self.paused_reason else "on"

    @property
    def cutoff(self) -> float:
        """The cut-off in force, on the scale of the signal in use."""
        return self.fit_below if self.signal == SIGNAL_FIT else self.reject_below

    @property
    def watch_since_at(self) -> dt.datetime | None:
        return _parse_time(self.watch_since)


def _parse_time(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _choice(value, options: tuple[float, ...]) -> float | None:
    """``value`` as the matching entry of ``options``, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return next((c for c in options if abs(c - number) < 1e-9), None)


def load_policy(session) -> Policy:
    row = session.get(AppToken, POLICY_TOKEN)
    if row is None or not row.value:
        return Policy()
    try:
        data = json.loads(row.value)
    except ValueError:
        log.warning("Unreadable automatic-triage settings; treating it as off")
        return Policy()
    if not isinstance(data, dict):
        return Policy()
    signal = str(data.get("signal") or SIGNAL_APPROVAL)
    return Policy(
        enabled=bool(data.get("enabled", False)),
        signal=signal if signal in SIGNALS else SIGNAL_APPROVAL,
        reject_below=_choice(data.get("reject_below"), CUTOFF_CHOICES) or DEFAULT_CUTOFF,
        fit_below=_choice(data.get("fit_below"), FIT_CUTOFF_CHOICES) or DEFAULT_FIT_CUTOFF,
        audit_rate=_choice(data.get("audit_rate"), AUDIT_CHOICES) or DEFAULT_AUDIT_RATE,
        paused_reason=str(data.get("paused_reason") or ""),
        paused_at=str(data.get("paused_at") or ""),
        watch_since=str(data.get("watch_since") or ""),
        updated_at=str(data.get("updated_at") or ""),
    )


def _save_policy(session, policy: Policy, now: dt.datetime) -> Policy:
    policy.updated_at = now.isoformat()
    row = session.get(AppToken, POLICY_TOKEN)
    if row is None:
        row = AppToken(name=POLICY_TOKEN)
        session.add(row)
    row.value = json.dumps(asdict(policy))
    row.updated_at = now
    return policy


def set_policy(session, *, enabled: bool | None = None, signal: str | None = None,
               reject_below=None, fit_below=None, audit_rate=None,
               now: dt.datetime | None = None) -> Policy:
    """Change the switch, the signal, a cut-off or the spot-check share.

    Switching on clears any pause. Switching on, changing the signal, or
    changing the cut-off in force starts a fresh spot-check window, since
    earlier verdicts judged another setting. Raises ValueError for a signal,
    cut-off or share that isn't on offer.
    """
    now = now or utcnow()
    policy = load_policy(session)
    before = (policy.signal, policy.cutoff)
    if signal is not None:
        if signal not in SIGNALS:
            raise ValueError(f"signal must be one of {', '.join(SIGNALS)}")
        policy.signal = signal
    if reject_below is not None:
        cutoff = _choice(reject_below, CUTOFF_CHOICES)
        if cutoff is None:
            raise ValueError("approve-chance cut-off must be one of "
                             + ", ".join(_pct(c) for c in CUTOFF_CHOICES))
        policy.reject_below = cutoff
    if fit_below is not None:
        cutoff = _choice(fit_below, FIT_CUTOFF_CHOICES)
        if cutoff is None:
            raise ValueError("fit cut-off must be one of "
                             + ", ".join(f"{c:.1f}" for c in FIT_CUTOFF_CHOICES))
        policy.fit_below = cutoff
    if audit_rate is not None:
        rate = _choice(audit_rate, AUDIT_CHOICES)
        if rate is None:
            raise ValueError("spot-check share must be one of "
                             + ", ".join(_pct(c) for c in AUDIT_CHOICES))
        policy.audit_rate = rate
    if (policy.signal, policy.cutoff) != before:
        policy.watch_since = now.isoformat()
    if enabled is not None and bool(enabled) != policy.enabled:
        policy.enabled = bool(enabled)
        policy.paused_reason = policy.paused_at = ""
        if policy.enabled:
            policy.watch_since = now.isoformat()
    return _save_policy(session, policy, now)


def resume(session, now: dt.datetime | None = None) -> Policy:
    """Clear a pause and start a fresh spot-check window."""
    now = now or utcnow()
    policy = load_policy(session)
    policy.paused_reason = policy.paused_at = ""
    policy.watch_since = now.isoformat()
    return _save_policy(session, policy, now)


def _pause(session, policy: Policy, reason: str, now: dt.datetime) -> Policy:
    policy.paused_reason = reason
    policy.paused_at = now.isoformat()
    log.warning("Automatic triage paused: %s", reason)
    return _save_policy(session, policy, now)


# --- Is the signal fit to act on? -----------------------------------------------------

def model_grade(model: TriageModel | None) -> dict:
    """The model's out-of-sample grade: the rolling evaluation when it has one
    (models trained before it existed only have the single backtest)."""
    if model is None:
        return {}
    try:
        backtest = json.loads(model.backtest or "{}")
    except ValueError:
        return {}
    return backtest.get("rolling") or backtest


def readiness(session, settings: Settings | None = None,
              predictor: triage_model.TriagePredictor | None = None,
              signal: str = SIGNAL_APPROVAL) -> tuple[TriageModel | None, str]:
    """``(model, "")`` when the signal is fit to act on, else the model (or
    None) and why it isn't. Fit needs no model: only that it's switched on."""
    settings = settings or load_settings()
    if signal == SIGNAL_FIT:
        if not positioning.enabled(settings):
            return None, "Positioning fit is switched off (positioning.enabled in settings.yaml)."
        return None, ""
    if not triage_model.enabled(settings):
        return None, "The approve predictor is switched off (triage_model.enabled in settings.yaml)."
    predictor = predictor or triage_model.load_predictor(session)
    if predictor is None:
        return None, "No approve predictor has been trained yet."
    model = session.get(TriageModel, predictor.model_pk)
    need = guardrail(settings, "min_model_decisions")
    if (model.n_decisions or 0) < need:
        return model, (f"Model #{model.id} has learned from {model.n_decisions} decisions; "
                       f"it needs {need} before it may reject anything.")
    auc = model_grade(model).get("auc")
    floor = guardrail(settings, "min_backtest_auc")
    if auc is None or auc < floor:
        shown = "no grade" if auc is None else f"{auc:.2f}"
        return model, (f"Model #{model.id} grades {shown} out of sample (AUC); "
                       f"it needs {floor:.2f} before it may reject anything.")
    return model, ""


# --- Which videos a pass may judge ---------------------------------------------------

def _untouched(settings: Settings):
    """Waiting and shown on the dashboard, never decided by the operator, never
    auto-triaged at this stage, and not from a first-party channel."""
    decided = select(TriageDecision.id).where(TriageDecision.candidate_pk == Candidate.id)
    triaged = select(AutoTriageAction.id).where(AutoTriageAction.candidate_pk == Candidate.id,
                                                AutoTriageAction.stage == STAGE)
    first_party = select(Channel.id).where(Channel.id == Candidate.channel_pk,
                                           Channel.first_party.is_(True))
    return (Candidate.status == STATUS_NEW) & visible(settings) & ~decided.exists() \
        & ~triaged.exists() & ~first_party.exists()


def _current(signal: str, key):
    """The video carries a current score for ``signal``: from model ``key``
    (approval) or from profile fingerprint ``key`` (fit)."""
    if signal == SIGNAL_FIT:
        return (Candidate.fit_basis == key) & Candidate.fit_score.is_not(None)
    return (Candidate.approve_model_pk == key) & Candidate.approve_prob.is_not(None)


def _stale(signal: str, key):
    """The opposite of ``_current``, spelled out: ``~_current`` would be NULL —
    and so silently false — for a video never scored at all."""
    if signal == SIGNAL_FIT:
        return (Candidate.fit_basis.is_(None) | (Candidate.fit_basis != key)
                | Candidate.fit_score.is_(None))
    return (Candidate.approve_model_pk.is_(None) | (Candidate.approve_model_pk != key)
            | Candidate.approve_prob.is_(None))


def _signal_key(session, settings: Settings, signal: str):
    """What makes a score current: the newest model's id, or the profile
    fingerprint. None when there's no model yet."""
    if signal == SIGNAL_FIT:
        return positioning.basis(settings)
    predictor = triage_model.load_predictor(session)
    return predictor.model_pk if predictor else None


def _set_status(session, ids: list[int], old: str, new: str) -> set[int]:
    """Move ``ids`` from ``old`` to ``new`` status, atomically per row; returns
    the ids actually moved. A row someone else moved first is left alone."""
    moved: set[int] = set()
    for i in range(0, len(ids), CHUNK):
        moved |= set(session.execute(
            update(Candidate)
            .where(Candidate.id.in_(ids[i:i + CHUNK]), Candidate.status == old)
            .values(status=new)
            .returning(Candidate.id)
            .execution_options(synchronize_session=False)
        ).scalars())
    # The bulk UPDATE bypasses the identity map: refresh any copy the caller's
    # session holds (the monitor's holds every video it just stored).
    for obj in list(session.identity_map.values()):
        if isinstance(obj, Candidate) and obj.id in moved:
            session.expire(obj, ["status"])
    return moved


def _audit_draw(video_id: str) -> float:
    """A fixed uniform draw per video, so re-running a pass can't reshuffle
    which rejections are set aside for a spot-check."""
    digest = hashlib.sha256(f"auto-triage-audit:{video_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def preview(session, settings: Settings | None = None, signal: str = SIGNAL_APPROVAL) -> dict:
    """What a pass would do right now at each of the signal's cut-offs, without
    writing.

    ``waiting``: videos a pass may judge (carrying a current score).
    ``below``: {cut-off: how many of them it would reject}. ``unscored``:
    waiting videos without a current score yet (a pass scores them first)."""
    settings = settings or load_settings()
    options = choices(signal)
    key = _signal_key(session, settings, signal)
    _, why = readiness(session, settings, signal=signal)
    out = {"signal": signal, "key": key, "not_ready": why, "waiting": 0, "unscored": 0,
           "below": {c: 0 for c in options}}
    if key is None:
        return out
    score = _score_column(signal)
    sums = [func.coalesce(func.sum(case((score < c, 1), else_=0)), 0) for c in options]
    row = session.execute(
        select(func.count(Candidate.id), *sums).where(_untouched(settings), _current(signal, key))
    ).one()
    out["waiting"] = int(row[0] or 0)
    out["below"] = {c: int(n or 0) for c, n in zip(options, row[1:])}
    out["unscored"] = int(session.execute(
        select(func.count(Candidate.id)).where(_untouched(settings), _stale(signal, key))
    ).scalar_one() or 0)
    return out


def arrivals(session, settings: Settings | None = None, signal: str = SIGNAL_APPROVAL,
             days: int = 28, now: dt.datetime | None = None) -> dict:
    """Videos shown on the dashboard per day over the last ``days``, and how
    many a day fell under each cut-off (by the score each carries; decided
    videos keep the score they had while waiting)."""
    settings = settings or load_settings()
    now = now or utcnow()
    since = now - dt.timedelta(days=days)
    options = choices(signal)
    score = _score_column(signal)
    sums = [func.coalesce(func.sum(case((score < c, 1), else_=0)), 0) for c in options]
    scored = func.coalesce(func.sum(case((score.is_not(None), 1), else_=0)), 0)
    row = session.execute(
        select(func.count(Candidate.id), scored, *sums)
        .where(Candidate.created_at >= since, visible(settings))
    ).one()
    return {"days": days, "per_day": (row[0] or 0) / days, "scored_per_day": (row[1] or 0) / days,
            "below_per_day": {c: (n or 0) / days for c, n in zip(options, row[2:])}}


# --- A pass ---------------------------------------------------------------------------

def run_pass(session, settings: Settings | None = None, *, trigger: str,
             now: dt.datetime | None = None) -> dict:
    """One pass under the current switch. Writes through ``session``; the
    caller commits. Returns what happened: ``state``, ``outcome`` (done |
    paused | waiting | off), ``considered``, ``rejected``, ``audited``,
    ``note``, ``run_pk``."""
    settings = settings or load_settings()
    now = now or utcnow()
    policy = load_policy(session)
    signal, cutoff = policy.signal, policy.cutoff
    result = {"state": policy.state, "outcome": policy.state, "signal": signal,
              "considered": 0, "rejected": 0, "audited": 0, "note": "", "run_pk": None}
    if policy.state == "off":
        return result
    if policy.state == "paused":
        result["note"] = policy.paused_reason
        return result

    predictor = triage_model.load_predictor(session) if signal == SIGNAL_APPROVAL else None
    _, why = readiness(session, settings, predictor, signal=signal)

    def record(outcome: str, note: str = "", considered: int = 0) -> AutoTriageRun:
        run = AutoTriageRun(stage=STAGE, signal=signal, trigger=trigger[:30],
                            model_pk=predictor.model_pk if predictor else None,
                            reject_below=cutoff, audit_rate=policy.audit_rate,
                            considered=considered, outcome=outcome, note=note,
                            started_at=now)
        session.add(run)
        session.flush()
        result.update(outcome=outcome, note=note, considered=considered, run_pk=run.id)
        return run

    if why:
        record(AutoTriageRun.OUTCOME_WAITING, why)
        return result

    tripped = check_spot_checks(session, settings, policy, now=now)
    if tripped:
        result["state"] = "paused"
        record(AutoTriageRun.OUTCOME_PAUSED, tripped)
        return result

    # Judge every waiting video on the same scale: score any that lack a
    # current score (the monitor only scores what it just stored).
    if signal == SIGNAL_FIT:
        positioning.score_waiting(session, settings)
        key = positioning.basis(settings)
    else:
        triage_model.score_new_candidates(session, predictor, only_missing=True)
        key = predictor.model_pk
    score = _score_column(signal)
    rows = session.execute(
        select(Candidate.id, Candidate.video_id, score, Candidate.approve_prob,
               Candidate.approve_model_pk, Candidate.fit_score, Candidate.fit_reason)
        .where(_untouched(settings), _current(signal, key))
        .order_by(Candidate.id)
    ).all()
    below = [r for r in rows if r[2] < cutoff]

    share_cap = guardrail(settings, "max_reject_share")
    if (len(rows) >= guardrail(settings, "max_reject_share_min_considered")
            and len(below) > share_cap * len(rows)):
        reason = (f"A pass would have rejected {len(below)} of the {len(rows)} waiting videos "
                  f"({len(below) / len(rows):.0%}); the limit is {share_cap:.0%}. A swing like "
                  f"that usually means the signal changed underneath it — check the model's "
                  f"grade, or the Brand & audience profile if it judges by fit, before resuming.")
        _pause(session, policy, reason, now)
        result["state"] = "paused"
        record(AutoTriageRun.OUTCOME_PAUSED, reason, considered=len(rows))
        return result

    run = record(AutoTriageRun.OUTCOME_DONE, considered=len(rows))
    # Claim atomically: a video the operator decided (or another pass
    # rejected) since the read above is no longer "new" and is left alone.
    claimed = _set_status(session, [r[0] for r in below], STATUS_NEW, STATUS_REJECTED)
    rejected = audited = 0
    for pk, vid, _, prob, prob_model, fit, fit_reason in below:
        if pk not in claimed:
            continue
        audit = _audit_draw(vid) < policy.audit_rate
        session.add(AutoTriageAction(
            run_pk=run.id, candidate_pk=pk, video_id=vid, stage=STAGE, action="reject",
            signal=signal, approve_prob=prob, model_pk=prob_model, fit_score=fit,
            fit_reason=fit_reason or "", reject_below=cutoff, audit=audit,
            audit_rate=policy.audit_rate, created_at=now))
        rejected += 1
        audited += int(audit)
    run.rejected, run.audited = rejected, audited
    session.flush()
    result.update(rejected=rejected, audited=audited)
    log.info("Automatic triage (%s): rejected %d of %d waiting videos under %s %s, "
             "%d set aside for a spot-check", trigger, rejected, len(rows), signal,
             cutoff_label(signal, cutoff), audited)
    return result


def undo_run(session, run_pk: int, now: dt.datetime | None = None) -> int:
    """Put a pass's rejections back in the queue. Videos you've since decided
    yourself (approved, or agreed with on a spot-check) stay as you left them.
    Returns how many went back. They won't be rejected automatically again."""
    now = now or utcnow()
    run = session.get(AutoTriageRun, run_pk)
    if run is None:
        raise LookupError(f"no automatic-triage pass #{run_pk}")
    actions = session.execute(
        select(AutoTriageAction).where(AutoTriageAction.run_pk == run_pk,
                                       AutoTriageAction.undone.is_(False))
    ).scalars().all()
    restorable = {a.candidate_pk: a for a in actions
                  if a.audit_verdict != AutoTriageAction.AUDIT_AGREE}
    restored = _set_status(session, list(restorable), STATUS_REJECTED, STATUS_NEW)
    for pk in restored:
        a = restorable[pk]
        a.undone, a.undone_reason, a.undone_at = True, "pass undone", now
    run.undone_at = now
    session.flush()
    return len(restored)


# --- The operator's calls ---------------------------------------------------------------

def _calls(session, candidate_pk: int) -> list[AutoTriageAction]:
    return session.execute(
        select(AutoTriageAction)
        .where(AutoTriageAction.candidate_pk == candidate_pk, AutoTriageAction.stage == STAGE)
        .order_by(AutoTriageAction.id.desc())
    ).scalars().all()


def standing_call(session, candidate_pk: int) -> AutoTriageAction | None:
    """The automatic rejection still in force on this video, if any."""
    return next((a for a in _calls(session, candidate_pk) if not a.undone), None)


def record_operator_decision(session, c: Candidate, action: str,
                             settings: Settings | None = None,
                             now: dt.datetime | None = None) -> TriageDecision:
    """Log the operator's approve/reject — the triage model's training record —
    and settle any automatic call on the video.

    A video drawn for a spot-check gets its verdict from the operator's first
    call on it, wherever it's made. Approving overrides the automatic
    rejection. A spot-check verdict can pause automatic triage on the spot.
    """
    now = now or utcnow()
    calls = _calls(session, c.id)
    spot = next((a for a in calls if a.audit and not a.audit_verdict), None)
    if spot is not None:
        spot.audit_verdict = (AutoTriageAction.AUDIT_APPROVE if action == "approve"
                              else AutoTriageAction.AUDIT_AGREE)
        spot.audited_at = now
    if action == "approve":
        for a in calls:
            if not a.undone:
                a.undone, a.undone_reason, a.undone_at = True, "approved", now
    decision = TriageDecision(
        candidate_pk=c.id,
        video_id=c.video_id,
        action=action,
        relevance_score=c.relevance_score,
        visual_score=c.visual_score,
        visual_traits=c.visual_traits or "",
        predicted_prob=c.approve_prob,
        model_pk=c.approve_model_pk,
        audit=spot is not None,
        decided_at=now,
    )
    session.add(decision)
    if spot is not None:
        session.flush()
        policy = load_policy(session)
        if policy.state == "on":
            check_spot_checks(session, settings or load_settings(), policy, now=now)
    return decision


def record_operator_undo(session, decision: TriageDecision) -> None:
    """The operator took a decision back (triage Undo): a spot-check verdict it
    carried is withdrawn too, and the video waits for a fresh one."""
    if not decision.audit:
        return
    spot = next((a for a in _calls(session, decision.candidate_pk)
                 if a.audit and a.audit_verdict), None)
    if spot is not None:
        spot.audit_verdict, spot.audited_at = "", None
        # Undo puts the video back in the queue, so the rejection no longer
        # stands either way; only the reason changes.
        if spot.undone_reason == "approved":
            spot.undone_reason = "restored"


def record_operator_restore(session, c: Candidate, now: dt.datetime | None = None) -> None:
    """The operator sent an automatically rejected video back to the queue."""
    now = now or utcnow()
    for a in _calls(session, c.id):
        if not a.undone:
            a.undone, a.undone_reason, a.undone_at = True, "restored", now


# --- Spot-checks ----------------------------------------------------------------------

def _wilson(p: float, n: float, z: float = Z) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def spot_check_summary(session, since: dt.datetime | None = None) -> dict:
    """What the answered spot-checks say about the share of its rejections
    you'd have approved.

    Each verdict counts 1/(spot-check share when it was drawn), so the estimate
    stays fair across a change of share. ``low``/``high`` is a 90% range
    (Wilson, on the effective sample size).
    """
    query = select(AutoTriageAction.audit_verdict, AutoTriageAction.audit_rate).where(
        AutoTriageAction.stage == STAGE, AutoTriageAction.audit.is_(True),
        AutoTriageAction.audit_verdict != "")
    if since is not None:
        query = query.where(AutoTriageAction.audited_at >= since)
    rows = session.execute(query).all()
    pending = session.execute(
        select(func.count(AutoTriageAction.id))
        .join(Candidate, Candidate.id == AutoTriageAction.candidate_pk)
        .where(AutoTriageAction.stage == STAGE, AutoTriageAction.audit.is_(True),
               AutoTriageAction.audit_verdict == "", AutoTriageAction.undone.is_(False),
               Candidate.status == STATUS_REJECTED)
    ).scalar_one()
    out = {"answered": len(rows), "approved": 0, "rate": None, "low": None, "high": None,
           "pending": int(pending or 0)}
    if not rows:
        return out
    weights = [1.0 / (rate or DEFAULT_AUDIT_RATE) for _, rate in rows]
    hits = [verdict == AutoTriageAction.AUDIT_APPROVE for verdict, _ in rows]
    total = sum(weights)
    rate = sum(w for w, y in zip(weights, hits) if y) / total
    n_eff = total * total / sum(w * w for w in weights)
    low, high = _wilson(rate, n_eff)
    out.update(approved=sum(hits), rate=rate, low=low, high=high)
    return out


def check_spot_checks(session, settings: Settings, policy: Policy,
                      now: dt.datetime | None = None) -> str:
    """Pause (and say why) when the spot-checks since ``watch_since`` show it
    clearly rejects more than it's allowed to. Returns the reason, or ""."""
    now = now or utcnow()
    since = policy.watch_since_at
    summary = spot_check_summary(session, since)
    need = guardrail(settings, "pause_min_audits")
    ceiling = guardrail(settings, "pause_above_rate")
    if summary["answered"] < need or summary["low"] is None or summary["low"] <= ceiling:
        return ""
    when = f" since {since:%b %-d}" if since else ""
    reason = (f"Spot-checks{when}: you'd have approved {summary['approved']} of "
              f"{summary['answered']} it rejected ({summary['rate']:.0%}, at least "
              f"{summary['low']:.0%} allowing for chance) — more than the {ceiling:.0%} it's "
              f"allowed. Lower the cut-off, or check the model's grade or the Brand & "
              f"audience profile, then resume.")
    _pause(session, policy, reason, now)
    return reason


# --- Reading it back ----------------------------------------------------------------------

def standing_rejections(session, *, run_pk: int | None = None, audit_only: bool = False,
                        limit: int = 50, offset: int = 0) -> list[tuple[AutoTriageAction, Candidate]]:
    """Automatic rejections still in force, newest first."""
    query = (
        select(AutoTriageAction, Candidate)
        .join(Candidate, Candidate.id == AutoTriageAction.candidate_pk)
        .where(AutoTriageAction.stage == STAGE, AutoTriageAction.undone.is_(False),
               Candidate.status == STATUS_REJECTED)
    )
    if run_pk is not None:
        query = query.where(AutoTriageAction.run_pk == run_pk)
    if audit_only:
        query = query.where(AutoTriageAction.audit.is_(True), AutoTriageAction.audit_verdict == "")
    query = (query.options(selectinload(Candidate.channel))
             .order_by(AutoTriageAction.id.desc()).limit(limit).offset(offset))
    return [(a, c) for a, c in session.execute(query).all()]


def count_standing(session, since: dt.datetime | None = None) -> int:
    query = (
        select(func.count(AutoTriageAction.id))
        .join(Candidate, Candidate.id == AutoTriageAction.candidate_pk)
        .where(AutoTriageAction.stage == STAGE, AutoTriageAction.undone.is_(False),
               Candidate.status == STATUS_REJECTED)
    )
    if since is not None:
        query = query.where(AutoTriageAction.created_at >= since)
    return int(session.execute(query).scalar_one() or 0)


def recent_runs(session, limit: int = 15) -> list[AutoTriageRun]:
    return session.execute(
        select(AutoTriageRun).where(AutoTriageRun.stage == STAGE)
        .order_by(AutoTriageRun.id.desc()).limit(limit)
    ).scalars().all()


def fit_grade(session, settings: Settings | None = None) -> dict:
    """Positioning fit graded against what you actually decided: every video
    you've decided that carries a current fit score (``run.py auto-triage
    grade-fit`` scores past decisions). Fit never saw the decisions, so this
    is out of sample by construction. {} until anything is graded."""
    settings = settings or load_settings()
    key = positioning.basis(settings)
    rows = session.execute(
        select(TriageDecision.candidate_pk, TriageDecision.action, Candidate.fit_score)
        .join(Candidate, Candidate.id == TriageDecision.candidate_pk)
        .where(TriageDecision.undone.is_(False), Candidate.fit_basis == key,
               Candidate.fit_score.is_not(None))
        .order_by(TriageDecision.decided_at, TriageDecision.id)
    ).all()
    latest = {pk: (action == "approve", fit) for pk, action, fit in rows}
    if not latest:
        return {}
    posted = triage_model._posted_candidates(session)
    pks = list(latest)
    return triage_model.evaluate([latest[pk][0] for pk in pks], [latest[pk][1] for pk in pks],
                                 thresholds=FIT_CUTOFF_CHOICES,
                                 posted=[pk in posted for pk in pks])


def cutoff_table(session, settings: Settings | None = None, signal: str = SIGNAL_APPROVAL,
                 now: dt.datetime | None = None) -> list[dict]:
    """One row per cut-off on offer for ``signal``: how much it takes off your
    plate (videos a day, waiting now) and what it would have cost on your own
    decisions."""
    settings = settings or load_settings()
    if signal == SIGNAL_FIT:
        grade = fit_grade(session, settings)
    else:
        predictor = triage_model.load_predictor(session)
        grade = model_grade(session.get(TriageModel, predictor.model_pk) if predictor else None)
    handoff = {round(h["threshold"], 6): h for h in grade.get("handoff", [])}
    waiting = preview(session, settings, signal)
    flow = arrivals(session, settings, signal, now=now)
    rows = []
    for c in choices(signal):
        h = handoff.get(round(c, 6))
        rows.append({
            "cutoff": c,
            "label": cutoff_label(signal, c),
            "per_day": flow["below_per_day"][c],
            "share_of_arrivals": (flow["below_per_day"][c] / flow["scored_per_day"]
                                  if flow["scored_per_day"] else None),
            "waiting_now": waiting["below"][c],
            "history_n": grade.get("n"),
            "history_auc": grade.get("auc"),
            "history_below": h["below"] if h else None,
            "approvals": grade.get("approved"),
            "approvals_lost": h["approvals_lost"] if h else None,
            "posted": grade.get("posted"),
            "posted_lost": h.get("posted_lost") if h else None,
        })
    return rows


def summary_note(result: dict | None) -> str:
    """A suffix for the monitor's one-line summary, or ""."""
    if not result:
        return ""
    if result.get("outcome") == "paused":
        return " — automatic triage paused"
    if result.get("outcome") == AutoTriageRun.OUTCOME_DONE and result.get("rejected"):
        spot = f", {result['audited']} set aside for a spot-check" if result.get("audited") else ""
        return f" — {result['rejected']} rejected automatically{spot}"
    return ""


def _pct(value: float) -> str:
    text = f"{value * 100:.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


pct = _pct


def cutoff_label(signal: str, value: float) -> str:
    """A cut-off as people read it: "2%" for the approve chance, "0.4" for fit."""
    return f"{value:.1f}" if signal == SIGNAL_FIT else _pct(value)
