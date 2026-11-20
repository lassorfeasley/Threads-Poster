"""The inbox: finished clips the system made on its own, waiting for one click.

The last automatic step before you. For videos automatic approval picked
(app/auto_select.py), the clip engine drafts clips and its critic reviews
each. The home machine takes the best drafts that clear the bar, cuts and
renders them, burns in subtitles, drafts a caption in the account's voice
and gut-checks it — then holds the result here. You approve (the clip joins
the posting queue for the next open slot), edit the caption first, or reject
it with the "What's wrong?" chips, which feed the rule proposals like any
dismissed suggestion.

Nothing here posts on its own. ``InboxItem`` is not a post; the post is
created only when you approve, through the same ``queue_clip`` the cut page
uses, so the scheduler, the caption ledger and the first reply all see an
ordinary queued post.

What reaches the inbox (``eligible``):

- a pending clip-engine draft the critic reviewed, scoring at least
  ``min_score`` (its score with the engine's own penalties, 0-1);
- not flagged as challenging the brand rules, and with no topic or language
  doubt from the engine;
- from a video automatic approval picked (``source: auto``), or from any
  downloaded video (``source: all``);
- at most ``per_video`` from one video, and ``per_day`` a day, paced through
  the day like automatic approval.

The switch and its settings live in the shared database (``AppToken``
``inbox:<workspace>``); the drafting runs on the machine that has the files
(``_inbox_tick`` in app/web/main.py).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass

from sqlalchemy import func, select, update

from . import auto_select
from .config import WORKSPACE, Settings, load_settings, scheduler_timezone
from .models import (
    STATUS_ARCHIVED,
    AppToken,
    AutoTriageAction,
    Candidate,
    ClipProposal,
    Cut,
    InboxItem,
    utcnow,
)

log = logging.getLogger("inbox")

POLICY_TOKEN = f"inbox:{WORKSPACE}"
if len(POLICY_TOKEN) > 40:  # AppToken.name is String(40)
    raise ValueError(f"workspace slug too long for the inbox switch: {POLICY_TOKEN!r}")

PER_DAY_CHOICES = (1, 2, 3, 5, 8, 10)
MIN_SCORE_CHOICES = (0.5, 0.6, 0.7, 0.8, 0.9)
PER_VIDEO_CHOICES = (1, 2, 3)
SOURCES = ("auto", "all")
# Drafts older than this aren't made into inbox clips: news goes stale.
FRESH_DAYS = 3
# A drafting that hasn't finished in this long was cut off (a restart).
STALE_DRAFTING = dt.timedelta(minutes=45)
MAX_ATTEMPTS = 2

OPEN_STATES = (InboxItem.STATE_DRAFTING, InboxItem.STATE_READY, InboxItem.STATE_APPROVED)


@dataclass
class Policy:
    enabled: bool = False
    per_day: int = 5
    min_score: float = 0.7
    per_video: int = 2
    source: str = "auto"
    subtitles: bool = True
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
        log.warning("Unreadable inbox settings; treating it as off")
        return Policy()
    if not isinstance(data, dict):
        return Policy()
    d = Policy()
    per_day = _choice(data.get("per_day"), PER_DAY_CHOICES)
    per_video = _choice(data.get("per_video"), PER_VIDEO_CHOICES)
    source = str(data.get("source") or d.source)
    return Policy(
        enabled=bool(data.get("enabled", False)),
        per_day=int(per_day) if per_day is not None else d.per_day,
        min_score=_choice(data.get("min_score"), MIN_SCORE_CHOICES) or d.min_score,
        per_video=int(per_video) if per_video is not None else d.per_video,
        source=source if source in SOURCES else d.source,
        subtitles=bool(data.get("subtitles", d.subtitles)),
        updated_at=str(data.get("updated_at") or ""),
    )


def set_policy(session, *, enabled: bool | None = None, per_day=None, min_score=None,
               per_video=None, source: str | None = None, subtitles: bool | None = None,
               now: dt.datetime | None = None) -> Policy:
    """Change any setting. Raises ValueError for a value that isn't on offer."""
    now = now or utcnow()
    policy = load_policy(session)
    for name, value, options in (("per_day", per_day, PER_DAY_CHOICES),
                                 ("min_score", min_score, MIN_SCORE_CHOICES),
                                 ("per_video", per_video, PER_VIDEO_CHOICES)):
        if value is None:
            continue
        chosen = _choice(value, options)
        if chosen is None:
            raise ValueError(f"{name.replace('_', ' ')} must be one of "
                             + ", ".join(str(o) for o in options))
        setattr(policy, name, chosen if name == "min_score" else int(chosen))
    if source is not None:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        policy.source = source
    if subtitles is not None:
        policy.subtitles = bool(subtitles)
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


# --- What qualifies ------------------------------------------------------------------

def _doubts(proposal_id: int) -> bool:
    """The engine itself doubted the draft's topic or language (its sidecar).
    A sidecar that can't be read raises no doubt."""
    try:
        from .clip_engine import sidecars

        meta = sidecars.load(proposal_id) or {}
    except Exception:
        return False
    return bool(meta.get("topic_doubt") or meta.get("language_doubt"))


def critic_line(proposal_id: int) -> str:
    """The card's one line from the critic ("Stops the scroll: ..."), or ""."""
    try:
        from .clip_engine import sidecars

        critique = (sidecars.load(proposal_id) or {}).get("critique") or {}
    except Exception:
        return ""
    reason = str(critique.get("stop_reason") or critique.get("first_impression") or "").strip()
    return f"Stops the scroll: {reason}" if critique.get("stops_scroll") and reason else reason


def eligible(session, settings: Settings, policy: Policy, now: dt.datetime) -> list[ClipProposal]:
    """Drafts that may become inbox clips, best first (critic score, then the
    video's approve chance)."""
    open_or_done = select(InboxItem.id).where(InboxItem.proposal_pk == ClipProposal.id)
    where = [ClipProposal.verdict == ClipProposal.VERDICT_PENDING,
             ClipProposal.confidence.is_not(None),
             ClipProposal.confidence >= policy.min_score,
             func.coalesce(ClipProposal.flag, "") == "",
             ClipProposal.created_at >= now - dt.timedelta(days=FRESH_DAYS),
             Candidate.status == STATUS_ARCHIVED,
             ~open_or_done.exists()]
    if policy.source == "auto":
        picked = select(AutoTriageAction.id).where(
            AutoTriageAction.candidate_pk == Candidate.id,
            AutoTriageAction.stage == auto_select.STAGE)
        where.append(picked.exists())
    rows = session.execute(
        select(ClipProposal)
        .join(Candidate, Candidate.id == ClipProposal.candidate_pk)
        .where(*where)
        .order_by(ClipProposal.confidence.desc(),
                  func.coalesce(Candidate.approve_prob, 0.0).desc(), ClipProposal.id)
    ).scalars().all()
    taken = dict(session.execute(
        select(InboxItem.candidate_pk, func.count())
        .where(InboxItem.state != InboxItem.STATE_FAILED)
        .group_by(InboxItem.candidate_pk)).all())
    out = []
    for row in rows:
        if taken.get(row.candidate_pk, 0) >= policy.per_video or _doubts(row.id):
            continue
        out.append(row)
        taken[row.candidate_pk] = taken.get(row.candidate_pk, 0) + 1
    return out


def made_since(session, since: dt.datetime) -> int:
    return int(session.execute(
        select(func.count(InboxItem.id))
        .where(InboxItem.created_at >= since, InboxItem.state != InboxItem.STATE_FAILED)
    ).scalar_one() or 0)


def claim(session, settings: Settings | None = None, *, now: dt.datetime | None = None,
          tz=None) -> list[int]:
    """Create ``drafting`` inbox items for what today's pace allows and return
    their ids. The caller drafts them (and commits first)."""
    settings = settings or load_settings()
    now = now or utcnow()
    policy = load_policy(session)
    if not policy.enabled:
        return []
    start, share = auto_select.day_window(now, tz or scheduler_timezone())
    allowed = auto_select.allowed_now(policy, made_since(session, start), share)
    if not allowed:
        return []
    ids = []
    for row in eligible(session, settings, policy, now)[:allowed]:
        item = InboxItem(candidate_pk=row.candidate_pk, proposal_pk=row.id,
                         state=InboxItem.STATE_DRAFTING, score=row.confidence,
                         story=row.story or "", critic_line=critic_line(row.id),
                         created_at=now)
        session.add(item)
        session.flush()
        ids.append(item.id)
    return ids


def stalled(session, now: dt.datetime | None = None) -> list[int]:
    """Items left ``drafting`` by a restart: retried while attempts remain,
    otherwise marked failed. Returns the ids to retry."""
    now = now or utcnow()
    rows = session.execute(
        select(InboxItem).where(InboxItem.state == InboxItem.STATE_DRAFTING,
                                func.coalesce(InboxItem.started_at, InboxItem.created_at)
                                < now - STALE_DRAFTING)
    ).scalars().all()
    retry = []
    for item in rows:
        if (item.attempts or 0) >= MAX_ATTEMPTS:
            item.state, item.error = InboxItem.STATE_FAILED, item.error or "Interrupted too often"
        else:
            retry.append(item.id)
    return retry


# --- Your call ------------------------------------------------------------------------

def chosen_clip(item: InboxItem, cut: Cut) -> str:
    from pathlib import Path

    if item.use_subtitles and cut.subtitled_clip_path and Path(cut.subtitled_clip_path).exists():
        return cut.subtitled_clip_path
    return cut.trimmed_clip_path


def approve(session, item: InboxItem, caption: str, now: dt.datetime | None = None) -> None:
    """Record the operator's approval with the caption as they left it. The
    post is created by ``queue_approved`` where the clip file is."""
    caption = (caption or "").strip()
    if not caption:
        raise ValueError("The caption is empty")
    if item.state != InboxItem.STATE_READY:
        raise ValueError(f"This clip is {item.state}, not waiting for you")
    item.final_caption = caption
    item.state = InboxItem.STATE_APPROVED
    item.decided_at = now or utcnow()


def queue_approved(session, item: InboxItem) -> bool:
    """Queue an approved item's clip for the next open slot, if this machine
    has the file. Returns whether it queued. The same ``queue_clip`` as the
    cut page: the post is ordinary from here on."""
    from pathlib import Path

    from .publishing import queue_clip

    if item.state != InboxItem.STATE_APPROVED or item.cut_pk is None:
        return False
    cut = session.get(Cut, item.cut_pk)
    if cut is None:
        item.state, item.error = InboxItem.STATE_FAILED, "The clip was deleted"
        return False
    path = chosen_clip(item, cut)
    if not path or not Path(path).exists():
        return False   # the home machine's next round queues it
    cut.use_subtitles = path == cut.subtitled_clip_path
    post = queue_clip(session, cut.candidate, path, item.final_caption, cut=cut)
    # What the operator approved — the AI draft stays in the caption ledger.
    cut.draft_caption = item.final_caption
    item.post_pk = post.id
    item.state = InboxItem.STATE_QUEUED
    return True


def reject(session, item: InboxItem, tags: list[str], note: str = "",
           now: dt.datetime | None = None) -> None:
    """Record the operator's rejection. The draft it came from is dismissed in
    the clip ledger (with the reason the chips map to); the caller deletes the
    rendered clip and saves the note."""
    from .clip_engine import notes

    if item.state not in (InboxItem.STATE_READY, InboxItem.STATE_APPROVED):
        raise ValueError(f"This clip is {item.state}")
    tags = notes.clean_tags(tags)
    item.state = InboxItem.STATE_REJECTED
    item.reject_tags = ",".join(tags)[:200]
    item.reject_note = " ".join((note or "").split())[:1000]
    item.decided_at = now or utcnow()
    session.execute(
        update(ClipProposal).where(ClipProposal.id == item.proposal_pk).values(
            verdict=ClipProposal.VERDICT_DISMISSED, cut_pk=None,
            dismiss_reason=notes.dismiss_reason_for(tags) or "other",
            decided_at=item.decided_at)
        .execution_options(synchronize_session=False))


def add_reasons(session, item: InboxItem, tags: list[str], note: str = "") -> tuple[str, str]:
    """Reasons given after a one-click rejection ("what was wrong?"), merged
    into the ones already recorded. The draft's dismiss reason follows the
    first tag when it had none. Returns the (tags, note) just added, cleaned,
    for the caller to save as a clip note."""
    from .clip_engine import notes

    if item.state != InboxItem.STATE_REJECTED:
        raise ValueError(f"This clip is {item.state}, not rejected")
    have = [t for t in (item.reject_tags or "").split(",") if t]
    added = [t for t in notes.clean_tags(tags) if t not in have]
    note = " ".join((note or "").split())[:1000]
    item.reject_tags = ",".join(have + added)[:200]
    if note:
        item.reject_note = (f"{item.reject_note} {note}" if item.reject_note else note)[:1000]
    if added and not have:
        session.execute(
            update(ClipProposal).where(ClipProposal.id == item.proposal_pk).values(
                dismiss_reason=notes.dismiss_reason_for(added) or "other")
            .execution_options(synchronize_session=False))
    return ",".join(added), note


# --- Reading it back ---------------------------------------------------------------------

def ready_items(session, limit: int = 50) -> list[InboxItem]:
    """What's waiting for you, best first."""
    return session.execute(
        select(InboxItem).where(InboxItem.state == InboxItem.STATE_READY)
        .order_by(InboxItem.score.desc().nullslast(), InboxItem.ready_at).limit(limit)
    ).scalars().all()


def counts(session, since: dt.datetime | None = None) -> dict:
    query = select(InboxItem.state, func.count()).group_by(InboxItem.state)
    if since is not None:
        query = query.where(InboxItem.created_at >= since)
    return dict(session.execute(query).all())


def recent(session, limit: int = 30) -> list[InboxItem]:
    return session.execute(
        select(InboxItem).where(InboxItem.state != InboxItem.STATE_READY)
        .order_by(InboxItem.id.desc()).limit(limit)
    ).scalars().all()
