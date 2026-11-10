"""Operator labels on posted clips, and the reminder to put them to use.

The labels are a small, slowly-growing eval set: the operator's own one-tap
call on whether a clip is visually gripping — would it stop a scroll? When
video models are good enough to judge that, this set is how to tell: run the
model over these clips and see how often it agrees. Until then it costs a
second per clip and nothing else.

The reminder points at ``docs/six-month-review.md``: on the review date the
notifications bell lights up until the operator marks the review done.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select

from .models import AppToken, ContentLabel, Cut, ThreadsPost, utcnow

log = logging.getLogger("labels")

VALUES = ("yes", "no")
# Labels needed before a model comparison means anything — and a reminder of
# what "enough" looks like on the labelling page.
TARGET = 30

REVIEW_NOTE = "docs/six-month-review.md"
DEFAULT_REVIEW_DATE = "2027-05-09"
_DONE_TOKEN = "_reminder:model_review_done"


def set_label(session, cut_pk: int, value: str,
              label: str = ContentLabel.LABEL_GRIPPING) -> bool:
    """Record (or overwrite, or with ``value == ""`` clear) one label."""
    if value and value not in VALUES:
        return False
    if session.get(Cut, cut_pk) is None:
        return False
    row = session.execute(
        select(ContentLabel).where(ContentLabel.cut_pk == cut_pk,
                                   ContentLabel.label == label)
    ).scalar_one_or_none()
    if not value:
        if row is not None:
            session.delete(row)
        return True
    now = utcnow()
    if row is None:
        session.add(ContentLabel(cut_pk=cut_pk, label=label, value=value,
                                 created_at=now, updated_at=now))
    else:
        row.value, row.updated_at = value, now
    return True


def counts(session, label: str = ContentLabel.LABEL_GRIPPING) -> dict:
    rows = session.execute(
        select(ContentLabel.value).where(ContentLabel.label == label)
    ).scalars().all()
    return {"yes": rows.count("yes"), "no": rows.count("no"), "total": len(rows)}


def queue(session, *, limit: int = 60,
          label: str = ContentLabel.LABEL_GRIPPING) -> list[dict]:
    """Published clips to label, unlabelled first, newest first within each.

    One entry per clip (a rerun is the same footage), carrying the post it
    aired as so the page can play and link it, and its latest views — shown
    AFTER labelling would be ideal, but views on the card are useful context
    for spotting which hits and flops are still unlabelled.
    """
    from .analytics import latest_metrics_bulk

    posts = session.execute(
        select(ThreadsPost)
        .where(ThreadsPost.status == "published", ThreadsPost.cut_pk.is_not(None))
        .order_by(ThreadsPost.published_at.desc())
        .limit(limit * 4)
    ).scalars().all()
    by_cut: dict[int, ThreadsPost] = {}
    for p in posts:
        # The first (newest) airing stands in for the clip; prefer an original
        # over a rerun when both are in range.
        if p.cut_pk not in by_cut or (by_cut[p.cut_pk].repost_of_post_pk and not p.repost_of_post_pk):
            by_cut[p.cut_pk] = p
    labelled = dict(session.execute(
        select(ContentLabel.cut_pk, ContentLabel.value)
        .where(ContentLabel.label == label, ContentLabel.cut_pk.in_(list(by_cut)))
    ).all())
    views = latest_metrics_bulk(session, [p.id for p in by_cut.values()])
    items = [{
        "cut_id": cut_pk,
        "post_id": p.id,
        "permalink": p.permalink,
        "caption": (p.caption or "")[:160],
        "published_at": p.published_at,
        "views": (views.get(p.id) or {}).get("views"),
        "value": labelled.get(cut_pk, ""),
    } for cut_pk, p in by_cut.items()]
    items.sort(key=lambda i: (bool(i["value"]),
                              -(i["published_at"].timestamp() if i["published_at"] else 0)))
    return items[:limit]


# ---- the six-month review reminder -----------------------------------------

def review_date(settings) -> dt.date | None:
    raw = str(settings.get("reminders.model_review_on", DEFAULT_REVIEW_DATE) or "").strip()
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        return None


def review_due(session, settings, today: dt.date | None = None) -> bool:
    """True from the review date until the operator marks the review done."""
    when = review_date(settings)
    if when is None or (today or dt.date.today()) < when:
        return False
    return session.get(AppToken, _DONE_TOKEN) is None


def mark_review_done(session) -> None:
    row = session.get(AppToken, _DONE_TOKEN)
    if row is None:
        session.add(AppToken(name=_DONE_TOKEN, value=utcnow().isoformat(),
                             updated_at=utcnow()))
