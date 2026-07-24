"""Advisory 'what's working' insights for caption style.

Correlates the structured caption attributes we already store on every post
(question / CTA / hashtags / length / tone) against engagement rate, so the
operator can see which habits move the needle and optionally promote one to a
style rule.

Deliberately ADVISORY only. Engagement is heavily confounded by topic, footage,
and timing, and a single operator's post volume is modest — so we:
  * measure lift against the account's recency-weighted MEDIAN (one viral post
    shouldn't set the bar),
  * gate on a minimum group size, flagging low-sample findings "still learning",
  * never auto-apply anything — promotion to a rule is always the operator's call.

Engagement rate = (likes + replies + reposts) / views, read at a fixed post age
so young and old posts are comparable (same convention as trait learning).
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from sqlalchemy import select

from .analytics import _weighted_median, metrics_at_age_bulk
from .config import load_settings
from .models import AppToken, ThreadsPost, utcnow

log = logging.getLogger("caption_insights")

_DISMISSED_TOKEN = "caption_insights_dismissed"
_MIN_GROUP_N = 5          # need at least this many posts in a group to report it
_MIN_LIFT = 0.10          # ignore patterns that move engagement < 10%
_CONFIDENT_N = 8          # group size for a "confident" (vs. "still learning") flag
_CONFIDENT_TOTAL = 20     # account-wide posts for confidence


def _dismissed_keys(session) -> set[str]:
    row = session.get(AppToken, _DISMISSED_TOKEN)
    if row is None or not row.value:
        return set()
    try:
        data = json.loads(row.value)
    except ValueError:
        return set()
    return set(data) if isinstance(data, list) else set()


def dismiss_insight(session, key: str) -> None:
    keys = _dismissed_keys(session)
    keys.add(key)
    payload = json.dumps(sorted(keys))
    row = session.get(AppToken, _DISMISSED_TOKEN)
    if row is None:
        session.add(AppToken(name=_DISMISSED_TOKEN, value=payload))
    else:
        row.value = payload
        row.updated_at = utcnow()
    session.flush()


def _length_bucket(n: int | None) -> str | None:
    if n is None:
        return None
    if n < 80:
        return "short"
    if n <= 180:
        return "medium"
    return "long"


def compute_insights(session) -> list[dict]:
    """Ranked advisory insights (strongest lift first). Empty when there isn't
    enough data yet. Never raises for the caller's sake — logs and returns []."""
    try:
        return _compute(session)
    except Exception as exc:  # advisory only; never break the page
        log.warning("Caption insight computation failed: %s", exc)
        return []


def _compute(session) -> list[dict]:
    settings = load_settings()
    age_hours = int(settings.get("learning.metric_age_hours", 48))
    halflife_days = float(settings.get("learning.halflife_days", 90))

    posts = session.execute(
        select(ThreadsPost).where(
            ThreadsPost.status == "published",
            ThreadsPost.published_at.is_not(None),
        )
    ).scalars().all()
    if not posts:
        return []

    views = metrics_at_age_bulk(session, posts, "views", age_hours)
    likes = metrics_at_age_bulk(session, posts, "likes", age_hours)
    replies = metrics_at_age_bulk(session, posts, "replies", age_hours)
    reposts = metrics_at_age_bulk(session, posts, "reposts", age_hours)

    now = utcnow()
    obs: list[dict] = []
    for p in posts:
        v = views.get(p.id)
        if not v or v <= 0:
            continue
        eng = (float(likes.get(p.id, 0) or 0)
               + float(replies.get(p.id, 0) or 0)
               + float(reposts.get(p.id, 0) or 0)) / float(v)
        published = p.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=dt.timezone.utc)
        age_days = max(0.0, (now - published).total_seconds() / 86400)
        weight = 0.5 ** (age_days / halflife_days) if halflife_days > 0 else 1.0
        obs.append({
            "eng": eng,
            "weight": weight,
            "has_question": bool(p.caption_has_question),
            "has_cta": bool(p.caption_has_cta),
            "uses_hashtags": (p.caption_hashtag_count or 0) > 0,
            "length": _length_bucket(p.caption_length),
            "tone": (p.caption_tone or "").strip().lower() or None,
        })

    total_n = len(obs)
    if total_n < _MIN_GROUP_N:
        return []
    baseline = _weighted_median([(o["eng"], o["weight"]) for o in obs])
    if not baseline:
        return []

    dismissed = _dismissed_keys(session)
    insights: list[dict] = []

    def consider(key, subset, noun, up_rule, down_rule):
        if key in dismissed:
            return
        n = len(subset)
        if n < _MIN_GROUP_N:
            return
        median = _weighted_median([(o["eng"], o["weight"]) for o in subset])
        if median is None:
            return
        lift = (median - baseline) / baseline
        if abs(lift) < _MIN_LIFT:
            return
        up = lift > 0
        insights.append({
            "key": key,
            "label": f"{noun} {'lifts' if up else 'lowers'} engagement",
            "suggested_rule": up_rule if up else down_rule,
            "lift": lift,
            "lift_pct": round(abs(lift) * 100),
            "direction": "up" if up else "down",
            "n": n,
            "confident": n >= _CONFIDENT_N and total_n >= _CONFIDENT_TOTAL,
        })

    consider("question",
             [o for o in obs if o["has_question"]],
             "Opening with a question",
             "Open with a question when it fits the clip.",
             "Lead with the striking fact, not a question.")
    consider("cta",
             [o for o in obs if o["has_cta"]],
             "Adding a call to action",
             "Close with a light call to action.",
             "Skip the call to action — let the clip stand on its own.")
    consider("hashtags",
             [o for o in obs if o["uses_hashtags"]],
             "Using hashtags",
             "Add a hashtag or two when relevant.",
             "Don't use hashtags.")

    for bucket, noun in (("short", "Short captions (under 80 chars)"),
                         ("medium", "Medium captions (80–180 chars)"),
                         ("long", "Long captions (180+ chars)")):
        consider(f"length:{bucket}",
                 [o for o in obs if o["length"] == bucket],
                 noun,
                 f"Aim for {bucket} captions.",
                 f"Avoid {bucket} captions.")

    tones: dict[str, list[dict]] = {}
    for o in obs:
        if o["tone"]:
            tones.setdefault(o["tone"], []).append(o)
    for tone, subset in tones.items():
        consider(f"tone:{tone}",
                 subset,
                 f"A {tone} tone",
                 f"Lean into a {tone} tone when the story allows.",
                 f"Ease off the {tone} tone.")

    insights.sort(key=lambda i: abs(i["lift"]), reverse=True)
    return insights
