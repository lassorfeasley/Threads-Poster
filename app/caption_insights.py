"""Advisory 'learned rules' for caption style.

Reads the operator's own published captions and, via the LLM, distills concrete
EDITORIAL / FORMATTING rules — the composition moves that make their strong posts
work (e.g. "lead with a one-line pull quote", "end with a wry question", "frame
denial viewpoints impartially"). The operator can promote the ones that ring true
into their style guide, or dismiss them.

Performance-aware but advisory: engagement rate = (likes + replies + reposts) /
views (read at a fixed post age) is used to decide WHICH captions to learn from —
the higher-performing ones — with lower performers passed as contrast. When there
aren't enough posts with metrics yet, it falls back to the most recent captions so
suggestions still work early. Nothing is ever auto-applied.

Suggestions are cached (the distillation is an LLM call) and only (re)generated on
explicit request from the Style guide page.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging

from sqlalchemy import select

from . import llm
from .analytics import metrics_at_age_bulk
from .config import load_caption_rules, load_settings
from .models import AppToken, ThreadsPost, utcnow

log = logging.getLogger("caption_insights")

_SUGGEST_TOKEN = "caption_rule_suggestions"     # cached LLM output
_DISMISSED_TOKEN = "caption_insights_dismissed"  # keys the operator hid
_MIN_CAPTIONS = 4          # need at least this many captions to suggest anything
_MIN_FOR_PERFORMANCE = 8   # below this, rank by recency instead of engagement
_STRONG_K = 15
_WEAK_K = 8


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _key_for(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:12]


# ---- dismissed keys ---------------------------------------------------------

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


# ---- caption selection ------------------------------------------------------

def _select_captions(session) -> dict | None:
    """Pick which captions to learn from. Returns {strong, weak, n,
    used_performance} or None when there isn't enough history."""
    settings = load_settings()
    age_hours = int(settings.get("learning.metric_age_hours", 48))

    posts = session.execute(
        select(ThreadsPost).where(
            ThreadsPost.status == "published",
            ThreadsPost.published_at.is_not(None),
            ThreadsPost.caption != "",
        )
    ).scalars().all()
    rows = [{"caption": p.caption.strip(), "post": p, "published_at": p.published_at}
            for p in posts if (p.caption or "").strip()]
    if len(rows) < _MIN_CAPTIONS:
        return None

    views = metrics_at_age_bulk(session, posts, "views", age_hours)
    likes = metrics_at_age_bulk(session, posts, "likes", age_hours)
    replies = metrics_at_age_bulk(session, posts, "replies", age_hours)
    reposts = metrics_at_age_bulk(session, posts, "reposts", age_hours)
    for r in rows:
        pid = r["post"].id
        v = views.get(pid)
        r["eng"] = (
            (float(likes.get(pid, 0) or 0) + float(replies.get(pid, 0) or 0)
             + float(reposts.get(pid, 0) or 0)) / float(v)
        ) if v and v > 0 else None

    scored = [r for r in rows if r["eng"] is not None]
    if len(scored) >= _MIN_FOR_PERFORMANCE:
        scored.sort(key=lambda r: r["eng"], reverse=True)
        strong = [r["caption"] for r in scored[:_STRONG_K]]
        weak = [r["caption"] for r in scored[-_WEAK_K:]] if len(scored) > _STRONG_K else []
        return {"strong": strong, "weak": weak, "n": len(scored), "used_performance": True}

    # Not enough metrics yet — learn from the most recent captions instead.
    rows.sort(key=lambda r: r["published_at"] or dt.datetime.min, reverse=True)
    return {"strong": [r["caption"] for r in rows[:_STRONG_K]], "weak": [],
            "n": len(rows), "used_performance": False}


# ---- suggestions cache ------------------------------------------------------

def _cached(session) -> dict:
    row = session.get(AppToken, _SUGGEST_TOKEN)
    if row and row.value:
        try:
            return json.loads(row.value)
        except ValueError:
            pass
    return {}


def _visible(items: list[dict], session) -> list[dict]:
    """Hide dismissed suggestions and any that duplicate an existing rule."""
    dismissed = _dismissed_keys(session)
    existing = {_norm(r["text"]) for r in load_caption_rules()}
    return [it for it in items
            if it.get("key") not in dismissed and _norm(it.get("rule", "")) not in existing]


def load_suggestions(session) -> list[dict]:
    """Cached suggestions still worth showing (no LLM call)."""
    return _visible(_cached(session).get("items", []), session)


def has_generated(session) -> bool:
    return bool(_cached(session))


def generate_suggestions(session) -> list[dict]:
    """(Re)distill editorial rules from the operator's captions via the LLM and
    cache them. Raises on hard failure so the caller can surface it."""
    picked = _select_captions(session)
    if not picked:
        # Cache the empty state so the UI can explain "not enough posts yet".
        _store(session, [], n=0, used_performance=False)
        return []

    settings = load_settings()
    model = settings.get("voice.model", settings.get("engagement.draft_model", "claude-sonnet-5"))
    existing = [r["text"] for r in load_caption_rules()]
    raw = llm.suggest_caption_rules(model, picked["strong"], picked["weak"], existing)

    items, seen = [], set()
    for r in raw:
        text = (r.get("rule") or "").strip()
        if not text:
            continue
        key = _key_for(text)
        if key in seen:
            continue
        seen.add(key)
        items.append({"key": key, "rule": text, "why": r.get("why", "")})

    _store(session, items, n=picked["n"], used_performance=picked["used_performance"])
    return _visible(items, session)


def _store(session, items: list[dict], *, n: int, used_performance: bool) -> None:
    payload = json.dumps({
        "items": items, "built_from_n": n,
        "used_performance": used_performance, "built_at": utcnow().isoformat(),
    })
    row = session.get(AppToken, _SUGGEST_TOKEN)
    if row is None:
        session.add(AppToken(name=_SUGGEST_TOKEN, value=payload))
    else:
        row.value = payload
        row.updated_at = utcnow()
    session.flush()
