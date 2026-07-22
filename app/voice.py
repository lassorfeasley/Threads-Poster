"""Voice profile: make caption suggestions sound like the operator.

Sources, strongest signal first:
- Captions imported from Threads history (``source='threads'``) — written
  entirely by hand, the purest voice sample.
- App captions the operator meaningfully edited away from the LLM draft
  (``suggested_caption`` vs. final ``caption``) — every edit is a correction
  toward their voice.
- App captions posted close to the draft — weak signal (that's the model's
  voice, not the operator's), weighted near zero.

Two artifacts feed ``llm.suggest_post_caption``:
- a handful of real example captions (few-shot beats abstract instructions),
- a distilled style guide, cached in ``app_tokens`` and rebuilt every
  ``voice.refresh_every`` new published captions.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from difflib import SequenceMatcher

from sqlalchemy import select

from . import llm
from .config import Settings
from .models import AppToken, ThreadsPost, utcnow

log = logging.getLogger("voice")

_STYLE_TOKEN_NAME = "voice_style_guide"

# Weight for app captions published before suggested_caption was recorded:
# unknown provenance, assume partially the operator's.
_LEGACY_WEIGHT = 0.6
# Floor for barely-edited drafts (still operator-approved, so not zero).
_UNEDITED_WEIGHT = 0.15


def _voice_weight(post: ThreadsPost) -> float:
    """How much of this caption is the operator's own voice, 0..1."""
    if post.source == "threads":
        return 1.0
    draft = (post.suggested_caption or "").strip()
    if not draft:
        return _LEGACY_WEIGHT
    similarity = SequenceMatcher(None, draft, post.caption.strip()).ratio()
    return max(_UNEDITED_WEIGHT, 1.0 - similarity)


def collect_voice_captions(session) -> list[dict]:
    """All published captions with their voice weight, newest first."""
    posts = session.execute(
        select(ThreadsPost)
        .where(ThreadsPost.status == "published", ThreadsPost.caption != "")
        .order_by(ThreadsPost.published_at.desc().nullslast())
    ).scalars().all()
    return [
        {"caption": p.caption.strip(), "weight": _voice_weight(p),
         "published_at": p.published_at}
        for p in posts if p.caption.strip()
    ]


def select_examples(captions: list[dict], k: int) -> list[str]:
    """Pick the ``k`` best voice examples: rank by voice weight with a mild
    recency boost, and drop near-duplicates so the examples show range."""
    now = utcnow()

    def score(c: dict) -> float:
        recency = 0.0
        if c["published_at"] is not None:
            published = c["published_at"]
            if published.tzinfo is None:
                published = published.replace(tzinfo=dt.timezone.utc)
            age_days = max(0.0, (now - published).total_seconds() / 86400)
            recency = 0.5 ** (age_days / 180)  # half-weight after ~6 months
        return c["weight"] * 2.0 + recency * 0.5

    picked: list[str] = []
    for c in sorted(captions, key=score, reverse=True):
        text = c["caption"]
        if any(SequenceMatcher(None, text, p).ratio() > 0.7 for p in picked):
            continue
        picked.append(text)
        if len(picked) >= k:
            break
    return picked


def _load_cached_guide(session) -> dict:
    row = session.get(AppToken, _STYLE_TOKEN_NAME)
    if row is None or not row.value:
        return {}
    try:
        return json.loads(row.value)
    except ValueError:
        return {}


def _store_guide(session, text: str, built_from_n: int) -> None:
    payload = json.dumps({"text": text, "built_from_n": built_from_n,
                          "built_at": utcnow().isoformat()})
    row = session.get(AppToken, _STYLE_TOKEN_NAME)
    if row is None:
        session.add(AppToken(name=_STYLE_TOKEN_NAME, value=payload))
    else:
        row.value = payload
        row.updated_at = utcnow()
    session.flush()


def get_style_guide(session, settings: Settings, captions: list[dict]) -> str:
    """Cached distilled style guide, rebuilt after ``voice.refresh_every`` new
    captions. Falls back to the stale cache (or empty) when the LLM call fails."""
    cached = _load_cached_guide(session)
    refresh_every = int(settings.get("voice.refresh_every", 5))
    stale = (not cached
             or len(captions) >= int(cached.get("built_from_n", 0)) + refresh_every)
    if not stale:
        return str(cached.get("text", ""))

    # Distill from the operator's most voice-heavy captions.
    sample = [c["caption"] for c in
              sorted(captions, key=lambda c: c["weight"], reverse=True)[:30]]
    try:
        text = llm.distill_style_guide(
            settings.get("voice.model", "claude-sonnet-5"), sample)
        _store_guide(session, text, len(captions))
        return text
    except Exception as exc:
        log.warning("Style guide distillation failed (using cached): %s", exc)
        return str(cached.get("text", ""))


def voice_context(session, settings: Settings) -> dict:
    """{"examples": [...], "style_guide": str} for caption drafting, or empty
    values when voice matching is disabled or there's not enough history yet."""
    if not settings.get("voice.enabled", True):
        return {"examples": [], "style_guide": ""}
    captions = collect_voice_captions(session)
    if len(captions) < int(settings.get("voice.min_captions", 3)):
        return {"examples": [], "style_guide": ""}
    examples = select_examples(captions, int(settings.get("voice.examples", 8)))
    guide = get_style_guide(session, settings, captions)
    return {"examples": examples, "style_guide": guide}
