"""Programming-category vocabulary + LLM auto-tagging.

The channel balances three kinds of programming (news / nature / culture),
defined in settings ``categories.options`` with an emoji, label and
description each. Every candidate video carries at most ONE category slug:
the LLM recommends one (monitor pass, video-page button, or the
``backfill-categories`` command) and the operator can override it by hand.
"""
from __future__ import annotations

import time

from . import llm
from .config import Settings, load_settings
from .models import Candidate, Channel

# Categories change only via settings.yaml, but templates look them up per
# row — cache briefly to avoid re-reading the YAML on every card render.
_CACHE_TTL_SECONDS = 60.0
_cache: tuple[float, list[dict]] | None = None


def _normalize(raw) -> list[dict]:
    out: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip().lower()
        if not slug:
            continue
        out.append({
            "slug": slug,
            "emoji": str(item.get("emoji") or "").strip(),
            "label": str(item.get("label") or slug).strip(),
            "description": str(item.get("description") or "").strip(),
        })
    return out


def invalidate_categories_cache() -> None:
    global _cache
    _cache = None


def category_options(settings: Settings | None = None) -> list[dict]:
    """The vocabulary as ``[{slug, emoji, label, description}]``, in the
    order defined in settings."""
    global _cache
    if settings is not None:
        return _normalize(settings.get("categories.options"))
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
        return list(_cache[1])
    options = _normalize(load_settings().get("categories.options"))
    _cache = (now, options)
    return list(options)


def category_by_slug(slug: str | None) -> dict | None:
    if not slug:
        return None
    for opt in category_options():
        if opt["slug"] == slug:
            return opt
    return None


def auto_tag_candidate(candidate: Candidate, settings: Settings | None = None,
                       channel: Channel | None = None) -> dict | None:
    """LLM-recommend a programming category for a candidate and store it on
    the row (``category`` + ``category_rationale``). Returns the
    ``{category, rationale}`` result, or None when the vocabulary is empty or
    the model abstained. ``channel`` lets the monitor pass supply the channel
    before the candidate row is flushed (its relationship isn't loaded yet).
    """
    settings = settings or load_settings()
    options = category_options(settings)
    if not options:
        return None
    ch = channel if channel is not None else candidate.channel
    channel_name = (ch.channel_title or ch.call_sign or "") if ch is not None else ""
    result = llm.suggest_category(
        settings.get("categories.model", "claude-haiku-4-5"),
        options,
        title=candidate.title or "",
        description=candidate.description or "",
        channel=channel_name,
        matched_keywords=[k for k in (candidate.matched_keywords or "").split(",") if k],
        transcript_excerpt=(candidate.transcript_text or "")[:2000],
    )
    if not result["category"]:
        return None
    candidate.category = result["category"]
    candidate.category_rationale = result["rationale"]
    return result
