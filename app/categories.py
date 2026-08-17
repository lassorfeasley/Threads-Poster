"""Programming-category vocabulary + LLM auto-tagging.

Categories separate the kinds of programming that go out, so the feed gives a
viewer variety instead of a run of the same thing. Every candidate video
carries at most ONE slug: the LLM recommends one (monitor pass or the
``backfill-categories`` command) and the operator can override it by hand.

The vocabulary has two halves. CONFIGURED categories are the genres a brand
programmes around — climate runs news / nature / culture — and live in
settings ``categories.options`` because a different brand needs a different
set. RESERVED categories are built into the app because every brand has them
whatever it covers; they are appended after the configured ones.

``promos`` is the only reserved category today: the brand's own promotional
footage. It is deliberately not a genre, and unlike a genre it cannot be read
off the video — a local-TV segment about solar financing reads as promotional
without being ours. So reserved categories are operator-assigned only: they
are withheld from the vocabulary the auto-tagger sees, and the auto-tagger
will not overwrite one it finds.

Categories now share the variety job with auto-tagged facets (format tags,
shelf life, first-party provenance — see app/placement.py). The category
filters across the library/video/post/cut pages stay until scored placement
ships (``scheduler.placement.mode: scored``) and the format-tag facet has
proven itself in the placement preview; retiring them earlier would remove
the operator's only working lens on the queue.
"""
from __future__ import annotations

import time

from . import llm
from .config import Settings, load_settings
from .models import Candidate, Channel

# Built-in categories, appended after whatever settings.yaml defines. A
# configured entry reusing one of these slugs is dropped so the definition
# here stays authoritative — these have behaviour attached, not just a label.
RESERVED_OPTIONS: list[dict] = [
    {
        "slug": "promos",
        "emoji": "📣",
        "label": "Promos & branded content",
        "description": (
            "First-party promotional footage made by or for the brand itself, "
            "rather than found footage from a monitored channel."
        ),
        "default_shelf_life": "evergreen",
        "reserved": True,
        "auto_tag": False,
    },
]

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
        shelf = str(item.get("default_shelf_life") or "").strip().lower()
        out.append({
            "slug": slug,
            "emoji": str(item.get("emoji") or "").strip(),
            "label": str(item.get("label") or slug).strip(),
            "description": str(item.get("description") or "").strip(),
            # Fallback shelf life for videos in this category when the
            # auto-tagger abstained (see app/placement.py). Empty = evergreen.
            "default_shelf_life": shelf if shelf in ("breaking", "timely", "evergreen") else "",
            "reserved": False,
            "auto_tag": True,
        })
    return out


def _compose(configured: list[dict]) -> list[dict]:
    """Configured categories followed by the reserved ones."""
    reserved_slugs = {opt["slug"] for opt in RESERVED_OPTIONS}
    out = [opt for opt in configured if opt["slug"] not in reserved_slugs]
    out.extend(dict(opt) for opt in RESERVED_OPTIONS)
    return out


def invalidate_categories_cache() -> None:
    global _cache
    _cache = None


def category_options(settings: Settings | None = None) -> list[dict]:
    """The full vocabulary as ``[{slug, emoji, label, description, reserved,
    auto_tag}]`` — configured categories in settings order, then reserved."""
    global _cache
    if settings is not None:
        return _compose(_normalize(settings.get("categories.options")))
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
        return list(_cache[1])
    options = _compose(_normalize(load_settings().get("categories.options")))
    _cache = (now, options)
    return list(options)


def auto_tag_options(settings: Settings | None = None) -> list[dict]:
    """The subset the LLM may pick from. Reserved categories record where
    footage came from, which no title or transcript reveals, so offering them
    to the model only invites confident mislabels."""
    return [opt for opt in category_options(settings) if opt["auto_tag"]]


def category_by_slug(slug: str | None) -> dict | None:
    if not slug:
        return None
    for opt in category_options():
        if opt["slug"] == slug:
            return opt
    return None


def reserved_slugs() -> set[str]:
    """Slugs of the built-in categories. Learning that reads the operator's
    published captions filters on these: reserved content is written to a
    different brief than the feed's own voice."""
    return {opt["slug"] for opt in RESERVED_OPTIONS}


def is_reserved(slug: str | None) -> bool:
    return bool(slug) and slug in reserved_slugs()


def default_shelf_life(slug: str | None) -> str:
    """Fallback shelf life for a category (empty when unknown/unset)."""
    opt = category_by_slug(slug)
    return (opt or {}).get("default_shelf_life", "") or ""


def is_first_party(candidate) -> bool:
    """Provenance: is this the brand's own footage rather than found footage?

    Derived, not tagged: a property of the SOURCE. A channel marked
    ``first_party`` covers every video it carries; the legacy reserved
    ``promos`` category survives as a per-video override so existing rows
    (and one-off brand videos on found-footage channels) keep working.
    """
    if candidate is None:
        return False
    channel = getattr(candidate, "channel", None)
    if channel is not None and getattr(channel, "first_party", False):
        return True
    return is_reserved(candidate.category)


def auto_tag_candidate(candidate: Candidate, settings: Settings | None = None,
                       channel: Channel | None = None) -> dict | None:
    """LLM-recommend a programming category for a candidate and store it on
    the row (``category`` + ``category_rationale``). Returns the
    ``{category, rationale}`` result, or None when the vocabulary is empty,
    the model abstained, or the candidate is already in a reserved category.
    ``channel`` lets the monitor pass supply the channel before the candidate
    row is flushed (its relationship isn't loaded yet).

    A reserved category is a fact the operator asserted, not a guess, so it
    outranks the model here — including under ``backfill-categories --force``,
    which would otherwise quietly relabel every promo as a genre.
    """
    settings = settings or load_settings()
    if is_reserved(candidate.category):
        return None
    options = auto_tag_options(settings)
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
    # Shelf life rides along in the same call; the scheduler resolves through
    # post override -> this -> the category's default_shelf_life -> evergreen.
    if result.get("shelf_life"):
        candidate.shelf_life = result["shelf_life"]
    return result
