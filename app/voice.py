"""Voice profile: make caption suggestions sound like the operator.

Sources, strongest signal first:
- Captions the operator wrote themselves — imported Threads history
  (``source='threads'``) and app posts where no model draft was involved.
- App captions the operator meaningfully rewrote away from the LLM draft
  (``suggested_caption`` vs. final ``caption``) — every edit is a correction
  toward their voice.
- App captions posted close to the draft — weak signal (that's the model's
  voice, not the operator's), weighted near zero.

Three artifacts feed ``llm.suggest_post_caption``:
- a handful of real example captions (few-shot beats abstract instructions),
- a distilled style guide, cached in ``app_tokens`` and rebuilt every
  ``voice.refresh_every`` new published captions,
- a length target learned from what the operator actually posts lately.

Two failure modes this module has to actively defend against, both of which
silently produced long, generic drafts:

1. Imported history outranking everything. History is hand-written so it earns
   weight 1.0, and it is also the operator's OLDEST material. Ranking purely by
   weight handed every example slot to it, so the model never saw a caption
   written in the app and kept reproducing the older, longer style.
   ``voice.history_example_share`` caps its share of the slots.
2. Length drifting up. The model treats a character ceiling as a target to
   fill, so a fixed ceiling can only ever be an upper bound. ``length_target``
   derives the actual goal from recent operator-written captions instead, which
   means it keeps tightening on its own as the operator keeps writing short.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics
from difflib import SequenceMatcher

from sqlalchemy import or_, select

from . import llm
from .categories import reserved_slugs
from .config import Settings
from .models import AppToken, Candidate, DraftProposal, ThreadsPost, utcnow

log = logging.getLogger("voice")

_STYLE_TOKEN_NAME = "voice_style_guide"

# Weight for app captions of unknown provenance — rows written before the
# caption ledger existed, when ``suggested_caption`` was mistakenly filled with
# the operator's own text. Deliberately mid-range: assume partially theirs.
_LEGACY_WEIGHT = 0.6
# Floor for barely-edited drafts (still operator-approved, so not zero).
_UNEDITED_WEIGHT = 0.15
# At/above this similarity a caption counts as the operator's own words for
# length learning (they wrote it, or rewrote the draft past recognition).
_OWN_VOICE_MIN = 0.6
# Rough English average including the trailing space, used only to keep the
# learned word target consistent with the character ceiling.
_CHARS_PER_WORD = 6


def _voice_weight(post: ThreadsPost, has_proposal: bool) -> float:
    """How much of this caption is the operator's own voice, 0..1.

    ``has_proposal`` says whether a ledger row exists for this post, which is
    what separates a genuine verbatim accept from a pre-ledger row where
    ``suggested_caption`` was wrongly a copy of the operator's own caption.
    Without that distinction every hand-written caption scored as "the model
    got it perfect", i.e. the lowest possible weight.
    """
    if post.source == "threads":
        return 1.0
    draft = (post.suggested_caption or "").strip()
    final = post.caption.strip()
    if not draft:
        # No model draft was recorded: the operator wrote this unaided.
        return 1.0
    if draft == final and not has_proposal:
        return _LEGACY_WEIGHT
    similarity = SequenceMatcher(None, draft, final).ratio()
    return max(_UNEDITED_WEIGHT, 1.0 - similarity)


def collect_voice_captions(session) -> list[dict]:
    """All published organic captions with their voice weight, newest first.

    Reserved categories are left out. Promos recycle their caption verbatim on
    every airing, so a single piece of ad copy would otherwise arrive as dozens
    of separate "hand-written" samples and drag the whole profile toward it.
    """
    posts = session.execute(
        select(ThreadsPost)
        # Outer join: imported Threads history has no candidate, and it's the
        # purest voice sample there is — an inner join would silently drop it.
        .outerjoin(Candidate, ThreadsPost.candidate_pk == Candidate.id)
        .where(
            ThreadsPost.status == "published",
            ThreadsPost.caption != "",
            or_(Candidate.category.is_(None),
                Candidate.category.not_in(reserved_slugs())),
        )
        .order_by(ThreadsPost.published_at.desc().nullslast())
    ).scalars().all()
    ledgered = set(session.execute(
        select(DraftProposal.post_pk).where(DraftProposal.post_pk.is_not(None))
    ).scalars().all())
    return [
        {"caption": p.caption.strip(),
         "weight": _voice_weight(p, p.id in ledgered),
         "imported": p.source == "threads",
         "published_at": p.published_at}
        for p in posts if p.caption.strip()
    ]


def _recency(published_at, half_life_days: float = 180.0) -> float:
    if published_at is None:
        return 0.0
    published = published_at
    if published.tzinfo is None:
        published = published.replace(tzinfo=dt.timezone.utc)
    age_days = max(0.0, (utcnow() - published).total_seconds() / 86400)
    return 0.5 ** (age_days / half_life_days)


def select_examples(captions: list[dict], k: int,
                    history_share: float = 0.5) -> list[str]:
    """Pick up to ``k`` voice examples, newest-and-most-operator-ish first.

    Imported history is capped at ``history_share`` of the slots. It is the
    purest voice sample but also the oldest, and left uncapped it takes every
    slot — which is how the drafter ended up trained exclusively on captions
    written before the operator's current style.

    Returns FEWER than ``k`` rather than topping up past that cap. Early on
    there are only a handful of app-written captions and dozens of imported
    ones, so backfilling would quietly restore the very imbalance the cap
    exists to prevent — right when it matters most. Five examples that show
    the current style beat eight that mostly show the old one. The cap stops
    binding on its own once enough recent captions exist.
    """
    def score(c: dict) -> float:
        return c["weight"] * 2.0 + _recency(c["published_at"]) * 0.5

    ranked = sorted(captions, key=score, reverse=True)
    history = [c for c in ranked if c["imported"]]
    own = [c for c in ranked if not c["imported"]]

    max_history = max(0, int(k * history_share))
    picked: list[str] = []

    def take(pool: list[dict], limit: int) -> None:
        for c in pool:
            if len(picked) >= k or limit <= 0:
                return
            text = c["caption"]
            if any(SequenceMatcher(None, text, p).ratio() > 0.7 for p in picked):
                continue
            picked.append(text)
            limit -= 1

    # The operator's own recent captions first (they carry the current style),
    # then history up to its cap. Own captions may take every slot; history
    # never exceeds its share.
    take(own, k - max_history)
    take(history, max_history)
    take(own, k - len(picked))
    return picked


def length_target(captions: list[dict], settings: Settings) -> int | None:
    """Median word count of the operator's own recent captions, or None when
    there isn't enough of their writing to infer one.

    This is the self-correcting half of the loop: the drafter is told to hit
    the length the operator has actually been posting, so as their captions get
    shorter the instruction tightens automatically — no config edit, no prompt
    rewrite. Only captions that are genuinely their words count, otherwise the
    model's own verbose drafts would feed the target back to itself.
    """
    if not settings.get("voice.learn_length", True):
        return None
    sample_n = int(settings.get("voice.length_sample", 8))
    min_sample = int(settings.get("voice.length_min_sample", 4))
    own = [c for c in captions if c["weight"] >= _OWN_VOICE_MIN]
    if settings.get("voice.length_prefer_app_captions", True):
        # Imported Threads history is hand-written, so it clears the voice bar —
        # but it predates the current posting strategy and is much longer. While
        # it's still in the sample it drags the median up and the target only
        # converges after enough new posts outnumber it. Once there are enough
        # captions written in the app, learn length from those alone.
        app_written = [c for c in own if not c["imported"]]
        if len(app_written) >= min_sample:
            own = app_written
    # captions arrive newest-first; the recent window IS the current preference.
    recent = own[:sample_n]
    if len(recent) < min_sample:
        return None
    words = [len(c["caption"].split()) for c in recent if c["caption"].split()]
    if not words:
        return None
    target = int(round(statistics.median(words)))
    floor = int(settings.get("voice.length_min_words", 3))
    ceiling = int(settings.get("voice.length_max_words", 40))
    # Never ask for more words than the character ceiling can hold, or the
    # drafter gets two instructions that contradict each other. Matters most
    # early on, when the only captions to learn from are imported history —
    # those are long, so the raw median lands above what max_chars allows.
    max_chars = int(settings.get("engagement.caption_max_chars", 220))
    ceiling = min(ceiling, max(floor, max_chars // _CHARS_PER_WORD))
    return max(floor, min(ceiling, target))


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

    # Distill from the operator's most voice-heavy captions, recency included
    # so the guide tracks how they write now rather than how they wrote first.
    sample = [c["caption"] for c in sorted(
        captions,
        key=lambda c: c["weight"] * 2.0 + _recency(c["published_at"]) * 0.5,
        reverse=True,
    )[:30]]
    try:
        text = llm.distill_style_guide(
            settings.get("voice.model", "claude-sonnet-5"), sample)
        _store_guide(session, text, len(captions))
        return text
    except Exception as exc:
        log.warning("Style guide distillation failed (using cached): %s", exc)
        return str(cached.get("text", ""))


def voice_context(session, settings: Settings) -> dict:
    """``{"examples": [...], "style_guide": str, "target_words": int | None}``
    for caption drafting, or empty values when voice matching is disabled or
    there isn't enough history yet.

    The length target is computed even below ``min_captions``: knowing how long
    the operator's captions run is useful well before there's enough material
    to imitate their voice.
    """
    if not settings.get("voice.enabled", True):
        return {"examples": [], "style_guide": "", "target_words": None}
    captions = collect_voice_captions(session)
    target = length_target(captions, settings)
    if len(captions) < int(settings.get("voice.min_captions", 3)):
        return {"examples": [], "style_guide": "", "target_words": target}
    examples = select_examples(
        captions,
        int(settings.get("voice.examples", 8)),
        history_share=float(settings.get("voice.history_example_share", 0.5)),
    )
    guide = get_style_guide(session, settings, captions)
    return {"examples": examples, "style_guide": guide, "target_words": target}
