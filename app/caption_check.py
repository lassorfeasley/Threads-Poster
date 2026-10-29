"""Gut check a caption against its clip before it's queued.

Two callers, two policies:
- Re-airs (``scheduler``): captions are drafted with no one watching, so a
  flagged draft is redrafted and, failing that, the clip re-airs under the
  caption the operator vetted on its first airing.
- The cut page: the operator is the final judge, so a flag is a warning they
  can override.

The reference for both is the FIRST airing's caption: the operator vets those
personally. A re-air's own caption is machine-written, so a chain of re-airs
must not anchor to the previous one.

Every check is logged to ``caption_checks``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select

from .config import load_settings
from .models import CaptionCheck, Cut, ThreadsPost

log = logging.getLogger("caption_check")


@dataclass
class Verdict:
    """``ok`` is None when the check itself couldn't run (``error`` says why)."""
    ok: bool | None
    reason: str = ""
    error: str = ""


def check_model() -> str:
    settings = load_settings()
    return str(settings.get("engagement.caption_check_model", "")
               or settings.get("engagement.draft_model", "claude-sonnet-5"))


def first_airing(session, post: ThreadsPost) -> ThreadsPost:
    """The original airing ``post`` descends from through re-airs (itself
    when it isn't a re-air)."""
    seen = {post.id}
    while post.repost_of_post_pk and post.repost_of_post_pk not in seen:
        parent = session.get(ThreadsPost, post.repost_of_post_pk)
        if parent is None:
            break
        seen.add(parent.id)
        post = parent
    return post


def vetted_airing(session, cut: Cut | None) -> ThreadsPost | None:
    """The cut's earliest published airing that wasn't a re-air, or None for a
    clip that has never aired."""
    if cut is None:
        return None
    return session.execute(
        select(ThreadsPost)
        .where(ThreadsPost.cut_pk == cut.id,
               ThreadsPost.status == "published",
               ThreadsPost.repost_of_post_pk.is_(None))
        .order_by(ThreadsPost.published_at.asc().nullslast())
        .limit(1)
    ).scalar_one_or_none()


def run(session, cut: Cut | None, caption: str, *, source: str,
        reference: ThreadsPost | None = None, rerun: bool = False) -> Verdict:
    """Check ``caption`` and log the result. Never raises."""
    from . import llm, spend
    from .publishing import _clip_transcript_text

    model = check_model()
    reference_caption = (reference.caption or "").strip() if reference else ""
    cand = cut.candidate if cut is not None else None
    if not spend.within_budget():
        verdict = Verdict(ok=None, error="daily LLM budget reached")
    else:
        try:
            data = llm.check_caption(
                model,
                caption=caption,
                clip_transcript=_clip_transcript_text(cut),
                title=(cand.title or "") if cand is not None else "",
                reference_caption=reference_caption,
                rerun=rerun,
            )
            verdict = Verdict(ok=data["ok"], reason=data["reason"])
        except Exception as exc:
            log.warning("Caption check failed for cut %s: %s",
                        cut.id if cut else None, exc)
            verdict = Verdict(ok=None, error=str(exc)[:500])
    try:
        with session.begin_nested():
            session.add(CaptionCheck(
                source=source,
                cut_pk=cut.id if cut is not None else None,
                reference_post_pk=reference.id if reference is not None else None,
                caption=caption,
                reference=reference_caption,
                passed=verdict.ok is not False,
                reason=verdict.reason,
                error=verdict.error,
                model=model,
            ))
    except Exception as exc:
        log.warning("Couldn't log caption check: %s", exc)
    if verdict.ok is False:
        log.info("Caption flagged (cut %s): %s — %r",
                 cut.id if cut else None, verdict.reason, caption[:120])
    return verdict
