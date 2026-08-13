"""The draft feedback ledger: what the model wrote, what the operator kept.

Covers both drafted texts — the Threads caption and the Instagram Reel hook.
Every draft is written here the moment it's generated, before the operator can
act on it, so a rejection persists as a rejection instead of vanishing.
Accepting or dismissing stamps a verdict; posting resolves the row against the
text that actually shipped.

Consumers:
- ``app/publishing.record_post`` / ``record_instagram_post`` read the proposed
  text back so the stored "suggestion" is the MODEL's draft rather than the
  operator's final text (the bug this module exists to fix).
- ``app/voice.py`` and the Style guide page read the resolved pairs to learn
  length and voice from real keep/reject behaviour.
"""
from __future__ import annotations

import hashlib
import logging
import statistics
from difflib import SequenceMatcher

from sqlalchemy import select, update

from .models import DraftProposal, utcnow

log = logging.getLogger("draft_proposals")

KIND_CAPTION = DraftProposal.KIND_CAPTION
KIND_HOOK = DraftProposal.KIND_HOOK

# At/above this similarity the operator effectively kept what the model wrote.
# Per kind, because the same ratio means very different things at different
# lengths: cutting a 12-word hook down to 8 words still scores ~0.66 on
# character overlap, which is a real rewrite — while 0.66 on a 200-character
# caption genuinely is "kept the draft and tightened it".
_KEPT_SIMILARITY = {KIND_CAPTION: 0.6, KIND_HOOK: 0.85}
_KEPT_DEFAULT = 0.6


def kept_threshold(kind: str) -> float:
    return _KEPT_SIMILARITY.get(kind, _KEPT_DEFAULT)


def _is_kept(row: DraftProposal) -> bool:
    return (row.similarity or 0) >= kept_threshold(row.kind)


def _words(text: str) -> int:
    return len((text or "").split())


def policy_version(*, model: str, max_chars: int, target_words: int | None,
                   style_guide: str, operator_guide: str, examples: int) -> str:
    """Short fingerprint of everything that shaped a draft.

    Prompts here are assembled from moving parts — the distilled style guide
    rebuilds every few captions, and the operator edits rules by hand. Without
    a version stamp, proposals from different regimes are indistinguishable
    later and no prompt change can be evaluated.
    """
    payload = "\x1f".join([
        model or "", str(max_chars), str(target_words or ""),
        style_guide or "", operator_guide or "", str(examples),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def log_proposal(session, cut_pk: int, proposed: str, *, kind: str = KIND_CAPTION,
                 model: str, policy: str, max_chars: int,
                 target_words: int | None, voice_examples: int) -> int:
    """Record a fresh draft and return its id.

    Any earlier draft of the same kind on this cut still awaiting a verdict is
    marked ``superseded`` — the operator redrafted rather than acting on it,
    which is a soft rejection and shouldn't be left looking undecided.
    """
    session.execute(
        update(DraftProposal)
        .where(DraftProposal.cut_pk == cut_pk, DraftProposal.kind == kind,
               DraftProposal.verdict == DraftProposal.VERDICT_PENDING)
        .values(verdict=DraftProposal.VERDICT_SUPERSEDED, decided_at=utcnow())
    )
    row = DraftProposal(
        kind=kind,
        cut_pk=cut_pk,
        proposed=proposed,
        proposed_chars=len(proposed or ""),
        proposed_words=_words(proposed),
        model=model,
        policy_version=policy,
        max_chars=max_chars,
        target_words=target_words,
        voice_examples=voice_examples,
    )
    session.add(row)
    session.flush()
    return row.id


def record_verdict(session, proposal_id: int, verdict: str) -> bool:
    """Stamp the operator's action on the proposal card. Only the first
    decision counts — a later redraft must not rewrite an existing verdict."""
    if verdict not in (DraftProposal.VERDICT_ACCEPTED,
                       DraftProposal.VERDICT_DISMISSED):
        return False
    row = session.get(DraftProposal, proposal_id)
    if row is None or row.verdict != DraftProposal.VERDICT_PENDING:
        return False
    row.verdict = verdict
    row.decided_at = utcnow()
    return True


def latest_for_cut(session, cut_pk: int, kind: str = KIND_CAPTION) -> DraftProposal | None:
    return session.execute(
        select(DraftProposal)
        .where(DraftProposal.cut_pk == cut_pk, DraftProposal.kind == kind)
        .order_by(DraftProposal.created_at.desc(), DraftProposal.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def attach_to_post(session, cut_pk: int | None, final_text: str, *,
                   kind: str = KIND_CAPTION, post_pk: int | None = None,
                   ig_post_pk: int | None = None) -> str:
    """Resolve this cut's newest unattached draft against what actually shipped,
    and return the proposed text for the caller to store alongside the post.

    Returns "" when the operator never asked for a draft — an empty stored
    suggestion correctly reads as "no model involvement" downstream rather than
    as "the model's draft shipped verbatim".
    """
    if not cut_pk:
        return ""
    attached = (DraftProposal.ig_post_pk if kind == KIND_HOOK
                else DraftProposal.post_pk)
    row = session.execute(
        select(DraftProposal)
        .where(DraftProposal.cut_pk == cut_pk, DraftProposal.kind == kind,
               attached.is_(None))
        .order_by(DraftProposal.created_at.desc(), DraftProposal.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        # Every draft of this kind is already tied to an earlier post (a
        # re-queue, or a reel row being refreshed). Reuse the text so the voice
        # signal survives, but leave the ledger alone so one draft isn't
        # counted against two posts.
        previous = latest_for_cut(session, cut_pk, kind)
        return previous.proposed if previous else ""

    final = (final_text or "").strip()
    if kind == KIND_HOOK:
        row.ig_post_pk = ig_post_pk
    else:
        row.post_pk = post_pk
    row.final_text = final
    row.final_chars = len(final)
    row.final_words = _words(final)
    row.similarity = SequenceMatcher(None, (row.proposed or "").strip(), final).ratio()
    if row.verdict == DraftProposal.VERDICT_PENDING:
        # Captions: posted without ever clicking Use/Dismiss, so the operator
        # wrote past the card. Hooks: there is no card at all, the draft lands
        # straight in the field. Either way the diff is the real verdict.
        row.verdict = (DraftProposal.VERDICT_ACCEPTED if _is_kept(row)
                       else DraftProposal.VERDICT_DISMISSED)
        row.decided_at = utcnow()
    return row.proposed or ""


def operator_written(session, kind: str, limit: int = 40) -> list[str]:
    """Resolved drafts the operator substantially rewrote, newest first.

    These are corrections toward how they actually write, which makes them the
    few-shot examples worth showing the model. Used for hooks, which have no
    other source of voice — captions have published ``ThreadsPost`` rows.
    """
    rows = session.execute(
        select(DraftProposal)
        .where(DraftProposal.kind == kind, DraftProposal.similarity.is_not(None),
               DraftProposal.final_text != "")
        .order_by(DraftProposal.created_at.desc(), DraftProposal.id.desc())
        .limit(limit * 3)
    ).scalars().all()
    return [r.final_text for r in rows if not _is_kept(r)][:limit]


# ---- reporting --------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stats(session, kind: str = KIND_CAPTION, recent_n: int = 20) -> dict:
    """Health of the loop for one draft kind, for the Style guide page.

    ``kept_rate`` is the share of drafts the operator used as-is or lightly
    edited. It's the honest measure — clicking "Use this caption" and then
    rewriting it is not an acceptance.
    """
    empty = {"total": 0, "resolved": 0, "dismissed": 0, "kept": 0,
             "kept_rate": None, "recent_n": 0, "recent_kept_rate": None,
             "median_draft_words": None, "median_posted_words": None,
             "policies": 0}
    rows = session.execute(
        select(DraftProposal).where(DraftProposal.kind == kind)
        .order_by(DraftProposal.created_at.desc(), DraftProposal.id.desc())
    ).scalars().all()
    if not rows:
        # Always the full key set: the template checks these individually, and
        # a missing key reads as Undefined rather than as "no data".
        return empty

    resolved = [r for r in rows if r.similarity is not None]
    kept = [r for r in resolved if _is_kept(r)]
    dismissed = [r for r in rows
                 if r.verdict in (DraftProposal.VERDICT_DISMISSED,
                                  DraftProposal.VERDICT_SUPERSEDED)]

    recent = resolved[:recent_n]
    recent_kept = [r for r in recent if _is_kept(r)]

    posted_words = [r.final_words for r in resolved if r.final_words]
    draft_words = [r.proposed_words for r in resolved if r.proposed_words]

    return {
        "total": len(rows),
        "resolved": len(resolved),
        "dismissed": len(dismissed),
        "kept": len(kept),
        "kept_rate": (len(kept) / len(resolved)) if resolved else None,
        "recent_n": len(recent),
        "recent_kept_rate": (len(recent_kept) / len(recent)) if recent else None,
        "median_draft_words": _median(draft_words),
        "median_posted_words": _median(posted_words),
        "policies": len({r.policy_version for r in rows if r.policy_version}),
    }


def recent_pairs(session, kind: str = KIND_CAPTION, limit: int = 12) -> list[dict]:
    """Most recent draft-vs-posted pairs, newest first — the concrete evidence
    behind the summary numbers."""
    rows = session.execute(
        select(DraftProposal)
        .where(DraftProposal.kind == kind, DraftProposal.similarity.is_not(None))
        .order_by(DraftProposal.created_at.desc(), DraftProposal.id.desc())
        .limit(limit)
    ).scalars().all()
    return [{
        "id": r.id,
        "proposed": r.proposed,
        "final": r.final_text,
        "proposed_words": r.proposed_words,
        "final_words": r.final_words,
        "similarity": r.similarity,
        "kept": _is_kept(r),
        "created_at": r.created_at,
    } for r in rows]
