"""The transcript suggester (``clips.*``), run without holding a DB session.
Spec §4.13.

``clip_proposals.propose`` reads its inputs, calls the model, refines each
opening and writes the ledger in ONE session. That suits the old synchronous
route, but not a background job: the model call plus the opening-frame
passes take a minute or more, and on Postgres the session would hold a
transaction (and a pooled connection) open the whole time. So the engine's
fallback, and the bench's transcript arm, use ``propose`` in two halves:

- ``suggest`` is the model half, on the plain values ``jobs.prepare_inputs``
  copied out of a session. It opens no session at all.
- ``write`` is the ledger half, for the caller's own short session.

Both mirror ``propose`` argument for argument, so the transcript arm of the
bench measures the suggester that actually runs, not a lookalike. Two
deliberate differences: an opening refinement that blows up keeps the clip's
transcript start instead of throwing away a paid-for suggestion, and an empty
result never reaches ``log_run``, which would otherwise supersede the
operator's pending suggestions with nothing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from .. import clip_proposals, llm, spend
from . import opt

log = logging.getLogger("clip_engine.legacy")


def suggester_model(settings) -> str:
    """The transcript suggester's model: ``clips.model``, else
    ``matching.model``, else Haiku — ``propose``'s chain, with blank values
    treated as unset."""
    return str(opt(settings, "clips.model",
                   opt(settings, "matching.model", "claude-haiku-4-5"))).strip()


def limits(settings, *, window: dict | None = None) -> dict:
    """``propose``'s clip budget, as configured (the raw values feed the
    policy fingerprint, so they are not coerced): fewer clips for a section."""
    return {
        "max_clips": (opt(settings, "clips.section_max_clips", 2) if window
                      else opt(settings, "clips.max_clips", 3)),
        "max_segments": opt(settings, "clips.max_segments", 4),
        "min_seconds": opt(settings, "clips.min_seconds", 15),
        "max_seconds": opt(settings, "clips.max_seconds", 40),
        "opening_hold": opt(settings, "clips.opening_hold_seconds", 3.0),
    }


def policy_for(settings, *, window: dict | None = None) -> str:
    """The ledger fingerprint ``propose`` would write for this pass."""
    lim = limits(settings, window=window)
    return clip_proposals.policy_version(
        model=suggester_model(settings), max_clips=lim["max_clips"],
        max_segments_per_clip=lim["max_segments"], min_seconds=lim["min_seconds"],
        max_seconds=lim["max_seconds"], scope="section" if window else "video",
        prompt=clip_proposals.PROMPT_VERSION)


def _transcript(inputs: dict, candidate: dict) -> list[dict]:
    if "transcript" in inputs:
        return list(inputs.get("transcript") or [])
    return clip_proposals.load_transcript(
        SimpleNamespace(transcript_path=candidate.get("transcript_path") or ""))


def _words(inputs: dict, candidate: dict) -> list[dict]:
    if "words" in inputs:
        return list(inputs.get("words") or [])
    return clip_proposals.load_words(
        SimpleNamespace(word_transcript_path=candidate.get("word_transcript_path") or ""))


def _video_path(inputs: dict, candidate: dict) -> str:
    """The file the opening refinement samples frames from: ``inputs["source"]``
    (the bench's in-flight hardlink) when given, else the candidate's own path
    (the jobs fallback: ``ensure_local`` restores the source to that path).
    "" when neither is on disk, so ``refine_opening`` returns at once."""
    for raw in (inputs.get("source"), candidate.get("local_video_path")):
        if raw and Path(raw).is_file():
            return str(raw)
    return ""


def _refine_openings(clips: list[dict], target: SimpleNamespace, settings,
                     used: list[dict]) -> None:
    """``propose``'s opening pass: each clip's start may move earlier onto a
    stronger frame, never into claimed material — and sibling clips from
    this run count as claimed, so two proposals can't reach back onto the
    same lead-in footage."""
    blocked = list(used)
    for clip in clips:
        try:
            clip["segments"] = clip_proposals.refine_opening(
                target, clip["segments"], settings, blocked, story=clip.get("story", ""))
        except Exception as exc:
            log.warning("Opening refinement failed for %s: %s", target.video_id, exc)
        blocked = blocked + list(clip["segments"])


def suggest(*, settings, inputs: dict, window: dict | None = None, topic: str = "",
            context: dict | None = None) -> dict:
    """The transcript suggester on plain inputs. Opens no DB session.

    ``inputs`` is ``jobs.prepare_inputs``' dict: ``candidate`` (plain fields),
    ``used``, ``context``, ``striking``, and optionally ``transcript`` /
    ``words`` (else read from the candidate's files) and ``source`` (a local
    video path for the opening pass). ``context`` overrides
    ``inputs["context"]`` (the bench passes its replay context).

    Returns ``{"clips", "model", "policy", "cost_usd"}``; ``cost_usd`` is what
    this thread spent on the model calls. Model failures propagate, as in
    ``propose``.
    """
    candidate = dict(inputs.get("candidate") or {})
    model = suggester_model(settings)
    lim = limits(settings, window=window)
    out = {"clips": [], "model": model, "policy": policy_for(settings, window=window),
           "cost_usd": 0.0}
    transcript = _transcript(inputs, candidate)
    if not transcript:
        log.info("No transcript for %s; nothing to suggest from", candidate.get("video_id"))
        return out
    used = list(inputs.get("used") or [])
    target = SimpleNamespace(local_video_path=_video_path(inputs, candidate),
                             title=candidate.get("title") or "",
                             video_id=candidate.get("video_id") or "")
    with spend.metered() as meter:
        clips = llm.suggest_clips(
            model, candidate.get("title") or "", transcript,
            max_clips=lim["max_clips"], max_segments_per_clip=lim["max_segments"],
            min_seconds=lim["min_seconds"], max_seconds=lim["max_seconds"],
            opening_hold=lim["opening_hold"],
            used_ranges=used,
            words=_words(inputs, candidate),
            description=candidate.get("description") or "",
            window=window, topic=topic,
            context=context if context is not None else (inputs.get("context") or {}),
            visual_moments=list(inputs.get("striking") or []),
        )
        clips = [c for c in (clips or []) if isinstance(c, dict)]
        _refine_openings(clips, target, settings, used)
    out["clips"] = clips
    out["cost_usd"] = round(float(meter.total or 0.0), 6)
    return out


def _stories(clips: list[dict]) -> int:
    """How many stories the clips tell: distinct ``story_key``s when every
    clip carries one (engine picks: variants of one story are one story),
    else one per clip (the transcript suggester's partition)."""
    keys = [" ".join(str(c.get("story_key") or "").lower().split()) for c in clips]
    if keys and all(keys):
        return len(set(keys))
    return len(clips)


def write(session, candidate, clips: list[dict], *, model: str, policy: str,
          window: dict | None = None) -> list[int]:
    """``propose``'s ledger half, in the caller's (short) session: log the
    run, then the whole-video side effects — the caption seed, and the
    multi-clip marker unless the operator set it by hand.

    Clips with no segments are dropped first, and nothing left means nothing
    written: ``log_run`` would otherwise supersede the pending suggestions
    with an empty run. Returns the new proposal ids."""
    clips = [c for c in (clips or []) if isinstance(c, dict) and c.get("segments")]
    if not clips:
        return []
    ids = clip_proposals.log_run(session, candidate.id, clips, model=str(model)[:60],
                                 policy=str(policy)[:40], window=window)
    # As in ``propose``: a run that wrote nothing (every segment cleaned away)
    # left the standing suggestions in place, so the markers that describe
    # them stay as they were too.
    if window or not ids:
        return ids
    if clips[0].get("draft_caption"):
        candidate.draft_caption = clips[0]["draft_caption"]
    # Mirror the partition onto the operator's multi-clip marker, but never
    # overwrite a marker they set by hand — only one a suggester set before.
    if not candidate.multi_clip_potential or candidate.multi_clip_auto:
        auto = _stories(clips) > 1
        candidate.multi_clip_potential = auto
        candidate.multi_clip_auto = auto
    return ids
