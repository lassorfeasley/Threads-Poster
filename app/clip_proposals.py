"""The clip feedback ledger: what the model proposed cutting, what got cut.

Written the moment a suggestion pass runs, before the operator can act, so a
rejection persists as a rejection instead of vanishing. Accepting a proposal
links it to the cut it seeded; exporting that cut resolves it against the
segments that actually shipped.

The resolution deliberately produces three numbers rather than one, because
they fail independently and a blended score hides which one is wrong:

- partition: how many stories the model found (``clips_in_run`` vs. the cuts
  the operator actually made from the video)
- compression: how many segments it joined and how long the result ran
- boundaries: ``iou`` plus SIGNED start/end deltas of the clip's overall span

Only the third is a "how close was it" score. The first two are where a
transcript-driven model usually goes wrong: it proposes one continuous take
where the operator would have joined three, and that reads as decent IoU.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path

from sqlalchemy import select, update

from . import llm, vision
from .llm import suggest_clips
from .models import ClipProposal, Cut, utcnow

log = logging.getLogger("clip_proposals")

# At/above this overlap the operator effectively cut what the model proposed.
# Deliberately stricter than the caption threshold: two clips sharing 80% of
# their timeline really are the same clip, while two captions sharing 80% of
# their characters can still be a rewrite.
KEPT_IOU = 0.8

# Below this, an exported cut and a pending proposal have so little in common
# that pinning them together would be an invention — the proposal stays pending
# for a later cut instead, and the miss shows up in the partition numbers.
_MATCH_FLOOR = 0.05

# After already-clipped material is removed from a proposal: slivers shorter
# than this aren't a usable shot, and a remainder under the floor isn't a clip
# worth offering at all.
_MIN_FRAGMENT = 1.5
_MIN_REMAINING = 5.0


# ---- interval math ----------------------------------------------------------

def normalize(segments: list[dict]) -> list[dict]:
    """Sort and merge segments into disjoint chronological intervals.

    Operator segments arrive in LIST order (the exported supercut order, which
    the trim editor lets you drag out of time order) and may overlap, so every
    comparison has to normalize first. Ordering is a real editorial choice but
    it isn't part of "did the model pick the right material", so it's dropped
    here rather than scored.
    """
    windows = []
    for s in segments or []:
        try:
            start, end = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            windows.append({"start": start, "end": end})
    windows.sort(key=lambda w: w["start"])
    merged: list[dict] = []
    for w in windows:
        if merged and w["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(dict(w))
    return merged


def duration(segments: list[dict]) -> float:
    return round(sum(s["end"] - s["start"] for s in normalize(segments)), 2)


def _overlap(a: list[dict], b: list[dict]) -> float:
    total = 0.0
    for x in a:
        for y in b:
            total += max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    return total


def iou(proposed: list[dict], final: list[dict]) -> float:
    """Temporal intersection-over-union of two clips on the source timeline.

    Treats each clip as a set of intervals, so it reads the same whether the
    clip is one window or four.
    """
    a, b = normalize(proposed), normalize(final)
    if not a or not b:
        return 0.0
    inter = _overlap(a, b)
    union = duration(a) + duration(b) - inter
    return round(inter / union, 4) if union > 0 else 0.0


def subtract(segments: list[dict], used: list[dict]) -> list[dict]:
    """Remove already-claimed time from a proposal's windows."""
    used = normalize(used)
    if not used:
        return normalize(segments)
    out: list[dict] = []
    for s in normalize(segments):
        pieces = [dict(s)]
        for u in used:
            nxt: list[dict] = []
            for p in pieces:
                if u["end"] <= p["start"] or u["start"] >= p["end"]:
                    nxt.append(p)
                    continue
                if u["start"] > p["start"]:
                    nxt.append({"start": p["start"], "end": u["start"]})
                if u["end"] < p["end"]:
                    nxt.append({"start": u["end"], "end": p["end"]})
            pieces = nxt
        out.extend(pieces)
    return [p for p in out if p["end"] - p["start"] >= _MIN_FRAGMENT]


def used_ranges(session, candidate_pk: int) -> list[dict]:
    """Time on this video already claimed by a clip.

    Every cut counts, exported or not — segments saved into a cut are a claim
    on that material, which is the same rule the trim editor's dashed overlay
    already draws.
    """
    spans: list[dict] = []
    rows = session.execute(
        select(Cut).where(Cut.candidate_pk == candidate_pk, Cut.trim_segments != "")
    ).scalars().all()
    for cut in rows:
        try:
            spans.extend(json.loads(cut.trim_segments) or [])
        except (ValueError, TypeError):
            continue
    return normalize(spans)


def visible_segments(proposed: list[dict], used: list[dict]) -> tuple[list[dict], bool]:
    """What's left of a proposal once already-clipped material is removed.

    Returns ``(segments, trimmed)``; an empty list means too little survived to
    be worth offering. This runs when proposals are *served*, not when they're
    made, because the archive-time pass predates every cut on the video — a
    proposal goes stale the moment the next clip claims part of it.

    The stored proposal is deliberately left alone. It's the ledger's record of
    what the model actually asked for, and proposing used material is a miss
    that the export-time score should keep reflecting.
    """
    kept = subtract(proposed, used)
    trimmed = duration(kept) < duration(proposed) - 0.05
    if duration(kept) < _MIN_REMAINING:
        return [], True
    return kept, trimmed


def span_deltas(proposed: list[dict], final: list[dict]) -> tuple[float | None, float | None]:
    """Signed shift of the clip's overall start and end, in seconds.

    Positive means the operator moved that boundary LATER than the model
    proposed: a positive start delta says the model came in too early, a
    negative end delta says it ran too long. The direction is the point —
    a consistent bias is a one-line prompt fix in a way a scalar score is not.
    """
    a, b = normalize(proposed), normalize(final)
    if not a or not b:
        return None, None
    return (round(b[0]["start"] - a[0]["start"], 2),
            round(b[-1]["end"] - a[-1]["end"], 2))


# ---- writing ----------------------------------------------------------------

def policy_version(*, model: str, max_clips: int, max_segments_per_clip: int,
                   min_seconds: int, max_seconds: int) -> str:
    """Short fingerprint of the regime that produced a partition, so a year of
    proposals can be split by prompt version instead of blended into one
    unanalyzable pile."""
    payload = "\x1f".join([
        model or "", str(max_clips), str(max_segments_per_clip),
        str(min_seconds), str(max_seconds),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def log_run(session, candidate_pk: int, clips: list[dict], *,
            model: str, policy: str) -> list[int]:
    """Record one suggestion pass and return the new row ids.

    Any earlier proposal on this video still awaiting a verdict is marked
    ``superseded`` — re-rolling is a soft rejection and shouldn't be left
    looking undecided. Already-accepted rows are untouched, so re-rolling a
    multi-story video keeps the clips already cut.
    """
    session.execute(
        update(ClipProposal)
        .where(ClipProposal.candidate_pk == candidate_pk,
               ClipProposal.verdict == ClipProposal.VERDICT_PENDING)
        .values(verdict=ClipProposal.VERDICT_SUPERSEDED, decided_at=utcnow())
    )
    run_id = uuid.uuid4().hex[:16]
    ids: list[int] = []
    for index, clip in enumerate(clips):
        segments = normalize(clip.get("segments") or [])
        if not segments:
            continue
        row = ClipProposal(
            candidate_pk=candidate_pk,
            run_id=run_id,
            clip_index=index,
            clips_in_run=len(clips),
            proposed_segments=json.dumps(segments),
            proposed_segment_count=len(segments),
            proposed_duration_s=duration(segments),
            story=str(clip.get("story", ""))[:200],
            why=str(clip.get("why", ""))[:300],
            confidence=clip.get("confidence"),
            model=model,
            policy_version=policy,
        )
        session.add(row)
        session.flush()
        ids.append(row.id)
    return ids


def _backstop(start: float, blocked: list[dict]) -> float:
    """Earliest time an opening may be pulled back to without entering footage
    another clip has claimed. Returns ``start`` itself when there's no room."""
    floor = 0.0
    for span in blocked:
        if span["end"] <= start:
            floor = max(floor, span["end"])
        elif span["start"] < start:
            return start  # the start sits inside claimed material
    return floor


def refine_opening(candidate, segments: list[dict], settings,
                   blocked: list[dict], story: str = "") -> list[dict]:
    """Pull a clip's start back onto a stronger image, using the actual frames.

    The transcript pass picks a start from the words alone, so it lands on the
    sentence rather than the shot — which is how a clip ends up opening on an
    anchor at a desk. Here the footage around that start is sampled and the
    model picks the frame worth opening on.

    The start only ever moves EARLIER, onto lead-in footage ahead of the
    speech; moving it later would truncate the first sentence. It never crosses
    into material another clip already claims. Any failure — no local file, no
    ffmpeg, no answer — leaves the transcript's start exactly as it was.
    """
    if not segments or not getattr(candidate, "local_video_path", ""):
        return segments
    if not settings.get("clips.opening_vision", True):
        return segments

    lead = float(settings.get("clips.opening_lead_seconds", 4.0))
    hold = float(settings.get("clips.opening_hold_seconds", 3.0))
    interval = float(settings.get("clips.opening_frame_interval", 0.5))
    model = settings.get("clips.opening_model",
                         settings.get("vision.model", "claude-haiku-4-5"))

    first = segments[0]
    start = float(first["start"])
    floor = _backstop(start, normalize(blocked))
    window_start = max(floor, start - lead)
    if start - window_start < interval:
        return segments  # nowhere to move

    # Sample past the start too: the model can only tell whether a shot HOLDS
    # by seeing what follows it.
    window_end = min(start + hold, float(first["end"])) + interval
    frames = vision.frames_between(candidate.local_video_path, window_start,
                                   window_end, interval)
    if len(frames) < 2:
        return segments
    latest = min(range(len(frames)), key=lambda i: abs(frames[i][0] - start))
    if latest <= 0:
        return segments

    try:
        picked = llm.pick_opening_frame(model, frames, latest, hold,
                                        title=candidate.title, story=story)
    except Exception as exc:
        log.info("Opening frame pick failed for %s: %s", candidate.video_id, exc)
        return segments

    index = picked.get("index")
    if index is None or index >= latest:
        return segments
    new_start = max(floor, frames[index][0])
    if new_start >= start - 0.05:
        return segments

    out = [dict(s) for s in segments]
    out[0]["start"] = round(new_start, 2)
    log.info("Opening for %s moved %.1fs -> %.1fs (%s)", candidate.video_id,
             start, new_start, picked.get("why", ""))
    return out


def load_transcript(candidate) -> list[dict]:
    """The video's timestamped transcript, or [] when it has none."""
    if not candidate.transcript_path:
        return []
    try:
        data = json.loads(Path(candidate.transcript_path).read_text())
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def propose(session, candidate, settings, transcript_segments: list[dict]) -> list[int]:
    """Run one suggestion pass for a video and record it. Returns new row ids.

    Callers pass the transcript they already have (the archive pass holds it in
    memory; a re-roll reads it back off disk). Model failures propagate so the
    caller can decide between logging a warning and returning an error — this
    only owns the ledger and the two candidate-level side effects.
    """
    if not transcript_segments:
        return []
    model = settings.get("clips.model",
                         settings.get("matching.model", "claude-haiku-4-5"))
    max_clips = settings.get("clips.max_clips", 3)
    max_segments = settings.get("clips.max_segments", 4)
    min_seconds = settings.get("clips.min_seconds", 15)
    max_seconds = settings.get("clips.max_seconds", 40)
    opening_hold = settings.get("clips.opening_hold_seconds", 3.0)
    claimed = used_ranges(session, candidate.id)

    clips = suggest_clips(
        model, candidate.title, transcript_segments,
        max_clips=max_clips, max_segments_per_clip=max_segments,
        min_seconds=min_seconds, max_seconds=max_seconds,
        opening_hold=opening_hold,
        used_ranges=claimed,
    )

    # Put each opening on a frame worth looking at. Sibling clips from this same
    # run count as blocked too, so two proposals can't reach back onto the same
    # lead-in footage.
    blocked = list(claimed)
    for clip in clips:
        clip["segments"] = refine_opening(candidate, clip["segments"], settings,
                                          blocked, story=clip.get("story", ""))
        blocked = blocked + clip["segments"]

    ids = log_run(session, candidate.id, clips, model=model, policy=policy_version(
        model=model, max_clips=max_clips, max_segments_per_clip=max_segments,
        min_seconds=min_seconds, max_seconds=max_seconds,
    ))

    if clips and clips[0].get("draft_caption"):
        candidate.draft_caption = clips[0]["draft_caption"]

    # Mirror the partition onto the operator's own multi-clip marker, but never
    # overwrite a marker they set by hand — only one the suggester set before.
    if not candidate.multi_clip_potential or candidate.multi_clip_auto:
        auto = len(clips) > 1
        candidate.multi_clip_potential = auto
        candidate.multi_clip_auto = auto
    return ids


def pending_for_candidate(session, candidate_pk: int) -> list[ClipProposal]:
    """Proposals still awaiting a verdict, in the order the model ranked them."""
    return list(session.execute(
        select(ClipProposal)
        .where(ClipProposal.candidate_pk == candidate_pk,
               ClipProposal.verdict == ClipProposal.VERDICT_PENDING)
        .order_by(ClipProposal.clip_index.asc(), ClipProposal.id.asc())
    ).scalars().all())


def accept(session, proposal_id: int, cut_pk: int) -> ClipProposal | None:
    """Mark a proposal as loaded into a cut. The boundary metrics stay empty
    until that cut is exported — accepting is an intent, not an outcome."""
    row = session.get(ClipProposal, proposal_id)
    if row is None or row.verdict != ClipProposal.VERDICT_PENDING:
        return None
    row.verdict = ClipProposal.VERDICT_ACCEPTED
    row.cut_pk = cut_pk
    row.decided_at = utcnow()
    return row


def dismiss(session, proposal_id: int) -> bool:
    row = session.get(ClipProposal, proposal_id)
    if row is None or row.verdict != ClipProposal.VERDICT_PENDING:
        return False
    row.verdict = ClipProposal.VERDICT_DISMISSED
    row.decided_at = utcnow()
    return True


def dismiss_pending(session, candidate_pk: int) -> int:
    """Reject every open proposal on a video in one call — "none of these".

    Run-level rejection is the partition signal, and it has to be one click:
    a three-clip miss that costs three clicks to reject is a miss that stops
    getting recorded.
    """
    return session.execute(
        update(ClipProposal)
        .where(ClipProposal.candidate_pk == candidate_pk,
               ClipProposal.verdict == ClipProposal.VERDICT_PENDING)
        .values(verdict=ClipProposal.VERDICT_DISMISSED, decided_at=utcnow())
    ).rowcount or 0


# ---- resolving --------------------------------------------------------------

def _score(row: ClipProposal, final: list[dict]) -> float:
    try:
        return iou(json.loads(row.proposed_segments or "[]"), final)
    except (ValueError, TypeError):
        return 0.0


def resolve_for_cut(session, cut_pk: int, candidate_pk: int,
                    final_segments: list[dict],
                    from_cut_pk: int | None = None) -> ClipProposal | None:
    """Resolve this cut's proposal against the segments that actually shipped.

    Prefers the proposal explicitly accepted into this cut. Failing that, the
    best-overlapping still-pending proposal on the video is resolved instead —
    the operator who cut past the panel without clicking anything still leaves
    a verdict, exactly as writing past the caption card does.

    ``from_cut_pk`` covers "Save as new clip": the proposal was accepted into
    the cut the operator had open, but the segments landed in a fresh sibling.
    The proposal follows the material, so it re-points — unless it has already
    been resolved by an earlier export of the original cut, which would make
    one proposal answer for two different clips.

    A pending proposal that doesn't overlap the export at all is left alone:
    on a multi-story video it's probably describing a clip that hasn't been
    made yet, and claiming it here would invent a rejection.
    """
    final = normalize(final_segments)
    if not final:
        return None

    lookup = [cut_pk] + ([from_cut_pk] if from_cut_pk and from_cut_pk != cut_pk else [])
    row = session.execute(
        select(ClipProposal)
        .where(ClipProposal.cut_pk.in_(lookup), ClipProposal.final_segments == "")
        .order_by(ClipProposal.id.desc()).limit(1)
    ).scalar_one_or_none()
    if row is not None:
        row.cut_pk = cut_pk

    if row is None:
        candidates = [(r, _score(r, final)) for r in
                      pending_for_candidate(session, candidate_pk)]
        candidates = [(r, s) for r, s in candidates if s > _MATCH_FLOOR]
        if not candidates:
            return None
        row = max(candidates, key=lambda pair: pair[1])[0]
        row.cut_pk = cut_pk

    try:
        proposed = json.loads(row.proposed_segments or "[]")
    except (ValueError, TypeError):
        proposed = []

    row.final_segments = json.dumps(final)
    row.final_segment_count = len(final)
    row.final_duration_s = duration(final)
    row.iou = iou(proposed, final)
    row.start_delta_s, row.end_delta_s = span_deltas(proposed, final)
    if row.verdict == ClipProposal.VERDICT_PENDING:
        row.verdict = (ClipProposal.VERDICT_ACCEPTED if row.iou >= KEPT_IOU
                       else ClipProposal.VERDICT_DISMISSED)
        row.decided_at = utcnow()
    return row
