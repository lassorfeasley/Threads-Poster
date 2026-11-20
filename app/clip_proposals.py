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
import math
import uuid
from pathlib import Path

from sqlalchemy import select, update

from . import llm, vision
from .llm import suggest_clips
from .models import ClipProposal, ClipRevision, Cut, utcnow

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

# Footage reuse between clips of ONE video. Clips may recycle each other's
# footage — a video about several artisans can reuse the same setup shots in
# every clip — so long as each clip opens on its own shot and tells its own
# story. Mechanically: a clip's OPENING is the first OPENING_SECONDS of its
# first played segment, and two openings clash when those windows overlap; a
# clip sharing more than DUPLICATE_IOU of its material with another clip is
# that clip again, not a new one.
OPENING_SECONDS = 3.0
DUPLICATE_IOU = 0.6
CONFLICT_SAME_OPENING = "same_opening"
CONFLICT_DUPLICATE = "duplicate"

# Why a dismissal happened — the one-click annotation offered right after
# dismissing, never required. Slugs are the stable vocabulary (labels live in
# the UI); each names a different fix, which is the point of collecting them:
# off_topic / not_clipworthy are partition misses, bad_boundaries and too_long
# are prompt fixes, weak_visuals indicts the opening pass, already_covered
# means the model ignored (or wasn't shown) what's been posted.
DISMISS_REASONS = frozenset({
    "off_topic", "weak_visuals", "bad_boundaries",
    "too_long", "already_covered", "not_clipworthy",
})


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


# ---- play order -------------------------------------------------------------
#
# A clip's segment list is in PLAY order: a cold open plays its payoff first
# and then cuts back to the setup. ``normalize`` above sorts and merges, which
# is right for scoring "did it pick the right material" and wrong for anything
# that will be played, stored as a proposal, or loaded into the trim editor —
# a sorted cold open is a different clip. Everything below keeps list order.

def _span(segment) -> tuple[float, float] | None:
    """``(start, end)`` as finite floats, or None when the entry is unusable."""
    if not isinstance(segment, dict):
        return None
    try:
        start, end = float(segment["start"]), float(segment["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    return start, end


def _minus(start: float, end: float,
           holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``[start, end)`` with every hole removed, as pieces in time order."""
    pieces = [(start, end)]
    for hole_start, hole_end in holes:
        nxt: list[tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if hole_end <= piece_start or hole_start >= piece_end:
                nxt.append((piece_start, piece_end))
                continue
            if hole_start > piece_start:
                nxt.append((piece_start, hole_start))
            if hole_end < piece_end:
                nxt.append((hole_end, piece_end))
        pieces = nxt
    return [p for p in pieces if p[1] > p[0]]


def clean_play_order(segments: list[dict], *, min_len: float = 0.5,
                     floor: float = 0.0, horizon: float | None = None) -> list[dict]:
    """Coerce, clamp and de-overlap a segment list WITHOUT reordering it.

    Each segment is clamped to ``[floor, horizon]``. A segment overlapping an
    EARLIER one in the list is cut down to the part not already played — the
    same footage twice in one clip is a mistake far more often than a choice —
    and when that splits it, the longer piece is kept so the segment stays one
    shot. Anything shorter than ``min_len`` afterwards is dropped. Extra keys
    (an engine draft's ``lead``) ride along; times are rounded to 2 dp, and
    overlap is judged on the rounded values so the output never overlaps.
    """
    lo = 0.0 if floor is None else float(floor)
    hi = None if horizon is None else float(horizon)
    shortest = float(min_len or 0.0)
    out: list[dict] = []
    for segment in segments or []:
        span = _span(segment)
        if span is None:
            continue
        start, end = max(span[0], lo), span[1]
        if hi is not None:
            end = min(end, hi)
        pieces = _minus(start, end, [(s["start"], s["end"]) for s in out])
        if not pieces:
            continue
        # max() keeps the first of equal pieces: the earlier one wins a tie.
        start, end = max(pieces, key=lambda p: p[1] - p[0])
        start, end = round(start, 2), round(end, 2)
        if end - start < shortest - 1e-9:
            continue
        kept = dict(segment)
        kept["start"], kept["end"] = start, end
        out.append(kept)
    return out


def strip_segments(segments: list[dict]) -> list[dict]:
    """``[{start, end}]`` only, in the order given — what the ledger stores.
    Engine-only keys such as ``lead`` stay out of ``proposed_segments``."""
    out: list[dict] = []
    for segment in segments or []:
        span = _span(segment)
        if span is not None:
            out.append({"start": span[0], "end": span[1]})
    return out


# ---- other clips of the same video ------------------------------------------
#
# The footage-reuse rule. A clip may recycle footage another clip of the same
# video already uses (shared setup, context, b-roll), but it must (a) OPEN on a
# different shot and (b) tell a meaningfully different story. (a) and the
# duplicate test are mechanical and live here; (b) is the model's judgment,
# which is why every suggester and reviser is shown the other clips' titles and
# words heard.

def opening_window(segments: list[dict]) -> tuple[float, float] | None:
    """``(start, start + OPENING_SECONDS)`` of the first PLAYED segment, or
    None when the clip has no usable segment."""
    for segment in segments or []:
        span = _span(segment)
        if span is not None and span[1] > span[0]:
            return span[0], span[0] + OPENING_SECONDS
    return None


def opening_windows(clips: list[dict]) -> list[dict]:
    """Every clip's opening window as ``{start, end}`` — the footage no other
    clip may open on (and what an opening pull-back may not reach into)."""
    out: list[dict] = []
    for clip in clips or []:
        window = opening_window((clip or {}).get("segments") or [])
        if window is not None:
            out.append({"start": window[0], "end": window[1]})
    return out


def words_heard(segments: list[dict], words: list[dict] | None,
                transcript: list[dict] | None, *, limit: int = 40) -> str:
    """The words a clip plays, in PLAY order, capped at ``limit`` words: enough
    for a model to tell what story another clip already tells."""
    heard: list[str] = []
    for segment in segments or []:
        if _span(segment) is None:
            continue
        heard.extend(_spoken_in(segment, words or [], transcript or []))
        if len(heard) > limit:
            break
    text = " ".join(heard[:limit])
    return text + ("…" if len(heard) > limit else "")


def clips_from_cuts(cuts, *, words: list[dict] | None = None,
                    transcript: list[dict] | None = None) -> list[dict]:
    """``Cut`` rows (or anything with ``id``, ``clip_title``, ``trim_segments``)
    as plain "other clip" dicts: ``{cut_pk, title, segments (play order),
    opening: [s, s+3], words_heard}``. Cuts without segments are skipped;
    ``words_heard`` is "" unless a word stream or transcript is given."""
    out: list[dict] = []
    for cut in cuts or []:
        try:
            raw = json.loads(getattr(cut, "trim_segments", "") or "[]")
        except (ValueError, TypeError):
            continue
        segments = strip_segments(raw if isinstance(raw, list) else [])
        window = opening_window(segments)
        if window is None:
            continue
        cut_pk = getattr(cut, "id", None)
        title = " ".join(str(getattr(cut, "clip_title", "") or "").split())
        out.append({
            "cut_pk": cut_pk,
            "title": title or (f"Clip {cut_pk}" if cut_pk is not None else "Another clip"),
            "segments": segments,
            "opening": [round(window[0], 2), round(window[1], 2)],
            "words_heard": (words_heard(segments, words, transcript)
                            if (words or transcript) else ""),
        })
    return out


def other_clips(session, candidate_pk: int, exclude_cut_pk: int | None = None, *,
                words: list[dict] | None = None,
                transcript: list[dict] | None = None) -> list[dict]:
    """The clips already made from this video (every cut with segments,
    exported or not), oldest first, as ``clips_from_cuts`` dicts.
    ``exclude_cut_pk`` leaves out the cut being edited: it is not another clip
    of itself."""
    where = [Cut.candidate_pk == candidate_pk, Cut.trim_segments != ""]
    if exclude_cut_pk is not None:
        where.append(Cut.id != exclude_cut_pk)
    rows = session.execute(select(Cut).where(*where).order_by(Cut.id.asc())).scalars().all()
    return clips_from_cuts(rows, words=words, transcript=transcript)


# A shot longer than this is a long take (a sit-down interview, a speech): two
# clips about different things can both open inside it, so only the 3-second
# opening window keeps them apart there.
LONG_TAKE_SECONDS = 20.0


def _shot_index(t: float, cut_times: list[float]) -> int:
    import bisect

    return bisect.bisect_right(cut_times, t)


def _shot_length(index: int, cut_times: list[float]) -> float:
    """Length of the shot ``index`` (between cut_times[index-1] and
    cut_times[index]); the first and last shots are open-ended -> infinite."""
    if index <= 0 or index >= len(cut_times):
        return math.inf
    return cut_times[index] - cut_times[index - 1]


def clip_conflict(segments: list[dict], others: list[dict], *, cuts=None,
                  min_shot_score: float = 20.0) -> tuple[str, dict | None]:
    """``(kind, other_clip)`` for the first rule ``segments`` breaks against
    the video's other clips, or ``("", None)``.

    - ``"duplicate"``: IoU above ``DUPLICATE_IOU`` with another clip — the
      same clip again. Checked first, against every clip.
    - ``"same_opening"``: its opening window overlaps another clip's, or —
      when a shot map is given (``cuts``: ``[{t, score}]``, only cuts scoring
      at least ``min_shot_score`` count) — the two openings start inside the
      same shot (between the same pair of cuts).

    Sharing footage further in is fine: that is recycled setup, not a repeat.
    """
    mine = opening_window(segments)
    if mine is None:
        return "", None
    clips = [c for c in (others or []) if isinstance(c, dict) and c.get("segments")]
    for other in clips:
        if iou(segments, other["segments"]) > DUPLICATE_IOU:
            return CONFLICT_DUPLICATE, other
    cut_times: list[float] = []
    for cut in cuts or []:
        try:
            t = float(cut["t"]) if isinstance(cut, dict) else float(cut)
            score = float(cut.get("score", min_shot_score)) if isinstance(cut, dict) else min_shot_score
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(t) and score >= min_shot_score:
            cut_times.append(t)
    cut_times.sort()
    for other in clips:
        theirs = opening_window(other["segments"])
        if theirs is None:
            continue
        if mine[0] < theirs[1] and theirs[0] < mine[1]:
            return CONFLICT_SAME_OPENING, other
        if cut_times:
            shot = _shot_index(mine[0], cut_times)
            if (shot == _shot_index(theirs[0], cut_times)
                    and _shot_length(shot, cut_times) <= LONG_TAKE_SECONDS):
                return CONFLICT_SAME_OPENING, other
    return "", None


def conflict_with_clips(segments: list[dict], others: list[dict], *, cuts=None,
                        min_shot_score: float = 20.0) -> str:
    """``clip_conflict``'s kind alone: "" | "same_opening" | "duplicate"."""
    return clip_conflict(segments, others, cuts=cuts, min_shot_score=min_shot_score)[0]


def visible_segments(proposed: list[dict], used: list[dict] | None = None) -> tuple[list[dict], bool]:
    """What a card offers of a proposal: its segments as proposed, in PLAY
    order (unusable entries dropped). Returns ``(segments, trimmed)``;
    ``trimmed`` is always False now.

    Proposals used to be cut down around footage other clips already use.
    Clips may now recycle each other's footage (see ``clip_conflict``), so
    nothing is subtracted: a proposal that opens on another clip's opening is
    flagged on its card and a duplicate is hidden, but neither is reshaped.
    ``used`` is accepted for old callers and ignored.
    """
    kept = [{**segment, "start": span[0], "end": span[1]}
            for segment, span in ((s, _span(s)) for s in proposed or [])
            if span is not None and span[1] > span[0]]
    return kept, False


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

# Bumped whenever the suggester's prompt or inputs change in a way that should
# be scored separately. Blank was the original transcript-only prompt;
# "v2" added the editorial rubric, account context and visual moments; "v3"
# stopped telling the model that most clips join 2-4 segments (single takes
# are often the better clip) and stores proposals in play order.
PROMPT_VERSION = "v3"

# The clip engine's own regime (app/clip_engine), joined with its setup in the
# policy fingerprint ("engine-v1:gemini") so engine proposals are never pooled
# with the transcript suggester's when the ledger is scored.
ENGINE_PROMPT_VERSION = "engine-v1"


def policy_version(*, model: str, max_clips: int, max_segments_per_clip: int,
                   min_seconds: int, max_seconds: int,
                   scope: str = "video", prompt: str = "") -> str:
    """Short fingerprint of the regime that produced a partition, so a year of
    proposals can be split by prompt version instead of blended into one
    unanalyzable pile.

    ``scope`` keeps section passes apart from whole-video ones: a different
    prompt against a hand-picked stretch of footage should not be averaged in
    with "find the clips in this hour". It only joins the payload when it isn't
    the default, so fingerprints already in the ledger keep their meaning.
    """
    parts = [model or "", str(max_clips), str(max_segments_per_clip),
             str(min_seconds), str(max_seconds)]
    if scope != "video":
        parts.append(scope)
    # Same rule for the prompt version: only joins when set.
    if prompt:
        parts.append(f"prompt={prompt}")
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


def log_run(session, candidate_pk: int, clips: list[dict], *,
            model: str, policy: str, window: dict | None = None) -> list[int]:
    """Record one suggestion pass and return the new row ids.

    Any earlier proposal on this video still awaiting a verdict is marked
    ``superseded`` — re-rolling is a soft rejection and shouldn't be left
    looking undecided. Already-accepted rows are untouched, so re-rolling a
    multi-story video keeps the clips already cut.

    A SECTION pass only supersedes what it actually replaces: the pending
    proposals overlapping ``window``. Asking again about chapter three is not
    an answer to a suggestion standing for chapter seven.

    A run with nothing to record — no clips, or none with a usable segment —
    writes nothing and supersedes NOTHING. An empty pass (a failed render
    batch, an off-topic video, a re-roll that found less) is not a verdict on
    the suggestions already standing, and wiping them would leave the operator
    with fewer clips than before they asked.

    Segments are stored in PLAY order (``clean_play_order``: clamped,
    de-overlapped, never sorted) and stripped to ``{start, end}``.
    """
    prepared = [strip_segments(clean_play_order(clip.get("segments") or []))
                if isinstance(clip, dict) else []
                for clip in clips or []]
    if not any(prepared):
        return []
    if window is None:
        session.execute(
            update(ClipProposal)
            .where(ClipProposal.candidate_pk == candidate_pk,
                   ClipProposal.verdict == ClipProposal.VERDICT_PENDING)
            .values(verdict=ClipProposal.VERDICT_SUPERSEDED, decided_at=utcnow())
        )
    else:
        span = normalize([window])
        for row in pending_for_candidate(session, candidate_pk):
            try:
                proposed = json.loads(row.proposed_segments or "[]")
            except (ValueError, TypeError):
                continue
            if _overlap(normalize(proposed), span) > 0:
                row.verdict = ClipProposal.VERDICT_SUPERSEDED
                row.decided_at = utcnow()
    run_id = uuid.uuid4().hex[:16]
    ids: list[int] = []
    for index, (clip, segments) in enumerate(zip(clips, prepared)):
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
            flag=str(clip.get("challenges_rules") or "")[:300],
            # String(60)/String(40): Postgres rejects an overlong value (SQLite
            # doesn't), which would fail the whole write in production only.
            model=str(model or "")[:60],
            policy_version=str(policy or "")[:40],
        )
        session.add(row)
        session.flush()
        ids.append(row.id)
    return ids


def _backstop(start: float, blocked: list[dict]) -> float:
    """Earliest time an opening may be pulled back to without entering blocked
    footage (another clip's opening window, or this clip's own later
    segments). Returns ``start`` itself when there's no room."""
    floor = 0.0
    for span in blocked:
        if span["end"] <= start:
            floor = max(floor, span["end"])
        elif span["start"] < start:
            return start  # the start sits inside blocked footage
    return floor


def _one_line(text: str, limit: int = 150) -> str:
    """The model's rationale, cut down to a receipt the rail can hold.

    ``pick_opening_frame`` asks for one line and often writes a paragraph, and
    a hard character cap ends it mid-word. Keep the first sentence — the claim
    is always there; the frame-by-frame inventory that follows is not worth a
    rail full of text. The full text still goes to the log.
    """
    text = " ".join((text or "").split())
    if not text:
        return ""
    head = text.split(". ")[0].rstrip(". ")
    if len(head) > limit:
        return head[:limit].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"
    return head + "."


def refine_opening(candidate, segments: list[dict], settings,
                   blocked: list[dict], story: str = "") -> list[dict]:
    """Pull a clip's start back onto a stronger image, using the actual frames.

    The transcript pass picks a start from the words alone, so it lands on the
    sentence rather than the shot — which is how a clip ends up opening on an
    anchor at a desk. Here the footage around that start is sampled and the
    model picks the frame worth opening on.

    The start only ever moves EARLIER, onto lead-in footage ahead of the
    speech; moving it later would truncate the first sentence. It never crosses
    into ``blocked`` footage — another clip's opening window (clips may reuse
    each other's footage, never each other's opening shot). Any failure — no local file, no
    ffmpeg, no answer — leaves the transcript's start exactly as it was.
    """
    return refine_opening_detail(candidate, segments, settings, blocked, story)[0]


def refine_opening_detail(candidate, segments: list[dict], settings,
                          blocked: list[dict],
                          story: str = "") -> tuple[list[dict], str]:
    """``refine_opening`` plus a one-line account of what happened.

    The suggester only needs the segments: a silent no-op is fine while it
    polishes its own proposal. An operator who pressed a button needs to know
    which no-op they got — the model looked and kept the start, or it never
    got to look at all — so every early return here carries a reason.
    """
    if not segments:
        return segments, "Mark a segment first."
    if not getattr(candidate, "local_video_path", ""):
        return segments, "The source video isn't on disk, so there are no frames to look at."
    if not settings.get("clips.opening_vision", True):
        return segments, "Opening-shot vision is switched off (clips.opening_vision)."

    lead = float(settings.get("clips.opening_lead_seconds", 4.0))
    hold = float(settings.get("clips.opening_hold_seconds", 3.0))
    interval = float(settings.get("clips.opening_frame_interval", 0.5))
    model = settings.get("clips.opening_model",
                         settings.get("vision.model", "claude-haiku-4-5"))

    first = segments[0]
    start = float(first["start"])
    floor = _backstop(start, normalize(blocked))
    window_start = max(floor, start - lead)
    no_room = ("No lead-in room before this start — the top of the video, or "
               "another clip's opening shot, is right there.")
    if start - window_start < interval:
        return segments, no_room

    # Sample past the start too: the model can only tell whether a shot HOLDS
    # by seeing what follows it.
    window_end = min(start + hold, float(first["end"])) + interval
    frames = vision.frames_between(candidate.local_video_path, window_start,
                                   window_end, interval)
    if len(frames) < 2:
        return segments, "Couldn't sample frames from the video file."
    latest = min(range(len(frames)), key=lambda i: abs(frames[i][0] - start))
    if latest <= 0:
        return segments, no_room

    try:
        picked = llm.pick_opening_frame(model, frames, latest, hold,
                                        title=candidate.title, story=story)
    except Exception as exc:
        log.info("Opening frame pick failed for %s: %s", candidate.video_id, exc)
        return segments, f"The vision model didn't answer: {exc}"

    raw_why = str(picked.get("why", "")).strip()
    why = _one_line(raw_why)
    index = picked.get("index")
    kept = (f"Kept your opening — nothing stronger in the {lead:g}s before it."
            + (f" {why}" if why else ""))
    if index is None or index >= latest:
        return segments, kept
    new_start = max(floor, frames[index][0])
    if new_start >= start - 0.05:
        return segments, kept

    out = [dict(s) for s in segments]
    out[0]["start"] = round(new_start, 2)
    log.info("Opening for %s moved %.1fs -> %.1fs (%s)", candidate.video_id,
             start, new_start, raw_why)
    moved = (f"Opened {start - new_start:.1f}s earlier, at "
             f"{int(new_start // 60)}:{int(new_start % 60):02d}.")
    return out, (f"{moved} {why}" if why else moved)


def load_transcript(candidate) -> list[dict]:
    """The video's timestamped transcript, or [] when it has none."""
    if not candidate.transcript_path:
        return []
    try:
        from .datasync import ensure_file

        ensure_file(candidate.transcript_path)
        data = json.loads(Path(candidate.transcript_path).read_text())
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def load_words(candidate) -> list[dict]:
    """The video's Whisper word stream, or [] when it has none."""
    from .scrape import load_word_transcript

    return load_word_transcript(candidate)


def _spoken_in(segment: dict, words: list[dict], transcript: list[dict]) -> list[str]:
    """The words heard inside one window, in order: Whisper words when the
    video has them (exact), else every transcript line overlapping it."""
    a, b = float(segment["start"]), float(segment["end"])
    out: list[str] = []
    if words:
        for w in words:
            try:
                mid = (float(w["start"]) + float(w["end"])) / 2
            except (KeyError, TypeError, ValueError):
                continue
            if a <= mid < b and str(w.get("word", "")).strip():
                out.append(str(w["word"]).strip())
        return out
    for line in transcript:
        try:
            if float(line["end"]) > a and float(line["start"]) < b:
                out.extend(str(line.get("text", "")).split())
        except (KeyError, TypeError, ValueError):
            continue
    return out


def excerpts(segments: list[dict], words: list[dict], transcript: list[dict],
             *, head: int = 22, tail: int = 16) -> dict:
    """What a proposal opens and closes on, as the words actually heard.

    The model's own ``story`` line says what it THINKS the clip is; these say
    what you'll hear in the first and last seconds — the two places a clip is
    won or lost — so it can be judged from the card without pressing play.
    """
    if not segments:
        return {"opening": "", "closing": ""}
    first = _spoken_in(segments[0], words, transcript)
    last = _spoken_in(segments[-1], words, transcript)
    opening = " ".join(first[:head]) + ("…" if len(first) > head else "")
    if len(segments) == 1 and len(first) <= head + tail:
        closing = ""   # the opening already shows all of it
    else:
        closing = ("…" if len(last) > tail else "") + " ".join(last[-tail:])
    return {"opening": opening.strip(), "closing": closing.strip()}


def pending_summary(session, candidate_pks: list[int]) -> dict[int, dict]:
    """``candidate_pk -> {count, story}`` for still-usable pending proposals.

    "Usable" means the same filter the editor applies: a proposal that is the
    same clip as one already made (``clip_conflict`` "duplicate") isn't
    counted, so a dashboard badge never promises a suggestion the editor won't
    show. One that only opens on another clip's shot still counts — its card
    shows it, flagged.
    """
    if not candidate_pks:
        return {}
    rows = session.execute(
        select(ClipProposal.candidate_pk, ClipProposal.proposed_segments,
               ClipProposal.story)
        .where(ClipProposal.candidate_pk.in_(candidate_pks),
               ClipProposal.verdict == ClipProposal.VERDICT_PENDING)
        .order_by(ClipProposal.clip_index.asc(), ClipProposal.id.asc())
    ).all()
    if not rows:
        return {}
    cuts: dict[int, list] = {}
    for cut in session.execute(
        select(Cut)
        .where(Cut.candidate_pk.in_({r.candidate_pk for r in rows}),
               Cut.trim_segments != "")
        .order_by(Cut.id.asc())
    ).scalars().all():
        cuts.setdefault(cut.candidate_pk, []).append(cut)
    others = {pk: clips_from_cuts(rows_) for pk, rows_ in cuts.items()}
    out: dict[int, dict] = {}
    for r in rows:
        try:
            proposed = json.loads(r.proposed_segments or "[]")
        except (ValueError, TypeError):
            continue
        kept, _ = visible_segments(proposed)
        if not kept:
            continue
        if conflict_with_clips(kept, others.get(r.candidate_pk, [])) == CONFLICT_DUPLICATE:
            continue
        entry = out.setdefault(r.candidate_pk, {"count": 0, "story": r.story or ""})
        entry["count"] += 1
    return out


def propose(session, candidate, settings, transcript_segments: list[dict], *,
            window: dict | None = None, topic: str = "") -> list[int]:
    """Run one suggestion pass for a video and record it. Returns new row ids.

    Callers pass the transcript they already have (the archive pass holds it in
    memory; a re-roll reads it back off disk). Model failures propagate so the
    caller can decide between logging a warning and returning an error — this
    only owns the ledger and the two candidate-level side effects.

    ``window`` scopes the pass to one stretch of the video — the chapter the
    operator is looking at. A section pass gets a smaller clip budget, keeps
    the proposals standing for the rest of the video, and touches neither of
    the candidate-level side effects: how many stories a SECTION holds says
    nothing about whether the VIDEO is worth several clips, and the caption
    seed belongs to the whole-video pass that wrote it.
    """
    if not transcript_segments:
        return []
    model = settings.get("clips.model",
                         settings.get("matching.model", "claude-haiku-4-5"))
    max_clips = (settings.get("clips.section_max_clips", 2) if window
                 else settings.get("clips.max_clips", 3))
    max_segments = settings.get("clips.max_segments", 4)
    min_seconds = settings.get("clips.min_seconds", 15)
    max_seconds = settings.get("clips.max_seconds", 40)
    opening_hold = settings.get("clips.opening_hold_seconds", 3.0)
    words = load_words(candidate)
    others = other_clips(session, candidate.id, words=words,
                         transcript=transcript_segments)
    # What this account has shipped, thrown back and posted lately — see
    # clip_context. Best-effort by construction; an empty dict is fine.
    from . import clip_context
    context = clip_context.build(session, candidate, settings)

    clips = suggest_clips(
        model, candidate.title, transcript_segments,
        max_clips=max_clips, max_segments_per_clip=max_segments,
        min_seconds=min_seconds, max_seconds=max_seconds,
        opening_hold=opening_hold,
        other_clips=others,
        # The archive-time Whisper word stream, when this video has one:
        # boundaries proposed against it land on word edges.
        words=words,
        description=candidate.description,
        window=window, topic=topic,
        context=context,
        # Frames an earlier "Striking frames" scan judged worth opening on.
        visual_moments=vision.load_striking(candidate),
    )

    # Put each opening on a frame worth looking at. Another clip's opening
    # shot is off limits, and so is a sibling's from this same run, so two
    # proposals can't reach back onto the same opening.
    blocked = opening_windows(others)
    for clip in clips:
        clip["segments"] = refine_opening(candidate, clip["segments"], settings,
                                          blocked, story=clip.get("story", ""))
        blocked = blocked + opening_windows([clip])

    ids = log_run(session, candidate.id, clips, model=model, window=window,
                  policy=policy_version(
                      model=model, max_clips=max_clips,
                      max_segments_per_clip=max_segments,
                      min_seconds=min_seconds, max_seconds=max_seconds,
                      scope="section" if window else "video",
                      prompt=PROMPT_VERSION,
                  ))
    # An empty pass left the standing suggestions in place (see log_run), so
    # the candidate-level markers that describe them stay as they were too.
    if window or not ids:
        return ids

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


def set_dismiss_reason(session, proposal_id: int, reason: str) -> bool:
    """Attach a why to a dismissal, after the fact.

    Deliberately a separate call from ``dismiss``: the dismissal itself must
    stay one click, and the reason is a bonus the operator may or may not
    volunteer. The latest click wins. Only dismissed/superseded rows take one —
    an accepted clip has nothing to explain.
    """
    if reason not in DISMISS_REASONS:
        return False
    row = session.get(ClipProposal, proposal_id)
    if row is None or row.verdict not in (ClipProposal.VERDICT_DISMISSED,
                                          ClipProposal.VERDICT_SUPERSEDED):
        return False
    row.dismiss_reason = reason
    return True


# The one canned instruction, logged verbatim so a year of revisions can be
# split into "what the operator typed" and "what the opening button did".
OPENING_INSTRUCTION = "Better opening shot"


def log_revision(session, cut_pk: int, candidate_pk: int, instruction: str, *,
                 before: list[dict], after: list[dict], note: str,
                 changed: bool, model: str) -> ClipRevision:
    """Record one prompted edit. Written when the revision is served — kept or
    not, the instruction is the signal (see ``ClipRevision``)."""
    row = ClipRevision(
        cut_pk=cut_pk,
        candidate_pk=candidate_pk,
        instruction=instruction[:500],
        before_segments=json.dumps(before),
        after_segments=json.dumps(after),
        note=note[:300],
        changed=changed,
        model=model,
    )
    session.add(row)
    return row


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
