"""What the clip suggester learns from this account's own history.

The suggester used to see one video and nothing else, so every pass started
from zero: no idea what this operator ships, what they keep throwing back, or
what the account posted yesterday. This module turns the ledgers the app
already keeps into a compact ``account_context`` for the prompt:

- shipped_examples: published clips, best-performing first, as the words
  actually heard — the bar to match, shown rather than described
- rejected_here / rejected_elsewhere: dismissals (and their reasons) on this
  video and recently across the account
- calibration: the signed boundary/length bias measured at export time,
  phrased as the correction it implies
- recently_posted: stories the account told lately, so a pass can avoid
  retelling one — or knowingly continue it

Everything here is best-effort. A failure anywhere drops that piece of
context; it must never stop a suggestion pass.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics

from sqlalchemy import select

from .models import Candidate, ClipProposal, Cut, ThreadsPost, utcnow

log = logging.getLogger("clip_context")

REASON_LABELS = {
    "off_topic": "off topic",
    "weak_visuals": "weak visuals",
    "bad_boundaries": "bad boundaries",
    "too_long": "too long",
    "already_covered": "already covered",
    "not_clipworthy": "not clipworthy",
}

# Resolved proposals needed before a measured bias is worth stating. Below
# this, a median is one bad afternoon, not a habit.
MIN_CALIBRATION_SAMPLES = 5

# Bias smaller than this is within the noise of hand-trimming.
_BOUNDARY_NOISE_S = 1.0


def _segments(raw: str) -> list[dict]:
    try:
        data = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []
    out = []
    for s in data if isinstance(data, list) else []:
        try:
            start, end = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            out.append({"start": start, "end": end})
    return out


def shipped_examples(session, *, limit: int = 6, exclude_candidate_pk: int | None = None,
                     lookback_days: int = 180) -> list[dict]:
    """Published clips worth imitating, best-performing first.

    Ranked by each post's latest view count; posts with no snapshot yet sort
    after the measured ones, newest first. The transcript is sliced from the
    source video's stored transcript (a DB column), so this keeps working after
    the files themselves are offloaded.
    """
    from .analytics import latest_metrics_bulk
    from .publishing import _slice_transcript

    cutoff = utcnow() - dt.timedelta(days=lookback_days)
    rows = session.execute(
        select(ThreadsPost.id, ThreadsPost.cut_pk, ThreadsPost.published_at)
        .where(ThreadsPost.status == "published",
               ThreadsPost.cut_pk.is_not(None),
               ThreadsPost.published_at >= cutoff)
        .order_by(ThreadsPost.published_at.desc())
        .limit(300)
    ).all()
    if not rows:
        return []
    views = {pk: (m.get("views") or 0)
             for pk, m in latest_metrics_bulk(session, [r.id for r in rows]).items()}
    # One post per cut: reruns of the same clip would otherwise crowd the list.
    best: dict[int, tuple[int, int]] = {}
    for r in rows:
        v = views.get(r.id)
        score = v if v is not None else -1
        if r.cut_pk not in best or score > best[r.cut_pk][0]:
            best[r.cut_pk] = (score, r.id)
    measured = sorted(s for s, _ in best.values() if s >= 0)

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    cut_ids = [cut_pk for cut_pk, _ in ranked[:limit * 3]]
    cuts = {c.id: c for c in session.execute(
        select(Cut).where(Cut.id.in_(cut_ids))).scalars().all()}
    cand_ids = {c.candidate_pk for c in cuts.values()}
    candidates = {c.id: c for c in session.execute(
        select(Candidate).where(Candidate.id.in_(cand_ids))).scalars().all()}

    out: list[dict] = []
    for cut_pk, (score, _post) in ranked:
        if len(out) >= limit:
            break
        cut = cuts.get(cut_pk)
        if cut is None or cut.candidate_pk == exclude_candidate_pk:
            continue
        cand = candidates.get(cut.candidate_pk)
        segs = _segments(cut.trim_segments)
        if cand is None or not segs:
            continue
        text = _slice_transcript(cand.transcript_text or "", cut.trim_segments or "")
        if not text:
            continue
        example = {
            "seconds": round(sum(s["end"] - s["start"] for s in segs), 1),
            "segments": len(segs),
            "words_heard": text[:600],
        }
        if score >= 0 and measured:
            below = sum(1 for m in measured if m < score)
            pct = round(100 * (1 - below / len(measured)))
            example["performance"] = f"top {max(1, pct)}% of recent posts by views"
        out.append(example)
    return out


def rejected(session, candidate_pk: int, *, limit: int = 8) -> tuple[list[dict], list[dict]]:
    """``(rejected_here, rejected_elsewhere)``.

    Here: every dismissal on this video, reason or not — re-proposing the
    thing the operator just threw back is the fastest way to lose their
    attention. Elsewhere: only dismissals that came with a reason, because a
    bare "no" from another video teaches nothing transferable.
    """
    here_rows = session.execute(
        select(ClipProposal)
        .where(ClipProposal.candidate_pk == candidate_pk,
               ClipProposal.verdict == ClipProposal.VERDICT_DISMISSED)
        .order_by(ClipProposal.id.desc()).limit(limit)
    ).scalars().all()
    here = [{"story": r.story,
             "segments": [{"start": round(s["start"], 1), "end": round(s["end"], 1)}
                          for s in _segments(r.proposed_segments)],
             **({"reason": REASON_LABELS.get(r.dismiss_reason, r.dismiss_reason)}
                if r.dismiss_reason else {})}
            for r in here_rows]

    else_rows = session.execute(
        select(ClipProposal)
        .where(ClipProposal.candidate_pk != candidate_pk,
               ClipProposal.dismiss_reason != "")
        .order_by(ClipProposal.id.desc()).limit(limit)
    ).scalars().all()
    elsewhere = [{"story": r.story,
                  "reason": REASON_LABELS.get(r.dismiss_reason, r.dismiss_reason),
                  "seconds": round(r.proposed_duration_s or 0, 1)}
                 for r in else_rows]
    return here, elsewhere


def calibration_notes(resolved: list[dict], reasons: dict[str, int]) -> list[str]:
    """Measured biases, phrased as the correction each one implies.

    ``resolved`` rows carry the export-time numbers (``start_delta_s``,
    ``end_delta_s``, proposed/final duration and segment counts). Deltas are
    signed: positive means the operator moved that boundary LATER.
    """
    notes: list[str] = []
    if len(resolved) >= MIN_CALIBRATION_SAMPLES:
        def med(key):
            vals = [r[key] for r in resolved if r.get(key) is not None]
            return statistics.median(vals) if vals else None

        start, end = med("start_delta_s"), med("end_delta_s")
        if start is not None and start >= _BOUNDARY_NOISE_S:
            notes.append(f"Past clips started about {start:.0f}s too early — "
                         "the operator cuts into the opening. Start later, on the hook.")
        elif start is not None and start <= -_BOUNDARY_NOISE_S:
            notes.append(f"Past clips started about {-start:.0f}s too late — "
                         "the operator pulls the start earlier.")
        if end is not None and end <= -_BOUNDARY_NOISE_S:
            notes.append(f"Past clips ran about {-end:.0f}s too long at the end — "
                         "end sooner, on the payoff.")
        elif end is not None and end >= _BOUNDARY_NOISE_S:
            notes.append(f"Past clips ended about {end:.0f}s too soon — "
                         "the operator lets the payoff finish.")
        p_dur, f_dur = med("proposed_duration_s"), med("final_duration_s")
        p_seg, f_seg = med("proposed_segment_count"), med("final_segment_count")
        if p_dur and f_dur and abs(p_dur - f_dur) >= 4:
            notes.append(f"Shipped clips run a median {f_dur:.0f}s; proposals "
                         f"ran {p_dur:.0f}s. Aim for {f_dur:.0f}s.")
        if p_seg and f_seg and abs(p_seg - f_seg) >= 1:
            notes.append(f"Shipped clips join a median {f_seg:.0f} segments; "
                         f"proposals used {p_seg:.0f}.")
    common = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if common and sum(n for _, n in common) >= 3:
        notes.append("Most common reasons proposals were dismissed: " + ", ".join(
            f"{REASON_LABELS.get(k, k)} ({n})" for k, n in common) + ".")
    return notes


def calibration(session, *, sample: int = 100, reason_days: int = 60) -> list[str]:
    rows = session.execute(
        select(ClipProposal)
        .where(ClipProposal.final_segments != "")
        .order_by(ClipProposal.id.desc()).limit(sample)
    ).scalars().all()
    resolved = [{"start_delta_s": r.start_delta_s, "end_delta_s": r.end_delta_s,
                 "proposed_duration_s": r.proposed_duration_s,
                 "final_duration_s": r.final_duration_s,
                 "proposed_segment_count": r.proposed_segment_count,
                 "final_segment_count": r.final_segment_count}
                for r in rows]
    since = utcnow() - dt.timedelta(days=reason_days)
    reasons: dict[str, int] = {}
    for (reason,) in session.execute(
        select(ClipProposal.dismiss_reason)
        .where(ClipProposal.dismiss_reason != "",
               ClipProposal.decided_at >= since)
    ).all():
        reasons[reason] = reasons.get(reason, 0) + 1
    return calibration_notes(resolved, reasons)


def recently_posted(session, *, days: int = 14, limit: int = 15) -> list[str]:
    """Titles of clips the account published lately, newest first."""
    since = utcnow() - dt.timedelta(days=days)
    rows = session.execute(
        select(Cut.clip_title, Cut.calendar_name, ThreadsPost.caption)
        .join(Cut, Cut.id == ThreadsPost.cut_pk)
        .where(ThreadsPost.status == "published", ThreadsPost.published_at >= since)
        .order_by(ThreadsPost.published_at.desc()).limit(limit * 2)
    ).all()
    seen, out = set(), []
    for title, name, caption in rows:
        label = (title or name or (caption or "")[:120]).strip()
        if label and label not in seen:
            seen.add(label)
            out.append(label)
        if len(out) >= limit:
            break
    return out


def build(session, candidate, settings) -> dict:
    """The ``account_context`` payload for one suggestion pass.

    Each piece is fetched independently inside its own SAVEPOINT and dropped
    on failure: on Postgres one failed statement poisons the whole
    transaction, and this runs in the same session that then writes the
    proposals. A broken metrics table should cost the examples' ranking, not
    the pass.
    """
    if not settings.get("clips.learn_from_history", True):
        return {}
    limit = int(settings.get("clips.context_examples", 6))

    def attempt(label, fn):
        try:
            with session.begin_nested():
                return fn()
        except Exception as exc:
            log.info("Clip context %s unavailable: %s", label, exc)
            return None

    here, elsewhere = attempt("rejections", lambda: rejected(session, candidate.id)) or ([], [])
    out = {
        "shipped_examples": attempt("examples", lambda: shipped_examples(
            session, limit=limit, exclude_candidate_pk=candidate.id)),
        "rejected_here": here,
        "rejected_elsewhere": elsewhere,
        "calibration": attempt("calibration", lambda: calibration(session)),
        "recently_posted": attempt("recent posts", lambda: recently_posted(session)),
    }
    return {k: v for k, v in out.items() if v}
