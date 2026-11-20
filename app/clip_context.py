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

Two suggesters now share these ledgers, so calibration is kept per suggester
(``calibration="legacy" | "engine" | "none"``): the transcript suggester's
habit of starting early (which the operator trims away) must not become an
instruction to the clip engine, whose editor is told to open on the person,
and vice versa. Engine rows are the ones whose ``model`` starts "engine:".

The replay benchmark rebuilds this context AS OF a video's archive time
(``as_of``): every window is measured back from that moment and only rows that
existed by then count, so a replayed pass sees what the original pass could
have seen and nothing the operator did later. ``replay=True`` additionally
drops everything tied to the replayed video itself (its own dismissals, its
own clips) and leaves calibration out, because the export-time numbers it
reads are stamped later than the rows they describe.

Everything here is best-effort. A failure anywhere drops that piece of
context; it must never stop a suggestion pass.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics

from sqlalchemy import func, or_, select

from .models import Candidate, ClipProposal, Cut, MetricSnapshot, ThreadsPost, utcnow

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

# Which suggester's ledger rows a calibration reads. "none" leaves it out.
CALIBRATION_KINDS = ("legacy", "engine", "none")

# ClipProposal.model of rows the clip engine wrote: f"engine:{setup}:{editor model}".
ENGINE_MODEL_PREFIX = "engine:"


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


def _moment(value) -> dt.datetime | None:
    """``as_of`` as an aware UTC datetime, or None for "now".

    Accepts a datetime (naive means UTC — SQLite hands timestamps back naive)
    or an ISO-8601 string, which is how the benchmark freezes archive times.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        value = dt.datetime.fromisoformat(text)
    if not isinstance(value, dt.datetime):
        if isinstance(value, dt.date):
            value = dt.datetime(value.year, value.month, value.day)
        else:
            raise TypeError(f"as_of must be a datetime or ISO string, not {type(value).__name__}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _calibration_kind(kind) -> str:
    value = str(kind or "legacy").strip().lower()
    if value not in CALIBRATION_KINDS:
        raise ValueError(f"calibration must be one of {', '.join(CALIBRATION_KINDS)}, not {kind!r}")
    return value


def _suggester_filter(kind: str):
    """WHERE clause selecting one suggester's ClipProposal rows."""
    engine = ClipProposal.model.startswith(ENGINE_MODEL_PREFIX, autoescape=True)
    if kind == "engine":
        return engine
    # Legacy: everything else, including rows from before ``model`` existed.
    return or_(ClipProposal.model.is_(None), ~engine)


def _latest_views_before(session, post_ids: list[int], when: dt.datetime) -> dict[int, dict]:
    """``post_pk -> {"views": n}`` from each post's newest snapshot taken at or
    before ``when`` — the ranking a pass at that moment could have seen. The
    same shape as ``analytics.latest_metrics_bulk`` (posts with no snapshot by
    then are absent)."""
    if not post_ids:
        return {}
    rn = (
        func.row_number()
        .over(partition_by=MetricSnapshot.post_pk,
              order_by=MetricSnapshot.captured_at.desc())
        .label("rn")
    )
    sub = (
        select(MetricSnapshot.post_pk, MetricSnapshot.views, rn)
        .where(MetricSnapshot.post_pk.in_(post_ids), MetricSnapshot.captured_at <= when)
        .subquery()
    )
    rows = session.execute(select(sub.c.post_pk, sub.c.views).where(sub.c.rn == 1)).all()
    return {row.post_pk: {"views": row.views} for row in rows}


def shipped_examples(session, *, limit: int = 6, exclude_candidate_pk: int | None = None,
                     lookback_days: int = 180, as_of=None) -> list[dict]:
    """Published clips worth imitating, best-performing first.

    Ranked by each post's latest view count; posts with no snapshot yet sort
    after the measured ones, newest first. The transcript is sliced from the
    source video's stored transcript (a DB column), so this keeps working after
    the files themselves are offloaded.

    With ``as_of`` the lookback runs back from that moment, only posts
    published before it count, and each is ranked by the newest snapshot
    taken by then — not by the views it went on to earn.
    """
    from .analytics import latest_metrics_bulk
    from .publishing import _slice_transcript

    when = _moment(as_of)
    cutoff = (when or utcnow()) - dt.timedelta(days=lookback_days)
    query = (
        select(ThreadsPost.id, ThreadsPost.cut_pk, ThreadsPost.published_at)
        .where(ThreadsPost.status == "published",
               ThreadsPost.cut_pk.is_not(None),
               ThreadsPost.published_at >= cutoff)
    )
    if when is not None:
        query = query.where(ThreadsPost.published_at < when)
    rows = session.execute(
        query.order_by(ThreadsPost.published_at.desc()).limit(300)
    ).all()
    if not rows:
        return []
    post_ids = [r.id for r in rows]
    metrics = (latest_metrics_bulk(session, post_ids) if when is None
               else _latest_views_before(session, post_ids, when))
    views = {pk: (m.get("views") or 0) for pk, m in metrics.items()}
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


def rejected(session, candidate_pk: int, *, limit: int = 8, as_of=None,
             include_here: bool = True) -> tuple[list[dict], list[dict]]:
    """``(rejected_here, rejected_elsewhere)``.

    Here: every dismissal on this video, reason or not — re-proposing the
    thing the operator just threw back is the fastest way to lose their
    attention. Elsewhere: only dismissals that came with a reason, because a
    bare "no" from another video teaches nothing transferable.

    ``as_of`` keeps only verdicts given before that moment. ``include_here=False``
    (replay) leaves this video's own verdicts out entirely: they are the
    operator's answer to the very question being replayed.
    """
    when = _moment(as_of)
    here: list[dict] = []
    if include_here:
        query = (select(ClipProposal)
                 .where(ClipProposal.candidate_pk == candidate_pk,
                        ClipProposal.verdict == ClipProposal.VERDICT_DISMISSED))
        if when is not None:
            query = query.where(ClipProposal.decided_at < when)
        here_rows = session.execute(
            query.order_by(ClipProposal.id.desc()).limit(limit)
        ).scalars().all()
        here = [{"story": r.story,
                 "segments": [{"start": round(s["start"], 1), "end": round(s["end"], 1)}
                              for s in _segments(r.proposed_segments)],
                 **({"reason": REASON_LABELS.get(r.dismiss_reason, r.dismiss_reason)}
                    if r.dismiss_reason else {})}
                for r in here_rows]

    query = (select(ClipProposal)
             .where(ClipProposal.candidate_pk != candidate_pk,
                    ClipProposal.dismiss_reason != ""))
    if when is not None:
        query = query.where(ClipProposal.decided_at < when)
    else_rows = session.execute(
        query.order_by(ClipProposal.id.desc()).limit(limit)
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


def calibration_for(session, *, kind: str = "legacy", sample: int = 100,
                    reason_days: int = 60, as_of=None) -> list[str]:
    """Calibration notes measured on ONE suggester's resolved proposals.

    ``kind="engine"`` stays silent until at least ``MIN_CALIBRATION_SAMPLES``
    engine proposals have been resolved: a new suggester has no habits yet,
    and borrowing the old one's would push it the wrong way.

    With ``as_of``, a resolved row only counts when the export that resolved it
    provably happened before that moment: exports stamp the cut's
    ``updated_at`` and nothing moves it backwards, so ``updated_at < as_of``
    bounds the export time. Stricter than strictly necessary (a later caption
    edit also bumps it), which errs toward saying less rather than leaking.
    """
    kind = _calibration_kind(kind)
    if kind == "none":
        return []
    when = _moment(as_of)
    query = select(ClipProposal).where(ClipProposal.final_segments != "",
                                       _suggester_filter(kind))
    if when is not None:
        query = (query.join(Cut, Cut.id == ClipProposal.cut_pk)
                 .where(ClipProposal.decided_at < when, Cut.updated_at < when))
    rows = session.execute(
        query.order_by(ClipProposal.id.desc()).limit(sample)
    ).scalars().all()
    resolved = [{"start_delta_s": r.start_delta_s, "end_delta_s": r.end_delta_s,
                 "proposed_duration_s": r.proposed_duration_s,
                 "final_duration_s": r.final_duration_s,
                 "proposed_segment_count": r.proposed_segment_count,
                 "final_segment_count": r.final_segment_count}
                for r in rows]
    if kind == "engine" and len(resolved) < MIN_CALIBRATION_SAMPLES:
        return []
    since = (when or utcnow()) - dt.timedelta(days=reason_days)
    reason_query = (select(ClipProposal.dismiss_reason)
                    .where(ClipProposal.dismiss_reason != "",
                           ClipProposal.decided_at >= since,
                           _suggester_filter(kind)))
    if when is not None:
        reason_query = reason_query.where(ClipProposal.decided_at < when)
    reasons: dict[str, int] = {}
    for (reason,) in session.execute(reason_query).all():
        reasons[reason] = reasons.get(reason, 0) + 1
    return calibration_notes(resolved, reasons)


def calibration(session, *, sample: int = 100, reason_days: int = 60) -> list[str]:
    """The transcript suggester's calibration, measured now (the original
    entry point; see ``calibration_for``)."""
    return calibration_for(session, kind="legacy", sample=sample, reason_days=reason_days)


def recently_posted(session, *, days: int = 14, limit: int = 15, as_of=None,
                    exclude_candidate_pk: int | None = None) -> list[str]:
    """Titles of clips the account published lately, newest first.

    ``as_of`` measures "lately" back from that moment. ``exclude_candidate_pk``
    (replay) drops clips cut from that video: telling a replayed pass "already
    posted" about its own answer would steer it away from being right.
    """
    when = _moment(as_of)
    since = (when or utcnow()) - dt.timedelta(days=days)
    query = (
        select(Cut.clip_title, Cut.calendar_name, ThreadsPost.caption)
        .join(Cut, Cut.id == ThreadsPost.cut_pk)
        .where(ThreadsPost.status == "published", ThreadsPost.published_at >= since)
    )
    if when is not None:
        query = query.where(ThreadsPost.published_at < when)
    if exclude_candidate_pk is not None:
        query = query.where(Cut.candidate_pk != exclude_candidate_pk)
    rows = session.execute(
        query.order_by(ThreadsPost.published_at.desc()).limit(limit * 2)
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


def build(session, candidate, settings, *, as_of=None, replay: bool = False,
          calibration: str = "legacy") -> dict:
    """The ``account_context`` payload for one suggestion pass.

    Each piece is fetched independently inside its own SAVEPOINT and dropped
    on failure: on Postgres one failed statement poisons the whole
    transaction, and this runs in the same session that then writes the
    proposals. A broken metrics table should cost the examples' ranking, not
    the pass. (The SAVEPOINTs need a writer session: on the AUTOCOMMIT
    read-only engine Postgres refuses them and every piece comes back empty.)

    ``as_of`` rebuilds the context as it stood at that moment; ``replay`` also
    drops everything tied to this video and leaves calibration out (see the
    module docstring); ``calibration`` picks whose ledger rows calibrate
    ("legacy", the default, keeps the original suggester's behavior).
    """
    if not settings.get("clips.learn_from_history", True):
        return {}
    limit = int(settings.get("clips.context_examples", 6))
    kind = _calibration_kind(calibration)
    when = _moment(as_of)
    candidate_pk = candidate.id
    # A silently missing piece skews a replay comparison, so say so loudly
    # there; in a live pass it's routine best-effort and stays at info.
    strict = replay or when is not None
    failed: list[str] = []

    def attempt(label, fn):
        try:
            with session.begin_nested():
                return fn()
        except Exception as exc:
            failed.append(label)
            (log.warning if strict else log.info)("Clip context %s unavailable: %s", label, exc)
            return None

    here, elsewhere = attempt("rejections", lambda: rejected(
        session, candidate_pk, as_of=when, include_here=not replay)) or ([], [])
    want_calibration = not replay and kind != "none"
    out = {
        "shipped_examples": attempt("examples", lambda: shipped_examples(
            session, limit=limit, exclude_candidate_pk=candidate_pk, as_of=when)),
        "rejected_here": here,
        "rejected_elsewhere": elsewhere,
        "calibration": (attempt("calibration", lambda: calibration_for(
            session, kind=kind, as_of=when)) if want_calibration else None),
        "recently_posted": attempt("recent posts", lambda: recently_posted(
            session, as_of=when, exclude_candidate_pk=candidate_pk if replay else None)),
    }
    if failed and strict:
        log.warning("Replayed context for candidate %s is missing: %s",
                    candidate_pk, ", ".join(failed))
    return {k: v for k, v in out.items() if v}
