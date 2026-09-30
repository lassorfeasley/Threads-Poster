"""Rerun review: a second look at proven posts that aren't evergreen.

Both rerun paths (the repost rotation and just-in-time filler) only take
evergreen clips, and shelf life is judged once — at monitor time, from the
source video, before anything aired. Plenty of clips tagged timely hold up
long after their news peg, and they're locked out of reruns for good.

This surfaces the ones worth a second look: published, non-evergreen, old
enough that their moment has passed, and above the account's view baseline.
Performance only decides WHO gets reviewed — a hit about one specific tornado
is still dated. The call itself is a hindsight re-judgment by the model,
confirmed by the operator. An "evergreen" decision writes the post's
shelf-life override, which is exactly what the rerun paths already read.
"""
from __future__ import annotations

import bisect
import datetime as dt
import logging
import threading

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from . import llm, spend
from .analytics import metrics_at_age_bulk
from .categories import is_first_party
from .config import load_settings
from .db import session_scope
from .models import Candidate, RerunReview, ThreadsPost, utcnow
from .placement import SHELF_EVERGREEN
from .scheduler import invalidate_recycle_overview, resolve_shelf_life

log = logging.getLogger("rerun_review")

DECISION_EVERGREEN = "evergreen"
DECISION_NEW_CAPTION = "new_caption"
DECISION_KEEP = "keep"
DECISIONS = (DECISION_EVERGREEN, DECISION_NEW_CAPTION, DECISION_KEEP)

# Same floor as the repost rotation: below this many published posts with
# metrics, a percentile says nothing.
_MIN_BASELINE = 10

# Pending cards sort by what the model said, most actionable first.
_VERDICT_ORDER = {"evergreen": 0, "new_caption": 1, "": 2, "dated": 3}


def review_settings() -> dict:
    settings = load_settings()

    def g(key: str, default):
        return settings.get(f"scheduler.placement.rerun_review.{key}", default)

    return {
        "min_age_days": float(g("min_age_days", 30) or 0),
        "percentile": float(g("percentile", 50) or 0),
        "model": str(g("model", "claude-haiku-4-5") or "claude-haiku-4-5"),
        "max_per_run": int(g("max_per_run", 60) or 0),
        "metric_age_hours": int(settings.get("learning.metric_age_hours", 48)),
    }


def _aware(ts: dt.datetime) -> dt.datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=dt.timezone.utc)


def review_queue(session) -> dict:
    """Posts awaiting review, and the ones already decided.

    ``pending`` holds the latest airing of every cut that is published, not
    evergreen, at least ``min_age_days`` past that airing, clone-able (a
    re-air copies caption and clip), not first-party, and whose best views at
    the learning comparison age clear the ``percentile`` of the account
    baseline. ``decided`` holds every post with a decision, whatever it
    resolves to now, so a decision can always be undone.
    """
    cfg = review_settings()
    out = {"pending": [], "decided": [], "threshold": None, "baseline_n": 0, "cfg": cfg}

    posts = session.execute(
        select(ThreadsPost)
        .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel),
                 selectinload(ThreadsPost.cut))
        .where(ThreadsPost.status == "published",
               ThreadsPost.published_at.is_not(None))
    ).scalars().all()
    views = metrics_at_age_bulk(session, posts, "views", cfg["metric_age_hours"])
    ranked = sorted(views.values())
    out["baseline_n"] = len(ranked)
    if len(ranked) < _MIN_BASELINE:
        return out
    idx = min(len(ranked) - 1, int(cfg["percentile"] * (len(ranked) - 1) // 100))
    threshold = ranked[idx]
    out["threshold"] = threshold

    reviews = {r.post_pk: r for r in session.execute(select(RerunReview)).scalars()}

    by_cut: dict[int, list[ThreadsPost]] = {}
    for p in posts:
        if p.cut_pk is not None:
            by_cut.setdefault(p.cut_pk, []).append(p)

    now = utcnow()
    for airings in by_cut.values():
        airings.sort(key=lambda p: (_aware(p.published_at), p.id))
        last = airings[-1]
        best = max((views.get(p.id, 0) for p in airings), default=0)
        review = reviews.get(last.id)
        fact = {
            "post": last,
            "cut": last.cut,
            "candidate": last.candidate,
            "shelf_life": resolve_shelf_life(last, last.candidate),
            "views_at_age": best,
            "rank": bisect.bisect_left(ranked, best) / (len(ranked) - 1),
            "aired": _aware(last.published_at).date(),
            "age_days": (now - _aware(last.published_at)).days,
            "airings": len(airings),
            "review": review,
        }
        if review is not None and review.decision:
            out["decided"].append(fact)
            continue
        if last.cut is None or is_first_party(last.candidate):
            continue
        if fact["shelf_life"] == SHELF_EVERGREEN:
            continue
        if not (last.caption or "").strip():
            continue
        if not (last.clip_object_path or last.clip_local_path):
            continue
        if fact["age_days"] < cfg["min_age_days"]:
            continue
        if best <= 0 or best < threshold:
            continue
        out["pending"].append(fact)

    out["pending"].sort(key=lambda f: (
        _VERDICT_ORDER.get(f["review"].verdict if f["review"] else "", 2),
        -f["views_at_age"], f["post"].id))
    out["decided"].sort(key=lambda f: (f["review"].decided_at or now), reverse=True)
    return out


# --- Model re-judgment (background) -----------------------------------------

_job_lock = threading.Lock()
_job = {"running": False, "done": 0, "total": 0, "error": ""}


def judging_status() -> dict:
    with _job_lock:
        return dict(_job)


def _judge_inputs(fact: dict) -> dict:
    from .publishing import _clip_transcript_text

    post, cut, cand = fact["post"], fact["cut"], fact["candidate"]
    channel = cand.channel if cand is not None else None
    transcript = _clip_transcript_text(cut) or ((cand.transcript_text or "") if cand else "")
    published = cand.published_at if cand is not None else None
    return {
        "title": ((cut.clip_title if cut else "") or (cand.title if cand else "") or ""),
        "channel": ((channel.channel_title or channel.call_sign) if channel else ""),
        "caption": post.caption or "",
        "clip_transcript": transcript,
        "shelf_life": fact["shelf_life"],
        "video_published": published.date().isoformat() if published else "",
        "first_aired": fact["aired"].isoformat(),
    }


def start_judging(on_done=None) -> bool:
    """Judge every pending post the model hasn't seen yet, off the request
    path; ``on_done`` runs when the pass ends. False when a run is already
    going."""
    with _job_lock:
        if _job["running"]:
            return False
        _job.update(running=True, done=0, total=0, error="")

    def _run() -> None:
        try:
            _judge_all()
        finally:
            if on_done is not None:
                on_done()

    threading.Thread(target=_run, daemon=True, name="rerun-review").start()
    return True


def _judge_all() -> None:
    cfg = review_settings()
    try:
        with session_scope(read_only=True) as session:
            queue = review_queue(session)
            todo = [(f["post"].id, _judge_inputs(f)) for f in queue["pending"]
                    if not (f["review"] and f["review"].verdict)]
        if cfg["max_per_run"] > 0:
            todo = todo[:cfg["max_per_run"]]
        with _job_lock:
            _job["total"] = len(todo)
        today = utcnow().date().isoformat()
        for post_id, inputs in todo:
            if not spend.within_budget():
                with _job_lock:
                    _job["error"] = "Stopped early: today's LLM budget is used up."
                break
            try:
                result = llm.judge_rerun(cfg["model"], today=today, **inputs)
            except Exception as exc:
                log.warning("Rerun judgment failed for post %s: %s", post_id, exc)
                with _job_lock:
                    _job["error"] = f"Some posts couldn't be judged: {exc}"[:300]
                continue
            if result["verdict"]:
                with session_scope() as session:
                    review = _review_row(session, post_id)
                    review.verdict = result["verdict"]
                    review.reason = result["reason"]
                    review.model = cfg["model"]
                    review.judged_at = utcnow()
            with _job_lock:
                _job["done"] += 1
    except Exception as exc:
        log.exception("Rerun review run failed")
        with _job_lock:
            _job["error"] = str(exc)[:300]
    finally:
        with _job_lock:
            _job["running"] = False


# --- Operator decisions -------------------------------------------------------

def _review_row(session, post_id: int) -> RerunReview:
    review = session.execute(
        select(RerunReview).where(RerunReview.post_pk == post_id)
    ).scalar_one_or_none()
    if review is None:
        review = RerunReview(post_pk=post_id)
        session.add(review)
    return review


def decide(session, post_id: int, decision: str) -> bool:
    """Record the operator's call. ``evergreen`` writes the post's shelf-life
    override so both rerun paths take it; the others only take it out of the
    review list. False when the post doesn't exist or the decision is unknown."""
    if decision not in DECISIONS:
        return False
    post = session.get(ThreadsPost, post_id)
    if post is None:
        return False
    review = _review_row(session, post_id)
    if decision == DECISION_EVERGREEN:
        if review.decision != DECISION_EVERGREEN:
            review.prior_override = post.shelf_life or ""
        post.shelf_life = SHELF_EVERGREEN
    elif review.decision == DECISION_EVERGREEN:
        post.shelf_life = review.prior_override or ""
    review.decision = decision
    review.decided_at = utcnow()
    invalidate_recycle_overview()
    return True


def undo(session, post_id: int) -> bool:
    """Send a decided post back to the review list, restoring the shelf-life
    override an ``evergreen`` decision replaced."""
    review = session.execute(
        select(RerunReview).where(RerunReview.post_pk == post_id)
    ).scalar_one_or_none()
    if review is None or not review.decision:
        return False
    if review.decision == DECISION_EVERGREEN:
        post = session.get(ThreadsPost, post_id)
        if post is not None and post.shelf_life == SHELF_EVERGREEN:
            post.shelf_life = review.prior_override or ""
    review.decision = ""
    review.decided_at = None
    review.prior_override = ""
    invalidate_recycle_overview()
    return True


def accept_evergreen_verdicts(session) -> int:
    """Apply every pending "re-airs as-is" verdict in one go."""
    ids = [f["post"].id for f in review_queue(session)["pending"]
           if f["review"] and f["review"].verdict == "evergreen"]
    for post_id in ids:
        decide(session, post_id, DECISION_EVERGREEN)
    return len(ids)
