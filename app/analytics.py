"""Part 3 analytics: time-series metric snapshots + attribute slicing + digest."""
from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from .config import load_settings
from .llm import write_digest
from .models import Candidate, MetricSnapshot, ThreadsComment, ThreadsPost, TraitWeight, utcnow
from .threads_api import fetch_insights

log = logging.getLogger("analytics")

METRICS = ["views", "likes", "replies", "reposts", "quotes", "shares"]


def _last_snapshot_times(session, post_ids: list[int]) -> dict[int, dt.datetime]:
    """post_pk -> newest snapshot capture time, in one grouped query."""
    if not post_ids:
        return {}
    rows = session.execute(
        select(MetricSnapshot.post_pk, func.max(MetricSnapshot.captured_at))
        .where(MetricSnapshot.post_pk.in_(post_ids))
        .group_by(MetricSnapshot.post_pk)
    ).all()
    return {pk: last for pk, last in rows if last is not None}


def snapshot_metrics(session) -> int:
    """Re-pull Threads insights for published posts that are due for a snapshot."""
    settings = load_settings()
    interval = dt.timedelta(hours=settings.get("analytics.snapshot_interval_hours", 12))
    max_age = dt.timedelta(days=settings.get("analytics.max_post_age_days", 60))
    now = utcnow()

    posts = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == "published")
    ).scalars().all()
    last_by_post = _last_snapshot_times(session, [p.id for p in posts])

    taken = 0
    for post in posts:
        published = post.published_at
        if published and published.tzinfo is None:
            published = published.replace(tzinfo=dt.timezone.utc)
        if published and now - published > max_age:
            continue
        last = last_by_post.get(post.id)
        if last is not None:
            if last.tzinfo is None:
                last = last.replace(tzinfo=dt.timezone.utc)
            if now - last < interval:
                continue

        data = fetch_insights(post.threads_media_id)
        if not data:
            continue
        session.add(
            MetricSnapshot(
                post_pk=post.id,
                views=data.get("views"),
                likes=data.get("likes"),
                replies=data.get("replies"),
                reposts=data.get("reposts"),
                quotes=data.get("quotes"),
                shares=data.get("shares"),
            )
        )
        taken += 1
    session.flush()
    log.info("Metric snapshots taken: %d", taken)
    return taken


def poll_recent_metrics(session) -> int:
    """Frequent insights pull for recently published posts (feeds hotness checks).

    Independent of the long-term ``snapshot_interval_hours`` cadence: only posts
    published within ``scheduler.metrics_poll_recency_hours`` are considered, and
    each is re-snapshotted at most every ``scheduler.metrics_poll_interval_minutes``.
    """
    settings = load_settings()
    interval = dt.timedelta(
        minutes=settings.get("scheduler.metrics_poll_interval_minutes", 15)
    )
    recency = dt.timedelta(
        hours=settings.get("scheduler.metrics_poll_recency_hours", 6)
    )
    now = utcnow()
    cutoff = now - recency

    posts = session.execute(
        select(ThreadsPost).where(
            ThreadsPost.status == "published",
            ThreadsPost.published_at.is_not(None),
            ThreadsPost.published_at >= cutoff,
            ThreadsPost.threads_media_id != "",
        ).order_by(ThreadsPost.published_at.desc())
    ).scalars().all()
    last_by_post = _last_snapshot_times(session, [p.id for p in posts])

    taken = 0
    for post in posts:
        last = last_by_post.get(post.id)
        if last is not None:
            if last.tzinfo is None:
                last = last.replace(tzinfo=dt.timezone.utc)
            if now - last < interval:
                continue
        try:
            data = fetch_insights(post.threads_media_id)
        except Exception as exc:
            log.warning("Recent metrics poll failed for post %s: %s", post.id, exc)
            continue
        if not data:
            continue
        session.add(
            MetricSnapshot(
                post_pk=post.id,
                views=data.get("views"),
                likes=data.get("likes"),
                replies=data.get("replies"),
                reposts=data.get("reposts"),
                quotes=data.get("quotes"),
                shares=data.get("shares"),
            )
        )
        taken += 1
    if taken:
        session.flush()
        log.info("Recent metric snapshots taken: %d", taken)
    return taken


def likes_delta_trailing(session, post: ThreadsPost, window_minutes: int) -> int | None:
    """Likes gained by ``post`` over the trailing ``window_minutes``.

    Diffs the latest MetricSnapshot against the newest snapshot at least
    ``window_minutes`` old (or the oldest available if the post is younger).
    Returns None when there aren't enough snapshots to compute a delta.
    """
    snaps = session.execute(
        select(MetricSnapshot)
        .where(MetricSnapshot.post_pk == post.id, MetricSnapshot.likes.is_not(None))
        .order_by(MetricSnapshot.captured_at.desc())
    ).scalars().all()
    if len(snaps) < 2:
        return None

    latest = snaps[0]
    latest_at = latest.captured_at
    if latest_at.tzinfo is None:
        latest_at = latest_at.replace(tzinfo=dt.timezone.utc)
    cutoff = latest_at - dt.timedelta(minutes=window_minutes)

    older = None
    for snap in snaps[1:]:
        at = snap.captured_at
        if at.tzinfo is None:
            at = at.replace(tzinfo=dt.timezone.utc)
        older = snap
        if at <= cutoff:
            break

    if older is None or older.likes is None or latest.likes is None:
        return None
    return max(0, int(latest.likes) - int(older.likes))


def is_last_post_hot(session) -> tuple[bool, int | None]:
    """Whether the most recently published post is 'hot' per scheduler settings.

    Returns ``(is_hot, likes_delta)``. When there is no published post or not
    enough snapshot history, treats as not hot (``False, None``) so the queue
    can proceed rather than stall forever on missing data.
    """
    settings = load_settings()
    threshold = int(settings.get("scheduler.hot.threshold", 100))
    window_minutes = int(settings.get("scheduler.hot.window_minutes", 60))
    # metric is fixed to likes for v1; settings keep the knob for later.

    post = session.execute(
        select(ThreadsPost).where(
            ThreadsPost.status == "published",
            ThreadsPost.published_at.is_not(None),
        ).order_by(ThreadsPost.published_at.desc()).limit(1)
    ).scalar_one_or_none()
    if post is None:
        return False, None

    delta = likes_delta_trailing(session, post, window_minutes)
    if delta is None:
        return False, None
    return delta > threshold, delta


def _latest_metrics_bulk(session, post_ids: list[int]) -> dict[int, dict]:
    """post_pk -> {metric: value} from each post's newest snapshot.

    One window-function query instead of one query per post — essential against
    a remote DB, where per-post lookups cost a network round trip each.
    """
    if not post_ids:
        return {}
    rn = (
        func.row_number()
        .over(partition_by=MetricSnapshot.post_pk,
              order_by=MetricSnapshot.captured_at.desc())
        .label("rn")
    )
    sub = (
        select(MetricSnapshot.post_pk,
               *[getattr(MetricSnapshot, m) for m in METRICS], rn)
        .where(MetricSnapshot.post_pk.in_(post_ids))
        .subquery()
    )
    rows = session.execute(select(sub).where(sub.c.rn == 1)).all()
    return {row.post_pk: {m: getattr(row, m) for m in METRICS} for row in rows}


def _comment_outcomes_bulk(session, post_ids: list[int]) -> dict[int, dict]:
    """post_pk -> comment-outcome counts, from two grouped queries total."""
    if not post_ids:
        return {}
    class_rows = session.execute(
        select(ThreadsComment.post_pk, ThreadsComment.classification,
               func.count(ThreadsComment.id))
        .where(ThreadsComment.post_pk.in_(post_ids))
        .group_by(ThreadsComment.post_pk, ThreadsComment.classification)
    ).all()
    posted_rows = session.execute(
        select(ThreadsComment.post_pk, func.count(ThreadsComment.id))
        .where(ThreadsComment.post_pk.in_(post_ids),
               ThreadsComment.reply_status == "posted")
        .group_by(ThreadsComment.post_pk)
    ).all()
    posted = {pk: n for pk, n in posted_rows}

    by_post: dict[int, dict[str, int]] = defaultdict(dict)
    for pk, classification, n in class_rows:
        by_post[pk][classification] = n
    out: dict[int, dict] = {}
    for pk in post_ids:
        by_class = by_post.get(pk, {})
        out[pk] = {
            "supportive_comments": by_class.get("supportive", 0) + by_class.get("genuine_question", 0),
            "hostile_comments": by_class.get("hostile_or_argumentative", 0) + by_class.get("bait_or_trolling", 0),
            "renewables_replies_posted": posted.get(pk, 0),
        }
    return out


_EMPTY_METRICS = {m: None for m in METRICS}
_EMPTY_OUTCOMES = {"supportive_comments": 0, "hostile_comments": 0,
                   "renewables_replies_posted": 0}


def build_post_rows(session) -> list[dict]:
    """One flat row per published post: attributes + latest metrics + comment outcomes."""
    posts = session.execute(
        select(ThreadsPost)
        .options(selectinload(ThreadsPost.candidate).selectinload(Candidate.channel))
        .where(ThreadsPost.status == "published")
        .order_by(ThreadsPost.published_at.desc())
    ).scalars().all()

    post_ids = [p.id for p in posts]
    metrics_by_post = _latest_metrics_bulk(session, post_ids)
    outcomes_by_post = _comment_outcomes_bulk(session, post_ids)

    rows = []
    for post in posts:
        candidate: Candidate | None = post.candidate
        row = {
            "post_id": post.threads_media_id,
            "permalink": post.permalink,
            "caption": post.caption[:120],
            "published_at": post.published_at.isoformat() if post.published_at else None,
            # Attributes
            "market": candidate.channel.market if candidate else None,
            "region": candidate.channel.region if candidate else None,
            "station": candidate.channel.call_sign if candidate else None,
            "matched_keywords": candidate.matched_keywords if candidate else None,
            "visual_traits": candidate.visual_traits if candidate else None,
            "visual_score": candidate.visual_score if candidate else None,
            "footage_traits": post.footage_traits,
            "clip_length_seconds": post.clip_length_seconds,
            "caption_length": post.caption_length,
            "caption_has_question": post.caption_has_question,
            "caption_has_cta": post.caption_has_cta,
            "caption_tone": post.caption_tone,
            "caption_hashtag_count": post.caption_hashtag_count,
            "day_of_week": post.post_day_of_week,
            "hour_local": post.post_hour_local,
        }
        row.update(metrics_by_post.get(post.id, _EMPTY_METRICS))
        row.update(outcomes_by_post.get(post.id, _EMPTY_OUTCOMES))
        rows.append(row)
    return rows


def slice_summaries(rows: list[dict]) -> dict:
    """Mean of each metric grouped by each attribute (simple correlational slices)."""
    attributes = [
        "region", "matched_keywords", "visual_traits", "footage_traits",
        "caption_tone", "day_of_week",
        "caption_has_question", "caption_has_cta",
    ]
    multi_value = {"matched_keywords", "visual_traits", "footage_traits"}
    summaries: dict = {}
    for attr in attributes:
        groups: dict = defaultdict(list)
        for row in rows:
            key = row.get(attr)
            if key is None:
                continue
            if attr in multi_value:
                # CSV of tags: count the post under every tag it carries.
                for t in str(key).split(","):
                    if t.strip():
                        groups[t.strip()].append(row)
                continue
            groups[str(key)].append(row)
        attr_summary = {}
        for key, grp in groups.items():
            metric_means = {}
            for m in METRICS:
                vals = [r[m] for r in grp if r.get(m) is not None]
                metric_means[m] = round(sum(vals) / len(vals), 1) if vals else None
            attr_summary[key] = {"n_posts": len(grp), **metric_means}
        summaries[attr] = attr_summary
    return summaries


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def daily_timeseries(rows: list[dict], window: int = 7) -> dict:
    """Group posts by publish date across the full calendar span, then compute
    per-day figures plus a trailing ``window``-day rolling average.

    Rolling per-post metrics pool every post published in the trailing window
    (not an average-of-daily-averages), which is the honest reading when posting
    cadence is uneven. Returns arrays aligned to a contiguous list of days.
    """
    dated: list[tuple[dt.date, dict]] = []
    for r in rows:
        pub = r.get("published_at")
        if not pub:
            continue
        try:
            day = dt.date.fromisoformat(pub[:10])
        except ValueError:
            continue
        dated.append((day, r))

    empty = {"days": [], "posts_per_day": [], "rolling_posts_7d": [],
             "metrics": {m: {"daily_avg": [], "rolling_7d": []} for m in METRICS},
             "window": window}
    if not dated:
        return empty

    by_date: dict[dt.date, list[dict]] = defaultdict(list)
    for day, r in dated:
        by_date[day].append(r)
    start, end = min(by_date), max(by_date)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]

    posts_per_day = [len(by_date.get(d, [])) for d in days]
    rolling_posts = [
        round(sum(posts_per_day[max(0, i - window + 1): i + 1]) / min(i + 1, window), 2)
        for i in range(len(days))
    ]

    metrics_out: dict = {}
    for m in METRICS:
        daily_avg, rolling = [], []
        for i, d in enumerate(days):
            daily_avg.append(_mean([r.get(m) for r in by_date.get(d, [])]))
            pool: list = []
            for j in range(max(0, i - window + 1), i + 1):
                pool.extend(r.get(m) for r in by_date.get(days[j], []))
            rolling.append(_mean(pool))
        metrics_out[m] = {"daily_avg": daily_avg, "rolling_7d": rolling}

    return {
        "days": [d.isoformat() for d in days],
        "posts_per_day": posts_per_day,
        "rolling_posts_7d": rolling_posts,
        "metrics": metrics_out,
        "window": window,
    }


def summary_kpis(rows: list[dict]) -> dict:
    """Headline totals + per-post averages across all posts with metrics."""
    with_metrics = [r for r in rows if r.get("views") is not None]
    kpi = {"total_posts": len(rows), "posts_with_metrics": len(with_metrics)}
    for m in METRICS:
        vals = [r[m] for r in rows if r.get(m) is not None]
        kpi[f"total_{m}"] = sum(vals) if vals else 0
        kpi[f"avg_{m}"] = _mean(vals)
    return kpi


def _metric_at_age_from_series(
    series: list[tuple[dt.datetime, int]],
    published: dt.datetime | None,
    age_hours: int,
) -> int | None:
    """Value when the post was ``age_hours`` old, from its (captured_at, value)
    series in ascending capture order. Comparing every post at the same age
    keeps old posts (with months of accumulated views) from always beating
    young ones.

    Uses the newest snapshot captured at or before the target age; when
    snapshots only start later (or the post is still younger than the target),
    falls back to the closest one available.
    """
    if not series or published is None:
        return None
    if published.tzinfo is None:
        published = published.replace(tzinfo=dt.timezone.utc)
    target = published + dt.timedelta(hours=age_hours)

    best = None
    for at, value in series:
        if at.tzinfo is None:
            at = at.replace(tzinfo=dt.timezone.utc)
        if at <= target:
            best = value
        else:
            break
    return best if best is not None else series[0][1]


def metrics_at_age_bulk(session, posts: list[ThreadsPost], metric: str,
                        age_hours: int) -> dict[int, int]:
    """post.id -> ``metric`` value at fixed age, for all posts in ONE query
    (instead of one snapshot-series query per post)."""
    ids = [p.id for p in posts]
    if not ids:
        return {}
    col = getattr(MetricSnapshot, metric)
    rows = session.execute(
        select(MetricSnapshot.post_pk, MetricSnapshot.captured_at, col)
        .where(MetricSnapshot.post_pk.in_(ids), col.is_not(None))
        .order_by(MetricSnapshot.post_pk, MetricSnapshot.captured_at.asc())
    ).all()
    series: dict[int, list[tuple[dt.datetime, int]]] = defaultdict(list)
    for pk, at, value in rows:
        series[pk].append((at, value))

    out: dict[int, int] = {}
    for post in posts:
        value = _metric_at_age_from_series(
            series.get(post.id, []), post.published_at, age_hours
        )
        if value is not None:
            out[post.id] = value
    return out


def metric_at_age(session, post: ThreadsPost, metric: str, age_hours: int) -> int | None:
    """Single-post convenience wrapper around the bulk fixed-age lookup."""
    return metrics_at_age_bulk(session, [post], metric, age_hours).get(post.id)


def _weighted_median(pairs: list[tuple[float, float]]) -> float | None:
    """Median of (value, weight) pairs; the value where cumulative weight
    crosses half the total."""
    pairs = [(v, w) for v, w in pairs if v is not None and w > 0]
    if not pairs:
        return None
    pairs.sort(key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return float(v)
    return float(pairs[-1][0])


def learn_trait_weights(session, rows: list[dict] | None = None, metric: str = "views") -> list[dict]:
    """Self-improvement loop: recompute each footage trait's performance verdict
    from the operator's own published posts, and upsert TraitWeight rows.

    Design (all knobs under ``learning.*`` in settings):
    - Trains on post-level ``footage_traits`` (annotated from the posted clip),
      never on pre-download predictions.
    - Every post's metric is read at the same fixed age (``metric_age_hours``)
      so old and new posts are comparable.
    - Lift is measured against the account's recency-weighted MEDIAN (one viral
      post shouldn't define the baseline), with posts decaying at
      ``halflife_days`` so verdicts drift with the audience.
    - Threshold-gated verdict ``status``: influence requires BOTH
      ``min_total_posts`` account-wide AND ``min_trait_posts`` observations of
      the trait. Below that, traits just collect data.

    Returns per-trait summaries for the UI. Purely correlational.
    """
    del rows  # legacy arg; verdicts need the snapshot series, not latest rows
    settings = load_settings()
    min_total = int(settings.get("learning.min_total_posts", 100))
    min_trait = int(settings.get("learning.min_trait_posts", 20))
    provisional_frac = float(settings.get("learning.provisional_fraction", 0.5))
    halflife_days = float(settings.get("learning.halflife_days", 90))
    age_hours = int(settings.get("learning.metric_age_hours", 48))

    posts = session.execute(
        select(ThreadsPost).where(
            ThreadsPost.status == "published",
            ThreadsPost.published_at.is_not(None),
        )
    ).scalars().all()

    now = utcnow()
    values_by_post = metrics_at_age_bulk(session, posts, metric, age_hours)
    observations: list[dict] = []  # one per post with a usable metric value
    for post in posts:
        value = values_by_post.get(post.id)
        if value is None:
            continue
        published = post.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=dt.timezone.utc)
        age_days = max(0.0, (now - published).total_seconds() / 86400)
        weight = 0.5 ** (age_days / halflife_days) if halflife_days > 0 else 1.0
        traits = [t.strip() for t in (post.footage_traits or "").split(",") if t.strip()]
        observations.append({"value": float(value), "weight": weight, "traits": traits})

    total_n = len(observations)
    baseline = _weighted_median([(o["value"], o["weight"]) for o in observations])
    overall_mean = _mean([o["value"] for o in observations])

    groups: dict[str, list[dict]] = defaultdict(list)
    for o in observations:
        for t in o["traits"]:
            groups[t].append(o)

    existing = {
        w.trait: w for w in session.execute(
            select(TraitWeight).where(TraitWeight.metric == metric)
        ).scalars().all()
    }
    results = []
    for trait, obs in groups.items():
        n = len(obs)
        eff_n = sum(o["weight"] for o in obs)
        median = _weighted_median([(o["value"], o["weight"]) for o in obs])
        avg = _mean([o["value"] for o in obs])
        lift = ((median - baseline) / baseline) if (baseline and median is not None) else None

        if total_n >= min_total and n >= min_trait and lift is not None:
            status = TraitWeight.STATUS_ACTIVE
        elif n >= max(1, int(min_trait * provisional_frac)):
            status = TraitWeight.STATUS_PROVISIONAL
        else:
            status = TraitWeight.STATUS_COLLECTING

        row = existing.pop(trait, None)
        if row is None:
            row = TraitWeight(trait=trait, metric=metric)
            session.add(row)
        row.n_posts = n
        row.effective_n = round(eff_n, 2)
        row.avg_metric = avg
        row.overall_avg = overall_mean
        row.median_metric = median
        row.baseline = baseline
        row.lift = lift
        row.status = status
        row.updated_at = utcnow()
        results.append({"trait": trait, "n_posts": n, "effective_n": round(eff_n, 2),
                        "avg_metric": avg, "overall_avg": overall_mean,
                        "median_metric": median, "baseline": baseline,
                        "lift": lift, "status": status, "metric": metric})

    # Traits that vanished from the data (e.g. annotations re-run) must not keep
    # a stale verdict.
    for row in existing.values():
        row.n_posts = 0
        row.effective_n = 0.0
        row.lift = None
        row.status = TraitWeight.STATUS_COLLECTING
        row.updated_at = utcnow()

    session.flush()
    results.sort(key=lambda d: (d["lift"] if d["lift"] is not None else -99), reverse=True)
    return results


def generate_report(session) -> dict:
    """Full analytics payload: per-post rows, attribute slices, and LLM digest."""
    settings = load_settings()
    rows = build_post_rows(session)
    slices = slice_summaries(rows)
    timeseries = daily_timeseries(rows)
    summary = summary_kpis(rows)
    trait_weights = learn_trait_weights(session, rows)
    payload = {
        "total_posts": len(rows),
        "posts": rows,
        "attribute_slices": slices,
        "visual_trait_performance": trait_weights,
        "note": "All slice comparisons are correlational; small samples likely.",
    }
    digest = ""
    if rows:
        try:
            digest = write_digest(
                settings.get("analytics.digest_model", "claude-sonnet-5"),
                payload,
                settings.get("analytics.min_sample_size", 8),
            )
        except Exception as exc:
            log.warning("Digest generation failed: %s", exc)
            digest = f"(digest generation failed: {exc})"
    return {"rows": rows, "slices": slices, "digest": digest,
            "timeseries": timeseries, "summary": summary,
            "trait_weights": trait_weights}
