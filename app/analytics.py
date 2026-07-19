"""Part 3 analytics: time-series metric snapshots + attribute slicing + digest."""
from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict

from sqlalchemy import func, select

from .config import load_settings
from .llm import write_digest
from .models import Candidate, MetricSnapshot, ThreadsComment, ThreadsPost, TraitWeight, utcnow
from .threads_api import fetch_insights

log = logging.getLogger("analytics")

METRICS = ["views", "likes", "replies", "reposts", "quotes", "shares"]


def snapshot_metrics(session) -> int:
    """Re-pull Threads insights for published posts that are due for a snapshot."""
    settings = load_settings()
    interval = dt.timedelta(hours=settings.get("analytics.snapshot_interval_hours", 12))
    max_age = dt.timedelta(days=settings.get("analytics.max_post_age_days", 60))
    now = utcnow()

    posts = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == "published")
    ).scalars().all()

    taken = 0
    for post in posts:
        published = post.published_at
        if published and published.tzinfo is None:
            published = published.replace(tzinfo=dt.timezone.utc)
        if published and now - published > max_age:
            continue
        last = session.execute(
            select(func.max(MetricSnapshot.captured_at)).where(MetricSnapshot.post_pk == post.id)
        ).scalar_one()
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


def _latest_metrics(session, post: ThreadsPost) -> dict:
    snap = session.execute(
        select(MetricSnapshot)
        .where(MetricSnapshot.post_pk == post.id)
        .order_by(MetricSnapshot.captured_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if snap is None:
        return {m: None for m in METRICS}
    return {m: getattr(snap, m) for m in METRICS}


def _comment_outcomes(session, post: ThreadsPost) -> dict:
    rows = session.execute(
        select(ThreadsComment.classification, func.count(ThreadsComment.id))
        .where(ThreadsComment.post_pk == post.id)
        .group_by(ThreadsComment.classification)
    ).all()
    by_class = {c: n for c, n in rows}
    replies_posted = session.execute(
        select(func.count(ThreadsComment.id)).where(
            ThreadsComment.post_pk == post.id, ThreadsComment.reply_status == "posted"
        )
    ).scalar_one()
    return {
        "supportive_comments": by_class.get("supportive", 0) + by_class.get("genuine_question", 0),
        "hostile_comments": by_class.get("hostile_or_argumentative", 0) + by_class.get("bait_or_trolling", 0),
        "renewables_replies_posted": replies_posted,
    }


def build_post_rows(session) -> list[dict]:
    """One flat row per published post: attributes + latest metrics + comment outcomes."""
    posts = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == "published").order_by(ThreadsPost.published_at.desc())
    ).scalars().all()

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
            "climate_topic": candidate.climate_topic if candidate else None,
            "matched_keywords": candidate.matched_keywords if candidate else None,
            "visual_traits": candidate.visual_traits if candidate else None,
            "visual_score": candidate.visual_score if candidate else None,
            "clip_length_seconds": post.clip_length_seconds,
            "caption_length": post.caption_length,
            "caption_has_question": post.caption_has_question,
            "caption_has_cta": post.caption_has_cta,
            "caption_tone": post.caption_tone,
            "caption_hashtag_count": post.caption_hashtag_count,
            "day_of_week": post.post_day_of_week,
            "hour_local": post.post_hour_local,
        }
        row.update(_latest_metrics(session, post))
        row.update(_comment_outcomes(session, post))
        rows.append(row)
    return rows


def slice_summaries(rows: list[dict]) -> dict:
    """Mean of each metric grouped by each attribute (simple correlational slices)."""
    attributes = [
        "region", "climate_topic", "visual_traits", "caption_tone", "day_of_week",
        "caption_has_question", "caption_has_cta",
    ]
    multi_value = {"climate_topic", "visual_traits"}
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


def learn_trait_weights(session, rows: list[dict] | None = None, metric: str = "views") -> list[dict]:
    """Self-improvement loop: recompute how each visual trait's posts perform
    vs. the overall average, and upsert TraitWeight rows. Ranking reads these to
    drift toward footage traits that correlate with more of ``metric``.

    Returns the per-trait summaries (also handy for the analytics UI). Purely
    correlational; ``ranking`` gates influence on sample size.
    """
    if rows is None:
        rows = build_post_rows(session)
    with_metric = [r for r in rows if r.get(metric) is not None]
    overall = _mean([r[metric] for r in with_metric])

    groups: dict[str, list] = defaultdict(list)
    for r in with_metric:
        for t in str(r.get("visual_traits") or "").split(","):
            t = t.strip()
            if t:
                groups[t].append(r[metric])

    existing = {
        w.trait: w for w in session.execute(
            select(TraitWeight).where(TraitWeight.metric == metric)
        ).scalars().all()
    }
    results = []
    for trait, vals in groups.items():
        avg = _mean(vals)
        lift = ((avg - overall) / overall) if (overall and avg is not None) else None
        row = existing.get(trait)
        if row is None:
            row = TraitWeight(trait=trait, metric=metric)
            session.add(row)
        row.n_posts = len(vals)
        row.avg_metric = avg
        row.overall_avg = overall
        row.lift = lift
        row.updated_at = utcnow()
        results.append({"trait": trait, "n_posts": len(vals), "avg_metric": avg,
                        "overall_avg": overall, "lift": lift, "metric": metric})
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
