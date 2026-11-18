"""This account's own hits and flops, as examples the ranking critic watches.

A generic "would this stop a scroll" judge doesn't know that this audience
rewards a recognizable person's charged line and scrolls past aerial
spectacle. Showing the ranker a few of the account's real hits (and clips the
operator labelled not gripping) next to the drafts teaches that in the one
language a video model reads well: video.

How examples are chosen, and why:

- Only first airings (no reposts, not the brand's own promos): a re-air's
  views say more about timing than about the clip.
- Views at a FIXED age (6 h — the scheduler polls densely until then), as a
  percentile among posts from the same ±30 days. "Latest" views compare a
  post polled for a month with one polled for six hours, and a pool-wide
  percentile mixes in months of follower growth.
- Hits come from the top decile, one per source video, picked by a shuffle
  seeded with the source being edited, so different videos see different
  hits instead of the same three celebrities every time.
- Flops are only clips the operator labelled "not gripping": a low view
  count alone can't separate a weak clip from a bad posting slot.

The pool is read in the caller's writer session (the caller rolls it back);
nothing here writes to the database. Example files are cut from the
pre-outro export when it is still on disk, else from the published copy with
the appended outro (end card) trimmed off by duration, file by file.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import engine_dir, media, opt, util

log = logging.getLogger("clip_engine.taste")

VIEWS_AGE_HOURS = 6
LOCAL_DAYS = 30            # percentile neighbourhood: posts within ±30 days
MIN_POOL = 10              # fewer eligible posts than this: no examples at all
_MIN_NEIGHBOURS = 5        # a sparser neighbourhood falls back to the whole population
_TOP_DECILE = 0.10
_OUTRO_MIN, _OUTRO_MAX = 2.0, 4.5   # an appended outro adds ~2.8 s
_WORDS_CHARS = 600


def _aware(when: dt.datetime | None) -> dt.datetime | None:
    if when is not None and when.tzinfo is None:
        return when.replace(tzinfo=dt.timezone.utc)
    return when


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _segments(raw: str) -> list[dict]:
    """Cut.trim_segments in play order; invalid entries dropped."""
    try:
        data = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    out = []
    for item in data if isinstance(data, list) else []:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            out.append({"start": start, "end": end})
    return out


def local_percentiles(points: list[tuple[Any, dt.datetime, float]], *,
                      days: int = LOCAL_DAYS) -> dict[Any, float]:
    """``{key: percentile}`` for ``(key, published_at, views)`` points: the
    share (0-100) of posts published within ±``days`` (the post included)
    that got FEWER views. A neighbourhood under 5 posts falls back to the
    whole population. "Top X%" is then ``100 - percentile`` (the best of 50
    is the top 2%)."""
    points = [(k, _aware(t), float(v)) for k, t, v in points]
    window = dt.timedelta(days=days)
    everyone = [v for _, _, v in points]
    out: dict[Any, float] = {}
    for key, when, views in points:
        near = [v for _, t, v in points if abs(t - when) <= window]
        if len(near) < _MIN_NEIGHBOURS:
            near = everyone
        below = sum(1 for v in near if v < views)
        out[key] = round(100.0 * below / len(near), 1)
    return out


def scored_first_airings(session, *, as_of: dt.datetime, lookback_days: int,
                         min_age_hours: float, exclude_candidate_pks=()) -> list[dict]:
    """First airings published in [as_of - lookback, as_of - min_age] with
    their views at 6 h and local percentile, as plain dicts, best first.

    First airing = the earliest published, non-repost post of its cut. The
    brand's own footage (``categories.is_first_party``) is left out of both
    the pool and the percentile neighbourhoods; posts of excluded candidates
    still count as neighbours (they are part of how the account performed),
    they just can't be picked. Shared with ``playbook.generate``."""
    from sqlalchemy import select

    from .. import analytics, categories
    from ..models import Candidate, Channel, Cut, ThreadsPost

    as_of = _aware(as_of)
    until = as_of - dt.timedelta(hours=float(min_age_hours))
    since = as_of - dt.timedelta(days=float(lookback_days))
    neighbours_since = since - dt.timedelta(days=LOCAL_DAYS)
    rows = session.execute(
        select(ThreadsPost.id, ThreadsPost.cut_pk, ThreadsPost.published_at, ThreadsPost.caption,
               ThreadsPost.clip_object_path, ThreadsPost.clip_local_path,
               ThreadsPost.format_tags, ThreadsPost.footage_traits,
               Cut.candidate_pk, Cut.clip_title, Cut.trim_segments,
               Cut.format_tags.label("cut_format_tags"),
               Cut.footage_traits.label("cut_footage_traits"),
               Candidate.title, Candidate.category, Channel.first_party)
        .join(Cut, ThreadsPost.cut_pk == Cut.id)
        .join(Candidate, Cut.candidate_pk == Candidate.id)
        .outerjoin(Channel, Candidate.channel_pk == Channel.id)
        .where(ThreadsPost.status == "published",
               ThreadsPost.cut_pk.is_not(None),
               ThreadsPost.repost_of_post_pk.is_(None),
               ThreadsPost.published_at.is_not(None),
               ThreadsPost.published_at <= until)
        .order_by(ThreadsPost.published_at, ThreadsPost.id)
    ).all()

    firsts: dict[int, Any] = {}
    for row in rows:
        firsts.setdefault(row.cut_pk, row)   # rows come oldest first
    population = []
    for row in firsts.values():
        published = _aware(row.published_at)
        if published < neighbours_since:
            continue
        source = SimpleNamespace(channel=SimpleNamespace(first_party=bool(row.first_party)),
                                 category=row.category or "")
        if categories.is_first_party(source):
            continue
        segments = _segments(row.trim_segments)
        if not segments:
            continue
        population.append((row, published, segments))
    if not population:
        return []
    probe = [SimpleNamespace(id=row.id, published_at=published)
             for row, published, _ in population]
    views = analytics.metrics_at_age_bulk(session, probe, "views", VIEWS_AGE_HOURS)
    measured = [(row, published, segments) for row, published, segments in population
                if views.get(row.id) is not None]
    percentiles = local_percentiles([(row.id, published, views[row.id])
                                     for row, published, _ in measured])
    excluded = {int(pk) for pk in exclude_candidate_pks or () if pk is not None}
    out = []
    for row, published, segments in measured:
        if published < since or row.candidate_pk in excluded:
            continue
        out.append({
            "post_pk": row.id,
            "cut_pk": row.cut_pk,
            "candidate_pk": row.candidate_pk,
            "title": (row.clip_title or row.title or "").strip(),
            "caption": row.caption or "",
            "views": int(views[row.id]),
            "percentile": percentiles[row.id],
            "segments": segments,
            "seconds": round(sum(s["end"] - s["start"] for s in segments), 1),
            "trim_segments": row.trim_segments or "",
            "format_tags": row.format_tags or row.cut_format_tags or "",
            "footage_traits": row.footage_traits or row.cut_footage_traits or "",
            "published_at": published.isoformat(),
            "clip_object_path": row.clip_object_path or "",
            "clip_local_path": row.clip_local_path or "",
        })
    out.sort(key=lambda p: (-p["percentile"], -p["views"], p["post_pk"]))
    return out


def words_heard(session, posts: list[dict]) -> dict[int, str]:
    """``{post_pk: words heard}`` (≤ 600 chars), sliced from each source's
    stored transcript — loaded only for these posts, since transcripts are
    the heaviest column in the table."""
    from sqlalchemy import select

    from ..models import Candidate
    from ..publishing import _slice_transcript

    ids = {p["candidate_pk"] for p in posts}
    if not ids:
        return {}
    texts = dict(session.execute(
        select(Candidate.id, Candidate.transcript_text).where(Candidate.id.in_(ids))).all())
    return {p["post_pk"]: _slice_transcript(texts.get(p["candidate_pk"]) or "",
                                            p.get("trim_segments") or "")[:_WORDS_CHARS]
            for p in posts}


def _one_per_candidate(posts: list[dict]) -> list[dict]:
    seen: set[int] = set()
    out = []
    for post in posts:
        if post["candidate_pk"] in seen:
            continue
        seen.add(post["candidate_pk"])
        out.append(post)
    return out


def _gripping_no(session, cut_pks: list[int], as_of: dt.datetime | None = None) -> set[int]:
    """The cuts among ``cut_pks`` labelled not gripping — with ``as_of``,
    only labels given (or last changed: a relabel overwrites the row) by
    then, so a replay of an earlier day never learns from a verdict the
    operator hadn't given yet."""
    from sqlalchemy import select

    from ..models import ContentLabel

    if not cut_pks:
        return set()
    query = select(ContentLabel.cut_pk).where(ContentLabel.label == ContentLabel.LABEL_GRIPPING,
                                              ContentLabel.value == "no",
                                              ContentLabel.cut_pk.in_(cut_pks))
    if as_of is not None:
        # UTC: the label times are written as UTC, and SQLite compares the
        # wall-clock digits, whatever zone the bound value says it is in.
        query = query.where(ContentLabel.updated_at <= _aware(as_of).astimezone(dt.timezone.utc))
    return set(session.execute(query).scalars().all())


def select_examples(session, *, as_of=None, exclude_candidate_pk=None, hits=3, flops=2,
                    lookback_days=180, min_age_hours=24, seed="") -> list[dict]:
    """Up to ``hits`` hits then up to ``flops`` flops (spec §4.9 shape).

    Hits: one per source, from the top decile by local percentile, chosen by
    a shuffle seeded with ``seed`` (callers pass the video id). Flops: posts
    whose cut is labelled gripping = "no", one per source, lowest percentile
    first (none when nothing is labelled) — with an explicit ``as_of`` (a
    replay), only labels given by then. [] when fewer than 10 posts are
    eligible."""
    labels_as_of = _aware(as_of)        # None: every label there is now
    as_of = _aware(as_of) or _now()
    pool = scored_first_airings(
        session, as_of=as_of, lookback_days=lookback_days, min_age_hours=min_age_hours,
        exclude_candidate_pks=() if exclude_candidate_pk is None else (exclude_candidate_pk,))
    if len(pool) < MIN_POOL:
        log.info("taste: only %d eligible posts before %s; no examples", len(pool), as_of.date())
        return []
    picked: list[dict] = []
    if hits > 0:
        decile = _one_per_candidate(pool)[:max(int(hits), math.ceil(len(pool) * _TOP_DECILE))]
        shuffled = sorted(decile, key=lambda p: util.sha1(seed, p["post_pk"]))
        chosen = sorted(shuffled[:int(hits)], key=lambda p: (-p["percentile"], -p["views"]))
        picked += [{**p, "label": "hit"} for p in chosen]
    if flops > 0:
        labelled = _gripping_no(session, [p["cut_pk"] for p in pool], as_of=labels_as_of)
        used = {p["candidate_pk"] for p in picked}
        worst = sorted((p for p in pool if p["cut_pk"] in labelled and p["candidate_pk"] not in used),
                       key=lambda p: (p["percentile"], p["views"], p["post_pk"]))
        picked += [{**p, "label": "flop"} for p in _one_per_candidate(worst)[:int(flops)]]
    words = words_heard(session, picked)
    keys = ("post_pk", "cut_pk", "candidate_pk", "title", "caption", "views", "percentile",
            "label", "segments", "seconds", "published_at", "clip_object_path", "clip_local_path")
    out = []
    for post in picked:
        example = {k: post[k] for k in keys}
        example["transcript"] = words.get(post["post_pk"], "")
        out.append(example)
    log.info("taste: %d hits, %d flops from %d posts (seed %s)",
             sum(1 for e in out if e["label"] == "hit"),
             sum(1 for e in out if e["label"] == "flop"), len(pool), seed)
    return out


def outro_trim(duration: float | None, segments) -> float:
    """Seconds to cut off a published copy's end: its duration minus the
    clip's own length, when that difference looks like the appended outro
    (2.0-4.5 s); else 0. Decided per file — re-uploads gave old posts the
    outro too, so the publish date says nothing."""
    try:
        total = float(duration)
    except (TypeError, ValueError):
        return 0.0
    length = sum(max(0.0, float(s["end"]) - float(s["start"])) for s in segments or [])
    extra = total - length
    return round(extra, 2) if _OUTRO_MIN <= extra <= _OUTRO_MAX else 0.0


def _tmp_name(dest: Path, tag: str) -> Path:
    return dest.with_name(f".{dest.stem}.{tag}.{os.getpid()}.{threading.get_ident()}.mp4")


def _transcode(src: Path, trim: float, dest: Path, height: int) -> Path:
    tmp = _tmp_name(dest, "tmp")
    try:
        made = Path(media.trim_tail(src, trim, tmp, height=height))
        os.replace(made, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def example_file(example, settings, *, height=360) -> Path | None:
    """A small video of the example, cached at engine_dir()/taste/<post>-<h>.mp4.

    From the local pre-outro export when it still exists (nothing to trim),
    else from the published copy in the bucket with the outro trimmed by
    ``outro_trim``. None on any failure (no copy, download or ffmpeg error).
    A cached file is reused through ``media.reuse_cached``, which leaves its
    mtime alone unless it is old enough to need keeping from the prune: the
    mtime is part of the Gemini upload cache's key, so refreshing it on every
    reuse uploaded the same example again on every ranking run."""
    try:
        post_pk = int(example["post_pk"])
        dest = engine_dir() / "taste" / f"{post_pk}-{int(height)}.mp4"
        if media.reuse_cached(dest):
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = str(example.get("clip_local_path") or "").strip()
        if local and Path(local).is_file():
            return _transcode(Path(local), 0.0, dest, int(height))
        key = str(example.get("clip_object_path") or "").strip()
        if not key:
            return None
        from .. import storage_supabase

        bucket = opt(settings, "storage.trimmed_clip_bucket", "trimmed-clips")
        download = _tmp_name(dest, "download")
        try:
            storage_supabase.download_to_file(bucket, key, download)
            trim = outro_trim(media.probe_duration(download), example.get("segments"))
            return _transcode(download, trim, dest, int(height))
        finally:
            download.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - an example is optional; the ranker goes without it
        log.warning("taste example for post %s unavailable: %s", example.get("post_pk"), exc)
        return None


def performance_line(example) -> str:
    """How the example did, as the ranker reads it."""
    if (example or {}).get("label") == "flop":
        return "you labelled this one not gripping"
    percentile = (example or {}).get("percentile")
    views = (example or {}).get("views")
    if percentile is None:
        text = "one of this account's best posts from the same weeks"
    else:
        top = max(1, int(round(100 - float(percentile))))
        text = f"top {top}% of this account's posts from the same weeks"
    if views is not None:
        text += f" ({int(views):,} views at {VIEWS_AGE_HOURS} h)"
    return text
