"""Self-healing for monitored channels.

After each monitor pass, every enabled channel is assessed against three
signals, and channels that fail one are *retired* (disabled, with the reason
kept on the row so the operator can see it and restore with one click):

  - broken:       YouTube said "not found" on several passes in a row
                  (deleted channel, dead handle, bad URL), or the row is a
                  duplicate of another channel.
  - unproductive: the operator kept rejecting its candidates without approving
                  any (counted since its last approved clip; channels with an
                  approval history get double the allowance).
  - silent:       no keyword-matching uploads at all for a long stretch.

For each retired channel the system then looks for a stand-in: an LLM
proposes channels for the same slot (market / scope), each proposal is
resolved against the YouTube API (filtering out hallucinated handles) and
trial-run over its recent uploads with the live keyword list. The best one
that clears the bar is added — on probation by the same rules as any other.

Thresholds live under ``channel_health`` in settings.yaml.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from sqlalchemy import case, func, or_, select

from . import youtube
from .llm import suggest_replacement_channels
from .matching import find_keyword_matches
from .models import STATUS_APPROVED, STATUS_ARCHIVED, STATUS_REJECTED, Candidate, Channel, utcnow

log = logging.getLogger("channel_health")

# Fraction of a limit at which the Channels page flags a channel "at risk".
_AT_RISK = 0.75


@dataclass
class Health:
    approved: int = 0
    misses: int = 0              # rejections since the last approved clip
    miss_limit: int = 0
    last_candidate_at: dt.datetime | None = None
    days_silent: int | None = None
    verdict: str = "ok"          # ok | at_risk | retire
    reason: str = ""


def _aware(value: dt.datetime | None) -> dt.datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=dt.timezone.utc)


def _cfg(settings, key: str, default):
    return settings.get(f"channel_health.{key}", default)


def is_monitored(channel: Channel) -> bool:
    """Real YouTube sources only — not the synthetic upload/pasted-URL owners."""
    return (channel.url or "").startswith("http")


def evidence_start(channel: Channel) -> dt.datetime:
    return _aware(channel.health_since) or _aware(channel.created_at) or utcnow()


_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def assess(session, settings, now: dt.datetime | None = None,
           channels: list[Channel] | None = None) -> dict[int, Health]:
    """Health for every enabled monitored channel, keyed by channel id.
    Two queries total — this runs on every Channels page load against a
    possibly remote database. Pass ``channels`` if already loaded."""
    now = now or utcnow()
    max_failures = int(_cfg(settings, "max_consecutive_failures", 4))
    base_misses = int(_cfg(settings, "max_rejections_without_approval", 12))
    silent_days = int(_cfg(settings, "max_days_without_match", 45))

    if channels is None:
        channels = session.execute(select(Channel).where(Channel.enabled)).scalars().all()
    channels = [c for c in channels if c.enabled and is_monitored(c)]

    picked = or_(Candidate.approved_at.isnot(None),
                 Candidate.status.in_([STATUS_APPROVED, STATUS_ARCHIVED]))
    stats = (
        select(Candidate.channel_pk.label("pk"),
               func.count(case((picked, 1))).label("approved"),
               func.max(case((picked, Candidate.published_at))).label("last_pick"),
               func.max(Candidate.published_at).label("last_any"))
        .group_by(Candidate.channel_pk)
        .subquery()
    )
    approvals = {pk: (n, _aware(lp), _aware(la))
                 for pk, n, lp, la in session.execute(select(stats)).all()}
    # Misses: rejections published after both the last approval and the last
    # restore. Each bound is compared separately to stay portable (no GREATEST
    # in SQLite).
    misses = dict(session.execute(
        select(Candidate.channel_pk, func.count())
        .join(Channel, Channel.id == Candidate.channel_pk)
        .outerjoin(stats, stats.c.pk == Candidate.channel_pk)
        .where(Candidate.status == STATUS_REJECTED,
               Candidate.published_at > func.coalesce(stats.c.last_pick, _EPOCH),
               Candidate.published_at > func.coalesce(Channel.health_since, _EPOCH))
        .group_by(Candidate.channel_pk)
    ).all())

    out: dict[int, Health] = {}
    for ch in channels:
        start = evidence_start(ch)
        n_approved, _, last_any = approvals.get(ch.id, (0, None, None))
        h = Health(
            approved=n_approved,
            misses=misses.get(ch.id, 0),
            miss_limit=base_misses * (2 if n_approved else 1),
            last_candidate_at=last_any,
        )
        quiet_since = max(start, h.last_candidate_at) if h.last_candidate_at else start
        h.days_silent = (now - quiet_since).days if ch.last_checked_at else None

        if ch.first_party:
            out[ch.id] = h
            continue
        if (ch.last_error or "").startswith("duplicate:"):
            h.verdict, h.reason = "retire", ch.last_error
        elif (ch.consecutive_failures or 0) >= max_failures:
            h.verdict = "retire"
            h.reason = (f"Broken: YouTube couldn't find this channel on "
                        f"{ch.consecutive_failures} passes in a row ({ch.last_error[:160]})")
        elif h.misses >= h.miss_limit:
            h.verdict = "retire"
            h.reason = (f"Unproductive: {h.misses} candidates rejected since its last approved clip"
                        if n_approved else
                        f"Unproductive: {h.misses} candidates rejected, none ever approved")
        elif h.days_silent is not None and h.days_silent >= silent_days:
            h.verdict = "retire"
            h.reason = (f"Silent: no keyword-matching uploads in {h.days_silent} days"
                        if ch.last_seen_published_at else
                        f"Silent: no uploads at all in {h.days_silent} days — likely the wrong "
                        f"or an abandoned account (resolves to “{ch.channel_title}”)")
        elif (h.misses >= h.miss_limit * _AT_RISK
              or (ch.consecutive_failures or 0) > 0
              or (h.days_silent is not None and h.days_silent >= silent_days * _AT_RISK)):
            h.verdict = "at_risk"
        out[ch.id] = h
    return out


def retire(channel: Channel, reason: str, now: dt.datetime | None = None) -> None:
    channel.enabled = False
    channel.retired_at = now or utcnow()
    channel.retired_reason = reason
    if reason.startswith("duplicate:"):
        # Nothing to replace: the channel it duplicates is still monitored.
        channel.replacement_searched_at = channel.retired_at


def restore(channel: Channel, now: dt.datetime | None = None) -> None:
    """Re-enable with a clean slate: evidence before now no longer counts."""
    channel.enabled = True
    channel.retired_at = None
    channel.retired_reason = ""
    channel.consecutive_failures = 0
    channel.last_error = ""
    channel.health_since = now or utcnow()


def retire_unhealthy(session, settings, now: dt.datetime | None = None) -> list[dict]:
    """Retire channels whose verdict is 'retire', worst first, up to the
    per-pass cap (a guard against one bad signal emptying the list at once)."""
    now = now or utcnow()
    cap = int(_cfg(settings, "max_retirements_per_pass", 10))
    health = assess(session, settings, now)
    doomed = [(cid, h) for cid, h in health.items() if h.verdict == "retire"]
    # Broken/duplicate first (certain), then by how far past the limit.
    doomed.sort(key=lambda item: (not item[1].reason.startswith(("duplicate", "Broken")),
                                  -(item[1].misses / max(item[1].miss_limit, 1))))
    retired = []
    for cid, h in doomed[:cap]:
        ch = session.get(Channel, cid)
        retire(ch, h.reason, now)
        retired.append({"id": ch.id, "call_sign": ch.call_sign, "reason": h.reason})
        log.info("Retired %s: %s", ch.call_sign, h.reason)
    return retired


def _trial(info: dict, keywords: list[str], days: int) -> tuple[int, int]:
    """(keyword-matching uploads, total uploads) over the last ``days``.
    Even proven producers match only a few percent of uploads, so the sample
    has to be deep: 150 uploads is ~3 days of a national outlet."""
    uploads = youtube.list_recent_uploads(
        info["uploads_playlist_id"], since=utcnow() - dt.timedelta(days=days), max_results=150,
    )
    hits = sum(1 for u in uploads if find_keyword_matches(f"{u.title}\n{u.description}", keywords))
    return hits, len(uploads)


def find_replacement(session, settings, retired: Channel, keywords: list[str]) -> Channel | None:
    """Search for, verify, and add one stand-in for ``retired``. Marks the
    search done either way so a miss isn't retried every pass."""
    retired.replacement_searched_at = utcnow()
    model = _cfg(settings, "discovery_model", None) or settings.get(
        "engagement.draft_model", "claude-sonnet-5")
    trial_days = int(_cfg(settings, "trial_days", 30))
    min_hits = int(_cfg(settings, "min_trial_matches", 2))

    all_channels = session.execute(select(Channel)).scalars().all()
    known_ids = {c.channel_id for c in all_channels if c.channel_id}
    known_urls = {(c.url or "").rstrip("/").lower() for c in all_channels}

    health = assess(session, settings)
    exemplars = sorted(
        ({"call_sign": c.call_sign, "network": c.network, "market": c.market,
          "scope": c.scope, "approved": health[c.id].approved}
         for c in all_channels if c.id in health and health[c.id].approved),
        key=lambda e: -e["approved"],
    )
    slot = {"call_sign": retired.call_sign, "network": retired.network,
            "market": retired.market, "region": retired.region,
            "country": retired.country, "scope": retired.scope,
            "reason": retired.retired_reason}
    existing = [f"{c.call_sign} {c.url}" for c in all_channels if is_monitored(c)]
    try:
        proposals = suggest_replacement_channels(model, slot, exemplars, existing)
    except ValueError:  # occasional empty/garbled reply; one retry
        proposals = suggest_replacement_channels(model, slot, exemplars, existing)

    best = None
    for p in proposals:
        if p["url"].rstrip("/").lower() in known_urls:
            continue
        try:
            info = youtube.resolve_channel(p["url"])
            if info["channel_id"] in known_ids:
                continue
            hits, total = _trial(info, keywords, trial_days)
        except youtube.YouTubeAPIError as exc:
            log.info("Replacement proposal %s rejected: %s", p["url"], exc)
            continue
        log.info("Trial %s for %s: %d/%d uploads matched", p["url"], retired.call_sign, hits, total)
        if hits >= min_hits and (best is None or hits > best[1]):
            best = (p, hits, total, info)

    if best is None:
        log.info("No replacement found for %s (%d proposals)", retired.call_sign, len(proposals))
        return None
    p, hits, total, info = best
    ch = Channel(
        call_sign=p["call_sign"] or info["title"][:40], network=p["network"],
        market=p["market"], region=p["region"], country=p["country"], scope=p["scope"],
        url=p["url"], channel_id=info["channel_id"],
        uploads_playlist_id=info["uploads_playlist_id"], channel_title=info["title"],
        replaces_pk=retired.id,
        added_note=(f"Auto-added to replace {retired.call_sign}: {hits} of its last {total} "
                    f"uploads ({trial_days} days) matched your keywords. {p['why']}").strip(),
    )
    session.add(ch)
    session.flush()
    log.info("Added %s (%s) to replace %s", ch.call_sign, ch.url, retired.call_sign)
    return ch


def find_replacements(session, settings, keywords: list[str]) -> list[dict]:
    cap = int(_cfg(settings, "max_replacements_per_pass", 3))
    pending = session.execute(
        select(Channel).where(Channel.retired_at.isnot(None),
                              Channel.replacement_searched_at.is_(None))
        .order_by(Channel.retired_at).limit(cap)
    ).scalars().all()
    added = []
    for retired in pending:
        try:
            ch = find_replacement(session, settings, retired, keywords)
        except Exception as exc:  # one failed search shouldn't stop the others
            log.warning("Replacement search for %s failed: %s", retired.call_sign, exc)
            retired.replacement_searched_at = utcnow()
            continue
        if ch is not None:
            added.append({"id": ch.id, "call_sign": ch.call_sign, "replaces": retired.call_sign})
        session.commit()
    return added


def heal(session, settings, keywords: list[str]) -> dict:
    """One self-healing step: retire unhealthy channels, then search for
    replacements (capped per pass; the backlog drains over later passes)."""
    if not _cfg(settings, "enabled", True):
        return {"retired": [], "added": []}
    retired = retire_unhealthy(session, settings)
    session.commit()
    added = []
    if _cfg(settings, "find_replacements", True):
        added = find_replacements(session, settings, keywords)
    return {"retired": retired, "added": added}
