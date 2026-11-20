"""Turn one workspace's database into what the admin pages show.

Everything here reads; nothing writes. Each part is fenced off so one broken
query shows as a gap on the page rather than a blank site.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics
from collections import Counter
from typing import Any

from sqlalchemy import func, select

from ..models import (AppToken, Candidate, Channel, ClipProposal, InboxItem, InstagramPost,
                      MetricSnapshot, MonitorRun, SchedulerState, ThreadsPost)
from .sources import Workspace, session

log = logging.getLogger("admin.collect")

# How long without a heartbeat before a worker counts as down.
STALE = {"scheduler": dt.timedelta(minutes=10), "home-worker": dt.timedelta(minutes=45)}
TOKEN_ROWS = ("threads", "instagram", "facebook", "linkedin")   # X rotates hourly by design
OVERDUE_AFTER = dt.timedelta(minutes=30)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(t: dt.datetime | None) -> dt.datetime | None:
    if t is None:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _count(s, stmt) -> int:
    return int(s.execute(stmt).scalar_one() or 0)


def _part(out: dict, name: str, fn, *args) -> None:
    try:
        out[name] = fn(*args)
    except Exception as exc:
        log.exception("Admin: %s failed", name)
        out.setdefault("errors", []).append(f"{name}: {str(exc)[:160]}")
        out[name] = None


# --- Parts ------------------------------------------------------------------------

def _heartbeats(s) -> list[dict]:
    now = _now()
    out = []
    for row in s.execute(select(AppToken).where(AppToken.name.like("hb:%"))).scalars():
        try:
            data = json.loads(row.value or "{}")
        except ValueError:
            continue
        at = _aware(dt.datetime.fromisoformat(data["at"])) if data.get("at") else None
        role = data.get("role", "")
        limit = STALE.get(role, dt.timedelta(minutes=30))
        out.append({**data, "at": at, "age": (now - at) if at else None,
                    "alive": bool(at and now - at < limit)})
    return sorted(out, key=lambda h: (h.get("role", ""), h.get("host", "")))


def _monitor(s) -> dict | None:
    run = s.execute(select(MonitorRun).order_by(MonitorRun.started_at.desc()).limit(1)).scalar()
    if run is None:
        return None
    last_ok = s.execute(select(func.max(MonitorRun.finished_at))
                        .where(MonitorRun.status == MonitorRun.STATUS_DONE)).scalar()
    return {"status": run.status, "started_at": _aware(run.started_at),
            "finished_at": _aware(run.finished_at), "found": run.candidates_stored,
            "checked": run.channels_checked, "error": (run.error or "")[:300],
            "last_ok": _aware(last_ok)}


def _scheduler(s) -> dict | None:
    st = s.get(SchedulerState, 1)
    if st is None:
        return None
    return {"updated_at": _aware(st.updated_at), "last_publish_at": _aware(st.last_publish_at),
            "last_action": st.last_action}


def _publishing(s) -> dict:
    now = _now()
    failed = s.execute(select(ThreadsPost).where(ThreadsPost.status == "failed",
                                                 ThreadsPost.attention_dismissed_at.is_(None))
                       .order_by(ThreadsPost.updated_at.desc()).limit(8)).scalars().all()
    overdue = _count(s, select(func.count(ThreadsPost.id)).where(
        ThreadsPost.status == "queued", ThreadsPost.scheduled_at < now - OVERDUE_AFTER))
    ig_failed = _count(s, select(func.count(InstagramPost.id)).where(InstagramPost.status == "failed"))
    queued = _count(s, select(func.count(ThreadsPost.id)).where(ThreadsPost.status == "queued"))
    last = s.execute(select(func.max(ThreadsPost.published_at))).scalar()
    return {"failed": [{"id": p.id, "at": _aware(p.updated_at), "error": (p.error or "")[:200],
                        "name": p.calendar_name or (p.caption or "")[:60]} for p in failed],
            "failed_count": _count(s, select(func.count(ThreadsPost.id)).where(
                ThreadsPost.status == "failed", ThreadsPost.attention_dismissed_at.is_(None))),
            "ig_failed": ig_failed, "overdue": overdue, "queued": queued,
            "last_published_at": _aware(last)}


def _tokens(s) -> list[dict]:
    now = _now()
    out = []
    for name in TOKEN_ROWS:
        row = s.get(AppToken, name)
        if row is None or not row.value:
            continue
        try:
            data = json.loads(row.value)
            # A token that renews itself lives as long as its refresh token
            # (LinkedIn); one without an expiry never expires (a Facebook Page).
            if data.get("refresh_token_expires_in"):
                start, life = data.get("refresh_obtained_at") or data["obtained_at"], \
                    data["refresh_token_expires_in"]
            elif data.get("expires_in"):
                start, life = data["obtained_at"], data["expires_in"]
            else:
                out.append({"name": name, "expires": None, "days_left": None, "never": True})
                continue
            expires = _aware(dt.datetime.fromisoformat(start)) + dt.timedelta(seconds=int(life))
        except (ValueError, KeyError, TypeError):
            out.append({"name": name, "expires": None, "days_left": None})
            continue
        # Only the expiry is read; the token itself never leaves this function.
        out.append({"name": name, "expires": expires,
                    "days_left": round((expires - now).total_seconds() / 86400, 1)})
    return out


def _pipeline(s, days: int = 7) -> dict:
    since = _now() - dt.timedelta(days=days)
    return {
        "days": days,
        "found": _count(s, select(func.count(Candidate.id)).where(Candidate.created_at >= since)),
        "approved": _count(s, select(func.count(Candidate.id)).where(Candidate.approved_at >= since)),
        "downloaded": _count(s, select(func.count(Candidate.id)).where(Candidate.archived_at >= since)),
        "drafted": _count(s, select(func.count(ClipProposal.id)).where(ClipProposal.created_at >= since)),
        "inbox_made": _count(s, select(func.count(InboxItem.id)).where(InboxItem.created_at >= since)),
        "inbox_waiting": _count(s, select(func.count(InboxItem.id)).where(
            InboxItem.state == InboxItem.STATE_READY)),
        "inbox_failed": _count(s, select(func.count(InboxItem.id)).where(
            InboxItem.state == InboxItem.STATE_FAILED, InboxItem.created_at >= since)),
        "published": _count(s, select(func.count(ThreadsPost.id)).where(
            ThreadsPost.published_at >= since)),
        "channels": _count(s, select(func.count(Channel.id)).where(
            Channel.enabled.is_(True), ~Channel.url.like("upload://%"), ~Channel.url.like("youtube://%"))),
        "retired": _count(s, select(func.count(Channel.id)).where(Channel.retired_at.is_not(None))),
        "oldest_waiting": _aware(s.execute(select(func.min(InboxItem.ready_at)).where(
            InboxItem.state == InboxItem.STATE_READY)).scalar()),
    }


def _decisions(s, days: int = 30) -> dict:
    since = _now() - dt.timedelta(days=days)
    items = s.execute(select(InboxItem).where(InboxItem.decided_at >= since)).scalars().all()
    approved = [i for i in items if i.state in (InboxItem.STATE_APPROVED, InboxItem.STATE_QUEUED)
                or (i.post_pk and i.state != InboxItem.STATE_REJECTED)]
    rejected = [i for i in items if i.state == InboxItem.STATE_REJECTED]
    reasons = Counter(t.strip() for i in rejected for t in (i.reject_tags or "").split(",") if t.strip())
    waits = [(_aware(i.decided_at) - _aware(i.ready_at)).total_seconds() / 3600
             for i in items if i.decided_at and i.ready_at]
    edited = sum(1 for i in approved if i.final_caption and i.caption
                 and i.final_caption.strip() != i.caption.strip())
    notes = [{"at": _aware(i.decided_at), "tags": i.reject_tags, "note": i.reject_note[:200]}
             for i in sorted(rejected, key=lambda i: i.decided_at, reverse=True) if i.reject_note][:8]
    return {"days": days, "approved": len(approved), "rejected": len(rejected),
            "approve_rate": (len(approved) / len(items)) if items else None,
            "reasons": reasons.most_common(8), "notes": notes,
            "median_hours": round(statistics.median(waits), 1) if waits else None,
            "captions_edited": edited}


def _results(s, days: int = 30) -> dict:
    since = _now() - dt.timedelta(days=days)
    posts = s.execute(select(ThreadsPost).where(ThreadsPost.published_at >= since)
                      .order_by(ThreadsPost.published_at)).scalars().all()
    if not posts:
        return {"days": days, "posts": 0, "views": 0, "likes": 0, "replies": 0, "top": [], "daily": []}
    ids = [p.id for p in posts]
    latest = s.execute(select(MetricSnapshot.post_pk, func.max(MetricSnapshot.captured_at))
                       .where(MetricSnapshot.post_pk.in_(ids)).group_by(MetricSnapshot.post_pk)).all()
    want = {(pk, at) for pk, at in latest}
    snaps = {m.post_pk: m for m in s.execute(select(MetricSnapshot).where(
        MetricSnapshot.post_pk.in_(ids))).scalars() if (m.post_pk, m.captured_at) in want}
    rows, daily = [], Counter()
    for p in posts:
        m = snaps.get(p.id)
        views = (m.views or 0) if m else 0
        rows.append({"id": p.id, "name": p.calendar_name or (p.caption or "")[:70],
                     "published_at": _aware(p.published_at), "permalink": p.permalink,
                     "views": views, "likes": (m.likes or 0) if m else 0,
                     "replies": (m.replies or 0) if m else 0})
        daily[_aware(p.published_at).date()] += views
    start = since.date()
    series = [{"day": start + dt.timedelta(days=n), "views": daily.get(start + dt.timedelta(days=n), 0)}
              for n in range(days + 1)]
    return {"days": days, "posts": len(rows), "views": sum(r["views"] for r in rows),
            "likes": sum(r["likes"] for r in rows), "replies": sum(r["replies"] for r in rows),
            "top": sorted(rows, key=lambda r: -r["views"])[:6], "daily": series}


# --- Health ------------------------------------------------------------------------

def _health(snap: dict) -> tuple[str, list[str]]:
    """("ok" | "warn" | "bad", reasons), worst first."""
    bad, warn = [], []
    now = _now()
    if snap.get("errors") and not snap.get("pipeline"):
        return "bad", ["Couldn't read the database: " + snap["errors"][0]]
    beats = snap.get("heartbeats") or []
    for role in ("scheduler", "home-worker"):
        mine = [h for h in beats if h.get("role") == role]
        if mine and not any(h["alive"] for h in mine):
            bad.append(f"{role.replace('-', ' ').capitalize()} hasn't checked in since "
                       f"{_ago(max(h['at'] for h in mine if h['at']))}")
        elif not mine:
            warn.append(f"No heartbeat from the {role.replace('-', ' ')} yet")
    pub = snap.get("publishing") or {}
    if pub.get("failed_count"):
        bad.append(f"{pub['failed_count']} post{'s' if pub['failed_count'] != 1 else ''} failed to publish")
    if pub.get("ig_failed"):
        warn.append(f"{pub['ig_failed']} Instagram post{'s' if pub['ig_failed'] != 1 else ''} failed")
    if pub.get("overdue"):
        bad.append(f"{pub['overdue']} post{'s' if pub['overdue'] != 1 else ''} overdue")
    mon = snap.get("monitor")
    if mon is None:
        warn.append("The monitor has never run")
    else:
        if mon["status"] == MonitorRun.STATUS_FAILED:
            bad.append("The last monitor run failed")
        if not mon.get("last_ok") or now - mon["last_ok"] > dt.timedelta(hours=14):
            warn.append(f"No successful monitor run since {_ago(mon.get('last_ok'))}")
    for t in snap.get("tokens") or []:
        if t["days_left"] is not None and t["days_left"] < 0:
            bad.append(f"{t['name'].capitalize()} sign-in expired")
        elif t["days_left"] is not None and t["days_left"] < 7:
            warn.append(f"{t['name'].capitalize()} sign-in expires in {t['days_left']:.0f} days")
    pipe = snap.get("pipeline") or {}
    if pipe.get("oldest_waiting") and now - pipe["oldest_waiting"] > dt.timedelta(days=2):
        warn.append(f"{pipe['inbox_waiting']} clips waiting in the inbox since {_ago(pipe['oldest_waiting'])}")
    if pipe and pipe.get("found", 0) and not pipe.get("published"):
        warn.append("Videos are coming in but nothing was published this week")
    for m in snap.get("machines") or []:
        if m.get("state") not in ("started", "stopped", "suspended"):
            bad.append(f"Fly {m['app']} machine is {m.get('state')}")
    for e in snap.get("errors") or []:
        warn.append(f"Couldn't read {e.split(':')[0]}")
    return ("bad" if bad else "warn" if warn else "ok"), bad + warn


def _ago(t: dt.datetime | None) -> str:
    if t is None:
        return "never"
    secs = (_now() - t).total_seconds()
    if secs < 90:
        return "just now"
    if secs < 5400:
        return f"{secs / 60:.0f} min ago"
    if secs < 172800:
        return f"{secs / 3600:.0f} h ago"
    return f"{secs / 86400:.0f} days ago"


def snapshot(ws: Workspace, *, detail: bool = False, machines: list[dict] | None = None) -> dict:
    """Everything the pages show for one workspace. ``detail`` adds the
    decisions and results the account page needs."""
    out: dict[str, Any] = {"slug": ws.slug, "label": ws.label, "machines": machines or []}
    try:
        with session(ws) as s:
            for name, fn in (("heartbeats", _heartbeats), ("monitor", _monitor),
                             ("scheduler", _scheduler), ("publishing", _publishing),
                             ("tokens", _tokens), ("pipeline", _pipeline)):
                _part(out, name, fn, s)
            if detail:
                _part(out, "decisions", _decisions, s)
                _part(out, "results", _results, s)
    except Exception as exc:
        log.exception("Admin: can't open %s", ws.slug)
        out.setdefault("errors", []).append(f"database: {str(exc)[:200]}")
    out["spend_today"] = round(sum(h.get("spend_today") or 0 for h in out.get("heartbeats") or []), 2)
    out["status"], out["reasons"] = _health(out)
    return out


ago = _ago
