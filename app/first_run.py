"""What a brand-new account shows before its first videos arrive.

Right after onboarding, the dashboard and Today's posts would otherwise say
"no new matches" or "all caught up", which reads like something is broken.
Until the account has its first video, they show where things stand instead:
how many channels are being watched, whether the first check has run, and
that a video of your own gets clips right away.
"""
from __future__ import annotations

from sqlalchemy import func, select

from .config import load_settings
from .models import Candidate, Channel, InboxItem, MonitorRun

# Channels that hold videos added by hand, not watched ones.
_SYNTHETIC = ("upload://", "youtube://pasted")


def status(session) -> dict | None:
    """None once the account has a video from its channels or anything in
    the inbox; otherwise the numbers the placeholder shows."""
    synthetic = [Channel.url.like(f"{p}%") for p in _SYNTHETIC]
    from sqlalchemy import or_

    found = session.execute(
        select(Candidate.id).join(Channel, Channel.id == Candidate.channel_pk)
        .where(~or_(*synthetic)).limit(1)).first()
    if found or session.execute(select(InboxItem.id).limit(1)).first():
        return None
    channels = int(session.execute(
        select(func.count(Channel.id)).where(Channel.enabled.is_(True), ~or_(*synthetic))
    ).scalar_one() or 0)
    ran = session.execute(
        select(MonitorRun.id).where(MonitorRun.status == MonitorRun.STATUS_DONE).limit(1)).first()
    running = session.execute(
        select(MonitorRun.id).where(MonitorRun.status == MonitorRun.STATUS_RUNNING).limit(1)).first()
    uploads = int(session.execute(
        select(func.count(Candidate.id)).join(Channel, Channel.id == Candidate.channel_pk)
        .where(Channel.url.like("upload://%"))).scalar_one() or 0)
    hours = int(load_settings().get("monitor.poll_interval_minutes", 360)) // 60 or 6
    return {"channels": channels, "monitor_ran": bool(ran), "monitor_running": bool(running),
            "uploads": uploads, "poll_hours": hours}
