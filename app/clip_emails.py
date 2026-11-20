"""The "clips ready to approve" email.

Clips wait in Today's posts until someone approves or rejects them, so an
inbox nobody opens just grows. Each login on the account gets an email when
clips are waiting: daily by default, or weekly, every two weeks or monthly,
as they choose (Settings > Email updates). It goes out at 9 AM in the
account's time zone, and only when something is waiting.

Runs from the scheduler tick (``maybe_send``), on every machine that ticks.
Each recipient is claimed in the database before the email goes out, so two
machines never send the same email twice. Sent through Resend from
Continuity Marketing; needs RESEND_API_KEY, PUBLIC_URL and the login list
(SUPABASE_URL / SUPABASE_SERVICE_KEY).
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger("clip_emails")

TOKEN = "clip_emails"
FREQUENCIES = {"daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30}
FREQUENCY_LABELS = {"daily": "Every day", "weekly": "Once a week",
                    "biweekly": "Every two weeks", "monthly": "Once a month"}
DEFAULT = "daily"
SEND_HOUR = 9                 # local time, in the account's time zone
CHECK_EVERY = 900.0           # seconds between checks on one machine
SENDER = "Continuity Marketing <hello@continuity.marketing>"
SHOWN = 5

_last_check = 0.0
_lock = threading.Lock()


# --- Preferences ------------------------------------------------------------------

def _row(session, *, lock: bool = False):
    from .models import AppToken

    if lock:
        return session.get(AppToken, TOKEN, with_for_update=True)
    return session.get(AppToken, TOKEN)


def _load(row) -> dict[str, dict]:
    try:
        data = json.loads(row.value) if row is not None and row.value else {}
    except ValueError:
        data = {}
    return {k.lower(): v for k, v in data.items() if isinstance(v, dict)}


def preferences() -> dict[str, dict]:
    """{email: {"frequency", "last_sent"}} for everyone with a saved choice."""
    from .db import session_scope

    with session_scope(read_only=True) as session:
        return _load(_row(session))


def frequency_for(email: str, prefs: dict | None = None) -> str:
    pref = (prefs if prefs is not None else preferences()).get(email.lower()) or {}
    freq = pref.get("frequency")
    return freq if freq in FREQUENCIES else DEFAULT


def set_frequency(email: str, frequency: str) -> None:
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {', '.join(FREQUENCIES)}")
    from .db import session_scope
    from .models import AppToken, utcnow

    with session_scope() as session:
        row = _row(session, lock=True)
        if row is None:
            row = AppToken(name=TOKEN)
            session.add(row)
        data = _load(row)
        data.setdefault(email.lower(), {})["frequency"] = frequency
        row.value = json.dumps(data)
        row.updated_at = utcnow()


# --- When ---------------------------------------------------------------------------

def due(frequency: str, last_sent: str | None, now_local: dt.datetime) -> bool:
    """Whether an email is due: past the send hour, and at least the
    frequency's number of days since the last one (counted in calendar days,
    so a daily email goes out every morning, not every 24h to the minute)."""
    if now_local.hour < SEND_HOUR:
        return False
    if not last_sent:
        return True
    try:
        last = dt.datetime.fromisoformat(last_sent).astimezone(now_local.tzinfo).date()
    except ValueError:
        return True
    return (now_local.date() - last).days >= FREQUENCIES.get(frequency, 1)


def _claim(email: str, frequency: str, now: dt.datetime, tz) -> str | None:
    """Mark ``email`` as sent now if it's due, atomically. Returns the previous
    send time ("" for never) when this machine won the claim, else None."""
    from .db import session_scope
    from .models import AppToken, utcnow

    with session_scope() as session:
        row = _row(session, lock=True)
        if row is None:
            row = AppToken(name=TOKEN, value="{}")
            session.add(row)
            session.flush()
        data = _load(row)
        pref = data.setdefault(email.lower(), {})
        if not due(frequency, pref.get("last_sent"), now.astimezone(tz)):
            return None
        previous = pref.get("last_sent") or ""
        pref["last_sent"] = now.isoformat()
        row.value = json.dumps(data)
        row.updated_at = utcnow()
        return previous


def _unclaim(email: str, previous: str) -> None:
    """Put the last send time back after an email failed to go out."""
    from .db import session_scope

    try:
        with session_scope() as session:
            row = _row(session, lock=True)
            data = _load(row)
            if email.lower() in data:
                data[email.lower()]["last_sent"] = previous or None
                row.value = json.dumps(data)
    except Exception:
        log.exception("Couldn't undo the claim for %s", email)


# --- What ---------------------------------------------------------------------------

def waiting(since: str | None = None) -> dict:
    """Clips waiting in Today's posts: how many, how many arrived since
    ``since``, and the best few to show."""
    from sqlalchemy import func, select

    from .db import session_scope
    from .inbox import ready_items
    from .models import Cut, InboxItem

    with session_scope(read_only=True) as session:
        total = int(session.execute(select(func.count(InboxItem.id)).where(
            InboxItem.state == InboxItem.STATE_READY)).scalar_one() or 0)
        new = total
        if since:
            try:
                after = dt.datetime.fromisoformat(since)
                new = int(session.execute(select(func.count(InboxItem.id)).where(
                    InboxItem.state == InboxItem.STATE_READY,
                    InboxItem.ready_at > after)).scalar_one() or 0)
            except ValueError:
                pass
        top = []
        for item in ready_items(session, limit=SHOWN):
            cut = session.get(Cut, item.cut_pk) if item.cut_pk else None
            top.append({"title": (cut.clip_title if cut else "") or item.story or "A new clip",
                        "caption": (item.final_caption or item.caption or "")[:220]})
    return {"total": total, "new": new, "top": top}


def compose(*, brand: str, frequency: str, clips: dict, app_url: str) -> tuple[str, str, str]:
    """(subject, html, text) for one email."""
    n, new = clips["total"], clips["new"]
    subject = (f"{n} clip{'s' if n != 1 else ''} ready to approve"
               + (f" ({new} new)" if 0 < new < n else ""))
    inbox, prefs = f"{app_url}/inbox", f"{app_url}/account/emails"
    period = {"daily": "since yesterday", "weekly": "this week", "biweekly": "in the last two weeks",
              "monthly": "this month"}[frequency]
    lead = (f"{new} new clip{'s' if new != 1 else ''} {period}, " if new and new < n else "") + \
           f"{n} waiting for you in Today's posts."
    items_html = "".join(
        f'<tr><td style="padding:12px 0;border-top:1px solid #e8e6e1">'
        f'<div style="font-weight:700;font-size:15px;color:#111">{html.escape(c["title"])}</div>'
        + (f'<div style="font-size:14px;color:#555;margin-top:4px">{html.escape(c["caption"])}</div>' if c["caption"] else "")
        + "</td></tr>" for c in clips["top"])
    more = n - len(clips["top"])
    body = f"""<!doctype html><html><body style="margin:0;background:#f3f2ef;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:32px 16px">
<table role="presentation" width="100%" style="max-width:560px;background:#fff;border-radius:16px" cellpadding="0" cellspacing="0">
<tr><td style="padding:28px 28px 8px">
  <div style="font-size:13px;color:#888">{html.escape(brand)}</div>
  <h1 style="margin:6px 0 8px;font-size:24px;color:#111">{html.escape(subject)}</h1>
  <p style="margin:0 0 18px;font-size:15px;color:#444">{html.escape(lead)} Each one is cut, subtitled and captioned. Nothing posts until you approve it.</p>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">{items_html}</table>
  {f'<p style="font-size:14px;color:#777;margin:8px 0 0">…and {more} more.</p>' if more > 0 else ''}
  <p style="margin:24px 0 8px"><a href="{inbox}" style="display:inline-block;background:#111;color:#fff;text-decoration:none;font-weight:700;padding:13px 24px;border-radius:999px">Review your clips</a></p>
</td></tr>
<tr><td style="padding:16px 28px 26px;font-size:12px;color:#999">You get this {FREQUENCY_LABELS[frequency].lower()} while clips are waiting. <a href="{prefs}" style="color:#999">Change how often</a>.</td></tr>
</table></td></tr></table></body></html>"""
    text = (f"{subject}\n\n{lead} Nothing posts until you approve it.\n\n"
            + "\n".join(f"- {c['title']}" for c in clips["top"])
            + (f"\n…and {more} more." if more > 0 else "")
            + f"\n\nReview your clips: {inbox}\n\nChange how often you get this: {prefs}\n")
    return subject, body, text


def send(to: str, subject: str, body_html: str, body_text: str) -> None:
    import httpx

    r = httpx.post("https://api.resend.com/emails", timeout=20,
                   headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}"},
                   json={"from": SENDER, "to": [to], "subject": subject,
                         "html": body_html, "text": body_text})
    r.raise_for_status()


# --- The round --------------------------------------------------------------------------

def recipients() -> list[str]:
    """Every login with this account in its workspaces."""
    from .config import WORKSPACE
    from .web import auth

    return sorted({u.email.lower() for u in auth.list_users() if WORKSPACE in (u.workspaces or [])})


def ready() -> str:
    """"" when emails can go out from this machine, else why not."""
    for var in ("RESEND_API_KEY", "PUBLIC_URL", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
        if not os.environ.get(var, "").strip():
            return f"{var} not set"
    return ""


def run(now: dt.datetime | None = None) -> list[str]:
    """Send whatever is due. Returns who got an email."""
    problem = ready()
    if problem:
        log.debug("Clip emails off on this machine: %s", problem)
        return []
    from .config import load_brand, scheduler_timezone

    now = now or dt.datetime.now(dt.timezone.utc)
    tz = scheduler_timezone()
    if now.astimezone(tz).hour < SEND_HOUR:
        return []
    prefs = preferences()
    clips_now = waiting()
    if not clips_now["total"]:
        return []
    brand = load_brand().get("name") or "Throughline"
    app_url = os.environ["PUBLIC_URL"].rstrip("/")
    sent = []
    for email in recipients():
        freq = frequency_for(email, prefs)
        previous = _claim(email, freq, now, tz)
        if previous is None:
            continue
        try:
            subject, body, text = compose(brand=brand, frequency=freq,
                                          clips=waiting(previous or None), app_url=app_url)
            send(email, subject, body, text)
            sent.append(email)
            log.info("Clips-ready email (%s) sent to %s", freq, email)
        except Exception:
            log.exception("Clips-ready email to %s failed", email)
            _unclaim(email, previous)
    return sent


def maybe_send() -> None:
    """From the scheduler tick: at most one check every 15 minutes. Never raises."""
    global _last_check
    with _lock:
        if time.monotonic() - _last_check < CHECK_EVERY:
            return
        _last_check = time.monotonic()
    try:
        run()
    except Exception:
        log.exception("Clip emails round failed")
