"""Mail forwarding for continuity.marketing, through Resend receiving.

Resend receives mail for the domain (its MX record) and calls our webhook with
an ``email.received`` event. The event carries only metadata, so we fetch the
message, and the original .eml when it has attachments, then send a copy to
the forwarding target from the address it was sent to. The original sender
goes in Reply-To, so replying from the target answers them directly.

Configuration (on the admin app):
  RESEND_API_KEY         the same key the clip emails use
  RESEND_WEBHOOK_SECRET  ``whsec_...`` from the webhook's page in Resend
  MAIL_FORWARDS          ``lassor@continuity.marketing=feasley@lassor.com``;
                         comma-separate several. ``*@continuity.marketing=...``
                         catches every other address at the domain.
Mail for an address with no rule is left in Resend's Receiving tab.
"""
from __future__ import annotations

import base64
import email
import email.policy
import email.utils
import hashlib
import hmac
import logging
import os
import time

import httpx

log = logging.getLogger("admin.inbound_mail")

API = "https://api.resend.com"
TOLERANCE_S = 5 * 60
MAX_ATTACHMENTS_BYTES = 30 * 1024 * 1024   # Resend's limit is 40 MB once base64-encoded


class BadSignature(Exception):
    pass


def verify(secret: str, msg_id: str, timestamp: str, signatures: str, body: bytes,
           *, now: float | None = None) -> None:
    """Svix-style check: HMAC-SHA256 of ``{id}.{timestamp}.{body}`` keyed with the
    base64 part of ``whsec_...``; the header lists ``v1,<base64>`` signatures."""
    if not (secret and msg_id and timestamp and signatures):
        raise BadSignature("missing signature headers or secret")
    try:
        ts = int(timestamp)
    except ValueError:
        raise BadSignature("bad timestamp") from None
    if abs((now if now is not None else time.time()) - ts) > TOLERANCE_S:
        raise BadSignature("timestamp outside tolerance")
    try:
        key = base64.b64decode(secret.removeprefix("whsec_"), validate=True)
    except ValueError:
        raise BadSignature("RESEND_WEBHOOK_SECRET isn't a valid whsec_ secret") from None
    signed = f"{msg_id}.{timestamp}.".encode() + body
    expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    for part in signatures.split():
        version, _, sig = part.partition(",")
        if version == "v1" and hmac.compare_digest(sig, expected):
            return
    raise BadSignature("no matching signature")


def rules(raw: str | None = None) -> dict[str, str]:
    """``MAIL_FORWARDS`` as {address or "*@domain": target}."""
    raw = os.environ.get("MAIL_FORWARDS", "") if raw is None else raw
    out = {}
    for pair in raw.split(","):
        src, _, dst = pair.partition("=")
        if src.strip() and dst.strip():
            out[src.strip().lower()] = dst.strip()
    return out


def route(recipients: list[str], table: dict[str, str]) -> list[tuple[str, str]]:
    """(address it was sent to, target) for each recipient with a rule; one per target."""
    seen, out = set(), []
    for addr in recipients:
        addr = email.utils.parseaddr(addr)[1].lower()
        if not addr or "@" not in addr:
            continue
        target = table.get(addr) or table.get("*@" + addr.split("@", 1)[1])
        if target and target.lower() not in seen:
            seen.add(target.lower())
            out.append((addr, target))
    return out


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}"}


def _attachments(raw_url: str) -> list[dict]:
    """Attachments from the original .eml, as Resend's send API wants them."""
    r = httpx.get(raw_url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    msg = email.message_from_bytes(r.content, policy=email.policy.default)
    out, total = [], 0
    for part in msg.iter_attachments():
        data = part.get_payload(decode=True) or b""
        total += len(data)
        if total > MAX_ATTACHMENTS_BYTES:
            log.warning("Attachments too large to forward; the rest are left in Resend")
            break
        item = {"filename": part.get_filename() or "attachment",
                "content": base64.b64encode(data).decode(),
                "content_type": part.get_content_type()}
        if part.get("Content-ID"):
            item["content_id"] = part["Content-ID"].strip("<>")
        out.append(item)
    return out


def handle(event: dict) -> list[str]:
    """Forward one ``email.received`` event; returns the targets it went to."""
    if event.get("type") != "email.received":
        return []
    data = event.get("data") or {}
    email_id = data.get("email_id") or ""
    recipients = list(data.get("received_for") or []) + list(data.get("to") or []) + list(data.get("cc") or [])
    routes = route(recipients, rules())
    if not email_id or not routes:
        log.info("Received mail %s for %s: no forwarding rule", email_id, recipients)
        return []

    r = httpx.get(f"{API}/emails/receiving/{email_id}", params={"html_format": "cid"},
                  headers=_headers(), timeout=30)
    r.raise_for_status()
    mail = r.json()
    original_from = (mail.get("headers") or {}).get("from") or mail.get("from") or data.get("from") or ""
    name, addr = email.utils.parseaddr(original_from)
    attachments = []
    if mail.get("attachments") and (mail.get("raw") or {}).get("download_url"):
        attachments = _attachments(mail["raw"]["download_url"])

    sent = []
    for to_addr, target in routes:
        label = f"{name or addr} via Continuity Marketing".replace('"', "'")
        body = {"from": email.utils.formataddr((label, to_addr)),
                "to": [target],
                "subject": mail.get("subject") or data.get("subject") or "(no subject)",
                "reply_to": [original_from or addr]}
        if mail.get("html"):
            body["html"] = mail["html"]
        if mail.get("text"):
            body["text"] = mail["text"]
        if "html" not in body and "text" not in body:
            body["text"] = "(empty message)"
        if attachments:
            body["attachments"] = attachments
        s = httpx.post(f"{API}/emails", json=body, timeout=60,
                       headers={**_headers(), "Idempotency-Key": f"fwd-{email_id}-{target.lower()}"})
        s.raise_for_status()
        log.info("Forwarded mail %s for %s to %s", email_id, to_addr, target)
        sent.append(target)
    return sent
