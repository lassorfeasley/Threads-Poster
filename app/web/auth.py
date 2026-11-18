"""Dashboard login: who may use a hosted dashboard.

Off unless configured, so the local single-operator setup is unchanged. It turns
on when DASHBOARD_USERS is set:

  DASHBOARD_USERS   comma-separated "email:hash" entries. Make a hash with
                    `python run.py user hash` (prompts for the password).
  AUTH_SECRET       signs the session cookie. Required whenever users are set;
                    rotating it signs everyone out.

Passwords are hashed with scrypt; the session is an HMAC-signed cookie holding
the email and an expiry, so there is no session table and a restart keeps
everyone signed in. Standard library only.

A dashboard that listens on anything but loopback refuses to start without a
login (see run.py `dashboard --host`), so a hosted copy can never come up open.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time

COOKIE_NAME = "cm_session"
SESSION_SECONDS = 14 * 24 * 3600

# scrypt cost: ~50 ms per check, ~16 MB of memory. Stored with the hash so it
# can be raised later without invalidating existing hashes.
_N, _R, _P = 2 ** 14, 8, 1

# Reachable without a session: the login form itself, the assets it uses, and
# the Fly health check.
PUBLIC_PATHS = frozenset({"/login", "/healthz", "/favicon.ico"})
PUBLIC_PREFIXES = ("/static/",)


# --- passwords ---------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(
        _N, _R, _P, _b64(salt), _b64(digest))


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        want = _unb64(digest)
        got = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=int(n), r=int(r),
                             p=int(p), dklen=len(want))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, want)


# --- configuration -----------------------------------------------------------

def users() -> dict[str, str]:
    """email (lowercased) -> password hash, from DASHBOARD_USERS. Read on every
    call so tests and a changed env take effect without a restart of state."""
    out: dict[str, str] = {}
    for entry in os.environ.get("DASHBOARD_USERS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        email, sep, stored = entry.partition(":")
        if sep and email.strip() and stored.strip():
            out[email.strip().lower()] = stored.strip()
    return out


def enabled() -> bool:
    return bool(users())


def config_problem() -> str:
    """Why the login can't protect this dashboard, or '' when it can."""
    if not enabled():
        return "DASHBOARD_USERS is not set"
    if len(_secret()) < 32:
        return "AUTH_SECRET must be set to at least 32 characters"
    return ""


def _secret() -> bytes:
    return os.environ.get("AUTH_SECRET", "").encode()


# --- sessions ----------------------------------------------------------------

def make_session(email: str, now: float | None = None) -> str:
    expires = int((now if now is not None else time.time()) + SESSION_SECONDS)
    payload = _b64(f"{email.lower()}|{expires}".encode())
    return f"{payload}.{_sign(payload)}"


def read_session(token: str | None, now: float | None = None) -> str | None:
    """The signed-in email, or None for a missing, forged, expired or revoked
    session (a user removed from DASHBOARD_USERS is signed out at once)."""
    if not token or "." not in token or len(_secret()) < 32:
        return None
    payload, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(payload)):
        return None
    try:
        email, expires = _unb64(payload).decode().rsplit("|", 1)
        expired = int(expires) < (now if now is not None else time.time())
    except (ValueError, UnicodeDecodeError):
        return None
    if expired or email not in users():
        return None
    return email


def authenticate(email: str, password: str) -> str | None:
    email = (email or "").strip().lower()
    stored = users().get(email)
    if stored is None:
        # Same work as a real check, so response time doesn't reveal which
        # emails exist.
        verify_password(password or "", _DUMMY_HASH)
        return None
    return email if verify_password(password or "", stored) else None


def _sign(payload: str) -> str:
    return _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())


def safe_next(target: str | None) -> str:
    """Only same-site paths: never let ?next= bounce a login to another site."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


# --- login throttling --------------------------------------------------------

_FAIL_WINDOW = 15 * 60
_FAIL_LIMIT = 10
_fails: dict[str, list[float]] = {}
_fails_lock = threading.Lock()


def throttled(key: str, now: float | None = None) -> bool:
    now = now if now is not None else time.time()
    with _fails_lock:
        recent = [t for t in _fails.get(key, []) if now - t < _FAIL_WINDOW]
        _fails[key] = recent
        return len(recent) >= _FAIL_LIMIT


def record_failure(key: str, now: float | None = None) -> None:
    with _fails_lock:
        _fails.setdefault(key, []).append(now if now is not None else time.time())


def clear_failures(key: str) -> None:
    with _fails_lock:
        _fails.pop(key, None)


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
