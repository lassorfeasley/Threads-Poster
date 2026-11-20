"""Dashboard login, backed by Supabase Auth.

Off unless DASHBOARD_AUTH=supabase, so the local single-operator setup is
unchanged. When on:

- **Accounts** are Supabase Auth users (email + password), so password resets,
  invites and account management come from Supabase.
- **Access** is per workspace: a user may use this dashboard only when their
  ``app_metadata.workspaces`` lists it. ``app_metadata`` can only be written
  with the service key, so a stranger who signs up through Supabase's public
  API gets an account that opens nothing. Grant with ``python run.py user add``
  or ``user invite``.
- **The session** is an HttpOnly cookie holding the Supabase access and refresh
  tokens. Each request asks Supabase who the access token belongs to (cached a
  minute per token); an expired one is refreshed transparently.

The server talks to Supabase Auth's REST API directly (httpx), with the service
key, because pages are rendered here rather than in a browser SDK.

A dashboard that listens on anything but loopback refuses to start without a
login (see run.py `dashboard --host`), so a hosted copy can never come up open.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("auth")

COOKIE_NAME = "cm_session"
SESSION_SECONDS = 30 * 24 * 3600  # the refresh token's life; access tokens renew hourly

# Reachable without a session: the sign-in and recovery pages, the assets they
# use, and the Fly health check.
PUBLIC_PATHS = frozenset({"/login", "/forgot", "/auth/callback", "/auth/session",
                          "/healthz", "/favicon.ico"})
PUBLIC_PREFIXES = ("/static/",)

_USER_CACHE_TTL = 60.0
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)


class AuthError(RuntimeError):
    """Supabase refused, with a message fit for the person signing in."""


@dataclass
class User:
    id: str
    email: str
    workspaces: list[str] = field(default_factory=list)
    role: str = ""

    def can_use(self, workspace: str) -> bool:
        return workspace in self.workspaces or "*" in self.workspaces


@dataclass
class Session:
    access_token: str
    refresh_token: str


# --- configuration -----------------------------------------------------------

def enabled() -> bool:
    return os.environ.get("DASHBOARD_AUTH", "").strip().lower() == "supabase"


def config_problem() -> str:
    """Why the login can't protect this dashboard, or '' when it can."""
    if not enabled():
        return "DASHBOARD_AUTH is not set to supabase"
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_SERVICE_KEY"):
        return "SUPABASE_URL and SUPABASE_SERVICE_KEY are required for the login"
    return ""


def _workspace() -> str:
    from ..config import WORKSPACE

    return WORKSPACE


def _base() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/") + "/auth/v1"


def _key() -> str:
    return os.environ.get("SUPABASE_SERVICE_KEY", "")


def _request(method: str, path: str, *, token: str | None = None,
             json_body: dict | None = None, params: dict | None = None) -> httpx.Response:
    """One call to Supabase Auth. ``token`` is a user's access token; without
    one the service key authorizes (admin endpoints)."""
    headers = {"apikey": _key(), "Authorization": f"Bearer {token or _key()}"}
    return httpx.request(method, _base() + path, headers=headers, json=json_body,
                         params=params, timeout=_TIMEOUT)


def _message(r: httpx.Response, fallback: str) -> str:
    try:
        data = r.json()
    except ValueError:
        return fallback
    return str(data.get("msg") or data.get("error_description") or data.get("message")
               or fallback)


def _user_from(data: dict) -> User:
    meta = data.get("app_metadata") or {}
    spaces = meta.get("workspaces") or []
    return User(id=str(data.get("id") or ""), email=str(data.get("email") or "").lower(),
                workspaces=[str(s) for s in spaces] if isinstance(spaces, list) else [],
                role=str(meta.get("role") or ""))


# --- signing in ----------------------------------------------------------------

def sign_in(email: str, password: str) -> tuple[Session, User] | None:
    """Tokens and user for a correct email and password, else None."""
    r = _request("POST", "/token", params={"grant_type": "password"},
                 json_body={"email": (email or "").strip().lower(), "password": password or ""})
    if r.status_code != 200:
        if r.status_code >= 500:
            raise AuthError("Sign-in is unavailable right now. Try again in a minute.")
        return None
    data = r.json()
    return (Session(data["access_token"], data["refresh_token"]),
            _user_from(data.get("user") or {}))


def refresh(refresh_token: str) -> Session | None:
    r = _request("POST", "/token", params={"grant_type": "refresh_token"},
                 json_body={"refresh_token": refresh_token})
    if r.status_code != 200:
        return None
    data = r.json()
    return Session(data["access_token"], data["refresh_token"])


_cache: dict[str, tuple[float, User | None]] = {}
_cache_lock = threading.Lock()


def user_for(access_token: str) -> User | None:
    """Who the access token belongs to (None when invalid or expired). Cached
    briefly: a page load makes many requests with the same token."""
    if not access_token:
        return None
    key = hashlib.sha256(access_token.encode()).hexdigest()
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _USER_CACHE_TTL:
            return hit[1]
    r = _request("GET", "/user", token=access_token)
    if r.status_code >= 500:
        raise AuthError("Sign-in is unavailable right now.")
    user = _user_from(r.json()) if r.status_code == 200 else None
    with _cache_lock:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[key] = (now, user)
    return user


def resolve(cookie: str | None) -> tuple[User | None, Session | None]:
    """The signed-in user for a session cookie, and a replacement session when
    the access token had to be refreshed (the caller re-sets the cookie).
    Users without access to this workspace resolve to None."""
    session = read_cookie(cookie)
    if session is None:
        return None, None
    renewed = None
    user = user_for(session.access_token)
    if user is None and session.refresh_token:
        renewed = refresh(session.refresh_token)
        if renewed is not None:
            user = user_for(renewed.access_token)
    if user is None or not user.can_use(_workspace()):
        return None, None
    return user, renewed


def sign_out(access_token: str) -> None:
    try:
        _request("POST", "/logout", token=access_token)
    except httpx.HTTPError:
        pass


def send_reset(email: str, redirect_to: str) -> None:
    """Ask Supabase to email a password-reset link. Says nothing about whether
    the address has an account."""
    r = _request("POST", "/recover", params={"redirect_to": redirect_to},
                 json_body={"email": (email or "").strip().lower()})
    if r.status_code == 429:
        raise AuthError("Too many reset emails. Wait a few minutes and try again.")
    if r.status_code >= 500:
        raise AuthError("Couldn't send the email right now. Try again in a minute.")


def set_password(access_token: str, password: str) -> None:
    r = _request("PUT", "/user", token=access_token, json_body={"password": password})
    if r.status_code != 200:
        raise AuthError(_message(r, "Couldn't change the password."))


# --- the cookie ----------------------------------------------------------------

def make_cookie(session: Session) -> str:
    raw = json.dumps([session.access_token, session.refresh_token], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def read_cookie(value: str | None) -> Session | None:
    if not value:
        return None
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        access, refresh_token = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(access, str) or not isinstance(refresh_token, str):
        return None
    return Session(access, refresh_token)


# --- administration (CLI) --------------------------------------------------------

def _admin_find(email: str) -> dict | None:
    email = email.strip().lower()
    page = 1
    while True:
        r = _request("GET", "/admin/users", params={"page": page, "per_page": 200})
        if r.status_code != 200:
            raise AuthError(_message(r, f"Supabase refused ({r.status_code})"))
        users = r.json().get("users") or []
        for u in users:
            if str(u.get("email") or "").lower() == email:
                return u
        if len(users) < 200:
            return None
        page += 1


def list_users() -> list[User]:
    out, page = [], 1
    while True:
        r = _request("GET", "/admin/users", params={"page": page, "per_page": 200})
        if r.status_code != 200:
            raise AuthError(_message(r, f"Supabase refused ({r.status_code})"))
        users = r.json().get("users") or []
        out += [_user_from(u) for u in users]
        if len(users) < 200:
            return out
        page += 1


def _with_workspace(meta: dict, workspace: str, role: str) -> dict:
    spaces = [s for s in (meta.get("workspaces") or []) if s != workspace] + [workspace]
    return {**meta, "workspaces": spaces, "role": role or meta.get("role") or "member"}


def add_user(email: str, password: str, workspace: str, role: str = "member") -> User:
    """Create a confirmed account with a password (or give an existing one the
    workspace and the new password)."""
    existing = _admin_find(email)
    if existing is None:
        r = _request("POST", "/admin/users", json_body={
            "email": email.strip().lower(), "password": password, "email_confirm": True,
            "app_metadata": _with_workspace({}, workspace, role)})
    else:
        r = _request("PUT", f"/admin/users/{existing['id']}", json_body={
            "password": password,
            "app_metadata": _with_workspace(existing.get("app_metadata") or {}, workspace, role)})
    if r.status_code not in (200, 201):
        raise AuthError(_message(r, f"Supabase refused ({r.status_code})"))
    return _user_from(r.json())


def invite(email: str, workspace: str, redirect_to: str, role: str = "member") -> User:
    """Email an invitation; following it lands on the set-password page."""
    existing = _admin_find(email)
    if existing is not None:
        grant(email, workspace, role)
        send_reset(email, redirect_to)
        return _user_from({**existing, "app_metadata": _with_workspace(
            existing.get("app_metadata") or {}, workspace, role)})
    r = _request("POST", "/invite", params={"redirect_to": redirect_to},
                 json_body={"email": email.strip().lower()})
    if r.status_code not in (200, 201):
        raise AuthError(_message(r, f"Supabase refused ({r.status_code})"))
    return grant(email, workspace, role)


def grant(email: str, workspace: str, role: str = "member") -> User:
    u = _admin_find(email)
    if u is None:
        raise AuthError(f"No account for {email}")
    r = _request("PUT", f"/admin/users/{u['id']}", json_body={
        "app_metadata": _with_workspace(u.get("app_metadata") or {}, workspace, role)})
    if r.status_code != 200:
        raise AuthError(_message(r, f"Supabase refused ({r.status_code})"))
    with _cache_lock:  # a new role applies on the next request, not when a cache expires
        _cache.clear()
    return _user_from(r.json())


def revoke(email: str, workspace: str) -> User:
    u = _admin_find(email)
    if u is None:
        raise AuthError(f"No account for {email}")
    meta = u.get("app_metadata") or {}
    spaces = [s for s in (meta.get("workspaces") or []) if s != workspace]
    r = _request("PUT", f"/admin/users/{u['id']}",
                 json_body={"app_metadata": {**meta, "workspaces": spaces}})
    if r.status_code != 200:
        raise AuthError(_message(r, f"Supabase refused ({r.status_code})"))
    with _cache_lock:
        _cache.clear()
    return _user_from(r.json())


# --- helpers -------------------------------------------------------------------

def safe_next(target: str | None) -> str:
    """Only same-site paths: never let ?next= bounce a login to another site."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


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
