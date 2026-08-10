"""Instagram Graph API client for Reels publishing (Instagram Login flavor).

Uses the "Instagram API with Instagram Login" (Business Login): a professional
Instagram account, ``instagram_business_basic`` + ``instagram_business_content_publish``
scopes, and the ``graph.instagram.com`` host — no linked Facebook Page needed.

Same shape as app/threads_api.py on purpose: the token lives in the shared
database (``app_tokens``, name ``instagram``) so headless runners can publish
without this machine, with data/instagram_token.json as a gitignored backup.
Long-lived tokens last ~60 days and are refreshed when older than 7 days.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

from .config import ROOT, env, load_settings

log = logging.getLogger("instagram")

GRAPH = "https://graph.instagram.com/v23.0"
TOKEN_FILE = ROOT / "data" / "instagram_token.json"

SCOPES = "instagram_business_basic,instagram_business_content_publish"

TOKEN_NAME = "instagram"

# Meta's per-account cap on API-published posts in a rolling 24h window
# (all media types combined; enforced at media_publish).
PUBLISH_LIMIT_PER_DAY = 100

# In-process cache mirroring threads_api: pages call is_authenticated() often
# and shouldn't pay a remote DB round trip each time.
_token_cache: dict | None = None
_token_cache_at: float = 0.0
_TOKEN_CACHE_TTL = 60.0


class InstagramError(RuntimeError):
    pass


# --- Token storage (DB-first, file fallback) ---------------------------------

def _save_token(token: dict) -> None:
    """Persist the token to the DB (canonical) and the local file (backup)."""
    global _token_cache, _token_cache_at
    payload = json.dumps(token, indent=1)
    try:
        from .db import session_scope
        from .models import AppToken, utcnow

        with session_scope() as session:
            row = session.get(AppToken, TOKEN_NAME)
            if row is None:
                row = AppToken(name=TOKEN_NAME)
                session.add(row)
            row.value = payload
            row.updated_at = utcnow()
    except Exception as exc:
        log.warning("Could not save Instagram token to DB (file copy still written): %s", exc)
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(payload)
    except Exception as exc:
        log.warning("Could not write local Instagram token file: %s", exc)
    _token_cache = dict(token)
    _token_cache_at = time.monotonic()


def _peek_token() -> dict | None:
    """Return the stored token or None. Migrates a legacy file into the DB."""
    global _token_cache, _token_cache_at
    if (
        _token_cache is not None
        and (time.monotonic() - _token_cache_at) < _TOKEN_CACHE_TTL
    ):
        return dict(_token_cache)
    if (
        _token_cache is None
        and _token_cache_at
        and (time.monotonic() - _token_cache_at) < _TOKEN_CACHE_TTL
    ):
        return None
    try:
        from .db import session_scope
        from .models import AppToken

        with session_scope() as session:
            row = session.get(AppToken, TOKEN_NAME)
            if row is not None and row.value:
                token = json.loads(row.value)
                _token_cache = token
                _token_cache_at = time.monotonic()
                return dict(token)
    except Exception as exc:
        log.warning("Could not read Instagram token from DB (trying local file): %s", exc)
    if TOKEN_FILE.exists():
        token = json.loads(TOKEN_FILE.read_text())
        _save_token(token)  # one-time migration into the DB
        return token
    _token_cache = None
    _token_cache_at = time.monotonic()
    return None


# --- OAuth -------------------------------------------------------------------

def authorize_url() -> str:
    params = {
        "client_id": env("INSTAGRAM_APP_ID"),
        "redirect_uri": env("INSTAGRAM_REDIRECT_URI"),
        "scope": SCOPES,
        "response_type": "code",
    }
    return f"https://www.instagram.com/oauth/authorize?{urlencode(params)}"


def exchange_code(code: str) -> dict:
    """Auth code -> short-lived token -> long-lived token. Saves to DB+disk."""
    # Instagram appends '#_' to the redirect; operators paste the raw param.
    code = code.strip().removesuffix("#_")
    resp = requests.post(
        "https://api.instagram.com/oauth/access_token",
        data={
            "client_id": env("INSTAGRAM_APP_ID"),
            "client_secret": env("INSTAGRAM_APP_SECRET"),
            "grant_type": "authorization_code",
            "redirect_uri": env("INSTAGRAM_REDIRECT_URI"),
            "code": code,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise InstagramError(f"code exchange failed: {resp.text[:300]}")
    short = resp.json()
    # Business Login can return {"data": [{...}]} or a flat object.
    if isinstance(short.get("data"), list) and short["data"]:
        short = short["data"][0]

    resp = requests.get(
        f"{GRAPH}/access_token",
        params={
            "grant_type": "ig_exchange_token",
            "client_secret": env("INSTAGRAM_APP_SECRET"),
            "access_token": short["access_token"],
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise InstagramError(f"long-lived exchange failed: {resp.text[:300]}")
    data = resp.json()
    token = {
        "access_token": data["access_token"],
        "user_id": str(short.get("user_id", "")),
        "obtained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expires_in": data.get("expires_in", 5183944),
    }
    _save_token(token)
    return token


def _load_token() -> dict:
    token = _peek_token()
    if token is None:
        raise InstagramError(
            "Not authenticated with Instagram. Use the Accounts page to connect.")
    return token


def _maybe_refresh(token: dict) -> dict:
    obtained = dt.datetime.fromisoformat(token["obtained_at"])
    age = (dt.datetime.now(dt.timezone.utc) - obtained).total_seconds()
    # Refresh when older than 7 days (must be >24h old; expires ~60 days).
    if age < 7 * 86400:
        return token
    resp = requests.get(
        f"{GRAPH}/refresh_access_token",
        params={"grant_type": "ig_refresh_token", "access_token": token["access_token"]},
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        token["access_token"] = data["access_token"]
        token["expires_in"] = data.get("expires_in", token.get("expires_in"))
        token["obtained_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _save_token(token)
        log.info("Refreshed Instagram token")
    else:
        log.warning("Instagram token refresh failed (will keep current token): %s",
                    resp.text[:200])
    return token


def is_authenticated() -> bool:
    try:
        return _peek_token() is not None
    except Exception:
        return False


def _auth() -> tuple[str, str]:
    token = _maybe_refresh(_load_token())
    user_id = token.get("user_id") or _me(token["access_token"])["id"]
    return token["access_token"], str(user_id)


def _me(access_token: str) -> dict:
    resp = requests.get(f"{GRAPH}/me",
                        params={"fields": "user_id,username", "access_token": access_token},
                        timeout=30)
    if resp.status_code != 200:
        raise InstagramError(f"me lookup failed: {resp.text[:300]}")
    data = resp.json()
    # Normalize: the professional-account ID is "user_id" on this API flavor.
    data["id"] = str(data.get("user_id") or data.get("id") or "")
    return data


def account_username() -> str:
    """The connected account's handle, cached in the stored token payload after
    the first lookup. Empty string when unavailable (offline / not connected)."""
    try:
        token = _peek_token()
        if not token:
            return ""
        if token.get("username"):
            return str(token["username"])
        me = _me(token["access_token"])
        token["username"] = me.get("username", "")
        token["user_id"] = token.get("user_id") or me.get("id")
        _save_token(token)
        return str(token.get("username") or "")
    except Exception:
        return ""


def _api(method: str, path: str, **params) -> dict:
    access_token, user_id = _auth()
    params["access_token"] = access_token
    path = path.replace("{user_id}", user_id)
    url = f"{GRAPH}/{path}"
    resp = requests.request(method, url, params=params if method == "GET" else None,
                            data=None if method == "GET" else params, timeout=60)
    if resp.status_code != 200:
        raise InstagramError(f"{method} {path} failed: {resp.text[:400]}")
    return resp.json()


# --- Publishing --------------------------------------------------------------

def publishing_quota_usage() -> int | None:
    """API posts published in the current rolling 24h window (None if the
    lookup fails). Meta caps this at PUBLISH_LIMIT_PER_DAY."""
    try:
        data = _api("GET", "{user_id}/content_publishing_limit", fields="quota_usage")
        entries = data.get("data") or []
        if entries:
            return int(entries[0].get("quota_usage", 0))
    except Exception as exc:
        log.warning("Instagram quota lookup failed: %s", exc)
    return None


def publish_reel(video_url: str, caption: str,
                 poll_timeout_seconds: int | None = None) -> dict:
    """Create a REELS media container from a public URL, wait for Meta to
    process it, then publish. Returns {media_id, permalink}."""
    settings = load_settings()
    if poll_timeout_seconds is None:
        poll_timeout_seconds = settings.get("instagram.publish_poll_timeout_seconds", 600)
    interval = max(5, settings.get("instagram.publish_poll_interval_seconds", 10))

    usage = publishing_quota_usage()
    if usage is not None and usage >= PUBLISH_LIMIT_PER_DAY:
        raise InstagramError(
            f"Instagram publishing quota reached ({usage}/{PUBLISH_LIMIT_PER_DAY} "
            f"posts in the rolling 24h window) — try again later."
        )

    container = _api("POST", "{user_id}/media",
                     media_type="REELS", video_url=video_url, caption=caption)
    container_id = container["id"]

    # Poll container status until FINISHED (video processing takes ~30s+,
    # larger clips a few minutes).
    deadline = time.time() + poll_timeout_seconds
    while time.time() < deadline:
        time.sleep(interval)
        status = _api("GET", container_id, fields="status_code,status")
        state = status.get("status_code")
        if state == "FINISHED":
            break
        if state in ("ERROR", "EXPIRED"):
            raise InstagramError(f"Media container failed: {status.get('status')}")
    else:
        raise InstagramError(
            f"Timed out waiting for Instagram to process the video after "
            f"{poll_timeout_seconds}s. Larger clips take longer; you can raise "
            f"instagram.publish_poll_timeout_seconds."
        )

    published = _api("POST", "{user_id}/media_publish", creation_id=container_id)
    media_id = published["id"]
    permalink = ""
    try:
        info = _api("GET", media_id, fields="id,permalink")
        permalink = info.get("permalink", "")
    except Exception as exc:
        # The reel is live at this point; a permalink hiccup must not fail it.
        log.warning("Could not fetch permalink for reel %s: %s", media_id, exc)
    return {"media_id": media_id, "permalink": permalink}
