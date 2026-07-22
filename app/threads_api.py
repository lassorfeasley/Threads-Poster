"""Official Threads API (Meta) client: OAuth, publishing, replies, insights.

The token lives in the shared database (``app_tokens`` table) so headless
runners (GitHub Actions / cron) can publish without this machine. A local copy
in data/threads_token.json (gitignored) is kept as a backup and migrated into
the DB automatically. Long-lived tokens last ~60 days and are refreshed when
older than 7 days.
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

log = logging.getLogger("threads")

GRAPH = "https://graph.threads.net/v1.0"
TOKEN_FILE = ROOT / "data" / "threads_token.json"

SCOPES = "threads_basic,threads_content_publish,threads_manage_replies,threads_read_replies,threads_manage_insights"


TOKEN_NAME = "threads"

# In-process cache so pages that call ``is_authenticated()`` (and every API
# call via ``_auth``) don't pay a remote DB round trip each time. Invalidated
# on save; TTL covers the rare case of an external writer updating the token.
_token_cache: dict | None = None
_token_cache_at: float = 0.0
_TOKEN_CACHE_TTL = 60.0


class ThreadsError(RuntimeError):
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
        log.warning("Could not save Threads token to DB (file copy still written): %s", exc)
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(payload)
    except Exception as exc:
        log.warning("Could not write local token file: %s", exc)
    _token_cache = dict(token)
    _token_cache_at = time.monotonic()


def _peek_token() -> dict | None:
    """Return the stored token or None. Migrates the legacy file into the DB."""
    global _token_cache, _token_cache_at
    if (
        _token_cache is not None
        and (time.monotonic() - _token_cache_at) < _TOKEN_CACHE_TTL
    ):
        return dict(_token_cache)
    # Negative cache: remember "no token" briefly so unauthenticated pages
    # don't hammer the DB either.
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
        log.warning("Could not read Threads token from DB (trying local file): %s", exc)
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
        "client_id": env("THREADS_APP_ID"),
        "redirect_uri": env("THREADS_REDIRECT_URI"),
        "scope": SCOPES,
        "response_type": "code",
    }
    return f"https://threads.net/oauth/authorize?{urlencode(params)}"


def exchange_code(code: str) -> dict:
    """Auth code -> short-lived token -> long-lived token. Saves to disk."""
    resp = requests.post(
        "https://graph.threads.net/oauth/access_token",
        data={
            "client_id": env("THREADS_APP_ID"),
            "client_secret": env("THREADS_APP_SECRET"),
            "grant_type": "authorization_code",
            "redirect_uri": env("THREADS_REDIRECT_URI"),
            "code": code,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise ThreadsError(f"code exchange failed: {resp.text[:300]}")
    short = resp.json()

    resp = requests.get(
        "https://graph.threads.net/access_token",
        params={
            "grant_type": "th_exchange_token",
            "client_secret": env("THREADS_APP_SECRET"),
            "access_token": short["access_token"],
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise ThreadsError(f"long-lived exchange failed: {resp.text[:300]}")
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
        raise ThreadsError("Not authenticated with Threads. Use the Publish page to connect.")
    return token


def _maybe_refresh(token: dict) -> dict:
    obtained = dt.datetime.fromisoformat(token["obtained_at"])
    age = (dt.datetime.now(dt.timezone.utc) - obtained).total_seconds()
    # Refresh when older than 7 days (must be >24h old; expires ~60 days).
    if age < 7 * 86400:
        return token
    resp = requests.get(
        "https://graph.threads.net/refresh_access_token",
        params={"grant_type": "th_refresh_token", "access_token": token["access_token"]},
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        token["access_token"] = data["access_token"]
        token["expires_in"] = data.get("expires_in", token.get("expires_in"))
        token["obtained_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _save_token(token)
        log.info("Refreshed Threads token")
    else:
        log.warning("Token refresh failed (will keep current token): %s", resp.text[:200])
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
    resp = requests.get(f"{GRAPH}/me", params={"fields": "id,username", "access_token": access_token}, timeout=30)
    if resp.status_code != 200:
        raise ThreadsError(f"me lookup failed: {resp.text[:300]}")
    return resp.json()


def _api(method: str, path: str, **params) -> dict:
    access_token, user_id = _auth()
    params["access_token"] = access_token
    path = path.replace("{user_id}", user_id)
    url = f"{GRAPH}/{path}"
    resp = requests.request(method, url, params=params if method == "GET" else None,
                            data=None if method == "GET" else params, timeout=60)
    if resp.status_code != 200:
        raise ThreadsError(f"{method} {path} failed: {resp.text[:400]}")
    return resp.json()


# --- Publishing --------------------------------------------------------------

def publish_video(video_url: str, caption: str, reply_to_id: str | None = None,
                  poll_timeout_seconds: int | None = None) -> dict:
    """Create a video media container from a public URL, wait for Meta to
    process it, then publish. Returns {media_id, permalink}."""
    settings = load_settings()
    if poll_timeout_seconds is None:
        poll_timeout_seconds = settings.get("threads.publish_poll_timeout_seconds", 600)
    interval = max(5, settings.get("threads.publish_poll_interval_seconds", 10))

    params = {"media_type": "VIDEO", "video_url": video_url, "text": caption}
    if reply_to_id:
        params["reply_to_id"] = reply_to_id
    container = _api("POST", "{user_id}/threads", **params)
    container_id = container["id"]

    # Poll container status until FINISHED (Meta needs ~30s+ for video; larger
    # clips can take a few minutes).
    deadline = time.time() + poll_timeout_seconds
    while time.time() < deadline:
        time.sleep(interval)
        status = _api("GET", container_id, fields="status,error_message")
        state = status.get("status")
        if state == "FINISHED":
            break
        if state == "ERROR":
            raise ThreadsError(f"Media container failed: {status.get('error_message')}")
    else:
        raise ThreadsError(
            f"Timed out waiting for Threads to process the video after "
            f"{poll_timeout_seconds}s. Try posting again — larger clips can take "
            f"longer; you can also raise threads.publish_poll_timeout_seconds."
        )

    published = _api("POST", "{user_id}/threads_publish", creation_id=container_id)
    media_id = published["id"]
    info = _api("GET", media_id, fields="id,permalink")
    return {"media_id": media_id, "permalink": info.get("permalink", "")}


def publish_text_reply(text: str, reply_to_id: str) -> dict:
    """Publish a text reply to a post or comment (``reply_to_id`` is either)."""
    container = _api("POST", "{user_id}/threads", media_type="TEXT", text=text, reply_to_id=reply_to_id)
    published = _api("POST", "{user_id}/threads_publish", creation_id=container["id"])
    return {"media_id": published["id"]}


# --- Reading -----------------------------------------------------------------

def fetch_user_posts(limit: int = 200) -> list[dict]:
    """List the authenticated account's own Threads posts (newest first), paging
    until ``limit`` is reached. Covers posts made outside this tool."""
    out: list[dict] = []
    params = {
        "fields": "id,media_type,permalink,text,timestamp,is_quote_post",
        "limit": 50,
    }
    while len(out) < limit:
        data = _api("GET", "{user_id}/threads", **params)
        out.extend(data.get("data", []))
        paging = data.get("paging") or {}
        after = (paging.get("cursors") or {}).get("after")
        if not paging.get("next") or not after:
            break
        params["after"] = after
    return out[:limit]


def fetch_replies(media_id: str) -> list[dict]:
    """Top-level replies to one of the operator's own posts."""
    out: list[dict] = []
    params = {"fields": "id,text,username,timestamp,hide_status"}
    path = f"{media_id}/replies"
    while True:
        data = _api("GET", path, **params)
        out.extend(data.get("data", []))
        cursors = (data.get("paging") or {}).get("cursors") or {}
        after = cursors.get("after")
        next_url = (data.get("paging") or {}).get("next")
        if not next_url or not after:
            break
        params["after"] = after
    return out


def fetch_insights(media_id: str) -> dict:
    """Post metrics. Returns {views, likes, replies, reposts, quotes, shares} (missing -> None)."""
    try:
        data = _api("GET", f"{media_id}/insights", metric="views,likes,replies,reposts,quotes,shares")
    except ThreadsError as exc:
        log.warning("Insights fetch failed for %s: %s", media_id, exc)
        return {}
    result: dict = {}
    for item in data.get("data", []):
        name = item.get("name")
        values = item.get("values") or [{}]
        result[name] = values[0].get("value")
    return result
