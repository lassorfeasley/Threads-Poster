"""X (Twitter) API v2 client: video posts on the connected account.

OAuth 2.0 authorization code flow with PKCE. Scopes: ``tweet.write`` and
``media.write`` to post, ``users.read``/``tweet.read`` to look up the account
name at connect time, ``offline.access`` for a refresh token.

Same storage shape as app/linkedin_api.py: the token lives in the shared
database (``app_tokens``, name ``x``) so headless runners can publish without
this machine, with data/x_token.json as a gitignored backup. Access tokens
last two hours and X rotates the refresh token on every use, so a refresh is
done under a row lock on the token: two runners refreshing the same token at
once would otherwise invalidate each other's copy and disconnect the account.

X bills per request (pay-per-use credits). A post containing a URL costs over
ten times a plain one, which is why ``contains_link`` exists: the mirror never
sends a caption with a link in it.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import logging
import re
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from sqlalchemy import select

from .config import DATA_DIR, env, load_settings

log = logging.getLogger("x")

API = "https://api.x.com/2"
AUTHORIZE = "https://x.com/i/oauth2/authorize"
TOKEN_URL = f"{API}/oauth2/token"
TOKEN_FILE = DATA_DIR / "x_token.json"
TOKEN_NAME = "x"

DEFAULT_SCOPES = "tweet.read tweet.write users.read media.write offline.access"

# Refresh when the access token has less than this long to live.
_REFRESH_WINDOW = dt.timedelta(minutes=10)

_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024

MAX_WEIGHTED_LENGTH = 280

_token_cache: dict | None = None
_token_cache_at: float = 0.0
_TOKEN_CACHE_TTL = 60.0

# state -> PKCE code_verifier for the authorize -> callback round trip.
# In-process only, and X's authorization code expires in 30 seconds, so the
# redirect has to land on this dashboard; there is no paste-the-code fallback.
_pending: dict[str, str] = {}
_MAX_PENDING = 20


class XError(RuntimeError):
    """An X API failure, carrying the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class XDuplicateError(XError):
    """X refused the post as a duplicate of one already on the account."""


# --- Token storage (DB-first, file fallback) ---------------------------------

def _cache(token: dict | None) -> None:
    global _token_cache, _token_cache_at
    _token_cache = dict(token) if token is not None else None
    _token_cache_at = time.monotonic()


def _write_file(payload: str) -> None:
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(payload)
    except Exception as exc:
        log.warning("Could not write local X token file: %s", exc)


def _save_token(token: dict) -> None:
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
        log.warning("Could not save X token to DB (file copy still written): %s", exc)
    _write_file(payload)
    _cache(token)


def _peek_token() -> dict | None:
    if _token_cache_at and (time.monotonic() - _token_cache_at) < _TOKEN_CACHE_TTL:
        return dict(_token_cache) if _token_cache is not None else None
    try:
        from .db import session_scope
        from .models import AppToken

        with session_scope() as session:
            row = session.get(AppToken, TOKEN_NAME)
            if row is not None and row.value:
                token = json.loads(row.value)
                _cache(token)
                return dict(token)
    except Exception as exc:
        log.warning("Could not read X token from DB (trying local file): %s", exc)
    if TOKEN_FILE.exists():
        token = json.loads(TOKEN_FILE.read_text())
        _save_token(token)
        return token
    _cache(None)
    return None


def _expires_at(token: dict) -> dt.datetime | None:
    try:
        obtained = dt.datetime.fromisoformat(token["obtained_at"])
        return obtained + dt.timedelta(seconds=int(token["expires_in"]))
    except (KeyError, TypeError, ValueError):
        return None


def _needs_refresh(token: dict) -> bool:
    expires = _expires_at(token)
    return expires is None or expires - dt.datetime.now(dt.timezone.utc) < _REFRESH_WINDOW


# --- OAuth -------------------------------------------------------------------

def is_configured() -> bool:
    return bool(env("X_CLIENT_ID") and env("X_REDIRECT_URI"))


def authorize_url() -> str:
    state = secrets.token_urlsafe(16)
    verifier = secrets.token_urlsafe(64)
    _pending[state] = verifier
    for stale in list(_pending)[:-_MAX_PENDING]:
        del _pending[stale]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    params = {
        "response_type": "code",
        "client_id": env("X_CLIENT_ID"),
        "redirect_uri": env("X_REDIRECT_URI"),
        "scope": env("X_SCOPES") or DEFAULT_SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTHORIZE}?{urlencode(params)}"


def pop_verifier(state: str) -> str | None:
    """Consume the PKCE verifier for a ``state`` issued by ``authorize_url``."""
    return _pending.pop(state, None)


def _token_request(data: dict) -> dict:
    # Confidential clients (an app with a client secret) authenticate with
    # HTTP Basic; public clients identify themselves by client_id alone.
    secret = env("X_CLIENT_SECRET")
    resp = requests.post(
        TOKEN_URL,
        data={**data, "client_id": env("X_CLIENT_ID")},
        auth=(env("X_CLIENT_ID"), secret) if secret else None,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise XError(f"token request failed: {resp.text[:300]}", status=resp.status_code)
    return resp.json()


def _token_from_response(data: dict, previous: dict | None = None) -> dict:
    token = dict(previous or {})
    token.update({
        "access_token": data["access_token"],
        "expires_in": data.get("expires_in", 7200),
        "obtained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": data.get("scope", token.get("scope", "")),
    })
    if data.get("refresh_token"):
        token["refresh_token"] = data["refresh_token"]
    return token


def exchange_code(code: str, verifier: str) -> dict:
    """Auth code -> access + refresh token. Looks the account up right away so
    the Accounts page can say which account it will post as."""
    data = _token_request({
        "grant_type": "authorization_code",
        "code": code.strip(),
        "redirect_uri": env("X_REDIRECT_URI"),
        "code_verifier": verifier,
    })
    token = _token_from_response(data)
    if not token.get("refresh_token"):
        raise XError("X issued no refresh token — the app must request the "
                     "offline.access scope, or the connection dies in two hours.")
    me = _request("GET", "users/me", token["access_token"], timeout=30).json().get("data", {})
    token["user_id"] = str(me.get("id", ""))
    token["username"] = str(me.get("username", ""))
    _save_token(token)
    return token


def _refresh_locked() -> dict:
    """Refresh under a row lock on the stored token, re-reading it first: if
    another runner refreshed while this one waited, its fresh token is used
    rather than spending the (now rotated, dead) refresh token again."""
    from .db import session_scope
    from .models import AppToken, utcnow

    with session_scope() as session:
        row = session.execute(
            select(AppToken).where(AppToken.name == TOKEN_NAME).with_for_update()
        ).scalar_one_or_none()
        if row is None or not row.value:
            raise XError("Not connected to X. Use the Accounts page to connect.")
        token = json.loads(row.value)
        if not _needs_refresh(token):
            _cache(token)
            return token
        if not token.get("refresh_token"):
            raise XError("The X token has expired. Reconnect X from the Accounts page.")
        try:
            data = _token_request({"grant_type": "refresh_token",
                                   "refresh_token": token["refresh_token"]})
        except XError as exc:
            if exc.status in (400, 401):
                raise XError("X rejected the refresh token, so the connection is gone. "
                             "Reconnect X from the Accounts page.", status=exc.status) from exc
            raise
        token = _token_from_response(data, previous=token)
        row.value = json.dumps(token, indent=1)
        row.updated_at = utcnow()
    _write_file(json.dumps(token, indent=1))
    _cache(token)
    log.info("Refreshed X token")
    return token


def is_authenticated() -> bool:
    try:
        token = _peek_token()
    except Exception:
        return False
    if not token or not token.get("access_token"):
        return False
    return bool(token.get("refresh_token")) or not _needs_refresh(token)


def connection_info() -> dict:
    token = _peek_token() or {}
    return {"username": token.get("username", ""), "user_id": token.get("user_id", "")}


def _access_token() -> tuple[str, dict]:
    token = _peek_token()
    if token is None:
        raise XError("Not connected to X. Use the Accounts page to connect.")
    if _needs_refresh(token):
        token = _refresh_locked()
    return token["access_token"], token


# --- REST plumbing -----------------------------------------------------------

def _request(method: str, path: str, access_token: str, **kwargs) -> requests.Response:
    resp = requests.request(method, f"{API}/{path}",
                            headers={"Authorization": f"Bearer {access_token}"},
                            timeout=kwargs.pop("timeout", 60), **kwargs)
    if resp.status_code >= 400:
        try:
            body = resp.json()
            errors = body.get("errors") or [{}]
            detail = (body.get("detail") or errors[0].get("message")
                      or body.get("title") or resp.text)
        except ValueError:
            detail = resp.text
        log.warning("X API %s %s -> %s: %s", method, path.split("?")[0],
                    resp.status_code, resp.text[:1000])
        if resp.status_code == 402:
            raise XError("X API credits are used up. Top up at console.x.com, then retry.",
                         status=402)
        message = (f"X {method} {path.split('?')[0]} failed "
                   f"({resp.status_code}): {str(detail)[:300]}")
        if resp.status_code == 403 and "duplicate" in str(detail).lower():
            raise XDuplicateError(message, status=403)
        raise XError(message, status=resp.status_code)
    return resp


# --- Text --------------------------------------------------------------------

# Anything X would turn into a link: a scheme or www. prefix, or a bare domain
# like "Renewables.org". Errs toward matching — a false match only means the
# post skips X, a miss means paying the link rate.
_LINK = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.[a-z]{2,24}\b(?:/\S*)?",
    re.IGNORECASE,
)


def contains_link(text: str) -> bool:
    return bool(_LINK.search(text or ""))


def weighted_length(text: str) -> int:
    """X's character count: most Latin/punctuation counts 1, everything else
    (CJK, emoji) counts 2. Counts each emoji code point, so ZWJ sequences
    come out longer than X counts them — safe for a limit check."""
    total = 0
    for ch in text or "":
        cp = ord(ch)
        light = (cp <= 4351 or 8192 <= cp <= 8205 or 8208 <= cp <= 8223
                 or 8242 <= cp <= 8247)
        total += 1 if light else 2
    return total


# --- Publishing ----------------------------------------------------------------

def _upload_video(access_token: str, video: Path) -> str:
    """initialize -> append each chunk -> finalize. Returns the media id.
    Reads one chunk at a time for the same reason as the LinkedIn upload: the
    Fly worker can't hold a whole clip in memory."""
    media_id = _request("POST", "media/upload/initialize", access_token, json={
        "media_type": "video/mp4",
        "total_bytes": video.stat().st_size,
        "media_category": "tweet_video",
    }).json()["data"]["id"]
    with open(video, "rb") as f:
        index = 0
        while chunk := f.read(_UPLOAD_CHUNK_BYTES):
            _request("POST", f"media/upload/{media_id}/append", access_token,
                     files={"media": ("chunk", chunk, "application/octet-stream")},
                     data={"segment_index": str(index)}, timeout=300)
            index += 1
    info = _request("POST", f"media/upload/{media_id}/finalize", access_token
                    ).json().get("data", {}).get("processing_info")
    _wait_until_processed(access_token, media_id, info)
    return media_id


def _wait_until_processed(access_token: str, media_id: str, info: dict | None) -> None:
    timeout = int(load_settings().get("x.publish_poll_timeout_seconds", 600))
    deadline = time.time() + timeout
    while info:
        state = info.get("state")
        if state == "succeeded":
            return
        if state == "failed":
            err = info.get("error") or {}
            raise XError(f"X could not process the video: "
                         f"{err.get('message') or err.get('name') or 'unknown error'}")
        if time.time() >= deadline:
            raise XError(
                f"Timed out waiting for X to process the video after {timeout}s; "
                "raise x.publish_poll_timeout_seconds if this keeps happening.")
        time.sleep(max(1, int(info.get("check_after_secs") or 5)))
        info = _request("GET", "media/upload", access_token, timeout=30,
                        params={"command": "STATUS", "media_id": media_id}
                        ).json().get("data", {}).get("processing_info")


def permalink_for(tweet_id: str, username: str = "") -> str:
    if not tweet_id:
        return ""
    return f"https://x.com/{username or 'i'}/status/{tweet_id}"


def publish_video(video: Path, text: str) -> dict:
    """Post a video with ``text`` to the connected account. Returns
    {tweet_id, media_id, permalink}."""
    access_token, token = _access_token()
    media_id = _upload_video(access_token, video)
    resp = _request("POST", "tweets", access_token, json={
        "text": text,
        "media": {"media_ids": [media_id]},
    })
    tweet_id = str(resp.json().get("data", {}).get("id", ""))
    return {"tweet_id": tweet_id, "media_id": media_id,
            "permalink": permalink_for(tweet_id, token.get("username", ""))}
