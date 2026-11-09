"""Facebook Pages API client: video posts on a Facebook Page.

Posts as the PAGE, with a Page access token, not as the person who connected
it — they only have to manage the Page. Scopes: ``pages_show_list`` to find the
Page, ``pages_manage_posts`` to post, ``pages_read_engagement`` to read a
video back (status, permalink, copyright check), ``pages_manage_engagement``
to post the first comment as the Page.

Connect flow: Facebook Login code -> short-lived user token -> long-lived user
token -> ``/me/accounts`` -> the Page's token. A Page token derived from a
long-lived user token has no expiry, so there is nothing to refresh and the
headless runners need no app secret. It dies only if the person loses their
role on the Page, changes their password, or removes the app; the API then
answers code 190 and the Accounts page asks for a reconnect.

Same storage shape as app/linkedin_api.py: the token lives in the shared
database (``app_tokens``, name ``facebook``) with data/facebook_token.json as
a gitignored backup.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

from . import meta_errors
from .config import DATA_DIR, env, load_settings

log = logging.getLogger("facebook")

GRAPH_VERSION = "v23.0"
GRAPH = f"https://graph.facebook.com/{GRAPH_VERSION}"
GRAPH_VIDEO = f"https://graph-video.facebook.com/{GRAPH_VERSION}"
DIALOG = f"https://www.facebook.com/{GRAPH_VERSION}/dialog/oauth"
TOKEN_FILE = DATA_DIR / "facebook_token.json"
TOKEN_NAME = "facebook"

DEFAULT_SCOPES = ("pages_show_list,pages_manage_posts,pages_read_engagement,"
                  "pages_manage_engagement")

_token_cache: dict | None = None
_token_cache_at: float = 0.0
_TOKEN_CACHE_TTL = 60.0

# CSRF state for the authorize -> callback round trip (see linkedin_api).
_pending_states: list[str] = []
_MAX_PENDING_STATES = 20

# Anything in an error that reads as a rights claim. Facebook has no stable
# code for these, so the wording is all there is to go on.
_COPYRIGHT = re.compile(r"copyright|rights manager|intellectual property|\bIP\b violation",
                        re.IGNORECASE)


class FacebookError(RuntimeError):
    """A Graph API failure, carrying the HTTP status and Meta error code."""

    def __init__(self, message: str, *, status: int | None = None,
                 code: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code

    @property
    def is_copyright(self) -> bool:
        return bool(_COPYRIGHT.search(str(self)))

    @property
    def is_missing(self) -> bool:
        """The object isn't there (deleted, or never was)."""
        return self.code == 100 and "does not exist" in str(self).lower()


# --- Token storage (DB-first, file fallback) ---------------------------------

def _cache(token: dict | None) -> None:
    global _token_cache, _token_cache_at
    _token_cache = dict(token) if token is not None else None
    _token_cache_at = time.monotonic()


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
        log.warning("Could not save Facebook token to DB (file copy still written): %s", exc)
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(payload)
    except Exception as exc:
        log.warning("Could not write local Facebook token file: %s", exc)
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
        log.warning("Could not read Facebook token from DB (trying local file): %s", exc)
    if TOKEN_FILE.exists():
        token = json.loads(TOKEN_FILE.read_text())
        _save_token(token)
        return token
    _cache(None)
    return None


# --- REST plumbing -----------------------------------------------------------

def _request(method: str, path: str, *, host: str = GRAPH, ids: dict | None = None,
             **kwargs) -> requests.Response:
    """``path`` is a template (``{page_id}/videos``); ``ids`` fills it in. The
    template, not the filled path, is what reaches error messages."""
    url = f"{host}/{path.format(**(ids or {}))}"
    resp = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
    if resp.status_code >= 400:
        message, code, _ = meta_errors.describe(method, path, resp)
        log.warning("Facebook API %s %s -> %s: %s", method, path, resp.status_code,
                    meta_errors.raw_body(resp))
        if code == 190:
            message = ("Facebook rejected the Page token, so the connection is gone. "
                       "Reconnect Facebook from the Accounts page. " + message)
        raise FacebookError(message, status=resp.status_code, code=code)
    return resp


# --- OAuth -------------------------------------------------------------------

def is_configured() -> bool:
    return bool(env("FACEBOOK_APP_ID") and env("FACEBOOK_APP_SECRET")
                and env("FACEBOOK_REDIRECT_URI"))


def authorize_url() -> str:
    state = secrets.token_urlsafe(16)
    _pending_states.append(state)
    del _pending_states[:-_MAX_PENDING_STATES]
    params = {
        "client_id": env("FACEBOOK_APP_ID"),
        "redirect_uri": env("FACEBOOK_REDIRECT_URI"),
        "state": state,
        "response_type": "code",
    }
    # Facebook Login for Business can insist on a saved configuration (which
    # carries its own permission list) instead of a scope list.
    if env("FACEBOOK_CONFIG_ID"):
        params["config_id"] = env("FACEBOOK_CONFIG_ID")
    else:
        params["scope"] = env("FACEBOOK_SCOPES") or DEFAULT_SCOPES
    return f"{DIALOG}?{urlencode(params)}"


def check_state(state: str) -> bool:
    """Consume an OAuth ``state`` issued by ``authorize_url`` in this process."""
    if state in _pending_states:
        _pending_states.remove(state)
        return True
    return False


def _choose_page(pages: list[dict]) -> dict:
    """The configured Page, or the only one the person manages."""
    wanted = str(load_settings().get("facebook.page_id", "") or "").strip()
    if wanted:
        for page in pages:
            if str(page.get("id")) == wanted:
                return page
        raise FacebookError(
            f"facebook.page_id is {wanted}, but that Page wasn't among the ones this "
            f"login manages ({_page_list(pages) or 'none'}). Check the ID, or grant the "
            "app access to that Page on Facebook's consent screen.")
    if len(pages) == 1:
        return pages[0]
    if not pages:
        raise FacebookError(
            "This login manages no Pages the app was given access to. Reconnect and "
            "tick the Page on Facebook's consent screen.")
    raise FacebookError(
        f"This login manages several Pages ({_page_list(pages)}). Set facebook.page_id "
        "in settings.yaml to the one to post to, then connect again.")


def _page_list(pages: list[dict]) -> str:
    return ", ".join(f"{p.get('name', '?')} = {p.get('id', '?')}" for p in pages)


def exchange_code(code: str) -> dict:
    """Auth code -> the Page's token, stored. Returns the stored token."""
    app = {"client_id": env("FACEBOOK_APP_ID"), "client_secret": env("FACEBOOK_APP_SECRET")}
    short = _request("GET", "oauth/access_token", params={
        **app, "redirect_uri": env("FACEBOOK_REDIRECT_URI"), "code": code.strip(),
    }, timeout=30).json()["access_token"]
    long_lived = _request("GET", "oauth/access_token", params={
        **app, "grant_type": "fb_exchange_token", "fb_exchange_token": short,
    }, timeout=30).json()["access_token"]
    pages = _request("GET", "me/accounts", params={
        "access_token": long_lived, "fields": "id,name,access_token", "limit": 100,
    }, timeout=30).json().get("data", [])
    page = _choose_page(pages)
    token = {
        "page_id": str(page["id"]),
        "page_name": page.get("name", ""),
        "access_token": page["access_token"],
        "obtained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _save_token(token)
    return token


def is_authenticated() -> bool:
    try:
        token = _peek_token()
    except Exception:
        return False
    return bool(token and token.get("access_token") and token.get("page_id"))


def connection_info() -> dict:
    token = _peek_token() or {}
    return {"page_name": token.get("page_name", ""), "page_id": token.get("page_id", "")}


def _page() -> tuple[str, str]:
    token = _peek_token()
    if not token or not token.get("access_token"):
        raise FacebookError("Not connected to Facebook. Use the Accounts page to connect.")
    return token["page_id"], token["access_token"]


# --- Publishing ----------------------------------------------------------------

def permalink_for(page_id: str, video_id: str) -> str:
    if not video_id:
        return ""
    return f"https://www.facebook.com/{page_id}/videos/{video_id}"


def publish_video(video: Path, description: str) -> dict:
    """Post a video to the Page with ``description`` as its text, through the
    resumable upload (start -> transfer each chunk -> finish). Facebook picks
    the chunk boundaries; one chunk at a time is read, never the whole file —
    the Fly worker can't hold a clip in memory. Returns {video_id, permalink}.

    The post goes live once Facebook finishes processing; ``video_status``
    is how the scheduler later learns whether that worked."""
    page_id, access_token = _page()
    ids = {"page_id": page_id}
    start = _request("POST", "{page_id}/videos", host=GRAPH_VIDEO, ids=ids, data={
        "access_token": access_token, "upload_phase": "start",
        "file_size": str(video.stat().st_size),
    }).json()
    session_id = start["upload_session_id"]
    video_id = str(start["video_id"])
    offset, end = int(start["start_offset"]), int(start["end_offset"])
    with open(video, "rb") as f:
        while offset < end:
            f.seek(offset)
            chunk = f.read(end - offset)
            nxt = _request("POST", "{page_id}/videos", host=GRAPH_VIDEO, ids=ids, data={
                "access_token": access_token, "upload_phase": "transfer",
                "upload_session_id": session_id, "start_offset": str(offset),
            }, files={"video_file_chunk": ("chunk", chunk, "application/octet-stream")},
                timeout=300).json()
            offset, end = int(nxt["start_offset"]), int(nxt["end_offset"])
    _request("POST", "{page_id}/videos", host=GRAPH_VIDEO, ids=ids, data={
        "access_token": access_token, "upload_phase": "finish",
        "upload_session_id": session_id, "description": description,
    })
    return {"video_id": video_id, "permalink": permalink_for(page_id, video_id)}


def video_status(video_id: str) -> dict:
    """Where a posted video stands. Returns {state, copyright_match, permalink,
    detail}:

    - ``state``: ``ready``, ``processing``, ``error`` (processing failed, nothing
      went out) or ``missing`` (gone from the Page: deleted by hand, or taken
      down by a rights claim).
    - ``copyright_match``: True when Facebook's copyright check matched the
      video against someone's reference content.
    """
    page_id, access_token = _page()
    try:
        data = _request("GET", "{video_id}", ids={"video_id": video_id}, params={
            "access_token": access_token, "fields": "status,permalink_url",
        }, timeout=30).json()
    except FacebookError as exc:
        if exc.is_missing:
            return {"state": "missing", "copyright_match": False, "permalink": ""}
        raise
    status = data.get("status") or {}
    copyright_check = status.get("copyright_check_status") or {}
    state = str(status.get("video_status") or "processing").lower()
    if state not in ("ready", "error"):
        state = "processing"
    link = data.get("permalink_url") or ""
    if link.startswith("/"):
        link = f"https://www.facebook.com{link}"
    errors = (status.get("processing_phase") or {}).get("errors") or []
    detail = "; ".join(str(e.get("message", "")) for e in errors if isinstance(e, dict))
    return {"state": state, "copyright_match": bool(copyright_check.get("matches_found")),
            "permalink": link or permalink_for(page_id, video_id), "detail": detail}


def publish_comment(video_id: str, message: str) -> str:
    """Comment on a Page video as the Page. Returns the comment id."""
    _, access_token = _page()
    resp = _request("POST", "{video_id}/comments", ids={"video_id": video_id}, data={
        "access_token": access_token, "message": message,
    }, timeout=30)
    return str(resp.json().get("id", ""))
