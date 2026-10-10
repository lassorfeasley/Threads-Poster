"""LinkedIn Community Management API client: video posts on a company page.

Posts as the ORGANIZATION (``urn:li:organization:<id>``), not as the member
who authorized the app — that member only has to be an admin of the page.
Scopes: ``w_organization_social`` to post, ``rw_organization_admin`` to find
the page the member administers (skipped when ``linkedin.organization_id`` is
set in settings.yaml).

Same storage shape as app/instagram_api.py: the token lives in the shared
database (``app_tokens``, name ``linkedin``) so headless runners can publish
without this machine, with data/linkedin_token.json as a gitignored backup.
Access tokens last 60 days. Apps approved for programmatic refresh also get a
refresh token (~1 year), used when the access token is within 7 days of
expiry; without one the operator reconnects from the Accounts page.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import secrets
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import requests

from .config import DATA_DIR, env, load_settings

log = logging.getLogger("linkedin")

API = "https://api.linkedin.com/rest"
OAUTH = "https://www.linkedin.com/oauth/v2"
TOKEN_FILE = DATA_DIR / "linkedin_token.json"
TOKEN_NAME = "linkedin"

DEFAULT_SCOPES = "w_organization_social r_organization_social rw_organization_admin"

# Refresh (or warn) when the access token has less than this long to live.
_REFRESH_WINDOW = dt.timedelta(days=7)

_token_cache: dict | None = None
_token_cache_at: float = 0.0
_TOKEN_CACHE_TTL = 60.0

# CSRF state for the authorize -> callback round trip. In-process only: a
# code pasted by hand into the Accounts page carries no state to check.
# Every Accounts page render mints one, so only the newest few are kept.
_pending_states: list[str] = []
_MAX_PENDING_STATES = 20


class LinkedInError(RuntimeError):
    """A LinkedIn API failure, carrying the HTTP status when there was one."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


# --- Token storage (DB-first, file fallback) ---------------------------------

def _save_token(token: dict) -> None:
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
        log.warning("Could not save LinkedIn token to DB (file copy still written): %s", exc)
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(payload)
    except Exception as exc:
        log.warning("Could not write local LinkedIn token file: %s", exc)
    _token_cache = dict(token)
    _token_cache_at = time.monotonic()


def _peek_token() -> dict | None:
    global _token_cache, _token_cache_at
    if _token_cache_at and (time.monotonic() - _token_cache_at) < _TOKEN_CACHE_TTL:
        return dict(_token_cache) if _token_cache is not None else None
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
        log.warning("Could not read LinkedIn token from DB (trying local file): %s", exc)
    if TOKEN_FILE.exists():
        token = json.loads(TOKEN_FILE.read_text())
        _save_token(token)
        return token
    _token_cache = None
    _token_cache_at = time.monotonic()
    return None


def _expires_at(token: dict, key: str = "expires_in") -> dt.datetime | None:
    try:
        obtained = dt.datetime.fromisoformat(token["obtained_at"])
        return obtained + dt.timedelta(seconds=int(token[key]))
    except (KeyError, TypeError, ValueError):
        return None


# --- OAuth -------------------------------------------------------------------

def is_configured() -> bool:
    return bool(env("LINKEDIN_CLIENT_ID") and env("LINKEDIN_CLIENT_SECRET")
                and env("LINKEDIN_REDIRECT_URI"))


def authorize_url() -> str:
    state = secrets.token_urlsafe(16)
    _pending_states.append(state)
    del _pending_states[:-_MAX_PENDING_STATES]
    params = {
        "response_type": "code",
        "client_id": env("LINKEDIN_CLIENT_ID"),
        "redirect_uri": env("LINKEDIN_REDIRECT_URI"),
        "state": state,
        "scope": env("LINKEDIN_SCOPES") or DEFAULT_SCOPES,
    }
    return f"{OAUTH}/authorization?{urlencode(params)}"


def check_state(state: str) -> bool:
    """Consume an OAuth ``state`` issued by ``authorize_url`` in this process."""
    if state in _pending_states:
        _pending_states.remove(state)
        return True
    return False


def _token_request(data: dict) -> dict:
    resp = requests.post(
        f"{OAUTH}/accessToken",
        data={**data,
              "client_id": env("LINKEDIN_CLIENT_ID"),
              "client_secret": env("LINKEDIN_CLIENT_SECRET")},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise LinkedInError(f"token request failed: {resp.text[:300]}",
                            status=resp.status_code)
    return resp.json()


def _token_from_response(data: dict, previous: dict | None = None) -> dict:
    token = dict(previous or {})
    token.update({
        "access_token": data["access_token"],
        "expires_in": data.get("expires_in", 5184000),
        "obtained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": data.get("scope", token.get("scope", "")),
    })
    if data.get("refresh_token"):
        token["refresh_token"] = data["refresh_token"]
        token["refresh_token_expires_in"] = data.get("refresh_token_expires_in")
        token["refresh_obtained_at"] = token["obtained_at"]
    return token


def exchange_code(code: str) -> dict:
    """Auth code -> access token (+ refresh token when the app is approved for
    it). Resolves the company page right away so a wrong account fails here,
    on the Accounts page, rather than at the first scheduled publish."""
    data = _token_request({
        "grant_type": "authorization_code",
        "code": code.strip(),
        "redirect_uri": env("LINKEDIN_REDIRECT_URI"),
    })
    token = _token_from_response(data)
    _save_token(token)
    token["organization_urn"] = _resolve_organization(token["access_token"])
    token["organization_name"] = _organization_name(token["access_token"],
                                                    token["organization_urn"])
    _save_token(token)
    return token


def _maybe_refresh(token: dict) -> dict:
    expires = _expires_at(token)
    if expires is None or expires - dt.datetime.now(dt.timezone.utc) > _REFRESH_WINDOW:
        return token
    refresh = token.get("refresh_token")
    if not refresh or not env("LINKEDIN_CLIENT_SECRET"):
        if expires <= dt.datetime.now(dt.timezone.utc):
            raise LinkedInError(
                "The LinkedIn token has expired. Reconnect LinkedIn from the "
                "Accounts page.")
        return token
    try:
        data = _token_request({"grant_type": "refresh_token", "refresh_token": refresh})
    except LinkedInError as exc:
        log.warning("LinkedIn token refresh failed (keeping current token): %s", exc)
        return token
    token = _token_from_response(data, previous=token)
    _save_token(token)
    log.info("Refreshed LinkedIn token")
    return token


def is_authenticated() -> bool:
    try:
        token = _peek_token()
    except Exception:
        return False
    if not token:
        return False
    expires = _expires_at(token)
    if expires is not None and expires <= dt.datetime.now(dt.timezone.utc) \
            and not token.get("refresh_token"):
        return False
    return True


def connection_info() -> dict:
    """What the Accounts page shows: page name, URN, and when the operator
    will have to reconnect (None when a refresh token keeps it alive)."""
    token = _peek_token() or {}
    expires = _expires_at(token)
    refresh_expires = None
    if token.get("refresh_token"):
        refresh_expires = _expires_at(
            {"obtained_at": token.get("refresh_obtained_at", token.get("obtained_at")),
             "expires_in": token.get("refresh_token_expires_in")})
    reconnect_by = refresh_expires if token.get("refresh_token") else expires
    days_left = None
    if reconnect_by is not None:
        days_left = max(0, (reconnect_by - dt.datetime.now(dt.timezone.utc)).days)
    return {
        "organization_name": token.get("organization_name", ""),
        "organization_urn": token.get("organization_urn", ""),
        "reconnect_in_days": days_left,
        "auto_refresh": bool(token.get("refresh_token")),
    }


# --- REST plumbing -----------------------------------------------------------

def _headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "LinkedIn-Version": str(load_settings().get("linkedin.api_version", "202609")),
        "X-Restli-Protocol-Version": "2.0.0",
    }


def _request(method: str, path: str, access_token: str, **kwargs) -> requests.Response:
    resp = requests.request(method, f"{API}/{path}", headers=_headers(access_token),
                            timeout=kwargs.pop("timeout", 60), **kwargs)
    if resp.status_code >= 400:
        try:
            body = resp.json()
            detail = body.get("message") or body.get("error_description") or resp.text
        except ValueError:
            detail = resp.text
        log.warning("LinkedIn API %s %s -> %s: %s", method, path.split("?")[0],
                    resp.status_code, resp.text[:1000])
        raise LinkedInError(f"LinkedIn {method} {path.split('?')[0]} failed "
                            f"({resp.status_code}): {str(detail)[:300]}",
                            status=resp.status_code)
    return resp


def _resolve_organization(access_token: str) -> str:
    """The company page to post as: ``linkedin.organization_id`` when set,
    else the single page this member administers."""
    configured = str(load_settings().get("linkedin.organization_id", "") or "").strip()
    if configured:
        return configured if configured.startswith("urn:") else f"urn:li:organization:{configured}"
    resp = _request("GET", "organizationAcls?q=roleAssignee&role=ADMINISTRATOR&state=APPROVED",
                    access_token)
    orgs = sorted({e.get("organization") for e in resp.json().get("elements", [])
                   if e.get("organization")})
    if len(orgs) == 1:
        return orgs[0]
    if not orgs:
        raise LinkedInError(
            "This LinkedIn member isn't an approved admin of any company page. "
            "Reconnect as a page admin, or set linkedin.organization_id in settings.yaml.")
    raise LinkedInError(
        "This LinkedIn member administers several pages "
        f"({', '.join(orgs)}). Set linkedin.organization_id in settings.yaml to "
        "the one to post to, then reconnect.")


def _organization_name(access_token: str, org_urn: str) -> str:
    try:
        org_id = org_urn.rsplit(":", 1)[-1]
        resp = _request("GET", f"organizations/{org_id}", access_token, timeout=30)
        return str(resp.json().get("localizedName") or "")
    except Exception as exc:
        log.info("Could not look up LinkedIn page name for %s: %s", org_urn, exc)
        return ""


def _auth() -> tuple[str, str]:
    token = _peek_token()
    if token is None:
        raise LinkedInError("Not connected to LinkedIn. Use the Accounts page to connect.")
    token = _maybe_refresh(token)
    org = token.get("organization_urn")
    if not org:
        org = _resolve_organization(token["access_token"])
        token["organization_urn"] = org
        _save_token(token)
    return token["access_token"], org


# --- Commentary ----------------------------------------------------------------

# Characters LinkedIn's "little text" format reserves. Left unescaped, the API
# silently truncates the post at the first one (a parenthesis is enough).
_RESERVED = re.compile(r"([\\|{}@\[\]()<>#*_~])")
_HASHTAG = re.compile(r"(?<![\w#])#(\w+)")


def to_commentary(caption: str) -> str:
    """A Threads caption as LinkedIn post commentary: reserved characters
    escaped, ``#tags`` kept as real hashtags."""
    out: list[str] = []
    pos = 0
    for m in _HASHTAG.finditer(caption):
        out.append(_RESERVED.sub(r"\\\1", caption[pos:m.start()]))
        out.append("{hashtag|\\#|" + m.group(1) + "}")
        pos = m.end()
    out.append(_RESERVED.sub(r"\\\1", caption[pos:]))
    return "".join(out)


# --- Publishing ----------------------------------------------------------------

def _upload_video(access_token: str, org: str, data: bytes) -> str:
    """initializeUpload -> PUT each part -> finalizeUpload. Returns the video URN."""
    init = _request("POST", "videos?action=initializeUpload", access_token, json={
        "initializeUploadRequest": {
            "owner": org,
            "fileSizeBytes": len(data),
            "uploadCaptions": False,
            "uploadThumbnail": False,
        }
    }).json()["value"]
    video_urn = init["video"]
    etags: list[str] = []
    for part in init.get("uploadInstructions", []):
        first, last = int(part["firstByte"]), int(part["lastByte"])
        resp = requests.put(part["uploadUrl"], data=data[first:last + 1],
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=300)
        if resp.status_code >= 400:
            raise LinkedInError(f"Video part upload failed ({resp.status_code}): "
                                f"{resp.text[:200]}", status=resp.status_code)
        etags.append(resp.headers.get("ETag", "").strip('"'))
    _request("POST", "videos?action=finalizeUpload", access_token, json={
        "finalizeUploadRequest": {
            "video": video_urn,
            "uploadToken": init.get("uploadToken", ""),
            "uploadedPartIds": etags,
        }
    })
    return video_urn


def _wait_until_available(access_token: str, video_urn: str) -> None:
    settings = load_settings()
    timeout = int(settings.get("linkedin.publish_poll_timeout_seconds", 600))
    interval = max(3, int(settings.get("linkedin.publish_poll_interval_seconds", 10)))
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = _request("GET", f"videos/{quote(video_urn, safe='')}",
                          access_token, timeout=30).json().get("status")
        if status == "AVAILABLE":
            return
        if status == "PROCESSING_FAILED":
            raise LinkedInError("LinkedIn could not process the video.")
        time.sleep(interval)
    raise LinkedInError(
        f"Timed out waiting for LinkedIn to process the video after {timeout}s; "
        "raise linkedin.publish_poll_timeout_seconds if this keeps happening.")


def permalink_for(post_urn: str) -> str:
    return f"https://www.linkedin.com/feed/update/{post_urn}/" if post_urn else ""


def publish_video(video: bytes | Path, caption: str) -> dict:
    """Post a video to the company page. Returns {post_urn, video_urn, permalink}."""
    access_token, org = _auth()
    data = video.read_bytes() if isinstance(video, Path) else video
    video_urn = _upload_video(access_token, org, data)
    _wait_until_available(access_token, video_urn)
    resp = _request("POST", "posts", access_token, json={
        "author": org,
        "commentary": to_commentary(caption),
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "content": {"media": {"id": video_urn}},
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    })
    post_urn = resp.headers.get("x-restli-id", "")
    return {"post_urn": post_urn, "video_urn": video_urn,
            "permalink": permalink_for(post_urn)}
