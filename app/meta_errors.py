"""Readable messages for Meta Graph API failures.

Shared by the Threads and Instagram clients: different hosts, identical error
envelope. Raw ``resp.text`` is unusable in the UI — it is JSON carrying
``\\uXXXX``-escaped, Meta-localized prose, and the request path has already had
the account id substituted into it, so a stored error read as 400 characters of
escapes with the operator's own user id in front of it.

Codes worth knowing:
- 24 / 4279009: the creation_id handed to a publish call isn't there. Meta
  hasn't finished registering the container yet — poll it to FINISHED first.
"""
from __future__ import annotations

import json

# Meta's transient "that container doesn't exist (yet)" pair. Nothing was
# published when this comes back, which is what makes a retry safe.
MISSING_RESOURCE = (24, 4279009)

# Graph paths -> what the operator would call them.
_OPERATIONS = {
    "threads": "Create post",
    "threads_publish": "Publish post",
    "media": "Create reel",
    "media_publish": "Publish reel",
    "content_publishing_limit": "Check publishing quota",
    "access_token": "Exchange access token",
    "refresh_access_token": "Refresh access token",
    "insights": "Fetch insights",
    "replies": "Fetch replies",
    "conversation": "Fetch conversation",
}


def envelope(resp) -> dict:
    """The ``error`` object from a Graph response, or {} when it isn't JSON."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def operation(method: str, path: str) -> str:
    """A label for the call. ``path`` must be the *un-substituted* template, so
    the account id can never reach an error string the operator reads."""
    parts = [p for p in path.replace("{user_id}", "me").split("/") if p]
    named = _OPERATIONS.get(parts[-1]) if parts else None
    if named:
        return named
    if len(parts) == 1 and parts[0].isdigit():
        return "Read media"
    return f"{method} {'/'.join(parts) or 'me'}"


def describe(method: str, path: str, resp) -> tuple[str, int | None, int | None]:
    """Render a failure as (message, code, subcode).

    The message is one line fit for the notifications list and the post page.
    Callers log the raw body separately — nothing here should be so long that
    it buries the next error in a 1000-character database column.
    """
    error = envelope(resp)
    code = error.get("code")
    subcode = error.get("error_subcode")
    detail = (error.get("message") or error.get("error_user_title") or "").strip()
    if not detail:
        detail = f"HTTP {resp.status_code}"
    codes = ", ".join(
        f"{label} {value}" for label, value in (("code", code), ("subcode", subcode))
        if value is not None
    )
    message = f"{operation(method, path)} failed: {detail}"
    return (f"{message} ({codes})" if codes else message,
            code if isinstance(code, int) else None,
            subcode if isinstance(subcode, int) else None)


def raw_body(resp, limit: int = 600) -> str:
    """The full envelope with escapes decoded, for the log rather than the UI."""
    error = envelope(resp)
    if not error:
        return resp.text[:limit]
    return json.dumps(error, ensure_ascii=False)[:limit]
