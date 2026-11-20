"""Gemini over plain REST: schema-constrained generateContent and the Files API.

Why REST rather than the google-genai SDK: no new dependency (the scheduler
image installs only requirements-scheduler.txt), and the few endpoints used
here are small, stable JSON.

What callers (shotlog, editor, critic) can rely on:

- ``generate_json`` returns ``(data, usage)`` or raises a ``GeminiError``
  subclass. Spend is recorded from ``usageMetadata`` BEFORE anything else is
  checked, so blocked, truncated and unparseable answers are still counted, and
  the error carries ``.usage`` for the engine's cost meter — an error-prone run
  must not look cheaper than it was.
- Transient failures (408/429/5xx, dropped connections) are retried with
  backoff, but a costly one only once: a client read timeout, a 504, or any
  failure that came after a minute or more of waiting. And no retry starts
  once a call has run for twice its read timeout. A 900 s timeout retried
  four times would hold the dashboard's engine worker for over an hour.
- A 429 that names a per-DAY quota won't clear by retrying, so it opens a
  process-wide breaker for 30 minutes: every call fails fast with
  ``GeminiQuota`` and jobs go straight to the transcript suggester instead of
  burning retries all day. A 402 (prepaid credits depleted — seen live on
  both generateContent and the upload start) opens the same breaker: it is
  never transient, and its message tells the operator where to add credit.
  ``reset_breaker`` closes it early, for a run the operator asks for after
  adding credit.
- A 400 that names one of the newer request fields (``systemInstruction``,
  thinking, ``mediaResolution``, ``responseJsonSchema``) is retried once
  without it, so a model that lacks a feature degrades instead of failing every
  run.
- Never temperature/topP/topK: Gemini 3.6+ ignores them today and will reject
  them later.

Video goes inline (base64) while a request's raw video bytes stay under a
budget — Google's docs disagree on 20 MB vs 100 MB per request, so ~14 MiB by
default — and through the resumable Files API otherwise. Uploads are cached by
(path, size, mtime) for 46 h (Google keeps files 48 h), so taste examples and a
source's proxy go up once and are reused across drafts, critics and jobs. An
upload belongs to the API key's project, so it is reused only under the key
that made it, and one that a request's 4xx names is forgotten.
"""
from __future__ import annotations

import base64
import datetime as dt
import email.utils
import hashlib
import json
import logging
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
import urllib3

from .. import spend
from ..config import env
from . import engine_dir, opt, util

log = logging.getLogger("clip_engine.gemini")

BASE = "https://generativelanguage.googleapis.com"
_API = f"{BASE}/v1beta"
_UPLOAD_URL = f"{BASE}/upload/v1beta/files"

_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_MAX_RETRIES = 4
_MAX_BACKOFF = 60.0
_CONNECT_TIMEOUT = 15.0
# A failure that kept a request waiting this long (a 5xx after minutes of
# server-side work, a connection dropped mid-answer) cost as much as a read
# timeout, so it shares the read timeout's single retry.
_SLOW_FAILURE = 60.0
# No retry starts once a call has run for twice its read timeout, or for this
# long if that is more: room for four full backoffs, so quick 503s and
# per-minute 429s keep all their retries on short-timeout calls.
_MIN_RETRY_WINDOW = 5 * 60.0
_BREAKER_SECONDS = 30 * 60
# Google deletes uploads after 48 h; stop reusing them comfortably before that.
_FILE_TTL = 46 * 3600
# Older than this, an upload is confirmed ACTIVE (GET) before it is reused...
_FILE_RECHECK = 3600
# ...unless this process confirmed it within the last few minutes.
_CONFIRM_MEMO = 600
_POLL_SECONDS = 3.0
_UPLOAD_ATTEMPTS = 3
_DEFAULT_INLINE_MB = 14
_BLOCKED_FINISHES = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
                               "RECITATION", "IMAGE_SAFETY"})
_DAILY_QUOTA_MARKERS = ("perday", "per day", "requestsperday")
# Why the breaker is open, worded for the operator: it reaches the job status
# and the suggestion page through the fallback note, so the cause comes first.
_DAILY_QUOTA = "Gemini's daily quota is exhausted"
_CREDITS_DEPLETED = ("Gemini prepaid credits are depleted — add credit at "
                     "https://ai.studio/projects")


class GeminiError(RuntimeError):
    """A Gemini call failed. ``usage`` is set when the failed response was
    billed (it has already been recorded in the spend ledger); ``status`` is
    the HTTP status when there was one."""

    usage: dict | None = None

    def __init__(self, message: str = "", *, usage: dict | None = None,
                 status: int | None = None) -> None:
        super().__init__(message)
        self.usage = usage
        self.status = status


class GeminiBlocked(GeminiError):
    """The prompt or the answer was blocked (promptFeedback.blockReason or a
    safety/recitation finishReason)."""


class GeminiTruncated(GeminiError):
    """The answer hit maxOutputTokens (thinking counts toward it)."""


class GeminiQuota(GeminiError):
    """The daily quota or the prepaid credits are exhausted, or the breaker
    that opened is still open."""


# --- seams (tests replace these) ------------------------------------------------

def _send(method: str, url: str, **kwargs: Any) -> requests.Response:
    """The one place this module touches the network."""
    return requests.request(method, url, **kwargs)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _now() -> float:
    """Monotonic clock for the breaker, polling and confirm memos."""
    return time.monotonic()


# --- key, breaker ----------------------------------------------------------------

def is_configured() -> bool:
    return bool(env("GEMINI_API_KEY", "").strip())


def api_key() -> str:
    key = env("GEMINI_API_KEY", "").strip()
    if not key:
        raise GeminiError("GEMINI_API_KEY is not set (see .env.example)")
    return key


_breaker_lock = threading.Lock()
_breaker_until = 0.0
_breaker_reason = ""          # Google's own message
_breaker_why = _DAILY_QUOTA   # the operator's version of it
_breaker_status = 429


def breaker_open() -> bool:
    """True while a per-day quota or depleted-credit error from the last 30
    minutes stands."""
    return _now() < _breaker_until


def _minutes_left(until: float) -> int:
    """Whole minutes until ``until`` on the ``_now`` clock, rounded up (at least 1)."""
    return max(1, math.ceil((until - _now()) / 60))


def breaker_reason() -> str:
    """Why the breaker is open and when Gemini will be tried again, worded
    for the operator (a skipped run shows it as its fallback note); "" while
    it is closed."""
    with _breaker_lock:
        until, why, status = _breaker_until, _breaker_why, _breaker_status
    if _now() >= until:
        return ""
    reason = f"{why}; trying Gemini again in {_minutes_left(until)} min"
    if status == 402:
        # Credit takes a minute to add, unlike waiting out a day's quota, and
        # a run the operator starts closes this breaker first (jobs).
        reason += ", or at once when you press Suggest"
    return reason


def reset_breaker(*, status: int | None = None) -> bool:
    """Close the breaker now; with ``status``, only if that HTTP status
    opened it. True when it was open.

    For a run the operator asks for after fixing the cause, such as adding
    prepaid credit (402). If the cause is still there, that run's first call
    opens the breaker again."""
    global _breaker_until
    with _breaker_lock:
        if _now() >= _breaker_until or (status is not None and status != _breaker_status):
            return False
        _breaker_until = 0.0
        opened_by = _breaker_status
    log.info("Gemini breaker closed early (it opened on HTTP %d)", opened_by)
    return True


def _open_breaker(why: str, reason: str, status: int) -> None:
    global _breaker_until, _breaker_reason, _breaker_why, _breaker_status
    with _breaker_lock:
        _breaker_until = _now() + _BREAKER_SECONDS
        _breaker_reason = reason
        _breaker_why = why
        _breaker_status = status
    log.warning("%s; skipping Gemini for %d min: %s", why, _BREAKER_SECONDS // 60, reason)


def _check_breaker(label: str) -> None:
    if breaker_open():
        raise GeminiQuota(f"{_breaker_why}; not calling Gemini for another "
                          f"{_minutes_left(_breaker_until)} min ({label}: {_breaker_reason})",
                          status=_breaker_status)


# --- HTTP with retries -----------------------------------------------------------------

def _error_text(resp: requests.Response) -> str:
    try:
        return resp.text or ""
    except Exception:  # pragma: no cover - undecodable body
        return ""


def _error_message(resp: requests.Response) -> str:
    """Google's ``error.message`` when the body has one, else the raw text."""
    try:
        body = resp.json()
        message = body.get("error", {}).get("message") if isinstance(body, dict) else None
        if isinstance(message, str) and message:
            return message[:300]
    except Exception:
        pass
    return _error_text(resp)[:300]


def _is_daily_quota(resp: requests.Response) -> bool:
    text = _error_text(resp).lower()
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)


def _backoff(retry: int) -> float:
    return min(_MAX_BACKOFF, 2 ** retry + random.random())


def _retry_delay(resp: requests.Response, retry: int) -> float:
    """Retry-After (seconds or an HTTP date), else google.rpc.RetryInfo's
    ``retryDelay`` in the error body, else exponential backoff; capped."""
    header = (resp.headers.get("Retry-After") or "").strip()
    if header:
        seconds: float | None
        try:
            seconds = float(header)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(header)
                seconds = (when - dt.datetime.now(dt.timezone.utc)).total_seconds()
            except (TypeError, ValueError):
                seconds = None
        if seconds is not None:
            return min(_MAX_BACKOFF, max(0.0, seconds))
    match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', _error_text(resp))
    if match:
        return min(_MAX_BACKOFF, float(match.group(1)))
    return _backoff(retry)


def _is_read_timeout(exc: BaseException) -> bool:
    """A client read timeout, including one that struck while the body was
    being read: requests raises that as a ConnectionError around urllib3's
    ReadTimeoutError."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return False
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    return any(isinstance(arg, urllib3.exceptions.ReadTimeoutError) for arg in exc.args)


def _request(method: str, url: str, *, label: str, timeout: tuple[float, float],
             **kwargs: Any) -> requests.Response:
    """``_send`` under this module's retry policy.

    Retries HTTP 408/429/5xx and connection failures up to ``_MAX_RETRIES``
    times, but a costly failure only once between them: a read timeout, a
    504 (the server's deadline passed, and the same request won't fit it the
    next time either) or any failure that kept us waiting ``_SLOW_FAILURE``
    seconds or more. No retry starts once the call has run for twice its read
    timeout (``_MIN_RETRY_WINDOW`` at least): a slow, failing server must not
    hold the engine's one worker for an hour. Returns the last response
    (whatever its status) for the caller to judge. Raises ``GeminiQuota`` (and
    opens the breaker) on a per-day quota 429 or a 402 (prepaid credits
    depleted), ``GeminiError`` on transport failure."""
    started = _now()
    window = max(_MIN_RETRY_WINDOW, 2.0 * float(timeout[1]))
    retries = 0
    costly = 0
    while True:
        _check_breaker(label)
        sent = _now()
        resp: requests.Response | None = None
        error: Exception | None = None
        try:
            resp = _send(method, url, timeout=timeout, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as exc:
            error = exc
        except requests.exceptions.RequestException as exc:
            raise GeminiError(f"{label}: request to Gemini failed: {exc}") from exc
        waited = _now() - sent
        if resp is not None:
            if resp.status_code == 402:
                # Prepaid credits ran out. Not transient: every call fails the
                # same way until someone adds credit, so fail fast for all.
                message = _error_message(resp)
                _open_breaker(_CREDITS_DEPLETED, message, 402)
                raise GeminiQuota(f"{_CREDITS_DEPLETED} ({label}: {message})", status=402)
            if resp.status_code == 429 and _is_daily_quota(resp):
                message = _error_message(resp)
                _open_breaker(_DAILY_QUOTA, message, 429)
                raise GeminiQuota(f"{_DAILY_QUOTA} ({label}: {message})", status=429)
            if resp.status_code not in _RETRY_STATUS:
                return resp
            delay, why, problem = _retry_delay(resp, retries), f"HTTP {resp.status_code}", ""
            is_costly = resp.status_code == 504
        else:
            delay = _backoff(retries)
            is_costly = _is_read_timeout(error)
            if isinstance(error, requests.exceptions.ConnectTimeout):
                why, problem = "connect timeout", f"could not connect to Gemini: {error}"
            elif is_costly:
                why, problem = "read timeout", f"Gemini sent no answer within {timeout[1]:.0f}s"
            else:
                why, problem = f"connection error: {error}", f"could not reach Gemini: {error}"
        if waited >= _SLOW_FAILURE:
            is_costly = True
            why += f" after {waited:.0f}s"
        if is_costly:
            costly += 1
        elapsed = _now() - started
        cut_short = costly > 1 or elapsed + delay > window
        if cut_short or retries >= _MAX_RETRIES:
            attempts = f"{retries + 1} attempt{'s' if retries else ''} in {elapsed:.0f}s"
            if cut_short:
                log.warning("%s: %s; giving up after %s", label, why, attempts)
            if resp is not None:
                return resp
            raise GeminiError(f"{label}: {problem} ({attempts})") from error
        retries += 1
        log.warning("%s: %s; retry %d of %d in %.1fs", label, why, retries, _MAX_RETRIES, delay)
        _sleep(delay)


def _json_body(resp: requests.Response, label: str) -> dict:
    try:
        body = resp.json()
    except ValueError as exc:
        raise GeminiError(f"{label}: Gemini answered with non-JSON: {_error_text(resp)[:200]!r}",
                          status=resp.status_code) from exc
    if not isinstance(body, dict):
        raise GeminiError(f"{label}: unexpected Gemini response: {str(body)[:200]!r}",
                          status=resp.status_code)
    return body


# --- Files API ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_upload_locks: dict[str, threading.Lock] = {}
_confirmed: dict[str, float] = {}


def _cache_path() -> Path:
    return engine_dir() / "gemini_files.json"


def _cache_load() -> dict:
    data = util.read_json(_cache_path(), {})
    return data if isinstance(data, dict) else {}


def _age_seconds(entry: dict) -> float | None:
    try:
        uploaded = dt.datetime.fromisoformat(str(entry.get("uploaded_at")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if uploaded.tzinfo is None:
        uploaded = uploaded.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - uploaded).total_seconds()


def _expired(entry: Any) -> bool:
    """Past Google's 48 h retention, or not an entry we can date at all."""
    if not isinstance(entry, dict):
        return True
    age = _age_seconds(entry)
    return age is None or age > 48 * 3600


def _cache_update(mutate: Callable[[dict], None]) -> None:
    """Read-modify-write the upload cache under a thread lock and a
    cross-process lock (the bench CLI and the dashboard share it); entries past
    Google's 48 h retention are dropped. Never raises: the cache only saves
    re-uploads."""
    path = _cache_path()
    try:
        with _cache_lock, spend.file_lock(path.with_name(path.name + ".lock")):
            data = _cache_load()
            mutate(data)
            for key in [k for k, v in data.items() if _expired(v)]:
                data.pop(key, None)
            util.atomic_write_json(path, data)
    except Exception as exc:
        log.warning("Could not update %s: %s", path.name, exc)


def _forget(file_name: str) -> None:
    _confirmed.pop(file_name, None)

    def drop(data: dict) -> None:
        for key in [k for k, v in data.items() if isinstance(v, dict) and v.get("name") == file_name]:
            data.pop(key, None)

    _cache_update(drop)


def _upload_lock(cache_key: str) -> threading.Lock:
    with _cache_lock:
        return _upload_locks.setdefault(cache_key, threading.Lock())


def _file_name(value: Any) -> str:
    """``files/<id>`` from a name, a bare id or a file URI; "" if none."""
    text = str(value or "").strip()
    match = re.search(r"(files/[a-z0-9-]+)", text)
    if match:
        return match.group(1)
    return f"files/{text}" if re.fullmatch(r"[a-z0-9-]+", text) else ""


def _forget_named_uploads(parts: list[dict], error: str, label: str) -> None:
    """Forget each of a request's uploads that its 4xx names ("You do not
    have permission to access the File <id> or it may not exist"): another
    project's file, or one already gone. The next call then uploads the video
    again instead of reusing the dead one until the hourly recheck."""
    for part in parts:
        data = part.get("fileData") if isinstance(part, dict) else None
        name = _file_name(data.get("fileUri")) if isinstance(data, dict) else ""
        file_id = name[len("files/"):]
        if file_id and re.search(rf"(?<![a-z0-9-]){re.escape(file_id)}(?![a-z0-9-])", error):
            log.info("%s: Gemini can't use %s; forgetting it so it's uploaded again", label, name)
            _forget(name)


def _get_file(file_name: str, key: str, label: str) -> requests.Response:
    return _request("GET", f"{_API}/{file_name}", label=label, timeout=(_CONNECT_TIMEOUT, 30.0),
                    headers={"x-goog-api-key": key})


def _still_active(file_name: str, key: str, label: str) -> bool:
    if _now() - _confirmed.get(file_name, float("-inf")) < _CONFIRM_MEMO:
        return True
    try:
        resp = _get_file(file_name, key, label)
    except GeminiQuota:
        raise
    except GeminiError as exc:
        log.info("%s: could not confirm %s (%s); uploading again", label, file_name, exc)
        return False
    if resp.status_code != 200:
        return False
    try:
        state = str(resp.json().get("state") or "").upper()
    except Exception:
        return False
    if state == "ACTIVE":
        _confirmed[file_name] = _now()
        return True
    return False


def _key_fingerprint(key: str) -> str:
    """A short one-way tag for an API key, kept with each cached upload in
    place of the key itself."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _cached_upload(cache_key: str, key: str, label: str) -> dict | None:
    entry = _cache_load().get(cache_key)
    if not isinstance(entry, dict) or not entry.get("name") or not entry.get("uri"):
        return None
    if entry.get("key_fingerprint") != _key_fingerprint(key):
        # Uploaded under another API key (a file belongs to its key's project,
        # and another project's key can't read it), or by a version that
        # didn't record the key: upload again rather than risk it.
        return None
    age = _age_seconds(entry)
    if age is None or age >= _FILE_TTL:
        return None
    if age > _FILE_RECHECK and not _still_active(entry["name"], key, label):
        _forget(entry["name"])
        return None
    return {"name": entry["name"], "uri": entry["uri"],
            "mime_type": entry.get("mime_type") or "video/mp4"}


def _upload_timeout(size: int) -> float:
    return max(120.0, 60.0 + size / 1_000_000 * 2.0)


def _upload_bytes(path: Path, size: int, mime: str, display_name: str, key: str,
                  label: str) -> dict:
    """Resumable upload: start a session, then send every byte with
    ``upload, finalize``. A failed byte transfer starts a fresh session (a
    half-sent session can't simply be restarted at offset 0), but one that
    Google never answered only once: each try may wait
    ``_upload_timeout(size)``, many minutes for a whole-source proxy."""
    attempt = 0
    read_timeouts = 0
    while True:
        attempt += 1
        start = _request(
            "POST", _UPLOAD_URL, label=label, timeout=(_CONNECT_TIMEOUT, 60.0),
            headers={
                "x-goog-api-key": key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": display_name[:120]}},
        )
        if start.status_code != 200:
            raise GeminiError(f"{label}: upload could not start: HTTP {start.status_code}: "
                              f"{_error_message(start)}", status=start.status_code)
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise GeminiError(f"{label}: upload start returned no upload URL")
        try:
            with open(path, "rb") as fh:
                resp = _send("POST", upload_url, timeout=(_CONNECT_TIMEOUT, _upload_timeout(size)),
                             headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                      "X-Goog-Upload-Command": "upload, finalize"},
                             data=fh)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            if _is_read_timeout(exc):
                read_timeouts += 1
            if attempt >= _UPLOAD_ATTEMPTS or read_timeouts > 1:
                raise GeminiError(f"{label}: upload failed: {exc}") from exc
            log.warning("%s: upload interrupted (%s); starting again", label, exc)
            _sleep(_backoff(attempt - 1))
            continue
        except OSError as exc:
            raise GeminiError(f"{label}: could not read {path}: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise GeminiError(f"{label}: upload failed: {exc}") from exc
        if resp.status_code == 402:   # as in _request: credits ran out mid-upload
            message = _error_message(resp)
            _open_breaker(_CREDITS_DEPLETED, message, 402)
            raise GeminiQuota(f"{_CREDITS_DEPLETED} ({label}: {message})", status=402)
        if resp.status_code in _RETRY_STATUS and attempt < _UPLOAD_ATTEMPTS:
            log.warning("%s: upload got HTTP %d; starting again", label, resp.status_code)
            _sleep(_retry_delay(resp, attempt - 1))
            continue
        if resp.status_code != 200:
            raise GeminiError(f"{label}: upload failed: HTTP {resp.status_code}: "
                              f"{_error_message(resp)}", status=resp.status_code)
        body = _json_body(resp, label)
        info = body.get("file") if isinstance(body.get("file"), dict) else body
        if not info.get("name") or not info.get("uri"):
            raise GeminiError(f"{label}: upload response had no file name/uri: {str(body)[:200]}")
        return info


def _wait_active(info: dict, *, key: str, label: str, wait_timeout: float) -> dict:
    """Poll the file every few seconds until Google has processed it."""
    name = info["name"]
    deadline = _now() + max(0.0, float(wait_timeout))
    while True:
        state = str(info.get("state") or "").upper()
        if state == "ACTIVE":
            return info
        if state == "FAILED":
            error = info.get("error") if isinstance(info.get("error"), dict) else {}
            raise GeminiError(f"{label}: Gemini could not process {name}: "
                              f"{error.get('message') or 'state FAILED'}")
        if _now() >= deadline:
            raise GeminiError(f"{label}: {name} still {state or 'not ready'} after "
                              f"{float(wait_timeout):.0f}s")
        _sleep(_POLL_SECONDS)
        resp = _get_file(name, key, label)
        if resp.status_code != 200:
            raise GeminiError(f"{label}: checking {name} failed: HTTP {resp.status_code}: "
                              f"{_error_message(resp)}", status=resp.status_code)
        body = _json_body(resp, label)
        info = {**info, **(body.get("file") if isinstance(body.get("file"), dict) else body)}


def upload_file(path, *, mime: str = "video/mp4", display_name: str = "",
                wait_timeout: float = 600.0) -> dict:
    """Upload ``path`` through the Files API (or reuse an upload of the same
    bytes from the last 46 h) and wait until it can be used.

    Returns ``{"name", "uri", "mime_type"}``. Raises ``GeminiError`` when the
    file can't be read, the upload fails, Google can't process it, or it isn't
    ACTIVE within ``wait_timeout`` seconds; ``GeminiQuota`` while the breaker
    is open."""
    source = Path(path).expanduser()
    label = f"upload {source.name}"
    _check_breaker(label)
    key = api_key()
    try:
        source = source.resolve()
        st = source.stat()
    except OSError as exc:
        raise GeminiError(f"{label}: cannot read {source}: {exc}") from exc
    cache_key = f"{source}|{st.st_size}|{st.st_mtime_ns}"
    # One upload per file at a time: parallel critics often want the same render.
    with _upload_lock(cache_key):
        hit = _cached_upload(cache_key, key, label)
        if hit:
            return hit
        started = _now()
        info = _upload_bytes(source, st.st_size, mime, display_name or source.name, key, label)
        info = _wait_active(info, key=key, label=label, wait_timeout=wait_timeout)
        out = {"name": info["name"], "uri": info["uri"],
               "mime_type": info.get("mimeType") or info.get("mime_type") or mime}
        _confirmed[out["name"]] = _now()
        entry = {**out, "uploaded_at": util.utcnow_iso(), "size": st.st_size,
                 "key_fingerprint": _key_fingerprint(key)}
        _cache_update(lambda data: data.__setitem__(cache_key, entry))
        log.info("%s: uploaded %.1f MB as %s in %.1fs", label, st.st_size / 1e6, out["name"],
                 _now() - started)
        return out


def delete_file(name: str) -> None:
    """Delete an upload (``files/<id>``, a bare id, or its URI) and forget it
    in the cache. Best effort: never raises — a file left behind expires on
    its own after 48 h."""
    file_name = _file_name(name)
    if not file_name:
        log.warning("delete_file: not a Gemini file name: %r", name)
        return
    _forget(file_name)
    if not is_configured():
        return
    try:
        resp = _send("DELETE", f"{_API}/{file_name}", headers={"x-goog-api-key": api_key()},
                     timeout=(_CONNECT_TIMEOUT, 30.0))
        if resp.status_code not in (200, 204, 404):
            log.warning("Could not delete Gemini file %s: HTTP %d: %s", file_name,
                        resp.status_code, _error_message(resp))
    except Exception as exc:
        log.warning("Could not delete Gemini file %s: %s", file_name, exc)


# --- request parts -----------------------------------------------------------------------

def text_part(text: str) -> dict:
    return {"text": str(text)}


def _inline_budget() -> int:
    """``clip_engine.gemini.inline_request_mb`` (MiB) as bytes."""
    try:
        from ..config import load_settings

        mb = float(opt(load_settings(), "clip_engine.gemini.inline_request_mb", _DEFAULT_INLINE_MB))
    except (TypeError, ValueError):
        mb = float(_DEFAULT_INLINE_MB)
    return max(0, int(mb * 1024 * 1024))


def _offset(seconds: Any) -> str:
    """Seconds as a videoMetadata offset string: 3 -> "3.0s"."""
    return f"{round(max(0.0, float(seconds)), 3)}s"


def _seconds(item: dict, key: str) -> float | None:
    value = item.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise GeminiError(f"invalid {key} {value!r} for {item.get('path')}") from None


def _video_metadata(item: dict) -> dict:
    """``videoMetadata`` with only the keys the item sets."""
    meta: dict[str, Any] = {}
    fps = _seconds(item, "fps")
    if fps is not None:
        if fps <= 0:
            raise GeminiError(f"invalid fps {fps} for {item.get('path')}")
        meta["fps"] = min(fps, 24.0)  # Gemini's documented maximum
    start, end = _seconds(item, "start"), _seconds(item, "end")
    if end is not None and end <= max(0.0, start or 0.0):
        raise GeminiError(f"empty video span {start}-{end} for {item.get('path')}")
    if start is not None:
        meta["startOffset"] = _offset(start)
    if end is not None:
        meta["endOffset"] = _offset(end)
    return meta


def video_parts(items: list[dict], *, inline_budget_bytes: int | None = None) -> list[dict]:
    """Request parts for videos, in the order given.

    Each item: ``{"path", "fps"?, "start"?, "end"?, "upload"?, "mime"?}``. A
    video is sent inline while the request's running total of raw inline bytes
    stays within ``inline_budget_bytes`` (default
    ``clip_engine.gemini.inline_request_mb`` MiB) and the item doesn't ask for
    ``upload``; every other video goes through ``upload_file`` (cached).
    Raises ``GeminiError`` when a file can't be read or uploaded."""
    budget = _inline_budget() if inline_budget_bytes is None else max(0, int(inline_budget_bytes))
    inline_total = 0
    parts: list[dict] = []
    for item in items:
        path = Path(item["path"])
        mime = str(item.get("mime") or "video/mp4")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise GeminiError(f"cannot read video {path}: {exc}") from exc
        meta = _video_metadata(item)
        if not item.get("upload") and inline_total + size <= budget:
            try:
                data = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError as exc:
                raise GeminiError(f"cannot read video {path}: {exc}") from exc
            part: dict[str, Any] = {"inlineData": {"mimeType": mime, "data": data}}
            inline_total += size
        else:
            uploaded = upload_file(path, mime=mime)
            part = {"fileData": {"mimeType": uploaded["mime_type"], "fileUri": uploaded["uri"]}}
        if meta:
            part["videoMetadata"] = meta
        parts.append(part)
    return parts


# --- generateContent ---------------------------------------------------------------------

# Keys Gemini's older OpenAPI-style ``responseSchema`` understands.
_OPENAPI_KEYS = frozenset({
    "type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems",
    "properties", "required", "minProperties", "maxProperties", "minLength", "maxLength",
    "pattern", "example", "anyOf", "propertyOrdering", "default", "items", "minimum", "maximum",
})


def _openapi_schema(node: Any) -> Any:
    """A JSON Schema rewritten for ``responseSchema`` (the fallback when a
    model rejects ``responseJsonSchema``): type lists become one upper-case
    type plus ``nullable``, ``oneOf`` becomes ``anyOf``, ``const`` an enum, and
    keys it doesn't know (``additionalProperties``, ``$schema``, …) are dropped."""
    if isinstance(node, list):
        return [_openapi_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    nullable = False
    for key, value in node.items():
        if key == "oneOf":
            key = "anyOf"
        if key == "const":
            out["enum"] = [value]
            continue
        if key not in _OPENAPI_KEYS:
            continue
        if key == "type":
            names = [str(t).lower() for t in (value if isinstance(value, list) else [value])]
            rest = [t for t in names if t != "null"]
            nullable = nullable or ("null" in names and bool(rest))
            out["type"] = (rest[0] if rest else "null").upper()
        elif key == "properties" and isinstance(value, dict):
            out[key] = {name: _openapi_schema(sub) for name, sub in value.items()}
        elif key in ("items", "anyOf"):
            out[key] = _openapi_schema(value)
        else:
            out[key] = value
    if nullable or out.get("nullable"):
        out["nullable"] = True
        if isinstance(out.get("enum"), list):
            out["enum"] = [v for v in out["enum"] if v is not None]
    return out


@dataclass
class _Shape:
    """Which request features a 400 has made us drop (each at most once)."""

    system_in_parts: bool = False
    drop_thinking: bool = False
    drop_media_resolution: bool = False
    legacy_schema: bool = False

    def downgrade(self, error: str, *, has_system: bool, has_thinking: bool,
                  has_media: bool) -> list[str]:
        """Apply every not-yet-applied downgrade the 400 ``error`` names;
        returns what changed ([] = nothing left to try)."""
        text = error.lower()
        changed: list[str] = []
        if has_system and not self.system_in_parts and (
                "systeminstruction" in text or "system_instruction" in text):
            self.system_in_parts = True
            changed.append("systemInstruction")
        if has_thinking and not self.drop_thinking and "thinking" in text:
            self.drop_thinking = True
            changed.append("thinkingConfig")
        if has_media and not self.drop_media_resolution and (
                "mediaresolution" in text or "media_resolution" in text):
            self.drop_media_resolution = True
            changed.append("mediaResolution")
        if not self.legacy_schema and (
                "responsejsonschema" in text or "response_json_schema" in text):
            self.legacy_schema = True
            changed.append("responseJsonSchema")
        return changed


def _request_body(*, system: str, parts: list[dict], schema: dict, thinking: str | None,
                  media_resolution: str | None, max_output_tokens: int, shape: _Shape) -> dict:
    from ..config import with_american_english

    contents = list(parts)
    body: dict[str, Any] = {}
    system = with_american_english(system)
    if system:
        if shape.system_in_parts:
            contents = [{"text": system}] + contents
        else:
            body["systemInstruction"] = {"parts": [{"text": system}]}
    body["contents"] = [{"role": "user", "parts": contents}]
    config: dict[str, Any] = {"responseMimeType": "application/json"}
    if shape.legacy_schema:
        config["responseSchema"] = _openapi_schema(schema)
    else:
        config["responseJsonSchema"] = schema
    config["maxOutputTokens"] = int(max_output_tokens)
    if media_resolution and not shape.drop_media_resolution:
        config["mediaResolution"] = media_resolution
    if thinking and not shape.drop_thinking:
        config["thinkingConfig"] = {"thinkingLevel": thinking}
    body["generationConfig"] = config
    return body


def _model_id(model: str) -> str:
    value = str(model or "").strip()
    if value.startswith("models/"):
        value = value[len("models/"):]
    if not value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise GeminiError(f"not a Gemini model id: {model!r}")
    return value


def _tokens(meta: Any) -> int:
    try:
        return max(0, int(meta or 0))
    except (TypeError, ValueError):
        return 0


def _record_usage(meta: Any, *, model: str, bucket: str | None, label: str) -> dict:
    """Price and record one response's ``usageMetadata``; returns the usage dict.

    input = promptTokenCount + toolUsePromptTokenCount; output =
    candidatesTokenCount + thoughtsTokenCount (thinking bills as output). If the
    named counts are missing, totalTokenCount is priced as output; anything the
    total has beyond the named counts is too, so the guard errs high."""
    meta = meta if isinstance(meta, dict) else {}
    tokens_in = _tokens(meta.get("promptTokenCount")) + _tokens(meta.get("toolUsePromptTokenCount"))
    tokens_out = _tokens(meta.get("candidatesTokenCount")) + _tokens(meta.get("thoughtsTokenCount"))
    total = _tokens(meta.get("totalTokenCount"))
    named = any(k in meta for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"))
    if not named:
        log.warning("%s: usageMetadata has no prompt/candidate counts (keys %s); "
                    "pricing totalTokenCount=%d as output", label, sorted(meta), total)
        tokens_out = total
    elif total > tokens_in + tokens_out:
        tokens_out += total - (tokens_in + tokens_out)
    cost = spend.record(model, tokens_in, tokens_out,
                        bucket=bucket or spend.current_bucket() or "engine")
    return {"model": model, "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": round(cost, 6)}


def _answer_text(candidate: dict) -> str:
    """The answer's text: every text part except thought summaries."""
    content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    return "".join(p["text"] for p in parts
                   if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought"))


def parse_json_object(text: str) -> dict | None:
    """The JSON object in a model's answer text, or None.

    Tries the text as is; else the outermost ``{...}`` (which also unwraps a
    code fence or surrounding prose); else the first object that decodes from
    the first ``{`` (prose after it that contains braces). Valid JSON of the
    wrong shape (an array, a string) is a wrong answer, not an object to dig
    out of. Shared with ``claude.json_call``."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    first, last = text.find("{"), text.rfind("}")
    if first < 0:
        return None
    try:
        value = json.loads(text[first:last + 1])
    except ValueError:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[first:])
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _parse_response(payload: dict, *, model: str, bucket: str | None,
                    label: str) -> tuple[dict, dict]:
    """Usage first (recorded even when the answer is unusable), then the
    block / truncation checks, then the JSON."""
    usage = _record_usage(payload.get("usageMetadata"), model=model, bucket=bucket, label=label)
    feedback = payload.get("promptFeedback") if isinstance(payload.get("promptFeedback"), dict) else {}
    if feedback.get("blockReason"):
        raise GeminiBlocked(f"{label}: Gemini blocked the prompt ({feedback['blockReason']})",
                            usage=usage)
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    if not candidates or not isinstance(candidates[0], dict):
        raise GeminiError(f"{label}: Gemini returned no answer", usage=usage)
    candidate = candidates[0]
    finish = str(candidate.get("finishReason") or "").upper()
    if finish == "MAX_TOKENS":
        raise GeminiTruncated(f"{label}: the answer hit maxOutputTokens "
                              f"({usage['output_tokens']} output tokens incl. thinking)", usage=usage)
    if finish in _BLOCKED_FINISHES:
        raise GeminiBlocked(f"{label}: Gemini stopped the answer ({finish})", usage=usage)
    text = _answer_text(candidate)
    data = parse_json_object(text)
    if data is None:
        why = f" (finishReason {finish})" if finish and finish != "STOP" else ""
        raise GeminiError(f"{label}: Gemini did not return a JSON object{why}: {text[:200]!r}",
                          usage=usage)
    return data, usage


def generate_json(model: str, *, system: str, parts: list[dict], schema: dict,
                  thinking: str | None = "LOW",
                  media_resolution: str | None = "MEDIA_RESOLUTION_LOW",
                  max_output_tokens: int = 32768, timeout_s: float = 900.0,
                  label: str = "", bucket: str | None = "engine") -> tuple[dict, dict]:
    """One schema-constrained generateContent call.

    ``parts`` go in one user turn, in order (videos first, text last, as
    Google recommends — callers build them with ``video_parts``/``text_part``).
    Returns ``(data, usage)`` with usage ``{"model", "input_tokens",
    "output_tokens", "cost_usd"}``. Spend is recorded under ``bucket`` (else
    the process default, else "engine") before any check, and every
    ``GeminiError`` raised after a response arrived carries ``.usage``."""
    model = _model_id(model)
    label = label or model
    _check_breaker(label)
    key = api_key()
    url = f"{_API}/models/{model}:generateContent"
    shape = _Shape()
    started = _now()
    while True:
        body = _request_body(system=system, parts=parts, schema=schema, thinking=thinking,
                             media_resolution=media_resolution,
                             max_output_tokens=max_output_tokens, shape=shape)
        resp = _request("POST", url, label=label, timeout=(_CONNECT_TIMEOUT, float(timeout_s)),
                        headers={"x-goog-api-key": key}, json=body)
        if resp.status_code != 400:
            break
        changed = shape.downgrade(_error_text(resp), has_system=bool(system),
                                  has_thinking=bool(thinking),
                                  has_media=bool(media_resolution))
        if not changed:
            break
        log.warning("%s: Gemini rejected %s (%s); retrying without it", label,
                    ", ".join(changed), _error_message(resp))
    if resp.status_code != 200:
        if resp.status_code in (400, 403, 404):
            _forget_named_uploads(parts, _error_text(resp), label)
        raise GeminiError(f"{label}: Gemini HTTP {resp.status_code}: {_error_message(resp)}",
                          status=resp.status_code)
    data, usage = _parse_response(_json_body(resp, label), model=model, bucket=bucket, label=label)
    log.info("%s: %s in=%d out=%d $%.4f %.1fs", label, model, usage["input_tokens"],
             usage["output_tokens"], usage["cost_usd"], _now() - started)
    return data, usage
