"""Supabase Storage: host the operator's trimmed clip at a URL Threads can fetch.

The Threads API only accepts video by URL, so the trimmed clip is uploaded to a
private bucket and served via a time-limited signed URL that outlives Meta's
fetch + processing window. Objects are retained afterward as the canonical
record of exactly what was published.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import httpx

from supabase import create_client

from .config import env, load_settings
from .outro import with_outro

log = logging.getLogger("storage")


def _client():
    url, key = env("SUPABASE_URL"), env("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not set (see .env.example)")
    return create_client(url, key)


def _ensure_bucket(client, bucket: str) -> None:
    """Create the private clip bucket on first use so publishing works without
    any manual Supabase dashboard setup. Per-file size is governed by the
    project's global upload limit, so we don't set a bucket-level cap here."""
    try:
        client.storage.get_bucket(bucket)
        return  # already exists
    except Exception:
        pass  # not found (or transient) — try to create it below
    try:
        client.storage.create_bucket(bucket, options={"public": False})
        log.info("Created Supabase storage bucket %r", bucket)
    except Exception as exc:
        # A concurrent create (or an existing-but-unreadable bucket) is fine.
        if "exist" in str(exc).lower() or "duplicate" in str(exc).lower():
            return
        raise RuntimeError(
            f"Supabase bucket {bucket!r} is missing and could not be created "
            f"automatically: {exc}. Create it once in the Supabase dashboard "
            f"(Storage \u2192 New bucket, name {bucket!r}, keep it private)."
        ) from exc


def ensure_bucket(bucket: str) -> None:
    """Create a private bucket on first use (see ``_ensure_bucket``)."""
    _ensure_bucket(_client(), bucket)


# Supabase's resumable (TUS) endpoint requires exactly 6 MB chunks (the last
# one may be shorter).
_TUS_CHUNK = 6 * 1024 * 1024
_TUS_MAX_RETRIES = 6


def _auth_headers() -> dict[str, str]:
    key = env("SUPABASE_SERVICE_KEY")
    if not env("SUPABASE_URL") or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not set (see .env.example)")
    return {"Authorization": f"Bearer {key}", "apikey": key}


def _object_url(kind: str, bucket: str, object_key: str) -> str:
    return (f"{env('SUPABASE_URL').rstrip('/')}/storage/v1/{kind}/"
            f"{quote(bucket)}/{quote(object_key)}")


def upload_file_resumable(local_path: str | Path, bucket: str, object_key: str,
                          content_type: str = "video/mp4") -> None:
    """Upload a large file in resumable chunks, overwriting any existing object.

    The standard upload holds the whole file in memory and fails outright on a
    dropped connection; source videos run to gigabytes, so they go up over TUS,
    resuming from the server's offset after a transient failure. An object over
    the project's global file size limit fails with a clear error."""
    path = Path(local_path)
    size = path.stat().st_size
    auth = {**_auth_headers(), "Tus-Resumable": "1.0.0"}

    def b64(value: str) -> str:
        return base64.b64encode(value.encode()).decode()

    endpoint = f"{env('SUPABASE_URL').rstrip('/')}/storage/v1/upload/resumable"
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as http:
        created = http.post(endpoint, headers={
            **auth,
            "Upload-Length": str(size),
            "x-upsert": "true",
            "Upload-Metadata": ",".join([
                f"bucketName {b64(bucket)}", f"objectName {b64(object_key)}",
                f"contentType {b64(content_type)}", f"cacheControl {b64('3600')}",
            ]),
        })
        if created.status_code == 413:
            raise RuntimeError(
                f"{path.name} ({size / 1e9:.2f} GB) is over the Supabase project's "
                f"file size limit (Storage \u2192 Settings).")
        if created.status_code != 201 or "location" not in created.headers:
            raise RuntimeError(f"Resumable upload refused ({created.status_code}): "
                               f"{created.text[:300]}")
        location = urljoin(endpoint, created.headers["location"])

        offset, failures = 0, 0
        with open(path, "rb") as f:
            while offset < size:
                f.seek(offset)
                chunk = f.read(_TUS_CHUNK)
                try:
                    r = http.patch(location, content=chunk, headers={
                        **auth, "Upload-Offset": str(offset),
                        "Content-Type": "application/offset+octet-stream",
                    })
                    if r.status_code == 204:
                        offset = int(r.headers.get("upload-offset", offset + len(chunk)))
                        failures = 0
                        continue
                    if r.status_code == 413:
                        raise RuntimeError(f"{path.name} is over the Supabase file size limit")
                    if r.status_code < 500 and r.status_code != 409:
                        raise RuntimeError(f"Upload chunk refused ({r.status_code}): "
                                           f"{r.text[:300]}")
                    problem = f"HTTP {r.status_code}"
                except httpx.TransportError as exc:
                    problem = str(exc) or type(exc).__name__
                failures += 1
                if failures > _TUS_MAX_RETRIES:
                    raise RuntimeError(f"Upload of {path.name} kept failing ({problem})")
                log.info("Upload of %s hiccuped at %.0f%% (%s); resuming",
                         path.name, 100 * offset / size, problem)
                time.sleep(min(60, 2 ** failures))
                head = http.head(location, headers=auth)
                if head.status_code != 200 or "upload-offset" not in head.headers:
                    raise RuntimeError(f"Upload of {path.name} can't be resumed "
                                       f"({head.status_code})")
                offset = int(head.headers["upload-offset"])


def download_to_file(bucket: str, object_key: str, dest: str | Path,
                     expected_size: int | None = None) -> Path:
    """Stream a stored object to ``dest`` without holding it in memory. Written
    to a temp name and renamed into place, so a reader never sees half a file."""
    dest = Path(dest)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as http:
            with http.stream("GET", _object_url("object/authenticated", bucket, object_key),
                             headers=_auth_headers()) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"Download of {object_key} failed ({r.status_code})")
                with open(tmp, "wb") as f:
                    for block in r.iter_bytes(1024 * 1024):
                        f.write(block)
        got = tmp.stat().st_size
        if expected_size is not None and got != expected_size:
            raise RuntimeError(f"Download of {object_key} was cut short "
                               f"({got} of {expected_size} bytes)")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def upload_trimmed_clip(local_path: str | Path, object_key: str) -> str:
    """Upload the clip and return a signed URL. Idempotent via upsert.

    What goes up is what publishes, so the workspace's post-roll outro (see
    app/outro.py) is appended here, on every path that ships a clip.

    Clips larger than ``storage.max_upload_mb`` are transparently recompressed to
    fit first: Supabase rejects oversized uploads with a 413 that otherwise fails
    the whole publish (and the post silently drops out of the queue)."""
    settings = load_settings()
    bucket = settings.get("storage.trimmed_clip_bucket", "trimmed-clips")
    ttl = settings.get("storage.signed_url_ttl_seconds", 3600)
    max_mb = settings.get("storage.max_upload_mb", 45)
    local_path = with_outro(local_path)

    upload_path = local_path
    compressed: Path | None = None
    max_bytes = int(max_mb * 1024 * 1024)
    if max_bytes > 0 and local_path.stat().st_size > max_bytes:
        from .clipper import compress_to_fit

        log.info(
            "Clip %s is %.1fMB, over the %sMB upload cap — recompressing to fit",
            local_path.name, local_path.stat().st_size / 1e6, max_mb,
        )
        compressed = compress_to_fit(local_path, max_bytes)
        upload_path = compressed

    client = _client()
    _ensure_bucket(client, bucket)
    storage = client.storage.from_(bucket)
    try:
        with open(upload_path, "rb") as f:
            storage.upload(object_key, f.read(), {"content-type": "video/mp4", "upsert": "true"})
    finally:
        if compressed is not None:
            compressed.unlink(missing_ok=True)
    signed = storage.create_signed_url(object_key, ttl)
    url = signed.get("signedURL") or signed.get("signedUrl") or ""
    if not url:
        raise RuntimeError(f"Could not create signed URL for {object_key}: {signed}")
    log.info("Uploaded clip %s (signed URL valid %ss)", object_key, ttl)
    return url


def signed_clip_url(object_key: str) -> str:
    """Fresh signed URL for a clip already uploaded to storage.

    Lets a headless runner publish a queued post without the operator's disk —
    the clip was uploaded at queue time."""
    settings = load_settings()
    bucket = settings.get("storage.trimmed_clip_bucket", "trimmed-clips")
    ttl = settings.get("storage.signed_url_ttl_seconds", 3600)
    storage = _client().storage.from_(bucket)
    signed = storage.create_signed_url(object_key, ttl)
    url = signed.get("signedURL") or signed.get("signedUrl") or ""
    if not url:
        raise RuntimeError(f"Could not create signed URL for {object_key}: {signed}")
    return url
