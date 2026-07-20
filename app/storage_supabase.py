"""Supabase Storage: host the operator's trimmed clip at a URL Threads can fetch.

The Threads API only accepts video by URL, so the trimmed clip is uploaded to a
private bucket and served via a time-limited signed URL that outlives Meta's
fetch + processing window. Objects are retained afterward as the canonical
record of exactly what was published.
"""
from __future__ import annotations

import logging
from pathlib import Path

from supabase import create_client

from .config import env, load_settings

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


def upload_trimmed_clip(local_path: str | Path, object_key: str) -> str:
    """Upload the clip and return a signed URL. Idempotent via upsert."""
    settings = load_settings()
    bucket = settings.get("storage.trimmed_clip_bucket", "trimmed-clips")
    ttl = settings.get("storage.signed_url_ttl_seconds", 3600)
    local_path = Path(local_path)

    client = _client()
    _ensure_bucket(client, bucket)
    storage = client.storage.from_(bucket)
    with open(local_path, "rb") as f:
        storage.upload(object_key, f.read(), {"content-type": "video/mp4", "upsert": "true"})
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
