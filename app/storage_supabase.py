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


def upload_trimmed_clip(local_path: str | Path, object_key: str) -> str:
    """Upload the clip and return a signed URL. Idempotent via upsert."""
    settings = load_settings()
    bucket = settings.get("storage.trimmed_clip_bucket", "trimmed-clips")
    ttl = settings.get("storage.signed_url_ttl_seconds", 3600)
    local_path = Path(local_path)

    client = _client()
    storage = client.storage.from_(bucket)
    with open(local_path, "rb") as f:
        storage.upload(object_key, f.read(), {"content-type": "video/mp4", "upsert": "true"})
    signed = storage.create_signed_url(object_key, ttl)
    url = signed.get("signedURL") or signed.get("signedUrl") or ""
    if not url:
        raise RuntimeError(f"Could not create signed URL for {object_key}: {signed}")
    log.info("Uploaded clip %s (signed URL valid %ss)", object_key, ttl)
    return url
