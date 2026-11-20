"""Where the live universal rules are kept: one private JSON object in the
platform's Supabase Storage (the project that holds the logins). The admin
site reads and writes it; every account's workers get it through the admin
site's /api/universal-rules (app/universal_rules.py)."""
from __future__ import annotations

import datetime as dt
import json
import os

import httpx

from .. import universal_rules

BUCKET = "platform"
OBJECT = "universal_rules.json"


def _base() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/") + "/storage/v1"


def _headers() -> dict:
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    return {"Authorization": f"Bearer {key}", "apikey": key}


def _ensure_bucket() -> None:
    r = httpx.post(f"{_base()}/bucket", headers=_headers(), timeout=15,
                   json={"id": BUCKET, "name": BUCKET, "public": False})
    if r.status_code not in (200, 201, 400, 409):   # 400/409: it already exists
        r.raise_for_status()


def read() -> dict:
    """The live rules, or the shipped defaults until the first save."""
    r = httpx.get(f"{_base()}/object/authenticated/{BUCKET}/{OBJECT}", headers=_headers(), timeout=15)
    if r.status_code in (400, 404):
        return universal_rules.defaults()
    r.raise_for_status()
    return universal_rules.clean(r.json())


def write(data: dict) -> dict:
    data = universal_rules.clean(data)
    data["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    _ensure_bucket()
    r = httpx.post(f"{_base()}/object/{BUCKET}/{OBJECT}", timeout=15,
                   headers={**_headers(), "Content-Type": "application/json", "x-upsert": "true"},
                   content=json.dumps(data).encode())
    r.raise_for_status()
    return data
