"""Machine state from Fly's Machines API (read-only).

Hosted, the token is the ``FLY_API_TOKEN`` secret (a read-only org token).
Locally it falls back to ``fly auth token`` from the signed-in CLI.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import subprocess
import time

import httpx

log = logging.getLogger("admin.fly")
API = "https://api.machines.dev/v1"
_token_cache: str | None = None
_cache: dict[str, tuple[float, list[dict]]] = {}
TTL = 30.0


def _token() -> str:
    global _token_cache
    if _token_cache is None:
        tok = os.environ.get("FLY_API_TOKEN", "").strip()
        if not tok:
            try:
                tok = subprocess.run(["fly", "auth", "token"], capture_output=True, text=True,
                                     timeout=10).stdout.strip().splitlines()[-1]
            except Exception:
                tok = ""
        _token_cache = tok
    return _token_cache


def machines(app: str) -> list[dict]:
    """Each machine of ``app``: id, state, region, image version, last update.
    An error comes back as one entry with state "unknown" and the reason."""
    hit = _cache.get(app)
    if hit and time.monotonic() - hit[0] < TTL:
        return hit[1]
    tok = _token()
    if not tok:
        return [{"app": app, "state": "unknown", "error": "no Fly token"}]
    try:
        r = httpx.get(f"{API}/apps/{app}/machines", timeout=10,
                      headers={"Authorization": f"Bearer {tok}"})
        r.raise_for_status()
        out = []
        for m in r.json():
            updated = m.get("updated_at")
            out.append({"app": app, "id": m.get("id"), "state": m.get("state"),
                        "region": m.get("region"), "name": m.get("name"),
                        "version": ((m.get("image_ref") or {}).get("tag") or "")[-26:],
                        "updated_at": dt.datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        if updated else None})
    except Exception as exc:
        log.warning("Fly machines for %s: %s", app, exc)
        out = [{"app": app, "state": "unknown", "error": str(exc)[:160]}]
    _cache[app] = (time.monotonic(), out)
    return out
