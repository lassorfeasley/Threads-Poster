"""Giphy search proxy for attaching GIFs to Threads replies.

Uses the free Beta API key (``GIPHY_API_KEY``). Keep the key server-side —
the picker UI hits our ``/giphy/*`` routes, not Giphy directly.
"""
from __future__ import annotations

import logging

import requests

from .config import env

log = logging.getLogger("giphy")

GIPHY_API = "https://api.giphy.com/v1/gifs"


class GiphyError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(env("GIPHY_API_KEY", "").strip())


def _key() -> str:
    key = env("GIPHY_API_KEY", "").strip()
    if not key:
        raise GiphyError("GIPHY_API_KEY is not set (see .env.example)")
    return key


def _normalize(items: list[dict]) -> list[dict]:
    out = []
    for g in items:
        images = g.get("images") or {}
        preview = (images.get("fixed_width_small") or images.get("fixed_width")
                   or images.get("downsized") or {})
        full = (images.get("fixed_width") or images.get("downsized")
                or images.get("original") or {})
        url = preview.get("url") or full.get("url")
        if not g.get("id") or not url:
            continue
        out.append({
            "id": str(g["id"]),
            "title": g.get("title") or "",
            "url": url,
            "preview_url": url,
            "width": int(preview.get("width") or full.get("width") or 0),
            "height": int(preview.get("height") or full.get("height") or 0),
        })
    return out


def search(query: str, *, limit: int = 24) -> list[dict]:
    """Search Giphy. Returns [{id, title, url, preview_url, width, height}]."""
    q = (query or "").strip()
    if not q:
        return trending(limit=limit)
    resp = requests.get(
        f"{GIPHY_API}/search",
        params={
            "api_key": _key(),
            "q": q[:50],
            "limit": max(1, min(limit, 50)),
            "rating": "pg-13",
            "lang": "en",
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise GiphyError(f"Giphy search failed: {resp.text[:200]}")
    return _normalize(resp.json().get("data") or [])


def trending(*, limit: int = 24) -> list[dict]:
    """Trending GIFs (used when the search box is empty)."""
    resp = requests.get(
        f"{GIPHY_API}/trending",
        params={
            "api_key": _key(),
            "limit": max(1, min(limit, 50)),
            "rating": "pg-13",
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise GiphyError(f"Giphy trending failed: {resp.text[:200]}")
    return _normalize(resp.json().get("data") or [])
