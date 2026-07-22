"""YouTube Data API v3 client (discovery only — no downloading here).

Uses plain HTTPS requests with an API key. Endpoints used:
  - channels.list  (resolve handle/username -> channel id, uploads playlist)
  - playlistItems.list  (page recent uploads from the UU... playlist)
  - videos.list  (durations for a batch of video ids)
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import requests

from .config import env

API_BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeAPIError(RuntimeError):
    pass


def _get(endpoint: str, **params) -> dict:
    key = env("YOUTUBE_API_KEY")
    if not key:
        raise YouTubeAPIError("YOUTUBE_API_KEY is not set (see .env.example)")
    params["key"] = key
    resp = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=30)
    if resp.status_code != 200:
        raise YouTubeAPIError(f"{endpoint} HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def is_short(video_id: str) -> bool:
    """True if the video is a YouTube Short (portrait/vertical).

    The Data API doesn't expose orientation, but youtube.com/shorts/<id>
    returns 200 for Shorts and redirects to /watch for regular videos.
    Uncertain (network error) counts as not-a-Short so we never drop
    a real candidate by accident.
    """
    try:
        resp = requests.head(
            f"https://www.youtube.com/shorts/{video_id}",
            allow_redirects=False, timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def parse_channel_url(url: str) -> tuple[str, str]:
    """Return (kind, value) where kind is 'id' | 'handle' | 'username' | 'custom'."""
    url = url.strip().rstrip("/")
    m = re.search(r"youtube\.com/channel/(UC[\w-]+)", url)
    if m:
        return "id", m.group(1)
    m = re.search(r"youtube\.com/@([\w.\-]+)", url)
    if m:
        return "handle", m.group(1)
    m = re.search(r"youtube\.com/user/([\w.\-]+)", url)
    if m:
        return "username", m.group(1)
    m = re.search(r"youtube\.com/c/([\w.\-]+)", url)
    if m:
        return "custom", m.group(1)
    # Bare path like youtube.com/fox5sd — treat as a handle-ish custom name.
    m = re.search(r"youtube\.com/([\w.\-]+)$", url)
    if m:
        return "custom", m.group(1)
    raise YouTubeAPIError(f"Unrecognized channel URL: {url}")


def parse_video_url(url: str) -> str:
    """Extract the 11-char video id from any YouTube video URL form.

    Handles watch?v=, youtu.be/, /shorts/, /embed/, /live/, and a bare id.
    Raises YouTubeAPIError if nothing looks like a video id.
    """
    url = url.strip()
    patterns = (
        r"[?&]v=([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"youtube\.com/shorts/([\w-]{11})",
        r"youtube\.com/embed/([\w-]{11})",
        r"youtube\.com/live/([\w-]{11})",
    )
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[\w-]{11}", url):
        return url
    raise YouTubeAPIError(f"Unrecognized YouTube video URL: {url}")


def resolve_channel(url: str) -> dict:
    """Resolve any channel URL form to canonical info.

    Returns {channel_id, uploads_playlist_id, title}.
    """
    kind, value = parse_channel_url(url)
    params: dict = {"part": "id,snippet,contentDetails"}
    if kind == "id":
        params["id"] = value
    elif kind == "handle":
        params["forHandle"] = value
    elif kind == "username":
        params["forUsername"] = value
    else:
        # Legacy /c/ or bare custom URLs have no direct lookup; forHandle often
        # works because most custom names were migrated to handles.
        params["forHandle"] = value

    data = _get("channels", **params)
    items = data.get("items") or []
    if not items and kind in ("custom", "username"):
        # Fall back to search (costs more quota, used rarely).
        sdata = _get("search", part="snippet", q=value, type="channel", maxResults=1)
        sitems = sdata.get("items") or []
        if sitems:
            cid = sitems[0]["snippet"]["channelId"]
            data = _get("channels", part="id,snippet,contentDetails", id=cid)
            items = data.get("items") or []
    if not items:
        raise YouTubeAPIError(f"Could not resolve channel for URL: {url}")

    item = items[0]
    return {
        "channel_id": item["id"],
        "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
        "title": item["snippet"]["title"],
    }


@dataclass
class Upload:
    video_id: str
    title: str
    description: str
    published_at: dt.datetime
    thumbnail_url: str
    duration_seconds: int | None = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def _parse_ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_iso8601_duration(value: str) -> int:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value or "")
    if not m:
        return 0
    d, h, mi, s = (int(g) if g else 0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def list_recent_uploads(uploads_playlist_id: str, since: dt.datetime, max_results: int = 25) -> list[Upload]:
    """Page the uploads playlist newest-first, stopping once items are older than `since`."""
    uploads: list[Upload] = []
    page_token = None
    while len(uploads) < max_results:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": min(50, max_results),
        }
        if page_token:
            params["pageToken"] = page_token
        data = _get("playlistItems", **params)
        items = data.get("items") or []
        if not items:
            break
        hit_older = False
        for item in items:
            snip = item["snippet"]
            published = _parse_ts(item["contentDetails"].get("videoPublishedAt") or snip["publishedAt"])
            if published <= since:
                hit_older = True
                continue
            thumbs = snip.get("thumbnails") or {}
            thumb = (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url", "")
            uploads.append(
                Upload(
                    video_id=item["contentDetails"]["videoId"],
                    title=snip.get("title", ""),
                    description=snip.get("description", ""),
                    published_at=published,
                    thumbnail_url=thumb,
                )
            )
        page_token = data.get("nextPageToken")
        if hit_older or not page_token:
            break

    # Batch-fetch durations.
    if uploads:
        ids = ",".join(u.video_id for u in uploads[:50])
        vdata = _get("videos", part="contentDetails", id=ids)
        # Live streams / premieres / upcoming videos may have no duration.
        durations = {
            item["id"]: parse_iso8601_duration(item.get("contentDetails", {}).get("duration", ""))
            for item in vdata.get("items", [])
        }
        for u in uploads:
            u.duration_seconds = durations.get(u.video_id)
    return uploads[:max_results]
