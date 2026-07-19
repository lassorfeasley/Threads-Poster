"""Fetch YouTube storyboard (scrub-preview) sprite sheets for a video.

This is a metadata-only fetch via yt-dlp (no video download), used by triage
mode to show a filmstrip so the operator can judge the visuals before
approving. Results are cached in memory for the process lifetime.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("storyboard")

_cache: dict[str, dict] = {}
_lock = threading.Lock()


def get_storyboard(video_id: str) -> dict:
    with _lock:
        if video_id in _cache:
            return _cache[video_id]

    result: dict = {"available": False}
    try:
        import yt_dlp

        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

        storyboards = [
            f for f in info.get("formats", [])
            if str(f.get("format_id", "")).startswith("sb") and f.get("fragments")
        ]
        if storyboards:
            # Highest-resolution storyboard track.
            sb = max(storyboards, key=lambda f: (f.get("height") or 0))
            result = {
                "available": True,
                "tile_width": sb.get("width"),
                "tile_height": sb.get("height"),
                "rows": sb.get("rows"),
                "columns": sb.get("columns"),
                "duration": info.get("duration"),
                "fragments": [
                    {"url": frag["url"], "duration": frag.get("duration")}
                    for frag in sb["fragments"]
                ],
            }
    except Exception as exc:
        log.info("Storyboard unavailable for %s: %s", video_id, exc)

    with _lock:
        _cache[video_id] = result
    return result
