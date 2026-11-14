"""Watch pages: link a post's first reply to the full segment on Renewables.org.

The page lives on the Renewables.org platform at renewables.org/watch/{slug}.
It embeds the station's own YouTube video starting at the clip, credits the
station, and follows it with Renewables.org explainer copy and a call to
action. This app only publishes the row behind the page
(PUT {api_url}/api/internal/watch-pages/{slug}, bearer
WATCH_PAGES_SERVICE_TOKEN). The footage is never uploaded anywhere: the row
holds only the YouTube video id.

The order matters, because Threads builds the link preview at the moment the
reply goes up:
  1. check the video is still public and embeddable (YouTube oEmbed),
  2. publish the row,
  3. fetch the page and require a 200,
  4. only then put the link in the reply.
If any step fails, the reply goes out exactly as before, with the plain
Renewables.org mention, and the reason is recorded on the post. A watch page
can make a first reply better but can never block or break one.

Settings (settings.yaml, ``watch_pages``):
  enabled         - master switch (default off)
  site_url        - public origin the link points at (https://renewables.org)
  api_url         - where the backend's /api lives (defaults to site_url;
                    the site proxies /api to the backend)
  lead_in_seconds - start the player this much before the clip begins
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import unicodedata
from urllib.parse import quote

import requests

from .categories import is_first_party
from .config import env, load_settings
from .models import Candidate, Cut, ThreadsPost

log = logging.getLogger(__name__)

TOKEN_ENV = "WATCH_PAGES_SERVICE_TOKEN"
# Explicit placeholder an operator can put in first-reply text. It becomes the
# watch link, or "Renewables.org" when there is no page.
PLACEHOLDER = "{watch_url}"
FALLBACK_MENTION = "Renewables.org"
HTTP_TIMEOUT = 15

_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
# A bare brand mention, not already part of a URL, path, handle or email.
_BARE_MENTION = re.compile(r"(?<![\w/@.:-])renewables\.org(?![\w/.-])", re.IGNORECASE)


def config() -> dict:
    s = load_settings()
    site = (s.get("watch_pages.site_url", "") or "https://renewables.org").rstrip("/")
    return {
        "enabled": bool(s.get("watch_pages.enabled", False)),
        "site_url": site,
        "api_url": (s.get("watch_pages.api_url", "") or site).rstrip("/"),
        "lead_in_seconds": max(0, int(s.get("watch_pages.lead_in_seconds", 2) or 0)),
    }


def slugify(text: str, max_len: int = 48) -> str:
    """Lowercase ASCII words joined by single hyphens (the backend's pattern)."""
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", ascii_text.lower())
    out = ""
    for w in words:
        nxt = f"{out}-{w}" if out else w
        if len(nxt) > max_len:
            break
        out = nxt
    return out


def page_slug(post: ThreadsPost, title: str) -> str:
    """Readable and unique: the title's leading words, then the post id. Kept
    short because it shows up in the reply text. Stable for a post, so
    publishing again updates the same page instead of making a new one."""
    base = slugify(title) or "segment"
    return f"{base}-{post.id}"


def clip_start_seconds(cut: Cut | None, lead_in: int = 0) -> int:
    """Where the clip begins in the source video, minus a short lead-in so the
    viewer lands just before the moment they saw, not mid-word."""
    if cut is None or not cut.trim_segments:
        return 0
    try:
        segments = json.loads(cut.trim_segments) or []
        start = min(float(s["start"]) for s in segments)
    except (ValueError, TypeError, KeyError):
        return 0
    return max(0, int(start) - lead_in)


def youtube_embeddable(video_id: str) -> tuple[bool, str, str]:
    """``(ok, reason, author)`` from YouTube's oEmbed endpoint: 401/403 means
    the uploader turned embedding off, 400/404 means private or deleted. A
    YouTube outage counts as not ok, because a reply that links to a dead player
    is worse than one with the plain mention. ``author`` is the uploading
    channel's name as YouTube shows it, which is the credit the page uses."""
    url = "https://www.youtube.com/oembed?format=json&url=" + quote(
        f"https://www.youtube.com/watch?v={video_id}", safe="")
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        return False, f"couldn't reach YouTube to check the video ({exc.__class__.__name__})", ""
    if resp.status_code == 200:
        try:
            author = resp.json().get("author_name", "")
        except ValueError:
            author = ""
        return True, "", author.strip() if isinstance(author, str) else ""
    if resp.status_code in (401, 403):
        return False, "the station has turned off embedding for this video", ""
    if resp.status_code in (400, 404):
        return False, "the video is private or has been removed", ""
    return False, f"YouTube answered {resp.status_code} when checking the video", ""


def _channel_name(candidate: Candidate) -> str:
    """Fallback credit from our own channel record. Videos pasted in by URL hang
    off a placeholder channel ("Pasted URLs"), which is why YouTube's own
    author name is preferred over this."""
    channel = candidate.channel
    if channel is None:
        return ""
    return (channel.channel_title or channel.call_sign or "").strip()


def _iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()


def ensure_watch_page(post: ThreadsPost) -> str:
    """Publish (or refresh) this post's watch page and return its URL, or ""
    when the post doesn't get one. Never raises. The outcome is written to
    ``post.watch_page_url`` / ``post.watch_page_error`` (the caller flushes)."""
    cfg = config()
    if not cfg["enabled"]:
        return ""
    url, error = _publish(post, cfg)
    post.watch_page_url = url
    post.watch_page_error = error
    if error:
        log.info("No watch page for post %s: %s", post.id, error)
    return url


def _publish(post: ThreadsPost, cfg: dict) -> tuple[str, str]:
    candidate = post.candidate
    if candidate is None:
        return "", "No watch page: this post has no source video."
    if is_first_party(candidate):
        # The brand's own promos are the ad themselves, and there's no
        # station footage to credit.
        return "", "No watch page: this is the brand's own footage."
    video_id = (candidate.video_id or "").strip()
    if not _YOUTUBE_ID.match(video_id):
        return "", "No watch page: the source isn't a YouTube video."
    token = env(TOKEN_ENV)
    if not token:
        return "", f"No watch page: {TOKEN_ENV} isn't set on this machine."

    ok, reason, author = youtube_embeddable(video_id)
    if not ok:
        return "", f"No watch page: {reason}."
    source = author or _channel_name(candidate)
    if not source:
        return "", "No watch page: there's no channel name to credit the video to."

    title = (candidate.title or "").strip()[:300] or "The full segment"
    slug = post.watch_page_slug or page_slug(post, title)
    body = {
        "youtube_video_id": video_id,
        "start_seconds": clip_start_seconds(post.cut, cfg["lead_in_seconds"]),
        "title": title,
        "source_name": source[:200],
        "source_published_at": _iso(candidate.published_at),
        "status": "published",
    }
    try:
        resp = requests.put(
            f"{cfg['api_url']}/api/internal/watch-pages/{slug}",
            json=body, timeout=HTTP_TIMEOUT,
            headers={"Authorization": f"Bearer {token}"},
        )
    except requests.RequestException as exc:
        return "", f"No watch page: couldn't reach Renewables.org ({exc.__class__.__name__})."
    if resp.status_code != 200:
        return "", f"No watch page: Renewables.org refused it ({resp.status_code}: {resp.text[:200]})."
    post.watch_page_slug = slug

    # The link only goes out once the page actually loads. Threads fetches
    # the preview when the reply is posted, and a broken card can't be fixed
    # afterwards.
    page_url = f"{cfg['site_url']}/watch/{slug}"
    try:
        check = requests.get(page_url, timeout=30)
    except requests.RequestException as exc:
        return "", f"No watch page: published, but the page didn't load ({exc.__class__.__name__})."
    if check.status_code != 200:
        return "", f"No watch page: published, but the page answered {check.status_code}."
    return page_url, ""


def display_url(url: str) -> str:
    """The link as written in the reply: no scheme, matching the house style
    for the bare mention ("Renewables.org", never "https://...")."""
    return re.sub(r"^https?://", "", url or "")


def apply_watch_link(text: str, url: str) -> str:
    """Put the watch link into first-reply text.

    The ``{watch_url}`` placeholder always resolves: to the link, or to
    "Renewables.org" when there's no page. Without a placeholder, the first
    bare "Renewables.org" mention becomes the link, so the existing copy works
    unchanged. Text with neither is left alone (it may be a citation with no
    call to action).
    """
    if PLACEHOLDER in text:
        return text.replace(PLACEHOLDER, display_url(url) if url else FALLBACK_MENTION)
    if not url:
        return text
    return _BARE_MENTION.sub(display_url(url), text, count=1)
