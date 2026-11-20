"""Onboarding's video sources: find the YouTube channels a new account watches.

A climate account watches ~190 channels; nobody should have to add those by
hand. After the keywords screen, this runs in the background:

1. Claude proposes channels from the customer's answers (what the brand does,
   its throughline, the kinds of clips that carry it, the keywords), in two
   halves: channels about the topic itself, and channels whose footage carries
   the throughline beyond it. It also names a few YouTube searches.
2. Every proposal and search result is resolved against the YouTube API, so a
   channel that doesn't exist drops out, and trial-run: its recent uploads are
   checked against the keywords, the same test the monitor applies.
3. The channels with matching uploads are kept, best first.

The customer sees a cross-section, can drop any, add their own, or say what to
find more or less of and run it again. Saving the screen adds the channels to
the account. From then on channel health (app/channel_health.py) curates them:
channels whose videos never get approved are retired and replaced.

State is its own shared document (``obsources``), separate from the onboarding
record, so progress updates from the search never race the customer's saves.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import spend, youtube
from .config import load_keywords, with_american_english
from .matching import find_keyword_matches

log = logging.getLogger("onboarding.sources")

DOC_KEY = "obsources"
TRIAL_DAYS = 90
TRIAL_UPLOADS = 50          # one page of the uploads playlist: one quota unit
MAX_KEPT = 150
SAMPLE_SIZE = 12
WORKERS = 8
_lock = threading.Lock()


# --- State ------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _blank() -> dict[str, Any]:
    return {"status": "idle", "checked": 0, "total": 0, "error": "", "feedback": "",
            "channels": [], "removed": []}


def load() -> dict[str, Any]:
    from .shared_docs import read_text

    try:
        data = json.loads(read_text(DOC_KEY) or "null")
    except ValueError:
        data = None
    data = data if isinstance(data, dict) else _blank()
    for k, v in _blank().items():
        data.setdefault(k, v)
    return data


def _save(state: dict[str, Any]) -> None:
    from .shared_docs import write_text

    state["updated_at"] = _now()
    write_text(DOC_KEY, json.dumps(state, ensure_ascii=False))


def mutate(fn) -> dict[str, Any]:
    """Read, change and write the state under a lock (the search thread and
    the customer's clicks both write it)."""
    with _lock:
        state = load()
        fn(state)
        _save(state)
        return state


def reset() -> None:
    with _lock:
        _save(_blank())


def chosen(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in state["channels"] if c.get("on")]


def sample(state: dict[str, Any], size: int = SAMPLE_SIZE) -> list[dict[str, Any]]:
    """A cross-section: the best channel of each kind in turn, so the sample
    shows the range rather than the top dozen of one kind."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in state["channels"]:
        groups.setdefault(c.get("kind") or "Other", []).append(c)
    out: list[dict[str, Any]] = []
    while len(out) < size and any(groups.values()):
        for kind in list(groups):
            if groups[kind] and len(out) < size:
                out.append(groups[kind].pop(0))
    return out


# --- Proposing --------------------------------------------------------------------

_PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "channels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"},
                    "kind": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["name", "url", "kind", "why"],
                "additionalProperties": False,
            },
        },
        "searches": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["channels", "searches"],
    "additionalProperties": False,
}

_PROPOSE_SYSTEM = """\
You pick the YouTube channels a Throughline account will watch. Throughline \
finds new uploads on those channels whose title or description contains one of \
the account's keywords, cuts short clips from the ones that carry the account's \
throughline, and posts them daily. So a good channel uploads often, is real \
footage (news, archives, documentaries, talks, sports, everyday video) rather \
than talk-show punditry, and regularly carries the throughline.

Propose real channels you are confident exist, each with its canonical \
https://www.youtube.com/@handle URL. A channel that doesn't exist is checked \
and thrown away, so precision beats volume, but aim for the number asked. \
Label each with the kind of clip it supplies, using the account's kinds of \
clips where one fits, and say in one short sentence why it fits. Also name up \
to three YouTube search queries that would find more channels like these.
"""

_HALVES = (
    ("topic", "Channels about the topic itself: specialist channels, practitioners, "
              "institutions, outlets that cover it, archives of it."),
    ("beyond", "Channels whose footage carries the throughline beyond the topic: "
               "news outlets, archives, documentaries, talks, sports, pop culture and "
               "everyday video where the throughline shows up."),
)


def _brief(answers: dict[str, str], feedback: str, avoid: list[str]) -> str:
    return json.dumps({
        "brand": {k: answers.get(k, "") for k in
                  ("name", "mission", "audience", "throughline", "format", "off_limits")},
        "keywords": load_keywords()[:80],
        "feedback_from_the_customer": feedback,
        "already_have_or_removed": avoid[:300],
    }, ensure_ascii=False)


def propose(answers: dict[str, str], *, feedback: str = "", avoid: list[str] | None = None,
            client=None, per_half: int = 45) -> tuple[list[dict], list[str]]:
    """Proposed channels ({name, url, kind, why}) and search queries."""
    from .llm import client as default_client
    from .onboarding import _model

    client = client or default_client()
    model = _model()
    user_brief = _brief(answers, feedback, avoid or [])

    def half(spec: tuple[str, str]) -> dict:
        name, what = spec
        resp = client.messages.create(
            model=model, max_tokens=16000,
            system=with_american_english(_PROPOSE_SYSTEM),
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": _PROPOSE_SCHEMA}},
            messages=[{"role": "user", "content":
                       f"{user_brief}\n\nPropose {per_half} channels. This half: {what}"}],
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            spend.record(model, getattr(usage, "input_tokens", 0) or 0,
                         getattr(usage, "output_tokens", 0) or 0)
        data = json.loads("".join(b.text for b in resp.content if b.type == "text") or "{}")
        for c in data.get("channels") or []:
            c["half"] = name
        return data

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(half, _HALVES))
    channels, searches = [], []
    for data in results:
        channels += [c for c in data.get("channels") or []
                     if isinstance(c, dict) and "youtube.com/" in str(c.get("url", ""))]
        searches += [str(q).strip() for q in data.get("searches") or [] if str(q).strip()]
    return channels, searches[:4]


def search(queries: list[str]) -> list[dict]:
    """Channels YouTube's own search returns for the queries (100 quota units each)."""
    out = []
    for q in queries:
        try:
            data = youtube._get("search", part="snippet", q=q, type="channel", maxResults=10)
        except youtube.YouTubeAPIError as exc:
            log.info("Channel search %r failed: %s", q, exc)
            continue
        for item in data.get("items") or []:
            cid = (item.get("id") or {}).get("channelId") or item.get("snippet", {}).get("channelId")
            if cid:
                out.append({"name": item["snippet"].get("title", ""),
                            "url": f"https://www.youtube.com/channel/{cid}",
                            "kind": "More like these", "why": f"YouTube search: “{q}”"})
    return out


# --- Checking ----------------------------------------------------------------------

def check(proposal: dict, keywords: list[str]) -> dict | None:
    """Resolve one channel and trial-run its recent uploads. None when it
    doesn't exist or has nothing that matches."""
    try:
        info = youtube.resolve_channel(proposal["url"])
        uploads = youtube.list_recent_uploads(
            info["uploads_playlist_id"],
            since=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=TRIAL_DAYS),
            max_results=TRIAL_UPLOADS)
    except (youtube.YouTubeAPIError, KeyError, ValueError) as exc:
        log.info("Source %s dropped: %s", proposal.get("url"), exc)
        return None
    hits = [u for u in uploads if find_keyword_matches(f"{u.title}\n{u.description}", keywords)]
    if not hits:
        return None
    return {"channel_id": info["channel_id"], "uploads_playlist_id": info["uploads_playlist_id"],
            "title": info.get("title") or proposal.get("name", ""),
            "url": proposal["url"], "kind": (proposal.get("kind") or "Other").strip()[:60],
            "why": str(proposal.get("why", "")).strip()[:200],
            "hits": len(hits), "total": len(uploads), "example": hits[0].title[:140],
            "on": True}


def _score(c: dict) -> float:
    return c["hits"] / max(c["total"], 1) + min(c["hits"], 10) / 20


def discover(answers: dict[str, str], *, feedback: str = "", client=None) -> None:
    """The whole search, run off the request. Keeps the customer's own
    channels, and never brings back one they removed."""
    def start(s):
        s.update(status="running", checked=0, total=0, error="", feedback=feedback,
                 started_at=_now())
        s["channels"] = [c for c in s["channels"] if c.get("own")]
    state = mutate(start)
    avoid = [c["url"] for c in state["channels"]] + list(state.get("removed") or [])
    keywords = load_keywords()
    try:
        if not keywords:
            raise ValueError("Add some keywords first: they're how channels are checked.")
        proposals, queries = propose(answers, feedback=feedback, avoid=avoid, client=client)
        proposals += search(queries)
        seen, todo = set(avoid), []
        for p in proposals:
            key = p["url"].rstrip("/").lower()
            if key not in seen:
                seen.add(key)
                todo.append(p)
        mutate(lambda s: s.update(total=len(todo)))

        removed_ids = set(state.get("removed_ids") or [])
        kept: list[dict] = []
        kept_ids: set[str] = set()
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for result in pool.map(lambda p: check(p, keywords), todo):
                done += 1
                # The same channel can come from a proposal and a search under
                # different URLs; keep the first (it has the better label).
                if result and result["channel_id"] not in removed_ids | kept_ids:
                    kept.append(result)
                    kept_ids.add(result["channel_id"])
                if done % 6 == 0 or done == len(todo):
                    snapshot = sorted(kept, key=_score, reverse=True)[:MAX_KEPT]

                    def progress(s, n=done, snap=snapshot):
                        own = [c for c in s["channels"] if c.get("own")]
                        ids = {c["channel_id"] for c in own}
                        s["checked"] = n
                        s["channels"] = own + [c for c in snap if c["channel_id"] not in ids]
                    mutate(progress)
        mutate(lambda s: s.update(status="done", finished_at=_now()))
        log.info("Source search: %d of %d proposals kept", len(kept), len(todo))
    except Exception as exc:
        log.exception("Source search failed")
        mutate(lambda s: s.update(status="error", error=str(exc)[:300]))


def add_own(url: str) -> dict:
    """A channel the customer adds by URL: resolved and checked at once.
    Kept even when nothing matches yet (it's theirs), but they're told."""
    url = url.strip()
    if "youtube.com/" not in url and not url.startswith("@"):
        raise ValueError("Paste a YouTube channel link, like youtube.com/@channel")
    if url.startswith("@"):
        url = f"https://www.youtube.com/{url}"
    info = youtube.resolve_channel(url)
    uploads = youtube.list_recent_uploads(
        info["uploads_playlist_id"],
        since=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=TRIAL_DAYS),
        max_results=TRIAL_UPLOADS)
    keywords = load_keywords()
    hits = [u for u in uploads if find_keyword_matches(f"{u.title}\n{u.description}", keywords)]
    entry = {"channel_id": info["channel_id"], "uploads_playlist_id": info["uploads_playlist_id"],
             "title": info.get("title", ""), "url": url, "kind": "Your pick", "why": "Added by you",
             "hits": len(hits), "total": len(uploads),
             "example": hits[0].title[:140] if hits else "", "on": True, "own": True}

    def put(s):
        s["channels"] = [entry] + [c for c in s["channels"] if c["channel_id"] != entry["channel_id"]]
        s["removed_ids"] = [i for i in s.get("removed_ids") or [] if i != entry["channel_id"]]
    mutate(put)
    return entry


def toggle(channel_id: str, on: bool) -> None:
    def flip(s):
        for c in s["channels"]:
            if c["channel_id"] == channel_id:
                c["on"] = bool(on)
                ids = set(s.get("removed_ids") or [])
                urls = set(s.get("removed") or [])
                (ids.discard if on else ids.add)(channel_id)
                (urls.discard if on else urls.add)(c["url"].rstrip("/").lower())
                s["removed_ids"], s["removed"] = sorted(ids), sorted(urls)
    mutate(flip)


def commit() -> int:
    """Add the chosen channels to the account. Returns how many were new."""
    from sqlalchemy import select

    from .db import session_scope
    from .models import Channel

    picks = chosen(load())
    added = 0
    with session_scope() as session:
        existing = session.execute(select(Channel)).scalars().all()
        ids = {c.channel_id for c in existing if c.channel_id}
        urls = {(c.url or "").rstrip("/").lower() for c in existing}
        for c in picks:
            if c["channel_id"] in ids or c["url"].rstrip("/").lower() in urls:
                continue
            note = (f"Added in onboarding: {c['hits']} of its last {c['total']} uploads matched "
                    f"your keywords. {c.get('why', '')}").strip()
            session.add(Channel(
                call_sign=(c["title"] or c["url"])[:40], network=c.get("kind", "")[:40],
                market="", region="", country="", scope="national", url=c["url"][:300],
                channel_id=c["channel_id"], uploads_playlist_id=c["uploads_playlist_id"],
                channel_title=c["title"][:200], added_note=note))
            ids.add(c["channel_id"])
            added += 1
    return added
