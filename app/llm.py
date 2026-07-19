"""LLM helpers: relevance scoring, comment classification, reply drafting,
highlight suggestion, analytics digest. All calls go through Anthropic's API.
"""
from __future__ import annotations

import json
import re

from anthropic import Anthropic

from .config import env

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=env("ANTHROPIC_API_KEY"))
    return _client


def _text_chat(model: str, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    kwargs = dict(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        resp = client().messages.create(**kwargs)
    except Exception as exc:
        # Some newer models deprecate/reject `temperature`; retry without it.
        if "temperature" in str(exc).lower():
            kwargs.pop("temperature", None)
            resp = client().messages.create(**kwargs)
        else:
            raise
    return "".join(block.text for block in resp.content if block.type == "text")


def _json_chat(model: str, system: str, user: str, max_tokens: int = 1000) -> dict:
    system = system + "\nRespond with a single JSON object only — no prose, no code fences."
    text = _text_chat(model, system, user, max_tokens=max_tokens).strip()
    # Strip code fences and any stray text around the JSON object.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


TOPIC_TAXONOMY = [
    "wildfire", "flood", "heat", "drought", "hurricane", "sea_level",
    "storms", "clean_energy", "emissions_policy", "air_quality", "ecosystem", "other",
]


def score_relevance(model: str, title: str, description: str, matched_keywords: list[str]) -> dict:
    """Return {score: float 0-1, rationale: str, topics: [str, ...]}."""
    system = (
        "You score local TV news videos for genuine climate-change relevance. "
        "A video is relevant if it covers climate change, its impacts (extreme weather, "
        "wildfire, flood, heat, drought, sea level), clean energy, emissions, or climate "
        "policy. It is NOT relevant if the keyword is incidental ('political climate', "
        "'business climate', a sports team name, routine weather forecasts with no "
        "climate angle). Tag 1-3 topics that genuinely apply, most central first. "
        "JSON shape: {\"score\": 0.0-1.0, \"rationale\": \"one line\", "
        f"\"topics\": [\"1-3 of: {', '.join(TOPIC_TAXONOMY)}\"]}}"
    )
    user = json.dumps(
        {"title": title, "description": description[:2000], "matched_keywords": matched_keywords}
    )
    data = _json_chat(model, system, user)
    raw_topics = data.get("topics") or ([data["topic"]] if data.get("topic") else [])
    topics = [t for t in (str(x).strip() for x in raw_topics) if t in TOPIC_TAXONOMY][:3]
    return {
        "score": max(0.0, min(1.0, float(data.get("score", 0.0)))),
        "rationale": str(data.get("rationale", ""))[:500],
        "topics": topics or ["other"],
    }


def classify_comment(model: str, categories: list[str], post_caption: str, comment_text: str, username: str) -> dict:
    """Return {category, rationale, risk_flags: [..]}."""
    system = (
        "You classify comments on the operator's own Threads posts about climate news "
        f"clips. Pick exactly one category from: {', '.join(categories)}. "
        "Also list risk flags from: low_history_account_suspected, duplicate_or_coordinated_text, "
        "political_bait, sarcasm_suspected, inauthentic_or_sus, none. "
        "Be conservative: when uncertain whether a comment is genuinely supportive or a "
        "good-faith question, prefer a non-eligible category — a promotional reply to the "
        "wrong comment looks tone-deaf. JSON shape: "
        "{\"category\": \"...\", \"rationale\": \"one line\", \"risk_flags\": [\"...\"]}"
    )
    user = json.dumps({"post_caption": post_caption[:800], "comment": comment_text[:1000], "username": username})
    data = _json_chat(model, system, user)
    category = str(data.get("category", "off_topic"))
    if category not in categories:
        category = "off_topic"
    flags = [f for f in data.get("risk_flags", []) if f and f != "none"]
    return {"category": category, "rationale": str(data.get("rationale", ""))[:500], "risk_flags": flags}


def draft_reply(model: str, guidance: str, post_caption: str, comment_text: str, username: str, recent_replies: list[str]) -> str:
    """Draft one context-aware reply. `recent_replies` are the operator's recent
    posted replies, provided so wording does not repeat."""
    system = (
        "You draft a reply for the operator to review and edit before posting — it will "
        "never be posted automatically. Follow this guidance:\n" + guidance + "\n"
        "Avoid any wording similar to these recent replies (vary structure and phrasing):\n"
        + "\n".join(f"- {r[:200]}" for r in recent_replies[-8:])
        + "\nJSON shape: {\"reply\": \"...\"}"
    )
    user = json.dumps({"post_caption": post_caption[:800], "comment": comment_text[:1000], "username": username})
    data = _json_chat(model, system, user)
    return str(data.get("reply", "")).strip()[:500]


def suggest_highlight(model: str, title: str, transcript_segments: list[dict]) -> dict:
    """Given timestamped transcript segments, suggest the strongest 15-40s window
    and a draft caption. Returns {start, end, why, draft_caption}. DRAFT ONLY."""
    compact = [
        {"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"][:200]}
        for s in transcript_segments[:400]
    ]
    system = (
        "You find the single strongest 15-40 second window of a local TV news climate "
        "segment for a short social clip: the most vivid, concrete, human moment. "
        "Also draft a short caption (under 300 chars) the operator will rewrite. "
        "JSON shape: {\"start_seconds\": n, \"end_seconds\": n, \"why\": \"one line\", "
        "\"draft_caption\": \"...\"}"
    )
    user = json.dumps({"title": title, "segments": compact})
    data = _json_chat(model, system, user)
    return {
        "start": float(data.get("start_seconds", 0)),
        "end": float(data.get("end_seconds", 0)),
        "why": str(data.get("why", ""))[:300],
        "draft_caption": str(data.get("draft_caption", ""))[:400],
    }


def suggest_post_caption(model: str, title: str, station: str, market: str,
                         excerpt: str, clip_seconds: float | None) -> str:
    """Recommend Threads post text for the operator's trimmed clip. The operator
    reviews/edits before posting — this is a DRAFT, never auto-posted."""
    system = (
        "You draft a Threads caption for a short local-TV climate news clip. The "
        "operator will edit it before posting. Style: concrete and human, lead with "
        "the most striking fact from the excerpt, mention the place, no hype, no "
        "emojis unless truly fitting, at most one question, under 350 characters. "
        "Do not invent facts not in the excerpt. "
        "JSON shape: {\"caption\": \"...\"}"
    )
    user = json.dumps({
        "video_title": title,
        "station": station,
        "market": market,
        "clip_length_seconds": clip_seconds,
        "transcript_excerpt_of_clip": excerpt[:3000],
    })
    data = _json_chat(model, system, user)
    return str(data.get("caption", "")).strip()[:500]


def caption_attributes(model: str, caption: str) -> dict:
    """Tag a published caption's attributes for analytics. Returns
    {tone, has_question, has_cta, hashtag_count}."""
    system = (
        "Tag this social media caption. JSON shape: {\"tone\": \"one of: urgent, hopeful, "
        "informative, alarmed, neutral, humorous\", \"has_question\": bool, "
        "\"has_cta\": bool, \"hashtag_count\": int}"
    )
    data = _json_chat(model, system, json.dumps({"caption": caption[:1000]}))
    return {
        "tone": str(data.get("tone", "neutral")),
        "has_question": bool(data.get("has_question", False)),
        "has_cta": bool(data.get("has_cta", False)),
        "hashtag_count": int(data.get("hashtag_count", 0)),
    }


def write_digest(model: str, stats_payload: dict, min_sample_size: int) -> str:
    """Produce the periodic written performance digest (plain text/markdown)."""
    system = (
        "You are a careful social media analyst writing a performance digest for a "
        "single-operator Threads account posting climate news clips. Using ONLY the "
        "provided data: report top and bottom performers per metric; surface patterns "
        "across attribute slices (topic, region, clip length, caption traits, day/time); "
        "state hypotheses for WHY, clearly labeled as hypotheses, never as proven cause; "
        "label all patterns as correlational. "
        f"If total posts < {min_sample_size}, lead with a prominent small-sample caveat "
        "and avoid claiming any pattern. End with 2-3 lightweight experiment suggestions "
        "for upcoming posts, framed as tests, not guarantees. Write concise markdown."
    )
    return _text_chat(model, system, json.dumps(stats_payload), max_tokens=3000, temperature=0.4)
