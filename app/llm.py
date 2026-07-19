"""LLM helpers: relevance scoring, comment classification, reply drafting,
highlight suggestion, analytics digest. All calls go through Anthropic's API.
"""
from __future__ import annotations

import base64
import json
import re

from anthropic import Anthropic

from . import spend
from .config import env

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=env("ANTHROPIC_API_KEY"))
    return _client


def _create(model: str, system: str, content, max_tokens: int, temperature: float):
    """Single entry point for Anthropic message calls. `content` is either a
    plain string or a list of content blocks (for multimodal). Records token
    usage in the spend ledger."""
    kwargs = dict(
        model=model,
        system=system,
        messages=[{"role": "user", "content": content}],
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
    usage = getattr(resp, "usage", None)
    if usage is not None:
        spend.record(model, getattr(usage, "input_tokens", 0) or 0,
                     getattr(usage, "output_tokens", 0) or 0)
    return resp


def _text_from(resp) -> str:
    return "".join(block.text for block in resp.content if block.type == "text")


def _text_chat(model: str, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    return _text_from(_create(model, system, user, max_tokens, temperature))


def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip code fences and any stray text around the JSON object.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


def _json_chat(model: str, system: str, user: str, max_tokens: int = 1000) -> dict:
    system = system + "\nRespond with a single JSON object only — no prose, no code fences."
    return _parse_json(_text_chat(model, system, user, max_tokens=max_tokens))


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
    return str(data.get("reply", "")).strip()[:700]


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


def score_visuals(model: str, images: list[bytes], desirable_traits: list[str],
                  undesirable_traits: list[str] | None = None,
                  title: str = "", learned_guidance: str = "") -> dict:
    """Judge how visually engaging a clip's FOOTAGE looks from storyboard stills.

    ``images`` are JPEG bytes (YouTube storyboard sprite sheets — each is a
    contact-sheet of small stills sampled across the clip). The model tags which
    DESIRABLE traits (raise appeal) and UNDESIRABLE traits (lower appeal) are
    present, and scores overall visual punch. Returns
    {visual_score: 0-1, traits: [detected, both kinds], why: str}. The score
    reflects action/drama/human interest, NOT climate relevance.
    """
    undesirable_traits = undesirable_traits or []
    system = (
        "You rate how visually engaging a short news clip would be on social "
        "media, judging ONLY the footage shown in these storyboard stills (a "
        "grid of small preview frames sampled across the clip). This is about "
        "visual punch, not topic importance. Raise the score for DESIRABLE "
        f"traits and lower it for UNDESIRABLE traits.\n"
        f"DESIRABLE traits (score UP): {', '.join(desirable_traits)}.\n"
        f"UNDESIRABLE traits (score DOWN): {', '.join(undesirable_traits) or '(none)'}.\n"
        "In 'traits', list every trait from EITHER list that is visibly present "
        "(use the exact names given). "
        "JSON shape: {\"visual_score\": 0.0-1.0, \"traits\": [\"...\"], "
        "\"why\": \"one line\"}"
    )
    if learned_guidance:
        system += "\n\nObserved performance so far (use as a soft prior, not a rule):\n" + learned_guidance
    blocks: list = [{
        "type": "text",
        "text": (f"Clip title: {title}\n" if title else "")
        + "Rate the footage in these storyboard stills.",
    }]
    for img in images:
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(img).decode("ascii"),
            },
        })
    resp = _create(model, system + "\nRespond with a single JSON object only — no prose, no code fences.",
                   blocks, max_tokens=500, temperature=0.2)
    data = _parse_json(_text_from(resp))
    allowed = set(desirable_traits) | set(undesirable_traits)
    found = [t for t in (str(x).strip() for x in data.get("traits", [])) if t in allowed]
    return {
        "visual_score": max(0.0, min(1.0, float(data.get("visual_score", 0.0)))),
        "traits": found,
        "why": str(data.get("why", ""))[:300],
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
        "across attribute slices (topic, region, clip length, caption traits, day/time, "
        "and visual/footage traits such as fire, flood, crowds, action); "
        "state hypotheses for WHY, clearly labeled as hypotheses, never as proven cause; "
        "label all patterns as correlational. "
        f"If total posts < {min_sample_size}, lead with a prominent small-sample caveat "
        "and avoid claiming any pattern. End with 2-3 lightweight experiment suggestions "
        "for upcoming posts, framed as tests, not guarantees. Write concise markdown."
    )
    return _text_chat(model, system, json.dumps(stats_payload), max_tokens=3000, temperature=0.4)
