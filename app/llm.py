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


def score_relevance(model: str, title: str, description: str, matched_keywords: list[str]) -> dict:
    """Return {score: float 0-1, rationale: str}."""
    system = (
        "You score local TV news videos for genuine climate-change relevance. "
        "A video is relevant if it covers climate change, its impacts (extreme weather, "
        "wildfire, flood, heat, drought, sea level), clean energy, emissions, or climate "
        "policy. It is NOT relevant if the keyword is incidental ('political climate', "
        "'business climate', a sports team name, routine weather forecasts with no "
        "climate angle). "
        "JSON shape: {\"score\": 0.0-1.0, \"rationale\": \"one line\"}"
    )
    user = json.dumps(
        {"title": title, "description": description[:2000], "matched_keywords": matched_keywords}
    )
    data = _json_chat(model, system, user)
    return {
        "score": max(0.0, min(1.0, float(data.get("score", 0.0)))),
        "rationale": str(data.get("rationale", ""))[:500],
    }


def suggest_channel_fields(model: str, url: str, title: str = "", description: str = "",
                           country_code: str = "", recent_titles: list[str] | None = None) -> dict:
    """Infer editorial channel metadata from a YouTube channel's public info.

    Given the channel URL plus whatever the Data API returned (title,
    description, ISO country code, and a few recent upload titles), guess the
    fields the operator would otherwise type by hand. Everything is a best-effort
    DRAFT the operator reviews before saving.

    Returns {call_sign, network, market, region, country, scope} where scope is
    one of local | national | international.
    """
    system = (
        "You help catalog news/media YouTube channels. From a channel's public "
        "info, infer these fields for a media-monitoring database:\n"
        "- call_sign: the station call sign or short brand name (e.g. 'KXYZ', "
        "'BBC News', 'Al Jazeera'). Prefer an official call sign for US/Canada "
        "broadcast stations; otherwise the common brand name.\n"
        "- network: parent network/affiliation if clear (e.g. 'ABC', 'NBC', "
        "'CBS', 'FOX', 'CNN', 'BBC'), else empty.\n"
        "- market: the primary city/metro the outlet covers (e.g. "
        "'Springfield', 'San Diego'), empty for national/international outlets.\n"
        "- region: state/province or broader region (e.g. 'California', "
        "'Midwest'), else empty.\n"
        "- country: full country name (e.g. 'United States', 'United Kingdom'). "
        "Convert any ISO country code to its full name.\n"
        "- scope: 'local' for a single-market station, 'national' for a "
        "country-wide outlet, 'international' for a global outlet.\n"
        "Only assert what the info supports; leave a field as an empty string "
        "when genuinely unknown rather than guessing wildly. "
        "JSON shape: {\"call_sign\": \"...\", \"network\": \"...\", "
        "\"market\": \"...\", \"region\": \"...\", \"country\": \"...\", "
        "\"scope\": \"local|national|international\"}"
    )
    user = json.dumps({
        "url": url,
        "channel_title": title,
        "channel_description": (description or "")[:1500],
        "country_code": country_code,
        "recent_video_titles": [t[:120] for t in (recent_titles or [])[:10]],
    })
    data = _json_chat(model, system, user)
    scope = str(data.get("scope", "local")).strip().lower()
    if scope not in ("local", "national", "international"):
        scope = "local"
    return {
        "call_sign": str(data.get("call_sign", "")).strip()[:40],
        "network": str(data.get("network", "")).strip()[:40],
        "market": str(data.get("market", "")).strip()[:80],
        "region": str(data.get("region", "")).strip()[:80],
        "country": str(data.get("country", "")).strip()[:60],
        "scope": scope,
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


def tag_footage(model: str, images: list[bytes], traits: list[str],
                title: str = "") -> dict:
    """Tag which traits from the vocabulary are visibly present in footage stills.

    Neutral observation only — no good/bad score. ``images`` are JPEG bytes
    (YouTube storyboard sheets or a contact sheet from a posted clip). Returns
    {traits: [detected], why: str}.
    """
    vocab = [t for t in traits if t]
    system = (
        "You label footage stills with a fixed vocabulary. These may be YouTube "
        "storyboard grids or a contact sheet of frames from a short clip. "
        "List ONLY traits from the vocabulary that are clearly visible — do not "
        "guess, and do not invent new trait names. Do not judge quality or "
        "appeal; observation only.\n"
        f"Vocabulary: {', '.join(vocab) or '(empty)'}.\n"
        "JSON shape: {\"traits\": [\"...\"], \"why\": \"one line\"}"
    )
    blocks: list = [{
        "type": "text",
        "text": (f"Clip title: {title}\n" if title else "")
        + "Tag the footage in these stills.",
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
    allowed = set(vocab)
    found = [t for t in (str(x).strip() for x in data.get("traits", [])) if t in allowed]
    return {
        "traits": found,
        "why": str(data.get("why", ""))[:300],
    }


def score_visuals(model: str, images: list[bytes], desirable_traits: list[str],
                  undesirable_traits: list[str] | None = None,
                  title: str = "", learned_guidance: str = "") -> dict:
    """Backward-compatible wrapper: tag-only (score dropped)."""
    del learned_guidance
    traits = list(desirable_traits or []) + list(undesirable_traits or [])
    result = tag_footage(model, images, traits, title=title)
    return {"visual_score": None, "traits": result["traits"], "why": result["why"]}


def suggest_post_caption(model: str, title: str, station: str, market: str,
                         excerpt: str, clip_seconds: float | None,
                         examples: list[str] | None = None,
                         style_guide: str = "", operator_guide: str = "") -> str:
    """Recommend Threads post text for the operator's trimmed clip. The operator
    reviews/edits before posting — this is a DRAFT, never auto-posted.

    ``examples``/``style_guide`` come from ``app/voice.py``: real captions the
    operator wrote, so the draft matches their voice instead of a generic one.
    ``operator_guide`` is the hand-written style guide from the Configure page —
    explicit instructions that take priority over the auto-learned voice.
    """
    system = (
        "You draft a Threads caption for a short local-TV climate news clip. The "
        "operator will edit it before posting. Hard constraints: do not invent "
        "facts not in the excerpt, mention the place, under 350 characters. "
    )
    if examples:
        system += (
            "\n\nVOICE: Write in the operator's own voice. Below are real captions "
            "they published — study the sentence rhythm, openings, punctuation, "
            "emoji/hashtag habits, and attitude, then write the new caption as if "
            "they wrote it. Match voice, never reuse their facts.\n\n"
            + "\n".join(f"<example>\n{e[:500]}\n</example>" for e in examples)
        )
        if style_guide:
            system += "\n\nStyle notes distilled from their full history:\n" + style_guide[:2000]
    else:
        system += (
            "Style: concrete and human, lead with the most striking fact from the "
            "excerpt, no hype, no emojis unless truly fitting, at most one question."
        )
    if operator_guide:
        system += (
            "\n\nOPERATOR STYLE GUIDE — the operator's preferred way of writing. "
            "Treat these as guidance, not rigid rules: apply each one when it fits "
            "this clip's transcript and context, and skip or adapt any that would "
            "feel forced or don't suit the material. Favor them over the general "
            "style notes above, but never over the hard constraints. Above all, the "
            "caption must read naturally for this specific clip:\n" + operator_guide[:2000]
        )
    system += "\nJSON shape: {\"caption\": \"...\"}"
    user = json.dumps({
        "video_title": title,
        "station": station,
        "market": market,
        "clip_length_seconds": clip_seconds,
        "transcript_excerpt_of_clip": excerpt[:3000],
    })
    data = _json_chat(model, system, user, max_tokens=1500)
    return str(data.get("caption", "")).strip()[:500]


def distill_style_guide(model: str, captions: list[str]) -> str:
    """Distill the operator's caption-writing voice into a short reusable style
    guide (plain text bullets). Rebuilt occasionally as history grows."""
    system = (
        "You are a writing-voice analyst. Given social media captions all written "
        "by one person, produce a compact style guide (6-10 plain-text bullets, "
        "no headers) that would let a ghostwriter imitate them: sentence length "
        "and rhythm, how they open and close, punctuation and capitalization "
        "quirks, emoji/hashtag habits, tone and attitude, recurring moves (e.g. "
        "quotes, stats, questions). Describe only patterns actually present. "
        "JSON shape: {\"style_guide\": \"- bullet\\n- bullet\"}"
    )
    user = json.dumps({"captions": [c[:500] for c in captions[:30]]})
    data = _json_chat(model, system, user, max_tokens=1200)
    return str(data.get("style_guide", "")).strip()[:3000]


def suggest_caption_rules(model: str, strong_captions: list[str],
                          weak_captions: list[str] | None = None,
                          existing_rules: list[str] | None = None) -> list[dict]:
    """Distill concrete, reusable *editorial/formatting* rules from the operator's
    own captions — the composition moves that make their strong posts work, phrased
    as instructions they could apply to every future caption.

    ``strong_captions`` are their higher-performing (or, absent metrics, most
    recent hand-written) captions; ``weak_captions`` are lower-performing ones for
    contrast. Advisory only — the operator promotes the ones that ring true.
    """
    # Line-based output (not JSON): these rules are about pull quotes and framing,
    # so the text routinely contains quotation marks and apostrophes that break
    # strict JSON parsing. One rule per line with a rare ``:::`` delimiter sidesteps
    # all escaping issues.
    system = (
        "You are an editorial coach for someone who posts short local-TV climate "
        "news clips on Threads. You are shown captions they published; when "
        "available they're split into higher- and lower-performing sets. Infer a "
        "short list of CONCRETE, REUSABLE composition rules that capture what makes "
        "the strong captions work — structural and editorial patterns to apply to "
        "every future caption.\n\n"
        "Focus on FORMAT and FRAMING: how to open, how to close, how to use quotes "
        "or stats, how to frame contested/denial viewpoints, rhythm, and what to "
        "avoid. Each rule must be ONE imperative instruction, specific and "
        "actionable. Good examples of the style and specificity wanted:\n"
        "- Lead with a one-line pull quote from the transcript.\n"
        "- End with a short, wry question.\n"
        "- Frame climate-denial perspectives impartially, without editorializing.\n\n"
        "Avoid vague advice ('be engaging'), do NOT restate hard constraints "
        "(don't invent facts, mention the place, length limit), and do NOT "
        "duplicate the operator's existing rules. Base them only on patterns "
        "actually visible in the captions.\n\n"
        "OUTPUT FORMAT: 4-6 rules, strongest first, one per line, nothing else. "
        "Format each line exactly as:\n"
        "<imperative rule> ::: <short reason>\n"
        "No numbering, no bullets, no quotes around the line, no preamble, no code fences."
    )
    user = json.dumps({
        "existing_rules": [r[:200] for r in (existing_rules or [])][:40],
        "higher_performing_captions": [c[:500] for c in (strong_captions or [])][:15],
        "lower_performing_captions": [c[:500] for c in (weak_captions or [])][:8],
    })
    text = _text_chat(model, system, user, max_tokens=1200, temperature=0.4)
    out: list[dict] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        rule, _, why = line.partition(":::")
        rule = rule.strip().strip('"').strip("“”").strip()
        if not rule:
            continue
        out.append({"rule": rule[:300], "why": why.strip()[:200]})
    return out


def suggest_title(model: str, source_title: str, transcript_excerpt: str,
                  caption: str | None = None) -> str:
    """Generate a concise, human-readable title for a trimmed climate news clip.

    Draws on the clip's own transcript excerpt (the trimmed windows) plus the
    original source title/description and, optionally, the draft caption. Returns
    a single punchy plain-text title (no surrounding quotes), roughly <= 70 chars.
    """
    system = (
        "You write a short, punchy title for a local-TV climate news clip that has "
        "been trimmed to its strongest moment. Base it on what the clip actually "
        "says (the transcript excerpt), using the source title only for context. "
        "Rules: one line, plain text, no surrounding quotes, no emojis, no hashtags, "
        "at most ~70 characters, concrete and faithful to the clip — do not invent "
        "facts. Prefer the place and the striking detail over vague phrasing. "
        "JSON shape: {\"title\": \"...\"}"
    )
    user = json.dumps({
        "source_title": source_title,
        "transcript_excerpt_of_clip": transcript_excerpt[:3000],
        "draft_caption": (caption or "")[:800],
    })
    data = _json_chat(model, system, user)
    title = str(data.get("title", "")).strip().strip('"').strip("'").strip()
    return title[:120]


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
        "across attribute slices (keywords, region, clip length, caption traits, day/time, "
        "and visual/footage traits such as fire, flood, crowds, action); "
        "state hypotheses for WHY, clearly labeled as hypotheses, never as proven cause; "
        "label all patterns as correlational. "
        f"If total posts < {min_sample_size}, lead with a prominent small-sample caveat "
        "and avoid claiming any pattern. End with 2-3 lightweight experiment suggestions "
        "for upcoming posts, framed as tests, not guarantees. Write concise markdown."
    )
    return _text_chat(model, system, json.dumps(stats_payload), max_tokens=3000, temperature=0.4)
