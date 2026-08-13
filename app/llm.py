"""LLM helpers: relevance scoring, highlight suggestion, analytics digest.
All calls go through Anthropic's API.
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
    """Return {score: float 0-1}.

    A free-text rationale used to be returned alongside the score; it wasn't
    useful in the operator workflow and isn't a learning signal (numeric score
    + keyword hits + visual traits already drive ranking). Kept accepting an
    optional ``rationale`` key for old cached responses.
    """
    system = (
        "You score local TV news videos for genuine climate-change relevance. "
        "A video is relevant if it covers climate change, its impacts (extreme weather, "
        "wildfire, flood, heat, drought, sea level), clean energy, emissions, or climate "
        "policy. It is NOT relevant if the keyword is incidental ('political climate', "
        "'business climate', a sports team name, routine weather forecasts with no "
        "climate angle). "
        "JSON shape: {\"score\": 0.0-1.0}"
    )
    user = json.dumps(
        {"title": title, "description": description[:2000], "matched_keywords": matched_keywords}
    )
    data = _json_chat(model, system, user)
    return {
        "score": max(0.0, min(1.0, float(data.get("score", 0.0)))),
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


def suggest_category(model: str, categories: list[dict], title: str, description: str,
                     channel: str = "", matched_keywords: list[str] | None = None,
                     transcript_excerpt: str = "") -> dict:
    """Recommend ONE programming category for a video. ``categories`` come from
    settings ``categories.options`` ({slug, label, description}); the channel
    aims for a roughly equal mix of them, so this is genre/framing, not topic
    (every video is climate-related). Returns {category: slug or "", rationale};
    an off-vocabulary answer comes back as "" so the video stays untagged
    rather than mislabeled.
    """
    vocab = "\n".join(
        f"- {c['slug']}: {c['label']} — {c['description']}" for c in categories
    )
    system = (
        "You assign a programming category to a video for a climate-focused "
        "social channel. Every video is climate-related; the category captures "
        "the GENRE and framing of the footage, not the topic. Pick exactly one "
        "slug from:\n" + vocab + "\n"
        "Judge from the title, description, channel and transcript excerpt. "
        "JSON shape: {\"category\": \"slug\", \"rationale\": \"one line\"}"
    )
    user = json.dumps({
        "title": title,
        "description": (description or "")[:2000],
        "channel": channel,
        "matched_keywords": matched_keywords or [],
        "transcript_excerpt": (transcript_excerpt or "")[:2000],
    })
    data = _json_chat(model, system, user)
    slug = str(data.get("category", "")).strip().lower()
    if slug not in {c["slug"] for c in categories}:
        slug = ""
    return {"category": slug, "rationale": str(data.get("rationale", ""))[:500]}


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


def _tidy_caption(text: str) -> str:
    """Flatten a drafted caption to tight, single-spaced lines.

    Models sometimes answer a style guide with a multi-paragraph essay; blank
    lines and stray indentation are the tell. Collapsing them keeps the proposal
    readable in the review card. An overlong draft is left whole rather than cut
    mid-sentence — the operator sees the real draft and trims it themselves.
    """
    lines = [" ".join(ln.split()) for ln in text.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln)[:500]


def suggest_post_caption(model: str, title: str, station: str, market: str,
                         excerpt: str, clip_seconds: float | None,
                         examples: list[str] | None = None,
                         style_guide: str = "", operator_guide: str = "",
                         max_chars: int = 220,
                         target_words: int | None = None) -> str:
    """Recommend Threads post text for the operator's trimmed clip. The operator
    reviews/edits before posting — this is a DRAFT, never auto-posted.

    ``examples``/``style_guide`` come from ``app/voice.py``: real captions the
    operator wrote, so the draft matches their voice instead of a generic one.
    ``operator_guide`` is the hand-written style guide from the Configure page —
    a menu of preferred moves, deliberately framed so the model picks one rather
    than stacking them into a template-shaped caption.

    ``target_words`` is the length the operator has actually been posting
    lately (``voice.length_target``), and is the instruction that does the real
    work. ``max_chars`` is only a backstop: stated alone it behaves as a target
    to fill, which is how drafts drifted long enough that the operator rewrote
    nearly all of them.
    """
    system = (
        "You draft a Threads caption for a short local-TV climate news clip. The "
        "operator will edit it before posting.\n\n"
    )
    if target_words:
        # Framed as a goal to land on, with permission to go shorter. A ceiling
        # on its own reads as an allowance and gets spent.
        system += (
            f"LENGTH IS THE CONSTRAINT THAT MATTERS MOST. Aim for about "
            f"{target_words} words — that is the length this operator actually "
            f"posts, measured from their recent captions, not a guess. Coming in "
            f"UNDER it is always safe; going over is the single most common way "
            f"to get this wrong. A caption of just a few words is a success, not "
            f"an unfinished draft. Hard ceiling {max_chars} characters, but treat "
            f"{target_words} words as the goal and stop as soon as the line "
            f"lands.\n\n"
        )
    else:
        system += (
            "LENGTH IS THE CONSTRAINT THAT MATTERS MOST: one or two short lines, "
            f"two sentences at the absolute most, under {max_chars} characters. "
            "Coming in well under is always safe. A caption of just a few words "
            "is a success, not an unfinished draft.\n\n"
        )
    system += (
        "No paragraphs, no blank lines, no lists. The video carries the story — "
        "the caption only has to make someone stop and watch it. When torn "
        "between two good sentences, keep one.\n\n"
        "Also hard: do not invent facts not in the excerpt. "
    )
    # A mandatory place name can consume most of a very short caption, so it
    # becomes a preference once the target is tight.
    system += ("Mention the place when it fits the length.\n"
               if target_words and target_words <= 14
               else "Mention the place.\n")
    if examples:
        system += (
            "\n\nVOICE: Write in the operator's own voice. Below are real captions "
            "they published — study the sentence rhythm, openings, punctuation, "
            "emoji/hashtag habits, and attitude, then write the new caption as if "
            "they wrote it. Match their voice and diction, not their length: some "
            "of these run long, and yours must not. Never reuse their facts.\n\n"
            + "\n".join(f"<example>\n{e[:500]}\n</example>" for e in examples)
        )
        if style_guide:
            system += "\n\nStyle notes distilled from their full history:\n" + style_guide[:2000]
    else:
        system += (
            "Style: concrete and human, lead with the single most striking fact "
            "from the excerpt and stop there, no hype, no emojis unless truly "
            "fitting, at most one question."
        )
    if operator_guide:
        system += (
            "\n\nOPERATOR STYLE GUIDE — a MENU of moves the operator likes, not a "
            "checklist. Choose the ONE that best suits this clip and ignore the "
            "rest on purpose; if a move does not fit the length, skip it. Trying "
            "to satisfy several at once is the most common failure here: it "
            "produces a padded, template-shaped caption. A move that adds a "
            "sentence is not worth the sentence. These outrank the general style "
            "notes above, but never the length target or the hard "
            "constraints:\n" + operator_guide[:2000]
        )
    system += "\nJSON shape: {\"caption\": \"...\"}"
    user = json.dumps({
        "video_title": title,
        "station": station,
        "market": market,
        "clip_length_seconds": clip_seconds,
        "transcript_excerpt_of_clip": excerpt[:3000],
    })
    data = _json_chat(model, system, user, max_tokens=600)
    return _tidy_caption(str(data.get("caption", "")))


def suggest_hook_text(model: str, title: str, station: str, market: str,
                      excerpt: str, examples: list[str] | None = None) -> str:
    """Draft short on-video hook text for an Instagram Reel vertical composite.

    Rendered large in the brand font at the top of the 9:16 frame — so it must
    stay brief. DRAFT ONLY: the operator edits before regenerating the reel.

    ``examples`` are hooks the operator rewrote for themselves, taken from the
    draft ledger (``app/draft_proposals.operator_written``). The hook has no
    other voice source: unlike captions there is no published history to learn
    from, because the hook is burned into the video rather than posted as text.
    """
    system = (
        "You write a short HOOK line that appears as large on-screen text at the "
        "top of an Instagram Reel (climate / local-TV news clip). Hard rules:\n"
        "- 3–12 words, under 80 characters, ideally one line (two max).\n"
        "- Lead with the most striking fact or tension from the excerpt.\n"
        "- No hashtags, no URLs, no emojis, no quotation marks around the whole hook.\n"
        "- Do not invent facts not in the excerpt; mention the place when it matters.\n"
        "- Punchy and concrete — not a full caption, not a question unless irresistible.\n"
    )
    if examples:
        system += (
            "\nVOICE: hooks the operator WROTE THEMSELVES after discarding a "
            "draft like the one you're about to write. Each one is a correction "
            "toward how they actually write — study the rhythm, capitalisation, "
            "and how much they leave unsaid, then write as if they wrote it. "
            "Never reuse their facts.\n"
            + "\n".join(f"<example>{e[:120]}</example>" for e in examples) + "\n"
        )
    system += "JSON shape: {\"hook\": \"...\"}"
    user = json.dumps({
        "video_title": title,
        "station": station,
        "market": market,
        "transcript_excerpt_of_clip": excerpt[:3000],
    })
    data = _json_chat(model, system, user, max_tokens=400)
    return str(data.get("hook", "")).strip()[:300]


def suggest_attribution(model: str, channel: dict, video_title: str,
                        description: str = "", transcript: str = "",
                        published_at: str = "", video_url: str = "") -> str:
    """Draft a formal source citation for the first comment under a post,
    crediting the publisher (and the program/journalists when the source
    material establishes them). DRAFT ONLY: nothing posts until the operator
    accepts it into the attribution field.

    Returns "" when the available data cannot support a credible citation —
    the model is instructed to decline rather than guess, and callers surface
    that as "data not available" instead of a made-up credit.

    ``channel`` carries the station metadata (call_sign, network, market,
    region, country, channel_title). ``transcript`` is the FULL source-video
    transcript (not just the clipped segments), so the model sees the whole
    broadcast when identifying programs, segments, and journalists.
    """
    system = (
        "You write a formal source citation to be posted as the first comment "
        "under a short news clip on Threads, crediting the original publisher. "
        "The clip is a short excerpt; you are given the FULL source video's "
        "metadata and transcript — use all of it.\n\n"
        "Citation style — a formal credit line assembled from whichever of these "
        "elements the data clearly establishes, in this order:\n"
        "  Source: <Station/Publisher> (<Network>), \"<program or segment name>\", "
        "<Market, Region>, aired <date>. Reported by <journalist(s)>.\n"
        "Omit any element the data does not establish; never pad with guesses.\n\n"
        "Hard rules:\n"
        "- Only state facts the provided metadata, description, or transcript "
        "clearly establishes. NEVER guess or invent station names, programs, "
        "journalists, dates, or network affiliations.\n"
        "- If the data does not clearly establish at least the publisher, do NOT "
        "write a citation at all: return {\"available\": false, \"attribution\": \"\"}.\n"
        "- NEVER tag or mention any account: no @handles of any kind.\n"
        "- No hashtags, no URLs, no emojis.\n"
        "- Plain, factual tone; under 400 characters; one line.\n"
        "JSON shape: {\"available\": true|false, \"attribution\": \"...\"}"
    )
    user = json.dumps({
        "channel": {k: str(channel.get(k, "")) for k in
                    ("call_sign", "network", "market", "region", "country", "channel_title")},
        "video_title": (video_title or "")[:500],
        "video_published_date": (published_at or "")[:40],
        "video_url": (video_url or "")[:300],
        "video_description": (description or "")[:4000],
        "full_transcript": (transcript or "")[:24000],
    })
    data = _json_chat(model, system, user)
    text = str(data.get("attribution", "")).strip()
    if not data.get("available", bool(text)) or not text:
        return ""
    # Belt-and-braces: strip any @handle the model slipped in despite the rule.
    text = re.sub(r"@[\w.]+", "", text).strip()
    return " ".join(text.split())[:480]


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
        "Captions are only one or two short lines, and the drafter applies just "
        "one rule per caption — so each rule must stand on its own inside that "
        "space. Never propose a rule that requires extra sentences or a "
        "multi-part structure (e.g. 'open with X, then Y, then close with Z').\n\n"
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


def suggest_calendar_name(model: str, clip_title: str, caption: str | None = None) -> str:
    """Condense a clip's title into a 2-5 word label for the calendar's window
    slots, which have room for only a short phrase. Runs right after
    ``suggest_title`` produces (or regenerates) ``clip_title``.
    """
    system = (
        "You condense a video clip's title into a very short label for a small "
        "calendar tile. Rules: 2 to 5 words, plain text, no surrounding quotes, "
        "no emojis, no hashtags, no trailing punctuation, title case. Keep the "
        "single most identifying noun/place/subject from the title — do not "
        "invent facts or add words not implied by the title. "
        "JSON shape: {\"name\": \"...\"}"
    )
    user = json.dumps({
        "clip_title": clip_title[:300],
        "caption": (caption or "")[:400],
    })
    data = _json_chat(model, system, user)
    name = str(data.get("name", "")).strip().strip('"').strip("'").strip()
    # Defensive cap in case the model ignores the word-count rule.
    words = name.split()
    if len(words) > 5:
        name = " ".join(words[:5])
    return name[:48]


def suggest_short_title(model: str, source_title: str, description: str = "") -> str:
    """Distill a source video's (often long/clickbait) title into a punchy 2-5
    word clip title. Used when ingesting a pasted YouTube URL so the clip gets a
    concise human label instead of the raw YouTube title. Faithful to the
    source — no invented facts.
    """
    system = (
        "You write a very short title for a news video clip, distilled from its "
        "original (often long or clickbait) source title. Rules: 2 to 5 words, "
        "plain text, no surrounding quotes, no emojis, no hashtags, no trailing "
        "punctuation, title case. Keep the single most identifying subject/place "
        "from the source — do not invent facts or add words the source doesn't "
        "imply. JSON shape: {\"title\": \"...\"}"
    )
    user = json.dumps({
        "source_title": source_title[:300],
        "source_description": (description or "")[:600],
    })
    data = _json_chat(model, system, user)
    title = str(data.get("title", "")).strip().strip('"').strip("'").strip()
    words = title.split()
    if len(words) > 5:
        title = " ".join(words[:5])
    return title[:80]


def suggest_title_from_transcript(model: str, transcript_text: str,
                                  source_title: str = "") -> str:
    """Write a punchy 2-5 word clip title from what the video actually says.

    Preferred over :func:`suggest_short_title` whenever a transcript exists,
    because a publisher's own title can misdescribe its own footage — a San
    Diego station uploading a Napa council story under a San Diego headline, for
    one real case — and titling from the headline alone propagates that error
    into the clip, the calendar and the caption draft. ``source_title`` is passed
    only so proper nouns the transcript may have garbled can be spelled right;
    the transcript decides the facts.
    """
    if not (transcript_text or "").strip():
        return ""
    system = (
        "You write a very short title for a news video clip, based on what is "
        "actually said in its transcript. Rules: 2 to 5 words, plain text, no "
        "surrounding quotes, no emojis, no hashtags, no trailing punctuation, "
        "title case. Name the specific subject and place the transcript "
        "establishes. The transcript is the ONLY authority on the facts: where "
        "the supplied source_title disagrees with it about who, where or what, "
        "follow the transcript and ignore the source title. Use source_title "
        "only to spell proper nouns the transcript may have garbled. Never "
        "state anything the transcript does not support. "
        "JSON shape: {\"title\": \"...\"}"
    )
    user = json.dumps({
        "transcript": transcript_text[:6000],
        "source_title": (source_title or "")[:300],
    })
    data = _json_chat(model, system, user)
    title = str(data.get("title", "")).strip().strip('"').strip("'").strip()
    words = title.split()
    if len(words) > 5:
        title = " ".join(words[:5])
    return title[:80]


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
