"""LLM helpers: relevance scoring, clip suggestion, analytics digest.
All calls go through Anthropic's API.
"""
from __future__ import annotations

import base64
import json
import re

from anthropic import Anthropic

from . import spend
from .config import env, load_brand

_client: Anthropic | None = None

# Generic fallbacks so every prompt stays coherent when brand.yaml is blank.
# Each workspace overrides these on the Brand & audience page; the climate
# workspace's values reproduce the framing that used to be hardcoded here.
_BRAND_DEFAULTS = {
    "topic": "the channel's topic",
    "source_kind": "online video",
    "relevance_rules": "it substantively covers the topic",
    "false_positives": "the topic is only incidental to the video",
    "strong_openings": (
        "action in progress, people doing something concrete, a striking or "
        "scenic shot, movement, a person visibly reacting"
    ),
    "weak_openings": (
        "static graphics, title cards, logos, text-heavy frames"
    ),
    "clip_guidance": "",
}


def _brand() -> dict:
    """Brand context for prompt framing. Missing or blank fields fall back to
    generic phrasing, and this never raises — prompts must work with no
    brand.yaml at all."""
    try:
        raw = load_brand()
    except Exception:
        raw = {}
    out = dict(_BRAND_DEFAULTS)
    for key in _BRAND_DEFAULTS:
        value = str(raw.get(key) or "").strip()
        if value:
            out[key] = value
    # No generic default makes sense for these; empty just omits the context.
    for key in ("mission", "audience", "voice_notes"):
        out[key] = str(raw.get(key) or "").strip()
    return out


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


def _json_chat(model: str, system: str, user: str, max_tokens: int = 1000,
               temperature: float = 0.2) -> dict:
    system = system + "\nRespond with a single JSON object only — no prose, no code fences."
    return _parse_json(_text_chat(model, system, user, max_tokens=max_tokens,
                                  temperature=temperature))


def score_relevance(model: str, title: str, description: str, matched_keywords: list[str]) -> dict:
    """Return {score: float 0-1}.

    A free-text rationale used to be returned alongside the score; it wasn't
    useful in the operator workflow and isn't a learning signal (numeric score
    + keyword hits + visual traits already drive ranking). Kept accepting an
    optional ``rationale`` key for old cached responses.
    """
    b = _brand()
    system = (
        f"You score {b['source_kind']} videos for genuine relevance to "
        f"{b['topic']}. "
        f"A video is relevant if {b['relevance_rules']}. "
        f"It is NOT relevant if {b['false_positives']}. "
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
    """Recommend ONE programming category — and a shelf life — for a video.
    ``categories`` is the auto-taggable vocabulary ({slug, label, description})
    — the caller passes ``categories.auto_tag_options()``, which excludes
    reserved categories the model cannot infer. The channel aims for a roughly
    equal mix, so this is genre/framing, not topic (every video already
    matched the brand's focus).

    Shelf life rides along in the SAME call (the model already has everything
    it needs, so this costs nothing extra) and feeds the scheduler's urgency
    decay: ``breaking`` content is stale within days, ``timely`` within weeks,
    ``evergreen`` never. It is orthogonal to category — a nature clip about an
    active hurricane is timely; a news explainer is evergreen.

    Returns {category: slug or "", shelf_life: "" | breaking | timely |
    evergreen, rationale}; off-vocabulary answers come back as "" so the video
    stays untagged rather than mislabeled.
    """
    vocab = "\n".join(
        f"- {c['slug']}: {c['label']} — {c['description']}" for c in categories
    )
    b = _brand()
    system = (
        f"You assign a programming category to a video for a social channel "
        f"focused on {b['topic']}. Every video is already relevant to that "
        "focus; the category captures the GENRE and framing of the footage, "
        "not the topic. Pick exactly one slug from:\n" + vocab + "\n"
        "Also judge the content's SHELF LIFE — how fast it stops being worth "
        "posting, independent of its category. Shelf life is about being "
        "PEGGED TO A DATED EVENT, not about feeling newsy:\n"
        "- breaking: reports an event of the last day or two (a storm making "
        "landfall, a ruling just issued); visibly stale within days\n"
        "- timely: coverage of a specific ongoing or just-past event — an "
        "active wildfire or flood, a named storm, a specific policy "
        "announcement or official's statement, a vote, a summit; fades once "
        "that event leaves the news cycle\n"
        "- evergreen: EVERYTHING ELSE. Public figures voicing opinions or "
        "advocacy, interviews, profiles, explainers, science, history, "
        "features and trend pieces are evergreen even when they mention "
        "current politics or feel topical — a viewer months from now loses "
        "nothing. When unsure, answer evergreen; a wrongly-timely tag "
        "expires good content, a wrongly-evergreen tag merely posts it "
        "later.\n"
        "Judge from the title, description, channel and transcript excerpt. "
        "JSON shape: {\"category\": \"slug\", \"shelf_life\": "
        "\"breaking|timely|evergreen\", \"rationale\": \"one line\"}"
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
    shelf = str(data.get("shelf_life", "")).strip().lower()
    if shelf not in ("breaking", "timely", "evergreen"):
        shelf = ""
    return {"category": slug, "shelf_life": shelf,
            "rationale": str(data.get("rationale", ""))[:500]}


def _clean_clip_segments(raw, horizon: float, cap: int,
                         opening_hold: float = 0.0,
                         floor: float = 0.0) -> list[dict]:
    """Coerce, clamp, order and merge one proposed clip's windows.

    ``floor`` and ``horizon`` are the hard bounds every window is clamped into
    — 0 and the end of the transcript normally, the section's edges when the
    pass was scoped to one.

    Overlapping windows would play the same audio twice in the supercut, so
    they're merged here rather than left for the operator to discover after an
    export. Caption timings occasionally arrive as strings, hence the coercion.

    ``opening_hold`` enforces a minimum length on the FIRST segment so the
    clip's opening image stays on screen instead of cutting away immediately.
    The prompt asks for this too, but asking is not enforcing — the fix here
    extends the first segment forward (never backward, which would drag in the
    tail of whatever came before) and merges into the next segment if the
    extension would collide with it.
    """
    windows: list[dict] = []
    for item in (raw or []):
        try:
            start = max(floor, float(item["start"]))
            end = min(horizon, float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < 0.5:
            continue
        windows.append({"start": round(start, 2), "end": round(end, 2)})
    windows.sort(key=lambda w: w["start"])
    merged: list[dict] = []
    for w in windows:
        if merged and w["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(dict(w))
    merged = merged[:cap]

    if opening_hold > 0 and merged:
        first = merged[0]
        if first["end"] - first["start"] < opening_hold:
            first["end"] = round(min(first["start"] + opening_hold, horizon), 2)
            while len(merged) > 1 and merged[1]["start"] <= first["end"]:
                first["end"] = round(max(first["end"], merged[1]["end"]), 2)
                merged.pop(1)
    return merged


def _words_to_clauses(words: list[dict], max_words: int = 12,
                      max_gap: float = 0.8) -> list[dict]:
    """Compact a Whisper word stream into clause-sized ``{start, end, text}``.

    Each entry starts and ends exactly on a word boundary, so a clip cut at an
    entry's edge never clips a word in half — the precision the segment-level
    transcript (YouTube captions especially) can't offer. Splits at sentence
    punctuation, speech gaps, or ``max_words``, mirroring how the burned-in
    captions group the same stream.
    """
    clauses: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            clauses.append({
                "start": round(float(cur[0]["start"]), 1),
                "end": round(float(cur[-1]["end"]), 1),
                "text": " ".join(w["word"] for w in cur),
            })

    for w in words:
        try:
            start, end = float(w["start"]), float(w["end"])
            text = str(w.get("word", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not text:
            continue
        if cur and (
            len(cur) >= max_words
            or start - float(cur[-1]["end"]) > max_gap
            or cur[-1]["word"][-1] in ".?!"
        ):
            flush()
            cur = []
        cur.append({"word": text, "start": start, "end": end})
    flush()
    return clauses


def suggest_clips(model: str, title: str, transcript_segments: list[dict],
                  max_clips: int = 3, max_segments_per_clip: int = 4,
                  min_seconds: int = 15, max_seconds: int = 40,
                  opening_hold: float = 3.0,
                  used_ranges: list[dict] | None = None,
                  words: list[dict] | None = None,
                  description: str = "",
                  window: dict | None = None, topic: str = "") -> list[dict]:
    """Propose the clips worth cutting from one video. DRAFTS ONLY.

    The output is a partition, not a window, because clipping happens on two
    levels that the model has to keep apart:

    - Several SEGMENTS in one clip: the same story, compressed. Most clips are
      cut this way — the filler comes out and the vivid beats are joined.
    - Several CLIPS: separate stories that can't share a caption, each its own
      post and its own ``Cut`` row.

    ``used_ranges`` are windows already claimed by other clips from this video.
    They're shown to the model so it spends its proposals on fresh material;
    the caller still subtracts them afterwards, because a proposal made at
    archive time goes stale as soon as the next clip is cut.

    ``words`` is the full-video Whisper word stream (``[{word, start, end}]``).
    When present it replaces ``transcript_segments`` as the model's timeline:
    the words are regrouped into clause-sized entries whose timestamps sit
    exactly on word boundaries, so proposed cuts land between words instead of
    somewhere inside a multi-second caption block.

    ``window`` scopes the whole pass to one stretch of the video — a chapter,
    or whatever the operator has zoomed to. The transcript is cut down to that
    stretch and every boundary is clamped to it, so a section pass cannot
    wander into footage the operator wasn't looking at. Finding the clip inside
    a section someone already chose is a much smaller question than finding the
    clips in an hour of footage. ``topic`` is what that stretch is about, when
    something already knows — a chapter label.

    Returns ``[{segments, story, why, confidence, draft_caption}]`` where each
    ``segments`` list is already clamped to the transcript, chronological, and
    non-overlapping — i.e. droppable straight into ``Cut.trim_segments``. An
    empty list means the model found nothing worth clipping.
    """
    word_accurate = bool(words)
    if words:
        compact = _words_to_clauses(words)
    else:
        compact = []
        for s in transcript_segments:
            try:
                compact.append({"start": round(float(s["start"]), 1),
                                "end": round(float(s["end"]), 1),
                                "text": str(s.get("text", ""))[:200]})
            except (KeyError, TypeError, ValueError):
                continue

    floor, ceiling = 0.0, None
    if window:
        floor = max(0.0, float(window["start"]))
        ceiling = float(window["end"])
        compact = [w for w in compact if w["end"] > floor and w["start"] < ceiling]
    # The entry cap goes AFTER the window filter, never before: the first 400
    # clauses of a 57-minute video are its first quarter, and a section past
    # that would arrive here with an empty transcript and propose nothing.
    compact = compact[:400]
    if not compact:
        return []
    horizon = max(w["end"] for w in compact)
    if ceiling is not None:
        horizon = min(horizon, ceiling)
        if horizon - floor < min_seconds:
            return []   # nothing here long enough to be a clip

    used = [{"start": round(float(r["start"]), 1), "end": round(float(r["end"]), 1)}
            for r in (used_ranges or [])]

    b = _brand()
    system = (
        f"You cut short social clips out of {b['source_kind']} footage about "
        f"{b['topic']}. "
        "Given a timestamped transcript, propose the clips worth making.\n"
        "\n"
        + (f"You are looking at ONE SECTION of a longer video, "
           f"{floor:.0f}s to {horizon:.0f}s"
           + (f" — {topic}" if topic else "") + ". Every segment you propose "
           "must fall inside it. The rest of the video is not yours to cut, "
           "and it is being handled separately.\n\n" if window else "")
        + "Two levels, and they are NOT the same thing:\n"
        "- SEGMENTS within one clip: one story, compressed. Cut the set-up, "
        "filler, repetition and dead air; keep the vivid, concrete, "
        "human beats. Most good clips are 2-4 segments joined together, not "
        "one continuous take.\n"
        "- Separate CLIPS: separate stories that could not share a single "
        "caption. Only split when the video genuinely covers more than one "
        "story. Most videos yield exactly one clip.\n"
        "\n"
        + (f"EDITORIAL GUIDANCE for this channel: {b['clip_guidance']}\n\n"
           if b["clip_guidance"] else "")
        + "THE OPENING SHOT decides whether anyone watches the rest. Start on "
        "something worth looking at, and hold it: the first segment must run "
        f"at least {opening_hold:g} seconds without a cut. You can't see the "
        "footage, so infer it from what is being said. Strong openings: "
        f"{b['strong_openings']}. "
        f"Do NOT open on: {b['weak_openings']}.\n"
        "\n"
        f"Each clip totals {min_seconds}-{max_seconds} seconds across at most "
        f"{max_segments_per_clip} segments. Segments must be chronological and "
        f"must not overlap. Propose at most {max_clips} clips, best first, and "
        "return an empty list if nothing here is worth clipping.\n"
        + ("The transcript timestamps are word-accurate: every entry starts "
           "and ends exactly on a word boundary. Cut precisely — start a "
           "segment on the first word you want heard and end it right after "
           "the last one, using the entry timestamps as-is.\n"
           if word_accurate else "")
        + ("Some of this video is ALREADY published in other clips — the "
           "used_ranges below. Do not propose any of that material again, and "
           "do not build a clip that merely straddles it. Find fresh moments "
           "or return fewer clips.\n" if used else "")
        + "Also draft a short caption (under 300 chars) the operator will "
        "rewrite, and rate your own confidence in the clip.\n"
        "JSON shape: {\"clips\": [{\"story\": \"one line\", \"segments\": "
        "[{\"start\": n, \"end\": n}], \"why\": \"one line\", "
        "\"confidence\": 0.0-1.0, \"draft_caption\": \"...\"}]}"
    )
    payload = {"title": title, "segments": compact}
    if window:
        payload["section"] = {"start": round(floor, 1), "end": round(horizon, 1),
                              "about": topic}
    if description:
        payload["source_video_description"] = description[:2000]
    if used:
        payload["used_ranges"] = used
    user = json.dumps(payload)
    data = _json_chat(model, system, user, max_tokens=2000)

    clips: list[dict] = []
    for raw in (data.get("clips") or [])[:max_clips]:
        if not isinstance(raw, dict):
            continue
        segments = _clean_clip_segments(raw.get("segments"), horizon,
                                        max_segments_per_clip, opening_hold,
                                        floor=floor)
        if not segments:
            continue
        # A "clip" of a couple of seconds is a parse artifact, not a proposal.
        if sum(s["end"] - s["start"] for s in segments) < 3.0:
            continue
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        clips.append({
            "segments": segments,
            "story": str(raw.get("story", ""))[:200],
            "why": str(raw.get("why", ""))[:300],
            "confidence": confidence,
            "draft_caption": str(raw.get("draft_caption", ""))[:400],
        })
    return clips


def revise_clip(model: str, title: str, instruction: str,
                current_segments: list[dict], transcript_segments: list[dict],
                words: list[dict] | None = None,
                used_ranges: list[dict] | None = None,
                max_segments: int = 8) -> dict:
    """Apply ONE operator instruction to an existing clip's segments.

    The complement of ``suggest_clips``: that call has to find the story on its
    own, this one is handed the story and told exactly what's wrong with it.
    Constrained editing over a small search space — which is why it gets a
    looser rein than the suggester: no min/max duration and a higher segment
    cap, because the operator's instruction outranks the house rules.

    The segment list is in PLAY order, which the trim editor lets differ from
    time order. Output preserves whatever order the model returns and is only
    coerced/clamped, never sorted or merged — reordering someone's supercut to
    "fix" it would be a worse bug than any overlap.

    Returns ``{"segments": [...], "note": str, "changed": bool}``. When the
    model can't comply, ``segments`` is the input unchanged and ``note`` says
    why. Model/parse failures propagate to the caller.
    """
    word_accurate = bool(words)
    if words:
        compact = _words_to_clauses(words)[:400]
    else:
        compact = []
        for s in transcript_segments[:400]:
            try:
                compact.append({"start": round(float(s["start"]), 1),
                                "end": round(float(s["end"]), 1),
                                "text": str(s.get("text", ""))[:200]})
            except (KeyError, TypeError, ValueError):
                continue
    current = []
    for s in current_segments:
        try:
            current.append({"start": round(float(s["start"]), 2),
                            "end": round(float(s["end"]), 2)})
        except (KeyError, TypeError, ValueError):
            continue
    if not compact or not current:
        return {"segments": current, "note": "No transcript to edit against.",
                "changed": False}
    horizon = max(w["end"] for w in compact)

    used = [{"start": round(float(r["start"]), 1), "end": round(float(r["end"]), 1)}
            for r in (used_ranges or [])]

    b = _brand()
    system = (
        f"You edit a short social clip cut from {b['source_kind']} footage "
        f"about {b['topic']}. The clip is a list of segments — time windows on "
        "the source video, played in LIST order (which may differ from time "
        "order; that ordering is the operator's choice and must be kept).\n"
        "\n"
        "The operator gives you ONE instruction. Apply it and change nothing "
        "else: segments the instruction doesn't touch stay exactly as they "
        "are, to the hundredth of a second. Only add, remove, split, extend, "
        "trim or move what the instruction requires.\n"
        "\n"
        + ("The transcript timestamps are word-accurate: every entry starts "
           "and ends exactly on a word boundary. Cut precisely — start a "
           "segment on the first word you want heard and end it right after "
           "the last one, using the entry timestamps as-is.\n"
           if word_accurate else "")
        + ("Some of this video is already published in other clips — the "
           "used_ranges in the input. Do not pull any of that material in.\n"
           if used else "")
        + "If the instruction can't be done (the material isn't in this video, "
        "or it's already used elsewhere), return the segments UNCHANGED and "
        "say why in the note.\n"
        f"At most {max_segments} segments.\n"
        "JSON shape: {\"segments\": [{\"start\": n, \"end\": n}], "
        "\"note\": \"one line on what you changed, or why you couldn't\"}"
    )
    payload = {"title": title, "instruction": instruction[:500],
               "current_segments": current, "transcript": compact}
    if used:
        payload["used_ranges"] = used
    data = _json_chat(model, system, json.dumps(payload), max_tokens=1500)

    # Coerce and clamp only — no sorting, no merging (see docstring).
    segments: list[dict] = []
    for item in (data.get("segments") or [])[:max_segments]:
        try:
            start = max(0.0, float(item["start"]))
            end = min(float(horizon), float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < 0.5:
            continue
        segments.append({"start": round(start, 2), "end": round(end, 2)})
    note = str(data.get("note", ""))[:300]
    if not segments:
        return {"segments": current, "changed": False,
                "note": note or "The model returned nothing usable; clip left as-is."}
    return {"segments": segments, "note": note, "changed": segments != current}


def _transcript_blocks(segments: list[dict], target: int = 240,
                       char_cap: int = 260) -> list[dict]:
    """Compact a whole transcript into roughly ``target`` even blocks.

    Chapter finding needs the WHOLE timeline, so it can't truncate the way the
    clip suggester does — a 45-minute video would lose its second half and
    every chapter in it. Neighbouring lines are merged into blocks of roughly
    equal length instead, which keeps coverage end to end at a bounded token
    cost. Boundaries then land on block edges, which is fine: a chapter mark a
    few seconds out is still the right chapter.
    """
    clean: list[tuple[float, float, str]] = []
    for s in segments:
        try:
            start, end = float(s["start"]), float(s["end"])
            text = str(s.get("text", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if end >= start and text:
            clean.append((start, end, text))
    if not clean:
        return []
    clean.sort(key=lambda row: row[0])

    span = clean[-1][1] - clean[0][0]
    width = max(1.0, span / max(1, target))
    blocks: list[dict] = []
    for start, end, text in clean:
        if blocks and start - blocks[-1]["start"] < width:
            block = blocks[-1]
            block["end"] = max(block["end"], round(end, 1))
            if len(block["text"]) < char_cap:
                block["text"] = f"{block['text']} {text}"
        else:
            blocks.append({"start": round(start, 1), "end": round(end, 1),
                           "text": text})
    for block in blocks:
        block["text"] = block["text"][:char_cap]
    return blocks


def _chunk_blocks(blocks: list[dict], span: float) -> list[list[dict]]:
    """Split a compacted transcript into windows of at most ``span`` seconds."""
    if not blocks or blocks[-1]["end"] - blocks[0]["start"] <= span:
        return [blocks]
    windows: list[list[dict]] = []
    current: list[dict] = []
    edge = blocks[0]["start"] + span
    for block in blocks:
        if current and block["start"] >= edge:
            windows.append(current)
            current = []
            edge = block["start"] + span
        current.append(block)
    if current:
        windows.append(current)
    return windows


def _chapter_marks(model: str, title: str, blocks: list[dict], *, total: float,
                   max_marks: int, min_seconds: float, section: bool,
                   description: str = "") -> list[dict]:
    """One pass of chapter-start finding over one window of transcript."""
    first, last = blocks[0]["start"], blocks[-1]["end"]
    b = _brand()
    system = (
        f"You index {b['source_kind']} footage so an editor can find their way "
        "around it. Mark where this video changes subject: a new chapter "
        "begins where it moves to a different story, place, or speaker.\n"
        "\n"
        "Report only the START of each chapter, in seconds, in order. Each "
        "chapter runs until the next one begins, so they need no end times.\n"
        + (f"You are reading SECTION {first:.0f}s-{last:.0f}s of a "
           f"{total:.0f}s video. Mark only starts inside this section; the "
           "rest of the video is being indexed separately. Read to the end of "
           "the section — the last minutes of it matter as much as the "
           "first.\n" if section else
           "The first chapter starts at 0, and together they cover the whole "
           "video. Keep marking all the way to the end.\n")
        + f"At most {max_marks} chapters here, none shorter than "
        f"{min_seconds:g} seconds. This is a table of contents, not a shot "
        "list: footage that stays on one subject is one chapter.\n"
        "Label each with a specific topic of a few words — name the place, "
        "person or event ('Peat fires in Kalimantan', not 'Environmental "
        "impacts') — plus one line on what happens in it.\n"
        "JSON shape: {\"chapters\": [{\"start\": n, \"topic\": \"...\", "
        "\"summary\": \"one line\"}]}"
    )
    payload = {"title": title, "transcript": blocks}
    if description:
        payload["source_video_description"] = description[:1000]
    data = _json_chat(model, system, json.dumps(payload), max_tokens=2000)

    marks: list[dict] = []
    for item in (data.get("chapters") or []):
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
        except (KeyError, TypeError, ValueError):
            continue
        # A mark outside the window is the model reaching into a section it
        # wasn't shown; the pass that owns that footage gets to place it.
        if not (first - 0.5 <= start <= last):
            continue
        marks.append({"start": round(max(0.0, start), 2),
                      "topic": str(item.get("topic", "")).strip()[:80],
                      "summary": str(item.get("summary", "")).strip()[:200]})
    return marks


# Past this, one pass stops reading: on a 57-minute documentary the model
# marked eleven chapters in the first half hour and called the remaining
# twenty-three minutes a single subject, while the transcript there ran from a
# film anecdote to forest loss to a palm oil conglomerate. Windows keep every
# pass short enough to finish.
CHAPTER_WINDOW_SECONDS = 900


def segment_chapters(model: str, title: str, transcript_segments: list[dict],
                     max_chapters: int = 12, min_seconds: float = 30.0,
                     description: str = "") -> list[dict]:
    """Divide a video into consecutive topical chapters.

    The deliberately easy half of "what is in this video": finding WHERE the
    subject changes, with none of the judgement that makes clipping hard — no
    frames to pick, no boundaries to land on words, no self-contained story to
    tell. A chapter a few seconds out costs nothing, because the operator cuts
    inside one rather than shipping it.

    STARTS, not spans. Chapters that tile a video end to end are fully
    determined by where each begins, so a model that only reports starts cannot
    return a gap, an overlap, or a chapter running backwards. The ends, the 0.0
    opening and the final end are filled in here.

    Long videos are read in windows (see ``CHAPTER_WINDOW_SECONDS``) and the
    marks concatenated — which is safe precisely because marks are just points
    on a shared timeline, and the tiling is built here rather than by the
    model. Short videos still cost one call.

    Returns ``[{start, end, topic, summary}]``, chronological and contiguous.
    Returns [] when the video is too short to have chapters, or turns out to
    hold only one subject — a single band is furniture, not navigation.
    """
    blocks = _transcript_blocks(transcript_segments)
    if not blocks:
        return []
    horizon = max(b["end"] for b in blocks)
    if horizon < 2 * min_seconds:
        return []   # can't hold two chapters, so it doesn't have any

    windows = _chunk_blocks(blocks, CHAPTER_WINDOW_SECONDS)
    marks: list[dict] = []
    for window in windows:
        span = window[-1]["end"] - window[0]["start"]
        # A window's share of the budget, plus one so a busy stretch isn't
        # forced to under-report. The global cap below trims any excess by
        # merging the shortest chapters, which is the right thing to lose.
        share = (max_chapters * span / horizon) if horizon > 0 else max_chapters
        marks.extend(_chapter_marks(
            model, title, window, total=horizon,
            max_marks=max(2, round(share) + 1), min_seconds=min_seconds,
            section=len(windows) > 1, description=description,
        ))
    marks = [m for m in marks if m["start"] <= horizon]
    if not marks:
        return []
    marks.sort(key=lambda ch: ch["start"])
    marks[0]["start"] = 0.0

    chapters: list[dict] = []
    for i, mark in enumerate(marks):
        end = marks[i + 1]["start"] if i + 1 < len(marks) else horizon
        if end <= mark["start"]:
            continue   # two marks on the same second: the later label wins
        chapters.append({**mark, "end": round(end, 2)})

    def length(ch: dict) -> float:
        return ch["end"] - ch["start"]

    def absorb(index: int) -> None:
        """Fold one chapter into its neighbour, keeping the tiling intact."""
        gone = chapters.pop(index)
        if index > 0:
            chapters[index - 1]["end"] = gone["end"]
        else:
            chapters[0]["start"] = gone["start"]

    shrinking = True
    while shrinking and len(chapters) > 1:
        shrinking = False
        for i, ch in enumerate(chapters):
            if length(ch) < min_seconds:
                absorb(i)
                shrinking = True
                break
    while len(chapters) > max_chapters:
        absorb(min(range(len(chapters)), key=lambda i: length(chapters[i])))

    return chapters if len(chapters) > 1 else []


def rank_striking_frames(model: str, frames: list[tuple[float, bytes]], *,
                         title: str = "", topic: str = "", top: int = 6) -> list[dict]:
    """Out of a scan of the footage, the frames worth opening a clip on.

    The third and widest of the three ways this app looks at a video, and the
    only one that can answer "where are the images in here":

    - ``suggest_clips`` reads words and cannot see at all.
    - ``pick_opening_frame`` can see, but only a few seconds either side of a
      start the transcript already chose.
    - this scans a whole stretch and judges the pictures on their own terms.

    Judging the IMAGE, not the story, is the point. The most arresting shot in
    a video is rarely where its words begin, and an operator who can see it
    listed can open on it and cut to the talking afterwards.

    ``frames`` are ``(timestamp, jpeg)`` in order, each sent as its own
    labelled image rather than tiled into a sheet — a model asked which cell
    of a grid it picked is guessing (see ``vision.frames_between``).

    Returns ``[{"index": int, "why": str}]``, strongest first, at most ``top``.
    """
    if len(frames) < 2:
        return []
    b = _brand()
    system = (
        f"You are choosing the opening image for a short social video cut from "
        f"{b['source_kind']} footage about {b['topic']}. The first frame "
        "decides whether anyone stops scrolling.\n"
        "\n"
        "These frames are a scan of one stretch of a longer video, in order. "
        f"Pick the {top} worth opening on, strongest first.\n"
        "\n"
        f"STRONG openings: {b['strong_openings']}.\n"
        f"WEAK openings: {b['weak_openings']}, frames caught mid-transition or "
        "mid-dissolve, motion blur, black or near-black frames.\n"
        "\n"
        "Judge the PICTURE, not the story. It does not matter whether the "
        "clip could sensibly begin there — the editor decides that, and can "
        "open on an image and cut to the talking straight after.\n"
        "Never pick two frames from the same shot: near-identical images are "
        "one choice, not two. Spread your picks across different moments.\n"
        "If fewer than that many frames are worth anything, return fewer. If "
        "none are, return an empty list.\n"
        "Say in one line what is in the frame that earns it — name what is on "
        "screen, not an adjective.\n"
        "JSON shape: {\"frames\": [{\"index\": n, \"why\": \"one line\"}]}"
    )

    blocks: list = [{
        "type": "text",
        "text": (f"Video: {title}\n" if title else "")
        + (f"This stretch is about: {topic}\n" if topic else "")
        + f"{len(frames)} frames follow, each labelled with its index.",
    }]
    for i, (ts, image) in enumerate(frames):
        blocks.append({"type": "text", "text": f"Frame {i} (t={ts:.1f}s)"})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(image).decode("ascii"),
            },
        })

    resp = _create(model, system + "\nRespond with a single JSON object only — "
                                   "no prose, no code fences.",
                   blocks, max_tokens=800, temperature=0.2)
    data = _parse_json(_text_from(resp))

    out: list[dict] = []
    seen: set[int] = set()
    for item in (data.get("frames") or []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= index < len(frames)) or index in seen:
            continue
        seen.add(index)
        out.append({"index": index, "why": str(item.get("why", "")).strip()[:200]})
        if len(out) >= top:
            break
    return out


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


def pick_opening_frame(model: str, frames: list[tuple[float, bytes]],
                       latest_index: int, hold_seconds: float = 3.0,
                       title: str = "", story: str = "") -> dict:
    """Choose which sampled frame makes the strongest opening image.

    The transcript pass picks a start from the words alone, which is how clips
    end up opening on an anchor at a desk — the sentence is right and the frame
    is dead. This looks at the actual footage around that start and moves it.

    ``frames`` are ``(timestamp, jpeg)`` in chronological order. Only frames at
    or before ``latest_index`` may be chosen: earlier means adding lead-in
    footage ahead of the speech, while later would cut into the first sentence.
    The frames after it are still sent, because the model needs them to check
    that the shot HOLDS rather than cutting away half a second later.

    Returns ``{"index": int|None, "why": str}``; None means leave the start
    alone.
    """
    if not frames or latest_index < 0:
        return {"index": None, "why": ""}

    interval = round(frames[1][0] - frames[0][0], 2) if len(frames) > 1 else 0.5
    b = _brand()
    system = (
        f"You choose the opening frame of a short social video cut from "
        f"{b['source_kind']} footage. The first image decides whether anyone "
        "watches the rest.\n"
        f"The frames are consecutive stills, in order, {interval:g}s apart. "
        f"Frame {latest_index} is where the clip currently starts — where the "
        "speech begins.\n"
        f"Choose a frame at or BEFORE frame {latest_index}. Choosing an earlier "
        "one opens the clip on footage that runs before the talking starts; "
        "later frames are shown only so you can see what happens next, and must "
        "never be chosen.\n"
        "\n"
        f"STRONG openings: {b['strong_openings']}.\n"
        f"WEAK openings: {b['weak_openings']}, a frame caught mid-transition "
        "or mid-dissolve, motion blur, black frames.\n"
        "\n"
        f"The image must HOLD: the shot you pick should still be the same shot "
        f"about {hold_seconds:g}s later. Use the frames that follow yours to "
        "check. If it cuts away almost immediately, pick a different frame "
        "even if that one image is prettier.\n"
        "\n"
        f"If nothing beats frame {latest_index} itself, answer {latest_index}. "
        "If the frames are unusable, answer null.\n"
        "JSON shape: {\"index\": n or null, \"why\": \"one line\"}"
    )

    blocks: list = [{
        "type": "text",
        "text": (f"Story: {story}\n" if story else "")
        + (f"Video: {title}\n" if title else "")
        + "Frames follow, each labelled with its index.",
    }]
    for i, (ts, image) in enumerate(frames):
        marker = " <- clip currently starts here" if i == latest_index else ""
        suffix = " (context only, not selectable)" if i > latest_index else ""
        blocks.append({"type": "text",
                       "text": f"Frame {i} (t={ts:.1f}s){marker}{suffix}"})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(image).decode("ascii"),
            },
        })

    resp = _create(model, system + "\nRespond with a single JSON object only — "
                                   "no prose, no code fences.",
                   blocks, max_tokens=300, temperature=0.2)
    data = _parse_json(_text_from(resp))
    raw = data.get("index")
    why = str(data.get("why", ""))[:300]
    if raw is None:
        return {"index": None, "why": why}
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return {"index": None, "why": why}
    # A model that answers past the cutoff has ignored the one hard rule here;
    # trusting it would truncate the opening sentence.
    if idx < 0 or idx > latest_index:
        return {"index": None, "why": why}
    return {"index": idx, "why": why}


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
                         target_words: int | None = None,
                         description: str = "") -> str:
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
    b = _brand()
    system = (
        f"You draft a Threads caption for a short {b['source_kind']} clip "
        f"about {b['topic']}. The operator will edit it before posting.\n\n"
    )
    if b["mission"] or b["audience"]:
        context_bits = []
        if b["mission"]:
            context_bits.append(f"mission — {b['mission']}")
        if b["audience"]:
            context_bits.append(f"audience — {b['audience']}")
        system += "Channel context: " + "; ".join(context_bits) + ".\n\n"
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
        "Also hard: do not invent facts not in the excerpt or description. "
        "When a source_video_description is provided it may name the speaker, "
        "publisher, or topic — use it for context but the transcript is still "
        "the primary authority on what happened. "
    )
    # A mandatory place name can consume most of a very short caption, so it
    # becomes a preference once the target is tight. Only asked for at all
    # when the source actually has a place (news stations do; a fitness
    # creator's channel doesn't).
    if station or market:
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
    payload: dict = {
        "video_title": title,
        "station": station,
        "market": market,
        "clip_length_seconds": clip_seconds,
        "transcript_excerpt_of_clip": excerpt[:3000],
    }
    if description:
        payload["source_video_description"] = description[:2000]
    user = json.dumps(payload)
    data = _json_chat(model, system, user, max_tokens=600)
    return _tidy_caption(str(data.get("caption", "")))


def suggest_hook_text(model: str, title: str, station: str, market: str,
                      excerpt: str, examples: list[str] | None = None,
                      description: str = "") -> str:
    """Draft short on-video hook text for an Instagram Reel vertical composite.

    Rendered large in the brand font at the top of the 9:16 frame — so it must
    stay brief. DRAFT ONLY: the operator edits before regenerating the reel.

    ``examples`` are hooks the operator rewrote for themselves, taken from the
    draft ledger (``app/draft_proposals.operator_written``). The hook has no
    other voice source: unlike captions there is no published history to learn
    from, because the hook is burned into the video rather than posted as text.
    """
    b = _brand()
    system = (
        "You write a short HOOK line that appears as large on-screen text at the "
        f"top of an Instagram Reel (a {b['source_kind']} clip about {b['topic']}). "
        "Hard rules:\n"
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
    payload: dict = {
        "video_title": title,
        "station": station,
        "market": market,
        "transcript_excerpt_of_clip": excerpt[:3000],
    }
    if description:
        payload["source_video_description"] = description[:2000]
    user = json.dumps(payload)
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
    b = _brand()
    system = (
        "You write a formal source citation to be posted as the first comment "
        f"under a short {b['source_kind']} clip on Threads, crediting the "
        "original publisher. "
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


def suggest_first_reply(model: str, instruction: str, *, video_title: str = "",
                        description: str = "", transcript: str = "", caption: str = "",
                        recent_replies: list[str] | None = None) -> str:
    """Draft the call-to-action first comment for a post, following the
    operator's own plain-language ``instruction`` from first_reply.yaml.

    DRAFT ONLY: the text lands in the editable first-reply box and posts only
    once the operator has left it there. ``instruction`` is authoritative and is
    also the only permitted source of factual claims — the model is told not to
    invent figures, because this copy pitches a real investment product.

    ``recent_replies`` are the last replies actually posted; the model is told
    to avoid reusing their phrasing so the same boilerplate doesn't ride under
    every post. Returns "" when the model declines.
    """
    if not (instruction or "").strip():
        return ""
    b = _brand()
    system = (
        "You write the first comment posted under a short "
        f"{b['source_kind']} clip on Threads. The comment invites viewers to "
        "act, and its whole job is to connect what they just watched to that "
        "invitation.\n\n"
        "The operator's brief below is authoritative. Follow it exactly:\n"
        "--- BRIEF ---\n"
        f"{instruction.strip()}\n"
        "--- END BRIEF ---\n\n"
        "Hard rules:\n"
        "- The opening sentence is a short question tying the invitation to THIS "
        "specific clip — its subject, place, or stakes. Make it concrete: name "
        "what is actually on screen, not a generic 'want to help the planet?'\n"
        "- Every factual claim (figures, percentages, minimums, how the product "
        "works) must appear in the brief. NEVER invent, round, embellish, or "
        "infer a statistic, return, or guarantee. Nothing may read as a promise "
        "of financial return beyond what the brief states.\n"
        "- Vary the wording from the recent replies you are given: different "
        "opening, different sentence shape, different verbs. Never reuse a "
        "previous opening question.\n"
        "- Keep the question hook for environmental damage, wildlife decline, and "
        "species loss — that is the normal subject matter here. Drop it ONLY when "
        "PEOPLE have been killed or injured, or a disaster is actively unfolding; "
        "there, open with a plain, non-glib sentence instead so the reply never "
        "sounds like it is marketing off human tragedy.\n"
        "- No hashtags, no emojis, no @handles. Under 400 characters total.\n"
        "- If the brief is too thin to write from, return "
        "{\"available\": false, \"reply\": \"\"}.\n"
        "JSON shape: {\"available\": true|false, \"reply\": \"...\"}"
    )
    payload = {
        "clip_caption": (caption or "")[:1000],
        "video_title": (video_title or "")[:500],
        "video_description": (description or "")[:2000],
        "clip_transcript": (transcript or "")[:8000],
        "recent_replies_to_avoid_echoing": [r[:300] for r in (recent_replies or [])[:12]],
    }
    # Runs hotter than the rest of the prompts on purpose: this copy goes under
    # every single post, so near-deterministic sampling made every reply open
    # the same two or three ways. The flip side is the occasional empty or
    # non-JSON completion, so a blank draft gets one more roll before giving up
    # rather than surfacing as a failure on the post page.
    user = json.dumps(payload)
    text = ""
    for _ in range(2):
        try:
            data = _json_chat(model, system, user, max_tokens=1000, temperature=0.9)
        except ValueError:
            continue
        text = str(data.get("reply", "")).strip()
        if not data.get("available", bool(text)):
            return ""
        if text:
            break
    if not text:
        return ""
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
    b = _brand()
    system = (
        f"You are an editorial coach for someone who posts short "
        f"{b['source_kind']} clips about {b['topic']} on Threads. "
        "You are shown captions they published; when "
        "available they're split into higher- and lower-performing sets. Infer a "
        "short list of CONCRETE, REUSABLE composition rules that capture what makes "
        "the strong captions work — structural and editorial patterns to apply to "
        "every future caption.\n\n"
        "Focus on FORMAT and FRAMING: how to open, how to close, how to use quotes "
        "or stats, how to frame contested viewpoints, rhythm, and what to "
        "avoid. Each rule must be ONE imperative instruction, specific and "
        "actionable. Good examples of the style and specificity wanted:\n"
        "- Lead with a one-line pull quote from the transcript.\n"
        "- End with a short, wry question.\n"
        "- Frame contested claims impartially, without editorializing.\n\n"
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
    """Generate a concise, human-readable title for a trimmed clip.

    Draws on the clip's own transcript excerpt (the trimmed windows) plus the
    original source title/description and, optionally, the draft caption. Returns
    a single punchy plain-text title (no surrounding quotes), roughly <= 70 chars.
    """
    b = _brand()
    system = (
        f"You write a short, punchy title for a {b['source_kind']} clip about "
        f"{b['topic']} that has "
        "been trimmed to its strongest moment. Base it on what the clip actually "
        "says (the transcript excerpt), using the source title only for context. "
        "Rules: one line, plain text, no surrounding quotes, no emojis, no hashtags, "
        "at most ~70 characters, concrete and faithful to the clip — do not invent "
        "facts. Prefer the specific subject and the striking detail over vague "
        "phrasing. "
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
    b = _brand()
    system = (
        f"You write a very short title for a {b['source_kind']} clip, distilled from its "
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
    b = _brand()
    system = (
        f"You write a very short title for a {b['source_kind']} clip, based on what is "
        "actually said in its transcript. Rules: 2 to 5 words, plain text, no "
        "surrounding quotes, no emojis, no hashtags, no trailing punctuation, "
        "title case. Name the specific subject the transcript "
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
    b = _brand()
    system = (
        "You are a careful social media analyst writing a performance digest for a "
        f"single-operator Threads account posting short {b['topic']} clips. Using ONLY the "
        "provided data: report top and bottom performers per metric; surface patterns "
        "across attribute slices (keywords, region, clip length, caption traits, day/time, "
        "and visual/footage traits); "
        "state hypotheses for WHY, clearly labeled as hypotheses, never as proven cause; "
        "label all patterns as correlational. "
        f"If total posts < {min_sample_size}, lead with a prominent small-sample caveat "
        "and avoid claiming any pattern. End with 2-3 lightweight experiment suggestions "
        "for upcoming posts, framed as tests, not guarantees. Write concise markdown."
    )
    return _text_chat(model, system, json.dumps(stats_payload), max_tokens=3000, temperature=0.4)
