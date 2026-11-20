"""Onboarding: the first thing a new account sees, and what fills its settings.

A new workspace starts empty (``run.py workspace create`` writes blank config).
Onboarding asks a handful of friendly questions, one per screen: the brand's
name and website, what it does, who its customers are, what the account should
post, and the keywords to watch for. Answers save to the same settings the
Settings pages edit.

The helper does the heavy lifting. After the first screen it reads the website
(and any upload) in the background and drafts the remaining answers, so most
screens arrive filled in, marked as suggestions. A chat with it is a tap away
on every screen. When the last screen is saved it writes the behind-the-scenes
rules the prompts need (where the footage comes from, on/off-topic notes, good
and weak openings) from the answers, so nobody has to fill those in.

State (status, finished screens, drafts, the conversation, background work) is
one shared document per workspace (app/shared_docs.py, key ``onboarding``).

A workspace without an onboarding record counts as onboarded when its brand
already names a topic. Existing accounts never get sent through onboarding, and
no account needs a flag to make that true.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import logging
from typing import Any

from . import spend
from .config import (load_brand, load_keywords, load_settings, save_brand, save_keywords,
                     with_american_english)

log = logging.getLogger("onboarding")

DOC_KEY = "onboarding"
DEFAULT_MODEL = "claude-opus-5-5"

# --- Steps -------------------------------------------------------------------
# One question per screen, one to three answers each. Only what a person can
# answer off the top of their head is asked; the prompt-tuning fields (where the
# footage comes from, on/off-topic rules, good and weak openings) are written by
# the helper from these answers when onboarding finishes (``derive_format``).
#
# A field: name, kind ("line", "text" or "chips"), label (may be blank when the
# question says it all), placeholder, required.

def _f(name, kind, label="", placeholder="", required=False, rows=3):
    return {"name": name, "kind": kind, "label": label, "placeholder": placeholder,
            "required": required, "rows": rows}


STEPS: list[dict[str, Any]] = [
    {"key": "brand", "title": "Your brand",
     "question": "Hi there! What's your brand called?",
     "sub": "Add your website too and we'll read it to get a head start.",
     "upload": True,
     "fields": [_f("logo", "avatar", "Your icon"),
                _f("name", "line", "Brand name", "e.g. Core & Flow", True),
                _f("cta_url", "line", "Website", "e.g. coreandflow.com")]},
    {"key": "about", "title": "What you do",
     "question": "What does {name} do?",
     "sub": "A sentence or two, the way you'd explain it to a friend.",
     "fields": [_f("mission", "text", "", "e.g. We're a pilates studio in Austin that makes "
                                         "reformer classes feel friendly, not intimidating.")]},
    {"key": "customers", "title": "Your customers",
     "question": "Who are {name}'s customers?",
     "sub": "The people you're here for. Every caption is written with them in mind.",
     "fields": [_f("audience", "text", "", "e.g. Busy professionals in their 30s and 40s, "
                                          "mostly new to pilates.")]},
    {"key": "throughline", "title": "Your throughline",
     "question": "What's your throughline?",
     "sub": "One idea that ties every clip together, whatever the footage. "
            "It's what people will come to know you for.",
     "teach": True,
     "fields": [_f("throughline", "throughline", "",
                   "e.g. Every conflict has a moment where it could have gone differently.",
                   True, rows=2),
                _f("format", "lenses", "Clips that carry it", "Add your own"),
                _f("off_limits", "text", "Anything off limits?",
                   "e.g. no politics, no body-shaming, nothing graphic", rows=2),
                _f("caption_brief", "brief", "Give your captions more to work with",
                   "e.g. How our app would respond to the conflict in a clip: it names what "
                   "each person needs, reflects it back without blame, and suggests one "
                   "next step both could agree to.", rows=6)]},
    {"key": "keywords", "title": "Keywords",
     "question": "Here's what we'll watch for",
     "sub": "New videos that mention any of these get a closer look. "
            "Tap one to remove it, or add your own.",
     "fields": [_f("keywords", "chips", "", "Add a keyword", True)]},
    {"key": "subtitles", "title": "Subtitles",
     "question": "Pick your subtitle style",
     "sub": "Words appear on your clips as people speak, and the word being said "
            "lights up in your color.",
     "illustration": "subtitles",
     "fields": [_f("font", "fonts", "Font"),
                _f("color", "colors", "Highlight color"),
                _f("tab_text", "tab", "Corner tab", "e.g. yourwebsite.com or @yourhandle")]},
    {"key": "endcard", "title": "End card",
     "question": "Write your end card",
     "sub": "Every clip ends on a short card with your message and your website, "
            "so people know where to find you.",
     "illustration": "endcard",
     "fields": [_f("outro", "text", "Up to three short lines",
                   "e.g. Pilates for every body\nClasses from the best teachers\nTry a class free",
                   rows=3),
                _f("end_color", "colors", "Card color")]},
    {"key": "schedule", "title": "Your schedule",
     "question": "When should your posts go out?",
     "sub": "We spread your posts through the day. You approve each one first, and "
            "can move any of them on the calendar.",
     "fields": [_f("tz", "timezone", "Your time zone", required=True),
                _f("rhythm", "rhythm", "", required=True)]},
    {"key": "sources", "title": "Video sources",
     "question": "Here's where we'll look",
     "sub": "YouTube channels whose recent videos carry your throughline. "
            "Here's a cross-section; drop any you don't want.",
     "fields": [_f("sources", "sources", "")]},
    {"key": "connect", "title": "Connect",
     "question": "Connect your accounts",
     "sub": "Where your clips get posted. Nothing posts until you approve it.",
     "fields": [_f("connect", "connect", "")]},
    {"key": "launch", "title": "Launch",
     "question": "Start with a video of your own?",
     "sub": "Got a talk, an interview or a webinar? Drop it in and we'll cut it "
            "into clips right away. They'll be waiting in Today's posts for you to "
            "approve. Optional: your channels will start sending videos either way.",
     "fields": [_f("upload", "upload", "")]},
]
STEP_KEYS = [s["key"] for s in STEPS]

# What's still set on its own page after onboarding. Shown on the last screen
# until each gets a step of its own.
# Nothing is left for later pages any more: every step is in the flow.
LATER_STEPS: list[tuple[str, str]] = []

# Fields the customer answers, and the ones the helper writes at the end.
# Screens whose answers aren't text the helper writes (they save to their own
# settings: the caption look, the posting plan, the channels, the connections).
_WIDGET_KINDS = ("fonts", "timezone", "rhythm", "sources", "connect", "upload", "avatar")
ASKED_FIELDS = tuple(f["name"] for s in STEPS for f in s["fields"]
                     if f["kind"] not in _WIDGET_KINDS)
DERIVED_FIELDS = ("topic", "positioning", "source_kind", "relevance_rules", "false_positives",
                  "strong_openings", "weak_openings")
# Answers kept only for the helper (not settings of their own): "off limits"
# becomes part of the positioning and the off-topic notes.
STATE_ONLY_FIELDS = ("off_limits",)
# Answers that save to their own settings rather than the brand.
LOOK_FIELDS = ("color", "outro", "end_color", "tab_text")
BRAND_FIELDS = tuple(f for f in ASKED_FIELDS
                     if f != "keywords" and f not in STATE_ONLY_FIELDS + LOOK_FIELDS) + DERIVED_FIELDS
ALL_FIELDS = ASKED_FIELDS


def step(key: str) -> dict[str, Any] | None:
    return next((s for s in STEPS if s["key"] == key), None)


def step_of(field: str) -> str | None:
    return next((s["key"] for s in STEPS if any(f["name"] == field for f in s["fields"])), None)


def next_step(key: str) -> str | None:
    i = STEP_KEYS.index(key)
    return STEP_KEYS[i + 1] if i + 1 < len(STEP_KEYS) else None


def prev_step(key: str) -> str | None:
    i = STEP_KEYS.index(key)
    return STEP_KEYS[i - 1] if i > 0 else None


# Answers for trying the flow quickly (admins only: "Fill sample" on each screen).
SAMPLE_ANSWERS = {
    "name": "Core & Flow",
    "cta_url": "coreandflow.example",
    "mission": "We're a pilates studio in Austin that makes reformer classes feel friendly, "
               "not intimidating.",
    "audience": "Busy professionals in their 30s and 40s, mostly new to pilates, who want "
                "to feel strong without a gym.",
    "format": "📰 Pilates in the news — a new study on pilates and back pain, explained\n"
              "🎞️ From the archive — Joseph Pilates teaching on a 1930s rooftop\n"
              "🎯 One move, explained — a great teacher breaking down the Hundred",
    "throughline": "Strength is something you learn, not something you're born with.",
    "color": "#7B5CD6",
    "end_color": "#3D2B78",
    "tab_text": "coreandflow.example",
    "outro": "Learn the why behind every move\nClasses for every body\nTry a class free",
    "caption_brief": "We teach the why behind every move: name the muscle that's working, "
                     "the cue that makes it click, and one way to try it at home.",
    "off_limits": "body shaming, weight-loss promises, medical advice",
    "keywords": "pilates\nreformer pilates\nmat pilates\njoseph pilates\npilates for beginners\n"
                "pilates study\ncore strength\nback pain exercise",
}


def skip_step(key: str) -> dict[str, Any]:
    """Mark a screen done without saving anything (admins testing the flow)."""
    state = load()
    if key not in state["steps_done"]:
        state["steps_done"].append(key)
    save(state)
    return state


# --- State -------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _blank() -> dict[str, Any]:
    return {"status": "pending", "steps_done": [], "drafts": {}, "messages": [],
            "transcript": []}


def load() -> dict[str, Any]:
    """The workspace's onboarding state. Without a record, an account whose
    brand already names a topic was set up before onboarding existed: done."""
    from .shared_docs import read_text

    text = read_text(DOC_KEY)
    data: dict[str, Any] | None = None
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            log.warning("Onboarding state unreadable; starting over")
    if not isinstance(data, dict):
        data = _blank()
        if load_brand().get("topic"):
            data["status"] = "done"
            data["derived"] = True
    for k, v in _blank().items():
        data.setdefault(k, v)
    return data


def save(state: dict[str, Any]) -> None:
    from .shared_docs import write_text

    state.pop("derived", None)
    state["updated_at"] = _now()
    write_text(DOC_KEY, json.dumps(state, ensure_ascii=False))


def is_done() -> bool:
    try:
        return load().get("status") == "done"
    except Exception:
        # Never lock someone out of their app because this record can't be read.
        log.exception("Onboarding state unavailable; treating as done")
        return True


def current_step(state: dict[str, Any]) -> str:
    """The first step not yet finished (the last step once all are)."""
    done = set(state.get("steps_done") or [])
    return next((k for k in STEP_KEYS if k not in done), STEP_KEYS[-1])


def reset() -> None:
    """Start over: an admin's action. Saved settings stay; the helper forgets."""
    save(_blank())


def mark_done(state: dict[str, Any] | None = None) -> None:
    state = state or load()
    state["status"] = "done"
    state["finished_at"] = _now()
    save(state)


# --- Form values -------------------------------------------------------------

def saved_values() -> dict[str, str]:
    values = {f: v for f, v in load_brand().items() if f in BRAND_FIELDS}
    values["keywords"] = "\n".join(load_keywords())
    values.update(_settings_values())
    return values


def _settings_values() -> dict[str, str]:
    """What the later screens show: the account's saved look, time zone and
    posting rhythm. Never raises; an unreadable setting just shows blank."""
    out = {"font": "", "color": "", "outro": "", "end_color": "", "tab_text": "", "tab_on": "1",
           "tz": "", "rhythm": ""}
    try:
        from . import caption_look, outro_card

        look = caption_look.load()
        out["font"], out["color"] = look["font"], look["box"]
        card = outro_card.load() or {}
        out["outro"] = "\n".join(l for l in card.get("lines") or [] if l)
        out["end_color"] = card.get("color") or ""
        if card:
            tab = outro_card.tab_settings(card)
            out["tab_text"] = tab["text"] if tab["custom_text"] else ""
            out["tab_on"] = "1" if tab["on"] else ""
    except Exception:
        log.exception("Couldn't read the look")
    try:
        from .config import scheduler_timezone, timezone_saved

        out["tz"] = str(scheduler_timezone()) if timezone_saved() else ""
    except Exception:
        log.exception("Couldn't read the time zone")
    try:
        from . import posting_plan
        from .db import session_scope

        with session_scope(read_only=True) as session:
            out["rhythm"] = posting_plan.load_mode(session) or ""
    except Exception:
        log.exception("Couldn't read the posting plan")
    return out


def _save_settings_step(key: str, values: dict[str, str]) -> None:
    """The screens that save to their own settings rather than the brand."""
    if key == "subtitles":
        from . import caption_look, outro_card

        color = values.get("color", "").strip()
        caption_look.save(font=values.get("font", "").strip(), box=color)
        outro_card.save_tab(on=values.get("tab_on") == "1",
                            text=values.get("tab_text", "").strip() or load_brand().get("cta_url", ""),
                            color=color or outro_card.DEFAULT_COLOR)
    elif key == "endcard":
        from . import caption_look, outro_card

        lines = [l.strip() for l in values.get("outro", "").splitlines() if l.strip()][:3]
        website = load_brand().get("cta_url", "")
        color = (values.get("end_color") or caption_look.load()["box"] or "#111111").strip()
        outro_card.save(enabled=bool(lines and website), lines=lines, website=website,
                        color=color)
    elif key == "schedule":
        from . import posting_plan
        from .config import save_timezone, scheduler_timezone
        from .db import session_scope

        if values.get("tz"):
            save_timezone(values["tz"])
        if values.get("rhythm") in posting_plan.MODES:
            today = dt.datetime.now(scheduler_timezone()).date()
            with session_scope() as session:
                posting_plan.save_mode(session, values["rhythm"])
                posting_plan.ensure(session, today=today)
    elif key == "sources":
        from . import onboarding_sources

        onboarding_sources.commit()


def answers_with_state(state: dict[str, Any]) -> dict[str, str]:
    values = saved_values()
    values.update({k: str(v) for k, v in (state.get("answers") or {}).items()
                   if k in STATE_ONLY_FIELDS})
    return values


def form_values(state: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
    """What each field shows: the helper's draft if there is one, else what's
    saved. Also returns which fields hold an unsaved draft."""
    values = answers_with_state(state)
    drafts = {k: v for k, v in (state.get("drafts") or {}).items() if k in ALL_FIELDS}
    drafted = {k for k, v in drafts.items() if v != values.get(k)}
    values.update(drafts)
    return values, drafted


def _keyword_list(text: str) -> list[str]:
    seen: set[str] = set()
    out = []
    for line in str(text or "").replace(",", "\n").splitlines():
        word = " ".join(line.split()).strip().lower()
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def save_step(key: str, form: dict[str, str]) -> dict[str, Any]:
    """Save one step's answers to the real settings and mark it finished."""
    spec = step(key)
    if spec is None:
        raise KeyError(key)
    names = [f["name"] for f in spec["fields"]]
    values = {n: str(form.get(n, "")).strip() for n in names if n in form}
    brand_values = {n: v for n, v in values.items() if n in BRAND_FIELDS}
    if "cta_url" in brand_values:
        import re

        brand_values["cta_url"] = re.sub(r"^\s*https?://(www\.)?", "",
                                         brand_values["cta_url"]).strip().strip("/")
    if brand_values:
        save_brand(brand_values)
    if "keywords" in values:
        save_keywords(_keyword_list(values["keywords"]))
    _save_settings_step(key, values)
    state = load()
    for n in names:
        (state.get("drafts") or {}).pop(n, None)
        if n in STATE_ONLY_FIELDS and n in values:
            state.setdefault("answers", {})[n] = values[n]
    if key not in state["steps_done"]:
        state["steps_done"].append(key)
    save(state)
    return state


def set_busy(what: str | None, error: str = "") -> None:
    """What the helper is doing in the background ("reading", "finishing"),
    for the pages to show and wait on."""
    state = load()
    state["busy"] = what
    state["busy_error"] = error
    save(state)


# --- The helper ----------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the setup helper for Throughline, a service from Continuity Marketing that \
runs a short-video social account for a brand. Throughline watches YouTube channels, \
picks up new uploads whose title or description contains one of the account's \
keywords, has a model judge whether each video fits the account, cuts short clips \
from the good ones, writes captions, and posts them to Threads and Instagram after \
the customer approves them.

A new customer is setting up. They answer a few short questions, one per screen, \
and you draft the answers for them so most screens arrive already filled in. They \
can also open a chat with you on any screen.

The answers (field name: what it holds):
- name: the brand's name.
- cta_url: the brand's website, bare domain.
- mission: what the brand does, in a sentence or two, the way they would tell a \
friend. This is about the brand and its business, not the social account.
- audience: the brand's customers: who buys from them or who they serve. Not \
"followers"; the real customers.
- throughline: the one idea every clip comes back to, whatever the footage, in \
one short sentence the customer would say out loud. Every caption carries it.
- format: the kinds of clips that carry the throughline, one per line, each \
written "emoji Title — one-line example". Pick two to four for them.
- caption_brief: optional background every caption draws on: a framework, how \
the brand's product would respond to what a clip shows, sample takes. Only fill it \
from material the customer gives you (a document, a description of their method); \
never invent it.
- color: the brand's main color as a hex code like #1A4A7D, from its website or \
logo: a real brand color, not a neutral gray. Highlights the word being said in \
subtitles.
- tab_text: the small label in the top-left corner of every clip: the website, \
a handle, or two to five words (under 40 characters).
- end_color: the end card's background color as a hex code; usually the same \
brand color, or a darker partner to it.
- outro: the end card on every clip: up to three short lines (one per line, each \
under 30 characters), in the brand's voice, ending on what people should do.
- off_limits: what the account never posts, as a short comma-separated list on one line.
- keywords: one keyword or short phrase per line, usually 25 to 50, chosen to \
find videos that carry the throughline. When the throughline reaches past the \
topic, so do the keywords (a pilates account about learned strength also needs \
"stroke recovery" and "athlete training"). Be inclusive: \
every match is then checked by a model, so recall matters more than precision. \
Leave out words so common they would match most uploads, and qualify ambiguous \
ones ("pilates teaser", not "teaser").

The throughline is the heart of setup, and the product is named after it. A \
brand earns thought leadership with found footage by having one idea it keeps \
coming back to across very different clips. The variety of footage is the point: \
the same idea showing up in the news, in archival film, in pop culture and in a \
stranger's phone video is what makes the brand the voice that sees it. Examples:
- A climate nonprofit mixes climate news, archival footage, celebrities, animals \
and conservation, and big-idea talks. Throughline: the climate is changing right \
now, and people are doing something about it.
- An AI conflict-mediation firm posts arguments, standoffs, heated meetings, \
referees and coaches, family blowups. Throughline: every conflict has a moment \
where it could have gone differently.
- A pilates studio might post a new back-pain study, Joseph Pilates on a 1930s \
rooftop, an athlete's core training and an 80-year-old's teaser. Throughline: \
strength is something you learn, not something you're born with.
A good throughline is short, true to the brand, and a point of view rather than \
a topic ("conflict" is a topic; "every conflict has a moment where it could have \
gone differently" is a throughline). Every throughline you offer must cast a wide net: the account posts every day, \
so YouTube has to keep producing plenty of videos that carry it. The throughline \
does more than shape captions: it decides which videos the critic picks, which \
keywords are searched and which channels are watched. So three options differ in \
where the footage comes from, not in how narrow they are:
- Footage of the topic itself, in the news and through history (the climate \
nonprofit's is essentially "a cross-section of climate in video, today and \
through history"; for pilates, "pilates in the news and through history").
- The topic and the wider world around it (for pilates, "strength is something \
you learn, not something you're born with": pilates, but also athletes, aging, \
rehab).
- An idea that reaches well past the topic (for pilates, "stories of personal \
transformation through mind and body practice"; for the mediation firm, any \
interpersonal conflict caught on camera).
A throughline only a handful of videos could ever carry is too narrow, however \
good it sounds.
Offer kinds of clips that carry it with the suggest_formats tool: four to six, \
mixing the obvious with at least one surprising source.

Three options for every answer: for each of mission, audience, throughline and \
off_limits, offer exactly three options with the suggest_options tool and fill the \
field with fill_fields using the one you recommend. Make the three genuinely \
different, not rewordings. For throughline, give one of each kind above, in that \
order, labeled by where the footage comes from in two or three words (for \
pilates: "Pilates footage", "Pilates and beyond", "Beyond pilates"), each with a \
note saying what it draws on and naming two very different clips it would connect. For the others, label each \
option in two or three words (for example "Plain and direct", "Warmer", \
"Mission-first") and leave the note short or empty. When the customer changes \
direction in chat, offer fresh options.

How to work:
- Draft with the fill_fields tool as soon as you can make a reasonable draft, for \
any screen, not just the current one. A draft they can correct beats another question.
- Write answers in the customer's voice: warm, plain, specific, short. No marketing \
fluff.
- When they give a website or links, fetch the pages that tell you what the brand \
does and who it serves.
- In chat, keep replies to one to three short sentences of plain text (no \
markdown), and ask at most one question.
- A message may start with a <form_state> block showing the current screen and \
what the answers hold. It is context, not something they typed.
"""

# Written answers that come with three options to pick from.
OPTION_FIELDS = ("mission", "audience", "throughline", "off_limits")

OPTIONS_TOOL = {
    "name": "suggest_options",
    "description": (
        "Offer the customer three options for one or more written answers, shown as "
        "cards under the field. Picking one fills the field. Replaces that field's "
        "previous options."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": list(OPTION_FIELDS)},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string",
                                              "description": "Two or three words; for "
                                                             "throughline its reach."},
                                    "text": {"type": "string",
                                             "description": "The answer itself."},
                                    "note": {"type": "string",
                                             "description": "For throughline, two very "
                                                            "different clips it connects. "
                                                            "Otherwise short or empty."},
                                },
                                "required": ["label", "text", "note"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["field", "options"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["suggestions"],
        "additionalProperties": False,
    },
}


def _clean_options(raw: Any) -> dict[str, list[dict[str, str]]]:
    """The tool's suggestions as {field: [up to three {label, text, note}]}."""
    out: dict[str, list[dict[str, str]]] = {}
    items = raw.get("suggestions") if isinstance(raw, dict) else None
    for item in items or []:
        if not isinstance(item, dict) or item.get("field") not in OPTION_FIELDS:
            continue
        opts = [{"label": str(o.get("label", "")).strip()[:40],
                 "text": str(o.get("text", "")).strip()[:600],
                 "note": str(o.get("note", "")).strip()[:300]}
                for o in item.get("options") or [] if isinstance(o, dict)]
        opts = [o for o in opts if o["text"]][:3]
        if opts:
            out[item["field"]] = opts
    return out


FORMATS_TOOL = {
    "name": "suggest_formats",
    "description": (
        "Offer the customer kinds of clips that could carry their throughline, specific "
        "to their business. They appear as pills to pick from. Replaces the previous "
        "ideas, so include every idea worth showing (four to six)."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "emoji": {"type": "string"},
                        "title": {"type": "string"},
                        "example": {"type": "string"},
                    },
                    "required": ["emoji", "title", "example"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["options"],
        "additionalProperties": False,
    },
}


# Shown until the helper has read the brand and suggested its own.
GENERIC_FORMATS = [
    {"emoji": "🗞️", "title": "In the news", "example": "Fresh stories about your field, as they break"},
    {"emoji": "📼", "title": "From the archive", "example": "Old footage that shows how far things have come"},
    {"emoji": "🎬", "title": "Pop culture", "example": "Your subject in movies, TV and celebrity moments"},
    {"emoji": "🎤", "title": "Big ideas", "example": "Talks and interviews from people leading the field"},
    {"emoji": "📱", "title": "Real life", "example": "Everyday moments caught on camera that make your point"},
]


def format_cards(state: dict[str, Any], selected_text: str) -> list[dict[str, Any]]:
    """The concept cards for the format screen: the helper's ideas (or the
    generic ones), plus anything already picked, each marked picked or not."""
    picked = parse_format(selected_text)
    picked_titles = {p["title"].lower() for p in picked}
    cards = [dict(o, on=o["title"].lower() in picked_titles)
             for o in (state.get("format_options") or GENERIC_FORMATS)]
    known = {c["title"].lower() for c in cards}
    cards += [dict(p, on=True) for p in picked if p["title"].lower() not in known]
    for c in cards:
        c["line"] = format_line(c)
    return cards


def format_line(option: dict[str, str]) -> str:
    """One concept idea as it's stored in the brand's ``format`` field."""
    head = " ".join(x for x in (option.get("emoji", "").strip(), option.get("title", "").strip()) if x)
    example = option.get("example", "").strip()
    return f"{head} — {example}" if example else head


def parse_format(text: str) -> list[dict[str, str]]:
    """The brand's ``format`` field back into ideas (emoji, title, example)."""
    out = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, example = line.partition(" — ")
        first, _, rest = head.partition(" ")
        emoji, title = (first, rest) if rest and not first.isascii() else ("", head)
        out.append({"emoji": emoji, "title": title.strip(), "example": example.strip()})
    return out


FILL_TOOL = {
    "name": "fill_fields",
    "description": (
        "Draft answers for the customer's setup screens. They appear as suggestions for "
        "the customer to review. Only include the fields you are setting; a value "
        "replaces the field's current content entirely."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": list(ALL_FIELDS)},
                        "value": {"type": "string"},
                    },
                    "required": ["field", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["fields"],
        "additionalProperties": False,
    },
}

WEB_FETCH_TOOL = {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 3}

MAX_TOOL_ROUNDS = 6

# Uploads the helper can read, by extension.
_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}
_TEXT_TYPES = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".html"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class UploadError(ValueError):
    pass


def attachment_block(filename: str, data: bytes) -> dict[str, Any]:
    """A content block for an uploaded file, or UploadError for one the helper
    can't read."""
    name = (filename or "file").strip()
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"{name} is over 5 MB")
    if ext == ".pdf":
        return {"type": "document", "title": name[:200],
                "source": {"type": "base64", "media_type": "application/pdf",
                           "data": base64.standard_b64encode(data).decode()}}
    if ext in _IMAGE_TYPES:
        return {"type": "image",
                "source": {"type": "base64", "media_type": _IMAGE_TYPES[ext],
                           "data": base64.standard_b64encode(data).decode()}}
    if ext in _TEXT_TYPES:
        return {"type": "document", "title": name[:200],
                "source": {"type": "text", "media_type": "text/plain",
                           "data": data.decode("utf-8", errors="replace")}}
    raise UploadError(f"Can't read {name}: upload a PDF, an image, or a text file")


def _model() -> str:
    return str(load_settings().get("onboarding.model") or DEFAULT_MODEL)


def _form_state_block(step_key: str, values: dict[str, str]) -> str:
    spec = step(step_key) or STEPS[0]
    shown = {f["name"]: values.get(f["name"], "") for f in spec["fields"]}
    others = {k: v for k, v in values.items() if k not in shown and v}
    return ("<form_state>\n"
            f"Current screen: {spec['title']} ({spec['key']})\n"
            f"Fields on this screen: {json.dumps(shown, ensure_ascii=False)}\n"
            f"Other fields already filled: {json.dumps(others, ensure_ascii=False)}\n"
            "</form_state>")


_ONE_LINE_FIELDS = tuple(f["name"] for s in STEPS for f in s["fields"] if f["kind"] == "line") + ("off_limits",)


def _apply_fills(raw: Any, fills: dict[str, str]) -> int:
    n = 0
    items = raw.get("fields") if isinstance(raw, dict) else None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name, value = item.get("field"), item.get("value")
        if name in ALL_FIELDS and isinstance(value, str):
            value = value.strip()
            if name in _ONE_LINE_FIELDS:
                value = ", ".join(p.strip(" -•") for p in value.splitlines() if p.strip(" -•"))
            fills[name] = value
            n += 1
    return n


def _plain(text: str) -> str:
    """The chat panel shows plain text: drop the markdown emphasis and heading
    marks the model sometimes uses anyway."""
    import re

    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    return text.strip()


def chat(step_key: str, text: str, attachments: list[dict[str, Any]] | None = None,
         *, form: dict[str, str] | None = None, client=None,
         keep_answered: bool = True) -> dict[str, Any]:
    """One turn with the helper. ``form`` is what the customer's form holds right
    now (unsaved typing included). Returns {reply, fills, error}; the fills are
    already stored as drafts. With ``keep_answered`` off (the background read),
    drafts for screens the customer has already answered are dropped: they
    moved on while the helper was reading, and their answer stands."""
    from .llm import client as default_client

    client = client or default_client()
    state = load()
    values, _ = form_values(state)
    if form:
        values.update({k: str(v) for k, v in form.items() if k in ALL_FIELDS})

    content: list[dict[str, Any]] = [{"type": "text", "text": _form_state_block(step_key, values)}]
    content.extend(attachments or [])
    content.append({"type": "text", "text": text.strip() or "(I uploaded a file.)"})
    messages: list[dict[str, Any]] = list(state.get("messages") or [])
    messages.append({"role": "user", "content": content})

    model = _model()
    fills: dict[str, str] = {}
    options: list[dict[str, str]] | None = None
    field_options: dict[str, list[dict[str, str]]] = {}
    reply_parts: list[str] = []
    error = ""
    for _round in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=model,
            max_tokens=16000,
            system=[{"type": "text", "text": with_american_english(SYSTEM_PROMPT),
                     "cache_control": {"type": "ephemeral"}}],
            tools=[FILL_TOOL, OPTIONS_TOOL, FORMATS_TOOL, WEB_FETCH_TOOL],
            output_config={"effort": "medium"},
            messages=messages,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            spend.record(model, getattr(usage, "input_tokens", 0) or 0,
                         getattr(usage, "output_tokens", 0) or 0)
        if resp.stop_reason == "refusal":
            # The declined turn isn't kept: the conversation stays as it was
            # before this message, so they can rephrase.
            error = "The helper couldn't answer that one. Try putting it another way."
            messages = list(state.get("messages") or [])
            break
        messages.append({"role": "assistant",
                         "content": [b.to_dict() for b in resp.content]})
        reply_parts.extend(b.text for b in resp.content if b.type == "text" and b.text.strip())
        if resp.stop_reason == "pause_turn":
            continue   # a long server-side fetch; let it carry on
        if resp.stop_reason != "tool_use":
            break
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "suggest_options":
                got = _clean_options(block.input)
                field_options.update(got)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "Showing options for " + (", ".join(got) or "nothing")
                                           + " as cards."})
                continue
            if block.name == "suggest_formats":
                opts = [o for o in (block.input or {}).get("options") or []
                        if isinstance(o, dict) and str(o.get("title", "")).strip()]
                options = [{k: str(o.get(k, "")).strip()[:160] for k in ("emoji", "title", "example")}
                           for o in opts[:8]]
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Showing {len(options)} ideas as cards."})
                continue
            if block.name == "fill_fields":
                n = _apply_fills(block.input, fills)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Drafted {n} field(s); they're in the form for the customer to review."})
            else:
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": f"Unknown tool {block.name}"})
        messages.append({"role": "user", "content": results})
    else:
        reply_parts.append("(I ran out of steps on that one. Say “go on” and I'll pick it up.)")

    reply = _plain("\n\n".join(reply_parts))
    if not error and not reply and fills:
        reply = "I've drafted those in the form. Have a look and change anything that's off."

    # Re-read before writing: the customer may have saved a step while the
    # helper was thinking.
    state = load()
    if not keep_answered:
        fills = {k: v for k, v in fills.items() if step_of(k) not in state["steps_done"]}
    state["messages"] = messages
    state["drafts"] = {**(state.get("drafts") or {}), **fills}
    if options:
        state["format_options"] = options
    if field_options:
        state["options"] = {**(state.get("options") or {}), **field_options}
    names = [a.get("title") or a.get("type") for a in (attachments or [])]
    state["transcript"].append({"role": "user", "text": text.strip(), "files": names,
                                "step": step_key, "at": _now()})
    if reply or error:
        state["transcript"].append({"role": "assistant", "text": reply or error,
                                    "fields": sorted(fills), "error": bool(error),
                                    "step": step_key, "at": _now()})
    save(state)
    return {"reply": reply, "fills": fills, "error": error,
            "format_options": options, "options": field_options}


# --- Background work -------------------------------------------------------------

def read_brand(name: str, website: str, attachments: list[dict[str, Any]] | None = None,
               *, client=None) -> None:
    """After the first screen: read the website (and any upload) and draft the
    rest of the answers, so the next screens arrive filled in. Runs off the
    request; the pages poll ``status``."""
    set_busy("reading")
    parts = [f"My brand is {name}."]
    if website:
        parts.append(f"Our website is {website}.")
    if attachments:
        parts.append("I've attached something about us.")
    parts.append("Please draft the rest of my answers with three options for each, and suggest the kinds of clips that could carry my throughline.")
    try:
        chat("brand", " ".join(parts), attachments, client=client, keep_answered=False)
        set_busy(None)
    except Exception as exc:
        log.exception("Reading the brand failed")
        set_busy(None, f"Couldn't read that: {exc.__class__.__name__}")


def refresh_for_throughline(throughline: str, *, client=None) -> None:
    """After the throughline screen: redraft the keywords (and the kinds of
    clips) for the throughline they chose, since it decides which videos are
    searched for. Runs off the request; the keywords screen waits on it."""
    set_busy("refreshing")
    try:
        chat("throughline",
             f"I've chosen my throughline: “{throughline}”. Redraft my keywords so they "
             "find the videos that carry it, reaching past the topic if the throughline "
             "does, and update the kinds of clips if they no longer fit.",
             client=client, keep_answered=False)
        set_busy(None)
    except Exception as exc:
        log.exception("Refreshing for the throughline failed")
        set_busy(None, f"Couldn't update the keywords: {exc.__class__.__name__}")


_DERIVE_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in DERIVED_FIELDS},
    "required": list(DERIVED_FIELDS),
    "additionalProperties": False,
}

_DERIVE_SYSTEM = """\
You set up the behind-the-scenes rules for a Throughline account (a service that \
finds YouTube videos for a brand, cuts short clips and posts them). From the \
customer's answers, above all their throughline (the one idea every clip comes \
back to), the kinds of clips that carry it (format) and what is off limits, write the fields other models read when they judge videos \
and pick moments:
- topic: what the account posts about, two to five words. When the throughline \
reaches past the brand's own subject, the topic follows the throughline.
- positioning: what makes a clip right for this account and what is a no, in two \
or three concrete sentences. Lead with the throughline: a clip is right when it \
carries that idea, whichever kind of footage it is.
- source_kind: the kinds of videos the clips come from, a short noun phrase.
- relevance_rules: completes "A video is on topic when ...". One or two concrete \
sentences covering every kind of clip in the format, without the lead-in.
- false_positives: completes "... and off topic when ...". The near-misses its \
keywords will attract and everything off limits, named concretely. Without the lead-in.
- strong_openings: what the first seconds of a strong clip show, as a comma-separated list.
- weak_openings: what makes people scroll past in the first seconds, as a comma-separated list.
Be specific to this brand. Plain text, no markdown.
"""


def derive_format(*, client=None) -> dict[str, str]:
    """Write the prompt-tuning fields from the customer's answers and save
    them to the brand. Returns what was written."""
    from .llm import client as default_client

    client = client or default_client()
    state = load()
    answers = {k: v for k, v in answers_with_state(state).items() if k in ASKED_FIELDS}
    answers["keywords"] = ", ".join(load_keywords()[:80])
    said = [m["text"] for m in state.get("transcript") or [] if m.get("role") == "user"]
    user = json.dumps({"answers": answers, "what_they_told_the_helper": said[-12:]},
                      ensure_ascii=False)
    model = _model()
    resp = client.messages.create(
        model=model, max_tokens=4000, system=with_american_english(_DERIVE_SYSTEM),
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": _DERIVE_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    usage = getattr(resp, "usage", None)
    if usage is not None:
        spend.record(model, getattr(usage, "input_tokens", 0) or 0,
                     getattr(usage, "output_tokens", 0) or 0)
    text = "".join(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    out = {f: _plain(str(data.get(f) or "")) for f in DERIVED_FIELDS}
    save_brand({k: v for k, v in out.items() if v})
    return out


def finish(*, client=None) -> None:
    """The last screen: write the behind-the-scenes rules, then mark the
    account onboarded. If the rules can't be written, onboarding still
    finishes: the prompts fall back to generic wording, and an admin can fill
    them in under Brand & audience."""
    set_busy("finishing")
    try:
        derive_format(client=client)
    except Exception:
        log.exception("Deriving the format failed; finishing without it")
    state = load()
    state["busy"] = None
    mark_done(state)
