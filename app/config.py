"""Config loading: .env secrets + YAML settings/keywords/channels/first-reply."""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Which environment this process is: every workspace has its own config/,
# data/, database, and Meta credentials under workspaces/<slug>/. One process
# serves exactly one workspace — isolation comes from the process boundary,
# not from in-app tenancy.
WORKSPACE = os.environ.get("WORKSPACE", "climate")
WORKSPACE_DIR = ROOT / "workspaces" / WORKSPACE
if not WORKSPACE_DIR.is_dir():
    _known = sorted(
        p.name for p in (ROOT / "workspaces").glob("*") if p.is_dir()
    ) if (ROOT / "workspaces").is_dir() else []
    # Fail loudly: _load_yaml() returns {} for missing files, so a typo'd
    # workspace would otherwise boot silently with all-default config against
    # an empty database — and could publish with the wrong settings.
    raise RuntimeError(
        f"Unknown workspace {WORKSPACE!r}: {WORKSPACE_DIR} does not exist. "
        f"Available workspaces: {', '.join(_known) or '(none)'}. "
        f"Set WORKSPACE or pass --workspace to run.py."
    )
CONFIG_DIR = WORKSPACE_DIR / "config"
DATA_DIR = WORKSPACE_DIR / "data"

# Workspace .env first, root .env second: load_dotenv never overrides an
# already-set variable, so precedence is real environment (e.g. CI) > the
# workspace's .env > the shared root .env.
load_dotenv(WORKSPACE_DIR / ".env")
load_dotenv(ROOT / ".env")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_doc(key: str) -> dict[str, Any]:
    """A workspace document shared across machines (app/shared_docs.py)."""
    from .shared_docs import read_text

    text = read_text(key)
    if not text:
        return {}
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _save_doc(key: str, text: str) -> None:
    from .shared_docs import write_text

    write_text(key, text)


@dataclass
class Settings:
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


# Parsed-settings cache, keyed on the file's (mtime, size). settings.yaml is
# ~500 lines and load_settings() is called liberally — the calendar's window
# plan alone reads it 100+ times, which made YAML parsing rival the database
# as a page-load cost. Nothing in the app writes settings.yaml (it's edited
# by hand), and the stat check picks up out-of-band edits, so callers keep
# their re-read-every-call semantics at the price of one stat() instead of a
# full parse. Callers never mutate Settings.raw, so sharing one object is safe.
_settings_cache: tuple[tuple[int, int], Settings] | None = None


def load_settings() -> Settings:
    global _settings_cache
    path = CONFIG_DIR / "settings.yaml"
    try:
        st = path.stat()
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        key = (0, 0)
    cached = _settings_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    settings = Settings(raw=_load_yaml(path))
    _settings_cache = (key, settings)
    return settings


DEFAULT_TIMEZONE = "America/New_York"
TIMEZONE_TOKEN = f"timezone:{WORKSPACE}"

# scheduler_timezone() runs hundreds of times per calendar build and on every
# scheduler tick: the saved zone is read from the database at most once a
# minute (a failed read is retried after a few seconds, not every call).
_TIMEZONE_CACHE_TTL = 60.0
_TIMEZONE_RETRY_AFTER = 5.0
_timezone_cache: tuple[float, str | None] | None = None  # (expires, zone)


def _timezone_from_db() -> str | None:
    """The zone saved from the Posting schedule page, or None when none is.

    When the database can't be read, the zone from the last good read stands
    (a blip mustn't put this machine back on settings.yaml's clock while the
    others keep the saved one); with no good read yet, None."""
    global _timezone_cache
    import time

    now = time.monotonic()
    cached = _timezone_cache
    if cached is not None and now < cached[0]:
        return cached[1]
    try:
        from .db import session_scope
        from .models import AppToken

        name: str | None = None
        with session_scope(read_only=True) as session:
            row = session.get(AppToken, TIMEZONE_TOKEN)
            if row is not None and row.value:
                name = str(row.value).strip() or None
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Could not read the saved time zone: %s", exc)
        last = cached[1] if cached is not None else None
        _timezone_cache = (now + _TIMEZONE_RETRY_AFTER, last)
        return last
    _timezone_cache = (now + _TIMEZONE_CACHE_TTL, name)
    return name


def scheduler_timezone() -> ZoneInfo:
    """The account's time zone: posting windows are defined in it.

    Also the zone each post's weekday/hour analytics are recorded in, so those
    numbers mean the same thing no matter which machine did the publishing.

    The zone chosen on the Posting schedule page (kept in the shared database,
    so every machine agrees), else ``scheduler.timezone`` from settings.yaml."""
    settings = load_settings()
    for name in (_timezone_from_db(), str(settings.get("scheduler.timezone", DEFAULT_TIMEZONE)),
                 DEFAULT_TIMEZONE):
        if not name:
            continue
        try:
            return ZoneInfo(name)
        except Exception:
            continue
    return ZoneInfo(DEFAULT_TIMEZONE)


def timezone_saved() -> bool:
    """A zone has been chosen in the app (and is in force)."""
    return _timezone_from_db() is not None


def save_timezone(name: str) -> str:
    """Save the account's time zone (an IANA name such as ``Europe/Berlin``).
    Returns the name saved. Raises ValueError for anything that isn't a zone."""
    global _timezone_cache
    import time
    from zoneinfo import available_timezones

    from .db import session_scope
    from .models import AppToken, utcnow

    name = (name or "").strip()
    if len(TIMEZONE_TOKEN) > 40:  # AppToken.name is String(40)
        raise ValueError(f"workspace slug too long to save a time zone: {TIMEZONE_TOKEN!r}")
    if name not in available_timezones():
        raise ValueError(f"“{name}” isn't a time zone the app knows.")
    with session_scope() as session:
        row = session.get(AppToken, TIMEZONE_TOKEN)
        if row is None:
            row = AppToken(name=TIMEZONE_TOKEN)
            session.add(row)
        row.value = name
        row.updated_at = utcnow()
    # This machine follows at once; the others within a minute (their cache).
    _timezone_cache = (time.monotonic() + _TIMEZONE_CACHE_TTL, name)
    return name


def load_keywords() -> list[str]:
    data = _load_doc("keywords")
    return [str(k).strip() for k in data.get("keywords", []) if str(k).strip()]


KEYWORDS_HEADER = """\
# Climate keyword list used for the first-pass filter on video title + description.
# Matching is case-insensitive, whole-word/phrase. Edit freely (or via the
# dashboard's Keywords page); no code changes needed.
#
# Every keyword hit is then scored by the LLM for genuine climate relevance,
# which is what filters out "political climate" / "business climate" style
# false positives. So err on the side of inclusive keywords here.

"""


def save_keywords(keywords: list[str]) -> None:
    cleaned = sorted({k.strip().lower() for k in keywords if k.strip()})
    body = yaml.safe_dump({"keywords": cleaned}, default_flow_style=False, allow_unicode=True)
    _save_doc("keywords", KEYWORDS_HEADER + body)


FIRST_REPLY_HEADER = """\
# The first reply posted under every found-footage Threads post. Edit it on the
# First reply page (Engagement); no code changes needed.
#
# The page saves to the shared database AND to this file. Every runner reads
# the database copy whenever one exists, so editing this file by hand changes
# nothing until it's saved again from the page. The file only matters for a
# workspace that has never been saved from there.
#
# Two texts, one per case:
#   watch_text    - when the post's full-segment page on Renewables.org was
#                   published and loads. Must contain {watch_url}, which
#                   becomes renewables.org/watch/<slug>.
#   fallback_text - when there's no page: the source isn't an embeddable
#                   YouTube video, or the page couldn't be published.
# enabled: false and no first reply posts at all.
#
# It posts right after the post publishes. The brand's own promos never get
# one: on those the clip is the ad, so the call to action rides in the caption.

"""

WATCH_URL_PLACEHOLDER = "{watch_url}"

# The first-reply settings live in the shared database as well as in
# first_reply.yaml, for the same reason the posting windows do (see
# models.PostingSchedule): the Fly worker that actually posts the comments runs
# from a baked image, so a YAML-only edit made on the dashboard never reached it
# until the next deploy. The DB copy wins whenever it exists; the YAML is the
# seed for a workspace that has never saved from the dashboard, and the mirror
# a save writes for the repo.
FIRST_REPLY_TOKEN = f"first_reply_config:{WORKSPACE}"
# load_first_reply() runs once per published post and on the post page; a short
# cache keeps that from being a DB round trip each time while a dashboard save
# still reaches the worker within a minute.
_FIRST_REPLY_CACHE_TTL = 60.0
_first_reply_cache: tuple[float, dict[str, Any] | None] | None = None


def _first_reply_from_db() -> dict[str, Any] | None:
    """The DB copy of the first-reply settings, or None when there is none (or
    the DB can't be reached — the YAML then stands in rather than failing the
    caller)."""
    global _first_reply_cache
    import json
    import logging
    import time

    now = time.monotonic()
    if _first_reply_cache and now - _first_reply_cache[0] < _FIRST_REPLY_CACHE_TTL:
        return _first_reply_cache[1]
    data: dict[str, Any] | None = None
    try:
        from .db import session_scope
        from .models import AppToken

        with session_scope(read_only=True) as session:
            row = session.get(AppToken, FIRST_REPLY_TOKEN)
            if row is not None and row.value:
                parsed = json.loads(row.value)
                data = parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not read first-reply settings from the DB (using %s): %s",
            CONFIG_DIR / "first_reply.yaml", exc)
        return None  # not cached: retry on the next call
    _first_reply_cache = (now, data)
    return data


def load_first_reply() -> dict[str, Any]:
    """Return ``{enabled, watch_text, fallback_text}``.

    Read from the shared DB when a copy has been saved there, else from
    first_reply.yaml. Settings saved before the two-text version carry a
    single ``text``: it becomes the fallback as-is, and the watch text is the
    same line with its Renewables.org mention swapped for the link, which is
    what those posts were already getting."""
    data = _first_reply_from_db()
    if data is None:
        data = _load_yaml(CONFIG_DIR / "first_reply.yaml")

    def _text(key: str) -> str:
        val = data.get(key) or ""
        return val.strip() if isinstance(val, str) else str(val).strip()

    fallback = _text("fallback_text") or _text("text")
    watch = _text("watch_text")
    if not watch and fallback:
        import re
        watch = re.sub(r"(?<![\w/@.:-])renewables\.org(?![\w/.-])",
                       WATCH_URL_PLACEHOLDER, fallback, count=1, flags=re.IGNORECASE)
        if WATCH_URL_PLACEHOLDER not in watch:
            watch = ""
    return {
        "enabled": bool(data.get("enabled", False)),
        "watch_text": watch,
        "fallback_text": fallback,
    }


def save_first_reply(*, enabled: bool, watch_text: str, fallback_text: str) -> None:
    payload = {
        "enabled": bool(enabled),
        "watch_text": (watch_text or "").strip(),
        "fallback_text": (fallback_text or "").strip(),
    }
    _save_first_reply_to_db(payload)
    body = yaml.safe_dump(payload, default_flow_style=False, allow_unicode=True, width=88,
                          sort_keys=False)
    (CONFIG_DIR / "first_reply.yaml").write_text(FIRST_REPLY_HEADER + body)


def _save_first_reply_to_db(payload: dict[str, Any]) -> None:
    """Write the settings to the shared DB so every runner sees them. Raises on
    failure: a save that only reached the local YAML would look successful on
    the dashboard while the worker kept posting the old settings."""
    global _first_reply_cache
    import json

    from .db import session_scope
    from .models import AppToken, utcnow

    with session_scope() as session:
        row = session.get(AppToken, FIRST_REPLY_TOKEN)
        if row is None:
            row = AppToken(name=FIRST_REPLY_TOKEN)
            session.add(row)
        row.value = json.dumps(payload)
        row.updated_at = utcnow()
    _first_reply_cache = None


BRAND_HEADER = """\
# Who this workspace is: identity + audience fields that parameterize the LLM
# prompts (relevance scoring, clip selection, captions, titles, digest), plus
# white-label appearance for the app chrome. Editable via the Brand & audience
# page under Configure; no code changes needed.

"""

# Every model writes for an American audience. Appended to every system prompt
# (app/llm.py, app/clip_engine/claude.py and gemini.py, app/onboarding.py), so
# captions, rules and drafts never drift into British spelling.
AMERICAN_ENGLISH = ("Write in American English: American spelling and usage "
                    "(color, favorite, organize, center, canceled, labeled, gray).")


def with_american_english(system: str | None) -> str:
    system = (system or "").rstrip()
    return f"{system}\n\n{AMERICAN_ENGLISH}" if system else AMERICAN_ENGLISH


# White-label defaults: what the sidebar/title show when brand.yaml is unset.
DEFAULT_APP_NAME = "Throughline"

_BRAND_TEXT_FIELDS = ("name", "cta_url", "mission", "audience", "positioning",
                      "voice_notes",
                      "topic", "app_name", "logo_file",
                      # Prompt-framing fields consumed by app/llm.py. Blank
                      # fields fall back to generic phrasing there.
                      "source_kind", "relevance_rules", "false_positives",
                      "strong_openings", "weak_openings", "clip_guidance",
                      # The channel's concept (set in onboarding): its
                      # throughline, the one idea every clip comes back to,
                      # and the kinds of clips that carry it, one per line.
                      "throughline", "format",
                      # Optional background every caption can draw on: a
                      # framework, how the brand's product would respond to
                      # what a clip shows, sample takes.
                      "caption_brief")


def load_brand() -> dict[str, str]:
    """Brand & audience config as ``{field: str}`` with every field present."""
    data = _load_doc("brand")
    return {f: str(data.get(f) or "").strip() for f in _BRAND_TEXT_FIELDS}


def _write_brand(fields: dict[str, str], rules: list[dict[str, Any]]) -> None:
    body = yaml.safe_dump({**fields, "rules": rules}, default_flow_style=False,
                          allow_unicode=True, width=88, sort_keys=False)
    _save_doc("brand", BRAND_HEADER + body)


def save_brand(values: dict[str, str]) -> None:
    current = load_brand()
    for f in _BRAND_TEXT_FIELDS:
        if f in values:
            current[f] = str(values[f] or "").strip()
    _write_brand(current, load_brand_rules())


# ---- Brand rules -------------------------------------------------------------
# The operator's hard constraints, kept with the brand because they say what
# the account IS ("write as we", "never present climate denial as credible"),
# not how a caption should sound — that style is learned. Unlike the caption
# style guide's menu of moves, every enabled rule applies, always.
#
# ``applies_to`` scopes a rule: "everything" reaches the decisions about
# content too (relevance scoring, clip selection), "captions" only the text
# written about it (captions, hooks, re-air captions, the caption gut check),
# "clips" only the decisions about material (clip selection — through the
# "content" block, which relevance scoring shares) and never the words.
#
# One store for every rule, whoever wrote it: a rule the operator typed, one
# he added from his notes on clip suggestions, and one learned from his past
# edits (the handbook) are the same kind of rule and live in the same list.
# Optional metadata says where it came from — ``source`` ("manual", the
# default, "notes" or "handbook"), ``evidence`` (a short line: what it was
# drawn from) and ``added_at`` — and what the house formatter made of it
# (clip_engine.notes.format_rule): ``rule`` (the sentence), ``when``, ``why``,
# ``markdown`` (the page's rendering) and ``original`` (the words as first
# written). ``text`` stays what every prompt reads. Metadata is kept only
# when set, so a rule without it loads exactly as it always has.
RULE_SCOPES = ("everything", "captions", "clips")
RULE_SOURCES = ("manual", "notes", "handbook")
# Formatter metadata kept on a rule, with its length cap.
_RULE_META = {"evidence": 300, "added_at": 40, "rule": 400, "when": 160, "why": 300,
              "original": 1000}
# Which scopes each prompt kind takes. "content" is a decision about
# material; "captions" is text, and every stance ("everything") binds it.
# "screening" is relevance scoring of whole videos: clip-selection rules
# ("never use an interview in another language") must not lower a video's score.
_SCOPES_FOR = {"content": ("everything", "clips"), "screening": ("everything",),
               "captions": ("everything", "captions")}


def _coerce_brand_rule(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        item = {"text": item}
    if not isinstance(item, dict):
        return None
    text = " ".join(str(item.get("text") or "").split())
    if not text:
        return None
    scope = str(item.get("applies_to") or "everything").lower()
    rule: dict[str, Any] = {"text": text[:400],
                            "applies_to": scope if scope in RULE_SCOPES else "everything",
                            "enabled": bool(item.get("enabled", True))}
    source = str(item.get("source") or "").strip().lower()
    if source in RULE_SOURCES and source != "manual":
        rule["source"] = source
    for key, cap in _RULE_META.items():
        value = " ".join(str(item.get(key) or "").split())
        if value:
            rule[key] = value[:cap]
    # Markdown keeps its line breaks (one field per line).
    markdown = "\n".join(" ".join(line.split()) for line in str(item.get("markdown") or "")
                         .splitlines() if line.strip())
    if markdown:
        rule["markdown"] = markdown[:1200]
    return rule


def _rule_key(text: str) -> str:
    """Two rules are the same rule regardless of case, spacing or a closing
    full stop (the formatter adds one)."""
    return " ".join(str(text or "").lower().split()).rstrip(".")


def load_brand_rules() -> list[dict[str, Any]]:
    raw = _load_doc("brand").get("rules") or []
    return [r for r in (_coerce_brand_rule(i) for i in raw) if r]


def save_brand_rules(rules: list[Any]) -> list[dict[str, Any]]:
    cleaned = [r for r in (_coerce_brand_rule(i) for i in (rules or [])) if r]
    _write_brand(load_brand(), cleaned)
    return cleaned


def add_brand_rule(text: str, applies_to: str = "everything", *, source: str = "manual",
                   evidence: str = "", **meta: Any) -> dict[str, Any] | None:
    """Append one rule (from the caption editor, a note-drawn suggestion or
    the handbook). Returns it, or None when it's empty or already there.
    ``source``/``evidence`` record where it came from; a rule from anywhere
    but the operator's own hand is stamped with when it was added. ``meta``
    carries the formatter's fields (rule, when, why, markdown, original)."""
    item: dict[str, Any] = {k: v for k, v in meta.items()
                            if k in ("rule", "when", "why", "markdown", "original")}
    item.update(text=text, applies_to=applies_to, source=source, evidence=evidence)
    if str(source or "manual").strip().lower() != "manual":
        item["added_at"] = (_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
                            .isoformat().replace("+00:00", "Z"))
    rule = _coerce_brand_rule(item)
    if rule is None:
        return None
    rules = load_brand_rules()
    if any(_rule_key(r["text"]) == _rule_key(rule["text"]) for r in rules):
        return None
    save_brand_rules(rules + [rule])
    return rule


def render_brand_rules(scope: str) -> str:
    """Enabled rules that reach ``scope`` as a bullet list, or "".

    ``scope="content"`` is a decision about material — rules that apply to
    everything, and the clip-selection ("clips") rules. ``scope="captions"``
    is text — every enabled rule but the clip-selection ones, since a rule
    about the account's stance binds what it writes too.
    """
    scopes = _SCOPES_FOR.get(scope, ("everything",))
    rules = [r for r in load_brand_rules() if r["enabled"] and r["applies_to"] in scopes]
    return "\n".join(f"- {r['text']}" for r in rules)


CAPTION_STYLE_HEADER = """\
# Operator style guide for AI-drafted post captions — a list of `rules`.
# Editable via the Style guide page under Configure; no code changes needed.
#
# Each rule: { text, enabled, priority }. Enabled rules go into the drafting
# prompt as a MENU (high-priority ones first), not a checklist: the drafter
# picks the one that fits the clip and skips the rest, so rules never stack into
# a padded caption. Hard rules still win — the model won't invent facts, always
# mentions the place, and keeps the caption to one or two short lines
# (`engagement.caption_max_chars` in settings.yaml).

"""

_VALID_PRIORITY = ("high", "normal")


def _coerce_rule(item: Any) -> dict[str, Any] | None:
    """Normalize a raw entry (string or dict) into {text, enabled, priority}."""
    if isinstance(item, str):
        text = item.strip().lstrip("-*•").strip()
        return {"text": text, "enabled": True, "priority": "normal"} if text else None
    if isinstance(item, dict):
        text = str(item.get("text") or "").strip()
        if not text:
            return None
        priority = str(item.get("priority") or "normal").lower()
        if priority not in _VALID_PRIORITY:
            priority = "normal"
        return {"text": text, "enabled": bool(item.get("enabled", True)), "priority": priority}
    return None


def load_caption_rules() -> list[dict[str, Any]]:
    """The operator's caption style rules, in order. Back-compatible with the
    original single ``text:`` blob (split into one rule per line)."""
    data = _load_doc("captions")
    raw = data.get("rules")
    if raw is None:
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            raw = [ln for ln in text.splitlines() if ln.strip()]
        else:
            raw = []
    return [r for r in (_coerce_rule(item) for item in (raw or [])) if r]


def save_caption_rules(rules: list[dict[str, Any]]) -> None:
    cleaned = [r for r in (_coerce_rule(item) for item in (rules or [])) if r]
    body = yaml.safe_dump({"rules": cleaned}, default_flow_style=False,
                          allow_unicode=True, width=88, sort_keys=False)
    _save_doc("captions", CAPTION_STYLE_HEADER + body)


def render_caption_guide() -> str:
    """Enabled rules as a bullet list for the drafting prompt (high priority
    first), or an empty string when there are none."""
    rules = [r for r in load_caption_rules() if r["enabled"]]
    if not rules:
        return ""
    rules.sort(key=lambda r: 0 if r["priority"] == "high" else 1)
    return "\n".join(f"- {r['text']}" for r in rules)


def load_channel_seed() -> list[dict[str, Any]]:
    data = _load_yaml(CONFIG_DIR / "channels.yaml")
    return data.get("channels", [])


_WORKSPACE_DEFAULTS = {"label": "", "port": 8321, "accent": "", "enabled": True}


def load_workspaces() -> list[dict[str, Any]]:
    """The workspace registry from ``workspaces.yaml`` at the repo root, as
    ``[{slug, label, port, accent, enabled}]``. Falls back to a single entry
    for the current workspace when the registry is missing, so a checkout
    without the file still boots."""
    data = _load_yaml(ROOT / "workspaces.yaml")
    out: list[dict[str, Any]] = []
    for item in data.get("workspaces", []) or []:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        if not slug:
            continue
        entry = dict(_WORKSPACE_DEFAULTS)
        entry.update({k: item[k] for k in _WORKSPACE_DEFAULTS if k in item})
        entry["slug"] = slug
        entry["label"] = str(entry["label"] or slug)
        entry["port"] = int(entry["port"] or 8321)
        entry["enabled"] = bool(entry["enabled"])
        out.append(entry)
    if not out:
        out = [dict(_WORKSPACE_DEFAULTS, slug=WORKSPACE, label=WORKSPACE)]
    return out


def current_workspace() -> dict[str, Any]:
    """This process's registry entry (or a synthesized one)."""
    for ws in load_workspaces():
        if ws["slug"] == WORKSPACE:
            return ws
    return dict(_WORKSPACE_DEFAULTS, slug=WORKSPACE, label=WORKSPACE)


def storage_dir(settings: Settings, key: str, default: str) -> Path:
    """Resolve a ``storage.*`` directory setting against this workspace's data
    tree. Values are stored relative (e.g. ``data/videos``); absolute values
    are honored as-is for operators who point storage elsewhere."""
    value = str(settings.get(key, default))
    path = Path(value)
    if path.is_absolute():
        return path
    # Historical settings values are prefixed "data/"; strip it so the same
    # settings.yaml works before and after the workspace move.
    parts = path.parts
    if parts and parts[0] == "data":
        path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return DATA_DIR / path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def database_url() -> str:
    url = env("DATABASE_URL")
    if url:
        return url
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DATA_DIR / 'app.db'}"
