"""Config loading: .env secrets + YAML settings/keywords/channels/first-reply."""
from __future__ import annotations

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


def scheduler_timezone() -> ZoneInfo:
    """The zone posting windows are defined in (``scheduler.timezone``).

    Also the zone each post's weekday/hour analytics are recorded in, so those
    numbers mean the same thing no matter which machine did the publishing.
    """
    return ZoneInfo(str(load_settings().get("scheduler.timezone", "America/New_York")))


def load_keywords() -> list[str]:
    data = _load_yaml(CONFIG_DIR / "keywords.yaml")
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
    (CONFIG_DIR / "keywords.yaml").write_text(KEYWORDS_HEADER + body)


FIRST_REPLY_HEADER = """\
# Auto first-reply posted under every Threads post this app publishes.
# Editable via the cog on the Replies page; no code changes needed.
#
# The Replies page saves these settings to the shared database AND to this file.
# Every runner reads the database copy whenever one exists, so editing this
# file by hand changes nothing until it's saved again from the Replies page.
# The file only matters for a workspace that has never been saved from there.
#
# mode decides what the "Suggest a draft" button writes into the first-reply box
# on the cut and post pages:
#   citation   - a formal credit line for the source publisher (drafted only
#                when you click Suggest; nothing is written automatically).
#   invitation - a call to action written from `instruction` below, drafted
#                automatically for every post so none goes out without one.
#   static     - nothing is drafted: every post's first comment is `text`
#                below, unless you type something else into its box.
# Drafted text lands in an editable box you review before the post publishes;
# the first comment is only ever the text sitting in that box, or `text` when
# the box is empty.
#
# instruction is your own plain-language brief for invitation mode, passed to
# the model as the authoritative spec. It is also the ONLY source of factual
# claims the model may make, so any statistic or figure you want used has to
# appear here verbatim.
#
# attribution_enabled: when false, the first comment never posts, whatever is in
# the box. When enabled is true and text is non-empty, that static text is the
# fallback for posts whose box is empty. A reply failure never rolls back the
# post — check the post page and retry there.
#
# first_party_enabled: whether the brand's OWN promos get a first comment at all.
# On those the clip IS the ad, so the call to action belongs in the caption where
# it lands with the post, not in a comment the timing gates below can hold back
# for hours — long enough for a viewer to ask where to go and get no answer. Set
# it false and promo captions have to carry the URL themselves; found footage is
# unaffected either way.
#
# WHEN the first comment posts. With all four gates at 0 it goes out seconds
# after the post, which is the recommended setting: a post gets ~70% of its
# lifetime views in its first 2 hours and ~99% within 6 (measured across 400+
# of this account's posts, Nov 2026), so every hour of holding back hides the
# comment from most of the people who will ever see the post. Holding until
# other people reply also showed no benefit: posts commented under immediately
# drew as many audience replies as posts that waited. The gates remain for
# anyone who wants the comment held back until the post has a conversation of
# its own:
#   delay_minutes  - don't comment until the post is at least this old
#   min_replies    - ...and until this many OTHER people have replied
#   min_likes      - ...and until it has at least this many likes
#   deadline_hours - comment anyway once the post is this old, so a post that
#                    never got replies still gets its call to action
#                    (0 = wait indefinitely)
# All four at 0: comment immediately on publish.
#
# The deadline fires only on posts that FLOPPED — a post that got its replies
# was commented under long before. Left unqualified it therefore attaches the
# pitch to your weakest posts and only your weakest posts, which is backwards:
# nobody is reading the comments on a post nobody engaged with, so the ask earns
# nothing, while a Threads post that revives days later shows new arrivals a
# pitch before anyone has had a chance to reply. These two put a floor under it:
#   deadline_min_views - at the deadline, comment only if the post cleared this
#   deadline_min_likes - ...or this (either one is enough)
# A post under both is skipped for good rather than commented on. Both at 0
# means no floor: the deadline comments on everything it reaches.

"""

FIRST_REPLY_MODES = ("citation", "invitation", "static")


def _positive_int(value: Any, default: int = 0) -> int:
    """Non-negative int from YAML/form input; anything unparseable is 0."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


# The first-reply settings live in the shared database as well as in
# first_reply.yaml, for the same reason the posting windows do (see
# models.PostingSchedule): the Fly worker that actually posts the comments runs
# from a baked image, so a YAML-only edit made on the dashboard never reached it
# until the next deploy — which is how a replaced first-reply text kept going
# out for weeks. The DB copy wins whenever it exists; the YAML is the seed for a
# workspace that has never saved from the dashboard, and the mirror a save
# writes for the repo.
FIRST_REPLY_TOKEN = f"first_reply_config:{WORKSPACE}"
# load_first_reply() runs per post in the reply sweep and on every cut/post
# page; a short cache keeps that from being a DB round trip each time while a
# dashboard save still reaches the worker within a minute.
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
    """Return the auto first-reply config: ``{enabled, text, attribution_enabled,
    first_party_enabled, mode, instruction, delay_minutes, min_replies,
    min_likes, deadline_hours, deadline_min_views, deadline_min_likes}``.

    Read from the shared DB when a copy has been saved there, else from
    first_reply.yaml."""
    data = _first_reply_from_db()
    if data is None:
        data = _load_yaml(CONFIG_DIR / "first_reply.yaml")

    def _text(key: str) -> str:
        val = data.get(key) or ""
        return val.strip() if isinstance(val, str) else str(val).strip()

    mode = _text("mode").lower() or "citation"
    if mode not in FIRST_REPLY_MODES:
        mode = "citation"
    return {
        "enabled": bool(data.get("enabled", False)),
        "text": _text("text"),
        "attribution_enabled": bool(data.get("attribution_enabled", True)),
        # Absent from older config files, where every post got a first comment
        # regardless of provenance — the behavior those files were written for.
        "first_party_enabled": bool(data.get("first_party_enabled", True)),
        "mode": mode,
        "instruction": _text("instruction"),
        # Timing gates. Absent from older config files, and 0 there means
        # "post on publish" — the behavior those files were written for.
        "delay_minutes": _positive_int(data.get("delay_minutes")),
        "min_replies": _positive_int(data.get("min_replies")),
        "min_likes": _positive_int(data.get("min_likes")),
        "deadline_hours": _positive_int(data.get("deadline_hours")),
        # Traction floor under the deadline. Absent from older config files,
        # where 0 means the deadline comments unconditionally — what those
        # files were written for.
        "deadline_min_views": _positive_int(data.get("deadline_min_views")),
        "deadline_min_likes": _positive_int(data.get("deadline_min_likes")),
    }


def save_first_reply(*, enabled: bool, text: str, attribution_enabled: bool = True,
                     first_party_enabled: bool = True,
                     mode: str = "citation", instruction: str = "",
                     delay_minutes: int = 0, min_replies: int = 0,
                     min_likes: int = 0, deadline_hours: int = 0,
                     deadline_min_views: int = 0,
                     deadline_min_likes: int = 0) -> None:
    mode = (mode or "citation").strip().lower()
    if mode not in FIRST_REPLY_MODES:
        mode = "citation"
    payload = {
        "mode": mode,
        "enabled": bool(enabled),
        "attribution_enabled": bool(attribution_enabled),
        "first_party_enabled": bool(first_party_enabled),
        "delay_minutes": _positive_int(delay_minutes),
        "min_replies": _positive_int(min_replies),
        "min_likes": _positive_int(min_likes),
        "deadline_hours": _positive_int(deadline_hours),
        "deadline_min_views": _positive_int(deadline_min_views),
        "deadline_min_likes": _positive_int(deadline_min_likes),
        "instruction": (instruction or "").strip(),
        "text": (text or "").strip(),
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

# White-label defaults: what the sidebar/title show when brand.yaml is unset.
DEFAULT_APP_NAME = "Clip Monitor"

_BRAND_TEXT_FIELDS = ("name", "cta_url", "mission", "audience", "voice_notes",
                      "topic", "app_name", "logo_file",
                      # Prompt-framing fields consumed by app/llm.py. Blank
                      # fields fall back to generic phrasing there.
                      "source_kind", "relevance_rules", "false_positives",
                      "strong_openings", "weak_openings", "clip_guidance")


def load_brand() -> dict[str, str]:
    """Brand & audience config as ``{field: str}`` with every field present."""
    data = _load_yaml(CONFIG_DIR / "brand.yaml")
    return {f: str(data.get(f) or "").strip() for f in _BRAND_TEXT_FIELDS}


def _write_brand(fields: dict[str, str], rules: list[dict[str, Any]]) -> None:
    body = yaml.safe_dump({**fields, "rules": rules}, default_flow_style=False,
                          allow_unicode=True, width=88, sort_keys=False)
    (CONFIG_DIR / "brand.yaml").write_text(BRAND_HEADER + body)


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
# written about it (captions, hooks, re-air captions, the caption gut check).
RULE_SCOPES = ("everything", "captions")


def _coerce_brand_rule(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        item = {"text": item}
    if not isinstance(item, dict):
        return None
    text = " ".join(str(item.get("text") or "").split())
    if not text:
        return None
    scope = str(item.get("applies_to") or "everything").lower()
    return {"text": text[:400],
            "applies_to": scope if scope in RULE_SCOPES else "everything",
            "enabled": bool(item.get("enabled", True))}


def load_brand_rules() -> list[dict[str, Any]]:
    raw = _load_yaml(CONFIG_DIR / "brand.yaml").get("rules") or []
    return [r for r in (_coerce_brand_rule(i) for i in raw) if r]


def save_brand_rules(rules: list[Any]) -> list[dict[str, Any]]:
    cleaned = [r for r in (_coerce_brand_rule(i) for i in (rules or [])) if r]
    _write_brand(load_brand(), cleaned)
    return cleaned


def add_brand_rule(text: str, applies_to: str = "everything") -> dict[str, Any] | None:
    """Append one rule (from the caption editor, say). Returns it, or None when
    it's empty or already there."""
    rule = _coerce_brand_rule({"text": text, "applies_to": applies_to})
    if rule is None:
        return None
    rules = load_brand_rules()
    if any(r["text"].lower() == rule["text"].lower() for r in rules):
        return None
    save_brand_rules(rules + [rule])
    return rule


def render_brand_rules(scope: str) -> str:
    """Enabled rules that reach ``scope`` as a bullet list, or "".

    ``scope="content"`` is a decision about material — only rules that apply
    to everything. ``scope="captions"`` is text — every enabled rule, since a
    rule about the account's stance binds what it writes too.
    """
    rules = [r for r in load_brand_rules() if r["enabled"]
             and (scope == "captions" or r["applies_to"] == "everything")]
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
    data = _load_yaml(CONFIG_DIR / "caption_style.yaml")
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
    (CONFIG_DIR / "caption_style.yaml").write_text(CAPTION_STYLE_HEADER + body)


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
