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
# attribution_enabled: when true, an attribution comment the OPERATOR set on a
# post (typed, or accepted from the "Suggest a draft" formal citation) is
# published as the first comment after the post goes live. Nothing is ever
# drafted or posted automatically — a post whose attribution field is empty
# publishes without one.
#
# When enabled is true and text is non-empty, that static text is the reply
# instead — used as the fallback for posts without an attribution. A reply
# failure never rolls back the post — check the post page and retry there.

"""


def load_first_reply() -> dict[str, Any]:
    """Return ``{enabled: bool, text: str, attribution_enabled: bool}`` for the
    auto first-reply."""
    data = _load_yaml(CONFIG_DIR / "first_reply.yaml")
    text = data.get("text") or ""
    if isinstance(text, str):
        text = text.strip()
    else:
        text = str(text).strip()
    return {
        "enabled": bool(data.get("enabled", False)),
        "text": text,
        "attribution_enabled": bool(data.get("attribution_enabled", True)),
    }


def save_first_reply(*, enabled: bool, text: str, attribution_enabled: bool = True) -> None:
    payload = {
        "enabled": bool(enabled),
        "attribution_enabled": bool(attribution_enabled),
        "text": (text or "").strip(),
    }
    body = yaml.safe_dump(payload, default_flow_style=False, allow_unicode=True, width=88)
    (CONFIG_DIR / "first_reply.yaml").write_text(FIRST_REPLY_HEADER + body)


BRAND_HEADER = """\
# Who this workspace is: identity + audience fields that parameterize the LLM
# prompts (relevance scoring, clip selection, captions, titles, digest), plus
# white-label appearance for the app chrome. Editable via the Brand & audience
# page under Configure; no code changes needed.

"""

# White-label defaults: what the sidebar/title show when brand.yaml is unset.
DEFAULT_APP_NAME = "Clip Monitor"

_BRAND_TEXT_FIELDS = ("name", "mission", "audience", "voice_notes", "topic",
                      "app_name", "logo_file",
                      # Prompt-framing fields consumed by app/llm.py. Blank
                      # fields fall back to generic phrasing there.
                      "source_kind", "relevance_rules", "false_positives",
                      "strong_openings", "weak_openings", "clip_guidance")


def load_brand() -> dict[str, str]:
    """Brand & audience config as ``{field: str}`` with every field present."""
    data = _load_yaml(CONFIG_DIR / "brand.yaml")
    return {f: str(data.get(f) or "").strip() for f in _BRAND_TEXT_FIELDS}


def save_brand(values: dict[str, str]) -> None:
    current = load_brand()
    for f in _BRAND_TEXT_FIELDS:
        if f in values:
            current[f] = str(values[f] or "").strip()
    body = yaml.safe_dump(current, default_flow_style=False, allow_unicode=True,
                          width=88, sort_keys=False)
    (CONFIG_DIR / "brand.yaml").write_text(BRAND_HEADER + body)


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
