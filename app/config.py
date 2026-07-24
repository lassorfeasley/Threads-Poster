"""Config loading: .env secrets + YAML settings/keywords/channels/first-reply."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

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


def load_settings() -> Settings:
    return Settings(raw=_load_yaml(CONFIG_DIR / "settings.yaml"))


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
# When enabled is true and text is non-empty, the reply is published
# immediately after the main post succeeds. A reply failure never rolls
# back the post — check the post page and retry there if needed.

"""


def load_first_reply() -> dict[str, Any]:
    """Return ``{enabled: bool, text: str}`` for the auto first-reply."""
    data = _load_yaml(CONFIG_DIR / "first_reply.yaml")
    text = data.get("text") or ""
    if isinstance(text, str):
        text = text.strip()
    else:
        text = str(text).strip()
    return {"enabled": bool(data.get("enabled", False)), "text": text}


def save_first_reply(*, enabled: bool, text: str) -> None:
    payload = {"enabled": bool(enabled), "text": (text or "").strip()}
    body = yaml.safe_dump(payload, default_flow_style=False, allow_unicode=True, width=88)
    (CONFIG_DIR / "first_reply.yaml").write_text(FIRST_REPLY_HEADER + body)


CAPTION_STYLE_HEADER = """\
# Operator style guide for AI-drafted post captions — a list of `rules`.
# Editable via the Style guide page under Configure; no code changes needed.
#
# Each rule: { text, enabled, priority }. Enabled rules are injected into the
# caption-drafting prompt as authoritative style instructions (high-priority
# ones first). Hard safety rules still apply — the model won't invent facts,
# always mentions the place, and stays under Threads' character limit — so an
# aggressive rule can't override those.

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


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def database_url() -> str:
    url = env("DATABASE_URL")
    if url:
        return url
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    return f"sqlite:///{data_dir / 'app.db'}"
