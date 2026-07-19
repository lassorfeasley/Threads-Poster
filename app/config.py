"""Config loading: .env secrets + YAML settings/keywords/channels."""
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
