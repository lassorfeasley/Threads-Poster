"""The account's own outro card: three lines and its website on a background
colour (Settings > Outro).

Stored in the shared database (``AppToken`` ``outro_card:<workspace>``) so
every machine that renders agrees. ``apply`` lays it over the outro design
(app/outro.py): the workspace's config/outro.yaml when it has one, else the
bundled app/outro_template.yaml. The lines replace the design's three text
lines, the website is the line typed out at the end and the tab on the
footage, and every piece of text turns black or white to read on the colour. With nothing saved, the design file stands.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from pathlib import Path

from .config import WORKSPACE

log = logging.getLogger("outro_card")

TOKEN = f"outro_card:{WORKSPACE}"
if len(TOKEN) > 40:  # AppToken.name is String(40)
    raise ValueError(f"workspace slug too long for an outro card: {TOKEN!r}")

TEMPLATE = Path(__file__).with_name("outro_template.yaml")
MAX_LINE = 40
MAX_WEBSITE = 40
DEFAULT_COLOR = "#1A4A7D"

_HEX = re.compile(r"#[0-9a-fA-F]{6}")
_SITE = re.compile(r"[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+(/[^\s]*)?")
_cache: tuple[float, dict | None] | None = None
_CACHE_TTL = 30.0


# --- The saved card ---------------------------------------------------------------------

def load() -> dict | None:
    """The saved card, or None when the account never made one (the design
    file decides). Cached briefly; an unreadable database reads as None."""
    global _cache
    now = time.monotonic()
    if _cache is not None and now < _cache[0]:
        return copy.deepcopy(_cache[1])
    card = None
    try:
        from .db import session_scope
        from .models import AppToken

        with session_scope(read_only=True) as session:
            row = session.get(AppToken, TOKEN)
            if row is not None and row.value:
                card = json.loads(row.value)
    except Exception as exc:
        log.warning("Couldn't read the outro card: %s", exc)
    _cache = (now + _CACHE_TTL, card)
    return copy.deepcopy(card)


def clean(*, enabled: bool, lines: list[str], website: str, color: str) -> dict:
    """Check and tidy a card. Raises ValueError with a message for the page."""
    lines = [" ".join((line or "").split()) for line in (lines or [])][:3]
    if any(len(line) > MAX_LINE for line in lines):
        raise ValueError(f"Keep each line to {MAX_LINE} characters so it fits on one line")
    if enabled and not any(lines):
        raise ValueError("Write at least one line before switching the outro on")
    website = (website or "").strip()
    website = re.sub(r"^https?://", "", website, flags=re.I).rstrip("/")
    website = re.sub(r"^www\.", "", website, flags=re.I)
    if website and (len(website) > MAX_WEBSITE or not _SITE.fullmatch(website)):
        raise ValueError("Enter your website like example.org")
    if enabled and not website:
        raise ValueError("Enter your website before switching the outro on")
    color = (color or "").strip()
    if color and not color.startswith("#"):
        color = "#" + color
    if not _HEX.fullmatch(color):
        raise ValueError(f"“{color}” isn't a colour like #1A4A7D")
    return {"enabled": bool(enabled), "lines": lines + [""] * (3 - len(lines)),
            "website": website, "color": color.upper()}


def save(**fields) -> dict:
    global _cache
    from .db import session_scope
    from .models import AppToken, utcnow

    card = clean(**fields)
    with session_scope() as session:
        row = session.get(AppToken, TOKEN)
        if row is None:
            row = AppToken(name=TOKEN)
            session.add(row)
        row.value = json.dumps({**card, "updated_at": utcnow().isoformat()})
        row.updated_at = utcnow()
    _cache = None
    return card


# --- Laying it over the design ------------------------------------------------------------

def _contrast(hex_color: str) -> str:
    from .caption_look import _contrast_text

    return "#FFFFFF" if _contrast_text(hex_color) == "#FFFFFF" else "#111111"


def _darker(hex_color: str, by: float = 0.55) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(*(round(c * by) for c in (r, g, b)))


def apply(raw: dict, card: dict) -> dict:
    """``raw`` (an outro design, as read from YAML) with ``card`` laid over it."""
    raw = copy.deepcopy(raw)
    raw["enabled"] = bool(card.get("enabled"))
    color = card.get("color") or DEFAULT_COLOR
    ink = _contrast(color)
    raw["background"] = {**(raw.get("background") or {}), "color": color, "edge_color": _darker(color)}

    elements = list(raw.get("elements") or [])
    texts = [i for i, el in enumerate(elements) if el.get("type", "text") == "text"][:3]
    typed = next((i for i, el in enumerate(elements) if el.get("type") == "typed"), None)
    lines = list(card.get("lines") or [])
    for n, i in enumerate(texts):
        elements[i] = {**elements[i], "text": lines[n] if n < len(lines) else ""}
    if typed is not None:
        elements[typed] = {**elements[typed], "text": card.get("website", "")}
    for el in elements:
        if el.get("type", "text") in ("text", "typed", "pill"):
            el["color"] = ink
            if el.get("type") == "typed":
                el["cursor_color"] = ink

    # Blank lines go, and the per-orientation tweaks (keyed by element index)
    # follow the elements that stay.
    keep = [i for i, el in enumerate(elements)
            if not (i in texts and not str(el.get("text", "")).strip())]
    kept_texts = [i for i in texts if i in keep]
    if texts and kept_texts and kept_texts[-1] != texts[-1]:
        # The design spaces the last line off from the website; carry that
        # space to whichever line is last now.
        last, now = texts[-1], kept_texts[-1]
        elements[now] = {**elements[now], "gap": elements[last].get("gap", 0.0)}
        for over in (raw.get("overrides") or {}).values():
            tweaks = over.get("elements") if isinstance(over, dict) else None
            if isinstance(tweaks, dict):
                moved = {int(k): v for k, v in tweaks.items()}
                if "gap" in moved.get(last, {}):
                    moved[now] = {**moved.get(now, {}), "gap": moved[last]["gap"]}
                over["elements"] = moved
    index_map = {old: new for new, old in enumerate(keep)}
    elements = [elements[i] for i in keep]
    raw["elements"] = elements
    for orientation, over in (raw.get("overrides") or {}).items():
        if isinstance(over, dict) and isinstance(over.get("elements"), dict):
            over["elements"] = {index_map[int(k)]: v for k, v in over["elements"].items()
                                if int(k) in index_map}

    tab = raw.get("tab") or {}
    raw["tab"] = {**tab, "text": card.get("website", ""), "box_color": color, "color": ink,
                  "enabled": bool(card.get("website")) and tab.get("enabled", True)}
    return raw


def fingerprint(card: dict | None) -> str:
    """What the rendered outro depends on beyond the design file, for the
    render cache: a new card renders fresh copies."""
    if not card:
        return ""
    return json.dumps({k: card.get(k) for k in ("enabled", "lines", "website", "color")}, sort_keys=True)


# --- For the settings page ------------------------------------------------------------------

def _design_file() -> Path:
    from .outro import CONFIG_FILE

    return CONFIG_FILE if CONFIG_FILE.exists() else TEMPLATE


def current() -> dict:
    """The card to show on the page: the saved one, else what the account's
    design file says today (so climate opens on its existing outro)."""
    card = load()
    if card:
        return clean(**{k: card.get(k) for k in ("enabled", "lines", "website", "color")})
    import yaml

    raw = yaml.safe_load(_design_file().read_text()) or {}
    elements = raw.get("elements") or []
    lines = [str(e.get("text", "")) for e in elements if e.get("type", "text") == "text"][:3]
    website = next((str(e.get("text", "")) for e in elements if e.get("type") == "typed"), "")
    if _design_file() == TEMPLATE:
        lines, website = ["", "", ""], ""
    color = str((raw.get("background") or {}).get("color") or DEFAULT_COLOR)
    return {"enabled": bool(raw.get("enabled", True)) and _design_file() != TEMPLATE,
            "lines": (lines + ["", "", ""])[:3], "website": website,
            "color": color.upper() if _HEX.fullmatch(color) else DEFAULT_COLOR}


def preview_png(card: dict, orientation: str) -> bytes:
    """The card's final frame for ``card`` (unsaved, straight from the form),
    as the renderer draws it."""
    import io

    import yaml

    from .outro import Card, config_from

    path = _design_file()
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = config_from(apply(raw, {**card, "enabled": True}), str(path.parent), orientation)
    w, h = (720, 1280) if orientation == "vertical" else (1280, 720)
    frame = Card(w, h, cfg).frame(float(cfg["duration"]) - 0.01).convert("RGB")
    out = io.BytesIO()
    frame.save(out, format="PNG", optimize=True)
    return out.getvalue()
