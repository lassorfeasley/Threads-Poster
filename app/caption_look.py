"""How the burned-in captions look: the font and the highlight box's color.

Each account picks from nine fonts and nine colors, or any Google font and
any color of its own (Settings → Subtitles). The choice lives in the shared
database (``AppToken`` ``caption_look:<workspace>``) so every machine that
renders captions agrees; with no choice saved, settings.yaml's ``subtitles``
section stands, exactly as before.

Fonts: Funnel Display is bundled (assets/fonts). Any other family is fetched
from Google Fonts the first time it's needed — the CSS API hands back a
TrueType URL — and cached under data/fonts/, so rendering never waits on the
network twice. Bold is preferred (captions are short and need weight); a
family without it falls back to the weights it has.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from .config import DATA_DIR, ROOT, WORKSPACE

log = logging.getLogger("caption_look")

TOKEN = f"caption_look:{WORKSPACE}"
if len(TOKEN) > 40:  # AppToken.name is String(40)
    raise ValueError(f"workspace slug too long for caption looks: {TOKEN!r}")

BUNDLED_FAMILY = "Funnel Display"
BUNDLED_FILE = ROOT / "assets" / "fonts" / "FunnelDisplay-SemiBold.ttf"
FONT_CACHE = DATA_DIR / "fonts"

# Nine faces that hold up as big, short, on-screen words.
FONTS = (
    "Funnel Display",
    "Inter",
    "Montserrat",
    "Poppins",
    "Anton",
    "Bebas Neue",
    "Oswald",
    "Archivo Black",
    "Playfair Display",
)

# The weight each preset is shown and rendered in, as a Google Fonts CSS
# suffix. Single-weight display faces have none to ask for (asking would get
# the whole request refused).
FONT_WEIGHT_SPEC = {"Funnel Display": ":wght@600", "Anton": "", "Bebas Neue": "", "Archivo Black": ""}
DEFAULT_WEIGHT_SPEC = ":wght@700"

# Nine highlight colors: brand-neutral, each readable behind its auto text color.
COLORS = (
    ("#FFFFFF", "White"),
    ("#111111", "Black"),
    ("#1A4A7D", "Navy"),
    ("#2F6FED", "Blue"),
    ("#16794C", "Green"),
    ("#F2C230", "Yellow"),
    ("#F07F3C", "Orange"),
    ("#E2445C", "Red"),
    ("#7B5CD6", "Purple"),
)

_HEX = re.compile(r"#[0-9a-fA-F]{6}")
_FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ]{0,60}")
_WEIGHTS = (700, 600, 800, 400, None)

_cache: tuple[float, dict] | None = None
_CACHE_TTL = 30.0
_fetch_lock = threading.Lock()


class FontError(ValueError):
    pass


# --- The saved choice ---------------------------------------------------------------

def load() -> dict:
    """``{"font": family or "", "box": "#RRGGBB" or ""}`` — "" where the account
    hasn't chosen (settings.yaml decides). Cached briefly; an unreadable
    database reads as no choice."""
    global _cache
    now = time.monotonic()
    if _cache is not None and now < _cache[0]:
        return dict(_cache[1])
    look = {"font": "", "box": ""}
    try:
        from .db import session_scope
        from .models import AppToken

        with session_scope(read_only=True) as session:
            row = session.get(AppToken, TOKEN)
            if row is not None and row.value:
                saved = json.loads(row.value)
                look["font"] = str(saved.get("font") or "")
                look["box"] = str(saved.get("box") or "")
    except Exception as exc:
        log.warning("Couldn't read the caption look: %s", exc)
    _cache = (now + _CACHE_TTL, look)
    return dict(look)


def clean_color(value: str) -> str:
    value = (value or "").strip()
    if value and not value.startswith("#"):
        value = "#" + value
    if not _HEX.fullmatch(value):
        raise ValueError(f"“{value}” isn't a color like #1A4A7D")
    return value.upper()


def clean_family(value: str) -> str:
    value = " ".join((value or "").split())
    if not _FAMILY.fullmatch(value):
        raise FontError("Font names are letters, numbers and spaces, like “Space Grotesk”")
    return value


def save(*, font: str, box: str) -> dict:
    """Save the account's choice. The font must exist on Google Fonts (it's
    fetched now, so a typo is caught here rather than at render time)."""
    global _cache
    from .db import session_scope
    from .models import AppToken, utcnow

    font = clean_family(font)
    box = clean_color(box)
    font_file(font)   # raises FontError if Google Fonts doesn't have it
    look = {"font": font, "box": box}
    with session_scope() as session:
        row = session.get(AppToken, TOKEN)
        if row is None:
            row = AppToken(name=TOKEN)
            session.add(row)
        row.value = json.dumps({**look, "updated_at": utcnow().isoformat()})
        row.updated_at = utcnow()
    _cache = None
    return look


# --- Fonts ------------------------------------------------------------------------------

def _slug(family: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")


def _google_ttf_url(family: str) -> str:
    """The TrueType URL Google Fonts serves for ``family``, boldest-first."""
    import requests

    name = family   # requests encodes the spaces; a hand-made "+" would arrive as %2B
    for weight in _WEIGHTS:
        spec = f"{name}:wght@{weight}" if weight else name
        try:
            r = requests.get("https://fonts.googleapis.com/css2", params={"family": spec},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        except requests.RequestException as exc:
            raise FontError(f"Couldn't reach Google Fonts: {exc}") from exc
        if r.status_code != 200:
            continue
        m = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)", r.text)
        if m:
            return m.group(1)
    raise FontError(f"Google Fonts has no font called “{family}”")


def font_file(family: str) -> Path:
    """A TrueType file for ``family`` on this machine: the bundled one for
    Funnel Display, else Google's, downloaded once into data/fonts/."""
    family = clean_family(family)
    if family == BUNDLED_FAMILY and BUNDLED_FILE.exists():
        return BUNDLED_FILE
    dest = FONT_CACHE / f"{_slug(family)}.ttf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import requests

    with _fetch_lock:
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        url = _google_ttf_url(family)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise FontError(f"Couldn't download “{family}”: {exc}") from exc
        FONT_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
    log.info("Fetched the %s font from Google Fonts", family)
    return dest


# --- What the renderers use ---------------------------------------------------------------

def _contrast_text(box_hex: str) -> str:
    """Black or white, whichever reads better on ``box_hex`` (WCAG luminance)."""
    r, g, b = (int(box_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#111111" if lum > 0.36 else "#FFFFFF"


def resolve(settings) -> dict:
    """The font file and colors to render with: the account's choice where it
    made one, settings.yaml's ``subtitles`` section for the rest. A saved
    font that can't be had right now falls back to the configured one rather
    than failing the render."""
    look = load()
    font_path: Path | str = settings.get("subtitles.font_file", "FunnelDisplay-SemiBold.ttf")
    if look["font"]:
        try:
            font_path = font_file(look["font"])
        except FontError as exc:
            log.warning("Caption font %s unavailable (%s); using the default", look["font"], exc)
    box = settings.get("subtitles.highlight_box_color", "#FFFFFF")
    box_text = settings.get("subtitles.highlight_text_color", "#1A4A7D")
    if look["box"]:
        box, box_text = look["box"], _contrast_text(look["box"])
    return {"font": font_path, "text": settings.get("subtitles.text_color", "#FFFFFF"),
            "box": box, "box_text": box_text}
