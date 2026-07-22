"""Burned-in, word-by-word stylized captions for exported clips.

Pipeline: faster-whisper (word timestamps) -> Pillow renders one transparent
PNG per "caption state" (current word highlighted + popped) in Funnel Display
(the Renewables.org brand font, bundled in assets/fonts) -> ffmpeg overlays
the PNG sequence onto the clip via the concat demuxer.

Pillow does the text rendering because the system ffmpeg is built without
libass/freetype; this also gives full control over the style. Captions are
optional per clip — generation writes a separate ``*_subs.mp4`` next to the
original and the operator chooses which file to post.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .clipper import CLIPS_DIR, ClipExportError, _run_ffmpeg
from .config import ROOT, load_settings

log = logging.getLogger("subtitles")

FONT_DIR = ROOT / "assets" / "fonts"

# Fraction of the video height used for the caption strip (rendered PNGs are
# strip-sized, not full-frame, to keep the temp files small).
STRIP_FRAC = 0.42


class SubtitleError(RuntimeError):
    pass


def _hex_to_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    v = value.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16), alpha)


def _video_size(path: str | Path) -> tuple[int, int]:
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        w, h = proc.stdout.strip().split(",")
        return int(w), int(h)
    except Exception as exc:
        raise SubtitleError(f"Could not probe video size: {proc.stdout!r}") from exc


def transcribe_words(clip_path: str | Path) -> list[dict]:
    """Word-level timestamps for the exported clip: [{word, start, end}]."""
    from .scrape import _get_whisper_model

    settings = load_settings()
    model = _get_whisper_model(settings)
    segments, _info = model.transcribe(str(clip_path), word_timestamps=True, vad_filter=True)
    words: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            text = (w.word or "").strip()
            if text:
                words.append({"word": text, "start": float(w.start), "end": float(w.end)})
    return words


def group_words(words: list[dict], max_words: int = 4, max_gap: float = 0.8) -> list[list[dict]]:
    """Split the word stream into short display groups (one on-screen line)."""
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and (
            len(cur) >= max_words
            or w["start"] - cur[-1]["end"] > max_gap
            or cur[-1]["word"][-1] in ".?!"
        ):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def _load_fonts(px: int, font_name: str) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    font_file = FONT_DIR / font_name
    if not font_file.exists():
        raise SubtitleError(f"Caption font missing: {font_file}")
    base = ImageFont.truetype(str(font_file), px)
    big = ImageFont.truetype(str(font_file), int(px * 1.08))  # active-word "pop"
    return base, big


def _render_state(texts: list[str], active: int, width: int, strip_h: int,
                  fonts: tuple, colors: dict, position: str = "bottom") -> Image.Image:
    """One caption state: the group's words, with the ``active`` word set on a
    solid rounded box in inverted colors (the "talks Renewables.org" look)."""
    from PIL import ImageFilter

    base_f, big_f = fonts
    img = Image.new("RGBA", (width, strip_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def word_font(i: int):
        return big_f if i == active else base_f

    ascent, descent = base_f.getmetrics()
    pad_x = int(base_f.size * 0.22)   # box side padding around the active word
    pad_y = int(base_f.size * 0.10)
    space = draw.textlength(" ", font=base_f) * 1.15
    widths = [draw.textlength(t, font=word_font(i)) for i, t in enumerate(texts)]
    # The box makes the active word occupy extra horizontal room.
    slots = [w + (2 * pad_x if i == active else 0) for i, w in enumerate(widths)]

    # Wrap to two lines when the single line would overflow the safe width.
    max_w = width * 0.92
    lines: list[list[int]] = [[i for i in range(len(texts))]]
    total = sum(slots) + space * (len(texts) - 1)
    if total > max_w and len(texts) > 1:
        split, acc = 1, slots[0]
        for i in range(1, len(texts)):
            if acc + space + slots[i] > total / 2:
                split = i
                break
            acc += space + slots[i]
        lines = [list(range(split)), list(range(split, len(texts)))]

    line_h = (ascent + descent) * 1.06
    if position == "top":
        # Mirror of the bottom layout: same edge margin, lines flow downward
        # from the top of the strip (which sits at the top of the frame).
        first = line_h * 0.55 - descent + ascent
        baselines = [first] if len(lines) == 1 else [first, first + line_h]
    else:
        baselines = ([strip_h - line_h * 0.55] if len(lines) == 1
                     else [strip_h - line_h * 1.55, strip_h - line_h * 0.55])

    # Soft drop shadow (separate blurred layer) keeps white text readable on
    # bright footage without the hard outline of the old style.
    shadow = Image.new("RGBA", (width, strip_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)

    for line, base_y in zip(lines, baselines):
        line_w = sum(slots[i] for i in line) + space * (len(line) - 1)
        x = (width - line_w) / 2
        for i in line:
            f = word_font(i)
            if i == active:
                x0 = x
                box = (x0, base_y - ascent - pad_y, x0 + slots[i], base_y + descent + pad_y)
                radius = int(base_f.size * 0.18)
                sdraw.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 170))
                draw.rounded_rectangle(box, radius=radius, fill=colors["box"])
                draw.text((x0 + pad_x, base_y), texts[i], font=f,
                          fill=colors["box_text"], anchor="ls")
            else:
                sdraw.text((x + 3, base_y + 3), texts[i], font=f,
                           fill=(0, 0, 0, 190), anchor="ls")
                draw.text((x, base_y), texts[i], font=f, fill=colors["text"], anchor="ls")
            x += slots[i] + space

    shadow = shadow.filter(ImageFilter.GaussianBlur(base_f.size * 0.06))
    return Image.alpha_composite(shadow, img)


def create_subtitled_clip(clip_path: str | Path, position: str | None = None) -> Path:
    """Generate ``<clip>_subs.mp4`` with burned-in word captions. Returns path.

    ``position`` ("top"/"bottom") overrides the ``subtitles.position`` setting
    for this run — the web UI passes the operator's per-clip choice here.
    """
    clip = Path(clip_path)
    if not clip.exists():
        raise SubtitleError(f"Clip not found: {clip}")

    settings = load_settings()
    uppercase = bool(settings.get("subtitles.uppercase", False))
    max_words = int(settings.get("subtitles.max_words_per_group", 3))
    font_frac = float(settings.get("subtitles.font_size_frac", 0.11))
    font_name = settings.get("subtitles.font_file", "FunnelDisplay-SemiBold.ttf")
    dwell = max(0.0, float(settings.get("subtitles.dwell_seconds", 2.0)))
    position = str(position or settings.get("subtitles.position", "bottom")).strip().lower()
    if position not in ("top", "bottom"):
        raise SubtitleError(f"subtitles.position must be 'top' or 'bottom', got {position!r}")
    colors = {
        "text": _hex_to_rgba(settings.get("subtitles.text_color", "#FFFFFF")),
        "box": _hex_to_rgba(settings.get("subtitles.highlight_box_color", "#FFFFFF")),
        "box_text": _hex_to_rgba(settings.get("subtitles.highlight_text_color", "#1A4A7D")),
    }

    words = transcribe_words(clip)
    if not words:
        raise SubtitleError("No speech detected in the clip — nothing to caption.")
    groups = group_words(words, max_words=max_words)

    width, height = _video_size(clip)
    strip_h = int(height * STRIP_FRAC)
    font_px = max(18, int(height * font_frac))
    fonts = _load_fonts(font_px, font_name)

    out = CLIPS_DIR / f"{clip.stem}_subs.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        blank = tmpdir / "blank.png"
        Image.new("RGBA", (width, strip_h), (0, 0, 0, 0)).save(blank)

        # Timeline of (png, duration) entries covering the whole clip.
        entries: list[tuple[Path, float]] = []
        t = 0.0
        n_png = 0
        for gi, group in enumerate(groups):
            texts = [w["word"].upper() if uppercase else w["word"] for w in group]
            g_start, g_end = group[0]["start"], group[-1]["end"]
            if g_start > t + 0.01:
                entries.append((blank, g_start - t))
            last_png: Path | None = None
            for i, w in enumerate(group):
                # A word stays highlighted until the next word starts (no flicker).
                end = group[i + 1]["start"] if i + 1 < len(group) else g_end
                dur = max(0.05, end - w["start"])
                png = tmpdir / f"s{n_png:04d}.png"
                _render_state(texts, i, width, strip_h, fonts, colors, position).save(png)
                entries.append((png, dur))
                last_png = png
                n_png += 1
            t = g_end
            # Hold the finished phrase on screen through short pauses so text
            # doesn't vanish the instant the speaker stops. Cap at ``dwell``,
            # or cut short when the next phrase is ready to take over.
            if last_png is not None and dwell > 0:
                next_start = (
                    groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
                )
                gap = (next_start - t) if next_start is not None else dwell
                hold = min(dwell, max(0.0, gap))
                if hold > 0.01:
                    entries.append((last_png, hold))
                    t += hold
        entries.append((blank, 1.0))

        concat = tmpdir / "list.txt"
        lines = ["ffconcat version 1.0"]
        for png, dur in entries:
            lines.append(f"file '{png}'")
            lines.append(f"duration {max(0.05, dur):.3f}")
        lines.append(f"file '{blank}'")  # concat demuxer needs a trailing entry
        concat.write_text("\n".join(lines) + "\n")

        try:
            _run_ffmpeg([
                "-i", str(clip),
                "-safe", "0", "-f", "concat", "-i", str(concat),
                "-filter_complex",
                f"[1:v]format=rgba[cap];[0:v][cap]overlay=x=0:y={0 if position == 'top' else height - strip_h}:eof_action=pass",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "copy", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out),
            ])
        except ClipExportError as exc:
            raise SubtitleError(str(exc)) from exc

    log.info("Burned captions into %s (%d words, %d groups)", out.name, len(words), len(groups))
    return out
