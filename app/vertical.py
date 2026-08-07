"""1080x1920 vertical composite for Instagram Reels.

Layout, top to bottom: an operator-written hook text block, the 16:9 trimmed
clip full-width, the word-by-word caption strip, and a bottom band kept clear
of the Reels UI overlays — all over a background made from the clip itself,
scaled to fill the frame, blurred and darkened. Text is rendered with Pillow
(hook block + the caption PNG stream shared with app/subtitles.py) and
composited in a single ffmpeg pass; every layout coordinate comes from the
``vertical:`` section of config/settings.yaml rather than being hardcoded in
the filter graph, so a different canvas (e.g. a 1:1 variant) is a settings
change, not a rewrite.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .clipper import CLIPS_DIR, ClipExportError, _run_ffmpeg
from .config import load_settings
from .models import utcnow
from .subtitles import (
    FONT_DIR,
    SubtitleError,
    _hex_to_rgba,
    _load_fonts,
    ensure_clip_words,
    group_words,
    render_caption_concat,
)

log = logging.getLogger("vertical")


class VerticalCompositeError(RuntimeError):
    pass


def _wrap_hook_lines(draw: ImageDraw.ImageDraw, text: str,
                     font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Greedy word-wrap honoring explicit newlines."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        words = raw_line.split()
        if not words:
            continue
        cur = words[0]
        for word in words[1:]:
            trial = f"{cur} {word}"
            if draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def render_hook_png(text: str, width: int, max_height: int, *, font_name: str,
                    font_px: int, color: str) -> Image.Image:
    """Render the hook text block: left-aligned, wrapped, shrunk to fit.

    Returns a transparent-background RGBA image exactly ``width`` wide whose
    height is the rendered block (<= ``max_height``). Rendering happens here in
    Pillow — never typed into ffmpeg drawtext — so the brand font, wrapping and
    sizing behave like the caption renderer.
    """
    text = (text or "").strip()
    if not text:
        raise VerticalCompositeError("Hook text is empty")
    font_file = FONT_DIR / font_name
    if not font_file.exists():
        raise VerticalCompositeError(f"Hook font missing: {font_file}")

    # Same safe width as the caption strip so both share one left rail.
    max_w = width * 0.92
    fill = _hex_to_rgba(color)
    probe = ImageDraw.Draw(Image.new("RGBA", (width, 8), (0, 0, 0, 0)))

    px = int(font_px)
    while True:
        font = ImageFont.truetype(str(font_file), px)
        lines = _wrap_hook_lines(probe, text, font, max_w)
        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * 1.12)
        block_h = line_h * len(lines)
        if block_h <= max_height or px <= 24:
            break
        # Too tall for the space above the video: step the size down and rewrap.
        px = max(24, int(px * 0.92))

    img = Image.new("RGBA", (width, block_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = (width - max_w) / 2  # same side gutter the wrap width leaves
    for i, line in enumerate(lines):
        draw.text((margin, i * line_h + ascent),
                  line, font=font, fill=fill, anchor="ls")
    return img


def create_vertical_composite(clip_path: str | Path, hook_text: str,
                              transcript_path: str | Path | None = None,
                              out_path: str | Path | None = None) -> Path:
    """Compose the trimmed 16:9 clip into a 1080x1920 Reels-ready mp4.

    Captions reuse the clip's cached Whisper word sidecar (``transcript_path``
    when the cut has one) — never re-transcribed. ``out_path`` overrides the
    default timestamped name so callers can version outputs.
    """
    clip = Path(clip_path)
    if not clip.exists():
        raise VerticalCompositeError(f"Clip not found: {clip}")

    settings = load_settings()
    width = int(settings.get("vertical.width", 1080))
    height = int(settings.get("vertical.height", 1920))
    bg_blur = int(settings.get("vertical.background_blur", 32))
    bg_brightness = float(settings.get("vertical.background_brightness", 0.35))
    hook_y = int(settings.get("vertical.hook_y", 150))
    video_y = int(settings.get("vertical.video_y", 560))
    caption_y = int(settings.get("vertical.caption_y", 1210))
    strip_h = int(settings.get("vertical.caption_strip_px", 360))
    safe_px = int(settings.get("vertical.bottom_safe_px", 350))
    hook_font = settings.get("vertical.hook_font_file", "FunnelDisplay-ExtraBold.ttf")
    hook_px = int(settings.get("vertical.hook_font_px", 84))
    hook_color = str(settings.get("vertical.hook_text_color", "#FFFFFF"))
    caption_px = int(settings.get("vertical.caption_font_px", 78))

    if caption_y + strip_h > height - safe_px:
        # Keep every overlay out of the platform-UI band rather than failing:
        # the strip is transparent apart from the text lines at its top.
        log.warning("vertical: caption strip (y=%d h=%d) crosses the %dpx safe "
                    "zone; captions may sit under Reels UI", caption_y, strip_h, safe_px)

    # Caption styling follows the subtitles section (same brand look), with the
    # size in canvas pixels since the strip no longer scales with clip height.
    # The group size and line count are the vertical's own: bigger phrases
    # than the 16:9 export, wrapped naturally across up to caption_max_lines.
    uppercase = bool(settings.get("subtitles.uppercase", False))
    max_words = int(settings.get("vertical.caption_max_words", 9))
    caption_lines = int(settings.get("vertical.caption_max_lines", 3))
    caption_font_name = settings.get("subtitles.font_file", "FunnelDisplay-SemiBold.ttf")
    dwell = max(0.0, float(settings.get("subtitles.dwell_seconds", 2.0)))
    colors = {
        "text": _hex_to_rgba(settings.get("subtitles.text_color", "#FFFFFF")),
        "box": _hex_to_rgba(settings.get("subtitles.highlight_box_color", "#FFFFFF")),
        "box_text": _hex_to_rgba(settings.get("subtitles.highlight_text_color", "#1A4A7D")),
    }

    try:
        words, _sidecar = ensure_clip_words(clip, transcript_path)
    except SubtitleError as exc:
        raise VerticalCompositeError(str(exc)) from exc
    groups = group_words(words, max_words=max_words)
    fonts = _load_fonts(max(18, caption_px), caption_font_name)

    hook_text = (hook_text or "").strip()
    hook_img: Image.Image | None = None
    if hook_text:
        hook_img = render_hook_png(
            hook_text, width, max(60, video_y - hook_y - 20),
            font_name=hook_font, font_px=hook_px, color=hook_color,
        )

    if out_path:
        out = Path(out_path)
    else:
        stamp = utcnow().strftime("%Y%m%dT%H%M%S")
        out = CLIPS_DIR / f"{clip.stem}_vertical_{stamp}.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        concat = render_caption_concat(
            groups, tmpdir, width=width, strip_h=strip_h, fonts=fonts,
            colors=colors, position="top", uppercase=uppercase, dwell=dwell,
            max_lines=caption_lines, align="left",
        )

        # Single pass. The background is the clip itself scaled to cover the
        # full frame, blurred and darkened; the sharp clip sits on top at its
        # configured offset, then the hook and caption stream are overlaid.
        # Blur trick: cover-scale to quarter resolution, boxblur there, and
        # upscale back — visually a heavy gaussian at a fraction of the cost.
        bw, bh = max(2, width // 4), max(2, height // 4)
        dim = max(0.0, min(1.0, bg_brightness))
        chains = [
            "[0:v]split=2[bgsrc][fgsrc]",
            f"[bgsrc]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
            f"crop={bw}:{bh},boxblur={max(1, bg_blur // 4)}:2,"
            f"scale={width}:{height},setsar=1,"
            f"colorchannelmixer=rr={dim}:gg={dim}:bb={dim}[bg]",
            f"[fgsrc]scale={width}:-2[fg]",
            f"[bg][fg]overlay=x=(main_w-overlay_w)/2:y={video_y}[base]",
            "[1:v]format=rgba[cap]",
        ]
        args = [
            "-i", str(clip),
            "-safe", "0", "-f", "concat", "-i", str(concat),
        ]
        current = "[base]"
        if hook_img is not None:
            hook_png = tmpdir / "hook.png"
            hook_img.save(hook_png)
            args += ["-i", str(hook_png)]
            chains.append("[2:v]format=rgba[hook]")
            chains.append(
                f"{current}[hook]overlay=x=(main_w-overlay_w)/2:y={hook_y}:"
                f"eof_action=repeat[withhook]"
            )
            current = "[withhook]"
        chains.append(f"{current}[cap]overlay=x=0:y={caption_y}:eof_action=pass[outv]")

        try:
            _run_ffmpeg([
                *args,
                "-filter_complex", ";".join(chains),
                "-map", "[outv]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(out),
            ])
        except ClipExportError as exc:
            raise VerticalCompositeError(str(exc)) from exc

    log.info("Composed vertical %s (%dx%d, %d caption groups, hook=%s)",
             out.name, width, height, len(groups), "yes" if hook_img else "no")
    return out
