"""Burned-in, word-by-word stylized captions for exported clips.

Pipeline: faster-whisper (word timestamps) -> Pillow renders one transparent
PNG per "caption state" (current word highlighted + popped) in Funnel Display
(the Renewables.org brand font, bundled in assets/fonts) -> ffmpeg overlays
the PNG sequence onto the clip via the concat demuxer.

Pillow does the text rendering because the system ffmpeg is built without
libass/freetype; this also gives full control over the style. Captions are
optional per clip — generation writes a separate ``*_subs.mp4`` next to the
original and the operator chooses which file to post.

The same Whisper word stream is persisted next to the trimmed clip
(``*_transcript.json``) so Suggest caption / Copy transcript use what was
actually said in the export — not the source video's YouTube captions.
"""
from __future__ import annotations

import json
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


def clip_transcript_path_for(clip_path: str | Path) -> Path:
    """Sidecar JSON for the Whisper word stream of an exported trim."""
    clip = Path(clip_path)
    return clip.with_name(f"{clip.stem}_transcript.json")


def transcribe_words(clip_path: str | Path) -> list[dict]:
    """Word-level timestamps for the exported clip: [{word, start, end}]."""
    from .scrape import _get_whisper_model

    settings = load_settings()
    model = _get_whisper_model(settings)
    # language pinned to English: Whisper's auto-detect samples the first ~30s
    # and infamously misreads noisy/archival British audio as Welsh, then
    # "transcribes" the whole clip in fluent Welsh. All channel content is
    # English, so force it.
    segments, _info = model.transcribe(str(clip_path), language="en",
                                       word_timestamps=True, vad_filter=True)
    words: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            text = (w.word or "").strip()
            if text:
                words.append({"word": text, "start": float(w.start), "end": float(w.end)})
    return words


def save_clip_transcript(clip_path: str | Path, words: list[dict]) -> Path:
    """Persist Whisper words next to the trimmed clip; returns the JSON path."""
    path = clip_transcript_path_for(clip_path)
    path.write_text(json.dumps(words, indent=1))
    return path


def load_clip_words(transcript_path: str | Path) -> list[dict]:
    """Load a previously saved Whisper word stream."""
    try:
        data = json.loads(Path(transcript_path).read_text())
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [w for w in data if isinstance(w, dict) and (w.get("word") or "").strip()]


def slice_source_words(source_words: list[dict],
                       segments: list[dict]) -> list[dict]:
    """Map the full-video Whisper word stream onto an exported trim's timeline.

    ``segments`` are the cut's trim windows in LIST order — the order
    ``export_supercut`` joins them, which the trim editor lets drift out of
    time order — so each window's words land at the cumulative offset of the
    windows before it. A word belongs to a window when its midpoint falls
    inside it, so a word straddling a cut boundary lands in whichever window
    holds most of it. Timestamps are clamped to the window so a straddler
    can't overhang the joined clip's cut points.

    This is what lets an export reuse the archive-time transcription instead
    of running Whisper again on the trimmed file.
    """
    out: list[dict] = []
    offset = 0.0
    for seg in segments:
        try:
            seg_start, seg_end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        length = seg_end - seg_start
        if length <= 0:
            continue
        for w in source_words:
            try:
                w_start, w_end = float(w["start"]), float(w["end"])
                text = str(w.get("word", "")).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not text:
                continue
            mid = (w_start + w_end) / 2
            if not (seg_start <= mid < seg_end):
                continue
            start = max(0.0, w_start - seg_start)
            end = min(length, w_end - seg_start)
            if end <= start:
                continue
            out.append({"word": text,
                        "start": round(offset + start, 3),
                        "end": round(offset + end, 3)})
        offset += length
    return out


def ensure_clip_words(clip_path: str | Path,
                      transcript_path: str | Path | None = None) -> tuple[list[dict], Path]:
    """Load cached Whisper words for a trim, or transcribe and cache them.

    ``transcript_path`` is preferred when set (the cut's stored sidecar); otherwise
    the default sidecar next to ``clip_path`` is used.

    Exports of videos archived with a word sidecar write the trim's transcript
    up front (see ``_export_cut_in_thread``), so the transcription fallback
    here only runs for clips whose source predates archive-time word streams.
    """
    clip = Path(clip_path)
    path = Path(transcript_path) if transcript_path else clip_transcript_path_for(clip)
    if path.exists():
        words = load_clip_words(path)
        if words:
            return words, path
    words = transcribe_words(clip)
    if not words:
        raise SubtitleError("No speech detected in the clip — nothing to caption.")
    return words, save_clip_transcript(clip, words)


def words_to_lines(words: list[dict], max_words: int = 12) -> list[dict]:
    """Group Whisper words into readable lines: [{start, text, clip_start}]."""
    groups = group_words(words, max_words=max_words)
    lines: list[dict] = []
    for group in groups:
        text = " ".join(w["word"] for w in group).strip()
        if not text:
            continue
        start = float(group[0]["start"])
        lines.append({"start": start, "text": text, "clip_start": round(start, 2)})
    return lines


def words_to_plain(words: list[dict], max_words: int = 12) -> str:
    """Newline-joined readable transcript from a Whisper word stream."""
    return "\n".join(line["text"] for line in words_to_lines(words, max_words=max_words))


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


def _fonts_at_px(font_path: str | Path, px: int) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    px = max(12, int(px))
    base = ImageFont.truetype(str(font_path), px)
    big = ImageFont.truetype(str(font_path), int(px * 1.08))  # active-word "pop"
    return base, big


def _load_fonts(px: int, font_name: str) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    font_file = FONT_DIR / font_name
    if not font_file.exists():
        raise SubtitleError(f"Caption font missing: {font_file}")
    return _fonts_at_px(font_file, px)


def _measure_slots(texts: list[str], fonts: tuple,
                   active: int) -> tuple[list[float], float, int]:
    """Horizontal room each word needs, the inter-word space, and box padding.

    The active word is set in the larger font and sits on a rounded box, so it
    claims its own width plus padding on both sides."""
    base_f, big_f = fonts
    pad_x = int(base_f.size * 0.22)
    slots = [
        (big_f if i == active else base_f).getlength(t) + (2 * pad_x if i == active else 0)
        for i, t in enumerate(texts)
    ]
    return slots, base_f.getlength(" ") * 1.15, pad_x


def _line_width(line: list[int], slots: list[float], space: float) -> float:
    return sum(slots[i] for i in line) + space * max(0, len(line) - 1)


def _layout_lines(slots: list[float], space: float, max_w: float,
                  max_lines: int, align: str) -> list[list[int]]:
    """Group word indices into display lines.

    ``align="left"`` wraps greedily, filling each line to ``max_w``.
    ``align="center"`` keeps one line, split into two balanced ones on overflow.

    Neither mode can always honor ``max_w``: once the line budget is spent the
    remaining words have nowhere to go but the final line. Callers size the font
    with ``fit_fonts`` first so that case doesn't arise."""
    if align == "left":
        lines: list[list[int]] = [[]]
        acc = 0.0
        for i in range(len(slots)):
            add = slots[i] if not lines[-1] else slots[i] + space
            if lines[-1] and acc + add > max_w and len(lines) < max_lines:
                lines.append([i])
                acc = slots[i]
            else:
                lines[-1].append(i)
                acc += add
        return lines

    lines = [list(range(len(slots)))]
    total = _line_width(lines[0], slots, space)
    if total > max_w and len(slots) > 1:
        split, acc = 1, slots[0]
        for i in range(1, len(slots)):
            if acc + space + slots[i] > total / 2:
                split = i
                break
            acc += space + slots[i]
        lines = [list(range(split)), list(range(split, len(slots)))]
    return lines


def _block_height(n_lines: int, base_f: ImageFont.FreeTypeFont) -> float:
    """Ink height of an ``n_lines`` block, matching _render_state's baselines."""
    ascent, descent = base_f.getmetrics()
    line_h = (ascent + descent) * 1.06
    return line_h * 0.55 + ascent + (n_lines - 1) * line_h


def _phrase_fits(texts: list[str], fonts: tuple, max_w: float, strip_h: int,
                 max_lines: int, align: str) -> bool:
    """True when every state of the phrase stays inside the strip.

    Checked for all active words, not just one: the highlight box widens
    whichever word is current, so a phrase can fit early in its animation and
    overflow later."""
    for active in range(len(texts)):
        slots, space, _pad = _measure_slots(texts, fonts, active)
        lines = _layout_lines(slots, space, max_w, max_lines, align)
        if _block_height(len(lines), fonts[0]) > strip_h:
            return False
        if any(_line_width(ln, slots, space) > max_w for ln in lines):
            return False
    return True


def fit_fonts(texts: list[str], fonts: tuple, *, width: int, strip_h: int,
              max_lines: int, align: str, safe_frac: float,
              min_frac: float = 0.62) -> tuple:
    """Shrink the caption font until the whole phrase fits the safe rail.

    Without this a long phrase silently runs off the side of the frame, because
    wrapping can only break a line while lines remain in the budget. Returns the
    configured ``fonts`` untouched whenever the phrase already fits — most do —
    and never shrinks past ``min_frac`` of the configured size, since unreadably
    small text is a worse outcome than a slight overhang.

    Callers apply the result per phrase, so the size holds steady while a phrase
    animates word by word.
    """
    if not texts:
        return fonts
    base_f = fonts[0]
    max_w = width * max(0.4, min(1.0, safe_frac))
    start_px = base_f.size
    min_px = max(18, int(start_px * min_frac))
    px = start_px
    while True:
        trial = fonts if px == start_px else _fonts_at_px(base_f.path, px)
        if px <= min_px or _phrase_fits(texts, trial, max_w, strip_h, max_lines, align):
            if px != start_px:
                log.debug("captions: shrank %dpx -> %dpx to fit %r", start_px, px,
                          " ".join(texts)[:60])
            return trial
        px = max(min_px, int(px * 0.94))


def _render_state(texts: list[str], active: int, width: int, strip_h: int,
                  fonts: tuple, colors: dict, position: str = "bottom",
                  max_lines: int = 2, align: str = "center",
                  safe_frac: float = 0.92) -> Image.Image:
    """One caption state: the group's words, with the ``active`` word set on a
    solid rounded box in inverted colors (the "talks Renewables.org" look).

    ``align="center"`` (the 16:9 export) keeps the original layout: one
    centered line, split into two balanced lines only on overflow.
    ``align="left"`` (the vertical composite) wraps greedily instead — each
    line fills the safe width and breaks naturally, up to ``max_lines``.

    ``safe_frac`` is the share of ``width`` text may occupy. The vertical
    composite passes a smaller value than the 16:9 export because Instagram
    crops the sides of a 9:16 frame on phones taller than 16:9."""
    from PIL import ImageFilter

    base_f, big_f = fonts
    img = Image.new("RGBA", (width, strip_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def word_font(i: int):
        return big_f if i == active else base_f

    ascent, descent = base_f.getmetrics()
    pad_y = int(base_f.size * 0.10)
    slots, space, pad_x = _measure_slots(texts, fonts, active)

    max_w = width * max(0.4, min(1.0, safe_frac))
    lines = _layout_lines(slots, space, max_w, max_lines, align)

    line_h = (ascent + descent) * 1.06
    if position == "top":
        # Lines flow downward from the top of the strip (which sits at the
        # top of the frame).
        first = line_h * 0.55 - descent + ascent
        baselines = [first + k * line_h for k in range(len(lines))]
    else:
        # Lines stack upward from the bottom edge of the strip.
        last = strip_h - line_h * 0.55
        baselines = [last - (len(lines) - 1 - k) * line_h for k in range(len(lines))]

    # Soft drop shadow (separate blurred layer) keeps white text readable on
    # bright footage without the hard outline of the old style.
    shadow = Image.new("RGBA", (width, strip_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)

    for line, base_y in zip(lines, baselines):
        line_w = _line_width(line, slots, space)
        x = (width - max_w) / 2 if align == "left" else (width - line_w) / 2
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


def render_caption_concat(groups: list[list[dict]], tmpdir: Path, *, width: int,
                          strip_h: int, fonts: tuple, colors: dict, position: str,
                          uppercase: bool, dwell: float,
                          max_lines: int = 2, align: str = "center",
                          safe_frac: float = 0.92) -> Path:
    """Render one PNG per caption state plus an ffconcat list covering the whole
    clip: blank strips fill silences, and each finished phrase dwells on screen
    (up to ``dwell`` seconds, or until the next phrase). Returns the concat list
    path; the PNGs live in ``tmpdir``. Shared by the 16:9 burned-caption export
    and the vertical composite (which overlays the strip mid-frame)."""
    blank = tmpdir / "blank.png"
    Image.new("RGBA", (width, strip_h), (0, 0, 0, 0)).save(blank)

    # Timeline of (png, duration) entries covering the whole clip.
    entries: list[tuple[Path, float]] = []
    t = 0.0
    n_png = 0
    for gi, group in enumerate(groups):
        texts = [w["word"].upper() if uppercase else w["word"] for w in group]
        # Size the phrase once, then hold it for every state: a font chosen per
        # state would resize the text as the highlight moves along the line.
        g_fonts = fit_fonts(texts, fonts, width=width, strip_h=strip_h,
                            max_lines=max_lines, align=align, safe_frac=safe_frac)
        g_start, g_end = group[0]["start"], group[-1]["end"]
        if g_start > t + 0.01:
            entries.append((blank, g_start - t))
        last_png: Path | None = None
        for i, w in enumerate(group):
            # A word stays highlighted until the next word starts (no flicker).
            end = group[i + 1]["start"] if i + 1 < len(group) else g_end
            dur = max(0.05, end - w["start"])
            png = tmpdir / f"s{n_png:04d}.png"
            _render_state(texts, i, width, strip_h, g_fonts, colors, position,
                          max_lines=max_lines, align=align,
                          safe_frac=safe_frac).save(png)
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
    return concat


def create_subtitled_clip(clip_path: str | Path, position: str | None = None,
                          out_path: str | Path | None = None) -> Path:
    """Generate ``<clip>_subs.mp4`` with burned-in word captions. Returns path.

    ``position`` ("top"/"bottom") overrides the ``subtitles.position`` setting
    for this run — the web UI passes the operator's per-clip choice here.
    ``out_path`` overrides the default output name, letting callers version the
    file so a regeneration can't overwrite one a queued post already points at.
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

    # Reuse a cached Whisper pass when re-rendering at a new position; the
    # sidecar is also what Suggest caption / Copy transcript read from.
    words, _transcript_path = ensure_clip_words(clip)
    groups = group_words(words, max_words=max_words)

    width, height = _video_size(clip)
    strip_h = int(height * STRIP_FRAC)
    font_px = max(18, int(height * font_frac))
    fonts = _load_fonts(font_px, font_name)

    out = Path(out_path) if out_path else CLIPS_DIR / f"{clip.stem}_subs.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        concat = render_caption_concat(
            groups, tmpdir, width=width, strip_h=strip_h, fonts=fonts,
            colors=colors, position=position, uppercase=uppercase, dwell=dwell,
        )

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
