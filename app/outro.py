"""Post-roll outro: dissolve the end of every published clip into a branded
CTA card, with a label tab riding on the footage until then.

The card is drawn frame by frame with Pillow and piped into ffmpeg, which
xfades it onto the clip. Design lives in the workspace's config/outro.yaml (no
file, or ``enabled: false``, means no outro); iterate on it with the live
preview in lab/outro/.

``with_outro`` is the entry point for publishing: it returns a cached copy of a
clip with the outro appended, rendering it on first use.
"""
from __future__ import annotations

import array
import bisect
import copy
import glob
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

import yaml

from .config import CONFIG_DIR, DATA_DIR, ROOT

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont
except ImportError:  # the headless scheduler image; it never holds clips to render
    Image = None

log = logging.getLogger("outro")

FONT_DIR = ROOT / "assets" / "fonts"
CONFIG_FILE = CONFIG_DIR / "outro.yaml"
OUTRO_DIR = DATA_DIR / "outro"
MAX_DURATION = 4.0
# Bump when a code change alters the rendered output, so cached copies rebuild.
ENGINE_VERSION = 1

ENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-c:a", "aac", "-b:a", "128k",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
]


# ── config ────────────────────────────────────────────────────────────────

def _merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else v
        return out
    if isinstance(base, list) and isinstance(over, dict):
        out = list(base)
        for k, v in over.items():
            out[int(k)] = _merge(out[int(k)], v)
        return out
    return over


def enabled(config_path: Path = CONFIG_FILE) -> bool:
    if not config_path.exists():
        return False
    raw = yaml.safe_load(config_path.read_text()) or {}
    return bool(raw.get("enabled", True))


def load_config(orientation: str, config_path: Path = CONFIG_FILE) -> dict:
    raw = yaml.safe_load(config_path.read_text()) or {}
    overrides = (raw.pop("overrides", None) or {}).get(orientation) or {}
    cfg = _merge(raw, copy.deepcopy(overrides))
    duration = float(cfg.get("duration", 3.5))
    dissolve = float(cfg.get("dissolve", 0.7))
    if duration > MAX_DURATION:
        raise ValueError(f"duration {duration}s is over the {MAX_DURATION}s cap")
    if not 0 < dissolve < duration:
        raise ValueError("dissolve must be > 0 and shorter than duration")
    cfg["_base"] = str(config_path.parent)  # image paths in the config resolve from here
    return cfg


# ── card rendering ────────────────────────────────────────────────────────

def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _background(w: int, h: int, bg: dict) -> Image.Image:
    color = _rgb(bg.get("color", "#1A4A7D"))
    style = bg.get("style", "flat")
    if style == "flat":
        return Image.new("RGBA", (w, h), color + (255,))
    edge = _rgb(bg.get("edge_color", "#0E2F55"))
    if style == "radial":
        side = max(w, h)
        mask = Image.radial_gradient("L").resize((side, side), Image.BICUBIC)
        mask = mask.crop(((side - w) // 2, (side - h) // 2, (side + w) // 2, (side + h) // 2))
    else:
        mask = Image.linear_gradient("L").resize((w, h), Image.BICUBIC)
    img = Image.composite(Image.new("RGB", (w, h), edge), Image.new("RGB", (w, h), color), mask)
    # A ramp between two close colors has only a few dozen 8-bit steps, which
    # show as rings; a level or two of noise dithers them away.
    sigma = float(bg.get("dither", 1.5))
    if sigma > 0:
        noise = Image.effect_noise((w, h), sigma).convert("L")
        img = ImageChops.add(img, Image.merge("RGB", (noise, noise, noise)), 1.0, -128)
    return img.convert("RGBA")


def _font(name: str, px: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Font missing: {path}")
    font = ImageFont.truetype(str(path), max(8, int(px)))
    if weight is not None:
        font.set_variation_by_axes([float(weight)])
    return font


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def _fit(text: str, font_name: str, px: int, max_w: float, max_lines: int, weight: int | None = None):
    """Largest font (down to 55% of ``px``) whose wrap fits ``max_lines``."""
    for scale in [1 - i * 0.03 for i in range(16)]:
        font = _font(font_name, px * scale, weight)
        lines = _wrap(text, font, max_w)
        if len(lines) <= max_lines and all(font.getlength(ln) <= max_w for ln in lines):
            return font, lines
    return font, lines


def _layer(el: dict, short: int, max_w: float, align: str, base: Path) -> Image.Image:
    kind = el.get("type", "text")
    color = _rgb(el.get("color", "#FFFFFF"))
    px = float(el.get("size", 0.05)) * short

    if kind == "rule":
        w = int(float(el.get("width", 0.12)) * short)
        t = max(1, int(float(el.get("thickness", 0.008)) * short))
        img = Image.new("RGBA", (w, t))
        ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, t - 1), radius=t // 2, fill=color + (255,))
        return img

    if kind == "logo":
        img = Image.open(base / el["file"]).convert("RGBA")
        if el.get("mask") == "circle":
            # Supersampled so the cut edge is antialiased; `inset` trims a
            # fraction off each side to drop a square backdrop's fringe.
            ss = 4
            inset = float(el.get("inset", 0.0)) * min(img.size) * ss
            mask = Image.new("L", (img.width * ss, img.height * ss))
            ImageDraw.Draw(mask).ellipse((inset, inset, mask.width - 1 - inset, mask.height - 1 - inset), fill=255)
            img.putalpha(ImageChops.multiply(img.getchannel("A"), mask.resize(img.size, Image.LANCZOS)))
        target_h = int(float(el.get("height", 0.09)) * short)
        return img.resize((max(1, round(img.width * target_h / img.height)), target_h), Image.LANCZOS)

    font_name = el.get("font", "FunnelDisplay-Bold.ttf")
    if kind == "pill":
        font = _font(font_name, px)
        pad_x, pad_y = px * float(el.get("pad_x", 0.55)), px * float(el.get("pad_y", 0.32))
        left, top, right, bottom = font.getbbox(el["text"], anchor="ls")
        w = int(right - left + 2 * pad_x)
        h = int(bottom - top + 2 * pad_y)
        img = Image.new("RGBA", (w, h))
        d = ImageDraw.Draw(img)
        radius = h * float(el.get("radius", 0.5))
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=_rgb(el.get("box_color", "#FFFFFF")) + (255,))
        d.text((pad_x - left, pad_y - top), el["text"], font=font, fill=color + (255,), anchor="ls")
        return img

    font, lines = _fit(el["text"], font_name, px, max_w, int(el.get("max_lines", 2)), el.get("weight"))
    ascent, descent = font.getmetrics()
    line_h = (ascent + descent) * float(el.get("line_spacing", 1.02))
    w = int(max_w) if align in ("center", "right") else int(max(font.getlength(ln) for ln in lines)) + 2
    h = int(line_h * (len(lines) - 1) + ascent + descent)
    img = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(img)
    x, anchor = {"center": (w / 2, "ma"), "right": (w, "ra")}.get(align, (0, "la"))
    for i, line in enumerate(lines):
        d.text((x, i * line_h), line, font=font, fill=color + (255,), anchor=anchor)
    return img


class Typed:
    """One line typed out a character at a time, followed by a text cursor.

    The cursor blinks before typing starts and once it finishes, and holds solid
    while keys are landing, the way a real caret does."""

    def __init__(self, el: dict, short: int, max_w: float):
        px = float(el.get("size", 0.1)) * short
        self.cursor_w = max(2, round(px * float(el.get("cursor_width", 0.07))))
        self.gap = px * float(el.get("cursor_gap", 0.05))
        self.text = el["text"]
        self.font, _ = _fit(self.text, el.get("font", "FunnelDisplay-Bold.ttf"), px,
                            max_w - self.gap - self.cursor_w, 1, el.get("weight"))
        self.color = _rgb(el.get("color", "#FFFFFF")) + (255,)
        self.cursor_color = _rgb(el.get("cursor_color", el.get("color", "#FFFFFF"))) + (255,)
        ascent, descent = self.font.getmetrics()
        self.cursor_top = self.font.getbbox("l", anchor="la")[1]
        self.cursor_bottom = ascent + descent * 0.2
        self.width = int(self.font.getlength(self.text) + self.gap + self.cursor_w) + 1
        self.height = ascent + descent

        # Slightly uneven keystrokes read as a person typing rather than a ticker.
        rng = random.Random(self.text)
        cps, jitter = float(el.get("cps", 12)), float(el.get("jitter", 0.35))
        t, self.times = float(el.get("delay", 0.8)), []
        for _ in self.text:
            self.times.append(t)
            t += (1 / cps) * (1 + jitter * (rng.random() * 2 - 1))
        self.blink = float(el.get("blink", 1.0))
        self.solid_until = self.times[-1] + float(el.get("solid_after", 0.4))

    def draw(self, out: Image.Image, x: float, y: float, t: float) -> None:
        prefix = self.text[:bisect.bisect_right(self.times, t)]
        d = ImageDraw.Draw(out)
        if prefix:
            d.text((x, y), prefix, font=self.font, fill=self.color, anchor="la")
        typing = self.times[0] <= t < self.solid_until
        if typing or (t - self.solid_until) % self.blink < self.blink / 2:
            cx = x + self.font.getlength(prefix) + (self.gap if prefix else 0)
            d.rectangle((round(cx), round(y + self.cursor_top),
                         round(cx) + self.cursor_w - 1, round(y + self.cursor_bottom)),
                        fill=self.cursor_color)


def render_tab(tab: dict, short: int) -> Image.Image:
    """A label tab hanging from the frame's top edge: square top corners,
    rounded bottom ones. Placed at (``x``, ``y``) fractions of the frame."""
    px = float(tab.get("size", 0.034)) * short
    font = _font(tab.get("font", "FunnelDisplay-Bold.ttf"), px, tab.get("weight"))
    pad_x, pad_y = px * float(tab.get("pad_x", 0.6)), px * float(tab.get("pad_y", 0.38))
    left, _top, right, _bottom = font.getbbox(tab["text"], anchor="ls")
    text_h = -font.getbbox("H", anchor="ls")[1]
    w = int(pad_x + (right - left) + pad_x)
    h = int(text_h + 2 * pad_y)
    img = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(img)
    alpha = round(255 * float(tab.get("opacity", 1.0)))
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h * float(tab.get("radius", 0.3)),
                        fill=_rgb(tab.get("box_color", "#1A4A7D")) + (alpha,),
                        corners=(False, False, True, True))
    # Centered on cap height so lowercase descenders hang naturally below.
    d.text((pad_x - left, pad_y + text_h), tab["text"], font=font,
           fill=_rgb(tab.get("color", "#FFFFFF")) + (255,), anchor="ls")
    return img


def _keystroke(rng: random.Random, sr: int) -> list[float]:
    """One synthetic mechanical key press: a bright click, a low thock, and the
    softer tick of the key bottoming out a few milliseconds later."""
    n = int(0.06 * sr)
    out = [0.0] * n
    pitch = rng.uniform(150, 230)
    bottom = int(rng.uniform(0.008, 0.016) * sr)
    gain = rng.uniform(0.75, 1.0)
    prev_x = lp = 0.0
    alpha = 0.55  # one-pole lowpass on differentiated noise ~ a 2-6 kHz band
    for i in range(n):
        t = i / sr
        x = rng.uniform(-1, 1)
        hp, prev_x = x - prev_x, x
        lp += alpha * (hp - lp)
        click = lp * math.exp(-t / 0.0035) * min(1.0, t / 0.0004)
        thock = 0.55 * math.sin(2 * math.pi * pitch * t) * math.exp(-t / 0.012)
        tick = 0.0
        if i >= bottom:
            tb = (i - bottom) / sr
            tick = 0.4 * lp * math.exp(-tb / 0.002)
        out[i] = gain * (click + thock + tick)
    return out


def typing_track(times: list[float], duration: float, dest: Path, *,
                 volume: float = 0.5, seed: str = "", sr: int = 48000) -> Path:
    """Mono WAV ``duration`` long with a keystroke at each of ``times``."""
    rng = random.Random(seed)
    buf = [0.0] * int(duration * sr)
    for t in times:
        start = int(t * sr)
        for i, v in enumerate(_keystroke(rng, sr)):
            if start + i < len(buf):
                buf[start + i] += v
    peak = max((abs(v) for v in buf), default=0) or 1.0
    scale = 32767 * volume / peak
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(array.array("h", (int(v * scale) for v in buf)).tobytes())
    return dest


class Card:
    """Precomputed background and element layers; ``frame(t)`` composites one moment."""

    def __init__(self, w: int, h: int, cfg: dict):
        self.w, self.h = w, h
        short = min(w, h)
        layout = cfg.get("layout") or {}
        align = layout.get("align", "center")
        max_w = min(float(layout.get("safe_frac", 0.8)) * w,
                    float(layout.get("max_width", 1.1)) * short)
        self.bg = _background(w, h, cfg.get("background") or {})

        self.items = []
        y = 0.0
        for el in cfg.get("elements") or []:
            kind = el.get("type", "text")
            if kind == "typed":
                img, typed = None, Typed(el, short, max_w)
                el_w, el_h = typed.width, typed.height
            else:
                img, typed = _layer(el, short, max_w, align, Path(cfg["_base"])), None
                el_w, el_h = img.size
            x = {"center": (w - el_w) / 2,
                 "right": (w + max_w) / 2 - el_w}.get(align, (w - max_w) / 2)
            self.items.append({
                "img": img, "typed": typed, "x": x, "y": y, "kind": kind, "align": align,
                # No `delay` means the element is simply on the card from the first frame.
                "delay": float(el["delay"]) if "delay" in el and kind != "typed" else None,
                "fade": max(0.01, float(el.get("fade", 0.45))),
                "rise": float(el.get("rise", 0.022)) * short,
                "opacity": float(el.get("opacity", 1.0)),
            })
            y += el_h + float(el.get("gap", 0.03)) * short
        self.keystrokes = sorted(t for it in self.items if it["typed"] for t in it["typed"].times)
        stack_h = y - (float((cfg.get("elements") or [{}])[-1].get("gap", 0.03)) * short if self.items else 0)
        top = float(layout.get("anchor_y", 0.5)) * h - stack_h / 2
        for it in self.items:
            it["y"] += top

    def frame(self, t: float) -> Image.Image:
        out = self.bg.copy()
        for it in self.items:
            if it["typed"]:
                it["typed"].draw(out, it["x"], it["y"], t)
                continue
            p = 1.0 if it["delay"] is None else _ease_out((t - it["delay"]) / it["fade"])
            if p <= 0:
                continue
            img = it["img"]
            if it["kind"] == "rule":
                full, cw = img.width, max(1, int(img.width * p))
                left = {"center": (full - cw) // 2, "right": full - cw}.get(it["align"], 0)
                img = img.crop((left, 0, left + cw, img.height))
                x, y = it["x"] + left, it["y"]
            else:
                x, y = it["x"], it["y"] + it["rise"] * (1 - p)
            alpha = p * it["opacity"]
            if alpha < 1:
                img = img.copy()
                img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha)))
            out.alpha_composite(img, (int(round(x)), int(round(y))))
        return out.convert("RGB")


# ── ffmpeg ────────────────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {
        "w": int(video["width"]), "h": int(video["height"]),
        "fps": Fraction(video["r_frame_rate"]),
        "duration": float(info["format"]["duration"]),
        "audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def render(clip: Path, dest: Path, *, lead_in: float | None = None,
           config_path: Path = CONFIG_FILE, card_png: Path | None = None) -> dict:
    """Write ``clip`` with the outro appended to ``dest``.

    ``lead_in`` keeps only that many seconds of clip before the dissolve (the
    lab's quick previews); None keeps the whole clip. ``card_png`` saves the
    card's final frame. Returns the timing of the result."""
    meta = probe(clip)
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    orientation = "vertical" if h > w else "landscape"
    cfg = load_config(orientation, config_path)
    duration, dissolve = float(cfg["duration"]), float(cfg["dissolve"])

    lead = meta["duration"] if lead_in is None else min(lead_in + dissolve, meta["duration"])
    if lead <= dissolve:
        raise ValueError(f"{clip.name} is too short ({meta['duration']:.1f}s) for a {dissolve}s dissolve")
    offset = lead - dissolve
    total = lead + duration - dissolve
    n_frames = round(duration * fps)
    fps_s = f"{fps.numerator}/{fps.denominator}"
    card = Card(w, h, cfg)

    with tempfile.TemporaryDirectory(prefix="outro_") as tmpdir:
        tmpdir = Path(tmpdir)
        seek = [] if lead_in is None else ["-sseof", f"-{lead:.3f}"]
        afade = f",afade=t=out:st={offset:.3f}:d={dissolve:.3f}" if cfg.get("audio_fade", True) else ""
        stereo = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
        audio = (
            f"[0:a]{stereo},asetpts=PTS-STARTPTS,apad,atrim=duration={total:.3f}{afade}[ca]"
            if meta["audio"] else f"anullsrc=r=48000:cl=stereo,atrim=duration={total:.3f},{stereo}[ca]"
        )
        extra_inputs: list[str] = []
        sound = cfg.get("typing_sound") or {}
        # Keys sound on the frame their letter appears, not between frames.
        keystrokes = [math.ceil(k * fps - 1e-6) / float(fps) for k in card.keystrokes]
        if sound.get("enabled", True) and keystrokes:
            track = typing_track(keystrokes, duration, tmpdir / "typing.wav",
                                 volume=float(sound.get("volume", 0.35)), seed=clip.stem)
            idx = 2 + len(extra_inputs) // 2
            extra_inputs += ["-i", str(track)]
            audio += (f";[{idx}:a]{stereo},adelay={offset * 1000:.0f}:all=1,apad,"
                      f"atrim=duration={total:.3f}[typ]"
                      ";[ca][typ]amix=inputs=2:normalize=0:duration=first[a]")
        else:
            audio += ";[ca]anull[a]"

        clip_chain = (f"[0:v]fps={fps_s},scale={w}:{h},setsar=1,setpts=PTS-STARTPTS,"
                      f"tpad=stop_mode=clone:stop_duration=1,trim=duration={lead:.3f}")
        tab = cfg.get("tab") or {}
        if tab.get("enabled", True) and tab.get("text"):
            tab_png = tmpdir / "tab.png"
            render_tab(tab, min(w, h)).save(tab_png)
            idx = 2 + len(extra_inputs) // 2
            extra_inputs += ["-i", str(tab_png)]
            clip_chain += (f"[c0];[c0][{idx}:v]overlay=x={round(float(tab.get('x', 0.07)) * w)}"
                           f":y={round(float(tab.get('y', 0.0)) * h)}:eof_action=repeat")
        graph = ";".join([
            f"{clip_chain},format=yuv420p,settb=AVTB[clip]",
            "[1:v]setsar=1,format=yuv420p,settb=AVTB[card]",
            f"[clip][card]xfade=transition={cfg.get('transition', 'fade')}:"
            f"duration={dissolve:.3f}:offset={offset:.3f},format=yuv420p[v]",
            audio,
        ])

        tmp_out = tmpdir / "out.mp4"
        # stderr goes to a file, not a pipe: a damaged clip can log decode
        # errors faster than anyone reads them, and a full stderr pipe stalls
        # ffmpeg while we block feeding it frames — a deadlock, not a failure.
        err_path = tmpdir / "ffmpeg.err"
        with open(err_path, "wb") as err_file:
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 *seek, "-i", str(clip),
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps_s, "-i", "-",
                 *extra_inputs,
                 "-filter_complex", graph, "-map", "[v]", "-map", "[a]", *ENCODE_ARGS, str(tmp_out)],
                stdin=subprocess.PIPE, stderr=err_file)
            last = None
            try:
                for i in range(n_frames):
                    last = card.frame(i / float(fps))
                    proc.stdin.write(last.tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            code = proc.wait(timeout=900)
        if code != 0:
            err = err_path.read_text(errors="replace")
            raise RuntimeError(f"ffmpeg failed adding the outro to {clip.name}: {err[-800:]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Moved into place whole, so a concurrent reader never sees a partial file.
        staged = dest.with_name(f".{dest.name}.{os.getpid()}.part")
        shutil.move(tmp_out, staged)
        os.replace(staged, dest)

    if card_png is not None and last is not None:
        last.save(card_png)
    return {
        "orientation": orientation, "w": w, "h": h,
        "outro_start": round(offset, 3), "total": round(total, 3),
        "duration": duration, "dissolve": dissolve,
        "added_seconds": round(duration - dissolve, 3),
    }


def _cache_key(clip: Path, config_path: Path) -> str:
    st = clip.stat()
    digest = hashlib.sha1()
    for part in (str(ENGINE_VERSION), config_path.read_text(), str(clip.resolve()),
                 str(st.st_size), str(st.st_mtime_ns)):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def is_outro_copy(path: str | Path) -> bool:
    return Path(path).resolve().parent == OUTRO_DIR.resolve()


def with_outro(clip_path: str | Path, *, config_path: Path = CONFIG_FILE) -> Path:
    """``clip_path`` with the outro appended, cached under data/outro/.

    Returns the clip itself when the outro is disabled, already applied, or
    fails to render: a publish never fails over its outro. The cache key covers
    the source file, the design, and ENGINE_VERSION, so editing outro.yaml or
    re-exporting the clip renders a fresh copy."""
    clip = Path(clip_path).expanduser()
    try:
        if Image is None or not enabled(config_path) or is_outro_copy(clip):
            return clip
        dest = OUTRO_DIR / f"{clip.stem}_outro_{_cache_key(clip, config_path)}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        info = render(clip, dest, config_path=config_path)
        for stale in OUTRO_DIR.glob(f"{glob.escape(clip.stem)}_outro_*.mp4"):
            if stale != dest:
                stale.unlink(missing_ok=True)
        log.info("Added %.1fs outro to %s -> %s", info["added_seconds"], clip.name, dest.name)
        return dest
    except Exception:
        log.exception("Outro failed for %s; using the clip without it", clip.name)
        return clip
