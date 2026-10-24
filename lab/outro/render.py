"""Live design lab for the post-roll outro (the engine is app/outro.py).

    python lab/outro/render.py                 # render every clip in samples/
    python lab/outro/render.py --serve         # live preview at :8340, re-renders on save
    python lab/outro/render.py --grab CLIP     # add the last 10 s of CLIP to samples/
    python lab/outro/render.py --full CLIP     # whole CLIP + outro -> out/full_<name>.mp4

The design is the workspace's config/outro.yaml (WORKSPACE env, default
climate) -- the same file publishing uses, so what you see here is what ships.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from app import outro  # noqa: E402

CONFIG = outro.CONFIG_FILE
SAMPLES = LAB / "samples"
OUT = LAB / "out"
PORT = 8340
WATCHED_CODE = [Path(__file__), Path(outro.__file__)]


def render(clip: Path, dest: Path, *, full: bool = False) -> dict:
    lead_in = None if full else float((yaml.safe_load(CONFIG.read_text()) or {}).get("lead_in", 2.5))
    card_png = dest.with_name(dest.stem + "_card.png")
    info = outro.render(clip, dest, lead_in=lead_in, card_png=card_png)

    sheet = dest.with_name(dest.stem + "_sheet.jpg")
    cells = 8
    start = max(0.0, info["outro_start"] - 0.3)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(dest),
         "-vf", f"fps={cells / (info['total'] - start):.4f},"
                f"scale=-2:{480 if info['h'] > info['w'] else 270},"
                f"tile={cells}x1:padding=6:color=0x111111",
         "-frames:v", "1", str(sheet)], check=True)
    return {**info, "name": dest.stem, "video": f"out/{dest.name}",
            "card": f"out/{card_png.name}", "sheet": f"out/{sheet.name}"}


def samples() -> list[Path]:
    return sorted(p for p in SAMPLES.glob("*.mp4") if not p.name.startswith("."))


def render_all() -> dict:
    OUT.mkdir(exist_ok=True)
    started = time.time()
    manifest = {"version": started, "samples": [], "error": None}
    try:
        if not samples():
            raise RuntimeError("samples/ is empty; add one with --grab CLIP")
        for clip in samples():
            manifest["samples"].append(render(clip, OUT / f"{clip.stem}.mp4"))
        manifest["config"] = CONFIG.read_text()
    except Exception as exc:
        traceback.print_exc()
        manifest["error"] = str(exc)
        old = OUT / "manifest.json"
        if old.exists():
            manifest["samples"] = json.loads(old.read_text()).get("samples", [])
    manifest["render_seconds"] = round(time.time() - started, 2)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    status = f"error: {manifest['error']}" if manifest["error"] else "ok"
    print(f"[{time.strftime('%H:%M:%S')}] rendered {len(manifest['samples'])} sample(s) "
          f"in {manifest['render_seconds']}s ({status})", flush=True)
    return manifest


def grab(clip: Path, name: str | None, seconds: float = 10.0) -> Path:
    SAMPLES.mkdir(exist_ok=True)
    stem = re.sub(r"[^\w-]+", "_", name or clip.stem)[:60]
    dest = SAMPLES / f"{stem}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", f"-{seconds}", "-i", str(clip),
                    *outro.ENCODE_ARGS, str(dest)], check=True)
    print(f"Saved {dest.relative_to(ROOT)}")
    return dest


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static server with byte-range support so <video> can seek."""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(LAB), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass

    def send_head(self):
        path = Path(self.translate_path(self.path))
        rng = self.headers.get("Range")
        if not rng or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        start = int(m.group(1)) if m and m.group(1) else 0
        end = int(m.group(2)) if m and m.group(2) else size - 1
        end = min(end, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def _watched_state():
    paths = [CONFIG, *samples(), *outro.FONT_DIR.glob("*.ttf"), *CONFIG.parent.glob("*.png"),
             *CONFIG.parent.glob("*.jpg")]
    return tuple((str(p), p.stat().st_mtime) for p in paths if p.exists())


def serve() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Preview: http://127.0.0.1:{PORT}/preview.html  (edit {CONFIG.relative_to(ROOT)}, Ctrl-C to stop)",
          flush=True)
    state = None
    code_mtimes = [p.stat().st_mtime for p in WATCHED_CODE]
    try:
        while True:
            if [p.stat().st_mtime for p in WATCHED_CODE] != code_mtimes:
                print("code changed; restarting", flush=True)
                server.server_close()
                os.execv(sys.executable, [sys.executable, *sys.argv])
            now = _watched_state()
            if now != state:
                time.sleep(0.25)
                state = _watched_state()
                render_all()
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="live preview + re-render on save")
    ap.add_argument("--grab", type=Path, metavar="CLIP", help="add the tail of CLIP to samples/")
    ap.add_argument("--name", help="sample name for --grab")
    ap.add_argument("--full", type=Path, metavar="CLIP", help="render all of CLIP with the outro")
    args = ap.parse_args()

    if args.grab:
        grab(args.grab, args.name)
    elif args.full:
        OUT.mkdir(exist_ok=True)
        info = render(args.full, OUT / f"full_{args.full.stem[:60]}.mp4", full=True)
        print(json.dumps(info, indent=2))
    elif args.serve:
        serve()
    else:
        sys.exit(1 if render_all()["error"] else 0)


if __name__ == "__main__":
    main()
