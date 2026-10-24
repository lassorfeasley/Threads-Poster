"""Tests for the post-roll outro (app/outro.py).

Run with:  python -m unittest tests.test_outro -v

Renders tiny synthetic clips through the real ffmpeg pipeline against the
climate workspace's design, with the cache pointed at a temp dir.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import outro

DESIGN = outro.CONFIG_FILE


def _make_clip(path: Path, w: int, h: int, seconds: float = 2.0, audio: bool = True) -> Path:
    inputs = ["-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=25:duration={seconds}"]
    if audio:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    *(["-c:a", "aac"] if audio else []), "-shortest", str(path)], check=True)
    return path


@unittest.skipUnless(shutil.which("ffmpeg") and DESIGN.exists(), "needs ffmpeg and the climate outro design")
class WithOutroTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = self.tmp / "outro"
        self.config = self.tmp / "outro.yaml"
        shutil.copy(DESIGN, self.config)
        patcher = mock.patch.object(outro, "OUTRO_DIR", self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _added(self, clip: Path) -> float:
        meta = outro.probe(clip)
        cfg = outro.load_config("vertical" if meta["h"] > meta["w"] else "landscape", self.config)
        return float(cfg["duration"]) - float(cfg["dissolve"])

    def test_appends_outro_to_both_orientations(self):
        for w, h in ((320, 180), (180, 320)):
            clip = _make_clip(self.tmp / f"clip_{w}x{h}.mp4", w, h)
            out = outro.with_outro(clip, config_path=self.config)
            self.assertNotEqual(out, clip)
            self.assertTrue(outro.is_outro_copy(out))
            got = outro.probe(out)
            self.assertEqual((got["w"], got["h"]), (w, h))
            self.assertTrue(got["audio"])
            self.assertAlmostEqual(got["duration"], 2.0 + self._added(clip), delta=0.15)

    def test_silent_clip_gets_an_audio_track(self):
        clip = _make_clip(self.tmp / "silent.mp4", 320, 180, audio=False)
        self.assertTrue(outro.probe(outro.with_outro(clip, config_path=self.config))["audio"])

    def test_cached_and_never_applied_twice(self):
        clip = _make_clip(self.tmp / "clip.mp4", 320, 180)
        first = outro.with_outro(clip, config_path=self.config)
        mtime = first.stat().st_mtime_ns
        self.assertEqual(outro.with_outro(clip, config_path=self.config), first)
        self.assertEqual(first.stat().st_mtime_ns, mtime)
        self.assertEqual(outro.with_outro(first, config_path=self.config), first)

    def test_design_change_rerenders_and_drops_the_stale_copy(self):
        clip = _make_clip(self.tmp / "clip.mp4", 320, 180)
        first = outro.with_outro(clip, config_path=self.config)
        self.config.write_text(self.config.read_text().replace("dissolve: 0.7", "dissolve: 0.5"))
        second = outro.with_outro(clip, config_path=self.config)
        self.assertNotEqual(first, second)
        self.assertFalse(first.exists())

    def test_disabled_or_missing_design_passes_the_clip_through(self):
        clip = _make_clip(self.tmp / "clip.mp4", 320, 180)
        self.assertEqual(outro.with_outro(clip, config_path=self.tmp / "absent.yaml"), clip)
        self.config.write_text("enabled: false\n" + self.config.read_text().replace("enabled: true\n", "", 1))
        self.assertEqual(outro.with_outro(clip, config_path=self.config), clip)

    def test_render_failure_falls_back_to_the_plain_clip(self):
        broken = self.tmp / "broken.mp4"
        broken.write_bytes(b"not a video")
        with self.assertLogs("outro", level="ERROR"):
            self.assertEqual(outro.with_outro(broken, config_path=self.config), broken)

    def test_outro_over_four_seconds_is_rejected(self):
        self.config.write_text(self.config.read_text().replace("duration: 3.5", "duration: 4.5"))
        with self.assertRaises(ValueError):
            outro.load_config("landscape", self.config)


if __name__ == "__main__":
    unittest.main()
