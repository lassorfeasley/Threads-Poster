"""Every exported clip is levelled to one loudness (app/clipper.py: level_audio).

Run with:  python -m unittest tests.test_audio_level -v   (needs ffmpeg)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app import clipper


def _make(path: Path, audio: str | None) -> Path:
    """A 4-second clip: a test card plus ``audio`` (a lavfi source), or no audio track."""
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=4"]
    if audio:
        args += ["-f", "lavfi", "-i", audio, "-shortest", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True, timeout=120)
    return path


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class LevelAudioTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def lufs(self, path: Path) -> float:
        return float(clipper._loudness(path)["input_i"])

    def test_a_quiet_clip_is_brought_up_to_the_target(self):
        clip = _make(self.dir / "quiet.mp4", "sine=frequency=440:duration=4,volume=0.1")
        self.assertLess(self.lufs(clip), -30)
        video_before = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,nb_frames",
             "-of", "csv=p=0", str(clip)], capture_output=True, text=True).stdout
        self.assertTrue(clipper.level_audio(clip))
        self.assertAlmostEqual(self.lufs(clip), clipper.TARGET_LUFS, delta=1.0)
        video_after = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,nb_frames",
             "-of", "csv=p=0", str(clip)], capture_output=True, text=True).stdout
        self.assertEqual(video_before, video_after)          # the picture is copied, not re-encoded
        self.assertFalse(clipper.level_audio(clip))          # already level: left alone
        self.assertEqual([p.name for p in self.dir.iterdir()], ["quiet.mp4"])   # no temp file left

    def test_silence_and_no_audio_are_left_alone(self):
        silent = _make(self.dir / "silent.mp4", "anullsrc=r=48000:cl=stereo")
        mute = _make(self.dir / "noaudio.mp4", None)
        for clip in (silent, mute):
            with self.subTest(clip=clip.name):
                before = clip.read_bytes()
                self.assertFalse(clipper.level_audio(clip))
                self.assertEqual(clip.read_bytes(), before)

    def test_a_failure_keeps_the_clip_as_it_was(self):
        clip = _make(self.dir / "quiet.mp4", "sine=frequency=440:duration=4,volume=0.1")
        before = clip.read_bytes()
        original = clipper._run_ffmpeg

        def broken(args):
            raise clipper.ClipExportError("ffmpeg failed: boom")

        clipper._run_ffmpeg = broken
        try:
            self.assertFalse(clipper.level_audio(clip))
        finally:
            clipper._run_ffmpeg = original
        self.assertEqual(clip.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
