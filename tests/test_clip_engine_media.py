"""Tests for the clip engine's ffmpeg layer (app/clip_engine/media.py).

The ffmpeg-backed tests render tiny synthetic sources through the real
pipeline (skipped when ffmpeg isn't installed): red for 3 s then a hard cut to
blue for 3 s, under a stereo tone that drops to silence from 1.0 to 2.0 s;
and a silent green source. Failure paths patch ``subprocess.run``. Every test
points CLIP_ENGINE_DIR at its own temp dir.

Run with:  DATABASE_URL="sqlite://" python -m unittest tests.test_clip_engine_media -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from app.clip_engine import media

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _make_cut_source(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=red:s=320x180:r=25:d=3",
         "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
         "-filter_complex",
         "[0:v][1:v]concat=n=2:v=1:a=0[v];"
         "[2:a]volume=enable='between(t,1,2)':volume=0,aformat=channel_layouts=stereo[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)
    return path


def _make_silent_source(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=green:s=320x180:r=25:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(path)], check=True, capture_output=True)
    return path


def _streams(path: Path) -> list[dict]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,channels,width,height", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)["streams"]


def _rgb(data: bytes) -> tuple[int, int, int]:
    return data[0], data[1], data[2]


def _colour_at(video: Path, seconds: float) -> str:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{seconds:.3f}",
                          "-i", str(video), "-frames:v", "1", "-vf", "scale=1:1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return _name(_rgb(raw))


def _colour_of_jpeg(data: bytes) -> str:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "image2pipe",
                          "-i", "-", "-vf", "scale=1:1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"],
                         input=data, check=True, capture_output=True).stdout
    return _name(_rgb(raw))


def _name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    if r > 150 and b < 100:
        return "red"
    if b > 150 and r < 100:
        return "blue"
    if g > 80 and r < 80 and b < 80:
        return "green"
    return f"rgb{rgb}"


class _EngineDir(unittest.TestCase):
    """A private engine_dir() per test."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.tmp / "engine")})
        patcher.start()
        self.addCleanup(patcher.stop)


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
class FfmpegTest(_EngineDir):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = Path(tempfile.mkdtemp())
        cls.source = _make_cut_source(cls.fixtures / "cut.mp4")
        cls.silent = _make_silent_source(cls.fixtures / "silent.mp4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.fixtures, True)

    # ---- probing ----------------------------------------------------------

    def test_probe(self):
        self.assertAlmostEqual(media.probe_duration(self.source), 6.0, delta=0.1)
        self.assertTrue(media.has_audio(self.source))
        self.assertFalse(media.has_audio(self.silent))
        self.assertIsNone(media.probe_duration(self.tmp / "missing.mp4"))
        with self.assertRaises(media.MediaError):
            media.has_audio(self.tmp / "missing.mp4")
        broken = self.tmp / "broken.mp4"
        broken.write_bytes(b"not a video")
        self.assertIsNone(media.probe_duration(broken))
        with self.assertRaises(media.MediaError):
            media.has_audio(broken)

    # ---- renders ----------------------------------------------------------

    def test_render_keeps_play_order(self):
        out = media.render_segments(self.source, [{"start": 3.5, "end": 5.0},
                                                  {"start": 0.5, "end": 2.5}],
                                    self.tmp / "out" / "ba.mp4")
        self.assertEqual(_colour_at(out, 0.2), "blue")
        self.assertEqual(_colour_at(out, 2.5), "red")
        self.assertAlmostEqual(media.probe_duration(out), 3.5, delta=0.1)
        kinds = {s["codec_type"]: s for s in _streams(out)}
        self.assertEqual(kinds["video"]["height"], 360)
        self.assertEqual(kinds["audio"]["channels"], 2)
        self.assertEqual(sorted(p.name for p in out.parent.iterdir()), ["ba.mp4"])

    def test_fades_do_not_change_the_length(self):
        segments = [{"start": 0.5, "end": 2.5}, {"start": 3.2, "end": 4.4},
                    {"start": 4.6, "end": 5.9}]
        with_fades = media.render_segments(self.source, segments, self.tmp / "f.mp4")
        without = media.render_segments(self.source, segments, self.tmp / "n.mp4", fade=0)
        self.assertAlmostEqual(media.probe_duration(with_fades),
                               media.probe_duration(without), delta=0.05)
        self.assertAlmostEqual(media.probe_duration(with_fades), 4.5, delta=0.1)

    def test_silent_source_renders_with_an_audio_track(self):
        out = media.render_segments(self.silent, [{"start": 0.5, "end": 1.5},
                                                  {"start": 2.0, "end": 2.8}],
                                    self.tmp / "silent-out.mp4")
        self.assertIn("audio", {s["codec_type"] for s in _streams(out)})
        self.assertAlmostEqual(media.probe_duration(out), 1.8, delta=0.1)
        self.assertEqual(_colour_at(out, 0.5), "green")

    def test_render_rejects_bad_segments(self):
        for segments in ([], [{"start": 1.0, "end": 1.2}], [{"start": 1.0}],
                         [{"start": 7.0, "end": 9.0}], [{"start": "x", "end": 2}]):
            with self.subTest(segments=segments), self.assertRaises(media.MediaError):
                media.render_segments(self.source, segments, self.tmp / "bad.mp4")
        self.assertFalse((self.tmp / "bad.mp4").exists())

    def test_render_clamps_a_segment_running_past_the_end(self):
        out = media.render_segments(self.source, [{"start": 5.0, "end": 9.0}],
                                    self.tmp / "tail.mp4")
        self.assertAlmostEqual(media.probe_duration(out), 1.0, delta=0.1)

    def test_render_cached_is_keyed_and_reused(self):
        first = media.render_cached(self.source, [{"start": 3.5, "end": 5.0},
                                                  {"start": 0.5, "end": 2.5}], kind="draft")
        self.assertEqual(first.parent, self.tmp / "engine" / "renders")
        self.assertTrue(first.name.startswith("draft-"))
        before = first.stat()
        os.utime(first, (before.st_atime - 3600, before.st_mtime - 3600))
        with mock.patch.object(media.subprocess, "run",
                               side_effect=AssertionError("re-rendered")):
            again = media.render_cached(self.source, [{"start": 3.501, "end": 4.999},
                                                      {"start": 0.5, "end": 2.5}])
        self.assertEqual(again, first)
        self.assertEqual(first.stat().st_ino, before.st_ino)          # the same file...
        self.assertGreaterEqual(first.stat().st_mtime, before.st_mtime)  # ...touched for prune
        swapped = media.render_cached(self.source, [{"start": 0.5, "end": 2.5},
                                                    {"start": 3.5, "end": 5.0}], height=240,
                                      kind="judge/../x")
        self.assertNotEqual(swapped, first)
        self.assertEqual(swapped.parent, first.parent)
        self.assertEqual(_colour_at(swapped, 0.2), "red")

    # ---- proxies ----------------------------------------------------------

    def test_windowed_proxy_starts_its_clock_at_the_window(self):
        proxy = media.perception_proxy(self.source, start=2.0, end=5.0)
        self.assertIn("-2.00-5.00-v1.mp4", proxy.name)
        self.assertAlmostEqual(media.probe_duration(proxy), 3.0, delta=0.15)
        self.assertEqual(_colour_at(proxy, 0.5), "red")    # source 2.5 s
        self.assertEqual(_colour_at(proxy, 1.5), "blue")   # source 3.5 s
        streams = {s["codec_type"]: s for s in _streams(proxy)}
        self.assertEqual(streams["video"]["height"], 180)  # never upscaled
        self.assertEqual(streams["audio"]["channels"], 1)
        with mock.patch.object(media.subprocess, "run", side_effect=AssertionError("rebuilt")):
            self.assertEqual(media.perception_proxy(self.source, start=2.0, end=5.0), proxy)

    def test_whole_proxy_and_bad_windows(self):
        proxy = media.perception_proxy(self.source)
        self.assertIn("-0.00-end-v1.mp4", proxy.name)
        self.assertAlmostEqual(media.probe_duration(proxy), 6.0, delta=0.15)
        silent = media.perception_proxy(self.silent)
        self.assertNotIn("audio", {s["codec_type"] for s in _streams(silent)})
        with self.assertRaises(media.MediaError):
            media.perception_proxy(self.source, start=4.0, end=4.0)
        with self.assertRaises(media.MediaError):
            media.perception_proxy(self.tmp / "missing.mp4")

    # ---- shot maps --------------------------------------------------------

    def test_shot_map_finds_the_cut_and_the_gap(self):
        result = media.shot_map(self.source)
        self.assertEqual(result["error"], "")
        self.assertAlmostEqual(result["duration"], 6.0, delta=0.1)
        self.assertEqual(len(result["cuts"]), 1, result["cuts"])
        self.assertAlmostEqual(result["cuts"][0]["t"], 3.0, delta=0.05)
        self.assertGreaterEqual(result["cuts"][0]["score"], 8.0)
        self.assertEqual(len(result["quiet"]), 1, result["quiet"])
        start, end = result["quiet"][0]
        self.assertAlmostEqual(start, 1.0, delta=0.1)
        self.assertAlmostEqual(end, 2.0, delta=0.1)
        cached = list((self.tmp / "engine" / "shotmaps").glob("*.json"))
        self.assertEqual(len(cached), 1)
        with mock.patch.object(media.subprocess, "run", side_effect=AssertionError("re-ran")):
            self.assertEqual(media.shot_map(self.source)["cuts"], result["cuts"])

    def test_shot_map_of_a_silent_source(self):
        result = media.shot_map(self.silent)
        self.assertEqual((result["error"], result["cuts"], result["quiet"]), ("", [], []))
        self.assertEqual(len(list((self.tmp / "engine" / "shotmaps").glob("*.json"))), 1)

    def test_shot_map_error_is_returned_not_cached(self):
        def failing(cmd, **kwargs):
            if cmd[0] == "ffmpeg":
                return subprocess.CompletedProcess(cmd, 1, b"", b"decoder exploded")
            return real_run(cmd, **kwargs)

        real_run = subprocess.run
        with mock.patch.object(media.subprocess, "run", side_effect=failing):
            result = media.shot_map(self.source)
        self.assertIn("decoder exploded", result["error"])
        self.assertEqual(result["cuts"], [])
        self.assertFalse(list((self.tmp / "engine" / "shotmaps").glob("*.json")))
        # The next call tries again and caches the good result.
        self.assertEqual(media.shot_map(self.source)["error"], "")
        self.assertEqual(len(list((self.tmp / "engine" / "shotmaps").glob("*.json"))), 1)

    def test_shot_map_never_raises(self):
        self.assertTrue(media.shot_map(self.tmp / "missing.mp4")["error"])
        with mock.patch.object(media, "_build_shot_map", side_effect=KeyError("boom")):
            self.assertIn("boom", media.shot_map(self.source)["error"])

    def test_corrupt_cached_map_is_rebuilt(self):
        path = (self.tmp / "engine" / "shotmaps" /
                f"{media.file_key(self.source)}-v{media.VERSION}.json")
        path.parent.mkdir(parents=True)
        path.write_text('{"version": 1, "cuts": [], "quiet": [], "error": "", "threshold": "x"}')
        self.assertEqual(len(media.shot_map(self.source)["cuts"]), 1)

    # ---- frames -----------------------------------------------------------

    def test_sample_frames_on_a_grid(self):
        frames = media.sample_frames(self.source, start=1.0, end=5.0, interval=1.0, width=160)
        self.assertEqual([t for t, _ in frames], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual([_colour_of_jpeg(j) for _, j in frames], ["red", "red", "blue", "blue"])

    def test_a_frame_labelled_t_shows_what_is_on_screen_at_t(self):
        # The cut to blue is at 3.0: 0.5 s after the 2.5 slot. Picking the
        # frame nearest a slot's END (round-to-nearest) would label blue 2.5.
        frames = media.sample_frames(self.source, start=2.5, end=6.0, interval=2.0, width=160)
        self.assertEqual([(t, _colour_of_jpeg(j)) for t, j in frames],
                         [(2.5, "red"), (4.5, "blue")])
        sparse = media.sample_frames(self.source, start=0, end=6, interval=2000, width=160)
        self.assertEqual([t for t, _ in sparse], [0.0])

    def test_sample_frames_in_chunks_keeps_one_clock(self):
        calls = []
        real = media._frames_chunk

        def spy(*args, **kwargs):
            calls.append(args[1:3])
            return real(*args, **kwargs)

        with mock.patch.object(media, "_FRAME_CHUNK_SECONDS", 2.0), \
                mock.patch.object(media, "_frames_chunk", side_effect=spy):
            frames = media.sample_frames(self.source, start=0.25, end=5.0, interval=0.5,
                                         width=160)
        self.assertEqual(len(calls), 3)
        self.assertEqual([t for t, _ in frames], [0.25 + 0.5 * i for i in range(10)])

    def test_sample_frames_edges(self):
        self.assertEqual(media.sample_frames(self.source, start=4, end=4, interval=1,
                                             width=160), [])
        with self.assertRaises(media.MediaError):   # nothing there: never an empty "success"
            media.sample_frames(self.source, start=10, end=12, interval=1, width=160)
        with self.assertRaises(media.MediaError):
            media.sample_frames(self.source, start=0, end=2, interval=0, width=160)
        clamped = media.sample_frames(self.source, start=4.5, end=60, interval=0.5, width=160)
        self.assertEqual([t for t, _ in clamped], [4.5, 5.0, 5.5])

    def test_clip_frames_spread_to_max_frames(self):
        frames = media.clip_frames(self.source, interval=0.5, width=160, max_frames=4)
        self.assertEqual([t for t, _ in frames], [0.0, 1.5, 3.0, 4.5])
        self.assertEqual(len(media.clip_frames(self.source, interval=2.0, width=160,
                                               max_frames=60)), 3)
        self.assertEqual(media.clip_frames(self.source, interval=1, width=160, max_frames=0), [])

    # ---- trim -------------------------------------------------------------

    def test_trim_tail(self):
        out = media.trim_tail(self.source, 2.0, self.tmp / "trimmed.mp4", height=240)
        self.assertAlmostEqual(media.probe_duration(out), 4.0, delta=0.1)
        self.assertEqual({s["codec_type"]: s for s in _streams(out)}["video"]["height"], 240)
        whole = media.trim_tail(self.source, 0, self.tmp / "whole.mp4")
        self.assertAlmostEqual(media.probe_duration(whole), 6.0, delta=0.1)
        with self.assertRaises(media.MediaError):
            media.trim_tail(self.source, 5.9, self.tmp / "nothing.mp4")
        self.assertFalse((self.tmp / "nothing.mp4").exists())

    # ---- failures -----------------------------------------------------------

    def test_timeouts_and_missing_binaries_are_media_errors(self):
        dest = self.tmp / "out" / "x.mp4"
        for error in (subprocess.TimeoutExpired(["ffmpeg"], 5), FileNotFoundError("ffmpeg"),
                      PermissionError("denied")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(media.subprocess, "run", side_effect=error), \
                        self.assertRaises(media.MediaError) as caught:
                    media.render_segments(self.source, [{"start": 0, "end": 2}], dest)
                if isinstance(error, subprocess.TimeoutExpired):
                    self.assertIn("timed out", str(caught.exception))
        self.assertFalse(dest.exists())
        self.assertFalse(list(dest.parent.glob(".*")) if dest.parent.exists() else [])

    def test_failed_encode_leaves_no_tmp_and_no_cache(self):
        real_run = subprocess.run

        def failing(cmd, **kwargs):
            if cmd[0] == "ffmpeg":
                Path(cmd[-1]).write_bytes(b"partial")
                return subprocess.CompletedProcess(cmd, 1, b"", b"x264 fell over")
            return real_run(cmd, **kwargs)

        with mock.patch.object(media.subprocess, "run", side_effect=failing), \
                self.assertRaises(media.MediaError) as caught:
            media.render_cached(self.source, [{"start": 0, "end": 2}])
        self.assertIn("x264 fell over", str(caught.exception))
        self.assertEqual(list((self.tmp / "engine" / "renders").iterdir()), [])


class TimeoutTest(unittest.TestCase):
    def test_table(self):
        for seconds, expected in ((None, 600.0), (0, 600.0), (-5, 600.0), ("x", 600.0),
                                  (float("nan"), 600.0), (float("inf"), 600.0),
                                  (60, 150.0), (3600, 1920.0), ("10", 125.0)):
            with self.subTest(seconds=seconds):
                self.assertEqual(media.timeout_for(seconds), expected)


class FileKeyTest(_EngineDir):
    def test_key_follows_the_file(self):
        path = self.tmp / "a.mp4"
        path.write_bytes(b"x" * 10)
        key = media.file_key(path)
        self.assertEqual(len(key), 16)
        self.assertEqual(media.file_key(str(path)), key)
        os.utime(path, ns=(1, 2_000_000_000))
        self.assertNotEqual(media.file_key(path), key)
        with self.assertRaises(media.MediaError):
            media.file_key(self.tmp / "missing.mp4")


class InflightTest(_EngineDir):
    def setUp(self):
        super().setUp()
        self.source = self.tmp / "videos" / "Vid_1.MP4"
        self.source.parent.mkdir()
        self.source.write_bytes(b"video bytes")

    def inflight(self) -> Path:
        return self.tmp / "engine" / "inflight"

    def holds(self) -> list[str]:
        return sorted(p.name for p in self.inflight().glob(".*.hold"))

    def test_hardlink_pin_is_shared_and_released_by_the_last_holder(self):
        first = media.link_inflight(self.source, "vid/1")
        self.assertEqual(first.parent, self.inflight())
        self.assertTrue(first.name.startswith("vid-1-") and first.name.endswith(".mp4"))
        self.assertTrue(os.path.samefile(first, self.source))
        second = media.link_inflight(self.source, "vid/1")
        self.assertEqual(second, first)
        self.assertEqual(len(self.holds()), 1)
        media.release_inflight(first)
        self.assertTrue(first.exists())
        media.release_inflight(first)
        self.assertFalse(first.exists())
        self.assertEqual(self.holds(), [])
        self.assertTrue(self.source.exists())

    def test_pin_survives_the_original_being_deleted(self):
        pin = media.link_inflight(self.source, "v")
        self.source.unlink()
        self.assertEqual(pin.read_bytes(), b"video bytes")
        media.release_inflight(pin)
        self.assertFalse(pin.exists())

    def test_copy_fallback_keeps_the_cache_key(self):
        with mock.patch.object(media.os, "link", side_effect=OSError(18, "cross-device")):
            pin = media.link_inflight(self.source, "v")
        self.assertFalse(os.path.samefile(pin, self.source))
        self.assertEqual(pin.stat().st_mtime_ns, self.source.stat().st_mtime_ns)
        key = media.file_key(pin)
        media.release_inflight(pin)
        with mock.patch.object(media.os, "link", side_effect=OSError(18, "cross-device")):
            again = media.link_inflight(self.source, "v")
        self.assertEqual(media.file_key(again), key)
        media.release_inflight(again)

    def test_another_live_process_keeps_the_pin(self):
        pin = media.link_inflight(self.source, "v")
        other = pin.with_name(f".{pin.name}.{os.getppid()}.hold")
        other.write_text("x")
        media.release_inflight(pin)
        self.assertTrue(pin.exists())
        # Once that process is gone (or its marker is stale), the pin goes.
        done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                              capture_output=True, text=True, check=True)
        other.rename(pin.with_name(f".{pin.name}.{int(done.stdout)}.hold"))
        media.release_inflight(pin)
        self.assertFalse(pin.exists())
        self.assertEqual(self.holds(), [])

    def test_release_refuses_anything_that_is_not_a_pin(self):
        media.release_inflight(self.source)
        media.release_inflight(None)
        media.release_inflight("")
        self.assertTrue(self.source.exists())

    def test_pinning_a_missing_source_is_a_media_error(self):
        with self.assertRaises(media.MediaError):
            media.link_inflight(self.tmp / "missing.mp4", "v")


class HousekeepingTest(_EngineDir):
    def make(self, sub: str, name: str, age_days: float = 0.0) -> Path:
        path = self.tmp / "engine" / sub / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        stamp = time.time() - age_days * 86400
        os.utime(path, (stamp, stamp))
        return path

    def test_prune_by_age(self):
        old = [self.make(sub, f"old-{sub}.bin", 20) for sub in
               ("proxies", "renders", "shotmaps", "taste")]
        fresh = [self.make(sub, f"new-{sub}.bin", 1) for sub in
                 ("proxies", "renders", "shotmaps", "taste")]
        stale_tmp = self.make("renders", ".draft-x.123.456.tmp.mp4", 2)
        live_tmp = self.make("renders", ".draft-y.123.456.tmp.mp4", 0)
        elsewhere = self.make("shotlogs", "old.json", 400)   # not media's to prune
        self.assertEqual(media.prune(14), 5)
        for path in old + [stale_tmp]:
            self.assertFalse(path.exists(), path)
        for path in fresh + [live_tmp, elsewhere]:
            self.assertTrue(path.exists(), path)
        self.assertEqual(media.prune(14), 0)

    def test_prune_inflight_only_takes_unheld_pins_over_a_day_old(self):
        source = self.tmp / "src.mp4"
        source.write_bytes(b"v")
        held = media.link_inflight(source, "held")
        orphan = self.tmp / "engine" / "inflight" / "orphan-1234.mp4"
        os.link(source, orphan)
        dead_marker = orphan.with_name(".gone-0000.mp4.99999999.hold")
        dead_marker.write_text("x")
        self.assertEqual(media._prune_inflight(time.time()), 0)    # too young
        self.assertTrue(orphan.exists())
        removed = media._prune_inflight(time.time() + 2 * 86400)
        self.assertEqual(removed, 1)
        self.assertFalse(orphan.exists())
        self.assertTrue(held.exists())
        self.assertFalse(dead_marker.exists())
        media.release_inflight(held)

    def test_prune_never_raises(self):
        with mock.patch.object(media, "engine_dir", side_effect=OSError("disk gone")):
            self.assertEqual(media.prune(), 0)

    def test_drop(self):
        a = self.make("renders", "a.mp4")
        b = self.make("renders", "b.mp4")
        directory = self.tmp / "engine" / "renders"
        media.drop([a, None, "", self.tmp / "missing.mp4", directory])
        media.drop(str(b))
        media.drop(None)
        media.drop(12)
        self.assertFalse(a.exists() or b.exists())
        self.assertTrue(directory.is_dir())

    def test_disk_free_gb(self):
        self.assertGreater(media.disk_free_gb(self.tmp), 0.0)
        self.assertGreater(media.disk_free_gb(self.tmp / "not" / "there.mp4"), 0.0)
        with mock.patch.object(media.shutil, "disk_usage", side_effect=PermissionError("no")):
            self.assertEqual(media.disk_free_gb(self.tmp), 0.0)


class QuietRunsTest(unittest.TestCase):
    def windows(self, levels, start=0.0):
        return [(round(start + i * 0.05, 3), db) for i, db in enumerate(levels)]

    def test_relative_to_the_median(self):
        loud = [-20.0] * 20
        levels = loud + [-40.0, -45.0, float("-inf"), -50.0] + loud + [-35.0, -40.0] + loud
        runs = media._quiet_runs(self.windows(levels))
        # The 4-window dip (0.2 s) counts; the 2-window one (0.1 s) is too short.
        self.assertEqual(runs, [[1.0, 1.2]])

    def test_a_silent_track_is_one_long_pause(self):
        self.assertEqual(media._quiet_runs(self.windows([float("-inf")] * 10)), [[0.0, 0.5]])

    def test_end_clamped_to_duration_and_holes_split_runs(self):
        levels = [-20.0] * 10 + [-60.0] * 4
        self.assertEqual(media._quiet_runs(self.windows(levels), duration=0.68),
                         [[0.5, 0.68]])
        # Timestamps jumping from 1.1 to 2.0: two separate pauses, not one.
        loud = [(round(i * 0.05, 2), -20.0) for i in range(20)]
        gap = loud + [(1.0, -60.0), (1.05, -60.0), (1.1, -60.0), (2.0, -60.0), (2.05, -60.0),
                      (2.1, -60.0)] + [(round(3.0 + i * 0.05, 2), -20.0) for i in range(20)]
        self.assertEqual(media._quiet_runs(gap), [[1.0, 1.15], [2.0, 2.15]])

    def test_metadata_parsing(self):
        text = ("frame:0    pts:0       pts_time:0\nlavfi.scd.score=0.000\n"
                "frame:1    pts:512     pts_time:0.04\nlavfi.scd.score=15.625\n"
                "frame:2    pts:NOPTS   pts_time:NOPTS\nlavfi.scd.score=99\n"
                "frame:3    pts:2048    pts_time:0.12\nlavfi.scd.score=nan\n"
                "frame:4    pts:2560    pts_time:0.16\nlavfi.astats.Overall.RMS_level=-inf\n")
        values = media._metadata_values(text, "lavfi.scd.score")
        self.assertEqual(values[:2], [(0.0, 0.0), (0.04, 15.625)])
        self.assertEqual(len(values), 3)   # the NOPTS frame is skipped; nan parses
        self.assertEqual(media._metadata_values(text, "lavfi.astats.Overall.RMS_level"),
                         [(0.16, float("-inf"))])


if __name__ == "__main__":
    unittest.main()
