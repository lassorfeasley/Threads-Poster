"""app/datasync.py: sharing working files between machines through Storage."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from app import datasync


class DataSyncTestBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name).resolve()
        for patch in (mock.patch.object(datasync, "DATA_DIR", self.data),
                      mock.patch.object(datasync, "WORKSPACE", "ws"),
                      mock.patch.object(datasync, "_bucket", return_value="bucket"),
                      mock.patch.dict(os.environ, {"DATA_MIRROR": "1"})):
            patch.start()
            self.addCleanup(patch.stop)
        datasync._misses.clear()

    def write(self, rel: str, body: bytes = b"x", age: float = 3600) -> Path:
        p = self.data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        t = time.time() - age
        os.utime(p, (t, t))
        return p


class KeyTests(DataSyncTestBase):
    def test_keys(self):
        self.assertEqual(datasync.key_for(self.data / "clips/a.mp4"), "ws/clips/a.mp4")
        self.assertEqual(datasync.key_for(self.data / "transcripts/c/d/v.json"),
                         "ws/transcripts/c/d/v.json")
        self.assertEqual(datasync.key_for(self.data / "clip_engine/proposals/7.json"),
                         "ws/clip_engine/proposals/7.json")

    def test_not_mirrored(self):
        for rel in ("videos/a.mp4", "outro/a.mp4", "clip_engine/proxies/a.mp4", "app.db",
                    "clips/.a.mp4.123.part", "clips/a.lock", "clipsx/a.mp4"):
            self.assertIsNone(datasync.key_for(self.data / rel), rel)
        self.assertIsNone(datasync.key_for("/elsewhere/clips/a.mp4"))


class PullTests(DataSyncTestBase):
    def test_present_file_needs_no_storage(self):
        p = self.write("thumbs/1.jpg")
        with mock.patch("app.offload.remote_size") as remote:
            self.assertTrue(datasync.ensure_file(p))
        remote.assert_not_called()

    def test_fetches_missing_file_and_remembers_it(self):
        target = self.data / "transcripts/c/v.json"

        def fake_download(bucket, key, dest, expected_size=None):
            self.assertEqual((bucket, key, expected_size), ("bucket", "ws/transcripts/c/v.json", 2))
            Path(dest).write_bytes(b"[]")
            return Path(dest)

        with mock.patch("app.offload.remote_size", return_value=2), \
                mock.patch("app.storage_supabase.download_to_file", side_effect=fake_download):
            self.assertEqual(datasync.ensure_text(target), "[]")
        # Pulled files are recorded so the push sweep doesn't send them back.
        self.assertEqual(datasync.push_sweep(dry_run=True).pending, [])

    def test_miss_is_cached(self):
        target = self.data / "thumbs/9.jpg"
        with mock.patch("app.offload.remote_size", return_value=None) as remote:
            self.assertFalse(datasync.ensure_file(target))
            self.assertFalse(datasync.ensure_file(target))
        self.assertEqual(remote.call_count, 1)

    def test_storage_errors_never_raise(self):
        with mock.patch("app.offload.remote_size", side_effect=RuntimeError("down")):
            self.assertFalse(datasync.ensure_file(self.data / "thumbs/9.jpg"))

    def test_off_without_data_mirror(self):
        with mock.patch.dict(os.environ, {"DATA_MIRROR": ""}), \
                mock.patch("app.offload.remote_size") as remote:
            self.assertFalse(datasync.ensure_file(self.data / "thumbs/9.jpg"))
        remote.assert_not_called()

    def test_unmirrored_and_empty_paths(self):
        with mock.patch("app.offload.remote_size") as remote:
            self.assertFalse(datasync.ensure_file(self.data / "videos/a.mp4"))
            self.assertFalse(datasync.ensure_file(""))
            self.assertFalse(datasync.ensure_file(None))
        remote.assert_not_called()


class PushTests(DataSyncTestBase):
    def test_uploads_new_files_once(self):
        self.write("clips/a.mp4", b"abc")
        self.write("transcripts/c/v.json", b"[]")
        self.write("videos/source.mp4", b"big")       # not mirrored
        self.write("clips/fresh.mp4", b"new", age=1)   # still settling
        uploaded = []
        with mock.patch.object(datasync, "upload", side_effect=lambda p, k: uploaded.append(k)):
            res = datasync.push_sweep()
            self.assertEqual(sorted(uploaded), ["ws/clips/a.mp4", "ws/transcripts/c/v.json"])
            self.assertEqual((res.uploaded, res.uploaded_bytes, res.failed), (2, 5, 0))
            uploaded.clear()
            datasync.push_sweep()
            self.assertEqual(uploaded, [])

    def test_changed_file_uploads_again(self):
        p = self.write("thumbs/1.jpg", b"a")
        with mock.patch.object(datasync, "upload"):
            datasync.push_sweep()
        p.write_bytes(b"bb")
        t = time.time() - 3600
        os.utime(p, (t, t))
        with mock.patch.object(datasync, "upload") as up:
            datasync.push_sweep()
        up.assert_called_once()

    def test_failure_retries_next_sweep(self):
        self.write("thumbs/1.jpg")
        with mock.patch.object(datasync, "upload", side_effect=RuntimeError("down")):
            self.assertEqual(datasync.push_sweep().failed, 1)
        with mock.patch.object(datasync, "upload") as up:
            self.assertEqual(datasync.push_sweep().uploaded, 1)
        up.assert_called_once()

    def test_dry_run_and_limit(self):
        for i in range(3):
            self.write(f"thumbs/{i}.jpg")
        with mock.patch.object(datasync, "upload") as up:
            self.assertEqual(len(datasync.push_sweep(dry_run=True).pending), 3)
            up.assert_not_called()
            self.assertEqual(datasync.push_sweep(limit=2).uploaded, 2)
            self.assertEqual(datasync.push_sweep().uploaded, 1)


if __name__ == "__main__":
    unittest.main()
