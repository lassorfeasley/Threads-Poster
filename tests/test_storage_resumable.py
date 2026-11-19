"""The resumable (TUS) upload's retry rules (app/storage_supabase.py).

Run with:  python -m unittest tests.test_storage_resumable -v

A fake HTTP client stands in for Supabase Storage; sleeps are skipped.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import storage_supabase


class _Resp:
    def __init__(self, status: int, headers: dict | None = None, text: str = ""):
        self.status_code = status
        self.headers = headers or {}
        self.text = text


class _FakeClient:
    """POST creates the upload; PATCH answers from ``patches`` in turn; HEAD
    reports the server's offset."""

    def __init__(self, patches: list[int], server_offset: int = 0):
        self.patches = list(patches)
        self.server_offset = server_offset
        self.patch_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None):
        return _Resp(201, {"location": "/storage/v1/upload/resumable/abc"})

    def patch(self, url, content=b"", headers=None):
        self.patch_calls += 1
        status = self.patches.pop(0)
        if status == 204:
            return _Resp(204, {"upload-offset": str(int(headers["Upload-Offset"]) + len(content))})
        return _Resp(status, text="conflict")

    def head(self, url, headers=None):
        return _Resp(200, {"upload-offset": str(self.server_offset)})


class ResumableUploadTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"x" * 1000)
        tmp.close()
        self.path = Path(tmp.name)
        self.addCleanup(self.path.unlink)
        for patcher in (mock.patch.object(storage_supabase.time, "sleep"),
                        mock.patch.object(storage_supabase, "_auth_headers", return_value={}),
                        mock.patch.object(storage_supabase, "env", return_value="https://x.supabase.co")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def upload(self, client):
        with mock.patch.object(storage_supabase.httpx, "Client", return_value=client):
            storage_supabase.upload_file_resumable(self.path, "bucket", "key.mp4")

    def test_a_conflict_that_repeats_at_the_same_offset_gives_up_fast(self):
        client = _FakeClient([409, 409, 409, 409, 409, 409, 409, 409])
        with self.assertRaises(RuntimeError) as err:
            self.upload(client)
        self.assertIn("keeps conflicting", str(err.exception))
        self.assertEqual(client.patch_calls, 2)

    def test_a_single_conflict_resumes_from_the_server_offset(self):
        client = _FakeClient([409, 204])
        self.upload(client)
        self.assertEqual(client.patch_calls, 2)

    def test_server_errors_still_get_the_full_retries(self):
        client = _FakeClient([500] * (storage_supabase._TUS_MAX_RETRIES + 1))
        with self.assertRaises(RuntimeError) as err:
            self.upload(client)
        self.assertIn("kept failing", str(err.exception))
        self.assertEqual(client.patch_calls, storage_supabase._TUS_MAX_RETRIES + 1)


if __name__ == "__main__":
    unittest.main()
