"""Adding a video to clip from Today's posts (app/web/inbox.py). Run with:
python -m unittest tests.test_inbox_add_video -v"""
from __future__ import annotations

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web import inbox, main


class AddVideoRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(inbox.router)
        self.client = TestClient(app)
        p = mock.patch("threading.Thread")
        self.thread = p.start()
        self.addCleanup(p.stop)

    def test_a_bad_link_is_refused(self):
        with mock.patch.object(main, "add_youtube_video", return_value=("invalid", None)):
            r = self.client.post("/inbox/add-url", data={"url": "nope"})
        self.assertEqual(r.status_code, 400)
        self.thread.assert_not_called()

    def test_a_new_link_is_downloaded_then_clipped(self):
        with mock.patch.object(main, "add_youtube_video", return_value=("queued", 7)):
            r = self.client.post("/inbox/add-url", data={"url": "https://youtu.be/x"})
        self.assertEqual(r.json(), {"ok": True, "id": 7})
        kwargs = self.thread.call_args.kwargs
        self.assertIs(kwargs["target"], main.clip_upload_to_inbox)
        self.assertEqual(kwargs["args"], (7, "link"))

    def test_a_file_needs_a_file(self):
        r = self.client.post("/inbox/add-file", data={})
        self.assertEqual(r.status_code, 400)

    def test_a_file_the_machine_cant_take_says_why(self):
        async def refuse(*a, **k):
            raise main.UploadRejected("Uploading isn't available here yet")
        with mock.patch.object(main, "store_upload", side_effect=refuse):
            r = self.client.post("/inbox/add-file", files={"file": ("a.mp4", b"x", "video/mp4")})
        self.assertEqual((r.status_code, r.json()["error"]), (400, "Uploading isn't available here yet"))


if __name__ == "__main__":
    unittest.main()
