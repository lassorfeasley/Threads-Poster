"""Tests for the Facebook Page mirror (app/facebook_api.py and the mirror
functions in app/publishing.py).

Run with:  python -m unittest tests.test_facebook_mirror -v

Nothing here talks to Facebook: HTTP is mocked at ``requests.request`` and the
database is a throwaway in-memory SQLite with ``session_scope`` patched.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import facebook_api, publishing
from app.models import AppToken, Base, FacebookPost, ThreadsPost

NOW = dt.datetime(2026, 11, 9, 12, tzinfo=dt.timezone.utc)


class FakeSettings(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def response(status=200, body=None):
    resp = mock.Mock()
    resp.status_code = status
    resp.json.return_value = body or {}
    resp.text = str(body)
    return resp


class ChoosePageTest(unittest.TestCase):
    PAGES = [{"id": "1", "name": "Renewables.org"}, {"id": "2", "name": "Legends Solar"}]

    def choose(self, pages, page_id=""):
        with mock.patch.object(facebook_api, "load_settings",
                               return_value=FakeSettings({"facebook.page_id": page_id})):
            return facebook_api._choose_page(pages)

    def test_single_page_is_used(self):
        self.assertEqual(self.choose(self.PAGES[:1])["id"], "1")

    def test_several_pages_need_page_id(self):
        with self.assertRaisesRegex(facebook_api.FacebookError, "Renewables.org = 1"):
            self.choose(self.PAGES)

    def test_configured_page_id_wins(self):
        self.assertEqual(self.choose(self.PAGES, "2")["name"], "Legends Solar")

    def test_configured_page_id_missing(self):
        with self.assertRaisesRegex(facebook_api.FacebookError, "facebook.page_id is 9"):
            self.choose(self.PAGES, "9")

    def test_no_pages(self):
        with self.assertRaisesRegex(facebook_api.FacebookError, "no Pages"):
            self.choose([])


class ApiTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(facebook_api, "_page", return_value=("PAGE", "TOKEN"))
        p.start()
        self.addCleanup(p.stop)

    def test_resumable_upload_follows_server_offsets(self):
        calls = []

        def fake(method, url, **kw):
            data = kw.get("data", {})
            chunk = kw.get("files", {}).get("video_file_chunk")
            calls.append((data.get("upload_phase"), data.get("start_offset"),
                          chunk[1] if chunk else None, data.get("description")))
            if data["upload_phase"] == "start":
                return response(body={"upload_session_id": "S", "video_id": 77,
                                      "start_offset": "0", "end_offset": "4"})
            if data["upload_phase"] == "transfer":
                start = int(data["start_offset"])
                return response(body={"start_offset": str(start + 4),
                                      "end_offset": str(min(start + 8, 10))})
            return response(body={"success": True})

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"0123456789")
            f.flush()
            with mock.patch("app.facebook_api.requests.request", side_effect=fake):
                result = facebook_api.publish_video(Path(f.name), "caption text")

        self.assertEqual(result, {"video_id": "77",
                                  "permalink": "https://www.facebook.com/PAGE/videos/77"})
        self.assertEqual([c[0] for c in calls],
                         ["start", "transfer", "transfer", "transfer", "finish"])
        self.assertEqual([c[2] for c in calls if c[0] == "transfer"],
                         [b"0123", b"4567", b"89"])
        self.assertEqual(calls[-1][3], "caption text")

    def test_status_copyright_match(self):
        body = {"status": {"video_status": "ready",
                           "copyright_check_status": {"status": "complete",
                                                      "matches_found": True}},
                "permalink_url": "/PAGE/videos/77/"}
        with mock.patch("app.facebook_api.requests.request", return_value=response(body=body)):
            info = facebook_api.video_status("77")
        self.assertEqual(info["state"], "ready")
        self.assertTrue(info["copyright_match"])
        self.assertEqual(info["permalink"], "https://www.facebook.com/PAGE/videos/77/")

    def test_status_missing_video(self):
        body = {"error": {"code": 100, "message": "Object with ID '77' does not exist"}}
        with mock.patch("app.facebook_api.requests.request",
                        return_value=response(400, body)):
            info = facebook_api.video_status("77")
        self.assertEqual(info["state"], "missing")

    def test_error_message_hides_ids_and_flags_copyright(self):
        body = {"error": {"code": 6000, "message": "Video blocked: copyright content"}}
        with mock.patch("app.facebook_api.requests.request",
                        return_value=response(400, body)):
            with self.assertRaises(facebook_api.FacebookError) as ctx:
                facebook_api._request("POST", "{page_id}/videos", ids={"page_id": "SECRET"})
        self.assertNotIn("SECRET", str(ctx.exception))
        self.assertIn("Upload video failed", str(ctx.exception))
        self.assertTrue(ctx.exception.is_copyright)


class MirrorTest(unittest.TestCase):
    """The mirror against an in-memory database."""

    def setUp(self):
        engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(engine, future=True, expire_on_commit=False)
        self.tmp = tempfile.TemporaryDirectory()
        self.clip = Path(self.tmp.name) / "clip.mp4"
        self.clip.write_bytes(b"video")

        @contextlib.contextmanager
        def scope(read_only=False):
            s = self.Session()
            try:
                yield s
                s.commit()
            finally:
                s.close()

        settings = FakeSettings({"scheduler.publish_retries": 1,
                                 "scheduler.publish_retry_delay_seconds": 0})
        self.publish = mock.Mock(return_value={"video_id": "77", "permalink": "https://fb/77"})
        for p in (mock.patch("app.db.session_scope", scope),
                  mock.patch.object(publishing, "load_settings", return_value=settings),
                  mock.patch.object(publishing, "with_outro", side_effect=lambda c: c),
                  mock.patch.object(facebook_api, "is_authenticated", return_value=True),
                  mock.patch.object(facebook_api, "publish_video", self.publish)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        with self.Session() as s:
            s.add(ThreadsPost(id=1, caption="Hello", status="published",
                              threads_media_id="T1", clip_local_path=str(self.clip)))
            s.commit()

    def fb_row(self) -> FacebookPost:
        with self.Session() as s:
            return s.query(FacebookPost).one()

    def test_publishes_once(self):
        fb = publishing.publish_facebook_mirror(1)
        self.assertEqual((fb.status, fb.video_id, fb.caption), ("published", "77", "Hello"))
        self.publish.assert_called_once_with(self.clip, "Hello")
        # A second publish path for the same post must not post again.
        publishing.publish_facebook_mirror(1)
        self.publish.assert_called_once()

    def test_unpublished_threads_post_is_not_mirrored(self):
        with self.Session() as s:
            s.get(ThreadsPost, 1).status = "queued"
            s.commit()
        self.assertIsNone(publishing.publish_facebook_mirror(1))
        self.publish.assert_not_called()

    def test_disabled_skips(self):
        with mock.patch.object(publishing, "load_settings",
                               return_value=FakeSettings({"facebook.enabled": False})):
            self.assertIsNone(publishing.publish_facebook_mirror(1))
        self.publish.assert_not_called()

    def test_copyright_refusal_fails_and_pauses(self):
        self.publish.side_effect = facebook_api.FacebookError(
            "Upload video failed: copyright", status=400)
        fb = publishing.publish_facebook_mirror(1)
        self.assertEqual(fb.status, "failed")
        self.assertEqual(self.publish.call_count, 1)  # not retried
        self.assertIsNotNone(publishing.facebook_pause_info())
        self.assertFalse(publishing.facebook_mirror_enabled())

        # Operator retry goes through the pause; resume re-enables the mirror.
        self.publish.side_effect = None
        fb = publishing.publish_facebook_mirror(1, force=True)
        self.assertEqual(fb.status, "published")
        publishing.resume_facebook_mirror()
        self.assertTrue(publishing.facebook_mirror_enabled())

    def test_server_error_is_retried(self):
        self.publish.side_effect = [facebook_api.FacebookError("boom", status=500),
                                    {"video_id": "78", "permalink": "https://fb/78"}]
        fb = publishing.publish_facebook_mirror(1)
        self.assertEqual((fb.status, fb.video_id), ("published", "78"))

    def test_pause_keeps_first_reason(self):
        publishing.pause_facebook_mirror("first", 1)
        publishing.pause_facebook_mirror("second", 2)
        self.assertEqual(publishing.facebook_pause_info()["reason"], "first")
        with self.Session() as s:
            self.assertEqual(s.query(AppToken).filter_by(
                name=publishing.FACEBOOK_PAUSE_TOKEN).count(), 1)

    # --- check_facebook_videos ---------------------------------------------

    def mirrored(self, published_at=NOW - dt.timedelta(hours=2), checked_at=None):
        with self.Session() as s:
            s.add(FacebookPost(threads_post_pk=1, video_id="77", status="published",
                               published_at=published_at, checked_at=checked_at))
            s.commit()

    def check(self, info):
        settings = FakeSettings({"facebook.copyright_watch_hours": 72,
                                 "facebook.copyright_check_minutes": 60})
        status = mock.Mock(return_value={"permalink": "", "detail": "", **info})
        with mock.patch.object(publishing, "load_settings", return_value=settings), \
                mock.patch.object(facebook_api, "video_status", status):
            changed = publishing.check_facebook_videos(now=NOW)
        return changed, status

    def test_copyright_match_flags_and_pauses(self):
        self.mirrored()
        changed, _ = self.check({"state": "ready", "copyright_match": True})
        self.assertEqual(changed, 1)
        self.assertEqual(self.fb_row().status, "flagged")
        self.assertIsNotNone(publishing.facebook_pause_info())

    def test_missing_video_flags(self):
        self.mirrored()
        self.check({"state": "missing", "copyright_match": False})
        self.assertEqual(self.fb_row().status, "flagged")

    def test_processing_error_fails_without_pause(self):
        self.mirrored()
        self.check({"state": "error", "copyright_match": False, "detail": "bad codec"})
        row = self.fb_row()
        self.assertEqual(row.status, "failed")
        self.assertIn("bad codec", row.error)
        self.assertIsNone(publishing.facebook_pause_info())

    def test_clean_video_just_records_check(self):
        self.mirrored()
        changed, _ = self.check({"state": "ready", "copyright_match": False})
        self.assertEqual(changed, 0)
        row = self.fb_row()
        self.assertEqual(row.status, "published")
        self.assertIsNotNone(row.checked_at)

    def test_recently_checked_and_old_videos_are_skipped(self):
        self.mirrored(checked_at=NOW - dt.timedelta(minutes=10))
        _, status = self.check({"state": "ready", "copyright_match": False})
        status.assert_not_called()
        with self.Session() as s:
            row = s.query(FacebookPost).one()
            row.checked_at = None
            row.published_at = NOW - dt.timedelta(hours=100)
            s.commit()
        _, status = self.check({"state": "ready", "copyright_match": False})
        status.assert_not_called()

    # --- first comment -------------------------------------------------------

    def with_first_reply(self, text="Support solar at Renewables.org"):
        with self.Session() as s:
            post = s.get(ThreadsPost, 1)
            post.first_reply_id = "R1"
            post.first_reply_text = text
            s.commit()

    def check_with_comment(self, info, comment=None):
        comment = comment or mock.Mock(return_value="C1")
        with mock.patch.object(facebook_api, "publish_comment", comment):
            changed, status = self.check(info)
        return changed, status, comment

    def test_comment_posted_once_video_is_ready(self):
        self.mirrored()
        self.with_first_reply()
        _, _, comment = self.check_with_comment({"state": "processing", "copyright_match": False})
        comment.assert_not_called()
        changed, _, comment = self.check_with_comment({"state": "ready", "copyright_match": False})
        comment.assert_called_once_with("77", "Support solar at Renewables.org")
        self.assertEqual(changed, 1)
        self.assertEqual(self.fb_row().comment_id, "C1")
        # Posted: the next sweep leaves it alone (the copyright check is not due yet).
        _, status, comment = self.check_with_comment({"state": "ready", "copyright_match": False})
        status.assert_not_called()
        comment.assert_not_called()

    def test_no_threads_reply_means_no_comment(self):
        self.mirrored()
        _, _, comment = self.check_with_comment({"state": "ready", "copyright_match": False})
        comment.assert_not_called()

    def test_comment_due_even_when_copyright_check_is_not(self):
        # Deferred Threads reply: checked recently, but the comment is now due.
        self.mirrored(checked_at=NOW - dt.timedelta(minutes=10))
        self.with_first_reply()
        _, status, comment = self.check_with_comment({"state": "ready", "copyright_match": False})
        status.assert_called_once()
        comment.assert_called_once()

    def test_no_comment_on_copyright_match(self):
        self.mirrored()
        self.with_first_reply()
        _, _, comment = self.check_with_comment({"state": "ready", "copyright_match": True})
        comment.assert_not_called()

    def test_failed_comment_waits_for_operator_retry(self):
        self.mirrored()
        self.with_first_reply()
        failing = mock.Mock(side_effect=facebook_api.FacebookError("Post comment failed: no"))
        self.check_with_comment({"state": "ready", "copyright_match": False}, failing)
        row = self.fb_row()
        self.assertEqual(row.comment_id, "")
        self.assertIn("no", row.comment_error)
        # The sweep doesn't keep hammering it...
        self.check_with_comment({"state": "ready", "copyright_match": False}, failing)
        self.assertEqual(failing.call_count, 1)
        # ...the operator's retry does.
        with mock.patch.object(facebook_api, "publish_comment", return_value="C2"):
            self.assertTrue(publishing.post_facebook_comment(row.id, force=True))
        self.assertEqual(self.fb_row().comment_id, "C2")


if __name__ == "__main__":
    unittest.main()
