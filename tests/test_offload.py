"""Tests for moving source videos off disk (app/offload.py) and the storage
helpers it uploads/downloads with (app/storage_supabase.py).

Run with:  python -m unittest tests.test_offload -v

Nothing here talks to Supabase: database tests use a throwaway in-memory
SQLite database, storage calls are mocked, and the TUS/download helpers run
against an in-process ``httpx.MockTransport``.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import offload, storage_supabase
from app.models import Base, Candidate, Channel, Cut, ThreadsPost

NOW = dt.datetime(2026, 11, 6, 12, tzinfo=dt.timezone.utc)
GRACE = dt.timedelta(days=3)
OLD = NOW - dt.timedelta(days=10)


def row(**kw) -> offload.SourceRow:
    base = dict(candidate_id=1, video_id="v", status="archived", local_path="/x.mp4",
                object_key="", multi_clip=False)
    base.update(kw)
    return offload.SourceRow(**base)


class InToTrimTest(unittest.TestCase):
    """Mirrors the dashboard's "To trim" bucket."""

    def test_downloaded_never_clipped_is_waiting(self):
        self.assertTrue(offload.in_to_trim("archived", False, set(), False))

    def test_exported_cut_is_clipped(self):
        self.assertFalse(offload.in_to_trim("archived", False, set(), True))

    def test_published_or_queued_is_clipped(self):
        for st in ("published", "queued", "draft", "publishing"):
            self.assertFalse(offload.in_to_trim("archived", False, {st}, False), st)

    def test_failed_post_is_clipped(self):
        self.assertFalse(offload.in_to_trim("archived", False, {"failed"}, True))

    def test_multi_clip_marker_always_waits(self):
        self.assertTrue(offload.in_to_trim("archived", True, {"published"}, True))
        self.assertTrue(offload.in_to_trim("archived", True, {"failed"}, True))

    def test_deleted_draft_puts_it_back(self):
        # Operator deleted their only post: post statuses empty, no export.
        self.assertTrue(offload.in_to_trim("archived", False, set(), False))


class DecideTest(unittest.TestCase):
    def decide(self, r, mtime=OLD):
        return offload.decide(r, file_mtime=mtime, now=NOW, grace=GRACE)[0]

    def test_published_and_settled_is_offloaded(self):
        self.assertEqual(self.decide(row(post_statuses={"published"}, has_exported_cut=True)),
                         offload.OFFLOAD)

    def test_waiting_to_be_clipped_stays(self):
        self.assertEqual(self.decide(row()), offload.KEEP)

    def test_recent_activity_holds_it(self):
        r = row(post_statuses={"published"}, has_exported_cut=True,
                last_activity=NOW - dt.timedelta(hours=5))
        self.assertEqual(self.decide(r), offload.KEEP)

    def test_fresh_file_holds_it(self):
        # A just-restored file restarts the grace period.
        r = row(post_statuses={"published"}, has_exported_cut=True)
        self.assertEqual(self.decide(r, mtime=NOW - dt.timedelta(hours=1)), offload.KEEP)

    def test_export_in_progress_stays(self):
        r = row(post_statuses={"published"}, has_exported_cut=True, exporting=True)
        self.assertEqual(self.decide(r), offload.KEEP)

    def test_rejected_is_deleted_once_settled(self):
        self.assertEqual(self.decide(row(status="rejected")), offload.DELETE)
        self.assertEqual(self.decide(row(status="rejected"), mtime=NOW), offload.KEEP)

    def test_downloading_or_failed_stays(self):
        for st in ("approved", "failed", "new"):
            self.assertEqual(self.decide(row(status=st, post_statuses={"published"})),
                             offload.KEEP, st)


class _DB:
    """In-memory SQLite with ``session_scope`` patched to use it."""

    def setUp(self):
        engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(engine, future=True, expire_on_commit=False)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

        @contextlib.contextmanager
        def scope(read_only=False):
            s = self.Session()
            try:
                yield s
                s.commit()
            finally:
                s.close()

        self.patches = [mock.patch("app.db.session_scope", scope)]
        for p in self.patches:
            p.start()
        with self.Session() as s:
            s.add(Channel(id=1, call_sign="KXYZ", url="https://example.com/c"))
            s.commit()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def add_video(self, cid, status="archived", posts=(), exported=False, size=1000,
                  age_days=10, multi=False):
        path = self.dir / f"{cid}.mp4"
        path.write_bytes(b"x" * size)
        t = (NOW - dt.timedelta(days=age_days)).timestamp()
        os.utime(path, (t, t))
        with self.Session() as s:
            s.add(Candidate(id=cid, channel_pk=1, video_id=f"vid{cid}", title="t",
                            url=f"https://y/{cid}", status=status,
                            local_video_path=str(path), multi_clip_potential=multi))
            if exported:
                s.add(Cut(candidate_pk=cid, trimmed_clip_path="/clip.mp4",
                          updated_at=NOW - dt.timedelta(days=age_days)))
            for st in posts:
                s.add(ThreadsPost(candidate_pk=cid, status=st,
                                  created_at=NOW - dt.timedelta(days=age_days)))
            s.commit()
        return path


class SweepTest(_DB, unittest.TestCase):
    def run_sweep(self, **kw):
        uploaded = {}

        def fake_upload(path, bucket, key):
            uploaded[key] = Path(path).stat().st_size

        with mock.patch.object(offload, "utcnow", return_value=NOW), \
                mock.patch.object(offload, "remote_size", side_effect=lambda k: uploaded.get(k)), \
                mock.patch.object(offload, "object_key_for", side_effect=lambda p, v: f"ws/{v}.mp4"), \
                mock.patch.object(offload, "_precache"), \
                mock.patch("app.storage_supabase.ensure_bucket"), \
                mock.patch("app.storage_supabase.upload_file_resumable", side_effect=fake_upload):
            return offload.sweep(**kw), uploaded

    def test_offloads_clipped_keeps_waiting_deletes_rejected(self):
        done = self.add_video(1, posts=["published"], exported=True)
        waiting = self.add_video(2)
        rejected = self.add_video(3, status="rejected")
        res, uploaded = self.run_sweep()

        self.assertFalse(done.exists())
        self.assertTrue(waiting.exists())
        self.assertFalse(rejected.exists())
        self.assertEqual(uploaded, {"ws/vid1.mp4": 1000})  # rejected never uploaded
        self.assertEqual((res.offloaded, res.deleted, res.failed), (1, 1, 0))
        with self.Session() as s:
            c = s.get(Candidate, 1)
            self.assertEqual(c.source_object_key, "ws/vid1.mp4")
            self.assertIsNotNone(c.source_uploaded_at)
            self.assertEqual(c.local_video_path, str(done))  # kept as restore target
            self.assertEqual(s.get(Candidate, 3).source_object_key, "")

    def test_dry_run_changes_nothing(self):
        done = self.add_video(1, posts=["published"], exported=True)
        res, uploaded = self.run_sweep(dry_run=True)
        self.assertTrue(done.exists())
        self.assertEqual(uploaded, {})
        self.assertEqual([v for v, _, _ in res.planned], ["offload"])

    def test_size_mismatch_keeps_the_file(self):
        done = self.add_video(1, posts=["published"], exported=True)
        with mock.patch.object(offload, "utcnow", return_value=NOW), \
                mock.patch.object(offload, "remote_size", return_value=5), \
                mock.patch.object(offload, "object_key_for", return_value="ws/vid1.mp4"), \
                mock.patch.object(offload, "_precache"), \
                mock.patch("app.storage_supabase.ensure_bucket"), \
                mock.patch("app.storage_supabase.upload_file_resumable"):
            res = offload.sweep()
        self.assertTrue(done.exists())
        self.assertEqual(res.failed, 1)
        with self.Session() as s:
            self.assertEqual(s.get(Candidate, 1).source_object_key, "")

    def test_upload_failure_keeps_the_file(self):
        done = self.add_video(1, posts=["published"], exported=True)
        with mock.patch.object(offload, "utcnow", return_value=NOW), \
                mock.patch.object(offload, "remote_size", return_value=None), \
                mock.patch.object(offload, "object_key_for", return_value="ws/vid1.mp4"), \
                mock.patch.object(offload, "_precache"), \
                mock.patch("app.storage_supabase.ensure_bucket"), \
                mock.patch("app.storage_supabase.upload_file_resumable",
                           side_effect=RuntimeError("413")):
            res = offload.sweep()
        self.assertTrue(done.exists())
        self.assertEqual(res.failed, 1)

    def test_already_uploaded_is_not_reuploaded(self):
        done = self.add_video(1, posts=["published"], exported=True)
        with self.Session() as s:
            s.get(Candidate, 1).source_object_key = "ws/vid1.mp4"
            s.commit()
        upload = mock.Mock()
        with mock.patch.object(offload, "utcnow", return_value=NOW), \
                mock.patch.object(offload, "remote_size", return_value=1000), \
                mock.patch.object(offload, "_precache"), \
                mock.patch("app.storage_supabase.upload_file_resumable", upload):
            res = offload.sweep()
        upload.assert_not_called()
        self.assertFalse(done.exists())
        self.assertEqual(res.offloaded, 1)

    def test_limit_caps_actions(self):
        self.add_video(1, posts=["published"], exported=True)
        self.add_video(2, posts=["published"], exported=True)
        res, uploaded = self.run_sweep(limit=1)
        self.assertEqual(len(uploaded), 1)


class ClipSweepTest(_DB, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.clips = self.dir / "clips"
        self.outro = self.dir / "outro"
        self.clips.mkdir()
        self.outro.mkdir()
        with self.Session() as s:
            s.add(Candidate(id=1, channel_pk=1, video_id="v", title="t", url="u",
                            status="archived"))
            s.commit()

    def file(self, directory, name, age_days=10, size=100):
        path = directory / name
        path.write_bytes(b"x" * size)
        t = (NOW - dt.timedelta(days=age_days)).timestamp()
        os.utime(path, (t, t))
        return path

    def add_cut(self, cid, posts=(), age_days=10, **paths):
        with self.Session() as s:
            s.add(Cut(id=cid, candidate_pk=1, updated_at=NOW - dt.timedelta(days=age_days),
                      **{k: str(v) for k, v in paths.items()}))
            for st, clip in posts:
                s.add(ThreadsPost(candidate_pk=1, cut_pk=cid, status=st,
                                  clip_local_path=str(clip),
                                  created_at=NOW - dt.timedelta(days=age_days)))
            s.commit()

    def run_sweep(self, **kw):
        uploaded = {}

        def fake_upload(path, bucket, key):
            uploaded[key] = Path(path).stat().st_size

        with mock.patch.object(offload, "utcnow", return_value=NOW), \
                mock.patch.object(offload, "_clips_dir", return_value=self.clips), \
                mock.patch("app.outro.OUTRO_DIR", self.outro), \
                mock.patch.object(offload, "remote_size", side_effect=lambda k: uploaded.get(k)), \
                mock.patch.object(offload, "_precache_clip"), \
                mock.patch("app.storage_supabase.ensure_bucket"), \
                mock.patch("app.storage_supabase.upload_file_resumable", side_effect=fake_upload):
            return offload.sweep_clips(**kw), uploaded

    def test_published_cut_is_archived_pending_cut_stays(self):
        done = self.file(self.clips, "a_cut1.mp4")
        done_subs = self.file(self.clips, "a_cut1_subs.mp4")
        queued = self.file(self.clips, "b_cut2.mp4")
        unposted = self.file(self.clips, "c_cut3.mp4")
        self.add_cut(1, posts=[("published", done_subs)], trimmed_clip_path=done,
                     subtitled_clip_path=done_subs)
        self.add_cut(2, posts=[("queued", queued)], trimmed_clip_path=queued)
        self.add_cut(3, trimmed_clip_path=unposted)
        res, uploaded = self.run_sweep()

        self.assertEqual(set(uploaded), {offload.clip_object_key(done),
                                         offload.clip_object_key(done_subs)})
        self.assertFalse(done.exists() or done_subs.exists())
        self.assertTrue(queued.exists() and unposted.exists())
        self.assertEqual(res.offloaded, 2)

    def test_cut_with_published_and_failed_posts_stays(self):
        clip = self.file(self.clips, "a_cut1.mp4")
        self.add_cut(1, posts=[("published", clip), ("failed", clip)], trimmed_clip_path=clip)
        res, uploaded = self.run_sweep()
        self.assertTrue(clip.exists())
        self.assertEqual(uploaded, {})

    def test_recently_touched_cut_stays(self):
        clip = self.file(self.clips, "a_cut1.mp4")
        self.add_cut(1, posts=[("published", clip)], age_days=1, trimmed_clip_path=clip)
        self.run_sweep()
        self.assertTrue(clip.exists())

    def test_orphans_and_stale_outros_deleted_pending_outro_kept(self):
        queued = self.file(self.clips, "b_cut2.mp4")
        self.add_cut(2, posts=[("queued", queued)], trimmed_clip_path=queued)
        orphan = self.file(self.clips, "old_vertical.mp4")
        fresh_orphan = self.file(self.clips, "being_exported.mp4", age_days=0)
        keep_outro = self.file(self.outro, "b_cut2_outro_abc123.mp4")
        stale_outro = self.file(self.outro, "z_cut9_outro_def456.mp4")
        res, uploaded = self.run_sweep()

        self.assertFalse(orphan.exists() or stale_outro.exists())
        self.assertTrue(fresh_orphan.exists() and keep_outro.exists())
        self.assertEqual(uploaded, {})  # deleted, never uploaded
        self.assertEqual(res.deleted, 2)

    def test_transcript_sidecar_of_finished_cut_stays(self):
        clip = self.file(self.clips, "a_cut1.mp4")
        sidecar = self.file(self.clips, "a_cut1_transcript.json")
        self.add_cut(1, posts=[("published", clip)], trimmed_clip_path=clip,
                     clip_transcript_path=sidecar)
        with self.Session() as s:
            s.get(Cut, 1).clip_transcript_path = str(sidecar)
            s.commit()
        self.run_sweep()
        self.assertFalse(clip.exists())
        self.assertTrue(sidecar.exists())

    def test_dry_run_changes_nothing(self):
        clip = self.file(self.clips, "a_cut1.mp4")
        self.add_cut(1, posts=[("published", clip)], trimmed_clip_path=clip)
        orphan = self.file(self.clips, "orphan.mp4")
        res, uploaded = self.run_sweep(dry_run=True)
        self.assertTrue(clip.exists() and orphan.exists())
        self.assertEqual(sorted(v for v, _, _ in res.planned), ["archive", "delete-orphan"])


class EnsureClipFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clips = Path(self.tmp.name)
        offload._clip_misses.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_restores_archived_clip_by_path(self):
        path = self.clips / "a_cut1.mp4"

        def fake_download(bucket, key, dest, expected_size=None):
            self.assertEqual(key, offload.clip_object_key(path))
            Path(dest).write_bytes(b"clip")

        with mock.patch.object(offload, "_clips_dir", return_value=self.clips), \
                mock.patch.object(offload, "remote_size", return_value=4), \
                mock.patch("app.storage_supabase.download_to_file", side_effect=fake_download):
            self.assertTrue(offload.ensure_clip_file(str(path)))
        self.assertEqual(path.read_bytes(), b"clip")

    def test_not_archived_is_remembered(self):
        path = self.clips / "gone.mp4"
        with mock.patch.object(offload, "_clips_dir", return_value=self.clips), \
                mock.patch.object(offload, "remote_size", return_value=None) as rs:
            self.assertFalse(offload.ensure_clip_file(str(path)))
            self.assertFalse(offload.ensure_clip_file(str(path)))
        self.assertEqual(rs.call_count, 1)

    def test_outside_clips_dir_is_never_fetched(self):
        with mock.patch.object(offload, "_clips_dir", return_value=self.clips), \
                mock.patch.object(offload, "remote_size") as rs:
            self.assertFalse(offload.ensure_clip_file("/elsewhere/x.mp4"))
        rs.assert_not_called()


class EnsureLocalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sub" / "v.mp4"

    def tearDown(self):
        self.tmp.cleanup()

    def test_present_file_is_returned_without_storage(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b"abc")
        with mock.patch.object(offload, "remote_size") as rs:
            self.assertEqual(offload.ensure_local(str(self.path), "k"), self.path)
        rs.assert_not_called()

    def test_missing_without_key_raises(self):
        with self.assertRaises(offload.SourceUnavailable):
            offload.ensure_local(str(self.path), "")

    def test_restores_from_storage(self):
        def fake_download(bucket, key, dest, expected_size=None):
            Path(dest).write_bytes(b"abcd")

        with mock.patch.object(offload, "remote_size", return_value=4), \
                mock.patch("app.storage_supabase.download_to_file", side_effect=fake_download):
            out = offload.ensure_local(str(self.path), "k")
        self.assertEqual(out.read_bytes(), b"abcd")

    def test_refuses_when_disk_too_full(self):
        usage = mock.Mock(free=10)
        with mock.patch.object(offload, "remote_size", return_value=4), \
                mock.patch.object(offload.shutil, "disk_usage", return_value=usage), \
                mock.patch("app.storage_supabase.download_to_file") as dl:
            with self.assertRaises(offload.SourceUnavailable):
                offload.ensure_local(str(self.path), "k")
        dl.assert_not_called()


class _Transport:
    """Patch httpx.Client in storage_supabase onto a MockTransport handler."""

    def __init__(self, handler):
        real = httpx.Client
        self.patch = mock.patch.object(
            storage_supabase.httpx, "Client",
            lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        self.env = mock.patch.dict(os.environ, {"SUPABASE_URL": "https://p.supabase.co",
                                                "SUPABASE_SERVICE_KEY": "svc"})

    def __enter__(self):
        self.env.start()
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()
        self.env.stop()


class ResumableUploadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = Path(self.tmp.name) / "big.mp4"
        self.data = os.urandom(storage_supabase._TUS_CHUNK * 2 + 123)
        self.file.write_bytes(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_uploads_in_chunks_and_resumes_after_a_drop(self):
        received = bytearray()
        state = {"dropped": False}

        def handler(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.headers["authorization"], "Bearer svc")
            if req.method == "POST":
                self.assertEqual(req.headers["upload-length"], str(len(self.data)))
                return httpx.Response(201, headers={"Location": "/storage/v1/upload/resumable/abc"})
            if req.method == "HEAD":
                return httpx.Response(200, headers={"Upload-Offset": str(len(received))})
            self.assertEqual(req.method, "PATCH")
            self.assertEqual(int(req.headers["upload-offset"]), len(received))
            body = req.read()
            self.assertLessEqual(len(body), storage_supabase._TUS_CHUNK)
            if len(received) == storage_supabase._TUS_CHUNK and not state["dropped"]:
                state["dropped"] = True
                raise httpx.ConnectError("connection reset")
            received.extend(body)
            return httpx.Response(204, headers={"Upload-Offset": str(len(received))})

        with _Transport(handler), mock.patch.object(storage_supabase.time, "sleep"):
            storage_supabase.upload_file_resumable(self.file, "b", "ws/big.mp4")
        self.assertTrue(state["dropped"])
        self.assertEqual(bytes(received), self.data)

    def test_over_size_limit_is_a_clear_error(self):
        with _Transport(lambda req: httpx.Response(413)):
            with self.assertRaisesRegex(RuntimeError, "file size limit"):
                storage_supabase.upload_file_resumable(self.file, "b", "k")


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "v.mp4"

    def tearDown(self):
        self.tmp.cleanup()

    def test_streams_to_dest(self):
        def handler(req):
            self.assertIn("/storage/v1/object/authenticated/b/ws/a%20b.mp4", str(req.url))
            return httpx.Response(200, content=b"video-bytes")

        with _Transport(handler):
            storage_supabase.download_to_file("b", "ws/a b.mp4", self.dest, expected_size=11)
        self.assertEqual(self.dest.read_bytes(), b"video-bytes")

    def test_short_download_leaves_nothing_behind(self):
        with _Transport(lambda req: httpx.Response(200, content=b"short")):
            with self.assertRaises(RuntimeError):
                storage_supabase.download_to_file("b", "k", self.dest, expected_size=99)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
