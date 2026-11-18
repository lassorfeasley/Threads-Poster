"""Clip players: the file from disk when it's here, else a redirect to Storage.

  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false .venv/bin/python -m pytest -q tests/test_media_fallback.py
"""
from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, Candidate, Channel, Cut, ThreadsPost


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class MediaFallbackTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.web import main

        self.main = main
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        engine = create_engine("sqlite://", poolclass=StaticPool,
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.addCleanup(engine.dispose)
        self.Session = sessionmaker(engine, expire_on_commit=False)

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            session = self.Session()
            try:
                yield session
                if not read_only:
                    session.commit()
            finally:
                session.close()

        for patch in (mock.patch.object(main, "session_scope", scope),
                      mock.patch.dict(os.environ, {"DASHBOARD_USERS": ""})):
            patch.start()
            self.addCleanup(patch.stop)
        self.client = TestClient(main.app, follow_redirects=False)

        self.trim = self.dir / "trim.mp4"
        self.subs = self.dir / "trim_subs.mp4"
        self.own = self.dir / "post.mp4"
        with self.Session() as s:
            ch = Channel(call_sign="KXYZ", url="https://example.com/c")
            s.add(ch)
            s.flush()
            cand = Candidate(video_id="v1", channel_pk=ch.id, title="t", url="u")
            s.add(cand)
            s.flush()
            cut = Cut(candidate_pk=cand.id, trimmed_clip_path=str(self.trim),
                      subtitled_clip_path=str(self.subs), use_subtitles=True)
            s.add(cut)
            s.flush()
            s.add(ThreadsPost(cut_pk=cut.id, candidate_pk=cand.id, status="queued",
                              clip_local_path=str(self.own), clip_object_path="pub/post.mp4"))
            s.commit()
            self.cut_id = cut.id
            self.post_id = s.query(ThreadsPost).one().id

    def stored(self, available: dict[str, str]):
        """Patch Storage: path or publish key -> signed URL."""
        def fake(path, publish_key=""):
            return available.get(path) or available.get(publish_key) or ""
        return mock.patch.object(self.main, "_stored_clip_url", side_effect=fake)

    def test_local_file_served_from_disk(self):
        self.trim.write_bytes(b"clip")
        with self.stored({}) as stored:
            r = self.client.get(f"/media/clip/{self.cut_id}")
        self.assertEqual((r.status_code, r.content), (200, b"clip"))
        stored.assert_not_called()

    def test_missing_file_redirects_to_storage(self):
        with self.stored({str(self.trim): "https://storage/trim"}):
            r = self.client.get(f"/media/clip/{self.cut_id}")
        self.assertEqual((r.status_code, r.headers["location"]), (307, "https://storage/trim"))

    def test_missing_everywhere_is_404(self):
        with self.stored({}):
            self.assertEqual(self.client.get(f"/media/vertical/{self.cut_id}").status_code, 404)
            self.assertEqual(self.client.get("/media/clip/999").status_code, 404)

    def test_post_prefers_any_local_file_in_the_old_order(self):
        self.trim.write_bytes(b"plain")
        with self.stored({str(self.own): "https://storage/own"}) as stored:
            r = self.client.get(f"/media/post/{self.post_id}")
        self.assertEqual(r.content, b"plain")   # disk first, as on the Mac today
        stored.assert_not_called()
        self.subs.write_bytes(b"captioned")
        self.assertEqual(self.client.get(f"/media/post/{self.post_id}").content, b"captioned")
        self.own.write_bytes(b"own")
        self.assertEqual(self.client.get(f"/media/post/{self.post_id}").content, b"own")

    def test_post_from_storage_prefers_its_own_clip(self):
        with self.stored({"pub/post.mp4": "https://storage/publish",
                          str(self.trim): "https://storage/trim"}):
            r = self.client.get(f"/media/post/{self.post_id}")
        self.assertEqual(r.headers["location"], "https://storage/publish")
        with self.stored({str(self.subs): "https://storage/subs", str(self.trim): "https://storage/trim"}):
            r = self.client.get(f"/media/post/{self.post_id}")
        self.assertEqual(r.headers["location"], "https://storage/subs")


if __name__ == "__main__":
    unittest.main()
