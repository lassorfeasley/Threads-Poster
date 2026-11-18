"""app/shared_docs.py: workspace documents shared through the database."""
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

from app import config, shared_docs
from app.models import AppToken, Base


class SharedDocsTestBase(unittest.TestCase):
    shared = True

    def setUp(self):
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

        def path_for(key):
            return self.dir / shared_docs.DOC_FILES[key][1]

        for patch in (mock.patch("app.db.session_scope", scope),
                      mock.patch.object(shared_docs, "default_path", side_effect=path_for),
                      mock.patch.object(shared_docs, "WORKSPACE", "ws"),
                      mock.patch.dict(os.environ, {"SHARED_WORKSPACE": "on" if self.shared else ""})):
            patch.start()
            self.addCleanup(patch.stop)
        shared_docs._cache.clear()

    def db_value(self, key):
        with self.Session() as s:
            row = s.get(AppToken, f"doc:ws:{key}")
            return row.value if row else None

    def file(self, key) -> Path:
        return self.dir / shared_docs.DOC_FILES[key][1]


class SharedOnTests(SharedDocsTestBase):
    def test_file_seeds_until_first_save(self):
        self.file("brand").write_text("from: file\n")
        self.assertEqual(shared_docs.read_text("brand"), "from: file\n")
        shared_docs.write_text("brand", "from: save\n")
        self.assertEqual(self.db_value("brand"), "from: save\n")
        self.assertEqual(self.file("brand").read_text(), "from: save\n")

    def test_database_wins_over_a_stale_file(self):
        shared_docs.write_text("keywords", "keywords: [a]\n")
        self.file("keywords").write_text("keywords: [stale]\n")   # e.g. a baked image
        shared_docs._cache.clear()
        self.assertEqual(shared_docs.read_text("keywords"), "keywords: [a]\n")

    def test_another_machines_save_shows_up_after_the_cache(self):
        shared_docs.write_text("brand", "v: 1\n")
        self.assertEqual(shared_docs.read_text("brand"), "v: 1\n")
        with self.Session() as s:
            s.get(AppToken, "doc:ws:brand").value = "v: 2\n"
            s.commit()
        self.assertEqual(shared_docs.read_text("brand"), "v: 1\n")    # cached
        shared_docs._cache.clear()
        self.assertEqual(shared_docs.read_text("brand"), "v: 2\n")

    def test_update_appends_onto_the_seed(self):
        # Cross-machine concurrency rests on the row lock (checked against
        # Postgres by hand); here, read-modify-write semantics.
        self.file("notes").write_text("seed\n")
        for i in range(3):
            shared_docs.update_text("notes", lambda cur, i=i: (cur or "") + f"{i}\n")
        self.assertEqual(self.db_value("notes"), "seed\n0\n1\n2\n")
        self.assertEqual(self.file("notes").read_text(), "seed\n0\n1\n2\n")
        self.assertEqual(shared_docs.update_text("notes", lambda cur: None), "seed\n0\n1\n2\n")

    def test_unreachable_database_falls_back_to_file_on_read(self):
        self.file("captions").write_text("rules: []\n")
        with mock.patch.object(shared_docs, "_db_get", side_effect=RuntimeError("down")):
            self.assertEqual(shared_docs.read_text("captions"), "rules: []\n")

    def test_failed_database_save_raises_and_leaves_the_file(self):
        self.file("brand").write_text("old\n")
        with mock.patch.object(shared_docs, "_db_update", side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                shared_docs.write_text("brand", "new\n")
        self.assertEqual(self.file("brand").read_text(), "old\n")

    def test_push_pull_status(self):
        self.file("brand").write_text("mine\n")
        self.assertTrue(shared_docs.push("brand"))
        self.assertFalse(shared_docs.push("playbook"))
        self.file("brand").write_text("edited by hand\n")
        row = next(r for r in shared_docs.status() if r["key"] == "brand")
        self.assertEqual((row["in_db"], row["same"]), (True, False))
        self.assertTrue(shared_docs.pull("brand"))
        self.assertEqual(self.file("brand").read_text(), "mine\n")
        self.assertFalse(shared_docs.pull("playbook"))

    def test_config_loaders_read_the_shared_copy(self):
        shared_docs.write_text("keywords", "keywords: [Flood, heat dome]\n")
        self.assertEqual(config.load_keywords(), ["Flood", "heat dome"])
        config.save_keywords(["wildfire"])
        self.assertIn("wildfire", self.db_value("keywords"))

    def test_row_name_length_guard(self):
        with mock.patch.object(shared_docs, "WORKSPACE", "x" * 40):
            with self.assertRaises(ValueError):
                shared_docs.row_name("brand")


class SharedOffTests(SharedDocsTestBase):
    shared = False

    def test_files_only(self):
        shared_docs.write_text("brand", "a: 1\n")
        self.assertEqual(self.file("brand").read_text(), "a: 1\n")
        self.assertIsNone(self.db_value("brand"))
        self.assertEqual(shared_docs.update_text("brand", lambda cur: cur + "b: 2\n"), "a: 1\nb: 2\n")
        self.assertIsNone(self.db_value("brand"))
        self.assertIsNone(shared_docs.read_text("playbook"))


if __name__ == "__main__":
    unittest.main()
