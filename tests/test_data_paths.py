"""models.DataPath: stored file paths read as this machine's paths."""
from __future__ import annotations

import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import DATA_DIR, WORKSPACE
from app.models import Base, Cut, localize_data_path


class LocalizeTests(unittest.TestCase):
    def test_other_machine_path_maps_onto_this_data_dir(self):
        for root in ("/Users/someone/Threads-Poster", "/app", "/home/runner/work/x/x"):
            stored = f"{root}/workspaces/{WORKSPACE}/data/clips/a_subs_1.mp4"
            self.assertEqual(localize_data_path(stored), str(DATA_DIR / "clips/a_subs_1.mp4"))

    def test_nested(self):
        stored = f"/x/workspaces/{WORKSPACE}/data/transcripts/chan/2026-01-01/v.json"
        self.assertEqual(localize_data_path(stored),
                         str(DATA_DIR / "transcripts/chan/2026-01-01/v.json"))

    def test_left_alone(self):
        for value in ("", None, "/tmp/elsewhere.mp4", "relative.mp4",
                      "/x/workspaces/another-workspace/data/clips/a.mp4",
                      f"/x/workspaces/{WORKSPACE}/data/"):
            self.assertEqual(localize_data_path(value), value)


class ColumnTests(unittest.TestCase):
    def test_round_trip_through_the_database(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        stored = f"/Users/someone/repo/workspaces/{WORKSPACE}/data/clips/c.mp4"
        with Session(engine) as s:
            s.add(Cut(candidate_pk=1, trimmed_clip_path=stored))
            s.commit()
        with Session(engine) as s:
            self.assertEqual(s.scalars(select(Cut)).one().trimmed_clip_path,
                             str(DATA_DIR / "clips/c.mp4"))
            self.assertEqual(s.execute(select(Cut.trimmed_clip_path)).scalar_one(),
                             str(DATA_DIR / "clips/c.mp4"))
            # SQL comparisons see the stored form: match on the file name.
            self.assertEqual(s.execute(select(Cut.id).where(
                Cut.trimmed_clip_path.endswith("/c.mp4", autoescape=True))).scalar_one(), 1)


if __name__ == "__main__":
    unittest.main()
