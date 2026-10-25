"""Tests for slicing a source transcript down to a cut's trim windows
(app/publishing.py ``_slice_transcript``).

Run with:  python -m unittest tests.test_clip_transcript -v
"""
from __future__ import annotations

import json
import unittest

from app.publishing import _slice_transcript

TRANSCRIPT = """\
[00:06] STORY ONE OPENS
[00:20] STILL STORY ONE
[00:49] STORY TWO STARTS
[01:01] STORY TWO ENDS
[01:14] BACK TO STORY ONE
[1:00:05] LATE LINE
"""


def windows(*spans: tuple[float, float]) -> str:
    return json.dumps([{"start": s, "end": e} for s, e in spans])


class SliceTranscriptTest(unittest.TestCase):
    def test_keeps_only_lines_inside_the_windows(self):
        self.assertEqual(_slice_transcript(TRANSCRIPT, windows((50, 70))),
                         "STORY TWO STARTS STORY TWO ENDS")

    def test_a_line_runs_until_the_next_timestamp(self):
        # 0:30 falls inside the line that starts at 0:20.
        self.assertEqual(_slice_transcript(TRANSCRIPT, windows((30, 40))),
                         "STILL STORY ONE")

    def test_windows_are_joined_in_order(self):
        self.assertEqual(_slice_transcript(TRANSCRIPT, windows((50, 55), (3606, 3607))),
                         "STORY TWO STARTS LATE LINE")

    def test_nothing_without_windows_or_timestamps(self):
        self.assertEqual(_slice_transcript(TRANSCRIPT, ""), "")
        self.assertEqual(_slice_transcript("no timestamps here", windows((0, 10))), "")
        self.assertEqual(_slice_transcript(TRANSCRIPT, "not json"), "")


if __name__ == "__main__":
    unittest.main()
