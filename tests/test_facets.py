"""Tests for facet resolution (app/scheduler.py resolve_facets).

Run with:  python -m unittest tests.test_facets -v

resolve_facets only reads ``format_tags`` off the post/candidate and
``category`` off the candidate, so two-field fakes drive everything.
"""
from __future__ import annotations

import unittest

from app.scheduler import resolve_facets


class FakePost:
    def __init__(self, format_tags: str = ""):
        self.format_tags = format_tags


class FakeCandidate:
    def __init__(self, category: str = "", format_tags: str = ""):
        self.category = category
        self.format_tags = format_tags


class TestUnionMode(unittest.TestCase):
    def test_union_combines_category_and_format_tags(self):
        got = resolve_facets(FakePost("archival_footage, found_footage"),
                             FakeCandidate(category="culture"), "union")
        self.assertEqual(got, frozenset({"culture", "archival_footage", "found_footage"}))

    def test_union_untagged_post_keeps_the_category_floor(self):
        # No format tags anywhere: the set degrades to the category alone,
        # so untagged posts still overlap 0.5 with tagged same-category ones
        # instead of 0.0 (which would wave them past the variety gate).
        got = resolve_facets(FakePost(""), FakeCandidate(category="nature"), "union")
        self.assertEqual(got, frozenset({"nature"}))

    def test_union_falls_back_to_candidate_storyboard_tags(self):
        # Post not yet annotated: the candidate's storyboard prediction fills in.
        got = resolve_facets(FakePost(""),
                             FakeCandidate(category="news", format_tags="local_news_segment"),
                             "union")
        self.assertEqual(got, frozenset({"news", "local_news_segment"}))

    def test_union_post_tags_shadow_candidate_tags(self):
        # Ground truth from the trimmed clip wins over the storyboard guess.
        got = resolve_facets(FakePost("produced_segment"),
                             FakeCandidate(category="news", format_tags="local_news_segment"),
                             "union")
        self.assertEqual(got, frozenset({"news", "produced_segment"}))

    def test_union_without_category(self):
        got = resolve_facets(FakePost("archival_footage"), FakeCandidate(category=""), "union")
        self.assertEqual(got, frozenset({"archival_footage"}))

    def test_union_empty_everything(self):
        self.assertEqual(resolve_facets(None, None, "union"), frozenset())


class TestLegacyModes(unittest.TestCase):
    """category and format modes must keep their pre-union behavior."""

    def test_category_mode_ignores_tags(self):
        got = resolve_facets(FakePost("archival_footage"),
                             FakeCandidate(category="culture"), "category")
        self.assertEqual(got, frozenset({"culture"}))

    def test_format_mode_prefers_tags_alone(self):
        got = resolve_facets(FakePost("archival_footage"),
                             FakeCandidate(category="culture"), "format")
        self.assertEqual(got, frozenset({"archival_footage"}))

    def test_format_mode_falls_back_to_category_when_untagged(self):
        got = resolve_facets(FakePost(""), FakeCandidate(category="culture"), "format")
        self.assertEqual(got, frozenset({"culture"}))


if __name__ == "__main__":
    unittest.main()
