"""Play order through the clip ledger, and the legacy suggester's quick wins.

A clip's segment list is in PLAY order — a cold open plays its payoff first.
These pin down the order-preserving interval helpers (clean_play_order,
strip_segments), visible_segments, and log_run (which
stores play order and never supersedes anything on an empty run), plus the v3
prompt wording and the no-temperature rule in llm._create.

Run with:  DATABASE_URL="sqlite://" python -m unittest tests.test_play_order -v
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app import clip_proposals, llm, spend
from app.models import Base, Candidate, Channel, ClipProposal


def seg(start, end, **extra):
    return {"start": start, "end": end, **extra}


def spans(segments):
    return [(s["start"], s["end"]) for s in segments]


class CleanPlayOrderTest(unittest.TestCase):
    CASES = [
        # (name, input, kwargs, expected spans)
        ("cold open keeps its order",
         [seg(30, 40), seg(10, 20)], {}, [(30, 40), (10, 20)]),
        ("strings are coerced",
         [seg("1.5", "3.25")], {}, [(1.5, 3.25)]),
        ("unusable entries are dropped",
         [seg(None, 5), {"start": 1}, "x", seg("a", 3), seg(math.nan, 4),
          seg(0, math.inf), seg(5, 5), seg(9, 6), seg(1, 2)], {}, [(1, 2)]),
        ("clamped to floor and horizon",
         [seg(-3, 4), seg(8, 12)], {"floor": 1, "horizon": 10}, [(1, 4), (8, 10)]),
        ("short after clamping is dropped",
         [seg(9.8, 30)], {"horizon": 10, "min_len": 0.5}, []),
        ("overlap with an earlier segment keeps the remainder",
         [seg(10, 20), seg(15, 25)], {}, [(10, 20), (20, 25)]),
        ("a later segment split by an earlier one keeps its longer piece",
         [seg(12, 14), seg(10, 20)], {}, [(12, 14), (14, 20)]),
        ("equal pieces: the earlier one wins",
         [seg(14, 16), seg(10, 20)], {}, [(14, 16), (10, 14)]),
        ("overlaps with several earlier segments",
         [seg(30, 32), seg(10, 12), seg(5, 40)], {}, [(30, 32), (10, 12), (12, 30)]),
        ("fully replayed footage is dropped",
         [seg(10, 20), seg(12, 18)], {}, [(10, 20)]),
        ("a remainder under min_len is dropped",
         [seg(10, 20), seg(19, 20.4)], {"min_len": 0.5}, [(10, 20)]),
        ("touching segments are not merged",
         [seg(10, 20), seg(20, 30)], {}, [(10, 20), (20, 30)]),
        ("rounded to 2 dp",
         [seg(1.23456, 4.98765)], {}, [(1.23, 4.99)]),
        ("rounding never creates an overlap",
         [seg(10.004, 20.006), seg(20.004, 30)], {}, [(10.0, 20.01), (20.01, 30.0)]),
        ("floor None means 0, horizon None means no cap",
         [seg(-1, 5000)], {"floor": None, "horizon": None}, [(0, 5000)]),
    ]

    def test_cases(self):
        for name, segments, kwargs, expected in self.CASES:
            with self.subTest(name):
                self.assertEqual(spans(clip_proposals.clean_play_order(segments, **kwargs)),
                                 expected)

    def test_extra_keys_ride_along_and_input_is_untouched(self):
        original = [seg(30, 40, lead="picture"), seg(35, 50, lead="word", note="x")]
        out = clip_proposals.clean_play_order(original)
        self.assertEqual(out, [seg(30.0, 40.0, lead="picture"),
                               seg(40.0, 50.0, lead="word", note="x")])
        self.assertEqual(original[1]["start"], 35)

    def test_legacy_sorted_input_is_unchanged(self):
        legacy = [seg(10.5, 22.25), seg(30.0, 41.75), seg(50.1, 52.0)]
        self.assertEqual(clip_proposals.clean_play_order(legacy),
                         clip_proposals.normalize(legacy))


class StripSegmentsTest(unittest.TestCase):
    def test_keeps_order_and_only_start_end(self):
        self.assertEqual(
            clip_proposals.strip_segments([seg(30, 40, lead="picture"), seg("1", 5, x=1),
                                           {"start": 2}, None]),
            [{"start": 30.0, "end": 40.0}, {"start": 1.0, "end": 5.0}])


class VisibleSegmentsTest(unittest.TestCase):
    def test_play_order_survives(self):
        kept, trimmed = clip_proposals.visible_segments([seg(100, 115), seg(10, 25)], [])
        self.assertEqual(spans(kept), [(100, 115), (10, 25)])
        self.assertFalse(trimmed)

    def test_footage_other_clips_use_is_never_cut_out(self):
        # Clips may recycle each other's footage: the old "used" argument is
        # ignored and nothing is subtracted, trimmed or dropped.
        for used in ([seg(115, 120)], [seg(0, 1000)], [seg(4, 30)]):
            with self.subTest(used=used):
                kept, trimmed = clip_proposals.visible_segments(
                    [seg(100, 130), seg(10, 25)], used)
                self.assertEqual(spans(kept), [(100, 130), (10, 25)])
                self.assertFalse(trimmed)

    def test_unusable_entries_are_dropped_and_extra_keys_kept(self):
        kept, trimmed = clip_proposals.visible_segments(
            [seg(None, 3), seg(5, 4), seg(1, 4, lead="picture")])
        self.assertEqual(kept, [seg(1.0, 4.0, lead="picture")])
        self.assertFalse(trimmed)
        self.assertEqual(clip_proposals.visible_segments([]), ([], False))

    def test_no_subtraction_helpers_remain(self):
        for name in ("subtract", "subtract_play_order", "used_ranges"):
            self.assertFalse(hasattr(clip_proposals, name), name)


class _LedgerDb(unittest.TestCase):
    """An in-memory SQLite schema with one candidate."""

    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        channel = Channel(call_sign="KTST", url="https://example.com")
        self.session.add(channel)
        self.session.flush()
        self.candidate = Candidate(video_id="vid1", channel_pk=channel.id, title="t", url="u")
        self.session.add(self.candidate)
        self.session.flush()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def pending(self, segments, **kw):
        row = ClipProposal(candidate_pk=self.candidate.id, proposed_segments=json.dumps(segments),
                           verdict=ClipProposal.VERDICT_PENDING, **kw)
        self.session.add(row)
        self.session.flush()
        return row

    def log(self, clips, **kw):
        kw.setdefault("model", "m")
        kw.setdefault("policy", "p")
        return clip_proposals.log_run(self.session, self.candidate.id, clips, **kw)

    def stored(self, row_id):
        row = self.session.get(ClipProposal, row_id)
        return row, json.loads(row.proposed_segments)

    def verdicts(self):
        return [r.verdict for r in self.session.execute(
            select(ClipProposal).order_by(ClipProposal.id)).scalars()]


class LogRunTest(_LedgerDb):
    def test_cold_open_is_stored_in_play_order_without_lead(self):
        [row_id] = self.log([{"segments": [seg(100, 112, lead="picture"), seg(20, 38, lead="word")],
                              "story": "s"}])
        row, segments = self.stored(row_id)
        self.assertEqual(segments, [{"start": 100.0, "end": 112.0}, {"start": 20.0, "end": 38.0}])
        self.assertEqual(row.proposed_segment_count, 2)
        self.assertEqual(row.proposed_duration_s, 30.0)

    def test_overlapping_segments_are_resolved_before_storing(self):
        [row_id] = self.log([{"segments": [seg(30, 35), seg(20, 40)]}])
        _, segments = self.stored(row_id)
        self.assertEqual(spans(segments), [(30, 35), (20, 30)])

    def test_order_survives_through_visible_segments(self):
        [row_id] = self.log([{"segments": [seg(100, 130), seg(10, 40)]}])
        _, segments = self.stored(row_id)
        kept, trimmed = clip_proposals.visible_segments(segments, [seg(115, 120)])
        self.assertEqual(spans(kept), [(100, 130), (10, 40)])
        self.assertFalse(trimmed)

    def test_empty_run_supersedes_nothing(self):
        standing = self.pending([seg(0, 30)])
        for clips in ([], None, [{"segments": []}],
                      [{"segments": [seg(5, 5.2)]}, {"segments": [{"start": "x"}]}, "junk"]):
            with self.subTest(clips=clips):
                self.assertEqual(self.log(clips), [])
                self.assertEqual(self.log(clips, window={"start": 0, "end": 60}), [])
        self.session.refresh(standing)
        self.assertEqual(standing.verdict, ClipProposal.VERDICT_PENDING)
        self.assertIsNone(standing.decided_at)

    def test_real_run_still_supersedes_pending(self):
        self.pending([seg(0, 30)])
        ids = self.log([{"segments": [seg(40, 60)]}])
        self.assertEqual(self.verdicts(), [ClipProposal.VERDICT_SUPERSEDED,
                                           ClipProposal.VERDICT_PENDING])
        self.assertEqual(len(ids), 1)

    def test_section_run_supersedes_only_what_it_overlaps(self):
        inside = self.pending([seg(100, 120)])
        outside = self.pending([seg(500, 520)])
        self.log([{"segments": [seg(110, 130)]}], window={"start": 90, "end": 200})
        self.session.refresh(inside)
        self.session.refresh(outside)
        self.assertEqual(inside.verdict, ClipProposal.VERDICT_SUPERSEDED)
        self.assertEqual(outside.verdict, ClipProposal.VERDICT_PENDING)

    def test_legacy_sorted_input_is_stored_as_before(self):
        legacy = [seg(10.5, 22.25), seg(30.0, 41.75)]
        [row_id] = self.log([{"segments": legacy}])
        _, segments = self.stored(row_id)
        self.assertEqual(segments, clip_proposals.normalize(legacy))

    def test_skipped_clip_keeps_index_semantics(self):
        ids = self.log([{"segments": []}, {"segments": [seg(0, 20)]}])
        row, _ = self.stored(ids[0])
        self.assertEqual((row.clip_index, row.clips_in_run), (1, 2))

    def test_overlong_model_is_truncated_to_the_column(self):
        [row_id] = self.log([{"segments": [seg(0, 20)]}], model="engine:hybrid:" + "x" * 80)
        row, _ = self.stored(row_id)
        self.assertEqual(len(row.model), 60)


class ProposeTest(_LedgerDb):
    """The legacy suggester's whole-video pass, with the model mocked."""

    def propose_with(self, clips):
        settings = SimpleNamespace(get=lambda key, default=None: default)
        with mock.patch.object(clip_proposals, "suggest_clips", return_value=clips) as model, \
                mock.patch("app.clip_context.build", return_value={}), \
                mock.patch.object(clip_proposals, "load_words", return_value=[]), \
                mock.patch.object(clip_proposals.vision, "load_striking", return_value=[]):
            ids = clip_proposals.propose(self.session, self.candidate, settings,
                                         [{"start": 0, "end": 60, "text": "x"}])
        self.assertEqual(model.call_count, 1)
        return ids

    def test_an_empty_pass_changes_nothing(self):
        standing = self.pending([seg(0, 30)])
        self.candidate.multi_clip_potential = self.candidate.multi_clip_auto = True
        self.candidate.draft_caption = "kept"
        self.assertEqual(self.propose_with([]), [])
        self.session.refresh(standing)
        self.assertEqual(standing.verdict, ClipProposal.VERDICT_PENDING)
        self.assertTrue(self.candidate.multi_clip_potential)
        self.assertEqual(self.candidate.draft_caption, "kept")

    def test_a_real_pass_records_play_order_and_updates_the_markers(self):
        self.candidate.multi_clip_potential = self.candidate.multi_clip_auto = True
        [row_id] = self.propose_with([{"segments": [seg(40, 50), seg(10, 20)], "story": "s",
                                       "draft_caption": "new caption"}])
        _, segments = self.stored(row_id)
        self.assertEqual(spans(segments), [(40, 50), (10, 20)])
        self.assertEqual(self.candidate.draft_caption, "new caption")
        self.assertFalse(self.candidate.multi_clip_potential)


class VersionTest(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(clip_proposals.PROMPT_VERSION, "v3")
        self.assertEqual(clip_proposals.ENGINE_PROMPT_VERSION, "engine-v1")
        base = dict(model="m", max_clips=3, max_segments_per_clip=4, min_seconds=15,
                    max_seconds=40)
        self.assertNotEqual(clip_proposals.policy_version(**base, prompt="v2"),
                            clip_proposals.policy_version(**base, prompt="v3"))


class LegacySuggesterPromptTest(unittest.TestCase):
    def test_single_takes_are_not_discouraged(self):
        captured = {}

        def fake(model, system, user, max_tokens=0, temperature=0.2):
            captured["system"] = system
            return {"clips": []}

        transcript = [{"start": float(i * 5), "end": i * 5 + 5.0, "text": f"line {i}"}
                      for i in range(20)]
        with mock.patch.object(llm, "_json_chat", side_effect=fake):
            llm.suggest_clips("m", "Title", transcript)
        self.assertIn("Many good clips are one continuous take; join segments only when "
                      "cutting makes the clip tighter.", captured["system"])
        self.assertNotIn("2-4 segments joined together", captured["system"])


class CreateTemperatureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(spend, "_LEDGER", Path(self.tmp.name) / "llm_spend.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls: list[dict] = []
        self.failures: list[Exception] = []

        def create(**kwargs):
            self.calls.append(dict(kwargs))
            if self.failures:
                raise self.failures.pop(0)
            return SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=3,
                                                                     output_tokens=4))

        fake = SimpleNamespace(messages=SimpleNamespace(create=create))
        patcher = mock.patch.object(llm, "client", return_value=fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_thinking_models_get_no_temperature(self):
        for model in ("claude-opus-5-5", "claude-sonnet-5", "claude-sonnet-5-5",
                      "claude-haiku-5-5", "claude-fable-1", "claude-mythos-preview",
                      "claude-opus-4-7", "claude-opus-4-8-20260101", " Claude-Opus-5-5 "):
            with self.subTest(model):
                self.calls.clear()
                llm._create(model, "sys", "hi", 100, 0.2)
                self.assertNotIn("temperature", self.calls[0])
                self.assertEqual(len(self.calls), 1)

    def test_older_models_keep_temperature(self):
        for model in ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-6"):
            with self.subTest(model):
                self.calls.clear()
                llm._create(model, "sys", "hi", 100, 0.3)
                self.assertEqual(self.calls[0]["temperature"], 0.3)

    def test_retry_without_temperature_when_rejected(self):
        self.failures.append(RuntimeError("temperature is not supported for this model"))
        llm._create("claude-haiku-4-5", "sys", "hi", 100, 0.2)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("temperature", self.calls[0])
        self.assertNotIn("temperature", self.calls[1])

    def test_no_pointless_retry_when_temperature_was_never_sent(self):
        self.failures.append(RuntimeError("temperature and top_p are deprecated"))
        with self.assertRaises(RuntimeError):
            llm._create("claude-opus-5-5", "sys", "hi", 100, 0.2)
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
