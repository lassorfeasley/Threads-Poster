"""Tests for the session-free transcript suggester (app/clip_engine/legacy.py)
and the per-proposal sidecar files (app/clip_engine/sidecars.py).

``legacy.suggest`` must ask the model exactly what ``clip_proposals.propose``
asks — the bench's transcript arm and the engine's fallback are only honest
if they are the suggester that really runs — so the central test runs both
under the same fakes and compares the calls. ``legacy.write`` is checked
against a real in-memory SQLite ledger. Sidecars are checked on disk,
including every way a reader can meet a bad file.

Run with:  python -m pytest -q tests/test_clip_engine_legacy_sidecars.py
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app import clip_context, clip_proposals, llm, spend, vision
from app.clip_engine import legacy, sidecars, util
from app.config import Settings
from app.models import Base, Candidate, Channel, ClipProposal

SETTINGS = Settings(raw={"clips": {
    "model": "claude-sonnet-5", "max_clips": 3, "section_max_clips": 2, "max_segments": 4,
    "min_seconds": 15, "max_seconds": 40, "opening_hold_seconds": 3.0}})
TRANSCRIPT = [{"start": float(i * 5), "end": i * 5 + 5.0, "text": f"line {i}"} for i in range(60)]
WORDS = [{"word": "Hello", "start": 10.0, "end": 10.4}, {"word": "there.", "start": 10.5, "end": 11.0}]
USED = [{"start": 200.0, "end": 230.0}]
CONTEXT = {"shipped_examples": [{"words": "a shipped clip"}], "rejected_here": []}
STRIKING = [{"t": 12.0, "why": "a burning roof"}]
CLIPS = [
    {"segments": [{"start": 10.0, "end": 30.0}], "story": "first story", "hook": "", "payoff": "",
     "why": "w1", "confidence": 0.8, "draft_caption": "caption one", "challenges_rules": ""},
    {"segments": [{"start": 100.0, "end": 120.0}, {"start": 125.0, "end": 130.0}],
     "story": "second story", "hook": "", "payoff": "", "why": "w2", "confidence": 0.6,
     "draft_caption": "caption two", "challenges_rules": "a flag"},
]


class _TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        for p in (mock.patch.object(spend, "_LEDGER", self.tmp / "llm_spend.json"),
                  mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.tmp / "engine")})):
            p.start()
            self.addCleanup(p.stop)


class LegacySuggestTest(_TempDirTest):
    def setUp(self):
        super().setUp()
        self.video = self.tmp / "source.mp4"
        self.video.write_bytes(b"\x00" * 64)
        self.model_calls: list[tuple] = []
        self.refine_calls: list[dict] = []

    def fake_suggest(self, *args, **kwargs):
        self.model_calls.append((copy.deepcopy(args), copy.deepcopy(kwargs)))
        return copy.deepcopy(CLIPS)

    def fake_refine(self, candidate, segments, settings, blocked, story=""):
        self.refine_calls.append({
            "path": candidate.local_video_path, "title": candidate.title,
            "video_id": candidate.video_id, "segments": copy.deepcopy(segments),
            "blocked": copy.deepcopy(blocked), "story": story, "settings": settings})
        out = [dict(s) for s in segments]
        out[0]["start"] = round(out[0]["start"] - 1.0, 2)
        return out

    def patch_model(self):
        """The model call, wherever ``propose`` and ``legacy`` look it up."""
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(llm, "suggest_clips", new=self.fake_suggest))
        if hasattr(clip_proposals, "suggest_clips"):
            stack.enter_context(mock.patch.object(clip_proposals, "suggest_clips",
                                                  new=self.fake_suggest))
        stack.enter_context(mock.patch.object(clip_proposals, "refine_opening",
                                              new=self.fake_refine))
        return stack

    def inputs(self, **extra):
        out = {"candidate": {"id": 7, "video_id": "vid7", "title": "A title",
                             "description": "A description",
                             "local_video_path": str(self.video), "transcript_path": "",
                             "word_transcript_path": ""},
               "used": list(USED), "context": dict(CONTEXT), "striking": list(STRIKING),
               "transcript": list(TRANSCRIPT), "words": list(WORDS)}
        out.update(extra)
        return out

    def run_propose(self, window=None, topic=""):
        candidate = SimpleNamespace(
            id=7, video_id="vid7", title="A title", description="A description",
            local_video_path=str(self.video), transcript_path="", word_transcript_path="",
            striking_frames="", draft_caption="", multi_clip_potential=False,
            multi_clip_auto=False)
        logged = {}

        def fake_log_run(session, candidate_pk, clips, *, model, policy, window=None):
            logged.update(model=model, policy=policy, clips=copy.deepcopy(clips))
            return [1, 2]

        with self.patch_model(), \
                mock.patch.object(clip_proposals, "used_ranges", return_value=list(USED)), \
                mock.patch.object(clip_context, "build", return_value=dict(CONTEXT)), \
                mock.patch.object(clip_proposals, "load_words", return_value=list(WORDS)), \
                mock.patch.object(vision, "load_striking", return_value=list(STRIKING)), \
                mock.patch.object(clip_proposals, "log_run", new=fake_log_run):
            clip_proposals.propose(mock.Mock(), candidate, SETTINGS, list(TRANSCRIPT),
                                   window=window, topic=topic)
        return logged

    def test_asks_the_model_exactly_what_propose_asks(self):
        for window, topic in ((None, ""), ({"start": 50.0, "end": 200.0}, "Chapter 2 — floods")):
            with self.subTest(window=window):
                self.model_calls.clear()
                self.refine_calls.clear()
                logged = self.run_propose(window=window, topic=topic)
                with self.patch_model():
                    out = legacy.suggest(settings=SETTINGS, inputs=self.inputs(),
                                         window=window, topic=topic)
                self.assertEqual(len(self.model_calls), 2)
                self.assertEqual(self.model_calls[1], self.model_calls[0])
                self.assertEqual(len(self.refine_calls), 4)
                self.assertEqual(self.refine_calls[2:], self.refine_calls[:2])
                self.assertEqual(out["policy"], logged["policy"])
                self.assertEqual(out["model"], logged["model"])
                self.assertEqual(out["clips"], logged["clips"])

    def test_section_pass_budget_and_scope(self):
        with self.patch_model():
            out = legacy.suggest(settings=SETTINGS, inputs=self.inputs(),
                                 window={"start": 50.0, "end": 200.0}, topic="floods")
        args, kwargs = self.model_calls[0]
        self.assertEqual(kwargs["max_clips"], 2)
        self.assertEqual(kwargs["window"], {"start": 50.0, "end": 200.0})
        self.assertEqual(kwargs["topic"], "floods")
        self.assertEqual(out["policy"], clip_proposals.policy_version(
            model="claude-sonnet-5", max_clips=2, max_segments_per_clip=4, min_seconds=15,
            max_seconds=40, scope="section", prompt=clip_proposals.PROMPT_VERSION))
        self.assertNotEqual(out["policy"], legacy.policy_for(SETTINGS))

    def test_openings_use_the_inflight_source_and_block_siblings(self):
        link = self.tmp / "inflight.mp4"
        link.write_bytes(b"\x00")
        with self.patch_model():
            out = legacy.suggest(settings=SETTINGS, inputs=self.inputs(source=str(link)))
        first, second = self.refine_calls
        self.assertEqual(first["path"], str(link))
        self.assertEqual(first["blocked"], USED)
        self.assertEqual(second["blocked"], USED + out["clips"][0]["segments"])
        self.assertEqual(second["story"], "second story")
        self.assertEqual(out["clips"][0]["segments"][0]["start"], 9.0)

    def test_no_video_on_disk_skips_frame_sampling(self):
        inputs = self.inputs(source=str(self.tmp / "gone.mp4"))
        inputs["candidate"]["local_video_path"] = str(self.tmp / "offloaded.mp4")
        with self.patch_model():
            legacy.suggest(settings=SETTINGS, inputs=inputs)
        self.assertEqual({c["path"] for c in self.refine_calls}, {""})

    def test_a_refinement_failure_keeps_the_paid_for_clip(self):
        def broken(*a, **k):
            raise RuntimeError("ffmpeg vanished")

        with self.patch_model(), mock.patch.object(clip_proposals, "refine_opening", new=broken):
            out = legacy.suggest(settings=SETTINGS, inputs=self.inputs())
        self.assertEqual([c["segments"] for c in out["clips"]], [c["segments"] for c in CLIPS])

    def test_no_transcript_no_model_call(self):
        with self.patch_model():
            out = legacy.suggest(settings=SETTINGS, inputs=self.inputs(transcript=[]))
        self.assertEqual(out["clips"], [])
        self.assertEqual(self.model_calls, [])
        self.assertEqual(out["policy"], legacy.policy_for(SETTINGS))
        self.assertEqual(out["cost_usd"], 0.0)

    def test_reads_transcript_and_words_from_the_candidates_files(self):
        transcript_path = self.tmp / "t.json"
        words_path = self.tmp / "t_words.json"
        transcript_path.write_text(json.dumps(TRANSCRIPT))
        words_path.write_text(json.dumps(WORDS))
        inputs = self.inputs()
        del inputs["transcript"], inputs["words"]
        inputs["candidate"].update(transcript_path=str(transcript_path),
                                   word_transcript_path=str(words_path))
        with self.patch_model():
            legacy.suggest(settings=SETTINGS, inputs=inputs)
        args, kwargs = self.model_calls[0]
        self.assertEqual(args[2], TRANSCRIPT)
        self.assertEqual(kwargs["words"], WORDS)

    def test_explicit_context_wins(self):
        replay = {"shipped_examples": []}
        with self.patch_model():
            legacy.suggest(settings=SETTINGS, inputs=self.inputs(), context=replay)
            legacy.suggest(settings=SETTINGS, inputs=self.inputs(context=None))
        self.assertEqual(self.model_calls[0][1]["context"], replay)
        self.assertEqual(self.model_calls[1][1]["context"], {})

    def test_cost_is_what_this_thread_spent(self):
        class Meter:
            total = 0.0

        meter = Meter()

        @contextlib.contextmanager
        def metered():
            yield meter

        def paid_suggest(*args, **kwargs):
            meter.total += 0.0421
            return copy.deepcopy(CLIPS)

        with mock.patch.object(spend, "metered", new=metered, create=True), \
                mock.patch.object(llm, "suggest_clips", new=paid_suggest), \
                mock.patch.object(clip_proposals, "refine_opening", new=self.fake_refine):
            out = legacy.suggest(settings=SETTINGS, inputs=self.inputs())
        self.assertEqual(out["cost_usd"], 0.0421)

    def test_cost_comes_from_the_real_spend_meter(self):
        """Through the real spend.metered: what the model calls record on this
        thread is the run's cost; a call recorded on another thread is not."""
        def recording_suggest(*args, **kwargs):
            spend.record("claude-haiku-4-5", 1_000_000, 0, bucket="engine")   # $1.00
            other = threading.Thread(target=spend.record,
                                     args=("claude-haiku-4-5", 1_000_000, 1_000_000))
            other.start()
            other.join()
            return copy.deepcopy(CLIPS)

        with mock.patch.object(spend, "load_settings", return_value=Settings(raw={})), \
                mock.patch.object(llm, "suggest_clips", new=recording_suggest), \
                mock.patch.object(clip_proposals, "refine_opening", new=self.fake_refine):
            out = legacy.suggest(settings=SETTINGS, inputs=self.inputs())
        self.assertEqual(out["cost_usd"], 1.0)

    def test_model_errors_propagate(self):
        def down(*a, **k):
            raise RuntimeError("overloaded")

        with mock.patch.object(llm, "suggest_clips", new=down):
            with self.assertRaises(RuntimeError):
                legacy.suggest(settings=SETTINGS, inputs=self.inputs())

    def test_model_chain_treats_blanks_as_unset(self):
        cases = [
            ({"clips": {"model": "claude-sonnet-5"}}, "claude-sonnet-5"),
            ({"clips": {"model": ""}, "matching": {"model": "claude-haiku-4-5"}}, "claude-haiku-4-5"),
            ({"clips": {"model": None}}, "claude-haiku-4-5"),
            ({}, "claude-haiku-4-5"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(legacy.suggester_model(Settings(raw=raw)), expected)


class LegacyWriteTest(unittest.TestCase):
    def setUp(self):
        self.db = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.db)
        self.session = Session(self.db)
        channel = Channel(call_sign="KTST", url="https://example.com/c")
        self.session.add(channel)
        self.session.flush()
        self.video = Candidate(video_id="vid1", channel_pk=channel.id, title="T", url="u")
        self.session.add(self.video)
        self.session.flush()

    def tearDown(self):
        self.session.close()
        self.db.dispose()

    def pending(self, start=300.0):
        row = ClipProposal(candidate_pk=self.video.id, verdict=ClipProposal.VERDICT_PENDING,
                           proposed_segments=json.dumps([{"start": start, "end": start + 20}]))
        self.session.add(row)
        self.session.flush()
        return row

    def rows(self):
        return self.session.execute(
            select(ClipProposal).where(ClipProposal.candidate_pk == self.video.id)
            .order_by(ClipProposal.id)).scalars().all()

    def test_logs_the_run_and_the_whole_video_side_effects(self):
        ids = legacy.write(self.session, self.video, copy.deepcopy(CLIPS),
                           model="claude-sonnet-5", policy="abc123")
        self.assertEqual(len(ids), 2)
        new = [r for r in self.rows() if r.id in ids]
        self.assertEqual([r.model for r in new], ["claude-sonnet-5"] * 2)
        self.assertEqual([r.policy_version for r in new], ["abc123"] * 2)
        self.assertEqual(new[1].flag, "a flag")
        self.assertEqual(self.video.draft_caption, "caption one")
        self.assertTrue(self.video.multi_clip_potential)
        self.assertTrue(self.video.multi_clip_auto)

    def test_engine_picks_count_stories_not_clips(self):
        cases = [(["nye-fire", "nye-fire"], False), (["nye-fire", "polar-bear"], True),
                 (["nye-fire", ""], True), ([None, None], True)]
        for keys, multi in cases:
            with self.subTest(keys=keys):
                self.video.multi_clip_potential = False
                self.video.multi_clip_auto = False
                clips = [dict(c, story_key=k) for c, k in zip(copy.deepcopy(CLIPS), keys)]
                legacy.write(self.session, self.video, clips, model="m", policy="p")
                self.assertEqual(self.video.multi_clip_potential, multi)

    def test_a_marker_set_by_hand_is_kept(self):
        self.video.multi_clip_potential = True
        self.video.multi_clip_auto = False
        legacy.write(self.session, self.video, copy.deepcopy(CLIPS[:1]), model="m", policy="p")
        self.assertTrue(self.video.multi_clip_potential)
        self.assertFalse(self.video.multi_clip_auto)

    def test_an_automatic_marker_follows_the_new_run(self):
        self.video.multi_clip_potential = True
        self.video.multi_clip_auto = True
        legacy.write(self.session, self.video, copy.deepcopy(CLIPS[:1]), model="m", policy="p")
        self.assertFalse(self.video.multi_clip_potential)

    def test_a_section_write_leaves_the_video_alone(self):
        self.video.draft_caption = "seed"
        ids = legacy.write(self.session, self.video, copy.deepcopy(CLIPS), model="m",
                           policy="p", window={"start": 0.0, "end": 150.0})
        self.assertEqual(len(ids), 2)
        self.assertEqual(self.video.draft_caption, "seed")
        self.assertFalse(self.video.multi_clip_potential)

    def test_an_empty_run_supersedes_nothing(self):
        standing = self.pending()
        for clips in ([], None, [{"segments": []}, {"story": "no segments"}]):
            with self.subTest(clips=clips):
                self.assertEqual(legacy.write(self.session, self.video, clips, model="m",
                                              policy="p"), [])
                self.session.refresh(standing)
                self.assertEqual(standing.verdict, ClipProposal.VERDICT_PENDING)
        self.assertEqual(self.video.draft_caption, "")

    def test_a_run_that_cleans_away_to_nothing_changes_nothing(self):
        """Every segment a sliver ``log_run`` cleans away: no rows, the standing
        suggestion stays pending — and so do the caption seed and the
        multi-clip marker that describe it (as in ``propose``)."""
        standing = self.pending()
        self.video.draft_caption = "the standing caption"
        self.video.multi_clip_potential = False
        self.video.multi_clip_auto = False
        clips = [{"segments": [{"start": 5.0, "end": 5.3}], "story": "a",
                  "draft_caption": "NEW CAPTION"},
                 {"segments": [{"start": 9.0, "end": 9.2}], "story": "b"}]
        self.assertEqual(legacy.write(self.session, self.video, clips, model="m", policy="p"), [])
        self.session.refresh(standing)
        self.assertEqual(standing.verdict, ClipProposal.VERDICT_PENDING)
        self.assertEqual(self.video.draft_caption, "the standing caption")
        self.assertFalse(self.video.multi_clip_potential)
        self.assertFalse(self.video.multi_clip_auto)

    def test_clips_without_segments_are_dropped_not_counted(self):
        clips = [{"segments": []}] + copy.deepcopy(CLIPS[:1])
        ids = legacy.write(self.session, self.video, clips, model="m", policy="p")
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.rows()[-1].clips_in_run, 1)

    def test_labels_fit_their_columns(self):
        legacy.write(self.session, self.video, copy.deepcopy(CLIPS[:1]),
                     model="engine:gemini:" + "x" * 80, policy="p" * 50)
        row = self.rows()[-1]
        self.assertEqual(len(row.model), 60)
        self.assertEqual(len(row.policy_version), 40)


class SidecarsTest(_TempDirTest):
    def render(self, name="draft.mp4", data=b"\x00\x01video bytes"):
        path = self.tmp / name
        path.write_bytes(data)
        return path

    META = {"setup": "gemini", "archetype": "voice", "lens": "person", "hook": "Nye, glasses off",
            "rank": 1, "score": 7.5, "critique": {"stop_reason": "a famous face", "error": ""},
            "credits": "0-12 s: Bill Nye", "segments": [{"start": 20.0, "end": 30.0},
                                                       {"start": 5.0, "end": 12.0}],
            "notes": ["d2 skipped"]}

    def test_directory_lives_under_the_engine_dir(self):
        self.assertEqual(sidecars.directory(), self.tmp / "engine" / "proposals")
        self.assertTrue(sidecars.directory().is_dir())

    def test_save_load_and_preview(self):
        render = self.render()
        sidecars.save(12, meta=dict(self.META), render=render)
        meta = sidecars.load(12)
        self.assertEqual(meta["proposal_id"], 12)
        self.assertEqual(meta["segments"], self.META["segments"])     # play order kept
        self.assertEqual(meta["critique"]["stop_reason"], "a famous face")
        self.assertTrue(meta["created_at"].endswith("Z"))
        self.assertEqual(meta["preview_file"], {"name": "12.mp4", "bytes": render.stat().st_size})
        preview = sidecars.preview_path(12)
        self.assertEqual(preview, sidecars.directory() / "12.mp4")
        self.assertEqual(preview.read_bytes(), render.read_bytes())
        self.assertEqual(sidecars.load("12")["proposal_id"], 12)

    def test_created_at_is_kept_when_given(self):
        sidecars.save(3, meta={"created_at": "2026-11-18T10:00:00Z"}, render=None)
        self.assertEqual(sidecars.load(3)["created_at"], "2026-11-18T10:00:00Z")

    def test_preview_is_a_hardlink_when_possible_else_a_copy(self):
        render = self.render()
        sidecars.save(4, meta={}, render=render)
        self.assertEqual(os.stat(sidecars.preview_path(4)).st_ino, os.stat(render).st_ino)
        # The engine dropping its render afterwards leaves the preview intact.
        render.unlink()
        self.assertIsNotNone(sidecars.preview_path(4))
        render = self.render("other.mp4")
        with mock.patch.object(os, "link", side_effect=OSError("cross-device")):
            sidecars.save(5, meta={}, render=render)
        self.assertNotEqual(os.stat(sidecars.preview_path(5)).st_ino, os.stat(render).st_ino)
        self.assertEqual(sidecars.preview_path(5).read_bytes(), render.read_bytes())

    def test_saving_without_a_render_removes_an_old_preview(self):
        sidecars.save(6, meta={}, render=self.render())
        self.assertIsNotNone(sidecars.preview_path(6))
        for render in (None, self.tmp / "missing.mp4", self.render("empty.mp4", b"")):
            with self.subTest(render=render):
                sidecars.save(6, meta={"setup": "gemini"}, render=render)
                self.assertIsNone(sidecars.preview_path(6))
                self.assertFalse((sidecars.directory() / "6.mp4").exists())
                self.assertIsNone(sidecars.load(6)["preview_file"])
                self.assertEqual(sidecars.load(6)["setup"], "gemini")

    def test_a_preview_must_match_its_sidecar(self):
        sidecars.save(7, meta={}, render=self.render())
        mp4 = sidecars.directory() / "7.mp4"
        mp4.write_bytes(b"someone else's footage, longer")
        self.assertIsNone(sidecars.preview_path(7))
        sidecars.save(8, meta={}, render=self.render())
        (sidecars.directory() / "8.json").unlink()
        self.assertIsNone(sidecars.preview_path(8))       # an mp4 with no sidecar
        self.assertIsNone(sidecars.preview_path(9))       # nothing at all

    def test_bad_files_read_as_nothing(self):
        directory = sidecars.directory()
        cases = {21: "{half a fi", 22: "", 23: "[1, 2]", 24: "null"}
        for pid, text in cases.items():
            with self.subTest(text=text):
                (directory / f"{pid}.json").write_text(text)
                (directory / f"{pid}.mp4").write_bytes(b"\x00")
                self.assertIsNone(sidecars.load(pid))
                self.assertIsNone(sidecars.preview_path(pid))
        (directory / "25.json").write_text(json.dumps({"preview_file": {"name": "25.mp4",
                                                                        "bytes": "lots"}}))
        (directory / "25.mp4").write_bytes(b"\x00")
        self.assertIsNone(sidecars.preview_path(25))

    def test_bad_ids_never_raise_or_escape_the_directory(self):
        for bad in ("../../etc/passwd", "12.json", None, -3, 0, "abc", 3.5, True, "012"):
            with self.subTest(bad=bad):
                self.assertIsNone(sidecars.load(bad))
                self.assertIsNone(sidecars.preview_path(bad))
                sidecars.save(bad, meta={"x": 1}, render=self.render())
        self.assertEqual(sorted(p.name for p in sidecars.directory().iterdir()), [])

    def test_save_never_raises(self):
        with mock.patch.object(util, "atomic_write_json", side_effect=OSError("disk full")):
            sidecars.save(30, meta={}, render=self.render())
        self.assertIsNone(sidecars.load(30))
        with mock.patch.object(sidecars, "directory", side_effect=PermissionError("ro")):
            sidecars.save(31, meta={}, render=None)
            self.assertIsNone(sidecars.load(31))
            self.assertIsNone(sidecars.preview_path(31))
            self.assertEqual(sidecars.prune(), 0)

    def test_meta_with_paths_and_odd_values_still_saves(self):
        sidecars.save(32, meta={"render": self.tmp / "r.mp4", "when": time.gmtime(0)},
                      render=None)
        self.assertEqual(sidecars.load(32)["render"], str(self.tmp / "r.mp4"))

    def test_prune_by_the_newest_file_of_each_pair(self):
        old = time.time() - 50 * 86400
        for pid in (1, 2, 3):
            sidecars.save(pid, meta={}, render=self.render(f"r{pid}.mp4"))
        directory = sidecars.directory()
        for name in ("1.json", "1.mp4", "3.json"):
            os.utime(directory / name, (old, old))
        stale_tmp = directory / ".9.json.123.456.tmp"
        fresh_tmp = directory / ".10.json.123.456.tmp"
        stale_tmp.write_text("x")
        fresh_tmp.write_text("x")
        os.utime(stale_tmp, (old, old))
        self.assertEqual(sidecars.prune(45), 3)
        self.assertIsNone(sidecars.load(1))
        self.assertFalse((directory / "1.mp4").exists())
        self.assertIsNotNone(sidecars.preview_path(2))
        self.assertIsNotNone(sidecars.preview_path(3))    # its mp4 is fresh, so the pair stays
        self.assertFalse(stale_tmp.exists())
        self.assertTrue(fresh_tmp.exists())
        self.assertEqual(sidecars.prune(45), 0)


if __name__ == "__main__":
    unittest.main()
