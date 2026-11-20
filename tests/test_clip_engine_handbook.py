"""The clip handbook (app/clip_engine/handbook.py) and its report page
(app/web/clip_handbook.py).

No real database, model, storage or ffmpeg: ``examples`` runs on in-memory
SQLite, Gemini/Claude/shotlog/offload/media/notes/engine are mocks, sources
are temp files, and the engine directory is a temp dir. The page is mounted on
a bare FastAPI app (never app.web.main).

Run with the suite's §0 prefix:  python -m pytest tests/test_clip_engine_handbook.py -q
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import spend
from app.clip_engine import engine, handbook
from app.config import Settings
from app.models import Base, Candidate, Channel, ClipProposal, Cut, ThreadsPost
from app.web import clip_handbook

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "app" / "web" / "templates"
REAL_PERFORMANCE = handbook._performance


def video_ids(holdout: bool, n: int, prefix: str = "vid") -> list[str]:
    out, i = [], 0
    while len(out) < n:
        vid = f"{prefix}{i:04d}"
        if handbook.is_holdout(vid) == holdout:
            out.append(vid)
        i += 1
    return out


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.engine_dir = self.tmp / "engine"
        self.start(mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.engine_dir)}))
        self.start(mock.patch.object(spend, "_LEDGER", self.tmp / "llm_spend.json"))
        self.report_file = self.tmp / "config" / "clip_handbook.yaml"
        self.start(mock.patch.object(handbook, "path", return_value=self.report_file))

    def start(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value


# ---- pure helpers -----------------------------------------------------------------------

class HelpersTest(unittest.TestCase):
    def test_holdout_is_deterministic_and_about_a_fifth(self):
        ids = [f"v{i}" for i in range(2000)]
        first = [handbook.is_holdout(v) for v in ids]
        self.assertEqual(first, [handbook.is_holdout(v) for v in ids])
        share = sum(first) / len(ids)
        self.assertGreater(share, 0.16)
        self.assertLess(share, 0.24)

    def test_weights(self):
        self.assertEqual(handbook.weight_for(published=True, seeded=False), 1.0)
        self.assertEqual(handbook.weight_for(published=False, seeded=False), 0.5)
        self.assertEqual(handbook.weight_for(published=True, seeded=True), 0.3)
        self.assertEqual(handbook.weight_for(published=False, seeded=True), 0.3)

    def test_segments_keep_play_order_and_drop_slivers_and_junk(self):
        raw = json.dumps([{"start": 30, "end": 41.5}, {"start": 10, "end": 10.2},
                          {"start": 5, "end": 9}, {"start": "x", "end": 3}, {"start": 8, "end": 2}])
        self.assertEqual(handbook.segments_of(raw),
                         [{"start": 30.0, "end": 41.5}, {"start": 5.0, "end": 9.0}])
        self.assertEqual(handbook.segments_of("not json"), [])
        self.assertEqual(handbook.segments_of(None), [])

    def test_play_map(self):
        self.assertEqual(handbook.play_map([{"start": 30, "end": 40}, {"start": 5, "end": 8}]), [
            {"source_start": 30, "source_end": 40, "clip_start": 0.0, "clip_end": 10.0},
            {"source_start": 5, "source_end": 8, "clip_start": 10.0, "clip_end": 13.0}])

    def test_skipped_stretches_cover_the_opening_the_run_up_and_the_gaps(self):
        beats = [{"start": 0, "end": 8, "shot": "anchor_desk", "speaker": "anchor", "line": "Good evening"},
                 {"start": 100, "end": 120, "shot": "b_roll", "line": "x" * 300},
                 {"start": 300, "end": 310, "shot": "interview"}]
        out = handbook.skipped_stretches([{"start": 200, "end": 210}, {"start": 230, "end": 240}],
                                         beats, duration=600)
        self.assertEqual([(s["start"], s["end"]) for s in out],
                         [(0.0, 60.0), (110.0, 200), (210, 230), (240, 330.0)])
        self.assertEqual(out[0]["beats"][0]["shot"], "anchor_desk")
        self.assertEqual(out[1]["beats"][0]["t"], 100.0)
        self.assertLessEqual(len(out[1]["beats"][0]["line"]), 100)
        self.assertEqual(out[3]["beats"][0]["shot"], "interview")
        # A clip near the start: one stretch from 0.
        near = handbook.skipped_stretches([{"start": 40, "end": 50}], [], duration=100)
        self.assertEqual([(s["start"], s["end"]) for s in near], [(0.0, 40), (50, 100.0)])

    def test_skipped_beats_are_capped(self):
        beats = [{"start": i, "end": i + 1, "shot": "b_roll"} for i in range(40)]
        out = handbook.skipped_stretches([{"start": 50, "end": 60}], beats, duration=60)
        self.assertEqual(len(out[0]["beats"]), handbook.MAX_SKIPPED_BEATS)
        self.assertEqual(out[0]["more_beats"], 40 - handbook.MAX_SKIPPED_BEATS)

    def test_clean_edit_coerces_to_the_vocabulary(self):
        edit = handbook.clean_edit({
            "opening": {"kind": "Anchor Intro", "what": "w", "beat_shot": "anchor_desk",
                        "speaker": "", "why_visible": "v"},
            "kept": [{"what": "a", "kind": "made_up"}, {"what": "", "kind": "b_roll"}, "junk"],
            "dropped": [{"what": "d", "kind": "non-english-interview"}],
            "structure": {"cold_open": 1, "segments": "3", "seconds": "41.26",
                          "ending_kind": "nonsense"},
            "language_choices": "Hindi dropped", "notes": None})
        self.assertEqual(edit["opening"]["kind"], "anchor_intro")
        self.assertEqual(edit["kept"], [{"what": "a", "kind": "other"}])
        self.assertEqual(edit["dropped"][0]["kind"], "non_english_interview")
        self.assertEqual(edit["structure"], {"cold_open": True, "segments": 3, "seconds": 41.3,
                                             "ending_kind": "other"})
        self.assertEqual(edit["notes"], "")
        self.assertEqual(handbook.clean_edit(None)["opening"]["kind"], "other")

    def test_strength_thresholds(self):
        table = [((5, 0), "rule"), ((5, 2), "rule"), ((7, 3), "rule"), ((4, 0), "tendency"),
                 ((5, 3), "tendency"), ((20, 9), "tendency"), ((0, 0), "tendency")]
        for (support, exceptions), want in table:
            with self.subTest(support=support, exceptions=exceptions):
                self.assertEqual(handbook.strength_for(support, exceptions), want)

    def test_schemas_have_the_specified_shape(self):
        props = handbook.EDIT_SCHEMA["properties"]
        self.assertEqual(set(props), {"opening", "kept", "dropped", "structure",
                                      "language_choices", "notes"})
        self.assertEqual(set(props["opening"]["properties"]),
                         {"kind", "what", "beat_shot", "speaker", "why_visible"})
        self.assertEqual(set(props["structure"]["properties"]),
                         {"cold_open", "segments", "seconds", "ending_kind"})
        self.assertIn("non_english_interview", props["kept"]["items"]["properties"]["kind"]["enum"])
        rule = handbook.RULES_SCHEMA["properties"]["rules"]["items"]["properties"]
        self.assertEqual(set(rule), {"id", "text", "when", "support", "exceptions", "examples",
                                     "strength"})

    def test_prompts_describe_choices_without_motives(self):
        system = handbook.edit_system({"name": "Climate Clips", "topic": "climate"})
        self.assertIn("Never guess at why", system)
        self.assertIn("non_english_interview", system)
        self.assertIn("Climate Clips", system)
        synth = handbook.synth_system({})
        self.assertIn("WITHOUT its condition", synth)
        self.assertIn("never round up", synth)

    def test_prompt_hash_is_stable(self):
        self.assertEqual(handbook.prompt_hash(), handbook.prompt_hash())


# ---- examples (in-memory SQLite) ----------------------------------------------------------

class ExamplesTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.db = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.db)
        self.session = sessionmaker(bind=self.db, future=True)()
        self.addCleanup(self.db.dispose)
        self.addCleanup(self.session.close)
        self.channel = Channel(call_sign="F24", url="https://example.com/f24")
        self.own = Channel(call_sign="OWN", url="https://example.com/own", first_party=True)
        self.session.add_all([self.channel, self.own])
        self.session.commit()
        self.train = video_ids(False, 8)
        self.held = video_ids(True, 2)
        self.perf = self.start(mock.patch.object(handbook, "_performance", return_value={}))

    def candidate(self, vid, *, channel=None, transcript=True, local=True, key="", category=""):
        tpath = self.tmp / f"{vid}.json"
        if transcript:
            tpath.write_text("[]")
        lpath = self.tmp / f"{vid}.mp4"
        if local:
            lpath.write_bytes(b"\0" * 8)
        c = Candidate(video_id=vid, channel_pk=(channel or self.channel).id, title=f"Title {vid}",
                      url="u", status="archived", transcript_path=str(tpath),
                      local_video_path=str(lpath), source_object_key=key, duration_seconds=300,
                      category=category, archived_at=dt.datetime(2026, 10, 1))
        self.session.add(c)
        self.session.commit()   # examples() rolls its session back
        return c

    def cut(self, cand, segments, *, exported=False, published=False, seeded=False, fmt=""):
        cut = Cut(candidate_pk=cand.id, trim_segments=json.dumps(segments), clip_title="Clip",
                  trimmed_clip_path="/x.mp4" if exported else "", format_tags=fmt)
        self.session.add(cut)
        self.session.flush()
        if published:
            self.session.add(ThreadsPost(candidate_pk=cand.id, cut_pk=cut.id, status="published",
                                         threads_media_id="m", format_tags="package",
                                         published_at=dt.datetime(2026, 10, 2)))
        if seeded:
            self.session.add(ClipProposal(candidate_pk=cand.id, cut_pk=cut.id,
                                          verdict=ClipProposal.VERDICT_ACCEPTED))
        self.session.commit()
        return cut

    def test_selection_weights_and_order(self):
        seg = [{"start": 30, "end": 45}, {"start": 5, "end": 9}]
        a, b, c, d = (self.candidate(v) for v in self.train[:4])
        published = self.cut(a, seg, published=True, fmt="package")
        exported = self.cut(b, seg, exported=True)
        seeded = self.cut(c, seg, published=True, seeded=True)
        self.cut(d, seg)                                                # never exported
        self.cut(self.candidate(self.train[4], channel=self.own), seg, exported=True)  # first party
        self.cut(self.candidate(self.train[5], category="promos"), seg, exported=True)
        self.cut(self.candidate(self.train[6], transcript=False), seg, exported=True)
        offloaded = self.cut(self.candidate(self.train[7], local=False, key="src/k.mp4"),
                             seg, exported=True)
        self.cut(self.candidate("gone", local=False), seg, exported=True)  # nowhere
        self.cut(a, [{"start": 1, "end": 1.1}], exported=True)          # slivers only

        out = handbook.examples(self.session)
        by_pk = {e["cut_pk"]: e for e in out}
        self.assertEqual(set(by_pk), {published.id, exported.id, seeded.id, offloaded.id}
                         - {e["cut_pk"] for e in out if e["holdout"]})
        self.assertEqual(by_pk[published.id]["weight"], 1.0)
        self.assertEqual(by_pk[exported.id]["weight"], 0.5)
        self.assertEqual(by_pk[seeded.id]["weight"], 0.3)
        self.assertTrue(by_pk[seeded.id]["seeded"])
        self.assertEqual(by_pk[published.id]["segments"], seg)   # play order kept
        self.assertEqual(by_pk[published.id]["seconds"], 19.0)
        self.assertEqual(by_pk[published.id]["channel"], "F24")
        self.assertEqual(by_pk[published.id]["format"], "package")
        self.assertEqual(by_pk[exported.id]["format"], "")
        self.assertEqual([e["weight"] for e in out], sorted((e["weight"] for e in out), reverse=True))
        self.assertEqual(len(handbook.examples(self.session, limit=2)), 2)

    def test_holdout_videos_are_never_in_the_training_set(self):
        seg = [{"start": 0, "end": 10}]
        train = self.cut(self.candidate(self.train[0]), seg, exported=True)
        held = self.cut(self.candidate(self.held[0]), seg, exported=True)
        held2 = self.cut(self.candidate(self.held[1]), seg, published=True)
        self.assertEqual([e["cut_pk"] for e in handbook.examples(self.session)], [train.id])
        self.assertEqual({e["cut_pk"] for e in handbook.examples(self.session, holdout=True)},
                         {held.id, held2.id})
        self.assertTrue(all(e["holdout"] for e in handbook.examples(self.session, holdout=True)))

    def test_performance_orders_ties_and_is_attached(self):
        seg = [{"start": 0, "end": 10}]
        low = self.cut(self.candidate(self.train[0]), seg, published=True)
        high = self.cut(self.candidate(self.train[1]), seg, published=True)
        self.perf.return_value = {low.id: {"views": 10, "percentile": 20.0},
                                  high.id: {"views": 900, "percentile": 95.0}}
        out = handbook.examples(self.session)
        self.assertEqual([e["cut_pk"] for e in out], [high.id, low.id])
        self.assertEqual(out[0]["views"], 900)

    def test_session_is_rolled_back(self):
        with mock.patch.object(self.session, "rollback", wraps=self.session.rollback) as rb:
            handbook.examples(self.session)
        rb.assert_called()

    def test_performance_reads_taste_and_survives_failure(self):
        with mock.patch("app.clip_engine.taste.scored_first_airings",
                        return_value=[{"cut_pk": 7, "views": 5, "percentile": 50.0}]):
            self.assertEqual(REAL_PERFORMANCE(self.session), {7: {"views": 5, "percentile": 50.0}})
        with mock.patch("app.clip_engine.taste.scored_first_airings", side_effect=RuntimeError("x")):
            self.assertEqual(REAL_PERFORMANCE(self.session), {})


# ---- analyse --------------------------------------------------------------------------------

SHOT_LOG = {"source_seconds": 300.0, "cost_usd": 0.05, "summary": "s", "format": "package",
            "beats": [{"start": 0.0, "end": 8.0, "shot": "anchor_desk", "speaker": "anchor",
                       "line": "Good evening", "on_screen": "", "energy": 2, "visual": 1,
                       "topic": 1, "face": "close", "sound": "", "note": ""},
                      {"start": 8.0, "end": 60.0, "shot": "aerial", "speaker": "narrator",
                       "line": "Floods in Assam", "on_screen": "", "energy": 4, "visual": 5,
                       "topic": 2, "face": "none", "sound": "", "note": ""}],
            "people": [], "highlights": []}

EDIT_ANSWER = {"opening": {"kind": "aerial", "what": "flood aerial at 8.0 s", "beat_shot": "aerial",
                           "speaker": "narrator", "why_visible": "water to the horizon"},
               "kept": [{"what": "aerial 8-20", "kind": "aerial"}],
               "dropped": [{"what": "anchor intro 0-8", "kind": "anchor_intro"}],
               "structure": {"cold_open": False, "segments": 1, "seconds": 12, "ending_kind": "line_lands"},
               "language_choices": "", "notes": ""}

USAGE = {"model": "gemini-3.8-flash", "input_tokens": 1000, "output_tokens": 200, "cost_usd": 0.01}


def example(cut_pk=11, vid="vidA", **kw) -> dict:
    out = {"cut_pk": cut_pk, "candidate_pk": 3, "video_id": vid, "title": "Clip",
           "source_title": "Assam floods", "channel": "F24", "format": "package",
           "published": True, "seeded": False, "weight": 1.0, "views": 100, "percentile": 80.0,
           "segments": [{"start": 20.0, "end": 30.0}, {"start": 8.0, "end": 12.0}],
           "seconds": 14.0, "duration": 300.0, "local_video_path": "", "source_object_key": "",
           "transcript_path": "/t.json", "word_transcript_path": "", "archived_at": None}
    out.update(kw)
    return out


class AnalyseTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.settings = Settings(raw={})
        self.start(mock.patch.object(handbook, "_texts", return_value=(
            [{"start": 0.0, "end": 8.0, "text": "Good evening"},
             {"start": 8.0, "end": 20.0, "text": "Floods in Assam"}], [])))
        self.start(mock.patch("app.clip_engine.brand", return_value={"name": "Climate Clips"}))
        self.load_cached = self.start(mock.patch("app.clip_engine.shotlog.load_cached",
                                                 return_value=SHOT_LOG))
        self.build = self.start(mock.patch("app.clip_engine.shotlog.build", return_value=SHOT_LOG))
        self.gen = self.start(mock.patch("app.clip_engine.gemini.generate_json",
                                         return_value=(EDIT_ANSWER, USAGE)))
        self.ensure = self.start(mock.patch("app.offload.ensure_local"))
        self.link = self.start(mock.patch("app.clip_engine.media.link_inflight"))
        self.release = self.start(mock.patch("app.clip_engine.media.release_inflight"))
        self.free = self.start(mock.patch("app.clip_engine.media.disk_free_gb", return_value=500.0))

    def test_one_text_only_gemini_call_with_the_edit_laid_out(self):
        meter = engine.CostMeter("bench")
        record = handbook.analyse(example(), self.settings, meter)
        self.gen.assert_called_once()
        args, kwargs = self.gen.call_args
        self.assertEqual(args[0], "gemini-3.8-flash")          # model_for gemini editor
        self.assertEqual(kwargs["thinking"], "MEDIUM")
        self.assertIsNone(kwargs["media_resolution"])
        self.assertEqual(kwargs["bucket"], "bench")
        self.assertIs(kwargs["schema"], handbook.EDIT_SCHEMA)
        self.assertNotIn("temperature", kwargs)
        self.assertIn("Never guess", kwargs["system"])
        parts = kwargs["parts"]
        self.assertEqual(len(parts), 1)
        self.assertNotIn("inlineData", parts[0])
        self.assertNotIn("fileData", parts[0])
        payload = json.loads(parts[0]["text"])
        self.assertEqual(payload["editor_segments_play_order"][0]["source_start"], 20.0)
        self.assertEqual(payload["editor_segments_play_order"][1]["clip_start"], 10.0)
        self.assertEqual(payload["clip_seconds"], 14.0)
        self.assertEqual(len(payload["shot_log"]["beats"]), 2)
        self.assertEqual(payload["transcript"][1]["text"], "Floods in Assam")
        self.assertEqual(payload["skipped_stretches"][0]["start"], 0.0)
        self.assertEqual(payload["skipped_stretches"][0]["beats"][0]["shot"], "anchor_desk")
        # A cached log: the source is never restored, its stored cost is noted.
        self.ensure.assert_not_called()
        self.build.assert_not_called()
        self.assertEqual(meter.by_stage.get("perceive_cached"), 0.05)
        self.assertAlmostEqual(meter.total, 0.01)
        self.assertEqual(record["edit"]["opening"]["kind"], "aerial")
        self.assertEqual(record["weight"], 1.0)
        self.assertEqual(record["prompt_hash"], handbook.prompt_hash())

    def test_cached_per_cut_until_the_segments_change(self):
        handbook.analyse(example(), self.settings, engine.CostMeter("bench"))
        self.assertTrue(handbook.edit_path(11).is_file())
        meter = engine.CostMeter("bench")
        again = handbook.analyse(example(), self.settings, meter)
        self.assertEqual(self.gen.call_count, 1)
        self.assertEqual(again["edit"]["opening"]["kind"], "aerial")
        self.assertEqual(meter.total, 0.0)
        self.assertAlmostEqual(meter.by_stage["handbook_analyse_cached"], 0.01)
        handbook.analyse(example(segments=[{"start": 20.0, "end": 31.0}]), self.settings, None)
        self.assertEqual(self.gen.call_count, 2)

    def test_without_a_cached_log_a_restored_source_is_logged_then_deleted(self):
        self.load_cached.return_value = None
        local = self.tmp / "src.mp4"

        def restore(path, key):
            Path(path).write_bytes(b"\0" * 10)
            return Path(path)

        self.ensure.side_effect = restore
        self.link.return_value = self.engine_dir / "inflight" / "x.mp4"
        handbook.analyse(example(local_video_path=str(local), source_object_key="k"),
                         self.settings, engine.CostMeter("bench"))
        self.assertEqual(self.build.call_args.kwargs["source"], self.link.return_value)
        self.assertEqual(self.build.call_args.kwargs["perceiver"], "gemini")
        self.release.assert_called_once_with(self.link.return_value)
        self.assertFalse(local.exists())          # restored by us -> removed

    def test_a_source_already_on_disk_is_kept(self):
        self.load_cached.return_value = None
        local = self.tmp / "src.mp4"
        local.write_bytes(b"\0" * 10)
        self.ensure.return_value = local
        self.link.return_value = local     # link failed over to the file itself
        handbook.analyse(example(local_video_path=str(local)), self.settings, None)
        self.assertTrue(local.exists())
        self.release.assert_not_called()

    def test_low_disk_stops_before_any_restore(self):
        self.load_cached.return_value = None
        self.free.return_value = 2.0
        with self.assertRaises(handbook.LowDisk):
            handbook.analyse(example(local_video_path=str(self.tmp / "missing.mp4"),
                                     source_object_key="k"), self.settings, None)
        self.ensure.assert_not_called()
        self.gen.assert_not_called()

    def test_a_failed_call_still_counts_its_spend(self):
        err = RuntimeError("truncated")
        err.usage = USAGE
        self.gen.side_effect = err
        meter = engine.CostMeter("bench")
        with self.assertRaises(RuntimeError):
            handbook.analyse(example(), self.settings, meter)
        self.assertAlmostEqual(meter.total, 0.01)
        self.assertFalse(handbook.edit_path(11).exists())

    def test_no_transcript_is_refused(self):
        with mock.patch.object(handbook, "_texts", return_value=([], [])):
            with self.assertRaises(handbook.HandbookError):
                handbook.analyse(example(), self.settings, None)
        self.gen.assert_not_called()


# ---- synthesize -------------------------------------------------------------------------------

def edits(n: int) -> list[dict]:
    return [{"cut_pk": 100 + i, "channel": "F24", "format": "package", "published": i % 2 == 0,
             "percentile": 50.0, "weight": 1.0 if i % 2 == 0 else 0.5, "seconds": 30.0,
             "edit": handbook.clean_edit(EDIT_ANSWER)} for i in range(n)]


class SynthesizeTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.start(mock.patch("app.clip_engine.brand", return_value={}))
        self.call = self.start(mock.patch("app.clip_engine.claude.json_call"))
        self.call.return_value = ({
            "summary": "Opens on pictures.",
            "rules": [
                {"id": "open-aerial", "text": "Open on the aerial b-roll, not the anchor",
                 "when": "France 24 packages", "support": 8, "exceptions": 1,
                 "examples": [100, 101, 999], "strength": "rule"},
                {"id": "open-aerial", "text": "Drop the studio toss", "when": "always",
                 "support": 6, "exceptions": 4, "examples": [102], "strength": "rule"},
                {"id": "", "text": "Keep it under 40 seconds", "when": "", "support": 4,
                 "exceptions": 0, "examples": [], "strength": "rule"},
                {"id": "x", "text": "Nothing backs this", "when": "always", "support": 0,
                 "exceptions": 0, "examples": [], "strength": "rule"},
                {"id": "y", "text": "", "when": "always", "support": 9, "exceptions": 0,
                 "examples": [], "strength": "rule"},
                {"id": "big", "text": "End on the line that lands", "when": "Always",
                 "support": 500, "exceptions": 0, "examples": [], "strength": "tendency"},
            ]}, {"model": "claude-opus-5-5", "input_tokens": 9000, "output_tokens": 900,
                 "cost_usd": 0.054})

    def test_too_few_edits_never_calls_claude(self):
        with self.assertRaises(handbook.HandbookError):
            handbook.synthesize(edits(handbook.MIN_EDITS - 1), Settings(raw={}))
        self.call.assert_not_called()

    def test_one_high_effort_call_and_thresholds_from_the_counts(self):
        meter = engine.CostMeter("bench")
        report = handbook.synthesize(edits(12), Settings(raw={}), meter)
        args, kwargs = self.call.call_args
        self.assertEqual(args[0], "claude-opus-5-5")
        self.assertEqual(kwargs["effort"], "high")
        self.assertIs(kwargs["schema"], handbook.RULES_SCHEMA)
        self.assertEqual(kwargs["bucket"], "bench")
        records = json.loads(kwargs["content"][0]["text"])["records"]
        self.assertEqual(len(records), 12)
        self.assertEqual(set(records[0]), {"id", "channel", "format", "published", "percentile",
                                           "weight", "seconds", "opening", "kept", "dropped",
                                           "structure", "language_choices", "notes"})
        self.assertAlmostEqual(meter.total, 0.054)

        by_text = {c["text"]: c for c in report["candidates"]}
        self.assertNotIn("Nothing backs this", by_text)
        self.assertNotIn("", by_text)
        self.assertEqual(by_text["Open on the aerial b-roll, not the anchor"]["strength"], "rule")
        self.assertEqual(by_text["Open on the aerial b-roll, not the anchor"]["examples"], [100, 101])
        self.assertEqual(by_text["Drop the studio toss"]["strength"], "tendency")   # 6/10 < 0.7
        self.assertEqual(by_text["Drop the studio toss"]["when"], "Always")
        self.assertEqual(by_text["Keep it under 40 seconds"]["strength"], "tendency")  # support 4
        self.assertEqual(by_text["End on the line that lands"]["support"], 12)  # capped at records
        self.assertEqual(by_text["End on the line that lands"]["strength"], "rule")
        ids = [c["id"] for c in report["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        strengths = [c["strength"] for c in report["candidates"]]
        self.assertEqual(strengths, sorted(strengths, key=lambda s: s != "rule"))
        self.assertEqual(report["examples_used"], 12)
        self.assertEqual(report["summary"], "Opens on pictures.")
        self.assertIn("8 of your 9 past edits where this applied",
                      by_text["Open on the aerial b-roll, not the anchor"]["why"])
        self.assertNotIn("enabled", report["candidates"][0])

    def test_an_answer_without_usable_rules_is_an_error(self):
        self.call.return_value = ({"rules": [{"text": "", "support": 3}], "summary": ""}, {})
        with self.assertRaises(handbook.HandbookError):
            handbook.synthesize(edits(12), Settings(raw={}))


# ---- proposing into the shared queue -----------------------------------------------------------

def report(**kw) -> dict:
    out = {"generated_at": "2026-11-18T10:00:00Z", "examples_used": 40, "holdout": 9,
           "holdout_videos": 6, "summary": "He opens on pictures.", "validation": None,
           "candidates": [
               {"id": "open-aerial", "text": "Open on the aerial b-roll, not the anchor",
                "when": "France 24 packages", "support": 8, "exceptions": 1,
                "examples": [100, 101], "strength": "rule"},
               {"id": "toss", "text": "Drop the studio toss", "when": "Always", "support": 6,
                "exceptions": 4, "examples": [102], "strength": "tendency"},
               {"id": "thin", "text": "Use a map", "when": "Always", "support": 2,
                "exceptions": 0, "examples": [], "strength": "tendency"},
           ]}
    out.update(kw)
    return out


class ProposeTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.add = self.start(mock.patch("app.clip_engine.notes.add_proposals"))
        self.queue = self.start(mock.patch("app.clip_engine.notes.proposals", return_value=[]))
        self.rules = self.start(mock.patch("app.config.load_brand_rules", return_value=[]))

    def test_items_go_to_the_shared_queue_as_handbook_suggestions(self):
        self.add.side_effect = lambda items, **kw: [
            {"text": f"When x: {i['text']}", "original": i["text"], "rule": i["text"],
             "markdown": f"**Rule:** {i['text']}", "status": "pending", "idx": n}
            for n, i in enumerate(items)]
        rep = report()
        queued = handbook.propose(rep, settings="S")
        self.assertEqual(len(queued), 2)
        items = self.add.call_args.args[0]
        self.assertEqual(self.add.call_args.kwargs, {"source": "handbook", "settings": "S"})
        self.assertEqual([i["text"] for i in items],
                         ["Open on the aerial b-roll, not the anchor",
                          "Usually drop the studio toss"])     # thin one (support 2) left out
        first = items[0]
        self.assertEqual(first["when"], "France 24 packages")
        self.assertEqual(first["count"], 8)
        self.assertEqual(first["evidence"], "8 of 9 edits; examples cut 100, 101")
        self.assertIn("8 of your 9 past edits", first["why"])
        self.assertEqual(items[1]["when"], "Always")
        cands = {c["id"]: c for c in rep["candidates"]}
        self.assertTrue(cands["open-aerial"]["proposed"])
        self.assertFalse(cands["thin"]["proposed"])
        self.assertEqual(cands["open-aerial"]["formatted"],
                         "**Rule:** Open on the aerial b-roll, not the anchor")
        self.assertEqual(cands["open-aerial"]["rule_text"],
                         "When x: Open on the aerial b-roll, not the anchor")
        self.assertEqual(rep["proposed"], 2)

    def test_already_queued_ones_keep_their_existing_format(self):
        self.add.return_value = []
        self.queue.return_value = [{"text": "Rule text", "original": "Usually drop the studio toss",
                                    "markdown": "**Rule:** md", "status": "dismissed", "idx": 0}]
        rep = report()
        handbook.propose(rep)
        cands = {c["id"]: c for c in rep["candidates"]}
        self.assertFalse(cands["toss"]["proposed"])
        self.assertEqual(cands["toss"]["formatted"], "**Rule:** md")

    def test_a_queue_failure_leaves_the_report_usable(self):
        self.add.side_effect = OSError("disk")
        rep = report()
        self.assertEqual(handbook.propose(rep), [])
        self.assertEqual(rep["proposed"], 0)

    def test_statuses_from_the_queue_and_the_brand_rules(self):
        self.queue.return_value = [
            {"text": "When France 24 packages: Open on the aerial b-roll, not the anchor",
             "original": "Open on the aerial b-roll, not the anchor", "status": "pending"},
            {"text": "Usually drop the studio toss", "status": "dismissed"}]
        rows = {c["id"]: c["status"] for c in handbook.statuses(report())}
        self.assertEqual(rows, {"open-aerial": "pending", "toss": "dismissed",
                                "thin": "not proposed"})
        self.rules.return_value = [{"text": "When France 24 packages: Open on the aerial b-roll, "
                                            "not the anchor",
                                    "original": "Open on the aerial b-roll, not the anchor"}]
        rows = {c["id"]: c["status"] for c in handbook.statuses(report())}
        self.assertEqual(rows["open-aerial"], "added")

    def test_statuses_never_raise(self):
        self.queue.side_effect = RuntimeError("x")
        self.rules.side_effect = RuntimeError("y")
        self.assertEqual({c["status"] for c in handbook.statuses(report())}, {"not proposed"})
        self.assertEqual(handbook.statuses(None), [])

    def test_rule_line_is_what_the_engine_would_read(self):
        c = report()["candidates"]
        self.assertEqual(handbook.rule_line(c[0]),
                         "When France 24 packages: Open on the aerial b-roll, not the anchor")
        self.assertEqual(handbook.rule_line(c[1]), "Usually drop the studio toss")
        self.assertEqual(handbook.rule_line({**c[0], "when": "The source is a hearing"}),
                         "When the source is a hearing: Open on the aerial b-roll, not the anchor")
        self.assertEqual(handbook.rule_line({**c[0], "rule_text": "Formatted."}), "Formatted.")


# ---- storage, rendering --------------------------------------------------------------------------

class StorageTest(TempDirCase):
    def test_round_trip_and_header(self):
        rep = report()
        rep["candidates"][0]["formatted"] = "**Rule:** x"
        handbook.save(rep)
        text = self.report_file.read_text()
        self.assertTrue(text.startswith("# Clip handbook report"))
        self.assertIn("not a rule store", text)
        loaded = handbook.load()
        self.assertEqual(loaded["examples_used"], 40)
        self.assertEqual([c["id"] for c in loaded["candidates"]], ["open-aerial", "toss", "thin"])
        self.assertEqual(loaded["candidates"][0]["formatted"], "**Rule:** x")
        self.assertNotIn("enabled", loaded["candidates"][0])

    def test_hand_edits_are_normalized(self):
        self.report_file.parent.mkdir(parents=True)
        self.report_file.write_text(yaml.safe_dump({
            "junk": 1, "examples_used": 3,
            "candidates": [{"text": "A", "support": "7", "exceptions": None, "strength": "rule"},
                           {"text": "B", "support": 2, "strength": "rule", "enabled": True},
                           "not a dict", {"text": "", "support": 3}]}))
        loaded = handbook.load()
        self.assertNotIn("junk", loaded)
        self.assertEqual([(c["text"], c["support"], c["strength"]) for c in loaded["candidates"]],
                         [("A", 7, "rule"), ("B", 2, "tendency")])
        self.assertNotIn("enabled", loaded["candidates"][1])

    def test_missing_or_broken_file_is_none(self):
        self.assertIsNone(handbook.load())
        self.report_file.parent.mkdir(parents=True)
        self.report_file.write_text("candidates: [unclosed")
        self.assertIsNone(handbook.load())

    def test_set_validation_keeps_the_rest(self):
        with self.assertRaises(handbook.HandbookError):
            handbook.set_validation({"videos": 1})
        handbook.save(report())
        handbook.set_validation({"videos": 15, "current": {"moment_hit1": 0.4}})
        loaded = handbook.load()
        self.assertEqual(loaded["validation"]["videos"], 15)
        self.assertEqual(len(loaded["candidates"]), 3)

    def test_backup(self):
        self.assertIsNone(handbook.backup())
        handbook.save(report())
        copy = handbook.backup()
        self.assertEqual(copy.read_text(), self.report_file.read_text())
        self.assertEqual(copy.parent, self.engine_dir / "handbook" / "backups")

    def test_render_lists_rules_first_with_conditions_and_evidence(self):
        text = handbook.render(report())
        lines = text.splitlines()
        self.assertEqual(lines[0], handbook.HEADER_TITLE)
        self.assertIn("Open on the aerial b-roll, not the anchor (when: France 24 packages)",
                      lines[1])
        self.assertIn("Usually drop the studio toss", lines[2])
        self.assertIn("8 of 9 edits", lines[1])
        self.assertEqual(handbook.render(None), "")

    def test_render_active_is_always_empty(self):
        """Approved rules are brand rules now; the engine reads those itself."""
        self.assertEqual(handbook.render_active(), "")
        handbook.save(report())
        self.assertEqual(handbook.render_active(), "")


# ---- build ----------------------------------------------------------------------------------------

class BuildTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.start(mock.patch.object(handbook, "_db", contextlib.nullcontext))
        self.train = [example(cut_pk=i, vid=f"v{i}") for i in range(14)]
        self.held = [example(cut_pk=90, vid="h1"), example(cut_pk=91, vid="h1")]
        self.start(mock.patch.object(handbook, "examples", side_effect=lambda s, limit=None, holdout=False:
                                     self.held if holdout else self.train[:limit] if limit else self.train))
        self.synth = self.start(mock.patch.object(handbook, "synthesize",
                                                  side_effect=lambda e, s, m: {"candidates": [],
                                                                               "examples_used": len(e)}))

    def test_analyses_then_synthesizes_and_counts_failures(self):
        def fake(ex, settings, meter):
            if ex["cut_pk"] == 3:
                raise RuntimeError("blocked")
            meter.add("handbook_analyse", {"model": "g", "cost_usd": 0.1})
            return {"cut_pk": ex["cut_pk"], "edit": {}}

        said = []
        with mock.patch.object(handbook, "analyse", side_effect=fake):
            rep = handbook.build(Settings(raw={}), say=said.append)
        self.assertEqual(rep["examples_used"], 13)
        self.assertEqual(rep["failed"], 1)
        self.assertEqual(rep["holdout"], 2)
        self.assertEqual(rep["holdout_videos"], 1)
        self.assertEqual(rep["stopped"], "")
        self.assertAlmostEqual(rep["cost_usd"], 1.3)
        self.assertTrue(any("blocked" in s for s in said))

    def test_stops_at_the_cap_and_on_low_disk(self):
        def costly(ex, settings, meter):
            meter.add("handbook_analyse", {"model": "g", "cost_usd": 1.0})
            return {"cut_pk": ex["cut_pk"], "edit": {}}

        with mock.patch.object(handbook, "analyse", side_effect=costly) as analyse:
            rep = handbook.build(Settings(raw={}), max_usd=3.0, say=lambda s: None)
        self.assertEqual(analyse.call_count, 3)
        self.assertIn("$3", rep["stopped"])

        with mock.patch.object(handbook, "analyse", side_effect=handbook.LowDisk("only 2 GB")) as analyse:
            rep = handbook.build(Settings(raw={}), say=lambda s: None)
        self.assertEqual(analyse.call_count, 1)
        self.assertIn("only 2 GB", rep["stopped"])

    def test_limit(self):
        with mock.patch.object(handbook, "analyse",
                               side_effect=lambda ex, s, m: {"cut_pk": ex["cut_pk"], "edit": {}}) as a:
            handbook.build(Settings(raw={}), limit=5, say=lambda s: None)
        self.assertEqual(a.call_count, 5)


# ---- validate -------------------------------------------------------------------------------------

class ValidateTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.start(mock.patch.object(handbook, "_db", contextlib.nullcontext))
        self.held = [example(cut_pk=1, vid="h1", segments=[{"start": 10.0, "end": 30.0}]),
                     example(cut_pk=2, vid="h1", segments=[{"start": 100.0, "end": 120.0}]),
                     example(cut_pk=3, vid="h2", segments=[{"start": 50.0, "end": 70.0}])]
        self.start(mock.patch.object(handbook, "examples", return_value=self.held))
        self.start(mock.patch.object(handbook, "_texts",
                                     return_value=([{"start": 0, "end": 1, "text": "x"}], [])))
        self.prepare = self.start(mock.patch("app.clip_engine.jobs.prepare_inputs", return_value={
            "candidate": {"title": "T", "description": "D"}, "context": {"c": 1},
            "examples": [{"post_pk": 1}], "notes": []}))

        @contextlib.contextmanager
        def source(fields, *, min_free_gb, key):
            yield Path("/tmp/link.mp4")

        self.start(mock.patch.object(handbook, "source_for", source))
        self.start(mock.patch("app.clip_engine.brand", return_value={}))
        self.start(mock.patch("app.clip_engine.playbook.load", return_value={"archetypes": []}))
        self.queue = self.start(mock.patch("app.clip_engine.notes.proposals", return_value=[]))
        self.start(mock.patch("app.config.load_brand_rules", return_value=[]))
        self.runs = []

        def run(**kw):
            extra = kw["settings"].get("clip_engine.extra_rules")
            self.runs.append((kw["video_id"], list(extra)))
            kw["meter"].add("draft", {"model": "g", "cost_usd": 0.5})
            # With the candidates the engine finds the operator's moment; without, it misses.
            seg = {"start": 12.0, "end": 28.0} if extra else {"start": 200.0, "end": 220.0}
            return SimpleNamespace(top=[{"segments": [seg]}], outcome="proposals")

        self.engine_run = self.start(mock.patch("app.clip_engine.engine.run", side_effect=run))

    def test_current_rules_vs_current_plus_candidates(self):
        summary = handbook.validate(report(), Settings(raw={}), limit=15, say=lambda s: None)
        self.assertEqual(len(self.runs), 4)
        expected = ["When France 24 packages: Open on the aerial b-roll, not the anchor",
                    "Usually drop the studio toss", "Usually use a map"]
        self.assertEqual({tuple(extra) for _, extra in self.runs}, {(), tuple(expected)})
        self.assertEqual(summary["rules"], expected)
        self.assertEqual(summary["rules_tested"], 3)
        self.assertEqual(summary["videos"], 2)
        self.assertEqual(summary["candidates"]["moment_hit1"], 0.5)   # h1 hit, h2 missed
        self.assertEqual(summary["current"]["moment_hit1"], 0.0)
        self.assertEqual(summary["candidates"]["n"], 2)
        self.assertAlmostEqual(summary["cost_usd"], 2.0)
        kw = self.engine_run.call_args.kwargs
        self.assertEqual(kw["used"], [])
        self.assertEqual(kw["rejected"], [])
        self.assertEqual(kw["meter"].bucket, "bench")
        self.assertEqual(kw["context"], {"c": 1})
        self.assertTrue(self.prepare.call_args.kwargs["replay"])

    def test_added_and_dismissed_candidates_are_not_retested(self):
        self.queue.return_value = [{"text": "Usually drop the studio toss", "status": "dismissed"},
                                   {"text": "Usually use a map", "status": "added"}]
        summary = handbook.validate(report(), Settings(raw={}), say=lambda s: None)
        self.assertEqual(summary["rules"],
                         ["When France 24 packages: Open on the aerial b-roll, not the anchor"])
        self.queue.return_value = [{"text": c, "status": "dismissed"}
                                   for c in ("Usually drop the studio toss", "Usually use a map",
                                             "Open on the aerial b-roll, not the anchor")]
        with self.assertRaises(handbook.HandbookError):
            handbook.validate(report(), Settings(raw={}), say=lambda s: None)

    def test_replays_are_cached_and_errors_count_as_misses(self):
        handbook.validate(report(), Settings(raw={}), say=lambda s: None)
        self.assertEqual(len(self.runs), 4)
        summary = handbook.validate(report(), Settings(raw={}), say=lambda s: None)
        self.assertEqual(len(self.runs), 4)                 # nothing paid twice
        self.assertEqual(summary["cost_usd"], 0.0)
        self.assertEqual(summary["candidates"]["moment_hit1"], 0.5)

        self.engine_run.side_effect = RuntimeError("boom")
        fresh = report(candidates=report()["candidates"][:1])
        summary = handbook.validate(fresh, Settings(raw={}), say=lambda s: None)
        self.assertEqual(summary["current"]["errors"], 2)
        self.assertEqual(summary["current"]["moment_hit1"], 0.0)

    def test_the_cap_stops_between_videos(self):
        summary = handbook.validate(report(), Settings(raw={}), max_usd=0.5, say=lambda s: None)
        self.assertEqual(len(self.runs), 2)         # one video (two arms), then the cap
        self.assertIn("cap", summary["stopped"])
        self.assertEqual(summary["videos"], 1)

    def test_no_holdout_videos(self):
        with mock.patch.object(handbook, "examples", return_value=[]):
            with self.assertRaises(handbook.HandbookError):
                handbook.validate(report(), Settings(raw={}), say=lambda s: None)

    def test_holdout_grouping_and_order(self):
        videos = handbook.holdout_videos(self.held, limit=None)
        self.assertEqual(sorted(v["video_id"] for v in videos), ["h1", "h2"])
        h1 = next(v for v in videos if v["video_id"] == "h1")
        self.assertEqual(len(h1["targets"]), 2)
        self.assertEqual(len(handbook.holdout_videos(self.held, limit=1)), 1)
        reordered = handbook.holdout_videos(list(reversed(self.held)), limit=None)
        self.assertEqual([v["video_id"] for v in videos], [v["video_id"] for v in reordered])


# ---- the report page ------------------------------------------------------------------------------

def make_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals.update({
        "brand_name": lambda: "Test Brand",
        "ws_current": lambda: {"slug": "test", "label": "Test", "accent": ""},
        "brand_logo": lambda: "", "workspace_options": lambda: [],
        "attention_count": lambda: 0, "threads_authenticated": lambda: True,
        "static_v": "1", "thumb_for": lambda candidate: "", "category_options": lambda: [],
        "category_by_slug": lambda slug: None, "digest_meta": lambda *a, **k: {},
        "default_app_name": "Throughline",
        "is_admin": lambda request: True,
    })
    return templates


class PageTest(TempDirCase):
    def setUp(self):
        super().setUp()
        app = FastAPI()
        app.state.templates = make_templates()
        app.include_router(clip_handbook.router)
        self.client = TestClient(app)
        self.queue = self.start(mock.patch("app.clip_engine.notes.proposals", return_value=[]))
        self.start(mock.patch("app.config.load_brand_rules", return_value=[]))

    def test_empty_state(self):
        r = self.client.get("/clip-handbook")
        self.assertEqual(r.status_code, 200)
        self.assertIn("No handbook yet", r.text)
        self.assertIn("clip-engine handbook build", r.text)

    def test_report_with_statuses_and_validation(self):
        rep = report(validation={"at": "2026-11-18T12:00:00Z", "videos": 15, "rules_tested": 3,
                                 "setup": "gemini", "cost_usd": 20.5, "stopped": "",
                                 "current": {"n": 15, "moment_hit1": 0.4, "moment_hit3": 0.6,
                                             "hit1": 0.2, "hit3": 0.3, "errors": 1, "empty": 0},
                                 "candidates": {"n": 15, "moment_hit1": 0.533,
                                                "moment_hit3": 0.667, "hit1": 0.2, "hit3": 0.4,
                                                "errors": 0, "empty": 0}})
        handbook.save(rep)
        self.queue.return_value = [{"text": "Usually drop the studio toss", "status": "pending"}]
        r = self.client.get("/clip-handbook")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Open on the aerial b-roll, not the anchor", r.text)
        self.assertIn("France 24 packages", r.text)
        self.assertIn("Waiting for you", r.text)
        self.assertIn("Not proposed", r.text)
        self.assertIn("Review suggested rules (1)", r.text)
        self.assertIn('href="/brand#rule-proposals"', r.text)
        self.assertIn("53%", r.text)
        self.assertIn("Current + candidates", r.text)
        self.assertIn("89%", r.text)          # 8 of 9 holds
        self.assertNotIn("<form", r.text)     # a report: nothing to post

    def test_model_text_is_escaped(self):
        rep = report()
        rep["candidates"][0]["text"] = "<script>alert(1)</script> open"
        rep["summary"] = "<b>bold</b>"
        handbook.save(rep)
        r = self.client.get("/clip-handbook")
        self.assertNotIn("<script>alert(1)</script>", r.text)
        self.assertIn("&lt;script&gt;", r.text)
        self.assertNotIn("<b>bold</b>", r.text)

    def test_not_validated_yet(self):
        handbook.save(report())
        self.assertIn("Not validated yet", self.client.get("/clip-handbook").text)

    def test_a_failure_is_a_500_page_not_a_crash(self):
        with mock.patch.object(handbook, "load", side_effect=RuntimeError("bad yaml")):
            r = self.client.get("/clip-handbook")
        self.assertEqual(r.status_code, 500)
        self.assertIn("bad yaml", r.text)

    def test_no_write_routes(self):
        methods = {m for route in clip_handbook.router.routes for m in getattr(route, "methods", ())}
        self.assertEqual(methods, {"GET"})

    def test_views(self):
        self.assertEqual(clip_handbook.percent(0.4), "40%")
        self.assertEqual(clip_handbook.percent(None), "–")
        self.assertEqual(clip_handbook.percent("x"), "–")
        row = clip_handbook.candidate_view({"text": "t", "support": 3, "exceptions": 1,
                                            "examples": [1, None, 2], "status": "added"})
        self.assertEqual((row["share"], row["examples_count"], row["status_label"]),
                         ("75%", 2, "Added as a rule"))
        self.assertIsNone(clip_handbook.validation_view(None))

    def test_analytics_links_the_page(self):
        text = (TEMPLATES_DIR / "analytics.html").read_text()
        self.assertIn('href="/clip-handbook"', text)


class ImportSafetyTest(unittest.TestCase):
    def test_the_page_module_loads_no_engine_at_import(self):
        probe = ("import json, sys, importlib; importlib.import_module('app.web.clip_handbook'); "
                 "print(json.dumps(sorted(m for m in sys.modules if m.startswith('app.clip_engine'))))")
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, DATABASE_URL="sqlite://", SCHEDULER_EMBEDDED="false",
                       GEMINI_API_KEY="", ANTHROPIC_API_KEY="", SUPABASE_SERVICE_KEY="",
                       CLIP_ENGINE_DIR=str(Path(tmp) / "e"), CLIPBENCH_DIR=str(Path(tmp) / "b"))
            proc = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                                  capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(json.loads(proc.stdout.strip().splitlines()[-1]), [])


if __name__ == "__main__":
    unittest.main()
