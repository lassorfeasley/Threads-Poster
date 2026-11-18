"""Tests for the replay benchmark (app/clip_engine/bench.py): arm specs,
frozen selection, the resumable run loop, file-only evaluation, blind pairs
and judging, the statistics behind the verdict, the report, the critic check.

Every collaborator that would spend money, touch the network or run ffmpeg is
mocked; the database is an in-memory SQLite (StaticPool, shared across the
bench's own sessions and worker threads).

Run with:  DATABASE_URL="sqlite://" python -m pytest -q tests/test_clip_bench.py
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import clip_proposals, offload, spend
from app.clip_engine import (bench, critic, engine, jobs, legacy, media, playbook, shotlog, taste,
                             util)
from app.config import Settings
from app.models import (Base, Candidate, Channel, ClipProposal, ContentLabel, Cut,
                        MetricSnapshot, ThreadsPost)

NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def S(a, b):
    return {"start": float(a), "end": float(b)}


def ago(days=0.0, hours=0.0):
    return NOW - dt.timedelta(days=days, hours=hours)


class FakeMeter:
    """Stands in for engine.CostMeter (Agent 4a's), with the spec'd surface."""

    def __init__(self, bucket="engine"):
        self.bucket = bucket
        self.total = 0.0
        self.by_stage: dict = {}
        self.tokens: dict = {}

    def add(self, stage, usage):
        cost = float((usage or {}).get("cost_usd") or 0.0)
        self.total += cost
        self.by_stage[stage] = self.by_stage.get(stage, 0.0) + cost
        model = (usage or {}).get("model")
        if model:
            t = self.tokens.setdefault(model, {"input": 0, "output": 0})
            t["input"] += int(usage.get("input_tokens", 0))
            t["output"] += int(usage.get("output_tokens", 0))

    def add_cached(self, stage, cost):
        self.by_stage[f"{stage}_cached"] = self.by_stage.get(f"{stage}_cached", 0.0) + cost


# ---- fixtures -------------------------------------------------------------------

class _Env(unittest.TestCase):
    """Fresh bench/engine dirs, a temp spend ledger, hermetic git."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="clip-bench-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ, {"CLIPBENCH_DIR": str(self.tmp / "bench"),
                                           "CLIP_ENGINE_DIR": str(self.tmp / "engine")})
        env.start()
        self.addCleanup(env.stop)
        self.patch(spend, "_LEDGER", self.tmp / "llm_spend.json")
        self.patch(bench, "_git_sha", lambda root: "testsha")
        self.buckets: list = []
        self.patch(spend, "set_bucket", lambda name: self.buckets.append(name))
        self.patch(media, "prune", lambda *a, **k: 0)
        self.said: list[str] = []

    def patch(self, obj, name, value):
        patcher = mock.patch.object(obj, name, value, create=True)
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def say(self, line):
        self.said.append(str(line))


class _DB(_Env):
    def setUp(self):
        super().setUp()
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, future=True, expire_on_commit=False)
        self.s = self.Session()
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.s.close)
        self.patch(bench, "_db", self.fake_db)
        self.files = self.tmp / "files"
        self.files.mkdir()
        self.channel = self.add(Channel(call_sign="KTST", url="https://youtube.com/@ktst"))

    @contextlib.contextmanager
    def fake_db(self):
        session = self.Session()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    def add(self, row):
        self.s.add(row)
        self.s.flush()
        return row

    def commit(self):
        self.s.commit()

    def video(self, vid, *, duration=200, words=True, local=True, key="", status="archived",
              url=None, channel=None, transcript=True, archived_at=None):
        tpath = self.files / f"{vid}.json"
        if transcript:
            tpath.write_text(json.dumps([{"start": 0.0, "end": float(duration), "text": "hi"}]))
        wpath = self.files / f"{vid}_words.json"
        if words:
            wpath.write_text(json.dumps([{"word": "hi", "start": 1.0, "end": 1.4}]))
        lpath = self.files / f"{vid}.mp4"
        if local:
            lpath.write_bytes(b"not really a video")
        return self.add(Candidate(
            video_id=vid, channel_pk=(channel or self.channel).id, title=f"Title {vid}",
            url=url or f"https://youtu.be/{vid}", status=status, duration_seconds=duration,
            transcript_path=str(tpath) if transcript else "",
            word_transcript_path=str(wpath) if words else "",
            local_video_path=str(lpath), source_object_key=key,
            archived_at=archived_at or ago(30)))

    def cut(self, cand, segs, *, exported=True, created=None, title=""):
        return self.add(Cut(candidate_pk=cand.id, trim_segments=json.dumps(segs),
                            trimmed_clip_path="/clips/x.mp4" if exported else "",
                            created_at=created or ago(20), clip_title=title))

    def post(self, cut, *, views=None, published=None, status="published", repost_of=None,
             object_path="2026/10/x.mp4", local_path=""):
        post = self.add(ThreadsPost(candidate_pk=cut.candidate_pk, cut_pk=cut.id, status=status,
                                    published_at=published or ago(19),
                                    repost_of_post_pk=repost_of,
                                    clip_object_path=object_path, clip_local_path=local_path))
        if views is not None:
            self.add(MetricSnapshot(post_pk=post.id, views=views,
                                    captured_at=post.published_at + dt.timedelta(hours=6)))
        return post


def write_run(name, *, videos, results, arms=("gemini", "transcript"), ledger=None, **extra):
    """A synthetic run on disk, as bench.run would have left it."""
    run_dir = bench.bench_dir() / name
    specs = {}
    for label in arms:
        arm = bench.parse_arm(label.split("@")[0])
        specs[label] = {"spec": arm["label"], "setup": arm["setup"],
                        "overrides": arm["overrides"],
                        "display": bench._display_label(label, arm["setup"]),
                        "added_at": "2026-11-01T00:00:00Z"}
    manifest = {"name": name, "seed": "test", "mode": "sampled", "arms": list(arms),
                "arm_specs": specs, "created_at": "2026-11-01T00:00:00Z",
                "fingerprint": {"hash": "f" * 16}, "max_usd": 40.0, "invocations": [],
                "warnings": [], "stopped": None, "revealed_at": None}
    manifest.update(extra)
    util.atomic_write_json(run_dir / "manifest.json", manifest)
    util.atomic_write_json(run_dir / "videos.json", videos)
    for label, per_video in results.items():
        for vid, result in per_video.items():
            util.atomic_write_json(run_dir / label / f"{vid}.json", result)
    if ledger is not None:
        util.atomic_write_json(run_dir / "ledger.json", {"label": bench.LEDGER_LABEL, "runs": ledger})
    return run_dir


def ventry(vid, targets, *, duration=100.0, negative=False, bucket="<120s", kind="channel",
           local=""):
    return {"candidate_pk": abs(hash(vid)) % 10_000, "video_id": vid, "title": vid,
            "kind": kind, "duration": duration, "length_bucket": bucket,
            "stratum": f"{bucket}|{kind}", "as_of": "2026-10-01T00:00:00Z",
            "targets": targets, "negative": negative, "local_video_path": local,
            "source_object_key": ""}


def target(cut_pk, segs, *, seeded=False, published=True, views=100):
    return {"cut_pk": cut_pk, "segments": segs, "seeded": seeded, "published": published,
            "views": views, "title": "", "members": [cut_pk]}


def result(top, *, error="", cost=0.1, outcome=None, drafts=None, orders=None, by_stage=None,
           tokens=None, seconds=30.0, source_seconds=None):
    return {"top": [{"id": f"d{i + 1}", "segments": s} for i, s in enumerate(top)],
            "drafts": drafts or [], "orders": orders or {}, "error": error, "cost_usd": cost,
            "outcome": outcome or ("error" if error else ("proposals" if top else "empty")),
            "by_stage": by_stage or {}, "tokens": tokens or {}, "seconds": seconds,
            "source_seconds": source_seconds}


# ---- arms --------------------------------------------------------------------------

class ParseArmTest(unittest.TestCase):
    def test_specs(self):
        cases = [
            ("gemini", "gemini", {}),
            ("transcript", "transcript", {}),
            ("gemini+editor_thinking=HIGH", "gemini",
             {"clip_engine.gemini.editor_thinking": "HIGH"}),
            ("gemini+editor_thinking=HIGH+critic_media_resolution=MEDIA_RESOLUTION_MEDIUM", "gemini",
             {"clip_engine.gemini.editor_thinking": "HIGH",
              "clip_engine.gemini.critic_media_resolution": "MEDIA_RESOLUTION_MEDIUM"}),
            ("gemini+editor_model=gemini-4-hydrogen", "gemini",
             {"clip_engine.gemini.editor_model": "gemini-4-hydrogen"}),
            ("gemini+judge_thinking=MEDIUM", "gemini", {"clip_engine.gemini.judge_thinking": "MEDIUM"}),
            ("hybrid+editor_effort=high", "hybrid", {"clip_engine.claude.editor_effort": "high"}),
            ("hybrid+judge_effort=low", "hybrid", {"clip_engine.claude.judge_effort": "low"}),
            ("hybrid+editor_keyframes=60", "hybrid", {"clip_engine.claude.editor_keyframes": 60}),
            # Claude reviews (and draws keyframes) at this width in hybrid; Gemini logs.
            ("hybrid+frame_width=320", "hybrid", {"clip_engine.claude.frame_width": 320}),
            ("gemini+perceiver_model=gemini-4-carbon", "gemini",
             {"clip_engine.gemini.perceiver_model": "gemini-4-carbon"}),
            ("gemini+drafts=12+revise_top=0", "gemini",
             {"clip_engine.drafts": 12, "clip_engine.revise_top": 0}),
            ("gemini+word_window=0.8", "gemini", {"clip_engine.snap.word_window": 0.8}),
            ("hybrid+gemini.model=gemini-4-carbon", "hybrid",
             {"clip_engine.gemini.model": "gemini-4-carbon"}),
            ("gemini+clip_engine.gemini.critic_fps=4", "gemini",
             {"clip_engine.gemini.critic_fps": 4}),
            ("gemini+taste_hits=0+taste_flops=1+job_timeout_seconds=600", "gemini",
             {"clip_engine.taste_hits": 0, "clip_engine.taste_flops": 1,
              "clip_engine.job_timeout_seconds": 600}),
            ("gemini+editor_video_max_seconds=600", "gemini",
             {"clip_engine.editor_video_max_seconds": 600}),
            ("transcript+clips.model=claude-sonnet-5", "transcript",
             {"clips.model": "claude-sonnet-5"}),
            ("transcript+clips.max_clips=2+clips.opening_vision=false", "transcript",
             {"clips.max_clips": 2, "clips.opening_vision": False}),
            ("gemini+clips.learn_from_history=false", "gemini",
             {"clips.learn_from_history": False}),
            ("transcript+clips.context_examples=2", "transcript", {"clips.context_examples": 2}),
        ]
        for spec, setup, overrides in cases:
            with self.subTest(spec):
                arm = bench.parse_arm(spec)
                self.assertEqual(arm, {"label": spec, "setup": setup, "overrides": overrides})

    def test_bad_specs(self):
        cases = [
            "",                                     # empty
            "vimeo",                                # unknown setup
            "gemini+editor_effort=high",            # a Claude key on the Gemini editor
            "hybrid+critic_media_resolution=X",     # a Gemini key on the Claude critic
            "claude+critic_fps=2",                  # ditto
            "hybrid+model=x",                       # ambiguous: both providers have it
            "gemini+bogus=1",                       # unknown key
            "gemini+editor_thinking",               # no value
            "gemini+=HIGH",                         # no key
            "gemini+drafts=1+drafts=2",             # set twice
            "gemini/../x",                          # unsafe as a directory name
            "transcript+model=x",                   # bare keys are engine-only
            "gemini@abc12345",                      # '@' is reserved for version tags
        ]
        for spec in cases:
            with self.subTest(spec):
                with self.assertRaises(bench.BenchError):
                    bench.parse_arm(spec)

    def test_overrides_the_arm_would_not_run_with_are_refused(self):
        # Each of these used to be accepted and then silently ignored, so the
        # label claimed a configuration that never ran.
        cases = [
            ("gemini+taste_lookback_days=30", "frozen per video"),         # frozen at creation
            ("gemini+gemini.edtor_thinking=HIGH", "unknown setting"),      # typo, dotted
            ("gemini+clip_engine.drafs=4", "unknown setting"),             # typo, dotted top key
            ("gemini+snap.word_windo=1", "unknown setting"),
            ("transcript+clips.lern_from_history=false", "isn't a clips setting"),
            ("gemini+clips.max_clips=2", "isn't a clips setting"),         # the engine reads keep
            ("gemini+bench.limit=5", "configures the run"),
            ("gemini+clip_engine.setup=claude", "unknown setting"),
            ("claude+gemini.model=gemini-4-carbon", "no gemini model"),    # no Gemini role
            ("gemini+claude.editor_effort=high", "no claude model"),
            ("claude+perception_window_seconds=60", "only a Gemini perceiver"),
            ("hybrid+editor_video_max_seconds=600", "only a Gemini editor"),
            ("transcript+gemini.editor_thinking=HIGH", "reads clips.*"),
            ("gemini+matching.model=x", "clip_engine.* and clips.*"),
            ("gemini+inline_request_mb=5", "settings.yaml"),               # read globally
            ("gemini+taste_hits=-1", "whole number"),
            ("gemini+taste_flops=two", "whole number"),
            ("gemini+clips.context_examples=1.5", "whole number"),
            ("gemini+clips.learn_from_history=maybe", "true or false"),
            ("gemini+job_timeout_seconds=0", "above 0"),
        ]
        for spec, why in cases:
            with self.subTest(spec):
                with self.assertRaises(bench.BenchError) as ctx:
                    bench.parse_arm(spec)
                self.assertIn(why, str(ctx.exception))

    PERCEPTION_ARMS = [
        "gemini+perception_fps=2.0", "gemini+media_resolution=MEDIA_RESOLUTION_MEDIUM",
        "gemini+perceiver_thinking=HIGH", "gemini+long_video_fps=1",
        "gemini+perception_window_seconds=120", "gemini+gemini.perception_fps=2",
        "hybrid+perceiver_thinking=HIGH", "hybrid+media_resolution=MEDIA_RESOLUTION_MEDIUM",
        "claude+frame_interval=1.0", "claude+frame_width=320", "claude+perceiver_effort=high",
        "claude+max_total_frames=100", "claude+max_frames_per_call=50"]

    def test_perception_settings_are_refused_while_shot_logs_are_not_keyed_by_them(self):
        # Logs cached per (video, perceiver, model) only: an arm changing how
        # its perceiver logs would reuse a log made without the change.
        def unkeyed(video_id, perceiver, model, *, part=None):
            return Path(f"/logs/{video_id}__{perceiver}__{model}.json")

        def ignores_settings(video_id, perceiver, model, *, part=None, settings=None, seconds=None):
            return unkeyed(video_id, perceiver, model)

        for fake in (unkeyed, ignores_settings):
            with mock.patch.object(shotlog, "cache_path", fake):
                for spec in self.PERCEPTION_ARMS:
                    with self.subTest(fake=fake.__name__, spec=spec):
                        with self.assertRaises(bench.BenchError) as ctx:
                            bench.parse_arm(spec)
                        self.assertIn("not cached per", str(ctx.exception))
                # A different perceiver model is its own log; Claude reviewing
                # at another width in hybrid (Gemini logs) isn't perception.
                bench.parse_arm("gemini+perceiver_model=gemini-4-carbon")
                bench.parse_arm("hybrid+frame_width=320")
        # Routing is unchanged: the perceiver's provider owns the key.
        self.assertEqual(bench._override_path("hybrid", "perceiver_thinking"),
                         "clip_engine.gemini.perceiver_thinking")
        self.assertEqual(bench._override_path("claude", "frame_interval"),
                         "clip_engine.claude.frame_interval")

    def test_perception_settings_run_once_shot_logs_are_keyed_by_them(self):
        def keyed(video_id, perceiver, model, *, part=None, settings=None, seconds=None):
            knobs = json.dumps((settings.raw if settings is not None else {}), sort_keys=True)
            return Path(f"/logs/{video_id}__{perceiver}__{model}__{knobs}.json")

        with mock.patch.object(shotlog, "cache_path", keyed):
            for spec in self.PERCEPTION_ARMS:
                with self.subTest(spec):
                    self.assertTrue(bench.parse_arm(spec)["overrides"])
        # The real shot log is keyed by every one of them now: each arm runs,
        # and gets a log of its own rather than its baseline's.
        for spec in self.PERCEPTION_ARMS:
            with self.subTest(real=spec):
                arm = bench.parse_arm(spec)
                setup = arm["setup"]
                perceiver = "claude" if setup == "claude" else "gemini"
                own = bench.settings_for(arm, Settings(raw={}))
                self.assertTrue(any(
                    shotlog.cache_path("v", perceiver, "m", settings=own, seconds=seconds)
                    != shotlog.cache_path("v", perceiver, "m", settings=Settings(raw={}),
                                          seconds=seconds)
                    for seconds in (None, 3600.0)))

    def test_values_are_typed(self):
        cases = [("12", 12), ("0.75", 0.75), ("true", True), ("False", False),
                 ("HIGH", "HIGH"), ("gemini-4-hydrogen", "gemini-4-hydrogen"), ("nan", "nan")]
        for raw, expected in cases:
            with self.subTest(raw):
                self.assertEqual(bench._scalar(raw), expected)


class SettingsForTest(unittest.TestCase):
    def test_overrides_apply_to_a_copy(self):
        base = Settings(raw={"clip_engine": {"drafts": 10, "gemini": {"editor_thinking": "LOW"}}})
        arm = bench.parse_arm("gemini+editor_thinking=HIGH+word_window=0.8+drafts=12")
        out = bench.settings_for(arm, base)
        self.assertEqual(out.get("clip_engine.gemini.editor_thinking"), "HIGH")
        self.assertEqual(out.get("clip_engine.snap.word_window"), 0.8)
        self.assertEqual(out.get("clip_engine.drafts"), 12)
        self.assertEqual(base.get("clip_engine.gemini.editor_thinking"), "LOW")
        self.assertIsNone(base.get("clip_engine.snap"))

    def test_override_through_a_scalar_is_refused(self):
        base = Settings(raw={"clip_engine": "oops"})
        with self.assertRaises(bench.BenchError):
            bench.settings_for(bench.parse_arm("gemini+drafts=3"), base)

    def test_model_override_reaches_model_for(self):
        from app.clip_engine import model_for

        out = bench.settings_for(bench.parse_arm("gemini+editor_model=gemini-4-hydrogen"),
                                 Settings(raw={}))
        self.assertEqual(model_for(out, "gemini", "editor"), "gemini-4-hydrogen")
        self.assertEqual(model_for(out, "gemini", "critic"), "gemini-3.8-flash")


class FingerprintTest(_Env):
    def test_components_and_hash(self):
        settings = Settings(raw={"clip_engine": {"drafts": 10, "workers": 1}})
        fp = bench.fingerprint(settings, {"archetypes": []})
        for key in ("git", "prompts", "code", "playbook", "settings", "clips_settings", "brand",
                    "models", "versions", "hash"):
            self.assertIn(key, fp)
        self.assertEqual(fp["git"], "testsha")
        self.assertEqual(fp["models"]["hybrid"]["editor"], "claude-opus-5-5")
        self.assertEqual(fp["models"]["hybrid"]["perceiver"], "gemini-3.8-flash")

        same = bench.fingerprint(Settings(raw={"clip_engine": {"drafts": 10, "workers": 4}}),
                                 {"archetypes": []})
        self.assertEqual(same["hash"], fp["hash"], "operational settings don't version a run")
        changed = bench.fingerprint(Settings(raw={"clip_engine": {"drafts": 12}}), {"archetypes": []})
        self.assertNotEqual(changed["hash"], fp["hash"])
        self.assertEqual(bench._fingerprint_diff(fp, changed), ["settings"])
        other_pb = bench.fingerprint(settings, {"archetypes": [{"key": "voice"}]})
        self.assertIn("playbook", bench._fingerprint_diff(fp, other_pb))


class IntervalMathTest(unittest.TestCase):
    def test_iou_matches_clip_proposals(self):
        cases = [
            ([S(0, 10)], [S(5, 15)]),
            ([S(0, 10), S(20, 30)], [S(5, 25)]),
            ([S(50, 60), S(10, 20)], [S(10, 20), S(50, 60)]),   # order-blind
            ([S(0, 10)], [S(10, 20)]),
            ([], [S(0, 1)]),
        ]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(bench.iou(a, b), clip_proposals.iou(a, b), places=4)

    def test_overlap_share_is_relative_to_the_shorter_clip(self):
        cases = [([S(12, 16)], [S(10, 40)], 1.0), ([S(0, 10)], [S(5, 25)], 0.5),
                 ([S(0, 10)], [S(20, 30)], 0.0)]
        for a, b, expected in cases:
            with self.subTest(a=a):
                self.assertAlmostEqual(bench.overlap_share(a, b), expected)

    def test_slivers_dropped_order_kept(self):
        self.assertEqual(bench._drop_slivers([S(50, 60), S(10, 10.2), S(20, 30)]),
                         [S(50, 60), S(20, 30)])

    def test_chance_baseline(self):
        # Window [s, s+20] vs target [10, 30] in a 100 s source: IoU ≥ 0.5 iff
        # |s - 10| ≤ 20/3, i.e. 13.3 s of the 80 s of room.
        self.assertAlmostEqual(bench.chance_hit(20, [[S(10, 30)]], 100), (40 / 3) / 80, delta=0.01)
        cases = [
            (0, [[S(10, 30)]], 100, 0.0),          # no proposal
            (4, [[S(0, 20)]], 100, 0.0),           # too short to ever reach 0.5
            (100, [[S(0, 60)]], 100, 1.0),         # the whole video
            (100, [[S(0, 20)]], 100, 0.0),
            (20, [], 100, 0.0),
        ]
        for length, targets, duration, expected in cases:
            with self.subTest(length=length, targets=targets):
                self.assertAlmostEqual(bench.chance_hit(length, targets, duration), expected)
        # Short sources make a hit cheap.
        self.assertGreater(bench.chance_hit(30, [[S(0, 40)]], 60), 0.5)


# ---- selection -----------------------------------------------------------------------

class SelectVideosTest(_DB):
    def setUp(self):
        super().setUp()
        pasted = self.add(Channel(call_sign="pasted", url="youtube://pasted"))
        ours = self.add(Channel(call_sign="Ours", url="https://youtube.com/@ours", first_party=True))
        v = self.video
        self.ok = v("ok", duration=200)
        self.cut(self.ok, [S(10, 30)])
        self.cut(v("unarchived", status="approved"), [S(0, 10)])
        self.cut(v("firstparty", channel=ours), [S(0, 10)])
        self.cut(v("notranscript", transcript=False), [S(0, 10)])
        self.cut(v("nowords", words=False), [S(0, 10)])
        self.cut(v("nosource", local=False), [S(0, 10)])
        self.cut(v("cloud", local=False, key="climate/cloud.mp4", duration=1800), [S(0, 10)])
        self.cut(v("unexported"), [S(0, 10)], exported=False)
        self.post(self.cut(v("postonly", duration=60), [S(0, 10)], exported=False), views=10)
        self.post(self.cut(v("queued"), [S(0, 10)], exported=False), status="queued")
        self.cut(v("slivers"), [S(0, 0.2), S(5, 5.1)])
        self.cut(v("pasted", channel=pasted, duration=600), [S(0, 10)])
        self.cut(v("upload", url="upload://upl", duration=90), [S(0, 10)])
        self.nocut = v("nocut")
        self.commit()

    def select(self, **kw):
        kw.setdefault("limit", 50)
        with self.fake_db() as s:
            return bench.select_videos(s, **kw)

    def test_eligibility(self):
        cases = [
            ({}, {"ok", "cloud", "postonly", "pasted", "upload"}),
            ({"require_words": False}, {"ok", "cloud", "postonly", "pasted", "upload", "nowords"}),
            ({"exclude_video_ids": {"ok", "cloud"}}, {"postonly", "pasted", "upload"}),
        ]
        for kw, expected in cases:
            with self.subTest(kw=kw):
                self.assertEqual({v["video_id"] for v in self.select(**kw)}, expected)

    def test_strata_and_kinds(self):
        got = {v["video_id"]: (v["length_bucket"], v["kind"], v["stratum"]) for v in self.select()}
        self.assertEqual(got["ok"], ("120-300s", "channel", "120-300s|channel"))
        self.assertEqual(got["cloud"][0], ">1200s")
        self.assertEqual(got["postonly"][0], "<120s")
        self.assertEqual(got["pasted"][:2], ("300-1200s", "pasted"))
        self.assertEqual(got["upload"][:2], ("<120s", "upload"))

    def test_entry_shape(self):
        entry = next(v for v in self.select() if v["video_id"] == "ok")
        self.assertEqual(entry["candidate_pk"], self.ok.id)
        self.assertFalse(entry["negative"])
        self.assertTrue(entry["has_words"])
        self.assertEqual(entry["as_of_source"], "archived_at")
        self.assertEqual(bench._parse_time(entry["as_of"]), self.ok.archived_at.replace(
            tzinfo=dt.timezone.utc) if self.ok.archived_at.tzinfo is None else self.ok.archived_at)
        self.assertEqual(entry["targets"][0]["segments"], [S(10, 30)])

    def test_selection_is_deterministic_and_limited(self):
        first = [v["video_id"] for v in self.select(limit=3, seed="test")]
        again = [v["video_id"] for v in self.select(limit=3, seed="test")]
        self.assertEqual(first, again)
        self.assertEqual(len(first), 3)

    def test_hand_picked(self):
        picks = [str(self.ok.id), "nocut", "unarchived"]
        with self.assertLogs("clip_engine.bench", "WARNING"):
            got = self.select(candidate_ids=picks)
        self.assertEqual([v["video_id"] for v in got], ["ok", "nocut"])
        self.assertEqual([v["negative"] for v in got], [False, True])

    def test_negatives_are_never_cut(self):
        with self.fake_db() as s:
            got = bench.select_negatives(s, count=5, seed="test")
            self.assertEqual([v["video_id"] for v in got], ["nocut"])
            self.assertTrue(got[0]["negative"])
            self.assertEqual(bench.select_negatives(s, count=0), [])
            self.assertEqual(bench.select_negatives(s, count=5, exclude_video_ids={"nocut"}), [])


class TargetsTest(_DB):
    def test_exported_cuts_cluster_and_flag(self):
        multi = self.video("multi", duration=300)
        a = self.cut(multi, [S(10, 40)], created=ago(25))
        b = self.cut(multi, [S(12, 38)], created=ago(24))
        c = self.cut(multi, [S(100, 120)], created=ago(23))
        self.cut(multi, [S(12, 40)], exported=False)                 # never exported
        self.post(a, views=500)
        self.post(b, views=2000)
        self.add(ClipProposal(candidate_pk=multi.id, cut_pk=a.id, verdict="accepted",
                              proposed_segments="[]"))
        self.add(ClipProposal(candidate_pk=multi.id, cut_pk=c.id, verdict="dismissed",
                              proposed_segments="[]"))
        cold = self.cut(multi, [S(200, 210), S(150, 150.1), S(160, 170)], created=ago(22))
        self.commit()
        with self.fake_db() as s:
            targets = bench._exported_targets(s, [multi.id])[multi.id]
        self.assertEqual(len(targets), 3)
        first, second, third = targets
        # Re-cuts of one moment: one target, represented by the published, most-viewed cut.
        self.assertEqual(first["cut_pk"], b.id)
        self.assertEqual(first["segments"], [S(12, 38)])
        self.assertEqual(sorted(first["members"]), sorted([a.id, b.id]))
        self.assertTrue(first["published"])
        self.assertEqual(first["views"], 2000)
        self.assertTrue(first["seeded"], "a member started from an accepted suggestion")
        self.assertEqual((second["cut_pk"], second["published"], second["views"], second["seeded"]),
                         (c.id, False, None, False))
        # Play order (a cold open) kept; the hand-made sliver dropped.
        self.assertEqual(third["cut_pk"], cold.id)
        self.assertEqual(third["segments"], [S(200, 210), S(160, 170)])

    def test_buckets_and_kinds(self):
        cases = [(None, "unknown"), (0, "<120s"), (119.9, "<120s"), (120, "120-300s"),
                 (299, "120-300s"), (300, "300-1200s"), (1200, "300-1200s"), (1201, ">1200s")]
        for seconds, expected in cases:
            with self.subTest(seconds=seconds):
                self.assertEqual(bench._length_bucket(seconds), expected)
        kinds = [("upload://abc", "https://x", "upload"), ("https://youtu.be/x", "youtube://pasted",
                  "pasted"), ("https://youtu.be/x", "https://youtube.com/@a", "channel")]
        for url, channel_url, expected in kinds:
            with self.subTest(url=url, channel=channel_url):
                self.assertEqual(bench._source_kind(url, channel_url), expected)


class StratifyTest(unittest.TestCase):
    def videos(self, stratum, n, prefix):
        return [{"video_id": f"{prefix}{i}", "stratum": stratum} for i in range(n)]

    def test_long_sources_oversampled_and_every_stratum_seated(self):
        pool = (self.videos(">1200s|channel", 10, "long") + self.videos("120-300s|channel", 88, "mid")
                + self.videos("<120s|upload", 2, "up"))
        got = bench._stratified(pool, limit=20, seed="test")
        self.assertEqual(len(got), 20)
        count = lambda p: sum(1 for v in got if v["video_id"].startswith(p))  # noqa: E731
        self.assertGreaterEqual(count("long"), 3, "×2 weight: proportional would give 2")
        self.assertEqual(count("up"), 1)

    def test_seed_changes_the_draw_and_limit_caps(self):
        pool = self.videos("120-300s|channel", 40, "v")
        a = [v["video_id"] for v in bench._stratified(pool, limit=10, seed="test")]
        b = [v["video_id"] for v in bench._stratified(pool, limit=10, seed="dev")]
        self.assertNotEqual(a, b)
        self.assertEqual(len(bench._stratified(pool, limit=100, seed="x")), 40)
        self.assertEqual(bench._stratified(pool, limit=0, seed="x"), [])

    def test_inside_a_stratum_order_is_sha1_of_seed_and_id(self):
        pool = self.videos("120-300s|channel", 5, "v")
        got = bench._stratified(pool, limit=2, seed="s")
        expected = sorted((v["video_id"] for v in pool),
                          key=lambda vid: hashlib.sha1(f"s{vid}".encode()).hexdigest())[:2]
        self.assertEqual(sorted(v["video_id"] for v in got), sorted(expected))


class TouchedTest(_Env):
    def test_touch_and_read(self):
        bench.touch_videos(["a", "b"], source="try")
        bench.touch_videos(["b", ""], source="bench:x", seed="dev")
        self.assertEqual(bench.touched_video_ids(), {"a", "b"})
        data = json.loads((bench.bench_dir() / "touched.json").read_text())
        self.assertEqual(len(data["videos"]["b"]), 2)

    def test_tolerates_other_shapes(self):
        cases = [(["x", "y"], {"x", "y"}), ({"x": {}, "y": []}, {"x", "y"}),
                 ({"videos": ["z"]}, {"z"}), ([{"video_id": "q"}], {"q"}), ("junk", set())]
        for data, expected in cases:
            with self.subTest(data=data):
                (bench.bench_dir() / "touched.json").write_text(json.dumps(data))
                self.assertEqual(bench.touched_video_ids(), expected)
        # Writing upgrades an older shape without losing what it held.
        (bench.bench_dir() / "touched.json").write_text(json.dumps([{"video_id": "q"}]))
        bench.touch_videos(["w"], source="try")
        self.assertEqual(bench.touched_video_ids(), {"w", "q"})


# ---- run -------------------------------------------------------------------------------

def fake_engine_result(setup, cost):
    top = [{"id": "d1", "segments": [{"start": 10.0, "end": 30.0, "lead": "picture"}],
            "story": "the story", "story_key": "nye-fire", "archetype": "voice", "lens": "person",
            "hook": "hook", "score": 7.5, "rank": 1, "critique": {"score": 8, "stops_scroll": True},
            "render": Path("/tmp/render.mp4"), "credits": "Bill Nye"}]
    drafts = [{"id": "d1", "segments": [{"start": 10.0, "end": 30.0, "lead": "picture"}],
               "score": 7.5, "render": Path("/tmp/render.mp4")},
              {"id": "d2", "segments": [S(50, 70)], "dropped_reason": "length"}]
    return SimpleNamespace(
        setup=setup, top=top, drafts=drafts, moments=[{"id": "m1", "start": 10, "end": 30}],
        shotlog={"beats": []}, cost_usd=cost, by_stage={"draft": cost},
        tokens={"gemini-3.8-flash": {"input": 1000, "output": 100}}, timings={"draft": 1.0},
        notes=["a note"], orders={"editor": ["d2", "d1"], "cold": ["d1"], "rank": ["d1"],
                                  "final": ["d1"]},
        model="engine:gemini:gemini-3.8-flash", policy="pol123", outcome="proposals")


class _RunFixture(_DB):
    def setUp(self):
        super().setUp()
        self.vids = [self.video(f"v{i}", archived_at=ago(30 + i)) for i in (1, 2, 3)]
        for c in self.vids:
            self.post(self.cut(c, [S(10, 30)]), views=100)
        self.commit()
        self.settings = Settings(raw={"clip_engine": {"min_free_gb": 10, "drafts": 10},
                                      "clips": {"model": "claude-sonnet-5"}})
        self.engine_cost = 0.5
        self.engine_error: Exception | None = None
        self.engine_calls: list = []
        self.legacy_calls: list = []
        self.inputs_calls: list = []
        self.inputs_fail: set = set()
        self.restores: list = []
        self.free_gb = 500.0
        self.db_lock = threading.Lock()

        def fake_run(**kw):
            self.engine_calls.append(kw)
            kw["meter"].add("draft", {"cost_usd": self.engine_cost, "model": "gemini-3.8-flash",
                                      "input_tokens": 1000, "output_tokens": 100})
            if self.engine_error is not None:
                raise self.engine_error
            return fake_engine_result(kw["setup"], kw["meter"].total)

        def fake_suggest(**kw):
            self.legacy_calls.append(kw)
            return {"clips": [{"segments": [S(12, 32)], "story": "legacy story", "why": "w",
                               "confidence": 0.7}],
                    "model": "claude-sonnet-5", "policy": "legacypol", "cost_usd": 0.05}

        self.patch(engine, "CostMeter", FakeMeter)
        self.patch(engine, "run", fake_run)
        self.patch(engine, "estimate_usd", lambda setup, settings, duration: 0.5)
        self.patch(legacy, "suggest", fake_suggest)
        self.patch(jobs, "prepare_inputs", self.fake_inputs)
        self.patch(taste, "select_examples", lambda session, **kw: [
            {"post_pk": 11, "title": "frozen hit", "published_at": NOW},
            {"post_pk": 12, "title": "frozen flop"}])
        self.patch(playbook, "load", lambda: {"archetypes": [{"key": "voice", "examples": ["x"]}],
                                              "flops": [], "notes": ""})
        self.patch(playbook, "strip_examples", lambda pb: {
            **pb, "archetypes": [{k: v for k, v in a.items() if k != "examples"}
                                 for a in pb["archetypes"]]})
        self.patch(playbook, "fingerprint", lambda pb: "pbfp")
        self.patch(media, "link_inflight", lambda source, key: Path(source))
        self.patch(media, "release_inflight", lambda path: None)
        self.patch(media, "probe_duration", lambda path: 200.0)
        self.patch(media, "disk_free_gb", lambda path: self.free_gb)
        self.patch(offload, "ensure_local", self.fake_ensure_local)

    def fake_inputs(self, candidate_id, *, window=None, as_of=None, replay=False):
        self.inputs_calls.append({"candidate_id": candidate_id, "as_of": as_of, "replay": replay})
        if candidate_id in self.inputs_fail:
            raise RuntimeError("database went away")
        # One shared in-memory SQLite connection (StaticPool) can't serve
        # several worker threads at once; the real prepare_inputs gets its own
        # pooled connection per call.
        with self.db_lock, self.fake_db() as s:
            c = s.get(Candidate, candidate_id)
            cand = {"id": c.id, "video_id": c.video_id, "title": c.title, "description": "desc",
                    "duration_seconds": c.duration_seconds, "local_video_path": c.local_video_path,
                    "source_object_key": c.source_object_key, "transcript_path": c.transcript_path,
                    "word_transcript_path": c.word_transcript_path, "archived_at": c.archived_at,
                    "status": c.status}
        return {"candidate": cand, "used": [], "context": {"shipped_examples": [{"seconds": 10}]},
                "rejected": [], "striking": [], "notes": [],
                "examples": [{"post_pk": 11, "title": "fresh hit"}, {"post_pk": 99, "title": "?"}]}

    def fake_ensure_local(self, local, key):
        path = Path(local)
        if path.is_file():
            return path
        if not key:
            raise offload.SourceUnavailable("gone")
        path.write_bytes(b"restored")
        self.restores.append(path)
        return path

    def run_bench(self, name="t1", **kw):
        kw.setdefault("arms", ["gemini", "transcript"])
        kw.setdefault("limit", 10)
        kw.setdefault("max_usd", 100)
        kw.setdefault("settings", self.settings)
        return bench.run(name, log=self.say, **kw)

    def result(self, name, label, vid):
        return json.loads((bench.bench_dir() / name / label / f"{vid}.json").read_text())

    def manifest(self, name="t1"):
        return json.loads((bench.bench_dir() / name / "manifest.json").read_text())

    def fail_engine_on(self, video_id):
        """Engine arms error on one video (so a resume has work); returns the
        engine.run that was in place, to put back."""
        real = engine.run

        def flaky(**kw):
            if kw["video_id"] == video_id:
                raise RuntimeError("quota")
            return real(**kw)

        self.patch(engine, "run", flaky)
        return real


class RunTest(_RunFixture):
    def test_creates_the_run_and_its_results(self):
        out = self.run_bench()
        run_dir = bench.bench_dir() / "t1"
        manifest = self.manifest()
        self.assertEqual(manifest["arms"], ["gemini", "transcript"])
        self.assertEqual((manifest["mode"], manifest["seed"]), ("sampled", "test"))
        self.assertEqual(manifest["arm_specs"]["transcript"]["display"],
                         f"transcript-{clip_proposals.PROMPT_VERSION}")
        self.assertEqual(manifest["playbook"]["source"], "default")
        self.assertIn("calibration omitted", manifest["context"])
        videos = json.loads((run_dir / "videos.json").read_text())
        self.assertEqual({v["video_id"] for v in videos}, {"v1", "v2", "v3"})
        self.assertTrue(all(v["taste_post_pks"] == [11, 12] for v in videos))
        self.assertEqual(out["complete"], {"gemini": 3, "transcript": 3})
        self.assertEqual(bench.touched_video_ids(), {"v1", "v2", "v3"})
        self.assertEqual(self.buckets, ["bench", None])

        res = self.result("t1", "gemini", "v1")
        self.assertEqual(res["top"][0]["segments"], [S(10, 30)], "lead stripped, play order kept")
        self.assertEqual(res["top"][0]["story_key"], "nye-fire")
        self.assertEqual(res["fingerprint"], manifest["fingerprint"]["hash"])
        self.assertEqual(res["examples"], [11, 12])
        self.assertEqual(res["context_keys"], ["shipped_examples"])
        self.assertEqual(res["source_seconds"], 200.0)
        self.assertEqual(res["cost_usd"], 0.5)
        self.assertEqual(res["orders"]["editor"], ["d2", "d1"])
        self.assertEqual(res["drafts"][0]["render"], "/tmp/render.mp4")
        self.assertEqual((res["error"], res["outcome"], res["attempts"]), ("", "proposals", 1))

        legacy_res = self.result("t1", "transcript", "v1")
        self.assertEqual(legacy_res["label"], f"transcript-{clip_proposals.PROMPT_VERSION}")
        self.assertIsNone(legacy_res["top"][0]["score"])
        self.assertEqual(legacy_res["policy"], "legacypol")
        self.assertEqual(legacy_res["arm"], "transcript")

    def test_arms_get_replay_inputs(self):
        self.run_bench()
        self.assertEqual(len(self.engine_calls), 3)
        call = next(c for c in self.engine_calls if c["video_id"] == "v1")
        self.assertEqual((call["used"], call["other_clips"], call["rejected"], call["window"]),
                         ([], [], [], None))     # the archive moment: no clip made yet
        self.assertEqual(call["setup"], "gemini")
        self.assertEqual(call["playbook"], {"archetypes": [{"key": "voice"}], "flops": [], "notes": ""})
        self.assertEqual(call["meter"].bucket, "bench")
        self.assertEqual([e["title"] for e in call["examples"]], ["fresh hit", "frozen flop"])
        self.assertEqual(call["context"], {"shipped_examples": [{"seconds": 10}]})
        self.assertEqual(call["duration"], 200.0)
        self.assertEqual(call["source"], Path(self.vids[0].local_video_path))
        self.assertIsInstance(call["deadline"], float)
        self.assertEqual(call["words"][0]["word"], "hi")
        self.assertTrue(call["transcript"])
        inputs = next(c for c in self.inputs_calls if c["candidate_id"] == self.vids[0].id)
        self.assertTrue(inputs["replay"])
        self.assertEqual(inputs["as_of"], self.vids[0].archived_at.replace(tzinfo=dt.timezone.utc)
                         if self.vids[0].archived_at.tzinfo is None else self.vids[0].archived_at)

        legacy_call = self.legacy_calls[0]
        li = legacy_call["inputs"]
        self.assertEqual((li["used"], li["striking"], li["rejected"]), ([], [], []))
        self.assertTrue(li["transcript"])
        self.assertTrue(li["words"])
        self.assertIsNone(legacy_call["window"])
        self.assertEqual(legacy_call["context"], {"shipped_examples": [{"seconds": 10}]})
        self.assertIn(li["candidate"]["local_video_path"],
                      {c.local_video_path for c in self.vids})

    def test_resume_skips_finished_results(self):
        self.run_bench()
        self.run_bench()
        self.assertEqual(len(self.engine_calls), 3)
        self.assertEqual(len(self.legacy_calls), 3)
        self.assertEqual(len(self.manifest()["invocations"]), 2)

    def test_changed_fingerprint_refuses_then_versions(self):
        self.run_bench()
        changed = Settings(raw={"clip_engine": {"min_free_gb": 10, "drafts": 12},
                                "clips": {"model": "claude-sonnet-5"}})
        with self.assertRaises(bench.BenchError) as ctx:
            self.run_bench(settings=changed)
        self.assertIn("settings", str(ctx.exception))
        out = self.run_bench(settings=changed, new_version=True)
        manifest = self.manifest()
        tag = manifest["fingerprint"]["hash"][:8]
        self.assertEqual(manifest["arms"], [f"gemini@{tag}", f"transcript@{tag}"])
        self.assertEqual(manifest["retired_arms"], ["gemini", "transcript"])
        self.assertEqual(out["complete"], {f"gemini@{tag}": 3, f"transcript@{tag}": 3})
        self.assertEqual(len(self.engine_calls), 6)
        self.assertEqual(len(manifest["fingerprint_history"]), 1)
        # Resuming the new version works without the flag.
        self.run_bench(settings=changed)
        self.assertEqual(len(self.engine_calls), 6)

    def test_new_version_takes_the_new_cap(self):
        self.run_bench(max_usd=40)
        changed = Settings(raw={"clip_engine": {"min_free_gb": 10, "drafts": 12},
                                "clips": {"model": "claude-sonnet-5"}})
        self.run_bench(settings=changed, new_version=True, max_usd=80)
        self.assertEqual(self.manifest()["max_usd"], 80.0)

    def test_errors_are_retried_unless_kept(self):
        self.engine_error = RuntimeError("quota")
        out = self.run_bench()
        self.assertEqual(out["errors"], {"gemini": 3, "transcript": 0})
        res = self.result("t1", "gemini", "v2")
        self.assertEqual(res["error"], "RuntimeError: quota")
        self.assertEqual(res["cost_usd"], 0.5, "billed-but-failed calls still count")
        self.assertEqual(res["by_stage"], {"draft": 0.5})
        self.run_bench(keep_errors=True)
        self.assertEqual(len(self.engine_calls), 3)
        self.engine_error = None
        out = self.run_bench()
        self.assertEqual(out["complete"], {"gemini": 3, "transcript": 3})
        res = self.result("t1", "gemini", "v2")
        self.assertEqual((res["attempts"], res["prior_cost_usd"]), (2, 0.5))
        self.assertAlmostEqual(out["cost_usd"], 3 * 0.5 + 3 * 0.5 + 3 * 0.05)

    def test_unexpected_failure_is_recorded_and_the_run_goes_on(self):
        real = bench._run_arm

        def flaky(label, arm, video, **kw):
            if video["video_id"] == "v2" and label == "transcript":
                raise OSError("disk full")
            return real(label, arm, video, **kw)

        self.patch(bench, "_run_arm", flaky)
        with self.assertLogs("clip_engine.bench", "ERROR"):
            out = self.run_bench()
        self.assertIn("OSError: disk full", out["skipped"]["v2"]["reason"])
        self.assertEqual(out["complete"], {"gemini": 3, "transcript": 2})
        # v2's engine arm finished (and was paid for) before the failure.
        self.assertAlmostEqual(out["cost_this_run"], 3 * 0.5 + 2 * 0.05)
        self.patch(bench, "_run_arm", real)
        out = self.run_bench()
        self.assertEqual(out["skipped"], {})
        self.assertEqual(out["complete"], {"gemini": 3, "transcript": 3})

    def test_spend_bucket_is_restored_for_the_cli(self):
        self.patch(spend, "current_bucket", lambda: "cli-set")
        self.run_bench()
        self.assertEqual(self.buckets, ["bench", "cli-set"])

    def test_error_notes_are_kept(self):
        class Noted(RuntimeError):
            def __init__(self, message, notes):
                super().__init__(message)
                self.notes = notes

        self.engine_error = Noted("perceiver failed", ["shot log: quota", "fell over"])
        self.run_bench(arms=["gemini"])
        res = self.result("t1", "gemini", "v1")
        self.assertEqual(res["notes"], ["shot log: quota", "fell over"])
        self.assertEqual(res["outcome"], "error")

    def test_deleted_video_is_skipped(self):
        real = self.fake_inputs

        def inputs(candidate_id, **kw):
            out = real(candidate_id, **kw)
            if candidate_id == self.vids[0].id:
                return {**out, "candidate": None, "notes": ["video not found"]}
            return out

        self.patch(jobs, "prepare_inputs", inputs)
        out = self.run_bench()
        self.assertEqual(out["skipped"]["v1"]["reason"], "inputs: video not found")
        self.assertEqual(out["complete"], {"gemini": 2, "transcript": 2})

    def test_input_notes_are_recorded(self):
        real = self.fake_inputs
        self.patch(jobs, "prepare_inputs",
                   lambda cid, **kw: {**real(cid, **kw), "notes": ["taste examples: boom"]})
        self.run_bench(arms=["gemini"])
        self.assertEqual(self.result("t1", "gemini", "v1")["input_notes"], ["taste examples: boom"])

    def test_legacy_failure_keeps_its_cost(self):
        @contextlib.contextmanager
        def metered():
            yield SimpleNamespace(total=0.07, calls=1)

        self.patch(spend, "metered", metered)
        self.patch(legacy, "suggest", mock.Mock(side_effect=RuntimeError("anthropic down")))
        self.run_bench(arms=["transcript"])
        res = self.result("t1", "transcript", "v1")
        self.assertEqual(res["error"], "RuntimeError: anthropic down")
        self.assertEqual(res["cost_usd"], 0.07)

    def test_max_usd_stops_after_a_whole_video(self):
        self.engine_cost = 1.0
        out = self.run_bench(max_usd=1.5)
        stopped = self.manifest()["stopped"]
        self.assertEqual(stopped["reason"], "max_usd")
        self.assertEqual((stopped["videos_done"], stopped["videos_total"]), (2, 3))
        self.assertEqual(out["complete"], {"gemini": 2, "transcript": 2}, "both arms of each video")
        self.assertEqual(len(self.engine_calls), 2)
        # Raising the cap on resume finishes the run and clears the banner.
        out = self.run_bench(max_usd=100)
        self.assertEqual(out["complete"], {"gemini": 3, "transcript": 3})
        self.assertIsNone(self.manifest()["stopped"])
        self.assertEqual(self.manifest()["max_usd"], 100.0)

    def test_restored_sources_are_removed_afterwards(self):
        moved = self.vids[1]
        Path(moved.local_video_path).unlink()
        with self.fake_db() as s:
            s.get(Candidate, moved.id).source_object_key = "climate/v2.mp4"
            s.commit()
        self.run_bench()
        self.assertEqual(self.restores, [Path(moved.local_video_path)])
        self.assertFalse(Path(moved.local_video_path).exists())
        self.assertTrue(Path(self.vids[0].local_video_path).exists(), "never touch local files")
        restored = self.manifest()["restored"]
        self.assertEqual([r["video_id"] for r in restored], ["v2"])

    def test_unavailable_video_is_skipped_then_retried(self):
        self.inputs_fail = {self.vids[1].id}
        out = self.run_bench()
        self.assertIn("v2", out["skipped"])
        self.assertIn("database went away", out["skipped"]["v2"]["reason"])
        self.assertEqual(out["complete"], {"gemini": 2, "transcript": 2})
        self.inputs_fail = set()
        out = self.run_bench()
        self.assertEqual(out["skipped"], {})
        self.assertEqual(out["complete"], {"gemini": 3, "transcript": 3})

    def test_missing_source_skips_the_video(self):
        # Selected (it has a cloud key) but the cloud copy is gone by run time.
        with self.fake_db() as s:
            s.get(Candidate, self.vids[2].id).source_object_key = "climate/gone.mp4"
            s.commit()
        Path(self.vids[2].local_video_path).unlink()
        real = self.fake_ensure_local

        def ensure(local, key):
            if key == "climate/gone.mp4":
                raise offload.SourceUnavailable("its cloud copy is gone")
            return real(local, key)

        self.patch(offload, "ensure_local", ensure)
        out = self.run_bench()
        self.assertIn("source: SourceUnavailable", out["skipped"]["v3"]["reason"])
        self.assertEqual(out["complete"], {"gemini": 2, "transcript": 2})
        self.assertIsNone(out["stopped"])

    def test_source_missing_everywhere_is_never_selected(self):
        Path(self.vids[2].local_video_path).unlink()
        self.run_bench()
        videos = json.loads((bench.bench_dir() / "t1" / "videos.json").read_text())
        self.assertNotIn("v3", {v["video_id"] for v in videos})

    def test_low_disk_stops_before_restoring(self):
        with self.fake_db() as s:
            for c in self.vids:
                s.get(Candidate, c.id).source_object_key = f"climate/{c.video_id}.mp4"
            s.commit()
        for c in self.vids:
            Path(c.local_video_path).unlink()
        self.free_gb = 2.0
        out = self.run_bench()
        self.assertEqual(out["stopped"]["reason"], "low_disk")
        self.assertEqual(self.restores, [])
        self.assertEqual(self.engine_calls, [])

    def test_test_seed_excludes_touched_videos_dev_does_not(self):
        bench.touch_videos(["v1"], source="try")
        self.run_bench("holdout", seed="test")
        videos = json.loads((bench.bench_dir() / "holdout" / "videos.json").read_text())
        self.assertNotIn("v1", {v["video_id"] for v in videos})
        self.run_bench("dev1", seed="dev")
        videos = json.loads((bench.bench_dir() / "dev1" / "videos.json").read_text())
        self.assertIn("v1", {v["video_id"] for v in videos})
        self.assertEqual(self.manifest("dev1")["excluded_touched"], 0)

    def test_nothing_eligible_creates_nothing(self):
        bench.touch_videos(["v1", "v2", "v3"], source="try")
        with self.assertRaises(bench.BenchError):
            self.run_bench("empty")
        self.assertFalse((bench.bench_dir() / "empty" / "manifest.json").exists())

    def test_hand_picked_run(self):
        self.run_bench("picked", candidate_ids=[self.vids[0].id])
        manifest = self.manifest("picked")
        self.assertEqual(manifest["mode"], "hand-picked")
        self.assertEqual(len(self.engine_calls), 1)
        self.assertIn("DEV RUN", bench.report("picked"))

    def test_parallel_workers(self):
        for i in range(5):    # repeated: thread interleavings vary run to run
            out = self.run_bench(f"par{i}", workers=3, seed="dev")
            self.assertEqual(out["skipped"], {})
            self.assertEqual(out["complete"], {"gemini": 3, "transcript": 3})

    def test_parallel_workers_respect_the_cap(self):
        self.engine_cost = 1.0
        out = self.run_bench(workers=2, max_usd=1.0)
        self.assertEqual(out["stopped"]["reason"], "max_usd")
        # Two videos may be in flight when the cap is hit, never all three.
        self.assertLessEqual(len(self.engine_calls), 2)

    def test_caps_and_limits_must_be_positive(self):
        cases = [{"max_usd": 0}, {"max_usd": -5}, {"max_usd": "lots"}, {"max_usd": float("nan")},
                 {"limit": 0}, {"workers": -1}]
        for kw in cases:
            with self.subTest(kw=kw):
                with self.assertRaises(bench.BenchError):
                    self.run_bench("caps", **kw)
        self.assertFalse((bench.bench_dir() / "caps" / "manifest.json").exists())
        self.assertEqual(self.engine_calls, [])
        self.run_bench()
        with self.assertRaises(bench.BenchError):
            self.run_bench(max_usd=0)    # a resume can't lift the cap off either

    def test_ledger_is_the_earliest_pre_cut_legacy_run(self):
        c = self.vids[0]
        first_cut_at = ago(20)
        rows = [
            ("r-early", "claude-sonnet-5", ago(29), [S(10, 30)]),
            ("r-early", "claude-sonnet-5", ago(29), [S(60, 80)]),
            ("r-engine", "engine:gemini:gemini-3.8-flash", ago(29.5), [S(0, 5)]),
            ("r-late", "claude-sonnet-5", ago(10), [S(90, 99)]),
        ]
        for i, (run_id, model, created, segs) in enumerate(rows):
            self.add(ClipProposal(candidate_pk=c.id, run_id=run_id, model=model, clip_index=i % 2,
                                  proposed_segments=json.dumps(segs), created_at=created,
                                  policy_version="pol-v2"))
        self.commit()
        self.assertLess(ago(29), first_cut_at)
        self.run_bench()
        ledger = json.loads((bench.bench_dir() / "t1" / "ledger.json").read_text())
        self.assertEqual(ledger["label"], bench.LEDGER_LABEL)
        entry = ledger["runs"]["v1"]
        self.assertEqual(entry["run_id"], "r-early")
        self.assertEqual([cl["segments"] for cl in entry["clips"]], [[S(10, 30)], [S(60, 80)]])
        self.assertEqual(entry["policy_version"], "pol-v2")
        self.assertNotIn("v2", ledger["runs"])

    def test_generated_playbook_newer_than_the_videos_is_flagged(self):
        self.patch(playbook, "load", lambda: {"archetypes": [], "flops": [], "notes": "",
                                              "generated_at": NOW.isoformat(),
                                              "as_of": NOW.isoformat()})
        self.run_bench()
        manifest = self.manifest()
        self.assertEqual(manifest["playbook"]["source"], "generated")
        self.assertTrue(any("clip_playbook.yaml" in w for w in manifest["warnings"]))

    def test_a_reveal_while_a_resume_runs_stays_revealed(self):
        real = self.fail_engine_on("v2")
        self.run_bench()

        def render(source, segments, dest, **kw):
            Path(dest).write_bytes(b"render")
            return Path(dest)

        self.patch(media, "render_segments", render)
        self.assertEqual(bench.prepare_pairs("t1"), 2)
        item = bench.next_item("t1")
        revealed = {}

        def reveal_midway(**kw):
            # The operator presses "Reveal results" on the page (another
            # process) while the resume retries v2.
            if not revealed:
                revealed.update(bench.reveal("t1"))
            return real(**kw)

        self.patch(engine, "run", reveal_midway)
        out = self.run_bench()
        self.assertEqual(out["complete"]["gemini"], 3, "the resume did its work")
        manifest = self.manifest()
        self.assertEqual(manifest["revealed_at"], revealed["revealed_at"])
        self.assertEqual(len(manifest["invocations"]), 2)
        self.assertIsNone(bench.next_item("t1"))
        with self.assertRaises(bench.BenchError):
            bench.judge("t1", item["item_id"], "a")
        self.assertTrue(bench.results("t1")["revealed"])
        with self.assertRaises(bench.BenchError):
            bench.prepare_pairs("t1")

    def test_entries_written_by_others_while_the_run_goes_survive_it(self):
        real = engine.run
        entry = {"video_id": "v9", "path": "/elsewhere/v9.mp4",
                 "removed_at": "2026-11-18T00:00:00Z"}

        def pairs_meanwhile(**kw):
            # prepare_pairs in another process records a source it restored.
            if kw["video_id"] == "v1":
                path = bench.bench_dir() / "t1" / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest.setdefault("restored", []).append(entry)
                util.atomic_write_json(path, manifest)
            return real(**kw)

        self.patch(engine, "run", pairs_meanwhile)
        moved = self.vids[1]
        Path(moved.local_video_path).unlink()
        with self.fake_db() as s:
            s.get(Candidate, moved.id).source_object_key = "climate/v2.mp4"
            s.commit()
        self.run_bench()
        restored = self.manifest()["restored"]
        self.assertIn(entry, restored)
        self.assertEqual(sorted(r["video_id"] for r in restored), ["v2", "v9"],
                         "the run's own restored source is recorded too")

    def test_a_second_process_on_the_same_run_refuses_at_once(self):
        real = engine.run
        busy, release = threading.Event(), threading.Event()

        def slow(**kw):
            busy.set()
            release.wait(2)
            return real(**kw)

        self.patch(engine, "run", slow)
        first: dict = {}
        worker = threading.Thread(target=lambda: first.update(self.run_bench()))
        worker.start()
        try:
            self.assertTrue(busy.wait(10))
            with self.assertRaises(bench.BenchError) as ctx:
                self.run_bench()
            self.assertIn("already running", str(ctx.exception))
        finally:
            release.set()
            worker.join(30)
        self.assertEqual(first["complete"], {"gemini": 3, "transcript": 3})
        self.assertEqual(sorted(c["video_id"] for c in self.engine_calls), ["v1", "v2", "v3"])
        self.assertEqual(len(self.legacy_calls), 3)
        self.run_bench()   # the lock is released once the run is done
        self.assertEqual(len(self.engine_calls), 3)

    def test_new_version_warns_that_it_starts_over_against_the_same_cap(self):
        self.engine_cost = 1.0
        out = self.run_bench(max_usd=1.5)
        self.assertEqual(out["stopped"]["reason"], "max_usd")
        changed = Settings(raw={"clip_engine": {"min_free_gb": 10, "drafts": 12},
                                "clips": {"model": "claude-sonnet-5"}})
        self.said.clear()
        before = len(self.engine_calls)
        self.run_bench(settings=changed, new_version=True, max_usd=1.5)
        self.assertEqual(len(self.engine_calls), before, "the old version's spend fills the cap")
        text = "\n".join(self.said)
        self.assertIn("re-runs every arm on all 3 videos", text)
        self.assertIn("judged again", text)
        self.assertIn("already spent $2.10 of its $1.50 cap", text)
        self.assertIn("Nothing will run until you raise --max-usd", text)
        # With room under the cap, it says how much more the new version needs.
        self.said.clear()
        changed.raw["clip_engine"]["drafts"] = 14
        self.run_bench(settings=changed, new_version=True, max_usd=3.0)
        text = "\n".join(self.said)
        self.assertIn("estimated at about $1.80", text)   # 3 videos × ($0.50 + $0.10)
        self.assertIn("Raise --max-usd to about $3.90", text)


class ArmInputsTest(_RunFixture):
    """The inputs are read once per video with the run's settings; each arm's
    own settings decide how much of them it gets."""

    ARMS = ["gemini", "gemini+taste_hits=1+taste_flops=0+job_timeout_seconds=5",
            "gemini+clips.context_examples=1", "transcript+clips.learn_from_history=false"]

    def setUp(self):
        super().setUp()
        other = self.video("other", status="approved")      # never selected
        flop_cut = self.cut(other, [S(0, 10)])
        self.add(ContentLabel(cut_pk=flop_cut.id, label="gripping", value="no",
                              created_at=ago(90), updated_at=ago(90)))
        self.commit()
        examples = [{"post_pk": 11, "label": "hit", "title": "hit one"},
                    {"post_pk": 13, "label": "hit", "title": "hit two"},
                    {"post_pk": 12, "label": "flop", "cut_pk": flop_cut.id, "title": "flop"}]
        self.patch(taste, "select_examples", lambda session, **kw: [dict(e) for e in examples])
        real_inputs = self.fake_inputs
        self.context = {"shipped_examples": [{"seconds": 10}, {"seconds": 20}],
                        "recently_posted": [{"story": "x"}]}
        self.patch(jobs, "prepare_inputs", lambda cid, **kw: {
            **real_inputs(cid, **kw), "context": dict(self.context), "examples": []})
        real_run = engine.run
        self.left: dict = {}

        def timed(**kw):
            self.left.setdefault(self.arm_of(kw), kw["deadline"] - time.monotonic())
            return real_run(**kw)

        self.patch(engine, "run", timed)

    @staticmethod
    def arm_of(call) -> str:
        settings = call["settings"]
        if settings.get("clip_engine.taste_hits") is not None:
            return ArmInputsTest.ARMS[1]
        if settings.get("clips.context_examples") is not None:
            return ArmInputsTest.ARMS[2]
        return "gemini"

    def test_each_arm_gets_what_its_settings_say(self):
        self.run_bench(arms=self.ARMS)
        by_arm: dict = {}
        for call in self.engine_calls:
            by_arm.setdefault(self.arm_of(call), []).append(call)
        self.assertEqual({k: len(v) for k, v in by_arm.items()},
                         {"gemini": 3, self.ARMS[1]: 3, self.ARMS[2]: 3})
        for call in by_arm["gemini"]:
            self.assertEqual([e["post_pk"] for e in call["examples"]], [11, 13, 12])
            self.assertEqual(call["context"], self.context)
        for call in by_arm[self.ARMS[1]]:
            self.assertEqual([e["post_pk"] for e in call["examples"]], [11], "1 hit, no flops")
            self.assertEqual(call["context"], self.context)
        for call in by_arm[self.ARMS[2]]:
            self.assertEqual(call["context"], {"shipped_examples": [{"seconds": 10}],
                                               "recently_posted": [{"story": "x"}]})
            self.assertEqual(len(call["examples"]), 3)
        self.assertLess(self.left[self.ARMS[1]], 10, "its own job_timeout_seconds")
        self.assertGreater(self.left["gemini"], 1000)
        self.assertEqual(len(self.legacy_calls), 3)
        self.assertTrue(all(c["context"] == {} for c in self.legacy_calls),
                        "learn_from_history=false: no account context")
        # The records say what each arm was actually given.
        self.assertEqual(self.result("t1", self.ARMS[1], "v1")["examples"], [11])
        self.assertEqual(self.result("t1", "gemini", "v1")["context_keys"],
                         ["recently_posted", "shipped_examples"])
        transcript = self.result("t1", self.ARMS[3], "v1")
        self.assertEqual((transcript["examples"], transcript["context_keys"]), ([], []))

    def test_an_arm_cannot_ask_for_more_than_the_run_holds(self):
        for arms in (["gemini+taste_hits=4"], ["gemini+taste_flops=3"],
                     ["gemini+clips.context_examples=7"]):
            with self.subTest(arms=arms):
                with self.assertRaises(bench.BenchError) as ctx:
                    self.run_bench("more", arms=arms)
                self.assertIn("only show fewer", str(ctx.exception))
        no_history = Settings(raw={**self.settings.raw, "clips": {
            "model": "claude-sonnet-5", "learn_from_history": False}})
        with self.assertRaises(bench.BenchError):
            self.run_bench("more", arms=["transcript+clips.learn_from_history=true"],
                           settings=no_history)
        self.assertFalse((bench.bench_dir() / "more" / "manifest.json").exists())
        self.assertEqual(self.engine_calls, [])
        # A resume can't add one either: the run froze 3 hits per video.
        self.run_bench(arms=["gemini"])
        self.assertEqual(self.manifest()["taste"], {"hits": 3, "flops": 2, "lookback_days": 180})
        with self.assertRaises(bench.BenchError):
            self.run_bench(arms=["gemini+taste_hits=5"])
        self.assertEqual(self.manifest()["arms"], ["gemini"])

    def test_an_arm_the_bench_now_refuses_does_not_keep_running(self):
        self.run_bench(arms=["gemini"])
        manifest = self.manifest()    # an arm an earlier bench accepted, then ignored
        manifest["arms"].append("gemini+taste_lookback_days=30")
        manifest["arm_specs"]["gemini+taste_lookback_days=30"] = {
            "spec": "gemini+taste_lookback_days=30", "setup": "gemini",
            "overrides": {"clip_engine.taste_lookback_days": 30}}
        util.atomic_write_json(bench.bench_dir() / "t1" / "manifest.json", manifest)
        with self.assertRaises(bench.BenchError) as ctx:
            self.run_bench(arms=None)
        self.assertIn("can't run as labelled", str(ctx.exception))
        self.assertEqual(len(self.engine_calls), 3)


class TasteAsOfTest(_RunFixture):
    def test_flops_labelled_after_the_archive_time_are_left_out(self):
        other = self.video("other", status="approved")
        stamps = {"early": (ago(90), ago(90)), "late": (ago(1), ago(1)),
                  "relabelled": (ago(90), ago(2))}    # a relabel overwrites the row
        cuts = {}
        for i, (name, (created, updated)) in enumerate(stamps.items()):
            cuts[name] = self.cut(other, [S(i * 10, i * 10 + 5)])
            self.add(ContentLabel(cut_pk=cuts[name].id, label="gripping", value="no",
                                  created_at=created, updated_at=updated))
        self.commit()
        examples = [{"post_pk": 11, "label": "hit"},
                    {"post_pk": 21, "label": "flop", "cut_pk": cuts["late"].id},
                    {"post_pk": 22, "label": "flop", "cut_pk": cuts["early"].id},
                    {"post_pk": 23, "label": "flop", "cut_pk": cuts["relabelled"].id},
                    {"post_pk": 24, "label": "flop", "cut_pk": None}]    # can't be dated
        self.patch(taste, "select_examples", lambda session, **kw: [dict(e) for e in examples])
        self.run_bench(arms=["gemini"])
        videos = json.loads((bench.bench_dir() / "t1" / "videos.json").read_text())
        self.assertEqual([v["taste_post_pks"] for v in videos], [[11, 22]] * 3)
        self.assertTrue(any("after the video was archived" in n for n in self.manifest()["notes"]))
        self.assertTrue(all([e["post_pk"] for e in c["examples"]] == [11, 22]
                            for c in self.engine_calls))


class CliResumeTest(_RunFixture):
    """run.py's `clipbench run` on the real bench.run (engine mocked)."""

    def setUp(self):
        super().setUp()
        settings = Settings(raw={**self.settings.raw, "clip_engine": {
            **self.settings.raw["clip_engine"],
            "bench": {"arms": ["gemini", "transcript"], "limit": 60, "max_usd": 40}}})
        for target, value in (("app.config.load_settings", lambda: settings),
                              ("app.db.init_db", lambda *a, **k: None)):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def cli(self, argv) -> str:
        import run as run_cli

        args = run_cli.build_parser().parse_args(argv)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            args.func(args)
        return out.getvalue()

    def test_a_bare_resume_keeps_the_runs_own_arms_and_cap(self):
        real = self.fail_engine_on("v2")
        self.cli(["clipbench", "run", "--name", "t1", "--arms",
                  "gemini+editor_thinking=HIGH,transcript", "--max-usd", "10"])
        self.patch(engine, "run", real)
        before = len(self.engine_calls)
        out = self.cli(["clipbench", "run", "--name", "t1"])     # "run the same name again"
        manifest = self.manifest()
        self.assertEqual(manifest["arms"], ["gemini+editor_thinking=HIGH", "transcript"])
        self.assertEqual(manifest["max_usd"], 10.0)
        resumed = self.engine_calls[before:]
        self.assertEqual([c["video_id"] for c in resumed], ["v2"], "only the errored video")
        self.assertTrue(all(c["settings"].get("clip_engine.gemini.editor_thinking") == "HIGH"
                            for c in resumed))
        self.assertIn("Resuming bench run 't1'", out)
        self.assertIn("stops once results pass $10.00", out)
        self.assertIn("max_usd: 10.0", out)


class FrozenExamplesTest(unittest.TestCase):
    def test_fresh_where_frozen_else_thawed_frozen_copy(self):
        video = {"taste_post_pks": [11, 12, 13],
                 "taste_examples": [{"post_pk": 11, "title": "old",
                                     "published_at": "2026-10-01T00:00:00Z"},
                                    {"post_pk": 12, "title": "frozen"}]}
        fresh = [{"post_pk": 12, "title": "fresh"}, {"post_pk": 99, "title": "not frozen"}]
        out = bench._frozen_examples(video, fresh)
        self.assertEqual([e["title"] for e in out], ["old", "fresh"])
        self.assertEqual(out[0]["published_at"], dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(video["taste_examples"][0]["published_at"], "2026-10-01T00:00:00Z")
        self.assertEqual(bench._frozen_examples({}, fresh), fresh, "runs frozen without examples")
        self.assertEqual(bench._frozen_examples({"taste_post_pks": []}, fresh), [])


class LegacyArmTest(_Env):
    def test_shape_and_replay_inputs(self):
        seen = {}

        def fake_suggest(**kw):
            seen.update(kw)
            return {"clips": [{"segments": [S(5, 25)], "story": "s"}, {"segments": []}],
                    "model": "m", "policy": "p", "cost_usd": 0.04}

        self.patch(legacy, "suggest", fake_suggest)
        inputs = {"candidate": {"video_id": "v", "local_video_path": "/orig.mp4"},
                  "used": [S(0, 3)], "striking": [{"t": 1}], "context": {"x": 1},
                  "other_clips": [{"title": "later clip", "segments": [S(0, 3)]}]}
        out = bench.legacy_arm(video={"video_id": "v", "candidate_pk": 7}, source=Path("/link.mp4"),
                               settings=Settings(raw={}), inputs=inputs)
        self.assertEqual(seen["inputs"]["used"], [])
        self.assertEqual(seen["inputs"]["other_clips"], [])
        self.assertEqual(seen["inputs"]["striking"], [])
        self.assertEqual(seen["inputs"]["candidate"]["local_video_path"], "/link.mp4")
        self.assertEqual(seen["inputs"]["source"], "/link.mp4")
        self.assertEqual(inputs["used"], [S(0, 3)], "caller's inputs untouched")
        self.assertEqual(seen["context"], {"x": 1})
        self.assertEqual(len(out["top"]), 1)
        self.assertIsNone(out["top"][0]["score"])
        self.assertEqual((out["cost_usd"], out["outcome"], out["candidate_pk"]), (0.04, "proposals", 7))

    def test_empty_answer(self):
        self.patch(legacy, "suggest", lambda **kw: {"clips": [], "cost_usd": 0.01})
        out = bench.legacy_arm(video={"video_id": "v", "candidate_pk": 1}, source="/x",
                               settings=Settings(raw={}), inputs={})
        self.assertEqual((out["outcome"], out["top"]), ("empty", []))


# ---- evaluate ----------------------------------------------------------------------------

class EvaluateTest(_Env):
    def setUp(self):
        super().setUp()
        drafts = [{"id": "d1", "segments": [S(11, 31)]}, {"id": "d2", "segments": [S(80, 90)]},
                  {"id": "d3", "segments": [S(40, 45)]}]
        orders = {"editor": ["d2", "d1", "d3"], "cold": ["d1", "d2"], "rank": ["d1"],
                  "final": ["d1", "d2"]}
        self.videos = [
            ventry("v1", [target(1, [S(10, 30)])]),
            ventry("v2", [target(2, [S(50, 70)], seeded=True)]),
            ventry("v3", [target(3, [S(0, 20)]), target(4, [S(60, 80)])], bucket="300-1200s",
                   kind="pasted"),
            ventry("v4", [target(5, [S(40, 60)])]),
            ventry("n1", [], negative=True),
        ]
        self.results = {
            "gemini": {
                "v1": result([[S(11, 31)], [S(80, 90)]], cost=0.5, drafts=drafts, orders=orders,
                             by_stage={"draft": 0.5, "perceive_cached": 0.1}),
                "v2": result([], error="EngineError: boom", cost=0.3),
                "v3": result([[S(62, 66)]], cost=0.4),
                "n1": result([[S(5, 15)]], cost=0.2),
            },
            "transcript": {
                "v1": result([[S(40, 50)], [S(10, 30)]], cost=0.05),
                "v2": result([[S(50, 70)]], cost=0.05),
                "v3": result([], cost=0.05),
                "v4": result([[S(40, 60)]], cost=0.05),
            },
        }
        write_run("e1", videos=self.videos, results=self.results,
                  ledger={"v1": {"clips": [{"segments": [S(10, 30)]}], "policy_version": "p2"}},
                  invocations=[{"results_cost": 1.5, "ledger_delta": 1.55}])
        self.ev = bench.evaluate("e1")

    def test_common_set_and_attempts(self):
        self.assertEqual((self.ev["videos"], self.ev["negatives"], self.ev["common"]), (4, 1, 3))
        self.assertEqual(self.ev["attempted"], {"gemini": 3, "transcript": 4})
        self.assertEqual(self.ev["transcript_arm"], "transcript")
        self.assertEqual(self.ev["engine_arms"], ["gemini"])

    def test_hits_on_all_targets(self):
        view = self.ev["views"]["all"]
        self.assertEqual(view["n_videos"], 3)
        g, t = view["arms"]["gemini"], view["arms"]["transcript"]
        cases = [
            (g, "hit1", 1 / 3), (g, "hit3", 1 / 3), (g, "moment1", 2 / 3), (g, "moment3", 2 / 3),
            (g, "error_rate", 1 / 3), (g, "empty_rate", 0.0), (g, "recall3", 1 / 3),
            (g, "open1", 1.0), (g, "props_per_video", 1.0),
            (t, "hit1", 1 / 3), (t, "hit3", 2 / 3), (t, "empty_rate", 1 / 3), (t, "open3", 1.0),
            (t, "error_rate", 0.0),
        ]
        for agg, key, expected in cases:
            with self.subTest(key=key):
                self.assertAlmostEqual(agg[key], expected, places=4)
        self.assertEqual((g["n"], g["errors"], g["open1_n"]), (3, 1, 1))

    def test_chance_and_lift(self):
        g = self.ev["views"]["all"]["arms"]["gemini"]
        expected = (bench.chance_hit(20, [[S(10, 30)]], 100) + 0 + 0) / 3
        self.assertAlmostEqual(g["chance1"], expected, places=6)
        self.assertAlmostEqual(g["lift1"], 1 / 3 - expected, places=6)

    def test_unseeded_view_is_the_headline(self):
        view = self.ev["views"]["unseeded"]
        self.assertEqual(view["n_videos"], 2, "v2's only target was seeded")
        self.assertAlmostEqual(view["arms"]["gemini"]["hit1"], 0.5)
        self.assertAlmostEqual(view["arms"]["transcript"]["hit1"], 0.0)

    def test_engine_orders(self):
        orders = self.ev["views"]["all"]["orders"]["gemini"]
        self.assertAlmostEqual(orders["editor"]["hit1"], 0.0)
        self.assertAlmostEqual(orders["editor"]["hit3"], 1 / 3)
        self.assertAlmostEqual(orders["cold"]["hit1"], 1 / 3)
        self.assertEqual(orders["final"]["n"], 3)

    def test_fallback_row_substitutes_the_transcript_result(self):
        row = self.ev["views"]["all"]["fallback"]["gemini+fallback"]
        self.assertAlmostEqual(row["hit1"], 2 / 3)
        self.assertAlmostEqual(row["error_rate"], 0.0)

    def test_ledger_is_reported_apart(self):
        ledger = self.ev["views"]["all"]["ledger"]
        self.assertEqual(ledger["n"], 1)
        self.assertEqual(ledger["hit1"], 1.0)
        self.assertIsNone(ledger["open1"], "ledger rows are stored sorted: no opening")
        self.assertEqual(self.ev["views"]["all"]["ledger_by_policy"], {"p2": 1})

    def test_strata(self):
        strata = self.ev["views"]["all"]["arms"]["gemini"]["strata"]
        self.assertEqual(strata["length_bucket"]["<120s"]["n"], 2)
        self.assertEqual(strata["length_bucket"]["300-1200s"]["n"], 1)
        self.assertEqual(strata["kind"]["pasted"]["n"], 1)
        self.assertAlmostEqual(strata["length_bucket"]["<120s"]["hit1"], 0.5)

    def test_costs(self):
        g = self.ev["views"]["all"]["arms"]["gemini"]
        self.assertAlmostEqual(g["cost_per_video"], (0.5 + 0.3 + 0.4) / 3)
        self.assertAlmostEqual(g["cost_full_per_video"], (0.6 + 0.3 + 0.4) / 3)
        self.assertAlmostEqual(g["seconds_per_video"], 30.0)
        self.assertEqual(self.ev["reconciliation"],
                         {"results_cost": 1.5, "ledger_delta": 1.55, "invocations": 1})
        self.assertAlmostEqual(self.ev["spent"]["gemini"], 1.4)
        self.assertEqual(self.ev["views"]["all"]["operator_median_seconds"], 20.0)

    def test_intro_pricing(self):
        res = {"cost_usd": 1.0, "tokens": {"gemini-3.8-flash": {"input": 100_000, "output": 10_000}}}
        with mock.patch.object(spend, "estimate_cost", return_value=0.4):
            with mock.patch.object(bench, "_INTRO_PRICING",
                                   {"gemini-3.8-flash": ("2999-12-31", 0.5)}):
                self.assertAlmostEqual(bench._cost_now(res), 0.8)
            with mock.patch.object(bench, "_INTRO_PRICING",
                                   {"gemini-3.8-flash": ("2000-01-01", 0.5)}):
                self.assertAlmostEqual(bench._cost_now(res), 1.0)

    def test_no_results_yet(self):
        write_run("blank", videos=self.videos, results={})
        ev = bench.evaluate("blank")
        self.assertEqual(ev["common"], 0)
        self.assertIsNone(ev["views"]["all"]["arms"]["gemini"]["hit1"])
        self.assertIn("Results stay hidden", bench.report("blank"))

    def test_unknown_run(self):
        with self.assertRaises(bench.BenchError):
            bench.evaluate("nope")
        with self.assertRaises(bench.BenchError):
            bench.evaluate("../escape")


# ---- pairs and judging ----------------------------------------------------------------------

class _PairsFixture(_Env):
    NAME = "p1"

    def setUp(self):
        super().setUp()
        self.media_dir = self.tmp / "media"
        self.media_dir.mkdir()
        self.videos = []
        for vid in ("v1", "v2", "v3", "v4", "n1"):
            src = self.media_dir / f"{vid}.mp4"
            src.write_bytes(b"src")
            self.videos.append({**ventry(vid, [], negative=vid == "n1"), "local_video_path": str(src)})
        targets = {"v1": [target(1, [S(10, 30)], views=100)],
                   "v2": [target(2, [S(50, 70)])],
                   "v3": [target(3, [S(60, 70), S(70, 70.2), S(75, 80)], views=5),
                          target(33, [S(0, 10)], published=False, views=None)],
                   "v4": [target(4, [S(0, 10)])]}
        for v in self.videos:
            v["targets"] = targets.get(v["video_id"], [])
        self.results = {
            "gemini": {"v1": result([[S(11, 31)]]), "v2": result([[S(0, 20)]]),
                       "v3": result([[S(61, 79)]]), "v4": result([], error="EngineError: x"),
                       "n1": result([[S(5, 15)]])},
            "transcript": {"v1": result([[S(40, 60)]]), "v2": result([[S(0, 20.5)]]),
                           "v3": result([]), "v4": result([[S(0, 10)]])},
        }
        write_run(self.NAME, videos=self.videos, results=self.results)
        self.renders: list = []
        self.fail_on: set = set()

        def fake_render(source, segments, dest, *, height=360, crf=28, fade=0.03):
            self.renders.append({"source": Path(source), "segments": segments, "dest": Path(dest),
                                 "height": height})
            if any(s["start"] in self.fail_on for s in segments):
                raise media.MediaError("ffmpeg exploded")
            Path(dest).write_bytes(b"render")
            return Path(dest)

        self.links: list = []
        self.released: list = []
        self.free_gb = 500.0
        self.keys: dict = {}

        def fake_link(source, key):
            link = self.media_dir / f"link-{key}.mp4"
            link.write_bytes(Path(source).read_bytes())
            self.links.append(link)
            return link

        def fake_ensure(local, key):
            path = Path(local)
            if not path.is_file():
                path.write_bytes(b"restored")
            return path

        self.patch(media, "render_segments", fake_render)
        self.patch(media, "link_inflight", fake_link)
        self.patch(media, "release_inflight", lambda path: self.released.append(Path(path)))
        self.patch(media, "disk_free_gb", lambda path: self.free_gb)
        self.patch(offload, "ensure_local", fake_ensure)
        self.patch(bench, "_source_fields", lambda videos: {
            v["video_id"]: (v["local_video_path"], self.keys.get(v["video_id"], ""))
            for v in videos})

    def items(self):
        data = json.loads((bench.bench_dir() / self.NAME / "pairs.json").read_text())
        return {(it["kind"], it["video_id"]): it for it in data["items"]}, data


class PairsTest(_PairsFixture):
    def test_kinds_auto_ties_and_renders(self):
        waiting = bench.prepare_pairs(self.NAME, height=480)
        items, data = self.items()
        self.assertEqual(set(items), {("arms", "v1"), ("ceiling", "v1"), ("arms", "v2"),
                                      ("ceiling", "v2"), ("ceiling", "v3"), ("solo", "n1")})
        # IoU ≥ 0.9: scored a tie, never shown.
        self.assertEqual(items[("ceiling", "v1")]["auto"], "tie")
        self.assertEqual(items[("arms", "v2")]["auto"], "tie")
        self.assertIsNone(items[("arms", "v1")]["auto"])
        self.assertEqual(waiting, 4)
        self.assertEqual(data["judgeable"], 4)
        self.assertEqual(data["no_top1"], {"gemini": ["v4"]})
        self.assertEqual(data["kinds"], {"arms": 2, "ceiling": 3, "solo": 1})
        self.assertEqual(len(self.renders), 2 + 2 + 2 + 1)
        self.assertTrue(all(r["height"] == 480 for r in self.renders))
        for item in items.values():
            for side in item["sides"].values():
                exists = (bench.bench_dir() / self.NAME / side["render"]).is_file()
                self.assertEqual(exists, not item["auto"])

    def test_ids_sides_and_reference(self):
        bench.prepare_pairs(self.NAME)
        items, _ = self.items()
        item = items[("arms", "v1")]
        expected_id = hashlib.sha1(f"{self.NAME}|arms|v1|gemini".encode()).hexdigest()[:12]
        self.assertEqual(item["item_id"], expected_id)
        on_b = int(hashlib.sha1((self.NAME + expected_id).encode()).hexdigest(), 16) % 2 == 1
        arm_side, other_side = ("b", "a") if on_b else ("a", "b")
        self.assertEqual(item["sides"][arm_side]["source"], "arm")
        self.assertEqual(item["sides"][arm_side]["segments"], [S(11, 31)])
        self.assertEqual(item["sides"][other_side]["source"], "transcript")
        # The reference is the published, most-viewed target; slivers dropped.
        ceiling = items[("ceiling", "v3")]
        ref = next(s for s in ceiling["sides"].values() if s["source"] == "reference")
        self.assertEqual(ref["ref"], 3)
        self.assertEqual(ref["segments"], [S(60, 70), S(75, 80)])
        self.assertEqual(ref["dropped_slivers"], 1)
        solo = items[("solo", "n1")]
        self.assertEqual(list(solo["sides"]), ["a"])

    def test_render_failures_are_counted_not_judged(self):
        self.fail_on = {61.0}
        waiting = bench.prepare_pairs(self.NAME)
        items, data = self.items()
        self.assertEqual(waiting, 3)
        self.assertEqual(data["render_failures"], 1)
        self.assertIn("ffmpeg exploded", items[("ceiling", "v3")]["render_error"])

    def test_idempotent_and_requeued_when_results_change(self):
        bench.prepare_pairs(self.NAME)
        first = len(self.renders)
        bench.prepare_pairs(self.NAME)
        self.assertEqual(len(self.renders), first, "unchanged items keep their renders")
        nxt = bench.next_item(self.NAME)
        bench.judge(self.NAME, nxt["item_id"], "a" if nxt["kind"] == "pair" else "yes")
        self.assertEqual(bench.next_item(self.NAME)["progress"]["judged"], 1)
        # Re-run the arm on every video: every judgment goes stale.
        for vid, res in self.results["gemini"].items():
            if res["top"]:
                res["top"][0]["segments"] = [S(s["start"] + 1, s["end"] + 1)
                                             for s in res["top"][0]["segments"]]
            util.atomic_write_json(bench.bench_dir() / self.NAME / "gemini" / f"{vid}.json", res)
        bench.prepare_pairs(self.NAME)
        _, data = self.items()
        self.assertGreater(len(self.renders), first)
        self.assertGreaterEqual(data["stale_items"], 1)
        self.assertEqual(bench.next_item(self.NAME)["progress"]["judged"], 0)

    def test_renders_run_on_the_in_flight_link(self):
        bench.prepare_pairs(self.NAME)
        self.assertTrue(all(r["source"] in self.links for r in self.renders))
        self.assertEqual(sorted(self.released), sorted(self.links))

    def test_a_new_height_rerenders_but_keeps_judgments(self):
        bench.prepare_pairs(self.NAME, height=540)
        nxt = bench.next_item(self.NAME)
        bench.judge(self.NAME, nxt["item_id"], "tie" if nxt["kind"] == "pair" else "yes")
        first = len(self.renders)
        bench.prepare_pairs(self.NAME, height=720)
        self.assertEqual(len(self.renders), 2 * first)
        self.assertTrue(all(r["height"] == 720 for r in self.renders[first:]))
        self.assertEqual(bench.next_item(self.NAME)["progress"]["judged"], 1)

    def test_restored_sources_are_removed_and_recorded(self):
        source = Path(self.videos[0]["local_video_path"])
        source.unlink()
        self.keys["v1"] = "climate/v1.mp4"
        bench.prepare_pairs(self.NAME)
        self.assertFalse(source.exists())
        manifest = json.loads((bench.bench_dir() / self.NAME / "manifest.json").read_text())
        self.assertEqual([r["video_id"] for r in manifest["restored"]], ["v1"])

    def test_low_disk_marks_what_wasnt_rendered(self):
        for v in self.videos:
            Path(v["local_video_path"]).unlink()
            self.keys[v["video_id"]] = f"climate/{v['video_id']}.mp4"
        self.free_gb = 1.0
        self.assertEqual(bench.prepare_pairs(self.NAME), 0)
        items, data = self.items()
        self.assertEqual(self.renders, [])
        self.assertEqual(data["render_failures"], 4)
        self.assertTrue(all("not rendered" in it["render_error"]
                            for it in items.values() if not it["auto"]))
        self.free_gb = 500.0
        self.assertEqual(bench.prepare_pairs(self.NAME), 4, "retried once there's room")

    def test_revealed_runs_take_no_new_pairs(self):
        bench.prepare_pairs(self.NAME)
        bench.reveal(self.NAME)
        with self.assertRaises(bench.BenchError):
            bench.prepare_pairs(self.NAME)


class JudgingTest(_PairsFixture):
    def setUp(self):
        super().setUp()
        bench.prepare_pairs(self.NAME)

    def test_next_item_reveals_nothing_about_the_arms(self):
        item = bench.next_item(self.NAME)
        self.assertEqual(set(item), {"item_id", "kind", "mode", "sides", "choices", "reasons",
                                     "media", "progress"})
        self.assertIn(item["kind"], ("pair", "solo"))
        self.assertEqual(item["media"]["a"], f"/clip-bench/{self.NAME}/media/{item['item_id']}/a")
        self.assertEqual(item["progress"], {"judged": 0, "total": 4, "remaining": 4})

    def test_judging_order_spaces_a_videos_items(self):
        items, _ = self.items()
        order = bench._judging_order(bench._judgeable(list(items.values())))
        videos = [it["video_id"] for it in order]
        for a, b in zip(videos, videos[1:]):
            self.assertNotEqual(a, b)
        synthetic = [{"item_id": f"{v}{k}", "video_id": v, "order_key": f"{i:03d}"}
                     for i, (v, k) in enumerate([("x", 1), ("x", 2), ("x", 3), ("y", 1), ("z", 1)])]
        order = [it["video_id"] for it in bench._judging_order(synthetic)]
        self.assertEqual(order, ["x", "y", "z", "x", "x"])

    def test_judge_everything(self):
        seen = []
        while True:
            item = bench.next_item(self.NAME)
            if item is None:
                break
            self.assertNotIn(item["item_id"], seen)
            seen.append(item["item_id"])
            out = bench.judge(self.NAME, item["item_id"], "yes" if item["kind"] == "solo" else "tie",
                              recognised="on", reason="")
            self.assertTrue(out["ok"])
        self.assertEqual(len(seen), 4)
        lines = (bench.bench_dir() / self.NAME / "judgments.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(json.loads(lines[0])["recognised"])

    def test_validation(self):
        items, _ = self.items()
        pair = items[("arms", "v1")]["item_id"]
        solo = items[("solo", "n1")]["item_id"]
        auto = items[("ceiling", "v1")]["item_id"]
        cases = [(pair, "yes", ""), (pair, "c", ""), (solo, "a", ""), (pair, "a", "too_long"),
                 ("nope", "a", ""), (auto, "a", "")]
        for item_id, choice, reason in cases:
            with self.subTest(item=item_id, choice=choice, reason=reason):
                with self.assertRaises(bench.BenchError):
                    bench.judge(self.NAME, item_id, choice, reason=reason)
        bench.judge(self.NAME, pair, "A", reason="weak_opening")
        bench.judge(self.NAME, solo, "no", reason="boring")

    def test_latest_wins_and_reveal_locks(self):
        items, _ = self.items()
        pair = items[("arms", "v1")]
        bench.judge(self.NAME, pair["item_id"], "a")
        bench.judge(self.NAME, pair["item_id"], "b")
        hidden = bench.results(self.NAME)
        self.assertEqual(set(hidden), {"name", "revealed", "revealed_at", "progress", "prepared"})
        self.assertFalse(hidden["revealed"])
        out = bench.reveal(self.NAME)
        self.assertTrue(out["revealed_at"])
        self.assertEqual(bench.reveal(self.NAME)["revealed_at"], out["revealed_at"], "idempotent")
        with self.assertRaises(bench.BenchError):
            bench.judge(self.NAME, pair["item_id"], "a")
        self.assertIsNone(bench.next_item(self.NAME))
        res = bench.results(self.NAME)
        arm_on_b = pair["sides"]["b"]["source"] == "arm"
        stats = res["arms"]["gemini"]
        self.assertEqual((stats["wins"], stats["losses"]), (1, 0) if arm_on_b else (0, 1))
        self.assertEqual(stats["auto_ties"], 1)

    def test_media_path(self):
        items, _ = self.items()
        pair = items[("arms", "v1")]["item_id"]
        path = bench.media_path(self.NAME, pair, "a")
        self.assertTrue(path.is_file())
        self.assertIsNone(bench.media_path(self.NAME, pair, "c"))
        self.assertIsNone(bench.media_path(self.NAME, items[("ceiling", "v1")]["item_id"], "a"))
        self.assertIsNone(bench.media_path(self.NAME, items[("solo", "n1")]["item_id"], "b"))
        self.assertIsNone(bench.media_path("missing-run", pair, "a"))


# ---- statistics and results -------------------------------------------------------------------

class StatsTest(unittest.TestCase):
    def test_wilson(self):
        cases = [(26, 50, bench.Z95, (0.3851, 0.6520)), (0, 10, bench.Z95, (0.0, 0.2775)),
                 (10, 10, bench.Z95, (0.7225, 1.0)), (0, 0, bench.Z95, (0.0, 1.0)),
                 (26, 50, bench.Z90, (0.4058, 0.6322))]
        for k, n, z, (lo, hi) in cases:
            with self.subTest(k=k, n=n, z=z):
                got = bench.wilson(k, n, z)
                self.assertAlmostEqual(got[0], lo, places=3)
                self.assertAlmostEqual(got[1], hi, places=3)

    def test_sign_test(self):
        cases = [(9, 1, 22 / 1024), (15, 5, 2 * 21700 / 2 ** 20), (5, 5, 1.0), (0, 0, 1.0),
                 (6, 2, 74 / 256), (0, 6, 2 / 64)]
        for wins, losses, expected in cases:
            with self.subTest(wins=wins, losses=losses):
                self.assertAlmostEqual(bench.sign_test(wins, losses), expected, places=6)

    def test_ceiling_verdict(self):
        cases = [(70, 150, "pass"), (69, 150, "inconclusive"), (26, 50, "pass"),
                 (25, 50, "inconclusive"), (10, 100, "fail"), (0, 0, "no data"),
                 # The printed 95% CI (24.5%–40.6%) straddles the bar: not a fail.
                 (40, 125, "inconclusive"),
                 # ...and once the whole printed interval is below it (upper 39.8%), fail.
                 (39, 125, "fail")]
        for k, n, expected in cases:
            with self.subTest(k=k, n=n):
                self.assertEqual(bench.ceiling_verdict(k, n), expected)

    def test_a_fail_never_sits_beside_a_printed_interval_that_reaches_the_bar(self):
        for n in range(1, 201):
            for k in range(n + 1):
                verdict = bench.ceiling_verdict(k, n)
                low, high = bench.wilson(k, n)   # the 95% CI the report prints
                if verdict == "fail":
                    self.assertLess(high, bench.CEILING_BAR, (k, n))
                elif verdict == "inconclusive":
                    self.assertGreaterEqual(high, bench.CEILING_BAR, (k, n))

    def test_spearman(self):
        cases = [([1, 2, 3, 4], [10, 20, 30, 40], 1.0), ([1, 2, 3, 4], [4, 3, 2, 1], -1.0),
                 ([1, 2, 3], [1, 1, 1], None), ([1, 2], [1, 2], None),
                 ([1, 2, 2, 3], [1, 2, 3, 4], 0.9487)]
        for xs, ys, expected in cases:
            with self.subTest(xs=xs, ys=ys):
                got = bench.spearman(xs, ys)
                if expected is None:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, expected, places=4)


class ResultsTest(_Env):
    NAME = "r1"

    def item(self, kind, idx, *, other="transcript", auto=None, solo=False):
        sides = {"a": {"source": "arm", "render": f"renders/{kind}{idx}-a.mp4", "segments": []}}
        if not solo:
            sides["b"] = {"source": other, "render": f"renders/{kind}{idx}-b.mp4", "segments": []}
        return {"item_id": f"{kind}{idx}", "kind": kind, "video_id": f"v{idx}", "arm": "gemini",
                "sides": sides, "auto": auto, "source_hash": "h", "order_key": f"{kind}{idx}",
                "render_error": ""}

    def setUp(self):
        super().setUp()
        items, judgments = [], []

        def add(kind, idx, choice=None, reason="", recognised=False, **kw):
            items.append(self.item(kind, idx, **kw))
            if choice:
                judgments.append({"item_id": f"{kind}{idx}", "choice": choice, "reason": reason,
                                  "recognised": recognised, "source_hash": "h",
                                  "at": "2026-11-10T00:00:00Z"})

        for i in range(6):
            add("arms", i, "a", reason="boring" if i == 0 else "")
        add("arms", 6, "b", reason="weak_opening")
        add("arms", 7, "b")
        add("arms", 8, "tie", reason="off_topic")
        add("arms", 9, "neither")
        add("arms", 10, auto="tie")
        add("arms", 11)                                   # pending
        add("ceiling", 0, "a", reason="boring")           # engine beat your cut
        add("ceiling", 1, "b", reason="no_sense")
        add("ceiling", 2, "b", recognised=True)
        add("ceiling", 3, "tie")
        add("ceiling", 4, "neither")
        add("ceiling", 5, other="reference", auto="tie")
        add("solo", 0, "yes", solo=True)
        add("solo", 1, "yes", solo=True)
        add("solo", 2, "no", reason="boring", solo=True)
        for it in items:
            if it["kind"] == "ceiling":
                it["sides"]["b"]["source"] = "reference"
        write_run(self.NAME, videos=[], results={}, revealed_at="2026-11-12T00:00:00Z")
        run_dir = bench.bench_dir() / self.NAME
        util.atomic_write_json(run_dir / "pairs.json", {"items": items, "render_failures": 2,
                                                        "no_top1": {"gemini": ["x", "y"]}})
        # A stale judgment (old results) and one after the reveal never count.
        judgments.append({"item_id": "arms7", "choice": "a", "source_hash": "old",
                          "at": "2026-11-11T00:00:00Z"})
        judgments.append({"item_id": "arms7", "choice": "a", "source_hash": "h",
                          "at": "2026-11-13T00:00:00Z"})
        (run_dir / "judgments.jsonl").write_text(
            "\n".join(json.dumps(j) for j in judgments) + "\nnot json\n")
        self.res = bench.results(self.NAME)

    def test_arms(self):
        a = self.res["arms"]["gemini"]
        self.assertEqual((a["judged"], a["wins"], a["losses"], a["ties"], a["neither"],
                          a["auto_ties"], a["pending"]), (10, 6, 2, 1, 1, 1, 1))
        self.assertAlmostEqual(a["win_rate"], 0.75)
        self.assertAlmostEqual(a["sign_p"], 74 / 256)
        self.assertEqual(a["ci"], list(bench.wilson(6, 8)))

    def test_ceiling(self):
        c = self.res["ceiling"]["gemini"]
        overall, unrec = c["overall"], c["unrecognised"]
        # n = judged 5 + auto 1 + no top-1 2; win-or-tie = wins 1 + ties 1 + auto 1.
        self.assertEqual((overall["n"], overall["wins"], overall["ties"], overall["auto_ties"],
                          overall["losses"], overall["neither"], overall["no_top1"]),
                         (8, 1, 1, 1, 2, 1, 2))
        self.assertAlmostEqual(overall["win_or_tie"], 3 / 8)
        self.assertEqual(overall["verdict"], bench.ceiling_verdict(3, 8))
        self.assertEqual((unrec["n"], unrec["losses"]), (7, 1))
        self.assertAlmostEqual(unrec["win_or_tie"], 3 / 7)
        self.assertEqual(c["recognised"], 1)

    def test_solo_and_reasons(self):
        solo = self.res["solo"]["gemini"]
        self.assertEqual((solo["judged"], solo["yes"]), (3, 2))
        self.assertAlmostEqual(solo["share_yes"], 2 / 3)
        reasons = self.res["loss_reasons"]
        self.assertEqual(reasons["gemini"]["arms"], {"weak_opening": 1, "off_topic": 1})
        self.assertEqual(reasons["transcript"]["arms"], {"boring": 1, "off_topic": 1})
        self.assertEqual(reasons["gemini"]["ceiling"], {"no_sense": 1})
        self.assertEqual(reasons["gemini"]["solo"], {"boring": 1})
        self.assertEqual(self.res["render_failures"], 2)

    def test_report_after_reveal(self):
        text = bench.report(self.NAME)
        for phrase in ("PRIMARY", "win rate", "Win-or-tie", "Never-cut videos", "Why clips lost",
                       bench.RUBRIC_NOTE, "Anchored history", "1-sided low",
                       "fail needs the whole 95% CI below 40%"):
            with self.subTest(phrase):
                self.assertIn(phrase, text)


class ReportTest(_Env):
    def setUp(self):
        super().setUp()
        videos = [ventry("v1", [target(1, [S(10, 30)])]), ventry("v2", [target(2, [S(0, 20)])])]
        results = {"gemini": {"v1": result([[S(10, 30)]]), "v2": result([[S(0, 20)]])},
                   "transcript": {"v1": result([[S(50, 60)]]), "v2": result([], error="X: y")}}
        self.stopped = {"reason": "max_usd", "max_usd": 40.0, "videos_done": 2, "videos_total": 60}
        write_run("rep", videos=videos, results=results, stopped=self.stopped, seed="dev",
                  ledger={"v1": {"clips": [{"segments": [S(10, 30)]}], "policy_version": "p"}})

    def test_hidden_until_reveal(self):
        text = bench.report("rep")
        self.assertIn("Results stay hidden", text)
        self.assertIn(bench.DEV_BANNER, text)
        self.assertIn("Stopped at the $40.00 cap after 2 of 60 videos", text)
        self.assertIn(bench.RUBRIC_NOTE, text)
        self.assertNotIn("hit@1", text)
        self.assertIn("pairs not prepared", text)

    def test_tables_after_reveal(self):
        bench.reveal("rep")
        text = bench.report("rep")
        for phrase in ("hit@1", "unseeded targets (headline)", "Engine orders", "By source length",
                       "Anchored history", f"transcript-{clip_proposals.PROMPT_VERSION}",
                       "gemini + fallback", "Reconciliation", "Cost per video"):
            with self.subTest(phrase):
                self.assertIn(phrase, text)
        hit_line = next(line for line in text.splitlines() if line.startswith("gemini  "))
        self.assertIn("100%", hit_line)


# ---- critic check -------------------------------------------------------------------------------

class CriticCheckTest(_DB):
    def setUp(self):
        super().setUp()
        ours = self.add(Channel(call_sign="Ours", url="https://x/@ours", first_party=True))
        self.labels = {}
        video = self.video("src", duration=600)
        for i, value in enumerate(["yes", "yes", "yes", "no", "no", "no"]):
            cut = self.cut(video, [S(i * 20, i * 20 + 15)], title=f"clip {i}")
            post = self.post(cut, views=1000 * (6 - i) if value == "yes" else 100 * i,
                             published=ago(10 + i))
            self.add(ContentLabel(cut_pk=cut.id, label="gripping", value=value))
            self.labels[post.id] = value
            if i == 0:   # a re-air of the same cut is not a first airing
                self.post(cut, views=5, published=ago(2), repost_of=post.id)
        for i in range(2):
            self.post(self.cut(video, [S(200 + i * 20, 210 + i * 20)]), views=50, published=ago(1 + i))
        own = self.video("own", channel=ours)
        own_cut = self.cut(own, [S(0, 10)])
        self.post(own_cut, views=10)
        self.add(ContentLabel(cut_pk=own_cut.id, label="gripping", value="yes"))
        self.commit()
        self.watched = []

        def fake_example_file(example, settings, *, height=360):
            path = self.tmp / f"example-{example['post_pk']}.mp4"
            path.write_bytes(b"clip")
            return path

        def fake_cold_watch(**kw):
            self.watched.append(kw)
            name = Path(kw["render"]).stem
            pk = int(name.split("-")[1])
            value = self.labels.get(pk)
            flipped = value == "no" and pk == max(k for k, v in self.labels.items() if v == "no")
            stops = value == "yes" or flipped
            return {"stops_scroll": stops, "score": 8 if value == "yes" else 3, "error": ""}

        self.patch(taste, "example_file", fake_example_file)
        self.patch(media, "probe_duration", lambda path: 15.0)
        self.patch(critic, "cold_watch", fake_cold_watch)
        self.patch(engine, "CostMeter", FakeMeter)

    def test_agreement_spearman_and_file(self):
        out = bench.critic_check(None, roles=("gemini",), limit=40,
                                 settings=Settings(raw={}), log=self.say)
        role = out["roles"]["gemini"]
        self.assertEqual(out["labelled"], 6)
        self.assertEqual(role["n"], 8, "6 labelled + 2 recent unlabelled; no reposts, no first-party")
        agree = role["agreement"]
        self.assertEqual((agree["yes"]["n"], agree["yes"]["stops_scroll_agree"]), (3, 3))
        self.assertEqual((agree["no"]["n"], agree["no"]["stops_scroll_agree"]), (3, 2))
        self.assertEqual(agree["yes"]["score6_agree"], 3)
        self.assertEqual(agree["gate"], "too few labels")
        self.assertIsNotNone(role["spearman"])
        self.assertGreaterEqual(role["n_spearman"], 3)
        path = Path(out["path"])
        self.assertEqual(path.parent, bench.bench_dir())
        self.assertTrue(path.name.startswith("critic_check-"))
        self.assertEqual(json.loads(path.read_text())["labelled"], 6)
        self.assertEqual(self.buckets, ["bench", None])
        self.assertTrue(all(w["clip_seconds"] == 15.0 for w in self.watched))
        self.assertTrue(all(w["meter"].bucket == "bench" for w in self.watched))

    def test_words_reach_the_critic_and_roles_are_checked(self):
        bench.critic_check(None, roles="claude", settings=Settings(raw={}), log=self.say)
        self.assertTrue(all(w["role"] == "claude" for w in self.watched))
        first = next(w for w in self.watched if w["clip_words"])
        self.assertEqual(first["clip_words"][0]["word"], "hi")
        for roles in (("gpt",), (), ""):
            with self.subTest(roles=roles):
                with self.assertRaises(bench.BenchError):
                    bench.critic_check(None, roles=roles, settings=Settings(raw={}))

    def test_written_into_a_named_run(self):
        write_run("cc", videos=[], results={})
        bench.critic_check("cc", settings=Settings(raw={}), log=self.say)
        self.assertTrue((bench.bench_dir() / "cc" / "critic_check.json").is_file())

    def test_agreement_gate(self):
        rows = ([{"label": "yes", "stops_scroll": True, "score": 7}] * 5
                + [{"label": "no", "stops_scroll": False, "score": 2}] * 4
                + [{"label": "no", "stops_scroll": True, "score": 6}])
        self.assertEqual(bench._agreement(rows)["gate"], "pass")
        rows[-2] = {"label": "no", "stops_scroll": True, "score": 2}
        self.assertEqual(bench._agreement(rows)["gate"], "fail")


class ListRunsTest(_Env):
    def test_lists_newest_first_and_skips_broken(self):
        write_run("old", videos=[ventry("v1", [])], results={"gemini": {"v1": result([])}},
                  created_at="2026-10-01T00:00:00Z")
        write_run("new", videos=[], results={}, created_at="2026-11-10T00:00:00Z", seed="dev",
                  revealed_at="2026-11-11T00:00:00Z")
        broken = bench.bench_dir() / "broken"
        broken.mkdir(parents=True)
        (broken / "manifest.json").write_text("{not json")
        (bench.bench_dir() / "touched.json").write_text("[]")
        runs = bench.list_runs()
        self.assertEqual([r["name"] for r in runs], ["new", "old"])
        old = runs[1]
        self.assertEqual((old["videos"], old["results"], old["revealed"], old["dev"]),
                         (1, {"gemini": 1, "transcript": 0}, False, False))
        self.assertTrue(runs[0]["revealed"])
        self.assertTrue(runs[0]["dev"])
        self.assertEqual(old["arms"], ["gemini", f"transcript-{clip_proposals.PROMPT_VERSION}"])


if __name__ == "__main__":
    unittest.main()
