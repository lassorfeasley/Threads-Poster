"""Tests for the clip engine's orchestrator (app/clip_engine/engine.py).

Every collaborator — perceiver, editor, snapper, renderer, critic, taste
files — is replaced by a scripted fake, so these pin down the engine's OWN
decisions: stage order, the filters, the compare set, how a ranking is
merged, when a revision may replace its original, top selection, the empty
outcomes, the failure paths, and the cost/cancel/deadline checks. No model,
ffmpeg or network call runs.

Run with:  python -m pytest -q tests/test_clip_engine_engine.py
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

from app import clip_proposals, llm, spend
from app.clip_engine import critic, editor, engine, gemini, media, shotlog, snap, taste, util
from app.clip_engine import playbook as playbook_mod
from app.config import Settings

# The real collaborators, captured before any test patches them, so the
# contract test can check the engine's calls against their signatures.
REAL = {
    "build": shotlog.build, "credits_for": shotlog.credits_for, "shot_map": media.shot_map,
    "perception_proxy": media.perception_proxy, "render_cached": media.render_cached,
    "probe_duration": media.probe_duration, "entries_for": editor.entries_for,
    "keyframes": editor.keyframes, "draft": editor.draft, "revise": editor.revise,
    "snap_segments": snap.snap_segments, "cold_watch": critic.cold_watch,
    "cold_score": critic.cold_score, "rank": critic.rank, "prefer": critic.prefer,
    "example_file": taste.example_file, "open_rule": playbook_mod.open_rule,
    "render": playbook_mod.render,
}

BRAND = {"name": "Renewables.org", "topic": "climate change"}
PLAYBOOK = {"archetypes": [{"key": "voice", "name": "The Voice", "open": "On the person."}],
            "flops": [], "notes": ""}
WORDS = [
    {"word": "Hello", "start": 10.0, "end": 10.4},
    {"word": "world.", "start": 10.5, "end": 11.0},
    {"word": "Later", "start": 105.0, "end": 105.5},
]
TRANSCRIPT = [
    {"start": 10.0, "end": 14.0, "text": "Hello world."},
    {"start": 104.0, "end": 108.0, "text": "Later on."},
]


def seg(a, b, lead="word"):
    return {"start": float(a), "end": float(b), "lead": lead}


def make_draft(did, *spans, story_key=None, archetype="voice", lens="person"):
    return {"id": did, "moment_id": "m1", "archetype": archetype, "lens": lens,
            "story": f"the {did} story",
            "story_key": did if story_key is None else story_key,
            "hook": f"hook {did}", "payoff": "", "segments": [seg(a, b) for a, b in spans],
            "why": "", "risk": "", "draft_caption": f"caption {did}",
            "challenges_rules": "", "origin": "draft", "parent": None}


def crit(score, *, makes_sense=True, ends_well=True, topic="central", error="", **extra):
    out = {"first_impression": "a face", "stops_scroll_muted": True, "muted_reason": "",
           "stops_scroll": True, "stop_reason": "a famous face, mid-line",
           "person_at_start": True, "about": "", "makes_sense": makes_sense,
           "dangling": "", "bored_at": None, "dead_air": [], "ends_well": ends_well,
           "topic": topic, "topic_note": "", "person": "", "visual": 3, "audio": 3,
           "score": score, "fix": "start on the face", "error": error}
    out.update(extra)
    return out


def usage(cost, model="test-model", tokens=(100, 10)):
    return {"model": model, "input_tokens": tokens[0], "output_tokens": tokens[1],
            "cost_usd": cost}


def segkey(segments):
    return "|".join(f"{float(s['start']):.2f}-{float(s['end']):.2f}" for s in segments)


def make_settings(dotted=None):
    raw: dict = {}
    for key, value in (dotted or {}).items():
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Settings(raw=raw)


class ProviderError(RuntimeError):
    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage


class Harness:
    """Scripted stand-ins for every collaborator ``engine.run`` calls, and a
    record of what ran, in order."""

    def __init__(self, test: unittest.TestCase) -> None:
        self.test = test
        self.events: list[str] = []
        self.calls: dict[str, list] = defaultdict(list)
        self.labels: list[str] = []
        self.settings: dict = {}
        self.drafts: list[dict] = []
        self.moments = [{"id": "m1", "start": 0.0, "end": 120.0, "kind": "person", "why": "x"}]
        self.critiques: dict[str, dict] = {}
        self.shotlog = {"version": 2, "perceiver": "gemini", "complete": True,
                        "cost_usd": 0.05, "people": [],
                        "beats": [{"start": 0.0, "end": 600.0, "topic": 2, "shot": "interview"}]}
        self.shot_map = {"version": 1, "duration": 600.0, "error": "", "quiet": [[40.0, 41.0]],
                         "cuts": [{"t": 31.0, "score": 45.0}]}
        self.snap_fn = None
        self.render_fail: set[str] = set()
        self.rank_result = None
        self.revisions: dict = {}
        self.prefer_choice = "b"
        self.costs = {"perceive": 0.05, "draft": 0.10, "watch": 0.01, "rank": 0.02,
                      "revise": 0.03, "prefer": 0.01}
        self.cached_perception = None
        self.perceive_error = None
        self.meter_before_error = False
        self.draft_error = None
        self.example_files: dict = {}
        self.probe = 600.0
        self._start()

    def _start(self) -> None:
        patches = [
            mock.patch.object(shotlog, "build", new=self.build),
            mock.patch.object(shotlog, "credits_for", new=self.credits_for),
            mock.patch.object(media, "shot_map", new=self.shot_map_fn),
            mock.patch.object(media, "perception_proxy", new=self.proxy),
            mock.patch.object(media, "render_cached", new=self.render),
            mock.patch.object(media, "probe_duration", new=self.probe_duration),
            mock.patch.object(editor, "entries_for", new=self.entries_for),
            mock.patch.object(editor, "keyframes", new=self.keyframes),
            mock.patch.object(editor, "draft", new=self.draft),
            mock.patch.object(editor, "revise", new=self.revise),
            mock.patch.object(snap, "snap_segments", new=self.snap),
            mock.patch.object(critic, "cold_watch", new=self.cold_watch),
            mock.patch.object(critic, "cold_score", new=self.cold_score),
            mock.patch.object(critic, "rank", new=self.rank),
            mock.patch.object(critic, "prefer", new=self.prefer),
            mock.patch.object(taste, "example_file", new=self.example_file),
            mock.patch.object(playbook_mod, "load", new=lambda: copy.deepcopy(PLAYBOOK)),
            mock.patch.object(playbook_mod, "render", new=lambda pb: "PLAYBOOK TEXT"),
            mock.patch.object(playbook_mod, "open_rule", new=lambda pb, key: f"{key}: open on it"),
            mock.patch.object(llm, "_operator_rules", new=lambda scope: "OPERATOR RULES"),
        ]
        for p in patches:
            p.start()
            self.test.addCleanup(p.stop)

    # -- the fakes ---------------------------------------------------------------

    def build(self, **kw):
        self.events.append("perceive")
        self.calls["build"].append(kw)
        meter = kw["meter"]
        if self.perceive_error is not None:
            if self.meter_before_error:
                meter.add("perceive", usage(self.costs["perceive"]))
            raise self.perceive_error
        if self.cached_perception is not None:
            meter.add_cached("perceive", self.cached_perception)
        else:
            meter.add("perceive", usage(self.costs["perceive"], model="gemini-3.8-flash"))
        return copy.deepcopy(self.shotlog)

    def shot_map_fn(self, source, **kw):
        self.events.append("shot_map")
        self.calls["shot_map"].append({"source": source, "thread": threading.current_thread()})
        return copy.deepcopy(self.shot_map)

    def proxy(self, source, **kw):
        self.events.append("proxy")
        self.calls["proxy"].append(dict(kw, source=source))
        return Path("/proxy/whole.mp4")

    def probe_duration(self, path):
        self.calls["probe"].append(path)
        return self.probe

    def keyframes(self, source, log, shot_map, *, settings, limit):
        self.calls["keyframes"].append({"limit": limit})
        return [(1.0, b"jpeg")]

    def entries_for(self, words, transcript):
        return list(transcript or []), bool(words)

    def draft(self, **kw):
        self.events.append("draft")
        self.calls["draft"].append(kw)
        if self.draft_error is not None:
            raise self.draft_error
        kw["meter"].add("draft", usage(self.costs["draft"]))
        return copy.deepcopy(self.drafts), copy.deepcopy(self.moments)

    def snapped(self, segments):
        segs = [dict(s) for s in segments]
        return self.snap_fn(segs) if self.snap_fn else segs

    def snap(self, segments, **kw):
        self.calls["snap"].append(dict(kw, segments=copy.deepcopy(segments)))
        return self.snapped(segments)

    def all_drafts(self):
        return list(self.drafts) + [r for r in self.revisions.values() if isinstance(r, dict)]

    def id_for(self, segments):
        key = segkey(segments)
        for d in self.all_drafts():
            if segkey(self.snapped(d["segments"])) == key:
                return d["id"]
        raise AssertionError(f"no scripted draft has segments {key}")

    def render(self, source, segments, *, height=360, kind="draft"):
        did = self.id_for(segments)
        self.events.append(f"render:{did}")
        self.calls["render"].append({"id": did, "segments": copy.deepcopy(segments),
                                     "height": height, "source": source})
        if did in self.render_fail:
            raise media.MediaError("ffmpeg exploded")
        return Path(f"/renders/{did}.mp4")

    def credits_for(self, log, segments):
        return f"credits:{self.id_for(segments)}"

    def cold_watch(self, **kw):
        did = Path(kw["render"]).stem
        self.events.append(f"watch:{did}")
        self.calls["watch"].append(dict(kw, id=did))
        kw["meter"].add("watch", usage(self.costs["watch"]))
        return copy.deepcopy(self.critiques[did])

    def cold_score(self, critique, *, clip_seconds, archetype=""):
        return -1e9 if critique.get("error") else float(critique["score"])

    def rank(self, **kw):
        self.events.append("rank")
        self.calls["rank"].append(kw)
        kw["meter"].add("rank", usage(self.costs["rank"]))
        if isinstance(self.rank_result, BaseException):
            raise self.rank_result
        if self.rank_result is not None:
            return list(self.rank_result)
        return [d["id"] for d in kw["drafts"]]

    def revise(self, **kw):
        did = kw["draft"]["id"]
        self.events.append(f"revise:{did}")
        self.calls["revise"].append(kw)
        out = self.revisions.get(did)
        if isinstance(out, BaseException):
            raise out
        if out:
            kw["meter"].add("revise", usage(self.costs["revise"]))
        return copy.deepcopy(out)

    def prefer(self, **kw):
        self.events.append("prefer")
        self.calls["prefer"].append(kw)
        kw["meter"].add("prefer", usage(self.costs["prefer"]))
        return self.prefer_choice

    def example_file(self, example, settings, *, height=360):
        self.calls["example_file"].append({"post_pk": example["post_pk"], "height": height})
        return self.example_files.get(example["post_pk"])

    def progress(self, label):
        self.labels.append(label)
        self.events.append(f"stage:{label}")

    # -- running -------------------------------------------------------------------

    def run(self, **overrides) -> engine.EngineResult:
        kwargs = dict(setup="gemini", settings=make_settings(self.settings), brand=BRAND,
                      playbook=None, video_id="vid123", title="A title",
                      description="A description", duration=600.0,
                      source=Path("/videos/source.mp4"), words=list(WORDS),
                      transcript=list(TRANSCRIPT), context={"shipped_examples": []},
                      rejected=[], examples=[], used=[], window=None, section_topic="",
                      keep=None, meter=None, progress=self.progress, cancel=None,
                      deadline=None)
        kwargs.update(overrides)
        return engine.run(**kwargs)

    def first(self, prefix):
        return min(i for i, e in enumerate(self.events) if e.startswith(prefix))

    def last(self, prefix):
        return max(i for i, e in enumerate(self.events) if e.startswith(prefix))

    def rendered(self):
        return [c["id"] for c in self.calls["render"]]


class _SpendSafe(unittest.TestCase):
    """Nothing here should record spend; if something did, it lands in a temp
    ledger rather than the workspace's real one."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = mock.patch.object(spend, "_LEDGER", Path(tmp.name) / "llm_spend.json")
        p.start()
        self.addCleanup(p.stop)

    def harness(self, drafts=(), critiques=None, **attrs) -> Harness:
        h = Harness(self)
        h.drafts = list(drafts)
        h.critiques = dict(critiques or {})
        for key, value in attrs.items():
            setattr(h, key, value)
        return h

    def three(self, **attrs) -> Harness:
        """Three distinct stories, scored 8 / 7 / 6, no revision on offer."""
        return self.harness(
            [make_draft("d1", (10, 30)), make_draft("d2", (100, 120)),
             make_draft("d3", (200, 220))],
            {"d1": crit(8), "d2": crit(7), "d3": crit(6)}, **attrs)


# ---- accounting ----------------------------------------------------------------------

class CostMeterTest(unittest.TestCase):
    def test_add_counts_cost_by_stage_and_tokens_by_model(self):
        meter = engine.CostMeter()
        meter.add("draft", usage(0.10, model="gemini-3.8-flash", tokens=(1000, 200)))
        meter.add("draft", usage(0.05, model="gemini-3.8-flash", tokens=(500, 100)))
        meter.add("watch", usage(0.01, model="claude-opus-5-5", tokens=(10, 1)))
        self.assertAlmostEqual(meter.total, 0.16)
        self.assertAlmostEqual(meter.by_stage["draft"], 0.15)
        self.assertEqual(meter.tokens["gemini-3.8-flash"], {"input": 1500, "output": 300})
        self.assertEqual(meter.tokens["claude-opus-5-5"], {"input": 10, "output": 1})
        self.assertEqual(meter.calls, 3)
        self.assertEqual(meter.bucket, "engine")
        self.assertEqual(engine.CostMeter("bench").bucket, "bench")

    def test_cached_cost_is_reported_but_not_spent(self):
        meter = engine.CostMeter()
        meter.add("draft", usage(0.10))
        meter.add_cached("perceive", 0.25)
        meter.add_cached("perceive", 0.05)
        self.assertAlmostEqual(meter.total, 0.10)
        self.assertAlmostEqual(meter.by_stage["perceive_cached"], 0.30)
        self.assertAlmostEqual(meter.total_with_cached, 0.40)
        self.assertNotIn("perceive", meter.by_stage)

    def test_malformed_usage_never_raises(self):
        meter = engine.CostMeter()
        for bad in (None, "x", 3, {}, {"cost_usd": "abc"}, {"cost_usd": -1.0},
                    {"cost_usd": float("nan")}, {"cost_usd": float("inf")},
                    {"cost_usd": 0.01, "input_tokens": "lots", "model": None}):
            meter.add("stage", bad)
        meter.add_cached("perceive", "not a number")
        self.assertAlmostEqual(meter.total, 0.01)
        self.assertEqual(meter.calls, 6)          # only dicts count as calls
        self.assertEqual(meter.tokens, {})        # no model, no token row
        self.assertEqual(meter.by_stage["perceive_cached"], 0.0)

    def test_thread_safe(self):
        meter = engine.CostMeter()

        def spend_some():
            for _ in range(500):
                meter.add("watch", usage(0.001, model="m", tokens=(2, 1)))

        threads = [threading.Thread(target=spend_some) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertAlmostEqual(meter.total, 4.0, places=6)
        self.assertEqual(meter.calls, 4000)
        self.assertEqual(meter.tokens["m"], {"input": 8000, "output": 4000})

    def test_snapshot_is_a_rounded_copy(self):
        meter = engine.CostMeter()
        meter.add("draft", usage(0.1234567891, model="m"))
        snap_ = meter.snapshot()
        self.assertEqual(snap_["cost_usd"], 0.123457)
        snap_["tokens"]["m"]["input"] = 0
        self.assertEqual(meter.tokens["m"]["input"], 100)


class ModelsAndEstimateTest(unittest.TestCase):
    def test_models_for_each_setup(self):
        s = make_settings()
        self.assertEqual(engine.models_for("gemini", s),
                         {"perceiver": "gemini-3.8-flash", "editor": "gemini-3.8-flash",
                          "critic": "gemini-3.8-flash", "judge": "gemini-3.8-flash"})
        self.assertEqual(engine.models_for("hybrid", s),
                         {"perceiver": "gemini-3.8-flash", "editor": "claude-opus-5-5",
                          "critic": "claude-opus-5-5", "judge": "claude-opus-5-5"})
        self.assertEqual(engine.models_for("claude", s)["perceiver"], "claude-opus-5-5")

    def test_role_overrides_blanks_and_judge_fallback(self):
        s = make_settings({"clip_engine.gemini.model": "gemini-4-hydrogen",
                           "clip_engine.gemini.editor_model": "",
                           "clip_engine.gemini.critic_model": "gemini-4-carbon",
                           "clip_engine.gemini.judge_model": None})
        models = engine.models_for("gemini", s)
        self.assertEqual(models["editor"], "gemini-4-hydrogen")    # blank -> provider model
        self.assertEqual(models["critic"], "gemini-4-carbon")
        self.assertEqual(models["judge"], "gemini-4-carbon")       # blank -> critic

    def test_transcript_and_unknown(self):
        s = make_settings({"clips.model": "claude-sonnet-5"})
        self.assertEqual(engine.models_for("transcript", s)["editor"], "claude-sonnet-5")
        with self.assertRaises(ValueError):
            engine.models_for("vibes", s)

    def test_estimate(self):
        s = make_settings()
        cases = [
            ("gemini", 600, 0.60 + 0.03 * 10),
            ("hybrid", 1800, 1.40 + 0.02 * 30),
            ("claude", 1800, 1.40 + 0.06 * 30),
            ("gemini", None, 0.60 + 0.03 * 20),     # unknown length priced high
            ("gemini", 0, 0.60 + 0.03 * 20),
            ("transcript", 600, 0.0),
        ]
        for setup, duration, expected in cases:
            with self.subTest(setup=setup, duration=duration):
                self.assertAlmostEqual(engine.estimate_usd(setup, s, duration), expected)

    def test_estimate_reads_settings_and_survives_bad_values(self):
        s = make_settings({"clip_engine.estimate_usd_base": {"gemini": 1.0, "claude": "lots"},
                           "clip_engine.estimate_usd_per_minute": {"gemini": 0.1}})
        self.assertAlmostEqual(engine.estimate_usd("gemini", s, 600), 2.0)
        self.assertAlmostEqual(engine.estimate_usd("claude", s, 600), 1.40 + 0.06 * 10)
        self.assertAlmostEqual(engine.estimate_usd("hybrid", s, 600), 1.40 + 0.02 * 10)


# ---- the selection rules, one table each -------------------------------------------------

class SelectionRulesTest(unittest.TestCase):
    def test_same_cut_only_for_true_duplicates(self):
        a = [seg(10, 30)]
        cases = [
            ("identical", [seg(10, 30)], True),
            ("nudged edges", [seg(10.3, 30.2)], True),
            ("opening moved more than 0.5 s", [seg(10.8, 30.4)], False),
            ("tightened take", [seg(10, 18), seg(20, 30)], False),
            ("trimmed by a fifth", [seg(10, 26)], False),
            ("elsewhere", [seg(100, 120)], False),
            ("empty", [], False),
        ]
        for name, b, expected in cases:
            with self.subTest(name):
                self.assertEqual(engine.same_cut(a, b), expected)
        straight = [seg(10, 20), seg(20.5, 30)]
        cold_open = [seg(20.5, 30), seg(10, 20)]
        self.assertFalse(engine.same_cut(straight, cold_open))
        # Same material, same play order, opening within 0.5 s: a duplicate.
        self.assertTrue(engine.same_cut(cold_open, [seg(20.6, 30), seg(10, 20)]))
        # Same count and opening but material played in a different order.
        self.assertFalse(engine.same_cut([seg(10, 15), seg(40, 50), seg(20, 30)],
                                         [seg(10, 15), seg(20, 30), seg(40, 50)]))

    def test_repeats_rejection(self):
        dismissed = [{"start": 10.0, "end": 30.0}]
        cases = [
            ("same moment", [seg(11, 31)], [{"segments": dismissed, "reason": "off_topic"}], True),
            ("no reason given", [seg(11, 31)], [{"segments": dismissed, "reason": ""}], True),
            ("under half shared", [seg(25, 45)], [{"segments": dismissed, "reason": ""}], False),
            ("boundaries moved 4 s", [seg(14, 31)],
             [{"segments": dismissed, "reason": "bad_boundaries"}], False),
            ("end moved 2 s", [seg(10, 32)],
             [{"segments": dismissed, "reason": "bad_boundaries"}], False),
            ("boundaries barely moved", [seg(11, 31)],
             [{"segments": dismissed, "reason": "bad_boundaries"}], True),
            ("stored as JSON text", [seg(11, 31)],
             [{"segments": json.dumps(dismissed), "reason": "weak_visuals"}], True),
            ("garbage rows", [seg(11, 31)], [None, {"segments": "{nope"}, {}], False),
            ("nothing dismissed", [seg(11, 31)], [], False),
        ]
        for name, segments, rejected, expected in cases:
            with self.subTest(name):
                self.assertEqual(engine.repeats_rejection(segments, rejected) is not None,
                                 expected)

    def test_material_topic_is_overlap_weighted(self):
        log = {"beats": [{"start": 0, "end": 20, "topic": 2},
                         {"start": 20, "end": 40, "topic": 0},
                         {"start": 40, "end": 50, "topic": "?"}]}
        self.assertEqual(engine.material_topic(log, [seg(0, 20)]), 2.0)
        self.assertEqual(engine.material_topic(log, [seg(10, 30)]), 1.0)
        # A sliver of the next beat barely moves it.
        self.assertAlmostEqual(engine.material_topic(log, [seg(0, 20.5)]), 1.951, places=3)
        self.assertEqual(engine.material_topic(log, [seg(30, 35), seg(5, 10)]), 1.0)
        self.assertIsNone(engine.material_topic(log, [seg(45, 50)]))     # unscored beat
        self.assertIsNone(engine.material_topic({}, [seg(0, 10)]))
        self.assertIsNone(engine.material_topic(None, [seg(0, 10)]))

    def test_topic_verdict_needs_two_witnesses(self):
        cases = [
            ("central", crit(5), 0.0, "keep"),
            ("allegory", crit(5, topic="allegory"), 0.0, "keep"),
            ("both say off topic", crit(5, topic="none"), 0.4, "drop"),
            ("incidental, log agrees", crit(5, topic="incidental"), 0.9, "drop"),
            ("critic alone", crit(5, topic="none"), 1.0, "doubt"),
            ("log has no opinion", crit(5, topic="incidental"), None, "doubt"),
            ("not reviewed", crit(0, topic="none", error="quota"), 0.0, "keep"),
            ("no critique", None, 0.0, "keep"),
        ]
        for name, critique, material, expected in cases:
            with self.subTest(name):
                verdict, why = engine.topic_verdict(critique, material)
                self.assertEqual(verdict, expected)
                if verdict != "keep":
                    self.assertTrue(why)

    def scored(self, did, score, story=None, rank=0, **critique):
        d = make_draft(did, (0, 20), story_key=story)
        error = critique.pop("error", "")
        d.update({"score": -1e9 if error else score, "editor_rank": rank,
                  "critique": crit(score, error=error, **critique)})
        return d

    def test_compare_set_best_per_story_then_runners_up(self):
        a1 = self.scored("a1", 8, "A", rank=0)
        a2 = self.scored("a2", 9, "A", rank=1)
        a3 = self.scored("a3", 7, "A", rank=2)
        b1 = self.scored("b1", 6, "B", rank=3)
        c1 = self.scored("c1", 9.5, "C", rank=4, makes_sense=False)
        d1 = self.scored("d1", 0, "D", rank=5, error="quota")
        pool = [a1, a2, a3, b1, c1, d1]
        ids = lambda ds: [d["id"] for d in ds]          # noqa: E731
        # Three stories pass the gate (A, B, D); A's runner-up fills a place.
        self.assertEqual(ids(engine.compare_set(pool, size=5)), ["a2", "a1", "b1", "d1"])
        self.assertEqual(ids(engine.compare_set(pool, size=3)), ["a2", "b1", "d1"])
        self.assertEqual(ids(engine.compare_set(pool, size=2)), ["a2", "b1"])
        self.assertEqual(engine.compare_set(pool, size=0), [])
        # Score ties go to the editor's own order; blank story keys stand alone.
        t1, t2 = self.scored("t1", 7, "", rank=1), self.scored("t2", 7, "", rank=0)
        self.assertEqual(ids(engine.compare_set([t1, t2], size=5)), ["t2", "t1"])

    def test_merge_ranking(self):
        group = [self.scored("a", 9), self.scored("b", 8), self.scored("c", 7)]
        others = [self.scored("x", 5), self.scored("y", 6)]
        ids = lambda ds: [d["id"] for d in ds]          # noqa: E731
        self.assertEqual(ids(engine.merge_ranking(["c", "a", "b"], group, others)),
                         ["c", "a", "b", "y", "x"])
        # Invented ids ignored, repeats ignored, forgotten ones follow by score.
        self.assertEqual(ids(engine.merge_ranking(["zz", "b", "b"], group, others)),
                         ["b", "a", "c", "y", "x"])
        self.assertEqual(ids(engine.merge_ranking([], group, [])), ["a", "b", "c"])

    def test_revision_rules(self):
        original = {"score": 7.0, "critique": crit(7)}
        cases = [
            ("better on every count", crit(8), 8.0, ""),
            ("as good", crit(7), 7.0, ""),
            ("lost the thread", crit(9, makes_sense=False), 9.0, "no longer makes sense"),
            ("lost the ending", crit(9, ends_well=False), 9.0, "no longer ends well"),
            ("drifted", crit(9, topic="incidental"), 9.0, "drifted off topic"),
            ("allegory is on topic", crit(8, topic="allegory"), 8.0, ""),
            ("scored lower", crit(6.5), 6.5, "scored lower"),
            ("not reviewed", crit(0, error="boom"), -1e9, "couldn't review"),
        ]
        for name, critique, score, expected in cases:
            with self.subTest(name):
                blocker = engine.revision_blocker(original, {"score": score, "critique": critique})
                if expected:
                    self.assertIn(expected, blocker)
                else:
                    self.assertEqual(blocker, "")
        # A failure the original already had is not a new one.
        weak = {"score": 4.0, "critique": crit(4, ends_well=False)}
        self.assertEqual(engine.revision_blocker(weak, {"score": 5.0,
                                                        "critique": crit(5, ends_well=False)}), "")
        # A legitimate score of 0 is a score, not "unreviewed".
        self.assertEqual(engine.revision_blocker({"score": 0.0, "critique": crit(0)},
                                                 {"score": 0.0, "critique": crit(0)}), "")

    def test_pick_top_skips_heavy_overlap(self):
        d1 = make_draft("d1", (10, 30))
        d2 = make_draft("d2", (12, 32))          # IoU 0.82 with d1
        d3 = make_draft("d3", (25, 45))          # IoU 0.14 with d1
        d4 = make_draft("d4", (100, 120))
        chosen, skipped = engine.pick_top([d1, d2, d3, d4], keep=3)
        self.assertEqual([d["id"] for d in chosen], ["d1", "d3", "d4"])
        self.assertEqual([(s[0]["id"], s[1]["id"]) for s in skipped], [("d2", "d1")])
        chosen, skipped = engine.pick_top([d1, d2, d3, d4], keep=1)
        self.assertEqual([d["id"] for d in chosen], ["d1"])
        self.assertEqual(skipped, [])

    def test_transcript_clip_entries_follow_play_order(self):
        transcript = [{"start": 10, "end": 14, "text": "first  line"},
                      {"start": 50, "end": 56, "text": "the payoff"},
                      {"start": 70, "end": 72, "text": ""}, {"start": "x"}]
        cold_open = [seg(52, 60), seg(10, 16)]
        self.assertEqual(engine.transcript_clip_entries(transcript, cold_open), [
            {"start": 0.0, "end": 4.0, "text": "the payoff"},
            {"start": 8.0, "end": 12.0, "text": "first line"},
        ])
        self.assertEqual(engine.transcript_clip_entries([], cold_open), [])

    def test_failed_critique_shape(self):
        out = engine.failed_critique(ProviderError("quota", usage=usage(0.02)))
        self.assertEqual(out["score"], 0)
        self.assertEqual(out["topic"], "central")
        self.assertFalse(out["stops_scroll"] or out["makes_sense"] or out["ends_well"])
        self.assertIn("quota", out["error"])
        self.assertEqual(out["usage"]["cost_usd"], 0.02)
        self.assertNotIn("usage", engine.failed_critique("timed out"))

    def test_clean_shot_map_tolerates_garbage(self):
        out = engine.clean_shot_map({"cuts": [{"t": "5", "score": 30}, {"t": 2.0}, "x", {}],
                                     "quiet": [[1, 2], [3, 3], "bad", [5]],
                                     "error": None}, 60.0)
        self.assertEqual(out["cuts"], [{"t": 2.0, "score": 0.0}, {"t": 5.0, "score": 30.0}])
        self.assertEqual(out["quiet"], [[1.0, 2.0]])
        self.assertEqual(out["error"], "")
        self.assertEqual(out["duration"], 60.0)
        self.assertEqual(engine.clean_shot_map(None, 60.0)["error"], "no shot map")

    def test_adopt_drafts_and_revisions(self):
        raw = [make_draft("d1", (10, 30)), make_draft("d1", (40, 60)), "junk",
               {"segments": [seg(70, 90)]}]
        drafts = engine.adopt_drafts(raw)
        self.assertEqual([d["id"] for d in drafts], ["d1", "d1-2", "d4"])
        self.assertEqual([d["editor_rank"] for d in drafts], [0, 1, 3])
        self.assertEqual(drafts[0]["editor_segments"], drafts[0]["segments"])
        self.assertIsNot(drafts[0]["editor_segments"], drafts[0]["segments"])
        self.assertEqual(drafts[0]["dropped_reason"], "")
        self.assertEqual(drafts[0]["seconds"], 20.0)
        original = dict(drafts[0], critique=crit(7), score=7.0, render="/r/d1.mp4")
        revision = engine.adopt_revision(original, {"segments": [seg(9, 29)],
                                                    "change": "open on the face"})
        self.assertEqual(revision["id"], "d1r")
        self.assertEqual((revision["origin"], revision["parent"]), ("revision", "d1"))
        self.assertEqual(revision["archetype"], "voice")
        self.assertEqual(revision["story_key"], "d1")
        self.assertEqual(revision["change"], "open on the face")
        self.assertIsNone(revision["critique"])
        self.assertEqual(revision["render"], "")
        self.assertEqual(revision["dropped_reason"], "revision not used")


# ---- full runs ----------------------------------------------------------------------------

class RunStagesTest(_SpendSafe):
    def test_stages_run_in_order_and_report_progress(self):
        h = self.three()
        result = h.run()
        self.assertEqual(h.labels, ["Logging the video", "Drafting clips", "Rendering drafts",
                                    "Watching the drafts", "Ranking", "Revising"])
        self.assertLess(h.first("perceive"), h.first("draft"))
        self.assertLess(h.first("draft"), h.first("render:"))
        self.assertLess(h.last("render:"), h.first("watch:"))
        self.assertLess(h.last("watch:"), h.first("rank"))
        self.assertLess(h.first("rank"), h.first("revise:"))
        self.assertEqual(result.outcome, "proposals")
        self.assertEqual([d["id"] for d in result.top], ["d1", "d2", "d3"])
        for key in ("perceive", "draft", "snap", "render", "watch", "rank", "revise", "total"):
            self.assertIn(key, result.timings)

    def test_shot_map_runs_on_a_daemon_thread(self):
        h = self.three()
        h.run()
        thread = h.calls["shot_map"][0]["thread"]
        self.assertTrue(thread.daemon)
        self.assertTrue(thread.name.startswith("clip-engine"))
        self.assertEqual(h.calls["shot_map"][0]["source"], Path("/videos/source.mp4"))

    def test_perceiver_gets_the_run_inputs(self):
        for setup, perceiver in (("gemini", "gemini"), ("hybrid", "gemini"), ("claude", "claude")):
            with self.subTest(setup):
                h = self.three()
                meter = engine.CostMeter()
                h.run(setup=setup, meter=meter)
                kw = h.calls["build"][0]
                self.assertEqual(kw["perceiver"], perceiver)
                self.assertEqual(kw["video_id"], "vid123")
                self.assertEqual(kw["duration"], 600.0)
                self.assertEqual(kw["words"], WORDS)
                self.assertIsNone(kw["window"])
                self.assertIs(kw["meter"], meter)
                self.assertEqual(kw["brand"], BRAND)

    def test_gemini_editor_watches_the_whole_source(self):
        h = self.three()
        h.run(used=[{"start": 400.0, "end": 420.0}])
        self.assertEqual(h.calls["proxy"], [{"source": Path("/videos/source.mp4")}])
        kw = h.calls["draft"][0]
        self.assertEqual(kw["role"], "gemini")
        self.assertEqual(kw["source_video"], Path("/proxy/whole.mp4"))
        self.assertEqual(kw["keyframe_frames"], [])
        self.assertEqual(kw["n"], 10)
        self.assertEqual((kw["min_s"], kw["max_s"], kw["max_segments"]), (12.0, 45.0, 6))
        self.assertEqual(kw["playbook_text"], "PLAYBOOK TEXT")
        self.assertEqual(kw["rules"], "OPERATOR RULES")
        self.assertEqual(kw["used"], [{"start": 400.0, "end": 420.0}])
        self.assertEqual(kw["context"], {"shipped_examples": []})
        self.assertEqual(kw["shot_map"]["cuts"], [{"t": 31.0, "score": 45.0}])
        self.assertTrue(kw["has_words"])
        self.assertEqual(h.calls["keyframes"], [])

    def test_long_source_editor_reads_the_log_alone(self):
        h = self.three(probe=4000.0)
        result = h.run(duration=4000.0)
        self.assertEqual(h.calls["proxy"], [])
        self.assertIsNone(h.calls["draft"][0]["source_video"])
        self.assertTrue(any("shot log alone" in n for n in result.notes))

    def test_proxy_failure_degrades_to_the_log(self):
        h = self.three()
        with mock.patch.object(media, "perception_proxy",
                               side_effect=media.MediaError("no ffmpeg")):
            result = h.run()
        self.assertIsNone(h.calls["draft"][0]["source_video"])
        self.assertTrue(any("couldn't watch the video" in n for n in result.notes))
        self.assertEqual(result.outcome, "proposals")

    def test_claude_editor_gets_keyframes_not_video(self):
        for setup in ("hybrid", "claude"):
            with self.subTest(setup):
                h = self.three()
                h.run(setup=setup)
                kw = h.calls["draft"][0]
                self.assertEqual(kw["role"], "claude")
                self.assertIsNone(kw["source_video"])
                self.assertEqual(kw["keyframe_frames"], [(1.0, b"jpeg")])
                self.assertEqual(h.calls["keyframes"], [{"limit": 40}])
                self.assertEqual(h.calls["proxy"], [])
                self.assertEqual(h.calls["watch"][0]["role"], "claude")
                self.assertEqual(h.calls["rank"][0]["role"], "claude")

    def test_section_run(self):
        h = self.harness([make_draft("d1", (110, 130)), make_draft("d2", (200, 220)),
                          make_draft("d3", (250, 270))],
                         {"d1": crit(8), "d2": crit(7), "d3": crit(6)})
        used = [{"start": 150.0, "end": 160.0}]
        result = h.run(window={"start": 100, "end": 300}, section_topic="Chapter 3", used=used)
        kw = h.calls["draft"][0]
        self.assertEqual(kw["n"], 6)
        self.assertEqual(kw["window"], {"start": 100.0, "end": 300.0})
        self.assertEqual(kw["section_topic"], "Chapter 3")
        self.assertEqual(h.calls["build"][0]["window"], {"start": 100.0, "end": 300.0})
        self.assertEqual(h.calls["proxy"], [{"source": Path("/videos/source.mp4")}])
        for call in h.calls["snap"]:
            self.assertEqual((call["floor"], call["horizon"]), (100.0, 300.0))
            self.assertEqual(call["blocked"], used)
        self.assertEqual(len(result.top), 2)               # clips.section_max_clips
        self.assertEqual(result.policy, clip_proposals.policy_version(
            model="engine:gemini:gemini-3.8-flash", max_clips=2, max_segments_per_clip=6,
            min_seconds=12, max_seconds=45, scope="section",
            prompt=f"{getattr(clip_proposals, 'ENGINE_PROMPT_VERSION', 'engine-v1')}:gemini"))

    def test_section_is_clamped_to_the_source(self):
        h = self.three()
        h.run(window={"start": 100, "end": 9999})
        self.assertEqual(h.calls["draft"][0]["window"], {"start": 100.0, "end": 600.0})
        h = self.three()
        with self.assertRaises(engine.EngineError):
            h.run(window={"start": 700, "end": 800})
        self.assertEqual(h.calls["build"], [])

    def test_snap_settings_and_inputs(self):
        h = self.three(settings={"clip_engine.snap.word_window": 0.9,
                                 "clip_engine.snap.min_shot_score": 30})
        h.run()
        call = h.calls["snap"][0]
        self.assertEqual(call["word_window"], 0.9)
        self.assertEqual(call["min_shot_score"], 30.0)
        self.assertEqual(call["picture_window"], 1.0)
        self.assertEqual((call["start_pad"], call["end_pad"], call["flash_guard"]),
                         (0.08, 0.25, 0.4))
        self.assertEqual(call["words"], WORDS)
        self.assertEqual(call["cuts"], [{"t": 31.0, "score": 45.0}])
        self.assertEqual(call["quiet"], [[40.0, 41.0]])
        self.assertEqual((call["floor"], call["horizon"]), (0.0, 600.0))
        self.assertEqual(call["segments"][0]["lead"], "word")

    def test_model_and_policy(self):
        h = self.three()
        result = h.run()
        self.assertEqual(result.model, "engine:gemini:gemini-3.8-flash")
        self.assertEqual(result.policy, clip_proposals.policy_version(
            model="engine:gemini:gemini-3.8-flash", max_clips=3, max_segments_per_clip=6,
            min_seconds=12, max_seconds=45, scope="video",
            prompt=f"{getattr(clip_proposals, 'ENGINE_PROMPT_VERSION', 'engine-v1')}:gemini"))
        h = self.three(settings={"clip_engine.claude.editor_model": "claude-" + "x" * 80})
        result = h.run(setup="hybrid")
        self.assertEqual(len(result.model), 60)
        self.assertTrue(result.model.startswith("engine:hybrid:claude-"))

    def test_duration_probed_when_unknown(self):
        h = self.three(probe=600.0)
        h.run(duration=None)
        self.assertEqual(h.calls["probe"], [Path("/videos/source.mp4")])
        self.assertEqual(h.calls["draft"][0]["duration"], 600.0)

    def test_unreadable_duration_raises_before_spending(self):
        h = self.three(probe=None)
        with self.assertRaises(engine.EngineError):
            h.run(duration=None)
        self.assertEqual(h.calls["build"], [])

    def test_playbook_given_or_loaded(self):
        h = self.three()
        with mock.patch.object(playbook_mod, "render", side_effect=lambda pb: f"N={len(pb)}") as r:
            h.run(playbook={"archetypes": []})
        self.assertEqual(r.call_args[0][0], {"archetypes": []})
        self.assertEqual(h.calls["draft"][-1]["playbook_text"], "N=1")

    def test_credits_and_clip_words_reach_the_critic(self):
        h = self.three()
        h.run()
        watch = {c["id"]: c for c in h.calls["watch"]}
        self.assertEqual(watch["d1"]["credits"], "credits:d1")
        self.assertEqual(watch["d1"]["clip_seconds"], 20.0)
        self.assertEqual(watch["d1"]["render"], Path("/renders/d1.mp4"))
        self.assertEqual(watch["d1"]["role"], "gemini")
        self.assertEqual(watch["d1"]["clip_words"][0], {"word": "Hello", "start": 0.0, "end": 0.4})
        self.assertEqual(watch["d2"]["clip_words"], [{"word": "Later", "start": 5.0, "end": 5.5}])

    def test_videos_without_words_give_the_critic_transcript_lines(self):
        h = self.three()
        h.run(words=[])
        watch = {c["id"]: c for c in h.calls["watch"]}
        self.assertEqual(watch["d1"]["clip_words"],
                         [{"start": 0.0, "end": 4.0, "text": "Hello world."}])
        self.assertFalse(h.calls["draft"][0]["has_words"])

    def test_renders_use_plain_segments_at_proxy_height(self):
        h = self.three(settings={"clip_engine.proxy_height": 240})
        h.run()
        for call in h.calls["render"]:
            self.assertEqual(call["height"], 240)
            self.assertEqual(set(call["segments"][0]), {"start", "end"})

    def test_result_is_json_ready(self):
        h = self.three()
        result = h.run()
        data = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(data["outcome"], "proposals")
        self.assertIsInstance(data["top"][0]["render"], str)
        self.assertEqual(set(data["orders"]), {"editor", "cold", "rank", "final"})
        self.assertAlmostEqual(result.cost_usd, 0.05 + 0.10 + 0.03 + 0.02)
        self.assertEqual(result.tokens["gemini-3.8-flash"], {"input": 100, "output": 10})
        self.assertEqual(result.moments[0]["id"], "m1")
        self.assertEqual(result.shotlog["beats"][0]["topic"], 2)


class RunFiltersTest(_SpendSafe):
    def test_drafts_outside_the_length_bounds_after_snapping_are_dropped(self):
        h = self.three()
        # d2 snaps down to 6 s, under 0.8 x 12 s.
        h.snap_fn = lambda segs: ([dict(segs[0], end=106.0)]
                                  if segs[0]["start"] == 100.0 else segs)
        result = h.run()
        d2 = next(d for d in result.drafts if d["id"] == "d2")
        self.assertEqual(d2["dropped_reason"], "length")
        self.assertEqual(d2["seconds"], 6.0)
        self.assertEqual(d2["editor_segments"], [seg(100, 120)])
        self.assertNotIn("d2", h.rendered())
        self.assertTrue(any("d2 (6.0 s)" in n for n in result.notes))

    def test_snapping_failure_keeps_the_editors_boundaries(self):
        h = self.three()

        def boom(segments, **kw):
            raise ValueError("bad words")

        with mock.patch.object(snap, "snap_segments", new=boom):
            result = h.run()
        self.assertEqual(result.outcome, "proposals")
        self.assertEqual(result.top[0]["segments"], [seg(10, 30)])
        self.assertTrue(any("snapping failed" in n for n in result.notes))

    def test_only_true_duplicates_are_dropped(self):
        h = self.harness(
            [make_draft("d1", (10, 30), story_key="s"),
             make_draft("d2", (10.3, 30.2), story_key="s"),                 # duplicate of d1
             {**make_draft("d3", story_key="s"), "segments": [seg(20, 30), seg(10, 20)]},
             {**make_draft("d4", story_key="s"), "segments": [seg(10, 18), seg(20, 30)]},
             make_draft("d5", (100, 120))],
            {"d1": crit(8), "d3": crit(7), "d4": crit(6), "d5": crit(5)})
        result = h.run()
        drafts = {d["id"]: d for d in result.drafts}
        self.assertEqual(drafts["d2"]["dropped_reason"], "duplicate")
        self.assertEqual(drafts["d2"]["duplicate_of"], "d1")
        self.assertEqual(sorted(h.rendered()), ["d1", "d3", "d4", "d5"])
        self.assertEqual(result.orders["editor"], ["d1", "d3", "d4", "d5"])

    def test_dismissed_moments_stay_dismissed(self):
        h = self.three()
        rejected = [{"segments": [{"start": 9.0, "end": 31.0}], "reason": "not_clipworthy"}]
        result = h.run(rejected=rejected)
        d1 = next(d for d in result.drafts if d["id"] == "d1")
        self.assertEqual(d1["dropped_reason"], "rejected before")
        self.assertNotIn("d1", h.rendered())
        self.assertTrue(any("dismissed" in n and "not_clipworthy" in n for n in result.notes))

    def test_boundary_complaints_allow_moved_boundaries(self):
        h = self.three()
        rejected = [{"segments": [{"start": 13.0, "end": 30.0}], "reason": "bad_boundaries"}]
        result = h.run(rejected=rejected)
        self.assertIn("d1", h.rendered())                   # start moved 3 s
        self.assertEqual(result.top[0]["id"], "d1")

    def test_nothing_left_after_filtering_is_a_failure(self):
        h = self.harness([make_draft("d1", (10, 30))], {"d1": crit(8)})
        with self.assertRaises(engine.EngineError) as caught:
            h.run(rejected=[{"segments": [{"start": 10, "end": 30}], "reason": ""}])
        self.assertIn("rejected before", str(caught.exception))
        self.assertEqual(h.rendered(), [])
        self.assertTrue(caught.exception.notes)

    def test_failed_renders_drop_their_drafts(self):
        h = self.three(render_fail={"d2"})
        result = h.run()
        d2 = next(d for d in result.drafts if d["id"] == "d2")
        self.assertEqual(d2["dropped_reason"], "render failed")
        self.assertIn("ffmpeg exploded", d2["render_error"])
        self.assertNotIn("watch:d2", h.events)
        self.assertEqual([d["id"] for d in result.top], ["d1", "d3"])
        self.assertTrue(any("couldn't be rendered" in n for n in result.notes))

    def test_no_render_at_all_is_a_failure(self):
        h = self.three(render_fail={"d1", "d2", "d3"})
        with self.assertRaises(engine.EngineError) as caught:
            h.run()
        self.assertIn("no draft could be rendered", str(caught.exception))
        self.assertEqual([e for e in h.events if e.startswith("watch:")], [])

    def test_unreviewed_drafts_are_kept_and_still_compared(self):
        h = self.three()
        h.critiques["d2"] = crit(0, error="GeminiQuota: daily quota")
        result = h.run()
        d2 = next(d for d in result.drafts if d["id"] == "d2")
        self.assertEqual(d2["dropped_reason"], "")
        self.assertEqual(d2["score"], -1e9)
        self.assertIn("d2", [d["id"] for d in h.calls["rank"][0]["drafts"]])
        self.assertTrue(any("weren't reviewed" in n for n in result.notes))

    def test_a_critic_that_raises_counts_as_not_reviewed(self):
        h = self.three()
        real = h.cold_watch

        def flaky(**kw):
            if Path(kw["render"]).stem == "d3":
                raise ProviderError("socket closed", usage=usage(0.01))
            return real(**kw)

        with mock.patch.object(critic, "cold_watch", new=flaky):
            result = h.run()
        d3 = next(d for d in result.drafts if d["id"] == "d3")
        self.assertIn("socket closed", d3["critique"]["error"])
        self.assertEqual(d3["critique"]["usage"]["cost_usd"], 0.01)
        self.assertEqual(d3["dropped_reason"], "")


class RunTopicGateTest(_SpendSafe):
    def setUp(self):
        super().setUp()
        self.log = {"complete": True, "cost_usd": 0.05, "people": [],
                    "beats": [{"start": 0, "end": 100, "topic": 2},
                              {"start": 100, "end": 300, "topic": 0}]}

    def test_dropped_only_when_critic_and_log_agree(self):
        h = self.harness(
            [make_draft("d1", (10, 30)), make_draft("d2", (110, 130)),
             make_draft("d3", (150, 170))],
            {"d1": crit(8, topic="none", topic_note="about a comet"),
             "d2": crit(7, topic="incidental", topic_note="climate in passing"),
             "d3": crit(6)},
            shotlog=self.log)
        result = h.run()
        drafts = {d["id"]: d for d in result.drafts}
        self.assertEqual(drafts["d1"]["dropped_reason"], "")
        self.assertEqual(drafts["d1"]["topic_doubt"], "about a comet")
        self.assertEqual(drafts["d2"]["dropped_reason"], "off topic")
        self.assertIn("climate in passing", drafts["d2"]["topic_doubt"])
        self.assertEqual(drafts["d3"]["dropped_reason"], "")      # critic says central
        self.assertEqual(drafts["d3"]["topic_doubt"], "")
        self.assertEqual(result.orders["cold"], ["d1", "d3"])
        self.assertEqual([d["id"] for d in result.top], ["d1", "d3"])
        self.assertEqual(result.top[0]["topic_doubt"], "about a comet")

    def test_all_off_topic_is_an_empty_topic_answer(self):
        h = self.harness([make_draft("d1", (110, 130)), make_draft("d2", (200, 220))],
                         {"d1": crit(8, topic="none"), "d2": crit(7, topic="incidental")},
                         shotlog=self.log)
        result = h.run()
        self.assertEqual(result.outcome, "empty_topic")
        self.assertEqual(result.top, [])
        self.assertNotIn("rank", h.events)
        self.assertEqual(h.calls["revise"], [])
        self.assertNotIn("Ranking", h.labels)

    def test_an_outage_is_never_an_off_topic_verdict(self):
        h = self.harness([make_draft("d1", (110, 130))],
                         {"d1": crit(0, topic="none", error="quota")}, shotlog=self.log)
        result = h.run()
        self.assertEqual(result.outcome, "proposals")
        self.assertEqual(result.top[0]["id"], "d1")


class RunRankingTest(_SpendSafe):
    def pool(self) -> Harness:
        return self.harness(
            [make_draft("a1", (10, 30), story_key="A"),
             make_draft("a2", (12, 32), story_key="A"),
             make_draft("b1", (100, 120), story_key="B"),
             make_draft("c1", (200, 220), story_key="C"),
             make_draft("d1", (300, 320), story_key="D")],
            {"a1": crit(8), "a2": crit(9), "b1": crit(6),
             "c1": crit(9.5, makes_sense=False), "d1": crit(0, error="quota")},
            settings={"clip_engine.revise_top": 0})

    def test_compare_set_goes_to_the_judge_with_its_files(self):
        h = self.pool()
        h.run()
        kw = h.calls["rank"][0]
        self.assertEqual([d["id"] for d in kw["drafts"]], ["a2", "a1", "b1", "d1"])
        self.assertEqual(kw["renders"], {i: Path(f"/renders/{i}.mp4")
                                         for i in ("a2", "a1", "b1", "d1")})
        self.assertEqual(kw["credits"]["b1"], "credits:b1")
        self.assertEqual(kw["scores"], {"a2": 9.0, "a1": 8.0, "b1": 6.0, "d1": -1e9})
        self.assertEqual(kw["playbook_text"], "PLAYBOOK TEXT")
        self.assertEqual(kw["role"], "gemini")

    def test_taste_examples_without_a_file_are_left_out(self):
        h = self.pool()
        h.example_files = {1: Path("/taste/1-360.mp4")}
        examples = [{"post_pk": 1, "title": "hit"}, {"post_pk": 2, "title": "flop"}]
        result = h.run(examples=examples)
        kw = h.calls["rank"][0]
        self.assertEqual(kw["examples"], [examples[0]])
        self.assertEqual(kw["example_files"], [Path("/taste/1-360.mp4")])
        self.assertEqual([c["height"] for c in h.calls["example_file"]], [360, 360])
        self.assertTrue(any("1 of 2 taste examples" in n for n in result.notes))

    def test_ranking_merged_with_the_rest(self):
        h = self.pool()
        h.rank_result = ["b1", "zz", "a2"]
        result = h.run()
        self.assertEqual(result.orders["rank"], ["b1", "a2", "a1", "d1", "c1"])
        self.assertEqual(result.orders["cold"], ["c1", "a2", "a1", "b1", "d1"])
        # a1 overlaps a2 too much to sit beside it in the top three.
        self.assertEqual([d["id"] for d in result.top], ["b1", "a2", "d1"])

    def test_ranking_failure_falls_back_to_cold_scores(self):
        h = self.pool()
        h.rank_result = ProviderError("judge down", usage=usage(0.02))
        meter = engine.CostMeter()
        result = h.run(meter=meter)
        self.assertEqual(result.orders["rank"], ["a2", "a1", "b1", "d1", "c1"])
        self.assertTrue(any("Ranking failed" in n for n in result.notes))
        self.assertAlmostEqual(meter.by_stage["rank"], 0.02)      # metered once by the fake

    def test_a_lone_candidate_skips_the_judge(self):
        h = self.harness([make_draft("d1", (10, 30)), make_draft("d2", (100, 120))],
                         {"d1": crit(8), "d2": crit(7, makes_sense=False)})
        result = h.run()
        self.assertNotIn("rank", h.events)
        self.assertEqual(result.orders["rank"], ["d1", "d2"])

    def test_compare_top_zero_ranks_by_score(self):
        h = self.three(settings={"clip_engine.compare_top": 0})
        result = h.run()
        self.assertNotIn("rank", h.events)
        self.assertEqual(result.orders["rank"], ["d1", "d2", "d3"])


class RunRevisionTest(_SpendSafe):
    def two(self, **attrs) -> Harness:
        h = self.harness([make_draft("d1", (10, 30)), make_draft("d2", (100, 120))],
                         {"d1": crit(8), "d2": crit(6), "d1r": crit(8.5)}, **attrs)
        h.revisions = {"d1": {"id": "d1r", "segments": [seg(9, 29, "picture")],
                              "change": "opens on the face", "archetype": "voice"}}
        return h

    def test_revision_adopted_when_it_wins_every_check(self):
        h = self.two()
        result = h.run()
        self.assertEqual([d["id"] for d in result.top], ["d1r", "d2"])
        top = result.top[0]
        self.assertEqual((top["origin"], top["parent"]), ("revision", "d1"))
        self.assertEqual(top["render"], "/renders/d1r.mp4")
        self.assertEqual(top["credits"], "credits:d1r")
        self.assertEqual(top["score"], 8.5)
        self.assertEqual(top["rank"], 1)
        self.assertEqual(top["segments"], [seg(9, 29, "picture")])
        original = next(d for d in result.drafts if d["id"] == "d1")
        self.assertEqual((original["dropped_reason"], original["revised_to"]), ("revised", "d1r"))
        self.assertEqual(result.orders["rank"], ["d1", "d2"])
        self.assertEqual(result.orders["final"], ["d1r", "d2"])
        prefer = h.calls["prefer"][0]
        self.assertEqual((prefer["a"]["id"], prefer["b"]["id"]), ("d1", "d1r"))
        self.assertEqual(prefer["render_a"], Path("/renders/d1.mp4"))
        self.assertEqual(prefer["render_b"], Path("/renders/d1r.mp4"))
        self.assertEqual(prefer["intent"], "voice: open on it")
        self.assertEqual(prefer["credits"], {"a": "credits:d1", "b": "credits:d1r"})
        self.assertEqual(prefer["role"], "gemini")
        revise = h.calls["revise"][0]
        self.assertEqual(revise["critique"]["score"], 8)
        self.assertEqual(revise["source_video"], Path("/proxy/whole.mp4"))
        self.assertEqual(revise["playbook_text"], "PLAYBOOK TEXT")
        self.assertEqual(revise["rules"], "OPERATOR RULES")
        self.assertTrue(any("adopted: opens on the face" in n for n in result.notes))

    def test_judges_hear_each_drafts_words_and_the_result_drops_them(self):
        """A Claude judge sees frames only, so rank and prefer get every
        draft's words on its clip clock (what the cold critic heard). They are
        derived and bulky, so the result keeps none of them."""
        from app.subtitles import slice_source_words

        h = self.two()
        harness_rank, harness_prefer = critic.rank, critic.prefer
        seen: dict = {}

        def rank(**kw):
            seen["rank"] = {d["id"]: copy.deepcopy(d.get("clip_words")) for d in kw["drafts"]}
            return harness_rank(**kw)

        def prefer(**kw):
            seen["prefer"] = {k: copy.deepcopy(kw[k].get("clip_words")) for k in ("a", "b")}
            return harness_prefer(**kw)

        with mock.patch.object(critic, "rank", new=rank), \
                mock.patch.object(critic, "prefer", new=prefer):
            result = h.run()
        expected = {did: slice_source_words(WORDS, segments)
                    for did, segments in (("d1", [seg(10, 30)]), ("d2", [seg(100, 120)]),
                                          ("d1r", [seg(9, 29, "picture")]))}
        self.assertEqual(seen["rank"], {"d1": expected["d1"], "d2": expected["d2"]})
        self.assertEqual(seen["prefer"], {"a": expected["d1"], "b": expected["d1r"]})
        self.assertTrue(expected["d1"] and expected["d2"])   # the case has words to pass
        heard = {c["id"]: c["clip_words"] for c in h.calls["watch"]}
        self.assertEqual(heard["d1"], expected["d1"])         # the same words the critic heard
        self.assertEqual([d["id"] for d in result.top], ["d1r", "d2"])
        for d in [*result.drafts, *result.top]:
            self.assertNotIn("clip_words", d, d["id"])
        self.assertNotIn("clip_words", json.dumps(result.to_dict()))

    def test_a_revision_never_shadows_another_drafts_id(self):
        h = self.two()
        h.revisions["d1"]["id"] = "d2"
        h.critiques["d1r3"] = crit(8.5)
        real = h.render

        def render(source, segments, **kw):
            if segments == [{"start": 9.0, "end": 29.0}]:
                h.calls["render"].append({"id": "d1r3"})
                return Path("/renders/d1r3.mp4")
            return real(source, segments, **kw)

        with mock.patch.object(media, "render_cached", new=render):
            result = h.run()
        ids = [d["id"] for d in result.drafts]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result.top[0]["id"], "d1r3")
        self.assertEqual(result.top[0]["parent"], "d1")

    def test_equal_cold_score_may_still_win(self):
        h = self.two()
        h.critiques["d1r"] = crit(8)
        result = h.run()
        self.assertEqual(result.top[0]["id"], "d1r")

    def test_judge_keeps_the_original(self):
        h = self.two(prefer_choice="a")
        result = h.run()
        self.assertEqual(result.top[0]["id"], "d1")
        revision = next(d for d in result.drafts if d["id"] == "d1r")
        self.assertEqual(revision["dropped_reason"], "revision not used")
        self.assertEqual(revision["revision_note"], "the judge kept the original")

    def test_same_counts_for_the_original(self):
        h = self.two(prefer_choice="same")
        self.assertEqual(h.run().top[0]["id"], "d1")

    def test_a_revision_that_breaks_something_never_reaches_the_judge(self):
        for name, critique, expected in (
                ("ending", crit(9, ends_well=False), "no longer ends well"),
                ("sense", crit(9, makes_sense=False), "no longer makes sense"),
                ("topic", crit(9, topic="none"), "drifted off topic"),
                ("score", crit(7.5), "scored lower"),
                ("outage", crit(0, error="quota"), "couldn't review")):
            with self.subTest(name):
                h = self.two()
                h.critiques["d1r"] = critique
                result = h.run()
                self.assertEqual(h.calls["prefer"], [])
                self.assertEqual(result.top[0]["id"], "d1")
                revision = next(d for d in result.drafts if d["id"] == "d1r")
                self.assertIn(expected, revision["revision_note"])

    def test_editor_keeping_its_cut(self):
        h = self.two()
        h.revisions = {"d1": None}
        result = h.run()
        self.assertEqual(result.top[0]["id"], "d1")
        self.assertNotIn("render:d1r", h.events)
        self.assertTrue(any("kept its cut" in n for n in result.notes))

    def test_revision_failures_keep_the_original(self):
        cases = {
            "editor raises": lambda h: h.revisions.update(d1=ProviderError("editor down")),
            "render fails": lambda h: h.render_fail.add("d1r"),
        }
        for name, setup in cases.items():
            with self.subTest(name):
                h = self.two()
                setup(h)
                result = h.run()
                self.assertEqual(result.outcome, "proposals")
                self.assertEqual(result.top[0]["id"], "d1")
                self.assertEqual(h.calls["prefer"], [])

    def test_failed_revision_call_is_metered(self):
        h = self.two()
        h.revisions = {"d1": ProviderError("truncated", usage=usage(0.04))}
        meter = engine.CostMeter()
        result = h.run(meter=meter)
        self.assertAlmostEqual(meter.by_stage["revise"], 0.04)
        self.assertTrue(any("Revising d1 failed" in n for n in result.notes))

    def test_unchanged_after_snapping_is_not_rendered(self):
        h = self.two()
        h.snap_fn = lambda segs: [dict(s, start=10.0, end=30.0) if s["start"] == 9.0 else s
                                  for s in segs]
        result = h.run()
        self.assertNotIn("render:d1r", h.events)
        revision = next(d for d in result.drafts if d["id"] == "d1r")
        self.assertIn("same as the original", revision["revision_note"])

    def test_revision_into_dismissed_material_is_not_used(self):
        h = self.two()
        h.revisions["d1"]["segments"] = [seg(40, 60)]
        h.critiques["d1r"] = crit(9)
        result = h.run(rejected=[{"segments": [{"start": 40, "end": 60}], "reason": ""}])
        self.assertEqual(result.top[0]["id"], "d1")
        self.assertNotIn("render:d1r", h.events)

    def test_revision_budget_counts_cached_perception(self):
        settings = {"clip_engine.max_usd_per_video": 0.30}
        h = self.two(settings=settings, cached_perception=0.25)
        result = h.run()
        self.assertEqual(h.calls["revise"], [])
        self.assertTrue(any("Revision skipped" in n for n in result.notes))
        self.assertAlmostEqual(result.by_stage["perceive_cached"], 0.25)
        self.assertAlmostEqual(result.cost_usd, 0.10 + 0.02 + 0.02)
        # The same video with nothing cached is under the line, so it revises.
        h = self.two(settings=settings)
        h.run()
        self.assertEqual(len(h.calls["revise"]), 1)

    def test_revise_top_zero_skips_the_round(self):
        h = self.two(settings={"clip_engine.revise_top": 0})
        result = h.run()
        self.assertEqual(h.calls["revise"], [])
        self.assertNotIn("Revising", h.labels)
        self.assertNotIn("revise", result.timings)

    def test_revise_top_two_revises_the_two_best(self):
        h = self.two(settings={"clip_engine.revise_top": 2})
        h.revisions["d2"] = None
        h.run()
        self.assertEqual([c["draft"]["id"] for c in h.calls["revise"]], ["d1", "d2"])

    def test_an_unreviewed_draft_is_not_revised(self):
        h = self.two(rank_result=["d1", "d2"])
        h.critiques["d1"] = crit(0, error="quota")
        result = h.run()
        self.assertEqual(h.calls["revise"], [])
        self.assertTrue(any("never reviewed" in n for n in result.notes))


class RunTopTest(_SpendSafe):
    def test_top_skips_heavy_overlap_and_numbers_its_picks(self):
        h = self.harness([make_draft("d1", (10, 30)), make_draft("d2", (12, 32)),
                          make_draft("d3", (100, 120)), make_draft("d4", (200, 220))],
                         {"d1": crit(9), "d2": crit(8), "d3": crit(7), "d4": crit(6)},
                         settings={"clip_engine.revise_top": 0})
        result = h.run()
        self.assertEqual([d["id"] for d in result.top], ["d1", "d3", "d4"])
        self.assertEqual([d["rank"] for d in result.top], [1, 2, 3])
        for d in result.top:
            self.assertIsInstance(d["render"], str)
            self.assertIn("critique", d)
            self.assertTrue(d["credits"].startswith("credits:"))
        self.assertTrue(any("d2 skipped" in n for n in result.notes))
        self.assertEqual(result.orders["final"], ["d1", "d3", "d4"])
        # Top items are copies: the rank isn't written back into the drafts list.
        self.assertNotIn("rank", next(d for d in result.drafts if d["id"] == "d1"))

    def test_keep_override(self):
        h = self.three()
        result = h.run(keep=1)
        self.assertEqual([d["id"] for d in result.top], ["d1"])

    def test_keep_setting(self):
        h = self.three(settings={"clip_engine.keep": 2})
        self.assertEqual(len(h.run().top), 2)

    def test_orders_for_the_bench(self):
        h = self.three(rank_result=["d3", "d2", "d1"], settings={"clip_engine.revise_top": 0})
        h.critiques = {"d1": crit(5), "d2": crit(9), "d3": crit(7)}
        result = h.run()
        self.assertEqual(result.orders, {"editor": ["d1", "d2", "d3"],
                                         "cold": ["d2", "d3", "d1"],
                                         "rank": ["d3", "d2", "d1"],
                                         "final": ["d3", "d2", "d1"]})


class CollaboratorContractTest(_SpendSafe):
    """Every call the engine makes binds to the REAL collaborator's
    signature (keyword names, required arguments), so the fakes above can't
    drift from the modules they stand in for."""

    def assertBinds(self, name, *args, **kwargs):
        fn = REAL[name]
        try:
            inspect.signature(fn).bind(*args, **kwargs)
        except TypeError as exc:
            self.fail(f"engine's call to {name} doesn't fit {fn.__module__}: {exc}")

    def run_everything(self, setup):
        h = self.harness([make_draft("d1", (10, 30)), make_draft("d2", (100, 120))],
                         {"d1": crit(8), "d2": crit(6), "d1r": crit(9)})
        h.revisions = {"d1": {"id": "d1r", "segments": [seg(9, 29)], "change": "x"}}
        h.example_files = {1: Path("/taste/1.mp4")}
        h.run(setup=setup, duration=None, examples=[{"post_pk": 1, "title": "hit"}])
        self.assertEqual(len(h.calls["prefer"]), 1)       # the whole pipeline ran
        return h

    def test_calls_fit_the_real_signatures(self):
        for setup in ("gemini", "hybrid"):
            with self.subTest(setup):
                h = self.run_everything(setup)
                for name in ("build", "draft", "revise", "rank", "prefer"):
                    self.assertBinds(name, **h.calls[name][0])
                watch = dict(h.calls["watch"][0])
                watch.pop("id")
                self.assertBinds("cold_watch", **watch)
                for call in h.calls["snap"]:
                    kwargs = dict(call)
                    self.assertBinds("snap_segments", kwargs.pop("segments"), **kwargs)
                for call in h.calls["render"]:
                    self.assertBinds("render_cached", call["source"], call["segments"],
                                     height=call["height"])
                self.assertBinds("shot_map", h.calls["shot_map"][0]["source"])
                self.assertBinds("probe_duration", h.calls["probe"][0])
                self.assertBinds("example_file", {"post_pk": 1}, make_settings(), height=360)
                self.assertBinds("credits_for", {}, [seg(1, 2)])
                self.assertBinds("entries_for", WORDS, TRANSCRIPT)
                self.assertBinds("cold_score", crit(5), clip_seconds=20.0, archetype="voice")
                self.assertBinds("open_rule", PLAYBOOK, "voice")
                self.assertBinds("render", PLAYBOOK)
                if setup == "gemini":
                    self.assertBinds("perception_proxy", h.calls["proxy"][0]["source"])
                else:
                    self.assertBinds("keyframes", Path("/v.mp4"), {}, {}, settings=make_settings(),
                                     **h.calls["keyframes"][0])


class RunOutcomesAndErrorsTest(_SpendSafe):
    def test_no_drafts_is_an_empty_drafts_answer(self):
        h = self.harness([])
        result = h.run()
        self.assertEqual(result.outcome, "empty_drafts")
        self.assertEqual(result.top, [])
        self.assertEqual(h.labels, ["Logging the video", "Drafting clips"])
        self.assertEqual(h.rendered(), [])
        self.assertTrue(result.notes)
        self.assertTrue(result.policy and result.model)

    def test_perceiver_failure_raises_and_meters_the_failed_call(self):
        h = self.three(perceive_error=ProviderError("MAX_TOKENS", usage=usage(0.07)))
        meter = engine.CostMeter()
        with self.assertRaises(engine.EngineError) as caught:
            h.run(meter=meter)
        self.assertIn("perceiver failed", str(caught.exception))
        self.assertAlmostEqual(meter.total, 0.07)
        self.assertEqual(h.calls["draft"], [])

    def test_a_failed_call_is_never_counted_twice(self):
        h = self.three(perceive_error=ProviderError("MAX_TOKENS", usage=usage(0.05)),
                       meter_before_error=True)
        meter = engine.CostMeter()
        with self.assertRaises(engine.EngineError):
            h.run(meter=meter)
        self.assertAlmostEqual(meter.total, 0.05)

    def test_perceiver_that_logged_nothing(self):
        h = self.three()
        h.shotlog = {"beats": [], "complete": True}
        with self.assertRaises(engine.EngineError) as caught:
            h.run()
        self.assertIn("logged nothing", str(caught.exception))

    def test_incomplete_log_is_noted_not_fatal(self):
        h = self.three()
        h.shotlog["complete"] = False
        result = h.run()
        self.assertEqual(result.outcome, "proposals")
        self.assertTrue(any("couldn't be logged" in n for n in result.notes))

    def test_shot_map_failure_degrades_to_words(self):
        h = self.three()
        h.shot_map = {"cuts": [], "quiet": [], "error": "ffmpeg not found"}
        result = h.run()
        self.assertTrue(any("No shot map (ffmpeg not found)" in n for n in result.notes))
        self.assertEqual(h.calls["snap"][0]["cuts"], [])
        self.assertEqual(result.outcome, "proposals")

    def test_drafting_failure_raises_and_meters_a_wrapped_error(self):
        h = self.three()
        try:
            raise ProviderError("blocked", usage=usage(0.08))
        except ProviderError as cause:
            try:
                raise ValueError("bad JSON") from cause
            except ValueError as wrapped:
                h.draft_error = wrapped
        meter = engine.CostMeter()
        with self.assertRaises(engine.EngineError) as caught:
            h.run(meter=meter)
        self.assertIn("drafting failed", str(caught.exception))
        self.assertAlmostEqual(meter.by_stage["draft"], 0.08)

    def test_hard_cap_stops_the_run(self):
        h = self.three(settings={"clip_engine.hard_cap_usd": 0.5})
        h.costs["draft"] = 0.6
        meter = engine.CostMeter()
        with self.assertRaises(engine.EngineError) as caught:
            h.run(meter=meter)
        self.assertIn("hard cap", str(caught.exception))
        self.assertEqual(h.rendered(), [])
        self.assertAlmostEqual(meter.total, 0.65)

    def test_cancel_stops_between_stages(self):
        h = self.three()
        cancel = threading.Event()

        def progress(label):
            h.labels.append(label)
            if label == "Rendering drafts":
                cancel.set()

        with self.assertRaises(util.Cancelled):
            h.run(progress=progress, cancel=cancel)
        self.assertEqual([e for e in h.events if e.startswith(("render:", "watch:"))], [])

    def test_cancel_during_drafting_stops_before_rendering(self):
        h = self.three()
        cancel = threading.Event()
        real = h.draft

        def draft_then_cancel(**kw):
            out = real(**kw)
            cancel.set()
            return out

        with mock.patch.object(editor, "draft", new=draft_then_cancel):
            with self.assertRaises(util.Cancelled):
                h.run(cancel=cancel)
        self.assertEqual(h.rendered(), [])

    def test_expired_deadline_spends_nothing(self):
        h = self.three()
        with self.assertRaises(util.Cancelled):
            h.run(deadline=time.monotonic() - 1)
        self.assertEqual(h.calls["build"], [])

    def test_deadline_passing_mid_run(self):
        h = self.three()
        real = h.build

        def slow_build(**kw):
            out = real(**kw)
            time.sleep(0.05)
            return out

        with mock.patch.object(shotlog, "build", new=slow_build):
            with self.assertRaises(util.Cancelled):
                h.run(deadline=time.monotonic() + 0.02)
        self.assertEqual(h.calls["draft"], [])

    def test_transcript_setup_has_no_engine_pipeline(self):
        h = self.three()
        for setup in ("transcript", "nonsense"):
            with self.subTest(setup), self.assertRaises(engine.EngineError):
                h.run(setup=setup)
        self.assertEqual(h.calls["build"], [])

    def test_broken_progress_callback_is_harmless(self):
        h = self.three()

        def broken(label):
            raise RuntimeError("status line down")

        self.assertEqual(h.run(progress=broken).outcome, "proposals")

    def test_engine_error_carries_the_notes(self):
        h = self.three(render_fail={"d1", "d2", "d3"})
        h.shot_map = {"error": "no ffmpeg"}
        with self.assertRaises(engine.EngineError) as caught:
            h.run()
        self.assertTrue(any("No shot map" in n for n in caught.exception.notes))
        self.assertEqual(caught.exception.renders, [])

    def test_a_run_stopped_after_rendering_hands_back_its_renders(self):
        """Nobody will be shown them: the caller deletes them (jobs does)."""
        # perceive $0.05 + draft $0.10, then three critiques at $0.01 tip it over.
        h = self.three(settings={"clip_engine.hard_cap_usd": 0.16})
        with self.assertRaises(engine.EngineError) as caught:
            h.run()
        self.assertIn("hard cap", str(caught.exception))
        self.assertEqual(sorted(caught.exception.renders),
                         ["/renders/d1.mp4", "/renders/d2.mp4", "/renders/d3.mp4"])
        self.assertEqual(h.calls["rank"], [])

    def test_a_deadline_during_rendering_hands_back_the_renders_too(self):
        h = self.three()
        real = h.render

        def slow_render(*args, **kwargs):
            out = real(*args, **kwargs)
            time.sleep(1.0)
            return out

        with mock.patch.object(media, "render_cached", new=slow_render):
            with self.assertRaises(util.Cancelled) as caught:
                h.run(deadline=time.monotonic() + 0.5)
        self.assertEqual(sorted(caught.exception.renders),
                         ["/renders/d1.mp4", "/renders/d2.mp4", "/renders/d3.mp4"])
        self.assertEqual(h.calls["watch"], [])
        self.assertIsInstance(caught.exception.notes, list)

    def real_editor(self, h: Harness, answer: dict) -> None:
        """The real ``editor.draft`` (cleaning included), its model answering ``answer``."""
        def draft(**kw):
            h.events.append("draft")
            with mock.patch.object(gemini, "generate_json",
                                   return_value=(copy.deepcopy(answer),
                                                 usage(0.08, model="gemini-3.8-flash"))), \
                    mock.patch.object(gemini, "video_parts", return_value=[{"fake": "video"}]):
                return REAL["draft"](**kw)

        p = mock.patch.object(editor, "draft", new=draft)
        p.start()
        self.addCleanup(p.stop)

    @staticmethod
    def raw_draft(i, *spans):
        return {"id": f"d{i}", "moment_id": "m1", "archetype": "voice", "lens": "person",
                "story": f"story {i}", "story_key": f"k{i}", "hook": "h", "payoff": "p",
                "segments": [{"start": a, "end": b, "lead": "word"} for a, b in spans],
                "why": "w", "risk": "", "draft_caption": "c", "challenges_rules": ""}

    def test_drafts_lost_to_cleaning_are_a_failure_not_nothing_worth_drafting(self):
        """The editor wrote drafts and its own cleaning dropped every one: a
        mechanical loss, which must fall back like any other — not the
        editorial answer "empty_drafts" that leaves an archived video with no
        suggestions at all."""
        moment = {"id": "m1", "start": 0, "end": 600, "kind": "person", "why": "x"}
        cases = [
            # 60 s each; max_seconds 45 allows up to 58.5 s.
            ("too long", {"moments": [moment],
                          "drafts": [self.raw_draft(i, (30.0 * i, 30.0 * i + 60.0))
                                     for i in range(10)]}, None),
            # A section pass whose drafts all fall outside the section.
            ("outside the section", {"moments": [],
                                     "drafts": [self.raw_draft(i, (300.0 + 20 * i, 330.0 + 20 * i))
                                                for i in range(6)]},
             {"start": 60.0, "end": 180.0}),
        ]
        for label, answer, window in cases:
            with self.subTest(label):
                h = self.harness()
                self.real_editor(h, answer)
                meter = engine.CostMeter()
                with self.assertRaises(engine.EngineError) as caught:
                    h.run(meter=meter, window=window)
                self.assertIn("drafting failed", str(caught.exception))
                self.assertAlmostEqual(meter.by_stage["draft"], 0.08)   # charged once
                self.assertEqual(h.rendered(), [])

    def test_an_editor_that_drafted_nothing_is_still_an_answer(self):
        h = self.harness()
        self.real_editor(h, {"moments": [], "drafts": []})
        result = h.run()
        self.assertEqual(result.outcome, "empty_drafts")
        self.assertEqual(result.top, [])


class PerceptionLimitsTest(_SpendSafe):
    """The hard cap, the deadline and a cancel reach INSIDE perception: the
    REAL ``shotlog.build`` logs an hour-long source window by window, only its
    Gemini calls faked. The checkpoint used to run once the whole log was
    back, so every window (and every split of a runaway answer) was paid for
    before the cap was even looked at: $12.75 against a $4 cap for a
    four-hour hearing, 91 calls for an hour of answers that all ran long."""

    COST = 0.25          # what each window call bills
    HOUR = 3600.0        # 13 windows of ~300 s

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.truncate = False
        self.on_call = None
        for patcher in (
            mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": tmp.name}),
            mock.patch.object(gemini, "generate_json", new=self.generate),
            mock.patch.object(gemini, "video_parts", new=self.video_parts),
            mock.patch.object(gemini, "delete_file", new=lambda uri: None),
            mock.patch.object(media, "drop", new=lambda paths: None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def video_parts(items, *, inline_budget_bytes=None):
        return [{"fileData": {"mimeType": "video/mp4", "fileUri": "https://x/v1beta/files/p"}}]

    def generate(self, model, *, system, parts, schema, label="", **kw):
        with self.lock:
            self.calls.append(label)
            n = len(self.calls)
        if self.on_call is not None:
            self.on_call(n)
        # 20k output tokens: long, but not so long it reads as a loop, so a
        # truncated window is split all the way down (1 + 2 + 4 calls).
        usage = {"model": model, "input_tokens": 20000, "output_tokens": 20000,
                 "cost_usd": self.COST}
        if self.truncate:
            raise gemini.GeminiTruncated("MAX_TOKENS", usage=usage)
        beat = {"start": "00:00", "end": "05:00", "shot": "interview", "on_screen": "a hearing",
                "face": "close", "speaker": "", "line": "", "sound": "", "energy": 3,
                "visual": 3, "topic": 2, "note": ""}
        return ({"summary": "a hearing", "format": "hearing", "structure": "", "people": [],
                 "beats": [beat], "highlights": []}, usage)

    def harness_with_the_real_shot_log(self, **settings) -> Harness:
        h = self.three(settings=settings)
        p = mock.patch.object(shotlog, "build", new=REAL["build"])
        p.start()
        self.addCleanup(p.stop)
        return h

    def test_the_hard_cap_stops_a_long_source_mid_perception(self):
        h = self.harness_with_the_real_shot_log(**{"clip_engine.hard_cap_usd": 1.0})
        meter = engine.CostMeter()
        with self.assertRaises(engine.EngineError) as caught:
            h.run(duration=self.HOUR, meter=meter)
        message = str(caught.exception)
        self.assertIn("past the hard cap", message)
        self.assertNotIn("perceiver failed", message)       # the cap's own words, not reworded
        # At most four $0.25 calls fit under $1.00, plus whatever the other
        # workers had in flight when the cap was crossed — not all 13 windows.
        self.assertLessEqual(len(self.calls), 4 + shotlog._PART_WORKERS)
        self.assertLessEqual(meter.total, 1.0 + shotlog._PART_WORKERS * self.COST + 1e-9)
        self.assertEqual(h.calls["draft"], [])

    def test_answers_that_all_run_long_cannot_split_past_the_cap(self):
        h = self.harness_with_the_real_shot_log(**{"clip_engine.hard_cap_usd": 1.0})
        self.truncate = True
        meter = engine.CostMeter()
        with self.assertRaises(engine.EngineError) as caught:
            h.run(duration=self.HOUR, meter=meter)
        self.assertIn("past the hard cap", str(caught.exception))
        self.assertLessEqual(len(self.calls), 4 + shotlog._PART_WORKERS)   # was 13 x 7 = 91
        self.assertEqual(h.calls["draft"], [])

    def test_the_deadline_stops_perception_between_windows(self):
        h = self.harness_with_the_real_shot_log()
        self.on_call = lambda n: time.sleep(0.6)
        meter = engine.CostMeter()
        with self.assertRaises(util.Cancelled) as caught:
            h.run(duration=self.HOUR, meter=meter, deadline=time.monotonic() + 0.4)
        self.assertIn("out of time", str(caught.exception))
        self.assertLessEqual(len(self.calls), shotlog._PART_WORKERS)   # the first round only
        self.assertEqual(h.calls["draft"], [])

    def test_a_cancel_stops_perception_between_windows(self):
        h = self.harness_with_the_real_shot_log()
        cancel = threading.Event()

        def first_call_cancels(n):
            if n == 1:
                cancel.set()
            time.sleep(0.05)

        self.on_call = first_call_cancels
        with self.assertRaises(util.Cancelled):
            h.run(duration=self.HOUR, cancel=cancel)
        self.assertLessEqual(len(self.calls), shotlog._PART_WORKERS)
        self.assertEqual(h.calls["draft"], [])

    def test_windows_logged_before_the_stop_are_kept_for_the_next_run(self):
        h = self.harness_with_the_real_shot_log(**{"clip_engine.hard_cap_usd": 1.0})
        with self.assertRaises(engine.EngineError):
            h.run(duration=self.HOUR)
        first = len(self.calls)
        # The next run (cap raised) pays only for the windows still missing.
        h.settings = {"clip_engine.hard_cap_usd": 10.0}
        self.calls.clear()
        result = h.run(duration=self.HOUR, meter=engine.CostMeter())
        self.assertEqual(result.outcome, "proposals")
        self.assertEqual(len(self.calls), 13 - first)


if __name__ == "__main__":
    unittest.main()
