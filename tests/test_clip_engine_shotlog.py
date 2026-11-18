"""Tests for the shot log (app/clip_engine/shotlog.py): parsing model times,
normalizing and merging window logs, the editor's compact view, credits, and
building logs window by window with every provider and ffmpeg call mocked.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
    SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
    .venv/bin/python -m pytest -q tests/test_clip_engine_shotlog.py
"""
from __future__ import annotations

import copy
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from app import spend
from app.clip_engine import claude, gemini, media, prompts, shotlog, util
from app.config import Settings

BRAND = {"name": "Renewables.org", "topic": "climate change"}


class FakeMeter:
    """The CostMeter interface the engine passes in (spec §4.12)."""

    def __init__(self, bucket="engine"):
        self.bucket = bucket
        self.added: list[tuple[str, dict]] = []
        self.cached: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def add(self, stage, usage):
        with self._lock:
            self.added.append((stage, dict(usage or {})))

    def add_cached(self, stage, cost):
        with self._lock:
            self.cached.append((stage, float(cost)))


def usage(cost=0.01, model="gemini-3.8-flash"):
    return {"model": model, "input_tokens": 1000, "output_tokens": 200, "cost_usd": cost}


def mmss(seconds: float) -> str:
    m, s = divmod(float(seconds), 60)
    return f"{int(m):02d}:{s:04.1f}"


class EngineDirCase(unittest.TestCase):
    """A private engine_dir per test and a throwaway spend ledger."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        patches = [mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.dir / "engine")})]
        if hasattr(spend, "_LEDGER"):
            patches.append(mock.patch.object(spend, "_LEDGER", self.dir / "llm_spend.json"))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


# ---- parsing and normalizing ---------------------------------------------------------------

class ParseTsTest(unittest.TestCase):
    CASES = [
        (75.5, 75.5), (0, 0.0), (12, 12.0), ("75.5", 75.5), ("75.5s", 75.5), ("t=75.5s", 75.5),
        (" 12 sec ", 12.0), ("01:15", 75.0), ("01:15.5", 75.5), ("1:15", 75.0),
        ("1:02:03", 3723.0), ("01:02:03.5", 3723.5), ("00:12,5", 12.5), ("75:30", 4530.0),
        ("-3", None), (-1, None), ("nan", None), (float("inf"), None), (None, None),
        (True, None), (False, None), ("", None), ("abc", None), ("12:30-12:40", None),
        ([1], None),
    ]

    def test_table(self):
        for value, expected in self.CASES:
            with self.subTest(value=value):
                self.assertEqual(shotlog.parse_ts(value), expected)


def raw_beat(start, end, **kw):
    beat = {"start": start, "end": end, "shot": "interview", "on_screen": "a man talks",
            "face": "close", "speaker": "narrator", "line": "", "sound": "", "energy": 3,
            "visual": 3, "topic": 2, "note": ""}
    beat.update(kw)
    return beat


class NormalizeLogTest(unittest.TestCase):
    def norm(self, raw, **kw):
        args = dict(duration=400.0, perceiver="gemini", model="gemini-3.8-flash", cost=0.05)
        args.update(kw)
        return shotlog.normalize_log(raw, **args)

    def test_offset_clamp_sort_and_invalid_beats(self):
        raw = {"beats": [raw_beat("00:20", "00:30"), raw_beat("00:00", "00:10"),
                         raw_beat("00:25", "00:20"),            # end before start
                         raw_beat("bad", "00:05"),              # unparseable
                         raw_beat("04:50", "05:30"),            # runs past the source
                         raw_beat("07:00", "07:10")]}           # entirely past it
        out = self.norm(raw, offset=100.0)
        self.assertEqual([(b["start"], b["end"]) for b in out["beats"]],
                         [(100.0, 110.0), (120.0, 130.0), (390.0, 400.0)])

    def test_field_coercion(self):
        raw = {"beats": [
            raw_beat(0, 5, shot="Wide Scene", face="CLOSE", energy=9, visual="2", topic=-1),
            raw_beat(5, 10, shot="drone", face="huge", energy="x", visual=None, topic=None),
            raw_beat(10, 15, shot="teleport", energy=2.6, visual=0, topic=5),
        ]}
        beats = self.norm(raw)["beats"]
        self.assertEqual([b["shot"] for b in beats], ["wide_scene", "aerial", "other"])
        self.assertEqual([b["face"] for b in beats], ["close", "none", "close"])
        self.assertEqual([b["energy"] for b in beats], [5, 3, 3])
        self.assertEqual([b["visual"] for b in beats], [2, 3, 1])
        self.assertEqual([b["topic"] for b in beats], [0, 1, 2])

    def test_gemini_clock_text_moves_to_source_seconds(self):
        raw = {"beats": [raw_beat("00:10", "00:20",
                                  on_screen="00:12.5 aerial of the town; 00:15.0 a man wading")],
               "structure": "00:00-00:12 anchor intro; 00:12-02:41 field report"}
        out = self.norm(raw, offset=100.0)
        self.assertEqual(out["beats"][0]["on_screen"],
                         "112.5s aerial of the town; 115s a man wading")
        self.assertEqual(out["structure"], "100s-112s anchor intro; 112s-261s field report")

    def test_frame_logs_keep_their_seconds(self):
        raw = {"beats": [raw_beat("75.5", "80", on_screen="75.5 aerial; 77.0 a man")],
               "structure": "0-12 intro"}
        out = self.norm(raw, perceiver="claude", model="claude-opus-5-5")
        self.assertEqual(out["beats"][0]["on_screen"], "75.5 aerial; 77.0 a man")
        self.assertEqual(out["beats"][0]["start"], 75.5)
        self.assertEqual(out["structure"], "0-12 intro")

    def test_people_deduped_by_name_keeping_the_most_prominent(self):
        raw = {"beats": [raw_beat(0, 5)], "people": [
            {"name": "bill nye", "role": "", "prominence": "public_figure", "evidence": "said"},
            {"name": "Bill Nye", "role": "science educator", "prominence": "famous",
             "evidence": "name shown in a lower third"},
            {"name": "", "role": "x", "prominence": "famous", "evidence": "title"},
            {"name": "A firefighter", "role": "witness", "prominence": "celebrity??",
             "evidence": "?"},
        ]}
        people = self.norm(raw)["people"]
        self.assertEqual(people, [
            {"name": "Bill Nye", "role": "science educator", "prominence": "famous",
             "evidence": "on_screen"},
            {"name": "A firefighter", "role": "witness", "prominence": "unknown",
             "evidence": "none"},
        ])

    def test_highlights(self):
        raw = {"beats": [raw_beat(0, 5)], "highlights": [
            {"start": "00:30", "end": "00:31.5", "kind": "image", "what": "a roof caves in"},
            {"start": "00:10", "end": "", "kind": "quote", "what": "we're all screwed"},
            {"start": "nope", "end": "00:05", "kind": "visual", "what": "dropped"},
        ]}
        out = self.norm(raw, offset=5.0)
        self.assertEqual(out["highlights"], [
            {"start": 15.0, "end": 17.0, "kind": "line", "what": "we're all screwed"},
            {"start": 35.0, "end": 36.5, "kind": "visual", "what": "a roof caves in"},
        ])

    def test_metadata_and_completeness(self):
        out = self.norm({"beats": [raw_beat(0, 5)], "summary": " Two  lines. ", "format": "news"})
        self.assertEqual(out["version"], shotlog.VERSION)
        self.assertEqual(out["perceiver"], "gemini")
        self.assertEqual(out["model"], "gemini-3.8-flash")
        self.assertEqual(out["prompt_hash"], shotlog.prompt_hash("video"))
        self.assertEqual(out["source_seconds"], 400.0)
        self.assertEqual(out["cost_usd"], 0.05)
        self.assertEqual(out["summary"], "Two lines.")
        self.assertTrue(out["complete"])
        self.assertFalse(self.norm({"beats": []})["complete"])
        self.assertFalse(self.norm("not a dict")["complete"])


def part_log(start, end, beats, *, people=(), highlights=(), complete=True, structure="",
             summary="", cost=0.01):
    return {"version": 2, "perceiver": "gemini", "model": "m", "prompt_hash": "h",
            "source_seconds": 456.0, "created_at": "x", "cost_usd": cost, "complete": complete,
            "summary": summary, "format": "news" if summary else "", "structure": structure,
            "people": list(people),
            "beats": [{"start": s, "end": e, "shot": "b_roll", "on_screen": f"{s}-{e}",
                       "face": "none", "speaker": "narrator", "line": "", "sound": "",
                       "energy": 3, "visual": 3, "topic": 2, "note": ""} for s, e in beats],
            "highlights": [{"start": s, "end": s + 1, "kind": "visual", "what": f"h{s}"}
                           for s in highlights],
            "span": [start, end]}


class MergeWindowsTest(unittest.TestCase):
    def test_overlap_hands_over_at_the_midpoint(self):
        a = part_log(0, 233, [(0, 100), (100, 230)], highlights=[50, 230], structure="intro",
                     summary="first", cost=0.02,
                     people=[{"name": "bill nye", "role": "", "prominence": "public_figure",
                              "evidence": "said"}])
        b = part_log(223, 456, [(223, 240), (240, 456)], highlights=[225, 300],
                     structure="report", summary="second", cost=0.03,
                     people=[{"name": "Bill Nye", "role": "educator", "prominence": "famous",
                              "evidence": "title"}])
        out = shotlog.merge_windows([(223.0, b), (0.0, a)])   # order of input doesn't matter
        self.assertEqual([(x["start"], x["end"]) for x in out["beats"]],
                         [(0, 100), (100, 228.0), (228.0, 240), (240, 456)])
        self.assertEqual([h["start"] for h in out["highlights"]], [50, 300])
        self.assertEqual(out["people"], [{"name": "Bill Nye", "role": "educator",
                                          "prominence": "famous", "evidence": "title"}])
        self.assertEqual(out["structure"], "intro; report")
        self.assertEqual(out["summary"], "first")
        self.assertEqual(out["cost_usd"], 0.05)
        self.assertTrue(out["complete"])

    def test_a_failed_part_leaves_a_gap_and_marks_the_log_incomplete(self):
        a = part_log(0, 300, [(0, 300)])
        failed = part_log(290, 590, [], complete=False)
        c = part_log(580, 880, [(580, 880)])
        out = shotlog.merge_windows([(0, a), (290, failed), (580, c)])
        self.assertFalse(out["complete"])
        self.assertEqual([(x["start"], x["end"]) for x in out["beats"]],
                         [(0, 295.0), (585.0, 880)])

    def test_back_to_back_parts_and_empty_input(self):
        a = part_log(0, 300, [(0, 300)])
        b = part_log(300, 600, [(300, 600)])
        out = shotlog.merge_windows([(0, a), (300, b)])
        self.assertEqual([(x["start"], x["end"]) for x in out["beats"]], [(0, 300), (300, 600)])
        empty = shotlog.merge_windows([])
        self.assertFalse(empty["complete"])
        self.assertEqual(empty["beats"], [])


class FilterAndCompactTest(unittest.TestCase):
    def log(self):
        beats = [
            {"start": 0.0, "end": 10.04, "shot": "anchor_desk", "on_screen": "anchor at desk",
             "face": "close", "speaker": "anchor", "line": "Good evening", "sound": "",
             "energy": 1, "visual": 1, "topic": 1, "note": "anchor toss"},
            {"start": 10.04, "end": 25.0, "shot": "aerial", "on_screen": "x" * 300,
             "face": "none", "speaker": "narrator", "line": "y" * 250, "sound": "sirens " * 30,
             "energy": 4, "visual": 5, "topic": 2, "note": ""},
            {"start": 25.0, "end": 40.0, "shot": "interview", "on_screen": "a farmer",
             "face": "close", "speaker": "farmer", "line": "It never rained", "sound": "",
             "energy": 3, "visual": 3, "topic": 2, "note": ""},
        ]
        return {"version": 2, "complete": True, "summary": "s", "format": "news package",
                "structure": "0s-10s intro", "beats": beats,
                "people": [{"name": "Ann Lee", "role": "farmer", "prominence": "local",
                            "evidence": "on_screen"}],
                "highlights": [{"start": 12.0, "end": 14.0, "kind": "visual", "what": "flood"},
                               {"start": 30.0, "end": 31.0, "kind": "line", "what": "rain"}]}

    def test_filter_to_window(self):
        out = shotlog.filter_to_window(self.log(), {"start": 12, "end": 30})
        self.assertEqual([b["start"] for b in out["beats"]], [10.04, 25.0])
        self.assertEqual([h["start"] for h in out["highlights"]], [12.0])
        self.assertEqual(out["window"], {"start": 12.0, "end": 30.0})
        self.assertEqual(len(out["people"]), 1)
        self.assertEqual(len(self.log()["beats"]), 3)
        self.assertIs(shotlog.filter_to_window(self.log(), None)["beats"][0]["start"], 0.0)

    def test_filter_to_window_keeps_only_people_the_window_can_hold(self):
        log = self.log()
        log["people"] += [
            {"name": "Tom Anchor", "role": "anchor", "prominence": "local", "evidence": "said"},
            {"name": "Joe Farmer", "role": "farmer", "prominence": "local", "evidence": "said"}]
        log["beats"][0]["speaker"] = "Tom Anchor"      # named only at 0-10 s
        log["beats"][2]["speaker"] = "Joe Farmer"      # named at 25-40 s
        out = shotlog.filter_to_window(log, {"start": 12, "end": 30})
        # Ann Lee: named by no beat, so she may be anywhere (kept). Tom Anchor: named
        # only outside the window (left out). Joe Farmer: named inside it (kept).
        self.assertEqual([p["name"] for p in out["people"]], ["Ann Lee", "Joe Farmer"])
        self.assertEqual(len(log["people"]), 3)                  # the input is untouched

    def test_compact_rounds_trims_and_attaches_cuts(self):
        cuts = [{"t": 9.7, "score": 40}, {"t": 10.04, "score": 35}, {"t": 18.33, "score": 25},
                {"t": 19.0, "score": 5}, {"t": 30.5, "score": 60}]
        out = shotlog.compact_for_prompt(self.log(), cuts=cuts, min_shot_score=20)
        first, second, third = out["beats"]
        self.assertEqual((second["start"], second["end"]), (10.0, 25.0))
        # 9.7 sits in the first beat and also leads the second (within 0.5 s of its start);
        # the weak 19.0 change is ignored.
        self.assertEqual(first["cuts"], [9.7])
        self.assertEqual(second["cuts"], [9.7, 10.04, 18.33])
        self.assertEqual(third["cuts"], [30.5])
        self.assertEqual(len(second["on_screen"]), 240)
        self.assertTrue(second["on_screen"].endswith("…"))
        self.assertEqual(len(second["line"]), 200)
        self.assertLessEqual(len(second["sound"]), 80)
        self.assertEqual(first["note"], "anchor toss")
        self.assertNotIn("note", second)        # empty fields are left out
        self.assertNotIn("sound", third)
        self.assertEqual(out["people"][0]["prominence"], "local")
        self.assertEqual(out["format"], "news package")
        self.assertEqual(len(out["highlights"]), 2)
        self.assertNotIn("warning", out)

    def test_compact_window_and_incomplete_warning(self):
        log = self.log()
        log["complete"] = False
        out = shotlog.compact_for_prompt(log, window={"start": 26, "end": 39})
        self.assertEqual([b["start"] for b in out["beats"]], [25.0])
        self.assertEqual([h["start"] for h in out["highlights"]], [30.0])
        self.assertIn("warning", out)

    def test_too_many_beats_merge_same_shot_neighbours_first(self):
        shots = ["aerial", "aerial", "interview", "interview", "b_roll", "aerial"]
        log = {"complete": True, "beats": [
            {"start": i * 10.0, "end": i * 10.0 + 10, "shot": s, "on_screen": f"shot {i}",
             "face": "none", "speaker": "narrator", "line": f"l{i}", "sound": "", "energy": i % 5 + 1,
             "visual": 3, "topic": 2, "note": ""} for i, s in enumerate(shots)]}
        out = shotlog.compact_for_prompt(log, max_beats=4)
        self.assertEqual([(b["start"], b["end"], b["shot"]) for b in out["beats"]],
                         [(0.0, 20.0, "aerial"), (20.0, 40.0, "interview"),
                          (40.0, 50.0, "b_roll"), (50.0, 60.0, "aerial")])
        self.assertEqual(out["beats"][0]["on_screen"], "shot 0; shot 1")
        self.assertEqual(out["beats"][0]["line"], "l0 … l1")
        self.assertEqual(out["beats"][0]["energy"], 2)
        out = shotlog.compact_for_prompt(log, max_beats=2)
        self.assertEqual(len(out["beats"]), 2)
        self.assertEqual((out["beats"][0]["start"], out["beats"][-1]["end"]), (0.0, 60.0))


class CreditsTest(unittest.TestCase):
    def log(self):
        def beat(start, end, speaker, on_screen=""):
            return {"start": start, "end": end, "shot": "interview", "on_screen": on_screen,
                    "face": "close", "speaker": speaker, "line": "", "sound": "", "energy": 3,
                    "visual": 3, "topic": 2, "note": ""}
        return {"beats": [beat(0, 10, "Bill Nye"),
                          beat(10, 20, "narrator", "Nye takes off his safety glasses"),
                          beat(20, 30, "narrator", "a burning bush by the road"),
                          beat(30, 40, "Jane Doe")],
                "people": [
                    {"name": "Bill Nye", "role": "science educator", "prominence": "famous",
                     "evidence": "title"},
                    {"name": "Jane Doe", "role": "resident", "prominence": "local",
                     "evidence": "said"},
                    {"name": "Dr. Ada Smith", "role": "climatologist", "prominence": "expert",
                     "evidence": "none"},
                    {"name": "George Bush", "role": "former president",
                     "prominence": "public_figure", "evidence": "said"},
                    {"name": "Al Gore", "role": "former vice president",
                     "prominence": "public_figure", "evidence": "on_screen"},
                ]}

    def test_mapping_to_clip_time_in_play_order(self):
        # Cold open: 20-30 plays first (clip 0-10), then 0-12 (clip 10-22).
        out = shotlog.credits_for(self.log(), [{"start": 20, "end": 30}, {"start": 0, "end": 12}])
        self.assertEqual(out, "10-22 s: Bill Nye, science educator (named in the title); "
                              "On screen at some point: George Bush, former president "
                              "(named aloud), Al Gore, former vice president "
                              "(name shown on screen)")

    def test_people_named_only_outside_the_clip_are_left_out(self):
        log = self.log()
        log["people"] = log["people"][:1]
        self.assertEqual(shotlog.credits_for(log, [{"start": 30, "end": 40}]), "")
        self.assertEqual(shotlog.credits_for(log, [{"start": 5, "end": 8}, {"start": 30, "end": 32}]),
                         "0-3 s: Bill Nye, science educator (named in the title)")

    def test_nobody_recognisable(self):
        self.assertEqual(shotlog.credits_for({"beats": [], "people": []}, [{"start": 0, "end": 5}]), "")
        self.assertEqual(shotlog.credits_for(None, []), "")

    def test_a_section_credits_like_the_whole_log(self):
        # A section pass reuses the cached whole-video log filtered to its window.
        # People named only outside the window must not turn into "on screen at
        # some point" on every clip of the section.
        def beat(start, end, speaker, on_screen=""):
            return {"start": start, "end": end, "shot": "interview", "on_screen": on_screen,
                    "face": "close", "speaker": speaker, "line": "", "sound": "", "energy": 3,
                    "visual": 3, "topic": 2, "note": ""}
        whole = {"beats": [beat(0, 10, "Bill Nye"), beat(10, 30, "narrator", "flood waters"),
                           beat(100, 110, "Jane Doe", "Jane Doe at a lectern"),
                           beat(200, 210, "Tom Roe")],
                 "people": [
                     {"name": "Bill Nye", "role": "science educator", "prominence": "famous",
                      "evidence": "title"},
                     {"name": "Jane Doe", "role": "senator", "prominence": "public_figure",
                      "evidence": "on_screen"},
                     {"name": "Tom Roe", "role": "climate scientist", "prominence": "expert",
                      "evidence": "said"},
                     {"name": "Al Gore", "role": "former vice president",
                      "prominence": "public_figure", "evidence": "description"}]}
        section = shotlog.filter_to_window(whole, {"start": 0, "end": 60})
        segments = [{"start": 5, "end": 30}]
        expected = ("0-5 s: Bill Nye, science educator (named in the title); On screen at some "
                    "point: Al Gore, former vice president (named in the video description)")
        self.assertEqual(shotlog.credits_for(whole, segments), expected)
        self.assertEqual(shotlog.credits_for(section, segments), expected)

    def test_names_are_matched_as_whole_words(self):
        def beat(start, end, speaker, on_screen):
            return {"start": start, "end": end, "shot": "b_roll", "on_screen": on_screen,
                    "face": "none", "speaker": speaker, "line": "", "sound": "", "energy": 3,
                    "visual": 3, "topic": 2, "note": ""}
        log = {"beats": [beat(0, 6, "narrator", "residents fleeing the flood"),
                         beat(6, 12, "Joann Leeds", "a woman on her porch"),
                         beat(12, 20, "narrator", "a burning bush by the road"),
                         beat(20, 28, "narrator", "protesters protesting outside the plant")],
               "people": [
                   {"name": "Lee", "role": "climate scientist", "prominence": "expert",
                    "evidence": "said"},
                   {"name": "Ann Lee", "role": "senator", "prominence": "public_figure",
                    "evidence": "on_screen"},
                   {"name": "Bush", "role": "former president", "prominence": "public_figure",
                    "evidence": "said"},
                   {"name": "Sting", "role": "musician", "prominence": "famous",
                    "evidence": "title"}]}
        # No beat names any of them, so nobody gets clip times.
        self.assertEqual(shotlog.credits_for(log, [{"start": 0, "end": 28}]),
                         "On screen at some point: Lee, climate scientist (named aloud), "
                         "Ann Lee, senator (name shown on screen), Bush, former president "
                         "(named aloud), Sting, musician (named in the title)")

    def test_names_table(self):
        def beat(speaker="", on_screen=""):
            return {"speaker": speaker, "on_screen": on_screen}
        cases = [
            ("Bill Nye", beat(speaker="Bill Nye"), True),
            ("Bill Nye", beat(speaker="bill nye (science educator)"), True),
            ("Bill Nye", beat(speaker="Nye"), True),                       # surname, speaker
            ("Bill Nye", beat(on_screen="Nye takes off his safety glasses"), True),
            ("Bill Nye", beat(on_screen="Bill Nye's lab"), True),
            ("Bill Nye", beat(on_screen="nye in lower case"), False),      # surname: capitalised
            ("George Bush", beat(on_screen="a burning bush by the road"), False),
            ("Lee", beat(on_screen="residents fleeing the flood"), False),
            ("Lee", beat(speaker="Joann Leeds"), False),
            ("Lee", beat(speaker="lee"), True),                            # speaker: any case
            ("Ann Lee", beat(speaker="Joann Leeds"), False),
            ("Ann Lee", beat(speaker="Senator Ann Lee"), True),
            ("Ann Lee", beat(on_screen="lower third: ANN LEE, senator"), True),
            ("Bush", beat(on_screen="a burning bush"), False),             # one word: capitalised
            ("Bush", beat(on_screen="Bush waves from the stage"), True),
            ("Sting", beat(on_screen="protesters protesting outside"), False),
            ("Sting", beat(on_screen="Sting sings"), True),
            ("Al Gore Jr.", beat(speaker="Al Gore Jr."), True),
            ("Al Gore Jr.", beat(speaker="Al Gore Jr.s"), False),
            ("", beat(speaker="anyone"), False),
        ]
        for name, b, expected in cases:
            with self.subTest(name=name, beat=b):
                self.assertIs(shotlog._names(b, {"name": name}), expected)


class CachePathTest(EngineDirCase):
    def test_layout(self):
        path = shotlog.cache_path("abc_123", "gemini", "gemini-3.8-flash")
        self.assertEqual(path.parent, self.dir / "engine" / "shotlogs")
        self.assertRegex(path.name, rf"^abc_123__gemini__gemini-3\.8-flash__v2__"
                                    rf"{shotlog.prompt_hash('video')}__s[0-9a-f]{{6}}\.json$")
        part = shotlog.cache_path("abc_123", "gemini", "gemini-3.8-flash", part=(290.0, 590.5))
        self.assertTrue(part.name.endswith("__p290-590.5.json"))
        self.assertEqual(part.name[:-len("__p290-590.5.json")], path.name[:-len(".json")])
        frames = shotlog.cache_path("abc", "claude", "claude-opus-5-5")
        self.assertIn(shotlog.prompt_hash("frames"), frames.name)
        self.assertNotEqual(shotlog.prompt_hash("video"), shotlog.prompt_hash("frames"))
        odd = shotlog.cache_path("up/load://x", "gemini", "models/gemini 3")
        self.assertNotIn("/", odd.name)
        self.assertEqual(odd.parent, self.dir / "engine" / "shotlogs")

    def test_load_cached_needs_a_complete_current_log(self):
        path = shotlog.cache_path("v", "gemini", "m")
        good = part_log(0, 10, [(0, 10)])
        util.atomic_write_json(path, good)
        self.assertEqual(shotlog.load_cached("v", "gemini", "m")["beats"], good["beats"])
        for bad in ({**good, "complete": False}, {**good, "beats": []}, {**good, "version": 1}):
            util.atomic_write_json(path, bad)
            self.assertIsNone(shotlog.load_cached("v", "gemini", "m"))
        path.write_text("{half a file")
        self.assertIsNone(shotlog.load_cached("v", "gemini", "m"))

    def test_perception_settings_are_part_of_the_key(self):
        def settings(**gemini_or_claude):
            raw = copy.deepcopy(GEMINI_SETTINGS)
            raw["clip_engine"].setdefault("claude", {})
            for key, value in gemini_or_claude.items():
                block, _, name = key.partition("__")
                if block == "top":
                    raw["clip_engine"][name] = value
                else:
                    raw["clip_engine"][block][name] = value
            return Settings(raw=raw)

        def path(perceiver, s, seconds=200.0):
            return shotlog.cache_path("v", perceiver, "m", settings=s, seconds=seconds)

        base = settings()
        # The workspace defaults and no settings at all give the same file.
        self.assertEqual(path("gemini", base), shotlog.cache_path("v", "gemini", "m"))
        self.assertEqual(path("gemini", base), path("gemini", settings()))
        for change in ({"gemini__perception_fps": 2.0},
                       {"gemini__media_resolution": "MEDIA_RESOLUTION_MEDIUM"},
                       {"gemini__perceiver_thinking": "MEDIUM"},
                       {"top__perception_window_seconds": 120}):
            with self.subTest(change=change):
                self.assertNotEqual(path("gemini", settings(**change)), path("gemini", base))
        # Past 1200 s Gemini samples at long_video_fps: that rate is what counts there.
        self.assertNotEqual(path("gemini", base, 1500.0), path("gemini", base, 200.0))
        self.assertEqual(path("gemini", settings(gemini__perception_fps=2.0), 1500.0),
                         path("gemini", base, 1500.0))
        self.assertNotEqual(path("gemini", settings(gemini__long_video_fps=1.0), 1500.0),
                            path("gemini", base, 1500.0))
        for change in ({"claude__frame_interval": 1.0}, {"claude__max_total_frames": 300},
                       {"claude__frame_width": 768}, {"claude__max_frames_per_call": 100},
                       {"claude__perceiver_effort": "high"}):
            with self.subTest(change=change):
                self.assertNotEqual(path("claude", settings(**change)), path("claude", base))
        # A Gemini knob doesn't touch Claude logs and vice versa.
        self.assertEqual(path("claude", settings(gemini__perception_fps=2.0)), path("claude", base))
        self.assertEqual(path("gemini", settings(claude__frame_interval=1.0)), path("gemini", base))
        self.assertEqual(shotlog.perception_settings("gemini", base, 200.0)["fps"], 1.0)
        self.assertEqual(shotlog.perception_settings("gemini", base, 1500.0)["fps"], 0.5)


class CoverageTest(unittest.TestCase):
    def test_share_of_the_span_the_beats_cover(self):
        def beats(*spans):
            return [{"start": s, "end": e} for s, e in spans]
        cases = [
            (beats((0, 50), (50, 100)), 0, 100, 0.0, 1.0),
            (beats((0, 40)), 0, 100, 0.0, 0.4),                       # stopped early
            (beats((60, 100)), 0, 100, 0.0, 0.4),                     # started late
            (beats((0, 30), (70, 100)), 0, 100, 0.0, 0.6),            # a hole in the middle
            (beats((0, 60), (20, 80)), 0, 100, 0.0, 0.8),             # overlaps count once
            (beats((90, 160)), 100, 200, 0.0, 0.6),                   # clipped to the span
            (beats((100, 194)), 100, 200, 6.0, 1.0),                  # last frame at 194 s
            (beats((0, 10)), 0, 100, 60.0, 10 / 75),                  # slack is at most 1/4
            ([], 0, 100, 0.0, 0.0),
            ([], 5, 5, 0.0, 1.0),
        ]
        for items, start, stop, slack, expected in cases:
            with self.subTest(items=items, start=start, stop=stop, slack=slack):
                self.assertAlmostEqual(shotlog._coverage(items, start, stop, slack=slack),
                                       expected)


# ---- building -------------------------------------------------------------------------------

GEMINI_SETTINGS = {"clip_engine": {"perception_window_seconds": 300, "gemini": {
    "perception_fps": 1.0, "long_video_fps": 0.5, "perceiver_thinking": "LOW",
    "media_resolution": "MEDIA_RESOLUTION_LOW", "perceiver_model": ""}}}


class GeminiBuildTest(EngineDirCase):
    def setUp(self):
        super().setUp()
        self.settings = Settings(raw=GEMINI_SETTINGS)
        self.calls: list[dict] = []
        self.video_items: list[list[dict]] = []
        self.proxy_args: list[tuple] = []
        self.fail: dict[tuple, Exception] = {}       # (start, end) -> error to raise
        self.always_truncate = False
        self.truncated_usage = usage(0.5)             # what a MAX_TOKENS answer bills
        self.coverage: dict = {}                      # (start, end) / "whole" -> share logged
        self.source_time_parts: set = set()
        self.lock = threading.Lock()
        for target, attr, fake in [
            (media, "perception_proxy", self.fake_proxy),
            (gemini, "video_parts", self.fake_video_parts),
            (gemini, "generate_json", self.fake_generate),
        ]:
            p = mock.patch.object(target, attr, side_effect=fake)
            p.start()
            self.addCleanup(p.stop)
        self.deleted = mock.patch.object(gemini, "delete_file").start()
        self.dropped = mock.patch.object(media, "drop").start()
        self.addCleanup(mock.patch.stopall)

    def fake_proxy(self, source, *, height=360, fps=10, start=None, end=None):
        with self.lock:
            self.proxy_args.append((start, end))
        return self.dir / f"proxy_{start}_{end}.mp4"

    def fake_video_parts(self, items, *, inline_budget_bytes=None):
        with self.lock:
            self.video_items.append([dict(i) for i in items])
        return [{"fileData": {"mimeType": "video/mp4",
                              "fileUri": f"https://x/v1beta/files/{Path(i['path']).stem}"},
                 "videoMetadata": {"fps": i.get("fps")}} for i in items]

    def span_of(self, parts) -> tuple:
        stem = parts[0]["fileData"]["fileUri"].rsplit("/", 1)[-1]
        _, start, end = stem.split("_")
        if start == "None":
            return None
        return float(start), float(end)

    def fake_generate(self, model, *, system, parts, schema, thinking, media_resolution,
                      max_output_tokens, label="", bucket="engine", **kw):
        span = self.span_of(parts)
        with self.lock:
            self.calls.append(dict(model=model, system=system, parts=parts, schema=schema,
                                   thinking=thinking, media_resolution=media_resolution,
                                   max_output_tokens=max_output_tokens, bucket=bucket, span=span))
        key = span or "whole"
        if self.always_truncate or key in self.fail:
            exc = self.fail.get(key) or gemini.GeminiTruncated("MAX_TOKENS")
            exc.usage = dict(self.truncated_usage)
            raise exc
        length = ((span[1] - span[0]) if span else self.duration) * self.coverage.get(key, 1.0)
        base = span[0] if (span and key in self.source_time_parts) else 0.0
        mid = round(length / 2, 1)
        return {"summary": f"part {key}", "format": "news", "structure": "",
                "people": [{"name": "Bill Nye", "role": "educator", "prominence": "famous",
                            "evidence": "title"}],
                "beats": [raw_beat(mmss(base), mmss(base + mid)),
                          raw_beat(mmss(base + mid), mmss(base + length))],
                "highlights": [{"start": mmss(base + 1), "end": mmss(base + 2), "kind": "visual",
                                "what": "x"}]}, usage(0.01)

    def build(self, duration, **kw):
        self.duration = duration or 200.0    # what the probe reports in the None cases
        args = dict(video_id="vid1", source=self.dir / "src.mp4", perceiver="gemini",
                    settings=self.settings, brand=BRAND, title="Fire season",
                    description="desc", duration=duration, words=[], transcript=[])
        args.update(kw)
        return shotlog.build(**args)

    def test_short_source_is_one_part_on_the_shared_whole_proxy(self):
        meter = FakeMeter(bucket="bench")
        log = self.build(200.0, meter=meter)
        self.assertEqual(self.proxy_args, [(None, None)])
        self.assertEqual(self.video_items, [[{"path": self.dir / "proxy_None_None.mp4", "fps": 1.0,
                                              "upload": True}]])
        (call,) = self.calls
        self.assertEqual(call["model"], "gemini-3.8-flash")       # blank perceiver_model
        self.assertEqual(call["system"], prompts.perceiver_system(BRAND, mode="video"))
        self.assertIs(call["schema"], shotlog.SCHEMA)
        self.assertEqual((call["thinking"], call["media_resolution"], call["max_output_tokens"]),
                         ("LOW", "MEDIA_RESOLUTION_LOW", 65536))
        self.assertEqual(call["bucket"], "bench")
        self.assertIn("Video title: Fire season", call["parts"][-1]["text"])
        self.assertNotIn("This is part", call["parts"][-1]["text"])
        self.assertEqual(meter.added, [("perceive", usage(0.01))])
        self.assertTrue(log["complete"])
        self.assertEqual([(b["start"], b["end"]) for b in log["beats"]], [(0, 100), (100, 200)])
        self.assertTrue(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash").is_file())
        self.deleted.assert_not_called()      # the whole proxy's upload is reused by the editor
        self.dropped.assert_not_called()

    def test_long_source_is_logged_in_overlapping_windows_and_cached(self):
        meter = FakeMeter()
        log = self.build(456.0, meter=meter)
        self.assertEqual(sorted(c["span"] for c in self.calls), [(0.0, 233.0), (223.0, 456.0)])
        texts = sorted(c["parts"][-1]["text"] for c in self.calls)
        self.assertIn("This is part 1 of 2: it covers 00:00 to 03:53", texts[0])
        self.assertIn("This is part 2 of 2", texts[1])
        # Part 2 answered from its own 00:00: its beats land on the source clock.
        self.assertEqual([(b["start"], b["end"]) for b in log["beats"]],
                         [(0.0, 116.5), (116.5, 228.0), (228.0, 339.5), (339.5, 456.0)])
        self.assertEqual(log["cost_usd"], 0.02)
        for span in [(0.0, 233.0), (223.0, 456.0)]:
            self.assertTrue(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash",
                                               part=span).is_file())
        self.assertEqual(sorted(c.args[0] for c in self.deleted.call_args_list),
                         ["https://x/v1beta/files/proxy_0.0_233.0",
                          "https://x/v1beta/files/proxy_223.0_456.0"])
        self.assertEqual(len(self.dropped.call_args_list), 2)
        # A rerun reuses the whole log: no calls, its stored cost reported as cached.
        self.calls.clear()
        again = FakeMeter()
        self.assertEqual(self.build(456.0, meter=again)["beats"], log["beats"])
        self.assertEqual(self.calls, [])
        self.assertEqual(again.added, [])
        self.assertEqual(again.cached, [("perceive", 0.02)])

    def test_finished_windows_are_reused_when_the_whole_log_is_missing(self):
        self.build(456.0)
        shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash").unlink()
        self.calls.clear()
        meter = FakeMeter()
        self.build(456.0, meter=meter)
        self.assertEqual(self.calls, [])
        self.assertEqual(sorted(meter.cached), [("perceive", 0.01), ("perceive", 0.01)])

    def test_a_truncated_window_is_split_in_half(self):
        self.fail[(0.0, 233.0)] = gemini.GeminiTruncated("MAX_TOKENS")
        meter = FakeMeter()
        log = self.build(456.0, meter=meter)
        self.assertEqual(sorted(c["span"] for c in self.calls),
                         [(0.0, 121.5), (0.0, 233.0), (111.5, 233.0), (223.0, 456.0)])
        self.assertIn(("perceive", usage(0.5)), meter.added)   # the truncated answer was billed
        self.assertTrue(log["complete"])
        self.assertEqual(log["beats"][0]["start"], 0.0)
        self.assertEqual(log["beats"][-1]["end"], 456.0)
        # Without the whole log, a rerun goes straight to the cached halves.
        shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash").unlink()
        self.calls.clear()
        self.build(456.0)
        self.assertEqual(self.calls, [])

    def test_splitting_stops_after_two_levels(self):
        self.always_truncate = True
        with self.assertRaises(shotlog.ShotLogError):
            self.build(200.0)
        self.assertEqual(len(self.calls), 1 + 2 + 4)

    def test_a_looping_answer_is_not_split_again(self):
        # 65,536 output tokens for a 233 s window (281 a second) could still be a
        # dense window: it is split. For a ~120 s half (over 500 a second) it can only
        # be the model repeating itself: that half fails rather than paying for quarters.
        self.always_truncate = True
        self.truncated_usage = {**usage(0.49), "output_tokens": 65536}
        meter = FakeMeter()
        with self.assertRaises(shotlog.ShotLogError) as caught:
            self.build(456.0, meter=meter)
        self.assertEqual(len(self.calls), 2 * (1 + 2))           # was 2 * (1 + 2 + 4)
        self.assertIn("looping", str(caught.exception))
        self.assertEqual(len(meter.added), 6)                    # every truncated call billed

    def test_check_runs_before_every_window_and_split(self):
        # Every answer runs past the output limit (billed $0.50 each). Without a
        # check, a 60-minute source pays 13 windows x 7 calls = 91 calls ($45.50)
        # inside this one stage, past any cap; the engine's checkpoint stops it.
        self.always_truncate = True
        meter = FakeMeter()

        class OverCap(Exception):
            pass

        def check():
            with meter._lock:
                spent = sum(u.get("cost_usd", 0.0) for _, u in meter.added)
            if spent > 1.0:
                raise OverCap(f"stopped at ${spent:.2f}")

        with self.assertRaises(OverCap):
            self.build(3600.0, meter=meter, check=check)
        self.assertLessEqual(len(self.calls), 6)
        self.assertGreaterEqual(len(self.calls), 3)
        self.assertFalse(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash",
                                            seconds=3600.0).is_file())

    def test_check_errors_are_not_swallowed_as_failed_windows(self):
        # A Cancelled (a RuntimeError) raised by the check must end the log, not
        # become one more failed window of a partial log.
        def check():
            raise util.Cancelled("ran out of time")

        with self.assertRaises(util.Cancelled):
            self.build(456.0, check=check)
        self.assertEqual(self.calls, [])

    def test_cancel_stops_windows_not_yet_started(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(util.Cancelled):
            self.build(456.0, cancel=cancel)
        self.assertEqual(self.calls, [])

    def test_a_window_that_stops_early_is_used_but_not_cached(self):
        # Part 2 answers for only its first 40%: finishReason STOP, beats that end
        # at 316 s of a window that runs to 456 s.
        self.coverage[(223.0, 456.0)] = 0.4
        log = self.build(456.0)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(log["complete"])
        self.assertEqual(log["beats"][-1]["end"], 316.2)         # the partial answer is used
        self.assertFalse(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash",
                                            part=(223.0, 456.0)).is_file())
        self.assertTrue(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash",
                                           part=(0.0, 233.0)).is_file())
        self.assertFalse(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash").is_file())
        self.assertIn("warning", shotlog.compact_for_prompt(log))
        # Only that window is asked again; a full answer completes the log.
        del self.coverage[(223.0, 456.0)]
        self.calls.clear()
        again = self.build(456.0)
        self.assertEqual([c["span"] for c in self.calls], [(223.0, 456.0)])
        self.assertTrue(again["complete"])
        self.assertTrue(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash").is_file())

    def test_a_change_in_perception_settings_logs_again(self):
        first = self.build(200.0)
        for change in ({"perception_fps": 2.0},
                       {"media_resolution": "MEDIA_RESOLUTION_MEDIUM"},
                       {"perceiver_thinking": "HIGH"}):
            with self.subTest(change=change):
                raw = copy.deepcopy(GEMINI_SETTINGS)
                raw["clip_engine"]["gemini"].update(change)
                self.settings = Settings(raw=raw)
                self.calls.clear()
                meter = FakeMeter()
                log = self.build(200.0, meter=meter)
                self.assertEqual(len(self.calls), 1)               # not the other arm's log
                self.assertEqual(meter.cached, [])
                key, value = next(iter(change.items()))
                if key == "perception_fps":
                    self.assertEqual(self.video_items[-1][0]["fps"], 2.0)
                elif key == "media_resolution":
                    self.assertEqual(self.calls[0]["media_resolution"], value)
                else:
                    self.assertEqual(self.calls[0]["thinking"], value)
                # The same settings again: the log just made is reused.
                self.calls.clear()
                self.assertEqual(self.build(200.0)["beats"], log["beats"])
                self.assertEqual(self.calls, [])
        raw = copy.deepcopy(GEMINI_SETTINGS)
        raw["clip_engine"]["perception_window_seconds"] = 120
        self.settings = Settings(raw=raw)
        self.calls.clear()
        self.build(200.0)
        self.assertEqual(len(self.calls), 2)                       # two 105 s windows now
        # Each log records what it was made with.
        self.assertEqual(first["perceiver_settings"],
                         shotlog.perception_settings("gemini", Settings(raw=GEMINI_SETTINGS),
                                                     200.0))

    def test_a_failed_window_leaves_an_uncached_incomplete_log(self):
        self.fail[(223.0, 456.0)] = gemini.GeminiBlocked("SAFETY")
        log = self.build(456.0)
        self.assertFalse(log["complete"])
        self.assertEqual(log["beats"][-1]["end"], 228.0)
        self.assertFalse(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash").is_file())
        self.assertTrue(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash",
                                           part=(0.0, 233.0)).is_file())
        # Only the failed window is asked again.
        del self.fail[(223.0, 456.0)]
        self.calls.clear()
        self.assertTrue(self.build(456.0)["complete"])
        self.assertEqual([c["span"] for c in self.calls], [(223.0, 456.0)])

    def test_quota_stops_the_whole_log(self):
        self.fail["whole"] = gemini.GeminiQuota("daily quota")
        with self.assertRaises(gemini.GeminiQuota):
            self.build(200.0)

    def test_every_window_failing_raises(self):
        self.fail["whole"] = gemini.GeminiError("HTTP 500")
        with self.assertRaises(shotlog.ShotLogError):
            self.build(200.0)

    def test_long_sources_sample_at_the_long_video_rate(self):
        self.build(1500.0)
        self.assertEqual({items[0]["fps"] for items in self.video_items}, {0.5})

    def test_section_without_a_whole_log_logs_only_the_window(self):
        log = self.build(1500.0, window={"start": 600, "end": 900})
        self.assertEqual([c["span"] for c in self.calls], [(600.0, 900.0)])
        self.assertEqual(log["window"], {"start": 600.0, "end": 900.0})
        self.assertEqual((log["beats"][0]["start"], log["beats"][-1]["end"]), (600.0, 900.0))
        self.assertFalse(shotlog.cache_path("vid1", "gemini", "gemini-3.8-flash").is_file())

    def test_section_reuses_the_whole_log(self):
        self.build(456.0)
        self.calls.clear()
        meter = FakeMeter()
        log = self.build(456.0, window={"start": 300, "end": 400}, meter=meter)
        self.assertEqual(self.calls, [])
        self.assertEqual([(b["start"], b["end"]) for b in log["beats"]], [(228.0, 339.5),
                                                                          (339.5, 456.0)])
        self.assertEqual(meter.cached, [("perceive", 0.02)])

    def test_force_ignores_every_cache(self):
        self.build(200.0)
        self.calls.clear()
        self.build(200.0, force=True)
        self.assertEqual(len(self.calls), 1)

    def test_answers_given_in_source_time_are_not_offset_twice(self):
        self.source_time_parts.add((223.0, 456.0))
        log = self.build(456.0)
        self.assertEqual(log["beats"][-1]["end"], 456.0)
        self.assertEqual(log["beats"][-2]["start"], 228.0)

    def test_unknown_duration_is_probed(self):
        with mock.patch.object(media, "probe_duration", return_value=200.0) as probe:
            self.build(None)
        probe.assert_called_once()
        with mock.patch.object(media, "probe_duration", return_value=None):
            with self.assertRaises(shotlog.ShotLogError):
                self.build(None, video_id="other")

    def test_unknown_perceiver(self):
        with self.assertRaises(shotlog.ShotLogError):
            self.build(200.0, perceiver="gpt")


class ClaudeBuildTest(EngineDirCase):
    def setUp(self):
        super().setUp()
        self.settings = Settings(raw={"clip_engine": {"claude": {
            "perceiver_model": "claude-opus-5-5", "perceiver_effort": "low", "frame_interval": 2.0,
            "max_total_frames": 600, "frame_width": 448, "max_frames_per_call": 150}}})
        self.samples: list[dict] = []
        self.calls: list[dict] = []
        self.zero_frames = False
        self.lock = threading.Lock()
        for target, attr, fake in [
            (media, "sample_frames", self.fake_sample),
            (claude, "frames_blocks", lambda frames, clock="video": [
                {"type": "image", "t": t, "clock": clock} for t, _ in frames]),
            (claude, "text_block", lambda text: {"type": "text", "text": text}),
            (claude, "json_call", self.fake_call),
        ]:
            p = mock.patch.object(target, attr, side_effect=fake)
            p.start()
            self.addCleanup(p.stop)

    def fake_sample(self, path, *, start, end, interval, width):
        with self.lock:
            self.samples.append(dict(start=start, end=end, interval=interval, width=width))
        if self.zero_frames:
            return []
        n = int(round((end - start) / interval))
        return [(round(start + i * interval, 2), b"jpg") for i in range(n)]

    def fake_call(self, model, *, system, content, schema, effort, max_tokens, label="",
                  bucket="engine"):
        times = [b["t"] for b in content if b["type"] == "image"]
        with self.lock:
            self.calls.append(dict(model=model, system=system, content=content, effort=effort,
                                   max_tokens=max_tokens, schema=schema))
        start, end = times[0], times[-1]
        return {"summary": "s", "format": "f", "structure": "", "people": [], "highlights": [],
                "beats": [raw_beat(str(start), str(end))]}, usage(0.02, "claude-opus-5-5")

    def build(self, duration, words=(), **kw):
        args = dict(video_id="vidc", source=self.dir / "src.mp4", perceiver="claude",
                    settings=self.settings, brand=BRAND, title="T", description="",
                    duration=duration, words=list(words), transcript=[])
        args.update(kw)
        return shotlog.build(**args)

    def test_chunks_of_frames_with_the_stretch_transcript(self):
        words = [{"word": "Hello", "start": 12.3, "end": 12.6},
                 {"word": "world.", "start": 12.7, "end": 13.0},
                 {"word": "Later", "start": 400.0, "end": 400.4}]
        log = self.build(1200.0, words=words)
        self.assertEqual(sorted((s["start"], s["end"]) for s in self.samples),
                         [(0.0, 300.0), (300.0, 600.0), (600.0, 900.0), (900.0, 1200.0)])
        self.assertEqual({(s["interval"], s["width"]) for s in self.samples}, {(2.0, 448)})
        first = next(c for c in self.calls if c["content"][1]["t"] == 0.0)
        self.assertEqual(first["model"], "claude-opus-5-5")
        self.assertEqual(first["effort"], "low")
        self.assertIs(first["schema"], shotlog.SCHEMA)
        self.assertEqual(first["system"],
                         prompts.perceiver_system(BRAND, mode="frames", frame_interval=2.0))
        text = first["content"][0]["text"]
        self.assertIn("Transcript for this stretch:\n[12.3] Hello world.", text)
        self.assertNotIn("Later", text)
        self.assertEqual(len(first["content"]), 1 + 150)
        self.assertTrue(log["complete"])
        self.assertEqual(log["beats"][0]["start"], 0.0)     # seconds as labelled, no offset

    def test_long_sources_are_sampled_more_sparsely(self):
        self.build(3600.0)
        self.assertEqual({s["interval"] for s in self.samples}, {6.0})
        # Still 600 frames in all, but in 5-minute chunks (50 frames each), not
        # four 15-minute quarters.
        self.assertEqual(sorted((s["start"], s["end"]) for s in self.samples),
                         [(i * 300.0, (i + 1) * 300.0) for i in range(12)])
        self.assertEqual({len(c["content"]) - 1 for c in self.calls}, {50})
        self.assertEqual(self.calls[0]["system"],
                         prompts.perceiver_system(BRAND, mode="frames", frame_interval=6.0))

    def test_long_sources_send_each_chunk_its_whole_transcript(self):
        # Steady speech at 2.8 words a second for an hour: a 15-minute chunk would
        # need ~17k characters of transcript and lose its last ~5 minutes.
        vocab = ("the water kept rising and nobody in the town had seen anything like it "
                 "before the levee finally gave way").split()
        words, t, i = [], 0.0, 0
        while t < 3600.0:
            word = vocab[i % len(vocab)] + ("." if i % 14 == 13 else "")
            words.append({"word": word, "start": round(t, 2), "end": round(t + 0.3, 2)})
            t += 1 / 2.8
            i += 1
        self.build(3600.0, words=words)
        self.assertEqual(len(self.calls), 12)
        for call in self.calls:
            times = [b["t"] for b in call["content"] if b["type"] == "image"]
            text = call["content"][0]["text"]
            stretch = text.split("Transcript for this stretch:\n", 1)[1]
            with self.subTest(start=times[0]):
                self.assertNotIn("[…]", stretch)
                self.assertNotIn("…", stretch)
                last = float(re.findall(r"^\[(\d+(?:\.\d+)?)\]", stretch, re.MULTILINE)[-1])
                self.assertGreater(last, times[0] + 300.0 - 30.0)

    def test_an_over_long_transcript_is_shortened_line_by_line_not_cut_at_the_end(self):
        entries = [{"start": float(t), "end": t + 2.0, "text": f"line {t} " + "word " * 60}
                   for t in range(0, 300, 2)]
        text = shotlog._stretch(entries, 0.0, 300.0)
        lines = text.split("\n")
        self.assertEqual(len(lines), 150)                         # every line is still there
        self.assertTrue(lines[-1].startswith("[298.0] line 298"))
        self.assertLessEqual(len(text), shotlog._STRETCH_CHARS)
        self.assertTrue(all(line.endswith("…") for line in lines))
        short = shotlog._stretch(entries[:3], 0.0, 300.0)
        self.assertNotIn("…", short)
        self.assertEqual(shotlog._stretch(entries, 400.0, 500.0), "")

    def test_a_change_in_perception_settings_logs_again(self):
        self.build(200.0)
        first = len(self.calls)
        self.settings = Settings(raw={"clip_engine": {"claude": {
            "perceiver_model": "claude-opus-5-5", "perceiver_effort": "high", "frame_interval": 1.0,
            "max_total_frames": 600, "frame_width": 768, "max_frames_per_call": 150}}})
        self.calls.clear()
        self.samples.clear()
        self.build(200.0)
        self.assertEqual(len(self.calls), 2 * first)               # 200 frames: two chunks
        self.assertEqual({(s["interval"], s["width"]) for s in self.samples}, {(1.0, 768)})
        self.assertEqual({c["effort"] for c in self.calls}, {"high"})
        self.calls.clear()
        self.build(200.0)
        self.assertEqual(self.calls, [])

    def test_zero_frames_raises(self):
        self.zero_frames = True
        with self.assertRaises(media.MediaError):
            self.build(100.0)
        self.assertFalse(shotlog.cache_path("vidc", "claude", "claude-opus-5-5").is_file())

    def test_truncated_chunks_split(self):
        state = {"n": 0}
        real = self.fake_call

        def flaky(*args, **kw):
            state["n"] += 1
            if state["n"] == 1:
                exc = claude.ClaudeTruncated("max_tokens")
                exc.usage = usage(0.3, "claude-opus-5-5")
                raise exc
            return real(*args, **kw)

        meter = FakeMeter()
        with mock.patch.object(claude, "json_call", side_effect=flaky):
            log = self.build(200.0, meter=meter)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(log["complete"])
        self.assertIn(("perceive", usage(0.3, "claude-opus-5-5")), meter.added)


if __name__ == "__main__":
    unittest.main()
