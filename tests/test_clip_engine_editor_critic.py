"""Tests for drafting/revising (app/clip_engine/editor.py) and for the critic
(app/clip_engine/critic.py): what each model call is sent, in what order,
and how the answers are cleaned, scored and merged. Every provider and ffmpeg
call is mocked.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
    SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
    .venv/bin/python -m pytest -q tests/test_clip_engine_editor_critic.py
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from app import llm, spend
from app.clip_engine import claude, critic, editor, engine, gemini, media, playbook, prompts, shotlog
from app.config import Settings

BRAND = {"name": "Renewables.org", "topic": "climate change"}
ARCHETYPES = ["voice", "scene", "revelation", "clash", "pop", "witness"]
PLAYBOOK_TEXT = playbook.render(playbook.DEFAULT_PLAYBOOK)

SETTINGS = Settings(raw={"clip_engine": {
    "snap": {"min_shot_score": 20},
    "gemini": {"model": "gemini-3.8-flash", "editor_model": "", "critic_model": "",
               "judge_model": "", "editor_thinking": "HIGH", "critic_thinking": "LOW",
               "judge_thinking": "MEDIUM", "media_resolution": "MEDIA_RESOLUTION_LOW",
               "critic_media_resolution": "MEDIA_RESOLUTION_LOW", "critic_fps": 2.0,
               "opening_fps": 8.0, "long_video_fps": 0.5},
    "claude": {"editor_model": "claude-opus-5-5", "critic_model": "claude-opus-5-5",
               "judge_model": "", "editor_effort": "high", "critic_effort": "medium",
               "judge_effort": "medium", "frame_width": 448, "critic_frame_interval": 1.0},
}})


class FakeMeter:
    def __init__(self, bucket="engine"):
        self.bucket = bucket
        self.added: list[tuple[str, dict]] = []
        self.cached: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def add(self, stage, usage):
        with self._lock:
            self.added.append((stage, dict(usage or {})))

    def add_cached(self, stage, cost):
        self.cached.append((stage, float(cost)))


def usage(cost=0.01, model="m"):
    return {"model": model, "input_tokens": 100, "output_tokens": 20, "cost_usd": cost}


class TempCase(unittest.TestCase):
    """Private engine dir + throwaway spend ledger; files for renders."""

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

    def file(self, name: str) -> Path:
        path = self.dir / name
        path.write_bytes(b"\x00" * 16)
        return path

    def patch(self, target, attr, **kw):
        p = mock.patch.object(target, attr, **kw)
        mocked = p.start()
        self.addCleanup(p.stop)
        return mocked


def fake_video_parts(items, *, inline_budget_bytes=None):
    return [{"video": str(i["path"]), **{k: i[k] for k in ("fps", "start", "end", "upload")
                                          if k in i}} for i in items]


# ---- editor ---------------------------------------------------------------------------------

class EntriesForTest(unittest.TestCase):
    def test_words_become_clauses_on_word_edges(self):
        words = [{"word": "The", "start": 1.0, "end": 1.2}, {"word": "water", "start": 1.25, "end": 1.6},
                 {"word": "rose.", "start": 1.65, "end": 2.0}, {"word": "Nobody", "start": 3.5, "end": 3.9},
                 {"word": "left", "start": 3.95, "end": 4.2}]
        entries, has_words = editor.entries_for(words, [{"start": 0, "end": 10, "text": "cap"}])
        self.assertTrue(has_words)
        self.assertEqual(entries, [{"start": 1.0, "end": 2.0, "text": "The water rose."},
                                   {"start": 3.5, "end": 4.2, "text": "Nobody left"}])

    def test_without_words_the_transcript_entries_are_used(self):
        transcript = [{"start": 0.04, "end": 5.06, "text": " Hi "}, {"start": 5, "end": 4, "text": "x"},
                      {"start": "x", "end": 1, "text": "x"}, {"start": 6, "end": 9, "text": ""}]
        for words in ([], None, [{"word": "", "start": 0, "end": 1}]):
            with self.subTest(words=words):
                entries, has_words = editor.entries_for(words, transcript)
                self.assertFalse(has_words)
                self.assertEqual(entries, [{"start": 0.0, "end": 5.1, "text": "Hi"}])

    def test_long_streams_are_coarsened_not_cut(self):
        words = [{"word": f"w{i}.", "start": float(i), "end": i + 0.5} for i in range(1500)]
        entries, _ = editor.entries_for(words, [])
        self.assertLessEqual(len(entries), 600)
        self.assertEqual((entries[0]["start"], entries[-1]["end"]), (0.0, 1499.5))

    def test_word_edges_keep_two_decimals(self):
        """The editor copies entry times into its cuts. At one decimal a
        clause starting at 226.64 s read 226.6 — inside "fuels." — and the
        cut opened on the previous sentence's last word."""
        words = [{"word": "fuels.", "start": 225.98, "end": 226.64},
                 {"word": "While", "start": 226.64, "end": 227.16},
                 {"word": "others", "start": 227.2, "end": 227.56}]
        entries, has_words = editor.entries_for(words, [])
        self.assertTrue(has_words)
        self.assertEqual(entries, [{"start": 225.98, "end": 226.64, "text": "fuels."},
                                   {"start": 226.64, "end": 227.56, "text": "While others"}])
        # The transcript suggester's prompts keep their one decimal.
        from app import llm

        self.assertEqual(llm._words_to_clauses(words)[1]["start"], 226.6)


def raw_draft(segments, **kw):
    draft = {"id": "x", "moment_id": "m1", "archetype": "voice", "lens": "person",
             "story": "Nye says the planet is on fire", "story_key": "Nye Planet Fire",
             "hook": "Nye on camera", "payoff": "glasses off", "segments": segments,
             "why": "famous face", "risk": "long", "draft_caption": "cap",
             "challenges_rules": ""}
    draft.update(kw)
    return draft


def seg(start, end, lead=None):
    out = {"start": start, "end": end}
    if lead:
        out["lead"] = lead
    return out


class CleanDraftsTest(unittest.TestCase):
    def clean(self, raw, **kw):
        args = dict(floor=0.0, horizon=300.0, min_s=12, max_s=45, max_segments=6, used=[],
                    archetypes=ARCHETYPES)
        args.update(kw)
        return editor.clean_drafts(raw, **args)

    def test_cold_open_order_and_leads_are_kept(self):
        (out,) = self.clean([raw_draft([seg(100, 112, "picture"), seg(80, 95, "bogus")])])
        self.assertEqual(out["segments"], [{"start": 100.0, "end": 112.0, "lead": "picture"},
                                           {"start": 80.0, "end": 95.0, "lead": "word"}])

    def test_overlap_with_an_earlier_segment_is_removed(self):
        (out,) = self.clean([raw_draft([seg(80, 100), seg(90, 110)])])
        self.assertEqual([(s["start"], s["end"]) for s in out["segments"]], [(80, 100), (100, 110)])

    def test_used_footage_is_subtracted_in_place(self):
        (out,) = self.clean([raw_draft([seg(120, 128, "picture"), seg(50, 80)])],
                            used=[{"start": 60, "end": 70}])
        self.assertEqual(out["segments"], [{"start": 120.0, "end": 128.0, "lead": "picture"},
                                           {"start": 50.0, "end": 60.0, "lead": "word"},
                                           {"start": 70.0, "end": 80.0, "lead": "word"}])

    def test_length_bounds(self):
        cases = [(9.5, False), (9.6, True), (58.5, True), (58.6, False)]
        for seconds, kept in cases:
            with self.subTest(seconds=seconds):
                out = self.clean([raw_draft([seg(10, 10 + seconds)])])
                self.assertEqual(bool(out), kept)

    def test_window_clamp(self):
        (out,) = self.clean([raw_draft([seg(90, 120), seg(190, 230)])], floor=100.0, horizon=200.0)
        self.assertEqual([(s["start"], s["end"]) for s in out["segments"]], [(100, 120), (190, 200)])

    def test_ids_ranks_and_vocabulary(self):
        raw = [raw_draft([seg(0, 2)]),                                   # too short: dropped
               raw_draft([seg(10, 30)], archetype="VOICE", lens="Picture"),
               "junk",
               raw_draft([seg(40, 60)], archetype="astrology", lens="smell", story_key=""),
               raw_draft([seg(70, 90)], story="", story_key="")]
        out = self.clean(raw)
        self.assertEqual([d["id"] for d in out], ["d1", "d2", "d3"])
        self.assertEqual([d["editor_rank"] for d in out], [0, 1, 2])
        self.assertEqual([d["archetype"] for d in out], ["voice", "other", "voice"])
        self.assertEqual([d["lens"] for d in out], ["picture", "line", "person"])
        self.assertEqual([d["story_key"] for d in out],
                         ["nye-planet-fire", "nye-says-the-planet-is-on-fire", "story-3"])
        self.assertTrue(all(d["origin"] == "draft" and d["parent"] is None for d in out))
        self.assertEqual(set(out[0]), {"id", "moment_id", "archetype", "lens", "story", "story_key",
                                       "hook", "payoff", "segments", "why", "risk", "draft_caption",
                                       "challenges_rules", "origin", "parent", "editor_rank"})

    def test_strings_are_trimmed(self):
        (out,) = self.clean([raw_draft([seg(10, 30)], story="s" * 500, hook="h" * 500,
                                       draft_caption="c" * 500, moment_id="m" * 50)])
        self.assertEqual((len(out["story"]), len(out["hook"]), len(out["draft_caption"]),
                          len(out["moment_id"])), (200, 300, 280, 20))

    def test_too_many_segments_keep_the_opening_and_the_payoff(self):
        segments = [seg(i * 10, i * 10 + 3) for i in range(8)]
        (out,) = self.clean([raw_draft(segments)], max_segments=4)
        self.assertEqual([s["start"] for s in out["segments"]], [0, 10, 20, 70])

    def test_string_times_and_garbage(self):
        (out,) = self.clean([raw_draft([{"start": "01:40", "end": "01:55.5", "lead": "word"},
                                        {"start": "x", "end": 5}, "nope"])])
        self.assertEqual(out["segments"], [{"start": 100.0, "end": 115.5, "lead": "word"}])
        self.assertEqual(self.clean(None), [])
        self.assertEqual(self.clean({"drafts": []}), [])


class ClipMapTest(unittest.TestCase):
    def test_clip_and_source_time_in_play_order(self):
        out = editor.clip_map([seg(100, 112, "picture"), seg(80, 95.5), {"start": 5, "end": 5}])
        self.assertEqual(out, [
            {"clip_start": 0.0, "clip_end": 12.0, "source_start": 100.0, "source_end": 112.0,
             "lead": "picture"},
            {"clip_start": 12.0, "clip_end": 27.5, "source_start": 80.0, "source_end": 95.5},
        ])


def shot_log():
    def beat(start, end, **kw):
        out = {"start": start, "end": end, "shot": "interview", "on_screen": f"beat {start}",
               "face": "close", "speaker": "narrator", "line": "", "sound": "", "energy": 3,
               "visual": 3, "topic": 2, "note": ""}
        out.update(kw)
        return out
    return {"version": 2, "complete": True, "source_seconds": 600.0, "summary": "s",
            "format": "news", "structure": "",
            "people": [{"name": "Bill Nye", "role": "educator", "prominence": "famous",
                        "evidence": "title"}],
            "beats": [beat(0, 10, visual=2), beat(10, 30, visual=4, shot="aerial"),
                      beat(30, 40, speaker="Bill Nye"), beat(40, 60, visual=5, shot="action_closeup"),
                      beat(60, 200), beat(200, 600)],
            "highlights": [{"start": 60.0, "end": 62.0, "kind": "visual", "what": "flood"},
                           {"start": 22.5, "end": 23.0, "kind": "visual", "what": "roof"}]}


ENTRIES = [{"start": float(t), "end": t + 4.0, "text": f"line {t}"} for t in range(0, 600, 5)]

MODEL_ANSWER = {
    "moments": [{"id": "m1", "start": 10, "end": 40, "kind": "person", "why": "Nye"},
                {"id": "m1", "start": -5, "end": 700, "kind": "weird", "why": "dup id"},
                {"id": "m3", "start": 50, "end": 40, "kind": "line", "why": "inverted"}],
    "drafts": [raw_draft([{"start": 30, "end": 52, "lead": "word"}]),
               raw_draft([{"start": 12.04, "end": 20, "lead": "picture"},
                          {"start": 30, "end": 40, "lead": "word"}], archetype="scene",
                         lens="picture", story_key="flood")],
}


class EditorCase(TempCase):
    """Provider mocks shared by the drafting and revising tests."""

    def setUp(self):
        super().setUp()
        self.calls = []
        self.answer = MODEL_ANSWER
        self.video_parts = self.patch(gemini, "video_parts", side_effect=fake_video_parts)
        self.patch(gemini, "generate_json", side_effect=self.fake_generate)
        self.patch(claude, "text_block", side_effect=lambda t: {"type": "text", "text": t})
        self.patch(claude, "frames_blocks", side_effect=lambda frames, clock="video": [
            {"type": "image", "t": t, "clock": clock} for t, _ in frames])
        self.patch(claude, "json_call", side_effect=self.fake_claude)
        self.proxy = self.file("proxy.mp4")

    def fake_generate(self, model, **kw):
        self.calls.append({"model": model, **kw})
        return self.answer, usage(0.04, model)

    def fake_claude(self, model, **kw):
        self.calls.append({"model": model, **kw})
        return self.answer, usage(0.3, model)

    def payload(self, call=None):
        call = call or self.calls[-1]
        if "parts" in call:
            return json.loads(call["parts"][-1]["text"])
        return json.loads(call["content"][0]["text"])


class DraftTest(EditorCase):
    def run_draft(self, **kw):
        args = dict(role="gemini", settings=SETTINGS, brand=BRAND, playbook_text=PLAYBOOK_TEXT,
                    rules="Never fake a quote.", title="Fire season", description="d" * 3000,
                    duration=600.0, shotlog=shot_log(),
                    shot_map={"duration": 600.0, "cuts": [{"t": 12.04, "score": 30},
                                                          {"t": 15.0, "score": 3}],
                              "quiet": [], "error": ""},
                    entries=ENTRIES, has_words=True, context={"rejected_here": [{"story": "x"}]},
                    used=[{"start": 500, "end": 520}], window=None, section_topic="", n=10,
                    min_s=12, max_s=45, max_segments=6, source_video=self.proxy,
                    keyframe_frames=[], meter=FakeMeter())
        args.update(kw)
        self.meter = args["meter"]
        return editor.draft(**args)

    def test_gemini_watches_the_video_and_the_payload_comes_last(self):
        drafts, moments = self.run_draft()
        (call,) = self.calls
        parts = call["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], {"video": str(self.proxy), "fps": 1.0})
        self.assertEqual(set(parts[-1]), {"text"})
        self.assertEqual(call["model"], "gemini-3.8-flash")
        self.assertEqual((call["thinking"], call["media_resolution"], call["max_output_tokens"]),
                         ("HIGH", "MEDIA_RESOLUTION_LOW", 32768))
        self.assertEqual(call["system"], prompts.editor_system(
            BRAND, PLAYBOOK_TEXT, "Never fake a quote.", n=10, min_s=12, max_s=45, max_segments=6,
            sees_video=True, section=None, section_topic="", has_words=True))
        schema = call["schema"]
        self.assertEqual(schema["properties"]["drafts"]["items"]["properties"]["archetype"]["enum"],
                         ARCHETYPES + ["other"])
        self.assertEqual(schema["properties"]["drafts"]["maxItems"], 10)
        self.assertNotIn("enum", editor.DRAFT_SCHEMA["properties"]["drafts"]["items"]["properties"]
                         ["archetype"])                     # the module constant is untouched
        self.assertEqual(self.meter.added, [("draft", usage(0.04, "gemini-3.8-flash"))])
        self.assertEqual([d["id"] for d in drafts], ["d1", "d2"])
        self.assertEqual(drafts[1]["segments"][0], {"start": 12.04, "end": 20.0, "lead": "picture"})
        self.assertEqual([(m["id"], m["start"], m["end"], m["kind"]) for m in moments],
                         [("m1", 10.0, 40.0, "person"), ("m2", 0.0, 600.0, "line")])

    def test_without_a_video_the_payload_is_the_only_part(self):
        self.run_draft(source_video=None)
        call = self.calls[-1]
        self.assertEqual(len(call["parts"]), 1)
        self.assertIn('"shot_log"', call["parts"][0]["text"])
        self.assertIn("You can't watch the video", call["system"])
        self.video_parts.assert_not_called()

    def test_long_sources_are_watched_at_the_long_video_rate(self):
        self.run_draft(duration=1500.0)
        self.assertEqual(self.calls[-1]["parts"][0]["fps"], 0.5)

    def test_payload_content(self):
        self.run_draft()
        payload = self.payload()
        self.assertEqual(list(payload), ["title", "description", "duration", "shot_log",
                                         "transcript", "account_context", "used_ranges"])
        self.assertEqual(payload["title"], "Fire season")
        self.assertEqual(len(payload["description"]), 2000)
        self.assertEqual(payload["duration"], 600.0)
        self.assertEqual(payload["transcript"], ENTRIES)
        self.assertEqual(payload["account_context"], {"rejected_here": [{"story": "x"}]})
        self.assertEqual(payload["used_ranges"], [{"start": 500, "end": 520}])
        beats = payload["shot_log"]["beats"]
        self.assertEqual(beats[1]["cuts"], [12.04])             # the weak change is left out
        self.assertEqual(payload["shot_log"]["people"][0]["name"], "Bill Nye")

    def test_sections(self):
        drafts, _ = self.run_draft(window={"start": 25, "end": 45}, context={})
        payload = self.payload()
        self.assertEqual(payload["section"], {"start": 25.0, "end": 45.0})
        self.assertNotIn("account_context", payload)
        self.assertEqual((payload["transcript"][0]["start"], payload["transcript"][-1]["start"]),
                         (0.0, 70.0))                           # the section ± 30 s
        self.assertEqual([b["start"] for b in payload["shot_log"]["beats"]], [10.0, 30.0, 40.0])
        self.assertIn("ONE SECTION of a longer video: 25s to 45s", self.calls[-1]["system"])
        # Clamped into the section: 30-52 -> 30-45; the other draft loses its 12-20 opening
        # and keeps 30-40 (10 s, still above 0.8 x 12 s).
        self.assertEqual([d["segments"] for d in drafts],
                         [[{"start": 30.0, "end": 45.0, "lead": "word"}],
                          [{"start": 30.0, "end": 40.0, "lead": "word"}]])
        # A tighter section leaves both at 30-38 (8 s): below 0.8 x 12 s, so both go.
        # The editor did draft, so that's a mechanical loss, not "nothing found".
        with self.assertRaises(editor.DraftsDropped) as caught:
            self.run_draft(window={"start": 25, "end": 38})
        self.assertEqual(caught.exception.reasons, {"too short": 2})

    def test_claude_reads_the_payload_then_keyframes(self):
        frames = [(12.2, b"a"), (45.0, b"b")]
        self.run_draft(role="claude", keyframe_frames=frames)
        call = self.calls[-1]
        self.assertEqual(call["model"], "claude-opus-5-5")
        self.assertEqual((call["effort"], call["max_tokens"]), ("high", 32000))
        content = call["content"]
        self.assertIn('"shot_log"', content[0]["text"])
        self.assertEqual(content[1:], [{"type": "text", "text": "Keyframes:"},
                                       {"type": "image", "t": 12.2, "clock": "video"},
                                       {"type": "image", "t": 45.0, "clock": "video"}])
        self.assertEqual(call["system"], prompts.editor_system(
            BRAND, PLAYBOOK_TEXT, "Never fake a quote.", n=10, min_s=12, max_s=45, max_segments=6,
            sees_video=False, section=None, section_topic="", has_words=True, has_keyframes=True))
        self.run_draft(role="claude", keyframe_frames=[])
        self.assertEqual(len(self.calls[-1]["content"]), 1)
        self.assertNotIn("Keyframes from the strongest shots", self.calls[-1]["system"])

    def test_provider_errors_propagate_with_their_cost(self):
        exc = gemini.GeminiBlocked("SAFETY")
        exc.usage = usage(0.02)
        with mock.patch.object(gemini, "generate_json", side_effect=exc):
            meter = FakeMeter()
            with self.assertRaises(gemini.GeminiBlocked):
                self.run_draft(meter=meter)
        self.assertEqual(meter.added, [("draft", usage(0.02))])

    def test_a_video_that_cannot_be_attached_is_left_out(self):
        # The whole proxy is too big to send inline and its upload times out (or
        # Google can't process it): draft from the shot log rather than failing a
        # run whose perception is already paid for.
        self.video_parts.side_effect = gemini.GeminiError(
            "upload proxy.mp4: upload failed: timed out")
        drafts, _ = self.run_draft()
        self.assertEqual([d["id"] for d in drafts], ["d1", "d2"])
        (call,) = self.calls
        self.assertEqual(len(call["parts"]), 1)
        self.assertIn('"shot_log"', call["parts"][0]["text"])
        self.assertIn("You can't watch the video", call["system"])
        self.assertEqual(call["system"], prompts.editor_system(
            BRAND, PLAYBOOK_TEXT, "Never fake a quote.", n=10, min_s=12, max_s=45, max_segments=6,
            sees_video=False, section=None, section_topic="", has_words=True))

    def test_an_exhausted_quota_still_stops_the_draft(self):
        self.video_parts.side_effect = gemini.GeminiQuota("Gemini prepaid credits are depleted")
        with self.assertRaises(gemini.GeminiQuota):
            self.run_draft()
        self.assertEqual(self.calls, [])

    def test_drafts_all_dropped_by_cleaning_are_a_failure_not_an_answer(self):
        # A reroll on a video whose best stretches are taken: every draft lies in
        # used footage, so cleaning leaves nothing (or too little) of any of them.
        self.answer = {"moments": [{"id": "m1", "start": 100, "end": 140, "kind": "line",
                                    "why": "x"}],
                       "drafts": [raw_draft([{"start": 100 + i, "end": 130 + i, "lead": "word"}])
                                  for i in range(10)] + ["junk"]}
        with self.assertRaises(editor.DraftsDropped) as caught:
            self.run_draft(used=[{"start": 95, "end": 150}])
        self.assertEqual(caught.exception.reasons, {"inside used footage": 10, "unreadable": 1})
        self.assertIn("11 drafts", str(caught.exception))
        self.assertEqual([s for s, _ in self.meter.added], ["draft"])   # the call was charged
        # Some survive: an answer, as before. Used footage to 128 s leaves 128-138
        # and 128-139 long enough (>= 9.6 s); the other eight are too short.
        drafts, _ = self.run_draft(used=[{"start": 95, "end": 128}])
        self.assertEqual([d["segments"] for d in drafts],
                         [[{"start": 128.0, "end": 138.0, "lead": "word"}],
                          [{"start": 128.0, "end": 139.0, "lead": "word"}]])

    def test_an_answer_with_no_drafts_is_empty_not_a_failure(self):
        self.answer = {"moments": [{"id": "m1", "start": 10, "end": 40, "kind": "line",
                                    "why": "x"}], "drafts": []}
        drafts, moments = self.run_draft()
        self.assertEqual(drafts, [])
        self.assertEqual([m["id"] for m in moments], ["m1"])

    def test_unknown_role(self):
        with self.assertRaises(ValueError):
            self.run_draft(role="gpt")


class DraftsDroppedInTheRunTest(TempCase):
    """The real editor inside ``engine.run`` (perception, the shot map and the
    model faked): drafts that cleaning drops to the last one stop the run with
    ``EngineError``, which jobs answers with the transcript suggester; an
    editor that drafted nothing is still the ``empty_drafts`` answer."""

    def setUp(self):
        super().setUp()
        self.answer = {"moments": [], "drafts": []}
        self.patch(shotlog, "build", side_effect=lambda **kw: shot_log())
        self.patch(media, "shot_map", return_value={"version": 1, "duration": 600.0,
                                                    "cuts": [], "quiet": [], "error": ""})
        self.patch(media, "perception_proxy", return_value=self.file("proxy.mp4"))
        self.patch(gemini, "video_parts", side_effect=fake_video_parts)
        self.patch(gemini, "generate_json",
                   side_effect=lambda model, **kw: (self.answer, usage(0.04, model)))
        self.patch(llm, "_operator_rules", return_value="")

    def run_engine(self, **kw):
        args = dict(setup="gemini", settings=SETTINGS, brand=BRAND,
                    playbook=playbook.DEFAULT_PLAYBOOK, video_id="vid", title="Fire season",
                    description="", duration=600.0, source=self.dir / "src.mp4", words=[],
                    transcript=list(ENTRIES), context={}, rejected=[], examples=[], used=[],
                    window=None)
        args.update(kw)
        return engine.run(**args)

    def test_every_draft_cleaned_away_is_a_failure(self):
        self.answer = {"moments": [],
                       "drafts": [raw_draft([{"start": 100 + i, "end": 130 + i, "lead": "word"}])
                                  for i in range(10)]}
        meter = engine.CostMeter()
        with self.assertRaises(engine.EngineError) as caught:
            self.run_engine(used=[{"start": 95, "end": 150}], meter=meter)
        self.assertIn("drafting failed: DraftsDropped", str(caught.exception))
        self.assertIn("10 inside used footage", str(caught.exception))
        self.assertAlmostEqual(meter.by_stage["draft"], 0.04)     # charged once

    def test_an_editor_that_drafted_nothing_is_an_answer(self):
        result = self.run_engine()
        self.assertEqual(result.outcome, "empty_drafts")
        self.assertEqual(result.top, [])


class ReviseTest(EditorCase):
    def original(self):
        return {"id": "d2", "moment_id": "m1", "archetype": "voice", "lens": "person",
                "story": "s", "story_key": "k", "hook": "h", "payoff": "p",
                "segments": [{"start": 100.0, "end": 120.0, "lead": "word"}], "why": "w",
                "risk": "r", "draft_caption": "c", "challenges_rules": "", "origin": "draft",
                "parent": None, "editor_rank": 1, "critique": {"score": 4}, "score": 3.5,
                "render": "/tmp/x.mp4", "credits": ""}

    def run_revise(self, answer, **kw):
        self.answer = answer
        args = dict(role="gemini", settings=SETTINGS, brand=BRAND, playbook_text=PLAYBOOK_TEXT,
                    rules="", draft=self.original(),
                    critique={"score": 4, "fix": "start at 0:03", "error": "", "usage": usage()},
                    shotlog=shot_log(), shot_map={"cuts": [{"t": 102.0, "score": 50}]},
                    entries=ENTRIES, has_words=True, used=[], window=None, min_s=12, max_s=45,
                    max_segments=6, source_video=self.proxy, meter=FakeMeter())
        args.update(kw)
        self.meter = args["meter"]
        return editor.revise(**args)

    def test_unchanged_cut_is_none(self):
        answer = {"change": "it already works", "archetype": "voice",
                  "segments": [{"start": 100.02, "end": 120.0, "lead": "word"}]}
        self.assertIsNone(self.run_revise(answer))
        self.assertEqual(self.meter.added[0][0], "revise")

    def test_invalid_answer_is_none(self):
        for segments in ([], [{"start": 100, "end": 105, "lead": "word"}]):
            with self.subTest(segments=segments):
                self.assertIsNone(self.run_revise({"change": "x", "archetype": "voice",
                                                   "segments": segments}))

    def test_a_real_revision(self):
        answer = {"change": "open on the flood", "archetype": "scene",
                  "segments": [{"start": 102, "end": 125, "lead": "picture"}]}
        out = self.run_revise(answer)
        self.assertEqual(out["id"], "d2r")
        self.assertEqual((out["origin"], out["parent"], out["archetype"]), ("revision", "d2", "scene"))
        self.assertEqual(out["segments"], [{"start": 102.0, "end": 125.0, "lead": "picture"}])
        self.assertEqual(out["change"], "open on the flood")
        self.assertEqual(out["editor_rank"], 1)
        for stale in ("critique", "score", "render", "credits"):
            self.assertNotIn(stale, out)
        answer["archetype"] = "nonsense"
        self.assertEqual(self.run_revise(answer)["archetype"], "voice")

    def test_payload_and_parts(self):
        self.run_revise({"change": "x", "archetype": "voice", "segments": []},
                        used=[{"start": 0, "end": 5}])
        call = self.calls[-1]
        self.assertEqual(call["parts"][0], {"video": str(self.proxy), "fps": 1.0})
        payload = self.payload(call)
        self.assertEqual(payload["draft"]["clip_map"], [
            {"clip_start": 0.0, "clip_end": 20.0, "source_start": 100.0, "source_end": 120.0,
             "lead": "word"}])
        self.assertEqual(payload["critique"], {"score": 4, "fix": "start at 0:03"})
        # Shot log and transcript cover the draft ± 60 s (40-180 s); the 30-40 beat only
        # touches it.
        self.assertEqual([b["start"] for b in payload["shot_log"]["beats"]], [40.0, 60.0])
        self.assertEqual((payload["transcript"][0]["start"], payload["transcript"][-1]["start"]),
                         (40.0, 175.0))
        self.assertEqual(payload["used_ranges"], [{"start": 0, "end": 5}])
        self.assertEqual(call["system"], prompts.revise_system(
            BRAND, PLAYBOOK_TEXT, "", min_s=12, max_s=45, max_segments=6, sees_video=True,
            has_words=True))
        self.assertEqual(call["schema"]["properties"]["archetype"]["enum"], ARCHETYPES + ["other"])

    def test_a_video_that_cannot_be_attached_is_left_out(self):
        self.video_parts.side_effect = gemini.GeminiError("files/abc: state FAILED")
        out = self.run_revise({"change": "open on the flood", "archetype": "scene",
                               "segments": [{"start": 102, "end": 125, "lead": "picture"}]})
        self.assertEqual(out["id"], "d2r")
        call = self.calls[-1]
        self.assertEqual(len(call["parts"]), 1)
        self.assertEqual(call["system"], prompts.revise_system(
            BRAND, PLAYBOOK_TEXT, "", min_s=12, max_s=45, max_segments=6, sees_video=False,
            has_words=True))
        self.video_parts.side_effect = gemini.GeminiQuota("daily quota")
        with self.assertRaises(gemini.GeminiQuota):
            self.run_revise({"change": "x", "archetype": "voice", "segments": []})

    def test_claude_reviser_reads_the_payload_only(self):
        self.run_revise({"change": "x", "archetype": "voice",
                         "segments": [{"start": 95, "end": 120, "lead": "word"}]}, role="claude")
        call = self.calls[-1]
        self.assertEqual(len(call["content"]), 1)
        self.assertIn("You can't watch the video", call["system"])

    def test_used_footage_is_subtracted_from_the_revision(self):
        out = self.run_revise({"change": "x", "archetype": "voice",
                               "segments": [{"start": 90, "end": 128, "lead": "word"}]},
                              used=[{"start": 120, "end": 130}])
        self.assertEqual(out["segments"], [{"start": 90.0, "end": 120.0, "lead": "word"}])


class KeyframesTest(TempCase):
    def test_strongest_shots_then_highlights(self):
        grabbed = []

        def fake_sample(path, *, start, end, interval, width):
            grabbed.append((start, end, interval, width))
            if abs(start - 60.5) < 1e-6:
                raise media.MediaError("no frames")
            return [(start, b"jpg")]

        log = shot_log()
        shot_map = {"cuts": [{"t": 22.0, "score": 30}, {"t": 41.0, "score": 25},
                             {"t": 45.0, "score": 2}]}
        with mock.patch.object(media, "sample_frames", side_effect=fake_sample):
            three = editor.keyframes(self.dir / "src.mp4", log, shot_map, settings=SETTINGS,
                                     limit=3)
            every = editor.keyframes(self.dir / "src.mp4", log, shot_map, settings=SETTINGS,
                                     limit=10)
        # Beats with visual >= 4 open on their first cut (+0.15 s); the famous speaker's beat
        # has no cut and uses its start + 0.5 s; the highlight at 22.5 is too close to 22.15.
        self.assertEqual([t for t, _ in three], [22.15, 30.5, 41.15])
        self.assertEqual([t for t, _ in every], [22.15, 30.5, 41.15])   # 60.5 failed: skipped
        self.assertIn((22.15, 22.65, 1.0, 448), grabbed)

    def test_nothing_to_grab(self):
        self.assertEqual(editor.keyframes(None, shot_log(), {}, settings=SETTINGS, limit=5), [])
        self.assertEqual(editor.keyframes(self.dir / "s.mp4", shot_log(), {}, settings=SETTINGS,
                                          limit=0), [])


# ---- critic ---------------------------------------------------------------------------------

class BordaTest(unittest.TestCase):
    CASES = [
        ([["a", "b", "c"], ["a", "c", "b"]], None, ["a", "b", "c"]),
        ([["a", "b", "c"], ["a", "c", "b"]], {"c": 7, "b": 5}, ["a", "c", "b"]),
        ([["a", "b", "c"], ["c", "b", "a"]], None, ["a", "b", "c"]),
        ([["a", "b", "c"], ["c", "b", "a"]], {"b": 9, "a": None}, ["b", "a", "c"]),
        ([["a", "b", "c"], ["c", "b", "a"]], {"c": 1.5, "a": "2"}, ["a", "c", "b"]),
        ([["a", "b"], ["b"]], None, ["b", "a"]),
        ([["a", "a", "b"]], None, ["a", "b"]),
        ([["x", "y", "z"]], None, ["x", "y", "z"]),
        ([], None, []),
    ]

    def test_table(self):
        for rankings, tiebreak, expected in self.CASES:
            with self.subTest(rankings=rankings, tiebreak=tiebreak):
                self.assertEqual(critic.borda(rankings, tiebreak=tiebreak), expected)


BASE_CRITIQUE = {"score": 7, "topic": "central", "makes_sense": True, "ends_well": True,
                 "stops_scroll_muted": True, "stops_scroll": True, "bored_at": None,
                 "visual": 3, "audio": 3, "error": ""}


class ColdScoreTest(unittest.TestCase):
    CASES = [
        ({}, 20, "", 7.0),
        ({"error": "boom"}, 20, "", -1e9),
        ({"topic": "incidental"}, 20, "", 5.0),
        ({"topic": "none"}, 20, "", 5.0),
        ({"topic": "allegory"}, 20, "", 7.0),
        ({"makes_sense": False}, 20, "", 5.0),
        ({"ends_well": False}, 20, "", 6.0),
        ({"stops_scroll_muted": False, "stops_scroll": False}, 20, "", 5.5),
        ({"stops_scroll": False}, 20, "", 6.5),
        ({"stops_scroll_muted": False}, 20, "", 6.5),
        ({"bored_at": 0}, 20, "", 5.0),
        ({"bored_at": 10}, 20, "", 6.0),
        ({"bored_at": 25}, 20, "", 7.0),
        ({"bored_at": 0.5}, 0, "", 6.0),
        ({"visual": 5}, 20, "scene", 7.5),
        ({"visual": 1}, 20, "revelation", 6.5),
        ({"visual": 5}, 20, "voice", 7.0),
        ({"audio": 5}, 20, "", 7.4),
        ({"audio": 1}, 20, "voice", 6.6),
        ({"score": 6, "topic": "none", "makes_sense": False, "ends_well": False,
          "stops_scroll_muted": False, "stops_scroll": False, "bored_at": 5, "visual": 2,
          "audio": 4}, 20, "scene", -2.05),
    ]

    def test_table(self):
        for change, seconds, archetype, expected in self.CASES:
            with self.subTest(change=change, archetype=archetype):
                critique = {**BASE_CRITIQUE, **change}
                self.assertAlmostEqual(critic.cold_score(critique, clip_seconds=seconds,
                                                         archetype=archetype), expected)


GOOD_ANSWER = {
    "first_impression": "A man in safety glasses", "stops_scroll_muted": "true",
    "muted_reason": "face", "stops_scroll": True, "stop_reason": "x" * 400,
    "person_at_start": True, "about": "fire", "makes_sense": True, "dangling": "",
    "bored_at": 40, "dead_air": [{"start": -1, "end": 5}, {"start": 30, "end": 50}, [10, 12],
                                 {"start": 9, "end": 3}, "junk"],
    "ends_well": False, "topic": "Allegory", "topic_note": "film", "person": "Bill Nye",
    "visual": 0, "audio": "4", "score": 15, "fix": "cut 0:12-0:18",
}


class ColdWatchTest(TempCase):
    def setUp(self):
        super().setUp()
        self.render = self.file("draft.mp4")
        self.calls = []
        self.answer = GOOD_ANSWER
        self.items = []
        self.patch(gemini, "video_parts", side_effect=self.fake_parts)
        self.patch(gemini, "generate_json", side_effect=self.fake_call)
        self.patch(claude, "text_block", side_effect=lambda t: {"type": "text", "text": t})
        self.patch(claude, "frames_blocks", side_effect=lambda frames, clock="video": [
            {"type": "image", "t": t, "clock": clock} for t, _ in frames])
        self.patch(claude, "json_call", side_effect=self.fake_call)
        self.clip_frames = self.patch(media, "clip_frames",
                                      return_value=[(0.0, b"a"), (1.0, b"b")])

    def fake_parts(self, items, **kw):
        self.items.append(items)
        return fake_video_parts(items)

    def fake_call(self, model, **kw):
        self.calls.append({"model": model, **kw})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer, usage(0.003, model)

    def watch(self, role="gemini", seconds=32.0, **kw):
        args = dict(role=role, settings=SETTINGS, brand=BRAND, render=self.render,
                    clip_seconds=seconds, clip_words=[], credits="", meter=FakeMeter())
        args.update(kw)
        self.meter = args["meter"]
        return critic.cold_watch(**args)

    def test_gemini_request(self):
        self.watch(credits="0-12 s: Bill Nye, science educator (named in the title)")
        self.assertEqual(self.items[-1], [
            {"path": self.render, "fps": 2.0},
            {"path": self.render, "fps": 8.0, "start": 0.0, "end": 3.0}])
        call = self.calls[-1]
        self.assertEqual(call["model"], "gemini-3.8-flash")
        self.assertEqual(call["system"], prompts.cold_system(BRAND, mode="video"))
        self.assertIs(call["schema"], critic.CRITIQUE_SCHEMA)
        self.assertEqual((call["thinking"], call["media_resolution"]), ("LOW", "MEDIA_RESOLUTION_LOW"))
        text = call["parts"][-1]["text"]
        self.assertTrue(text.startswith("Clip length: 32 s."))
        self.assertIn("Credits: 0-12 s: Bill Nye", text)
        self.assertEqual(len(call["parts"]), 3)
        self.assertEqual(self.meter.added, [("cold", usage(0.003, "gemini-3.8-flash"))])
        self.watch(seconds=2.5)
        self.assertEqual(self.items[-1][1]["end"], 2.5)
        self.assertIn("Credits: none", self.calls[-1]["parts"][-1]["text"])

    def test_answers_are_coerced(self):
        out = self.watch()
        self.assertEqual(list(out), list(critic.CRITIQUE_SCHEMA["properties"]) + ["error"])
        self.assertEqual(out["score"], 10)
        self.assertEqual((out["visual"], out["audio"]), (1, 4))
        self.assertIsNone(out["bored_at"])                       # past the end = watched it all
        self.assertEqual(out["dead_air"], [{"start": 0.0, "end": 5.0}, {"start": 30.0, "end": 32.0},
                                           {"start": 10.0, "end": 12.0}])
        self.assertEqual(out["topic"], "allegory")
        self.assertIs(out["stops_scroll_muted"], True)
        self.assertIs(out["ends_well"], False)
        self.assertEqual(len(out["stop_reason"]), 300)
        self.assertEqual(out["error"], "")
        for change, key, expected in [({"bored_at": -2}, "bored_at", 0.0),
                                      ({"bored_at": "0:12"}, "bored_at", 12.0),
                                      ({"topic": "weird"}, "topic", "central"),
                                      ({"stops_scroll": "no"}, "stops_scroll", False)]:
            with self.subTest(change=change):
                self.answer = {**GOOD_ANSWER, **change}
                self.assertEqual(self.watch()[key], expected)

    def test_no_score_is_a_failure_not_a_verdict(self):
        self.answer = {**GOOD_ANSWER, "score": None}
        out = self.watch()
        self.assertTrue(out["error"])
        self.assertEqual(critic.cold_score(out, clip_seconds=32), -1e9)

    def test_errors_become_a_not_reviewed_critique_with_their_cost(self):
        exc = gemini.GeminiBlocked("SAFETY")
        exc.usage = usage(0.002)
        self.answer = exc
        out = self.watch()
        self.assertIn("GeminiBlocked", out["error"])
        self.assertEqual((out["score"], out["topic"], out["stops_scroll"], out["makes_sense"]),
                         (0, "central", False, False))
        self.assertEqual(out["usage"], usage(0.002))
        self.assertEqual(self.meter.added, [("cold", usage(0.002))])
        self.answer = RuntimeError("network down")
        out = self.watch()
        self.assertNotIn("usage", out)
        self.assertTrue(self.watch(role="nobody")["error"])

    def test_claude_reads_frames_and_words(self):
        words = [{"word": "Hello", "start": 0.5, "end": 0.8}, {"word": "world.", "start": 0.9, "end": 1.2}]
        out = self.watch(role="claude", clip_words=words, credits="0-5 s: Bill Nye")
        self.clip_frames.assert_called_with(self.render, interval=1.0, width=448, max_frames=60)
        call = self.calls[-1]
        self.assertEqual(call["model"], "claude-opus-5-5")
        self.assertEqual(call["effort"], "medium")
        self.assertEqual(call["system"], prompts.cold_system(BRAND, mode="frames", frame_interval=1.0))
        text = call["content"][0]["text"]
        self.assertIn("Credits: 0-5 s: Bill Nye", text)
        self.assertIn("Words heard (clip seconds):\n[0.5] Hello world.", text)
        self.assertEqual(call["content"][1:], [{"type": "image", "t": 0.0, "clock": "clip"},
                                               {"type": "image", "t": 1.0, "clock": "clip"}])
        self.assertEqual(out["score"], 10)
        # A 90 s clip is sampled more sparsely (60 frames at most).
        self.watch(role="claude", seconds=90.0,
                   clip_words=[{"start": 1.0, "end": 3.0, "text": "caption entry"}])
        self.assertEqual(self.calls[-1]["system"],
                         prompts.cold_system(BRAND, mode="frames", frame_interval=1.5))
        self.assertIn("[1.0] caption entry", self.calls[-1]["content"][0]["text"])

    def test_claude_with_no_frames_is_not_reviewed(self):
        self.clip_frames.return_value = []
        self.assertTrue(self.watch(role="claude")["error"])
        self.assertEqual(self.calls, [])


class RankTest(TempCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.answers = {}                 # pass number -> ranking letters or an exception
        self.items = []
        self.patch(gemini, "video_parts", side_effect=self.fake_parts)
        self.patch(gemini, "generate_json", side_effect=self.fake_call)
        self.patch(claude, "text_block", side_effect=lambda t: {"type": "text", "text": t})
        self.patch(claude, "frames_blocks", side_effect=lambda frames, clock="video": [
            {"type": "image", "src": src, "clock": clock} for src, _ in frames])
        self.patch(claude, "json_call", side_effect=self.fake_call)
        self.clip_frames = self.patch(media, "clip_frames", side_effect=lambda path, **kw: [
            (Path(path).name, b"x")])
        self.drafts = [{"id": f"d{i}", "segments": [{"start": 0, "end": 20}],
                        "clip_words": [{"word": f"word{i}", "start": 0.5, "end": 0.9}]}
                       for i in range(1, 4)]
        self.renders = {d["id"]: self.file(f"{d['id']}.mp4") for d in self.drafts}
        self.examples = [{"post_pk": 11, "title": "Nye hit", "label": "hit", "percentile": 98.0,
                          "views": 21895},
                         {"post_pk": 12, "title": "Missing file", "label": "hit", "percentile": 95.0,
                          "views": 100},
                         {"post_pk": 13, "title": "Aerial flop", "label": "flop", "percentile": 10.0,
                          "views": 50}]
        self.example_files = {11: self.file("ex11.mp4"), 12: self.dir / "gone.mp4",
                              13: self.file("ex13.mp4")}

    def fake_parts(self, items, **kw):
        self.items.append(items)
        return fake_video_parts(items)

    def fake_call(self, model, **kw):
        self.calls.append({"model": model, **kw})
        label = kw.get("label", "")
        answer = self.answers.get(label[-1], ["A", "B", "C"])
        if isinstance(answer, Exception):
            raise answer
        return {"notes": [], "ranking": answer, "best_reason": "r"}, usage(0.02, model)

    def rank(self, role="gemini", **kw):
        args = dict(role=role, settings=SETTINGS, brand=BRAND, playbook_text=PLAYBOOK_TEXT,
                    drafts=self.drafts, renders=self.renders,
                    credits={"d1": "0-5 s: Bill Nye"}, scores={"d1": 5.0, "d2": 6.0, "d3": 4.0},
                    examples=self.examples, example_files=self.example_files, meter=FakeMeter())
        args.update(kw)
        self.meter = args["meter"]
        return critic.rank(**args)

    def call_for(self, n):
        return next(c for c in self.calls if c["label"].endswith(str(n)))

    def test_gemini_assembly_and_merge(self):
        self.answers = {"1": ["B", "A", "C"], "2": ["A", "B", "C"]}
        out = self.rank()
        # Pass 1 (d1 d2 d3): d2 d1 d3. Pass 2 (d3 d2 d1): d3 d2 d1. Borda: d2 5, d3 4, d1 3.
        self.assertEqual(out, ["d2", "d3", "d1"])
        (items,) = self.items      # parts are built once and reused by both passes
        self.assertEqual(items, [{"path": self.example_files[11], "upload": True},
                                 {"path": self.example_files[13], "upload": True},
                                 {"path": self.renders["d1"], "fps": 2.0},
                                 {"path": self.renders["d2"], "fps": 2.0},
                                 {"path": self.renders["d3"], "fps": 2.0}])
        texts = [p.get("text") or p["video"] for p in self.call_for(1)["parts"]]
        self.assertEqual(texts, [
            "Example 1 — Nye hit: top 2% of this account's posts from the same weeks "
            "(21,895 views at 6 h)", str(self.example_files[11]),
            "Example 2 — Aerial flop: you labelled this one not gripping", str(self.example_files[13]),
            "Draft A — Credits: 0-5 s: Bill Nye", str(self.renders["d1"]),
            "Draft B — Credits: none", str(self.renders["d2"]),
            "Draft C — Credits: none", str(self.renders["d3"]),
            "Rank all 3 drafts."])
        texts = [p.get("text") or p["video"] for p in self.call_for(2)["parts"]]
        self.assertEqual(texts[4:10], ["Draft A — Credits: none", str(self.renders["d3"]),
                                       "Draft B — Credits: none", str(self.renders["d2"]),
                                       "Draft C — Credits: 0-5 s: Bill Nye", str(self.renders["d1"])])
        call = self.call_for(1)
        self.assertEqual(call["model"], "gemini-3.8-flash")
        self.assertEqual(call["thinking"], "MEDIUM")
        self.assertEqual(call["system"], prompts.rank_system(BRAND, PLAYBOOK_TEXT, k=3,
                                                             has_examples=True, mode="video"))
        schema = call["schema"]
        self.assertEqual(list(schema["properties"]), ["notes", "ranking", "best_reason"])
        self.assertEqual(schema["properties"]["ranking"]["items"]["enum"], ["A", "B", "C"])
        self.assertEqual([stage for stage, _ in self.meter.added], ["rank", "rank"])

    def test_letters_are_read_forgivingly(self):
        self.answers = {"1": ["Draft b", "z", "A.", "a"], "2": ["C", "A", "B"]}
        # Pass 1: d2 d1 (+ d3 missing, appended). Pass 2 (d3 d2 d1): d1 d3 d2.
        self.assertEqual(self.rank(), ["d1", "d2", "d3"])

    def test_one_failed_pass_uses_the_other(self):
        self.answers = {"1": gemini.GeminiError("HTTP 500"), "2": ["A", "B", "C"]}
        self.assertEqual(self.rank(), ["d3", "d2", "d1"])

    def test_total_failure_keeps_the_input_order(self):
        self.answers = {"1": gemini.GeminiError("x"), "2": gemini.GeminiError("y")}
        with self.assertLogs("clip_engine.critic", level="WARNING"):
            self.assertEqual(self.rank(), ["d1", "d2", "d3"])
        with mock.patch.object(gemini, "video_parts", side_effect=gemini.GeminiError("upload")):
            with self.assertLogs("clip_engine.critic", level="WARNING"):
                self.assertEqual(self.rank(), ["d1", "d2", "d3"])

    def test_examples_that_cannot_be_attached_are_left_out(self):
        def picky(items, **kw):
            if any(i.get("upload") for i in items):
                raise gemini.GeminiError("upload failed")
            return self.fake_parts(items)

        self.answers = {"1": ["A", "B", "C"], "2": ["A", "B", "C"]}
        with mock.patch.object(gemini, "video_parts", side_effect=picky):
            with self.assertLogs("clip_engine.critic", level="WARNING"):
                out = self.rank()
        self.assertEqual(out, ["d2", "d1", "d3"])     # all tie; d2 has the best cold score
        call = self.call_for(1)
        self.assertFalse(any(str(p.get("text", "")).startswith("Example") for p in call["parts"]))
        self.assertEqual(call["system"], prompts.rank_system(BRAND, PLAYBOOK_TEXT, k=3,
                                                             has_examples=False, mode="video"))

    def test_claude_skips_unreadable_files(self):
        def frames(path, **kw):
            if Path(path).name in ("ex11.mp4", "d2.mp4"):
                raise media.MediaError("corrupt")
            return [(Path(path).name, b"x")]

        self.clip_frames.side_effect = frames
        self.answers = {"1": ["B", "A"], "2": ["B", "A"]}
        # Shown: d1 d3 (d2 has no frames). Pass 1: d3 d1; pass 2 (d3 d1): d1 d3 -> tie;
        # d1's cold score (5.0) beats d3's (4.0). d2 goes last.
        self.assertEqual(self.rank(role="claude"), ["d1", "d3", "d2"])
        content = [b.get("text") or b["src"] for b in self.call_for(1)["content"]]
        self.assertEqual(content[0], "Example 1 — Aerial flop: you labelled this one not gripping")
        self.assertNotIn("d2.mp4", content)

    def test_ten_videos_at_most_examples_dropped_first(self):
        drafts = [{"id": f"d{i}", "segments": []} for i in range(1, 13)]
        renders = {d["id"]: self.file(f"{d['id']}.mp4") for d in drafts}
        self.answers = {"1": [chr(65 + i) for i in range(10)], "2": [chr(65 + i) for i in range(10)]}
        out = self.rank(drafts=drafts, renders=renders)
        self.assertEqual(len(self.items[-1]), 10)
        self.assertFalse(any(i.get("upload") for i in self.items[-1]))
        self.assertEqual(out[-2:], ["d11", "d12"])          # beyond the cap: appended in order
        self.items.clear()
        self.rank(drafts=drafts[:8], renders=renders)
        self.assertEqual(sum(1 for i in self.items[-1] if i.get("upload")), 2)

    def test_drafts_without_renders_are_appended(self):
        renders = dict(self.renders)
        renders["d2"] = self.dir / "never-rendered.mp4"
        self.answers = {"1": ["B", "A"], "2": ["A", "B"]}
        # Shown: d1 d3. Pass 1: d3 d1; pass 2 (d3 d1): d3 d1.
        self.assertEqual(self.rank(renders=renders), ["d3", "d1", "d2"])
        self.assertEqual(self.rank(renders={"d1": self.renders["d1"]}), ["d1", "d2", "d3"])

    def test_list_inputs_and_draft_fallbacks(self):
        drafts = [dict(d, render=str(self.renders[d["id"]]), score=s)
                  for d, s in zip(self.drafts, (1.0, 2.0, 3.0))]
        self.answers = {"1": ["A", "B", "C"], "2": ["A", "B", "C"]}
        # Every draft ties on points; the cold scores (from the drafts) break it.
        self.assertEqual(self.rank(drafts=drafts, renders=None, scores=None,
                                   credits=["c1", "", ""]), ["d3", "d2", "d1"])
        self.assertIn("Draft A — Credits: c1",
                      [p.get("text") for p in self.call_for(1)["parts"]])

    def test_claude_assembly(self):
        self.answers = {"1": ["A", "B", "C"], "2": ["A", "B", "C"]}
        self.rank(role="claude")
        calls = self.clip_frames.call_args_list
        self.assertIn(mock.call(self.example_files[11], interval=3.0, width=448, max_frames=10),
                      calls)
        self.assertIn(mock.call(self.renders["d1"], interval=2.0, width=448, max_frames=20), calls)
        call = self.call_for(1)
        self.assertEqual(call["model"], "claude-opus-5-5")        # judge -> critic model
        self.assertEqual(call["effort"], "medium")
        self.assertEqual(call["system"], prompts.rank_system(BRAND, PLAYBOOK_TEXT, k=3,
                                                             has_examples=True, mode="frames",
                                                             frame_interval=2.0))
        content = [b.get("text") or b["src"] for b in call["content"]]
        self.assertEqual(content[:4], [
            "Example 1 — Nye hit: top 2% of this account's posts from the same weeks "
            "(21,895 views at 6 h)", "ex11.mp4",
            "Example 2 — Aerial flop: you labelled this one not gripping", "ex13.mp4"])
        self.assertEqual(content[4], "Draft A — Credits: 0-5 s: Bill Nye\n"
                                     "Words heard (clip seconds):\n[0.5] word1")
        self.assertEqual(content[5:8], ["d1.mp4", "Draft B — Credits: none\n"
                                        "Words heard (clip seconds):\n[0.5] word2", "d2.mp4"])
        self.assertEqual(content[-1], "Rank all 3 drafts.")

    def test_one_draft_needs_no_ranking(self):
        self.assertEqual(self.rank(drafts=self.drafts[:1]), ["d1"])
        self.assertEqual(self.calls, [])


class PreferTest(TempCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.answers = {"a": "2", "b": "1"}    # which version wins, by which draft is shown first
        self.patch(gemini, "video_parts", side_effect=fake_video_parts)
        self.patch(gemini, "generate_json", side_effect=self.fake_call)
        self.patch(claude, "text_block", side_effect=lambda t: {"type": "text", "text": t})
        self.patch(claude, "frames_blocks", side_effect=lambda frames, clock="video": [
            {"type": "image", "src": src, "clock": clock} for src, _ in frames])
        self.patch(claude, "json_call", side_effect=self.fake_call)
        self.clip_frames = self.patch(media, "clip_frames", side_effect=lambda path, **kw: [
            (Path(path).name, b"x")])
        self.a = {"id": "d1", "segments": [{"start": 0, "end": 20}]}
        self.b = {"id": "d1r", "segments": [{"start": 2, "end": 20}]}
        self.ra, self.rb = self.file("a.mp4"), self.file("b.mp4")

    def first_shown(self, kw) -> str:
        if "parts" in kw:
            first = next(p["video"] for p in kw["parts"] if "video" in p)
        else:
            first = next(b["src"] for b in kw["content"] if b["type"] == "image")
        return "a" if Path(first).name == "a.mp4" else "b"

    def fake_call(self, model, **kw):
        self.calls.append({"model": model, **kw})
        answer = self.answers[self.first_shown(kw)]
        if isinstance(answer, Exception):
            raise answer
        return {"choice": answer, "reason": "r"}, usage(0.01, model)

    def prefer(self, role="gemini", **kw):
        args = dict(role=role, settings=SETTINGS, brand=BRAND, a=self.a, b=self.b,
                    render_a=self.ra, render_b=self.rb, credits="0-5 s: Bill Nye",
                    intent="The Voice: On the person", meter=FakeMeter())
        args.update(kw)
        self.meter = args["meter"]
        return critic.prefer(**args)

    def test_the_revision_must_win_both_orders(self):
        cases = [({"a": "2", "b": "1"}, "b"), ({"a": "2", "b": "2"}, "a"),
                 ({"a": "1", "b": "2"}, "a"), ({"a": "same", "b": "1"}, "a"),
                 ({"a": "2", "b": "same"}, "a"), ({"a": "1", "b": "1"}, "a"),
                 ({"a": "2", "b": gemini.GeminiError("x")}, "a")]
        for answers, expected in cases:
            with self.subTest(answers=answers):
                self.answers = answers
                self.assertEqual(self.prefer(), expected)

    def test_gemini_request(self):
        self.prefer()
        self.assertEqual(len(self.calls), 2)
        a_first = next(c for c in self.calls if self.first_shown(c) == "a")
        texts = [p.get("text") or Path(p["video"]).name for p in a_first["parts"]]
        self.assertEqual(texts, ["Version 1 — Credits: 0-5 s: Bill Nye", "a.mp4",
                                 "Version 2 — Credits: 0-5 s: Bill Nye", "b.mp4",
                                 "Which version would you post?"])
        b_first = next(c for c in self.calls if self.first_shown(c) == "b")
        self.assertEqual([Path(p["video"]).name for p in b_first["parts"] if "video" in p],
                         ["b.mp4", "a.mp4"])
        self.assertEqual(a_first["system"], prompts.prefer_system(
            BRAND, mode="video", intent="The Voice: On the person"))
        self.assertEqual((a_first["model"], a_first["thinking"]), ("gemini-3.8-flash", "MEDIUM"))
        self.assertIs(a_first["schema"], critic.PREFER_SCHEMA)
        self.assertEqual([s for s, _ in self.meter.added], ["prefer", "prefer"])

    def test_separate_credits_per_version(self):
        self.prefer(credits=("0-5 s: Nye", "0-3 s: Nye"))
        a_first = next(c for c in self.calls if self.first_shown(c) == "a")
        self.assertIn("Version 2 — Credits: 0-3 s: Nye",
                      [p.get("text") for p in a_first["parts"]])

    def test_claude_request(self):
        self.a["clip_words"] = [{"word": "Hi.", "start": 0.2, "end": 0.4}]
        self.assertEqual(self.prefer(role="claude"), "b")
        a_first = next(c for c in self.calls if self.first_shown(c) == "a")
        content = [b.get("text") or b["src"] for b in a_first["content"]]
        self.assertEqual(content, [
            "Version 1 — Credits: 0-5 s: Bill Nye\nWords heard (clip seconds):\n[0.2] Hi.",
            "a.mp4", "Version 2 — Credits: 0-5 s: Bill Nye\nWords heard (clip seconds):\n(none)",
            "b.mp4", "Which version would you post?"])
        self.assertEqual(a_first["system"], prompts.prefer_system(
            BRAND, mode="frames", frame_interval=1.0, intent="The Voice: On the person"))
        self.assertEqual(self.clip_frames.call_count, 2)        # frames once per render

    def test_failures_keep_the_original(self):
        self.clip_frames.side_effect = media.MediaError("broken render")
        self.assertEqual(self.prefer(role="claude"), "a")
        self.assertEqual(self.prefer(role="martian"), "a")
        with mock.patch.object(gemini, "video_parts", side_effect=gemini.GeminiError("upload")):
            self.assertEqual(self.prefer(), "a")


if __name__ == "__main__":
    unittest.main()
