"""The footage-reuse rule between clips of one video (app/clip_proposals.py,
app/llm.py).

Clips may recycle each other's footage — on a video about several artisans,
every clip can reuse the same setup shots — but each must open on its own shot
and tell its own story, and a clip sharing more than 60% of its material with
another is that clip again. These pin down the mechanical half (opening
windows, the shot-map "same shot" test, duplicates), the "other clips" the
suggesters are shown, and the prompts that replaced "never reuse used footage".

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= .venv/bin/python -m pytest -q tests/test_footage_reuse.py
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import clip_proposals, llm, spend
from app.models import Base, Candidate, Channel, ClipProposal, Cut


def seg(start, end):
    return {"start": float(start), "end": float(end)}


def clip(title, *spans):
    return {"title": title, "segments": [seg(a, b) for a, b in spans]}


class OpeningWindowTest(unittest.TestCase):
    def test_the_first_played_segment_opens_the_clip(self):
        self.assertEqual(clip_proposals.opening_window([seg(50, 54), seg(10, 30)]), (50.0, 53.0))
        self.assertEqual(clip_proposals.opening_window([{"start": "x"}, seg(5, 4), seg(7, 9)]),
                         (7.0, 10.0))
        self.assertIsNone(clip_proposals.opening_window([]))

    def test_opening_windows(self):
        self.assertEqual(clip_proposals.opening_windows(
            [clip("a", (100, 115), (40, 50)), clip("empty"), {"segments": [seg(7, 9)]}]),
            [seg(100, 103), seg(7, 10)])


class ClipConflictTest(unittest.TestCase):
    LOOM = clip("The loom", (100, 115), (40, 50))
    DYER = clip("The dyer", (300, 320))
    # Shot changes at 98, 112 and 250 (strong) and 108 (too weak to count): the
    # loom opens in the 14-second shot 98-112.
    CUTS = [{"t": 98.0, "score": 45.0}, {"t": 112.0, "score": 30.0}, {"t": 108.0, "score": 5.0},
            {"t": 250.0, "score": 25.0}]

    CASES = [
        # (label, clip segments, shot-map cuts, kind, with)
        ("no other clips", [seg(100, 130)], None, "", ""),
        ("reuses the loom's footage further in", [seg(60, 80), seg(100, 112)], None, "", ""),
        ("opens inside the loom's opening window", [seg(101, 130)], None, "same_opening",
         "The loom"),
        ("its own window overlaps from before", [seg(97.5, 130)], None, "same_opening",
         "The loom"),
        ("windows that only touch", [seg(103, 130)], None, "", ""),
        ("windows that only touch, from before", [seg(97, 130)], None, "", ""),
        ("opening on the loom's later footage is fine", [seg(42, 60)], None, "", ""),
        ("same shot on the shot map", [seg(105, 130)], CUTS, "same_opening", "The loom"),
        ("a weak cut doesn't split the shot", [seg(109, 130)], CUTS, "same_opening", "The loom"),
        ("across a strong cut is another shot", [seg(113, 160)], CUTS, "", ""),
        ("before the shot starts is another shot", [seg(80, 96)], CUTS, "", ""),
        ("a shot map without cuts adds nothing", [seg(105, 130)], [], "", ""),
        ("the same clip again", [seg(100, 115), seg(40, 49)], None, "duplicate", "The loom"),
        ("the same material, opened elsewhere", [seg(40, 50), seg(100, 115)], None, "duplicate",
         "The loom"),
        ("over 60% shared but a different shape", [seg(301, 320)], None, "duplicate", "The dyer"),
        ("60% exactly is not a duplicate", [seg(300, 312)], None, "same_opening", "The dyer"),
    ]

    def test_cases(self):
        others = [self.LOOM, self.DYER]
        for label, segments, cuts, kind, title in self.CASES:
            with self.subTest(label):
                got, other = clip_proposals.clip_conflict(
                    segments, [] if label == "no other clips" else others, cuts=cuts)
                self.assertEqual(got, kind)
                self.assertEqual((other or {}).get("title", ""), title)
                self.assertEqual(clip_proposals.conflict_with_clips(
                    segments, [] if label == "no other clips" else others, cuts=cuts), kind)

    def test_a_long_take_is_not_one_opening(self):
        # A 50-second interview take: two different stories can both open in it,
        # as long as their 3-second opening windows don't overlap.
        long_take = [{"t": 90.0, "score": 45.0}, {"t": 140.0, "score": 30.0}]
        self.assertEqual(clip_proposals.conflict_with_clips(
            [seg(125, 150)], [self.LOOM], cuts=long_take), "")
        self.assertEqual(clip_proposals.conflict_with_clips(
            [seg(101, 150)], [self.LOOM], cuts=long_take), "same_opening")

    def test_min_shot_score_decides_which_cuts_count(self):
        segments = [seg(105, 130)]
        self.assertEqual(clip_proposals.conflict_with_clips(
            segments, [self.LOOM], cuts=self.CUTS, min_shot_score=50.0), "")
        self.assertEqual(clip_proposals.conflict_with_clips(
            segments, [self.LOOM], cuts=[120.0, (140.0, 99)]), "")   # bare times / pairs
        self.assertEqual(clip_proposals.conflict_with_clips(
            segments, [self.LOOM], cuts=["junk", {"t": "x"}]), "")

    def test_unusable_input(self):
        self.assertEqual(clip_proposals.clip_conflict([], [self.LOOM]), ("", None))
        self.assertEqual(clip_proposals.clip_conflict([seg(101, 130)], [None, "x", clip("e")]),
                         ("", None))


class OtherClipsTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.addCleanup(self.session.close)
        channel = Channel(call_sign="KTST", url="https://example.com")
        self.session.add(channel)
        self.session.flush()
        self.c = Candidate(video_id="v1", channel_pk=channel.id, title="Artisans",
                           url="https://youtu.be/v1")
        self.session.add(self.c)
        self.session.flush()

    def cut(self, segments, title=""):
        row = Cut(candidate_pk=self.c.id, clip_title=title,
                  trim_segments=json.dumps(segments) if segments is not None else "")
        self.session.add(row)
        self.session.flush()
        return row.id

    def test_every_cut_with_segments_in_play_order(self):
        loom = self.cut([seg(100, 115), seg(40, 50)], "  The   loom ")
        self.cut(None)                                        # nothing marked yet
        self.cut([seg(5, 4)])                                 # nothing usable
        bare = self.cut([seg(300, 320)])
        words = [{"word": "weave", "start": 101.0, "end": 101.4},
                 {"word": "dye", "start": 41.0, "end": 41.3},
                 {"word": "skip", "start": 200.0, "end": 200.4}]
        others = clip_proposals.other_clips(self.session, self.c.id, words=words)
        self.assertEqual(others, [
            {"cut_pk": loom, "title": "The loom", "segments": [seg(100, 115), seg(40, 50)],
             "opening": [100.0, 103.0], "words_heard": "weave dye"},
            {"cut_pk": bare, "title": f"Clip {bare}", "segments": [seg(300, 320)],
             "opening": [300.0, 303.0], "words_heard": ""}])
        self.assertEqual([c["cut_pk"] for c in clip_proposals.other_clips(
            self.session, self.c.id, exclude_cut_pk=loom)], [bare])

    def test_words_heard_falls_back_to_the_transcript_and_is_capped(self):
        transcript = [{"start": float(t), "end": float(t + 5), "text": f"w{t} x y z"}
                      for t in range(0, 200, 5)]
        text = clip_proposals.words_heard([seg(100, 200)], [], transcript, limit=6)
        self.assertEqual(text, "w100 x y z w105 x…")
        self.assertEqual(clip_proposals.words_heard([seg(0, 4)], [], transcript), "w0 x y z")

    def test_pending_summary_hides_duplicates_only(self):
        self.cut([seg(100, 115), seg(40, 50)], "The loom")
        for segments in ([seg(100, 115), seg(40, 49)],      # the loom again: hidden
                         [seg(101, 130)],                    # same opening: still offered
                         [seg(60, 80), seg(100, 112)]):      # shared footage: offered
            self.session.add(ClipProposal(candidate_pk=self.c.id, story="s",
                                          proposed_segments=json.dumps(segments)))
        self.session.flush()
        self.assertEqual(clip_proposals.pending_summary(self.session, [self.c.id]),
                         {self.c.id: {"count": 2, "story": "s"}})


class OtherClipsPayloadTest(unittest.TestCase):
    def test_shape(self):
        out = llm.other_clips_payload([
            {"cut_pk": 3, "title": "  The  loom ", "segments": [seg(100.04, 115), seg(40, 50)],
             "words_heard": " ".join(f"w{i}" for i in range(50))},
            {"title": "empty", "segments": [{"start": "x"}]}, None])
        self.assertEqual(out, [{
            "title": "The loom",
            "words_heard": " ".join(f"w{i}" for i in range(40)) + "…",
            "segments": [{"start": 100.0, "end": 115.0}, {"start": 40.0, "end": 50.0}],
            "opening": [100.0, 103.0]}])


class _Prompts(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for p in (mock.patch.object(spend, "_LEDGER", Path(tmp.name) / "spend.json"),
                  mock.patch.object(llm, "_operator_rules", return_value="")):
            p.start()
            self.addCleanup(p.stop)
        self.seen: dict = {}

    def chat(self, answer):
        def fake(model, system, user, max_tokens=0):
            self.seen.update(system=system, payload=json.loads(user))
            return answer
        return mock.patch.object(llm, "_json_chat", side_effect=fake)


TRANSCRIPT = [{"start": float(t), "end": float(t + 5), "text": f"line {t}"}
              for t in range(0, 600, 5)]
OTHERS = [{"title": "The loom", "words_heard": "she weaves every morning",
           "segments": [seg(100, 115), seg(40, 50)], "opening": [100.0, 103.0]}]


class SuggestClipsPromptTest(_Prompts):
    def suggest(self, **kw):
        with self.chat({"clips": []}):
            llm.suggest_clips("claude-sonnet-5", "Artisans", TRANSCRIPT, **kw)

    def test_other_clips_and_the_new_rule(self):
        self.suggest(other_clips=OTHERS, used_ranges=[seg(0, 600)])
        system, payload = self.seen["system"], self.seen["payload"]
        self.assertNotIn("used_ranges", payload)
        self.assertEqual(payload["other_clips"], [{
            "title": "The loom", "words_heard": "she weaves every morning",
            "segments": [seg(100, 115), seg(40, 50)], "opening": [100.0, 103.0]}])
        self.assertIn(llm.OTHER_CLIPS_RULE, system)
        self.assertIn("Reusing their footage is fine and expected", system)
        for gone in ("Do not propose any of that material again", "ALREADY published",
                     "used_ranges"):
            self.assertNotIn(gone, system)

    def test_no_other_clips_no_rule(self):
        self.suggest(used_ranges=[seg(0, 50)])
        self.assertNotIn("other_clips", self.seen["payload"])
        self.assertNotIn("used_ranges", self.seen["payload"])
        self.assertNotIn("DIFFERENT OPENING", self.seen["system"])

    def test_a_proposal_over_used_footage_comes_back_whole(self):
        answer = {"clips": [{"segments": [seg(100, 130)], "story": "s", "confidence": 0.9}]}
        with self.chat(answer):
            out = llm.suggest_clips("claude-sonnet-5", "Artisans", TRANSCRIPT,
                                    other_clips=OTHERS, used_ranges=[seg(0, 600)])
        self.assertEqual([(s["start"], s["end"]) for s in out[0]["segments"]], [(100.0, 130.0)])


class ReviseClipPromptTest(_Prompts):
    def revise(self, **kw):
        with self.chat({"segments": [seg(60, 80)], "note": "ok"}):
            return llm.revise_clip("claude-sonnet-5", "Artisans", "use the loom setup",
                                   [seg(60, 80)], TRANSCRIPT, **kw)

    def test_other_clips_and_the_new_rule(self):
        self.revise(other_clips=OTHERS, used_ranges=[seg(100, 115)])
        system, payload = self.seen["system"], self.seen["payload"]
        self.assertEqual([c["title"] for c in payload["other_clips"]], ["The loom"])
        self.assertEqual(payload["other_clips"][0]["opening"], [100.0, 103.0])
        self.assertNotIn("used_ranges", payload)
        self.assertIn(llm.OTHER_CLIPS_RULE, system)
        self.assertIn("Footage another clip also uses is never a reason to refuse", system)
        for gone in ("Do not pull any of that material in", "already used", "used_ranges"):
            self.assertNotIn(gone, system)

    def test_without_other_clips(self):
        out = self.revise()
        self.assertNotIn("other_clips", self.seen["payload"])
        self.assertNotIn("DIFFERENT OPENING", self.seen["system"])
        self.assertEqual(out["segments"], [seg(60, 80)])


class ProposeTest(unittest.TestCase):
    """The synchronous transcript pass: other clips go to the model, and the
    opening pull-back is blocked only by their openings (and its siblings')."""

    def test_propose(self):
        candidate = SimpleNamespace(id=7, video_id="v7", title="Artisans", description="",
                                    local_video_path="", transcript_path="",
                                    word_transcript_path="", striking_frames="",
                                    draft_caption="", multi_clip_potential=False,
                                    multi_clip_auto=False)
        model_kwargs, blocked = {}, []

        def fake_suggest(*args, **kw):
            model_kwargs.update(kw)
            return [{"segments": [seg(60, 80)], "story": "a"},
                    {"segments": [seg(200, 230)], "story": "b"}]

        def fake_refine(cand, segments, settings, blocked_now, story=""):
            blocked.append([dict(b) for b in blocked_now])
            return segments

        with mock.patch.object(clip_proposals, "suggest_clips", side_effect=fake_suggest), \
                mock.patch.object(clip_proposals, "other_clips", return_value=list(OTHERS)), \
                mock.patch.object(clip_proposals, "refine_opening", side_effect=fake_refine), \
                mock.patch.object(clip_proposals, "load_words", return_value=[]), \
                mock.patch("app.clip_context.build", return_value={}), \
                mock.patch("app.vision.load_striking", return_value=[]), \
                mock.patch.object(clip_proposals, "log_run", return_value=[]):
            clip_proposals.propose(mock.Mock(), candidate, {}, TRANSCRIPT)
        self.assertEqual(model_kwargs["other_clips"], OTHERS)
        self.assertNotIn("used_ranges", model_kwargs)
        self.assertEqual(blocked, [[seg(100, 103)], [seg(100, 103), seg(60, 63)]])


if __name__ == "__main__":
    unittest.main()
