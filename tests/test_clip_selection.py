"""Tests for clip selection: what the suggester is shown, what it learns from
the account's history, and what the suggestion cards display.

Run with:  python -m unittest tests.test_clip_selection -v
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import clip_context, clip_proposals, llm
from app.models import (Base, Candidate, Channel, ClipProposal, Cut, MetricSnapshot,
                        ThreadsPost, utcnow)


class FitEntriesTest(unittest.TestCase):
    def entries(self, n):
        return [{"start": float(i), "end": i + 0.9, "text": f"w{i}"} for i in range(n)]

    def test_under_the_cap_is_untouched(self):
        e = self.entries(10)
        self.assertEqual(llm._fit_entries(e, 400), e)

    def test_over_the_cap_merges_instead_of_dropping_the_tail(self):
        e = self.entries(1000)
        out = llm._fit_entries(e, 400)
        self.assertLessEqual(len(out), 400)
        # The end of the video is still there — the old [:400] cut it off.
        self.assertEqual(out[-1]["end"], e[-1]["end"])
        self.assertEqual(out[0]["start"], 0.0)
        self.assertEqual(out[0]["text"], "w0 w1 w2")

    def test_merged_edges_stay_on_original_boundaries(self):
        e = self.entries(9)
        out = llm._fit_entries(e, 3)
        starts = {x["start"] for x in e}
        ends = {x["end"] for x in e}
        for x in out:
            self.assertIn(x["start"], starts)
            self.assertIn(x["end"], ends)


class SuggestClipsPromptTest(unittest.TestCase):
    def run_suggest(self, reply, **kw):
        captured = {}

        def fake(model, system, user, max_tokens=0, temperature=0.2):
            captured["system"], captured["user"] = system, json.loads(user)
            return reply

        transcript = [{"start": float(i * 5), "end": i * 5 + 5.0, "text": f"line {i}"}
                      for i in range(40)]
        with mock.patch.object(llm, "_json_chat", side_effect=fake):
            clips = llm.suggest_clips("m", "Title", transcript, **kw)
        return clips, captured

    def test_hook_and_payoff_pass_through(self):
        reply = {"clips": [{"story": "s", "hook": "The water rose", "payoff": "It never left",
                            "segments": [{"start": 10, "end": 30}], "confidence": 0.8}]}
        clips, _ = self.run_suggest(reply)
        self.assertEqual(clips[0]["hook"], "The water rose")
        self.assertEqual(clips[0]["payoff"], "It never left")

    def test_context_and_moments_reach_the_model(self):
        ctx = {"shipped_examples": [{"words_heard": "x"}], "calibration": [], "recently_posted": []}
        moments = [{"t": 12.0, "why": "flooded street"}, {"t": 900.0, "why": "out of range"}]
        _, cap = self.run_suggest({"clips": []}, context=ctx, visual_moments=moments,
                                  window={"start": 0, "end": 60})
        # Empty context pieces are dropped; moments outside the section are too.
        self.assertEqual(cap["user"]["account_context"], {"shipped_examples": [{"words_heard": "x"}]})
        self.assertEqual(cap["user"]["visual_moments"], [{"t": 12.0, "what": "flooded street"}])
        self.assertIn("LEARN FROM THIS ACCOUNT", cap["system"])
        self.assertIn("visual_moments", cap["system"])

    def test_no_context_keeps_the_prompt_lean(self):
        _, cap = self.run_suggest({"clips": []})
        self.assertNotIn("account_context", cap["user"])
        self.assertNotIn("LEARN FROM THIS ACCOUNT", cap["system"])


class ExcerptsTest(unittest.TestCase):
    WORDS = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(200)]

    def test_words_give_exact_opening_and_closing(self):
        segs = [{"start": 0.0, "end": 20.0}, {"start": 60.0, "end": 80.0}]
        out = clip_proposals.excerpts(segs, self.WORDS, [], head=3, tail=2)
        self.assertEqual(out["opening"], "w0 w1 w2…")
        self.assertEqual(out["closing"], "…w158 w159")

    def test_short_single_segment_has_no_separate_closing(self):
        out = clip_proposals.excerpts([{"start": 0.0, "end": 2.0}], self.WORDS, [])
        self.assertEqual(out["opening"], "w0 w1 w2 w3")
        self.assertEqual(out["closing"], "")

    def test_falls_back_to_transcript_lines(self):
        transcript = [{"start": 0, "end": 5, "text": "first line"},
                      {"start": 5, "end": 10, "text": "second line"},
                      {"start": 30, "end": 35, "text": "last line"}]
        out = clip_proposals.excerpts([{"start": 1, "end": 6}, {"start": 31, "end": 33}],
                                      [], transcript)
        self.assertEqual(out["opening"], "first line second line")
        self.assertEqual(out["closing"], "last line")


class CalibrationNotesTest(unittest.TestCase):
    def rows(self, n, **vals):
        return [dict(vals) for _ in range(n)]

    def test_too_few_samples_states_no_bias(self):
        self.assertEqual(clip_context.calibration_notes(
            self.rows(2, start_delta_s=5, end_delta_s=-5), {}), [])

    def test_bias_becomes_a_correction(self):
        notes = clip_context.calibration_notes(self.rows(
            6, start_delta_s=3.0, end_delta_s=-6.0, proposed_duration_s=38,
            final_duration_s=27, proposed_segment_count=1, final_segment_count=3), {})
        text = " ".join(notes)
        self.assertIn("started about 3s too early", text)
        self.assertIn("ran about 6s too long", text)
        self.assertIn("median 27s", text)
        self.assertIn("median 3 segments", text)

    def test_noise_is_not_reported(self):
        notes = clip_context.calibration_notes(self.rows(
            6, start_delta_s=0.4, end_delta_s=-0.5, proposed_duration_s=30,
            final_duration_s=29, proposed_segment_count=3, final_segment_count=3), {})
        self.assertEqual(notes, [])

    def test_dismiss_reasons_are_summarised(self):
        notes = clip_context.calibration_notes([], {"too_long": 4, "weak_visuals": 2})
        self.assertEqual(notes, ["Most common reasons proposals were dismissed: "
                                 "too long (4), weak visuals (2)."])


class PolicyVersionTest(unittest.TestCase):
    base = dict(model="m", max_clips=3, max_segments_per_clip=4, min_seconds=15, max_seconds=40)

    def test_blank_prompt_keeps_old_fingerprints(self):
        self.assertEqual(clip_proposals.policy_version(**self.base),
                         clip_proposals.policy_version(**self.base, prompt=""))

    def test_prompt_version_splits_the_ledger(self):
        self.assertNotEqual(clip_proposals.policy_version(**self.base),
                            clip_proposals.policy_version(**self.base, prompt="v2"))


class LedgerTest(unittest.TestCase):
    """The DB-backed pieces, against an in-memory SQLite schema."""

    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        ch = Channel(call_sign="KTST", url="https://example.com")
        self.session.add(ch)
        self.session.flush()
        self.video = self.add_candidate(ch, "vid1")
        self.other = self.add_candidate(ch, "vid2", transcript="[00:00] THE FLOOD CAME\n"
                                        "[00:10] NOBODY LEFT\n[00:20] SPORTS NEXT")
        self.channel = ch

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def add_candidate(self, ch, vid, transcript=""):
        c = Candidate(video_id=vid, channel_pk=ch.id, title=vid, url="u",
                      transcript_text=transcript)
        self.session.add(c)
        self.session.flush()
        return c

    def propose(self, candidate, segments, verdict=ClipProposal.VERDICT_PENDING, **kw):
        row = ClipProposal(candidate_pk=candidate.id, proposed_segments=json.dumps(segments),
                           verdict=verdict, **kw)
        self.session.add(row)
        self.session.flush()
        return row

    def test_pending_summary_skips_proposals_already_clipped(self):
        self.propose(self.video, [{"start": 0, "end": 30}], story="used up")
        self.propose(self.video, [{"start": 100, "end": 130}], story="fresh")
        self.session.add(Cut(candidate_pk=self.video.id,
                             trim_segments=json.dumps([{"start": 0, "end": 30}])))
        self.session.flush()
        summary = clip_proposals.pending_summary(self.session, [self.video.id, self.other.id])
        self.assertEqual(summary, {self.video.id: {"count": 1, "story": "fresh"}})

    def test_rejections_split_here_from_elsewhere(self):
        self.propose(self.video, [{"start": 0, "end": 30}], ClipProposal.VERDICT_DISMISSED,
                     story="here, no reason")
        self.propose(self.other, [{"start": 0, "end": 30}], ClipProposal.VERDICT_DISMISSED,
                     story="elsewhere, no reason")
        self.propose(self.other, [{"start": 0, "end": 30}], ClipProposal.VERDICT_DISMISSED,
                     story="elsewhere", dismiss_reason="too_long", proposed_duration_s=38)
        here, elsewhere = clip_context.rejected(self.session, self.video.id)
        self.assertEqual([h["story"] for h in here], ["here, no reason"])
        self.assertEqual(elsewhere, [{"story": "elsewhere", "reason": "too long", "seconds": 38}])

    def test_shipped_examples_rank_by_views_and_skip_this_video(self):
        def ship(candidate, segs, views):
            cut = Cut(candidate_pk=candidate.id, trim_segments=json.dumps(segs))
            self.session.add(cut)
            self.session.flush()
            post = ThreadsPost(candidate_pk=candidate.id, cut_pk=cut.id, status="published",
                               published_at=utcnow() - dt.timedelta(days=1))
            self.session.add(post)
            self.session.flush()
            self.session.add(MetricSnapshot(post_pk=post.id, views=views))
            self.session.flush()

        ship(self.other, [{"start": 0, "end": 15}], 900)     # THE FLOOD CAME / NOBODY LEFT
        ship(self.other, [{"start": 20, "end": 25}], 50)     # SPORTS NEXT
        ship(self.video, [{"start": 0, "end": 10}], 5000)    # same video: excluded (and no transcript)
        out = clip_context.shipped_examples(self.session, exclude_candidate_pk=self.video.id)
        self.assertEqual([e["words_heard"] for e in out],
                         ["THE FLOOD CAME NOBODY LEFT", "SPORTS NEXT"])
        self.assertEqual(out[0]["seconds"], 15.0)
        self.assertIn("top", out[0]["performance"])

    def test_build_survives_a_broken_piece(self):
        settings = mock.Mock()
        settings.get = lambda key, default=None: default
        self.propose(self.video, [{"start": 0, "end": 30}], ClipProposal.VERDICT_DISMISSED,
                     story="nope")
        with mock.patch.object(clip_context, "shipped_examples", side_effect=RuntimeError("boom")):
            ctx = clip_context.build(self.session, self.video, settings)
        self.assertNotIn("shipped_examples", ctx)
        self.assertEqual(ctx["rejected_here"][0]["story"], "nope")
        # The session is still usable for the proposal write that follows.
        self.propose(self.video, [{"start": 40, "end": 60}])


if __name__ == "__main__":
    unittest.main()
