"""Tests for clip_context's replay modes: ``as_of`` (the context as it stood
at a past moment), ``replay`` (nothing tied to the replayed video), and the
per-suggester calibration kinds.

Run with:  DATABASE_URL="sqlite://" python -m pytest -q tests/test_clip_context_replay.py
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import clip_context
from app.models import (Base, Candidate, Channel, ClipProposal, Cut, MetricSnapshot,
                        ThreadsPost)

T0 = dt.datetime(2026, 10, 1, 12, 0, tzinfo=dt.timezone.utc)


def days(n: float) -> dt.timedelta:
    return dt.timedelta(days=n)


class _Settings:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class _ContextDB(unittest.TestCase):
    """An in-memory schema with a channel and a few videos."""

    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.channel = Channel(call_sign="KTST", url="https://example.com")
        self.session.add(self.channel)
        self.session.flush()
        self.video = self.candidate("vid-replayed", "[00:00] THE REPLAYED VIDEO")
        self.other = self.candidate("vid-other", "[00:00] FLOOD WATERS RISE\n[00:10] TOWN EVACUATES")
        self.third = self.candidate("vid-third", "[00:00] SOLAR FARM OPENS")

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def candidate(self, vid, transcript=""):
        c = Candidate(video_id=vid, channel_pk=self.channel.id, title=vid, url="u",
                      transcript_text=transcript)
        self.session.add(c)
        self.session.flush()
        return c

    def cut(self, candidate, segs, *, title="", updated_at=None):
        cut = Cut(candidate_pk=candidate.id, trim_segments=json.dumps(segs), clip_title=title,
                  updated_at=updated_at or (T0 - days(30)))
        self.session.add(cut)
        self.session.flush()
        return cut

    def post(self, cut, published_at, snapshots=(), status="published"):
        post = ThreadsPost(candidate_pk=cut.candidate_pk, cut_pk=cut.id, status=status,
                           published_at=published_at)
        self.session.add(post)
        self.session.flush()
        for captured_at, views in snapshots:
            self.session.add(MetricSnapshot(post_pk=post.id, captured_at=captured_at, views=views))
        self.session.flush()
        return post

    def proposal(self, candidate, **kw):
        kw.setdefault("proposed_segments", json.dumps([{"start": 0, "end": 20}]))
        row = ClipProposal(candidate_pk=candidate.id, **kw)
        self.session.add(row)
        self.session.flush()
        return row


class MomentTest(unittest.TestCase):
    def test_accepted_forms(self):
        cases = [
            (None, None),
            ("", None),
            ("2026-10-01T12:00:00Z", T0),
            ("2026-10-01T12:00:00+00:00", T0),
            ("2026-10-01T14:00:00+02:00", T0),
            (dt.datetime(2026, 10, 1, 12, 0), T0),       # naive = UTC (SQLite)
            (T0, T0),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(clip_context._moment(value), expected)

    def test_rejects_other_types(self):
        with self.assertRaises(TypeError):
            clip_context._moment(12345)


class ShippedExamplesAsOfTest(_ContextDB):
    def setUp(self):
        super().setUp()
        flood = self.cut(self.other, [{"start": 0, "end": 15}])
        solar = self.cut(self.third, [{"start": 0, "end": 5}])
        late = self.cut(self.other, [{"start": 10, "end": 15}])
        # Modest when the replayed pass ran; went viral afterwards.
        self.post(flood, T0 - days(2), [(T0 - days(1), 100), (T0 + days(1), 10_000)])
        self.post(solar, T0 - days(3), [(T0 - days(2), 500)])
        # Published after the replayed moment: invisible to it.
        self.post(late, T0 + days(1), [(T0 + days(2), 50_000)])

    def test_ranks_by_views_known_at_the_time(self):
        out = clip_context.shipped_examples(self.session, as_of=T0, exclude_candidate_pk=self.video.id)
        self.assertEqual([e["words_heard"] for e in out],
                         ["SOLAR FARM OPENS", "FLOOD WATERS RISE TOWN EVACUATES"])

    def test_without_as_of_today_wins(self):
        out = clip_context.shipped_examples(self.session, exclude_candidate_pk=self.video.id)
        # The late post leads now (50k), then the viral one; ranking by the
        # latest snapshot is unchanged for live callers.
        self.assertEqual(out[0]["words_heard"], "TOWN EVACUATES")
        self.assertEqual(len(out), 3)

    def test_lookback_is_measured_from_as_of(self):
        old = self.cut(self.third, [{"start": 0, "end": 5}])
        self.post(old, T0 - days(200), [(T0 - days(199), 9_999)])
        cases = [(180, 2), (365, 3)]
        for lookback, expected in cases:
            with self.subTest(lookback=lookback):
                out = clip_context.shipped_examples(self.session, as_of=T0, lookback_days=lookback,
                                                    exclude_candidate_pk=self.video.id)
                self.assertEqual(len(out), expected)

    def test_post_unmeasured_by_then_sorts_last(self):
        fresh = self.cut(self.third, [{"start": 0, "end": 5}])
        self.post(fresh, T0 - days(0.5), [(T0 + days(0.1), 80_000)])
        out = clip_context.shipped_examples(self.session, as_of=T0, exclude_candidate_pk=self.video.id)
        self.assertEqual(len(out), 3)
        self.assertNotIn("performance", out[-1])


class RejectedAsOfTest(_ContextDB):
    def setUp(self):
        super().setUp()
        D = ClipProposal.VERDICT_DISMISSED
        self.proposal(self.video, verdict=D, story="here before", decided_at=T0 - days(1))
        self.proposal(self.video, verdict=D, story="here after", decided_at=T0 + days(1))
        self.proposal(self.video, verdict=D, story="here with reason", dismiss_reason="too_long",
                      decided_at=T0 - days(2))
        self.proposal(self.other, verdict=D, story="else before", dismiss_reason="off_topic",
                      decided_at=T0 - days(1), proposed_duration_s=30)
        self.proposal(self.other, verdict=D, story="else after", dismiss_reason="too_long",
                      decided_at=T0 + days(1))

    def test_cases(self):
        cases = [
            ("live", {}, ["here after", "here with reason", "here before"],
             ["else after", "else before"]),
            ("as_of", {"as_of": T0}, ["here with reason", "here before"], ["else before"]),
            ("replay", {"as_of": T0, "include_here": False}, [], ["else before"]),
        ]
        for label, kw, here_expected, else_expected in cases:
            with self.subTest(label):
                here, elsewhere = clip_context.rejected(self.session, self.video.id, **kw)
                self.assertEqual(sorted(h["story"] for h in here), sorted(here_expected))
                self.assertEqual(sorted(e["story"] for e in elsewhere), sorted(else_expected))


class CalibrationKindsTest(_ContextDB):
    def resolved(self, candidate, n, *, model, start, end, decided=None, cut=None):
        for _ in range(n):
            self.proposal(candidate, model=model, final_segments='[{"start": 0, "end": 10}]',
                          start_delta_s=start, end_delta_s=end, proposed_duration_s=30,
                          final_duration_s=30, proposed_segment_count=2, final_segment_count=2,
                          verdict=ClipProposal.VERDICT_ACCEPTED,
                          decided_at=decided or (T0 - days(5)),
                          cut_pk=(cut or self.cut(candidate, [{"start": 0, "end": 10}])).id)

    def test_each_kind_reads_its_own_rows(self):
        self.resolved(self.other, 5, model="claude-sonnet-5", start=3.0, end=0.0)
        self.resolved(self.other, 5, model="engine:gemini:gemini-3.8-flash", start=-4.0, end=0.0)
        cases = [
            ("legacy", "started about 3s too early"),
            ("engine", "started about 4s too late"),
        ]
        for kind, phrase in cases:
            with self.subTest(kind):
                notes = clip_context.calibration_for(self.session, kind=kind)
                self.assertTrue(any(phrase in n for n in notes), notes)
        self.assertEqual(clip_context.calibration_for(self.session, kind="none"), [])

    def test_legacy_wrapper_matches_legacy_kind(self):
        self.resolved(self.other, 5, model="claude-sonnet-5", start=3.0, end=0.0)
        self.resolved(self.other, 5, model="engine:gemini:x", start=-4.0, end=0.0)
        self.assertEqual(clip_context.calibration(self.session),
                         clip_context.calibration_for(self.session, kind="legacy"))

    def test_engine_waits_for_enough_resolved_rows(self):
        self.resolved(self.other, 4, model="engine:gemini:x", start=-4.0, end=0.0)
        for _ in range(4):   # even the dismissal summary stays out
            self.proposal(self.other, model="engine:gemini:x", dismiss_reason="too_long",
                          verdict=ClipProposal.VERDICT_DISMISSED,
                          decided_at=dt.datetime.now(dt.timezone.utc) - days(1))
        self.assertEqual(clip_context.calibration_for(self.session, kind="engine"), [])

    def test_dismiss_reasons_are_per_suggester(self):
        now = dt.datetime.now(dt.timezone.utc)
        for _ in range(3):
            self.proposal(self.other, model="claude-sonnet-5", dismiss_reason="too_long",
                          verdict=ClipProposal.VERDICT_DISMISSED, decided_at=now - days(1))
        legacy = clip_context.calibration_for(self.session, kind="legacy")
        self.assertEqual(legacy, ["Most common reasons proposals were dismissed: too long (3)."])
        self.resolved(self.other, 5, model="engine:gemini:x", start=0.0, end=0.0)
        self.assertEqual(clip_context.calibration_for(self.session, kind="engine"), [])

    def test_rows_without_a_model_count_as_legacy(self):
        self.resolved(self.other, 5, model="", start=3.0, end=0.0)
        notes = clip_context.calibration_for(self.session, kind="legacy")
        self.assertTrue(any("too early" in n for n in notes))

    def test_as_of_needs_the_resolving_export_before_it(self):
        early_cut = self.cut(self.other, [{"start": 0, "end": 10}], updated_at=T0 - days(3))
        late_cut = self.cut(self.other, [{"start": 0, "end": 10}], updated_at=T0 + days(3))
        self.resolved(self.other, 5, model="m", start=3.0, end=0.0, cut=early_cut)
        # Decided before as_of, but the cut was exported (resolved) after it.
        self.resolved(self.other, 5, model="m", start=-9.0, end=0.0, cut=late_cut)
        # Decided after as_of.
        self.resolved(self.other, 5, model="m", start=-9.0, end=0.0, cut=early_cut,
                      decided=T0 + days(1))
        notes = clip_context.calibration_for(self.session, kind="legacy", as_of=T0)
        self.assertTrue(any("3s too early" in n for n in notes), notes)

    def test_unknown_kind_is_an_error(self):
        with self.assertRaises(ValueError):
            clip_context.calibration_for(self.session, kind="sometimes")


class RecentlyPostedTest(_ContextDB):
    def setUp(self):
        super().setUp()
        self.post(self.cut(self.other, [{"start": 0, "end": 5}], title="Flood"), T0 - days(2))
        self.post(self.cut(self.video, [{"start": 0, "end": 5}], title="Own clip"), T0 - days(1))
        self.post(self.cut(self.third, [{"start": 0, "end": 5}], title="Too old"), T0 - days(20))
        self.post(self.cut(self.third, [{"start": 0, "end": 5}], title="Later"), T0 + days(1))

    def test_cases(self):
        cases = [
            ("as_of", {"as_of": T0}, ["Own clip", "Flood"]),
            ("replay", {"as_of": T0, "exclude_candidate_pk": None}, ["Own clip", "Flood"]),
            ("replay excludes own", {"as_of": T0, "exclude_candidate_pk": "video"}, ["Flood"]),
        ]
        for label, kw, expected in cases:
            with self.subTest(label):
                if kw.get("exclude_candidate_pk") == "video":
                    kw = {**kw, "exclude_candidate_pk": self.video.id}
                self.assertEqual(clip_context.recently_posted(self.session, **kw), expected)


class BuildModesTest(_ContextDB):
    def setUp(self):
        super().setUp()
        D = ClipProposal.VERDICT_DISMISSED
        self.proposal(self.video, verdict=D, story="own dismissal", decided_at=T0 - days(1))
        self.post(self.cut(self.video, [{"start": 0, "end": 5}], title="Own clip"), T0 - days(1))
        self.post(self.cut(self.other, [{"start": 0, "end": 15}], title="Flood"), T0 - days(2),
                  [(T0 - days(1), 300)])
        for model, start in (("claude-sonnet-5", 3.0), ("engine:gemini:x", -4.0)):
            for _ in range(5):
                self.proposal(self.other, model=model, final_segments='[{"start":0,"end":9}]',
                              start_delta_s=start, end_delta_s=0.0,
                              verdict=ClipProposal.VERDICT_ACCEPTED, decided_at=T0 - days(4),
                              cut_pk=self.cut(self.other, [{"start": 0, "end": 9}]).id)
        self.settings = _Settings()

    def test_live_default_keeps_everything(self):
        # "Lately" is measured from now for live callers.
        now = dt.datetime.now(dt.timezone.utc)
        self.post(self.cut(self.video, [{"start": 5, "end": 9}], title="Own recent clip"),
                  now - days(1))
        ctx = clip_context.build(self.session, self.video, self.settings)
        self.assertEqual([h["story"] for h in ctx["rejected_here"]], ["own dismissal"])
        self.assertTrue(any("too early" in n for n in ctx["calibration"]))
        self.assertEqual(ctx["recently_posted"], ["Own recent clip"])
        self.assertIn("shipped_examples", ctx)

    def test_engine_calibration_kind(self):
        ctx = clip_context.build(self.session, self.video, self.settings, calibration="engine")
        self.assertTrue(any("too late" in n for n in ctx["calibration"]))
        ctx = clip_context.build(self.session, self.video, self.settings, calibration="none")
        self.assertNotIn("calibration", ctx)

    def test_replay_drops_everything_tied_to_the_video_and_calibration(self):
        for kind in ("legacy", "engine"):
            with self.subTest(kind):
                ctx = clip_context.build(self.session, self.video, self.settings,
                                         as_of="2026-10-01T12:00:00Z", replay=True,
                                         calibration=kind)
                self.assertNotIn("rejected_here", ctx)
                self.assertNotIn("calibration", ctx)
                self.assertEqual(ctx["recently_posted"], ["Flood"])
                self.assertEqual(len(ctx["shipped_examples"]), 1)

    def test_naive_and_string_as_of_agree(self):
        a = clip_context.build(self.session, self.video, self.settings,
                               as_of=dt.datetime(2026, 10, 1, 12, 0), replay=True)
        b = clip_context.build(self.session, self.video, self.settings,
                               as_of="2026-10-01T12:00:00Z", replay=True)
        self.assertEqual(a, b)

    def test_switched_off(self):
        settings = _Settings(**{"clips.learn_from_history": False})
        self.assertEqual(clip_context.build(self.session, self.video, settings, replay=True), {})

    def test_bad_calibration_kind(self):
        with self.assertRaises(ValueError):
            clip_context.build(self.session, self.video, self.settings, calibration="both")

    def test_replay_failures_are_loud(self):
        with mock.patch.object(clip_context, "shipped_examples", side_effect=RuntimeError("boom")):
            with self.assertLogs("clip_context", "WARNING") as logs:
                ctx = clip_context.build(self.session, self.video, self.settings,
                                         as_of=T0, replay=True)
        self.assertNotIn("shipped_examples", ctx)
        self.assertTrue(any("examples" in line for line in logs.output))
        # The session still works for whatever the caller does next.
        self.proposal(self.video, story="still writable")


if __name__ == "__main__":
    unittest.main()
