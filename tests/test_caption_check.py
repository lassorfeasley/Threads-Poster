"""Tests for the re-air caption gut check (app/caption_check.py and
``scheduler._rerun_caption``).

Run with:  python -m unittest tests.test_caption_check -v

Runs against a throwaway in-memory SQLite database; the drafter and the
checker are stubbed, so no model is called.
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import caption_check, scheduler
from app.caption_check import Verdict
from app.models import Base, Candidate, Channel, Cut, ThreadsPost, utcnow

VETTED = "The caption the operator vetted."


class RerunCaptionTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.session.add(channel)
        self.session.flush()
        cand = Candidate(video_id="v1", channel_pk=channel.id, title="Video",
                         url="https://example.com/v", category="news",
                         shelf_life="evergreen")
        self.session.add(cand)
        self.session.flush()
        self.cut = Cut(candidate_pk=cand.id)
        self.session.add(self.cut)
        self.session.flush()
        self.first = self.airing(VETTED, days_ago=90)
        self.second = self.airing("A machine-written re-air caption.", days_ago=30,
                                  repost_of=self.first)
        self.settings = mock.patch.object(scheduler, "_rerun_settings", return_value={
            "fresh_captions": True, "model": "m", "stage_ahead_hours": 24})
        self.settings.start()

    def tearDown(self):
        self.settings.stop()
        self.session.close()
        self.engine.dispose()

    def airing(self, caption: str, *, days_ago: int, repost_of=None) -> ThreadsPost:
        p = ThreadsPost(candidate_pk=self.cut.candidate_pk, cut_pk=self.cut.id,
                        status="published", caption=caption, clip_object_path="x.mp4",
                        published_at=utcnow() - dt.timedelta(days=days_ago),
                        repost_of_post_pk=repost_of.id if repost_of else None)
        self.session.add(p)
        self.session.flush()
        return p

    def rerun(self, drafts: list[str], verdicts: list[Verdict]):
        drafter = mock.patch.object(scheduler, "_draft_rerun_caption", side_effect=drafts)
        checker = mock.patch.object(caption_check, "run", side_effect=verdicts)
        with drafter as draft, checker as check:
            caption = scheduler._rerun_caption(self.session, self.cut, self.second)
        return caption, draft, check

    def test_first_airing_follows_the_rerun_chain(self):
        third = self.airing("Another re-air.", days_ago=1, repost_of=self.second)
        self.assertEqual(caption_check.first_airing(self.session, third).id, self.first.id)
        self.assertEqual(caption_check.vetted_airing(self.session, self.cut).id, self.first.id)

    def test_a_passing_draft_ships(self):
        caption, draft, check = self.rerun(["Fresh line."], [Verdict(ok=True)])
        self.assertEqual(caption, "Fresh line.")
        # Drafted from, and checked against, the first airing — not the re-air.
        self.assertEqual(draft.call_args.args[2].id, self.first.id)
        self.assertEqual(check.call_args.kwargs["reference"].id, self.first.id)

    def test_a_flagged_draft_is_redrafted(self):
        caption, _, _ = self.rerun(["Wrong story.", "Fresh line."],
                                   [Verdict(ok=False, reason="x"), Verdict(ok=True)])
        self.assertEqual(caption, "Fresh line.")

    def test_falls_back_to_the_vetted_caption(self):
        caption, _, _ = self.rerun(["Wrong story.", "Also wrong."],
                                   [Verdict(ok=False), Verdict(ok=False)])
        self.assertEqual(caption, VETTED)

    def test_an_unchecked_draft_never_ships(self):
        caption, draft, _ = self.rerun(["Fresh line."], [Verdict(ok=None, error="down")])
        self.assertEqual(caption, VETTED)
        self.assertEqual(draft.call_count, 1)

    def test_a_dated_clip_without_a_passing_draft_is_skipped(self):
        self.first.shelf_life = "timely"
        self.second.shelf_life = "timely"
        caption, _, _ = self.rerun(["Wrong.", "Wrong."],
                                   [Verdict(ok=False), Verdict(ok=False)])
        self.assertIsNone(caption)


if __name__ == "__main__":
    unittest.main()
