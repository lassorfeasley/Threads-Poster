"""Tests for rerun slots and fresh rerun captions (app/scheduler.py).

Run with:  python -m unittest tests.test_rerun_slots -v

Database-backed tests run against a throwaway in-memory SQLite database;
``session_scope`` is patched to hand out that session, and the LLM drafter is
always mocked.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import scheduler
from app.models import Base, Candidate, Channel, Cut, RerunReview, ThreadsPost, utcnow

DAY = dt.date(2026, 10, 1)
RERUN_KEY, NEW_KEY = f"{DAY}#0", f"{DAY}#1"


def _rerun_slot(key: str) -> bool:
    return key == RERUN_KEY


class ReservationTest(unittest.TestCase):
    """Organic posts stay out of rerun slots; pins and the fallback don't."""

    def assign(self, posts, open_keys=frozenset()):
        with mock.patch.object(scheduler, "is_rerun_slot", _rerun_slot), \
                mock.patch.object(scheduler, "build_placement_context", return_value=None):
            out, _ = scheduler._assign_with_mode(None, posts, [RERUN_KEY, NEW_KEY],
                                                 open_keys=open_keys)
        return out

    def test_new_clips_skip_the_rerun_slot(self):
        new = ThreadsPost(id=1, pinned_window_key="")
        self.assertEqual(self.assign([new]), [None, new])

    def test_a_pinned_rerun_takes_its_slot(self):
        new = ThreadsPost(id=1, pinned_window_key="")
        rerun = ThreadsPost(id=2, pinned_window_key=RERUN_KEY)
        self.assertEqual(self.assign([new, rerun]), [rerun, new])

    def test_fallback_opens_the_slot_to_new_clips(self):
        new = ThreadsPost(id=1, pinned_window_key="")
        self.assertEqual(self.assign([new], open_keys=frozenset({RERUN_KEY})), [new, None])


class DbTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        self.channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.session.add(self.channel)
        self.session.flush()
        self._n = 0

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            yield self.session
            self.session.flush()

        patches = [mock.patch.object(scheduler, "session_scope", scope),
                   mock.patch.object(scheduler, "is_rerun_slot", _rerun_slot)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def aired(self, *, shelf: str = "evergreen", days_ago: int = 120) -> ThreadsPost:
        self._n += 1
        cand = Candidate(video_id=f"v{self._n}", channel_pk=self.channel.id,
                         title=f"Video {self._n}", url="https://example.com/v",
                         category="news", shelf_life=shelf)
        self.session.add(cand)
        self.session.flush()
        cut = Cut(candidate_pk=cand.id)
        self.session.add(cut)
        self.session.flush()
        p = ThreadsPost(candidate_pk=cand.id, cut_pk=cut.id, status="published",
                        published_at=utcnow() - dt.timedelta(days=days_ago),
                        caption="original caption", clip_object_path="clips/x.mp4")
        self.session.add(p)
        self.session.flush()
        return p


class PendingTest(DbTest):
    def test_rerun_slot_reruns_dont_block_other_re_airs(self):
        prior = self.aired()
        self.session.add(ThreadsPost(status="queued", repost_of_post_pk=prior.id,
                                     pinned_window_key=RERUN_KEY))
        self.session.flush()
        self.assertFalse(scheduler._repost_pending(self.session))
        self.session.add(ThreadsPost(status="queued", repost_of_post_pk=prior.id,
                                     pinned_window_key=NEW_KEY))
        self.session.flush()
        self.assertTrue(scheduler._repost_pending(self.session))


class CaptionTest(DbTest):
    def caption(self, prior, *, fresh: bool, drafted: str):
        settings = {"fresh_captions": fresh, "model": "m", "stage_ahead_hours": 24}
        with mock.patch.object(scheduler, "_rerun_settings", return_value=settings), \
                mock.patch.object(scheduler, "_draft_rerun_caption",
                                  return_value=drafted) as draft:
            return scheduler._rerun_caption(self.session, prior.cut, prior), draft

    def test_fresh_caption_when_drafting_works(self):
        caption, _ = self.caption(self.aired(), fresh=True, drafted="a new line")
        self.assertEqual(caption, "a new line")

    def test_evergreen_falls_back_to_the_original(self):
        caption, _ = self.caption(self.aired(), fresh=True, drafted="")
        self.assertEqual(caption, "original caption")

    def test_fresh_off_reuses_the_original_without_drafting(self):
        caption, draft = self.caption(self.aired(), fresh=False, drafted="x")
        self.assertEqual(caption, "original caption")
        draft.assert_not_called()

    def test_dated_caption_never_airs_again(self):
        # Rerun Review cleared this timely clip only with a new caption.
        prior = self.aired(shelf="timely")
        caption, _ = self.caption(prior, fresh=False, drafted="")
        self.assertIsNone(caption)
        caption, _ = self.caption(prior, fresh=False, drafted="a new line")
        self.assertEqual(caption, "a new line")

    def test_new_caption_decision_joins_the_rerun_library(self):
        prior = self.aired(shelf="timely")
        cfg = {"min_quiet_days": 45.0, "airing_growth": 0.5}
        self.assertEqual(scheduler._filler_rotation(self.session, cfg)[0], [])
        self.session.add(RerunReview(post_pk=prior.id, decision="new_caption"))
        self.session.flush()
        facts, _ = scheduler._filler_rotation(self.session, cfg)
        self.assertEqual([f["prior"].id for f in facts], [prior.id])


class StagingTest(DbTest):
    def stage(self):
        win = utcnow() + dt.timedelta(hours=3)
        with mock.patch.object(scheduler, "_filler_config",
                               return_value={"min_quiet_days": 45.0, "airing_growth": 0.5}), \
                mock.patch.object(scheduler, "_upcoming_window_slots",
                                  return_value=[(RERUN_KEY, win, 0), (NEW_KEY, win, 1)]), \
                mock.patch.object(scheduler, "rerun_indices_for_date",
                                  return_value=frozenset({0})), \
                mock.patch.object(scheduler, "_rerun_caption", return_value="a new line"):
            return scheduler.ensure_rerun_slots_staged()

    def test_stages_once_with_the_fresh_caption(self):
        prior = self.aired()
        result = self.stage()
        self.assertTrue(result.startswith(f"rerun_staged:{RERUN_KEY}"))
        post = self.session.query(ThreadsPost).filter_by(status="queued").one()
        self.assertEqual((post.pinned_window_key, post.repost_of_post_pk), (RERUN_KEY, prior.id))
        self.assertEqual((post.caption, post.suggested_caption), ("a new line", "a new line"))

        # A deleted rerun leaves the slot to the at-window fallback.
        self.session.delete(post)
        self.session.flush()
        self.assertIsNone(self.stage())

    def test_empty_library_marks_the_slot_handled(self):
        self.aired(days_ago=5)  # too recent to rerun
        self.assertEqual(self.stage(), f"rerun_slot_unfilled:{RERUN_KEY}")
        self.assertIsNone(self.stage())


if __name__ == "__main__":
    unittest.main()
