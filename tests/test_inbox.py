"""Tests for the inbox's selection and decisions (app/inbox.py).

Run with:  python -m unittest tests.test_inbox -v

In-memory SQLite. Sidecars (the engine's doubts and critic lines) are mocked;
``queue_clip`` is mocked too — it uploads, and its own tests cover it.
"""
from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import inbox
from app.models import (STATUS_ARCHIVED, STATUS_NEW, AutoTriageAction, AutoTriageRun, Base,
                        Candidate, Channel, ClipProposal, Cut, InboxItem, ThreadsPost)

NY = ZoneInfo("America/New_York")
NOW = dt.datetime(2026, 11, 19, 18, 30, tzinfo=dt.timezone.utc)   # 13:30 in New York


class _Settings:
    def get(self, key, default=None):
        return default


class _Base(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        self.channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.session.add(self.channel)
        self.session.flush()
        self.sidecar: dict[int, dict] = {}
        patcher = mock.patch("app.clip_engine.sidecars.load",
                             side_effect=lambda pid: self.sidecar.get(pid))
        patcher.start()
        self.addCleanup(patcher.stop)
        self._n = 0

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def video(self, picked: bool = True, status: str = STATUS_ARCHIVED, prob: float = 0.3):
        self._n += 1
        c = Candidate(video_id=f"ib{self._n:04d}", channel_pk=self.channel.id, title=f"v{self._n}",
                      url="https://example.com/v", status=status, approve_prob=prob)
        self.session.add(c)
        self.session.flush()
        if picked:
            run = AutoTriageRun(stage="select", trigger="t")
            self.session.add(run)
            self.session.flush()
            self.session.add(AutoTriageAction(run_pk=run.id, candidate_pk=c.id, stage="select",
                                              action="approve"))
            self.session.flush()
        return c

    def draft(self, c, score=0.8, flag="", verdict="pending", age_days=0.5):
        row = ClipProposal(candidate_pk=c.id, proposed_segments='[{"start": 1, "end": 20}]',
                           confidence=score, flag=flag, verdict=verdict, story="a story",
                           created_at=NOW - dt.timedelta(days=age_days))
        self.session.add(row)
        self.session.flush()
        return row

    def on(self, **kw):
        return inbox.set_policy(self.session, enabled=True, **kw)


class EligibleTest(_Base):
    def test_takes_reviewed_unflagged_fresh_drafts_from_picked_videos_best_first(self):
        policy = self.on(min_score=0.7)
        a = self.video()
        b = self.video(prob=0.9)
        best = self.draft(a, 0.9)
        tie_better_video = self.draft(b, 0.8)
        tie = self.draft(a, 0.8)
        self.draft(a, 0.6)                          # under the bar
        self.draft(b, 0.9, flag="claims X")         # challenges the rules
        self.draft(b, 0.9, verdict="dismissed")     # already decided
        self.draft(b, 0.95, age_days=5)             # stale
        self.draft(self.video(picked=False), 0.95)  # a video automatic approval didn't pick
        self.draft(self.video(status=STATUS_NEW), 0.95)   # not downloaded
        doubtful = self.draft(b, 0.85)
        self.sidecar[doubtful.id] = {"topic_doubt": "maybe not climate"}
        rows = inbox.eligible(self.session, _Settings(), policy, NOW)
        # Two per video: a gets best + tie; b gets tie_better_video only.
        self.assertEqual([r.id for r in rows], [best.id, tie_better_video.id, tie.id])

    def test_source_all_takes_any_downloaded_video(self):
        policy = self.on(source="all")
        row = self.draft(self.video(picked=False), 0.9)
        self.assertEqual([r.id for r in inbox.eligible(self.session, _Settings(), policy, NOW)],
                         [row.id])

    def test_the_per_video_cap_counts_what_is_already_in_the_inbox(self):
        policy = self.on(per_video=1)
        c = self.video()
        done = self.draft(c, 0.9)
        self.session.add(InboxItem(candidate_pk=c.id, proposal_pk=done.id,
                                   state=InboxItem.STATE_REJECTED))
        self.draft(c, 0.8)
        self.session.flush()
        self.assertEqual(inbox.eligible(self.session, _Settings(), policy, NOW), [])


class ClaimTest(_Base):
    def test_paces_the_day_and_never_claims_twice(self):
        self.on(per_day=5)                 # 14/24 of 5 -> 3 by 13:30
        for _ in range(3):
            c = self.video()
            self.draft(c, 0.9)
            self.draft(c, 0.8)
        ids = inbox.claim(self.session, _Settings(), now=NOW, tz=NY)
        self.assertEqual(len(ids), 3)
        items = [self.session.get(InboxItem, i) for i in ids]
        self.assertTrue(all(i.state == InboxItem.STATE_DRAFTING for i in items))
        self.assertEqual(items[0].score, 0.9)
        self.assertEqual(inbox.claim(self.session, _Settings(), now=NOW, tz=NY), [])

    def test_off_claims_nothing(self):
        self.draft(self.video(), 0.9)
        self.assertEqual(inbox.claim(self.session, _Settings(), now=NOW, tz=NY), [])

    def test_stalled_drafts_retry_then_fail(self):
        c = self.video()
        old = NOW - dt.timedelta(hours=2)
        retry = InboxItem(candidate_pk=c.id, proposal_pk=self.draft(c).id, attempts=1,
                          started_at=old)
        give_up = InboxItem(candidate_pk=c.id, proposal_pk=self.draft(c).id, attempts=2,
                            started_at=old)
        fresh = InboxItem(candidate_pk=c.id, proposal_pk=self.draft(c).id, attempts=1,
                          started_at=NOW)
        self.session.add_all([retry, give_up, fresh])
        self.session.flush()
        self.assertEqual(inbox.stalled(self.session, now=NOW), [retry.id])
        self.assertEqual(give_up.state, InboxItem.STATE_FAILED)
        self.assertEqual(fresh.state, InboxItem.STATE_DRAFTING)


class DecisionTest(_Base):
    def ready(self, subs: bool = True):
        c = self.video()
        row = self.draft(c, 0.9)
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: [p.unlink() for p in tmp.iterdir()] and tmp.rmdir())
        plain, burned = tmp / "clip.mp4", tmp / "clip_subs.mp4"
        plain.write_bytes(b"x")
        burned.write_bytes(b"x")
        cut = Cut(candidate_pk=c.id, trim_segments='[{"start": 1, "end": 20}]',
                  trimmed_clip_path=str(plain), subtitled_clip_path=str(burned))
        self.session.add(cut)
        self.session.flush()
        row.verdict, row.cut_pk = "accepted", cut.id
        item = InboxItem(candidate_pk=c.id, proposal_pk=row.id, cut_pk=cut.id,
                         state=InboxItem.STATE_READY, caption="AI caption", use_subtitles=subs)
        self.session.add(item)
        self.session.flush()
        return item, cut, row

    def test_approve_then_queue_makes_an_ordinary_queued_post(self):
        item, cut, _ = self.ready()
        made = {}

        def fake_queue(session, candidate, path, caption, cut=None):
            post = ThreadsPost(candidate_pk=candidate.id, cut_pk=cut.id, caption=caption,
                               status="queued", clip_local_path=path)
            session.add(post)
            session.flush()
            made["path"] = path
            return post

        inbox.approve(self.session, item, "  My edit  ")
        with mock.patch("app.publishing.queue_clip", side_effect=fake_queue):
            self.assertTrue(inbox.queue_approved(self.session, item))
        self.assertEqual((item.state, item.final_caption), (InboxItem.STATE_QUEUED, "My edit"))
        self.assertEqual(cut.draft_caption, "My edit")
        self.assertEqual(made["path"], cut.subtitled_clip_path)
        self.assertTrue(cut.use_subtitles)
        self.assertEqual(self.session.get(ThreadsPost, item.post_pk).status, "queued")

    def test_without_the_file_here_approval_waits_for_the_home_machine(self):
        item, cut, _ = self.ready(subs=False)
        Path(cut.trimmed_clip_path).unlink()
        inbox.approve(self.session, item, "ok")
        with mock.patch("app.publishing.queue_clip") as queue:
            self.assertFalse(inbox.queue_approved(self.session, item))
        queue.assert_not_called()
        self.assertEqual(item.state, InboxItem.STATE_APPROVED)

    def test_approve_needs_a_caption_and_a_waiting_clip(self):
        item, _, _ = self.ready()
        with self.assertRaises(ValueError):
            inbox.approve(self.session, item, "   ")
        inbox.approve(self.session, item, "ok")
        with self.assertRaises(ValueError):
            inbox.approve(self.session, item, "again")

    def test_reject_dismisses_the_draft_with_the_chips_reason(self):
        item, _, row = self.ready()
        inbox.reject(self.session, item, ["boring", "not-a-tag"], "  dull   middle ")
        self.session.refresh(row)
        self.assertEqual((item.state, item.reject_tags, item.reject_note),
                         (InboxItem.STATE_REJECTED, "boring", "dull middle"))
        self.assertEqual((row.verdict, row.dismiss_reason, row.cut_pk),
                         ("dismissed", "not_clipworthy", None))


if __name__ == "__main__":
    unittest.main()
