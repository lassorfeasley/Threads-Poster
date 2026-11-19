"""Tests for positioning fit (app/positioning.py, llm.score_fit) and automatic
triage judging by it (app/auto_triage.py with signal "fit").

Run with:  python -m unittest tests.test_positioning -v

In-memory SQLite; the model call is mocked, and so is the Brand & audience
profile (``llm.fit_profile``), so the fingerprint is the test's to change.
"""
from __future__ import annotations

import json
import threading
import unittest
from unittest import mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import auto_triage, llm, positioning, ranking, triage_model
from app.models import (STATUS_NEW, STATUS_REJECTED, AutoTriageAction, Base, Candidate, Channel,
                        TriageDecision, TriageModel, utcnow)

PROFILE = {"name": "Acme Solar", "topic": "rooftop solar", "mission": "Get homes on solar",
           "audience": "US homeowners", "positioning": "Practical savings, no hype",
           "source_kind": "TV news footage", "relevance_rules": "it covers solar",
           "false_positives": "solar eclipse", "rules": ""}


class _Settings:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class _Base(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        triage_model._cache = None
        self.channel = Channel(call_sign="KXYZ", url="https://example.com/c")
        self.session.add(self.channel)
        self.session.flush()
        self.settings = _Settings(**{"matching.score_threshold": 0.5,
                                     "positioning.workers": 2})
        self.profile = dict(PROFILE)
        patcher = mock.patch.object(llm, "fit_profile", side_effect=lambda: dict(self.profile))
        patcher.start()
        self.addCleanup(patcher.stop)
        # Fit by title: "great" 0.9, "meh" 0.3, "fail" raises.
        self.calls: list[str] = []
        self.lock = threading.Lock()

        def fake_fit(model, title, description):
            with self.lock:
                self.calls.append(title)
            if "fail" in title:
                raise RuntimeError("model down")
            return {"fit": 0.9 if "great" in title else 0.3, "reason": f"because {title}"}

        patcher = mock.patch.object(llm, "score_fit", side_effect=fake_fit)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._n = 0

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def video(self, title: str, relevance: float | None = 0.8, status: str = STATUS_NEW,
              fit: float | None = None, basis: str = "") -> Candidate:
        self._n += 1
        c = Candidate(video_id=f"fit{self._n:04d}", channel_pk=self.channel.id, title=title,
                      description="", url="https://example.com/v", status=status,
                      relevance_score=relevance, fit_score=fit, fit_basis=basis)
        self.session.add(c)
        self.session.flush()
        return c

    def key(self) -> str:
        return positioning.basis(self.settings)


class BasisTest(_Base):
    def test_a_profile_edit_changes_the_fingerprint(self):
        before = self.key()
        self.assertEqual(before, self.key())
        self.profile["positioning"] = "Bold climate action"
        self.assertNotEqual(before, self.key())

    def test_the_model_is_part_of_it(self):
        other = _Settings(**{"positioning.model": "claude-sonnet-x"})
        self.assertNotEqual(self.key(), positioning.basis(other))


class ScoreWaitingTest(_Base):
    def test_scores_only_waiting_videos_the_dashboard_shows(self):
        shown = self.video("great rooftop install")
        unscored_relevance = self.video("meh solar story", relevance=None)
        hidden = self.video("great but off topic", relevance=0.2)
        decided = self.video("great old one", status=STATUS_REJECTED)
        out = positioning.score_waiting(self.session, self.settings)
        self.assertEqual((out["scored"], out["failed"], out["left"]), (2, 0, 0))
        for c in (shown, unscored_relevance, hidden, decided):
            self.session.refresh(c)
        self.assertEqual((shown.fit_score, shown.fit_reason, shown.fit_basis),
                         (0.9, "because great rooftop install", self.key()))
        self.assertEqual(unscored_relevance.fit_score, 0.3)
        self.assertIsNone(hidden.fit_score)
        self.assertIsNone(decided.fit_score)

    def test_a_current_score_is_not_redone_but_a_stale_one_is(self):
        fresh = self.video("great a", fit=0.5, basis=self.key())
        stale = self.video("great b", fit=0.5, basis="oldbasis")
        positioning.score_waiting(self.session, self.settings)
        self.assertEqual(self.calls, ["great b"])
        self.session.refresh(fresh)
        self.session.refresh(stale)
        self.assertEqual((fresh.fit_score, stale.fit_score), (0.5, 0.9))

    def test_a_failed_call_leaves_the_video_unscored_for_next_time(self):
        c = self.video("fail please")
        out = positioning.score_waiting(self.session, self.settings)
        self.assertEqual((out["scored"], out["failed"], out["left"]), (0, 1, 1))
        self.session.refresh(c)
        self.assertIsNone(c.fit_score)

    def test_limit_takes_the_newest_first(self):
        for i in range(5):
            self.video(f"meh {i}")
        out = positioning.score_waiting(self.session, self.settings, limit=2)
        self.assertEqual((out["scored"], out["left"]), (2, 3))
        self.assertEqual(sorted(self.calls), ["meh 3", "meh 4"])

    def test_switched_off_it_does_nothing(self):
        self.video("great")
        off = _Settings(**{"positioning.enabled": False})
        self.assertEqual(positioning.score_waiting(self.session, off)["scored"], 0)
        self.assertEqual(self.calls, [])

    def test_score_decided_scores_past_decisions(self):
        c = self.video("great past", status=STATUS_REJECTED)
        self.session.add(TriageDecision(candidate_pk=c.id, action="approve"))
        self.video("great waiting")
        self.session.flush()
        out = positioning.score_decided(self.session, self.settings, limit=10)
        self.assertEqual(out["scored"], 1)
        self.assertEqual(self.calls, ["great past"])


class ScoreFitPromptTest(unittest.TestCase):
    def run_fit(self, reply: dict, profile: dict | None = None) -> tuple[dict, str]:
        seen = {}

        def fake_chat(model, system, user, max_tokens=1000, temperature=0.2):
            seen["system"], seen["user"] = system, user
            return reply

        with mock.patch.object(llm, "fit_profile", return_value=profile or dict(PROFILE)), \
                mock.patch.object(llm, "_json_chat", side_effect=fake_chat):
            out = llm.score_fit("m", "Title here", "Desc")
        return out, seen["system"]

    def test_clamps_and_tidies(self):
        out, system = self.run_fit({"fit": 1.7, "reason": "  squarely   on\nangle "})
        self.assertEqual(out, {"fit": 1.0, "reason": "squarely on angle"})
        self.assertIn("Practical savings, no hype", system)
        self.assertIn("US homeowners", system)

    def test_empty_profile_fields_are_left_out(self):
        profile = dict(PROFILE, mission="", audience="", positioning="")
        _, system = self.run_fit({"fit": 0.5, "reason": "x"}, profile)
        self.assertNotIn("Mission:", system)
        self.assertNotIn("Positioning", system.split("Score fit")[0])

    def test_a_non_number_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            self.run_fit({"fit": "high", "reason": "x"})
        with self.assertRaises(ValueError):
            self.run_fit({"fit": float("nan"), "reason": "x"})


class FitTriageTest(_Base):
    def switch_on(self, cutoff: float = 0.4):
        return auto_triage.set_policy(self.session, enabled=True, signal="fit",
                                      fit_below=cutoff, audit_rate=0.1)

    def do_pass(self) -> dict:
        return auto_triage.run_pass(self.session, self.settings, trigger="test")

    def test_needs_no_model_and_rejects_under_the_fit_cutoff(self):
        self.switch_on(0.4)
        keep = self.video("great solar savings")
        drop = self.video("meh solar gossip")
        hidden = self.video("meh off topic", relevance=0.1)
        out = self.do_pass()
        self.assertEqual((out["outcome"], out["signal"], out["rejected"], out["considered"]),
                         ("done", "fit", 1, 2))
        for c in (keep, drop, hidden):
            self.session.refresh(c)
        self.assertEqual((keep.status, drop.status, hidden.status),
                         (STATUS_NEW, STATUS_REJECTED, STATUS_NEW))
        (call,) = self.session.execute(select(AutoTriageAction)).scalars().all()
        self.assertEqual((call.signal, call.fit_score, call.fit_reason, call.reject_below),
                         ("fit", 0.3, "because meh solar gossip", 0.4))
        self.assertEqual(self.session.query(TriageModel).count(), 0)

    def test_a_stale_fit_is_rescored_before_it_is_judged(self):
        self.switch_on(0.4)
        c = self.video("great after all", fit=0.1, basis="oldbasis")
        out = self.do_pass()
        self.assertEqual(out["rejected"], 0)
        self.session.refresh(c)
        self.assertEqual((c.status, c.fit_score), (STATUS_NEW, 0.9))

    def test_switching_signal_restarts_the_spot_check_window(self):
        import datetime as dt

        t0 = utcnow()
        auto_triage.set_policy(self.session, enabled=True, now=t0)
        later = t0 + dt.timedelta(days=1)
        policy = auto_triage.set_policy(self.session, signal="fit", now=later)
        self.assertEqual(policy.watch_since_at, later)
        self.assertEqual(policy.cutoff, auto_triage.DEFAULT_FIT_CUTOFF)
        with self.assertRaises(ValueError):
            auto_triage.set_policy(self.session, fit_below=0.45)
        with self.assertRaises(ValueError):
            auto_triage.set_policy(self.session, signal="vibes")

    def test_fit_waits_only_when_switched_off(self):
        self.switch_on()
        self.settings.values["positioning.enabled"] = False
        out = self.do_pass()
        self.assertEqual(out["outcome"], "waiting")
        self.assertIn("switched off", out["note"])

    def test_fit_grade_reads_decisions_with_a_current_fit(self):
        key = self.key()
        for title, fit, action in (("a", 0.9, "approve"), ("b", 0.3, "reject"),
                                   ("c", 0.25, "approve"), ("d", 0.8, "reject")):
            c = self.video(title, status=STATUS_REJECTED, fit=fit, basis=key)
            self.session.add(TriageDecision(candidate_pk=c.id, action=action))
        stale = self.video("e", status=STATUS_REJECTED, fit=0.1, basis="old")
        self.session.add(TriageDecision(candidate_pk=stale.id, action="approve"))
        self.session.flush()
        grade = auto_triage.fit_grade(self.session, self.settings)
        self.assertEqual((grade["n"], grade["approved"]), (4, 2))
        under_04 = next(h for h in grade["handoff"] if h["threshold"] == 0.4)
        self.assertEqual((under_04["below"], under_04["approvals_lost"]), (2, 1))
        rows = {r["cutoff"]: r for r in
                auto_triage.cutoff_table(self.session, self.settings, "fit")}
        self.assertEqual(rows[0.4]["approvals_lost"], 1)
        self.assertEqual(rows[0.4]["label"], "0.4")


class HiddenVideosTest(_Base):
    """The fix: automatic triage judges only what the dashboard shows."""

    def test_approval_signal_skips_videos_under_the_relevance_threshold(self):
        model = TriageModel(feature_version=triage_model.FEATURE_VERSION, n_decisions=1000,
                            n_approved=100, backtest=json.dumps({"auc": 0.8}),
                            params=json.dumps({"idf": {}, "weights": {}}))
        self.session.add(model)
        self.session.flush()
        shown = self.video("x")
        hidden = self.video("y", relevance=0.2)
        for c in (shown, hidden):
            c.approve_prob, c.approve_model_pk = 0.001, model.id
        self.session.flush()
        auto_triage.set_policy(self.session, enabled=True, reject_below=0.02)
        preview = auto_triage.preview(self.session, self.settings)
        self.assertEqual(preview["waiting"], 1)
        out = auto_triage.run_pass(self.session, self.settings, trigger="test")
        self.assertEqual(out["rejected"], 1)
        self.session.refresh(hidden)
        self.assertEqual(hidden.status, STATUS_NEW)


class RankingTest(unittest.TestCase):
    def test_fit_leads_when_there_is_no_approve_chance(self):
        settings = _Settings()

        def cand(prob, fit, rel):
            return Candidate(approve_prob=prob, fit_score=fit, relevance_score=rel,
                             visual_traits="", created_at=utcnow())

        a, b, c = cand(None, 0.2, 0.9), cand(None, 0.8, 0.5), cand(0.3, None, 0.1)
        ordered = ranking.sort_candidates([a, b, c], {}, settings)
        self.assertEqual(ordered, [c, b, a])


if __name__ == "__main__":
    unittest.main()


class ImageMediaTypeTest(unittest.TestCase):
    """Storyboard sheets arrive as WebP; declaring them JPEG fails the request."""

    def test_sniffs_the_real_format(self):
        self.assertEqual(llm.image_media_type(b"RIFF\x10\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertEqual(llm.image_media_type(b"\x89PNG\r\n\x1a\n\x00\x00"), "image/png")
        self.assertEqual(llm.image_media_type(b"GIF89a\x01\x00"), "image/gif")
        self.assertEqual(llm.image_media_type(b"\xff\xd8\xff\xe0\x00\x10JFIF"), "image/jpeg")

    def test_tag_footage_sends_what_it_got(self):
        sent = {}

        def fake_create(model, system, content, max_tokens, temperature):
            sent["blocks"] = content
            return mock.Mock(content=[mock.Mock(type="text", text='{"traits": [], "why": "x"}')])

        with mock.patch.object(llm, "_create", side_effect=fake_create):
            llm.tag_footage("m", [b"RIFF\x10\x00\x00\x00WEBPVP8 "], ["fire"])
        self.assertEqual(sent["blocks"][1]["source"]["media_type"], "image/webp")
