"""Tests for the operator's notes on clip suggestions (app/clip_engine/notes.py):
the append-only store, advice, the rule-proposal queue (from notes and from
the handbook), the house rule formatter and the advice tidier.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
      .venv/bin/python -m pytest -q tests/test_clip_engine_notes.py

Claude is always mocked (``claude.json_call`` / ``claude.is_configured``); no
database is touched.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from app import config, spend
from app.clip_engine import claude, notes
from app.config import Settings


class NotesTestBase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.engine_dir = self.dir / "engine"
        self.patch(mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.engine_dir)}))
        self.patch(mock.patch.object(spend, "_LEDGER", self.dir / "llm_spend.json"))
        conf = self.dir / "config"
        conf.mkdir()
        (conf / "brand.yaml").write_text("topic: climate change\n")
        self.patch(mock.patch.object(config, "CONFIG_DIR", conf))
        self.settings = Settings(raw={"clip_engine": {"claude": {"editor_model": "claude-test"}}})
        self.addCleanup(notes._reset_for_tests)

    def patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def claude_on(self, *answers):
        """Claude configured; json_call answers in turn (an exception raises)."""
        self.patch(mock.patch.object(claude, "is_configured", return_value=True))
        replies = list(answers)

        def call(model, **kwargs):
            reply = replies.pop(0) if len(replies) > 1 else replies[0]
            if isinstance(reply, BaseException):
                raise reply
            return (reply(model, **kwargs) if callable(reply) else reply), {"cost_usd": 0.01}

        return self.patch(mock.patch.object(claude, "json_call", side_effect=call))

    def write_lines(self, *lines: str) -> None:
        path = notes.notes_path()
        with path.open("a") as fh:
            for line in lines:
                fh.write(line + "\n")

    def old_note(self, days: float, **fields) -> dict:
        at = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).replace(microsecond=0)
        note = {"id": fields.pop("id", f"old{days}"), "at": at.isoformat().replace("+00:00", "Z"),
                "candidate_id": 1, "video_id": "v1", "proposal_id": None, "kind": "card",
                "tags": ["boring"], "text": ""}
        note.update(fields)
        self.write_lines(json.dumps(note))
        return note


# ---- the store ---------------------------------------------------------------------------

class StoreTest(NotesTestBase):
    def test_add_and_read_back(self):
        note = notes.add(7, video_id="abc", proposal_id=42, kind="card",
                         tags="wrong_language, bogus,monologue,wrong_language",
                         text="  the  interviews are in Hindi  ")
        self.assertEqual(set(note), {"id", "at", "candidate_id", "video_id", "proposal_id",
                                     "kind", "tags", "text"})
        self.assertEqual(note["tags"], ["wrong_language", "monologue"])   # known, unique, in order
        self.assertEqual(note["text"], "the interviews are in Hindi")
        self.assertEqual((note["candidate_id"], note["video_id"], note["proposal_id"]),
                         (7, "abc", 42))
        steer = notes.add(7, video_id="abc", kind="steer", text="open on the floods")
        self.assertIsNone(steer["proposal_id"])
        self.assertEqual(steer["tags"], [])
        self.assertEqual([n["id"] for n in notes.recent()], [note["id"], steer["id"]])
        self.assertEqual([n["id"] for n in notes.for_candidate(7)], [note["id"], steer["id"]])
        self.assertEqual(notes.for_candidate(8), [])
        self.assertEqual(notes.for_candidate("x"), [])
        # One JSON object per line, appended.
        lines = notes.notes_path().read_text().splitlines()
        self.assertEqual([json.loads(line)["id"] for line in lines], [note["id"], steer["id"]])

    def test_a_note_that_says_nothing_is_not_kept(self):
        self.assertIsNone(notes.add(1, tags=["bogus"], text="   "))
        self.assertIsNone(notes.add(1))
        self.assertFalse(notes.notes_path().exists())

    def test_unknown_kind_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            notes.add(1, kind="rant", text="x")

    def test_text_is_capped(self):
        note = notes.add(1, text="word " * 400)
        self.assertLessEqual(len(note["text"]), notes.MAX_TEXT)

    def test_reads_skip_torn_and_corrupt_lines_and_never_raise(self):
        good = notes.add(1, tags=["boring"])
        self.write_lines("{not json", "", "[1, 2]", json.dumps({"id": "", "at": ""}),
                         '{"id": "half", "at": "2026-')
        later = notes.add(2, tags=["too_long"])
        self.assertEqual([n["id"] for n in notes.recent()], [good["id"], later["id"]])
        self.assertEqual([n["id"] for n in notes.for_candidate(2)], [later["id"]])

    def test_missing_or_unreadable_file_reads_empty(self):
        self.assertEqual(notes.recent(), [])
        notes.notes_path().mkdir()           # a directory where the file should be
        self.assertEqual(notes.recent(), [])
        self.assertEqual(notes.for_candidate(1), [])
        # And a write that fails is logged, not raised.
        with self.assertLogs("clip_engine.notes", "WARNING"):
            self.assertIsNone(notes.add(1, tags=["boring"]))

    def test_recent_window(self):
        self.old_note(120, id="ancient")
        self.old_note(30, id="month")
        fresh = notes.add(1, tags=["boring"])
        self.assertEqual([n["id"] for n in notes.recent()], ["month", fresh["id"]])
        self.assertEqual([n["id"] for n in notes.recent(days=7)], [fresh["id"]])
        self.assertEqual(len(notes.for_candidate(1)), 3)   # every note on the video

    def test_concurrent_appends_never_interleave(self):
        def writer(n):
            for i in range(25):
                notes.add(n, tags=["boring"], text=f"writer {n} note {i} " + "x" * 200)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = notes.notes_path().read_text().splitlines()
        self.assertEqual(len(lines), 200)
        self.assertEqual(len({json.loads(line)["id"] for line in lines}), 200)


class VocabularyTest(NotesTestBase):
    def test_tag_labels(self):
        labels = notes.tag_labels()
        self.assertEqual([slug for slug, _ in labels],
                         ["wrong_language", "no_premise", "monologue", "weak_opening", "boring",
                          "off_topic", "bad_boundaries", "too_long", "other"])
        self.assertTrue(all(label for _, label in labels))

    def test_dismiss_reason_is_the_first_tag_that_maps(self):
        cases = [
            (["wrong_language"], "not_clipworthy"), (["monologue"], "not_clipworthy"),
            (["no_premise"], "not_clipworthy"), (["boring"], "not_clipworthy"),
            (["weak_opening"], "weak_visuals"), (["off_topic"], "off_topic"),
            (["bad_boundaries"], "bad_boundaries"), (["too_long"], "too_long"),
            (["other", "too_long", "off_topic"], "too_long"), (["other"], ""), ([], ""),
            (["bogus", "off_topic"], "off_topic"),
        ]
        for tags, reason in cases:
            with self.subTest(tags=tags):
                self.assertEqual(notes.dismiss_reason_for(tags), reason)
        # Every mapped reason is one the ledger accepts.
        from app.clip_proposals import DISMISS_REASONS

        self.assertTrue(set(notes.DISMISS_REASON.values()) <= set(DISMISS_REASONS))

    def test_advice_from(self):
        advice = notes.advice_from(["wrong_language", "monologue"], "open on the floods")
        self.assertIn("English voiceover", advice)
        self.assertIn("subtitles are not enough", advice)
        self.assertIn("reporter or anchor", advice)
        self.assertTrue(advice.endswith("open on the floods."))
        self.assertEqual(notes.advice_from("other", "Too dark!"), "Too dark!")
        self.assertEqual(notes.advice_from([], ""), "")
        self.assertEqual(notes.advice_from(["other"], ""), "")
        self.assertIn("premise", notes.advice_from("no_premise").lower())
        for slug, _label in notes.tag_labels():
            if slug != "other":
                with self.subTest(slug=slug):
                    self.assertTrue(notes.advice_from([slug]))


# ---- rule proposals from notes ------------------------------------------------------------

def theme_reply(evidence, text="Never open on a reporter talking to camera."):
    return {"proposals": [{"text": text, "evidence": evidence, "count": len(evidence)}]}


class ProposeRulesTest(NotesTestBase):
    def setUp(self) -> None:
        super().setUp()
        # Formatting is tested on its own; here it is the plain tidy.
        self.patch(mock.patch.object(notes, "format_rule", side_effect=lambda text, **kw:
                                     notes._tidy_rule(text, when=kw.get("when", ""),
                                                      why=kw.get("why", ""))))

    def add_notes(self, videos=("v1", "v2", "v1")):
        return [notes.add(i, video_id=v, tags=["monologue"], text=f"anchor again {i}")
                for i, v in enumerate(videos)]

    def test_waits_for_three_notes_since_the_last_run(self):
        call = self.claude_on(theme_reply([]))
        self.add_notes(("v1", "v2"))
        self.assertEqual(notes.propose_rules(settings=self.settings), [])
        call.assert_not_called()
        notes.add(9, video_id="v3", tags=["boring"])
        self.assertEqual(notes.propose_rules(settings=self.settings), [])   # nothing qualified
        call.assert_called_once()
        state = json.loads(notes.proposals_path().read_text())
        self.assertTrue(state["last_run_at"])
        # The run counted those three: it takes three more to ask again.
        notes.add(10, video_id="v3", tags=["boring"])
        notes.propose_rules(settings=self.settings)
        self.assertEqual(call.call_count, 1)
        notes.propose_rules(settings=self.settings, force=True)
        self.assertEqual(call.call_count, 2)

    def test_the_call(self):
        added = self.add_notes()
        config.add_brand_rule("Write as we", "captions")
        call = self.claude_on(theme_reply([n["id"] for n in added]))
        notes.propose_rules(settings=self.settings)
        model = call.call_args.args[0]
        kwargs = call.call_args.kwargs
        self.assertEqual(model, "claude-test")              # model_for(settings, claude, editor)
        self.assertEqual(kwargs["effort"], "medium")
        self.assertEqual(kwargs["bucket"], "engine")
        self.assertNotIn("temperature", kwargs)
        sent = json.loads(kwargs["content"])
        self.assertEqual(sent["existing_brand_rules"], ["Write as we"])
        self.assertEqual([n["id"] for n in sent["notes"]], [n["id"] for n in added])
        self.assertEqual(sent["notes"][0]["tags"], ["monologue"])
        self.assertIn("at least 3 notes from at least 2 different videos", kwargs["system"])
        self.assertEqual(kwargs["schema"], notes.PROPOSE_SCHEMA)
        # A model override wins.
        notes.propose_rules(settings=self.settings, model="claude-other", force=True)
        self.assertEqual(call.call_args.args[0], "claude-other")

    def test_a_theme_needs_three_real_notes_on_two_videos(self):
        same_video = self.add_notes(("v1", "v1", "v1"))
        spread = self.add_notes(("v1", "v2", "v3"))
        reply = {"proposals": [
            # Made-up evidence doesn't count.
            {"text": "Rule from thin air.", "evidence": ["nope1", "nope2", spread[0]["id"]],
             "count": 3},
            # Three notes, one video: a complaint about that video, not a rule.
            {"text": "Rule from one video.", "evidence": [n["id"] for n in same_video],
             "count": 3},
            {"text": "Never open on a reporter talking to camera.",
             "evidence": [n["id"] for n in spread] + [spread[0]["id"]], "count": 9},
            "garbage",
        ]}
        self.claude_on(reply)
        added = notes.propose_rules(settings=self.settings)
        self.assertEqual([p["rule"] for p in added], ["Never open on a reporter talking to camera."])
        p = added[0]
        self.assertEqual(p["evidence"], [n["id"] for n in spread])   # de-duplicated, real
        self.assertEqual((p["count"], p["videos"]), (3, 3))
        self.assertEqual((p["source"], p["status"]), ("notes", "pending"))
        self.assertEqual(p["summary"], "From 3 of your notes on 3 videos")
        self.assertEqual(len(p["examples"]), 3)
        self.assertEqual(notes.pending_proposals()[0]["idx"], 0)

    def test_existing_rules_and_earlier_proposals_are_not_proposed_again(self):
        added = self.add_notes(("v1", "v2", "v3"))
        ids = [n["id"] for n in added]
        config.add_brand_rule("Never open on a reporter talking to camera", "clips")
        self.claude_on(theme_reply(ids))
        self.assertEqual(notes.propose_rules(settings=self.settings, force=True), [])
        self.claude_on(theme_reply(ids, "Cut on complete sentences."))
        first = notes.propose_rules(settings=self.settings, force=True)
        self.assertEqual(len(first), 1)
        self.assertEqual(notes.propose_rules(settings=self.settings, force=True), [])  # pending
        notes.set_status(first[0]["idx"], "dismissed")
        self.assertEqual(notes.propose_rules(settings=self.settings, force=True), [])  # dismissed

    def test_long_rule_text_is_cut_at_a_word(self):
        added = self.add_notes(("v1", "v2", "v3"))
        self.claude_on(theme_reply([n["id"] for n in added], "Open on the footage " * 30))
        text = notes.propose_rules(settings=self.settings)[0]["original"]
        self.assertLessEqual(len(text), notes.MAX_RULE_CHARS)
        self.assertTrue(text.endswith("…"))

    def test_no_claude_no_call(self):
        self.add_notes(("v1", "v2", "v3"))
        with mock.patch.object(claude, "json_call") as call:
            self.assertEqual(notes.propose_rules(settings=self.settings), [])
        call.assert_not_called()          # ANTHROPIC_API_KEY is empty in the test env

    def test_a_failed_call_waits_an_hour(self):
        self.add_notes(("v1", "v2", "v3"))
        call = self.claude_on(claude.ClaudeError("overloaded"))
        self.assertEqual(notes.propose_rules(settings=self.settings), [])
        state = json.loads(notes.proposals_path().read_text())
        self.assertIn("overloaded", state["last_error"])
        self.assertNotIn("last_run_at", state)
        notes.add(5, video_id="v4", tags=["boring"])
        notes.propose_rules(settings=self.settings)
        self.assertEqual(call.call_count, 1)                  # still waiting
        stale = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat()
        notes.util.atomic_write_json(notes.proposals_path(), dict(state, last_error_at=stale))
        notes.propose_rules(settings=self.settings)
        self.assertEqual(call.call_count, 2)

    def test_one_run_at_a_time(self):
        self.add_notes(("v1", "v2", "v3"))
        call = self.claude_on(theme_reply([]))
        with notes._propose_lock:
            self.assertEqual(notes.propose_rules(settings=self.settings), [])
        call.assert_not_called()

    def test_never_raises(self):
        self.add_notes(("v1", "v2", "v3"))
        self.claude_on(theme_reply([]))
        with mock.patch.object(notes, "_propose", side_effect=RuntimeError("boom")), \
                self.assertLogs("clip_engine.notes", "ERROR"):
            self.assertEqual(notes.propose_rules(), [])

    def test_runs_in_a_daemon_thread_after_a_note(self):
        started = threading.Event()
        release = threading.Event()

        def slow(**kwargs):
            started.set()
            release.wait(5)
            return []

        with mock.patch.object(notes, "propose_rules", side_effect=slow):
            self.assertTrue(notes.propose_rules_soon())
            self.assertTrue(started.wait(5))
            self.assertTrue(notes._propose_thread.daemon)
            self.assertFalse(notes.propose_rules_soon())    # one already under way
            release.set()
            notes._propose_thread.join(5)
            self.assertTrue(notes.propose_rules_soon())
            notes._propose_thread.join(5)


# ---- the one proposal queue -------------------------------------------------------------

class QueueTest(NotesTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.formatted: list[str] = []

        def fake_format(text, **kw):
            self.formatted.append(text)
            f = notes._tidy_rule(text, when=kw.get("when", ""), why=kw.get("why", ""))
            return dict(f, formatted=True)

        self.patch(mock.patch.object(notes, "format_rule", side_effect=fake_format))

    def test_handbook_items_queue_beside_notes(self):
        added = notes.add_proposals([
            {"text": "open on the footage, not the anchor", "evidence": [101, 205],
             "count": 12, "when": "news packages with b-roll"},
            {"text": "Cut on complete sentences", "evidence": "support 9, exceptions 1",
             "count": 9, "when": "always", "summary": "In 9 of your edits"},
        ], source="handbook")
        self.assertEqual(len(added), 2)
        first, second = added
        self.assertEqual(first["source"], "handbook")
        self.assertEqual(first["evidence"], ["101", "205"])
        self.assertEqual(first["text"],
                         "When news packages with b-roll: Open on the footage, not the anchor.")
        self.assertEqual(first["when"], "News packages with b-roll")
        self.assertEqual(first["original"], "open on the footage, not the anchor")
        self.assertEqual(first["summary"], "Seen in 12 of your past edits")
        self.assertIn("**When:** News packages with b-roll", first["markdown"])
        self.assertEqual(second["evidence"], ["support 9, exceptions 1"])
        self.assertEqual((second["text"], second["when"]), ("Cut on complete sentences.", "Always"))
        self.assertEqual(second["summary"], "In 9 of your edits")
        self.assertEqual([p["idx"] for p in notes.pending_proposals()], [0, 1])
        self.assertEqual(self.formatted, ["open on the footage, not the anchor",
                                          "Cut on complete sentences"])

    def test_de_duplicated_against_rules_pending_dismissed_and_itself(self):
        config.add_brand_rule("Never use subtitles as a translation", "clips")
        notes.add_proposals([{"text": "Already pending"}, {"text": "Turned down"}],
                            source="notes")
        notes.set_status(1, "dismissed")
        added = notes.add_proposals([
            {"text": "never use SUBTITLES as a translation."},
            {"text": "already  pending"},
            {"text": "Turned down"},
            {"text": "Fresh rule"},
            {"text": "fresh rule."},
            {"text": "   "},
            "garbage",
        ], source="handbook")
        self.assertEqual([p["original"] for p in added], ["Fresh rule"])
        # Only the survivors of the first check were sent to the formatter.
        self.assertEqual(self.formatted[-2:], ["Fresh rule", "fresh rule."])
        # An added proposal no longer blocks (its rule does).
        notes.set_status(0, "added")
        self.assertEqual(len(notes.add_proposals([{"text": "Already pending"}],
                                                 source="notes")), 1)

    def test_bad_source(self):
        with self.assertRaises(ValueError):
            notes.add_proposals([{"text": "x"}], source="martians")

    def test_status_changes(self):
        notes.add_proposals([{"text": "One"}, {"text": "Two"}], source="notes")
        self.assertEqual(notes.set_status(1, "added")["status"], "added")
        self.assertIsNone(notes.set_status(1, "dismissed"))    # no longer pending
        self.assertIsNone(notes.set_status(7, "added"))
        with self.assertRaises(ValueError):
            notes.set_status(0, "maybe")
        self.assertEqual([p["idx"] for p in notes.pending_proposals()], [0])
        self.assertEqual([p["status"] for p in notes.proposals()], ["pending", "added"])
        self.assertTrue(notes.proposals("added")[0]["decided_at"])

    def test_a_corrupt_queue_reads_empty(self):
        notes.proposals_path().write_text("{oops")
        self.assertEqual(notes.proposals(), [])
        notes.proposals_path().write_text(json.dumps({"proposals": ["x", {"text": "ok",
                                                                         "status": "pending"}]}))
        self.assertEqual([(p["text"], p["source"], p["idx"]) for p in notes.proposals()],
                         [("ok", "notes", 0)])

    def test_evidence_line(self):
        self.assertEqual(notes.evidence_line({"summary": "From 4 of your notes", "when": "Always"}),
                         "From 4 of your notes")
        self.assertEqual(notes.evidence_line({"source": "handbook", "count": 7,
                                              "when": "Flood videos"}),
                         "Seen in 7 of your past edits (when: Flood videos)")


# ---- the formatter and the tidier -------------------------------------------------------------

class FormatRuleTest(NotesTestBase):
    def test_formatted_by_claude(self):
        call = self.claude_on({"rule": "Never use an interview in another language unless an "
                                       "English voiceover carries it",
                               "when": "always", "why": ""})
        f = notes.format_rule("never use foriegn interviews without english voice over",
                              settings=self.settings)
        self.assertEqual(f, {
            "rule": "Never use an interview in another language unless an English voiceover "
                    "carries it.",
            "when": "Always", "why": "",
            "markdown": "**Rule:** Never use an interview in another language unless an English "
                        "voiceover carries it.\n**When:** Always",
            "original": "never use foriegn interviews without english voice over",
            "formatted": True})
        kwargs = call.call_args.kwargs
        self.assertEqual(call.call_args.args[0], "claude-test")
        self.assertEqual((kwargs["effort"], kwargs["bucket"]), ("low", "engine"))
        self.assertEqual(kwargs["schema"], notes.FORMAT_SCHEMA)
        self.assertIn("Fix spelling and grammar", kwargs["system"])
        self.assertIn("**Rule:** Never use an interview in another language", kwargs["system"])
        self.assertEqual(json.loads(kwargs["content"])["rule_as_given"],
                         "never use foriegn interviews without english voice over")

    def test_condition_and_reason(self):
        self.claude_on({"rule": "Open on the disaster footage, not the anchor.",
                        "when": "When the video is a news package with b-roll.",
                        "why": "you cut the anchor intro in most edits"})
        f = notes.format_rule("for packages open on footage", why="12 edits",
                              settings=self.settings)
        self.assertEqual(f["when"], "The video is a news package with b-roll")
        self.assertEqual(f["why"], "You cut the anchor intro in most edits.")
        self.assertEqual(f["markdown"].splitlines(), [
            "**Rule:** Open on the disaster footage, not the anchor.",
            "**When:** The video is a news package with b-roll",
            "**Why:** You cut the anchor intro in most edits."])
        self.assertEqual(notes.rule_text(f), "When the video is a news package with b-roll: "
                                             "Open on the disaster footage, not the anchor.")
        self.assertEqual(notes.brand_rule_fields(f), {
            "rule": f["rule"], "when": f["when"], "why": f["why"], "markdown": f["markdown"],
            "original": "for packages open on footage"})

    def test_rule_is_capped_at_160(self):
        self.claude_on({"rule": "Open on the footage " * 20, "when": "", "why": ""})
        f = notes.format_rule("long", settings=self.settings)
        self.assertLessEqual(len(f["rule"]), notes.MAX_FORMATTED_RULE)

    def test_fallbacks(self):
        expected = {"rule": "No emojis or hashtags.", "when": "Always", "why": "",
                    "markdown": "**Rule:** No emojis or hashtags.\n**When:** Always",
                    "original": "no emojis or hashtags", "formatted": False}
        # No key: no call.
        with mock.patch.object(claude, "json_call") as call:
            self.assertEqual(notes.format_rule("no emojis   or hashtags"), expected)
        call.assert_not_called()
        # A failed call, or an empty answer.
        for answer in (claude.ClaudeError("refused"), {"rule": "", "when": "", "why": ""}):
            with self.subTest(answer=answer):
                self.claude_on(answer)
                self.assertEqual(notes.format_rule("no emojis   or hashtags",
                                                   settings=self.settings), expected)
        # Nothing to format.
        self.assertEqual(notes.format_rule("  ")["rule"], "")

    def test_plain_tidy(self):
        cases = [("open on action", "Open on action."), ("Why not?", "Why not?"),
                 ("  two   spaces. ", "Two spaces."), ("", "")]
        for raw, tidy in cases:
            with self.subTest(raw=raw):
                self.assertEqual(notes.tidy_sentence(raw), tidy)
        f = notes._tidy_rule("open on action", when="when it floods.", why="seen twice")
        self.assertEqual((f["when"], f["why"]), ("It floods", "Seen twice."))
        self.assertEqual(notes.rule_text(f), "When it floods: Open on action.")
        self.assertEqual(notes.rule_text({"rule": "", "when": "x"}), "")


class TidyAdviceTest(NotesTestBase):
    def test_tidied_by_claude(self):
        call = self.claude_on({"text": "Open on the disaster footage; no interviews."})
        self.assertEqual(notes.tidy_advice("open on teh disaster footage no intervews",
                                           settings=self.settings),
                         "Open on the disaster footage; no interviews.")
        kwargs = call.call_args.kwargs
        self.assertEqual((kwargs["effort"], kwargs["bucket"]), ("low", "engine"))
        self.assertEqual(kwargs["content"], "open on teh disaster footage no intervews")
        self.assertIn("spelling, grammar and punctuation only", kwargs["system"])

    def test_fallbacks(self):
        with mock.patch.object(claude, "json_call") as call:
            self.assertEqual(notes.tidy_advice("open on the floods"), "Open on the floods.")
        call.assert_not_called()
        self.claude_on(RuntimeError("down"))
        self.assertEqual(notes.tidy_advice("open on the floods", settings=self.settings),
                         "Open on the floods.")
        self.claude_on({"text": ""})
        self.assertEqual(notes.tidy_advice("open on the floods", settings=self.settings),
                         "Open on the floods.")
        self.assertEqual(notes.tidy_advice("   "), "")

    def test_engine_advice_carries_both(self):
        self.assertEqual(notes.engine_advice("open on teh floods", "Open on the floods."),
                         "Open on the floods.\n(Operator's words: open on teh floods)")
        self.assertEqual(notes.engine_advice(""), "")
        # Without a tidied text it tidies (here: the plain tidy, no key).
        self.assertEqual(notes.engine_advice("no interviews"),
                         "No interviews.\n(Operator's words: no interviews)")


if __name__ == "__main__":
    unittest.main()
