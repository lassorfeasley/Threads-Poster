"""Tests for brand rules: storage, scoping, where they reach the prompts, and
the Brand & audience form never blanking a field it didn't send.

Run with:  python -m unittest tests.test_brand_rules -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import config, llm


class TempConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "brand.yaml").write_text(
            "topic: climate change\nrelevance_rules: it covers climate\n")
        p = mock.patch.object(config, "CONFIG_DIR", self.dir)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)


class StorageTest(TempConfig):
    def test_add_dedupes_and_scopes(self):
        self.assertIsNotNone(config.add_brand_rule("Write as we", "captions"))
        self.assertIsNone(config.add_brand_rule("write AS we", "captions"))   # duplicate
        self.assertIsNone(config.add_brand_rule("   ", "captions"))          # empty
        config.add_brand_rule("Never present denial as credible", "bogus-scope")
        rules = config.load_brand_rules()
        self.assertEqual([r["applies_to"] for r in rules], ["captions", "everything"])

    def test_render_scopes(self):
        config.save_brand_rules([
            {"text": "Stance rule", "applies_to": "everything"},
            {"text": "Caption rule", "applies_to": "captions"},
            {"text": "Paused rule", "applies_to": "everything", "enabled": False},
        ])
        self.assertEqual(config.render_brand_rules("content"), "- Stance rule")
        self.assertEqual(config.render_brand_rules("captions"),
                         "- Stance rule\n- Caption rule")

    def test_saving_fields_keeps_rules_and_vice_versa(self):
        config.add_brand_rule("Keep me", "everything")
        config.save_brand({"topic": "clean energy"})
        self.assertEqual([r["text"] for r in config.load_brand_rules()], ["Keep me"])
        config.save_brand_rules([])
        self.assertEqual(config.load_brand()["topic"], "clean energy")
        self.assertEqual(config.load_brand()["relevance_rules"], "it covers climate")


class PromptTest(TempConfig):
    def capture(self, fn, **kw):
        seen = {}

        def fake(model, system, user, max_tokens=0, temperature=0.2):
            seen["system"] = system
            return {"score": 0.5, "caption": "x", "hook": "x", "ok": True, "clips": []}

        with mock.patch.object(llm, "_json_chat", side_effect=fake):
            fn(**kw)
        return seen["system"]

    def test_no_rules_no_block(self):
        system = self.capture(llm.score_relevance, model="m", title="t",
                              description="", matched_keywords=[])
        self.assertNotIn("OPERATOR'S RULES", system)

    def test_content_rules_reach_scoring_but_caption_rules_dont(self):
        config.save_brand_rules([{"text": "Never present denial as credible",
                                  "applies_to": "everything"},
                                 {"text": "Write as we", "applies_to": "captions"}])
        system = self.capture(llm.score_relevance, model="m", title="t",
                              description="", matched_keywords=[])
        self.assertIn("Never present denial as credible", system)
        self.assertNotIn("Write as we", system)
        self.assertIn("Score relevance, not agreement", system)

    def test_caption_prompts_get_every_rule(self):
        config.save_brand_rules([{"text": "Never present denial as credible"},
                                 {"text": "Write as we", "applies_to": "captions"}])
        system = self.capture(llm.suggest_rerun_caption, model="m", title="t",
                              channel="c", original_caption="o", clip_transcript="",
                              first_aired="2026-01-01", today="2026-11-09")
        self.assertIn("Write as we", system)
        self.assertIn("Never present denial as credible", system)
        system = self.capture(llm.check_caption, model="m", caption="c",
                              clip_transcript="", title="t")
        self.assertIn("breaks one of the operator's rules", system)

    def test_clip_suggestions_flag_rule_challenges(self):
        config.save_brand_rules([{"text": "Never present denial as credible"}])
        reply = {"clips": [{"story": "Host says warming is a hoax",
                            "segments": [{"start": 0, "end": 20}],
                            "challenges_rules": "Presents climate denial"}]}
        seen = {}

        def fake(model, system, user, max_tokens=0, temperature=0.2):
            seen["system"] = system
            return reply

        transcript = [{"start": float(i), "end": i + 1.0, "text": "w"} for i in range(30)]
        with mock.patch.object(llm, "_json_chat", side_effect=fake):
            clips = llm.suggest_clips("m", "T", transcript)
        self.assertIn("challenges_rules", seen["system"])
        self.assertEqual(clips[0]["challenges_rules"], "Presents climate denial")

    def test_broken_brand_file_never_breaks_a_prompt(self):
        (self.dir / "brand.yaml").write_text("rules: [unclosed")
        self.assertEqual(llm._operator_rules("content"), "")


def _throwaway_db() -> bool:
    from app import db
    return bool(db._is_sqlite)


# Importing the web app starts its scheduler thread, so these only run against
# a throwaway SQLite database — never the shared one .env points at.
@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL to a throwaway sqlite:// database")
class BrandFormTest(TempConfig):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from app.web import main
        self.client = TestClient(main.app)

    def test_missing_fields_are_left_alone(self):
        r = self.client.post("/brand", data={
            "topic": "climate change",
            "rules_json": json.dumps([{"text": "Write as we", "applies_to": "captions"}]),
        }, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(config.load_brand()["relevance_rules"], "it covers climate")
        self.assertEqual(config.load_brand_rules()[0]["text"], "Write as we")

    def test_add_rule_endpoint(self):
        r = self.client.post("/brand/rules/add", data={"text": "No emojis"})
        self.assertEqual(r.json()["rule"]["applies_to"], "captions")
        r = self.client.post("/brand/rules/add", data={"text": "No emojis"})
        self.assertEqual(r.status_code, 409)


if __name__ == "__main__":
    unittest.main()
