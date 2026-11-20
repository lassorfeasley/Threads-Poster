"""Throughline's universal rules (app/universal_rules.py). Run with:
python -m unittest tests.test_universal_rules -v"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import config, llm, universal_rules as ur


class ShapeTest(unittest.TestCase):
    def test_defaults(self):
        d = ur.defaults()
        self.assertEqual(ur.clip_length(d), (20, 45))
        self.assertEqual({r["id"] for r in d["rules"]},
                         {"striking-opening", "celebrities-first", "coherent-story"})

    def test_clean_rejects_bad_lengths_and_scopes(self):
        with self.assertRaises(ValueError):
            ur.clean({"clip_length": {"min_seconds": 50, "max_seconds": 40}})
        d = ur.clean({"clip_length": {"min_seconds": 20, "max_seconds": 45},
                      "rules": [{"text": " x  y ", "applies_to": "nonsense"}, {"text": ""}]})
        self.assertEqual(d["rules"], [{"id": "x-y", "text": "x y", "enabled": True, "applies_to": "everything"}])

    def test_scopes_reach_the_right_prompts(self):
        d = ur.clean({"clip_length": {"min_seconds": 20, "max_seconds": 45}, "rules": [
            {"text": "V", "applies_to": "videos"}, {"text": "C", "applies_to": "clips"},
            {"text": "K", "applies_to": "captions"}, {"text": "E", "applies_to": "everything"},
            {"text": "off", "applies_to": "everything", "enabled": False}]})
        self.assertEqual(ur.render("screening", d), "- V\n- E")
        self.assertEqual(ur.render("content", d), "- V\n- C\n- E")
        self.assertEqual(ur.render("captions", d), "- K\n- E")


class LoadTest(unittest.TestCase):
    def setUp(self):
        ur.forget()
        self.addCleanup(ur.forget)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.object(ur, "_last_good_path", return_value=Path(self.tmp.name) / "u.json")
        p.start()
        self.addCleanup(p.stop)

    def test_no_url_means_defaults(self):
        with mock.patch.dict(os.environ, {"UNIVERSAL_RULES_URL": "", "UNIVERSAL_RULES_TOKEN": ""}):
            self.assertEqual(ur.load(), ur.defaults())

    def test_live_copy_is_kept_and_used_when_the_site_is_down(self):
        live = {"clip_length": {"min_seconds": 25, "max_seconds": 50}, "rules": []}
        env = {"UNIVERSAL_RULES_URL": "https://admin.example/api/universal-rules", "UNIVERSAL_RULES_TOKEN": "t"}
        resp = mock.Mock(status_code=200, json=lambda: live, raise_for_status=lambda: None)
        with mock.patch.dict(os.environ, env), mock.patch("httpx.get", return_value=resp) as get:
            self.assertEqual(ur.clip_length(ur.load()), (25, 50))
            self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer t"})
        ur.forget()
        with mock.patch.dict(os.environ, env), mock.patch("httpx.get", side_effect=OSError("down")):
            self.assertEqual(ur.clip_length(ur.load()), (25, 50))   # last good copy

    def test_clip_length_wins_over_the_accounts_settings(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        (Path(d.name) / "settings.yaml").write_text("clips:\n  min_seconds: 15\n  max_seconds: 30\n")
        with mock.patch.object(config, "CONFIG_DIR", Path(d.name)), \
                mock.patch.object(config, "_settings_cache", None), \
                mock.patch.object(ur, "clip_length", return_value=(20, 45)):
            s = config.load_settings()
        self.assertEqual((s.get("clips.min_seconds"), s.get("clips.max_seconds")), (20, 45))
        self.assertEqual((s.get("clip_engine.min_seconds"), s.get("clip_engine.max_seconds")), (20, 45))


class PromptTest(unittest.TestCase):
    def test_universal_rules_come_first_and_win(self):
        with mock.patch("app.universal_rules.render", return_value="- Open striking"), \
                mock.patch("app.config.render_brand_rules", return_value="- Write as we"):
            block = llm._operator_rules("content")
        self.assertLess(block.index("THROUGHLINE'S RULES"), block.index("THE OPERATOR'S RULES"))
        self.assertIn("the account's own rules included", block)


if __name__ == "__main__":
    unittest.main()
