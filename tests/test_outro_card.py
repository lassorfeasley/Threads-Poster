"""The account's own outro card (app/outro_card.py), laid over the outro
design (app/outro.py), and its settings page.

Run with:  env DATABASE_URL="sqlite://" GEMINI_API_KEY= ANTHROPIC_API_KEY= \
             .venv/bin/python -m pytest -q tests/test_outro_card.py
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from app import outro, outro_card

CARD = dict(enabled=True, lines=["Plant a tree", "Feed a family", "Get repaid"],
            website="example.org", color="#16794C")


def template() -> dict:
    return yaml.safe_load(outro_card.TEMPLATE.read_text())


class CleanTest(unittest.TestCase):
    def test_websites_are_tidied(self):
        for given in ("https://www.Example.org/", "example.org", "http://example.org"):
            with self.subTest(given=given):
                self.assertEqual(outro_card.clean(**{**CARD, "website": given})["website"].lower(), "example.org")

    def test_what_is_refused(self):
        bad = [{"website": "not a site"}, {"lines": ["x" * 41, "", ""]}, {"color": "green"},
               {"lines": ["", "", ""]}, {"website": ""}]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                outro_card.clean(**{**CARD, **change})
        # Switched off, an empty card is fine.
        outro_card.clean(**{**CARD, "enabled": False, "lines": ["", "", ""], "website": ""})


class ApplyTest(unittest.TestCase):
    def test_the_card_replaces_the_words_and_colours(self):
        raw = outro_card.apply(template(), outro_card.clean(**CARD))
        texts = [e["text"] for e in raw["elements"] if e.get("type", "text") == "text"]
        self.assertEqual(texts, ["Plant a tree", "Feed a family", "Get repaid"])
        typed = next(e for e in raw["elements"] if e.get("type") == "typed")
        self.assertEqual(typed["text"], "example.org")
        self.assertEqual(raw["tab"]["text"], "example.org")
        self.assertEqual(raw["background"]["color"], "#16794C")
        self.assertEqual(typed["color"], "#FFFFFF")           # white on green
        light = outro_card.apply(template(), outro_card.clean(**{**CARD, "color": "#F2C230"}))
        self.assertEqual(next(e for e in light["elements"] if e.get("type") == "typed")["color"], "#111111")
        self.assertEqual(light["tab"]["color"], "#111111")

    def test_a_blank_line_goes_and_its_spacing_moves_up(self):
        raw = outro_card.apply(template(), outro_card.clean(**{**CARD, "lines": ["One", "Two", ""]}))
        texts = [e for e in raw["elements"] if e.get("type", "text") == "text"]
        self.assertEqual([e["text"] for e in texts], ["One", "Two"])
        self.assertEqual(texts[-1]["gap"], template()["elements"][2]["gap"])
        vertical = raw["overrides"]["vertical"]["elements"]
        self.assertEqual(vertical[1]["gap"], template()["overrides"]["vertical"]["elements"][2]["gap"])
        self.assertEqual(vertical[2], template()["overrides"]["vertical"]["elements"][3])   # the website moved up


class RendererTest(unittest.TestCase):
    def setUp(self):
        outro_card._cache = None
        self.addCleanup(setattr, outro_card, "_cache", None)
        self.missing = Path(tempfile.mkdtemp()) / "outro.yaml"     # an account without a design file

    def test_no_design_and_no_card_means_no_outro(self):
        with mock.patch.object(outro_card, "load", return_value=None), \
                mock.patch.object(outro, "CONFIG_FILE", self.missing):
            self.assertFalse(outro.enabled(self.missing))

    def test_a_saved_card_turns_it_on_from_the_template(self):
        card = outro_card.clean(**CARD)
        with mock.patch.object(outro_card, "load", return_value=card), \
                mock.patch.object(outro, "CONFIG_FILE", self.missing):
            self.assertTrue(outro.enabled(self.missing))
            cfg = outro.load_config("vertical", self.missing)
            self.assertEqual(cfg["_base"], str(outro_card.TEMPLATE.parent))
            frame = outro.Card(360, 640, cfg).frame(float(cfg["duration"]) - 0.01)
            self.assertEqual(frame.size, (360, 640))

    def test_cached_copies_survive_until_a_card_is_saved(self):
        design = Path(tempfile.mkdtemp()) / "outro.yaml"
        design.write_text(outro_card.TEMPLATE.read_text().replace("enabled: false", "enabled: true", 1))
        clip = Path(tempfile.mkdtemp()) / "c.mp4"
        clip.write_bytes(b"x")
        with mock.patch.object(outro, "CONFIG_FILE", design):
            with mock.patch.object(outro_card, "load", return_value=None):
                plain = outro._cache_key(clip, design)
            with mock.patch.object(outro_card, "load", return_value=outro_card.clean(**CARD)):
                carded = outro._cache_key(clip, design)
            with mock.patch.object(outro_card, "load",
                                   return_value=outro_card.clean(**{**CARD, "color": "#E2445C"})):
                recoloured = outro._cache_key(clip, design)
        self.assertEqual(len({plain, carded, recoloured}), 3)


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class OutroPageTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.web import main

        for p in (mock.patch.object(outro_card, "load", return_value=None),):
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app, follow_redirects=False)

    def test_the_page_opens_on_the_current_design(self):
        page = self.client.get("/outro-card")
        self.assertEqual(page.status_code, 200)
        self.assertIn('name="line1"', page.text)
        self.assertNotIn("logo", page.text.split('class="settings-body"')[1].lower())

    def test_the_preview_is_drawn_by_the_renderer(self):
        r = self.client.get("/outro-card/preview.png", params={"orientation": "vertical", "line1": "Hi",
                                                               "website": "example.org", "color": "#7B5CD6"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")
        self.assertTrue(r.content.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get("/outro-card/preview.png", params={"website": "nope"}).status_code, 400)

    def test_saving(self):
        with mock.patch.object(outro_card, "save", return_value={**CARD}) as save:
            r = self.client.post("/outro-card", data={"enabled": "1", "line1": "A", "line2": "B", "line3": "",
                                                      "website": "example.org", "color": "__custom",
                                                      "custom_color": "#123456"})
        self.assertEqual(r.status_code, 303)
        save.assert_called_once_with(enabled=True, lines=["A", "B", ""], website="example.org",
                                     color="#123456")

    def test_a_bad_card_says_why(self):
        r = self.client.post("/outro-card", data={"enabled": "1", "line1": "A", "website": "not a site",
                                                  "color": "#16794C"})
        self.assertIn("example.org", r.headers["location"])

    def test_customers_can_use_it(self):
        from app.web import access

        self.assertTrue(access.customer_may("/outro-card", "POST"))
        self.assertTrue(access.customer_may("/outro-card/preview.png"))


if __name__ == "__main__":
    unittest.main()
