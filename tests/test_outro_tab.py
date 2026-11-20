"""The corner tab's own settings (app/outro_card.py). Run with:
python -m unittest tests.test_outro_tab -v"""
from __future__ import annotations

import json
import unittest

from app import outro_card

RAW = {"tab": {"enabled": True, "text": "example.org", "x": 0.07}, "elements": [], "background": {}}
CARD = {"enabled": True, "lines": ["a", "", ""], "website": "core.example", "color": "#1A4A7D"}


class TabTest(unittest.TestCase):
    def test_without_tab_settings_nothing_changes(self):
        tab = outro_card.apply(RAW, CARD)["tab"]
        self.assertEqual((tab["text"], tab["box_color"], tab["enabled"]), ("core.example", "#1A4A7D", True))
        # Same render-cache key as before the tab could be set.
        old = json.dumps({k: CARD.get(k) for k in ("enabled", "lines", "website", "color")}, sort_keys=True)
        self.assertEqual(outro_card.fingerprint(CARD), old)

    def test_own_words_and_color(self):
        card = outro_card.clean(**CARD, tab_on=True, tab_text="Watch the full talk", tab_color="f2c230")
        tab = outro_card.apply(RAW, card)["tab"]
        self.assertEqual((tab["text"], tab["box_color"], tab["color"]),
                         ("Watch the full talk", "#F2C230", "#111111"))
        self.assertNotEqual(outro_card.fingerprint(card), outro_card.fingerprint(CARD))

    def test_switched_off(self):
        card = outro_card.clean(**CARD, tab_on=False)
        self.assertFalse(outro_card.apply(RAW, card)["tab"]["enabled"])

    def test_too_long(self):
        with self.assertRaises(ValueError):
            outro_card.clean(**CARD, tab_text="x" * 41)

    def test_settings_view(self):
        self.assertEqual(outro_card.tab_settings(CARD),
                         {"on": True, "text": "core.example", "color": "#1A4A7D", "custom_text": False})


if __name__ == "__main__":
    unittest.main()
