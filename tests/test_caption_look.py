"""The burned-in captions' font and highlight colour (app/caption_look.py) and
the Subtitles settings page.

Run with:  env DATABASE_URL="sqlite://" GEMINI_API_KEY= ANTHROPIC_API_KEY= \
             .venv/bin/python -m pytest -q tests/test_caption_look.py
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import caption_look
from app.models import AppToken, Base

SETTINGS = {"subtitles.font_file": "FunnelDisplay-SemiBold.ttf", "subtitles.text_color": "#FFFFFF",
            "subtitles.highlight_box_color": "#FFFFFF", "subtitles.highlight_text_color": "#1A4A7D"}


class _Resp:
    def __init__(self, status=200, text="", content=b""):
        self.status_code, self.text, self.content = status, text, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise caption_look_requests().HTTPError(str(self.status_code))


def caption_look_requests():
    import requests
    return requests


class CaptionLookTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(engine, future=True, expire_on_commit=False)

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            s = self.Session()
            try:
                yield s
                if not read_only:
                    s.commit()
            finally:
                s.close()

        for p in (mock.patch("app.db.session_scope", scope),
                  mock.patch.object(caption_look, "FONT_CACHE", Path(tempfile.mkdtemp()))):
            p.start()
            self.addCleanup(p.stop)
        caption_look._cache = None
        self.addCleanup(setattr, caption_look, "_cache", None)

    def fake_google(self, families_with=("Space Grotesk",), weights=(700,)):
        """requests.get standing in for Google Fonts: CSS for known families
        and weights, then the font bytes."""
        def get(url, params=None, headers=None, timeout=None):
            if url.startswith("https://fonts.googleapis.com/css2"):
                spec = params["family"]
                name, _, w = spec.partition(":wght@")
                if name in families_with and (not w or int(w) in weights):
                    return _Resp(text=f"src: url(https://fonts.gstatic.com/s/x/{name}.ttf) format('truetype');")
                return _Resp(status=400, text="bad")
            return _Resp(content=b"TTF-BYTES")
        return mock.patch("requests.get", side_effect=get)

    def test_nine_of_each(self):
        self.assertEqual(len(caption_look.FONTS), 9)
        self.assertEqual(len(caption_look.COLORS), 9)
        self.assertEqual(caption_look.FONTS[0], caption_look.BUNDLED_FAMILY)

    def test_colours_are_checked_and_tidied(self):
        self.assertEqual(caption_look.clean_color("1a4a7d"), "#1A4A7D")
        for bad in ("", "#12345", "red", "#GGGGGG"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                caption_look.clean_color(bad)

    def test_text_on_the_box_reads(self):
        self.assertEqual(caption_look._contrast_text("#FFFFFF"), "#111111")
        self.assertEqual(caption_look._contrast_text("#F2C230"), "#111111")   # yellow
        self.assertEqual(caption_look._contrast_text("#1A4A7D"), "#FFFFFF")   # navy
        self.assertEqual(caption_look._contrast_text("#E2445C"), "#FFFFFF")   # red

    def test_nothing_chosen_keeps_settings_yaml(self):
        look = caption_look.resolve(SETTINGS)
        self.assertEqual((look["font"], look["box"], look["box_text"]),
                         ("FunnelDisplay-SemiBold.ttf", "#FFFFFF", "#1A4A7D"))

    def test_the_bundled_font_needs_no_download(self):
        with mock.patch("requests.get", side_effect=AssertionError("no network")):
            self.assertEqual(caption_look.font_file("Funnel Display"), caption_look.BUNDLED_FILE)

    def test_any_google_font_is_fetched_once(self):
        with self.fake_google(weights=(600,)) as get:      # no 700: falls back to 600
            path = caption_look.font_file("Space Grotesk")
            self.assertEqual(path.read_bytes(), b"TTF-BYTES")
            calls = get.call_count
            self.assertEqual(caption_look.font_file("Space Grotesk"), path)
            self.assertEqual(get.call_count, calls)        # cached: no second fetch

    def test_save_and_render_with_the_choice(self):
        with self.fake_google():
            caption_look.save(font="Space Grotesk", box="#e2445c")
            look = caption_look.resolve(SETTINGS)
        self.assertTrue(str(look["font"]).endswith("space-grotesk.ttf"))
        self.assertEqual((look["box"], look["box_text"]), ("#E2445C", "#FFFFFF"))
        with self.Session() as s:
            self.assertIn("Space Grotesk", s.get(AppToken, caption_look.TOKEN).value)

    def test_a_font_google_doesnt_have_is_refused(self):
        with self.fake_google(), self.assertRaises(caption_look.FontError):
            caption_look.save(font="Comic Sanz", box="#FFFFFF")
        with self.Session() as s:
            self.assertIsNone(s.get(AppToken, caption_look.TOKEN))

    def test_a_font_that_cant_be_had_falls_back_rather_than_failing_the_render(self):
        with self.fake_google():
            caption_look.save(font="Space Grotesk", box="#FFFFFF")
        for f in caption_look.FONT_CACHE.glob("*.ttf"):
            f.unlink()
        with mock.patch("requests.get", side_effect=caption_look_requests().ConnectionError("offline")):
            look = caption_look.resolve(SETTINGS)
        self.assertEqual(look["font"], "FunnelDisplay-SemiBold.ttf")


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class SubtitlesPageTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.web import main

        p = mock.patch.object(caption_look, "load", return_value={"font": "", "box": ""})
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(main.app, follow_redirects=False)

    def test_the_page_offers_nine_fonts_and_nine_colours(self):
        page = self.client.get("/subtitle-style")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.text.count('name="font" value="') - 1, 9)   # plus "any Google font"
        self.assertEqual(page.text.count('name="box" value="') - 1, 9)    # plus "your own colour"
        self.assertIn('value="Funnel Display" data-weight="600" checked', page.text)

    def test_saving_a_preset_and_your_own(self):
        with mock.patch.object(caption_look, "save", return_value={"font": "Anton", "box": "#16794C"}) as save:
            r = self.client.post("/subtitle-style", data={"font": "Anton", "box": "#16794C"})
            self.assertEqual(r.status_code, 303)
            save.assert_called_with(font="Anton", box="#16794C")
            self.client.post("/subtitle-style", data={"font": "__custom", "custom_font": "Space Grotesk",
                                                      "box": "__custom", "custom_box": "#123456"})
            save.assert_called_with(font="Space Grotesk", box="#123456")

    def test_a_bad_choice_says_why(self):
        with mock.patch.object(caption_look, "save", side_effect=caption_look.FontError("Google Fonts has no font called “X”")):
            r = self.client.post("/subtitle-style", data={"font": "__custom", "custom_font": "X", "box": "#FFFFFF"})
        self.assertIn("Google%20Fonts%20has%20no%20font", r.headers["location"])

    def test_customers_can_use_it(self):
        from app.web import access

        self.assertTrue(access.customer_may("/subtitle-style", "POST"))


if __name__ == "__main__":
    unittest.main()
