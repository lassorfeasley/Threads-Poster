"""The First reply settings page (app/web/main.py): one reply every account
configures, and the watch-page style Renewables.org uses, which only admins
see (app/web/access.py).

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
      SUPABASE_SERVICE_KEY= .venv/bin/python -m pytest -q tests/test_first_reply_page.py
"""
from __future__ import annotations

import unittest
from unittest import mock

SAVED = {"enabled": True, "watch_text": "Watch it all 👉 {watch_url}", "fallback_text": "Act 👉 Renewables.org"}


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class FirstReplyPageTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.web import main

        for name, value in (("load_first_reply", mock.Mock(return_value=dict(SAVED))),):
            patcher = mock.patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(main, "save_first_reply")
        self.save = patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app, follow_redirects=False)

    def as_customer(self):
        self.client.cookies.set("view_as", "customer")

    def test_admins_see_both_replies(self):
        page = self.client.get("/engagement/first-reply")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Your first reply", page.text)
        self.assertIn("Video page reply", page.text)
        self.assertIn('name="watch_text"', page.text)

    def test_customers_see_only_their_reply(self):
        self.as_customer()
        page = self.client.get("/engagement/first-reply")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Your first reply", page.text)
        self.assertNotIn('name="watch_text"', page.text)
        self.assertNotIn("Renewables.org uses", page.text)

    def test_a_customer_save_keeps_the_hidden_reply(self):
        self.as_customer()
        r = self.client.post("/engagement/first-reply",
                             data={"enabled": "1", "fallback_text": "Fund solar 👉 example.org",
                                   "watch_text": "a customer can't set this"})
        self.assertEqual(r.status_code, 303)
        self.save.assert_called_once_with(enabled=True, watch_text=SAVED["watch_text"],
                                          fallback_text="Fund solar 👉 example.org")

    def test_an_admin_can_switch_video_pages_off(self):
        r = self.client.post("/engagement/first-reply",
                             data={"enabled": "1", "fallback_text": "Act now", "watch_text": ""})
        self.assertEqual(r.status_code, 303)
        self.save.assert_called_once_with(enabled=True, watch_text="", fallback_text="Act now")

    def test_switching_on_needs_the_reply(self):
        r = self.client.post("/engagement/first-reply", data={"enabled": "1", "fallback_text": " "})
        self.assertIn("Write%20the%20reply", r.headers["location"])
        self.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
