"""The clips-ready email (app/clip_emails.py). Run with:
python -m unittest tests.test_clip_emails -v"""
from __future__ import annotations

import datetime as dt
import os
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from app import clip_emails as ce

NY = ZoneInfo("America/New_York")


def at(day, hour):
    return dt.datetime(2026, 11, day, hour, 0, tzinfo=NY)


class DueTest(unittest.TestCase):
    def test_never_before_nine(self):
        self.assertFalse(ce.due("daily", None, at(20, 8)))
        self.assertTrue(ce.due("daily", None, at(20, 9)))

    def test_daily_is_once_a_calendar_day(self):
        sent = at(19, 9).isoformat()
        self.assertTrue(ce.due("daily", sent, at(20, 9)))
        self.assertFalse(ce.due("daily", at(20, 9).isoformat(), at(20, 17)))

    def test_weekly_biweekly_monthly(self):
        sent = at(1, 9).isoformat()
        self.assertFalse(ce.due("weekly", sent, at(7, 10)))
        self.assertTrue(ce.due("weekly", sent, at(8, 9)))
        self.assertFalse(ce.due("biweekly", sent, at(14, 9)))
        self.assertTrue(ce.due("biweekly", sent, at(15, 9)))
        self.assertFalse(ce.due("monthly", sent, at(30, 9)))

    def test_default_is_daily(self):
        self.assertEqual(ce.frequency_for("new@example.com", {}), "daily")
        self.assertEqual(ce.frequency_for("A@Example.com", {"a@example.com": {"frequency": "weekly"}}), "weekly")


class ComposeTest(unittest.TestCase):
    def test_subject_body_and_links(self):
        clips = {"total": 9, "new": 4, "top": [{"title": "Rooftop <class>", "caption": "One cue"}]}
        subject, body, text = ce.compose(brand="Core & Flow", frequency="weekly", clips=clips,
                                         app_url="https://app.example.com")
        self.assertEqual(subject, "9 clips ready to approve (4 new)")
        self.assertIn("Rooftop &lt;class&gt;", body)             # model text is escaped
        self.assertIn("https://app.example.com/inbox", body)
        self.assertIn("https://app.example.com/account/emails", text)
        self.assertIn("4 new clips this week", body)
        self.assertIn("…and 8 more", body)


class RunTest(unittest.TestCase):
    ENV = {"RESEND_API_KEY": "k", "PUBLIC_URL": "https://app.example.com",
           "SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SERVICE_KEY": "s"}

    def setUp(self):
        for p in (mock.patch.dict(os.environ, self.ENV),
                  mock.patch("app.config.scheduler_timezone", return_value=NY),
                  mock.patch("app.config.load_brand", return_value={"name": "Core"}),
                  mock.patch.object(ce, "preferences", return_value={}),
                  mock.patch.object(ce, "recipients", return_value=["a@example.com", "b@example.com"])):
            p.start()
            self.addCleanup(p.stop)

    def test_nothing_waiting_means_no_email(self):
        with mock.patch.object(ce, "waiting", return_value={"total": 0, "new": 0, "top": []}), \
                mock.patch.object(ce, "send") as send:
            self.assertEqual(ce.run(at(20, 10)), [])
        send.assert_not_called()

    def test_sends_to_each_claimed_recipient_and_undoes_a_failure(self):
        clips = {"total": 3, "new": 3, "top": []}

        def claim(email, freq, now, tz):
            return "" if email == "a@example.com" else None    # b was already sent today
        with mock.patch.object(ce, "waiting", return_value=clips), \
                mock.patch.object(ce, "_claim", side_effect=claim), \
                mock.patch.object(ce, "send") as send:
            self.assertEqual(ce.run(at(20, 10)), ["a@example.com"])
        self.assertEqual(send.call_args.args[0], "a@example.com")
        with mock.patch.object(ce, "waiting", return_value=clips), \
                mock.patch.object(ce, "_claim", return_value="2026-11-19T14:00:00+00:00"), \
                mock.patch.object(ce, "send", side_effect=OSError("resend down")), \
                mock.patch.object(ce, "_unclaim") as unclaim:
            self.assertEqual(ce.run(at(20, 10)), [])
        self.assertEqual(unclaim.call_count, 2)

    def test_off_without_settings(self):
        with mock.patch.dict(os.environ, {"RESEND_API_KEY": ""}), mock.patch.object(ce, "send") as send:
            self.assertEqual(ce.run(at(20, 10)), [])
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
