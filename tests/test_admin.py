"""Throughline Admin (app/admin): who gets in, and health from a snapshot.
Run with: python -m unittest tests.test_admin -v"""
from __future__ import annotations

import datetime as dt
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from app.admin import collect, main


def _user(email, role):
    return SimpleNamespace(email=email, role=role)


class AccessTest(unittest.TestCase):
    def test_only_listed_admins(self):
        with mock.patch.dict(os.environ, {"ADMIN_EMAILS": "me@example.com"}):
            self.assertTrue(main.is_operator(_user("Me@Example.com", "admin")))
            self.assertFalse(main.is_operator(_user("me@example.com", "member")))   # wrong role
            self.assertFalse(main.is_operator(_user("other@example.com", "admin")))  # not listed
            self.assertFalse(main.is_operator(None))

    def test_no_list_means_nobody(self):
        with mock.patch.dict(os.environ, {"ADMIN_EMAILS": ""}):
            self.assertFalse(main.is_operator(_user("me@example.com", "admin")))


class HealthTest(unittest.TestCase):
    def snap(self, **over):
        now = dt.datetime.now(dt.timezone.utc)
        base = {"heartbeats": [{"role": "scheduler", "alive": True, "at": now},
                               {"role": "home-worker", "alive": True, "at": now}],
                "publishing": {"failed_count": 0, "overdue": 0, "ig_failed": 0},
                "monitor": {"status": "done", "last_ok": now}, "tokens": [],
                "pipeline": {"found": 5, "published": 3, "inbox_waiting": 0, "oldest_waiting": None},
                "machines": []}
        base.update(over)
        return base

    def test_healthy(self):
        self.assertEqual(collect._health(self.snap()), ("ok", []))

    def test_dead_worker_and_failed_posts_are_bad(self):
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
        status, reasons = collect._health(self.snap(
            heartbeats=[{"role": "scheduler", "alive": False, "at": old},
                        {"role": "home-worker", "alive": True, "at": old}],
            publishing={"failed_count": 2, "overdue": 0, "ig_failed": 0}))
        self.assertEqual(status, "bad")
        self.assertTrue(any("Scheduler" in r for r in reasons))
        self.assertTrue(any("2 posts failed" in r for r in reasons))

    def test_expiring_sign_in_warns_and_never_expiring_doesnt(self):
        status, reasons = collect._health(self.snap(tokens=[
            {"name": "threads", "days_left": 3.0}, {"name": "facebook", "days_left": None, "never": True}]))
        self.assertEqual(status, "warn")
        self.assertEqual(reasons, ["Threads sign-in expires in 3 days"])


if __name__ == "__main__":
    unittest.main()
