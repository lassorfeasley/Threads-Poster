"""Meta sign-in that comes back to the dashboard (Threads, Instagram), and the
onboarding connect step. Run with: python -m unittest tests.test_meta_connect -v"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.web import main, onboarding as page

ENV = {"THREADS_REDIRECT_URI": "https://app.example.com/threads/callback",
       "THREADS_APP_SECRET": "s3cret", "THREADS_APP_ID": "123",
       "INSTAGRAM_REDIRECT_URI": "https://localhost/instagram-callback"}


class StateTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, ENV)
        p.start()
        self.addCleanup(p.stop)

    def test_round_trip(self):
        state = main.meta_state("threads", "/welcome/connect")
        self.assertEqual(main._meta_return("threads", state), "/welcome/connect")

    def test_forged_or_offsite_targets_go_to_connections(self):
        state = main.meta_state("threads", "/welcome/connect")
        packed, sig = state.rsplit(".", 1)
        self.assertEqual(main._meta_return("threads", packed + ".0000"), "/connections")
        self.assertEqual(main._meta_return("instagram", state), "/connections")   # other platform's key
        self.assertEqual(main._meta_return("threads", "garbage"), "/connections")
        self.assertEqual(main._meta_return("threads", main.meta_state("threads", "//evil.example")), "/")

    def test_redirect_mode_only_for_our_own_callback(self):
        self.assertTrue(main.meta_redirects_here("threads"))
        self.assertFalse(main.meta_redirects_here("instagram"))   # localhost: paste the code

    def test_the_sign_in_link_carries_the_state(self):
        url = main.meta_connect_url("threads", "/welcome/connect")
        self.assertIn("state=", url)
        self.assertIn("redirect_uri=https%3A%2F%2Fapp.example.com%2Fthreads%2Fcallback", url)


class CallbackTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, ENV)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def test_connected_and_back_where_they_started(self):
        state = main.meta_state("threads", "/welcome/connect")
        with mock.patch.object(main.threads_api, "exchange_code", return_value={"username": "coreflow"}) as ex:
            r = self.client.get(f"/threads/callback?code=abc%23_&state={state}", follow_redirects=False)
        ex.assert_called_once()
        self.assertTrue(r.headers["location"].startswith("/welcome/connect?msg="))
        self.assertIn("coreflow", r.headers["location"])

    def test_declined_says_so(self):
        state = main.meta_state("threads", "/inbox")
        r = self.client.get(f"/threads/callback?error=access_denied&state={state}", follow_redirects=False)
        self.assertTrue(r.headers["location"].startswith("/inbox?msg="))
        self.assertIn("wasn", r.headers["location"])


class ConnectStepTest(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(page.router)
        self.client = TestClient(app)
        for p in (mock.patch.object(page.onboarding, "save_step", return_value={"steps_done": []}),
                  mock.patch.object(page.onboarding, "save"),
                  mock.patch.object(page.onboarding, "saved_values", return_value={})):
            p.start()
            self.addCleanup(p.stop)

    def test_needs_threads_or_later(self):
        with mock.patch.object(page, "threads_connected", return_value=False):
            r = self.client.post("/welcome/connect", data={}, follow_redirects=False)
            self.assertTrue(r.headers["location"].startswith("/welcome/connect?msg="))
            r = self.client.post("/welcome/connect", data={"later": "1"}, follow_redirects=False)
            self.assertEqual(r.headers["location"], "/welcome/launch")
        with mock.patch.object(page, "threads_connected", return_value=True):
            r = self.client.post("/welcome/connect", data={}, follow_redirects=False)
            self.assertEqual(r.headers["location"], "/welcome/launch")


if __name__ == "__main__":
    unittest.main()
