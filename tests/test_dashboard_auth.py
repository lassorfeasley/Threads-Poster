"""Dashboard login (app/web/auth.py and the gate in app/web/main.py).

The unit tests run anywhere. The route tests import app.web.main, which
initialises whatever DATABASE_URL names and starts background threads, so they
are skipped unless that is a throwaway in-memory SQLite:

  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false .venv/bin/python -m pytest -q tests/test_dashboard_auth.py
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from app.web import auth

SECRET = "s" * 48


def _env(**extra: str) -> dict:
    return {"AUTH_SECRET": SECRET, **extra}


class PasswordTests(unittest.TestCase):
    def test_round_trip(self):
        stored = auth.hash_password("correct horse battery")
        self.assertTrue(auth.verify_password("correct horse battery", stored))
        self.assertFalse(auth.verify_password("wrong", stored))

    def test_salted(self):
        self.assertNotEqual(auth.hash_password("x" * 12), auth.hash_password("x" * 12))

    def test_malformed_hash_never_verifies(self):
        for stored in ("", "plain", "scrypt$1$2$3", "bcrypt$a$b$c$d$e", "scrypt$x$8$1$AA$AA"):
            self.assertFalse(auth.verify_password("anything", stored), stored)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.hash = auth.hash_password("correct horse battery")
        patcher = mock.patch.dict(os.environ, _env(DASHBOARD_USERS=f"Me@Example.org:{self.hash}"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_users_parsing_lowercases_and_skips_junk(self):
        with mock.patch.dict(os.environ, {"DASHBOARD_USERS": f" A@b.c:{self.hash} ,junk,:x,d@e:"}):
            self.assertEqual(list(auth.users()), ["a@b.c"])

    def test_authenticate(self):
        self.assertEqual(auth.authenticate(" ME@example.org ", "correct horse battery"), "me@example.org")
        self.assertIsNone(auth.authenticate("me@example.org", "nope"))
        self.assertIsNone(auth.authenticate("other@example.org", "correct horse battery"))

    def test_session_round_trip(self):
        token = auth.make_session("me@example.org")
        self.assertEqual(auth.read_session(token), "me@example.org")

    def test_tampered_session_rejected(self):
        token = auth.make_session("me@example.org")
        payload, sig = token.rsplit(".", 1)
        forged = auth._b64(b"admin@example.org|9999999999")
        self.assertIsNone(auth.read_session(f"{forged}.{sig}"))
        self.assertIsNone(auth.read_session(payload + ".x" + sig[1:]))
        self.assertIsNone(auth.read_session("garbage"))
        self.assertIsNone(auth.read_session(None))

    def test_expired_session_rejected(self):
        token = auth.make_session("me@example.org", now=1000)
        self.assertIsNone(auth.read_session(token, now=1000 + auth.SESSION_SECONDS + 1))

    def test_removed_user_is_signed_out(self):
        token = auth.make_session("me@example.org")
        with mock.patch.dict(os.environ, {"DASHBOARD_USERS": f"else@example.org:{self.hash}"}):
            self.assertIsNone(auth.read_session(token))

    def test_rotated_secret_signs_everyone_out(self):
        token = auth.make_session("me@example.org")
        with mock.patch.dict(os.environ, {"AUTH_SECRET": "t" * 48}):
            self.assertIsNone(auth.read_session(token))

    def test_short_secret_disables_sessions(self):
        with mock.patch.dict(os.environ, {"AUTH_SECRET": "short"}):
            token = auth.make_session("me@example.org")
            self.assertIsNone(auth.read_session(token))
            self.assertIn("AUTH_SECRET", auth.config_problem())

    def test_config_problem(self):
        self.assertEqual(auth.config_problem(), "")
        with mock.patch.dict(os.environ, {"DASHBOARD_USERS": ""}):
            self.assertIn("DASHBOARD_USERS", auth.config_problem())


class HelperTests(unittest.TestCase):
    def test_safe_next(self):
        self.assertEqual(auth.safe_next("/cut/3?x=1"), "/cut/3?x=1")
        for bad in (None, "", "https://evil.com", "//evil.com", "/\\evil.com", "evil"):
            self.assertEqual(auth.safe_next(bad), "/", bad)

    def test_throttle(self):
        key = "test:throttle"
        auth.clear_failures(key)
        for _ in range(auth._FAIL_LIMIT):
            self.assertFalse(auth.throttled(key, now=100))
            auth.record_failure(key, now=100)
        self.assertTrue(auth.throttled(key, now=100))
        self.assertFalse(auth.throttled(key, now=100 + auth._FAIL_WINDOW + 1))
        auth.clear_failures(key)


def _throwaway_db() -> bool:
    from app import db

    return db._url == "sqlite://"


@unittest.skipUnless(_throwaway_db(), "set DATABASE_URL=sqlite:// (a throwaway in-memory database)")
class GateTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.web import main

        self.hash = auth.hash_password("correct horse battery")
        patcher = mock.patch.dict(os.environ, _env(DASHBOARD_USERS=f"me@example.org:{self.hash}"))
        patcher.start()
        self.addCleanup(patcher.stop)
        auth._fails.clear()
        self.client = TestClient(main.app, follow_redirects=False)

    def test_pages_redirect_to_login(self):
        r = self.client.get("/calendar?week=2", headers={"accept": "text/html"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/login?next=%2Fcalendar%3Fweek%3D2")

    def test_api_and_writes_get_401(self):
        self.assertEqual(self.client.get("/media/clip/1").status_code, 401)
        self.assertEqual(self.client.post("/monitor/run").status_code, 401)

    def test_public_paths(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/login").status_code, 200)
        self.assertEqual(self.client.get("/static/style.css").status_code, 200)

    def test_login_flow(self):
        r = self.client.post("/login", data={"email": "me@example.org", "password": "nope", "next": "/x"})
        self.assertEqual(r.status_code, 303)
        self.assertIn("error=invalid", r.headers["location"])
        self.assertNotIn(auth.COOKIE_NAME, r.cookies)

        r = self.client.post("/login", data={"email": "ME@example.org", "password": "correct horse battery",
                                             "next": "https://evil.com"})
        self.assertEqual(r.headers["location"], "/")
        cookie = r.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/no-such-page").status_code, 404)

        r = self.client.post("/logout")
        self.assertEqual(r.headers["location"], "/login")
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/no-such-page").status_code, 401)

    def test_secure_cookie_behind_https_proxy(self):
        r = self.client.post("/login", data={"email": "me@example.org", "password": "correct horse battery"},
                             headers={"x-forwarded-proto": "https"})
        self.assertIn("secure", r.headers["set-cookie"].lower())

    def test_throttled_after_repeated_failures(self):
        for _ in range(auth._FAIL_LIMIT):
            self.client.post("/login", data={"email": "me@example.org", "password": "nope"})
        r = self.client.post("/login", data={"email": "me@example.org", "password": "correct horse battery"})
        self.assertIn("error=throttled", r.headers["location"])

    def test_off_when_unconfigured(self):
        with mock.patch.dict(os.environ, {"DASHBOARD_USERS": ""}):
            self.assertEqual(self.client.get("/healthz").status_code, 200)
            self.assertEqual(self.client.get("/no-such-page").status_code, 404)
            self.assertEqual(self.client.get("/login").status_code, 303)


if __name__ == "__main__":
    unittest.main()
