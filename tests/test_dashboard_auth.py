"""Dashboard login backed by Supabase Auth (app/web/auth.py, the gate and
pages in app/web/main.py). Supabase Auth is replaced by an in-memory fake.

The unit tests run anywhere. The route tests import app.web.main, which
initialises whatever DATABASE_URL names and starts background threads, so they
are skipped unless that is a throwaway in-memory SQLite:

  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false .venv/bin/python -m pytest -q tests/test_dashboard_auth.py
"""
from __future__ import annotations

import os
import secrets
import unittest
from unittest import mock

import httpx

from app.web import auth

ENV = {"DASHBOARD_AUTH": "supabase", "SUPABASE_URL": "https://x.supabase.co",
       "SUPABASE_SERVICE_KEY": "service"}


class FakeGoTrue:
    """Just enough of Supabase Auth's REST API."""

    def __init__(self):
        self.users: dict[str, dict] = {}       # email -> user record
        self.passwords: dict[str, str] = {}
        self.access: dict[str, str] = {}       # access token -> email
        self.refresh: dict[str, str] = {}      # refresh token -> email
        self.sent: list[tuple[str, str, str]] = []  # (kind, email, redirect_to)
        self.down = False

    def add(self, email, password, workspaces=("climate",), role="member"):
        self.users[email] = {"id": secrets.token_hex(4), "email": email,
                             "app_metadata": {"workspaces": list(workspaces), "role": role}}
        self.passwords[email] = password

    def _tokens(self, email):
        a, r = "a-" + secrets.token_hex(6), "r-" + secrets.token_hex(6)
        self.access[a], self.refresh[r] = email, email
        return {"access_token": a, "refresh_token": r, "user": self.users[email]}

    def expire(self, access_token):
        self.access.pop(access_token, None)

    def __call__(self, method, path, *, token=None, json_body=None, params=None):
        if self.down:
            return httpx.Response(503, json={"msg": "down"})
        body, params = json_body or {}, params or {}
        if path == "/token" and params.get("grant_type") == "password":
            email = body["email"]
            if email in self.users and self.passwords.get(email) == body["password"]:
                return httpx.Response(200, json=self._tokens(email))
            return httpx.Response(400, json={"msg": "Invalid login credentials"})
        if path == "/token" and params.get("grant_type") == "refresh_token":
            email = self.refresh.pop(body["refresh_token"], None)
            return httpx.Response(200, json=self._tokens(email)) if email else \
                httpx.Response(400, json={"msg": "Invalid Refresh Token"})
        if path == "/user" and method == "GET":
            email = self.access.get(token)
            return httpx.Response(200, json=self.users[email]) if email else \
                httpx.Response(401, json={"msg": "invalid JWT"})
        if path == "/user" and method == "PUT":
            email = self.access.get(token)
            if not email:
                return httpx.Response(401, json={"msg": "invalid JWT"})
            self.passwords[email] = body["password"]
            return httpx.Response(200, json=self.users[email])
        if path == "/logout":
            self.access.pop(token, None)
            return httpx.Response(204)
        if path == "/recover":
            self.sent.append(("recover", body["email"], params.get("redirect_to", "")))
            return httpx.Response(200, json={})
        if path == "/invite":
            self.add(body["email"], None, workspaces=())
            self.sent.append(("invite", body["email"], params.get("redirect_to", "")))
            return httpx.Response(200, json=self.users[body["email"]])
        if path == "/admin/users" and method == "GET":
            return httpx.Response(200, json={"users": list(self.users.values())})
        if path == "/admin/users" and method == "POST":
            self.add(body["email"], body["password"], workspaces=())
            self.users[body["email"]]["app_metadata"] = body["app_metadata"]
            return httpx.Response(200, json=self.users[body["email"]])
        if path.startswith("/admin/users/") and method == "PUT":
            uid = path.rsplit("/", 1)[1]
            user = next(u for u in self.users.values() if u["id"] == uid)
            if "app_metadata" in body:
                user["app_metadata"] = body["app_metadata"]
            if "password" in body:
                self.passwords[user["email"]] = body["password"]
            return httpx.Response(200, json=user)
        raise AssertionError(f"unexpected {method} {path}")


class AuthTestBase(unittest.TestCase):
    def setUp(self):
        self.fake = FakeGoTrue()
        self.fake.add("me@example.org", "correct horse battery")
        for patch in (mock.patch.dict(os.environ, ENV),
                      mock.patch.object(auth, "_request", side_effect=self.fake),
                      mock.patch.object(auth, "_workspace", return_value="climate")):
            patch.start()
            self.addCleanup(patch.stop)
        auth._cache.clear()
        auth._fails.clear()


class SessionTests(AuthTestBase):
    def test_sign_in(self):
        session, user = auth.sign_in(" ME@example.org ", "correct horse battery")
        self.assertEqual(user.email, "me@example.org")
        self.assertTrue(user.can_use("climate"))
        self.assertIsNone(auth.sign_in("me@example.org", "nope"))

    def test_sign_in_outage_raises(self):
        self.fake.down = True
        with self.assertRaises(auth.AuthError):
            auth.sign_in("me@example.org", "correct horse battery")

    def test_cookie_round_trip_and_garbage(self):
        s = auth.Session("a", "r")
        self.assertEqual(auth.read_cookie(auth.make_cookie(s)), s)
        for bad in (None, "", "not-base64!!", auth.make_cookie(auth.Session("a", "r"))[:-3] + "zzz"):
            self.assertIsInstance(auth.read_cookie(bad), (auth.Session, type(None)))
        self.assertIsNone(auth.read_cookie("bm90IGpzb24"))

    def test_resolve_and_refresh(self):
        session, _ = auth.sign_in("me@example.org", "correct horse battery")
        user, renewed = auth.resolve(auth.make_cookie(session))
        self.assertEqual((user.email, renewed), ("me@example.org", None))
        self.fake.expire(session.access_token)
        auth._cache.clear()
        user, renewed = auth.resolve(auth.make_cookie(session))
        self.assertEqual(user.email, "me@example.org")
        self.assertIsNotNone(renewed)
        self.assertNotEqual(renewed.access_token, session.access_token)

    def test_dead_session_resolves_to_nobody(self):
        self.assertEqual(auth.resolve(auth.make_cookie(auth.Session("bogus", "bogus"))), (None, None))
        self.assertEqual(auth.resolve(None), (None, None))

    def test_other_workspace_resolves_to_nobody(self):
        self.fake.add("other@example.org", "x" * 12, workspaces=("another",))
        session, _ = auth.sign_in("other@example.org", "x" * 12)
        self.assertEqual(auth.resolve(auth.make_cookie(session)), (None, None))

    def test_signup_without_metadata_opens_nothing(self):
        self.fake.add("stranger@example.org", "x" * 12, workspaces=())
        session, user = auth.sign_in("stranger@example.org", "x" * 12)
        self.assertFalse(user.can_use("climate"))

    def test_config_problem(self):
        self.assertEqual(auth.config_problem(), "")
        with mock.patch.dict(os.environ, {"DASHBOARD_AUTH": ""}):
            self.assertIn("DASHBOARD_AUTH", auth.config_problem())
        with mock.patch.dict(os.environ, {"SUPABASE_SERVICE_KEY": ""}):
            self.assertIn("SUPABASE", auth.config_problem())


class AdminTests(AuthTestBase):
    def test_add_user_then_grant_keeps_other_workspaces(self):
        auth.add_user("new@example.org", "y" * 12, "climate", "owner")
        self.assertEqual(self.fake.users["new@example.org"]["app_metadata"],
                         {"workspaces": ["climate"], "role": "owner"})
        auth.grant("new@example.org", "solar")
        self.assertEqual(self.fake.users["new@example.org"]["app_metadata"]["workspaces"],
                         ["climate", "solar"])
        auth.revoke("new@example.org", "climate")
        self.assertEqual(self.fake.users["new@example.org"]["app_metadata"]["workspaces"], ["solar"])

    def test_add_existing_user_sets_password(self):
        auth.add_user("me@example.org", "z" * 12, "climate")
        self.assertIsNotNone(auth.sign_in("me@example.org", "z" * 12))

    def test_invite(self):
        auth.invite("guest@example.org", "climate", "https://app/auth/callback")
        self.assertEqual(self.fake.sent, [("invite", "guest@example.org", "https://app/auth/callback")])
        self.assertEqual(self.fake.users["guest@example.org"]["app_metadata"]["workspaces"], ["climate"])

    def test_unknown_user_errors(self):
        with self.assertRaises(auth.AuthError):
            auth.grant("nobody@example.org", "climate")


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
class GateTests(AuthTestBase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient

        from app.web import main

        self.main = main
        self.client = TestClient(main.app, follow_redirects=False)

    def sign_in(self, email="me@example.org", password="correct horse battery"):
        return self.client.post("/login", data={"email": email, "password": password, "next": "/x"})

    def test_pages_redirect_to_login(self):
        r = self.client.get("/calendar?week=2", headers={"accept": "text/html"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/login?next=%2Fcalendar%3Fweek%3D2")

    def test_api_and_writes_get_401(self):
        self.assertEqual(self.client.get("/media/clip/1").status_code, 401)
        self.assertEqual(self.client.post("/monitor/run").status_code, 401)

    def test_public_paths(self):
        for path in ("/healthz", "/login", "/forgot", "/auth/callback", "/static/style.css"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_login_flow(self):
        r = self.sign_in(password="nope")
        self.assertIn("error=invalid", r.headers["location"])
        self.assertNotIn(auth.COOKIE_NAME, r.cookies)

        r = self.client.post("/login", data={"email": "ME@example.org", "password": "correct horse battery",
                                             "next": "https://evil.com"})
        self.assertEqual(r.headers["location"], "/")
        cookie = r.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        self.assertEqual(self.client.get("/no-such-page").status_code, 404)  # past the gate

        r = self.client.post("/logout")
        self.assertEqual(r.headers["location"], "/login?msg=signed_out")
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/no-such-page").status_code, 401)

    def test_expired_token_is_refreshed_and_cookie_reset(self):
        self.sign_in()
        session = auth.read_cookie(self.client.cookies.get(auth.COOKIE_NAME))
        self.fake.expire(session.access_token)
        auth._cache.clear()
        r = self.client.get("/no-such-page")
        self.assertEqual(r.status_code, 404)
        self.assertIn(auth.COOKIE_NAME, r.headers.get("set-cookie", ""))

    def test_account_without_workspace_access_is_refused(self):
        self.fake.add("stranger@example.org", "x" * 12, workspaces=())
        r = self.sign_in("stranger@example.org", "x" * 12)
        self.assertIn("error=noaccess", r.headers["location"])
        self.assertNotIn(auth.COOKIE_NAME, r.cookies)

    def test_secure_cookie_behind_https_proxy(self):
        r = self.client.post("/login", data={"email": "me@example.org", "password": "correct horse battery"},
                             headers={"x-forwarded-proto": "https"})
        self.assertIn("secure", r.headers["set-cookie"].lower())

    def test_throttled_after_repeated_failures(self):
        for _ in range(auth._FAIL_LIMIT):
            self.sign_in(password="nope")
        self.assertIn("error=throttled", self.sign_in().headers["location"])

    def test_supabase_outage(self):
        self.fake.down = True
        self.assertIn("error=unavailable", self.sign_in().headers["location"])

    def test_forgot_password_sends_link_to_callback(self):
        with mock.patch.dict(os.environ, {"PUBLIC_URL": "https://app.example"}):
            r = self.client.post("/forgot", data={"email": "me@example.org"})
        self.assertEqual(r.headers["location"], "/login?msg=reset_sent")
        self.assertEqual(self.fake.sent, [("recover", "me@example.org", "https://app.example/auth/callback")])

    def test_email_link_session_then_set_password(self):
        tokens = self.fake._tokens("me@example.org")
        r = self.client.post("/auth/session", data={"access_token": tokens["access_token"],
                                                    "refresh_token": tokens["refresh_token"],
                                                    "type": "recovery"})
        self.assertEqual(r.headers["location"], "/account/password?welcome=1")
        self.assertEqual(self.client.get("/account/password?welcome=1").status_code, 200)
        r = self.client.post("/account/password", data={"password": "short", "confirm": "short"})
        self.assertIn("error=short", r.headers["location"])
        r = self.client.post("/account/password", data={"password": "a brand new one", "confirm": "nope"})
        self.assertIn("error=mismatch", r.headers["location"])
        r = self.client.post("/account/password", data={"password": "a brand new one",
                                                        "confirm": "a brand new one"})
        self.assertTrue(r.headers["location"].startswith("/?msg="))
        self.assertEqual(self.fake.passwords["me@example.org"], "a brand new one")

    def test_bad_email_link(self):
        r = self.client.post("/auth/session", data={"access_token": "bogus", "refresh_token": ""})
        self.assertEqual(r.headers["location"], "/login?error=expired")

    def test_off_when_unconfigured(self):
        with mock.patch.dict(os.environ, {"DASHBOARD_AUTH": ""}):
            self.assertEqual(self.client.get("/no-such-page").status_code, 404)
            self.assertEqual(self.client.get("/login").status_code, 303)


if __name__ == "__main__":
    unittest.main()
