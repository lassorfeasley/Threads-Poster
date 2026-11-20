"""Who sees what (app/web/access.py). Run with: python -m unittest tests.test_access -v"""
from __future__ import annotations

import unittest

from app.web import access


class AccessTest(unittest.TestCase):
    def test_customer_paths(self):
        for path in ("/inbox", "/inbox/4/approve", "/inbox/4/reasons", "/analytics", "/analytics/body",
                     "/schedule", "/schedule/plan", "/schedule/timezone", "/brand", "/brand/rules/add",
                     "/style-guide", "/connections", "/appearance", "/account/password",
                     "/media/subtitled/3", "/static/style.css", "/settings", "/logout", "/view-as",
                     "/threads/connect", "/facebook/callback", "/clip-engine/rule-proposals", "/calendar"):
            self.assertTrue(access.customer_may(path, "POST" if "/" in path[1:] else "GET"), path)

    def test_admin_paths(self):
        for path in ("/", "/library", "/notifications", "/channels", "/keywords",
                     "/traits", "/auto-triage", "/video/3", "/cut/3", "/post/3", "/clip-bench",
                     "/labels", "/product-roadmap", "/inboxes", "/brandx", "/clip-engine/run"):
            self.assertFalse(access.customer_may(path), path)

    def test_admin_machinery_inside_customer_sections(self):
        for path in ("/inbox/settings", "/schedule", "/schedule/cancel", "/analytics/digest"):
            self.assertFalse(access.customer_may(path, "POST"), path)
        self.assertTrue(access.customer_may("/schedule", "GET"))

    def test_moving_a_post_on_the_calendar(self):
        self.assertTrue(access.customer_may("/post/12/pin-window", "POST"))
        for path, method in (("/post/12/pin-window", "GET"), ("/post/12", "GET"), ("/post/12/retry", "POST"),
                             ("/post/x/pin-window", "POST"), ("/post/12/pin-window/x", "POST")):
            self.assertFalse(access.customer_may(path, method), (path, method))

    def test_who_is_admin(self):
        self.assertTrue(access.real_admin(sign_in_on=False, role=None))   # the operator's own Mac
        self.assertTrue(access.real_admin(sign_in_on=True, role="admin"))
        self.assertTrue(access.real_admin(sign_in_on=True, role=" Admin "))
        for role in ("owner", "member", "", None):
            self.assertFalse(access.real_admin(sign_in_on=True, role=role), role)

    def test_looking_as_a_customer(self):
        self.assertTrue(access.sees_admin(real=True, view=None))
        self.assertFalse(access.sees_admin(real=True, view="customer"))
        self.assertFalse(access.sees_admin(real=False, view=None))
        self.assertFalse(access.sees_admin(real=False, view="admin"))


if __name__ == "__main__":
    unittest.main()
