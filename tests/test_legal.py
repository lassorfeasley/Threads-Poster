"""Privacy policy and terms of use (app/admin/legal_store.py and the admin pages).
Run with: python -m unittest tests.test_legal -v"""
from __future__ import annotations

import datetime as dt
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.admin import legal_store, main


class RenderTest(unittest.TestCase):
    def test_blocks(self):
        out = legal_store.render("# Title\n\nOne\nline.\n\n- a\n- **b**\n\n1. x\n2. *y*")
        self.assertIn("<h2>Title</h2>", out)
        self.assertIn("<p>One line.</p>", out)
        self.assertIn("<ul><li>a</li><li><strong>b</strong></li></ul>", out)
        self.assertIn("<ol><li>x</li><li><em>y</em></li></ol>", out)

    def test_escapes_markup_and_unsafe_links(self):
        out = legal_store.render('<script>alert(1)</script> [x](javascript:alert(1)) [ok](https://a.b/c?d=1&e=2)')
        self.assertNotIn("<script>", out)
        self.assertNotIn('href="javascript', out)
        self.assertIn('<a href="https://a.b/c?d=1&amp;e=2">ok</a>', out)

    def test_mailto_and_relative_links(self):
        out = legal_store.render("[mail](mailto:hi@x.com) and [p](/privacy)")
        self.assertIn('href="mailto:hi@x.com"', out)
        self.assertIn('href="/privacy"', out)

    def test_starters_render(self):
        for kind in legal_store.KINDS:
            text = legal_store._starter(kind)
            self.assertTrue(text.strip(), kind)
            self.assertIn("<h2>", legal_store.render(text))


class CurrentTest(unittest.TestCase):
    def test_newest_effective_published_wins(self):
        today = dt.date(2026, 11, 20)
        rows = [
            {"version": 4, "status": "draft", "effective_date": None},
            {"version": 3, "status": "published", "effective_date": "2026-12-01"},   # scheduled
            {"version": 2, "status": "published", "effective_date": "2026-11-20"},
            {"version": 1, "status": "published", "effective_date": "2026-01-01"},
        ]
        self.assertEqual(legal_store.current(rows, today)["version"], 2)
        self.assertEqual(legal_store.current(rows, dt.date(2026, 12, 1))["version"], 3)
        self.assertIsNone(legal_store.current(rows[:1], today))


class PublishRulesTest(unittest.TestCase):
    def draft(self, **over):
        row = {"kind": "privacy", "version": 1, "status": "draft", "summary": "First version.",
               "body_md": "Real text."}
        row.update(over)
        return row

    def publish(self, row):
        with mock.patch.object(legal_store, "get", return_value=row), \
             mock.patch.object(legal_store.httpx, "patch") as patch:
            patch.return_value = mock.Mock(status_code=204)
            legal_store.publish("privacy", 1, effective=dt.date.today(), by="me")
            return patch

    def test_publishes_a_complete_draft(self):
        patch = self.publish(self.draft())
        body = patch.call_args.kwargs["json"]
        self.assertEqual(body["status"], "published")
        self.assertEqual(body["published_by"], "me")

    def test_refuses_placeholders_empty_summary_and_published(self):
        for row in (self.draft(body_md="We are [LEGAL ENTITY NAME]."),
                    self.draft(summary=" "),
                    self.draft(body_md=""),
                    self.draft(status="published")):
            with self.assertRaises(legal_store.LegalError):
                self.publish(row)

    def test_lowercase_brackets_are_not_placeholders(self):
        self.publish(self.draft(body_md="See [our terms](/terms) and [hello@x.com](mailto:hello@x.com)."))


class PagesTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"ADMIN_LOCAL": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(main.app, client=("127.0.0.1", 5000))

    def test_overview_and_editor(self):
        draft = {"id": 2, "kind": "privacy", "version": 2, "status": "draft", "title": "Privacy policy",
                 "summary": "", "body_md": "# Hi", "body_html": "<h2>Hi</h2>", "effective_date": None,
                 "created_at": "2026-11-20T10:00:00+00:00", "updated_at": "2026-11-20T10:00:00+00:00",
                 "published_at": None, "published_by": None}
        live = {**draft, "version": 1, "status": "published", "summary": "First version.",
                "effective_date": "2026-11-01", "published_at": "2026-11-01T10:00:00+00:00"}
        with mock.patch.object(legal_store, "versions", return_value=[draft, live]), \
             mock.patch.object(legal_store, "get", side_effect=lambda k, v: draft if v == 2 else live):
            page = self.client.get("/legal")
            self.assertEqual(page.status_code, 200)
            self.assertIn("Edit draft v2", page.text)
            self.assertIn("In force", page.text)
            editor = self.client.get("/legal/privacy/2")
            self.assertIn("Publish version 2", editor.text)
            frozen = self.client.get("/legal/privacy/1")
            self.assertIn("can't be changed", frozen.text)
            self.assertNotIn("Publish version", frozen.text)

    def test_past_effective_date_is_refused(self):
        with mock.patch.object(legal_store, "publish") as publish:
            r = self.client.post("/legal/privacy/2/publish", data={"effective_date": "2020-01-01"},
                                 follow_redirects=False)
        publish.assert_not_called()
        self.assertIn("error=", r.headers["location"])


if __name__ == "__main__":
    unittest.main()
