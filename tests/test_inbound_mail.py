"""Mail forwarding (app/admin/inbound_mail): signature check, routing, the forward.
Run with: python -m unittest tests.test_inbound_mail -v"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.admin import inbound_mail, main

SECRET = "whsec_" + base64.b64encode(b"test-secret-key-0123456789").decode()


def sign(body: bytes, msg_id="msg_1", ts=None, secret=SECRET):
    ts = str(int(time.time()) if ts is None else ts)
    key = base64.b64decode(secret.removeprefix("whsec_"))
    sig = base64.b64encode(hmac.new(key, f"{msg_id}.{ts}.".encode() + body, hashlib.sha256).digest()).decode()
    return {"svix-id": msg_id, "svix-timestamp": ts, "svix-signature": f"v1,bogus v1,{sig}"}


class VerifyTest(unittest.TestCase):
    def test_good_signature(self):
        h = sign(b"{}")
        inbound_mail.verify(SECRET, h["svix-id"], h["svix-timestamp"], h["svix-signature"], b"{}")

    def test_tampered_body_or_old_timestamp_or_no_secret(self):
        h = sign(b"{}")
        with self.assertRaises(inbound_mail.BadSignature):
            inbound_mail.verify(SECRET, h["svix-id"], h["svix-timestamp"], h["svix-signature"], b"{ }")
        old = sign(b"{}", ts=int(time.time()) - 3600)
        with self.assertRaises(inbound_mail.BadSignature):
            inbound_mail.verify(SECRET, old["svix-id"], old["svix-timestamp"], old["svix-signature"], b"{}")
        with self.assertRaises(inbound_mail.BadSignature):
            inbound_mail.verify("", h["svix-id"], h["svix-timestamp"], h["svix-signature"], b"{}")


class RouteTest(unittest.TestCase):
    def test_exact_then_catch_all_one_per_target(self):
        table = inbound_mail.rules("Lassor@continuity.marketing=a@x.com, *@continuity.marketing=b@x.com")
        self.assertEqual(
            inbound_mail.route(["Lassor <LASSOR@continuity.marketing>", "hi@continuity.marketing",
                                "lassor@continuity.marketing", "x@other.com"], table),
            [("lassor@continuity.marketing", "a@x.com"), ("hi@continuity.marketing", "b@x.com")])

    def test_no_rules_nothing(self):
        self.assertEqual(inbound_mail.route(["lassor@continuity.marketing"], {}), [])


class _Resp:
    def __init__(self, data=None, content=b""):
        self._data, self.content = data, content

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


EVENT = {"type": "email.received",
         "data": {"email_id": "em_1", "from": "jo@example.com", "subject": "Hi",
                  "to": ["lassor@continuity.marketing"], "received_for": ["lassor@continuity.marketing"]}}
MAIL = {"id": "em_1", "subject": "Hi", "html": "<p>Hello</p>", "text": "Hello",
        "headers": {"from": "Jo Smith <jo@example.com>"}, "attachments": []}


class WebhookTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"RESEND_API_KEY": "re_test", "RESEND_WEBHOOK_SECRET": SECRET,
                                           "MAIL_FORWARDS": "lassor@continuity.marketing=feasley@lassor.com"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(main.app)

    def post(self, event, headers=None):
        body = json.dumps(event).encode()
        return self.client.post("/hooks/inbound-email", content=body,
                                headers=headers if headers is not None else sign(body))

    def test_forwards_with_reply_to_and_idempotency(self):
        with mock.patch.object(inbound_mail.httpx, "get", return_value=_Resp(MAIL)) as get, \
             mock.patch.object(inbound_mail.httpx, "post", return_value=_Resp({})) as post:
            r = self.post(EVENT)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["forwarded"], 1)
        self.assertIn("/emails/receiving/em_1", get.call_args.args[0])
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["to"], ["feasley@lassor.com"])
        self.assertEqual(sent["from"], 'Jo Smith via Continuity Marketing <lassor@continuity.marketing>')
        self.assertEqual(sent["reply_to"], ["Jo Smith <jo@example.com>"])
        self.assertEqual((sent["subject"], sent["html"], sent["text"]), ("Hi", "<p>Hello</p>", "Hello"))
        self.assertEqual(post.call_args.kwargs["headers"]["Idempotency-Key"], "fwd-em_1-feasley@lassor.com")

    def test_attachments_come_from_the_raw_message(self):
        eml = (b"From: jo@example.com\r\nTo: lassor@continuity.marketing\r\nSubject: Hi\r\n"
               b"MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=B\r\n\r\n"
               b"--B\r\nContent-Type: text/plain\r\n\r\nHello\r\n"
               b"--B\r\nContent-Type: application/pdf\r\nContent-Disposition: attachment; filename=\"a.pdf\"\r\n"
               b"Content-Transfer-Encoding: base64\r\n\r\n" + base64.b64encode(b"%PDF-data") + b"\r\n--B--\r\n")
        mail = {**MAIL, "attachments": [{"id": "at_1"}], "raw": {"download_url": "https://raw.example/x"}}
        with mock.patch.object(inbound_mail.httpx, "get",
                               side_effect=[_Resp(mail), _Resp(content=eml)]), \
             mock.patch.object(inbound_mail.httpx, "post", return_value=_Resp({})) as post:
            self.assertEqual(self.post(EVENT).status_code, 200)
        att = post.call_args.kwargs["json"]["attachments"]
        self.assertEqual([(a["filename"], base64.b64decode(a["content"])) for a in att], [("a.pdf", b"%PDF-data")])

    def test_unsigned_is_refused_and_sends_nothing(self):
        with mock.patch.object(inbound_mail.httpx, "post") as post:
            r = self.post(EVENT, headers={})
        self.assertEqual(r.status_code, 401)
        post.assert_not_called()

    def test_unrouted_address_is_left_alone(self):
        event = {**EVENT, "data": {**EVENT["data"], "to": ["nobody@continuity.marketing"],
                                   "received_for": ["nobody@continuity.marketing"]}}
        with mock.patch.object(inbound_mail.httpx, "get") as get:
            r = self.post(event)
        self.assertEqual(r.json()["forwarded"], 0)
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
