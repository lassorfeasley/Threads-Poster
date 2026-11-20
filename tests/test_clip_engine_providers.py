"""Tests for the clip engine's model providers: the Gemini REST client and the
streamed, schema-constrained Claude calls.

No network. Gemini's HTTP goes through a scripted stand-in for
``gemini._send``. Claude runs the real ``anthropic`` SDK against an in-memory
httpx transport (httpx2 from SDK 2.x), so request serialization and the SDK's
own stream accumulator are exercised for real. Spend goes to a temp ledger.

Run with:  python -m unittest tests.test_clip_engine_providers -v
"""
from __future__ import annotations

import base64
import copy
import datetime as dt
import email.utils
import json
import os
import tempfile
import threading
import unittest
from http.client import RemoteDisconnected
from pathlib import Path
from unittest import mock

import anthropic
import requests
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from app import config, llm, spend
from app.clip_engine import claude, gemini, util


def _sdk_http():
    """The HTTP package this anthropic SDK speaks: httpx up to 0.x, httpx2 (the
    same API) from 2.x. The fake transport must come from the same package, or
    the SDK refuses the client."""
    try:
        import httpx2
    except ImportError:
        import httpx

        return httpx
    probe = httpx2.Client()
    try:
        anthropic.Anthropic(api_key="probe", http_client=probe)
    except TypeError:
        import httpx

        return httpx
    finally:
        probe.close()
    return httpx2


httpx = _sdk_http()

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer", "minimum": 0}, "note": {"type": ["string", "null"]}},
    "required": ["a", "note"],
}


def keys_named(obj, names: set) -> list:
    """Every dict key in ``obj`` (recursively) that is in ``names``."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names:
                found.append(key)
            found += keys_named(value, names)
    elif isinstance(obj, list):
        for value in obj:
            found += keys_named(value, names)
    return found



# Every request carries the American English line after its own system prompt
# (app/config.py).
from app.config import AMERICAN_ENGLISH, with_american_english  # noqa: E402

SYS_US = with_american_english("SYS")

class ProviderCase(unittest.TestCase):
    """Temp engine dir + ledger, the code's own prices, provider state reset."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.engine_dir = self.dir / "engine"
        self.ledger = self.dir / "llm_spend.json"
        patches = [
            mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.engine_dir),
                                         "GEMINI_API_KEY": "test-key",
                                         "ANTHROPIC_API_KEY": "test-key"}),
            mock.patch.object(spend, "_LEDGER", self.ledger),
            mock.patch.object(spend, "load_settings", return_value=config.Settings(raw={})),
            mock.patch.object(gemini, "_breaker_until", 0.0),
            mock.patch.object(gemini, "_breaker_reason", ""),
            mock.patch.object(gemini, "_breaker_why", gemini._DAILY_QUOTA),
            mock.patch.object(gemini, "_breaker_status", 429),
            mock.patch.object(gemini, "_confirmed", {}),
            mock.patch.object(gemini, "_upload_locks", {}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        spend.set_bucket(None)
        self.addCleanup(spend.set_bucket, None)

    def ledger_day(self) -> dict:
        return json.loads(self.ledger.read_text())[spend._today()]


# =========================================================================================
# Gemini
# =========================================================================================

def http(status: int = 200, body=None, *, headers=None, text: str | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = (json.dumps(body) if body is not None else (text or "")).encode()
    r.headers.update(headers or {})
    r.encoding = "utf-8"
    return r


def api_error(status: int, message: str, *, name: str = "INVALID_ARGUMENT", details=None,
              headers=None) -> requests.Response:
    body = {"error": {"code": status, "message": message, "status": name}}
    if details:
        body["error"]["details"] = details
    return http(status, body, headers=headers)


USAGE = {"promptTokenCount": 1000, "candidatesTokenCount": 200, "thoughtsTokenCount": 300,
         "totalTokenCount": 1500}
COST = 1000 * 1.50 / 1e6 + 500 * 7.50 / 1e6   # gemini-3.8-flash


def answer_body(data=None, *, text: str | None = None, parts=None, finish: str = "STOP",
                usage=USAGE) -> dict:
    if parts is None:
        parts = [{"text": text if text is not None else json.dumps(data)}]
    body = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
            "modelVersion": "gemini-3.8-flash"}
    if usage is not None:
        body["usageMetadata"] = usage
    return body


def answer(data=None, **kw) -> requests.Response:
    return http(200, answer_body(data, **kw))


class GeminiHTTP:
    """Stands in for ``gemini._send``: replays scripted answers (responses or
    exceptions) in order and records every request."""

    def __init__(self):
        self.answers: list = []
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def add(self, *answers) -> None:
        self.answers.extend(answers)

    def __call__(self, method, url, **kwargs):
        call = {"method": method, "url": url, **kwargs}
        if hasattr(kwargs.get("data"), "read"):
            call["sent_bytes"] = kwargs["data"].read()
        with self.lock:
            self.calls.append(call)
            if not self.answers:
                raise AssertionError(f"unexpected request: {method} {url}")
            reply = self.answers.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class GeminiCase(ProviderCase):
    def setUp(self):
        super().setUp()
        self.http = GeminiHTTP()
        self.sleeps: list[float] = []
        for p in (mock.patch.object(gemini, "_send", self.http),
                  mock.patch.object(gemini, "_sleep", self.sleeps.append)):
            p.start()
            self.addCleanup(p.stop)
        self.video = self.dir / "clip.mp4"
        self.video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 1000)

    def generate(self, model: str = "gemini-3.8-flash", **kw):
        args = dict(system="SYS", parts=[gemini.text_part("payload")], schema=SCHEMA, label="test")
        args.update(kw)
        return gemini.generate_json(model, **args)

    def file_resource(self, name: str = "files/abc123", state: str = "ACTIVE") -> dict:
        return {"name": name, "uri": f"{gemini.BASE}/v1beta/{name}", "mimeType": "video/mp4",
                "state": state}

    def script_upload(self, name: str = "files/abc123", states=("ACTIVE",),
                      url: str = "https://upload.test/session-1") -> dict:
        """Start -> bytes (first state) -> one GET per further state."""
        first, *rest = states
        self.http.add(http(200, {}, headers={"X-Goog-Upload-URL": url}),
                      http(200, {"file": self.file_resource(name, first)}),
                      *[http(200, self.file_resource(name, s)) for s in rest])
        return self.file_resource(name)

    def cache(self) -> dict:
        return util.read_json(self.engine_dir / "gemini_files.json", {})


class GeminiRequestTest(GeminiCase):
    def test_request_body_headers_and_url(self):
        video = {"fileData": {"mimeType": "video/mp4", "fileUri": f"{gemini.BASE}/v1beta/files/v1"},
                 "videoMetadata": {"fps": 1.0}}
        self.http.add(answer({"a": 1, "note": None}))
        data, _ = gemini.generate_json(
            "gemini-3.8-flash", system="SYS", parts=[video, gemini.text_part("payload")],
            schema=SCHEMA, thinking="HIGH", media_resolution="MEDIA_RESOLUTION_LOW",
            max_output_tokens=1234, timeout_s=120, label="draft")
        self.assertEqual(data, {"a": 1, "note": None})
        (call,) = self.http.calls
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"],
                         "https://generativelanguage.googleapis.com/v1beta/models/"
                         "gemini-3.8-flash:generateContent")
        self.assertEqual(call["headers"], {"x-goog-api-key": "test-key"})
        self.assertEqual(call["timeout"], (15.0, 120.0))
        self.assertEqual(call["json"], {
            "systemInstruction": {"parts": [{"text": SYS_US}]},
            "contents": [{"role": "user", "parts": [video, {"text": "payload"}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": SCHEMA,
                "maxOutputTokens": 1234,
                "mediaResolution": "MEDIA_RESOLUTION_LOW",
                "thinkingConfig": {"thinkingLevel": "HIGH"},
            },
        })
        self.assertEqual(keys_named(call["json"], {"temperature", "topP", "topK"}), [])
        json.dumps(call["json"])  # what requests will send must serialize

    def test_optional_fields_are_left_out(self):
        self.http.add(answer({"a": 1}))
        self.generate(system="", thinking=None, media_resolution=None)
        body = self.http.calls[0]["json"]
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": AMERICAN_ENGLISH}]})
        self.assertEqual(sorted(body["generationConfig"]),
                         ["maxOutputTokens", "responseJsonSchema", "responseMimeType"])

    def test_usage_is_returned_and_recorded_in_a_bucket(self):
        cases = [
            # (bucket argument, process default, where it lands)
            ("engine", None, "engine"),
            ("bench", None, "bench"),
            (None, "bench", "bench"),
            (None, None, "engine"),   # provider spend never lands in the shared budget
        ]
        for bucket, default, expected in cases:
            with self.subTest(bucket=bucket, default=default):
                self.ledger.unlink(missing_ok=True)
                spend.set_bucket(default)
                self.http.add(answer({"a": 1}))
                _, usage = self.generate(bucket=bucket)
                self.assertEqual(usage, {"model": "gemini-3.8-flash", "input_tokens": 1000,
                                         "output_tokens": 500, "cost_usd": round(COST, 6)})
                day = self.ledger_day()
                self.assertEqual(day["calls"], 0)
                self.assertEqual(list(day["buckets"]), [expected])
                self.assertAlmostEqual(day["buckets"][expected]["spend_usd"], COST)

    def test_model_ids(self):
        self.http.add(answer({"a": 1}))
        self.generate(model="models/gemini-3.8-flash")
        self.assertTrue(self.http.calls[0]["url"].endswith("/v1beta/models/gemini-3.8-flash:generateContent"))
        for bad in ("", "   ", None, "../files", "gemini 3", "a/b:c"):
            with self.subTest(model=bad), self.assertRaises(gemini.GeminiError):
                self.generate(model=bad)
        self.assertEqual(len(self.http.calls), 1)

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "  "}):
            self.assertFalse(gemini.is_configured())
            with self.assertRaises(gemini.GeminiError) as ctx:
                self.generate()
            self.assertEqual(str(ctx.exception), "GEMINI_API_KEY is not set (see .env.example)")
            with self.assertRaises(gemini.GeminiError):
                gemini.upload_file(self.video)
        self.assertEqual(self.http.calls, [])
        self.assertTrue(gemini.is_configured())

    def test_text_part(self):
        self.assertEqual(gemini.text_part("hi"), {"text": "hi"})
        self.assertEqual(gemini.text_part(5), {"text": "5"})


class GeminiRetryTest(GeminiCase):
    def test_transient_failures_are_retried(self):
        rq = requests.exceptions
        cases = [
            ("503", [api_error(503, "unavailable", name="UNAVAILABLE")], 1),
            ("429 per minute", [api_error(429, "Quota exceeded for requests per minute",
                                          name="RESOURCE_EXHAUSTED")], 1),
            ("408 500 502 504", [api_error(s, "x") for s in (408, 500, 502, 504)], 4),
            ("dropped connection", [rq.ConnectionError("reset")], 1),
            ("connect timeouts", [rq.ConnectTimeout("slow"), rq.ConnectTimeout("slow")], 2),
            ("broken chunked body", [rq.ChunkedEncodingError("cut")], 1),
        ]
        for name, failures, retries in cases:
            with self.subTest(name):
                self.http.calls.clear()
                self.sleeps.clear()
                self.http.add(*failures, answer({"a": 1}))
                data, _ = self.generate()
                self.assertEqual(data, {"a": 1})
                self.assertEqual(len(self.http.calls), retries + 1)
                self.assertEqual(len(self.sleeps), retries)

    def test_backoff_doubles_with_jitter(self):
        self.http.add(*[api_error(503, "x") for _ in range(4)], answer({"a": 1}))
        self.generate()
        self.assertEqual(len(self.sleeps), 4)
        for n, slept in enumerate(self.sleeps):
            self.assertGreaterEqual(slept, 2 ** n)
            self.assertLess(slept, 2 ** n + 1)
        self.assertEqual(gemini._backoff(10), 60.0)

    def test_server_requested_delays(self):
        soon = email.utils.format_datetime(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30),
                                           usegmt=True)
        retry_info = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]
        cases = [
            ("Retry-After seconds", {"Retry-After": "7"}, None, (7.0, 7.0)),
            ("Retry-After capped", {"Retry-After": "3600"}, None, (60.0, 60.0)),
            ("Retry-After date", {"Retry-After": soon}, None, (25.0, 30.0)),
            ("Retry-After junk -> backoff", {"Retry-After": "soon"}, None, (1.0, 2.0)),
            ("RetryInfo in the body", {}, retry_info, (12.0, 12.0)),
        ]
        for name, headers, details, (low, high) in cases:
            with self.subTest(name):
                self.sleeps.clear()
                self.http.add(api_error(429, "Quota exceeded for requests per minute",
                                        name="RESOURCE_EXHAUSTED", headers=headers, details=details),
                              answer({"a": 1}))
                self.generate()
                (slept,) = self.sleeps
                self.assertGreaterEqual(slept, low)
                self.assertLessEqual(slept, high)

    def test_gives_up_after_four_retries(self):
        self.http.add(*[api_error(500, "internal", name="INTERNAL") for _ in range(5)])
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.generate()
        self.assertEqual(ctx.exception.status, 500)
        self.assertIn("internal", str(ctx.exception))
        self.assertIsNone(ctx.exception.usage)
        self.assertEqual((len(self.http.calls), len(self.sleeps)), (5, 4))

    def test_connection_failures_give_up_too(self):
        self.http.add(*[requests.exceptions.ConnectionError("down") for _ in range(5)])
        with self.assertRaises(gemini.GeminiError):
            self.generate()
        self.assertEqual(len(self.http.calls), 5)

    def test_client_errors_are_not_retried(self):
        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                self.http.calls.clear()
                self.http.add(api_error(status, "nope"))
                with self.assertRaises(gemini.GeminiError) as ctx:
                    self.generate()
                self.assertEqual(ctx.exception.status, status)
                self.assertEqual(len(self.http.calls), 1)
        self.assertEqual(self.sleeps, [])

    def test_read_timeout_is_retried_only_once(self):
        timeout = requests.exceptions.ReadTimeout
        self.http.add(timeout("slow"), answer({"a": 1}))
        self.assertEqual(self.generate()[0], {"a": 1})
        self.assertEqual(len(self.http.calls), 2)

        self.http.calls.clear()
        self.http.add(timeout("slow"), timeout("slow"), answer({"a": 1}))
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.generate(timeout_s=900)
        self.assertIn("900s", str(ctx.exception))
        self.assertEqual(len(self.http.calls), 2)
        self.http.answers.clear()

    def test_read_timeouts_share_the_retry_allowance(self):
        rq = requests.exceptions
        self.http.add(rq.ReadTimeout("slow"), api_error(503, "x"), api_error(503, "x"),
                      rq.ConnectionError("x"), rq.ReadTimeout("slow"))
        with self.assertRaises(gemini.GeminiError):
            self.generate()
        self.assertEqual(len(self.http.calls), 5)


class GeminiSlowFailureTest(GeminiCase):
    """A failure that arrives after minutes of waiting costs as much as a read
    timeout, so it gets the same single retry, and no retry starts once a call
    has run for twice its read timeout. Before, a 504 that took 10 minutes was
    tried five times: 50 minutes of the engine's only worker."""

    def setUp(self):
        super().setUp()
        self.wait = 0.0             # how long each scripted answer takes to arrive
        self.now = [1000.0]

        def send(method, url, **kwargs):
            self.now[0] += self.wait
            return self.http(method, url, **kwargs)

        def sleep(seconds):
            self.sleeps.append(seconds)
            self.now[0] += seconds

        for p in (mock.patch.object(gemini, "_send", send),
                  mock.patch.object(gemini, "_sleep", sleep),
                  mock.patch.object(gemini, "_now", lambda: self.now[0])):
            p.start()
            self.addCleanup(p.stop)

    def test_slow_failures_are_retried_only_once(self):
        rq = requests.exceptions
        cases = [
            # (name, a fresh failure, the status the error carries)
            ("504 DEADLINE_EXCEEDED", lambda: api_error(
                504, "Deadline expired before operation could complete.", name="DEADLINE_EXCEEDED"), 504),
            ("500 INTERNAL", lambda: api_error(500, "An internal error has occurred.",
                                               name="INTERNAL"), 500),
            ("503 UNAVAILABLE", lambda: api_error(503, "The model is overloaded.",
                                                  name="UNAVAILABLE"), 503),
            ("connection dropped mid-wait", lambda: rq.ConnectionError(ProtocolError(
                "Connection aborted.",
                RemoteDisconnected("Remote end closed connection without response"))), None),
            ("chunked body cut off", lambda: rq.ChunkedEncodingError("cut"), None),
        ]
        self.wait = 600.0
        for name, failure, status in cases:
            with self.subTest(name):
                self.http.calls.clear()
                self.http.answers.clear()
                self.sleeps.clear()
                self.http.add(*[failure() for _ in range(5)], answer({"a": 1}))
                started = self.now[0]
                with self.assertLogs("clip_engine.gemini", "WARNING"), \
                        self.assertRaises(gemini.GeminiError) as ctx:
                    self.generate(timeout_s=900)
                self.assertEqual(ctx.exception.status, status)
                self.assertEqual((len(self.http.calls), len(self.sleeps)), (2, 1))
                self.assertLess(self.now[0] - started, 21 * 60)

    def test_a_body_that_stalls_counts_as_a_read_timeout(self):
        """requests reports a read timeout during the body as a ConnectionError
        around urllib3's ReadTimeoutError; it still gets one retry, not four."""
        def stalled():
            return requests.exceptions.ConnectionError(
                ReadTimeoutError(None, None, "Read timed out. (read timeout=900)"))

        self.http.add(stalled(), stalled(), answer({"a": 1}))
        with self.assertLogs("clip_engine.gemini", "WARNING"), \
                self.assertRaises(gemini.GeminiError) as ctx:
            self.generate(timeout_s=900)
        self.assertIn("no answer within 900s", str(ctx.exception))
        self.assertEqual(len(self.http.calls), 2)

    def test_read_timeouts_and_slow_failures_share_one_retry(self):
        self.wait = 900.0
        self.http.add(requests.exceptions.ReadTimeout("slow"), api_error(500, "internal"),
                      answer({"a": 1}))
        with self.assertLogs("clip_engine.gemini", "WARNING"), \
                self.assertRaises(gemini.GeminiError) as ctx:
            self.generate(timeout_s=900)
        self.assertEqual(ctx.exception.status, 500)
        self.assertEqual(len(self.http.calls), 2)

    def test_a_quick_504_is_retried_once_too(self):
        """504 means the server's deadline passed; the same request won't fit
        it the next time either."""
        self.http.add(*[api_error(504, "Deadline expired", name="DEADLINE_EXCEEDED")
                        for _ in range(2)], answer({"a": 1}))
        with self.assertLogs("clip_engine.gemini", "WARNING"), \
                self.assertRaises(gemini.GeminiError) as ctx:
            self.generate()
        self.assertEqual(ctx.exception.status, 504)
        self.assertEqual(len(self.http.calls), 2)

    def test_quick_failures_keep_their_four_retries(self):
        def per_minute():
            return api_error(429, "Quota exceeded for requests per minute",
                             name="RESOURCE_EXHAUSTED", headers={"Retry-After": "60"})

        cases = [
            # (name, timeout_s, seconds each failure takes, failures)
            ("per-minute 429s on a short call", 30, 1.0, [per_minute() for _ in range(4)]),
            ("503s just under a minute", 900, 59.0,
             [api_error(503, "overloaded", name="UNAVAILABLE") for _ in range(4)]),
            ("dropped connections", 900, 59.0,
             [requests.exceptions.ConnectionError("reset") for _ in range(4)]),
        ]
        for name, timeout_s, wait, failures in cases:
            with self.subTest(name):
                self.http.calls.clear()
                self.sleeps.clear()
                self.wait = wait
                self.http.add(*failures, answer({"a": 1}))
                with self.assertLogs("clip_engine.gemini", "WARNING"):
                    data, _ = self.generate(timeout_s=timeout_s)
                self.assertEqual(data, {"a": 1})
                self.assertEqual((len(self.http.calls), len(self.sleeps)), (5, 4))

    def test_no_retry_starts_after_twice_the_read_timeout(self):
        # 50 s is not slow by itself, but with 60 s between tries a fourth
        # would start 330 s in, past twice the 150 s read timeout.
        self.wait = 50.0
        self.http.add(*[api_error(503, "overloaded", name="UNAVAILABLE",
                                  headers={"Retry-After": "60"}) for _ in range(5)],
                      answer({"a": 1}))
        with self.assertLogs("clip_engine.gemini", "WARNING"), \
                self.assertRaises(gemini.GeminiError) as ctx:
            self.generate(timeout_s=150)
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual((len(self.http.calls), self.sleeps), (3, [60.0, 60.0]))

        # An answer that trickled in past the whole window (no single read hit
        # the read timeout) and then broke off isn't tried again at all.
        self.http.calls.clear()
        self.http.answers.clear()
        self.wait = 400.0
        self.http.add(*[requests.exceptions.ConnectionError("reset") for _ in range(5)])
        with self.assertLogs("clip_engine.gemini", "WARNING"), \
                self.assertRaises(gemini.GeminiError) as ctx:
            self.generate(timeout_s=60)
        self.assertIn("could not reach Gemini", str(ctx.exception))
        self.assertEqual(len(self.http.calls), 1)


class GeminiQuotaTest(GeminiCase):
    def daily(self, message="Quota exceeded for quota metric 'Generate Content requests' and limit "
                            "'GenerateRequestsPerDayPerProjectPerModel'"):
        return api_error(429, message, name="RESOURCE_EXHAUSTED", details=[{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}])

    def test_a_daily_quota_opens_the_breaker_for_30_minutes(self):
        clock = [1000.0]
        with mock.patch.object(gemini, "_now", lambda: clock[0]):
            self.http.add(self.daily())
            with self.assertLogs("clip_engine.gemini", "WARNING"), \
                    self.assertRaises(gemini.GeminiQuota) as ctx:
                self.generate()
            self.assertEqual(ctx.exception.status, 429)
            self.assertEqual((len(self.http.calls), self.sleeps), (1, []))   # not retried
            self.assertTrue(gemini.breaker_open())
            # Every call now fails fast, without touching the network.
            with self.assertRaises(gemini.GeminiQuota):
                self.generate()
            with self.assertRaises(gemini.GeminiQuota):
                gemini.upload_file(self.video)
            with self.assertRaises(gemini.GeminiQuota):
                gemini.video_parts([{"path": self.video, "upload": True}])
            self.assertEqual(len(self.http.calls), 1)
            clock[0] += 29 * 60
            self.assertTrue(gemini.breaker_open())
            clock[0] += 2 * 60
            self.assertFalse(gemini.breaker_open())
            self.http.add(answer({"a": 1}))
            self.assertEqual(self.generate()[0], {"a": 1})

    def test_depleted_credits_open_the_breaker_with_a_clear_message(self):
        """A 402 (seen live on generateContent AND the upload start) is never
        transient: no retry, the breaker opens, and the message leads with
        what to do, because it reaches the job status and the page."""
        def depleted():
            return api_error(402, "Your prepayment credits are depleted. Please go to AI Studio "
                                  "at https://ai.studio/projects to manage your project and "
                                  "billing.", name="RESOURCE_EXHAUSTED")

        def started():
            return http(200, {}, headers={"X-Goog-Upload-URL": "https://upload.test/s1"})

        cases = [
            # (where, call, scripted answers)
            ("generate", self.generate, [depleted()]),
            ("upload start", lambda: gemini.upload_file(self.video), [depleted()]),
            ("upload bytes", lambda: gemini.upload_file(self.video), [started(), depleted()]),
            ("upload part", lambda: gemini.video_parts([{"path": self.video, "upload": True}]),
             [depleted()]),
        ]
        for where, call, answers in cases:
            with self.subTest(where=where), mock.patch.object(gemini, "_breaker_until", 0.0):
                self.http.calls.clear()
                self.sleeps.clear()
                self.http.add(*answers)
                with self.assertLogs("clip_engine.gemini", "WARNING"), \
                        self.assertRaises(gemini.GeminiQuota) as ctx:
                    call()
                message = str(ctx.exception)
                self.assertEqual(ctx.exception.status, 402)
                self.assertTrue(message.startswith("Gemini prepaid credits are depleted"), message)
                self.assertIn("https://ai.studio/projects", message)
                # Not retried: one request per scripted answer, no backoff.
                self.assertEqual((len(self.http.calls), self.sleeps), (len(answers), []))
                self.assertTrue(gemini.breaker_open())
                # Everything after fails fast with the same explanation.
                with self.assertRaises(gemini.GeminiQuota) as again:
                    self.generate()
                self.assertEqual(again.exception.status, 402)
                self.assertTrue(str(again.exception).startswith(
                    "Gemini prepaid credits are depleted"), str(again.exception))
                self.assertEqual(len(self.http.calls), len(answers))

    @staticmethod
    def depleted():
        return api_error(402, "Your prepayment credits are depleted. Please go to AI Studio at "
                              "https://ai.studio/projects to manage your project and billing.",
                         name="RESOURCE_EXHAUSTED")

    def test_the_breaker_reason_says_when_gemini_is_tried_again(self):
        """breaker_reason() is the fallback note a skipped run shows: it says
        when Gemini will be tried again and, for credit (which the operator can
        add in a minute), how to try it sooner."""
        clock = [1000.0]
        with mock.patch.object(gemini, "_now", lambda: clock[0]):
            self.assertEqual(gemini.breaker_reason(), "")
            self.http.add(self.depleted())
            with self.assertLogs("clip_engine.gemini", "WARNING"), \
                    self.assertRaises(gemini.GeminiQuota):
                self.generate()
            reason = gemini.breaker_reason()
            self.assertTrue(reason.startswith(gemini._CREDITS_DEPLETED), reason)
            self.assertIn("in 30 min", reason)
            self.assertIn("at once when you press Suggest", reason)
            clock[0] += 14 * 60 + 30
            self.assertIn("in 16 min", gemini.breaker_reason())
            clock[0] = 1000.0 + 30 * 60 - 1
            self.assertIn("in 1 min", gemini.breaker_reason())
            clock[0] += 1
            self.assertEqual(gemini.breaker_reason(), "")
            # Pressing Suggest won't bring back a day's quota: no such hint.
            self.http.add(self.daily())
            with self.assertLogs("clip_engine.gemini", "WARNING"), \
                    self.assertRaises(gemini.GeminiQuota):
                self.generate()
            reason = gemini.breaker_reason()
            self.assertTrue(reason.startswith(gemini._DAILY_QUOTA), reason)
            self.assertIn("in 30 min", reason)
            self.assertNotIn("Suggest", reason)

    def test_reset_breaker_lets_the_next_call_through(self):
        """For a run the operator starts after adding credit: closing the
        breaker early, and only when the named cause opened it."""
        clock = [1000.0]
        with mock.patch.object(gemini, "_now", lambda: clock[0]):
            self.assertFalse(gemini.reset_breaker())             # nothing to close
            self.http.add(self.depleted())
            with self.assertLogs("clip_engine.gemini", "WARNING"), \
                    self.assertRaises(gemini.GeminiQuota):
                self.generate()
            self.assertFalse(gemini.reset_breaker(status=429))   # opened by a 402, not a quota
            self.assertTrue(gemini.breaker_open())
            self.assertTrue(gemini.reset_breaker(status=402))
            self.assertFalse(gemini.breaker_open())
            self.assertEqual(gemini.breaker_reason(), "")
            self.http.add(answer({"a": 1}))
            self.assertEqual(self.generate()[0], {"a": 1})        # back on the network
            self.assertEqual(len(self.http.calls), 2)
            # Still no credit: the next call opens it again.
            self.http.add(self.depleted())
            with self.assertLogs("clip_engine.gemini", "WARNING"), \
                    self.assertRaises(gemini.GeminiQuota):
                self.generate()
            self.assertTrue(gemini.breaker_open())
            self.assertTrue(gemini.reset_breaker())              # any cause, when none is named

    def test_which_429s_count_as_daily(self):
        cases = [
            ("limit 'GenerateRequestsPerDayPerProjectPerModel'", True),
            ("You exceeded 1000 requests per day for this model", True),
            ("quota RequestsPerDay exhausted", True),
            ("Quota exceeded for requests per minute", False),
            ("Resource has been exhausted (e.g. check quota).", False),
        ]
        for message, daily in cases:
            with self.subTest(message=message), mock.patch.object(gemini, "_breaker_until", 0.0):
                self.http.calls.clear()
                if daily:
                    self.http.add(api_error(429, message, name="RESOURCE_EXHAUSTED"))
                    with self.assertLogs("clip_engine.gemini", "WARNING"), \
                            self.assertRaises(gemini.GeminiQuota):
                        self.generate()
                    self.assertTrue(gemini.breaker_open())
                else:
                    self.http.add(api_error(429, message, name="RESOURCE_EXHAUSTED"), answer({"a": 1}))
                    self.assertEqual(self.generate()[0], {"a": 1})
                    self.assertFalse(gemini.breaker_open())


class GeminiDowngradeTest(GeminiCase):
    PARTS = [{"inlineData": {"mimeType": "video/mp4", "data": "AAAA"}}, {"text": "payload"}]

    def first_body(self) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": SYS_US}]},
            "contents": [{"role": "user", "parts": self.PARTS}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseJsonSchema": SCHEMA, "maxOutputTokens": 32768,
                                 "mediaResolution": "MEDIA_RESOLUTION_LOW",
                                 "thinkingConfig": {"thinkingLevel": "LOW"}},
        }

    def run_downgrade(self, *errors) -> list[dict]:
        self.http.calls.clear()
        self.http.add(*errors, answer({"a": 1}))
        with self.assertLogs("clip_engine.gemini", "WARNING"):
            data, _ = self.generate(parts=self.PARTS)
        self.assertEqual(data, {"a": 1})
        bodies = [c["json"] for c in self.http.calls]
        self.assertEqual(bodies[0], self.first_body())
        return bodies

    def test_each_downgrade(self):
        def system_merged(b):
            b.pop("systemInstruction")
            b["contents"][0]["parts"] = [{"text": SYS_US}] + self.PARTS

        def legacy_schema(b):
            gc = b["generationConfig"]
            gc.pop("responseJsonSchema")
            gc["responseSchema"] = {"type": "OBJECT", "properties": {
                "a": {"type": "INTEGER", "minimum": 0}, "note": {"type": "STRING", "nullable": True}},
                "required": ["a", "note"]}

        cases = [
            ('Invalid JSON payload received. Unknown name "systemInstruction": Cannot find field.',
             system_merged),
            ("Thinking level is not supported for this model.",
             lambda b: b["generationConfig"].pop("thinkingConfig")),
            ('Invalid JSON payload received. Unknown name "mediaResolution" at \'generation_config\'',
             lambda b: b["generationConfig"].pop("mediaResolution")),
            ('Invalid JSON payload received. Unknown name "responseJsonSchema" at \'generation_config\'',
             legacy_schema),
        ]
        for message, change in cases:
            with self.subTest(message=message):
                bodies = self.run_downgrade(api_error(400, message))
                self.assertEqual(len(bodies), 2)
                expected = copy.deepcopy(self.first_body())
                change(expected)
                self.assertEqual(bodies[1], expected)

    def test_downgrades_accumulate_and_one_error_can_name_several(self):
        bodies = self.run_downgrade(
            api_error(400, "mediaResolution and thinkingConfig are not supported"),
            api_error(400, 'Unknown name "systemInstruction"'))
        self.assertEqual(len(bodies), 3)
        self.assertNotIn("mediaResolution", bodies[1]["generationConfig"])
        self.assertNotIn("thinkingConfig", bodies[1]["generationConfig"])
        self.assertIn("systemInstruction", bodies[1])
        self.assertNotIn("systemInstruction", bodies[2])
        self.assertNotIn("thinkingConfig", bodies[2]["generationConfig"])

    def test_each_downgrade_is_tried_only_once(self):
        self.http.add(api_error(400, "thinking is not supported"),
                      api_error(400, "thinking is not supported"))
        with self.assertLogs("clip_engine.gemini", "WARNING"), \
                self.assertRaises(gemini.GeminiError) as ctx:
            self.generate()
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(self.http.calls), 2)

    def test_a_400_naming_nothing_droppable_is_final(self):
        cases = [
            ("unrelated", "Request contains an invalid argument.", {}),
            ("thinking already off", "thinking is not supported", {"thinking": None}),
            ("no media resolution sent", "mediaResolution unsupported", {"media_resolution": None}),
        ]
        for name, message, kw in cases:
            with self.subTest(name):
                self.http.calls.clear()
                self.http.add(api_error(400, message))
                with self.assertRaises(gemini.GeminiError):
                    self.generate(**kw)
                self.assertEqual(len(self.http.calls), 1)

    def test_openapi_schema_for_the_legacy_field(self):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "additionalProperties": False,
            "properties": {
                "beats": {"type": "array", "minItems": 1, "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"start": {"type": "number"},
                                   "face": {"type": ["string", "null"], "enum": ["none", "close", None]},
                                   "kind": {"const": "visual"},
                                   "who": {"oneOf": [{"type": "string"}, {"type": "null"}]}},
                    "required": ["start", "face", "kind", "who"]}},
            },
            "required": ["beats"],
        }
        self.assertEqual(gemini._openapi_schema(schema), {
            "type": "OBJECT",
            "properties": {
                "beats": {"type": "ARRAY", "minItems": 1, "items": {
                    "type": "OBJECT",
                    "properties": {"start": {"type": "NUMBER"},
                                   "face": {"type": "STRING", "enum": ["none", "close"], "nullable": True},
                                   "kind": {"enum": ["visual"]},
                                   "who": {"anyOf": [{"type": "STRING"}, {"type": "NULL"}]}},
                    "required": ["start", "face", "kind", "who"]}},
            },
            "required": ["beats"],
        })


class GeminiParseTest(GeminiCase):
    def test_failed_answers_raise_typed_errors_with_usage_recorded(self):
        cases = [
            ("blocked prompt",
             {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
              "usageMetadata": {"promptTokenCount": 800, "totalTokenCount": 800}},
             gemini.GeminiBlocked, (800, 0)),
            ("no candidates", {"candidates": [], "usageMetadata": USAGE}, gemini.GeminiError, (1000, 500)),
            ("truncated", answer_body(text='{"a": [1, 2', finish="MAX_TOKENS"),
             gemini.GeminiTruncated, (1000, 500)),
            ("safety", answer_body(text="", finish="SAFETY"), gemini.GeminiBlocked, (1000, 500)),
            ("recitation", answer_body(text="", finish="RECITATION"), gemini.GeminiBlocked, (1000, 500)),
            ("spii", answer_body(text="", finish="SPII"), gemini.GeminiBlocked, (1000, 500)),
            ("not JSON", answer_body(text="I can't help with that."), gemini.GeminiError, (1000, 500)),
            ("a JSON array", answer_body(text="[1, 2]"), gemini.GeminiError, (1000, 500)),
            ("only thoughts", answer_body(parts=[{"text": '{"a": 1}', "thought": True}]),
             gemini.GeminiError, (1000, 500)),
            ("malformed, other finish", answer_body(text="{", finish="MALFORMED_RESPONSE"),
             gemini.GeminiError, (1000, 500)),
        ]
        for name, body, error_type, (tokens_in, tokens_out) in cases:
            with self.subTest(name):
                self.ledger.unlink(missing_ok=True)
                self.http.add(http(200, body))
                with self.assertRaises(gemini.GeminiError) as ctx:
                    self.generate()
                self.assertIs(type(ctx.exception), error_type)
                usage = ctx.exception.usage
                self.assertEqual((usage["input_tokens"], usage["output_tokens"]), (tokens_in, tokens_out))
                self.assertEqual(usage["model"], "gemini-3.8-flash")
                engine = self.ledger_day()["buckets"]["engine"]   # billed spend is on the books
                self.assertEqual(engine["calls"], 1)
                self.assertAlmostEqual(engine["spend_usd"], usage["cost_usd"], places=6)

    def test_answer_text(self):
        cases = [
            ("thought parts skipped",
             [{"text": 'Thinking: {"a": 0}', "thought": True}, {"text": '{"a": 1}'}], {"a": 1}),
            ("split across parts", [{"text": '{"a": '}, {"text": "1}"}], {"a": 1}),
            ("fenced", [{"text": '```json\n{"a": 1}\n```'}], {"a": 1}),
            ("prose around", [{"text": 'Here you go: {"a": 1} Hope that helps!'}], {"a": 1}),
            ("trailing braces in prose", [{"text": '{"a": 1} (see {notes})'}], {"a": 1}),
            ("non-text parts ignored", [{"inlineData": {}}, {"text": '{"a": 1}'}], {"a": 1}),
        ]
        for name, parts, expected in cases:
            with self.subTest(name):
                self.http.add(answer(parts=parts))
                self.assertEqual(self.generate()[0], expected)

    def test_parse_json_object(self):
        cases = [
            ('{"a": 1}', {"a": 1}),
            ('  {"a": 1}\n', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('Sure! {"a": {"b": 2}} Thanks', {"a": {"b": 2}}),
            ('{"a": 1} then {b}', {"a": 1}),
            ('{"a": 1} {"b": 2}', {"a": 1}),
            ('[{"a": 1}]', None),          # valid JSON, wrong shape: not dug into
            ('"text"', None),
            ("no braces at all", None),
            ('{"a": ', None),              # truncated
            ("", None),
            (None, None),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(gemini.parse_json_object(text), expected)

    def test_unparseable_error_quotes_the_first_200_characters(self):
        self.http.add(answer(text="x" * 500))
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.generate()
        self.assertIn("x" * 200, str(ctx.exception))
        self.assertNotIn("x" * 201, str(ctx.exception))

    def test_usage_accounting(self):
        cases = [
            ("all counts", {"promptTokenCount": 1000, "toolUsePromptTokenCount": 50,
                            "candidatesTokenCount": 200, "thoughtsTokenCount": 300,
                            "totalTokenCount": 1550}, (1050, 500), False),
            ("no thinking", {"promptTokenCount": 1000, "candidatesTokenCount": 200,
                             "totalTokenCount": 1200}, (1000, 200), False),
            ("cached prompt tokens are part of the prompt",
             {"promptTokenCount": 1000, "cachedContentTokenCount": 600,
              "candidatesTokenCount": 200, "totalTokenCount": 1200}, (1000, 200), False),
            ("total beyond the named counts", {"promptTokenCount": 1000, "candidatesTokenCount": 200,
                                               "totalTokenCount": 1300}, (1000, 300), False),
            ("only a total", {"totalTokenCount": 4000}, (0, 4000), True),
            ("no usage at all", None, (0, 0), True),
        ]
        for name, usage, expected, warns in cases:
            with self.subTest(name):
                self.http.add(answer({"a": 1}, usage=usage))
                if warns:
                    with self.assertLogs("clip_engine.gemini", "WARNING"):
                        _, got = self.generate()
                else:
                    with self.assertNoLogs("clip_engine.gemini", "WARNING"):
                        _, got = self.generate()
                self.assertEqual((got["input_tokens"], got["output_tokens"]), expected)
                self.assertAlmostEqual(got["cost_usd"],
                                       expected[0] * 1.5 / 1e6 + expected[1] * 7.5 / 1e6, places=6)

    def test_a_non_json_body(self):
        self.http.add(http(200, text="<html>oops</html>"))
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.generate()
        self.assertIsNone(ctx.exception.usage)


class GeminiUploadTest(GeminiCase):
    START = "https://generativelanguage.googleapis.com/upload/v1beta/files"

    def seed_cache(self, age_hours: float, name: str = "files/old1",
                   key: str | None = "test-key") -> None:
        """One cached upload of ``self.video``, made under API key ``key``
        (None: by a version that didn't record the key)."""
        st = self.video.stat()
        when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=age_hours)
        entry = {"name": name, "uri": f"{gemini.BASE}/v1beta/{name}", "mime_type": "video/mp4",
                 "uploaded_at": when.replace(microsecond=0).isoformat().replace("+00:00", "Z")}
        if key is not None:
            entry["key_fingerprint"] = gemini._key_fingerprint(key)
        util.atomic_write_json(self.engine_dir / "gemini_files.json", {
            f"{self.video.resolve()}|{st.st_size}|{st.st_mtime_ns}": entry})

    def test_resumable_upload_then_poll_until_active(self):
        file = self.script_upload(states=("PROCESSING", "PROCESSING", "ACTIVE"))
        out = gemini.upload_file(self.video, display_name="clip render")
        self.assertEqual(out, {"name": "files/abc123", "uri": file["uri"], "mime_type": "video/mp4"})
        size = str(self.video.stat().st_size)
        start, send, poll1, poll2 = self.http.calls
        self.assertEqual((start["method"], start["url"]), ("POST", self.START))
        self.assertEqual(start["headers"], {
            "x-goog-api-key": "test-key", "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start", "X-Goog-Upload-Header-Content-Length": size,
            "X-Goog-Upload-Header-Content-Type": "video/mp4", "Content-Type": "application/json"})
        self.assertEqual(start["json"], {"file": {"display_name": "clip render"}})
        self.assertEqual((send["method"], send["url"]), ("POST", "https://upload.test/session-1"))
        self.assertEqual(send["headers"], {"Content-Length": size, "X-Goog-Upload-Offset": "0",
                                           "X-Goog-Upload-Command": "upload, finalize"})
        self.assertEqual(send["sent_bytes"], self.video.read_bytes())
        for poll in (poll1, poll2):
            self.assertEqual((poll["method"], poll["url"]), ("GET", f"{gemini.BASE}/v1beta/files/abc123"))
            self.assertEqual(poll["headers"], {"x-goog-api-key": "test-key"})
        self.assertEqual(self.sleeps, [3.0, 3.0])
        st = self.video.stat()
        (key, entry), = self.cache().items()
        self.assertEqual(key, f"{self.video.resolve()}|{st.st_size}|{st.st_mtime_ns}")
        self.assertEqual({k: entry[k] for k in ("name", "uri", "mime_type")}, out)
        self.assertTrue(entry["uploaded_at"].endswith("Z"))
        # Which key made it is recorded, but never the key itself.
        self.assertEqual(entry["key_fingerprint"], gemini._key_fingerprint("test-key"))
        self.assertNotIn("test-key", (self.engine_dir / "gemini_files.json").read_text())

    def test_cache_reuse(self):
        cases = [
            # (age of the cached upload in hours, scripted answers, reused?, requests made)
            ("fresh: reused without asking", 0.5, [], True, []),
            ("older: confirmed ACTIVE first", 2, [http(200, {"name": "files/old1", "state": "ACTIVE"})],
             True, ["GET"]),
            ("older and gone: uploaded again", 2, [api_error(404, "not found", name="NOT_FOUND")],
             False, ["GET", "POST", "POST"]),
            ("older and failed: uploaded again", 2,
             [http(200, {"name": "files/old1", "state": "FAILED"})], False, ["GET", "POST", "POST"]),
            ("past 46 h: uploaded again without asking", 47, [], False, ["POST", "POST"]),
        ]
        for name, age, answers, reused, methods in cases:
            with self.subTest(name), mock.patch.object(gemini, "_confirmed", {}):
                self.http.calls.clear()
                self.seed_cache(age)
                self.http.add(*answers)
                if not reused:
                    self.script_upload(name="files/new1")
                out = gemini.upload_file(self.video)
                self.assertEqual(out["name"], "files/old1" if reused else "files/new1")
                self.assertEqual([c["method"] for c in self.http.calls], methods)
                names = [e["name"] for e in self.cache().values()]
                self.assertEqual(names, ["files/old1"] if reused else ["files/new1"])

    def test_uploads_are_reused_only_under_the_key_that_made_them(self):
        """Uploads belong to the API key's project. Once GEMINI_API_KEY points
        at another project (say, after the first one's credit ran out), the
        old uploads can't be read with it, so they mustn't be handed out, not
        even fresh ones that are reused without a check."""
        cases = [
            # (name, environment, key the cached upload was made with)
            ("another project's key", {"GEMINI_API_KEY": "other-project-key"}, "test-key"),
            ("an entry that didn't record its key", {}, None),
        ]
        for name, environ, made_with in cases:
            with self.subTest(name), mock.patch.dict(os.environ, environ), \
                    mock.patch.object(gemini, "_confirmed", {}):
                self.http.calls.clear()
                self.seed_cache(10 / 60, key=made_with)
                self.script_upload(name="files/new1")
                self.assertEqual(gemini.upload_file(self.video)["name"], "files/new1")
                self.assertEqual([c["method"] for c in self.http.calls], ["POST", "POST"])
                self.assertEqual(self.http.calls[0]["headers"]["x-goog-api-key"],
                                 os.environ["GEMINI_API_KEY"])
                self.assertEqual([e["name"] for e in self.cache().values()], ["files/new1"])
        # The new upload is reused, without asking, under the key that made it.
        self.http.calls.clear()
        self.assertEqual(gemini.upload_file(self.video)["name"], "files/new1")
        self.assertEqual(self.http.calls, [])

    def test_an_upload_gemini_cannot_read_is_forgotten(self):
        """A 4xx that names one of the request's uploads (another project's
        file, or one already gone) drops it from the cache, so the next call
        uploads the video again rather than failing until the hourly check."""
        cases = [
            ("another project's file", 403,
             "You do not have permission to access the File abc123 or it may not exist.", True),
            ("gone", 404, "Requested entity was not found: files/abc123", True),
            ("not ready", 400, "The File abc123 is not in an ACTIVE state and usage is not allowed.",
             True),
            ("about something else", 403, "Method doesn't allow unregistered callers.", False),
            ("another file", 403, "You do not have permission to access the File abc1234.", False),
        ]
        for name, status, message, forgotten in cases:
            with self.subTest(name), mock.patch.object(gemini, "_confirmed", {}):
                (self.engine_dir / "gemini_files.json").unlink(missing_ok=True)
                self.script_upload()
                uploaded = gemini.upload_file(self.video)
                self.http.add(api_error(status, message, name="PERMISSION_DENIED"))
                video = {"fileData": {"mimeType": "video/mp4", "fileUri": uploaded["uri"]}}
                with self.assertRaises(gemini.GeminiError) as ctx:
                    self.generate(parts=[video, gemini.text_part("payload")])
                self.assertEqual(ctx.exception.status, status)
                self.assertEqual([e["name"] for e in self.cache().values()],
                                 [] if forgotten else ["files/abc123"])
                self.assertEqual(gemini._confirmed, {} if forgotten else {"files/abc123": mock.ANY})
                self.assertEqual(self.http.answers, [])

    def test_a_recent_upload_and_a_recent_check_are_reused(self):
        self.script_upload()
        first = gemini.upload_file(self.video)
        self.assertEqual(gemini.upload_file(self.video), first)
        self.assertEqual(len(self.http.calls), 2)
        # An old entry confirmed a moment ago isn't confirmed again.
        self.seed_cache(5)
        self.http.add(http(200, {"name": "files/old1", "state": "ACTIVE"}))
        gemini.upload_file(self.video)
        gemini.upload_file(self.video)
        self.assertEqual([c["method"] for c in self.http.calls], ["POST", "POST", "GET"])

    def test_a_changed_file_is_uploaded_again(self):
        self.script_upload(name="files/one")
        gemini.upload_file(self.video)
        self.video.write_bytes(b"different bytes entirely")
        self.script_upload(name="files/two")
        self.assertEqual(gemini.upload_file(self.video)["name"], "files/two")
        self.assertEqual(len(self.cache()), 2)

    def test_processing_failures(self):
        failed = self.file_resource(state="FAILED")
        failed["error"] = {"code": 400, "message": "Unsupported codec"}
        cases = [
            ("FAILED", [http(200, {}, headers={"X-Goog-Upload-URL": "https://upload.test/s"}),
                        http(200, {"file": failed})], {}, "Unsupported codec"),
            ("never ACTIVE", [http(200, {}, headers={"X-Goog-Upload-URL": "https://upload.test/s"}),
                              http(200, {"file": self.file_resource(state="PROCESSING")})],
             {"wait_timeout": 0}, "still PROCESSING"),
            ("no upload URL", [http(200, {})], {}, "no upload URL"),
            ("start refused", [api_error(403, "API key not valid", name="PERMISSION_DENIED")], {},
             "API key not valid"),
            ("no file in the answer", [http(200, {}, headers={"X-Goog-Upload-URL": "https://upload.test/s"}),
                                       http(200, {"file": {}})], {}, "no file name"),
        ]
        for name, answers, kw, message in cases:
            with self.subTest(name):
                self.http.add(*answers)
                with self.assertRaises(gemini.GeminiError) as ctx:
                    gemini.upload_file(self.video, **kw)
                self.assertIn(message, str(ctx.exception))
                self.assertEqual(self.http.answers, [])
                self.assertEqual(self.cache(), {})

    def test_a_dropped_transfer_starts_a_fresh_session(self):
        self.http.add(http(200, {}, headers={"X-Goog-Upload-URL": "https://upload.test/s1"}),
                      requests.exceptions.ConnectionError("reset by peer"),
                      http(200, {}, headers={"X-Goog-Upload-URL": "https://upload.test/s2"}),
                      http(200, {"file": self.file_resource()}))
        with self.assertLogs("clip_engine.gemini", "WARNING"):
            out = gemini.upload_file(self.video)
        self.assertEqual(out["name"], "files/abc123")
        self.assertEqual([c["url"] for c in self.http.calls],
                         [self.START, "https://upload.test/s1", self.START, "https://upload.test/s2"])
        self.assertEqual(self.http.calls[3]["sent_bytes"], self.video.read_bytes())

    def test_a_transfer_gemini_never_answers_is_retried_only_once(self):
        """Each try can wait _upload_timeout(size) for Google's answer (many
        minutes for a whole-source proxy), so a read timeout gets one fresh
        session, as in _request; a dropped connection still gets two."""
        def started(n):
            return http(200, {}, headers={"X-Goog-Upload-URL": f"https://upload.test/s{n}"})

        rq = requests.exceptions
        cases = [
            ("read timeouts", [rq.ReadTimeout("slow"), rq.ReadTimeout("slow")], False),
            ("body stalled", [rq.ConnectionError(ReadTimeoutError(None, None, "timed out")),
                              rq.ReadTimeout("slow")], False),
            ("dropped connections", [rq.ConnectionError("reset"), rq.ConnectionError("reset")], True),
        ]
        for name, failures, uploaded in cases:
            with self.subTest(name), mock.patch.object(gemini, "_confirmed", {}):
                (self.engine_dir / "gemini_files.json").unlink(missing_ok=True)
                self.http.calls.clear()
                self.http.answers.clear()
                self.http.add(started(1), failures[0], started(2), failures[1],
                              started(3), http(200, {"file": self.file_resource()}))
                with self.assertLogs("clip_engine.gemini", "WARNING"):
                    if uploaded:
                        self.assertEqual(gemini.upload_file(self.video)["name"], "files/abc123")
                    else:
                        with self.assertRaises(gemini.GeminiError):
                            gemini.upload_file(self.video)
                self.assertEqual(len(self.http.calls), 6 if uploaded else 4)

    def test_a_missing_file(self):
        with self.assertRaises(gemini.GeminiError):
            gemini.upload_file(self.dir / "nope.mp4")
        self.assertEqual(self.http.calls, [])

    def test_parallel_requests_for_one_file_upload_it_once(self):
        self.script_upload()
        results = util.parallel_map(lambda _: gemini.upload_file(self.video), range(4), workers=4)
        for result in results:
            self.assertEqual(result, results[0])
        self.assertEqual(len(self.http.calls), 2)

    def test_delete_file(self):
        self.script_upload()
        out = gemini.upload_file(self.video)
        self.http.add(http(200, {}))
        gemini.delete_file(out["uri"])   # a URI works as well as a name
        call = self.http.calls[-1]
        self.assertEqual((call["method"], call["url"]), ("DELETE", f"{gemini.BASE}/v1beta/files/abc123"))
        self.assertEqual(call["headers"], {"x-goog-api-key": "test-key"})
        self.assertEqual(self.cache(), {})   # forgotten, so the next use uploads again
        self.script_upload(name="files/again")
        self.assertEqual(gemini.upload_file(self.video)["name"], "files/again")

    def test_delete_file_never_raises(self):
        for name, reply in (("network", requests.exceptions.ConnectionError("down")),
                            ("server", api_error(500, "internal"))):
            with self.subTest(name):
                self.http.add(reply)
                with self.assertLogs("clip_engine.gemini", "WARNING"):
                    gemini.delete_file("files/zzz")
        self.http.calls.clear()
        with self.assertLogs("clip_engine.gemini", "WARNING"):
            gemini.delete_file("not a file name!")
        self.assertEqual(self.http.calls, [])
        self.http.add(api_error(404, "gone"))
        gemini.delete_file("abc-1")          # a bare id; 404 = already gone, nothing to report
        self.assertEqual(self.http.calls[-1]["url"], f"{gemini.BASE}/v1beta/files/abc-1")


class VideoPartsTest(GeminiCase):
    def setUp(self):
        super().setUp()
        self.files = []
        for i in range(3):
            path = self.dir / f"v{i}.mp4"
            path.write_bytes(bytes([65 + i]) * 6000)
            self.files.append(path)

    @staticmethod
    def kinds(parts):
        return [next(iter(p)) for p in parts]

    def test_inline_budget_across_several_parts(self):
        file = self.script_upload(name="files/third")
        parts = gemini.video_parts([{"path": f, "fps": 2.0} for f in self.files],
                                   inline_budget_bytes=14000)
        self.assertEqual(self.kinds(parts), ["inlineData", "inlineData", "fileData"])
        self.assertEqual(parts[0]["inlineData"], {
            "mimeType": "video/mp4",
            "data": base64.b64encode(self.files[0].read_bytes()).decode("ascii")})
        self.assertEqual(parts[1]["inlineData"]["data"],
                         base64.b64encode(self.files[1].read_bytes()).decode("ascii"))
        self.assertEqual(parts[2]["fileData"], {"mimeType": "video/mp4", "fileUri": file["uri"]})
        self.assertEqual(self.http.calls[1]["sent_bytes"], self.files[2].read_bytes())
        for part in parts:
            self.assertEqual(part["videoMetadata"], {"fps": 2.0})

    def test_upload_flag_and_running_total(self):
        big = self.dir / "big.mp4"
        big.write_bytes(b"B" * 20000)
        cases = [
            # (items, budget, expected kinds)
            ([{"path": self.files[0], "upload": True}, {"path": self.files[1]}, {"path": self.files[2]}],
             12000, ["fileData", "inlineData", "inlineData"]),
            ([{"path": big}, {"path": self.files[0]}], 14000, ["fileData", "inlineData"]),
            ([{"path": self.files[0]}, {"path": self.files[0]}, {"path": self.files[0]}],
             12000, ["inlineData", "inlineData", "fileData"]),   # the same file counts each time
            ([{"path": self.files[0]}], 0, ["fileData"]),
        ]
        for i, (items, budget, expected) in enumerate(cases):
            with self.subTest(case=i), mock.patch.object(gemini, "_confirmed", {}):
                (self.engine_dir / "gemini_files.json").unlink(missing_ok=True)
                for _ in range(expected.count("fileData")):
                    self.script_upload(name=f"files/c{i}")
                parts = gemini.video_parts(items, inline_budget_bytes=budget)
                self.assertEqual(self.kinds(parts), expected)
                self.assertEqual(self.http.answers, [])

    def test_video_metadata(self):
        cases = [
            ({}, None),
            ({"fps": 1}, {"fps": 1.0}),
            ({"fps": 30}, {"fps": 24.0}),
            ({"start": 0.0, "end": 3.0}, {"startOffset": "0.0s", "endOffset": "3.0s"}),
            ({"fps": 8.0, "start": 1.25, "end": 2.5}, {"fps": 8.0, "startOffset": "1.25s", "endOffset": "2.5s"}),
            ({"start": 12.3456789}, {"startOffset": "12.346s"}),
            ({"start": -1.0}, {"startOffset": "0.0s"}),
            ({"end": 7}, {"endOffset": "7.0s"}),
            ({"fps": None, "start": None}, None),
        ]
        for meta, expected in cases:
            with self.subTest(meta=meta):
                (part,) = gemini.video_parts([{"path": self.files[0], **meta}], inline_budget_bytes=10 ** 9)
                self.assertEqual(part.get("videoMetadata"), expected)
        for meta in ({"fps": 0}, {"fps": -1}, {"fps": "fast"}, {"start": 3.0, "end": 3.0},
                     {"start": 5, "end": 2}, {"end": 0}):
            with self.subTest(bad=meta), self.assertRaises(gemini.GeminiError):
                gemini.video_parts([{"path": self.files[0], **meta}], inline_budget_bytes=10 ** 9)

    def test_default_budget_comes_from_settings(self):
        cases = [
            ({"inline_request_mb": 0.01}, "inlineData"),    # 10,485 bytes: 6,000 fit
            ({"inline_request_mb": 0.005}, "fileData"),     # 5,242 bytes: they don't
            ({"inline_request_mb": ""}, "inlineData"),      # blank -> 14 MiB
            ({}, "inlineData"),
        ]
        for gemini_settings, expected in cases:
            settings = config.Settings(raw={"clip_engine": {"gemini": gemini_settings}})
            with self.subTest(settings=gemini_settings), \
                    mock.patch("app.config.load_settings", return_value=settings), \
                    mock.patch.object(gemini, "_confirmed", {}):
                (self.engine_dir / "gemini_files.json").unlink(missing_ok=True)
                if expected == "fileData":
                    self.script_upload()
                self.assertEqual(self.kinds(gemini.video_parts([{"path": self.files[0]}])), [expected])

    def test_a_missing_video(self):
        with self.assertRaises(gemini.GeminiError):
            gemini.video_parts([{"path": self.dir / "missing.mp4"}])


# =========================================================================================
# Claude
# =========================================================================================

def text(value: str) -> dict:
    return {"type": "text", "text": value}


def thinking(value: str = "") -> dict:
    return {"type": "thinking", "thinking": value}


def fallback(source: str, target: str) -> dict:
    return {"type": "fallback", "from": {"model": source}, "to": {"model": target},
            "trigger": {"type": "refusal", "category": "cyber"}}


def sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def stream_events(content: list[dict], *, model: str = "claude-opus-5-5", stop_reason: str = "end_turn",
                  usage: dict | None = None, iterations: list | None = None,
                  stop_details: dict | None = None) -> list[dict]:
    """The SSE events of one streamed message whose blocks arrive whole."""
    usage = usage or {"input_tokens": 1000, "output_tokens": 300}
    usage = {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, **usage}
    events = [{"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {**usage, "output_tokens": 1}}}]
    for i, block in enumerate(content):
        if block["type"] == "text":
            events.append({"type": "content_block_start", "index": i, "content_block": text("")})
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "text_delta", "text": block["text"]}})
        elif block["type"] == "thinking":
            events.append({"type": "content_block_start", "index": i,
                           "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "signature_delta", "signature": "sig"}})
        else:
            events.append({"type": "content_block_start", "index": i, "content_block": block})
        events.append({"type": "content_block_stop", "index": i})
    final_usage = dict(usage)
    if iterations is not None:
        final_usage["iterations"] = iterations
    delta = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events.append({"type": "message_delta", "delta": delta, "usage": final_usage})
    events.append({"type": "message_stop"})
    return events


def streamed(content: list[dict], **kw) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          content=sse(stream_events(content, **kw)))


def claude_error(status: int, kind: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"type": "error", "error": {"type": kind, "message": message}})


class BrokenStream(httpx.SyncByteStream):
    """A response body that drops after its first chunk."""

    def __init__(self, first: bytes):
        self.first = first

    def __iter__(self):
        yield self.first
        raise httpx.ReadError("connection reset by peer")


class FakeAnthropic:
    """A real ``anthropic.Anthropic`` whose HTTP is answered from a script."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []
        self.client = anthropic.Anthropic(
            api_key="test-key", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handle)))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.answers:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        reply = self.answers.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


class ClaudeCase(ProviderCase):
    def fake(self, *answers) -> FakeAnthropic:
        fake = FakeAnthropic(*answers)
        p = mock.patch.object(llm, "client", return_value=fake.client)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(fake.client.close)
        return fake

    def call(self, model: str = "claude-opus-5-5", **kw):
        args = dict(system="SYS", content=[claude.text_block("payload")], schema=SCHEMA,
                    effort="high", max_tokens=4000, label="test")
        args.update(kw)
        return claude.json_call(model, **args)


class StrictSchemaTest(unittest.TestCase):
    def test_objects_are_closed_at_every_depth_and_constraints_dropped(self):
        schema = {
            "type": "object",
            "propertyOrdering": ["beats", "summary"],
            "properties": {
                "summary": {"type": "string", "maxLength": 400, "description": "one line"},
                "beats": {"type": "array", "minItems": 1, "maxItems": 400, "uniqueItems": True,
                          "items": {"type": "object", "properties": {
                              "start": {"type": "number", "minimum": 0, "exclusiveMaximum": 1e6},
                              "energy": {"type": "integer", "minimum": 1, "maximum": 5, "multipleOf": 1},
                              "shot": {"type": "string", "enum": ["aerial", "close"], "pattern": "^[a-z]+$"},
                          }, "required": ["start"]}},
                "meta": {"type": "object"},
            },
            "required": ["summary"],
        }
        self.assertEqual(claude.strict_schema(schema), {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "one line"},
                "beats": {"type": "array", "items": {"type": "object", "properties": {
                    "start": {"type": "number"},
                    "energy": {"type": "integer"},
                    "shot": {"type": "string", "enum": ["aerial", "close"]},
                }, "required": ["start", "energy", "shot"], "additionalProperties": False}},
                "meta": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            },
            "required": ["summary", "beats", "meta"],
            "additionalProperties": False,
        })

    def test_type_lists(self):
        cases = [
            ("nullable number", {"type": ["number", "null"], "description": "seconds", "minimum": 0},
             {"description": "seconds", "anyOf": [{"type": "number"}, {"type": "null"}]}),
            ("single-item list", {"type": ["string"]}, {"type": "string"}),
            ("enum split by type", {"type": ["string", "null"], "enum": ["a", "b", None]},
             {"anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]}),
            ("enum without null drops the null branch", {"type": ["string", "null"], "enum": ["a"]},
             {"anyOf": [{"type": "string", "enum": ["a"]}]}),
            ("nullable object keeps its properties closed",
             {"type": ["object", "null"], "properties": {"x": {"type": "integer", "maximum": 3}}},
             {"anyOf": [{"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"],
                         "additionalProperties": False}, {"type": "null"}]}),
            ("nullable array", {"type": ["array", "null"], "items": {"type": ["number", "null"]}, "maxItems": 2},
             {"anyOf": [{"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
                        {"type": "null"}]}),
            ("OpenAPI nullable", {"type": "string", "nullable": True, "title": "who"},
             {"title": "who", "anyOf": [{"type": "string"}, {"type": "null"}]}),
            ("oneOf", {"oneOf": [{"type": "string", "maxLength": 3}, {"type": "null"}]},
             {"anyOf": [{"type": "string"}, {"type": "null"}]}),
            ("$defs", {"$defs": {"seg": {"type": "object", "properties": {"s": {"type": "number"}}}},
                       "type": "object", "properties": {"seg": {"$ref": "#/$defs/seg"}}},
             {"$defs": {"seg": {"type": "object", "properties": {"s": {"type": "number"}},
                                "required": ["s"], "additionalProperties": False}},
              "type": "object", "properties": {"seg": {"$ref": "#/$defs/seg"}},
              "required": ["seg"], "additionalProperties": False}),
        ]
        for name, schema, expected in cases:
            with self.subTest(name):
                self.assertEqual(claude.strict_schema(schema), expected)

    def test_the_input_is_not_modified(self):
        schema = {"type": "object", "properties": {"t": {"type": ["number", "null"], "minimum": 0},
                                                   "o": {"type": "object", "properties": {}}}}
        before = copy.deepcopy(schema)
        out = claude.strict_schema(schema)
        self.assertEqual(schema, before)
        out["properties"]["o"]["properties"]["new"] = {}
        self.assertEqual(schema, before)


class BlocksTest(unittest.TestCase):
    def test_text_and_image_blocks(self):
        self.assertEqual(claude.text_block("hi"), {"type": "text", "text": "hi"})
        self.assertEqual(claude.image_block(b"\xff\xd8jpeg"), {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(b"\xff\xd8jpeg").decode("ascii")}})

    def test_frames_blocks(self):
        frames = [(0.0, b"a"), (12.04, b"b"), (13.5, b""), (59.96, b"c")]
        video = claude.frames_blocks(frames)
        self.assertEqual([b["text"] for b in video if b["type"] == "text"], ["t=0.0s", "t=12.0s", "t=60.0s"])
        self.assertEqual([b["type"] for b in video], ["text", "image"] * 3)   # empty frame skipped
        self.assertEqual(video[3], claude.image_block(b"b"))
        clip = claude.frames_blocks(frames[:2], clock="clip")
        self.assertEqual([b["text"] for b in clip if b["type"] == "text"], ["clip 0.0s", "clip 12.0s"])
        self.assertEqual(claude.frames_blocks([]), [])
        with self.assertRaises(ValueError):
            claude.frames_blocks(frames, clock="wall")

    def test_is_configured(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": " "}):
            self.assertFalse(claude.is_configured())
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}):
            self.assertTrue(claude.is_configured())


class ClaudeRequestTest(ClaudeCase):
    def test_request_shape(self):
        fake = self.fake(streamed([thinking(), text('{"a": 1, "note": null}')]))
        data, _ = self.call()
        self.assertEqual(data, {"a": 1, "note": None})
        (request,) = fake.requests
        self.assertEqual(request.url.path, "/v1/messages")
        self.assertEqual(request.headers["anthropic-beta"], claude.FALLBACK_BETA)
        body = json.loads(request.content)
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["max_tokens"], 4000)
        self.assertEqual(body["system"], SYS_US)
        self.assertEqual(body["messages"], [{"role": "user", "content": [{"type": "text", "text": "payload"}]}])
        self.assertIs(body["stream"], True)
        self.assertEqual(body["output_config"], {
            "effort": "high",
            "format": {"type": "json_schema", "schema": claude.strict_schema(SCHEMA)}})
        self.assertEqual(body["fallbacks"], "default")
        for key in ("temperature", "top_p", "top_k", "thinking"):
            self.assertNotIn(key, body)

    def test_blank_effort_is_left_out_and_system_is_just_the_spelling_line(self):
        fake = self.fake(streamed([text('{"a": 1}')]))
        self.call(effort="", system="", content="just text")
        body = fake.bodies()[0]
        self.assertEqual(list(body["output_config"]), ["format"])
        self.assertEqual(body["system"], AMERICAN_ENGLISH)
        self.assertEqual(body["messages"], [{"role": "user", "content": "just text"}])

    def test_usage_and_spend(self):
        self.fake(streamed([thinking(), text('{"a": 1}')], usage={
            "input_tokens": 1000, "output_tokens": 300, "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 50}))
        _, usage = self.call()
        cost = 1150 * 4.0 / 1e6 + 300 * 20.0 / 1e6
        self.assertEqual(usage, {"model": "claude-opus-5-5", "input_tokens": 1150, "output_tokens": 300,
                                 "cost_usd": round(cost, 6)})
        day = self.ledger_day()
        self.assertEqual(day["calls"], 0)
        self.assertAlmostEqual(day["buckets"]["engine"]["spend_usd"], cost)

    def test_buckets(self):
        cases = [("bench", None, "bench"), (None, "bench", "bench"), (None, None, "engine"),
                 ("engine", "bench", "engine")]
        for bucket, default, expected in cases:
            with self.subTest(bucket=bucket, default=default):
                self.ledger.unlink(missing_ok=True)
                spend.set_bucket(default)
                self.fake(streamed([text('{"a": 1}')]))
                self.call(bucket=bucket)
                self.assertEqual(list(self.ledger_day()["buckets"]), [expected])

    def test_no_model(self):
        fake = self.fake()
        for model in ("", "  ", None):
            with self.subTest(model=model), self.assertRaises(claude.ClaudeError):
                self.call(model=model)
        self.assertEqual(fake.requests, [])


class ClaudeFallbackTest(ClaudeCase):
    def test_retries_once_without_server_side_fallback(self):
        for message in ("fallbacks: Extra inputs are not permitted",
                        "Unexpected value(s) `server-side-fallback-2026-07-01` for the "
                        "`anthropic-beta` header."):
            with self.subTest(message=message):
                fake = self.fake(claude_error(400, "invalid_request_error", message),
                                 streamed([text('{"a": 1}')]))
                with self.assertLogs("clip_engine.claude", "WARNING"):
                    data, _ = self.call()
                self.assertEqual(data, {"a": 1})
                first, second = fake.requests
                self.assertEqual(first.headers.get("anthropic-beta"), claude.FALLBACK_BETA)
                self.assertEqual(json.loads(first.content)["fallbacks"], "default")
                self.assertNotIn("anthropic-beta", second.headers)
                second_body = json.loads(second.content)
                self.assertNotIn("fallbacks", second_body)
                self.assertEqual(second_body["output_config"], json.loads(first.content)["output_config"])

    def test_other_400s_are_not_retried(self):
        fake = self.fake(claude_error(400, "invalid_request_error", "max_tokens: must be at most 128000"))
        with self.assertRaises(claude.ClaudeError) as ctx:
            self.call()
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("max_tokens", str(ctx.exception))
        self.assertEqual(len(fake.requests), 1)

    def test_a_model_without_effort_control_runs_at_its_default(self):
        fake = self.fake(claude_error(400, "invalid_request_error",
                                      "output_config.effort: This model does not support the effort parameter."),
                         streamed([text('{"a": 1}')]))
        with self.assertLogs("clip_engine.claude", "WARNING"):
            self.assertEqual(self.call(model="claude-haiku-4-5")[0], {"a": 1})
        first, second = fake.bodies()
        self.assertEqual(first["output_config"]["effort"], "high")
        self.assertEqual(second["output_config"], {"format": first["output_config"]["format"]})
        self.assertEqual(second["fallbacks"], "default")   # only the named feature is dropped

    def test_downgrades_combine_and_each_happens_once(self):
        fake = self.fake(claude_error(400, "invalid_request_error", "fallbacks: not supported for this model"),
                         claude_error(400, "invalid_request_error", "effort: not supported"),
                         claude_error(400, "invalid_request_error", "effort: not supported"))
        with self.assertLogs("clip_engine.claude", "WARNING"), self.assertRaises(claude.ClaudeError):
            self.call()
        bodies = fake.bodies()
        self.assertEqual(len(bodies), 3)
        self.assertNotIn("fallbacks", bodies[1])
        self.assertIn("effort", bodies[1]["output_config"])
        self.assertNotIn("effort", bodies[2]["output_config"])

    def test_a_second_rejection_is_final(self):
        fake = self.fake(claude_error(400, "invalid_request_error", "fallbacks: not permitted"),
                         claude_error(400, "invalid_request_error", "beta not enabled for this org"))
        with self.assertLogs("clip_engine.claude", "WARNING"), self.assertRaises(claude.ClaudeError):
            self.call()
        self.assertEqual(len(fake.requests), 2)

    def test_the_answer_after_a_fallback_and_spend_per_attempt(self):
        iterations = [
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "output_tokens": 50,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 1100,
             "output_tokens": 200, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 10},
        ]
        self.fake(streamed(
            [thinking(), text('{"a": 0, "no'), fallback("claude-opus-5-5", "claude-opus-5"),
             thinking(), text('{"a": 2, "note": "after"}')],
            usage={"input_tokens": 1100, "output_tokens": 200, "cache_read_input_tokens": 10},
            iterations=iterations))
        data, usage = self.call()
        self.assertEqual(data, {"a": 2, "note": "after"})
        cost = 1000 * 4 / 1e6 + 50 * 20 / 1e6 + 1110 * 5 / 1e6 + 200 * 25 / 1e6
        self.assertEqual(usage, {"model": "claude-opus-5", "input_tokens": 2110, "output_tokens": 250,
                                 "cost_usd": round(cost, 6)})
        engine = self.ledger_day()["buckets"]["engine"]
        self.assertEqual(engine["calls"], 2)
        self.assertEqual(engine["by_model"]["claude-opus-5-5"]["output_tokens"], 50)
        self.assertEqual(engine["by_model"]["claude-opus-5"]["input_tokens"], 1110)
        self.assertAlmostEqual(engine["spend_usd"], cost)

    def test_a_chain_of_fallbacks_uses_the_last_answer(self):
        self.fake(streamed([text("{\"a\": 0"), fallback("claude-opus-5-5", "claude-opus-5"),
                            text("{\"a\": 1"), fallback("claude-opus-5", "claude-opus-4-8"),
                            text('{"a": 3}')]))
        data, usage = self.call()
        self.assertEqual(data, {"a": 3})
        self.assertEqual(usage["model"], "claude-opus-4-8")   # no iterations: top-level usage at its price
        self.assertAlmostEqual(usage["cost_usd"], 1000 * 5 / 1e6 + 300 * 25 / 1e6)

    def test_a_sticky_fallback_is_priced_at_the_model_that_served(self):
        self.fake(streamed([text('{"a": 1}')], model="claude-opus-5", iterations=[
            {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 1000,
             "output_tokens": 300, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]))
        _, usage = self.call()
        self.assertEqual(usage["model"], "claude-opus-5")
        self.assertAlmostEqual(usage["cost_usd"], 1000 * 5 / 1e6 + 300 * 25 / 1e6)


class ClaudeStopTest(ClaudeCase):
    def test_refusal_and_truncation_raise_with_spend_recorded(self):
        cases = [
            ("refused before any output",
             dict(content=[], stop_reason="refusal",
                  stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."}),
             claude.ClaudeRefused, "cyber"),
            ("refused mid-answer", dict(content=[text('{"a": ')], stop_reason="refusal"),
             claude.ClaudeRefused, "no category"),
            ("hit max_tokens", dict(content=[thinking(), text('{"a": [1, 2,')], stop_reason="max_tokens"),
             claude.ClaudeTruncated, "max_tokens"),
        ]
        for name, kw, error_type, message in cases:
            with self.subTest(name):
                self.ledger.unlink(missing_ok=True)
                self.fake(streamed(**kw))
                with self.assertRaises(error_type) as ctx:
                    self.call()
                self.assertIn(message, str(ctx.exception))
                self.assertEqual(ctx.exception.usage["input_tokens"], 1000)
                self.assertEqual(ctx.exception.usage["output_tokens"], 300)
                self.assertEqual(self.ledger_day()["buckets"]["engine"]["calls"], 1)

    def test_answers_that_are_not_a_json_object(self):
        cases = [
            ("prose", [text("Sorry, I can't do that.")]),
            ("an array", [text("[1, 2]")]),
            ("no text at all", [thinking()]),
            ("text only before the fallback", [text('{"a": 1}'), fallback("claude-opus-5-5", "claude-opus-5")]),
        ]
        for name, content in cases:
            with self.subTest(name):
                self.fake(streamed(content))
                with self.assertRaises(claude.ClaudeError) as ctx:
                    self.call()
                self.assertIs(type(ctx.exception), claude.ClaudeError)
                self.assertIsNotNone(ctx.exception.usage)

    def test_lenient_parsing(self):
        cases = [
            ("prose around", [text('Here it is: {"a": 1} done')], {"a": 1}),
            ("fenced", [text('```json\n{"a": 1}\n```')], {"a": 1}),
            ("split over two blocks", [text('{"a": '), text("5}")], {"a": 5}),
            ("the last block wins", [text('{"a": 1}'), text('{"a": 2}')], {"a": 2}),
        ]
        for name, content, expected in cases:
            with self.subTest(name):
                self.fake(streamed(content))
                self.assertEqual(self.call()[0], expected)


class ClaudeErrorsTest(ClaudeCase):
    def test_api_failures_become_claude_errors(self):
        cases = [
            ("server error", claude_error(500, "api_error", "Internal server error"), 500, "Internal"),
            ("overloaded", claude_error(529, "overloaded_error", "Overloaded"), 529, "Overloaded"),
            ("bad key", claude_error(401, "authentication_error", "invalid x-api-key"), 401,
             "ANTHROPIC_API_KEY"),
            ("network", httpx.ConnectError("connection refused"), None, "could not reach Claude"),
        ]
        for name, reply, status, message in cases:
            with self.subTest(name):
                fake = self.fake(reply)
                with self.assertRaises(claude.ClaudeError) as ctx:
                    self.call()
                self.assertEqual(ctx.exception.status, status)
                self.assertIn(message, str(ctx.exception))
                self.assertIsNone(ctx.exception.usage)
                self.assertIsInstance(ctx.exception.__cause__, Exception)
                self.assertEqual(len(fake.requests), 1)

    def test_an_error_event_mid_stream(self):
        events = stream_events([text('{"a": ')])[:3]
        body = sse(events) + b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}\n\n'
        self.fake(httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body))
        with self.assertRaises(claude.ClaudeError) as ctx:
            self.call()
        self.assertIn("overloaded_error", str(ctx.exception))
        self.assertIsNone(ctx.exception.status)   # not the stream's HTTP 200

    def test_a_stream_that_breaks_off(self):
        first = sse(stream_events([text('{"a": 1}')])[:2])
        self.fake(httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 stream=BrokenStream(first)))
        with self.assertRaises(claude.ClaudeError) as ctx:
            self.call()
        self.assertIn("connection reset", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
