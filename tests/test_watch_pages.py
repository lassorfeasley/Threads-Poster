"""Tests for watch pages (app/watch_pages.py) and how the first reply uses them.

Run with:  python -m unittest tests.test_watch_pages -v

Nothing here talks to YouTube, Renewables.org or Threads. HTTP is mocked at
``requests`` in app.watch_pages, publish_text_reply is mocked in
app.publishing, and the database is a throwaway in-memory SQLite.
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config, publishing, watch_pages
from app.models import Base, Candidate, Channel, Cut, ThreadsPost
from app.threads_api import ThreadsError

SITE = "https://renewables.org"
STATIC = "Take climate action 👉 Renewables.org — fund solar, create 5.5x the carbon impact, get repaid"
WATCH = "Watch the full segment 👉 {watch_url}"


class FakeSettings(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def settings(**overrides):
    base = {"watch_pages.enabled": True, "watch_pages.site_url": SITE,
            "watch_pages.lead_in_seconds": 2}
    base.update({f"watch_pages.{k}": v for k, v in overrides.items()})
    return FakeSettings(base)


def response(status=200, text="", body=None):
    resp = mock.Mock()
    resp.status_code = status
    resp.text = text
    resp.json.return_value = body or {}
    return resp


class PureHelpersTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(watch_pages.slugify("Heat Wave: 110°F in Phoenix — again!"),
                         "heat-wave-110f-in-phoenix-again")
        self.assertEqual(watch_pages.slugify("Café São Paulo"), "cafe-sao-paulo")
        self.assertEqual(watch_pages.slugify("!!!"), "")

    def test_slugify_cuts_on_word_boundary(self):
        slug = watch_pages.slugify("word " * 30)
        self.assertLessEqual(len(slug), 48)
        self.assertFalse(slug.endswith("-"))
        self.assertTrue(all(part == "word" for part in slug.split("-")))

    def test_slug_matches_backend_pattern(self):
        post = ThreadsPost(id=42)
        for title in ("Normal title", "", "  ", "ÀÉÎ ÕÜ", "a" * 200):
            slug = watch_pages.page_slug(post, title)
            self.assertRegex(slug, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
            self.assertTrue(slug.endswith("-42"))
            self.assertLessEqual(len(slug), 120)

    def test_clip_start_uses_earliest_segment_minus_lead_in(self):
        cut = Cut(trim_segments=json.dumps([{"start": 95.7, "end": 120}, {"start": 61.2, "end": 80}]))
        self.assertEqual(watch_pages.clip_start_seconds(cut, 2), 59)

    def test_clip_start_never_negative_and_tolerates_junk(self):
        self.assertEqual(watch_pages.clip_start_seconds(
            Cut(trim_segments=json.dumps([{"start": 1, "end": 5}])), 2), 0)
        self.assertEqual(watch_pages.clip_start_seconds(Cut(trim_segments="not json")), 0)
        self.assertEqual(watch_pages.clip_start_seconds(Cut(trim_segments="[]")), 0)
        self.assertEqual(watch_pages.clip_start_seconds(None), 0)



class FirstReplyTextTest(unittest.TestCase):
    CFG = {"enabled": True, "watch_text": WATCH, "fallback_text": STATIC}

    def test_watch_text_when_page_is_up(self):
        self.assertEqual(publishing.first_reply_text(self.CFG, f"{SITE}/watch/heat-wave-7"),
                         "Watch the full segment 👉 renewables.org/watch/heat-wave-7")

    def test_fallback_when_no_page(self):
        self.assertEqual(publishing.first_reply_text(self.CFG, ""), STATIC)

    def test_fallback_when_no_watch_text(self):
        cfg = dict(self.CFG, watch_text="")
        self.assertEqual(publishing.first_reply_text(cfg, f"{SITE}/watch/x-1"), STATIC)


class LoadFirstReplyTest(unittest.TestCase):
    def load(self, data):
        with mock.patch.object(config, "_first_reply_from_db", return_value=data):
            return config.load_first_reply()

    def test_two_text_settings(self):
        self.assertEqual(self.load({"enabled": True, "watch_text": WATCH, "fallback_text": STATIC}),
                         {"enabled": True, "watch_text": WATCH, "fallback_text": STATIC})

    def test_single_text_settings_carry_over(self):
        # Saved before the two-text version: the old line is the fallback, and
        # its Renewables.org mention becomes the link in the watch text.
        cfg = self.load({"enabled": True, "text": STATIC, "mode": "static",
                         "attribution_enabled": True, "delay_minutes": 0})
        self.assertEqual(cfg["fallback_text"], STATIC)
        self.assertEqual(cfg["watch_text"], STATIC.replace("Renewables.org", "{watch_url}"))

    def test_single_text_without_mention_has_no_watch_text(self):
        cfg = self.load({"enabled": True, "text": "Footage courtesy of KTST."})
        self.assertEqual(cfg["watch_text"], "")


class DbCase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, future=True)()
        self.channel = Channel(call_sign="KTST", channel_title="KTST News 4",
                               url="https://example.com/c")
        self.session.add(self.channel)
        self.session.flush()
        self.patches = [
            mock.patch.object(watch_pages, "load_settings", return_value=settings()),
            mock.patch.object(watch_pages, "env", side_effect=lambda name, default="":
                              "secret" if name == watch_pages.TOKEN_ENV else default),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def post(self, *, video_id="dQw4w9WgXcQ", title="Record heat strains the grid",
             segments=None, first_party=False) -> ThreadsPost:
        self.channel.first_party = first_party
        cand = Candidate(video_id=video_id, channel_pk=self.channel.id, title=title,
                         url="u", published_at=dt.datetime(2026, 11, 10, 14, tzinfo=dt.timezone.utc))
        self.session.add(cand)
        self.session.flush()
        cut = Cut(candidate_pk=cand.id,
                  trim_segments=json.dumps(segments or [{"start": 30, "end": 60}]))
        self.session.add(cut)
        self.session.flush()
        post = ThreadsPost(candidate_pk=cand.id, cut_pk=cut.id, status="published",
                           threads_media_id="m1", published_at=dt.datetime.now(dt.timezone.utc))
        self.session.add(post)
        self.session.flush()
        return post

    def http(self, oembed=200, put=200, page=200, author="KTST News 4 (YouTube)"):
        """Patch requests in app.watch_pages; returns the mocks to inspect."""
        get = mock.Mock(side_effect=lambda url, **kw:
                        response(oembed, body={"author_name": author}) if "oembed" in url
                        else response(page))
        put_mock = mock.Mock(return_value=response(put, "nope" if put != 200 else ""))
        p1 = mock.patch.object(watch_pages.requests, "get", get)
        p2 = mock.patch.object(watch_pages.requests, "put", put_mock)
        p1.start(), p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        return get, put_mock


class EnsureWatchPageTest(DbCase):
    def test_happy_path_publishes_then_verifies(self):
        post = self.post(segments=[{"start": 31.9, "end": 50}])
        get, put = self.http()
        url = watch_pages.ensure_watch_page(post)
        slug = f"record-heat-strains-the-grid-{post.id}"
        self.assertEqual(url, f"{SITE}/watch/{slug}")
        self.assertEqual((post.watch_page_slug, post.watch_page_url, post.watch_page_error),
                         (slug, url, ""))
        put_url = put.call_args.args[0]
        self.assertEqual(put_url, f"{SITE}/api/internal/watch-pages/{slug}")
        body = put.call_args.kwargs["json"]
        self.assertEqual(body["youtube_video_id"], "dQw4w9WgXcQ")
        self.assertEqual(body["start_seconds"], 29)
        # YouTube's own channel name wins over our channel record.
        self.assertEqual(body["source_name"], "KTST News 4 (YouTube)")
        self.assertEqual(body["source_published_at"], "2026-11-10T14:00:00+00:00")
        self.assertEqual(put.call_args.kwargs["headers"], {"Authorization": "Bearer secret"})
        # Page verified after publishing.
        self.assertEqual(get.call_args.args[0], url)

    def test_credit_falls_back_to_channel_record(self):
        post = self.post()
        _, put = self.http(author="")
        watch_pages.ensure_watch_page(post)
        self.assertEqual(put.call_args.kwargs["json"]["source_name"], "KTST News 4")

    def test_pasted_url_credits_the_real_channel(self):
        self.channel.channel_title = "Pasted URLs"
        post = self.post()
        _, put = self.http(author="Associated Press")
        watch_pages.ensure_watch_page(post)
        self.assertEqual(put.call_args.kwargs["json"]["source_name"], "Associated Press")

    def test_api_url_override(self):
        post = self.post()
        _, put = self.http()
        with mock.patch.object(watch_pages, "load_settings",
                               return_value=settings(api_url="https://api.example.com/")):
            watch_pages.ensure_watch_page(post)
        self.assertTrue(put.call_args.args[0].startswith("https://api.example.com/api/internal/"))

    def test_retry_reuses_slug(self):
        post = self.post()
        post.watch_page_slug = "kept-slug-9"
        self.http()
        self.assertTrue(watch_pages.ensure_watch_page(post).endswith("/watch/kept-slug-9"))

    def test_disabled_does_nothing(self):
        post = self.post()
        get, put = self.http()
        with mock.patch.object(watch_pages, "load_settings",
                               return_value=settings(enabled=False)):
            self.assertEqual(watch_pages.ensure_watch_page(post), "")
        get.assert_not_called()
        put.assert_not_called()
        self.assertEqual(post.watch_page_error or "", "")

    def assertNoPage(self, post, fragment):
        self.assertEqual(watch_pages.ensure_watch_page(post), "")
        self.assertEqual(post.watch_page_url, "")
        self.assertIn(fragment, post.watch_page_error)

    def test_embedding_disabled(self):
        post = self.post()
        _, put = self.http(oembed=401)
        self.assertNoPage(post, "turned off embedding")
        put.assert_not_called()

    def test_video_removed(self):
        post = self.post()
        _, put = self.http(oembed=404)
        self.assertNoPage(post, "private or has been removed")
        put.assert_not_called()

    def test_backend_refuses(self):
        self.http(put=401)
        self.assertNoPage(self.post(), "refused it (401")

    def test_page_not_loading_withholds_link(self):
        self.http(page=500)
        self.assertNoPage(self.post(), "page answered 500")

    def test_first_party_skipped(self):
        _, put = self.http()
        self.assertNoPage(self.post(first_party=True), "brand's own footage")
        put.assert_not_called()

    def test_non_youtube_source_skipped(self):
        _, put = self.http()
        self.assertNoPage(self.post(video_id="local-upload-1"), "isn't a YouTube video")
        put.assert_not_called()

    def test_missing_token_skipped(self):
        post = self.post()
        _, put = self.http()
        with mock.patch.object(watch_pages, "env", return_value=""):
            self.assertNoPage(post, "WATCH_PAGES_SERVICE_TOKEN")
        put.assert_not_called()

    def test_network_error_never_raises(self):
        post = self.post()
        with mock.patch.object(watch_pages.requests, "get",
                               side_effect=watch_pages.requests.ConnectionError("down")):
            self.assertNoPage(post, "couldn't reach YouTube")


class FirstReplyIntegrationTest(DbCase):
    CFG = {"enabled": True, "watch_text": WATCH, "fallback_text": STATIC}

    def setUp(self):
        super().setUp()
        self.cfg = dict(self.CFG)
        p = mock.patch.object(publishing, "load_first_reply", side_effect=lambda: self.cfg)
        p.start()
        self.addCleanup(p.stop)
        self.reply = mock.Mock(return_value={"media_id": "r1"})
        p = mock.patch.object(publishing, "publish_text_reply", self.reply)
        p.start()
        self.addCleanup(p.stop)

    def test_reply_links_to_watch_page_and_pins_card(self):
        post = self.post()
        self.http()
        self.assertTrue(publishing.maybe_post_first_reply(self.session, post))
        text = self.reply.call_args.args[0]
        url = f"{SITE}/watch/record-heat-strains-the-grid-{post.id}"
        self.assertEqual(text, f"Watch the full segment 👉 renewables.org/watch/record-heat-strains-the-grid-{post.id}")
        self.assertEqual(self.reply.call_args.kwargs, {"link_attachment": url})
        self.assertEqual(post.first_reply_text, text)
        self.assertEqual(post.watch_page_url, url)

    def test_reply_uses_fallback_text_when_no_page(self):
        post = self.post()
        self.http(oembed=401)
        self.assertTrue(publishing.maybe_post_first_reply(self.session, post))
        self.assertEqual(self.reply.call_args.args, (STATIC, "m1"))
        self.assertEqual(self.reply.call_args.kwargs, {})
        self.assertEqual(post.first_reply_text, STATIC)
        self.assertIn("embedding", post.watch_page_error)

    def test_no_watch_text_publishes_no_page(self):
        self.cfg["watch_text"] = ""
        post = self.post()
        get, put = self.http()
        self.assertTrue(publishing.maybe_post_first_reply(self.session, post))
        put.assert_not_called()
        self.assertEqual(self.reply.call_args.args, (STATIC, "m1"))

    def test_switched_off_posts_nothing_and_says_why(self):
        self.cfg["enabled"] = False
        post = self.post()
        self.http()
        self.assertFalse(publishing.maybe_post_first_reply(self.session, post))
        self.reply.assert_not_called()
        self.assertIn("switched off", post.first_reply_error)

    def test_posting_by_hand_overrides_the_switch(self):
        self.cfg["enabled"] = False
        post = self.post()
        self.http()
        self.assertTrue(publishing.maybe_post_first_reply(self.session, post, force=True))
        self.assertIn("renewables.org/watch/", self.reply.call_args.args[0])

    def test_brand_promo_gets_no_reply(self):
        post = self.post(first_party=True)
        self.http()
        self.assertFalse(publishing.maybe_post_first_reply(self.session, post))
        self.reply.assert_not_called()
        self.assertTrue(post.first_reply_skipped)
        self.assertIn("own promo", post.first_reply_error)

    def test_rejected_link_attachment_retries_without_it(self):
        post = self.post()
        self.http()
        self.reply.side_effect = [ThreadsError("Invalid parameter", code=100),
                                  {"media_id": "r2"}]
        self.assertTrue(publishing.maybe_post_first_reply(self.session, post))
        self.assertEqual(self.reply.call_count, 2)
        self.assertIn("link_attachment", self.reply.call_args_list[0].kwargs)
        self.assertEqual(self.reply.call_args_list[1].kwargs, {})
        self.assertIn("renewables.org/watch/", self.reply.call_args_list[1].args[0])
        self.assertEqual(post.first_reply_id, "r2")

    def test_other_threads_errors_are_not_retried(self):
        post = self.post()
        self.http()
        self.reply.side_effect = ThreadsError("rate limited", code=4)
        self.assertFalse(publishing.maybe_post_first_reply(self.session, post))
        self.assertEqual(self.reply.call_count, 1)
        self.assertIn("rate limited", post.first_reply_error)


if __name__ == "__main__":
    unittest.main()
