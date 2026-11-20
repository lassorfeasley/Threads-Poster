"""Onboarding (app/onboarding.py): state, saving steps, the helper's tool loop,
and who gets sent to it. Run with:  python -m unittest tests.test_onboarding -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app import config, onboarding
from app.web import access


class TempWorkspace(unittest.TestCase):
    brand = "topic: ''\n"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "brand.yaml").write_text(self.brand)
        (self.dir / "keywords.yaml").write_text("keywords: []\n")
        for p in (mock.patch.object(config, "CONFIG_DIR", self.dir),
                  mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.dir / "engine")})):
            p.start()
            self.addCleanup(p.stop)


class StateTest(TempWorkspace):
    def test_new_account_is_pending(self):
        state = onboarding.load()
        self.assertEqual(state["status"], "pending")
        self.assertEqual(onboarding.current_step(state), "brand")
        self.assertFalse(onboarding.is_done())

    def test_save_step_writes_settings_and_advances(self):
        onboarding.save({**onboarding.load(), "drafts": {"name": "Core", "mission": "x"}})
        state = onboarding.save_step("brand", {"name": " Core Studio ",
                                               "cta_url": "https://www.core.studio/"})
        brand = config.load_brand()
        self.assertEqual((brand["name"], brand["cta_url"]), ("Core Studio", "core.studio"))
        self.assertEqual(state["steps_done"], ["brand"])
        self.assertEqual(state["drafts"], {"mission": "x"})   # only this screen's drafts go
        self.assertEqual(onboarding.current_step(state), "about")

    def test_every_screen_is_short(self):
        for s in onboarding.STEPS:
            shown = [f for f in s["fields"] if f["kind"] != "brief"]   # briefs start folded
            self.assertTrue(1 <= len(shown) <= 3, s["key"])
        self.assertFalse(set(onboarding.DERIVED_FIELDS) & set(onboarding.ASKED_FIELDS))

    def test_throughline_screen(self):
        state = onboarding.save_step("throughline", {
            "format": "🗞️ In the news — fresh stories\n✏️ My own idea",
            "throughline": "Every conflict has a turning point", "off_limits": "no politics"})
        brand = config.load_brand()
        self.assertEqual(brand["throughline"], "Every conflict has a turning point")
        self.assertIn("In the news", brand["format"])
        self.assertNotIn("off_limits", brand)                      # kept for the helper only
        self.assertEqual(state["answers"]["off_limits"], "no politics")
        values, _ = onboarding.form_values(onboarding.load())
        self.assertEqual(values["off_limits"], "no politics")

    def test_format_cards(self):
        state = onboarding.load()
        cards = onboarding.format_cards(state, "")
        self.assertEqual(len(cards), len(onboarding.GENERIC_FORMATS))   # until the helper suggests some
        state["format_options"] = [{"emoji": "🧘", "title": "Teaching moments", "example": "one cue"}]
        cards = onboarding.format_cards(state, "🧘 Teaching moments — one cue\n✏️ Custom")
        self.assertEqual([(c["title"], c["on"]) for c in cards],
                         [("Teaching moments", True), ("Custom", True)])
        self.assertEqual(onboarding.parse_format(cards[0]["line"])[0]["title"], "Teaching moments")

    def test_skip_step_saves_nothing(self):
        state = onboarding.skip_step("brand")
        self.assertEqual(state["steps_done"], ["brand"])
        self.assertEqual(config.load_brand()["name"], "")

    def test_sample_answers_cover_every_question(self):
        self.assertEqual(set(onboarding.SAMPLE_ANSWERS), set(onboarding.ASKED_FIELDS))

    def test_keywords_step(self):
        onboarding.save_step("keywords", {"keywords": "Reformer\nreformer, mat pilates\n\n core "})
        self.assertEqual(config.load_keywords(), ["core", "mat pilates", "reformer"])

    def test_drafts_show_over_saved_values(self):
        config.save_brand({"topic": "pilates"})
        onboarding.save({**onboarding.load(), "drafts": {"topic": "pilates", "audience": "new movers"}})
        values, drafted = onboarding.form_values(onboarding.load())
        self.assertEqual(values["audience"], "new movers")
        self.assertEqual(drafted, {"audience"})   # a draft equal to the saved value isn't news

    def test_reset_and_done(self):
        onboarding.mark_done()
        self.assertTrue(onboarding.is_done())
        onboarding.reset()
        self.assertFalse(onboarding.is_done())


class ExistingAccountTest(TempWorkspace):
    brand = "topic: climate change\n"

    def test_set_up_before_onboarding_counts_as_done(self):
        self.assertTrue(onboarding.is_done())


def _resp(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class _Block(SimpleNamespace):
    def to_dict(self):
        return {k: v for k, v in vars(self).items()}


class HelperTest(TempWorkspace):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(onboarding.spend, "record")
        p.start()
        self.addCleanup(p.stop)

    def _client(self, *responses):
        """A fake client that records each request as it was sent (the
        message list keeps growing after the call)."""
        client = mock.Mock()
        client.sent = []
        queue = list(responses)

        def create(**kwargs):
            client.sent.append(json.loads(json.dumps(kwargs)))
            return queue.pop(0)

        client.messages.create.side_effect = create
        return client

    def test_tool_loop_drafts_fields(self):
        client = self._client(
            _resp("tool_use",
                  _Block(type="text", text="Got it."),
                  _Block(type="tool_use", id="t1", name="fill_fields",
                         input={"fields": [{"field": "mission", "value": "pilates"},
                                           {"field": "keywords", "value": "reformer\nmat"},
                                           {"field": "nonsense", "value": "x"}]})),
            _resp("end_turn", _Block(type="text", text="Anything you never want posted?")),
        )
        out = onboarding.chat("brand", "We run a pilates studio", client=client)
        self.assertEqual(out["fills"], {"mission": "pilates", "keywords": "reformer\nmat"})
        self.assertIn("Anything you never", out["reply"])

        second_call = client.sent[1]
        tool_result = second_call["messages"][-1]["content"][0]
        self.assertEqual((tool_result["type"], tool_result["tool_use_id"]), ("tool_result", "t1"))
        first_user = second_call["messages"][0]["content"]
        self.assertIn("<form_state>", first_user[0]["text"])
        self.assertEqual(first_user[-1]["text"], "We run a pilates studio")

        state = onboarding.load()
        self.assertEqual(state["drafts"]["mission"], "pilates")
        self.assertEqual([m["role"] for m in state["transcript"]], ["user", "assistant"])
        self.assertEqual(len(state["messages"]), 4)   # user, assistant, tool result, assistant

    def test_refusal_leaves_conversation_unchanged(self):
        client = self._client(_resp("refusal"))
        out = onboarding.chat("brand", "hello", client=client)
        self.assertTrue(out["error"])
        self.assertEqual(onboarding.load()["messages"], [])

    def test_background_read_keeps_answers_already_given(self):
        onboarding.save_step("brand", {"name": "Core"})
        onboarding.save_step("about", {"mission": "Mine"})   # answered while it was reading
        client = self._client(
            _resp("tool_use", _Block(type="tool_use", id="t1", name="fill_fields",
                                     input={"fields": [{"field": "mission", "value": "Theirs"},
                                                       {"field": "audience", "value": "Movers"}]})),
            _resp("end_turn", _Block(type="text", text="Done.")),
        )
        onboarding.read_brand("Core", "core.example", client=client)
        state = onboarding.load()
        self.assertEqual(state["drafts"], {"audience": "Movers"})
        self.assertIsNone(state["busy"])
        self.assertIn("core.example", client.sent[0]["messages"][0]["content"][-1]["text"])

    def test_refresh_redrafts_keywords_for_the_chosen_throughline(self):
        onboarding.save_step("throughline", {"throughline": "Strength is learned"})
        client = self._client(
            _resp("tool_use", _Block(type="tool_use", id="t1", name="fill_fields", input={"fields": [
                {"field": "keywords", "value": "pilates\nstroke recovery"},
                {"field": "throughline", "value": "something else"}]})),
            _resp("end_turn", _Block(type="text", text="Updated.")),
        )
        onboarding.refresh_for_throughline("Strength is learned", client=client)
        state = onboarding.load()
        self.assertEqual(state["drafts"], {"keywords": "pilates\nstroke recovery"})  # their throughline stands
        self.assertIsNone(state["busy"])
        self.assertIn("Strength is learned", client.sent[0]["messages"][0]["content"][-1]["text"])

    def test_caption_brief_is_saved_with_the_throughline(self):
        onboarding.save_step("throughline", {"throughline": "x", "caption_brief": "Our app names each need."})
        self.assertEqual(config.load_brand()["caption_brief"], "Our app names each need.")

    def test_finish_writes_the_rules_and_completes(self):
        config.save_brand({"name": "Core"})
        fields = {f: f"{f} text" for f in onboarding.DERIVED_FIELDS}
        client = self._client(_resp("end_turn", _Block(type="text", text=json.dumps(fields))))
        onboarding.finish(client=client)
        brand = config.load_brand()
        self.assertEqual(brand["relevance_rules"], "relevance_rules text")
        self.assertEqual(brand["topic"], "topic text")             # written, not asked
        self.assertEqual(brand["name"], "Core")
        self.assertTrue(onboarding.is_done())
        fmt = client.sent[0]["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")

    def test_finish_completes_even_when_the_rules_fail(self):
        client = self._client(_resp("end_turn", _Block(type="text", text="not json")))
        onboarding.finish(client=client)
        self.assertTrue(onboarding.is_done())

    def test_uploads(self):
        self.assertEqual(onboarding.attachment_block("brief.PDF", b"%PDF")["type"], "document")
        self.assertEqual(onboarding.attachment_block("a.png", b"x")["type"], "image")
        self.assertEqual(onboarding.attachment_block("notes.md", b"hi")["source"]["data"], "hi")
        with self.assertRaises(onboarding.UploadError):
            onboarding.attachment_block("clip.mp4", b"x")
        with self.assertRaises(onboarding.UploadError):
            onboarding.attachment_block("big.pdf", b"x" * (onboarding.MAX_UPLOAD_BYTES + 1))

    def test_helper_suggests_concept_ideas(self):
        client = self._client(
            _resp("tool_use", _Block(type="tool_use", id="t1", name="suggest_formats",
                                     input={"options": [{"emoji": "🥊", "title": "Conflict caught on camera",
                                                         "example": "a heated standoff"}]})),
            _resp("end_turn", _Block(type="text", text="Here are some ideas.")),
        )
        out = onboarding.chat("throughline", "We do AI conflict mediation", client=client)
        self.assertEqual(out["format_options"][0]["title"], "Conflict caught on camera")
        self.assertEqual(onboarding.load()["format_options"][0]["emoji"], "🥊")

    def test_helper_offers_three_options_per_answer(self):
        opts = lambda *texts: [{"label": t[:1], "text": t, "note": ""} for t in texts]
        client = self._client(
            _resp("tool_use", _Block(type="tool_use", id="t1", name="suggest_options", input={
                "suggestions": [
                    {"field": "throughline", "options": [
                        {"label": "Wide net", "text": "Pilates in the news and through history", "note": "a"},
                        {"label": "Focused", "text": "Strength is learned", "note": "b"},
                        {"label": "Niche", "text": "Transformation through mind and body", "note": "c"},
                        {"label": "Extra", "text": "a fourth is dropped", "note": ""}]},
                    {"field": "audience", "options": opts("A", "B", "")},
                    {"field": "name", "options": opts("not an option field")}]})),
            _resp("end_turn", _Block(type="text", text="Options are up.")),
        )
        out = onboarding.chat("throughline", "pilates", client=client)
        self.assertEqual(set(out["options"]), {"throughline", "audience"})
        self.assertEqual([o["label"] for o in out["options"]["throughline"]],
                         ["Wide net", "Focused", "Niche"])
        self.assertEqual([o["text"] for o in out["options"]["audience"]], ["A", "B"])   # blanks go
        saved = onboarding.load()["options"]
        self.assertEqual(saved["throughline"][2]["text"], "Transformation through mind and body")

    def test_tool_schema_is_strict_and_covers_every_field(self):
        schema = onboarding.FILL_TOOL["input_schema"]
        self.assertFalse(schema["additionalProperties"])
        enum = schema["properties"]["fields"]["items"]["properties"]["field"]["enum"]
        self.assertEqual(set(enum), set(onboarding.ALL_FIELDS))
        json.dumps(onboarding.FILL_TOOL)


class GateTest(unittest.TestCase):
    def test_what_onboarding_redirects(self):
        for path in ("/inbox", "/calendar", "/brand", "/settings", "/"):
            self.assertTrue(access.onboarding_applies(path), path)
        for path in ("/welcome", "/welcome/format", "/static/style.css", "/logout",
                     "/account/password", "/threads/connect", "/facebook/callback"):
            self.assertFalse(access.onboarding_applies(path), path)

    def test_customer_may_onboard_but_not_reset(self):
        self.assertTrue(access.customer_may("/welcome/account"))
        self.assertTrue(access.customer_may("/welcome/chat", "POST"))
        self.assertFalse(access.customer_may("/welcome/reset", "POST"))
        self.assertFalse(access.customer_may("/welcome/skip", "POST"))


if __name__ == "__main__":
    unittest.main()


class SourcesTest(TempWorkspace):
    def setUp(self):
        super().setUp()
        from app import onboarding_sources as src
        self.src = src
        config.save_keywords(["pilates", "reformer"])

    def _found(self, n, kind, hits=3):
        return [{"channel_id": f"{kind}{i}", "uploads_playlist_id": f"UU{kind}{i}", "title": f"{kind} {i}",
                 "url": f"https://www.youtube.com/@{kind}{i}", "kind": kind, "why": "", "hits": hits,
                 "total": 50, "example": "x", "on": True} for i in range(n)]

    def test_sample_is_a_cross_section(self):
        state = {"channels": self._found(10, "News") + self._found(3, "Archive")}
        kinds = [c["kind"] for c in self.src.sample(state, size=6)]
        self.assertEqual(kinds.count("Archive"), 3)          # not just the first kind

    def test_discover_keeps_matching_channels_and_skips_removed(self):
        proposals = [{"name": n, "url": f"https://www.youtube.com/@{n}", "kind": "News", "why": ""}
                     for n in ("good", "empty", "removed")]
        self.src.mutate(lambda s: s.update(removed=["https://www.youtube.com/@removed"]))

        def check(p, keywords):
            self.assertEqual(keywords, ["pilates", "reformer"])
            return None if "empty" in p["url"] else self._found(1, p["url"].rsplit("@", 1)[1])[0]

        with mock.patch.object(self.src, "propose", return_value=(proposals, [])), \
                mock.patch.object(self.src, "search", return_value=[]), \
                mock.patch.object(self.src, "check", side_effect=check):
            self.src.discover({"name": "Core"})
        state = self.src.load()
        self.assertEqual(state["status"], "done")
        self.assertEqual([c["title"] for c in state["channels"]], ["good 0"])
        self.assertEqual(state["total"], 2)                   # the removed one was never checked

    def test_discover_needs_keywords(self):
        config.save_keywords([])
        self.src.discover({"name": "Core"})
        self.assertEqual(self.src.load()["status"], "error")

    def test_toggle_remembers_removals(self):
        self.src.mutate(lambda s: s.update(channels=self._found(2, "News")))
        self.src.toggle("News0", False)
        state = self.src.load()
        self.assertEqual(len(self.src.chosen(state)), 1)
        self.assertIn("https://www.youtube.com/@news0", state["removed"])
        self.src.toggle("News0", True)
        self.assertEqual(self.src.load()["removed"], [])


class RouteTriggersTest(TempWorkspace):
    """Saving a screen starts the right background work (and skipping doesn't)."""

    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.web import onboarding as page
        self.page = page
        app = FastAPI()
        app.include_router(page.router)
        self.client = TestClient(app)
        p = mock.patch.object(page, "_background")
        self.bg = p.start()
        self.addCleanup(p.stop)

    def started(self):
        return [c.args[0].__name__ for c in self.bg.call_args_list]

    def test_saving_keywords_starts_the_source_search(self):
        r = self.client.post("/welcome/keywords", data={"keywords": "pilates\nreformer"},
                             follow_redirects=False)
        self.assertEqual(r.headers["location"], "/welcome/subtitles")
        self.assertEqual(self.started(), ["_run_sources"])

    def test_a_new_throughline_redrafts_the_keywords(self):
        self.client.post("/welcome/throughline", data={"throughline": "Strength is learned"},
                         follow_redirects=False)
        self.assertEqual(self.started(), ["refresh_for_throughline"])

    def test_skipping_starts_nothing(self):
        for key in ("brand", "throughline", "keywords"):
            self.client.post(f"/welcome/{key}/skip", follow_redirects=False)
        self.assertEqual(self.started(), [])
