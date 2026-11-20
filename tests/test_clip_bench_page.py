"""The clip bench judging page (app/web/clip_bench.py) on a bare FastAPI app.

Never imports app.web.main (that boots the scheduler and touches the DB): the
router is mounted on a fresh app whose ``state.templates`` is a Jinja
environment over the real templates, with stubs for the globals base.html
calls. bench.py is mocked throughout — these tests are about the page itself:
that it stays blind before the reveal, shows the report after, serves media
only from inside the bench tree, and validates and maps errors on its routes.

Run with:  python -m pytest tests/test_clip_bench_page.py -q
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from app.clip_engine import bench
from app.web import clip_bench

TEMPLATES_DIR = Path(clip_bench.__file__).parent / "templates"


def make_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals.update({
        "brand_name": lambda: "Test Brand",
        "ws_current": lambda: {"slug": "test", "label": "Test", "accent": ""},
        "brand_logo": lambda: "",
        "workspace_options": lambda: [],
        "attention_count": lambda: 0,
        "threads_authenticated": lambda: True,
        "static_v": "1",
        "thumb_for": lambda candidate: "",
        "category_options": lambda: [],
        "category_by_slug": lambda slug: None,
        "digest_meta": lambda *a, **k: {},
        "default_app_name": "Clip Monitor",
        "is_admin": lambda request: True,
    })
    return templates


def run_entry(name="test-1", **kw) -> dict:
    entry = {"name": name, "arms": ["gemini", "transcript"], "videos": 60, "judged": 12,
             "total": 58, "revealed": False, "seed": "test", "mode": "sampled",
             "created_at": "2026-11-18T10:00:00Z"}
    entry.update(kw)
    return entry


# What bench.next_item might hand back: everything the page must NOT show.
LEAKY_PAIR = {
    "item_id": "abc123def456", "kind": "ceiling", "video_id": "SECRETVID01",
    "title": "Secret Source Title", "arm": "gemini",
    "a": {"arm": "gemini", "render": "renders/abc123def456-a.mp4"},
    "b": {"arm": "reference-cut", "render": "renders/abc123def456-b.mp4"},
}

# Just enough DOM for the judging page's own script to run in node. The
# recognised box comes back ticked, as Firefox restores it after the reload
# that follows saving the previous item.
JUDGE_PAGE_STUB = """
function element(fields) {
  return Object.assign({ hidden: false, textContent: '', dataset: {}, handlers: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(type, fn) { this.handlers[type] = fn; },
    querySelectorAll: () => [] }, fields);
}
const choice = element({ dataset: { choice: 'a' } });
const elements = {
  'cb-judge': element({ dataset: { run: 'test-1', item: 'abc123def456', solo: '0' },
                        querySelectorAll: sel => (sel === '.cb-choice' ? [choice] : []) }),
  'cb-reason': element({}), 'cb-reason-q': element({}), 'cb-confirm': element({}),
  'cb-recognised': element({ checked: true }),
  'cb-save': element({}), 'cb-cancel': element({}),
};
const document = { getElementById: id => elements[id] || null, addEventListener() {},
                   querySelector: () => null };
let posted = null, reloaded = false;
async function fetch(url, opts) {
  posted = String(opts.body);
  return { ok: true, status: 200, json: async () => ({ ok: true }) };
}
const location = { reload() { reloaded = true; } };
function alert(message) { throw new Error('alert: ' + message); }
"""


class PageTestCase(unittest.TestCase):
    mock_bench = True   # False: the page runs against the real bench.py

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bench_dir = self.root / "clipbench"
        env = mock.patch.dict(os.environ, {"CLIPBENCH_DIR": str(self.bench_dir),
                                           "CLIP_ENGINE_DIR": str(self.root / "engine")})
        env.start()
        self.addCleanup(env.stop)

        self.templates = make_templates()
        app = FastAPI()
        app.state.templates = self.templates
        app.include_router(clip_bench.router)
        self.client = TestClient(app)
        if not self.mock_bench:
            return

        self.runs = [run_entry()]
        self.list_runs = self.patch("list_runs", side_effect=lambda: [dict(r) for r in self.runs])
        self.next_item = self.patch("next_item", return_value=dict(LEAKY_PAIR))
        self.report = self.patch("report", return_value="ARM TABLE\ngemini  wins 31  <b>x</b>")
        self.judge = self.patch("judge", return_value={"ok": True})
        self.reveal = self.patch("reveal", return_value={"revealed_at": "2026-11-18T12:00:00Z"})

    def patch(self, name, **kw):
        patcher = mock.patch.object(bench, name, create=True, **kw)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def write_render(self, rel: str, data: bytes = b"\x00\x00\x00\x18ftypmp42fake") -> Path:
        path = self.bench_dir / "test-1" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path


class IndexTest(PageTestCase):
    def test_lists_every_run_with_its_progress_and_state(self):
        self.runs = [
            run_entry(),
            run_entry("dev-2", arms=[{"label": "gemini+editor_thinking=HIGH"}],
                      videos=["a", "b", "c"], judged=3, total=3, seed="dev", dev=True),
            run_entry("old", revealed=None, revealed_at="2026-11-01T00:00:00Z",
                      stopped={"reason": "max_usd", "videos_done": 31}),
            run_entry("fresh", judged=0, total=0),
        ]
        r = self.client.get("/clip-bench")
        self.assertEqual(r.status_code, 200)
        html = r.text
        self.assertIn('href="/clip-bench?run=test-1"', html)
        self.assertIn("12 / 58", html)
        self.assertIn("gemini+editor_thinking=HIGH", html)
        self.assertIn("Ready to reveal", html)      # dev-2: everything judged
        self.assertIn("Revealed", html)             # old: revealed_at set
        self.assertIn("stopped early", html)        # old: stopped at its cap
        self.assertIn("no pairs yet", html)         # fresh
        self.assertEqual(html.count(">dev</span>"), 1)   # only the dev run is flagged
        self.next_item.assert_not_called()

    def test_empty_state_says_how_to_start(self):
        self.runs = []
        r = self.client.get("/clip-bench")
        self.assertEqual(r.status_code, 200)
        self.assertIn("No bench runs yet", r.text)
        self.assertIn("clipbench run --name", r.text)

    def test_a_broken_bench_module_is_an_error_page_not_a_crash(self):
        self.list_runs.side_effect = RuntimeError("manifest unreadable")
        r = self.client.get("/clip-bench")
        self.assertEqual(r.status_code, 500)
        self.assertIn("couldn", r.text)
        self.assertIn("manifest unreadable", r.text)

    def test_unknown_or_malformed_run_names_are_404(self):
        for name in ("nope", "../etc", ".hidden", "a b"):
            with self.subTest(name=name):
                r = self.client.get("/clip-bench", params={"run": name})
                self.assertEqual(r.status_code, 404)
                self.assertIn("There is no bench run named", r.text)
        self.next_item.assert_not_called()


class JudgingViewTest(PageTestCase):
    def test_a_pair_shows_only_two_players_and_the_choices(self):
        r = self.client.get("/clip-bench", params={"run": "test-1"})
        self.assertEqual(r.status_code, 200)
        html = r.text
        self.assertIn('src="/clip-bench/test-1/media/abc123def456/a"', html)
        self.assertIn('src="/clip-bench/test-1/media/abc123def456/b"', html)
        for text in ("A is better", "B is better", "Tie", "Neither",
                     "I recognised one of these as mine", "Reveal results",
                     "12 / 58 judged", 'data-reason="weak_opening"', 'data-reason="no_sense"'):
            self.assertIn(text, html)
        self.next_item.assert_called_once_with("test-1")

    def test_the_judging_view_is_blind(self):
        html = self.client.get("/clip-bench", params={"run": "test-1"}).text
        # Nothing about the arms, the pair's kind or the source video.
        for secret in ("SECRETVID01", "Secret Source Title", "ceiling", "reference-cut",
                       "gemini", "transcript", "renders/"):
            self.assertNotIn(secret, html)

    def test_the_recognised_box_is_not_restored_onto_the_next_item(self):
        # Saving reloads the page for the next item; Firefox restores a
        # checkbox's state across a reload unless the box opts out.
        html = self.client.get("/clip-bench", params={"run": "test-1"}).text
        box = re.search(r'<input[^>]*id="cb-recognised"[^>]*>', html).group(0)
        self.assertIn('autocomplete="off"', box)

    @unittest.skipUnless(shutil.which("node"), "node is needed to run the page's JavaScript")
    def test_a_tick_the_browser_restored_is_never_saved(self):
        html = self.client.get("/clip-bench", params={"run": "test-1"}).text
        script = next(s for s in re.findall(r"<script>(.*?)</script>", html, flags=re.S)
                      if "cb-judge" in s)
        harness = JUDGE_PAGE_STUB + script + """
choice.handlers.click();             // "A is better"
elements['cb-save'].handlers.click();
setTimeout(() => process.stdout.write(JSON.stringify({ posted, reloaded })), 20);
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge.js"
            path.write_text(harness)
            result = subprocess.run(["node", str(path)], capture_output=True, text=True,
                                    timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        out = json.loads(result.stdout)
        self.assertTrue(out["reloaded"])
        posted = parse_qs(out["posted"], keep_blank_values=True)
        self.assertEqual(posted["choice"], ["a"])
        self.assertEqual(posted["recognised"], [""])

    def test_a_solo_item_shows_one_player_and_would_post(self):
        self.next_item.return_value = {"item_id": "s0l0", "kind": "solo", "arm": "gemini"}
        html = self.client.get("/clip-bench", params={"run": "test-1"}).text
        self.assertIn('src="/clip-bench/test-1/media/s0l0/a"', html)
        self.assertNotIn("/media/s0l0/b", html)
        self.assertIn("Would post", html)
        self.assertIn("Wouldn't post", html)
        self.assertNotIn("I recognised one of these", html)
        self.assertNotIn("A is better", html)

    def test_everything_judged_offers_the_reveal(self):
        self.runs = [run_entry(judged=58, total=58)]
        self.next_item.return_value = None
        html = self.client.get("/clip-bench", params={"run": "test-1"}).text
        self.assertIn("All 58 judged", html)
        self.assertIn('id="cb-reveal"', html)
        self.assertIn('data-left="0"', html)
        self.assertNotIn("<video", html)

    def test_a_run_without_pairs_says_how_to_make_them(self):
        self.runs = [run_entry(judged=0, total=0)]
        self.next_item.return_value = None
        html = self.client.get("/clip-bench", params={"run": "test-1"}).text
        self.assertIn("No pairs yet", html)
        self.assertIn("clipbench pairs --name test-1", html)

    def test_a_malformed_item_is_reported_not_rendered(self):
        self.next_item.return_value = {"item_id": "../../etc/passwd", "kind": "arms"}
        r = self.client.get("/clip-bench", params={"run": "test-1"})
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("<video", r.text)
        self.assertNotIn("All 58 judged", r.text)
        self.assertIn("malformed", r.text)

    def test_a_revealed_run_shows_the_report_and_no_judging(self):
        self.runs = [run_entry(revealed=True)]
        r = self.client.get("/clip-bench", params={"run": "test-1"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("ARM TABLE", r.text)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", r.text)   # report text is escaped
        self.assertNotIn('id="cb-reveal"', r.text)
        self.assertNotIn("<video", r.text)
        self.next_item.assert_not_called()
        self.report.assert_called_once_with("test-1")

    def test_a_failing_report_is_an_error_page_with_the_cli_fallback(self):
        self.runs = [run_entry(revealed=True)]
        self.report.side_effect = ValueError("results.json is missing")
        r = self.client.get("/clip-bench", params={"run": "test-1"})
        self.assertEqual(r.status_code, 500)
        self.assertIn("results.json is missing", r.text)
        self.assertIn("clipbench report --name test-1", r.text)

    def test_a_failing_next_item_is_an_error_page(self):
        self.next_item.side_effect = OSError("pairs.json is half-written")
        r = self.client.get("/clip-bench", params={"run": "test-1"})
        self.assertEqual(r.status_code, 500)
        self.assertIn("pairs.json is half-written", r.text)


class MediaTest(PageTestCase):
    def test_serves_the_render_with_nothing_arm_specific_in_the_response(self):
        path = self.write_render("renders/gemini-arm-abc.mp4")
        resolver = self.patch("media_path", return_value=str(path))
        r = self.client.get("/clip-bench/test-1/media/abc123def456/a")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, path.read_bytes())
        self.assertEqual(r.headers["content-type"], "video/mp4")
        self.assertEqual(r.headers["cache-control"], "no-cache")
        self.assertNotIn("content-disposition", r.headers)
        self.assertNotIn("gemini", " ".join(f"{k}: {v}" for k, v in r.headers.items()))
        resolver.assert_called_once_with("test-1", "abc123def456", "a")

    def test_supports_range_requests_for_seeking(self):
        path = self.write_render("renders/x.mp4", b"0123456789")
        self.patch("media_path", return_value=str(path))
        r = self.client.get("/clip-bench/test-1/media/abc123def456/b",
                            headers={"Range": "bytes=2-5"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.content, b"2345")

    def test_relative_paths_resolve_inside_the_run(self):
        self.write_render("renders/one.mp4", b"one")
        self.patch("media_path", return_value="renders/one.mp4")
        self.assertEqual(self.client.get("/clip-bench/test-1/media/it1/a").content, b"one")

    def test_never_serves_a_file_outside_the_bench_tree(self):
        outside = self.root / "secret.mp4"
        outside.write_bytes(b"secret")
        link = self.bench_dir / "test-1" / "renders" / "sneaky.mp4"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside)
        for raw in (str(outside), "../../secret.mp4", str(link), "/etc/hosts"):
            with self.subTest(raw=raw):
                self.patch("media_path", return_value=raw)
                r = self.client.get("/clip-bench/test-1/media/abc123def456/a")
                self.assertEqual(r.status_code, 404)
                self.assertNotEqual(r.content, b"secret")

    def test_rejects_bad_names_sides_and_missing_files(self):
        resolver = self.patch("media_path", return_value=str(self.bench_dir / "test-1" / "gone.mp4"))
        cases = [("/clip-bench/test-1/media/abc123def456/c", False),
                 ("/clip-bench/test-1/media/abc.def/a", False),
                 ("/clip-bench/bad%20name/media/abc123def456/a", False),
                 ("/clip-bench/test-1/media/abc123def456/a", True)]   # valid, but no file
        for url, reaches_bench in cases:
            with self.subTest(url=url):
                resolver.reset_mock()
                self.assertEqual(self.client.get(url).status_code, 404)
                self.assertEqual(resolver.called, reaches_bench)

    def test_a_resolver_error_is_a_404(self):
        self.patch("media_path", side_effect=RuntimeError("pairs.json corrupt"))
        self.assertEqual(self.client.get("/clip-bench/test-1/media/abc123def456/a").status_code, 404)


class JudgeRouteTest(PageTestCase):
    def post(self, **data):
        form = {"item_id": "abc123def456", "choice": "a", "recognised": "", "reason": ""}
        form.update(data)
        return self.client.post("/clip-bench/test-1/judge", data=form)

    def test_passes_a_normalized_judgment_to_the_bench(self):
        r = self.post(choice=" A ", recognised="on", reason="Boring")
        self.assertEqual(r.status_code, 200)
        self.judge.assert_called_once_with("test-1", "abc123def456", "a",
                                           recognised=True, reason="boring")

    def test_every_valid_choice_and_reason_is_accepted(self):
        choices = clip_bench.PAIR_CHOICES + clip_bench.SOLO_CHOICES
        reasons = [""] + [key for key, _ in clip_bench.REASONS]
        for choice in choices:
            for reason in reasons:
                with self.subTest(choice=choice, reason=reason):
                    self.assertEqual(self.post(choice=choice, reason=reason).status_code, 200)
        self.assertEqual(self.judge.call_count, len(choices) * len(reasons))

    def test_the_answer_never_echoes_which_arm_was_where(self):
        cases = [  # bench.judge's shape (progress nested), and a flat one
            ({"ok": True, "item_id": "abc123def456", "choice": "a", "arm": "gemini",
              "progress": {"judged": 13, "total": 58, "remaining": 45},
              "next": {"item_id": "x", "arm": "transcript"}},
             {"ok": True, "judged": 13, "total": 58, "remaining": 45}),
            ({"arm_a": "gemini", "judged": 13, "total": 58, "remaining": True},
             {"ok": True, "judged": 13, "total": 58}),
            (None, {"ok": True}),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.judge.return_value = answer
                r = self.post()
                self.assertEqual(r.json(), expected)
                self.assertNotIn("gemini", r.text)
                self.assertNotIn("transcript", r.text)

    def test_bad_input_is_refused_before_the_bench(self):
        cases = [({"choice": "maybe"}, 400), ({"reason": "ugly"}, 400),
                 ({"item_id": ""}, 400), ({"item_id": "../x"}, 400), ({"choice": ""}, 400)]
        for data, status in cases:
            with self.subTest(data=data):
                r = self.post(**data)
                self.assertEqual(r.status_code, status)
                self.assertIn("error", r.json())
        self.judge.assert_not_called()

    def test_unknown_run_is_404(self):
        r = self.client.post("/clip-bench/other/judge", data={"item_id": "abc", "choice": "a"})
        self.assertEqual(r.status_code, 404)
        self.judge.assert_not_called()

    def test_judging_is_closed_after_the_reveal(self):
        self.runs = [run_entry(revealed=True)]
        r = self.post()
        self.assertEqual(r.status_code, 409)
        self.assertIn("revealed", r.json()["error"])
        self.judge.assert_not_called()

    def test_bench_errors_map_to_statuses(self):
        cases = [(ValueError("solo items take yes or no"), 400, "solo items take yes or no"),
                 (KeyError("abc123def456"), 400, "abc123def456"),
                 (FileNotFoundError("judgments.jsonl"), 404, "judgments.jsonl"),
                 (RuntimeError("disk full"), 500, "Couldn't save")]
        for exc, status, text in cases:
            with self.subTest(exc=exc):
                self.judge.side_effect = exc
                r = self.post()
                self.assertEqual(r.status_code, status)
                self.assertIn(text, r.json()["error"])


class RevealRouteTest(PageTestCase):
    def test_reveals_and_returns_json(self):
        r = self.client.post("/clip-bench/test-1/reveal")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True, "revealed_at": "2026-11-18T12:00:00Z"})
        self.reveal.assert_called_once_with("test-1")

    def test_unknown_run_and_refusals(self):
        self.assertEqual(self.client.post("/clip-bench/other/reveal").status_code, 404)
        self.reveal.side_effect = ValueError("already revealed")
        r = self.client.post("/clip-bench/test-1/reveal")
        self.assertEqual(r.status_code, 400)
        self.assertIn("already revealed", r.json()["error"])
        self.reveal.side_effect = RuntimeError("boom")
        self.assertEqual(self.client.post("/clip-bench/test-1/reveal").status_code, 500)


class RealBenchTest(PageTestCase):
    """The page against the real bench.py, nothing mocked but the report: a
    hand-built run in bench.py's own format (items made by its _make_item),
    judged through the routes from first item to reveal."""

    mock_bench = False
    NAME = "it-1"

    def setUp(self):
        super().setUp()
        run_dir = self.bench_dir / self.NAME
        run_dir.mkdir(parents=True)
        one, two = {"video_id": "VIDONE00001"}, {"video_id": "VIDTWO00002"}
        mine = ("gemini", None, [{"start": 0.0, "end": 12.0}])
        items = [
            bench._make_item(self.NAME, "arms", one, "gemini", mine,
                             ("transcript", None, [{"start": 50.0, "end": 70.0}]), ["r1"]),
            bench._make_item(self.NAME, "ceiling", two, "gemini", mine,
                             ("reference", 7, [{"start": 30.0, "end": 45.0}]), ["r2"]),
            bench._make_item(self.NAME, "solo", two, "gemini", mine, None, ["r3"]),
            # Near-identical sides: auto-scored a tie, never shown.
            bench._make_item(self.NAME, "arms", two, "gemini", mine,
                             ("transcript", None, [{"start": 0.0, "end": 12.0}]), ["r4"]),
        ]
        self.renders = {}
        for item in items:
            for side, info in item["sides"].items():
                path = run_dir / info["render"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{item['item_id']}-{side}".encode())
                self.renders[(item["item_id"], side)] = path.read_bytes()
        (run_dir / "manifest.json").write_text(json.dumps({
            "name": self.NAME, "seed": "test", "mode": "sampled",
            "arms": ["gemini", "transcript"], "created_at": "2026-11-18T09:00:00Z"}))
        (run_dir / "videos.json").write_text(json.dumps([one, two]))
        (run_dir / "pairs.json").write_text(json.dumps({"items": items}))
        self.run_dir = run_dir

    def judging_page(self) -> str:
        r = self.client.get("/clip-bench", params={"run": self.NAME})
        self.assertEqual(r.status_code, 200)
        return r.text

    def test_judge_every_item_then_reveal(self):
        index = self.client.get("/clip-bench").text
        self.assertIn('href="/clip-bench?run=it-1"', index)
        self.assertIn("0 / 3", index)                       # the auto-tie isn't counted

        seen = []
        for _ in range(3):
            html = self.judging_page()
            for secret in ("VIDONE00001", "VIDTWO00002", "gemini", "transcript",
                           "reference", "ceiling", "renders/"):
                self.assertNotIn(secret, html)
            item_id = html.split('data-item="', 1)[1].split('"', 1)[0]
            solo = "Would post" in html
            for side in (["a"] if solo else ["a", "b"]):
                r = self.client.get(f"/clip-bench/{self.NAME}/media/{item_id}/{side}")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.content, self.renders[(item_id, side)])
            r = self.client.post(f"/clip-bench/{self.NAME}/judge", data={
                "item_id": item_id, "choice": "yes" if solo else "b",
                "recognised": "" if solo else "1", "reason": "" if solo else "boring"})
            self.assertEqual(r.status_code, 200, r.text)
            seen.append(item_id)
            self.assertEqual(r.json()["judged"], len(seen))
            self.assertEqual(r.json()["total"], 3)
        self.assertEqual(len(set(seen)), 3)
        self.assertIn("All 3 judged", self.judging_page())

        # A pair's choice on a solo item is refused by bench.py itself.
        solo_id = bench._item_id(self.NAME, "solo", "VIDTWO00002", "gemini")
        r = self.client.post(f"/clip-bench/{self.NAME}/judge",
                             data={"item_id": solo_id, "choice": "a"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("choice must be one of yes, no", r.json()["error"])

        r = self.client.post(f"/clip-bench/{self.NAME}/reveal")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["revealed_at"])
        r = self.client.post(f"/clip-bench/{self.NAME}/judge",
                             data={"item_id": seen[0], "choice": "a"})
        self.assertEqual(r.status_code, 409)

        lines = (self.run_dir / "judgments.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 3)
        entries = [json.loads(line) for line in lines]
        pairs = [e for e in entries if e["kind"] != "solo"]
        self.assertTrue(all(e["recognised"] and e["reason"] == "boring" for e in pairs))
        with mock.patch.object(bench, "report", return_value="THE REPORT"):
            html = self.judging_page()
        self.assertIn("THE REPORT", html)
        self.assertNotIn("<video", html)

    def test_media_for_unknown_items_sides_and_auto_ties_is_404(self):
        tie_id = bench._item_id(self.NAME, "arms", "VIDTWO00002", "gemini")
        for url in (f"/clip-bench/{self.NAME}/media/{tie_id}/a",       # auto-tie: never judged
                    f"/clip-bench/{self.NAME}/media/000000000000/a",
                    f"/clip-bench/nope/media/{tie_id}/a"):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)


class HelpersTest(unittest.TestCase):
    def test_run_names(self):
        cases = {"test-1": True, "a.b_c": True, "Dev2": True, "x" * 64: True,
                 "": False, ".hidden": False, "../x": False, "a/b": False,
                 "a b": False, "x" * 65: False, "-lead": False}
        for name, ok in cases.items():
            with self.subTest(name=name):
                self.assertEqual(clip_bench.valid_run_name(name), ok)

    def test_run_names_are_the_ones_bench_accepts(self):
        self.assertEqual(clip_bench.RUN_NAME_RE.pattern, bench._NAME_RE.pattern)

    def test_choices_and_reasons_are_the_ones_bench_records(self):
        self.assertEqual(clip_bench.PAIR_CHOICES, tuple(bench.PAIR_CHOICES))
        self.assertEqual(clip_bench.SOLO_CHOICES, tuple(bench.SOLO_CHOICES))
        self.assertEqual(tuple(k for k, _ in clip_bench.REASONS), tuple(bench.REASONS))

    def test_truthy(self):
        for value, expected in [("1", True), ("on", True), ("TRUE", True), (" yes ", True),
                                ("", False), ("0", False), ("off", False), (None, False)]:
            with self.subTest(value=value):
                self.assertEqual(clip_bench.truthy(value), expected)

    def test_run_view_tolerates_shapes(self):
        cases = [
            ({"name": "a", "arms": "gemini", "videos": "7", "judged": None, "total": "x"},
             {"arms": ["gemini"], "videos": 7, "judged": 0, "total": 0, "revealed": False,
              "dev": False, "stopped": False}),
            ({"name": "b", "arms": [{"label": "g"}, "t", None], "videos": [1, 2],
              "revealed_at": "2026-11-01", "seed": "dev"},
             {"arms": ["g", "t"], "videos": 2, "revealed": True, "dev": True}),
            ({"name": "c", "judged": -3, "revealed": False, "revealed_at": "2026",
              "mode": "hand-picked", "seed": "test"},
             {"judged": 0, "revealed": False, "dev": True}),
            ({"name": "d", "seed": "dev", "dev": False, "stopped": {"reason": "max_usd"}},
             {"dev": False, "stopped": True}),   # bench.py's own flag wins
        ]
        for entry, expected in cases:
            with self.subTest(entry=entry):
                view = clip_bench.run_view(entry)
                for key, value in expected.items():
                    self.assertEqual(view[key], value, key)

    def test_item_view_keeps_only_id_solo_and_sides(self):
        self.assertEqual(clip_bench.item_view(dict(LEAKY_PAIR)),
                         {"item_id": "abc123def456", "solo": False, "sides": ["a", "b"]})
        cases = [  # (item, expected sides): bench.py's sides when sane, else derived
            ({"item_id": "s1", "kind": "solo", "sides": ["a"]}, ["a"]),
            ({"item_id": "s1", "mode": "solo"}, ["a"]),
            ({"item_id": "p1", "kind": "pair", "sides": ["b", "a"]}, ["b", "a"]),
            ({"item_id": "p1", "kind": "pair", "sides": ["a", "c"]}, ["a", "b"]),
            ({"item_id": "s1", "kind": "solo", "sides": ["a", "b"]}, ["a"]),
        ]
        for item, sides in cases:
            with self.subTest(item=item):
                self.assertEqual(clip_bench.item_view(item)["sides"], sides)
        for bad in (None, {}, {"item_id": "a/b"}, "abc"):
            with self.subTest(bad=bad):
                self.assertIsNone(clip_bench.item_view(bad))

    def test_analytics_links_the_bench_next_to_label_clips(self):
        html = make_templates().env.get_template("analytics.html").render(
            ready=False, active="analytics", msg="")
        self.assertIn('href="/clip-bench"', html)
        self.assertLess(html.index('href="/labels"'), html.index('href="/clip-bench"'))


if __name__ == "__main__":
    unittest.main()
