"""run.py's clip-engine, clipbench and retranscribe commands.

Nothing here reaches a real database, model or storage bucket: ``init_db`` is
patched, ``app.db.session_scope`` is swapped for one over an in-memory SQLite
schema, the bench/jobs/engine/shotlog/playbook calls are mocks, sources are
temp files, and the engine and bench directories are temp dirs. The tests pin
the argparse wiring and what each command does with the answers it gets —
above all the safety rules: words-only retranscription, sessions closed before
slow work, restored sources deleted, bench spend in its own bucket.

Run with:  python -m pytest tests/test_cli_clip_engine.py -q
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import run
from app import clip_engine, spend
from app.clip_engine import bench, engine, jobs, media, playbook, shotlog
from app.config import Settings
from app.models import Base, Candidate, Channel, Cut
from app.web import clip_bench


def parse(argv):
    return run.build_parser().parse_args(argv)


def parse_error(argv) -> str:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            run.build_parser().parse_args(argv)
        except SystemExit:
            return err.getvalue()
    raise AssertionError(f"{argv} parsed without an error")


class ParserTest(unittest.TestCase):
    def test_commands_reach_their_functions_with_their_defaults(self):
        cases = [
            (["clip-engine", "try", "42"], run.cmd_clip_engine_try,
             {"id": "42", "setup": None, "section": None, "log": False, "workspace": None}),
            (["clip-engine", "try", "KYp8dYJKP2s", "--setup", "hybrid", "--section", "1:30", "120",
              "--log"], run.cmd_clip_engine_try,
             {"id": "KYp8dYJKP2s", "setup": "hybrid", "section": [90.0, 120.0], "log": True}),
            (["clip-engine", "shotlog", "42", "--perceiver", "claude", "--force"],
             run.cmd_clip_engine_shotlog, {"perceiver": "claude", "force": True}),
            (["clip-engine", "playbook"], run.cmd_clip_engine_playbook,
             {"refresh": False, "model": None}),
            (["clip-engine", "playbook", "--refresh", "--model", "claude-opus-5-5"],
             run.cmd_clip_engine_playbook, {"refresh": True, "model": "claude-opus-5-5"}),
            (["clipbench", "run", "--name", "test-1"], run.cmd_clipbench_run,
             {"name": "test-1", "arms": None, "limit": None, "max_usd": None, "seed": "test",
              "workers": 1, "candidate": None, "allow_no_words": False, "negatives": 0,
              "new_version": False, "keep_errors": False}),
            (["clipbench", "run", "--name", "dev.2", "--arms", "gemini+editor_thinking=HIGH, transcript",
              "--limit", "5", "--max-usd", "3.5", "--seed", "dev", "--workers", "2",
              "--candidate", "12", "abc", "--candidate", "13", "--allow-no-words",
              "--negatives", "4", "--new-version", "--keep-errors"], run.cmd_clipbench_run,
             {"arms": ["gemini+editor_thinking=HIGH", "transcript"], "limit": 5, "max_usd": 3.5,
              "seed": "dev", "workers": 2, "candidate": ["12", "abc", "13"],
              "allow_no_words": True, "negatives": 4, "new_version": True, "keep_errors": True}),
            (["clipbench", "pairs", "--name", "t"], run.cmd_clipbench_pairs, {"name": "t"}),
            (["clipbench", "report", "--name", "t"], run.cmd_clipbench_report, {"name": "t"}),
            (["clipbench", "reveal", "--name", "t", "--yes"], run.cmd_clipbench_reveal,
             {"name": "t", "yes": True}),
            (["clipbench", "critic-check"], run.cmd_clipbench_critic_check,
             {"roles": None, "limit": 40, "name": None}),
            (["clipbench", "critic-check", "--roles", "claude,gemini,claude", "--limit", "10"],
             run.cmd_clipbench_critic_check, {"roles": ["claude", "gemini"], "limit": 10}),
            (["clipbench", "list", "--workspace", "climate"], run.cmd_clipbench_list,
             {"workspace": "climate"}),
            (["retranscribe", "--missing-words"], run.cmd_retranscribe,
             {"missing_words": True, "limit": None, "dry_run": False}),
            (["retranscribe", "--missing-words", "--limit", "3", "--dry-run"], run.cmd_retranscribe,
             {"limit": 3, "dry_run": True}),
        ]
        for argv, func, expected in cases:
            with self.subTest(argv=argv):
                args = parse(argv)
                self.assertIs(args.func, func)
                for key, value in expected.items():
                    self.assertEqual(getattr(args, key), value, key)

    def test_bad_input_is_refused_by_the_parser(self):
        cases = [
            ["clip-engine", "try", "42", "--setup", "transcript"],
            ["clip-engine", "try", "42", "--section", "1:75", "2:00"],
            ["clip-engine", "try", "42", "--section", "90"],
            ["clip-engine", "shotlog", "42", "--perceiver", "gpt"],
            ["clipbench", "run"],
            ["clipbench", "run", "--name", "../escape"],
            ["clipbench", "run", "--name", "t", "--limit", "0"],
            ["clipbench", "run", "--name", "t", "--max-usd", "-1"],
            ["clipbench", "run", "--name", "t", "--seed", "prod"],
            ["clipbench", "run", "--name", "t", "--negatives", "-1"],
            ["clipbench", "critic-check", "--roles", "gemini,gpt"],
            ["clipbench", "pairs"],
            ["retranscribe"],
            ["clip-engine"],
            ["clipbench"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertIn("error", parse_error(argv))

    def test_workspace_is_accepted_on_every_new_leaf(self):
        leaves = [["clip-engine", "try", "1"], ["clip-engine", "shotlog", "1"],
                  ["clip-engine", "playbook"], ["clipbench", "run", "--name", "t"],
                  ["clipbench", "pairs", "--name", "t"], ["clipbench", "report", "--name", "t"],
                  ["clipbench", "reveal", "--name", "t"], ["clipbench", "critic-check"],
                  ["clipbench", "list"], ["retranscribe", "--missing-words"]]
        for argv in leaves:
            with self.subTest(argv=argv):
                self.assertEqual(parse(argv + ["--workspace", "other"]).workspace, "other")

    def test_run_names_match_what_the_bench_and_the_judging_page_accept(self):
        self.assertEqual(run._RUN_NAME_RE.pattern, clip_bench.RUN_NAME_RE.pattern)
        self.assertEqual(run._RUN_NAME_RE.pattern, bench._NAME_RE.pattern)
        self.assertEqual(parse(["clipbench", "report", "--name", "x" * 64]).name, "x" * 64)
        self.assertIn("bad run name", parse_error(["clipbench", "report", "--name", "x" * 65]))

    def test_the_usage_docstring_lists_the_new_commands(self):
        for text in ("clip-engine try", "clip-engine shotlog", "clip-engine playbook",
                     "clipbench run", "clipbench pairs", "clipbench report", "clipbench reveal",
                     "clipbench critic-check", "clipbench list", "retranscribe --missing-words"):
            self.assertIn(text, run.__doc__)


class FormattingTest(unittest.TestCase):
    def test_parse_time(self):
        good = {"90": 90.0, "90.5": 90.5, "1:30": 90.0, "0:05.5": 5.5, "1:02:03": 3723.0, " 7 ": 7.0}
        for text, seconds in good.items():
            with self.subTest(text=text):
                self.assertEqual(run._parse_time(text), seconds)
        for text in ("", "abc", "1:75", "-5", "1:2:3:4", "inf", "nan", "1.5:30", "1::2"):
            with self.subTest(text=text):
                with self.assertRaises(Exception):
                    run._parse_time(text)

    def test_clock(self):
        cases = [((0,), "0:00.0"), ((75.3,), "1:15.3"), ((3723,), "1:02:03.0"),
                 ((59.96,), "1:00.0"), ((12, 0), "0:12"), ((-4,), "0:00.0"), (("x",), "?"),
                 ((None,), "?"), ((float("inf"),), "?"), ((float("nan"),), "?")]
        for args, text in cases:
            with self.subTest(args=args):
                self.assertEqual(run._clock(*args), text)

    def test_critic_line(self):
        cases = [
            (None, "not watched"),
            ({}, "not watched"),
            ({"error": "GeminiBlocked: SAFETY", "score": 0}, "not reviewed (GeminiBlocked: SAFETY)"),
            ({"stops_scroll": True, "stop_reason": "Nye pulls off his glasses", "bored_at": 21.0,
              "fix": "end on the laugh"},
             "Stops the scroll: Nye pulls off his glasses · slow at 0:21 · fix: end on the laugh"),
            ({"stops_scroll": False, "stop_reason": "aerial b-roll", "topic": "none",
              "bored_at": None}, "Wouldn't stop the scroll: aerial b-roll · topic: none"),
        ]
        for critique, text in cases:
            with self.subTest(critique=critique):
                self.assertEqual(run._critic_line(critique), text)

    def test_cost_lines_keep_reused_perception_apart(self):
        lines = run._cost_lines(0.4123, {"perceive": 0.11, "draft": 0.13, "perceive_cached": 0.06})
        self.assertEqual(lines[0], "Cost: $0.4123 — perceive $0.1100 · draft $0.1300")
        self.assertIn("perceive $0.0600 paid by an earlier run", lines[1])

    def test_segments_print_in_play_order(self):
        segs = [{"start": 30.0, "end": 41.5}, {"start": 10.0, "end": 14.0}]
        self.assertEqual(run._fmt_segments(segs), "0:30.0-0:41.5 -> 0:10.0-0:14.0")
        self.assertAlmostEqual(run._seg_seconds(segs + [{"start": "x"}]), 15.5)

    def test_shotlog_report(self):
        log = {"perceiver": "gemini", "model": "gemini-3.8-flash", "source_seconds": 456.2,
               "cost_usd": 0.061, "complete": False, "summary": "A science show segment.",
               "people": [{"name": "Bill Nye", "role": "science educator",
                           "prominence": "famous", "evidence": "title"}],
               "beats": [{"start": 0.0, "end": 6.5, "shot": "talking_head", "face": "close",
                          "speaker": "Bill Nye", "line": "The planet's on fire",
                          "on_screen": "lower third: Bill Nye", "energy": 4, "visual": 3,
                          "topic": 2}],
               "highlights": [{"start": 3.0, "end": 4.5, "kind": "person", "what": "glasses off"}]}
        text = run.format_shotlog_report(log, video_id="KYp8dYJKP2s")
        self.assertIn("Shot log KYp8dYJKP2s · gemini/gemini-3.8-flash · 7:36.2 · 1 beats · "
                      "1 highlights · $0.0610 · INCOMPLETE", text)
        self.assertIn("Bill Nye — science educator (famous; title)", text)
        self.assertIn('0:00.0-0:06.5  talking_head   face close E4 V3 T2  Bill Nye: "The '
                      "planet's on fire\" | on screen: lower third: Bill Nye", text)
        self.assertIn("0:03.0-0:04.5  person  glasses off", text)


class CliTestCase(unittest.TestCase):
    """Temp dirs, a patched spend ledger, init_db and an in-memory database."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.engine_dir = self.tmp / "engine"
        self.bench_dir = self.tmp / "bench"
        self.start(mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.engine_dir),
                                                "CLIPBENCH_DIR": str(self.bench_dir)}))
        self.start(mock.patch.object(spend, "_LEDGER", self.tmp / "llm_spend.json"))
        self.init_db = self.start(mock.patch("app.db.init_db"))
        self.db = create_engine("sqlite://", connect_args={"check_same_thread": False},
                                poolclass=StaticPool)
        self.addCleanup(self.db.dispose)
        Base.metadata.create_all(self.db)
        self.Session = sessionmaker(bind=self.db, expire_on_commit=False, future=True)
        self.open = 0   # sessions the command holds open right now
        self.start(mock.patch("app.db.session_scope", self._scope))
        self.settings = Settings(raw={})
        self.start(mock.patch("app.config.load_settings", lambda: self.settings))
        with self.Session() as s:
            channel = Channel(call_sign="KTST", url="https://example.com/ktst")
            s.add(channel)
            s.commit()
            self.channel_pk = channel.id

    def start(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    @contextlib.contextmanager
    def _scope(self, *, read_only=False):
        self.open += 1
        session = self.Session()
        try:
            yield session
            if not read_only:
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            self.open -= 1

    def add_candidate(self, video_id, *, status="archived", transcript=True, words=False,
                      local=True, key="", method="captions", duration=456,
                      archived_at=None, stale_words=False) -> int:
        tdir = self.tmp / "transcripts"
        tdir.mkdir(exist_ok=True)
        tpath = tdir / f"{video_id}.json"
        if transcript:
            tpath.write_text(json.dumps([{"start": 0.0, "end": 4.0, "text": "the planet is on fire"}]))
        wpath = tdir / f"{video_id}_words.json"
        if words:
            wpath.write_text(json.dumps([{"word": "planet", "start": 0.4, "end": 0.9}]))
        lpath = self.tmp / "videos" / f"{video_id}.mp4"
        if local:
            lpath.parent.mkdir(exist_ok=True)
            lpath.write_bytes(b"\0" * 32)
        with self.Session() as s:
            c = Candidate(video_id=video_id, channel_pk=self.channel_pk, title=f"Title {video_id}",
                          url="https://youtube.com/watch", status=status,
                          transcript_path=str(tpath) if transcript or stale_words else "",
                          transcript_text="[00:00] the planet is on fire",
                          word_transcript_path=str(wpath) if words or stale_words else "",
                          local_video_path=str(lpath), source_object_key=key,
                          transcription_method=method, duration_seconds=duration,
                          archived_at=archived_at or dt.datetime(2026, 11, 1, 12, 0))
            s.add(c)
            s.commit()
            return c.id

    def candidate(self, pk) -> Candidate:
        with self.Session() as s:
            return s.get(Candidate, pk)

    def cli(self, argv) -> tuple[str, object]:
        """Run one command; returns (stdout, SystemExit code or None)."""
        args = parse(argv)
        out = io.StringIO()
        code = None
        with contextlib.redirect_stdout(out):
            try:
                args.func(args)
            except SystemExit as exc:
                code = exc.code
        return out.getvalue(), code


def engine_result(**kw) -> engine.EngineResult:
    pick = {"id": "d2r", "origin": "revision", "parent": "d2", "archetype": "voice",
            "lens": "person", "segments": [{"start": 30.0, "end": 41.5}, {"start": 10.0, "end": 14.0}],
            "score": 7.25, "story": "Nye: the planet is on fire", "hook": "Glasses off",
            "critique": {"stops_scroll": True, "stop_reason": "Bill Nye on camera",
                         "fix": "end on the laugh", "bored_at": None, "topic": "central"},
            "credits": "0-12 s: Bill Nye, science educator", "render": "/tmp/r.mp4", "rank": 1}
    values = dict(
        setup="gemini", top=[pick],
        drafts=[{"id": "d1", "archetype": "scene", "lens": "picture",
                 "segments": [{"start": 50.0, "end": 70.0}], "score": 3.1,
                 "critique": {"stops_scroll": False, "stop_reason": "aerial b-roll",
                              "topic": "none"}, "dropped_reason": "off topic"},
                dict(pick),
                {"id": "d3", "archetype": "voice", "lens": "line",
                 "segments": [{"start": 80.0, "end": 100.0}], "score": -1e9,
                 "critique": {"error": "GeminiBlocked: SAFETY", "score": 0}}],
        moments=[], shotlog=None, cost_usd=0.4123,
        by_stage={"perceive": 0.11, "draft": 0.13, "perceive_cached": 0.06},
        tokens={"gemini-3.8-flash": {"input": 230000, "output": 45000}},
        timings={"perceive": 34.2, "draft": 51.0}, notes=["ranking failed; ordered by cold score"],
        orders={"editor": ["d1", "d2", "d3"], "final": ["d2r"]},
        model="engine:gemini:gemini-3.8-flash", policy="pol", outcome="proposals")
    values.update(kw)
    return engine.EngineResult(**values)


class TryCommandTest(CliTestCase):
    def setUp(self):
        super().setUp()
        self.pk = self.add_candidate("KYp8dYJKP2s", words=True)
        with self.Session() as s:
            fields = {k: getattr(s.get(Candidate, self.pk), k) for k in run._CANDIDATE_FIELDS}
        self.inputs = {"candidate": fields, "used": [{"start": 0.0, "end": 5.0}],
                       "context": {"shipped_examples": [1]},
                       "rejected": [{"segments": [{"start": 1, "end": 9}], "reason": "boring"}],
                       "examples": [{"post_pk": 7}], "striking": [], "notes": ["note from inputs"]}
        self.prepare = self.start(mock.patch.object(jobs, "prepare_inputs", return_value=self.inputs))
        self.write = self.start(mock.patch.object(jobs, "write_result", return_value=[101, 102]))
        self.setup = self.start(mock.patch.object(clip_engine, "effective_setup",
                                                  return_value=("gemini", "")))
        self.start(mock.patch.object(clip_engine, "brand", return_value={"name": "Test"}))
        self.source = Path(fields["local_video_path"])
        self.link = self.tmp / "inflight" / "KYp8dYJKP2s-cli.mp4"
        self.ensure = self.start(mock.patch("app.offload.ensure_local", return_value=self.source))
        self.link_inflight = self.start(mock.patch.object(media, "link_inflight",
                                                          return_value=self.link))
        self.release = self.start(mock.patch.object(media, "release_inflight"))
        self.run_engine = self.start(mock.patch.object(engine, "run", return_value=engine_result()))
        self.start(mock.patch.object(engine, "estimate_usd", return_value=0.83))
        self.start(mock.patch.object(spend, "bucket_spend", create=True, return_value=0.5))

    def test_a_dry_run_prints_every_draft_and_writes_nothing(self):
        out, code = self.cli(["clip-engine", "try", str(self.pk)])
        self.assertIsNone(code)
        kw = self.run_engine.call_args.kwargs
        self.assertEqual(kw["setup"], "gemini")
        self.assertEqual(kw["source"], self.link)
        self.assertEqual(kw["video_id"], "KYp8dYJKP2s")
        self.assertEqual(kw["transcript"], [{"start": 0.0, "end": 4.0, "text": "the planet is on fire"}])
        self.assertEqual(kw["words"], [{"word": "planet", "start": 0.4, "end": 0.9}])
        for key in ("context", "used", "rejected", "examples"):
            self.assertEqual(kw[key], self.inputs[key], key)
        self.assertIsNone(kw["window"])
        self.assertIsNone(kw["keep"])
        self.assertEqual(kw["duration"], 456.0)
        self.assertEqual(kw["meter"].bucket, "engine")
        self.assertIsInstance(kw["deadline"], float)
        self.prepare.assert_called_once_with(self.pk, window=None)
        self.link_inflight.assert_called_once_with(self.source, "KYp8dYJKP2s-cli")
        self.release.assert_called_once_with(self.link)
        self.write.assert_not_called()
        for text in ("note: note from inputs", "estimate ~$0.83", "engine spend today $0.50",
                     "dropped: off topic", "pick #1", "not reviewed (GeminiBlocked: SAFETY)",
                     "d2r (revision of d2)", "plays: 0:30.0-0:41.5 -> 0:10.0-0:14.0",
                     "credits: 0-12 s: Bill Nye", "Cost: $0.4123", "Order by editor: d1 d2 d3",
                     "ranking failed; ordered by cold score", "Dry run: nothing written"):
            self.assertIn(text, out)

    def test_no_session_is_open_while_the_engine_runs(self):
        open_during_run = []
        self.run_engine.side_effect = lambda **kw: open_during_run.append(self.open) or engine_result()
        self.cli(["clip-engine", "try", "KYp8dYJKP2s"])
        self.assertEqual(open_during_run, [0])

    def test_the_video_is_marked_touched_and_the_result_saved(self):
        out, _ = self.cli(["clip-engine", "try", "KYp8dYJKP2s"])   # by YouTube id
        self.prepare.assert_called_once_with(self.pk, window=None)
        self.assertIn("KYp8dYJKP2s", bench.touched_video_ids())
        saved = list((self.engine_dir / "tries").glob("KYp8dYJKP2s-gemini-*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text())["top"][0]["id"], "d2r")
        self.assertIn(str(saved[0]), out)

    def test_log_writes_the_top_picks(self):
        out, code = self.cli(["clip-engine", "try", str(self.pk), "--log"])
        self.assertIsNone(code)
        self.write.assert_called_once_with(self.pk, self.run_engine.return_value, window=None)
        self.assertIn("Logged 2 suggestion(s) [101, 102]", out)

    def test_log_with_no_picks_writes_nothing(self):
        self.run_engine.return_value = engine_result(top=[], outcome="empty_topic")
        out, _ = self.cli(["clip-engine", "try", str(self.pk), "--log"])
        self.write.assert_not_called()
        self.assertIn("Nothing to log (outcome: empty_topic)", out)

    def test_a_section_becomes_the_window_everywhere(self):
        self.cli(["clip-engine", "try", str(self.pk), "--section", "1:00", "120", "--log"])
        window = {"start": 60.0, "end": 120.0}
        self.prepare.assert_called_once_with(self.pk, window=window)
        self.assertEqual(self.run_engine.call_args.kwargs["window"], window)
        self.assertEqual(self.write.call_args.kwargs["window"], window)

    def test_refusals_before_any_spend(self):
        cases = [
            (["--section", "2:00", "1:00"], {}, "END must be after START"),
            ([], {"setup": ("transcript", "")}, "Pass --setup"),
            (["--setup", "hybrid"], {"setup": ("transcript", "ANTHROPIC_API_KEY is not set")},
             "Can't run the hybrid setup: ANTHROPIC_API_KEY is not set"),
            ([], {"candidate": {"status": "new"}}, "isn't archived"),
            ([], {"candidate": None}, "not found"),
            ([], {"candidate": {"transcript_path": ""}}, "has no transcript"),
        ]
        for extra, change, message in cases:
            with self.subTest(message=message):
                self.run_engine.reset_mock()
                self.setup.return_value = change.get("setup", ("gemini", ""))
                cand = dict(self.inputs["candidate"])
                if "candidate" in change:
                    cand = None if change["candidate"] is None else {**cand, **change["candidate"]}
                self.prepare.return_value = dict(self.inputs, candidate=cand)
                _, code = self.cli(["clip-engine", "try", str(self.pk)] + extra)
                self.assertIn(message, str(code))
                self.run_engine.assert_not_called()
        _, code = self.cli(["clip-engine", "try", "no-such-video"])
        self.assertIn("No video with id", str(code))

    def test_an_engine_failure_exits_1_and_releases_the_source(self):
        self.run_engine.side_effect = engine.EngineError("no draft could be rendered")
        out, code = self.cli(["clip-engine", "try", str(self.pk), "--log"])
        self.assertEqual(code, 1)
        self.assertIn("The engine run failed: EngineError: no draft could be rendered", out)
        self.release.assert_called_once_with(self.link)
        self.write.assert_not_called()

    def test_an_unavailable_source_stops_cleanly(self):
        from app.offload import SourceUnavailable

        for exc, text in ((SourceUnavailable("its cloud copy is gone"), "its cloud copy is gone"),
                          (ConnectionError("reset by peer"), "ConnectionError: reset by peer")):
            with self.subTest(exc=exc):
                self.ensure.side_effect = exc
                _, code = self.cli(["clip-engine", "try", str(self.pk)])
                self.assertIn(text, str(code))
                self.run_engine.assert_not_called()
                self.release.assert_not_called()

    def test_ctrl_c_reports_what_was_spent(self):
        def interrupted(**kw):
            kw["meter"].add("perceive", {"model": "gemini-3.8-flash", "cost_usd": 0.12,
                                         "input_tokens": 1000, "output_tokens": 10})
            raise KeyboardInterrupt

        self.run_engine.side_effect = interrupted
        out, code = self.cli(["clip-engine", "try", str(self.pk), "--log"])
        self.assertEqual(code, 130)
        self.assertIn("Interrupted. Spent so far:", out)
        self.assertIn("Cost: $0.1200 — perceive $0.1200", out)
        self.release.assert_called_once_with(self.link)
        self.write.assert_not_called()

    def test_a_failed_link_runs_on_the_source_in_place(self):
        self.link_inflight.side_effect = OSError("cross-device link")
        _, code = self.cli(["clip-engine", "try", str(self.pk)])
        self.assertIsNone(code)
        self.assertEqual(self.run_engine.call_args.kwargs["source"], self.source)
        self.release.assert_not_called()


class ShotlogCommandTest(CliTestCase):
    LOG = {"perceiver": "gemini", "model": "gemini-3.8-flash", "source_seconds": 60.0,
           "cost_usd": 0.02, "complete": True, "summary": "S", "people": [], "beats": [],
           "highlights": []}

    def setUp(self):
        super().setUp()
        self.pk = self.add_candidate("vidShot0001", words=True)
        self.start(mock.patch.object(clip_engine, "brand", return_value={}))
        self.load_cached = self.start(mock.patch.object(shotlog, "load_cached", return_value=None))
        self.build = self.start(mock.patch.object(shotlog, "build", return_value=dict(self.LOG)))
        self.shotlog_path = self.start(mock.patch.object(
            shotlog, "cache_path", return_value=self.engine_dir / "shotlogs" / "x.json"))
        self.ensure = self.start(mock.patch("app.offload.ensure_local",
                                            side_effect=lambda local, key: Path(local)))
        self.start(mock.patch.object(media, "link_inflight", side_effect=lambda src, key: src))
        self.release = self.start(mock.patch.object(media, "release_inflight"))
        self.gemini_ok = self.start(mock.patch("app.clip_engine.gemini.is_configured",
                                               return_value=True))

    def test_a_cached_log_is_shown_without_touching_the_source(self):
        self.load_cached.return_value = dict(self.LOG)
        out, code = self.cli(["clip-engine", "shotlog", str(self.pk)])
        self.assertIsNone(code)
        self.assertIn("Cached (no new spend)", out)
        self.build.assert_not_called()
        self.ensure.assert_not_called()

    def test_builds_with_the_configured_perceiver(self):
        self.settings = Settings(raw={"clip_engine": {"setup": "claude"}})
        self.start(mock.patch("app.clip_engine.claude.is_configured", return_value=True))
        out, code = self.cli(["clip-engine", "shotlog", "vidShot0001"])
        self.assertIsNone(code)
        kw = self.build.call_args.kwargs
        self.assertEqual(kw["perceiver"], "claude")
        self.assertEqual(kw["video_id"], "vidShot0001")
        self.assertFalse(kw["force"])
        self.assertIsNone(kw["window"])
        self.assertEqual(kw["meter"].bucket, "engine")
        self.assertEqual(kw["words"], [{"word": "planet", "start": 0.4, "end": 0.9}])
        self.release.assert_called_once()
        self.assertIn("Shot log vidShot0001", out)

    def test_the_cache_is_looked_up_with_this_runs_perception_settings(self):
        """Shot logs are keyed by perception settings, which depend on the
        source's length too: looked up without them, a long source's log is
        missed (the provider key demanded for nothing) and the wrong path printed."""
        self.load_cached.return_value = dict(self.LOG)
        self.cli(["clip-engine", "shotlog", str(self.pk)])
        kw = self.load_cached.call_args.kwargs
        self.assertIs(kw["settings"], self.settings)
        self.assertEqual(kw["seconds"], 456.0)
        self.assertEqual(self.shotlog_path.call_args.kwargs,
                         {"settings": self.settings, "seconds": 456.0})
        # After a build, the path printed is the one build() wrote: with the
        # length it logged (probed when the row has none).
        self.load_cached.return_value = None
        self.shotlog_path.reset_mock()
        self.cli(["clip-engine", "shotlog", str(self.pk), "--force"])
        self.assertEqual(self.build.call_args.kwargs["duration"], 456.0)
        self.assertEqual(self.shotlog_path.call_args.kwargs,
                         {"settings": self.settings, "seconds": 60.0})

    def test_force_skips_the_cache(self):
        self.load_cached.return_value = dict(self.LOG)
        self.cli(["clip-engine", "shotlog", str(self.pk), "--perceiver", "gemini", "--force"])
        self.load_cached.assert_not_called()
        self.assertTrue(self.build.call_args.kwargs["force"])

    def test_a_missing_key_or_a_failure_exits(self):
        self.gemini_ok.return_value = False
        _, code = self.cli(["clip-engine", "shotlog", str(self.pk), "--perceiver", "gemini"])
        self.assertIn("GEMINI_API_KEY is not set", str(code))
        self.build.assert_not_called()
        self.gemini_ok.return_value = True
        self.build.side_effect = RuntimeError("quota")
        out, code = self.cli(["clip-engine", "shotlog", str(self.pk), "--perceiver", "gemini"])
        self.assertEqual(code, 1)
        self.assertIn("Logging failed: RuntimeError: quota", out)
        self.release.assert_called()


class PlaybookCommandTest(CliTestCase):
    def setUp(self):
        super().setUp()
        self.file = self.tmp / "config" / "clip_playbook.yaml"
        self.start(mock.patch.object(playbook, "path", return_value=self.file))
        self.start(mock.patch.object(playbook, "load", return_value={"archetypes": []}))
        self.start(mock.patch.object(playbook, "render", return_value="ARCHETYPES (rendered)"))
        self.start(mock.patch.object(playbook, "fingerprint", return_value="abc123"))
        self.generate = self.start(mock.patch.object(playbook, "generate",
                                                     return_value={"archetypes": ["new"]}))
        self.save = self.start(mock.patch.object(playbook, "save", return_value=self.file))
        self.start(mock.patch("app.clip_engine.claude.is_configured", return_value=True))

    def test_prints_the_playbook_in_use(self):
        out, code = self.cli(["clip-engine", "playbook"])
        self.assertIsNone(code)
        self.assertIn("built-in default", out)
        self.assertIn("ARCHETYPES (rendered)", out)
        self.generate.assert_not_called()
        self.init_db.assert_not_called()

    def test_refresh_backs_up_the_old_file_then_saves(self):
        self.file.parent.mkdir(parents=True)
        self.file.write_text("hand: edited\n")
        out, code = self.cli(["clip-engine", "playbook", "--refresh", "--model", "claude-x"])
        self.assertIsNone(code)
        self.assertEqual(self.generate.call_args.kwargs["model"], "claude-x")
        self.assertEqual(self.generate.call_args.kwargs["meter"].bucket, "engine")
        self.save.assert_called_once_with({"archetypes": ["new"]})
        backups = list((self.engine_dir / "playbooks").glob("clip_playbook-*.yaml"))
        self.assertEqual([b.read_text() for b in backups], ["hand: edited\n"])
        self.assertIn("Saved", out)

    def test_a_failed_refresh_keeps_the_current_playbook(self):
        self.generate.side_effect = ValueError("only 2 archetypes")
        _, code = self.cli(["clip-engine", "playbook", "--refresh"])
        self.assertIn("only 2 archetypes", str(code))
        self.save.assert_not_called()

    def test_model_needs_refresh(self):
        _, code = self.cli(["clip-engine", "playbook", "--model", "m"])
        self.assertIn("--model only applies with --refresh", str(code))


class BenchCommandTest(CliTestCase):
    def setUp(self):
        super().setUp()
        self.calls = mock.Mock()
        self.set_bucket = self.start(mock.patch.object(spend, "set_bucket", create=True))
        self.calls.attach_mock(self.set_bucket, "set_bucket")
        self.parse_arm = self.start(mock.patch.object(bench, "parse_arm", create=True,
                                                      side_effect=lambda spec: {"label": spec}))
        self.bench_run = self.start(mock.patch.object(bench, "run", create=True,
                                                      return_value={"videos": 7, "cost_usd": 3.2}))
        self.calls.attach_mock(self.bench_run, "run")
        self.runs = [{"name": "t1", "arms": ["gemini", "transcript"], "videos": 7, "judged": 5,
                      "total": 7, "revealed": False, "seed": "test", "mode": "sampled"}]
        self.start(mock.patch.object(bench, "list_runs", create=True,
                                     side_effect=lambda: [dict(r) for r in self.runs]))
        self.report = self.start(mock.patch.object(bench, "report", create=True,
                                                   return_value="REPORT TEXT"))
        self.reveal = self.start(mock.patch.object(bench, "reveal", create=True,
                                                   return_value={"revealed_at": "now"}))
        self.pairs = self.start(mock.patch.object(bench, "prepare_pairs", create=True,
                                                  return_value=12))
        self.critic = self.start(mock.patch.object(bench, "critic_check", create=True,
                                                   return_value={"gemini": {"agree_yes": 0.8}}))
        self.start(mock.patch("app.config.current_workspace", return_value={"port": 8399}))

    def test_run_takes_its_defaults_from_settings_and_books_spend_to_bench(self):
        self.settings = Settings(raw={"clip_engine": {"bench": {
            "arms": ["gemini", "transcript"], "limit": 7, "max_usd": 12.5}}})
        out, code = self.cli(["clipbench", "run", "--name", "t1"])
        self.assertIsNone(code)
        self.set_bucket.assert_called_with("bench")
        names = [c[0] for c in self.calls.mock_calls]
        self.assertLess(names.index("set_bucket"), names.index("run"))
        args, kw = self.bench_run.call_args
        self.assertEqual(args, ("t1",))
        # Flags left out go through as None: bench.run applies clip_engine.bench.*
        # to a new run and keeps a resumed run's own arms and cap.
        self.assertEqual(kw, dict(arms=None, limit=None, max_usd=None,
                                  seed="test", workers=1, candidate_ids=None, require_words=True,
                                  negatives=0, new_version=False, keep_errors=False,
                                  settings=self.settings, log=run._say))
        self.assertEqual([c.args for c in self.parse_arm.call_args_list],
                         [("gemini",), ("transcript",)])
        self.init_db.assert_called_once()
        self.assertIn("arms gemini, transcript · up to 7 videos · seed test · "
                      "stops once results pass $12.50", out)
        self.assertIn("videos: 7", out)
        self.assertIn("clipbench pairs --name t1", out)

    def test_a_resume_leaves_the_runs_arms_and_cap_to_the_run(self):
        self.settings = Settings(raw={"clip_engine": {"bench": {
            "arms": ["gemini", "transcript"], "limit": 7, "max_usd": 40}}})
        run_dir = self.bench_dir / "t1"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps(
            {"name": "t1", "arms": ["gemini+editor_thinking=HIGH", "transcript"], "max_usd": 10.0}))
        out, code = self.cli(["clipbench", "run", "--name", "t1"])
        self.assertIsNone(code)
        kw = self.bench_run.call_args.kwargs
        self.assertEqual((kw["arms"], kw["limit"], kw["max_usd"]), (None, None, None))
        self.parse_arm.assert_not_called()   # the settings' arms aren't this run's
        self.assertIn("Resuming bench run 't1'", out)
        self.assertNotIn("$40.00", out)
        # Flags given on a resume still go through, checked first.
        self.cli(["clipbench", "run", "--name", "t1", "--arms", "claude", "--max-usd", "15"])
        kw = self.bench_run.call_args.kwargs
        self.assertEqual((kw["arms"], kw["max_usd"]), (["claude"], 15.0))
        self.assertEqual([c.args for c in self.parse_arm.call_args_list], [("claude",)])

    def test_run_flags_override_and_candidates_resolve_to_pks(self):
        a = self.add_candidate("vidAAAAAAAA")
        b = self.add_candidate("vidBBBBBBBB")
        self.cli(["clipbench", "run", "--name", "dev-1", "--arms", "claude", "--limit", "3",
                  "--max-usd", "2", "--seed", "dev", "--workers", "2", "--allow-no-words",
                  "--negatives", "5", "--new-version", "--keep-errors",
                  "--candidate", "vidBBBBBBBB", str(a), "vidBBBBBBBB"])
        kw = self.bench_run.call_args.kwargs
        self.assertEqual(kw["arms"], ["claude"])
        self.assertEqual((kw["limit"], kw["max_usd"], kw["seed"], kw["workers"]), (3, 2.0, "dev", 2))
        self.assertEqual(kw["candidate_ids"], [b, a])
        self.assertFalse(kw["require_words"])
        self.assertEqual(kw["negatives"], 5)
        self.assertTrue(kw["new_version"] and kw["keep_errors"])

    def test_run_refusals(self):
        self.parse_arm.side_effect = ValueError("unknown setup 'gpt'")
        _, code = self.cli(["clipbench", "run", "--name", "t1", "--arms", "gpt"])
        self.assertIn("Bad arm 'gpt': unknown setup 'gpt'", str(code))
        self.bench_run.assert_not_called()
        self.parse_arm.side_effect = lambda spec: {"label": spec}
        self.bench_run.side_effect = ValueError("the fingerprint changed; pass --new-version")
        _, code = self.cli(["clipbench", "run", "--name", "t1"])
        self.assertIn("The bench refused: the fingerprint changed", str(code))
        self.bench_run.side_effect = KeyboardInterrupt
        out, code = self.cli(["clipbench", "run", "--name", "t1"])
        self.assertEqual(code, 130)
        self.assertIn("run the same command again to resume", out)

    def test_pairs_uses_the_judge_height_and_points_at_the_page(self):
        self.settings = Settings(raw={"clip_engine": {"bench": {"judge_height": 720}}})
        out, _ = self.cli(["clipbench", "pairs", "--name", "t1"])
        self.pairs.assert_called_once_with("t1", height=720)
        self.assertIn("12 item(s) ready to judge: http://127.0.0.1:8399/clip-bench?run=t1", out)
        self.set_bucket.assert_called_with("bench")

    def test_report_prints_it(self):
        out, _ = self.cli(["clipbench", "report", "--name", "t1"])
        self.assertIn("REPORT TEXT", out)
        self.report.side_effect = FileNotFoundError("no manifest")
        _, code = self.cli(["clipbench", "report", "--name", "nope"])
        self.assertIn("no manifest", str(code))

    def test_reveal_asks_first(self):
        cases = [  # (argv extra, tty, answer, revealed?)
            (["--yes"], False, None, True),
            ([], False, None, False),
            ([], True, "y", True),
            ([], True, "", False),
            ([], True, "no", False),
        ]
        for extra, tty, answer, revealed in cases:
            with self.subTest(extra=extra, tty=tty, answer=answer):
                self.reveal.reset_mock()
                with mock.patch("sys.stdin") as stdin, \
                        mock.patch("builtins.input", return_value=answer or "") as ask:
                    stdin.isatty.return_value = tty
                    out, code = self.cli(["clipbench", "reveal", "--name", "t1"] + extra)
                self.assertEqual(self.reveal.called, revealed)
                if revealed:
                    self.assertIn("REPORT TEXT", out)
                if tty and not extra:
                    self.assertIn("5 of 7 judged", ask.call_args.args[0])
                if not tty and not extra:
                    self.assertIn("pass --yes", str(code))

    def test_reveal_of_a_revealed_or_unknown_run(self):
        self.runs[0]["revealed"] = True
        out, _ = self.cli(["clipbench", "reveal", "--name", "t1"])
        self.reveal.assert_not_called()
        self.assertIn("already revealed", out)
        self.assertIn("REPORT TEXT", out)
        _, code = self.cli(["clipbench", "reveal", "--name", "zzz", "--yes"])
        self.assertIn("No bench run named 'zzz'", str(code))

    def test_critic_check_checks_keys_and_passes_roles(self):
        self.start(mock.patch("app.clip_engine.gemini.is_configured", return_value=True))
        claude_ok = self.start(mock.patch("app.clip_engine.claude.is_configured",
                                          return_value=True))
        out, code = self.cli(["clipbench", "critic-check", "--roles", "gemini,claude",
                              "--limit", "10"])
        self.assertIsNone(code)
        self.critic.assert_called_once_with(None, roles=("gemini", "claude"), limit=10,
                                            settings=self.settings, log=run._say)
        self.assertIn("agree_yes: 0.8", out)
        claude_ok.return_value = False
        self.critic.reset_mock()
        _, code = self.cli(["clipbench", "critic-check", "--roles", "claude"])
        self.assertIn("ANTHROPIC_API_KEY is not set", str(code))
        self.critic.assert_not_called()

    def test_list(self):
        self.runs.append({"name": "picked", "arms": [{"label": "gemini"}], "videos": [1, 2],
                          "judged": 0, "total": 0, "mode": "hand-picked", "seed": "dev"})
        out, _ = self.cli(["clipbench", "list"])
        lines = out.splitlines()
        self.assertTrue(lines[1].startswith("t1"))
        self.assertIn("5/7", lines[1])
        self.assertIn("judging", lines[1])
        self.assertIn("picked", lines[2])
        self.assertIn("needs pairs", lines[2])
        self.runs.clear()
        out, _ = self.cli(["clipbench", "list"])
        self.assertIn("No bench runs yet", out)


class TouchedTest(CliTestCase):
    def test_bench_records_them_so_the_test_seed_skips_them(self):
        run._mark_touched(["v1", None, "v1", "v2"], by="clip-engine try")
        self.assertEqual(bench.touched_video_ids(), {"v1", "v2"})
        run._mark_touched(["v1"], by="clip-engine try")   # again: still one video
        self.assertEqual(bench.touched_video_ids(), {"v1", "v2"})

    def test_passes_ids_once_with_the_source(self):
        touch = self.start(mock.patch.object(bench, "touch_videos"))
        run._mark_touched(["v1", "", "v1", "v2"], by="clip-engine try")
        touch.assert_called_once_with(["v1", "v2"], source="clip-engine try")
        run._mark_touched([None, ""], by="x")
        touch.assert_called_once()

    def test_a_failure_warns_and_carries_on(self):
        self.start(mock.patch.object(bench, "touch_videos", side_effect=OSError("read-only disk")))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            run._mark_touched(["v1"], by="try")
        self.assertIn("couldn't record v1 as touched", out.getvalue())


class RetranscribeTest(CliTestCase):
    def setUp(self):
        super().setUp()
        self.local = self.add_candidate("vidLocal001", method="captions")
        self.remote = self.add_candidate("vidRemote01", local=False, key="climate/x/vidRemote01.mp4",
                                         method="whisper", stale_words=True,
                                         archived_at=dt.datetime(2026, 9, 1))
        self.add_candidate("vidHasWords", words=True)                    # has its word stream
        self.add_candidate("vidNotArchv", status="new")                  # not archived
        self.add_candidate("vidNoTrnscr", transcript=False, stale_words=True)  # transcript file gone
        self.nosource = self.add_candidate("vidNoSource", local=False, key="",
                                           archived_at=dt.datetime(2026, 8, 1))
        self.silent = self.add_candidate("vidSilent01", archived_at=dt.datetime(2026, 10, 1))
        with self.Session() as s:   # an exported clip puts the remote video first
            s.add(Cut(candidate_pk=self.remote, trimmed_clip_path="/clips/x.mp4"))
            s.commit()
        self.words = [{"word": "fire", "start": 1.0, "end": 1.4}]
        self.whispered: list[str] = []
        self.sessions_during_slow_work: list[int] = []   # open sessions at each download/Whisper

        def restore(local, key):
            self.sessions_during_slow_work.append(self.open)
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_bytes(b"restored")
            return Path(local)

        def whisper(path, settings):
            self.whispered.append(Path(path).name)
            self.sessions_during_slow_work.append(self.open)
            if "Silent" in str(path):
                return [], [], "small"
            return [{"start": 0, "end": 2, "text": "x"}], list(self.words), "base+vad"

        self.ensure = self.start(mock.patch("app.offload.ensure_local", side_effect=restore))
        self.whisper = self.start(mock.patch("app.scrape.transcribe_word_stream",
                                             side_effect=whisper))
        self.free = self.start(mock.patch.object(run, "_free_gb_near", return_value=100.0))

    def test_no_session_is_open_during_a_download_or_whisper(self):
        self.cli(["retranscribe", "--missing-words"])
        self.assertEqual(len(self.sessions_during_slow_work), 4)   # 1 restore + 3 Whisper runs
        self.assertEqual(set(self.sessions_during_slow_work), {0})

    def test_adds_word_streams_and_never_touches_the_transcript(self):
        before = {pk: self.candidate(pk) for pk in (self.local, self.remote)}
        out, code = self.cli(["retranscribe", "--missing-words"])
        self.assertIsNone(code)
        # Exported first, then newest: remote (Sep, exported), local (Nov 1), silent (Oct 1), no source.
        self.assertEqual(self.whispered, ["vidRemote01.mp4", "vidLocal001.mp4", "vidSilent01.mp4"])
        for pk, method in ((self.local, "captions+whisper"), (self.remote, "whisper")):
            c = self.candidate(pk)
            self.assertEqual(json.loads(Path(c.word_transcript_path).read_text()), self.words)
            self.assertEqual(Path(c.word_transcript_path).parent, Path(c.transcript_path).parent)
            self.assertEqual(c.transcription_method, method)
            self.assertEqual(c.transcript_path, before[pk].transcript_path)
            self.assertEqual(c.transcript_text, before[pk].transcript_text)
        silent = self.candidate(self.silent)
        self.assertEqual(silent.word_transcript_path, "")
        self.assertEqual(silent.transcription_method, "captions")
        self.assertIn("Added word streams to 2 video(s); 1 heard no words; 0 failed; 1 skipped.", out)
        self.assertIn("no source video on disk or in storage", out)

    def test_restored_sources_are_deleted_and_local_ones_kept(self):
        self.cli(["retranscribe", "--missing-words"])
        self.ensure.assert_called_once()
        self.assertFalse(Path(self.candidate(self.remote).local_video_path).exists())
        self.assertTrue(Path(self.candidate(self.local).local_video_path).exists())

    def test_dry_run_lists_in_order_and_changes_nothing(self):
        out, _ = self.cli(["retranscribe", "--missing-words", "--dry-run"])
        lines = [l for l in out.splitlines() if l.startswith("  ")]
        self.assertEqual([l.split()[1] for l in lines],
                         ["vidRemote01", "vidLocal001", "vidSilent01", "vidNoSource"])
        self.assertIn("(has an exported clip)", lines[0])
        self.assertIn("source: storage", lines[0])
        self.assertIn("source: none", lines[3])
        self.assertIn("Would add word streams to 4 video(s)", out)
        self.whisper.assert_not_called()
        self.ensure.assert_not_called()
        self.assertEqual(self.candidate(self.local).word_transcript_path, "")

    def test_limit(self):
        out, _ = self.cli(["retranscribe", "--missing-words", "--limit", "1"])
        self.assertEqual(self.whispered, ["vidRemote01.mp4"])
        self.assertIn("Added word streams to 1 video(s)", out)

    def test_stops_before_restoring_onto_a_full_disk(self):
        self.free.return_value = 4.0
        self.settings = Settings(raw={"clip_engine": {"min_free_gb": 10}})
        out, _ = self.cli(["retranscribe", "--missing-words"])
        self.ensure.assert_not_called()
        self.assertEqual(self.whispered, [])
        self.assertIn("Stopped early: only 4.0 GB free (clip_engine.min_free_gb is 10)", out)

    def test_a_whisper_failure_still_deletes_the_restored_source(self):
        self.whisper.side_effect = RuntimeError("model download failed")
        out, _ = self.cli(["retranscribe", "--missing-words", "--limit", "1"])
        self.assertIn("failed: Whisper failed: model download failed", out)
        self.assertFalse(Path(self.candidate(self.remote).local_video_path).exists())
        self.assertEqual(self.candidate(self.remote).transcription_method, "whisper")

    def test_an_unavailable_source_is_skipped(self):
        from app.offload import SourceUnavailable

        self.ensure.side_effect = SourceUnavailable("its cloud copy is gone")
        out, _ = self.cli(["retranscribe", "--missing-words", "--limit", "1"])
        self.assertIn("skipped: its cloud copy is gone", out)
        self.whisper.assert_not_called()

    def test_a_storage_error_fails_that_video_and_the_batch_goes_on(self):
        self.ensure.side_effect = ConnectionError("storage timed out")
        out, code = self.cli(["retranscribe", "--missing-words"])
        self.assertIsNone(code)
        self.assertIn("failed: couldn't restore the source: ConnectionError: storage timed out", out)
        self.assertEqual(self.whispered, ["vidLocal001.mp4", "vidSilent01.mp4"])
        self.assertIn("Added word streams to 1 video(s); 1 heard no words; 1 failed; 1 skipped.", out)

    def test_a_failed_save_leaves_the_row_alone(self):
        with mock.patch("app.scrape._write_word_transcript", side_effect=OSError("disk full")):
            out, _ = self.cli(["retranscribe", "--missing-words", "--limit", "1"])
        self.assertIn("failed: couldn't save the word stream: OSError: disk full", out)
        remote = self.candidate(self.remote)
        self.assertTrue(remote.word_transcript_path.endswith("vidRemote01_words.json"))  # stale, as before
        self.assertFalse(Path(remote.word_transcript_path).exists())
        self.assertFalse(Path(remote.local_video_path).exists())   # restored source still deleted

    def test_ctrl_c_prints_the_tally_and_exits_130(self):
        calls = []

        def whisper(path, settings):
            calls.append(path)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return [], list(self.words), "base+vad"

        self.whisper.side_effect = whisper
        out, code = self.cli(["retranscribe", "--missing-words"])
        self.assertEqual(code, 130)
        self.assertIn("Added word streams to 1 video(s)", out)
        self.assertIn("Stopped early: interrupted", out)
        self.assertFalse(Path(self.candidate(self.remote).local_video_path).exists())

    def test_nothing_to_do(self):
        with self.Session() as s:
            for c in s.query(Candidate).all():
                c.status = "rejected"
            s.commit()
        out, _ = self.cli(["retranscribe", "--missing-words"])
        self.assertIn("has a word stream", out)


if __name__ == "__main__":
    unittest.main()
