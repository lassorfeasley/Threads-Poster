"""Tests for the spend ledger: buckets kept out of the shared daily budget, the
process default bucket, meters, pricing rows, and cross-process locking.

Run with:  python -m unittest tests.test_spend_buckets -v
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

from app import config, spend

ROOT = Path(__file__).resolve().parent.parent


class LedgerCase(unittest.TestCase):
    """A temp ledger and the code's own price table (no settings rows)."""

    settings_raw: dict = {"llm": {"daily_budget_usd": 3.0}}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.ledger = self.dir / "llm_spend.json"
        patches = [
            mock.patch.object(spend, "_LEDGER", self.ledger),
            mock.patch.object(spend, "load_settings",
                              side_effect=lambda: config.Settings(raw=self.settings_raw)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        spend.set_bucket(None)
        self.addCleanup(spend.set_bucket, None)

    def data(self) -> dict:
        return json.loads(self.ledger.read_text())

    def today(self) -> dict:
        return self.data()[spend._today()]


class RecordTest(LedgerCase):
    def test_unbucketed_spend_goes_to_the_day_total(self):
        cost = spend.record("claude-opus-5-5", 1_000_000, 100_000)
        self.assertAlmostEqual(cost, 4.0 + 2.0)
        day = self.today()
        self.assertAlmostEqual(day["spend_usd"], 6.0)
        self.assertEqual(day["calls"], 1)
        self.assertEqual(day["by_model"]["claude-opus-5-5"],
                         {"spend_usd": 6.0, "calls": 1, "input_tokens": 1_000_000,
                          "output_tokens": 100_000})
        self.assertNotIn("buckets", day)
        self.assertAlmostEqual(spend.today_spend(), 6.0)

    def test_bucketed_spend_stays_out_of_the_shared_budget(self):
        spend.record("gemini-3.8-flash", 2_000_000, 1_000_000, bucket="engine")  # $10.50
        day = self.today()
        self.assertEqual((day["spend_usd"], day["calls"], day["by_model"]), (0.0, 0, {}))
        engine = day["buckets"]["engine"]
        self.assertAlmostEqual(engine["spend_usd"], 10.5)
        self.assertEqual(engine["calls"], 1)
        self.assertEqual(engine["by_model"]["gemini-3.8-flash"]["output_tokens"], 1_000_000)
        # The $3 budget that gates caption checks and rerun captions is untouched.
        self.assertEqual(spend.today_spend(), 0.0)
        self.assertTrue(spend.within_budget())
        self.assertEqual(spend.remaining_today(), 3.0)
        self.assertEqual(spend.recent(1)[0]["spend_usd"], 0.0)
        self.assertEqual(spend.recent(1)[0]["calls"], 0)
        self.assertAlmostEqual(spend.bucket_spend("engine"), 10.5)

    def test_buckets_and_main_total_accumulate_separately(self):
        spend.record("claude-haiku-4-5", 1_000_000, 0)                      # $1 main
        spend.record("claude-haiku-4-5", 1_000_000, 0, bucket="engine")     # $1 engine
        spend.record("claude-haiku-4-5", 2_000_000, 0, bucket="engine")     # $2 engine
        spend.record("claude-haiku-4-5", 1_000_000, 0, bucket="bench")      # $1 bench
        self.assertAlmostEqual(spend.today_spend(), 1.0)
        self.assertAlmostEqual(spend.bucket_spend("engine"), 3.0)
        self.assertAlmostEqual(spend.bucket_spend("bench"), 1.0)
        self.assertEqual(self.today()["buckets"]["engine"]["calls"], 2)

    def test_existing_positional_callers_still_work(self):
        cost = spend.record("claude-haiku-4-5", 1000, 2000)
        self.assertAlmostEqual(cost, 1000 / 1e6 * 1.0 + 2000 / 1e6 * 5.0)
        self.assertEqual(self.today()["calls"], 1)

    def test_unusable_token_counts_count_as_zero(self):
        self.assertEqual(spend.record("claude-opus-5-5", None, "lots"), 0.0)
        self.assertEqual(spend.record("claude-opus-5-5", -5, 0), 0.0)
        self.assertEqual(self.today()["calls"], 2)
        self.assertEqual(self.today()["by_model"]["claude-opus-5-5"]["input_tokens"], 0)

    def test_record_never_raises_when_the_ledger_cannot_be_written(self):
        with mock.patch.object(spend, "_write", side_effect=OSError("disk full")), \
                self.assertLogs("spend", "WARNING"):
            cost = spend.record("claude-opus-5-5", 1_000_000, 0, bucket="engine")
        self.assertAlmostEqual(cost, 4.0)
        self.assertFalse(self.ledger.exists())

    def test_no_tmp_files_are_left_behind(self):
        for _ in range(3):
            spend.record("claude-opus-5-5", 10, 10)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         ["llm_spend.json", "llm_spend.json.lock"])

    def test_a_corrupt_ledger_is_moved_aside_not_overwritten(self):
        self.ledger.write_text("{not json")
        with self.assertLogs("spend", "WARNING"):
            spend.record("claude-opus-5-5", 10, 10)
        backups = list(self.dir.glob("llm_spend.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), "{not json")
        self.assertEqual(self.today()["calls"], 1)

    def test_a_ledger_that_is_not_utf8_is_moved_aside_too(self):
        """Bytes that aren't UTF-8 used to fail every record() (logged, never
        repaired) while the readers saw an empty ledger: nothing was counted
        and both daily caps read $0 until someone noticed."""
        raw = json.dumps({spend._today(): {"spend_usd": 2.5, "calls": 10, "by_model": {}}},
                         indent=2).encode() + b"\xff\xfe"
        self.ledger.write_bytes(raw)
        self.assertEqual(spend.today_spend(), 0.0)    # unreadable reads as empty
        with self.assertLogs("spend", "WARNING") as logs:
            spend.record("claude-opus-5-5", 10, 10)
        self.assertIn("moved it to", logs.output[-1])
        backups = list(self.dir.glob("llm_spend.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), raw)
        self.assertEqual(self.today()["calls"], 1)
        spend.record("claude-opus-5-5", 10, 10, bucket="engine")
        self.assertEqual(self.today()["calls"], 1)
        self.assertEqual(self.today()["buckets"]["engine"]["calls"], 1)

    def test_a_ledger_saved_with_a_byte_order_mark_is_still_read(self):
        """A hand edit saved with a UTF-8 BOM is still a valid ledger: kept,
        and read the same by the budget checks and by record()."""
        self.ledger.write_bytes(b"\xef\xbb\xbf" + json.dumps(
            {spend._today(): {"spend_usd": 1.25, "calls": 3, "by_model": {}}}).encode())
        self.assertAlmostEqual(spend.today_spend(), 1.25)
        with self.assertNoLogs("spend", "WARNING"):
            spend.record("claude-haiku-4-5", 1_000_000, 0)
        self.assertEqual(list(self.dir.glob("llm_spend.json.corrupt-*")), [])
        self.assertEqual(self.today()["calls"], 4)
        self.assertAlmostEqual(spend.today_spend(), 2.25)

    def test_a_hand_damaged_day_is_repaired(self):
        self.ledger.write_text(json.dumps({spend._today(): {"spend_usd": "x", "buckets": []}}))
        spend.record("claude-opus-5-5", 1_000_000, 0, bucket="engine")
        spend.record("claude-opus-5-5", 1_000_000, 0)
        day = self.today()
        self.assertAlmostEqual(day["buckets"]["engine"]["spend_usd"], 4.0)
        self.assertAlmostEqual(day["spend_usd"], 4.0)

    def test_older_days_are_kept(self):
        self.ledger.write_text(json.dumps({"2026-01-01": {"spend_usd": 1.25, "calls": 3,
                                                          "by_model": {}}}))
        spend.record("claude-opus-5-5", 10, 10)
        self.assertEqual(self.data()["2026-01-01"]["spend_usd"], 1.25)


class ProcessBucketTest(LedgerCase):
    CASES = [
        # (process default, explicit bucket, where the spend lands)
        (None, None, None),
        (None, "engine", "engine"),
        ("bench", None, "bench"),
        ("bench", "engine", "engine"),   # an explicit bucket wins
        ("bench", "", "bench"),          # blank explicit = not given
        ("", None, None),                # blank default = no default
        ("  bench  ", None, "bench"),
    ]

    def test_cases(self):
        for default, explicit, expected in self.CASES:
            with self.subTest(default=default, explicit=explicit):
                self.ledger.unlink(missing_ok=True)
                spend.set_bucket(default)
                spend.record("claude-opus-5-5", 1_000_000, 0, bucket=explicit)
                day = self.today()
                if expected is None:
                    self.assertEqual(day["calls"], 1)
                    self.assertNotIn("buckets", day)
                else:
                    self.assertEqual(day["calls"], 0)
                    self.assertEqual(list(day["buckets"]), [expected])

    def test_current_bucket_reports_the_default(self):
        self.assertIsNone(spend.current_bucket())
        spend.set_bucket("bench")
        self.assertEqual(spend.current_bucket(), "bench")
        spend.set_bucket(None)
        self.assertIsNone(spend.current_bucket())


class BucketSpendTest(LedgerCase):
    def write(self, data):
        self.ledger.write_text(json.dumps(data))

    def test_dates(self):
        self.write({
            "2026-11-17": {"spend_usd": 0.5, "calls": 1, "by_model": {},
                           "buckets": {"engine": {"spend_usd": 2.25, "calls": 4, "by_model": {}}}},
            spend._today(): {"spend_usd": 0.0, "calls": 0, "by_model": {},
                             "buckets": {"engine": {"spend_usd": 1.5, "calls": 2, "by_model": {}}}},
        })
        cases = [
            (None, 1.5),
            ("2026-11-17", 2.25),
            (dt.date(2026, 11, 17), 2.25),
            (dt.datetime(2026, 11, 17, 23, 30), 2.25),
            (dt.datetime(2026, 11, 18, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=5))), 2.25),
            ("2026-11-16", 0.0),
        ]
        for date, expected in cases:
            with self.subTest(date=date):
                self.assertAlmostEqual(spend.bucket_spend("engine", date), expected)
        self.assertEqual(spend.bucket_spend("bench"), 0.0)

    def test_recent_reports_a_bucket_beside_the_shared_figures(self):
        """The analytics chart shows the clip engine's spend next to the
        shared budget's, since the shared figures never include it."""
        today = spend._today()
        yesterday = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        self.write({
            yesterday: {"spend_usd": 0.5, "calls": 1, "by_model": {},
                        "buckets": {"engine": {"spend_usd": 2.25, "calls": 4, "by_model": {}}}},
            today: {"spend_usd": 0.1, "calls": 2, "by_model": {},
                    "buckets": {"engine": "damaged", "bench": {"spend_usd": 9.0, "calls": 1}}},
        })
        rows = lambda **kw: [(r["date"], r["spend_usd"], r["calls"]) for r in spend.recent(3, **kw)]
        third = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).strftime("%Y-%m-%d")
        cases = [
            ({}, [(today, 0.1, 2), (yesterday, 0.5, 1), (third, 0.0, 0)]),
            ({"bucket": None}, [(today, 0.1, 2), (yesterday, 0.5, 1), (third, 0.0, 0)]),
            ({"bucket": "engine"}, [(today, 0.0, 0), (yesterday, 2.25, 4), (third, 0.0, 0)]),
            ({"bucket": "bench"}, [(today, 9.0, 1), (yesterday, 0.0, 0), (third, 0.0, 0)]),
            ({"bucket": "nobody"}, [(today, 0.0, 0), (yesterday, 0.0, 0), (third, 0.0, 0)]),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                self.assertEqual(rows(**kwargs), expected)

    def test_missing_or_damaged_ledger_reads_as_zero(self):
        self.assertEqual(spend.bucket_spend("engine"), 0.0)
        for raw in ("{oops", json.dumps([1, 2]),
                    json.dumps({spend._today(): {"buckets": {"engine": "x"}}}),
                    json.dumps({spend._today(): {"buckets": {"engine": {"spend_usd": None}}}})):
            with self.subTest(raw=raw):
                self.ledger.write_text(raw)
                self.assertEqual(spend.bucket_spend("engine"), 0.0)
                self.assertEqual(spend.today_spend(), 0.0)


class MeterTest(LedgerCase):
    def test_meter_counts_this_threads_calls_in_the_block(self):
        spend.record("claude-haiku-4-5", 1_000_000, 0)        # before: not counted
        with spend.metered() as m:
            spend.record("claude-haiku-4-5", 1_000_000, 0)
            spend.record("claude-opus-5-5", 1_000_000, 0, bucket="engine")
        spend.record("claude-haiku-4-5", 1_000_000, 0)        # after: not counted
        self.assertAlmostEqual(m.total, 5.0)
        self.assertEqual(m.calls, 2)
        self.assertEqual(sorted(m.by_model), ["claude-haiku-4-5", "claude-opus-5-5"])

    def test_meters_nest(self):
        with spend.metered() as outer:
            spend.record("claude-haiku-4-5", 1_000_000, 0)
            with spend.metered() as inner:
                spend.record("claude-haiku-4-5", 2_000_000, 0)
            spend.record("claude-haiku-4-5", 4_000_000, 0)
        self.assertAlmostEqual(inner.total, 2.0)
        self.assertEqual(inner.calls, 1)
        self.assertAlmostEqual(outer.total, 7.0)
        self.assertEqual(outer.calls, 3)

    def test_other_threads_are_not_counted(self):
        def elsewhere():
            spend.record("claude-haiku-4-5", 1_000_000, 0)

        with spend.metered() as m:
            t = threading.Thread(target=elsewhere)
            t.start()
            t.join()
        self.assertEqual((m.total, m.calls), (0.0, 0))
        self.assertEqual(self.today()["calls"], 1)

    def test_meter_counts_even_when_the_ledger_write_fails(self):
        with spend.metered() as m, mock.patch.object(spend, "_write", side_effect=OSError("ro")), \
                self.assertLogs("spend", "WARNING"):
            spend.record("claude-opus-5-5", 1_000_000, 0)
        self.assertAlmostEqual(m.total, 4.0)

    def test_meter_is_released_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with spend.metered():
                raise RuntimeError("boom")
        self.assertEqual(spend._meters(), [])


class PricingTest(LedgerCase):
    SPEC_ROWS = {  # SPEC §2.2, USD per 1M tokens
        "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
        "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5": (2.0, 10.0),
        "claude-haiku-4-5": (1.0, 5.0), "gemini-3.8-flash": (1.50, 7.50),
        "gemini-3.6-flash": (1.50, 7.50), "gemini-3.5-flash-lite": (0.30, 2.50),
        "gemini-4-hydrogen": (1.50, 7.50), "gemini-4-carbon": (4.00, 20.00),
        "gemini-4-argon": (4.00, 20.00), "gemini-3.1-pro-preview": (4.00, 18.00),
        # Server-side fallback targets for Opus 5.5.
        "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0),
        "default": (3.0, 15.0),
    }

    def test_default_rows(self):
        self.assertEqual(set(spend.DEFAULT_PRICING), set(self.SPEC_ROWS))
        for model, (inp, out) in self.SPEC_ROWS.items():
            with self.subTest(model=model):
                self.assertEqual(spend.DEFAULT_PRICING[model], {"input": inp, "output": out})
                self.assertAlmostEqual(spend.estimate_cost(model, 1_000_000, 1_000_000), inp + out)

    def test_lookup(self):
        cases = [
            ("claude-opus-5-5", 24.0),
            ("claude-opus-5-5-20261001", 24.0),       # dated snapshot -> its family row
            ("claude-sonnet-5-5", 12.0),
            ("gemini-3.8-flash", 9.0),
            ("gemini-3.8-flash-001", 9.0),
            ("some-new-model", 18.0),                 # unknown -> default
            ("claude-opus-5x", 18.0),                 # not a dash boundary -> default
        ]
        for model, expected in cases:
            with self.subTest(model=model):
                self.assertAlmostEqual(spend.estimate_cost(model, 1_000_000, 1_000_000), expected)

    def test_unknown_model_warns_once(self):
        spend._unpriced_warned.discard("brand-new-model")
        with self.assertLogs("spend", "WARNING") as logs:
            spend.estimate_cost("brand-new-model", 1, 1)
            spend.estimate_cost("brand-new-model", 1, 1)
        self.assertEqual(len([r for r in logs.records if "brand-new-model" in r.getMessage()]), 1)


class SettingsRowsTest(LedgerCase):
    settings_raw = {"llm": {"pricing": {
        "claude-opus-5-5": {"input": 1.0},                   # partial: keeps code output
        "claude-sonnet-5": {"input": 9.0, "output": 9.0},    # differs from sonnet-5-5
        "made-up": {"input": "abc", "output": -1},           # malformed values
        "broken-row": "not a mapping",
        "default": {"input": 10.0, "output": 10.0},
    }}}

    def test_settings_rows_lay_over_the_code_table(self):
        cases = [
            ("claude-opus-5-5", 1.0 + 20.0),
            ("claude-sonnet-5", 18.0),
            ("claude-sonnet-5-5", 12.0),        # exact row wins over the sonnet-5 prefix
            ("made-up", 3.0 + 15.0),            # malformed values -> code default rates
            ("broken-row", 20.0),               # ignored row -> settings default row
            ("gemini-4-carbon", 24.0),          # untouched code row
        ]
        for model, expected in cases:
            with self.subTest(model=model):
                self.assertAlmostEqual(spend.estimate_cost(model, 1_000_000, 1_000_000), expected)


class SettingsYamlTest(unittest.TestCase):
    def test_settings_yaml_rows_match_the_code_defaults(self):
        """settings.yaml rows replace the code rows, so a stale YAML row (the old
        claude-sonnet-5 3/15) would silently override the new price."""
        raw = yaml.safe_load((ROOT / "workspaces" / "climate" / "config" / "settings.yaml").read_text())
        rows = raw["llm"]["pricing"]
        self.assertEqual(set(rows), set(spend.DEFAULT_PRICING))
        for model, row in spend.DEFAULT_PRICING.items():
            with self.subTest(model=model):
                self.assertEqual({k: float(v) for k, v in rows[model].items()}, row)


class BudgetTest(LedgerCase):
    def test_daily_budget_tolerates_a_blank_value(self):
        for value, expected in ((None, 3.0), ("", 3.0), ("abc", 3.0), (5, 5.0), ("7.5", 7.5)):
            with self.subTest(value=value), \
                    mock.patch.object(spend, "load_settings",
                                      return_value=config.Settings(raw={"llm": {"daily_budget_usd": value}})):
                self.assertEqual(spend.daily_budget(), expected)

    def test_within_budget_counts_only_unbucketed_spend(self):
        spend.record("claude-opus-5-5", 500_000, 0)                     # $2 main
        spend.record("claude-opus-5-5", 5_000_000, 0, bucket="engine")  # $20 engine
        self.assertTrue(spend.within_budget())
        self.assertFalse(spend.within_budget(headroom=1.5))


class LockingTest(LedgerCase):
    def test_threads_never_lose_an_update(self):
        def work():
            for _ in range(25):
                spend.record("claude-haiku-4-5", 1000, 0, bucket="engine")

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        engine = self.today()["buckets"]["engine"]
        self.assertEqual(engine["calls"], 200)
        self.assertEqual(engine["by_model"]["claude-haiku-4-5"]["input_tokens"], 200_000)

    def test_processes_never_lose_an_update(self):
        """The dashboard and a CLI process record into one ledger."""
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "from app import spend\n"
            "spend._LEDGER = Path(sys.argv[1])\n"
            "for _ in range(int(sys.argv[2])):\n"
            "    spend.record('claude-haiku-4-5', 1000, 0, bucket='bench')\n"
        )
        env = {**os.environ, "DATABASE_URL": "sqlite://", "SCHEDULER_EMBEDDED": "false",
               "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "", "SUPABASE_SERVICE_KEY": ""}
        # Without the flock, four writers like these lose most of their updates.
        procs = [subprocess.Popen([sys.executable, "-c", script, str(self.ledger), "100"],
                                  cwd=ROOT, env=env, stderr=subprocess.PIPE)
                 for _ in range(4)]
        for p in procs:
            _, err = p.communicate(timeout=120)
            self.assertEqual(p.returncode, 0, err.decode()[-2000:])
        self.assertEqual(self.today()["buckets"]["bench"]["calls"], 400)

    def test_file_lock_excludes_a_second_holder(self):
        path = self.dir / "x.lock"
        results = []

        def try_lock():
            with spend.file_lock(path, timeout=0.05) as held:
                results.append(held)

        def in_thread():
            t = threading.Thread(target=try_lock)
            t.start()
            t.join()

        with spend.file_lock(path) as held:
            self.assertTrue(held)
            with self.assertLogs("spend", "WARNING"):
                in_thread()
        # It gave up after the timeout and let its caller proceed unlocked...
        self.assertEqual(results, [False])
        # ...and takes the lock once the first holder is gone.
        in_thread()
        self.assertEqual(results, [False, True])

    def test_file_lock_waits_for_a_short_holder(self):
        path = self.dir / "y.lock"
        release = threading.Event()
        holding = threading.Event()

        def holder():
            with spend.file_lock(path):
                holding.set()
                release.wait(5)

        t = threading.Thread(target=holder)
        t.start()
        holding.wait(5)
        threading.Timer(0.1, release.set).start()
        started = time.monotonic()
        with spend.file_lock(path, timeout=5.0) as held:
            self.assertTrue(held)
        t.join()
        self.assertGreaterEqual(time.monotonic() - started, 0.05)


if __name__ == "__main__":
    unittest.main()
