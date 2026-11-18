"""The headless processes never load the clip engine.

The scheduler (and the Fly worker built from requirements-scheduler.txt)
imports the app's core modules, not the dashboard. The clip engine drags in
ffmpeg plumbing, the Gemini client and the Anthropic streaming paths, and an
import-time error in any of it must cost the dashboard's engine, never the
scheduler's publishing loop. So every module the headless side imports keeps
its engine imports inside functions (spec §0.5). This imports them in a fresh
interpreter — the only way to see what a cold import really pulls in — and
checks that nothing under ``app.clip_engine`` came along.

Run with the suite's §0 prefix (the subprocess inherits its environment, so
it opens no database but in-memory SQLite and has no API keys).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Imported by the scheduler / monitor processes, directly or not.
HEADLESS = ("app.scheduler", "app.monitor", "app.llm", "app.vision", "app.spend",
            "app.analytics", "app.clip_proposals", "app.clip_context", "app.publishing")
# Import the engine lazily, inside functions: safe to import, and must stay so.
LAZY = ("app.scrape", "app.web.clip_bench")

_PROBE = """
import importlib, json, sys

def engine_modules():
    return sorted(m for m in sys.modules
                  if m == "app.clip_engine" or m.startswith("app.clip_engine."))

out = {}
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
    out[name] = engine_modules()
importlib.import_module("app.clip_engine.jobs")   # positive control
out["control"] = engine_modules()
print(json.dumps(out))
"""


class ImportSafetyTest(unittest.TestCase):
    def probe(self, modules: tuple[str, ...]) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env.update({"DATABASE_URL": "sqlite://", "SCHEDULER_EMBEDDED": "false",
                        "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "",
                        "SUPABASE_SERVICE_KEY": "",
                        "CLIP_ENGINE_DIR": str(Path(tmp) / "engine"),
                        "CLIPBENCH_DIR": str(Path(tmp) / "bench")})
            proc = subprocess.run([sys.executable, "-c", _PROBE, json.dumps(list(modules))],
                                  cwd=ROOT, env=env, capture_output=True, text=True,
                                  timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_headless_modules_never_import_the_engine(self):
        loaded = self.probe(HEADLESS + LAZY)
        for name in HEADLESS + LAZY:
            with self.subTest(module=name):
                self.assertEqual(loaded[name], [], f"importing {name} loaded the clip engine")
        # The probe can see the engine when it IS imported.
        self.assertIn("app.clip_engine.jobs", loaded["control"])


if __name__ == "__main__":
    unittest.main()
