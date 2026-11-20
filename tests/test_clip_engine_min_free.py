import os
import unittest
from unittest import mock

from app.clip_engine import min_free_gb
from app.config import Settings


class MinFreeGbTest(unittest.TestCase):
    def _settings(self, value=None):
        raw = {"clip_engine": {"min_free_gb": value}} if value is not None else {}
        return Settings(raw=raw)

    def test_default_and_setting(self):
        with mock.patch.dict(os.environ, {"CLIP_ENGINE_MIN_FREE_GB": ""}):
            self.assertEqual(min_free_gb(self._settings()), 10.0)
            self.assertEqual(min_free_gb(self._settings(25)), 25.0)

    def test_env_overrides_setting(self):
        with mock.patch.dict(os.environ, {"CLIP_ENGINE_MIN_FREE_GB": "3"}):
            self.assertEqual(min_free_gb(self._settings(10)), 3.0)

    def test_bad_values_fall_through(self):
        with mock.patch.dict(os.environ, {"CLIP_ENGINE_MIN_FREE_GB": "lots"}):
            self.assertEqual(min_free_gb(self._settings("30 min")), 10.0)
            self.assertEqual(min_free_gb(self._settings(12)), 12.0)


if __name__ == "__main__":
    unittest.main()
