"""Tests for the account's own time zone (app/config.py: scheduler_timezone,
timezone_editable, save_timezone).

Run with:  python -m unittest tests.test_timezone -v

The database is an in-memory SQLite patched in as ``app.db.session_scope``.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import config
from app.models import AppToken, Base


def _settings(**values):
    raw: dict = {"scheduler": {"timezone": "America/Chicago"}}
    raw["scheduler"].update(values)
    return config.Settings(raw=raw)


class _DbBase(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", future=True)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)

        @contextlib.contextmanager
        def scope(read_only: bool = False):
            session = Session()
            try:
                yield session
                if not read_only:
                    session.commit()
            finally:
                session.close()

        self.Session = Session
        for patcher in (mock.patch("app.db.session_scope", scope),
                        mock.patch.object(config, "load_settings", return_value=_settings())):
            patcher.start()
            self.addCleanup(patcher.stop)
        config._timezone_cache = None
        self.addCleanup(setattr, config, "_timezone_cache", None)

    def choosing(self, open_: bool):
        patcher = mock.patch.object(config, "TIMEZONE_CHOICE_OPEN", open_)
        patcher.start()
        self.addCleanup(patcher.stop)

    def store(self, name: str):
        with self.Session() as s:
            s.add(AppToken(name=config.TIMEZONE_TOKEN, value=name))
            s.commit()


class SchedulerTimezoneTest(_DbBase):
    def test_settings_zone_when_none_is_saved(self):
        self.assertEqual(config.scheduler_timezone(), ZoneInfo("America/Chicago"))
        self.assertFalse(config.timezone_saved())

    def test_a_saved_zone_is_in_force_whether_or_not_choosing_is_open(self):
        # The guard only stops new choices: every machine on this code follows
        # a zone that's been saved, so none of them is left on the old clock.
        for open_ in (False, True):
            with self.subTest(open_=open_):
                config._timezone_cache = None
                self.choosing(open_)
                with self.Session() as s:
                    s.query(AppToken).delete()
                    s.commit()
                self.store("Asia/Tokyo")
                self.assertEqual(config.scheduler_timezone(), ZoneInfo("Asia/Tokyo"))
                self.assertTrue(config.timezone_saved())

    def test_a_bad_saved_name_falls_back_to_the_settings_zone(self):
        self.store("Not/AZone")
        self.assertEqual(config.scheduler_timezone(), ZoneInfo("America/Chicago"))

    def test_the_saved_zone_is_read_at_most_once_a_minute(self):
        self.store("Asia/Tokyo")
        config.scheduler_timezone()
        with mock.patch("app.db.session_scope", side_effect=AssertionError("re-read")):
            self.assertEqual(config.scheduler_timezone(), ZoneInfo("Asia/Tokyo"))

    def test_an_unreadable_database_keeps_the_zone_last_read(self):
        self.store("Asia/Tokyo")
        self.assertEqual(config.scheduler_timezone(), ZoneInfo("Asia/Tokyo"))
        expires, _ = config._timezone_cache
        config._timezone_cache = (expires - config._TIMEZONE_CACHE_TTL - 1, "Asia/Tokyo")  # stale
        with mock.patch("app.db.session_scope", side_effect=RuntimeError("db down")):
            self.assertEqual(config.scheduler_timezone(), ZoneInfo("Asia/Tokyo"))
        # ...and asks again in seconds, not a minute.
        expires, name = config._timezone_cache
        self.assertEqual(name, "Asia/Tokyo")
        import time

        self.assertLessEqual(expires - time.monotonic(), config._TIMEZONE_RETRY_AFTER)

    def test_an_unreadable_database_with_nothing_read_yet_uses_the_settings_zone(self):
        with mock.patch("app.db.session_scope", side_effect=RuntimeError("db down")):
            self.assertEqual(config.scheduler_timezone(), ZoneInfo("America/Chicago"))
        self.assertIsNone(config._timezone_cache[1])


class SaveTimezoneTest(_DbBase):
    def test_the_guard_is_one_switch_for_every_account(self):
        # Not read from any account's settings.yaml: no account differs.
        with mock.patch.object(config, "load_settings",
                               return_value=_settings(timezone_editable=True)):
            self.assertEqual(config.timezone_editable(), config.TIMEZONE_CHOICE_OPEN)

    def test_refused_while_choosing_is_closed(self):
        self.choosing(False)
        with self.assertRaises(ValueError):
            config.save_timezone("Europe/Berlin")
        with self.Session() as s:
            self.assertIsNone(s.get(AppToken, config.TIMEZONE_TOKEN))

    def test_only_real_zones(self):
        self.choosing(True)
        for bad in ("", "Europe/Atlantis", "EST5EDTX"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                config.save_timezone(bad)

    def test_saves_and_takes_effect_at_once(self):
        self.choosing(True)
        self.assertEqual(config.scheduler_timezone(), ZoneInfo("America/Chicago"))
        self.assertEqual(config.save_timezone(" Europe/Berlin "), "Europe/Berlin")
        self.assertEqual(config.scheduler_timezone(), ZoneInfo("Europe/Berlin"))
        config.save_timezone("Asia/Kolkata")
        self.assertEqual(config.scheduler_timezone(), ZoneInfo("Asia/Kolkata"))
        with self.Session() as s:
            self.assertEqual(s.get(AppToken, config.TIMEZONE_TOKEN).value, "Asia/Kolkata")


if __name__ == "__main__":
    unittest.main()
