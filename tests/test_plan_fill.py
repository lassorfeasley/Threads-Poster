"""The window plan fills slots the way the tick does (app/scheduler.py:
_fill_open_windows): a window with neither a post nor a rerun goes to the
queue, so the calendar shows open only once the queue can't fill it.

Run with:  python -m unittest tests.test_plan_fill -v
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from app import scheduler

UPCOMING = [("2026-11-22#0", None, 0), ("2026-11-22#1", None, 1), ("2026-11-22#2", None, 2)]
KEYS = [k for k, _, _ in UPCOMING]


class FillOpenWindowsTest(unittest.TestCase):
    def test_an_empty_window_goes_to_the_queue(self):
        a, b = SimpleNamespace(id=1), SimpleNamespace(id=2)
        first = {KEYS[0]: None, KEYS[1]: a, KEYS[2]: b}          # #0: no post, no rerun
        calls = []

        def assign(session, posts, keys, open_keys=frozenset()):
            calls.append(open_keys)
            return [a, b, None], SimpleNamespace(decisions={"x": 1})   # the tick pulls ``a`` forward

        with mock.patch.object(scheduler, "_assign_with_mode", side_effect=assign), \
                mock.patch.object(scheduler, "_project_reruns", return_value={KEYS[2]: {"rerun": 1}}):
            post_by_key, decisions, reruns = scheduler._fill_open_windows(
                None, [a, b], UPCOMING, first, {}, {"cfg": 1})
        self.assertEqual(calls, [frozenset({KEYS[0]})])
        self.assertEqual(post_by_key, {KEYS[0]: a, KEYS[1]: b, KEYS[2]: None})
        self.assertEqual(reruns, {KEYS[2]: {"rerun": 1}})        # the freed slot takes a rerun
        self.assertEqual(decisions, {"x": 1})

    def test_nothing_to_do(self):
        a = SimpleNamespace(id=1)
        with mock.patch.object(scheduler, "_assign_with_mode", side_effect=AssertionError("no")):
            # Every window has a post or a rerun.
            self.assertIsNone(scheduler._fill_open_windows(
                None, [a], UPCOMING, {KEYS[0]: a}, {KEYS[1]: {}, KEYS[2]: {}}, None))
            # An empty queue can't fill anything.
            self.assertIsNone(scheduler._fill_open_windows(None, [], UPCOMING, {}, {}, None))


if __name__ == "__main__":
    unittest.main()
