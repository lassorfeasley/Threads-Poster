"""Models that think by default get room to answer (app/llm.py). Run with:
python -m unittest tests.test_llm_thinking_room -v"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from app import llm


def _resp(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))


class ThinkingRoomTest(unittest.TestCase):
    def setUp(self):
        self.sent = []
        fake = mock.Mock()
        fake.messages.create.side_effect = lambda **k: (self.sent.append(k), _resp("end_turn"))[1]
        for p in (mock.patch.object(llm, "client", return_value=fake),
                  mock.patch.object(llm.spend, "record")):
            p.start()
            self.addCleanup(p.stop)

    def test_thinking_models_get_the_floor(self):
        llm._create("claude-sonnet-5", "s", "u", 3000, 0.2)
        llm._create("claude-opus-5-5", "s", "u", 256, 0.2)
        llm._create("claude-sonnet-5", "s", "u", 20000, 0.2)        # bigger asks stand
        llm._create("claude-haiku-4-5", "s", "u", 3000, 0.2)        # doesn't think unasked
        self.assertEqual([k["max_tokens"] for k in self.sent], [16000, 16000, 20000, 3000])

    def test_a_reply_cut_off_mid_thought_says_so(self):
        cut = _resp("max_tokens", SimpleNamespace(type="thinking"))
        with self.assertRaises(llm.TruncatedReply):
            llm._text_from(cut)
        self.assertEqual(llm._text_from(_resp("end_turn", SimpleNamespace(type="text", text="{}"))), "{}")


if __name__ == "__main__":
    unittest.main()
