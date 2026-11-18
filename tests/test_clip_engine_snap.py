"""Tests for the boundary snapper (app/clip_engine/snap.py).

Table-driven: one row per rule, each with words, shot cuts and pauses laid
out so only that rule decides the outcome. Expected times are worked out by
hand from the rules in SPEC §4.5 (comments show the arithmetic).

Run with:  DATABASE_URL="sqlite://" python -m unittest tests.test_clip_engine_snap -v
"""
from __future__ import annotations

import copy
import random
import unittest

from app.clip_engine import snap


def W(*spans):
    """Words from (start, end) pairs."""
    return [{"word": f"w{i}", "start": s, "end": e} for i, (s, e) in enumerate(spans)]


def C(*times, score=40.0):
    """Strong shot-map cuts at these times."""
    return [{"t": t, "score": score} for t in times]


def S(start, end, lead=None, **extra):
    out = {"start": start, "end": end, **extra}
    if lead:
        out["lead"] = lead
    return out


# Narration running straight across the 10.2 cut (voice-over b-roll).
NARRATION = W((9.0, 9.4), (9.5, 9.9), (10.0, 10.5), (10.6, 11.0), (11.1, 11.5))
# A line ending at 6.5 s, the next one starting at 8.0 s (END cases; the
# segments start at 0, five seconds before any speech, so START stays put).
LINE = W((5.0, 5.4), (5.5, 6.0), (6.1, 6.5), (8.0, 8.4))
LINE_ONLY = W((5.0, 5.4), (5.5, 6.0), (6.1, 6.5))

CASES = [
    # ---- START, lead "picture" ------------------------------------------------
    dict(name="picture lead opens on the nearest cut",
         segments=[S(9.5, 20, "picture")], words=W((12.0, 12.4)), cuts=C(10.0),
         expected=[(10.0, 20.0)]),
    dict(name="voice-over across a cut: picture lead moves FORWARD to the next word gap",
         # cut 10.2 is inside (10.0, 10.5); 10.5 - 10.2 <= 0.8 -> 10.5 + 0.02
         segments=[S(10.0, 20, "picture")], words=NARRATION, cuts=C(10.2),
         expected=[(10.52, 20.0)]),
    dict(name="voice-over with a gap under 20 ms: opens inside the gap, not the next word",
         segments=[S(10.0, 20, "picture")], words=W((10.0, 10.5), (10.51, 10.9)),
         cuts=C(10.2), expected=[(10.51, 20.0)]),
    dict(name="same footage, word lead: opens on the first word, before the cut",
         # first word 10.0 -> max(10.0 - 0.08, 9.9 + 0.02) = 9.92; the 10.2 cut is
         # after the first word starts, so it can't be the opening
         segments=[S(10.0, 20)], words=NARRATION, cuts=C(10.2),
         expected=[(9.92, 20.0)]),
    dict(name="picture lead: a word running long past the cut is cut on the shot",
         segments=[S(10.3, 20, "picture")], words=W((9.8, 11.5)), cuts=C(10.0),
         expected=[(10.0, 20.0)]),
    dict(name="picture lead: nearest of several cuts within picture_window",
         segments=[S(10.0, 20, "picture")], words=[], cuts=C(9.2, 10.4, 11.3),
         expected=[(10.4, 20.0)]),
    dict(name="picture lead with no cut in reach falls back to the word rules",
         # cut 12.5 is 2.5 s away; first word 10.3 -> 10.3 - 0.08
         segments=[S(10.0, 20, "picture")], words=W((10.3, 10.6), (10.7, 11.0)),
         cuts=C(12.5), expected=[(10.22, 20.0)], leads=["picture"]),
    dict(name="picture lead ignores a cut that leaves under min_len",
         # 10.8 >= 11.5 - 1.0, and 0.8 s is outside the word rules' 0.6 window
         segments=[S(10.0, 11.5, "picture")], words=[], cuts=C(10.8),
         expected=[(10.0, 11.5)]),
    dict(name="cuts under min_shot_score are ignored",
         segments=[S(9.5, 20, "picture")], words=[], cuts=C(10.0, score=12.0),
         expected=[(9.5, 20.0)]),
    dict(name="J-cut soundbite, picture lead: face cut inside the first word -> next word gap",
         segments=[S(10.0, 20, "picture")], words=W((9.0, 9.6), (10.0, 10.5), (10.6, 11.0)),
         cuts=C(10.3), expected=[(10.52, 20.0)]),
    dict(name="J-cut soundbite, picture lead: face cut in the gap after the first word",
         segments=[S(10.0, 20, "picture")], words=W((10.0, 10.2), (10.5, 11.0)),
         cuts=C(10.35), expected=[(10.35, 20.0)]),
    dict(name="J-cut soundbite, word lead: opens on the first word over the b-roll",
         # 10.3 cut is inside the first word and after its start: not taken
         segments=[S(10.0, 20)], words=W((9.0, 9.6), (10.0, 10.5), (10.6, 11.0)),
         cuts=C(10.3), expected=[(9.92, 20.0)]),
    dict(name="picture lead: a word gap that would leave under min_len opens on the cut",
         # 10.6 + 0.02 is not < 11.6 - 1.0, so a = the cut itself
         segments=[S(10.0, 11.6, "picture")], words=W((10.0, 10.6)), cuts=C(10.1),
         expected=[(10.1, 11.6)]),

    # ---- START, lead "word" ---------------------------------------------------
    dict(name="a start inside a word backs up to it, then to just before it",
         # inside (10.0, 10.6) -> 10.0; max(10.0 - 0.08, 9.4 + 0.02) = 9.92
         segments=[S(10.3, 20)], words=W((9.0, 9.4), (10.0, 10.6), (10.7, 11.0)),
         cuts=[], expected=[(9.92, 20.0)]),
    dict(name="speech within 0.8 s: start start_pad before the first word",
         segments=[S(10.0, 20)], words=W((10.5, 10.9), (11.0, 11.3)), cuts=[],
         expected=[(10.42, 20.0)]),
    dict(name="never on the previous word's tail",
         # max(10.0 - 0.08, 9.95 + 0.02) = 9.97
         segments=[S(9.98, 20)], words=W((9.0, 9.95), (10.0, 10.4)), cuts=[],
         expected=[(9.97, 20.0)]),
    dict(name="speech more than 0.8 s away: a b-roll lead-in is kept",
         segments=[S(10.0, 20)], words=W((11.0, 11.4)), cuts=[],
         expected=[(10.0, 20.0)]),
    dict(name="a shot change in the gap before the first word wins",
         segments=[S(10.0, 20)], words=W((10.5, 10.9)), cuts=C(10.2),
         expected=[(10.2, 20.0)]),
    dict(name="a cut that would split a word is rejected",
         # 10.0 is inside (9.5, 10.3); start stays max(10.32, 10.3 + 0.02)
         segments=[S(10.35, 20)], words=W((9.5, 10.3), (10.4, 10.8)), cuts=C(10.0),
         expected=[(10.32, 20.0)]),
    dict(name="a cut after the first word starts is rejected",
         segments=[S(10.0, 20)], words=W((10.5, 10.9)), cuts=C(10.7),
         expected=[(10.42, 20.0)]),
    dict(name="a cut may not reach back across an earlier word",
         # 9.6 is within 0.6 s of 10.12 and splits no word, but taking it would
         # open on the stray (9.7, 9.85) word
         segments=[S(10.15, 20)], words=W((9.7, 9.85), (10.2, 10.6)), cuts=C(9.6),
         expected=[(10.12, 20.0)]),
    dict(name="no previous word: the floor bounds the pad",
         segments=[S(10.0, 20)], words=W((10.05, 10.4)), cuts=[], floor=10.0,
         expected=[(10.0, 20.0)]),
    dict(name="touching words: start on the first word itself",
         segments=[S(10.0, 20)], words=W((9.0, 10.0), (10.0, 10.4)), cuts=[],
         expected=[(10.0, 20.0)]),

    # ---- times rounded to a tenth of a second ---------------------------------
    # Transcript entries and the shot log show times to 0.1 s, so an edge the
    # editor copies arrives up to 0.05 s off, often inside the next word over.
    # Words below are from a real Whisper stream (A7MqyQ1TLXw).
    dict(name="rounded start: 'While' (226.64) copied as 226.6 doesn't open on 'fuels.'",
         # 226.6 is inside fuels. (225.98-226.64) but 0.62 s from its start and
         # 0.04 s from While's: While was meant; the words touch, so 226.64
         segments=[S(226.6, 230)],
         words=W((225.5, 225.98), (225.98, 226.64), (226.64, 227.16), (227.16, 227.5)),
         cuts=[], expected=[(226.64, 230.0)]),
    dict(name="rounded start mid-sentence: 'precedent' (6.05) as 6.0 doesn't open on 'a'",
         segments=[S(6.0, 20)], words=W((5.4, 5.87), (5.87, 6.05), (6.05, 6.61)),
         cuts=[], expected=[(6.05, 20.0)]),
    dict(name="rounded start with a short gap: opens in the gap, not on the word before",
         # 9.9 is inside (9.5, 9.93), 0.04 s from 9.94: min(max(9.86, 9.95), 9.94)
         segments=[S(9.9, 20)], words=W((9.5, 9.93), (9.94, 10.4)), cuts=[],
         expected=[(9.94, 20.0)]),
    dict(name="rounded end: 'temperatures.' (147.65) copied as 147.7 doesn't run into 'It's'",
         # 147.7 is inside It's (147.65-148.09), 0.05 s past where temperatures. ends:
         # the end goes back to 147.65 and the next-word cap makes it 147.63
         segments=[S(140, 147.7)],
         words=W((146.5, 147.07), (147.07, 147.65), (147.65, 148.09), (148.09, 148.5)),
         cuts=[], expected=[(140.0, 147.63)]),
    dict(name="rounded end mid-sentence: 'climate' (10.59) as 10.6 doesn't run into 'change.'",
         segments=[S(0, 10.6)], words=W((9.9, 10.31), (10.31, 10.59), (10.59, 11.03)),
         cuts=[], expected=[(0.0, 10.57)]),
    dict(name="a start in the first 0.05 s of a short word keeps that word",
         # 5.95 is 0.05 s from both edges of (5.9, 5.98): either could be meant,
         # and dropping a word is worse than keeping a short one
         segments=[S(5.95, 20)], words=W((5.0, 5.9), (5.9, 5.98), (5.98, 6.5)), cuts=[],
         expected=[(5.9, 20.0)]),
    dict(name="a start late in a word with a pause after it still backs up to it",
         segments=[S(10.57, 20)], words=W((10.0, 10.6), (11.0, 11.4)), cuts=[],
         expected=[(9.92, 20.0)]),
    dict(name="an end in the last 0.05 s of a short word keeps that word",
         # 5.55 is 0.05 s from 5.5 but also 0.03 s from 5.58: the word stays
         segments=[S(0, 5.55)], words=W((5.0, 5.5), (5.5, 5.58), (5.7, 6.5)), cuts=[],
         expected=[(0.0, 5.68)]),

    # ---- head flash guard -----------------------------------------------------
    dict(name="a flash shot at the head is skipped",
         segments=[S(10.0, 20, "picture")], words=[], cuts=C(10.0, 10.25),
         expected=[(10.25, 20.0)]),
    dict(name="a chain of flash shots walks to the first real shot",
         segments=[S(10.0, 20, "picture")], words=[], cuts=C(10.0, 10.2, 10.5, 10.8, 12.0),
         expected=[(10.8, 20.0)]),
    dict(name="the head guard won't split a word",
         segments=[S(10.0, 20, "picture")], words=W((9.5, 11.0)), cuts=C(10.0, 10.2),
         expected=[(10.0, 20.0)]),
    dict(name="the head guard never passes the first word",
         segments=[S(10.0, 20, "picture")], words=W((10.2, 10.25), (10.6, 11.0)),
         cuts=C(10.0, 10.3), expected=[(10.0, 20.0)]),
    dict(name="word lead without speech: flash after the chosen cut is skipped",
         segments=[S(10.0, 20)], words=[], cuts=C(10.1, 10.3),
         expected=[(10.3, 20.0)]),
    dict(name="word lead with speech: flash before the first word is skipped",
         # nearest to 10.42 is 10.40; 10.48 follows within 0.4 s, still <= 10.5
         segments=[S(10.0, 20)], words=W((10.5, 10.9)), cuts=C(10.40, 10.48),
         expected=[(10.48, 20.0)]),

    # ---- END, on the line -----------------------------------------------------
    dict(name="an end inside a word extends to it, plus end_pad",
         segments=[S(0, 6.3)], words=LINE, cuts=[], expected=[(0.0, 6.75)]),
    dict(name="an end on the line gets end_pad after the last word",
         segments=[S(0, 6.9)], words=LINE, cuts=[], expected=[(0.0, 6.75)]),
    dict(name="the next word caps the tail",
         segments=[S(0, 6.55)], words=W((5.0, 5.4), (6.1, 6.5), (6.6, 7.0)), cuts=[],
         expected=[(0.0, 6.58)]),
    dict(name="ends 0.15 s into the pause after the last word",
         segments=[S(0, 6.9)], words=LINE, cuts=[], quiet=[[6.55, 7.5]],
         expected=[(0.0, 6.7)]),
    dict(name="cut just after the last word: end before it, no flash frame",
         segments=[S(0, 6.9)], words=LINE, cuts=C(6.6), expected=[(0.0, 6.58)]),
    dict(name="cut just past the tail: end on the shot change",
         segments=[S(0, 6.9)], words=LINE, cuts=C(7.1), expected=[(0.0, 7.08)]),
    dict(name="the tail never stretches into the next word for a cut",
         segments=[S(0, 6.9)], words=W((5.0, 5.4), (6.1, 6.5), (7.0, 7.4)), cuts=C(7.1),
         expected=[(0.0, 6.75)]),

    # ---- END, held past the line ----------------------------------------------
    dict(name="a held silent ending is kept",
         segments=[S(0, 9.5)], words=LINE_ONLY, cuts=[], quiet=[[6.55, 10.0]],
         expected=[(0.0, 9.5)]),
    dict(name="a held ending lands on a cut next to it",
         segments=[S(0, 9.5)], words=LINE_ONLY, cuts=C(9.3), expected=[(0.0, 9.28)]),
    dict(name="a held ending takes the nearest cut",
         segments=[S(0, 9.5)], words=LINE_ONLY, cuts=C(9.25, 9.9), expected=[(0.0, 9.23)]),
    dict(name="a held ending won't reach a cut past the next word",
         segments=[S(0, 9.5)], words=W((5.0, 5.4), (6.1, 6.5), (9.6, 10.0)), cuts=C(9.8),
         expected=[(0.0, 9.5)]),

    # ---- tail flash guard -----------------------------------------------------
    dict(name="a flash of a new shot at the tail is cut off",
         # 19.65 is outside the hold reach (>= 19.7) but inside the flash guard
         segments=[S(0, 20)], words=[], cuts=C(19.65), expected=[(0.0, 19.63)]),
    dict(name="the tail guard won't cut into the last word",
         segments=[S(0, 20)], words=W((19.0, 19.75)), cuts=C(19.7), expected=[(0.0, 20.0)]),

    # ---- clamping and other clips' openings ------------------------------------
    dict(name="clamped to the window", segments=[S(8, 35)], words=[], cuts=[],
         floor=10, horizon=30, expected=[(10.0, 30.0)]),
    # Blocked = other clips' OPENING windows: only the clip's first played
    # segment may not START inside one. Reusing that footage later is fine.
    dict(name="an opening inside another clip's opening moves to its end",
         segments=[S(10, 30)], words=[], cuts=[], blocked=[S(9, 12)], expected=[(12.0, 30.0)]),
    dict(name="an end inside a blocked window is left alone",
         segments=[S(10, 30)], words=[], cuts=[], blocked=[S(25, 28)], expected=[(10.0, 30.0)]),
    dict(name="a segment straddling a blocked window is not split",
         segments=[S(10, 30)], words=[], cuts=[], blocked=[S(15, 18)], expected=[(10.0, 30.0)]),
    dict(name="only the first played segment is checked",
         segments=[S(50, 60, "picture"), S(10, 30)], words=[], cuts=[], blocked=[S(9, 12)],
         expected=[(50.0, 60.0), (10.0, 30.0)]),
    dict(name="the moved opening never starts mid-word",
         # 12.0 falls inside the word 11.8-12.4: the opening moves past it
         segments=[S(10, 30)], words=W((11.8, 12.4), (13.0, 13.5)), cuts=[],
         blocked=[S(9, 12)], expected=[(12.42, 30.0)]),
    dict(name="a first segment squeezed under min_len goes; the next opens instead",
         segments=[S(10, 12), S(40, 50)], words=[], cuts=[], blocked=[S(9, 11.5)],
         expected=[(40.0, 50.0)]),
    dict(name="the next opening is checked too",
         segments=[S(10, 12), S(40, 50)], words=[], cuts=[], blocked=[S(9, 11.5), S(39, 42)],
         expected=[(42.0, 50.0)]),
    dict(name="blocked windows as unsorted pairs are normalized",
         segments=[S(10, 30)], words=[], cuts=[], blocked=[[25, 40], [9, 12]],
         expected=[(12.0, 30.0)]),

    # ---- general --------------------------------------------------------------
    dict(name="cold-open order kept", segments=[S(50, 60, "picture"), S(10, 30)],
         words=[], cuts=[], expected=[(50.0, 60.0), (10.0, 30.0)]),
    dict(name="a segment replaying earlier footage is cut to the new part",
         segments=[S(10, 30), S(25, 40)], words=[], cuts=[], expected=[(10.0, 30.0), (30.0, 40.0)]),
    dict(name="no words: cuts still steer both ends",
         # start: 10.0 is within word_window of 9.7; end: 20.3 is within the hold reach
         segments=[S(9.7, 20.1)], words=[], cuts=C(10.0, 20.3), expected=[(10.0, 20.28)]),
    dict(name="no signals at all: kept as is", segments=[S(9.7, 20.1)], words=[], cuts=[],
         expected=[(9.7, 20.1)]),
    dict(name="rounded to 2 dp", segments=[S(1.23456, 7.65432)], words=[], cuts=[],
         expected=[(1.23, 7.65)]),
    dict(name="too short to keep", segments=[S(10, 10.8)], words=[], cuts=[], expected=[]),
    dict(name="unusable segments are skipped",
         segments=[None, {"start": "x", "end": 3}, S(5, 4), {"start": 1}, S(0, 10)],
         words=[], cuts=[], expected=[(0.0, 10.0)]),
    dict(name="cuts as bare times and pairs",
         segments=[S(9.5, 20, "picture"), S(30, 40, "picture")], words=[],
         cuts=[10.0, (30.4, 55.0), (29.0, 5.0)], expected=[(10.0, 20.0), (30.4, 40.0)]),
]


class SnapTableTest(unittest.TestCase):
    def test_rules(self):
        for case in CASES:
            with self.subTest(case["name"]):
                segments = copy.deepcopy(case["segments"])
                out = snap.snap_segments(
                    segments, words=case["words"], cuts=case["cuts"],
                    quiet=case.get("quiet", []), floor=case.get("floor", 0.0),
                    horizon=case.get("horizon", 1000.0), blocked=case.get("blocked", ()))
                self.assertEqual([(s["start"], s["end"]) for s in out], case["expected"])
                self.assertEqual(segments, case["segments"], "input was mutated")
                if "leads" in case:
                    self.assertEqual([s["lead"] for s in out], case["leads"])


class SnapContractTest(unittest.TestCase):
    def test_lead_and_extra_keys_are_kept(self):
        out = snap.snap_segments(
            [S(0, 10, "picture", note="x"), S(20, 30, "sideways"), S(40, 50)],
            words=[], cuts=[], quiet=[], floor=0, horizon=100)
        self.assertEqual(out, [{"start": 0.0, "end": 10.0, "lead": "picture", "note": "x"},
                               {"start": 20.0, "end": 30.0, "lead": "word"},
                               {"start": 40.0, "end": 50.0, "lead": "word"}])

    def test_settings_are_honoured(self):
        # A wider picture window reaches a cut the default (1.0) would not.
        args = dict(words=[], cuts=C(11.5), quiet=[], floor=0, horizon=100)
        self.assertEqual(snap.snap_segments([S(10, 20, "picture")], **args)[0]["start"], 10.0)
        self.assertEqual(snap.snap_segments([S(10, 20, "picture")], picture_window=2.0,
                                            **args)[0]["start"], 11.5)
        # A lower min_shot_score admits a weak cut.
        weak = dict(words=[], cuts=C(10.2, score=10), quiet=[], floor=0, horizon=100)
        self.assertEqual(snap.snap_segments([S(10, 20, "picture")], **weak)[0]["start"], 10.0)
        self.assertEqual(snap.snap_segments([S(10, 20, "picture")], min_shot_score=5,
                                            **weak)[0]["start"], 10.2)
        # min_len decides what survives.
        self.assertEqual(snap.snap_segments([S(10, 11.5)], min_len=2.0, words=[], cuts=[],
                                            quiet=[], floor=0, horizon=100), [])
        # start_pad / end_pad.
        out = snap.snap_segments([S(9.9, 12.0)], start_pad=0.2, end_pad=0.1,
                                 words=W((10.0, 10.5), (11.0, 11.8)), cuts=[], quiet=[],
                                 floor=0, horizon=100)
        self.assertEqual((out[0]["start"], out[0]["end"]), (9.8, 11.9))

    def test_no_horizon_means_no_upper_clamp(self):
        out = snap.snap_segments([S(0, 5000)], words=[], cuts=[], quiet=[], floor=0,
                                 horizon=None)
        self.assertEqual(out[0]["end"], 5000.0)

    def test_messy_words_are_tolerated(self):
        words = [{"word": "a", "start": "x", "end": 1}, None, {"word": "b", "start": 3},
                 {"word": "c", "start": 10.5, "end": 10.9}, {"word": "d", "start": 12, "end": 11}]
        out = snap.snap_segments([S(10.0, 20)], words=words, cuts=[], quiet=[[5, "y"], None],
                                 floor=0, horizon=100)
        self.assertEqual((out[0]["start"], out[0]["end"]), (10.42, 20.0))


class RoundedTimesTest(unittest.TestCase):
    """The editor copies its times off transcript entries and the shot log,
    which show them to a tenth of a second. Whisper's words mostly touch, so
    an edge rounded by up to 0.05 s often lands inside the neighbouring word.
    Snapping must still put each boundary on the edge that was meant: never
    a word from outside the stretch the times stand for, never one of its
    own words dropped."""

    def stream(self, seed: int, count: int = 800) -> list[dict]:
        """Whisper-like words (2 dp, each longer than 0.1 s, so a rounded
        edge can only have meant one of its neighbours): mostly touching,
        some short gaps, some pauses."""
        rng = random.Random(seed)
        words, t = [], 1.0
        for i in range(count):
            start = round(t + rng.choice([0.0, 0.0, 0.0, 0.0, 0.01, 0.03, 0.2, 0.9]), 2)
            end = round(start + rng.uniform(0.12, 0.7), 2)
            words.append({"word": f"w{i}", "start": start, "end": end})
            t = end
        return words

    def test_rounded_entry_times_snap_to_the_edges_they_stand_for(self):
        for seed in (1, 2, 3):
            words = self.stream(seed)
            rng = random.Random(seed)
            # Clause-sized entries (1-12 words), as the editor is shown them.
            bounds, i = [], 0
            while i < len(words):
                n = rng.randint(1, 12)
                bounds.append((i, min(i + n, len(words)) - 1))
                i += n
            # Every stretch of two neighbouring entries, cut at their shown
            # times (the ones too short to keep are skipped).
            for (first, _), (_, last) in zip(bounds, bounds[1:]):
                a = round(words[first]["start"], 1)
                b = round(words[last]["end"], 1)
                if b - a < 1.5:
                    continue
                with self.subTest(seed=seed, start=a, end=b):
                    out = snap.snap_segments([S(a, b)], words=words, cuts=[], quiet=[],
                                             floor=0.0, horizon=None)
                    self.assertEqual(len(out), 1)
                    start, end = out[0]["start"], out[0]["end"]
                    if first > 0:   # nothing of the word before
                        self.assertGreaterEqual(start, words[first - 1]["end"])
                    self.assertLessEqual(start, words[first]["start"])
                    if last + 1 < len(words):   # nothing of the word after
                        self.assertLessEqual(end, words[last + 1]["start"])
                    # At most the 0.02 s clearance off a last word that touches
                    # the next one.
                    self.assertGreaterEqual(end, round(words[last]["end"] - 0.02, 2))


class WordLookupTest(unittest.TestCase):
    def test_inside_next_prev_with_overlapping_words(self):
        words = snap._Words(W((1.0, 1.5), (1.4, 2.0), (3.0, 3.2)))
        self.assertEqual(words.inside(1.45), (1.4, 2.0))   # the later word wins
        self.assertEqual(words.inside(1.2), (1.0, 1.5))
        self.assertIsNone(words.inside(1.0))               # strictly inside only
        self.assertIsNone(words.inside(2.5))
        self.assertEqual(words.next(1.45), (3.0, 3.2))
        self.assertEqual(words.next(3.0), (3.0, 3.2))
        self.assertIsNone(words.next(3.1))
        self.assertEqual(words.prev(2.5), (1.4, 2.0))
        self.assertIsNone(words.prev(1.45))
        self.assertFalse(snap._Words([]))

    def test_a_long_word_far_back_is_still_found(self):
        words = snap._Words(W((0.0, 9.0), (1.0, 1.2), (2.0, 2.2)))
        self.assertEqual(words.inside(5.0), (0.0, 9.0))


if __name__ == "__main__":
    unittest.main()
