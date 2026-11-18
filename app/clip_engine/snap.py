"""Snap clip boundaries onto word edges, shot changes and pauses.

An editor model works from 1 fps pictures, transcript times rounded to a tenth
of a second and a shot log, so its boundaries land a few tenths off where a
human editor would cut. Those tenths are what make a clip look amateur: a word
clipped in half, three frames of the anchor before the b-roll, the next shot
blinking in after the last line. This moves each boundary to where an editor
would put it, using three exact signals: Whisper word edges, the shot map's
cuts (scdet) and its quiet stretches (RMS valleys).

How a segment STARTS depends on what it opens on (``lead``):

- ``"word"`` (default): someone speaking. Start just before the first word —
  never inside a word and never on the tail of the one before — and prefer a
  shot change in that gap, so the picture changes with the voice.
- ``"picture"``: an image (b-roll, an action, a face). Start ON the shot
  change. When narration runs across that cut (voice-over b-roll, or a J-cut
  where the speaker's face arrives a beat after their first word), move
  FORWARD to the next gap between words instead of back onto the previous
  shot — backing up would open on 0.1-1 s of the anchor, which is exactly the
  weak opening this engine exists to fix.

ENDS land just after the last word with a short tail (ended inside the pause
when there is one), never let the next shot flash in, and keep a deliberate
hold past the line — a reaction, a look, an image landing after the words.

Neither boundary is left on a flash of a shot shorter than ``flash_guard``
when moving off it splits no word. Blocked (already used) footage is never
entered. Pure: no I/O and no settings; the engine passes the knobs in.
"""
from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass

from .. import clip_proposals

log = logging.getLogger("clip_engine.snap")

LEADS = ("word", "picture")

# Speech this soon after a start makes the start word-led.
_SPEECH_WITHIN = 0.8
# A picture lead whose cut lands inside a word moves to that word's end when
# the word ends this soon after the cut; a longer word is cut on the shot.
_WORD_GAP_REACH = 0.8
# Clearance kept between a boundary and the word or shot it avoids.
_GAP = 0.02
# An end this close after the last word is "on the line" and gets tightened;
# farther out it is a deliberate hold and is kept.
_NEAR_LAST_WORD = 0.5
# Where a pause after the last word may begin, relative to the word's end...
_QUIET_BEFORE = 0.1
_QUIET_AFTER = 0.5
# ...how far into it the clip ends, and the least tail left after the word.
_INTO_QUIET = 0.15
_MIN_TAIL = 0.05
# A held ending moves onto a cut within this reach of it.
_HOLD_BACK = 0.3
_HOLD_FORWARD = 0.5


@dataclass(frozen=True)
class _Rules:
    floor: float
    horizon: float
    word_window: float
    picture_window: float
    flash_guard: float
    start_pad: float
    end_pad: float
    min_len: float


class _Words:
    """A Whisper word stream with the lookups the rules are written in.

    ``inside(t)``: the word with ``start < t < end``; ``next(t)``: the first
    word starting at or after ``t``; ``prev(t)``: the last word ending at or
    before ``t``. Words are ``(start, end)`` tuples; bad entries are dropped.
    """

    def __init__(self, words) -> None:
        spans: list[tuple[float, float]] = []
        for w in words or []:
            if not isinstance(w, dict):
                continue
            try:
                start, end = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and end >= start:
                spans.append((start, end))
        spans.sort()
        self._by_start = spans
        self._starts = [s for s, _ in spans]
        self._by_end = sorted(spans, key=lambda w: (w[1], w[0]))
        self._ends = [e for _, e in self._by_end]
        self._longest = max((e - s for s, e in spans), default=0.0)

    def __bool__(self) -> bool:
        return bool(self._by_start)

    def inside(self, t: float) -> tuple[float, float] | None:
        i = bisect.bisect_left(self._starts, t) - 1   # last word starting before t
        while i >= 0:
            start, end = self._by_start[i]
            if t - start > self._longest:
                break   # nothing that starts this early can still be running at t
            if end > t:
                return start, end
            i -= 1
        return None

    def next(self, t: float) -> tuple[float, float] | None:
        i = bisect.bisect_left(self._starts, t)
        return self._by_start[i] if i < len(self._by_start) else None

    def prev(self, t: float) -> tuple[float, float] | None:
        i = bisect.bisect_right(self._ends, t) - 1
        return self._by_end[i] if i >= 0 else None


def _pair(item, first_key: str, second_key: str):
    if isinstance(item, dict):
        return item.get(first_key), item.get(second_key)
    if isinstance(item, (list, tuple)):
        return (item[0] if item else None), (item[1] if len(item) > 1 else None)
    return item, None


def _cut_times(cuts, min_score: float) -> list[float]:
    """Sorted times of the cuts strong enough to snap to. Accepts the shot
    map's ``{"t", "score"}`` dicts, ``(t, score)`` pairs or bare times (a cut
    without a score counts as strong enough)."""
    times: set[float] = set()
    for cut in cuts or []:
        t, score = _pair(cut, "t", "score")
        try:
            t = float(t)
            score = float(min_score) if score is None else float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and score >= min_score:
            times.add(t)
    return sorted(times)


def _spans(items) -> list[tuple[float, float]]:
    """``[[s, e]]`` / ``[{start, end}]`` as sorted ``(s, e)`` tuples."""
    out: list[tuple[float, float]] = []
    for item in items or []:
        start, end = _pair(item, "start", "end")
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            out.append((start, end))
    out.sort()
    return out


def _between(times: list[float], lo: float, hi: float) -> list[float]:
    """The sorted ``times`` inside ``[lo, hi]``."""
    return times[bisect.bisect_left(times, lo):bisect.bisect_right(times, hi)]


def _nearest(candidates, target: float) -> float | None:
    """The candidate closest to ``target``; the earlier one on a tie."""
    return min(candidates, key=lambda t: (abs(t - target), t), default=None)


def _first_word(a: float, b: float, words: _Words):
    """``(first, speech)``: the first word starting inside the segment, and
    whether it starts soon enough after ``a`` to make the opening word-led."""
    first = words.next(a)
    if first is None or first[0] >= b:
        return None, False
    return first, first[0] - a <= _SPEECH_WITHIN


def _picture_start(a: float, b: float, words: _Words, cuts: list[float],
                   r: _Rules) -> float | None:
    """START for a picture lead, or None when no cut is close enough (the
    word rules then decide). Opens on the cut nearest ``a``; when a word spans
    that cut and ends soon after it, opens at the gap after that word instead —
    on the new shot, without half a word in front."""
    reach = _between(cuts, a - r.picture_window, a + r.picture_window)
    t = _nearest((c for c in reach if r.floor <= c < b - r.min_len), a)
    if t is None:
        return None
    word = words.inside(t)
    if word and word[1] - t <= _WORD_GAP_REACH:
        start = word[1] + _GAP
        following = words.next(word[1])
        if following is not None:
            # Continuous speech can leave a gap under 20 ms: stay inside it
            # rather than open on the first milliseconds of the next word.
            start = min(start, max(word[1], following[0]))
        # The gap must also leave the segment its minimum length; when it
        # wouldn't, opening on the cut mid-word beats dropping the segment.
        if start < b - r.min_len:
            return start
    return t


def _word_start(a: float, b: float, words: _Words, cuts: list[float],
                r: _Rules) -> tuple[float, tuple | None, bool]:
    """START for a word lead: ``(a, first, speech)``.

    A start inside a word backs up to that word's start. When speech begins
    within 0.8 s the start tightens to just before the first word — but never
    onto the previous word's tail. A cut near that start which splits no word
    and comes no later than the first word wins: the picture changes with the
    voice. The cut may not reach back across an earlier word either, which
    would open the clip on a stray "and".
    """
    word = words.inside(a)
    if word:
        a = word[0]
    first, speech = _first_word(a, b, words)
    if speech:
        before = words.prev(first[0])
        tightened = max(first[0] - r.start_pad,
                        before[1] + _GAP if before else r.floor)
        # Words that touch leave no gap: the first word's own start is the cut.
        tightened = min(tightened, first[0])
        latest = first[0]
    else:
        tightened, latest = a, b - r.min_len
    behind = words.prev(tightened)
    earliest = behind[1] if behind else -math.inf
    near = _between(cuts, tightened - r.word_window, tightened + r.word_window)
    t = _nearest((c for c in near
                  if max(earliest, r.floor) <= c <= latest and words.inside(c) is None),
                 tightened)
    return (tightened if t is None else t), first, speech


def _head_guard(a: float, b: float, first, speech: bool, words: _Words,
                cuts: list[float], r: _Rules) -> float:
    """Don't open on a flash: while a cut follows the start within
    ``flash_guard``, start on that cut instead — as long as doing so splits
    no word, skips no word, and leaves the segment its minimum length."""
    latest = b - r.min_len
    if speech:
        latest = min(latest, first[0])
    for _ in range(len(cuts) + 1):
        upcoming = words.next(a)
        t = next((c for c in _between(cuts, a, a + r.flash_guard)
                  if a < c < a + r.flash_guard and c <= latest
                  and words.inside(c) is None
                  and (upcoming is None or upcoming[0] >= c)), None)
        if t is None:
            break
        a = t
    return a


def _end(a: float, b: float, words: _Words, cuts: list[float],
         quiet: list[tuple[float, float]], r: _Rules) -> tuple[float, tuple | None]:
    """END: ``(b, last)`` where ``last`` is the segment's last word, if any.

    An end inside a word extends to the word's end. An end on the line (within
    0.5 s of the last word) is re-cut: ``end_pad`` after the word, or 0.15 s
    into the pause that follows it, never into the next word, and onto the
    shot change when one falls in that tail (ending before it — no flash of
    the next shot) or just after it. An end held well past the last word is a
    choice and stays, moving only onto a cut right next to it.
    """
    word = words.inside(b)
    if word:
        b = word[1]
    last = words.prev(b)
    if last is not None and last[1] <= a:
        last = None
    nxt = words.next(b)
    if last is not None and b - last[1] <= _NEAR_LAST_WORD:
        cap = nxt[0] - _GAP if nxt else r.horizon
        end = min(last[1] + r.end_pad, cap)
        pause = next((q for q in quiet
                      if last[1] - _QUIET_BEFORE <= q[0] <= last[1] + _QUIET_AFTER), None)
        if pause is not None:
            end = min(max(last[1] + _MIN_TAIL, pause[0] + _INTO_QUIET), cap)
        inner = next((c for c in _between(cuts, last[1], end) if c > last[1]), None)
        if inner is not None:
            end = max(inner - _GAP, last[1])
        else:
            after = next((c for c in _between(cuts, end, end + r.word_window)
                          if c > end and c - _GAP <= r.horizon
                          and (nxt is None or c <= nxt[0] - _GAP)), None)
            if after is not None:
                end = after - _GAP
        return end, last
    hold = [c for c in _between(cuts, b - _HOLD_BACK, b + _HOLD_FORWARD)
            if (nxt is None or c <= nxt[0]) and c - _GAP <= r.horizon
            and c - _GAP - a >= r.min_len]
    t = _nearest(hold, b)
    return (b if t is None else t - _GAP), last


def _tail_guard(a: float, b: float, last, cuts: list[float], r: _Rules) -> float:
    """Don't end on a flash: while a cut falls within ``flash_guard`` before
    the end, end before it — unless that would cut into the last word or take
    the segment under its minimum length."""
    for _ in range(len(cuts) + 1):
        t = next((c for c in _between(cuts, b - r.flash_guard, b)
                  if b - r.flash_guard < c < b
                  and (last is None or c >= last[1])
                  and c - _GAP - a >= r.min_len), None)
        if t is None:
            break
        b = t - _GAP if last is None else max(t - _GAP, last[1])
    return b


def _unblock(a: float, b: float,
             blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``[a, b]`` minus blocked footage, as pieces in time order: a start
    inside a blocked range moves to its end, an end inside one moves to its
    start, and a segment straddling one splits around it."""
    pieces = [(a, b)]
    for lo, hi in blocked:
        nxt: list[tuple[float, float]] = []
        for start, end in pieces:
            if hi <= start or lo >= end:
                nxt.append((start, end))
                continue
            if lo > start:
                nxt.append((start, lo))
            if hi < end:
                nxt.append((hi, end))
        pieces = nxt
    return pieces


def _coerce(segment) -> tuple[float, float] | None:
    if not isinstance(segment, dict):
        return None
    try:
        start, end = float(segment["start"]), float(segment["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        return None
    return start, end


def snap_segments(segments: list[dict], *, words: list[dict], cuts: list[dict],
                  quiet: list[list[float]], floor: float, horizon: float,
                  blocked=(), word_window: float = 0.6, picture_window: float = 1.0,
                  flash_guard: float = 0.4, min_shot_score: float = 20.0,
                  start_pad: float = 0.08, end_pad: float = 0.25,
                  min_len: float = 1.0) -> list[dict]:
    """Snap every segment's boundaries; PLAY order kept.

    ``segments``: ``[{start, end, lead?}]`` on the source timeline (other keys
    ride along; ``lead`` defaults to "word"). ``words``: Whisper words.
    ``cuts``: shot-map cuts ``[{t, score}]`` (only ``score >= min_shot_score``
    count). ``quiet``: shot-map pauses ``[[s, e]]``. ``floor``/``horizon``:
    the stretch the clip may use (a section, or the whole source).
    ``blocked``: footage other clips already use.

    Any signal can be empty: no words skips the word rules, no cuts or quiet
    skips theirs. Segments left shorter than ``min_len`` are dropped; the
    result goes through ``clip_proposals.clean_play_order`` (de-overlapped,
    2-dp times).
    """
    lo = 0.0 if floor is None else float(floor)
    hi = math.inf if horizon is None else float(horizon)
    rules = _Rules(floor=lo, horizon=hi, word_window=float(word_window),
                   picture_window=float(picture_window), flash_guard=float(flash_guard),
                   start_pad=float(start_pad), end_pad=float(end_pad),
                   min_len=float(min_len))
    stream = _Words(words)
    shot_cuts = _cut_times(cuts, float(min_shot_score))
    pauses = _spans(quiet)
    used = [(s["start"], s["end"]) for s in clip_proposals.normalize(
        [{"start": s, "end": e} for s, e in _spans(blocked)])]

    out: list[dict] = []
    for segment in segments or []:
        span = _coerce(segment)
        if span is None:
            continue
        a, b = span
        lead = segment.get("lead") if segment.get("lead") in LEADS else "word"

        start = (_picture_start(a, b, stream, shot_cuts, rules)
                 if lead == "picture" and shot_cuts else None)
        if start is not None:
            first, speech = _first_word(start, b, stream)
        else:
            start, first, speech = _word_start(a, b, stream, shot_cuts, rules)
        start = _head_guard(start, b, first, speech, stream, shot_cuts, rules)

        end, last = _end(start, b, stream, shot_cuts, pauses, rules)
        end = _tail_guard(start, end, last, shot_cuts, rules)

        start, end = max(start, lo), min(end, hi)
        for piece_start, piece_end in _unblock(start, end, used):
            if piece_end - piece_start >= rules.min_len - 1e-9:
                out.append({**segment, "start": piece_start, "end": piece_end,
                            "lead": lead})
        if log.isEnabledFor(logging.DEBUG):
            log.debug("snap %s %.2f-%.2f -> %.2f-%.2f", lead, a, b, start, end)

    return clip_proposals.clean_play_order(
        out, min_len=rules.min_len, floor=lo, horizon=None if math.isinf(hi) else hi)
