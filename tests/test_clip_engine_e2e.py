"""End to end: the clip engine on a real (synthetic) video through real ffmpeg,
with only the model calls faked.

Every module runs for real — shotlog, editor, snap, media (proxies, shot map,
renders, frames), critic, playbook, engine, jobs, legacy, sidecars,
clip_proposals and clip_context on an in-memory database. What is faked:

- the model calls: ``gemini.generate_json`` (one stand-in that answers as the
  perceiver, editor, critic, ranker, reviser or judge depending on the schema
  it is asked for), ``llm.suggest_clips`` and ``llm.pick_opening_frame`` (the
  transcript suggester's two calls);
- the network: ``gemini._send`` serves a fake Files API (uploads), and
  ``offload.ensure_local`` returns the local file;
- nothing else, except a pass-through spy on ``media.render_cached`` so the
  fake critic can tell which draft a render it is shown belongs to (the real
  render still runs).

The source is 40 s: a yellow shot, a hard cut to blue at 20 s and to white at
32 s, under a tone that drops out from 14 s to 15 s. A synthetic word stream
has a narrator's word spanning the 20 s cut (voice-over across b-roll), which
is exactly where a naive snapper opens on the previous shot.

Skipped unless ffmpeg and ffprobe are on PATH. Run with the suite's §0 prefix:
  env DATABASE_URL="sqlite://" ... .venv/bin/python -m pytest -q tests/test_clip_engine_e2e.py
"""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import requests
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import llm, offload, spend
from app.clip_engine import claude, engine, gemini, jobs, media, sidecars
from app.clip_engine import playbook as playbook_mod
from app.config import Settings
from app.models import STATUS_ARCHIVED, Base, Candidate, Channel, ClipProposal

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

DURATION = 40.0
CUTS = (20.0, 32.0)
TITLE = "Bill Nye: 'The Planet's on Fire' — Safety Glasses Off"
MODEL = "gemini-3.8-flash"
BRAND = {"name": "Renewables.org", "topic": "climate change"}


# ---- the synthetic source ---------------------------------------------------------------

def make_source(path: Path) -> Path:
    """Yellow 0-20 s, blue 20-32 s, white 32-40 s (hard cuts: big luma jumps,
    so scdet scores them far above the snapper's min_shot_score), under a
    440 Hz tone that is silent from 14 s to 15 s."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=yellow:s=320x180:r=25:d=20",
         "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=12",
         "-f", "lavfi", "-i", "color=c=white:s=320x180:r=25:d=8",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=40",
         "-filter_complex",
         "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];"
         "[3:a]volume=enable='between(t,14,15)':volume=0,aformat=channel_layouts=stereo[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True, timeout=120)
    return path


def make_words() -> list[dict]:
    """Nye speaks (pausing 14-15 s), a narrator's word spans the 20 s cut, the
    narration runs over the aerials, then an anchor reads sports scores."""
    words: list[dict] = []

    def stretch(start: float, stop: float, prefix: str) -> None:
        k = 0
        while start + 0.45 * k < stop:
            t = round(start + 0.45 * k, 2)
            words.append({"word": f"{prefix}{k}", "start": t, "end": round(t + 0.32, 2)})
            k += 1

    stretch(0.5, 13.9, "nye")
    stretch(15.1, 19.5, "nye")
    words.append({"word": "across", "start": 19.8, "end": 20.25})
    stretch(20.4, 31.8, "fire")
    stretch(32.3, 39.4, "score")
    return words


def make_transcript(words: list[dict]) -> list[dict]:
    """Caption-style lines of eight words."""
    lines = []
    for i in range(0, len(words), 8):
        group = words[i:i + 8]
        lines.append({"start": group[0]["start"], "end": group[-1]["end"],
                      "text": " ".join(w["word"] for w in group)})
    return lines


def colour_at(video: Path, seconds: float) -> str:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{seconds:.3f}",
                          "-i", str(video), "-frames:v", "1", "-vf", "scale=1:1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True, timeout=60).stdout
    r, g, b = raw[0], raw[1], raw[2]
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r > 150 and g > 150 and b < 110:
        return "yellow"
    if b > 150 and r < 110 and g < 110:
        return "blue"
    return f"rgb({r},{g},{b})"


def jpeg_colour(data: bytes) -> str:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "image2pipe", "-i", "-",
                          "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         input=data, check=True, capture_output=True, timeout=60).stdout
    r, g, b = raw[0], raw[1], raw[2]
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r > 150 and g > 150 and b < 110:
        return "yellow"
    if b > 150 and r < 110 and g < 110:
        return "blue"
    return f"rgb({r},{g},{b})"


def is_inside_a_word(t: float, words: list[dict]) -> bool:
    return any(w["start"] < t < w["end"] for w in words)


# ---- what the models say ------------------------------------------------------------------

def shot_log_answer(*, on_topic: bool = True) -> dict:
    """The perceiver's log, as Gemini writes it: MM:SS positions."""
    return {
        "summary": "Bill Nye warns the planet is on fire; aerials of a wildfire; sports scores.",
        "format": "local news package",
        "structure": "00:00-00:20 Nye speech; 00:20-00:32 wildfire aerials; 00:32-00:40 sports",
        "people": [{"name": "Bill Nye", "role": "science educator", "prominence": "famous",
                    "evidence": "title"}],
        "beats": [
            {"start": "00:00", "end": "00:20", "shot": "talking_head",
             "on_screen": "Bill Nye at a lectern, safety glasses on", "face": "close",
             "speaker": "Bill Nye", "line": "The planet's on fire.", "sound": "",
             "energy": 4, "visual": 3, "topic": 2 if on_topic else 0, "note": ""},
            {"start": "00:20", "end": "00:32", "shot": "aerial",
             "on_screen": "00:20.0 aerial of a wildfire cresting a ridge", "face": "none",
             "speaker": "narrator", "line": "Crews are losing the ridge.", "sound": "rotors",
             "energy": 4, "visual": 5, "topic": 2 if on_topic else 0, "note": ""},
            {"start": "00:32", "end": "00:40", "shot": "graphic",
             "on_screen": "full-screen baseball scores", "face": "none", "speaker": "anchor",
             "line": "And in baseball tonight...", "sound": "", "energy": 2, "visual": 1,
             "topic": 0, "note": "next story"},
        ],
        "highlights": [{"start": "00:21", "end": "00:24", "kind": "visual",
                        "what": "flames crest the ridge"}],
    }


def _draft(did, moment, archetype, lens, story_key, spans) -> dict:
    return {"id": did, "moment_id": moment, "archetype": archetype, "lens": lens,
            "story": f"the {story_key} clip", "story_key": story_key,
            "hook": f"hook of {did}", "payoff": f"payoff of {did}",
            "segments": [{"start": a, "end": b, "lead": lead} for a, b, lead in spans],
            "why": f"why {did}", "risk": f"risk {did}", "draft_caption": f"Caption {did}",
            "challenges_rules": ""}


DRAFTS_ANSWER = {
    "moments": [
        {"id": "m1", "start": 0.5, "end": 19.9, "kind": "person", "why": "Nye's warning"},
        {"id": "m2", "start": 20.0, "end": 31.9, "kind": "picture", "why": "the fire from the air"},
        {"id": "m3", "start": 32.0, "end": 39.6, "kind": "line", "why": "the scores"},
    ],
    "drafts": [
        _draft("d1", "m1", "voice", "person", "nye-fire", [(2.05, 17.6, "word")]),
        # A cold open: the aerials' narration first, then Nye's setup.
        _draft("d2", "m1", "voice", "line", "nye-fire",
               [(25.2, 31.0, "word"), (4.1, 11.9, "word")]),
        # Picture-led, a beat EARLY (the editor works from 1 fps pictures): the
        # start lands inside the narrator's word that spans the 20 s cut. The
        # word rule would back up onto the yellow shot; the picture rule opens
        # on the blue one, at the next word gap.
        _draft("d3", "m2", "scene", "picture", "wildfire-aerials", [(19.9, 31.6, "picture")]),
        _draft("d4", "m3", "other", "line", "sports-scores", [(32.6, 39.4, "picture")]),
        # The same cut as d1 to within a word: a true duplicate.
        _draft("d5", "m1", "voice", "person", "nye-fire", [(2.1, 17.55, "word")]),
    ],
}

REVISION_ANSWER = {"change": "end on the ridge line; drop the trailing narration",
                   "archetype": "scene",
                   "segments": [{"start": 19.9, "end": 29.9, "lead": "picture"}]}


def critique(score, *, topic="central", topic_note="", visual=3, stop_reason="", **extra) -> dict:
    out = {"first_impression": "what the first two seconds show", "stops_scroll_muted": True,
           "muted_reason": "motion in frame one", "stops_scroll": True,
           "stop_reason": stop_reason or "it opens on the point", "person_at_start": False,
           "about": "a clip", "makes_sense": True, "dangling": "", "bored_at": None,
           "dead_air": [], "ends_well": True, "topic": topic, "topic_note": topic_note,
           "person": "", "visual": visual, "audio": 3, "score": score,
           "fix": "start a beat later"}
    out.update(extra)
    return out


ON_TOPIC_CRITIQUES = {
    "voice": critique(8, person_at_start=True, stop_reason="Bill Nye mid-warning, face first"),
    "cold": critique(7, topic="incidental", topic_note="the fire is never tied to climate"),
    "picture": critique(6, visual=4, stop_reason="flames crest a ridge in the first second"),
    "offtopic": critique(5, topic="none", topic_note="baseball scores"),
    "revision": critique(7, visual=4, stop_reason="the fire, then straight to the line"),
}
OFF_TOPIC_CRITIQUES = {label: critique(5, topic="none", topic_note="nothing about climate")
                       for label in ("voice", "cold", "picture", "offtopic", "revision")}

# Who the fake judge prefers, best first.
JUDGE_ORDER = ("picture", "revision", "voice", "cold", "offtopic")

COSTS = {"perceive": 0.040, "draft": 0.060, "cold": 0.005, "rank": 0.012, "revise": 0.015,
         "prefer": 0.007}


def classify(segments: list[dict]) -> str:
    """Which scripted draft a snapped cut is, by where it opens."""
    first = float(segments[0]["start"])
    if first >= 31.5:
        return "offtopic"
    if first >= 24.0:
        return "cold"
    if first >= 19.0:
        return "picture"
    return "voice"


def _response(status: int = 200, body=None, headers=None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body if body is not None else {}).encode()
    r.headers.update(headers or {})
    r.encoding = "utf-8"
    return r


class FakeFilesAPI:
    """Google's Files API as ``gemini._send`` meets it: a resumable upload
    start, the byte transfer, GET and DELETE. Anything else (generateContent
    above all — it is faked whole) is a test failure. ``depleted`` answers the
    upload start the way a project with no prepaid credit does (seen live)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.files: dict[str, bytes] = {}
        self.uploads = 0
        self.unexpected: list[str] = []
        self.depleted = False

    def __call__(self, method: str, url: str, **kwargs):
        with self.lock:
            if method == "POST" and url == gemini._UPLOAD_URL:
                if self.depleted:
                    return _response(402, {"error": {
                        "code": 402, "status": "RESOURCE_EXHAUSTED",
                        "message": "Your prepayment credits are depleted. Please go to AI "
                                   "Studio at https://ai.studio/projects to manage your "
                                   "project and billing."}})
                self.uploads += 1
                return _response(200, {}, {"X-Goog-Upload-URL":
                                           f"https://upload.test/{self.uploads}"})
            if method == "POST" and url.startswith("https://upload.test/"):
                n = url.rsplit("/", 1)[1]
                name = f"files/e2e{n}"
                uri = f"{gemini.BASE}/v1beta/{name}"
                self.files[uri] = kwargs["data"].read()
                return _response(200, {"file": {"name": name, "uri": uri,
                                                "mimeType": "video/mp4", "state": "ACTIVE"}})
            if method == "GET" and "/v1beta/files/" in url:
                return _response(200, {"name": url.split("/v1beta/", 1)[1], "state": "ACTIVE"})
            if method == "DELETE":
                return _response(200, {})
            self.unexpected.append(f"{method} {url}")
        raise AssertionError(f"unexpected Gemini request: {method} {url}")


class FakeGemini:
    """``gemini.generate_json`` for every role, picked by the schema asked
    for. A critic, ranker or judge is shown real renders; which draft each
    one is comes from the render spy (sha1 of the bytes -> label)."""

    def __init__(self, files: FakeFilesAPI, *, on_topic: bool = True,
                 perceive_error: BaseException | None = None) -> None:
        self.files = files
        self.on_topic = on_topic
        self.perceive_error = perceive_error
        self.critiques = ON_TOPIC_CRITIQUES if on_topic else OFF_TOPIC_CRITIQUES
        self.lock = threading.Lock()
        self.calls: list[dict] = []
        self.renders: dict[str, str] = {}      # sha1(bytes) -> label
        self.render_paths: dict[str, Path] = {}
        self.revised = False

    # -- the render spy --
    def spy(self, real):
        def render_cached(source, segments, *, height=360, kind="draft"):
            path = real(source, segments, height=height, kind=kind)
            data = Path(path).read_bytes()
            with self.lock:
                label = "revision" if self.revised else classify(segments)
                self.renders[hashlib.sha1(data).hexdigest()] = label
                self.render_paths[label] = Path(path)
            return path
        return render_cached

    def video_label(self, part: dict) -> str:
        if "inlineData" in part:
            data = base64.b64decode(part["inlineData"]["data"])
        else:
            data = self.files.files[part["fileData"]["fileUri"]]
        return self.renders.get(hashlib.sha1(data).hexdigest(), "unknown")

    @staticmethod
    def kind(schema: dict) -> str:
        props = schema.get("properties", {})
        for key, kind in (("beats", "perceive"), ("drafts", "draft"), ("change", "revise"),
                          ("first_impression", "cold"), ("ranking", "rank"),
                          ("choice", "prefer")):
            if key in props:
                return kind
        raise AssertionError(f"unknown schema {sorted(props)}")

    def __call__(self, model, *, system, parts, schema, thinking="LOW",
                 media_resolution="MEDIA_RESOLUTION_LOW", max_output_tokens=32768,
                 timeout_s=900.0, label="", bucket="engine"):
        kind = self.kind(schema)
        call = {"kind": kind, "model": model, "system": system, "parts": parts,
                "bucket": bucket, "thinking": thinking, "label": label}
        if kind == "perceive" and self.perceive_error is not None:
            with self.lock:
                self.calls.append(call)
            raise self.perceive_error
        texts = [p["text"] for p in parts if "text" in p]
        videos = [p for p in parts if "inlineData" in p or "fileData" in p]
        if kind == "perceive":
            data = shot_log_answer(on_topic=self.on_topic)
        elif kind == "draft":
            data = copy.deepcopy(DRAFTS_ANSWER)
        elif kind == "revise":
            with self.lock:
                self.revised = True
            data = copy.deepcopy(REVISION_ANSWER)
        elif kind == "cold":
            call["shows"] = self.video_label(videos[0])
            call["credits"] = texts[-1]
            data = copy.deepcopy(self.critiques[call["shows"]])
        elif kind == "rank":
            letters = self._lettered(parts, r"^Draft ([A-Z]) — ")
            call["shows"] = letters
            ranking = sorted(letters, key=lambda l: JUDGE_ORDER.index(letters[l]))
            data = {"notes": [{"id": l, "note": f"{letters[l]} clip"} for l in letters],
                    "ranking": ranking, "best_reason": "the picture lands first"}
        else:   # prefer
            versions = self._lettered(parts, r"^Version ([12]) — ")
            call["shows"] = versions
            winner = next((n for n, l in versions.items() if l == "revision"), None)
            data = {"choice": winner or "same", "reason": "tighter ending"}
        usage = {"model": model, "input_tokens": 1000 + 10 * len(self.calls),
                 "output_tokens": 200, "cost_usd": COSTS[kind]}
        call["usage"] = usage
        with self.lock:
            self.calls.append(call)
        return data, usage

    def _lettered(self, parts: list[dict], pattern: str) -> dict[str, str]:
        """{letter: label} from "Draft X — ..." / "Version N — ..." text parts,
        each followed by its video."""
        out: dict[str, str] = {}
        for i, part in enumerate(parts):
            match = re.match(pattern, part.get("text", ""))
            if match:
                out[match.group(1)] = self.video_label(parts[i + 1])
        return out

    def of(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["kind"] == kind]


def heard(text: str) -> list[str]:
    """The words under "Words heard (clip seconds):" in a Claude request."""
    _head, _sep, lines = text.partition("Words heard (clip seconds):\n")
    return [w for line in lines.splitlines() if line.startswith("[")
            for w in line.split("] ", 1)[1].split()]


def classify_heard(words: list[str]) -> str:
    """Which scripted draft a clip is, by the first word it plays."""
    first = words[0] if words else ""
    if first.startswith("nye"):
        return "voice"
    if first == "fire0":
        return "picture"
    if first.startswith("fire"):
        return "cold"
    if first.startswith("score"):
        return "offtopic"
    return "unknown"


class FakeClaude:
    """``claude.json_call`` for every role, picked by the schema. A Claude
    critic or judge gets frames and the words heard, so that is how it tells
    the drafts apart — which also proves every judge was given the words."""

    def __init__(self, critiques: dict) -> None:
        self.critiques = critiques
        self.lock = threading.Lock()
        self.calls: list[dict] = []
        self.revised = False

    def __call__(self, model, *, system, content, schema, effort="medium", max_tokens=32000,
                 label="", bucket="engine"):
        kind = FakeGemini.kind(schema)
        texts = [b["text"] for b in content if b.get("type") == "text"]
        images = [b for b in content if b.get("type") == "image"]
        call = {"kind": kind, "model": model, "system": system, "texts": texts,
                "images": images, "bucket": bucket, "effort": effort}
        if kind == "perceive":
            data = shot_log_answer()
        elif kind == "draft":
            data = copy.deepcopy(DRAFTS_ANSWER)
        elif kind == "revise":
            with self.lock:
                self.revised = True
            data = copy.deepcopy(REVISION_ANSWER)
        elif kind == "cold":
            call["heard"] = heard(texts[0])
            call["shows"] = "revision" if self.revised else classify_heard(call["heard"])
            data = copy.deepcopy(self.critiques[call["shows"]])
        elif kind == "rank":
            letters = {}
            for text in texts:
                match = re.match(r"^Draft ([A-Z]) — ", text)
                if match:
                    letters[match.group(1)] = (classify_heard(heard(text)), heard(text))
            call["shows"] = {l: label for l, (label, _w) in letters.items()}
            call["words"] = {l: words for l, (_label, words) in letters.items()}
            ranking = sorted(letters, key=lambda l: JUDGE_ORDER.index(letters[l][0]))
            data = {"notes": [{"id": l, "note": "seen"} for l in letters], "ranking": ranking,
                    "best_reason": "the picture lands first"}
        else:   # prefer: the revision ends a sentence earlier than its original
            versions = {}
            for text in texts:
                match = re.match(r"^Version ([12]) — ", text)
                if match:
                    versions[match.group(1)] = "picture" if "fire24" in heard(text) else "revision"
            call["shows"] = versions
            winner = next((n for n, l in versions.items() if l == "revision"), None)
            data = {"choice": winner or "same", "reason": "tighter ending"}
        usage = {"model": model, "input_tokens": 2000, "output_tokens": 300,
                 "cost_usd": COSTS[kind]}
        call["usage"] = usage
        with self.lock:
            self.calls.append(call)
        return data, usage

    def of(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["kind"] == kind]


def e2e_settings() -> Settings:
    return Settings(raw={
        "clip_engine": {
            "setup": "gemini", "drafts": 6, "section_drafts": 4, "keep": 3, "compare_top": 5,
            "revise_top": 1, "min_seconds": 6, "max_seconds": 30, "max_segments": 4,
            "taste_hits": 3, "taste_flops": 2, "workers": 1, "daily_usd": 5.0,
            "max_usd_per_video": 1.5, "hard_cap_usd": 4.0, "max_source_seconds": 3600,
            "job_timeout_seconds": 900, "proxy_height": 180, "perception_window_seconds": 300,
            "editor_video_max_seconds": 3600, "min_free_gb": 0,
            "gemini": {"model": MODEL},
        },
        "clips": {"model": "claude-haiku-4-5", "max_clips": 3, "section_max_clips": 2,
                  "max_segments": 4, "min_seconds": 6, "max_seconds": 30,
                  "opening_model": "claude-haiku-4-5"},
    })


def make_db():
    """In-memory SQLite shared across threads, with real transactions (explicit
    BEGIN, so SAVEPOINT and rollback behave as on Postgres)."""
    db = create_engine("sqlite://", future=True, poolclass=StaticPool,
                       connect_args={"check_same_thread": False})

    @event.listens_for(db, "connect")
    def _no_implicit_transactions(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(db, "begin")
    def _begin(connection):
        connection.exec_driver_sql("BEGIN")

    Base.metadata.create_all(db)
    return db


# ---- the base: one synthetic source, a private engine dir per test -------------------------

@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
class _EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures = Path(tempfile.mkdtemp(prefix="clip-e2e-"))
        cls.source = make_source(cls.fixtures / "nye.mp4")
        cls.words = make_words()
        cls.transcript = make_transcript(cls.words)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.fixtures, True)

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.files = FakeFilesAPI()
        self.fake = FakeGemini(self.files)
        for patcher in (
            mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.dir / "engine"),
                                         "GEMINI_API_KEY": "test-key"}),
            mock.patch.object(spend, "_LEDGER", self.dir / "llm_spend.json"),
            mock.patch.object(gemini, "_send", self.files),
            mock.patch.object(gemini, "generate_json", self._generate),
            mock.patch.object(media, "render_cached", self.fake.spy(media.render_cached)),
            # The built-in playbook, whatever the workspace has: archetypes
            # feed the scores this test pins.
            mock.patch.object(playbook_mod, "load",
                              lambda: copy.deepcopy(playbook_mod.DEFAULT_PLAYBOOK)),
            # Restore the breaker whatever a test does to it (402 opens it).
            mock.patch.object(gemini, "_breaker_until", 0.0),
            mock.patch.object(gemini, "_breaker_reason", ""),
            mock.patch.object(gemini, "_breaker_why", gemini._DAILY_QUOTA),
            mock.patch.object(gemini, "_breaker_status", 429),
            mock.patch.object(gemini, "_confirmed", {}),
            mock.patch.object(llm, "suggest_clips", side_effect=AssertionError(
                "the transcript suggester ran unexpectedly")),
            mock.patch.object(llm, "pick_opening_frame", side_effect=AssertionError(
                "the opening-frame pass ran unexpectedly")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _generate(self, *args, **kwargs):
        return self.fake(*args, **kwargs)


# ---- the engine on its own ---------------------------------------------------------------

class EngineEndToEndTest(_EndToEnd):
    """engine.run over the synthetic source: snapping on real cuts and words,
    real renders, every filter, the ranking and the revision."""

    def setUp(self) -> None:
        super().setUp()
        self.stages: list[str] = []
        self.result = engine.run(
            setup="gemini", settings=e2e_settings(), brand=BRAND, playbook=None,
            video_id="e2e-nye", title=TITLE, description="A science educator warns.",
            duration=None, source=self.source, words=self.words, transcript=self.transcript,
            context={}, rejected=[], examples=[], used=[],
            meter=engine.CostMeter("engine"), progress=self.stages.append)
        self.by_id = {d["id"]: d for d in self.result.drafts}

    def draft_for(self, label: str) -> dict:
        ids = {"voice": "d1", "cold": "d2", "picture": "d3", "offtopic": "d4"}
        return self.by_id[ids[label]]

    def test_the_run_reaches_proposals_through_every_stage(self):
        self.assertEqual(self.result.outcome, "proposals", self.result.notes)
        self.assertEqual(self.stages, ["Logging the video", "Drafting clips", "Rendering drafts",
                                       "Watching the drafts", "Ranking", "Revising"])
        self.assertEqual(self.result.model, f"engine:gemini:{MODEL}")
        self.assertEqual(self.files.unexpected, [])
        # The perceiver watched an upload of the whole-source proxy; the editor
        # watched the same proxy inline, with the payload as the LAST part.
        self.assertEqual(self.files.uploads, 1)
        draft_call = self.fake.of("draft")[0]
        self.assertIn("inlineData", draft_call["parts"][0])
        payload = json.loads(draft_call["parts"][-1]["text"])
        beats = payload["shot_log"]["beats"]
        self.assertEqual([b["start"] for b in beats], [0.0, 20.0, 32.0])   # MM:SS -> seconds
        self.assertEqual([b.get("cuts") for b in beats], [None, [20.0], [32.0]])  # the shot map
        self.assertEqual(payload["transcript"][0]["start"], 0.5)   # word-accurate entries

    def test_renders_are_real_and_match_their_cuts(self):
        self.assertEqual(len(self.result.top), 3)
        for item in self.result.top:
            render = Path(item["render"])
            self.assertTrue(render.is_file() and render.stat().st_size > 0, render)
            self.assertAlmostEqual(media.probe_duration(render),
                                   engine.seconds_of(item["segments"]), delta=0.3)
        self.assertEqual(colour_at(Path(self.draft_for("voice")["render"]), 0.2), "yellow")

    def test_segments_are_snapped_in_play_order(self):
        # Every boundary sits on a word edge, never inside a word.
        for d in self.result.drafts:
            for s in d["segments"]:
                for t in (s["start"], s["end"]):
                    self.assertFalse(is_inside_a_word(t, self.words), (d["id"], t))
        voice = self.draft_for("voice")
        self.assertNotEqual(voice["segments"], voice["editor_segments"])   # snapping moved it
        self.assertEqual(voice["segments"], [{"start": 1.77, "end": 17.78, "lead": "word"}])
        # A cold open keeps its order: the later moment plays first, in the
        # stored segments and in the render itself.
        cold = self.draft_for("cold")
        self.assertEqual([s["start"] for s in cold["segments"]], [24.82, 4.02])
        first_len = cold["segments"][0]["end"] - cold["segments"][0]["start"]
        self.assertEqual(colour_at(Path(cold["render"]), 0.3), "blue")
        self.assertEqual(colour_at(Path(cold["render"]), first_len + 0.6), "yellow")
        # Picture-led across a cut a word spans: forward to the word gap, on
        # the NEW shot — never back onto the previous one.
        picture = self.draft_for("picture")
        start = picture["segments"][0]["start"]
        self.assertGreater(start, CUTS[0])
        self.assertLess(start, CUTS[0] + 0.8)
        self.assertEqual(colour_at(Path(picture["render"]), 0.04), "blue")

    def test_duplicate_and_off_topic_drafts_are_filtered(self):
        self.assertEqual(self.by_id["d5"]["dropped_reason"], "duplicate")
        self.assertEqual(self.by_id["d5"]["duplicate_of"], "d1")
        self.assertNotIn("d5", self.result.orders["editor"])
        off = self.draft_for("offtopic")
        self.assertEqual(off["dropped_reason"], "off topic")
        self.assertTrue(Path(off["render"]).is_file())          # rendered and watched first
        self.assertEqual(off["critique"]["topic"], "none")
        self.assertNotIn("d4", self.result.orders["cold"])
        self.assertNotIn("d4", [d["id"] for d in self.result.top])
        # A doubt with only one witness stays, flagged for the card.
        cold = self.draft_for("cold")
        self.assertEqual(cold["dropped_reason"], "")
        self.assertIn("never tied to climate", cold["topic_doubt"])

    def test_ranking_and_revision_change_the_order(self):
        orders = self.result.orders
        self.assertEqual(orders["cold"], ["d1", "d3", "d2"])     # 8 / 6.25 / 5 cold
        self.assertEqual(orders["rank"], ["d3", "d1", "d2"])     # the judge's order
        # Two passes (they run in parallel, so either may finish first): the
        # given order and its reverse, letters assigned by position in each.
        rank_calls = self.fake.of("rank")
        self.assertEqual(sorted(tuple(c["shows"].values()) for c in rank_calls),
                         [("cold", "picture", "voice"), ("voice", "picture", "cold")])
        revision = self.result.top[0]
        self.assertEqual((revision["id"], revision["origin"], revision["parent"]),
                         ("d3r", "revision", "d3"))
        self.assertEqual(revision["segments"], [{"start": 20.27, "end": 30.28, "lead": "picture"}])
        self.assertEqual(self.by_id["d3"]["dropped_reason"], "revised")
        self.assertTrue(self.by_id["d3r"]["adopted"])
        # The judge was asked in both orders (in parallel) and the revision won both.
        self.assertEqual(sorted((c["shows"]["1"], c["shows"]["2"]) for c in self.fake.of("prefer")),
                         [("picture", "revision"), ("revision", "picture")])
        self.assertEqual(orders["final"], ["d3r", "d1", "d2"])
        self.assertEqual([d["id"] for d in self.result.top], orders["final"])
        self.assertEqual([d["rank"] for d in self.result.top], [1, 2, 3])

    def test_judges_are_told_who_is_on_screen(self):
        credits = {c["shows"]: c["credits"] for c in self.fake.of("cold")}
        self.assertIn("0-17 s: Bill Nye, science educator (named in the title)", credits["voice"])
        self.assertIn("Bill Nye", credits["cold"])                # the setup half of the cold open
        self.assertIn("Credits: none", credits["picture"])        # nobody on the aerials

    def test_costs_are_summed_by_stage_and_model(self):
        spent = sum(c["usage"]["cost_usd"] for c in self.fake.calls)
        self.assertAlmostEqual(self.result.cost_usd, spent, places=6)
        self.assertEqual({k: round(v, 6) for k, v in self.result.by_stage.items()},
                         {"perceive": 0.04, "draft": 0.06, "cold": 0.025, "rank": 0.024,
                          "revise": 0.015, "prefer": 0.014})
        tokens = self.result.tokens[MODEL]
        self.assertEqual(tokens["output"], 200 * len(self.fake.calls))
        # Every call was billed to the engine's own bucket, never the shared budget.
        self.assertEqual({c["bucket"] for c in self.fake.calls}, {"engine"})

    def test_orders_cover_every_stage_for_the_bench(self):
        orders = self.result.orders
        self.assertEqual(set(orders), {"editor", "cold", "rank", "final"})
        self.assertEqual(orders["editor"], ["d1", "d2", "d3", "d4"])
        result = self.result.to_dict()
        json.dumps(result)   # the run record must serialize
        self.assertEqual(result["orders"], orders)


# ---- the Claude roles: frames and words instead of video ----------------------------------

class ClaudeRolesEndToEndTest(_EndToEnd):
    """hybrid (Gemini logs; Claude drafts, critiques and judges) and claude
    (Claude for everything): the same source, real frames from real renders."""

    def setUp(self) -> None:
        super().setUp()
        self.claude = FakeClaude(ON_TOPIC_CRITIQUES)
        patcher = mock.patch.object(claude, "json_call", self.claude)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_setup(self, setup: str) -> engine.EngineResult:
        result = engine.run(
            setup=setup, settings=e2e_settings(), brand=BRAND, playbook=None,
            video_id=f"e2e-{setup}", title=TITLE, description="A science educator warns.",
            duration=DURATION, source=self.source, words=self.words, transcript=self.transcript,
            context={}, rejected=[], examples=[], used=[], meter=engine.CostMeter("engine"))
        self.assertEqual(result.outcome, "proposals", result.notes)
        return result

    def assert_judged_like_gemini(self, result: engine.EngineResult) -> None:
        self.assertEqual(result.model, "engine:" + result.setup + ":claude-opus-5-5")
        self.assertEqual(result.orders["cold"], ["d1", "d3", "d2"])
        self.assertEqual(result.orders["rank"], ["d3", "d1", "d2"])
        self.assertEqual(result.orders["final"], ["d3r", "d1", "d2"])
        self.assertEqual(result.top[0]["segments"],
                         [{"start": 20.27, "end": 30.28, "lead": "picture"}])
        self.assertEqual({c["bucket"] for c in self.claude.calls}, {"engine"})

    def test_hybrid_claude_reads_frames_and_words(self):
        result = self.run_setup("hybrid")
        self.assert_judged_like_gemini(result)
        self.assertEqual([c["kind"] for c in self.fake.calls], ["perceive"])   # Gemini logs only
        self.assertEqual(sorted(c["kind"] for c in self.claude.calls),
                         ["cold"] * 5 + ["draft", "prefer", "prefer", "rank", "rank", "revise"])

        # The editor can't watch video: it gets keyframes of the strongest
        # shots, each labelled with its source time.
        draft = self.claude.of("draft")[0]
        labels = [t for t in draft["texts"] if t.startswith("t=")]
        self.assertTrue(draft["images"])
        self.assertEqual(len(labels), len(draft["images"]))
        frames = {float(label[2:-1]): image for label, image in zip(labels, draft["images"])}
        aerial = min((t for t in frames if t >= CUTS[0]), default=None)
        self.assertIsNotNone(aerial, labels)            # the visual-5 aerial beat is shown...
        self.assertEqual(jpeg_colour(base64.b64decode(frames[aerial]["source"]["data"])),
                         "blue")                         # ...from just after its cut

        # The critic gets frames of the render on the clip clock and the words.
        for call in self.claude.of("cold"):
            self.assertTrue(call["images"], call["shows"])
            self.assertTrue(call["heard"], call["shows"])
            self.assertTrue(any(t.startswith("clip ") for t in call["texts"]))
        # So does the judge, for every draft in both passes — never "(none)".
        for call in self.claude.of("rank"):
            self.assertEqual(sorted(call["shows"].values()), ["cold", "picture", "voice"])
            for letter, words in call["words"].items():
                self.assertTrue(words, f"draft {letter} reached the judge without its words")
        self.assertEqual(sorted((c["shows"]["1"], c["shows"]["2"])
                                for c in self.claude.of("prefer")),
                         [("picture", "revision"), ("revision", "picture")])

    def test_claude_logs_the_video_from_frames(self):
        result = self.run_setup("claude")
        self.assert_judged_like_gemini(result)
        self.assertEqual(self.fake.calls, [])            # no Gemini at all
        perceive = self.claude.of("perceive")
        self.assertEqual(len(perceive), 1)               # 40 s at 2 s a frame: one chunk
        call = perceive[0]
        labels = [t for t in call["texts"] if t.startswith("t=")]
        self.assertEqual(len(labels), 20)
        self.assertEqual((labels[0], labels[-1]), ("t=0.0s", "t=38.0s"))
        self.assertIn("Transcript for this stretch:\n[0.5] nye0", call["texts"][0])
        self.assertEqual(result.shotlog["perceiver"], "claude")
        self.assertEqual([b["start"] for b in result.shotlog["beats"]], [0.0, 20.0, 32.0])


# ---- the same flow through the dashboard's background runner ------------------------------

class JobsEndToEndTest(_EndToEnd):
    """jobs runs the real engine and writes through the real ledger."""

    def setUp(self) -> None:
        super().setUp()
        jobs._reset()
        self.addCleanup(jobs._reset)
        self.db = make_db()
        self.addCleanup(self.db.dispose)
        self.Session = sessionmaker(self.db, future=True, expire_on_commit=False)

        @contextlib.contextmanager
        def scope(read_only: bool = False):
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

        self.settings = e2e_settings()
        self.ensure_local = mock.MagicMock(side_effect=lambda path, key: Path(path))
        for patcher in (mock.patch.object(jobs, "session_scope", scope),
                        mock.patch.object(jobs, "load_settings", lambda: self.settings),
                        mock.patch.object(jobs, "brand", lambda: dict(BRAND)),
                        mock.patch.object(offload, "ensure_local", self.ensure_local)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cid = self.add_candidate()

    def add_candidate(self) -> int:
        transcript = self.dir / "nye.json"
        transcript.write_text(json.dumps(self.transcript))
        words = self.dir / "nye_words.json"
        words.write_text(json.dumps(self.words))
        with self.Session() as s:
            channel = Channel(call_sign="TEST", url="https://www.youtube.com/@test")
            s.add(channel)
            s.flush()
            c = Candidate(video_id="e2e-nye", channel_pk=channel.id, title=TITLE,
                          url="https://youtu.be/e2e-nye", status=STATUS_ARCHIVED,
                          archived_at=jobs.utcnow(), duration_seconds=40,
                          transcript_path=str(transcript), word_transcript_path=str(words),
                          local_video_path=str(self.source), source_object_key="src/nye.mp4")
            s.add(c)
            s.commit()
            return c.id

    def add_pending(self, segments: list[dict]) -> int:
        with self.Session() as s:
            row = ClipProposal(candidate_pk=self.cid, proposed_segments=json.dumps(segments),
                               verdict=ClipProposal.VERDICT_PENDING, story="an earlier pick",
                               model="claude-haiku-4-5")
            s.add(row)
            s.commit()
            return row.id

    def rows(self) -> list[ClipProposal]:
        with self.Session() as s:
            return list(s.execute(select(ClipProposal).where(ClipProposal.candidate_pk == self.cid)
                                  .order_by(ClipProposal.id)).scalars().all())

    def run_job(self, **kwargs) -> dict:
        """Queue one job with no worker thread, then run it on this thread."""
        with mock.patch.object(jobs, "_ensure_workers", lambda: None):
            queued = jobs.enqueue(self.cid, **kwargs)
        self.assertTrue(queued["started"], queued)
        _priority, _seq, job = jobs._queue.get_nowait()
        jobs._run(job)
        return jobs.status(self.cid)

    def test_an_archive_run_logs_its_picks_in_play_order_with_sidecars(self):
        status = self.run_job(reason="archive")
        self.assertEqual((status["state"], status["outcome"]), ("done", "proposals"), status)
        self.assertEqual(status["error"], "")
        self.ensure_local.assert_called_once_with(str(self.source), "src/nye.mp4")

        rows = self.rows()
        self.assertEqual([r.id for r in rows], status["new"])
        self.assertEqual(len(rows), 3)
        self.assertEqual({r.verdict for r in rows}, {ClipProposal.VERDICT_PENDING})
        self.assertEqual({r.model for r in rows}, {f"engine:gemini:{MODEL}"})
        result = json.loads(Path(status["result_path"]).read_text())
        self.assertEqual({r.policy_version for r in rows}, {result["policy"]})
        stored = [json.loads(r.proposed_segments) for r in rows]
        # The revised picture clip, Nye, then the cold open — stored in PLAY
        # order (the cold open's later moment first), without the lead hints.
        self.assertEqual(stored, [[{"start": 20.27, "end": 30.28}],
                                  [{"start": 1.77, "end": 17.78}],
                                  [{"start": 24.82, "end": 31.18}, {"start": 4.02, "end": 12.18}]])
        self.assertEqual([r.confidence for r in rows], [0.725, 0.8, 0.5])
        self.assertAlmostEqual(status["cost_usd"], round(result["cost_usd"], 4))

        for row, segments in zip(rows, stored):
            meta = sidecars.load(row.id)
            self.assertEqual(meta["segments"], segments)            # what the preview shows
            self.assertEqual(meta["setup"], "gemini")
            self.assertTrue(meta["critique"]["stop_reason"])
            preview = sidecars.preview_path(row.id)
            self.assertIsNotNone(preview)
            self.assertAlmostEqual(media.probe_duration(preview),
                                   engine.seconds_of(segments), delta=0.3)
        self.assertIn("Bill Nye", sidecars.load(rows[1].id)["credits"])

        with self.Session() as s:
            c = s.get(Candidate, self.cid)
            self.assertEqual(c.draft_caption, "Caption d3")
            self.assertTrue(c.multi_clip_potential)     # two stories among the picks
        # Renders nobody was shown are deleted; the chosen ones live on.
        dropped = {d["id"]: Path(d["render"]) for d in result["drafts"]
                   if d.get("render") and d["dropped_reason"]}
        self.assertTrue(dropped)
        for did, path in dropped.items():
            self.assertFalse(path.exists(), did)
        for item in result["top"]:
            self.assertTrue(Path(item["render"]).is_file())
        # The source pin was released.
        inflight = Path(os.environ["CLIP_ENGINE_DIR"]) / "inflight"
        self.assertEqual([p.name for p in inflight.iterdir() if not p.name.startswith(".")], [])

    def test_an_off_topic_video_writes_nothing_and_supersedes_nothing(self):
        earlier = self.add_pending([{"start": 1.0, "end": 15.0}])
        self.fake.on_topic = False
        self.fake.critiques = OFF_TOPIC_CRITIQUES
        status = self.run_job(reason="reroll")
        self.assertEqual((status["state"], status["outcome"]), ("done", "empty_topic"), status)
        self.assertEqual(status["error"], "found nothing on topic")
        self.assertEqual(status["new"], [])
        rows = self.rows()
        self.assertEqual([(r.id, r.verdict) for r in rows],
                         [(earlier, ClipProposal.VERDICT_PENDING)])
        result = json.loads(Path(status["result_path"]).read_text())
        self.assertEqual(result["outcome"], "empty_topic")
        self.assertEqual({d["dropped_reason"] for d in result["drafts"]},
                         {"off topic", "duplicate"})
        for d in result["drafts"]:
            if d.get("render"):
                self.assertFalse(Path(d["render"]).exists(), d["id"])
        llm.suggest_clips.assert_not_called()

    def test_an_engine_error_falls_back_to_the_transcript_suggester(self):
        """Gemini answers the upload with 402 (credits depleted): the perceiver
        fails, the engine raises, and the transcript suggester writes instead
        — saying why on the job."""
        earlier = self.add_pending([{"start": 1.0, "end": 15.0}])
        self.files.depleted = True
        legacy_clip = {"segments": [{"start": 2.0, "end": 17.6}],
                       "story": "Nye: the planet is on fire", "why": "a famous face",
                       "confidence": 0.7, "draft_caption": "Bill Nye: 'The planet's on fire'"}
        llm.suggest_clips.side_effect = None
        llm.suggest_clips.return_value = [legacy_clip]
        llm.pick_opening_frame.side_effect = None
        llm.pick_opening_frame.return_value = {"index": None, "why": "the start is strongest"}

        status = self.run_job(reason="reroll")
        self.assertEqual((status["state"], status["outcome"]), ("done", "legacy"), status)
        self.assertIn("perceiver failed", status["fallback"])
        self.assertIn("add credit at https://ai.studio/projects", status["fallback"])
        self.assertIn("add credit at https://ai.studio/projects", status["message"])
        self.assertTrue(gemini.breaker_open())
        self.assertEqual(self.fake.of("draft"), [])          # the engine stopped at perception

        call = llm.suggest_clips.call_args
        self.assertEqual(call.args[0], "claude-haiku-4-5")
        self.assertEqual(call.args[1], TITLE)
        self.assertEqual(call.args[2], self.transcript)
        self.assertEqual(call.kwargs["words"], self.words)
        # The opening pass looked at real frames sampled from the source.
        frames = llm.pick_opening_frame.call_args.args[1]
        self.assertGreaterEqual(len(frames), 2)
        self.assertTrue(all(isinstance(jpeg, bytes) and jpeg for _t, jpeg in frames))

        rows = self.rows()
        by_id = {r.id: r for r in rows}
        self.assertEqual(by_id[earlier].verdict, ClipProposal.VERDICT_SUPERSEDED)
        new = [by_id[i] for i in status["new"]]
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0].model, "claude-haiku-4-5")
        self.assertEqual(json.loads(new[0].proposed_segments), [{"start": 2.0, "end": 17.6}])
        self.assertIsNone(sidecars.load(new[0].id))          # no engine sidecar for these


if __name__ == "__main__":
    unittest.main()
