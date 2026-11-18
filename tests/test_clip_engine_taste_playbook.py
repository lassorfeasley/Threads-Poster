"""Tests for the account's taste examples (app/clip_engine/taste.py) and the
playbook (app/clip_engine/playbook.py), against an in-memory SQLite schema
with storage, ffmpeg and Claude mocked.

Run with:
  env DATABASE_URL="sqlite://" SCHEDULER_EMBEDDED=false GEMINI_API_KEY= ANTHROPIC_API_KEY= \
    SUPABASE_SERVICE_KEY= CLIP_ENGINE_DIR=$(mktemp -d) CLIPBENCH_DIR=$(mktemp -d) \
    .venv/bin/python -m pytest -q tests/test_clip_engine_taste_playbook.py
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.clip_engine as clip_engine
from app import spend, storage_supabase
from app.clip_engine import claude, media, playbook, prompts, taste
from app.config import Settings
from app.models import (Base, Candidate, Channel, ContentLabel, Cut, MetricSnapshot,
                        ThreadsPost)

BRAND = {"name": "Renewables.org", "topic": "climate change"}
NOW = dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.timezone.utc)
TRANSCRIPT = "[00:10] THE FLOOD CAME\n[00:20] NOBODY LEFT\n[00:40] SPORTS NEXT"


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        patches = [mock.patch.dict(os.environ, {"CLIP_ENGINE_DIR": str(self.dir / "engine")})]
        if hasattr(spend, "_LEDGER"):
            patches.append(mock.patch.object(spend, "_LEDGER", self.dir / "llm_spend.json"))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class DbCase(TempCase):
    """Posts with a 6 h snapshot each, on an in-memory schema."""

    def setUp(self):
        super().setUp()
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.session.close)
        self.channel = Channel(call_sign="KTST", url="https://example.com")
        self.ours = Channel(call_sign="OURS", url="https://example.org", first_party=True)
        self.session.add_all([self.channel, self.ours])
        self.session.flush()
        self.n = 0

    def candidate(self, *, channel=None, category="", title=None):
        self.n += 1
        cand = Candidate(video_id=f"vid{self.n}", channel_pk=(channel or self.channel).id,
                         title=title or f"Source {self.n}", url="u", category=category,
                         transcript_text=TRANSCRIPT)
        self.session.add(cand)
        self.session.flush()
        return cand

    def cut(self, cand, *, segments=((10.0, 30.0),), title=""):
        cut = Cut(candidate_pk=cand.id, clip_title=title,
                  trim_segments=json.dumps([{"start": s, "end": e} for s, e in segments]),
                  format_tags="talking-head", footage_traits="face")
        self.session.add(cut)
        self.session.flush()
        return cut

    def post(self, *, days_ago=3.0, views=100, later_views=None, cand=None, cut=None, title=None,
             repost_of=None, status="published", label=None, local="", key="", caption=None,
             when=None):
        cand = cand or self.candidate()
        cut = cut or self.cut(cand, title=title if title is not None else f"Clip {self.n}")
        published = when or NOW - dt.timedelta(days=days_ago)
        post = ThreadsPost(candidate_pk=cand.id, cut_pk=cut.id, status=status,
                           published_at=published, caption=caption or f"caption {cut.id}",
                           repost_of_post_pk=repost_of, clip_local_path=local,
                           clip_object_path=key)
        self.session.add(post)
        self.session.flush()
        if views is not None:
            self.session.add(MetricSnapshot(post_pk=post.id, views=views,
                                            captured_at=published + dt.timedelta(hours=5)))
        if later_views is not None:
            self.session.add(MetricSnapshot(post_pk=post.id, views=later_views,
                                            captured_at=published + dt.timedelta(days=20)))
        if label:
            self.session.add(ContentLabel(cut_pk=cut.id, label="gripping", value=label))
        self.session.flush()
        return post

    def ladder(self, n=20, *, start_days=2.0):
        """``n`` first airings, one per source, views 100, 200, ... (best is newest)."""
        return [self.post(days_ago=start_days + (n - i) * 0.5, views=100 * (i + 1))
                for i in range(n)]

    def select(self, **kw):
        args = dict(as_of=NOW, hits=3, flops=2, seed="vid-x")
        args.update(kw)
        return taste.select_examples(self.session, **args)


# ---- taste ------------------------------------------------------------------------------------

class LocalPercentilesTest(unittest.TestCase):
    def test_share_of_neighbours_with_fewer_views(self):
        t0 = NOW
        points = [(k, t0 + dt.timedelta(days=k), v) for k, v in
                  [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]]
        far = (6, t0 + dt.timedelta(days=200), 1)
        tied = [(7, t0, 20)]
        out = taste.local_percentiles(points + [far] + tied)
        # 7 ties 2 at 20 views; the far post has under 5 neighbours: whole population.
        self.assertEqual(out, {1: 0.0, 2: 16.7, 3: 50.0, 4: 66.7, 5: 83.3, 6: 0.0, 7: 16.7})

    def test_local_means_local(self):
        early = [(f"e{i}", NOW + dt.timedelta(days=i), 1000 * (i + 1)) for i in range(6)]
        late = [(f"l{i}", NOW + dt.timedelta(days=100 + i), 10 * (i + 1)) for i in range(6)]
        out = taste.local_percentiles(early + late)
        # The best late post is the best of ITS weeks, though it has fewer views than any early one.
        self.assertEqual(out["l5"], out["e5"])
        self.assertEqual(taste.local_percentiles([]), {})


class SelectExamplesTest(DbCase):
    def test_small_pool_gives_nothing(self):
        self.ladder(9)
        self.assertEqual(self.select(), [])

    def test_hits_are_the_top_decile_one_per_source(self):
        posts = self.ladder(20)
        out = self.select(flops=0)
        self.assertEqual([e["views"] for e in out], [2000, 1900, 1800])
        self.assertEqual([e["label"] for e in out], ["hit"] * 3)
        best = out[0]
        self.assertEqual(best["post_pk"], posts[-1].id)
        self.assertEqual(set(best), {"post_pk", "cut_pk", "candidate_pk", "title", "caption",
                                     "views", "percentile", "label", "segments", "seconds",
                                     "transcript", "published_at", "clip_object_path",
                                     "clip_local_path"})
        self.assertEqual(best["percentile"], 95.0)
        self.assertEqual(best["segments"], [{"start": 10.0, "end": 30.0}])
        self.assertEqual(best["seconds"], 20.0)
        self.assertEqual(best["transcript"], "THE FLOOD CAME NOBODY LEFT")
        self.assertEqual(best["title"], f"Clip {posts[-1].candidate_pk}")
        self.assertTrue(best["published_at"].startswith("2026-11-"))

    def test_only_first_airings_of_found_footage_count(self):
        self.ladder(20)
        top = self.post(days_ago=4, views=50)
        self.post(days_ago=3, views=99999, repost_of=top.id,
                  cand=self.session.get(Candidate, top.candidate_pk),
                  cut=self.session.get(Cut, top.cut_pk))                 # a declared re-air
        self.post(days_ago=2.5, views=88888, cand=self.session.get(Candidate, top.candidate_pk),
                  cut=self.session.get(Cut, top.cut_pk))                 # a second airing
        self.post(days_ago=3, views=77777, cand=self.candidate(channel=self.ours))  # our promo
        self.post(days_ago=3, views=66666, cand=self.candidate(category="promos"))
        self.post(days_ago=3, views=55555, status="queued")
        out = self.select(flops=0)
        self.assertEqual([e["views"] for e in out], [2000, 1900, 1800])

    def test_views_are_compared_at_six_hours(self):
        self.ladder(20)
        self.post(days_ago=3, views=150, later_views=1_000_000)
        self.assertEqual([e["views"] for e in self.select(flops=0)], [2000, 1900, 1800])

    def test_as_of_cutoff_and_minimum_age(self):
        self.ladder(20, start_days=6)
        self.post(days_ago=5.5, views=99999)          # after as_of
        self.post(days_ago=5.2, views=99998)          # before as_of but younger than 24 h
        out = self.select(as_of=NOW - dt.timedelta(days=5), flops=0)
        self.assertEqual([e["views"] for e in out], [2000, 1900, 1800])
        self.assertEqual(self.select(as_of=NOW - dt.timedelta(days=20)), [])

    def test_lookback_window(self):
        self.ladder(20)
        self.post(days_ago=200, views=99999)
        self.assertEqual(self.select(flops=0)[0]["views"], 2000)

    def test_excluded_candidate_and_one_hit_per_source(self):
        posts = self.ladder(20)
        best_source = self.session.get(Candidate, posts[-1].candidate_pk)
        self.post(days_ago=2.2, views=1950, cand=best_source)   # same source, another cut
        out = self.select(flops=0)
        self.assertEqual([e["views"] for e in out], [2000, 1900, 1800])
        out = self.select(flops=0, exclude_candidate_pk=best_source.id)
        self.assertEqual([e["views"] for e in out], [1900, 1800, 1700])

    def test_seeded_rotation_through_the_decile(self):
        self.ladder(60)
        picks = {seed: [e["post_pk"] for e in self.select(flops=0, seed=seed)]
                 for seed in ("a", "b", "c", "d", "e")}
        self.assertEqual(picks["a"], [e["post_pk"] for e in self.select(flops=0, seed="a")])
        self.assertGreater(len({tuple(sorted(p)) for p in picks.values()}), 1)
        decile = {e["post_pk"] for e in taste.scored_first_airings(
            self.session, as_of=NOW, lookback_days=180, min_age_hours=24)[:6]}
        for chosen in picks.values():
            self.assertEqual(len(chosen), 3)
            self.assertTrue(set(chosen) <= decile)

    def test_flops_are_only_labelled_clips(self):
        posts = self.ladder(20)
        self.assertEqual([e["label"] for e in self.select()], ["hit"] * 3)   # nothing labelled
        low = self.post(days_ago=3, views=150, label="no")
        lowest = self.post(days_ago=3.1, views=120, label="no")
        same_source = self.post(days_ago=3.2, views=130, label="no",
                                cand=self.session.get(Candidate, lowest.candidate_pk))
        self.post(days_ago=3.3, views=110, label="yes")
        hit_source = self.session.get(Candidate, posts[-1].candidate_pk)
        self.post(days_ago=3.4, views=101, label="no", cand=hit_source)
        out = self.select(flops=3)
        flops = [e for e in out if e["label"] == "flop"]
        self.assertEqual([e["post_pk"] for e in flops], [lowest.id, low.id])
        self.assertNotIn(same_source.id, [e["post_pk"] for e in out])


class PerformanceLineTest(unittest.TestCase):
    CASES = [
        ({"label": "hit", "percentile": 98.0, "views": 21895},
         "top 2% of this account's posts from the same weeks (21,895 views at 6 h)"),
        ({"label": "hit", "percentile": 99.9, "views": 5}, "top 1% of this account's posts from "
                                                           "the same weeks (5 views at 6 h)"),
        ({"label": "hit"}, "one of this account's best posts from the same weeks"),
        ({"label": "flop", "percentile": 3.0, "views": 40}, "you labelled this one not gripping"),
        ({}, "one of this account's best posts from the same weeks"),
    ]

    def test_table(self):
        for example, expected in self.CASES:
            with self.subTest(example=example):
                self.assertEqual(taste.performance_line(example), expected)


class OutroTrimTest(unittest.TestCase):
    SEGS = [{"start": 0, "end": 10}, {"start": 50, "end": 70}]   # 30 s of footage

    def test_table(self):
        cases = [(32.8, 2.8), (32.0, 2.0), (34.5, 4.5), (31.9, 0.0), (34.6, 0.0), (30.0, 0.0),
                 (29.0, 0.0), (None, 0.0), ("x", 0.0)]
        for duration, expected in cases:
            with self.subTest(duration=duration):
                self.assertEqual(taste.outro_trim(duration, self.SEGS), expected)
        self.assertEqual(taste.outro_trim(3.0, []), 3.0)


class ExampleFileTest(TempCase):
    def setUp(self):
        super().setUp()
        self.settings = Settings(raw={"storage": {"trimmed_clip_bucket": "clips-bucket"}})
        self.trims = []
        self.downloads = []

        def fake_trim(path, seconds, dest, *, height=360):
            self.trims.append((Path(path), seconds, height))
            Path(dest).write_bytes(b"mp4")
            return Path(dest)

        def fake_download(bucket, key, dest, expected_size=None):
            self.downloads.append((bucket, key))
            Path(dest).write_bytes(b"published copy")
            return Path(dest)

        for target, attr, fake in [(media, "trim_tail", fake_trim),
                                   (storage_supabase, "download_to_file", fake_download)]:
            p = mock.patch.object(target, attr, side_effect=fake)
            p.start()
            self.addCleanup(p.stop)
        self.probe = mock.patch.object(media, "probe_duration", return_value=32.8).start()
        self.addCleanup(mock.patch.stopall)

    def example(self, **kw):
        out = {"post_pk": 7, "segments": [{"start": 0, "end": 30}], "clip_local_path": "",
               "clip_object_path": "2026/11/clip_x.mp4"}
        out.update(kw)
        return out

    def taste_dir(self):
        return self.dir / "engine" / "taste"

    def test_the_local_export_needs_no_trim_and_is_cached(self):
        local = self.dir / "export.mp4"
        local.write_bytes(b"x")
        path = taste.example_file(self.example(clip_local_path=str(local)), self.settings)
        self.assertEqual(path, self.taste_dir() / "7-360.mp4")
        self.assertTrue(path.is_file())
        self.assertEqual(self.trims, [(local, 0.0, 360)])
        self.assertEqual(self.downloads, [])
        self.assertEqual(taste.example_file(self.example(clip_local_path=str(local)),
                                            self.settings), path)
        self.assertEqual(len(self.trims), 1)
        self.assertEqual(sorted(p.name for p in self.taste_dir().iterdir()), ["7-360.mp4"])

    def test_the_published_copy_loses_its_outro(self):
        path = taste.example_file(self.example(clip_local_path="/gone/export.mp4"), self.settings,
                                  height=240)
        self.assertEqual(path, self.taste_dir() / "7-240.mp4")
        self.assertEqual(self.downloads, [("clips-bucket", "2026/11/clip_x.mp4")])
        (src, seconds, height), = self.trims
        self.assertEqual((seconds, height), (2.8, 240))
        self.assertFalse(src.exists())                       # the download is cleaned up
        self.assertEqual(sorted(p.name for p in self.taste_dir().iterdir()), ["7-240.mp4"])

    def test_no_outro_means_no_trim(self):
        self.probe.return_value = 30.2
        taste.example_file(self.example(), self.settings)
        self.assertEqual(self.trims[-1][1], 0.0)

    def test_default_bucket(self):
        taste.example_file(self.example(), Settings(raw={"storage": {"trimmed_clip_bucket": ""}}))
        self.assertEqual(self.downloads[-1][0], "trimmed-clips")

    def test_failures_are_none(self):
        self.assertIsNone(taste.example_file(self.example(clip_object_path=""), self.settings))
        with mock.patch.object(storage_supabase, "download_to_file",
                               side_effect=RuntimeError("HTTP 404")):
            self.assertIsNone(taste.example_file(self.example(), self.settings))
        with mock.patch.object(media, "trim_tail", side_effect=media.MediaError("ffmpeg")):
            self.assertIsNone(taste.example_file(self.example(), self.settings))
        self.assertEqual(list(self.taste_dir().iterdir()), [])
        self.assertIsNone(taste.example_file({}, self.settings))


# ---- playbook ---------------------------------------------------------------------------------

class DefaultPlaybookTest(unittest.TestCase):
    def test_the_default_is_this_accounts_rubric(self):
        pb = playbook.DEFAULT_PLAYBOOK
        self.assertEqual(playbook.keys(pb), ["voice", "scene", "revelation", "clash", "pop",
                                             "witness"])
        self.assertEqual([len(a["examples"]) for a in pb["archetypes"]], [4, 3, 3, 2, 3, 2])
        voice, scene, revelation, clash, pop, witness = pb["archetypes"]
        self.assertEqual(voice["open"], "On the person, on or one beat before the line. Their "
                                        "face is the hook, so don't bury it under b-roll.")
        self.assertEqual(voice["examples"][0],
                         "Bill Nye: 'The Planet's on Fire' — Safety Glasses Off")
        self.assertEqual(pop["open"], "On the peak of the meltdown or rant, or on the last line of "
                                      "setup so the first laugh lands within about 3 seconds. "
                                      "Never open on a punchline whose setup plays after it.")
        self.assertEqual(clash["what"], "Confrontation — people in open conflict over climate or "
                                        "energy.")
        self.assertEqual(witness["avoid"], "The reporter paraphrasing what they said.")
        self.assertEqual(pb["flops"][0], "Spectacle with no person or stakes - fire, flood, crowd "
                                         "or aerial footage under narration (below this "
                                         "account's median).")
        self.assertEqual(len(pb["flops"]), 6)
        self.assertEqual(pb["notes"], "Clips of 15-45 seconds perform about the same here; what "
                                      "matters is the first two seconds and who is on screen.")
        self.assertEqual(playbook.validate(pb)["archetypes"], pb["archetypes"])


SMALL = {"archetypes": [{"key": "voice", "name": "The Voice", "what": "W", "open": "O",
                         "avoid": "A", "examples": ["x", "y"]},
                        {"key": "scene", "name": "The Scene", "what": "W2", "open": "O2",
                         "avoid": "", "examples": []}],
         "flops": ["f1", "f2"], "notes": "N"}


def valid(n=4, **kw):
    pb = {"archetypes": [{"key": f"k{i}", "name": f"Name {i}", "what": "w", "open": "o",
                          "avoid": "a", "examples": [f"t{i}"]} for i in range(n)],
          "flops": ["f"], "notes": "n"}
    pb.update(kw)
    return pb


class PlaybookFunctionsTest(TempCase):
    def test_render(self):
        self.assertEqual(playbook.render(SMALL), (
            "ARCHETYPES (from this account's own results):\n"
            "- voice — The Voice: W\n  Open: O\n  Avoid: A\n  Hits: \"x\", \"y\"\n"
            "- scene — The Scene: W2\n  Open: O2\n\n"
            "WHAT FLOPS HERE:\n- f1\n- f2\n\nN"))
        self.assertEqual(playbook.render({}), "ARCHETYPES (from this account's own results):")

    def test_strip_examples_and_fingerprint(self):
        original = copy.deepcopy(playbook.DEFAULT_PLAYBOOK)
        stripped = playbook.strip_examples(playbook.DEFAULT_PLAYBOOK)
        self.assertEqual(playbook.DEFAULT_PLAYBOOK, original)          # untouched
        self.assertTrue(all("examples" not in a for a in stripped["archetypes"]))
        self.assertNotIn("Hits:", playbook.render(stripped))
        fp = playbook.fingerprint(playbook.DEFAULT_PLAYBOOK)
        self.assertRegex(fp, r"^[0-9a-f]{12}$")
        self.assertEqual(fp, playbook.fingerprint(copy.deepcopy(original)))
        self.assertNotEqual(fp, playbook.fingerprint(stripped))
        self.assertEqual(playbook.fingerprint({**original, "as_of": "2026-01-01"}), fp)

    def test_open_rule_and_keys(self):
        pb = playbook.DEFAULT_PLAYBOOK
        self.assertEqual(playbook.open_rule(pb, "voice"),
                         "The Voice: On the person, on or one beat before the line. Their face "
                         "is the hook, so don't bury it under b-roll.")
        self.assertEqual(playbook.open_rule(pb, "VOICE "), playbook.open_rule(pb, "voice"))
        self.assertEqual(playbook.open_rule(pb, "other"), "")
        self.assertEqual(playbook.open_rule(SMALL | {"archetypes": [
            {"key": "x", "name": "X", "what": "w", "open": ""}]}, "x"), "")
        self.assertEqual(playbook.keys(SMALL), ["voice", "scene"])

    def test_validate(self):
        good = playbook.validate(valid(archetypes=[
            {"key": " The Voice ", "name": "V", "what": "w", "open": "o", "avoid": "a",
             "examples": ["a", "", "b"], "evidence": "12 posts"}] + valid(3)["archetypes"],
            as_of="2026-11-01", junk="dropped"))
        self.assertEqual(good["archetypes"][0]["key"], "the_voice")
        self.assertEqual(good["archetypes"][0]["examples"], ["a", "b"])
        self.assertEqual(good["archetypes"][0]["evidence"], "12 posts")
        self.assertEqual(good["as_of"], "2026-11-01")
        self.assertNotIn("junk", good)
        bad = [
            "not a mapping", valid(3), valid(8),
            valid(archetypes=valid(3)["archetypes"] + [{"key": "K0", "name": "n", "what": "w",
                                                        "open": "o"}]),
            valid(archetypes=valid(3)["archetypes"] + [{"key": "x", "name": "n", "what": "w",
                                                        "open": ""}]),
            valid(archetypes=valid(3)["archetypes"] + ["not a mapping"]),
            valid(archetypes=valid(3)["archetypes"] + [{"key": "", "name": "n", "what": "w",
                                                        "open": "o"}]),
        ]
        for data in bad:
            with self.subTest(data=str(data)[:60]):
                with self.assertRaises(playbook.PlaybookError):
                    playbook.validate(data)

    def test_load_falls_back_to_the_default(self):
        path = self.dir / "clip_playbook.yaml"
        with mock.patch.object(playbook, "path", return_value=path):
            loaded = playbook.load()
            self.assertEqual(loaded, playbook.DEFAULT_PLAYBOOK)
            loaded["archetypes"].clear()                    # a copy: the default is untouched
            self.assertEqual(len(playbook.DEFAULT_PLAYBOOK["archetypes"]), 6)
            for text in ("archetypes: [", yaml.safe_dump(valid(3)), "- just a list\n"):
                with self.subTest(text=text[:20]):
                    path.write_text(text)
                    with self.assertLogs("clip_engine.playbook", level="WARNING"):
                        self.assertEqual(playbook.load(), playbook.DEFAULT_PLAYBOOK)
            path.write_text(yaml.safe_dump(valid(5)))
            self.assertEqual(playbook.keys(playbook.load()), ["k0", "k1", "k2", "k3", "k4"])

    def test_save_writes_an_editable_file_that_loads_back(self):
        path = self.dir / "config" / "clip_playbook.yaml"
        data = valid(4, as_of="2026-11-01T00:00:00+00:00", generated_at="2026-11-18T09:00:00Z",
                     posts=40, excluded=2, excluded_candidate_pks=[3, 9], model="claude-opus-5-5")
        with mock.patch.object(playbook, "path", return_value=path):
            self.assertEqual(playbook.save(data), path)
            text = path.read_text()
            self.assertTrue(text.startswith("# Clip playbook — generated 2026-11-18 from posts up "
                                            "to 2026-11-01 (40 posts).\n"))
            self.assertIn("Edit it by hand", text)
            self.assertEqual(playbook.load(), playbook.validate(data))
            with self.assertRaises(playbook.PlaybookError):
                playbook.save(valid(2))


ANSWER = {"archetypes": [{"key": k, "name": k.title(), "what": "w", "open": "o", "avoid": "a",
                          "examples": ["t"], "evidence": "e"} for k in ("voice", "scene", "pop",
                                                                         "witness")],
          "flops": ["aerials"], "notes": "n"}


class GenerateTest(DbCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.answer = ANSWER
        self.settings = Settings(raw={"clip_engine": {"claude": {"editor_model": ""}}})
        for target, attr, fake in [
            (claude, "json_call", self.fake_call),
            (claude, "text_block", lambda t: {"type": "text", "text": t}),
            (clip_engine, "brand", lambda: BRAND),
        ]:
            p = mock.patch.object(target, attr, side_effect=fake)
            p.start()
            self.addCleanup(p.stop)

    def fake_call(self, model, **kw):
        self.calls.append({"model": model, "in_transaction": self.session.in_transaction(), **kw})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer, {"model": model, "input_tokens": 9000, "output_tokens": 900,
                             "cost_usd": 0.05}

    def generate(self, **kw):
        # Real rows are committed; generate() rolls its own reads back before the model call,
        # which would otherwise discard fixtures that were only flushed.
        self.session.commit()
        args = dict(as_of=NOW, meter=None)
        args.update(kw)
        return playbook.generate(self.session, self.settings, **args)

    def payload(self):
        return json.loads(self.calls[-1]["content"][0]["text"])

    def test_evidence_and_bookkeeping(self):
        posts = self.ladder(30)
        excluded = self.session.get(Candidate, posts[-1].candidate_pk)   # the best post's source
        self.post(days_ago=0.5, views=99999)                             # too young to judge
        self.post(days_ago=400, views=99999)                             # older than a year
        meter = mock.Mock(bucket="engine")
        out = self.generate(exclude_candidate_pks=[excluded.id, None], meter=meter)
        (call,) = self.calls
        self.assertFalse(call["in_transaction"])          # no transaction held across the call
        self.assertEqual(call["model"], "claude-opus-5-5")
        self.assertEqual(call["effort"], "high")
        self.assertIs(call["schema"], playbook.PLAYBOOK_SCHEMA)
        self.assertEqual(call["system"], prompts.playbook_system(BRAND))
        payload = self.payload()
        best, worst = payload["best_posts"], payload["worst_posts"]
        self.assertEqual((len(best), len(worst)), (18, 11))           # 29 posts, split 40:25
        # The excluded source can't be shown but still counts as a neighbour: 28 of the 30
        # posts from those weeks did worse than 2,900 views.
        self.assertEqual(best[0]["performance"],
                         "top 7% of posts from the same weeks (2,900 views at 6 h)")
        self.assertEqual(worst[-1]["seconds"], 20.0)
        self.assertEqual(best[0]["words_heard"], "THE FLOOD CAME NOBODY LEFT")
        self.assertEqual((best[0]["format_tags"], best[0]["footage_traits"]),
                         ("talking-head", "face"))
        titles = {p["title"] for p in best + worst}
        self.assertNotIn(f"Clip {excluded.id}", titles)
        self.assertEqual(out["excluded"], 1)
        self.assertEqual(out["excluded_candidate_pks"], [excluded.id])
        self.assertEqual(out["posts"], 29)
        self.assertEqual(out["as_of"], NOW.isoformat())
        self.assertTrue(out["generated_at"].endswith("Z"))
        self.assertEqual(playbook.keys(out), ["voice", "scene", "pop", "witness"])
        meter.add.assert_called_once_with("playbook", {"model": "claude-opus-5-5",
                                                       "input_tokens": 9000,
                                                       "output_tokens": 900, "cost_usd": 0.05})

    def test_large_pools_send_the_best_forty_and_worst_twenty_five(self):
        self.ladder(80)
        self.generate(model="claude-sonnet-5-5")
        payload = self.payload()
        self.assertEqual((len(payload["best_posts"]), len(payload["worst_posts"])), (40, 25))
        self.assertEqual(payload["worst_posts"][-1]["percentile"], 0.0)
        self.assertEqual(self.calls[-1]["model"], "claude-sonnet-5-5")

    def test_as_of_limits_the_evidence(self):
        self.ladder(20, start_days=40)
        self.ladder(20, start_days=1.5)
        self.generate(as_of=NOW - dt.timedelta(days=30))
        payload = self.payload()
        self.assertEqual(len(payload["best_posts"]) + len(payload["worst_posts"]), 20)

    def test_too_little_evidence(self):
        self.ladder(5)
        with self.assertRaises(playbook.PlaybookError):
            self.generate()
        self.assertEqual(self.calls, [])

    def test_an_invalid_answer_is_rejected(self):
        self.ladder(20)
        self.answer = {"archetypes": ANSWER["archetypes"][:2], "flops": [], "notes": ""}
        with self.assertRaises(playbook.PlaybookError):
            self.generate()

    def test_provider_errors_propagate_with_their_cost(self):
        self.ladder(20)
        exc = claude.ClaudeTruncated("max_tokens")
        exc.usage = {"model": "m", "input_tokens": 1, "output_tokens": 32000, "cost_usd": 0.64}
        self.answer = exc
        meter = mock.Mock(bucket="engine")
        with self.assertRaises(claude.ClaudeTruncated):
            self.generate(meter=meter)
        meter.add.assert_called_once_with("playbook", exc.usage)


if __name__ == "__main__":
    unittest.main()
