"""What works for this account, as archetypes the editor and the judges share.

One global rule ("open on the most striking image") gets this account's best
clips wrong: when a recognizable person delivering a charged line is the
hook, opening on b-roll hides it. So the rubric is a small set of archetypes
drawn from the account's own results, each with its own opening rule, plus
the patterns that flop. The editor names an archetype for every draft and
follows its opening rule; the ranker and the revision check read the same
text, so every stage pulls the same way.

The default below was written from this account's real posts. ``generate``
re-derives one from the posts' measured performance (views at a fixed age,
percentile among posts from the same weeks — never raw views) and ``save``
writes it to config/clip_playbook.yaml, where the operator can edit it; a
missing or invalid file means the default.

Benchmark hygiene: example titles are this account's own posts, so a replay
benchmark strips them (``strip_examples``), and ``generate`` takes ``as_of``
and the candidates to leave out, and records both, so a generated playbook
never describes the very clips it is being tested on.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from . import claude, model_for, prompts, util

log = logging.getLogger("clip_engine.playbook")

MIN_ARCHETYPES, MAX_ARCHETYPES = 4, 7
MIN_POSTS = 10                 # fewer measured posts than this: nothing to learn from
LOOKBACK_DAYS = 365
MIN_AGE_HOURS = 24
TOP_POSTS, BOTTOM_POSTS = 40, 25
_META_KEYS = ("as_of", "generated_at", "excluded", "excluded_candidate_pks", "model", "posts")

_DEFAULT_YAML = """\
archetypes:
  - key: voice
    name: The Voice
    what: A recognizable or compelling person delivering a charged line.
    open: On the person, on or one beat before the line. Their face is the hook, so don't bury it under b-roll.
    avoid: Unknown officials reading statements; long setups before the line.
    examples: ["Bill Nye: 'The Planet's on Fire' — Safety Glasses Off", "1989: Thatcher Tells UN No One Can Opt Out of Climate Change", "At 82, Jane Fonda Arrested Again on Capitol Hill for Climate", "Climate Alarm: 'We're All Screwed,' Meteorologist Warns"]
  - key: scene
    name: The Scene
    what: Something is happening on camera, with stakes.
    open: On the action itself with its natural sound; the first frame shows people or things in motion.
    avoid: A narrator explaining over generic b-roll before anything happens.
    examples: ["Sedated Polar Bear Mom Checked and Weighed: 385 Pounds, Healthy", "Chainsaw sounds lead rangers to illegal loggers in a protected forest", "Indonesia's Toxic Haze Crisis: Protesters Stage Mock Funeral"]
  - key: revelation
    name: The Revelation
    what: A jaw-dropping fact or number that reframes the issue.
    open: On the strongest image while the number lands, or on the number itself when the speaker is compelling. Never on the setup question.
    avoid: Charts, slideshows, a string of numbers with no picture.
    examples: ["Only 18% Remains: Illegal Roads, Burned Forests Expose Broken Pledge", "Ice Cores Reveal CO2 Is Already Off the Charts", "Deforestation Emits as Much CO2 as All Cars, Trucks and Planes Combined"]
  - key: clash
    name: The Clash
    what: Confrontation — people in open conflict over climate or energy.
    open: At the peak of the confrontation, not the setup.
    avoid: The anchor's summary of the fight instead of the fight.
    examples: ["Bastrop town hall erupts over data centers", "Texas Gov Accused of Data Center Flip-Flop as Grid Fight Escalates"]
  - key: pop
    name: The Pop Moment
    what: Fiction, comedy or culture touching climate, including allegories that never say the word.
    open: On the peak of the meltdown or rant, or on the last line of setup so the first laugh lands within about 3 seconds. Never open on a punchline whose setup plays after it.
    avoid: Explaining the joke.
    examples: ["Kate Dibiasky's on-air meltdown: the moment denial meets fury", "Colbert's Climate Change Pickup Lines", "Comedian: I'll Burn My Fortune on Private Jets Just to Spite Greta"]
  - key: witness
    name: The Witness
    what: An ordinary person's first-hand stakes.
    open: On their most emotional line or their face as they say it.
    avoid: The reporter paraphrasing what they said.
    examples: ["The father who helped shut down Italy's biggest steel plant", "How people's memories are helping document climate change"]
flops:
  - Spectacle with no person or stakes - fire, flood, crowd or aerial footage under narration (below this account's median).
  - A narrator explaining over generic b-roll, with no person, action or number to hold onto.
  - Anchor intros, studio tosses, and station graphics at the start.
  - Press-podium officials reading statements.
  - Slideshows, charts and maps.
  - Clips that only make sense with context the viewer doesn't have.
notes: "Clips of 15-45 seconds perform about the same here; what matters is the first two seconds and who is on screen."
"""

DEFAULT_PLAYBOOK: dict = yaml.safe_load(_DEFAULT_YAML)

PLAYBOOK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "archetypes": {
            "type": "array",
            "minItems": MIN_ARCHETYPES,
            "maxItems": MAX_ARCHETYPES,
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "one lowercase word"},
                    "name": {"type": "string"},
                    "what": {"type": "string"},
                    "open": {"type": "string",
                             "description": "exactly what the first two seconds show and say"},
                    "avoid": {"type": "string"},
                    "examples": {"type": "array", "items": {"type": "string"},
                                 "description": "2-4 post titles from the data, best first"},
                    "evidence": {"type": "string",
                                 "description": "what in the data supports this; say when it is thin"},
                },
                "required": ["key", "name", "what", "open", "avoid", "examples", "evidence"],
            },
        },
        "flops": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["archetypes", "flops", "notes"],
}


class PlaybookError(ValueError):
    """A playbook (from a model or a file) that doesn't hold together."""


def _line(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip() if len(text) > limit else text


def validate(data) -> dict:
    """A clean copy of ``data``: 4-7 archetypes with unique one-word
    lowercase keys and a name, what and opening rule each; flops and notes.
    Raises ``PlaybookError`` otherwise. Generation metadata is carried over."""
    if not isinstance(data, dict):
        raise PlaybookError("a playbook must be a mapping")
    archetypes, seen = [], set()
    for item in data.get("archetypes") or []:
        if not isinstance(item, dict):
            raise PlaybookError("every archetype must be a mapping")
        key = re.sub(r"[^a-z0-9_]+", "_", str(item.get("key") or "").strip().lower()).strip("_")
        if not key:
            raise PlaybookError("an archetype has no key")
        if key in seen:
            raise PlaybookError(f"archetype key {key!r} appears twice")
        seen.add(key)
        entry = {"key": key, "name": _line(item.get("name"), 80), "what": _line(item.get("what")),
                 "open": _line(item.get("open")), "avoid": _line(item.get("avoid"))}
        if not (entry["name"] and entry["what"] and entry["open"]):
            raise PlaybookError(f"archetype {key!r} needs a name, what and open")
        examples = item.get("examples") or []
        entry["examples"] = [_line(e, 200) for e in examples if _line(e, 200)][:6] \
            if isinstance(examples, list) else []
        if item.get("evidence"):
            entry["evidence"] = _line(item.get("evidence"))
        archetypes.append(entry)
    if not MIN_ARCHETYPES <= len(archetypes) <= MAX_ARCHETYPES:
        raise PlaybookError(f"a playbook needs {MIN_ARCHETYPES}-{MAX_ARCHETYPES} archetypes, "
                            f"not {len(archetypes)}")
    flops = data.get("flops") or []
    out: dict[str, Any] = {
        "archetypes": archetypes,
        "flops": [_line(f, 300) for f in flops if _line(f, 300)] if isinstance(flops, list) else [],
        "notes": _line(data.get("notes"), 1000),
    }
    for key in _META_KEYS:
        if key in data and data[key] is not None:
            out[key] = data[key]
    return out


def path() -> Path:
    """config/clip_playbook.yaml in the active workspace."""
    from ..config import CONFIG_DIR

    return CONFIG_DIR / "clip_playbook.yaml"


def load() -> dict:
    """The operator's playbook file when it is valid, else the default (a copy
    either way, so callers may annotate it)."""
    file = path()
    if file.is_file():
        try:
            return validate(yaml.safe_load(file.read_text()) or {})
        except Exception as exc:  # noqa: BLE001 - a broken file must not stop a run
            log.warning("%s is not a usable playbook (%s); using the built-in one", file, exc)
    return copy.deepcopy(DEFAULT_PLAYBOOK)


def render(playbook) -> str:
    """The playbook as prompt text (the form every stage reads)."""
    playbook = playbook or {}
    lines = ["ARCHETYPES (from this account's own results):"]
    for item in playbook.get("archetypes") or []:
        lines.append(f"- {item.get('key', '')} — {item.get('name', '')}: {item.get('what', '')}")
        if item.get("open"):
            lines.append(f"  Open: {item['open']}")
        if item.get("avoid"):
            lines.append(f"  Avoid: {item['avoid']}")
        examples = [str(e).strip() for e in item.get("examples") or [] if str(e).strip()]
        if examples:
            lines.append("  Hits: " + ", ".join(f'"{e}"' for e in examples))
    flops = [str(f).strip() for f in playbook.get("flops") or [] if str(f).strip()]
    if flops:
        lines += ["", "WHAT FLOPS HERE:"] + [f"- {f}" for f in flops]
    notes = str(playbook.get("notes") or "").strip()
    if notes:
        lines += ["", notes]
    return "\n".join(lines)


def keys(playbook) -> list[str]:
    return [str(a.get("key", "")).strip().lower() for a in (playbook or {}).get("archetypes") or []
            if str(a.get("key", "")).strip()]


def open_rule(playbook, key) -> str:
    """"<name>: <open>" for archetype ``key``; "" when unknown or blank."""
    wanted = str(key or "").strip().lower()
    for item in (playbook or {}).get("archetypes") or []:
        if str(item.get("key", "")).strip().lower() == wanted and item.get("open"):
            return f"{item.get('name') or wanted}: {item['open']}"
    return ""


def strip_examples(playbook) -> dict:
    """A copy without any archetype's example titles (they are this account's
    own posts — a benchmark replaying those videos must not see them)."""
    out = copy.deepcopy(playbook or {})
    for item in out.get("archetypes") or []:
        item.pop("examples", None)
    return out


def fingerprint(playbook) -> str:
    """sha1 of the rendered playbook, 12 hex: what the models actually read."""
    return hashlib.sha1(render(playbook).encode("utf-8")).hexdigest()[:12]


# ---- generation ----------------------------------------------------------------------------

def _aware(when: dt.datetime | None) -> dt.datetime | None:
    if when is not None and when.tzinfo is None:
        return when.replace(tzinfo=dt.timezone.utc)
    return when


def _split(posts: list[dict]) -> tuple[list[dict], list[dict]]:
    """The best 40 and the worst 25 (posts arrive best first); a small pool is
    split in the same proportion so there are always some of each."""
    n = len(posts)
    if n >= TOP_POSTS + BOTTOM_POSTS:
        return posts[:TOP_POSTS], posts[-BOTTOM_POSTS:]
    top = max(1, round(n * TOP_POSTS / (TOP_POSTS + BOTTOM_POSTS)))
    return posts[:top], posts[top:]


def _evidence(post: dict, words: str) -> dict:
    top = max(1, int(round(100 - float(post["percentile"]))))
    return {
        "title": post["title"],
        "caption": (post.get("caption") or "")[:500],
        "words_heard": words,
        "format_tags": post.get("format_tags") or "",
        "footage_traits": post.get("footage_traits") or "",
        "seconds": post["seconds"],
        "percentile": post["percentile"],
        "performance": f"top {top}% of posts from the same weeks "
                       f"({post['views']:,} views at 6 h)",
    }


def generate(session, settings, *, model=None, as_of=None, exclude_candidate_pks=(), meter=None) -> dict:
    """A playbook re-derived from this account's posts up to ``as_of``.

    Evidence: first airings published in the 365 days before ``as_of`` (and at
    least 24 h before it), minus ``exclude_candidate_pks``, ranked by views at
    6 h as a percentile among posts from the same ±30 days; the best 40 and
    worst 25 go to Claude with their titles, captions, words heard, footage
    tags and length. The session is only read, and is rolled back before the
    model call so no transaction stays open across it. Raises
    ``PlaybookError`` when there is too little evidence or the answer doesn't
    validate; provider errors propagate. Doesn't save (see ``save``)."""
    from . import brand as load_brand
    from .taste import scored_first_airings, words_heard

    as_of = _aware(as_of) or dt.datetime.now(dt.timezone.utc)
    excluded = sorted({int(pk) for pk in exclude_candidate_pks or () if pk is not None})
    try:
        posts = scored_first_airings(session, as_of=as_of, lookback_days=LOOKBACK_DAYS,
                                     min_age_hours=MIN_AGE_HOURS, exclude_candidate_pks=excluded)
        best, worst = _split(posts)
        words = words_heard(session, best + worst)
    finally:
        session.rollback()
    if len(posts) < MIN_POSTS:
        raise PlaybookError(f"only {len(posts)} measured posts before {as_of.date()}; "
                            f"need at least {MIN_POSTS}")
    payload = {
        "best_posts": [_evidence(p, words.get(p["post_pk"], "")) for p in best],
        "worst_posts": [_evidence(p, words.get(p["post_pk"], "")) for p in worst],
    }
    model_id = model or model_for(settings, "claude", "editor")
    try:
        data, usage = claude.json_call(
            model_id, system=prompts.playbook_system(load_brand()),
            content=[claude.text_block(json.dumps(payload, ensure_ascii=False))],
            schema=PLAYBOOK_SCHEMA, effort="high", max_tokens=32000, label="playbook",
            # No meter: the process default bucket, else "engine" (json_call).
            bucket=getattr(meter, "bucket", None) or None)
    except Exception as exc:
        if meter is not None and getattr(exc, "usage", None):
            meter.add("playbook", exc.usage)
        raise
    if meter is not None and usage:
        meter.add("playbook", usage)
    playbook = validate(data)
    playbook.update({
        "as_of": as_of.isoformat(),
        "generated_at": util.utcnow_iso(),
        "excluded": len(excluded),
        "excluded_candidate_pks": excluded,
        "model": model_id,
        "posts": len(best) + len(worst),
    })
    log.info("playbook: %d archetypes from %d posts (as of %s, %d candidates excluded)",
             len(playbook["archetypes"]), playbook["posts"], as_of.date(), len(excluded))
    return playbook


def save(playbook) -> Path:
    """Write ``playbook`` to config/clip_playbook.yaml (atomically), with a
    header saying where it came from and that it may be edited by hand."""
    data = validate(playbook)
    stamp = str(data.get("generated_at") or util.utcnow_iso())[:10]
    source = ""
    if data.get("as_of"):
        source = f" from posts up to {str(data['as_of'])[:10]}"
        if data.get("posts"):
            source += f" ({data['posts']} posts)"
    header = (
        f"# Clip playbook — generated {stamp}{source}.\n"
        "# The clip engine's editor, critic and ranker all read this file. Edit it by hand\n"
        "# as you like; delete it to go back to the built-in default.\n")
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    target = path()
    util.atomic_write_text(target, header + body)
    log.info("playbook saved to %s", target)
    return target
