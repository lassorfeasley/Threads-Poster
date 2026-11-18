"""The operator's notes on clip suggestions, and the brand rules they add up to.

The engine's first live run picked two non-English interviews and a reporter
monologue on a video whose story was the footage. The operator could dismiss
those cards, but a dismissal reason ("not clipworthy") says nothing the next
pass can act on, and nothing he said reached the run after it. This module is
where what he says goes:

* **A note per complaint** (``add``): a card he turned down with "What's
  wrong?" (kind ``card``), or advice he typed next to a Suggest button (kind
  ``steer``). Tags from a small fixed vocabulary, plus a line of his own words.
  ``advice_from`` turns one into the plain-English advice the engine's prompts
  carry for the re-suggest it triggers.
* **An append-only JSONL file** (``engine_dir()/notes.jsonl``), not a table:
  no schema change, and a history nobody edits. One line per note, appended
  with a single ``O_APPEND`` write under a thread lock and an ``flock`` (the
  dashboard and a CLI process share the directory), so concurrent writers
  never interleave. Reads skip a torn or corrupt line instead of raising — a
  broken note must not break a card or a page.
* **Rules from recurring notes** (``propose_rules``): the same complaint on
  three videos is a rule, not a one-off. After a note is saved, a daemon
  thread asks Claude — once at least three notes have arrived since the last
  time it asked — to cluster the last 90 days of notes into rules the brand
  rules don't already cover. A theme needs three notes across two videos;
  that is re-checked here, not left to the model. Proposals wait in
  ``engine_dir()/rule_proposals.json`` for the operator to add or dismiss on
  the Brand & audience page; nothing becomes a rule without him.
* **One queue, one rule store.** The handbook (rules learned from past
  edits) queues its proposals here too (``add_proposals(source="handbook")``),
  and an added proposal of either kind becomes an ordinary brand rule
  (``applies_to: clips``) that only remembers its ``source`` and evidence.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable

from . import engine_dir, util

log = logging.getLogger("clip_engine.notes")

KINDS = ("card", "steer")

# slug -> (chip label, the advice it stands for). Order is the chips' order.
TAGS: dict[str, tuple[str, str]] = {
    "wrong_language": (
        "Wrong language",
        "Only use speech the audience can understand: English, or another language carried by "
        "an English voiceover. Burned-in subtitles are not enough."),
    "no_premise": (
        "No premise",
        "Open on the premise: within the first few seconds a stranger must know what is "
        "happening and where — a picture plus a line that names the place or the stakes."),
    "monologue": (
        "Reporter monologue",
        "Don't build the clip on a reporter or anchor talking to camera; at most one short "
        "context beat."),
    "weak_opening": (
        "Weak opening",
        "Open on the strongest image — action already under way — not a slow or static shot."),
    "boring": (
        "Boring",
        "Pick a moment with real stakes or a surprise; skip routine footage that doesn't move."),
    "off_topic": (
        "Off topic",
        "Stay on the account's topic; skip moments that aren't about it."),
    "bad_boundaries": (
        "Bad cut points",
        "Cut cleanly: start and end on complete thoughts and settled shots, never mid-word."),
    "too_long": (
        "Too long",
        "Make it shorter and tighter; drop anything that doesn't earn its seconds."),
    "other": ("Something else", ""),
}

# The ledger's dismissal reason a tag stands for, where one fits. The first
# tag that maps wins; ``other`` (and an unmapped list) leaves it blank.
DISMISS_REASON = {
    "wrong_language": "not_clipworthy",
    "monologue": "not_clipworthy",
    "no_premise": "not_clipworthy",
    "boring": "not_clipworthy",
    "weak_opening": "weak_visuals",
    "off_topic": "off_topic",
    "bad_boundaries": "bad_boundaries",
    "too_long": "too_long",
}

MAX_TEXT = 500           # one line of the operator's words
MAX_ADVICE = 1200
MAX_RULE_CHARS = 200
PROPOSE_AFTER = 3        # notes since the last proposal run before asking again
THEME_MIN_NOTES = 3
THEME_MIN_VIDEOS = 2
LOOKBACK_DAYS = 90
PROPOSAL_SOURCES = ("notes", "handbook")
_RETRY_AFTER_FAILURE = 3600.0   # a failed proposal call waits this long before another
_MAX_NOTES_IN_PROMPT = 300

_lock = threading.Lock()            # notes file appends
_proposals_lock = threading.Lock()  # rule_proposals.json read-modify-write
_propose_lock = threading.Lock()    # one proposal run at a time
_propose_thread: threading.Thread | None = None


# ---- the notes file -----------------------------------------------------------------

def notes_path() -> Path:
    return engine_dir() / "notes.jsonl"


def proposals_path() -> Path:
    return engine_dir() / "rule_proposals.json"


def tag_labels() -> list[tuple[str, str]]:
    """``[(slug, label)]`` in chip order, for the pages."""
    return [(slug, label) for slug, (label, _advice) in TAGS.items()]


def clean_tags(tags: Iterable[str] | str | None) -> list[str]:
    """Known tags only, de-duplicated, in the order given. A string is a
    comma list (what the pages post)."""
    if isinstance(tags, str):
        tags = tags.split(",")
    out: list[str] = []
    for tag in tags or ():
        slug = str(tag or "").strip().lower()
        if slug in TAGS and slug not in out:
            out.append(slug)
    return out


def dismiss_reason_for(tags: Iterable[str]) -> str:
    """The ledger's dismissal reason for the first tag that maps to one, or ""."""
    for tag in clean_tags(list(tags or ())):
        if tag in DISMISS_REASON:
            return DISMISS_REASON[tag]
    return ""


def _clean_text(text: Any, limit: int = MAX_TEXT) -> str:
    return " ".join(str(text or "").split())[:limit]


def add(candidate_id: int, *, video_id: str = "", proposal_id: int | None = None,
        kind: str = "card", tags: Iterable[str] | str | None = (), text: str = "") -> dict | None:
    """Append one note; returns it, or None when it says nothing (no tag, no
    text) or couldn't be written. Never raises: a note that fails to save must
    not cost the operator the dismissal or the re-suggest it came with."""
    if kind not in KINDS:
        raise ValueError(f"unknown note kind {kind!r}")
    tags = clean_tags(tags)
    text = _clean_text(text)
    if not tags and not text:
        return None
    note = {"id": uuid.uuid4().hex[:12], "at": util.utcnow_iso(),
            "candidate_id": int(candidate_id), "video_id": str(video_id or "")[:40],
            "proposal_id": int(proposal_id) if proposal_id is not None else None,
            "kind": kind, "tags": tags, "text": text}
    line = (json.dumps(note, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        path = notes_path()
        with _lock:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                _flock(fd)
                os.write(fd, line)   # one write of one line: appends never interleave
            finally:
                os.close(fd)         # drops the flock too
    except Exception as exc:
        log.warning("Couldn't save a clip note for candidate %s: %s", candidate_id, exc)
        return None
    log.info("Clip note %s (%s) on candidate %s: %s%s", note["id"], kind, candidate_id,
             ",".join(tags) or "-", f" — {text[:80]}" if text else "")
    return note


def _flock(fd: int) -> None:
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
    except (ImportError, OSError):   # no flock here: the thread lock still holds
        pass


def _read_all() -> list[dict]:
    """Every readable note, oldest first. A torn or corrupt line is skipped."""
    try:
        raw = notes_path().read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except Exception as exc:
        log.warning("Clip notes unreadable: %s", exc)
        return []
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            note = json.loads(line)
        except ValueError:
            continue
        if isinstance(note, dict) and note.get("id") and note.get("at"):
            out.append(note)
    return out


def _parse_at(value: Any) -> dt.datetime | None:
    try:
        stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp


def recent(days: float = LOOKBACK_DAYS) -> list[dict]:
    """Notes from the last ``days``, oldest first. Never raises."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=float(days))
    out = []
    for note in _read_all():
        at = _parse_at(note.get("at"))
        if at is not None and at >= cutoff:
            out.append(note)
    return out


def for_candidate(candidate_id: int) -> list[dict]:
    """Every note on one video, oldest first. Never raises."""
    try:
        cid = int(candidate_id)
    except (TypeError, ValueError):
        return []
    return [n for n in _read_all() if _as_int(n.get("candidate_id")) == cid]


def advice_from(tags: Iterable[str] | str | None, text: str = "") -> str:
    """The plain-English advice a note stands for: one sentence per tag, then
    the operator's own words. "" when there is nothing to say."""
    parts = [TAGS[tag][1] for tag in clean_tags(tags) if TAGS[tag][1]]
    text = _clean_text(text)
    if text:
        parts.append(text if text.endswith((".", "!", "?")) else text + ".")
    return " ".join(parts)[:MAX_ADVICE]


# ---- rule proposals -------------------------------------------------------------------

PROPOSE_SYSTEM = """\
You read an editor's complaints about the clips an AI suggested from news videos, and turn the \
complaints that keep coming back into rules the AI clip editor can follow next time.

Each note is one complaint: tags from a fixed list and, sometimes, the editor's own words. A \
"card" note was written about one suggested clip the editor turned down; a "steer" note is \
advice the editor typed before asking for suggestions.

Find THEMES: the same underlying preference expressed in several notes. For each theme write \
one rule:
- an imperative sentence of at most 200 characters, about what to pick or how to cut — \
something the clip editor can check against a clip ("Never open on a reporter talking to \
camera; open on the footage the story is about.")
- specific to what the notes say, not generic advice about good video
- list the ids of the notes that support it as evidence, and how many there are

A theme needs at least 3 notes from at least 2 different videos. Skip one-offs and \
complaints about a single video. Skip anything the existing brand rules already cover, and \
anything already proposed earlier (whether the editor added or dismissed it). Return an empty \
list when no theme qualifies — that is a normal answer.
"""

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "The rule: imperative, at most 200 characters."},
                    "evidence": {"type": "array", "items": {"type": "string"},
                                 "description": "Ids of the notes behind it."},
                    "count": {"type": "integer"},
                },
                "required": ["text", "evidence", "count"],
            },
        },
    },
    "required": ["proposals"],
}


def _read_proposals_file() -> dict:
    data = util.read_json(proposals_path(), default=None)
    if not isinstance(data, dict):
        data = {}
    items = data.get("proposals")
    data["proposals"] = [p for p in items if isinstance(p, dict)] if isinstance(items, list) else []
    return data


def proposals(status: str | None = None) -> list[dict]:
    """Every rule proposal (with its ``idx``; ``source`` defaults to
    "notes"), or only those in ``status``. Never raises."""
    try:
        items = _read_proposals_file()["proposals"]
    except Exception as exc:
        log.warning("Rule proposals unreadable: %s", exc)
        return []
    out = []
    for idx, item in enumerate(items):
        if status is None or item.get("status") == status:
            out.append({"source": "notes", **item, "idx": idx})
    return out


def pending_proposals() -> list[dict]:
    return proposals("pending")


def set_status(idx: int, status: str) -> dict | None:
    """Mark one proposal ``added`` or ``dismissed``; returns it, or None when
    there is no such pending proposal."""
    if status not in ("added", "dismissed"):
        raise ValueError(f"unknown proposal status {status!r}")
    with _proposals_lock:
        data = _read_proposals_file()
        items = data["proposals"]
        if not (0 <= idx < len(items)) or items[idx].get("status") != "pending":
            return None
        items[idx]["status"] = status
        items[idx]["decided_at"] = util.utcnow_iso()
        util.atomic_write_json(proposals_path(), data)
        return {**items[idx], "idx": idx}


def propose_rules(*, model: str | None = None, settings=None, force: bool = False) -> list[dict]:
    """Ask Claude for rules the recent notes add up to; returns the NEW
    pending proposals ([] when it didn't ask or found none).

    Asks only when at least ``PROPOSE_AFTER`` notes have arrived since the last
    run (``force`` skips that, and the wait after a failure), Claude is
    configured, and no other run is under way. A failed call is logged and
    retried no sooner than an hour later, after another note. Never raises.
    """
    if not _propose_lock.acquire(blocking=False):
        return []
    try:
        return _propose(model=model, settings=settings, force=force)
    except Exception:
        log.exception("Rule proposals from clip notes failed")
        return []
    finally:
        _propose_lock.release()


def _propose(*, model: str | None, settings, force: bool) -> list[dict]:
    from . import claude, model_for

    with _proposals_lock:
        state = _read_proposals_file()
    last_run = _parse_at(state.get("last_run_at"))
    notes = recent(LOOKBACK_DAYS)
    fresh = [n for n in notes if last_run is None or _parse_at(n.get("at")) > last_run]
    if not force:
        if len(fresh) < PROPOSE_AFTER:
            return []
        failed = _parse_at(state.get("last_error_at"))
        if failed is not None and (dt.datetime.now(dt.timezone.utc) - failed).total_seconds() \
                < _RETRY_AFTER_FAILURE:
            return []
    if not claude.is_configured():
        log.info("Rule proposals skipped: ANTHROPIC_API_KEY is not set")
        return []
    if settings is None:
        from ..config import load_settings

        settings = load_settings()
    model = model or model_for(settings, "claude", "editor")
    from ..config import load_brand_rules

    try:
        brand_rules = [r["text"] for r in load_brand_rules()]
    except Exception as exc:
        log.warning("Brand rules unreadable for rule proposals: %s", exc)
        brand_rules = []
    earlier = [{"text": p.get("text", ""), "status": p.get("status", "")}
               for p in state["proposals"]]
    shown = notes[-_MAX_NOTES_IN_PROMPT:]
    content = json.dumps({
        "existing_brand_rules": brand_rules,
        "already_proposed": earlier,
        "notes": [{"id": n["id"], "video": n.get("video_id") or str(n.get("candidate_id")),
                   "kind": n.get("kind"), "tags": n.get("tags") or [],
                   "text": n.get("text") or ""} for n in shown],
    }, ensure_ascii=False, indent=1)
    started_at = util.utcnow_iso()
    try:
        data, _usage = claude.json_call(model, system=PROPOSE_SYSTEM, content=content,
                                        schema=PROPOSE_SCHEMA, effort="medium", max_tokens=16000,
                                        label="rule-proposals", bucket="engine")
    except Exception as exc:
        log.warning("Rule proposals: Claude call failed: %s", exc)
        with _proposals_lock:
            state = _read_proposals_file()
            state["last_error_at"] = util.utcnow_iso()
            state["last_error"] = str(exc)[:300]
            util.atomic_write_json(proposals_path(), state)
        return []
    by_id = {n["id"]: n for n in shown}
    new = _accept_proposals(data, by_id, brand_rules)
    added = add_proposals(new, source="notes")
    with _proposals_lock:
        state = _read_proposals_file()
        state["last_run_at"] = started_at
        state.pop("last_error_at", None)
        state.pop("last_error", None)
        util.atomic_write_json(proposals_path(), state)
    log.info("Rule proposals: %d new from %d notes", len(added), len(shown))
    return added


def _accept_proposals(data: Any, by_id: dict[str, dict], brand_rules: list[str]) -> list[dict]:
    """The model's themes that hold up: real evidence, enough of it, on
    enough videos, not already a brand rule."""
    rules = {_key(r) for r in brand_rules}
    out = []
    items = data.get("proposals") if isinstance(data, dict) else None
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = _clip_rule(item.get("text"))
        if not text or _key(text) in rules:
            continue
        evidence = []
        for ref in item.get("evidence") or []:
            ref = str(ref)
            if ref in by_id and ref not in evidence:
                evidence.append(ref)
        videos = {by_id[ref].get("video_id") or by_id[ref].get("candidate_id") for ref in evidence}
        if len(evidence) < THEME_MIN_NOTES or len(videos) < THEME_MIN_VIDEOS:
            continue
        examples = []
        for ref in evidence:
            note = by_id[ref]
            words = note.get("text") or ", ".join(TAGS[t][0] for t in note.get("tags") or ()
                                                  if t in TAGS)
            if words and words not in examples:
                examples.append(words[:160])
            if len(examples) == 3:
                break
        out.append({"text": text, "evidence": evidence, "count": len(evidence),
                    "videos": len(videos), "examples": examples,
                    "summary": (f"From {len(evidence)} of your notes on "
                                f"{len(videos)} videos")})
    return out


def add_proposals(items: Iterable[dict], *, source: str, settings=None) -> list[dict]:
    """Queue rule proposals for the operator; returns the ones queued (with ``idx``).

    THE queue for every proposed rule, whatever proposed it: rules drawn from
    the operator's notes (``source="notes"``, ``propose_rules``) and rules
    learned from his past edits (``source="handbook"``) wait side by side on
    the Brand & audience page, and become the same kind of brand rule when
    added. Each item: ``{text, evidence (list or line), count, when?, why?,
    summary?, examples?, videos?}``.

    Every item goes through ``format_rule`` first (one Claude call each, in
    parallel, never under the file lock), so a proposal reads like every other
    rule whatever wrote it; the proposal keeps the formatted fields (``rule``,
    ``when``, ``why``, ``markdown``, ``original``) and ``text`` is what the
    engine will read once added. A text that is already a brand rule, or
    already proposed and still pending or once dismissed, is skipped
    (case-insensitive, before and after formatting): a rebuild must not re-ask
    what the operator already turned down. Never raises (a bad ``source``
    does).
    """
    source = str(source or "").strip().lower()
    if source not in PROPOSAL_SOURCES:
        raise ValueError(f"unknown proposal source {source!r}")
    blocked = _blocked_keys()
    todo: list[dict] = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        raw_text = _clean_text(item.get("text"), 1000)
        if raw_text and _key(raw_text) not in blocked:
            todo.append(item)
    if not todo:
        return []

    def fmt(item: dict) -> dict:
        return format_rule(item.get("text") or "", when=str(item.get("when") or ""),
                           why=str(item.get("why") or item.get("summary") or ""),
                           source=source, settings=settings)

    formatted = util.parallel_map(fmt, todo, workers=4, name="clip-notes-format")
    added: list[dict] = []
    try:
        with _proposals_lock:
            state = _read_proposals_file()
            blocked = _blocked_keys(state)
            for item, f in zip(todo, formatted):
                if not isinstance(f, dict):   # format_rule never raises; belt and braces
                    f = _tidy_rule(item.get("text") or "", when=str(item.get("when") or ""),
                                   why="")
                text = rule_text(f)
                if not text or _key(text) in blocked or _key(f.get("original")) in blocked:
                    continue
                blocked.add(_key(text))
                raw = item.get("evidence")
                if isinstance(raw, (list, tuple)):
                    evidence = [str(x)[:120] for x in raw if str(x).strip()][:50]
                elif raw:
                    evidence = [_clean_text(raw, 300)]
                else:
                    evidence = []
                count = _as_int(item.get("count"), len(evidence))
                count = count if count >= 0 else len(evidence)
                entry = {"text": text, "source": source, "rule": f["rule"], "when": f["when"],
                         "why": f["why"], "markdown": f["markdown"], "original": f["original"],
                         "formatted": bool(f.get("formatted")),
                         "evidence": evidence, "count": count,
                         "summary": _clean_text(item.get("summary"), 200)
                         or _default_summary(source, count),
                         "examples": [_clean_text(x, 160) for x in (item.get("examples") or [])
                                      if _clean_text(x, 160)][:3],
                         "status": "pending", "created_at": util.utcnow_iso()}
                if item.get("videos") is not None:
                    entry["videos"] = _as_int(item.get("videos"), 0)
                state["proposals"].append(entry)
                added.append({**entry, "idx": len(state["proposals"]) - 1})
            if added:
                util.atomic_write_json(proposals_path(), state)
    except Exception as exc:
        log.warning("Couldn't queue %s rule proposals: %s", source, exc)
        return []
    if added:
        log.info("Queued %d rule proposal(s) from %s", len(added), source)
    return added


def _blocked_keys(state: dict | None = None) -> set[str]:
    """Texts a new proposal may not repeat: brand rules (as the engine reads
    them and as the operator first wrote them), and proposals pending or
    dismissed."""
    keys: set[str] = set()
    try:
        from ..config import load_brand_rules

        for r in load_brand_rules():
            keys.add(_key(r.get("text")))
            if r.get("original"):
                keys.add(_key(r.get("original")))
    except Exception as exc:
        log.warning("Brand rules unreadable while queueing proposals: %s", exc)
    if state is None:
        state = _read_proposals_file()
    for p in state["proposals"]:
        if p.get("status") in ("pending", "dismissed"):
            keys.add(_key(p.get("text")))
            if p.get("original"):
                keys.add(_key(p.get("original")))
    keys.discard("")
    return keys


# ---- one formatter for every rule, one tidier for every piece of advice ---------------

FORMAT_SYSTEM = """You format the house rules a video editing assistant follows when it picks and cuts clips from news footage. Every rule — typed by the editor, drawn from the editor's notes, or learned from the editor's past edits — goes through you, so all rules read alike.

Rewrite the rule you are given as:
- rule: ONE imperative sentence in plain words, at most 160 characters, ending with a period.
- when: the condition under which it applies ("Always" when it always applies). A short noun phrase or clause, no trailing period.
- why: a short reason or the evidence behind it, only if one was given; else "".

Fix spelling and grammar. Keep the editor's meaning EXACTLY: never add a claim, an example, a number, a place or a condition that is not in the input, and never soften or harden it. If the input already states a condition ("for flood videos, ..."), move it to "when".

Examples of the house style:
**Rule:** Never use an interview in another language unless an English voiceover carries it.
**When:** Always

**Rule:** Open on the disaster footage, not on the anchor introducing it.
**When:** The video is a news package with b-roll
**Why:** The operator cut the anchor intro in most past edits.

**Rule:** End on a complete sentence, never mid-word.
**When:** Always
"""

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "rule": {"type": "string"},
        "when": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["rule", "when", "why"],
}

TIDY_SYSTEM = """You tidy one piece of advice an editor typed for a video editing assistant. Fix spelling, grammar and punctuation only. Keep the meaning, the wording and the order as they are: add nothing, drop nothing, don't rephrase what is already correct. Return the tidied text.
"""

TIDY_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}},
               "required": ["text"]}

MAX_FORMATTED_RULE = 160


def tidy_sentence(text: Any, limit: int = 400) -> str:
    """The deterministic tidy: whitespace collapsed, first letter capitalised,
    a closing full stop. What every formatter falls back to."""
    text = _clean_text(text, limit)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if not text.endswith((".", "!", "?", "…")):
        text += "."
    return text


def _clean_when(when: Any) -> str:
    when = _clean_text(when, 120).rstrip(".;:")
    if when.lower().startswith("when "):
        when = when[5:].strip()
    if not when or when.lower() in ("always", "everywhere", "all videos", "any video"):
        return "Always"
    return when[0].upper() + when[1:]


def _markdown(rule: str, when: str, why: str) -> str:
    lines = [f"**Rule:** {rule}", f"**When:** {when}"]
    if why:
        lines.append(f"**Why:** {why}")
    return "\n".join(lines)


def _tidy_rule(text: Any, *, when: str = "", why: str = "") -> dict:
    original = _clean_text(text, 1000)
    rule = tidy_sentence(original, 1000)
    if len(rule) > MAX_FORMATTED_RULE:
        rule = _clip_to(rule, MAX_FORMATTED_RULE)
    when = _clean_when(when)
    why = tidy_sentence(why, 300) if _clean_text(why) else ""
    return {"rule": rule, "when": when, "why": why, "markdown": _markdown(rule, when, why),
            "original": original, "formatted": False}


def _clip_to(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"


def format_rule(text: Any, *, when: str = "", why: str = "", source: str = "manual",
                settings=None, model: str | None = None) -> dict:
    """Every rule in the house style: ``{rule, when, why, markdown, original,
    formatted}``.

    One Claude call (effort low) fixes spelling and grammar and splits out the
    condition, keeping the meaning; the markdown is assembled here so its
    shape never varies. Without Claude, or when the call fails, a
    deterministic tidy stands in and ``formatted`` is False. Never raises.
    """
    original = _clean_text(text, 1000)
    fallback = _tidy_rule(original, when=when, why=why)
    if not original:
        return fallback
    try:
        from . import claude, model_for

        if not claude.is_configured():
            return fallback
        if settings is None:
            from ..config import load_settings

            settings = load_settings()
        model = model or model_for(settings, "claude", "editor")
        content = json.dumps({"rule_as_given": original, "when_as_given": _clean_text(when, 200),
                              "why_as_given": _clean_text(why, 300), "source": source},
                             ensure_ascii=False)
        data, _usage = claude.json_call(model, system=FORMAT_SYSTEM, content=content,
                                        schema=FORMAT_SCHEMA, effort="low", max_tokens=4000,
                                        label="format-rule", bucket="engine")
        rule = tidy_sentence((data or {}).get("rule"), 1000)
        if not rule:
            return fallback
        rule = _clip_to(rule, MAX_FORMATTED_RULE)
        when_out = _clean_when((data or {}).get("when") or when)
        why_raw = (data or {}).get("why")
        why_out = tidy_sentence(why_raw, 300) if _clean_text(why_raw) else ""
        return {"rule": rule, "when": when_out, "why": why_out,
                "markdown": _markdown(rule, when_out, why_out), "original": original,
                "formatted": True}
    except Exception as exc:
        log.warning("Rule formatting failed (%s); using a plain tidy", exc)
        return fallback


def rule_text(formatted: dict) -> str:
    """What the engine reads for a formatted rule: the sentence, prefixed by
    its condition when it has one."""
    rule = _clean_text(formatted.get("rule"), 400)
    when = _clean_when(formatted.get("when"))
    if not rule:
        return ""
    if when == "Always":
        return rule
    return f"When {when[0].lower() + when[1:]}: {rule}"[:400]


def brand_rule_fields(formatted: dict) -> dict:
    """The keyword arguments ``config.add_brand_rule`` stores with a formatted rule."""
    return {"rule": formatted.get("rule", ""), "when": formatted.get("when", ""),
            "why": formatted.get("why", ""), "markdown": formatted.get("markdown", ""),
            "original": formatted.get("original", "")}


def tidy_advice(text: Any, *, settings=None, model: str | None = None) -> str:
    """The operator's advice with spelling and grammar fixed, meaning kept.
    One Claude call (effort low); the deterministic tidy when that can't run.
    Never raises."""
    original = _clean_text(text, MAX_ADVICE)
    if not original:
        return ""
    fallback = tidy_sentence(original, MAX_ADVICE)
    try:
        from . import claude, model_for

        if not claude.is_configured():
            return fallback
        if settings is None:
            from ..config import load_settings

            settings = load_settings()
        model = model or model_for(settings, "claude", "editor")
        data, _usage = claude.json_call(model, system=TIDY_SYSTEM, content=original,
                                        schema=TIDY_SCHEMA, effort="low", max_tokens=4000,
                                        label="tidy-advice", bucket="engine")
        tidied = _clean_text((data or {}).get("text"), MAX_ADVICE)
        return tidied or fallback
    except Exception as exc:
        log.warning("Advice tidy failed (%s); using a plain tidy", exc)
        return fallback


def engine_advice(original: Any, tidied: str | None = None) -> str:
    """The advice string the engine gets: the tidied text, then the
    operator's own words (they outrank any tidying)."""
    original = _clean_text(original, MAX_ADVICE)
    if not original:
        return ""
    tidied = _clean_text(tidied, MAX_ADVICE) if tidied is not None else tidy_advice(original)
    return f"{tidied or original}\n(Operator's words: {original})"


def _default_summary(source: str, count: int) -> str:
    if source == "handbook":
        return f"Seen in {count} of your past edits" if count else "Learned from your past edits"
    return f"From {count} of your notes" if count else "From your notes"


def evidence_line(proposal: dict) -> str:
    """The short line a brand rule keeps about where it came from."""
    line = _clean_text(proposal.get("summary"), 200) or _default_summary(
        str(proposal.get("source") or "notes"), _as_int(proposal.get("count"), 0))
    when = _clean_text(proposal.get("when"), 120)
    if when and when.lower() != "always":
        line += f" (when: {when})"
    return line[:300]


def _clip_rule(text: Any) -> str:
    return _clip_to(_clean_text(text, 1000), MAX_RULE_CHARS)


def _key(text: Any) -> str:
    return " ".join(str(text or "").lower().split()).rstrip(".")


def propose_rules_soon() -> bool:
    """Run ``propose_rules`` on a daemon thread (after a note is saved).
    Best effort; False when one is already running."""
    global _propose_thread
    with _proposals_lock:
        if _propose_thread is not None and _propose_thread.is_alive():
            return False
        _propose_thread = threading.Thread(target=propose_rules, daemon=True,
                                           name="clip-notes-rules")
        _propose_thread.start()
    return True


def _as_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _reset_for_tests() -> None:
    global _propose_thread
    thread = _propose_thread
    if thread is not None:
        thread.join(timeout=5)
    _propose_thread = None


__all__ = ["KINDS", "TAGS", "DISMISS_REASON", "add", "recent", "for_candidate", "tag_labels",
           "clean_tags", "dismiss_reason_for", "advice_from", "propose_rules",
           "propose_rules_soon", "proposals", "pending_proposals", "set_status",
           "add_proposals", "evidence_line", "format_rule", "rule_text", "brand_rule_fields",
           "tidy_advice", "tidy_sentence", "engine_advice"]
