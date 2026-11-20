"""Background clip-engine runs inside the dashboard process.

An engine run takes minutes: it restores the source, logs it with a video
model, renders and watches a dozen drafts and ranks them. None of that can
happen inside a request (the browser gives up, and a sync route holds one of
the threadpool's tokens the whole time) or inside a database session (a
production Postgres transaction would sit open for the whole run). So a request
only validates and queues; a worker thread does the slow part with no session
open, then writes in one short session.

Each design choice below answers a way this goes wrong in practice:

* **Daemon threads over a priority queue.** ``ThreadPoolExecutor`` workers are
  non-daemon and CPython joins them at exit, so a queue of engine work would
  hold every ``--reload`` restart and Ctrl-C hostage. Operator clicks
  ("Suggest", "Suggest a clip here") run ahead of unattended archive work and
  restart catch-up; otherwise a triage burst leaves the button "queued" for
  half an hour. A click on a video whose unattended run is still waiting
  moves that run up and makes it the operator's (no budget gate) rather than
  joining it where it stands.
* **A durable record per video** (``engine_dir()/runs/<candidate_id>.json``),
  written at queue, start, every stage and finish, stamped with this process's
  ``BOOT_ID``. In-memory state dies with every reload; the record is how a page
  opened afterwards learns a run was interrupted instead of polling forever,
  and how ``resume_recent`` re-queues only what an earlier boot left
  unfinished — never a run that legitimately ended empty, which would
  otherwise be paid for again on every restart. A record counts as an earlier
  boot's only once its process is gone (the lock it holds while it lives,
  ``_hold_boot_lock``): any process that imports the dashboard sees the
  records, and one that took over a live dashboard's runs would pay for them
  twice.
* **Never supersede with nothing.** ``log_run`` replaces the video's pending
  suggestions, so an empty result is never written. A run that failed or
  drafted nothing usable falls back to the transcript suggester; a run whose
  drafts were all judged off topic writes nothing and says so.
* **Budget gates for unattended runs only.** Archive-time and resumed runs are
  checked against the engine's own daily cap (``clip_engine.daily_usd``, kept
  apart from ``llm.daily_budget_usd`` so engine spend can't starve captions and
  tagging) and a source-length cap; a run that passes holds its estimate
  against the cap until it ends, so parallel workers can't all pass on the
  same figure. A person pressing Suggest is not refused, matching every other
  on-demand suggest route.
* **Writer sessions, rolled back, for every read.** ``clip_context.build``
  isolates its pieces in SAVEPOINTs, which Postgres refuses on the AUTOCOMMIT
  engine behind ``session_scope(read_only=True)`` — the context would silently
  come back empty. Plain values are copied out before the rollback.

``prepare_inputs`` and ``write_result`` are public because the CLI and the
replay bench reuse the same reads and the same write.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from .. import clip_proposals, spend
from ..config import load_settings
from ..db import session_scope
from ..models import STATUS_ARCHIVED, Candidate, ClipProposal, Cut, utcnow
from . import BOOT_ID, brand, effective_setup, engine_dir, min_free_gb, opt, util

log = logging.getLogger("clip_engine.jobs")

# Lower runs first. A person waiting on a button beats unattended work, and
# catching up after a restart comes last.
PRIORITY = {"reroll": 0, "section": 0, "archive": 1, "resume": 2}

ACTIVE_STATES = ("queued", "running")
# Runs nobody asked for in the moment: these are the ones the budget gates.
UNATTENDED = ("archive", "resume")
# A run an earlier boot left unfinished is resumed at most this many times in
# total — a video that crashes or wedges the process must not loop forever.
MAX_ATTEMPTS = 2
INTERRUPTED_ERROR = "interrupted — press Suggest again"

# Every key a job record (and so a status) carries. Spec §3.
_FIELDS = ("candidate_id", "job_id", "boot_id", "reason", "setup", "window", "topic",
           "advice", "state", "stage", "outcome", "error", "fallback", "attempts", "new",
           "cost_usd", "notes", "queued_at", "started_at", "finished_at", "result_path")

_POLL_SECONDS = 1.0          # how often an idle worker re-checks for shutdown
_RETRY_DELAY = 2.0           # before the one retry of a failed DB write
_PRUNE_EVERY = 24 * 3600.0   # at most one cache prune per day
_RUN_FILES_MAX_AGE_DAYS = 45
_MAX_NOTES = 20
MAX_ADVICE = 1200
ADVICE_IGNORED_NOTE = "your advice wasn't used: the transcript suggester can't take advice"


@dataclasses.dataclass
class _Job:
    candidate_id: int
    job_id: str
    reason: str
    window: dict | None
    topic: str
    setup: str | None          # requested setup; None = whatever settings say at run time
    on_done: Callable[[], Any] | None
    priority: int
    seq: int
    advice: str = ""           # the operator's advice for this pass, as given
    advice_tidied: str = ""    # ... and tidied (notes.tidy_advice), set when the run starts


class _Interrupted(Exception):
    """The process is shutting down mid-run: record it as interrupted, and
    don't start a fallback LLM call on the way out."""


class _JobFailed(RuntimeError):
    """Both the engine and the transcript suggester failed."""


_lock = threading.RLock()
_queue: queue.PriorityQueue = queue.PriorityQueue()
_seq = itertools.count()
_status: dict[int, dict] = {}              # candidate_id -> this boot's latest job
_queued: dict[str, _Job] = {}              # job_id -> the job, while it waits its turn
_running: dict[str, int] = {}              # job_id -> candidate_id while running
_workers: list[threading.Thread] = []
_worker_count = 1
_cancel = threading.Event()                # set by shutdown(); engine runs check it
_stopping = False
# What unattended runs that passed the budget gate are expected to cost, held
# against the cap until each run ends (job_id -> usd). Its own lock: the gate
# reads the ledger file, which status polls shouldn't wait on.
_budget_lock = threading.Lock()
_reserved: dict[str, float] = {}
# (path, fd) of this process's liveness lock; see _hold_boot_lock.
_boot_lock: tuple[Path, int] | None = None
_BOOT_LOCK_PRUNE_AGE = 3600.0


# ---- public API ----------------------------------------------------------------

def enqueue(candidate_id: int, *, window: dict | None = None, topic: str = "",
            reason: str = "reroll", setup: str | None = None,
            on_done: Callable[[], Any] | None = None, advice: str = "") -> dict:
    """Queue an engine run for one video. Returns ``{"started", "conflict", **status}``.

    One run per video at a time. Asking again for the same scope while one is
    queued or running joins it (``started`` False); asking for a DIFFERENT
    scope — a chapter while the whole video is being worked on — is a
    ``conflict``, so the route can say so instead of silently dropping the
    request. Joining a run that is still waiting for a reason that outranks
    its own (Suggest on a video whose archive run is queued) moves it up
    first; see ``_promote``.

    ``advice`` is the operator's word on this pass ("open on the disaster
    footage; no interviews"), passed to ``engine.run``. Asking with advice the
    queued or running run doesn't carry is a ``conflict`` too — joining would
    quietly drop it — except when that run is unattended and still waiting:
    then it moves up as the operator's, and takes the advice. Asking with no
    advice joins whatever is under way, advised or not.
    """
    candidate_id = int(candidate_id)
    if reason not in PRIORITY:
        raise ValueError(f"unknown clip engine job reason {reason!r}")
    window = _clean_window(window)
    advice = _clean_advice(advice)
    with _lock:
        current = _status.get(candidate_id)
        if current is not None and current.get("state") in ACTIVE_STATES:
            same = _same_window(current.get("window"), window)
            if same and advice and advice != _clean_advice(current.get("advice")):
                promoted = _promote(current, reason, on_done, advice=advice)
                if promoted is None:
                    return {"started": False, "conflict": True, **_public(current)}
                return {"started": False, "conflict": False, **_public(promoted)}
            if same:
                current = _promote(current, reason, on_done) or current
            return {"started": False, "conflict": not same, **_public(current)}
        if _stopping:
            entry = _new_entry(candidate_id, "", reason, window, topic, attempts=0,
                               advice=advice)
            entry.update(state="interrupted", error=INTERRUPTED_ERROR)
            return {"started": False, "conflict": False, **_public(entry)}
        attempts = 0
        if reason == "resume":
            # Carry the count across boots: this is the same unfinished run.
            prior = _read_record(candidate_id)
            attempts = _as_int(prior.get("attempts")) if prior else 0
        job = _Job(candidate_id=candidate_id, job_id=uuid.uuid4().hex[:12], reason=reason,
                   window=window, topic=str(topic or "")[:500], setup=setup,
                   on_done=on_done, priority=PRIORITY[reason], seq=next(_seq), advice=advice)
        entry = _new_entry(candidate_id, job.job_id, reason, window, job.topic, attempts=attempts,
                           advice=advice)
        _status[candidate_id] = entry
        _queued[job.job_id] = job
        _write_record(entry)
        _ensure_workers()
        _queue.put((job.priority, job.seq, job))
        out = {"started": True, "conflict": False, **_public(entry)}
    log.info("Queued clip engine job %s for candidate %s (%s%s%s)", job.job_id, candidate_id,
             reason, f", {window['start']:.0f}-{window['end']:.0f}s" if window else "",
             ", with advice" if advice else "")
    return out


def _promote(entry: dict, reason: str, on_done: Callable[[], Any] | None, *,
             advice: str | None = None) -> dict | None:
    """Move a job that is still waiting up to ``reason``'s place in the queue,
    as a run for ``reason``; returns its new entry. None when it has already
    left the queue (running, or about to: ``_run`` takes it out of
    ``_queued`` before it says so) or ``reason`` doesn't outrank it. Caller
    holds ``_lock``.

    The case it exists for: an archive run waits behind a triage burst (or a
    restart re-queued one as "resume") and the operator opens the video and
    presses Suggest. Joined unchanged, the click would wait behind every
    earlier unattended run and then meet the budget gate, which a person
    pressing Suggest never should. The run is re-queued as a new job (new id,
    new place, the new reason) that keeps the old one's scope, setup,
    attempts and callback; ``_run`` skips the old queue slot as superseded.
    ``advice`` (when given) replaces the old run's: an unattended run had
    nobody's advice, and the operator pressing Suggest is now its owner.
    """
    old = _queued.get(str(entry.get("job_id") or ""))
    if old is None or PRIORITY[reason] >= old.priority:
        return None
    if advice is not None and old.reason not in UNATTENDED:
        return None   # someone asked for that run as it is; their advice stands
    job = dataclasses.replace(old, job_id=uuid.uuid4().hex[:12], reason=reason,
                              priority=PRIORITY[reason], seq=next(_seq),
                              on_done=old.on_done or on_done,
                              advice=old.advice if advice is None else advice)
    promoted = dict(entry, job_id=job.job_id, reason=reason, advice=job.advice)
    _status[job.candidate_id] = promoted
    del _queued[old.job_id]
    _queued[job.job_id] = job
    _write_record(promoted)
    _queue.put((job.priority, job.seq, job))
    log.info("Clip engine job %s for candidate %s (%s) moved up as %s (%s)", old.job_id,
             job.candidate_id, old.reason, job.job_id, reason)
    return promoted


def status(candidate_id: int) -> dict:
    """Where this video's latest run stands.

    This boot's in-memory entry when there is one, else the durable record. A
    record an earlier boot left queued or running was killed by a restart, so
    it reports ``interrupted`` — a page polling it stops instead of spinning.
    (One whose process is still running — another process reading this
    one's records — reports what it says.) No record at all is ``idle``.
    """
    candidate_id = int(candidate_id)
    with _lock:
        entry = _status.get(candidate_id)
        if entry is not None:
            return _public(entry)
    record = _read_record(candidate_id)
    if record is None:
        return _public(_idle(candidate_id))
    entry = {**_idle(candidate_id), **{k: record[k] for k in _FIELDS if k in record}}
    if entry.get("state") in ACTIVE_STATES and not _owner_alive(entry):
        entry.update(state="interrupted", stage="", error=INTERRUPTED_ERROR)
    return _public(entry)


def resume_recent(*, hours: float = 24, on_done: Callable[[], Any] | None = None) -> int:
    """Queue engine runs for recently archived videos that never got suggestions.

    A restart (every save under app/ with reload on) kills the run an archive
    queued, and the archive no longer suggests inline when the engine is on —
    so without this such a video would never get suggestions. Picks videos
    archived within ``hours`` that have a transcript, no pending suggestion, no
    clip with segments yet, and either a job record an earlier boot left
    unfinished with fewer than ``MAX_ATTEMPTS`` starts, or no record and no
    suggestion of any verdict: a video with dismissed, superseded or accepted
    suggestions but no record was suggested before the engine ran here (the
    transcript suggester runs inline at archive time while the engine is
    off), and the operator has already dealt with them. Finished runs —
    including ones that legitimately found nothing — are never re-run
    automatically. Returns how many were queued.
    """
    settings = load_settings()
    try:
        setup, why = effective_setup(settings)
    except Exception:
        log.exception("Clip engine setup unreadable; not resuming engine runs")
        return 0
    if setup == "transcript":
        return 0
    cutoff = utcnow() - dt.timedelta(hours=float(hours))
    with session_scope() as session:
        try:
            rows = session.execute(
                select(Candidate.id, Candidate.transcript_path)
                .where(Candidate.status == STATUS_ARCHIVED,
                       Candidate.archived_at.is_not(None),
                       Candidate.archived_at >= cutoff)
                .order_by(Candidate.archived_at.asc())
            ).all()
            ids = [cid for cid, path in rows if path]
            verdicts: list = []
            cut_rows: list = []
            if ids:
                verdicts = session.execute(
                    select(ClipProposal.candidate_pk, ClipProposal.verdict)
                    .where(ClipProposal.candidate_pk.in_(ids)).distinct()
                ).all()
                cut_rows = session.execute(
                    select(Cut.candidate_pk, Cut.trim_segments).where(Cut.candidate_pk.in_(ids))
                ).all()
        finally:
            session.rollback()
    suggested = {pk for pk, _verdict in verdicts}
    pending = {pk for pk, verdict in verdicts if verdict == ClipProposal.VERDICT_PENDING}
    # The trim editor creates an EMPTY cut the moment a video is opened, so
    # only a cut that holds segments means the operator has started clipping.
    clipped = {pk for pk, segments in cut_rows if _has_segments(segments)}
    started = 0
    for cid, path in rows:
        if not path or cid in pending or cid in clipped or not Path(path).is_file():
            continue
        record = _read_record(cid)
        if record is None and cid in suggested:
            continue   # suggested before the engine ran here, and already dealt with
        if not _resumable(record):
            continue
        try:
            result = enqueue(cid, reason="resume", on_done=on_done)
        except Exception:
            log.exception("Couldn't resume clip engine run for candidate %s", cid)
            continue
        if result.get("started"):
            started += 1
    if started:
        log.info("Resumed %d clip engine run(s) left unfinished by a restart", started)
    return started


def prepare_inputs(candidate_id: int, *, window: dict | None = None, as_of=None,
                   replay: bool = False) -> dict:
    """Everything a run reads from the database, as plain values.

    One writer session, rolled back before it closes (see the module
    docstring for why not a read-only one); nothing ORM-bound escapes it, so no
    session is open during the minutes of model calls that follow. Each piece
    is read in its own SAVEPOINT and its own try/except: one failing query
    yields an empty value and a line in ``notes`` instead of poisoning the
    transaction for the rest, or the run.

    ``other_clips`` are the clips already made from this video, oldest first:
    ``[{cut_pk, title, segments (play order), opening, words_heard}]``. Their
    footage may be reused; a new clip must open on a different shot and tell a
    different story. ``used`` stays in the dict for old callers, always empty.

    ``replay`` (the bench) leaves out everything that leaks the operator's
    later verdicts on this video: other clips, dismissals and striking
    frames, and asks ``clip_context`` for its replay view without calibration.
    """
    from .. import clip_context, vision
    from . import taste

    settings = load_settings()
    notes: list[str] = []
    out: dict[str, Any] = {"candidate": None, "used": [], "other_clips": [], "context": {},
                           "rejected": [], "examples": [], "striking": [], "notes": notes,
                           "window": _clean_window(window)}
    with session_scope() as session:
        try:
            c = session.get(Candidate, int(candidate_id))
            if c is None:
                notes.append("video not found")
                return out
            fields = _candidate_fields(c)
            out["candidate"] = fields
            striking_raw = c.striking_frames or ""
            if not replay:
                out["other_clips"] = _piece(
                    session, "other clips", notes, [],
                    lambda: clip_proposals.other_clips(session, fields["id"]))
                out["rejected"] = _piece(session, "dismissed suggestions", notes, [],
                                         lambda: _rejected(session, fields["id"]))
                out["striking"] = _piece(
                    session, "striking frames", notes, [],
                    lambda: list(vision.load_striking(SimpleNamespace(striking_frames=striking_raw))))
            out["context"] = _piece(
                session, "account context", notes, {},
                lambda: clip_context.build(session, c, settings, as_of=as_of, replay=replay,
                                           calibration="none" if replay else "engine") or {})
            out["examples"] = _piece(
                session, "taste examples", notes, [],
                lambda: list(taste.select_examples(
                    session, as_of=as_of, exclude_candidate_pk=fields["id"],
                    hits=_as_int(opt(settings, "clip_engine.taste_hits", 3)),
                    flops=_as_int(opt(settings, "clip_engine.taste_flops", 2)),
                    lookback_days=_as_int(opt(settings, "clip_engine.taste_lookback_days", 180)),
                    seed=fields["video_id"]) or []))
        finally:
            session.rollback()
    if out["other_clips"] and out["candidate"]:
        _hear_other_clips(out["other_clips"], out["candidate"])
    return out


def _hear_other_clips(clips: list[dict], fields: dict) -> None:
    """Fill each other clip's ``words_heard`` from the video's word stream
    (or its transcript), read from local JSON after the session has closed.
    Best effort: a clip keeps "" when there is nothing to read."""
    try:
        files = SimpleNamespace(transcript_path=fields.get("transcript_path") or "",
                                word_transcript_path=fields.get("word_transcript_path") or "")
        words = clip_proposals.load_words(files)
        transcript = [] if words else clip_proposals.load_transcript(files)
        for clip in clips:
            clip["words_heard"] = clip_proposals.words_heard(clip.get("segments") or [],
                                                             words, transcript)
    except Exception as exc:
        log.warning("Couldn't read what the other clips of video %s say: %s",
                    fields.get("video_id"), exc)


def write_result(candidate_id: int, result, *, window: dict | None = None) -> list[int]:
    """Log an engine result's top picks as pending suggestions; returns the new ids.

    The same write the transcript suggester makes (``log_run`` plus the
    whole-video side effects), so the ledger, the cards and the trim editor
    need no second path — with three differences: confidence is the critic's
    score (None when no critic watched it, rather than a fake 0), the
    multi-clip marker counts distinct stories rather than clips (two cuts of
    one story are one clip's worth of material), and each proposal gets a
    sidecar holding its preview render and the critic's notes.

    An empty top writes nothing: ``log_run`` would otherwise supersede the
    video's pending suggestions with nothing. A dropped connection is retried
    once in a fresh session.
    """
    window = _clean_window(window)
    top = [item for item in (getattr(result, "top", None) or []) if isinstance(item, dict)]
    clips = [_clip_for_log(item) for item in top]
    if not any(clip["segments"] for clip in clips):
        return []
    model = str(getattr(result, "model", "") or "")[:60]
    policy = str(getattr(result, "policy", "") or "")[:40]
    stories = {_story_key(item, i) for i, item in enumerate(top)}

    def write(session) -> list[tuple[int, int | None]]:
        c = session.get(Candidate, int(candidate_id))
        if c is None:
            log.warning("Candidate %s vanished before its suggestions could be saved", candidate_id)
            return []
        ids = list(clip_proposals.log_run(session, c.id, clips, model=model, policy=policy,
                                          window=window) or [])
        if not ids:
            return []
        if window is None:
            caption = str(top[0].get("draft_caption") or "").strip()
            if caption:
                c.draft_caption = caption
            # Never overwrite a marker the operator set by hand.
            if not c.multi_clip_potential or c.multi_clip_auto:
                auto = len(stories) > 1
                c.multi_clip_potential = auto
                c.multi_clip_auto = auto
        # log_run skips a clip whose segments clean away to nothing, so map
        # rows back to their clip by clip_index rather than by position.
        index = dict(session.execute(
            select(ClipProposal.id, ClipProposal.clip_index).where(ClipProposal.id.in_(ids))
        ).all())
        return [(pid, index.get(pid)) for pid in ids]

    written = _db_write(write)
    for pid, idx in written:
        if isinstance(idx, int) and 0 <= idx < len(top):
            _save_sidecar(pid, top[idx], result, window=window)
    return [pid for pid, _ in written]


def shutdown() -> None:
    """Stop taking work and cancel the run in progress (the dashboard is going down).

    Queued jobs are dropped but their records still say "queued", so the next
    boot reports them interrupted and ``resume_recent`` can pick them up. The
    running job stops at its next stage boundary and records itself as
    interrupted. Workers are daemon threads, so nothing here waits on them.
    """
    global _stopping
    with _lock:
        _stopping = True
        _cancel.set()
        dropped = 0
        while True:
            try:
                _queue.get_nowait()
            except queue.Empty:
                break
            dropped += 1
        _queued.clear()
    if dropped:
        log.info("Clip engine shutting down; %d queued run(s) left for the next start", dropped)


# ---- the worker ----------------------------------------------------------------

def _ensure_workers() -> None:
    """Start the daemon workers on first use (importing this module starts nothing)."""
    global _worker_count
    with _lock:
        _workers[:] = [t for t in _workers if t.is_alive()]
        try:
            want = int(opt(load_settings(), "clip_engine.workers", 1))
        except (TypeError, ValueError):
            want = 1
        want = max(1, min(4, want))
        _worker_count = want
        while len(_workers) < want:
            t = threading.Thread(target=_worker_loop, args=(_cancel,), daemon=True,
                                 name=f"clip-engine-job-{len(_workers) + 1}")
            t.start()
            _workers.append(t)


def _worker_loop(stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            _priority, _seq_no, job = _queue.get(timeout=_POLL_SECONDS)
        except queue.Empty:
            continue
        if stop.is_set():
            break   # its record still says queued: the next boot resumes or reports it
        try:
            _run(job)
        except Exception:   # _run handles its own errors; this is a last guard
            log.exception("Clip engine worker crashed on job %s", getattr(job, "job_id", "?"))


def _run(job: _Job) -> None:
    with _lock:
        _queued.pop(job.job_id, None)
        current = _status.get(job.candidate_id)
        if current is None or current.get("job_id") != job.job_id:
            return   # superseded while waiting
        _running[job.job_id] = job.candidate_id
        attempts = _as_int(current.get("attempts")) + 1
    started = time.monotonic()
    final: dict = {}
    try:
        _update(job, state="running", stage="Getting ready", started_at=util.utcnow_iso(),
                attempts=attempts)
        log.info("Clip engine job %s started for candidate %s (%s, attempt %d)",
                 job.job_id, job.candidate_id, job.reason, attempts)
        try:
            final = _execute(job)
        except _Interrupted:
            final = {"state": "interrupted", "error": INTERRUPTED_ERROR}
        except _JobFailed as exc:
            log.warning("Clip engine job %s for candidate %s failed: %s",
                        job.job_id, job.candidate_id, exc)
            final = {"state": "failed", "outcome": "failed", "error": _short(exc, 400)}
        except Exception as exc:
            log.exception("Clip engine job %s for candidate %s failed",
                          job.job_id, job.candidate_id)
            final = {"state": "failed", "outcome": "failed", "error": _short(exc, 400)}
        final.setdefault("state", "done")
        final["stage"] = ""
        final["finished_at"] = util.utcnow_iso()
        if "notes" in final:
            final["notes"] = _clean_notes(final["notes"])
        _update(job, **final)
        log.info("Clip engine job %s for candidate %s: %s/%s, %d new, $%.2f, %.0fs%s",
                 job.job_id, job.candidate_id, final.get("state"), final.get("outcome", ""),
                 len(final.get("new") or []), float(final.get("cost_usd") or 0.0),
                 time.monotonic() - started,
                 f" — {final['error']}" if final.get("error") else "")
    finally:
        with _lock:
            _running.pop(job.job_id, None)
        _release_budget(job.job_id)
        if job.on_done is not None:
            try:
                job.on_done()
            except Exception:
                log.exception("Clip engine job %s: on_done callback failed", job.job_id)
        _maybe_prune()


def _execute(job: _Job) -> dict:
    """One run, start to finish. Returns the record fields to finish with."""
    settings = load_settings()
    unattended = job.reason in UNATTENDED
    inputs = prepare_inputs(job.candidate_id, window=job.window)
    notes: list[str] = list(inputs.get("notes") or [])
    cand = inputs.get("candidate")
    if not cand:
        return {"state": "skipped", "outcome": "failed", "error": "video not found", "notes": notes}
    if unattended and cand.get("status") != STATUS_ARCHIVED:
        # Rejected or reset while it waited: nobody wants clips from it now.
        return {"state": "skipped", "outcome": "failed",
                "error": "the video left the workflow before its turn", "notes": notes}

    # Local JSON files: read them before touching the source, so a missing
    # source still leaves the transcript suggester something to work with.
    files = SimpleNamespace(transcript_path=cand.get("transcript_path") or "",
                            word_transcript_path=cand.get("word_transcript_path") or "")
    transcript = clip_proposals.load_transcript(files)
    if not transcript:
        return {"state": "done", "outcome": "failed", "error": "no transcript", "notes": notes}
    words = clip_proposals.load_words(files)
    # The transcript suggester (legacy.suggest) reads these from ``inputs``
    # rather than parsing the files a second time.
    inputs["transcript"] = transcript
    inputs["words"] = words

    setup, why_not = effective_setup(settings, job.setup)
    _update(job, setup=setup)
    if setup == "transcript":
        return _legacy(job, settings, inputs, notes, unattended=unattended,
                       fallback=why_not or "the clip engine is set to the transcript suggester")
    try:
        from . import engine, media
    except Exception as exc:   # a broken engine module must still leave suggestions
        log.exception("Clip engine modules failed to import")
        return _legacy(job, settings, inputs, notes, unattended=unattended,
                       fallback=f"the clip engine failed to load: {_short(exc)}")
    blocked = _provider_blocked(setup, attended=not unattended)
    if blocked:
        # Fail fast: restoring the source and building proxies only to be
        # refused at the first Gemini call would delay the fallback for nothing.
        return _legacy(job, settings, inputs, notes, unattended=unattended, fallback=blocked)

    duration = _positive(cand.get("duration_seconds"))
    if unattended and duration:
        refusal = _budget_refusal(setup, settings, duration, job_id=job.job_id)
        if refusal:
            return _over_budget(job, settings, inputs, notes, refusal)

    # The operator's advice, spelling and grammar fixed (a cheap Claude call,
    # here rather than in the request so Suggest answers at once). The engine
    # gets both the tidied text and the words as typed.
    advice_for_engine = ""
    if job.advice:
        _update(job, stage="Reading your advice")
        try:
            from . import notes as notes_store   # ``notes`` here is the run's note list

            job.advice_tidied = notes_store.tidy_advice(job.advice, settings=settings)
            advice_for_engine = notes_store.engine_advice(job.advice, job.advice_tidied)
        except Exception as exc:
            log.warning("Couldn't tidy the advice for candidate %s: %s", job.candidate_id, exc)
            advice_for_engine = job.advice

    min_free = min_free_gb(settings)
    try:
        free = float(media.disk_free_gb(engine_dir()))
    except Exception as exc:
        log.warning("Couldn't measure free disk space: %s", exc)
        free = math.inf
    if free < min_free:
        return _legacy(job, settings, inputs, notes, unattended=unattended,
                       fallback=f"only {free:.1f} GB of disk free (clip_engine.min_free_gb is {min_free:g})")

    _update(job, stage="Fetching the source video")
    try:
        from .. import offload

        local = Path(offload.ensure_local(cand.get("local_video_path") or "",
                                          cand.get("source_object_key") or ""))
    except Exception as exc:
        return _legacy(job, settings, inputs, notes, unattended=unattended,
                       fallback=f"the source video is unavailable: {_short(exc)}")

    if not duration:
        try:
            duration = _positive(media.probe_duration(local))
        except Exception as exc:
            log.warning("Couldn't probe %s: %s", local, exc)
        if unattended and duration:
            refusal = _budget_refusal(setup, settings, duration, job_id=job.job_id)
            if refusal:
                return _over_budget(job, settings, inputs, notes, refusal)

    # Read before the pin below: from here to the try that releases it,
    # nothing may raise.
    timeout = _number_setting(settings, "clip_engine.job_timeout_seconds", 1800)
    link = None
    try:
        # A stable hardlink: an offload sweep or a re-archive can't pull the
        # file out from under ffmpeg mid-run, and the path (which the render
        # and upload caches key on) is the same on every run of this video.
        link = Path(media.link_inflight(local, cand["video_id"]))
    except Exception as exc:
        notes.append(f"couldn't pin the source for the run ({_short(exc)}); using it in place")

    meter, result, failure, stopped = None, None, "", None
    try:
        meter = engine.CostMeter("engine")
        result = engine.run(
            setup=setup, settings=settings, brand=brand(), playbook=None,
            video_id=cand["video_id"], title=cand.get("title") or "",
            description=cand.get("description") or "", duration=duration,
            source=link or local, words=words, transcript=transcript,
            context=inputs.get("context") or {}, rejected=inputs.get("rejected") or [],
            examples=inputs.get("examples") or [], used=[],
            other_clips=inputs.get("other_clips") or [],
            window=job.window, section_topic=job.topic, keep=None, meter=meter,
            progress=lambda label: _update(job, stage=str(label or "")[:80]),
            cancel=_cancel, deadline=time.monotonic() + timeout, advice=advice_for_engine)
    except util.Cancelled as exc:
        if _cancel.is_set():
            raise _Interrupted() from exc
        failure, stopped = f"the video model ran out of time ({timeout:.0f} s)", exc
    except Exception as exc:
        if _cancel.is_set():
            raise _Interrupted() from exc
        log.warning("Clip engine run failed for candidate %s: %s", job.candidate_id, exc)
        failure, stopped = f"the video model failed: {_short(exc)}", exc
    finally:
        if link is not None:
            try:
                media.release_inflight(link)
            except Exception as exc:
                log.debug("Couldn't release %s: %s", link, exc)
    if failure:
        return _engine_failed(job, settings, inputs, notes, failure=failure, error=stopped,
                              meter=meter, unattended=unattended)

    cost = _finite(getattr(result, "cost_usd", None))
    if cost is None:
        cost = _meter_total(meter)
    notes.extend(str(n) for n in (getattr(result, "notes", None) or []))
    top = [item for item in (getattr(result, "top", None) or []) if isinstance(item, dict)]
    outcome = str(getattr(result, "outcome", "") or "proposals")
    # On disk before the database write: a failed write loses nothing, and the
    # record points at it whatever happens next.
    result_path = _save_result(job, _engine_payload(result, job))
    _update(job, result_path=result_path, cost_usd=cost)
    if not top:
        _drop_renders(result, keep=())
        if outcome in ("empty_topic", "empty_drafts"):
            # An answer, not a malfunction: the editor found nothing worth
            # drafting, or the critic watched every draft and found none on
            # topic. (Mechanical losses — nothing survived snapping, nothing
            # rendered — raise EngineError and took the fallback above.) The
            # video's pending suggestions stay.
            if outcome == "empty_topic":
                return {"state": "done", "outcome": "empty_topic",
                        "error": "found nothing on topic", "new": [], "cost_usd": cost,
                        "notes": notes, "result_path": result_path}
            return {"state": "done", "outcome": "empty",
                    "error": "found nothing worth drafting", "new": [], "cost_usd": cost,
                    "notes": notes, "result_path": result_path}
        # "proposals" with no picks would be an engine bug; the video still
        # gets suggestions.
        return _legacy(job, settings, inputs, notes, unattended=unattended, engine_cost=cost,
                       result_path=result_path, fallback="the video model returned no picks")
    _update(job, stage="Saving suggestions")
    ids = write_result(job.candidate_id, result, window=job.window)
    _drop_renders(result, keep=top)
    return {"state": "done", "outcome": "proposals" if ids else "empty", "new": ids,
            "cost_usd": cost, "notes": notes, "result_path": result_path}


def _engine_failed(job: _Job, settings, inputs: dict, notes: list[str], *, failure: str,
                   error: BaseException | None, meter, unattended: bool) -> dict:
    """The engine stopped short of a result. Keep what it noted on the way
    (``engine.run`` attaches its notes: the best clue to why), put the failed
    pass on disk next to the job's other run files, delete the drafts it had
    rendered (nobody will be shown them; a retry renders them again), then
    let the transcript suggester stand in."""
    engine_notes = [str(n) for n in (getattr(error, "notes", None) or [])]
    notes.extend(engine_notes)
    _drop_files(getattr(error, "renders", None) or [])
    cost = _meter_total(meter)
    saved = _save_result(job, {
        "kind": "engine-error", "candidate_id": job.candidate_id, "job_id": job.job_id,
        "boot_id": BOOT_ID, "reason": job.reason, "window": job.window, "advice": job.advice,
        "error": failure,
        "notes": engine_notes, "cost_usd": round(cost, 6),
        "by_stage": dict(getattr(meter, "by_stage", None) or {}),
        "tokens": dict(getattr(meter, "tokens", None) or {}),
        "saved_at": util.utcnow_iso()}, suffix="-error")
    if saved:
        _update(job, result_path=saved, cost_usd=round(cost, 4))
    return _legacy(job, settings, inputs, notes, unattended=unattended, engine_cost=cost,
                   result_path=saved, fallback=failure)


def _legacy(job: _Job, settings, inputs: dict, notes: list[str], *, fallback: str,
            unattended: bool, engine_cost: float = 0.0, result_path: str = "") -> dict:
    """The transcript suggester in place of the engine. No session is open
    during its model call; its proposals are written in a short one after."""
    if _cancel.is_set():
        raise _Interrupted()
    from . import legacy

    if unattended and not _within_llm_budget():
        return {"state": "skipped", "outcome": "budget", "fallback": fallback,
                "error": "the daily LLM budget is spent", "cost_usd": engine_cost,
                "notes": notes, "result_path": result_path}
    log.info("Clip engine job %s for candidate %s: using the transcript suggester (%s)",
             job.job_id, job.candidate_id, fallback)
    if job.advice and ADVICE_IGNORED_NOTE not in notes:
        notes.append(ADVICE_IGNORED_NOTE)
    _update(job, stage="Reading the transcript instead", fallback=fallback)
    context = _legacy_context(job.candidate_id, settings, inputs.get("context") or {})
    try:
        res = legacy.suggest(settings=settings, inputs=inputs, window=job.window,
                             topic=job.topic, context=context) or {}
    except Exception as exc:
        if _cancel.is_set():
            raise _Interrupted() from exc
        raise _JobFailed(f"{fallback}; the transcript suggester failed too: {_short(exc)}") from exc
    clips = [clip for clip in (res.get("clips") or []) if isinstance(clip, dict)]
    cost = engine_cost + (_finite(res.get("cost_usd")) or 0.0)
    saved = _save_result(job, {"kind": "legacy", "candidate_id": job.candidate_id,
                               "job_id": job.job_id, "fallback": fallback,
                               "advice_ignored": bool(job.advice),
                               "saved_at": util.utcnow_iso(), **res}, suffix="-legacy")
    ids: list[int] = []
    if clips:
        _update(job, stage="Saving suggestions")

        def write(session) -> list[int]:
            c = session.get(Candidate, job.candidate_id)
            if c is None:
                return []
            return list(legacy.write(session, c, clips, model=str(res.get("model") or "")[:60],
                                     policy=str(res.get("policy") or "")[:40],
                                     window=job.window) or [])

        ids = _db_write(write)
    return {"state": "done", "outcome": "legacy", "new": ids, "fallback": fallback,
            "cost_usd": round(cost, 4), "notes": notes, "result_path": saved or result_path}


def _over_budget(job: _Job, settings, inputs: dict, notes: list[str], refusal: str) -> dict:
    """An unattended run the engine may not take: the transcript suggester
    runs instead while the general LLM budget allows, else nothing does."""
    if _within_llm_budget():
        return _legacy(job, settings, inputs, notes, unattended=True, fallback=refusal)
    return {"state": "skipped", "outcome": "budget", "error": refusal, "notes": notes}


def _budget_refusal(setup: str, settings, duration: float, *, job_id: str = "") -> str:
    """Why an unattended run may not use the engine, or "" when it may.

    A run that may holds its estimate against the cap (under ``job_id``)
    until it ends (``_release_budget``): with ``clip_engine.workers`` above 1,
    runs starting together would otherwise all pass on the same spent figure.
    The check and the hold are one step under ``_budget_lock``.

    Fails closed: a spend ledger that can't be read means no engine run, so a
    broken guard can never mean unlimited spend. (``spend.record`` moves a
    damaged ledger aside on its next write and counts the day afresh from
    there; this gate can't see past that.)
    """
    max_source = _number_setting(settings, "clip_engine.max_source_seconds", 3600)
    if duration > max_source:
        return (f"the video runs {duration / 60:.0f} min, longer than "
                f"clip_engine.max_source_seconds ({max_source / 60:.0f} min)")
    daily = _number_setting(settings, "clip_engine.daily_usd", 5.0)
    try:
        from . import engine

        estimate = float(engine.estimate_usd(setup, settings, duration))
    except Exception as exc:
        log.warning("Couldn't estimate the clip engine's cost: %s", exc)
        return f"the clip engine's budget couldn't be checked ({_short(exc)})"
    with _budget_lock:
        try:
            spent = _engine_spent_today()
        except Exception as exc:
            log.warning("Couldn't check the clip engine budget: %s", exc)
            return f"the clip engine's budget couldn't be checked ({_short(exc)})"
        held = sum(usd for other, usd in _reserved.items() if other != job_id)
        if spent + held + estimate > daily:
            return (f"the clip engine's daily budget is spent (${spent:.2f} of ${daily:.2f} today"
                    + (f", plus ${held:.2f} held for runs in progress" if held else "")
                    + f"; this video needs about ${estimate:.2f})")
        if job_id:
            _reserved[job_id] = estimate
    return ""


def _release_budget(job_id: str) -> None:
    """The run is over: what it spent is in the ledger now, not on hold."""
    with _budget_lock:
        _reserved.pop(job_id, None)


def _engine_spent_today() -> float:
    """Today's engine spend, for the budget gate. ``spend.bucket_spend``
    reads a ledger it can't parse as $0 spent — fine for a page, wrong for a
    guard — so the file is parsed here first and a damaged one raises. A
    ledger that doesn't exist yet is a day with nothing spent."""
    try:
        raw = Path(spend._LEDGER).read_bytes()
    except FileNotFoundError:
        raw = None
    if raw is not None and not isinstance(json.loads(raw), dict):
        raise ValueError("the spend ledger isn't a JSON object")
    return float(spend.bucket_spend("engine"))


def _provider_blocked(setup: str, *, attended: bool = False) -> str:
    """Why the engine can't run right now though its keys are set ("" when it
    can): Gemini's breaker is open (daily quota or prepaid credit exhausted).

    A run the operator asked for (``attended``) first closes a breaker that a
    402 opened: credit takes a minute to add, and Suggest is how they try
    again after adding it. If it is still missing, the run's first Gemini
    call opens the breaker again and the run falls back like any failure. A
    day's quota (429) can't be fixed that way, so that breaker stands."""
    from . import ROLES

    if "gemini" not in ROLES.get(setup, {}).values():
        return ""
    try:
        from . import gemini

        if attended:
            gemini.reset_breaker(status=402)
        return gemini.breaker_reason()
    except Exception as exc:   # the check is an optimisation; the run finds out anyway
        log.debug("Couldn't read the Gemini breaker: %s", exc)
        return ""


def _within_llm_budget() -> bool:
    try:
        return bool(spend.within_budget())
    except Exception as exc:
        log.warning("Couldn't read the LLM budget: %s", exc)
        return False


def _legacy_context(candidate_id: int, settings, default: dict) -> dict:
    """The account context exactly as the transcript suggester has always
    seen it (legacy calibration), rather than the engine's. Best effort: the
    engine's context stands in when it can't be read."""
    from .. import clip_context

    try:
        with session_scope() as session:
            try:
                c = session.get(Candidate, candidate_id)
                return (clip_context.build(session, c, settings) or {}) if c is not None else default
            finally:
                session.rollback()
    except Exception as exc:
        log.warning("Account context unavailable for the transcript suggester: %s", exc)
        return default


# ---- writing -------------------------------------------------------------------

def _db_write(fn: Callable[[Any], Any]) -> Any:
    """``fn(session)`` in a fresh writer session, retried once on a dropped
    connection. The retry is a whole new transaction, so nothing doubles."""
    for attempt in (1, 2):
        try:
            with session_scope() as session:
                return fn(session)
        except OperationalError as exc:
            if attempt == 2:
                raise
            log.warning("Saving suggestions hit a database error (%s); retrying once",
                        _short(exc, 300))
            time.sleep(_RETRY_DELAY)
    return None


def _clip_for_log(item: dict) -> dict:
    return {"segments": _plain_segments(item.get("segments")),
            "story": str(item.get("story") or item.get("hook") or ""),
            "why": str(item.get("why") or ""),
            "confidence": _confidence(item),
            "challenges_rules": str(item.get("challenges_rules") or "")}


def _confidence(item: dict) -> float | None:
    """The critic's score on the ledger's 0-1 scale; None when no critic
    watched it (an outage must not read as a verdict of 0)."""
    critique = item.get("critique") or {}
    if isinstance(critique, dict) and critique.get("error"):
        return None
    score = _finite(item.get("score"))
    if score is None:
        return None
    return round(min(1.0, max(0.0, score / 10.0)), 3)


def _story_key(item: dict, index: int) -> str:
    key = str(item.get("story_key") or "").strip().lower()
    return key or f"#{item.get('id') or index}"


def _topic_doubt(item: dict) -> str:
    value = item.get("topic_doubt")
    if isinstance(value, str):
        return value.strip()[:300]
    if value:
        note = ""
        critique = item.get("critique")
        if isinstance(critique, dict):
            note = str(critique.get("topic_note") or "").strip()
        return (note or "the critic doubted this is on topic")[:300]
    return ""


def _save_sidecar(proposal_id: int, item: dict, result, *, window: dict | None = None) -> None:
    from . import sidecars

    critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
    render = item.get("render")
    meta = {
        "proposal_id": proposal_id,
        "setup": str(getattr(result, "setup", "") or ""),
        "archetype": str(item.get("archetype") or ""),
        "lens": str(item.get("lens") or ""),
        "hook": str(item.get("hook") or ""),
        "payoff": str(item.get("payoff") or ""),
        "risk": str(item.get("risk") or ""),
        "rank": item.get("rank"),
        "score": _finite(item.get("score")),
        "critique": critique,
        "credits": str(item.get("credits") or ""),
        # What the preview render actually shows, in play order. The cards
        # compare it with what they offer and drop the preview once those differ.
        "segments": _plain_segments(item.get("segments")),
        "notes": [str(n)[:300] for n in (getattr(result, "notes", None) or [])][:_MAX_NOTES],
        "story_key": str(item.get("story_key") or ""),
        "topic_doubt": _topic_doubt(item),
        # The engine kept it although its speech may not be understood (every
        # draft had the problem); the card says so.
        "language_doubt": str(item.get("language_doubt") or "").strip()[:300],
        "draft_id": str(item.get("id") or ""),
        "origin": str(item.get("origin") or ""),
        # The pass's scope: "What's wrong?" on this card re-suggests the same
        # stretch (None = the whole video).
        "window": window,
        "created_at": util.utcnow_iso(),
    }
    try:
        sidecars.save(proposal_id, meta=meta, render=Path(render) if render else None)
    except Exception as exc:   # sidecars.save never raises by contract; belt and braces
        log.warning("Couldn't save the sidecar for proposal %s: %s", proposal_id, exc)


def _drop_renders(result, *, keep) -> None:
    """Delete this run's draft renders that no suggestion shows (the chosen
    ones were copied into their sidecars). Disk is tight; prune catches the rest."""
    kept = {str(item.get("render")) for item in keep if isinstance(item, dict) and item.get("render")}
    paths = []
    for draft in getattr(result, "drafts", None) or []:
        render = draft.get("render") if isinstance(draft, dict) else None
        if render and str(render) not in kept:
            paths.append(render)
    _drop_files(paths)


def _drop_files(paths) -> None:
    """Best-effort delete of draft renders nobody will be shown."""
    paths = [Path(p) for p in paths or () if p]
    if not paths:
        return
    try:
        from . import media

        media.drop(paths)
    except Exception as exc:
        log.debug("Couldn't drop draft renders: %s", exc)


def _engine_payload(result, job: _Job) -> dict:
    try:
        if callable(getattr(result, "to_dict", None)):
            data = dict(result.to_dict())          # JSON-safe: paths, non-finite floats
        elif dataclasses.is_dataclass(result):
            data = dataclasses.asdict(result)
        else:
            data = dict(vars(result))
    except Exception:
        data = {"repr": repr(result)[:2000]}
    data["job"] = {"candidate_id": job.candidate_id, "job_id": job.job_id, "boot_id": BOOT_ID,
                   "reason": job.reason, "window": job.window, "topic": job.topic,
                   "advice": job.advice, "advice_tidied": job.advice_tidied,
                   "saved_at": util.utcnow_iso()}
    return data


def _save_result(job: _Job, payload: dict, *, suffix: str = "") -> str:
    path = _runs_dir() / f"{job.candidate_id}-{job.job_id}{suffix}.json"
    try:
        util.atomic_write_json(path, payload)
        return str(path)
    except Exception as exc:
        log.warning("Couldn't save the run result for candidate %s: %s", job.candidate_id, exc)
        return ""


# ---- records and status --------------------------------------------------------

def _runs_dir() -> Path:
    return engine_dir() / "runs"


def _record_path(candidate_id: int) -> Path:
    return _runs_dir() / f"{int(candidate_id)}.json"


def _read_record(candidate_id: int) -> dict | None:
    data = util.read_json(_record_path(candidate_id))
    return data if isinstance(data, dict) else None


def _write_record(entry: dict) -> None:
    """Caller holds ``_lock``."""
    _hold_boot_lock()
    try:
        util.atomic_write_json(_record_path(entry["candidate_id"]),
                               {k: entry.get(k) for k in _FIELDS})
    except Exception as exc:
        log.warning("Couldn't write the job record for candidate %s: %s",
                    entry.get("candidate_id"), exc)


def _boot_lock_path(boot_id: str) -> Path:
    return _runs_dir() / f".boot-{boot_id}.lock"


def _hold_boot_lock() -> None:
    """Hold ``runs/.boot-<BOOT_ID>.lock`` (an exclusive ``flock``) for as long
    as this process lives, taken before its first record is written. Caller
    holds ``_lock``.

    It is how another process tells a run this one owns from one a restart
    killed (``_owner_alive``). The kernel drops the lock however the process
    ends — a reload, a crash, ``kill -9``, a reboot — and unlike a pid it
    can't pass to an unrelated process once this one is gone, which would
    leave a dead run looking alive (and its page spinning) for good. Best
    effort: without ``flock`` (another OS, an odd filesystem) this boot's
    records read as dead to other processes, as every earlier boot's did.
    """
    global _boot_lock
    path = _boot_lock_path(BOOT_ID)
    if _boot_lock is not None and _boot_lock[0] == path:
        return
    try:
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except (ImportError, OSError) as exc:
        log.debug("No clip engine liveness lock (%s)", exc)
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())   # for whoever looks
    except OSError as exc:
        os.close(fd)
        log.warning("Couldn't take the clip engine's liveness lock %s: %s", path, exc)
        return
    _release_boot_lock()   # one for an engine dir this process no longer uses (tests)
    _boot_lock = (path, fd)


def _release_boot_lock() -> None:
    global _boot_lock
    if _boot_lock is not None:
        try:
            os.close(_boot_lock[1])
        except OSError:
            pass
        _boot_lock = None


def _owner_alive(record: dict) -> bool:
    """Whether the process that wrote ``record`` is still running: this one,
    or another that still holds its boot lock (``_hold_boot_lock``). A record
    with no lock to test — written before there were locks, or where
    ``flock`` doesn't work — counts as dead, as every earlier boot's did."""
    boot = str(record.get("boot_id") or "")
    if boot == BOOT_ID:
        return True
    if not boot.isalnum():
        return False
    try:
        import fcntl

        fd = os.open(_boot_lock_path(boot), os.O_RDONLY)
    except (ImportError, OSError):
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True    # its owner still holds it
    except OSError:
        return False
    finally:
        os.close(fd)   # and with it the shared lock, if this took one
    return False


def _update(job: _Job, **fields) -> None:
    """Change this job's status in memory and in its record, unless a newer
    job has taken the video over."""
    with _lock:
        entry = _status.get(job.candidate_id)
        if entry is None or entry.get("job_id") != job.job_id:
            return
        entry.update(fields)
        _write_record(entry)


def _new_entry(candidate_id: int, job_id: str, reason: str, window: dict | None, topic: str,
               *, attempts: int, advice: str = "") -> dict:
    entry = _idle(candidate_id)
    entry.update(job_id=job_id, reason=reason, window=window, topic=topic, advice=advice,
                 state="queued", stage="Queued", attempts=attempts, queued_at=util.utcnow_iso())
    return entry


def _idle(candidate_id: int) -> dict:
    return {"candidate_id": int(candidate_id), "job_id": "", "boot_id": BOOT_ID, "reason": "",
            "setup": "", "window": None, "topic": "", "advice": "", "state": "idle", "stage": "",
            "outcome": "", "error": "", "fallback": "", "attempts": 0, "new": [],
            "cost_usd": 0.0, "notes": [], "queued_at": None, "started_at": None,
            "finished_at": None, "result_path": ""}


def _public(entry: dict) -> dict:
    out = {k: entry.get(k) for k in _FIELDS}
    out["new"] = list(out.get("new") or [])
    out["notes"] = list(out.get("notes") or [])
    out["advice"] = str(out.get("advice") or "")
    out["cost_usd"] = round(float(out.get("cost_usd") or 0.0), 4)
    out["queue_position"] = _queue_position(entry)
    out["message"] = _message(out)
    return out


def _queue_position(entry: dict) -> int:
    """How many runs this one waits behind (0 once it can start)."""
    if entry.get("state") != "queued":
        return 0
    with _lock:
        job = _queued.get(str(entry.get("job_id") or ""))
        if job is None:
            return 0
        rank = (job.priority, job.seq)
        ahead = (sum(1 for other in _queued.values() if (other.priority, other.seq) < rank)
                 + len(_running))
        return ahead if ahead >= _worker_count else 0


def _message(s: dict) -> str:
    """The sentence a page shows when a run ends, or "" when the clips speak
    for themselves (or nothing was found — the page words that per scope)."""
    state, outcome = s.get("state"), s.get("outcome")
    error = str(s.get("error") or "")
    if state == "interrupted":
        return "The suggestion run was interrupted — press Suggest again."
    if state == "failed":
        return f"Suggestion failed: {error or 'unknown error'}"
    if outcome == "empty_topic":
        return "The video model watched every draft and found nothing on topic."
    if outcome in ("budget", "failed") and error:
        return f"No suggestions: {error}."
    if outcome == "legacy":
        why = _short(s.get("fallback") or "", 160)
        # The transcript suggester takes no advice; say so, or the operator
        # reads its picks as the answer to what they asked for.
        unheard = " Your advice wasn't used." if s.get("advice") else ""
        if s.get("new"):
            return ("These came from the transcript suggester"
                    + (f" ({why})." if why else ".") + unheard)
        if why:
            # Otherwise the page says only "found nothing", and a cause the
            # operator must act on (e.g. Gemini credit ran out) goes unseen.
            return f"The transcript suggester stood in and found nothing ({why}).{unheard}"
        if unheard:
            return unheard.strip()
    return ""


def _resumable(record: dict | None) -> bool:
    if record is None:
        return True   # archived but never queued: e.g. the restart came first
    if record.get("boot_id") == BOOT_ID:
        return False  # this process already ran or queued it
    if record.get("state") not in ("queued", "running", "interrupted"):
        return False  # finished — an empty result is an answer, not an error
    if record.get("state") in ACTIVE_STATES and _owner_alive(record):
        return False  # another process that is still running has it in hand
    return _as_int(record.get("attempts")) < MAX_ATTEMPTS


# ---- inputs ----------------------------------------------------------------------

def _piece(session, label: str, notes: list[str], default, fn: Callable[[], Any]):
    try:
        with session.begin_nested():
            return fn()
    except Exception as exc:
        log.warning("Clip engine input %r unavailable: %s", label, exc)
        notes.append(f"{label} unavailable: {_short(exc)}")
        return default


def _candidate_fields(c) -> dict:
    archived = c.archived_at
    if archived is not None and archived.tzinfo is None:
        archived = archived.replace(tzinfo=dt.timezone.utc)
    return {"id": c.id, "video_id": c.video_id or "", "title": c.title or "",
            "description": c.description or "", "duration_seconds": c.duration_seconds,
            "local_video_path": c.local_video_path or "",
            "source_object_key": c.source_object_key or "",
            "transcript_path": c.transcript_path or "",
            "word_transcript_path": c.word_transcript_path or "",
            "archived_at": archived, "status": c.status or ""}


def _rejected(session, candidate_pk: int) -> list[dict]:
    """Suggestions dismissed on this video, so the engine doesn't offer the
    same moment again (the reason decides whether a re-cut may)."""
    rows = session.execute(
        select(ClipProposal.proposed_segments, ClipProposal.dismiss_reason)
        .where(ClipProposal.candidate_pk == candidate_pk,
               ClipProposal.verdict == ClipProposal.VERDICT_DISMISSED)
        .order_by(ClipProposal.id.desc()).limit(50)
    ).all()
    out = []
    for raw, reason in rows:
        try:
            parsed = json.loads(raw or "[]")
        except (TypeError, ValueError):
            continue
        segments = _plain_segments(parsed if isinstance(parsed, list) else [])
        if segments:
            out.append({"segments": segments, "reason": reason or ""})
    return out


# ---- pruning ---------------------------------------------------------------------

def _maybe_prune() -> None:
    """Prune engine caches at most once a day, across restarts (a marker
    file's age, not process memory: reload restarts happen many times a day)."""
    marker = _runs_dir() / ".pruned"
    try:
        if marker.exists() and time.time() - marker.stat().st_mtime < _PRUNE_EVERY:
            return
        util.atomic_write_text(marker, util.utcnow_iso())   # claim it before the slow part
    except OSError:
        return
    from . import media, sidecars

    for label, fn in (("media caches", media.prune), ("proposal sidecars", sidecars.prune),
                      ("run records", _prune_runs), ("boot locks", _prune_boot_locks)):
        try:
            removed = fn()
            if removed:
                log.info("Pruned %s %s", removed, label)
        except Exception as exc:
            log.warning("Pruning %s failed: %s", label, exc)


def _prune_runs(max_age_days: int = _RUN_FILES_MAX_AGE_DAYS) -> int:
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for path in _runs_dir().glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _prune_boot_locks(now: float | None = None) -> int:
    """Liveness locks whose process is gone (every dashboard start leaves
    one). Only ones over an hour old: a lock file exists a moment before its
    owner takes it, and one taken from under it would leave a live process
    looking dead."""
    try:
        import fcntl
    except ImportError:
        return 0
    now = time.time() if now is None else now
    removed = 0
    for path in _runs_dir().glob(".boot-*.lock"):
        if path == _boot_lock_path(BOOT_ID):
            continue
        try:
            if now - path.stat().st_mtime < _BOOT_LOCK_PRUNE_AGE:
                continue
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path.unlink()   # nobody held it: its process is gone
            removed += 1
        except OSError:
            continue        # held (its process is alive), or already gone
        finally:
            os.close(fd)
    return removed


# ---- small helpers -----------------------------------------------------------------

def _clean_window(window) -> dict | None:
    if not window:
        return None
    try:
        start, end = float(window["start"]), float(window["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad window {window!r}") from exc
    # Finiteness first: max(0.0, nan) is 0.0, which would hide a NaN start.
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"bad window {window!r}")
    start = max(0.0, start)
    if end <= start:
        raise ValueError(f"bad window {window!r}")
    return {"start": round(start, 2), "end": round(end, 2)}


def _clean_advice(advice) -> str:
    """One paragraph of the operator's words, whitespace collapsed, capped."""
    return " ".join(str(advice or "").split())[:MAX_ADVICE]


def _same_window(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return (abs(float(a["start"]) - float(b["start"])) <= 0.05
                and abs(float(a["end"]) - float(b["end"])) <= 0.05)
    except (KeyError, TypeError, ValueError):
        return False


def _plain_segments(segments) -> list[dict]:
    """``[{start, end}]`` in the order given, 2 dp, invalid spans dropped."""
    out = []
    for seg in segments or []:
        try:
            a, b = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b) and b > a:
            out.append({"start": round(a, 2), "end": round(b, 2)})
    return out


def _has_segments(raw: str | None) -> bool:
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, list) and bool(parsed)


def _finite(value) -> float | None:
    """A real number, or None (the critic's failure score of -1e9 included)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= -1e8:
        return None
    return number


def _positive(value) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _number_setting(settings, key: str, default: float) -> float:
    """A numeric setting; one that isn't a finite number (a hand edit like
    ``30 min``) warns and reads as ``default``. A typo in settings.yaml must
    not fail every run outright, with no transcript fallback either."""
    raw = opt(settings, key, default)
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(raw)
    except (TypeError, ValueError):
        log.warning("Setting %s=%r isn't a number; using %r", key, raw, default)
        value = float(default)
    return value


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _meter_total(meter) -> float:
    return _finite(getattr(meter, "total", 0.0)) or 0.0


def _short(exc_or_text, limit: int = 200) -> str:
    if isinstance(exc_or_text, BaseException):
        text = str(exc_or_text) or type(exc_or_text).__name__
    else:
        text = str(exc_or_text or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clean_notes(notes) -> list[str]:
    out: list[str] = []
    for note in notes or []:
        text = _short(note, 300)
        if text and text not in out:
            out.append(text)
    return out[:_MAX_NOTES]


def _reset() -> None:
    """Tests only: forget every job and stop the workers — and wait for them,
    so none is still polling when the next test queues its jobs (it would
    take one and drop it on seeing its stop flag)."""
    global _cancel, _stopping, _seq
    with _lock:
        _cancel.set()
        while True:
            try:
                _queue.get_nowait()
            except queue.Empty:
                break
        stopped = list(_workers)
        _status.clear()
        _queued.clear()
        _running.clear()
        _workers.clear()
        _cancel = threading.Event()
        _stopping = False
        _seq = itertools.count()
        _release_boot_lock()
    with _budget_lock:
        _reserved.clear()
    for worker in stopped:
        worker.join(timeout=3 * _POLL_SECONDS)
