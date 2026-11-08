"""Keep source videos (and finished clips) off the local disk once they're out
of the workflow. Clips and outro renders: see the section further down.

A downloaded source video is only needed locally while it is waiting to be
clipped (the dashboard's "To trim" bucket). After that it is uploaded to a
private Supabase bucket and the local copy is deleted; the database keeps the
object key, so the video can come back the moment it's needed again:

- playback streams straight from storage (``signed_source_url``), and
- anything that has to read the file itself (export, frame sampling,
  Whisper) calls ``ensure_local`` first, which downloads it back to the path
  it came from.

A restored file is fresh on disk, so it gets the full grace period again
before the sweep removes it; its cloud copy is kept, so that removal is just
a delete.

Rejected videos are different: nothing will be clipped from them, so their
local download is deleted without a cloud copy (un-rejecting one sends it back
to review, which re-downloads it).

The sweep runs in the dashboard process (``start_offload_thread``) and from
``python run.py offload``. It only ever acts on files present on THIS machine,
so it is harmless on a runner that has none (the Fly worker).
"""
from __future__ import annotations

import datetime as dt
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select

from .config import WORKSPACE, load_settings, storage_dir
from .models import (
    STATUS_APPROVED, STATUS_ARCHIVED, STATUS_FAILED, STATUS_REJECTED,
    Candidate, Cut, ThreadsPost, utcnow,
)

log = logging.getLogger("offload")

# Post statuses that take a video out of "To trim" (it was clipped and the clip
# is published or on its way). Mirrors the dashboard's ``_workflow_buckets``.
HANDLED_POST_STATUSES = frozenset({"published", "queued", "publishing", "draft"})

# Free space to leave on disk after a restore, on top of the file itself.
_RESTORE_HEADROOM_BYTES = 1024 ** 3


class SourceUnavailable(RuntimeError):
    """The source video is neither on disk nor restorable from storage."""


# --- Which videos can leave the disk ---------------------------------------------

def in_to_trim(status: str, multi_clip: bool, post_statuses: set[str],
               has_exported_cut: bool) -> bool:
    """Whether the dashboard would show this video under "To trim".

    Same rules as ``app.web.main._workflow_buckets``, except that the
    multi-clip marker always wins: the operator flagged the video for more
    clips, so it stays local even after one of its posts failed.
    """
    if status not in (STATUS_APPROVED, STATUS_ARCHIVED, STATUS_FAILED):
        return False
    if multi_clip:
        return True
    if "failed" in post_statuses:
        return False  # already clipped; shown under "Trimmed" for a retry
    if post_statuses & HANDLED_POST_STATUSES:
        return False
    # Archived with an exported cut = clipped, waiting to be posted.
    return not (status == STATUS_ARCHIVED and has_exported_cut)


@dataclass
class SourceRow:
    candidate_id: int
    video_id: str
    status: str
    local_path: str
    object_key: str
    multi_clip: bool
    post_statuses: set[str] = field(default_factory=set)
    has_exported_cut: bool = False
    exporting: bool = False
    # Latest operator activity on the video (cut edits, posts).
    last_activity: dt.datetime | None = None


# Decision verbs returned by ``decide``.
KEEP, OFFLOAD, DELETE = "keep", "offload", "delete"


def decide(row: SourceRow, *, file_mtime: dt.datetime, now: dt.datetime,
           grace: dt.timedelta) -> tuple[str, str]:
    """``(verb, reason)`` for one video whose file is on disk."""
    latest = max(t for t in (file_mtime, row.last_activity) if t is not None)
    settled = now - latest >= grace
    if row.status == STATUS_REJECTED:
        return (DELETE, "rejected") if settled else (KEEP, "rejected recently")
    if row.status != STATUS_ARCHIVED:
        return KEEP, f"status {row.status}"  # downloading, failed, or back in review
    if row.exporting:
        return KEEP, "export in progress"
    if in_to_trim(row.status, row.multi_clip, row.post_statuses, row.has_exported_cut):
        return KEEP, "waiting to be clipped"
    if not settled:
        return KEEP, "touched recently"
    return OFFLOAD, "out of the workflow"


def _load_rows(session) -> list[SourceRow]:
    """Every candidate that still names a local file, with what ``decide``
    needs, in three queries."""
    cands = session.execute(
        select(Candidate.id, Candidate.video_id, Candidate.status,
               Candidate.local_video_path, Candidate.source_object_key,
               Candidate.multi_clip_potential)
        .where(Candidate.local_video_path != "")
    ).all()
    rows = {c[0]: SourceRow(candidate_id=c[0], video_id=c[1] or "", status=c[2] or "",
                            local_path=c[3], object_key=c[4] or "",
                            multi_clip=bool(c[5])) for c in cands}
    if not rows:
        return []

    def _bump(row: SourceRow, t) -> None:
        if t is None:
            return
        if t.tzinfo is None:  # SQLite hands back naive UTC
            t = t.replace(tzinfo=dt.timezone.utc)
        if row.last_activity is None or t > row.last_activity:
            row.last_activity = t

    for pk, status, created in session.execute(
        select(ThreadsPost.candidate_pk, ThreadsPost.status, ThreadsPost.created_at)
        .where(ThreadsPost.candidate_pk.is_not(None))
    ).all():
        row = rows.get(pk)
        if row is not None:
            row.post_statuses.add(status)
            _bump(row, created)

    for pk, exported, updated in session.execute(
        select(Cut.candidate_pk,
               func.max(func.length(Cut.trimmed_clip_path)),
               func.max(Cut.updated_at))
        .group_by(Cut.candidate_pk)
    ).all():
        row = rows.get(pk)
        if row is None:
            continue
        row.has_exported_cut = bool(exported)
        _bump(row, updated)
    for (pk,) in session.execute(
        select(Cut.candidate_pk).where(Cut.export_status == "exporting")
    ).all():
        if pk in rows:
            rows[pk].exporting = True
    return list(rows.values())


# --- Storage plumbing ------------------------------------------------------------

def _bucket() -> str:
    return str(load_settings().get("storage.source_video_bucket", "source-videos"))


def object_key_for(local_path: str | Path, video_id: str) -> str:
    """Storage key mirroring the file's place under the download dir, prefixed
    by workspace so several workspaces can share one bucket."""
    path = Path(local_path)
    root = storage_dir(load_settings(), "storage.download_dir", "data/videos")
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = f"other/{video_id or path.stem}{path.suffix}"
    return f"{WORKSPACE}/{rel}"


def remote_size(object_key: str) -> int | None:
    """Byte size of the stored copy, or None when there is none."""
    from .storage_supabase import _client

    try:
        info = _client().storage.from_(_bucket()).info(object_key)
    except Exception as exc:
        if "not_found" in str(exc) or "404" in str(exc):
            return None
        raise
    size = info.get("size") if isinstance(info, dict) else None
    return int(size) if size is not None else None


def signed_source_url(object_key: str) -> str:
    """Time-limited URL the browser can stream the source video from."""
    from .storage_supabase import _client

    ttl = int(load_settings().get("storage.source_signed_url_ttl_seconds", 6 * 3600))
    signed = _client().storage.from_(_bucket()).create_signed_url(object_key, ttl)
    url = signed.get("signedURL") or signed.get("signedUrl") or ""
    if not url:
        raise SourceUnavailable(f"Could not sign {object_key}: {signed}")
    return url


_locks_guard = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _path_locks.setdefault(path, threading.Lock())


def ensure_local(local_path: str, object_key: str) -> Path:
    """The source video's local path, downloading it back from storage first
    when it was offloaded. Raises ``SourceUnavailable`` when it can't be had."""
    path = Path(local_path or "")
    if local_path and path.is_file() and path.stat().st_size > 0:
        return path
    if not local_path or not object_key:
        raise SourceUnavailable("The source video isn't on disk and has no cloud copy.")
    with _lock_for(str(path)):
        if path.is_file() and path.stat().st_size > 0:
            return path  # another thread restored it while we waited
        from .storage_supabase import download_to_file

        size = remote_size(object_key)
        if size is None:
            raise SourceUnavailable(
                f"The source video isn't on disk and its cloud copy ({object_key}) is gone.")
        path.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path.parent).free
        if free < size + _RESTORE_HEADROOM_BYTES:
            raise SourceUnavailable(
                f"Not enough disk space to bring the source video back "
                f"({size / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free).")
        started = time.monotonic()
        download_to_file(_bucket(), object_key, path, expected_size=size)
        log.info("Restored %s from storage (%.1f MB in %.0fs)", path.name,
                 size / 1e6, time.monotonic() - started)
    return path


_background_restores: set[str] = set()


def restore_in_background(local_path: str, object_key: str) -> bool:
    """Start bringing an offloaded source back, if it isn't already coming.
    Returns True when a restore is (now) in flight."""
    path = Path(local_path or "")
    if not object_key or not local_path or path.is_file():
        return False
    with _locks_guard:
        if local_path in _background_restores:
            return True
        _background_restores.add(local_path)

    def _run() -> None:
        try:
            ensure_local(local_path, object_key)
        except Exception as exc:
            log.warning("Background restore of %s failed: %s", path.name, exc)
        finally:
            with _locks_guard:
                _background_restores.discard(local_path)

    threading.Thread(target=_run, daemon=True, name="restore-source").start()
    return True


def is_restoring(local_path: str) -> bool:
    with _locks_guard:
        return local_path in _background_restores


# --- The sweep ---------------------------------------------------------------------

@dataclass
class SweepResult:
    offloaded: int = 0
    offloaded_bytes: int = 0
    deleted: int = 0
    deleted_bytes: int = 0
    kept: int = 0
    failed: int = 0
    planned: list[tuple[str, int, str]] = field(default_factory=list)  # (verb, bytes, path)


def _precache(row: SourceRow, path: Path) -> None:
    """Build the per-video caches the UI reads from the source, so cards and the
    trim editor keep working without bringing the file back."""
    from .clipper import cached_still, get_waveform

    for what, fn in (("still", lambda: cached_still(path, str(row.candidate_id))),
                     ("waveform", lambda: get_waveform(path, row.video_id))):
        try:
            fn()
        except Exception as exc:
            log.info("No %s cached for %s before offload: %s", what, row.video_id, exc)


def _offload_one(row: SourceRow, path: Path, size: int) -> None:
    from .db import session_scope
    from .storage_supabase import ensure_bucket, upload_file_resumable

    key = row.object_key or object_key_for(path, row.video_id)
    bucket = _bucket()
    if remote_size(key) != size:
        ensure_bucket(bucket)
        started = time.monotonic()
        upload_file_resumable(path, bucket, key)
        log.info("Uploaded %s (%.1f MB in %.0fs)", path.name, size / 1e6,
                 time.monotonic() - started)
    stored = remote_size(key)
    if stored != size:
        raise RuntimeError(f"stored copy of {path.name} is {stored} bytes, expected {size}")
    # Record the cloud copy BEFORE deleting: a crash in between then leaves a
    # file that is merely still on disk, never a video with no copy anywhere.
    with session_scope() as session:
        cand = session.get(Candidate, row.candidate_id)
        if cand is None or cand.local_video_path != row.local_path:
            raise RuntimeError("candidate changed during the upload; leaving the file")
        cand.source_object_key = key
        cand.source_uploaded_at = utcnow()
    row.object_key = key
    path.unlink()


def sweep(*, dry_run: bool = False, limit: int | None = None) -> SweepResult:
    """Offload or delete every eligible local source video (see ``decide``).

    Each file is handled on its own: one failure is logged and the sweep moves
    on. ``limit`` caps how many files are acted on (for a first cautious run).
    """
    from .db import session_scope

    settings = load_settings()
    grace = dt.timedelta(days=float(settings.get("storage.offload_after_days", 3)))
    result = SweepResult()
    with session_scope(read_only=True) as session:
        rows = _load_rows(session)
    now = utcnow()
    acted = 0
    seen: set[str] = set()
    for row in rows:
        path = Path(row.local_path)
        try:
            resolved = str(path.resolve())
            if resolved in seen:
                continue  # two rows naming one file: decide on the first only
            seen.add(resolved)
            st = path.stat()
        except OSError:
            continue  # already off disk
        if st.st_size == 0:
            continue  # a broken download; leave it for the operator to retry
        mtime = dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
        verb, reason = decide(row, file_mtime=mtime, now=now, grace=grace)
        if verb == KEEP:
            result.kept += 1
            continue
        if limit is not None and acted >= limit:
            break
        acted += 1
        result.planned.append((verb, st.st_size, str(path)))
        if dry_run:
            continue
        try:
            if verb == DELETE:
                path.unlink()
                result.deleted += 1
                result.deleted_bytes += st.st_size
                log.info("Deleted rejected video %s (%.1f MB)", path.name, st.st_size / 1e6)
            else:
                _precache(row, path)
                _offload_one(row, path, st.st_size)
                result.offloaded += 1
                result.offloaded_bytes += st.st_size
        except Exception as exc:
            result.failed += 1
            log.warning("Could not %s %s: %s", verb, path.name, exc)
    return result


# --- Exported clips and outro renders ------------------------------------------------
#
# A cut's exported files (plain, captioned, vertical) are needed locally while
# the cut is still headed for a post. Once every post made from it has
# published, the published copies live in the trimmed-clips bucket and the local
# files only serve the dashboard (playing the clip, redoing captions or the
# vertical version). They are then archived to the source-video bucket under
# ``<workspace>/clips/<filename>`` (filenames are timestamped, so unique) and
# deleted; ``ensure_clip_file`` brings one back by its path alone, so nothing in
# the database has to change. Outro renders are a pure cache of what was
# uploaded, so they are deleted once nothing waiting to publish uses them, as
# are clip files no row refers to any more.

# Post statuses after which a post never reads its local clip again.
FINISHED_POST_STATUSES = frozenset({"published"})
ARCHIVE, DELETE_ORPHAN, DELETE_OUTRO = "archive", "delete-orphan", "delete-outro"


def clip_object_key(path: str | Path) -> str:
    return f"{WORKSPACE}/clips/{Path(path).name}"


def _clips_dir() -> Path:
    from .clipper import CLIPS_DIR

    return CLIPS_DIR


def _as_utc(t: dt.datetime | None) -> dt.datetime | None:
    if t is not None and t.tzinfo is None:
        return t.replace(tzinfo=dt.timezone.utc)  # SQLite hands back naive UTC
    return t


@dataclass
class ClipPlan:
    """What the clip sweep found: per resolved file path, whether every row
    that refers to it is finished; plus the cut each file belongs to."""
    finished: dict[str, bool] = field(default_factory=dict)
    cut_of: dict[str, int] = field(default_factory=dict)
    # Stems of files something still waiting to publish uses (their outro
    # renders stay).
    pending_stems: set[str] = field(default_factory=set)


def plan_clips(session, *, now: dt.datetime, grace: dt.timedelta) -> ClipPlan:
    from .models import InstagramPost

    cut_posts: dict[int, list[tuple[str, dt.datetime | None]]] = {}
    post_refs: list[tuple[int | None, str, str]] = []  # (cut_pk, status, path)
    for M in (ThreadsPost, InstagramPost):
        for cut_pk, status, path, created in session.execute(
            select(M.cut_pk, M.status, M.clip_local_path, M.created_at)
        ).all():
            if cut_pk is not None:
                cut_posts.setdefault(cut_pk, []).append((status, _as_utc(created)))
            if path:
                post_refs.append((cut_pk, status, path))

    plan = ClipPlan()
    cut_done: dict[int, bool] = {}
    for cid, trimmed, subbed, vertical, sidecar, exp_status, updated in session.execute(
        select(Cut.id, Cut.trimmed_clip_path, Cut.subtitled_clip_path,
               Cut.vertical_clip_path, Cut.clip_transcript_path, Cut.export_status,
               Cut.updated_at)
    ).all():
        posts = cut_posts.get(cid, [])
        latest = max([t for t in [_as_utc(updated), *(c for _, c in posts)] if t],
                     default=None)
        done = (any(s in FINISHED_POST_STATUSES for s, _ in posts)
                and all(s in FINISHED_POST_STATUSES for s, _ in posts)
                and exp_status != "exporting"
                and (latest is None or now - latest >= grace))
        cut_done[cid] = done
        # The sidecar is referenced (so never an orphan) but, not being a
        # video, never archived: ``sweep_clips`` only archives .mp4 files.
        for p in (trimmed, subbed, vertical, sidecar):
            if p:
                _ref(plan, p, done, cid)
    # Belt and braces: any file a candidate names is never an orphan here.
    for row in session.execute(
        select(Candidate.local_video_path, Candidate.transcript_path,
               Candidate.word_transcript_path)
    ).all():
        for p in row:
            if p:
                _ref(plan, p, False, None)
    for cut_pk, status, path in post_refs:
        done = status in FINISHED_POST_STATUSES and (cut_pk is None
                                                    or cut_done.get(cut_pk, True))
        _ref(plan, path, done, cut_pk)
    return plan


def _ref(plan: ClipPlan, path: str, done: bool, cut_pk: int | None) -> None:
    key = str(Path(path).expanduser().resolve())
    plan.finished[key] = plan.finished.get(key, True) and done
    if cut_pk is not None:
        plan.cut_of.setdefault(key, cut_pk)
    if not done:
        plan.pending_stems.add(Path(key).stem)


def _precache_clip(path: Path, cut_pk: int | None) -> None:
    """Cache the clip card's still (served by /media/cut-thumb) before the
    clip leaves the disk."""
    if cut_pk is None:
        return
    from .clipper import cached_still

    try:
        cached_still(path, f"cut-{cut_pk}")
    except Exception as exc:
        log.info("No still cached for cut %s before archiving: %s", cut_pk, exc)


def _archive_clip(path: Path, size: int) -> None:
    from .storage_supabase import ensure_bucket, upload_file_resumable

    key = clip_object_key(path)
    if remote_size(key) != size:
        ensure_bucket(_bucket())
        upload_file_resumable(path, _bucket(), key)
    stored = remote_size(key)
    if stored != size:
        raise RuntimeError(f"stored copy of {path.name} is {stored} bytes, expected {size}")
    path.unlink()


def sweep_clips(*, dry_run: bool = False, limit: int | None = None) -> SweepResult:
    """Archive finished cuts' clips, delete unreferenced clip files and outro
    renders nothing pending needs. Same per-file failure handling as ``sweep``.
    Only video files are archived; the small transcript sidecars stay put."""
    from .db import session_scope
    from .outro import OUTRO_DIR

    settings = load_settings()
    grace = dt.timedelta(days=float(settings.get("storage.offload_after_days", 3)))
    now = utcnow()
    with session_scope(read_only=True) as session:
        plan = plan_clips(session, now=now, grace=grace)

    work: list[tuple[str, Path, int, int | None]] = []
    result = SweepResult()
    for directory, is_outro in ((_clips_dir(), False), (OUTRO_DIR, True)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            try:
                st = path.stat()
            except OSError:
                continue
            if not path.is_file() or path.name.startswith("."):
                continue
            settled = now - dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc) >= grace
            if not settled:
                result.kept += 1
                continue
            if is_outro:
                stem = path.name.split("_outro_", 1)[0]
                if stem in plan.pending_stems:
                    result.kept += 1
                else:
                    work.append((DELETE_OUTRO, path, st.st_size, None))
                continue
            key = str(path.resolve())
            if key not in plan.finished:
                work.append((DELETE_ORPHAN, path, st.st_size, None))
            elif plan.finished[key] and path.suffix.lower() == ".mp4":
                work.append((ARCHIVE, path, st.st_size, plan.cut_of.get(key)))
            else:
                result.kept += 1

    for i, (verb, path, size, cut_pk) in enumerate(work):
        if limit is not None and i >= limit:
            break
        result.planned.append((verb, size, str(path)))
        if dry_run:
            continue
        try:
            if verb == ARCHIVE:
                _precache_clip(path, cut_pk)
                _archive_clip(path, size)
                result.offloaded += 1
                result.offloaded_bytes += size
            else:
                path.unlink()
                result.deleted += 1
                result.deleted_bytes += size
        except Exception as exc:
            result.failed += 1
            log.warning("Could not %s %s: %s", verb, path.name, exc)
    return result


_clip_misses: dict[str, float] = {}
_CLIP_MISS_TTL = 300.0


def ensure_clip_file(path: str | Path) -> bool:
    """Bring an archived clip back to ``path`` if it's missing. True when the
    file is (now) on disk. Never raises: a clip that can't be restored leaves
    the caller's own missing-file handling in charge."""
    p = Path(path or "").expanduser()
    if not str(path or "") or p.is_file():
        return bool(str(path or ""))
    if p.parent.resolve() != _clips_dir().resolve():
        return False
    now = time.monotonic()
    if now - _clip_misses.get(str(p), -1e9) < _CLIP_MISS_TTL:
        return False  # checked recently: not in the archive either
    key = clip_object_key(p)
    try:
        with _lock_for(str(p)):
            if p.is_file():
                return True
            size = remote_size(key)
            if size is None:
                _clip_misses[str(p)] = now
                return False
            from .storage_supabase import download_to_file

            download_to_file(_bucket(), key, p, expected_size=size)
            log.info("Restored clip %s from storage", p.name)
            return True
    except Exception as exc:
        log.warning("Could not restore clip %s: %s", p.name, exc)
        _clip_misses[str(p)] = now
        return False


_sweep_lock = threading.Lock()
_thread: threading.Thread | None = None


def run_sweep_once(*, clips: bool = True, **kwargs) -> SweepResult | None:
    """Source sweep then clip sweep, unless one is already running in this
    process. The result merges both."""
    if not _sweep_lock.acquire(blocking=False):
        return None
    try:
        res = sweep(**kwargs)
        if clips:
            more = sweep_clips(**kwargs)
            res.offloaded += more.offloaded
            res.offloaded_bytes += more.offloaded_bytes
            res.deleted += more.deleted
            res.deleted_bytes += more.deleted_bytes
            res.kept += more.kept
            res.failed += more.failed
            res.planned += more.planned
        return res
    finally:
        _sweep_lock.release()


def start_offload_thread() -> None:
    """Background sweep for the dashboard process. Idempotent; a no-op unless
    ``storage.offload_sources`` is on."""
    global _thread
    if _thread and _thread.is_alive():
        return
    if not load_settings().get("storage.offload_sources", False):
        return

    def _loop() -> None:
        time.sleep(120)  # let the dashboard finish booting first
        while True:
            settings = load_settings()
            if settings.get("storage.offload_sources", False):
                try:
                    res = run_sweep_once()
                    if res and (res.offloaded or res.deleted or res.failed):
                        log.info("Offload sweep: %d uploaded (%.1f GB), %d deleted "
                                 "(%.1f GB), %d failed",
                                 res.offloaded, res.offloaded_bytes / 1e9,
                                 res.deleted, res.deleted_bytes / 1e9, res.failed)
                except Exception:
                    log.exception("Offload sweep failed")
            minutes = float(settings.get("storage.offload_interval_minutes", 60))
            time.sleep(max(60.0, minutes * 60))

    _thread = threading.Thread(target=_loop, daemon=True, name="offload-sources")
    _thread.start()


def free_bytes() -> int:
    """Free space on the volume holding the downloads (for CLI reporting)."""
    root = storage_dir(load_settings(), "storage.download_dir", "data/videos")
    return shutil.disk_usage(root if root.exists() else root.parent).free


__all__ = [
    "SourceUnavailable", "ensure_local", "restore_in_background", "is_restoring",
    "signed_source_url", "sweep", "run_sweep_once", "start_offload_thread",
    "in_to_trim", "decide", "object_key_for", "free_bytes",
    "sweep_clips", "plan_clips", "ensure_clip_file", "clip_object_key",
]
