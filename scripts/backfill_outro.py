"""One-shot backfill: give already-published clips the post-roll outro (end
card + Renewables.org tab), so re-airs and promo re-runs carry it.

A re-air clones the cut's latest published airing and reuses its storage
object, and the headless scheduler on Fly can't render the outro itself. So
the fix is upstream: re-upload that object with the outro, under the same key.

An object counts as already done when it runs more than 2 s longer than its
local source (the outro adds duration - dissolve, about 2.8 s). Rows whose
local file isn't on this machine are reported and skipped. Dry run by
default; pass --apply to upload.

Run from the repo root:
    WORKSPACE=climate .venv/bin/python scripts/backfill_outro.py            # dry run
    WORKSPACE=climate .venv/bin/python scripts/backfill_outro.py --apply
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import WORKSPACE  # noqa: E402
from app.db import session_scope  # noqa: E402
from app.models import ThreadsPost  # noqa: E402
from app.outro import enabled  # noqa: E402
from app.storage_supabase import signed_clip_url, upload_trimmed_clip  # noqa: E402

MIN_ADDED_SECONDS = 2.0


def _duration(src: str) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", src],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def _targets() -> list[tuple[int, int, str, str]]:
    """(post id, cut id, local path, object key) for each cut's latest
    published airing, plus every queued post."""
    with session_scope() as session:
        published = session.execute(
            select(ThreadsPost.id, ThreadsPost.cut_pk, ThreadsPost.clip_local_path,
                   ThreadsPost.clip_object_path)
            .where(ThreadsPost.status == "published", ThreadsPost.cut_pk.is_not(None))
            .order_by(ThreadsPost.published_at.desc().nullslast())
        ).all()
        queued = session.execute(
            select(ThreadsPost.id, ThreadsPost.cut_pk, ThreadsPost.clip_local_path,
                   ThreadsPost.clip_object_path)
            .where(ThreadsPost.status == "queued")
        ).all()
    seen: set[int] = set()
    rows = []
    for row in published:
        if row.cut_pk in seen:
            continue
        seen.add(row.cut_pk)
        rows.append(row)
    rows.extend(queued)
    # Several airings can share one object; check each object once.
    by_key: dict[str, tuple] = {}
    for r in rows:
        if r.clip_object_path and r.clip_object_path not in by_key:
            by_key[r.clip_object_path] = (r.id, r.cut_pk, r.clip_local_path or "",
                                          r.clip_object_path)
    return list(by_key.values())


def _check(target: tuple[int, int, str, str]) -> tuple[str, tuple]:
    post_id, cut_id, local, key = target
    path = Path(local).expanduser()
    if not local or not path.exists():
        return "no-local", target
    src = _duration(str(path))
    try:
        uploaded = _duration(signed_clip_url(key))
    except Exception:
        uploaded = None
    if src is None:
        return "no-local", target
    if uploaded is not None and uploaded - src > MIN_ADDED_SECONDS:
        return "done", target
    return "needs", target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="upload (default is a dry run)")
    args = parser.parse_args()

    if not enabled():
        sys.exit(f"The outro is disabled for workspace {WORKSPACE!r}; nothing to do.")

    targets = _targets()
    print(f"workspace: {WORKSPACE} — checking {len(targets)} storage objects")
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(_check, targets))
    needs = [t for s, t in results if s == "needs"]
    missing = [t for s, t in results if s == "no-local"]
    done = sum(1 for s, _ in results if s == "done")
    print(f"already have the outro: {done}")
    print(f"need it: {len(needs)}")
    if missing:
        print(f"skipped, local file not on this machine: {len(missing)} "
              f"(posts {', '.join(str(t[0]) for t in missing)})")

    if not args.apply:
        print("\nDry run. Re-run with --apply to upload.")
        return

    failed = 0
    for i, (post_id, cut_id, local, key) in enumerate(needs, 1):
        try:
            upload_trimmed_clip(Path(local).expanduser(), key)
            print(f"[{i}/{len(needs)}] post {post_id} (cut {cut_id}) uploaded")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(needs)}] post {post_id} (cut {cut_id}) FAILED: {exc}")
    print(f"\nuploaded {len(needs) - failed}, failed {failed}")


if __name__ == "__main__":
    main()
