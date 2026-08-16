"""One-shot migration: rewrite absolute local paths stored in DB rows after
the workspace move (ROOT/data -> ROOT/workspaces/<slug>/data).

Rows store absolute paths (e.g. Candidate.local_video_path), so moving the
data tree strands every one of them. This rewrites the old prefix to the
current workspace's DATA_DIR. Dry run by default; pass --apply to write.

Only the id + path columns are selected (full rows would drag every stored
transcript over the wire). Before writing, every affected (table, id, column,
old value) is saved to DATA_DIR/path_rewrite_backup.json for reversibility.

Run from the repo root, against the workspace whose rows you're fixing:
    WORKSPACE=climate python scripts/rewrite_data_paths.py            # dry run
    WORKSPACE=climate python scripts/rewrite_data_paths.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update  # noqa: E402

from app.config import DATA_DIR, ROOT, WORKSPACE  # noqa: E402
from app.db import session_scope  # noqa: E402
from app.models import Candidate, Cut, InstagramPost, ThreadsPost  # noqa: E402

# Every column that stores an absolute path on the local machine.
# (clip_object_path is a Supabase Storage key, not a local path.)
COLUMNS = [
    (Candidate, ("local_video_path", "transcript_path", "word_transcript_path")),
    (Cut, ("trimmed_clip_path", "subtitled_clip_path",
           "clip_transcript_path", "vertical_clip_path")),
    (ThreadsPost, ("clip_local_path",)),
    (InstagramPost, ("clip_local_path",)),
]

OLD_PREFIX = str(ROOT / "data")
NEW_PREFIX = str(DATA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write changes (default is a dry run)")
    args = parser.parse_args()

    print(f"workspace: {WORKSPACE}")
    print(f"rewriting prefix: {OLD_PREFIX}")
    print(f"             -> : {NEW_PREFIX}\n")

    backup: list[dict] = []
    planned = 0
    with session_scope() as session:
        for model, columns in COLUMNS:
            table = model.__tablename__
            cols = [getattr(model, c) for c in columns]
            rows = session.execute(select(model.id, *cols)).all()
            for row in rows:
                row_id = row[0]
                changes: dict[str, str] = {}
                for col_name, value in zip(columns, row[1:]):
                    value = value or ""
                    if not value.startswith(OLD_PREFIX):
                        continue
                    new_value = NEW_PREFIX + value[len(OLD_PREFIX):]
                    changes[col_name] = new_value
                    planned += 1
                    backup.append({"table": table, "id": row_id,
                                   "column": col_name, "old": value})
                    print(f"{table}.{col_name} id={row_id}\n  {value}\n  -> {new_value}")
                if changes and args.apply:
                    session.execute(
                        update(model).where(model.id == row_id).values(**changes)
                    )
        if args.apply and backup:
            backup_file = DATA_DIR / "path_rewrite_backup.json"
            backup_file.write_text(json.dumps(backup, indent=1))
            print(f"\nBacked up {len(backup)} old values to {backup_file}")

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{verb} {planned} path(s). "
          + ("" if args.apply else "Re-run with --apply to write."))


if __name__ == "__main__":
    main()
