"""One-off data migration: copy every row from the local SQLite database into
whatever DATABASE_URL currently points at (e.g. Supabase Postgres).

Run this BEFORE launching the dashboard against the new database, so channel
IDs are preserved and candidate/post foreign keys stay intact:

    # .env has DATABASE_URL=postgresql+psycopg2://...supabase...
    python run.py migrate-db
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, func, insert, select, text
from sqlalchemy.orm import Session

from .config import ROOT, database_url
from .db import engine as dest_engine, init_db
from .models import Candidate, Channel, MetricSnapshot, ThreadsComment, ThreadsPost

log = logging.getLogger("migrate")

# Parent-before-child so foreign keys resolve on insert.
ORDER = [Channel, Candidate, ThreadsPost, ThreadsComment, MetricSnapshot]


def migrate_sqlite_to_target(sqlite_path: str | None = None) -> dict:
    """Copy all rows from the local SQLite DB into the configured target DB.

    Skips any table that already has rows in the target (so it's safe to re-run
    and won't create duplicates). Resets Postgres identity sequences at the end.
    """
    target = database_url()
    if target.startswith("sqlite"):
        raise SystemExit(
            "DATABASE_URL still points at SQLite. Set it to your Supabase Postgres "
            "connection string (postgresql+psycopg2://...) before migrating."
        )

    src_path = sqlite_path or str(ROOT / "data" / "app.db")
    if not Path(src_path).exists():
        raise SystemExit(f"Source SQLite DB not found at {src_path}")
    src_engine = create_engine(f"sqlite:///{src_path}", future=True)

    # Ensure the destination schema exists (tables + additive columns).
    init_db()

    counts: dict[str, int] = {}
    with Session(src_engine) as src, Session(dest_engine) as dst:
        for Model in ORDER:
            table = Model.__tablename__
            existing = dst.execute(select(func.count()).select_from(Model)).scalar_one()
            if existing:
                log.warning("%s already has %d rows in target; skipping", table, existing)
                counts[table] = 0
                continue
            rows = src.execute(select(Model)).scalars().all()
            payload = [
                {col.name: getattr(r, col.name) for col in Model.__table__.columns}
                for r in rows
            ]
            if payload:
                dst.execute(insert(Model.__table__), payload)
                dst.commit()
            counts[table] = len(payload)
            log.info("Copied %d rows into %s", len(payload), table)

        # Keep future auto-increment ids from colliding with the copied ids.
        if dest_engine.dialect.name == "postgresql":
            for Model in ORDER:
                table = Model.__tablename__
                dst.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
                ))
            dst.commit()

    return counts
