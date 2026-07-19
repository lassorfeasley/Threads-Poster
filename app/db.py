"""Database engine/session setup and channel-seed sync."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from .config import database_url, load_channel_seed, load_settings
from .models import Base, Channel, Trait

_url = database_url()
_is_sqlite = _url.startswith("sqlite")

engine = create_engine(
    _url,
    future=True,
    connect_args={"timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        # WAL lets readers and one writer coexist; busy_timeout retries instead
        # of instantly raising "database is locked" under brief contention.
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _ensure_new_columns()
    _drop_removed_columns()
    _migrate_scheduled_to_queued()


def _ensure_new_columns() -> None:
    """Lightweight additive migration for columns added after first release."""
    from sqlalchemy import inspect, text

    bool_default = "FALSE" if not _is_sqlite else "0"
    tables = {
        "candidates": {
            "trim_segments": "TEXT DEFAULT ''",
            "trimmed_clip_path": "TEXT DEFAULT ''",
            "visual_score": "FLOAT",
            "visual_traits": "TEXT DEFAULT ''",
            "visual_rationale": "TEXT DEFAULT ''",
            "visual_scored_at": "TIMESTAMP",
        },
        "threads_posts": {
            "source": "VARCHAR(20) DEFAULT 'app'",
            "scheduled_at": "TIMESTAMP WITH TIME ZONE",
            "is_breaking": f"BOOLEAN DEFAULT {bool_default}",
            "defer_count": "INTEGER DEFAULT 0",
            "last_deferred_at": "TIMESTAMP WITH TIME ZONE",
            "pinned_window_key": "VARCHAR(40) DEFAULT ''",
        },
        "channels": {
            "country": "VARCHAR(60) DEFAULT ''",
            "scope": "VARCHAR(20) DEFAULT 'local'",
        },
    }
    added: list[tuple[str, str]] = []
    with engine.begin() as conn:
        insp = inspect(conn)
        for table, additions in tables.items():
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in additions.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    added.append((table, name))
                    existing.add(name)
        # One-time backfill: the original seed is all US local stations, so tag
        # pre-existing rows as United States when the country column is new.
        if ("channels", "country") in added:
            conn.execute(text(
                "UPDATE channels SET country='United States' WHERE country='' OR country IS NULL"
            ))


def _migrate_scheduled_to_queued() -> None:
    """One-time: move exact-time ``scheduled`` posts into the adaptive queue."""
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE threads_posts SET status='queued', scheduled_at=NULL "
            "WHERE status='scheduled'"
        ))


def _drop_removed_columns() -> None:
    """Drop columns retired after first release (best-effort, idempotent).

    SQLite supports ``ALTER TABLE ... DROP COLUMN`` since 3.35 and Postgres
    supports it natively; on older/unsupported engines the drop is skipped so a
    stale column simply lingers unused rather than breaking startup.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    removed = {
        "candidates": ["climate_topic"],
    }
    with engine.begin() as conn:
        for table, columns in removed.items():
            existing = {c["name"] for c in insp.get_columns(table)}
            for name in columns:
                if name not in existing:
                    continue
                try:
                    conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {name}"))
                except Exception:  # unsupported engine/version: leave column in place
                    pass


@contextmanager
def session_scope():
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def sync_channels_from_config(session: Session) -> int:
    """Upsert channels from config/channels.yaml into the DB (keyed by URL).

    Channels added via the dashboard live only in the DB; this never deletes
    rows, so dashboard-added channels survive a re-sync.
    """
    added = 0
    for entry in load_channel_seed():
        url = str(entry.get("url", "")).strip()
        if not url:
            continue
        existing = session.execute(select(Channel).where(Channel.url == url)).scalar_one_or_none()
        if existing is None:
            session.add(
                Channel(
                    call_sign=str(entry.get("call_sign", "")),
                    network=str(entry.get("network", "")),
                    market=str(entry.get("market", "")),
                    region=str(entry.get("region", "")),
                    country=str(entry.get("country") or "United States"),
                    scope=str(entry.get("scope") or "local"),
                    url=url,
                    channel_id=entry.get("channel_id") or None,
                    enabled=bool(entry.get("enabled", True)),
                )
            )
            added += 1
        else:
            # Refresh descriptive fields from config; keep runtime state.
            existing.call_sign = str(entry.get("call_sign", existing.call_sign))
            existing.network = str(entry.get("network", existing.network))
            existing.market = str(entry.get("market", existing.market))
            existing.region = str(entry.get("region", existing.region))
            # Only overwrite country/scope when the config actually specifies
            # them, so the migration backfill / dashboard edits aren't clobbered.
            if entry.get("country"):
                existing.country = str(entry["country"])
            if entry.get("scope"):
                existing.scope = str(entry["scope"])
    session.flush()
    return added


def sync_traits_from_config(session: Session) -> int:
    """Seed the trait database from settings (``vision.desirable_traits`` /
    ``vision.undesirable_traits``) on first run. Only adds missing names; never
    deletes or overwrites, so Traits-page edits survive a re-sync.
    """
    settings = load_settings()
    seed = [
        (settings.get("vision.desirable_traits") or settings.get("vision.traits") or [],
         Trait.KIND_DESIRABLE),
        (settings.get("vision.undesirable_traits") or [], Trait.KIND_UNDESIRABLE),
    ]
    existing = {t.name for t in session.execute(select(Trait)).scalars().all()}
    added = 0
    for names, kind in seed:
        for raw in names:
            name = str(raw).strip()
            if not name or name in existing:
                continue
            session.add(Trait(name=name, kind=kind, enabled=True))
            existing.add(name)
            added += 1
    session.flush()
    return added


def active_traits(session: Session) -> tuple[list[str], list[str]]:
    """Return (desirable_names, undesirable_names) for all enabled traits."""
    rows = session.execute(select(Trait).where(Trait.enabled)).scalars().all()
    desirable = [t.name for t in rows if t.kind == Trait.KIND_DESIRABLE]
    undesirable = [t.name for t in rows if t.kind == Trait.KIND_UNDESIRABLE]
    return desirable, undesirable
