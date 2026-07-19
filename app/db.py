"""Database engine/session setup and channel-seed sync."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from .config import database_url, load_channel_seed
from .models import Base, Channel

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


def _ensure_new_columns() -> None:
    """Lightweight additive migration for columns added after first release."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    tables = {
        "candidates": {
            "trim_segments": "TEXT DEFAULT ''",
            "trimmed_clip_path": "TEXT DEFAULT ''",
        },
        "threads_posts": {
            "source": "VARCHAR(20) DEFAULT 'app'",
        },
    }
    with engine.begin() as conn:
        for table, additions in tables.items():
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in additions.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


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
    session.flush()
    return added
