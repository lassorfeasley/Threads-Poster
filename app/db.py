"""Database engine/session setup and channel-seed sync."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from .config import database_url, load_channel_seed, load_settings
from .models import Base, Channel, Trait

_url = database_url()
_is_sqlite = _url.startswith("sqlite")

# Bump when additive migrations in ``_ensure_new_columns`` /
# ``_drop_removed_columns`` / ``_migrate_scheduled_to_queued`` change.
# Stored in ``app_tokens`` so remote Postgres startups skip the expensive
# inspection round trips after the first successful migrate.
SCHEMA_VERSION = "5"
_SCHEMA_TOKEN_NAME = "_schema_version"

_engine_kwargs: dict = {"future": True}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"timeout": 30}
else:
    # Supabase/Postgres: avoid 1–2s reconnects mid-session and drop stale
    # pooled connections after idle (Supabase closes them ~30 min).
    _engine_kwargs.update(
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=1800,
    )

engine = create_engine(_url, **_engine_kwargs)

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
    # Fast path: schema already migrated at SCHEMA_VERSION — skip create_all
    # inspection and additive migrations (each is a remote round trip).
    if _schema_is_current():
        return
    Base.metadata.create_all(engine)
    _ensure_new_columns()
    _drop_removed_columns()
    _migrate_scheduled_to_queued()
    _set_schema_version()


# Cheap probes that must succeed for the schema to really be at SCHEMA_VERSION.
# Update alongside every version bump. Guards against a stale/premature stamp
# (e.g. a dev-reload importing db.py between two source edits once wrote the
# new version while the column list was still old — the stamp then blocked the
# migration forever while queries crashed on missing columns).
_SCHEMA_SENTINELS = (
    "SELECT suggested_caption FROM threads_posts LIMIT 0",
    "SELECT status FROM trait_weights LIMIT 0",
)


def _schema_is_current() -> bool:
    """True when migrations already ran at SCHEMA_VERSION (skip re-inspection).
    Verifies sentinel columns instead of trusting the stamp alone; any failure
    falls through to a full (idempotent) migration pass."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT value FROM app_tokens WHERE name = :n"),
                {"n": _SCHEMA_TOKEN_NAME},
            ).first()
            if not (row and row[0] == SCHEMA_VERSION):
                return False
            for probe in _SCHEMA_SENTINELS:
                conn.execute(text(probe))
        return True
    except Exception:
        return False


def _set_schema_version() -> None:
    from sqlalchemy import text

    from .models import utcnow

    now = utcnow().isoformat()
    with engine.begin() as conn:
        # Upsert without depending on dialect-specific ON CONFLICT for SQLite.
        existing = conn.execute(
            text("SELECT name FROM app_tokens WHERE name = :n"),
            {"n": _SCHEMA_TOKEN_NAME},
        ).first()
        if existing:
            conn.execute(
                text("UPDATE app_tokens SET value = :v, updated_at = :t WHERE name = :n"),
                {"v": SCHEMA_VERSION, "t": now, "n": _SCHEMA_TOKEN_NAME},
            )
        else:
            conn.execute(
                text("INSERT INTO app_tokens (name, value, updated_at) VALUES (:n, :v, :t)"),
                {"n": _SCHEMA_TOKEN_NAME, "v": SCHEMA_VERSION, "t": now},
            )


def _ensure_new_columns() -> None:
    """Lightweight additive migration for columns added after first release."""
    from sqlalchemy import inspect, text

    bool_default = "FALSE" if not _is_sqlite else "0"
    tables = {
        "candidates": {
            "clip_title": "TEXT DEFAULT ''",
            "trim_segments": "TEXT DEFAULT ''",
            "trimmed_clip_path": "TEXT DEFAULT ''",
            "subtitled_clip_path": "TEXT DEFAULT ''",
            "use_subtitles": f"BOOLEAN DEFAULT {bool_default}",
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
            "first_reply_id": "VARCHAR(60) DEFAULT ''",
            "first_reply_text": "TEXT DEFAULT ''",
            "first_reply_error": "TEXT DEFAULT ''",
            "first_reply_at": "TIMESTAMP WITH TIME ZONE",
            "suggested_caption": "TEXT DEFAULT ''",
            "footage_traits": "TEXT DEFAULT ''",
            "footage_score": "FLOAT",
            "footage_rationale": "TEXT DEFAULT ''",
            "footage_scored_at": "TIMESTAMP WITH TIME ZONE",
        },
        "trait_weights": {
            "effective_n": "FLOAT",
            "median_metric": "FLOAT",
            "baseline": "FLOAT",
            "status": "VARCHAR(20) DEFAULT 'collecting'",
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

    Loads every existing channel in one query, then upserts in memory — critical
    when the DB is remote (Supabase), where one SELECT-per-channel was ~20s.
    """
    seed = load_channel_seed()
    if not seed:
        return 0

    by_url = {
        ch.url: ch
        for ch in session.execute(select(Channel)).scalars().all()
        if ch.url
    }
    added = 0
    for entry in seed:
        url = str(entry.get("url", "")).strip()
        if not url:
            continue
        existing = by_url.get(url)
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
            # Prevent duplicate inserts if the seed itself has duplicate URLs.
            by_url[url] = None  # type: ignore[assignment]
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
    """Seed the trait vocabulary from settings on first run. Only adds missing
    names; never deletes or overwrites, so Traits-page edits survive a re-sync.
    """
    settings = load_settings()
    names = settings.get("vision.traits") or []
    if not names:
        # Legacy keys: merge both old lists into one flat vocabulary.
        names = list(settings.get("vision.desirable_traits") or []) \
            + list(settings.get("vision.undesirable_traits") or [])
    existing = {t.name for t in session.execute(select(Trait)).scalars().all()}
    added = 0
    for raw in names:
        name = str(raw).strip()
        if not name or name in existing:
            continue
        session.add(Trait(name=name, kind=Trait.KIND_NEUTRAL, enabled=True))
        existing.add(name)
        added += 1
    session.flush()
    return added


def active_traits(session: Session) -> list[str]:
    """Enabled trait names (flat vocabulary)."""
    rows = session.execute(select(Trait).where(Trait.enabled)).scalars().all()
    return [t.name for t in rows]
