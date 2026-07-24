"""Database engine/session setup and channel-seed sync."""
from __future__ import annotations

import time
from contextlib import contextmanager

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from .config import database_url, load_channel_seed, load_settings
from .models import Base, Channel, Trait

_url = database_url()
_is_sqlite = _url.startswith("sqlite")

# Bump when additive migrations in ``_ensure_new_columns`` /
# ``_drop_removed_columns`` / ``_migrate_scheduled_to_queued`` /
# ``_ensure_indexes`` change.
# Stored in ``app_tokens`` so remote Postgres startups skip the expensive
# inspection round trips after the first successful migrate.
SCHEMA_VERSION = "9"
_SCHEMA_TOKEN_NAME = "_schema_version"

_engine_kwargs: dict = {"future": True}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"timeout": 30}
else:
    # Supabase/Postgres. No pool_pre_ping: it costs a network round trip on
    # every connection checkout (i.e. every request). Instead, recycle
    # connections well before Supabase's ~30-minute idle close so we never
    # hand out a stale one.
    _engine_kwargs.update(
        pool_size=10,
        max_overflow=20,
        pool_recycle=900,
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
    _migrate_cuts()          # must run before old candidate clip columns are dropped
    _drop_removed_columns()
    _migrate_scheduled_to_queued()
    _ensure_indexes()
    _set_schema_version()


# Cheap probes that must succeed for the schema to really be at SCHEMA_VERSION.
# Update alongside every version bump. Guards against a stale/premature stamp
# (e.g. a dev-reload importing db.py between two source edits once wrote the
# new version while the column list was still old — the stamp then blocked the
# migration forever while queries crashed on missing columns).
_SCHEMA_SENTINELS = (
    "SELECT suggested_caption FROM threads_posts LIMIT 0",
    "SELECT status FROM trait_weights LIMIT 0",
    "SELECT cut_pk FROM threads_posts LIMIT 0",
    "SELECT attention_dismissed_at FROM threads_posts LIMIT 0",
    "SELECT id FROM cuts LIMIT 0",
    "SELECT subs_position FROM cuts LIMIT 0",
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
            "visual_score": "FLOAT",
            "visual_traits": "TEXT DEFAULT ''",
            "visual_rationale": "TEXT DEFAULT ''",
            "visual_scored_at": "TIMESTAMP",
        },
        "threads_posts": {
            "cut_pk": "INTEGER",
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
            "attention_dismissed_at": "TIMESTAMP WITH TIME ZONE",
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
        "cuts": {
            "subs_position": "VARCHAR(10) DEFAULT 'bottom'",
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


def _migrate_cuts() -> None:
    """Promote the per-video trim columns to first-class ``cuts`` rows.

    Before this release the trimmed clip lived as columns on ``candidates``
    (one implicit cut per video). This backfills one ``Cut`` per candidate that
    had trim data, then points its posts at the new cut. Idempotent: runs only
    while the old columns still exist and skips candidates already migrated.
    """
    from sqlalchemy import inspect, text

    from .models import utcnow

    with engine.begin() as conn:
        insp = inspect(conn)
        cand_cols = {c["name"] for c in insp.get_columns("candidates")}
        # Old columns already dropped -> migration ran on a prior startup.
        if "trimmed_clip_path" not in cand_cols and "trim_segments" not in cand_cols:
            return

        already = {
            r[0] for r in conn.execute(
                text("SELECT DISTINCT candidate_pk FROM cuts")
            ).fetchall()
        }
        rows = conn.execute(text(
            "SELECT id, clip_title, trim_segments, trimmed_clip_path, "
            "subtitled_clip_path, use_subtitles, draft_caption "
            "FROM candidates "
            "WHERE (trim_segments IS NOT NULL AND trim_segments != '') "
            "   OR (trimmed_clip_path IS NOT NULL AND trimmed_clip_path != '')"
        )).fetchall()
        now = utcnow().isoformat()
        for r in rows:
            cid = r[0]
            if cid in already:
                continue
            conn.execute(text(
                "INSERT INTO cuts (candidate_pk, clip_title, draft_caption, "
                "trim_segments, trimmed_clip_path, subtitled_clip_path, "
                "use_subtitles, created_at, updated_at) VALUES "
                "(:cid, :title, :draft, :segs, :clip, :subs, :use_subs, :now, :now)"
            ), {
                "cid": cid,
                "title": r[1] or "",
                "draft": r[6] or "",
                "segs": r[2] or "",
                "clip": r[3] or "",
                "subs": r[4] or "",
                "use_subs": bool(r[5]),
                "now": now,
            })
            new_cut = conn.execute(text(
                "SELECT id FROM cuts WHERE candidate_pk = :cid ORDER BY id DESC LIMIT 1"
            ), {"cid": cid}).scalar()
            conn.execute(text(
                "UPDATE threads_posts SET cut_pk = :cut "
                "WHERE candidate_pk = :cid AND (cut_pk IS NULL)"
            ), {"cut": new_cut, "cid": cid})


def _ensure_indexes() -> None:
    """Indexes for the hot query filters. Without them a remote Postgres scans
    whole tables for every dashboard/analytics filter. ``IF NOT EXISTS`` works
    on both SQLite and Postgres, so this is idempotent."""
    from sqlalchemy import text

    statements = (
        "CREATE INDEX IF NOT EXISTS ix_threads_posts_status ON threads_posts (status)",
        "CREATE INDEX IF NOT EXISTS ix_threads_posts_candidate_pk ON threads_posts (candidate_pk)",
        "CREATE INDEX IF NOT EXISTS ix_threads_posts_cut_pk ON threads_posts (cut_pk)",
        "CREATE INDEX IF NOT EXISTS ix_threads_posts_published_at ON threads_posts (published_at)",
        "CREATE INDEX IF NOT EXISTS ix_cuts_candidate_pk ON cuts (candidate_pk)",
        "CREATE INDEX IF NOT EXISTS ix_metric_snapshots_post_captured "
        "ON metric_snapshots (post_pk, captured_at)",
        "CREATE INDEX IF NOT EXISTS ix_candidates_status ON candidates (status)",
        "CREATE INDEX IF NOT EXISTS ix_candidates_channel_pk ON candidates (channel_pk)",
        "CREATE INDEX IF NOT EXISTS ix_candidates_published_at ON candidates (published_at)",
        "CREATE INDEX IF NOT EXISTS ix_threads_comments_post_pk ON threads_comments (post_pk)",
        "CREATE INDEX IF NOT EXISTS ix_threads_comments_reply_status ON threads_comments (reply_status)",
    )
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


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
        # Trim columns promoted to the ``cuts`` table (see _migrate_cuts).
        "candidates": ["climate_topic", "clip_title", "trim_segments",
                       "trimmed_clip_path", "subtitled_clip_path", "use_subtitles"],
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


# Traits change only via the Traits page, but nearly every page reads them.
# A short in-process cache saves a remote round trip per request; trait
# mutations call invalidate_traits_cache() so edits still apply immediately.
_TRAITS_CACHE_TTL_SECONDS = 60.0
_traits_cache: tuple[float, list[str]] | None = None


def invalidate_traits_cache() -> None:
    global _traits_cache
    _traits_cache = None


def active_traits(session: Session) -> list[str]:
    """Enabled trait names (flat vocabulary)."""
    global _traits_cache
    now = time.monotonic()
    if _traits_cache is not None and now - _traits_cache[0] < _TRAITS_CACHE_TTL_SECONDS:
        return list(_traits_cache[1])
    rows = session.execute(select(Trait).where(Trait.enabled)).scalars().all()
    names = [t.name for t in rows]
    _traits_cache = (now, names)
    return list(names)
