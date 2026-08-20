"""One-shot latency diagnosis: DB round trips + pagecache loader timings.

Run:  .venv/bin/python scripts/diagnose_latency.py
Uses the climate workspace (remote Postgres) as configured; also times the
raw link so remote-DB cost is separable from query/loader cost.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WORKSPACE", "climate")


def timed(label: str, fn, n: int = 1):
    runs = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        runs.append((time.perf_counter() - t0) * 1000)
    if n > 1:
        print(f"{label:<44} {statistics.median(runs):8.0f} ms median "
              f"(min {min(runs):.0f} / max {max(runs):.0f}, n={n})")
    else:
        print(f"{label:<44} {runs[0]:8.0f} ms")
    return result


def main() -> None:
    from sqlalchemy import text

    from app.config import database_url
    from app.db import engine, session_scope

    url = database_url()
    backend = url.split("://", 1)[0]
    print(f"workspace={os.environ['WORKSPACE']}  backend={backend}\n")

    # 1. Raw link cost: one trivial round trip, repeated.
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))  # warm the pool/connection first
        timed("SELECT 1 (raw round trip)", lambda: conn.execute(text("SELECT 1")).scalar(), n=10)

    # 2. Table sizes, for context on loader costs.
    with engine.connect() as conn:
        for table in ("candidates", "cuts", "threads_posts", "metric_snapshots",
                      "threads_comments"):
            n_rows = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  rows in {table:<20} {n_rows:>8}")
    print()

    # 3. The pagecache loaders the plan calls out, timed cold (twice: the
    #    second run shows what a warm SQLAlchemy pool + OS cache buys).
    # Importing app.web.main normally starts the embedded scheduler (which
    # could publish a due post) and the pagecache refresher — neuter both so
    # this script only ever reads.
    import app.scheduler as _sched_mod
    import app.web.pagecache as _pc_mod
    _sched_mod.start_scheduler_thread = lambda *a, **k: None
    _pc_mod.start_refresher = lambda *a, **k: None

    from app.web.main import (_current_month_calendar_data, _default_dashboard_data,
                              _library_dataset, _notifications_data)

    for label, fn in (
        ("loader: dashboard", _default_dashboard_data),
        ("loader: library", _library_dataset),
        ("loader: calendar (build_window_plan)", _current_month_calendar_data),
        ("loader: notifications", _notifications_data),
    ):
        timed(label, fn)
        timed(f"{label} (2nd run)", fn)

    # 4. recycle_overview: bypass its module cache to see the true cost.
    from app import scheduler as sched

    def cold_recycle():
        sched.invalidate_recycle_overview()
        with session_scope() as s:
            return sched.recycle_overview(s)

    timed("recycle_overview (cold, cache bypassed)", cold_recycle)


if __name__ == "__main__":
    main()
