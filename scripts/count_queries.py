"""Count SQL statements each pagecache loader emits, and profile the calendar
loader's compute. Read-only; scheduler/refresher threads are disabled.

Run:  .venv/bin/python scripts/count_queries.py
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WORKSPACE", "climate")


def main() -> None:
    from sqlalchemy import event

    from app.db import engine

    import app.scheduler as _sched_mod
    import app.web.pagecache as _pc_mod
    _sched_mod.start_scheduler_thread = lambda *a, **k: None
    _pc_mod.start_refresher = lambda *a, **k: None

    from app.web.main import (_current_month_calendar_data, _default_dashboard_data,
                              _library_dataset, _notifications_data)

    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    for label, fn in (
        ("dashboard", _default_dashboard_data),
        ("library", _library_dataset),
        ("calendar", _current_month_calendar_data),
        ("notifications", _notifications_data),
    ):
        statements.clear()
        t0 = time.perf_counter()
        fn()
        ms = (time.perf_counter() - t0) * 1000
        top = Counter(s.split("\n")[0][:90] for s in statements).most_common(6)
        print(f"\n=== {label}: {len(statements)} queries, {ms:.0f} ms ===")
        for stmt, n in top:
            print(f"  {n:>3}x  {stmt}")

    # Where does calendar spend CPU (not I/O)?
    print("\n=== calendar cProfile (top cumulative) ===")
    pr = cProfile.Profile()
    pr.enable()
    _current_month_calendar_data()
    pr.disable()
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("cumulative").print_stats(18)
    print(out.getvalue())


if __name__ == "__main__":
    main()
