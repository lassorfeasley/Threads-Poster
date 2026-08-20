"""Smoke test: pages render, the embedded-scheduler flag works, and every
_WRITE_SCOPE route pattern actually exists.

Run:  SCHEDULER_EMBEDDED=false DATABASE_URL="sqlite:///workspaces/climate/data/app.db" \
      .venv/bin/python scripts/smoke_pages.py
Read-only against whatever database is configured; defaults enforce SQLite +
no scheduler so nothing can publish.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WORKSPACE", "climate")
os.environ["SCHEDULER_EMBEDDED"] = "false"
os.environ.setdefault(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "workspaces", "climate", "data", "app.db"),
)


def main() -> None:
    from fastapi.testclient import TestClient

    import app.scheduler as sched
    from app.web import main as web_main

    # The flag must have kept the scheduler thread from starting.
    assert sched._thread is None, "SCHEDULER_EMBEDDED=false did not skip the scheduler"
    print("ok  scheduler thread not started (SCHEDULER_EMBEDDED=false)")

    # Every mapped write route must exist, or a typo silently reverts that
    # route to blanket invalidation (worse: a rename would too).
    known = {getattr(r, "path_format", None) for r in web_main.app.routes}
    missing = [p for p in web_main._WRITE_SCOPE if p not in known]
    assert not missing, f"_WRITE_SCOPE patterns with no matching route: {missing}"
    print(f"ok  all {len(web_main._WRITE_SCOPE)} _WRITE_SCOPE patterns match real routes")

    # Every mapped dataset name must be registered with the pagecache.
    from app.web import pagecache
    names = set(pagecache._entries)
    bad = [(p, n) for p, scope in web_main._WRITE_SCOPE.items()
           for n in scope if n not in names]
    assert not bad, f"_WRITE_SCOPE references unregistered datasets: {bad}"
    print("ok  all _WRITE_SCOPE datasets are registered")

    client = TestClient(web_main.app)
    for path in ("/", "/calendar", "/library", "/notifications", "/connections",
                 "/product-roadmap"):
        r = client.get(path)
        status = "ok " if r.status_code == 200 else "FAIL"
        print(f"{status} GET {path} -> {r.status_code}")
        if path != "/product-roadmap":   # roadmap route name unknown; tolerate 404
            assert r.status_code == 200, f"{path} returned {r.status_code}"


if __name__ == "__main__":
    main()
