"""Which workspaces the admin site watches, and read-only connections to them.

Hosted, the list comes from the ``ADMIN_WORKSPACES`` secret, a JSON list:

    [{"slug": "climate", "label": "Climate Archive",
      "database_url": "postgresql://...", "fly_apps": ["throughline-climate", ...]}]

Locally it falls back to workspaces.yaml, each workspace's own .env
(``DATABASE_URL``, else its SQLite file) and ``ADMIN_FLY_APPS``
("climate=throughline-climate,threads-poster-scheduler;other=...").
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import ROOT


@dataclass
class Workspace:
    slug: str
    label: str
    database_url: str
    fly_apps: list[str] = field(default_factory=list)


def _fly_apps_env() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in os.environ.get("ADMIN_FLY_APPS", "").split(";"):
        slug, _, apps = part.partition("=")
        if slug.strip():
            out[slug.strip()] = [a.strip() for a in apps.split(",") if a.strip()]
    return out


def workspaces() -> list[Workspace]:
    raw = os.environ.get("ADMIN_WORKSPACES", "").strip()
    if raw:
        return [Workspace(slug=w["slug"], label=w.get("label") or w["slug"],
                          database_url=w["database_url"], fly_apps=list(w.get("fly_apps") or []))
                for w in json.loads(raw)]
    from dotenv import dotenv_values

    from ..config import load_workspaces

    fly = _fly_apps_env()
    out = []
    for ws in load_workspaces():
        env = dotenv_values(ROOT / "workspaces" / ws["slug"] / ".env")
        url = (env.get("DATABASE_URL") or "").strip()
        if not url:
            db = ROOT / "workspaces" / ws["slug"] / "data" / "app.db"
            if not db.exists():
                continue
            url = f"sqlite:///file:{db}?mode=ro&uri=true"
        out.append(Workspace(slug=ws["slug"], label=ws.get("label") or ws["slug"],
                             database_url=url, fly_apps=fly.get(ws["slug"], [])))
    return out


_engines: dict[str, Engine] = {}
_lock = threading.Lock()


def _engine(ws: Workspace) -> Engine:
    with _lock:
        eng = _engines.get(ws.slug)
        if eng is None:
            url = ws.database_url
            if url.startswith("postgres"):
                # Whatever driver the URL names (postgres://, postgresql://,
                # postgresql+psycopg://), use psycopg2: it's the one installed.
                url = "postgresql+psycopg2://" + url.split("://", 1)[1]
                eng = create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=1,
                                    connect_args={"options": "-c default_transaction_read_only=on",
                                                  "connect_timeout": 8})
            else:
                eng = create_engine(url)   # "?mode=ro&uri=true": SQLite opens it read-only
            _engines[ws.slug] = eng
        return eng


def session(ws: Workspace) -> Session:
    """A read-only session on the workspace's database."""
    return sessionmaker(bind=_engine(ws), future=True)()
