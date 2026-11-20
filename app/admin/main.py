"""The admin web app: sign-in (operator only), overview, one page per account.

Access is doubly fenced: the login must carry the admin role (Supabase
``app_metadata.role``) AND its email must be in ``ADMIN_EMAILS``. With
``ADMIN_EMAILS`` unset the site refuses everyone, so it can't be opened by
accident. Locally, ``ADMIN_LOCAL=1`` skips sign-in on a loopback address.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..web import auth
from . import collect, fly, sources

log = logging.getLogger("admin")

app = FastAPI(title="Throughline Admin", docs_url=None, redoc_url=None, openapi_url=None)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["ago"] = collect.ago
templates.env.filters["money"] = lambda v: f"${v:,.2f}" if v is not None else "—"
templates.env.filters["num"] = lambda v: f"{v:,}" if isinstance(v, int) else (v if v is not None else "—")
COOKIE = "tl_admin"


def _allowed_emails() -> set[str]:
    return {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}


def _local(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return os.environ.get("ADMIN_LOCAL") == "1" and host in ("127.0.0.1", "::1", "localhost")


def is_operator(user) -> bool:
    return bool(user and (user.role or "").lower() == "admin"
                and user.email.lower() in _allowed_emails())


@app.middleware("http")
async def _gate(request: Request, call_next):
    if request.url.path in ("/login", "/healthz") or _local(request):
        return await call_next(request)
    try:
        user, renewed = await run_in_threadpool(auth.resolve, request.cookies.get(COOKIE))
    except Exception:
        user, renewed = None, None
    if not is_operator(user):
        return RedirectResponse(f"/login?next={quote(request.url.path)}", status_code=303)
    request.state.user = user.email
    response = await call_next(request)
    if renewed is not None:
        _set_cookie(response, request, renewed)
    return response


def _set_cookie(response, request: Request, session) -> None:
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(COOKIE, auth.make_cookie(session), max_age=auth.SESSION_SECONDS,
                        httponly=True, samesite="strict", secure=secure, path="/")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/", error: str = ""):
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": error})


@app.post("/login")
def login(request: Request, email: str = Form(""), password: str = Form(""), next: str = Form("/")):
    try:
        result = auth.sign_in(email.strip(), password)
    except Exception:
        result = None
    # Same answer for a wrong password and a right password on the wrong
    # account: the page says nothing about who may use it.
    if not result or not is_operator(result[1]):
        return RedirectResponse("/login?error=1", status_code=303)
    session, _user = result
    dest = next if next.startswith("/") and not next.startswith("//") else "/"
    response = RedirectResponse(dest, status_code=303)
    _set_cookie(response, request, session)
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE, path="/")
    return response


def _machines(ws: sources.Workspace) -> list[dict]:
    return [m for a in ws.fly_apps for m in fly.machines(a)]


def _snap(ws: sources.Workspace, detail: bool = False) -> dict:
    return collect.snapshot(ws, detail=detail, machines=_machines(ws))


@app.get("/", response_class=HTMLResponse)
def overview(request: Request):
    spaces = sources.workspaces()
    with ThreadPoolExecutor(max_workers=6) as pool:
        snaps = list(pool.map(_snap, spaces))
    order = {"bad": 0, "warn": 1, "ok": 2}
    snaps.sort(key=lambda s: (order.get(s["status"], 3), s["label"].lower()))
    totals = {k: sum((s.get("pipeline") or {}).get(k, 0) or 0 for s in snaps)
              for k in ("found", "drafted", "published", "inbox_waiting")}
    return templates.TemplateResponse(request, "overview.html", {
        "snaps": snaps, "totals": totals, "now": dt.datetime.now(dt.timezone.utc),
        "spend": round(sum(s["spend_today"] for s in snaps), 2),
        "counts": {k: sum(1 for s in snaps if s["status"] == k) for k in ("bad", "warn", "ok")}})


@app.get("/account/{slug}", response_class=HTMLResponse)
def account(request: Request, slug: str):
    ws = next((w for w in sources.workspaces() if w.slug == slug), None)
    if ws is None:
        return RedirectResponse("/", status_code=303)
    snap = _snap(ws, detail=True)
    daily = (snap.get("results") or {}).get("daily") or []
    peak = max((d["views"] for d in daily), default=0) or 1
    return templates.TemplateResponse(request, "account.html", {"s": snap, "daily": daily, "peak": peak})
