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

import hmac
import json

from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..web import auth
from .. import universal_rules
from . import collect, fly, legal_store, rules_store, sources

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
    # The rules API has its own token (every account's workers call it).
    if request.url.path in ("/login", "/healthz", "/api/universal-rules") or _local(request):
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


# --- Universal rules -----------------------------------------------------------------

@app.get("/api/universal-rules")
def universal_rules_api(request: Request):
    """The live universal rules, for every account's workers. Bearer token:
    ``UNIVERSAL_RULES_TOKEN``."""
    want = os.environ.get("UNIVERSAL_RULES_TOKEN", "")
    got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not want or not hmac.compare_digest(want, got):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        return rules_store.read()
    except Exception:
        log.exception("Universal rules unreadable")
        return JSONResponse({"error": "unavailable"}, status_code=503)


@app.get("/rules", response_class=HTMLResponse)
def rules_page(request: Request, msg: str = "", error: str = ""):
    try:
        data, problem = rules_store.read(), ""
    except Exception as exc:
        data, problem = universal_rules.defaults(), f"Couldn't read the live rules ({exc}); showing the defaults."
    return templates.TemplateResponse(request, "rules.html", {
        "data": data, "scopes": universal_rules.SCOPE_LABELS, "msg": msg,
        "error": error or problem})


@app.post("/rules")
def rules_save(request: Request, payload: str = Form(...)):
    try:
        rules_store.write(json.loads(payload))
    except (ValueError, TypeError) as exc:
        return RedirectResponse(f"/rules?error={quote(str(exc))}", status_code=303)
    except Exception as exc:
        log.exception("Saving universal rules failed")
        return RedirectResponse(f"/rules?error={quote('Save failed: ' + str(exc)[:120])}", status_code=303)
    return RedirectResponse("/rules?msg=" + quote("Saved. Every account picks this up within ten minutes."),
                            status_code=303)


# --- Privacy policy and terms of use ---------------------------------------------------

def _back(path: str, *, msg: str = "", error: str = "") -> RedirectResponse:
    q = f"?msg={quote(msg)}" if msg else (f"?error={quote(error)}" if error else "")
    return RedirectResponse(path + q, status_code=303)


@app.get("/legal", response_class=HTMLResponse)
def legal_page(request: Request, msg: str = "", error: str = ""):
    docs, problem = [], ""
    for kind, label in legal_store.KINDS.items():
        try:
            rows = legal_store.versions(kind)
        except Exception as exc:
            rows, problem = [], f"Couldn't read the legal documents ({str(exc)[:160]})."
        live = legal_store.current(rows)
        docs.append({"kind": kind, "label": label, "rows": rows, "current": live,
                     "draft": next((r for r in rows if r["status"] == "draft"), None),
                     "public_path": legal_store.PUBLIC_PATHS[kind]})
    return templates.TemplateResponse(request, "legal.html", {
        "docs": docs, "msg": msg, "error": error or problem, "today_iso": dt.date.today().isoformat()})


@app.post("/legal/{kind}/new")
def legal_new(kind: str):
    try:
        row = legal_store.new_draft(kind)
    except legal_store.LegalError as exc:
        return _back("/legal", error=str(exc))
    except Exception as exc:
        log.exception("Starting a legal draft failed")
        return _back("/legal", error="Couldn't start a draft: " + str(exc)[:160])
    return RedirectResponse(f"/legal/{kind}/{row['version']}", status_code=303)


@app.get("/legal/{kind}/{version}", response_class=HTMLResponse)
def legal_version(request: Request, kind: str, version: int, msg: str = "", error: str = ""):
    try:
        row = legal_store.get(kind, version)
    except legal_store.LegalError:
        row = None
    if row is None:
        return _back("/legal", error="That version doesn't exist.")
    rows = legal_store.versions(kind)
    live = legal_store.current(rows)
    return templates.TemplateResponse(request, "legal_doc.html", {
        "doc": row, "label": legal_store.KINDS[kind], "msg": msg, "error": error,
        "is_current": bool(live and live["version"] == row["version"]),
        "public_path": legal_store.PUBLIC_PATHS[kind], "today": dt.date.today().isoformat()})


@app.post("/legal/{kind}/{version}/save")
def legal_save(kind: str, version: int, title: str = Form(""), summary: str = Form(""),
               body_md: str = Form("")):
    here = f"/legal/{kind}/{version}"
    try:
        legal_store.save_draft(kind, version, title=title, summary=summary, body_md=body_md)
    except legal_store.LegalError as exc:
        return _back(here, error=str(exc))
    except Exception as exc:
        log.exception("Saving a legal draft failed")
        return _back(here, error="Save failed: " + str(exc)[:160])
    return _back(here, msg="Draft saved.")


@app.post("/legal/{kind}/{version}/publish")
def legal_publish(request: Request, kind: str, version: int, effective_date: str = Form("")):
    here = f"/legal/{kind}/{version}"
    try:
        effective = dt.date.fromisoformat(effective_date) if effective_date else dt.date.today()
    except ValueError:
        return _back(here, error="Pick an effective date.")
    if effective < dt.date.today():
        return _back(here, error="The effective date can't be in the past.")
    who = getattr(request.state, "user", None) or "local"
    try:
        legal_store.publish(kind, version, effective=effective, by=who)
    except legal_store.LegalError as exc:
        return _back(here, error=str(exc))
    except Exception as exc:
        log.exception("Publishing a legal document failed")
        return _back(here, error="Publish failed: " + str(exc)[:160])
    when = "today" if effective == dt.date.today() else f"on {effective:%B %-d, %Y}"
    return _back("/legal", msg=f"{legal_store.KINDS[kind]} version {version} published. "
                               f"It takes effect {when}.")


@app.post("/legal/{kind}/{version}/delete")
def legal_delete(kind: str, version: int):
    try:
        legal_store.delete_draft(kind, version)
    except legal_store.LegalError as exc:
        return _back(f"/legal/{kind}/{version}", error=str(exc))
    except Exception as exc:
        log.exception("Deleting a legal draft failed")
        return _back(f"/legal/{kind}/{version}", error="Delete failed: " + str(exc)[:160])
    return _back("/legal", msg="Draft deleted.")
