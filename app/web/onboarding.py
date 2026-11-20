"""The onboarding pages (/welcome): one question per screen, with the helper a
tap away. The logic is app/onboarding.py; this is only the pages.

Customers whose account isn't onboarded yet are sent here by main.py's role
gate. Admins can open it on any account, start it over, or mark it done.

Two things run in the background while the customer carries on: reading their
website after the first screen (so the next screens arrive filled in), and
writing the behind-the-scenes rules after the last. Pages poll /welcome/status.
"""
from __future__ import annotations

import json
import logging
import threading
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .. import onboarding, onboarding_sources
from . import access

log = logging.getLogger("onboarding.page")

router = APIRouter()


def _to(url: str, msg: str = "") -> RedirectResponse:
    return RedirectResponse(url + (f"?msg={quote(msg)}" if msg else ""), status_code=303)


def _background(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True, name=f"onboarding-{fn.__name__}").start()


def _page(request: Request, ctx: dict):
    state = ctx["state"]
    base = {"steps": onboarding.STEPS, "later_steps": onboarding.LATER_STEPS,
            "is_admin": access.is_admin(request), "busy": state.get("busy"),
            "transcript": state.get("transcript") or [], "msg": ""}
    return request.app.state.templates.TemplateResponse(request, "onboarding.html", {**base, **ctx})


@router.get("/welcome")
def welcome(request: Request):
    state = onboarding.load()
    if state["status"] == "done" or state.get("busy") == "finishing":
        return _to("/welcome/finish")
    return _to(f"/welcome/{onboarding.current_step(state)}")


@router.get("/welcome/status")
def welcome_status():
    state = onboarding.load()
    values, drafted = onboarding.form_values(state)
    return {"status": state["status"], "busy": state.get("busy"),
            "busy_error": state.get("busy_error") or "",
            "drafts": {k: values[k] for k in drafted},
            "format_cards": onboarding.format_cards(state, values.get("format", "")),
            "options": state.get("options") or {}}


@router.get("/welcome/finish", response_class=HTMLResponse)
def welcome_finish(request: Request, msg: str = ""):
    state = onboarding.load()
    return _page(request, {"state": state, "spec": None, "msg": msg,
                           "name": onboarding.saved_values().get("name", "")})


@router.get("/welcome/{key}", response_class=HTMLResponse)
def welcome_step(request: Request, key: str, msg: str = ""):
    spec = onboarding.step(key)
    if spec is None:
        return _to("/welcome")
    state = onboarding.load()
    values, drafted = onboarding.form_values(state)
    name = values.get("name") or "your brand"
    return _page(request, {
        "state": state, "spec": spec, "values": values, "drafted": drafted, "msg": msg,
        "question": spec["question"].format(name=name),
        "step_number": onboarding.STEP_KEYS.index(key) + 1,
        "prev_key": onboarding.prev_step(key), "next_key": onboarding.next_step(key),
        "keywords": onboarding._keyword_list(values.get("keywords", "")),
        "format_cards": onboarding.format_cards(state, values.get("format", "")),
        "sample": onboarding.SAMPLE_ANSWERS if access.is_admin(request) else None,
        "options": state.get("options") or {}, "option_fields": onboarding.OPTION_FIELDS,
        **_screen_extras(key),
    })


def _screen_extras(key: str) -> dict:
    """What the widget screens need beyond the answers."""
    if key in ("subtitles", "endcard"):
        from .. import caption_look

        return {"fonts": caption_look.FONTS, "colors": caption_look.COLORS,
                "font_css": "https://fonts.googleapis.com/css2?" + "&".join(
                    "family=" + f.replace(" ", "+")
                    + caption_look.FONT_WEIGHT_SPEC.get(f, caption_look.DEFAULT_WEIGHT_SPEC)
                    for f in caption_look.FONTS) + "&display=swap"}
    if key == "sources":
        src = onboarding_sources.load()
        if src["status"] == "idle":
            _start_sources()
            src = onboarding_sources.load()
        return {"src": src, "src_sample": onboarding_sources.sample(src),
                "src_chosen": len(onboarding_sources.chosen(src))}
    if key == "connect":
        return {"connections": _connections()}
    if key == "launch":
        from .main import _archives_here

        return {"uploads_here": _archives_here()}
    return {}


def _connections() -> list[dict]:
    """Threads and Instagram, each connected or with its sign-in link."""
    from .. import instagram_api, threads_api

    out = []
    for name, api, action in (("Threads", threads_api, "/threads/connect"),
                              ("Instagram", instagram_api, "/instagram/connect")):
        try:
            done = api.is_authenticated()
            url = "" if done else api.authorize_url()
        except Exception:
            done, url = False, ""
        out.append({"name": name, "connected": done, "auth_url": url, "action": action})
    return out


def _start_sources(feedback: str = "") -> None:
    if onboarding_sources.load()["status"] == "running":
        return
    answers = onboarding.answers_with_state(onboarding.load())
    onboarding_sources.mutate(lambda s: s.update(status="running", checked=0, total=0))
    _background(_run_sources, answers, feedback)


def _run_sources(answers: dict, feedback: str) -> None:
    onboarding_sources.discover(answers, feedback=feedback)


@router.post("/welcome/upload")
async def welcome_upload(file: UploadFile = File(...)):
    """A video of their own, to clip right away: stored, transcribed, run
    through the clip engine, best clips to the inbox (main.clip_upload_to_inbox)."""
    from .main import UploadRejected, clip_upload_to_inbox, store_upload

    try:
        cid = await store_upload(file)
    except UploadRejected as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    _background(clip_upload_to_inbox, cid)

    def remember(state):
        state.setdefault("uploads", []).append(cid)
    st = onboarding.load()
    remember(st)
    onboarding.save(st)
    return {"ok": True, "id": cid}


@router.post("/welcome/logo")
async def welcome_logo(file: UploadFile = File(...)):
    """The account's icon, saved as soon as it's picked."""
    from .main import store_logo

    try:
        url = await run_in_threadpool(store_logo, file.filename or "", await file.read())
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"ok": True, "url": url}


@router.get("/welcome/upload/status")
def welcome_upload_status():
    from .main import upload_progress

    return {"uploads": [upload_progress(cid) for cid in onboarding.load().get("uploads") or []]}


@router.get("/welcome/sources/status")
def sources_status():
    src = onboarding_sources.load()
    return {"status": src["status"], "checked": src["checked"], "total": src["total"],
            "error": src["error"], "chosen": len(onboarding_sources.chosen(src)),
            "found": len(src["channels"]), "sample": onboarding_sources.sample(src),
            "channels": src["channels"]}


@router.post("/welcome/sources/find")
def sources_find(feedback: str = Form("")):
    _start_sources(feedback.strip()[:1000])
    return {"ok": True}


@router.post("/welcome/sources/toggle")
def sources_toggle(channel_id: str = Form(...), on: str = Form("")):
    onboarding_sources.toggle(channel_id, on in ("1", "true", "on"))
    return {"ok": True}


@router.post("/welcome/sources/add")
async def sources_add(url: str = Form("")):
    try:
        entry = await run_in_threadpool(onboarding_sources.add_own, url)
    except Exception as exc:
        return JSONResponse({"error": str(exc)[:200]}, status_code=400)
    return {"ok": True, "channel": entry}


async def _attachments(form) -> list[dict]:
    out = []
    for upload in form.getlist("files"):
        if getattr(upload, "filename", ""):
            out.append(onboarding.attachment_block(upload.filename, await upload.read()))
    return out


@router.post("/welcome/chat")
async def welcome_chat(request: Request):
    form = await request.form()
    key = str(form.get("step") or "")
    if onboarding.step(key) is None:
        return JSONResponse({"error": "Unknown step"}, status_code=400)
    text = str(form.get("text") or "")[:20000]
    try:
        current = json.loads(str(form.get("form_json") or "{}"))
        attachments = await _attachments(form)
    except onboarding.UploadError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except ValueError:
        current, attachments = {}, []
    if not text.strip() and not attachments:
        return JSONResponse({"error": "Say something or attach a file"}, status_code=400)
    try:
        result = await run_in_threadpool(
            onboarding.chat, key, text, attachments,
            form=current if isinstance(current, dict) else None)
    except Exception:
        log.exception("Onboarding helper failed")
        return JSONResponse({"error": "The helper is unavailable right now. "
                                      "Try again in a minute."}, status_code=502)
    return JSONResponse(result)


@router.post("/welcome/reset")
def welcome_reset(request: Request):
    onboarding.reset()
    onboarding_sources.reset()
    return _to("/welcome", "Onboarding started over. Saved settings are unchanged.")


@router.post("/welcome/skip")
def welcome_skip(request: Request):
    onboarding.mark_done()
    return _to("/welcome/finish", "Marked as onboarded")


@router.post("/welcome/{key}/skip")
def welcome_skip_step(request: Request, key: str):
    """Testing: on to the next screen without saving or checking anything.
    Admins only (locally, without sign-in, that's everyone)."""
    if not access.is_admin(request) or onboarding.step(key) is None:
        return _to("/welcome")
    onboarding.skip_step(key)
    nxt = onboarding.next_step(key)
    if nxt:
        return _to(f"/welcome/{nxt}")
    onboarding.mark_done()
    return _to("/welcome/finish")


@router.post("/welcome/{key}")
async def welcome_save(request: Request, key: str):
    spec = onboarding.step(key)
    if spec is None:
        return _to("/welcome")
    form = await request.form()
    values = {k: str(v) for k, v in form.items() if isinstance(v, str)}
    missing = [f for f in spec["fields"] if f["required"] and not values.get(f["name"], "").strip()]
    if missing:
        return _to(f"/welcome/{key}", "Fill this in to carry on")
    try:
        attachments = await _attachments(form) if spec.get("upload") else []
    except onboarding.UploadError as exc:
        return _to(f"/welcome/{key}", str(exc))
    before = onboarding.saved_values()
    state = await run_in_threadpool(onboarding.save_step, key, values)

    # Read the brand once, and again when they change its name or website or
    # upload something new on a second visit.
    changed = any(before.get(f, "") != onboarding.saved_values().get(f, "") for f in ("name", "cta_url"))
    if key == "brand" and state.get("busy") != "reading" and (changed or attachments):
        _background(onboarding.read_brand, values.get("name", ""),
                    values.get("cta_url", ""), attachments)
    if (key == "throughline" and state.get("busy") not in ("reading", "refreshing")
            and before.get("throughline", "") != values.get("throughline", "").strip()):
        _background(onboarding.refresh_for_throughline, values.get("throughline", "").strip())
    if key == "keywords" and (before.get("keywords") != onboarding.saved_values().get("keywords")
                              or onboarding_sources.load()["status"] in ("idle", "error")):
        # Channels are checked against the keywords, so a change means a new search.
        _start_sources()
    nxt = onboarding.next_step(key)
    if nxt:
        return _to(f"/welcome/{nxt}")
    if state.get("busy") != "finishing":
        await run_in_threadpool(onboarding.set_busy, "finishing")
        _background(onboarding.finish)
    return _to("/welcome/finish")
