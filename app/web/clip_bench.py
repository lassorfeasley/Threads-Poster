"""The clip bench judging page: blind A/B judging of a replay benchmark run.

``run.py clipbench run`` replays past videos through the clip engine and the
transcript suggester; ``clipbench pairs`` renders their top picks side by side.
This page is where the operator judges those pairs. It is the one step of the
benchmark a person does, so it is built around keeping that person blind:

- Before a run is revealed the page shows ONLY progress and the next item: no
  arm names on the item, no video title, no scores, no hit rates. Which arm
  made which clip stays on the server (``bench.py``); a media URL carries just
  an opaque item id and a side letter, and the media response has no filename.
- After the reveal it shows the report, and judging is closed (``bench.judge``
  refuses; the routes here refuse first with a clear message).

It lives in its own router so main.py only includes it, and it is light at
import: ``app.clip_engine.bench`` (and through it the engine, media and the
providers) is imported inside each handler, so an import-time error in the
engine breaks this page instead of the whole dashboard. It never touches the
database — everything comes from files under ``bench_dir()`` via bench.py.

Templates come from ``request.app.state.templates``: base.html calls template
globals (brand_name, ws_current, ...) that only main.py registers on its own
Jinja environment, so a router-local environment could not render the page.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

log = logging.getLogger("clip_engine.bench_page")

router = APIRouter()

# Run names become directory names under bench_dir() and a URL path segment,
# so only a conservative alphabet is served — the same pattern bench.py and
# ``run.py clipbench run --name`` enforce (a test keeps the three in step).
# Item ids are bench.py's sha1 prefixes.
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ITEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SIDES = ("a", "b")
PAIR_CHOICES = ("a", "b", "tie", "neither")
SOLO_CHOICES = ("yes", "no")
# Why the losing clip lost — the operator's three complaints (weak openings,
# boring stretches, off-topic picks) plus the two mechanical failures. Keys
# are what bench.judge stores; labels are what the page shows.
REASONS = (
    ("weak_opening", "Weak opening"),
    ("boring", "Boring"),
    ("off_topic", "Off topic"),
    ("bad_boundaries", "Bad boundaries"),
    ("no_sense", "Doesn't make sense"),
)
_REASON_KEYS = frozenset(k for k, _ in REASONS)
# The only parts of bench.judge's answer echoed to the browser (its progress
# counts): anything more risks naming an arm, and the page must stay blind.
_PROGRESS_KEYS = ("judged", "total", "remaining")


# --- small pure helpers (tested directly) -------------------------------------

def valid_run_name(name: str) -> bool:
    return bool(name) and RUN_NAME_RE.match(name) is not None


def valid_item_id(item_id: str) -> bool:
    return bool(item_id) and _ITEM_ID_RE.match(item_id) is not None


def truthy(value: Any) -> bool:
    """A checkbox/form flag: "1", "true", "on", "yes" (any case) are true."""
    return str(value or "").strip().lower() in {"1", "true", "on", "yes"}


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def run_view(entry: dict) -> dict:
    """One ``bench.list_runs()`` entry as the plain values the template shows.

    Tolerant of shape: ``arms`` may hold labels or dicts with a ``label``,
    ``videos`` may be a count or a list, ``revealed`` a flag or a timestamp.
    """
    arms = entry.get("arms") or []
    if isinstance(arms, (str, bytes)):
        arms = [arms]
    labels = []
    for arm in arms:
        label = arm.get("label") if isinstance(arm, dict) else arm
        if label:
            labels.append(str(label))
    videos = entry.get("videos")
    n_videos = len(videos) if isinstance(videos, (list, tuple, dict)) else _as_int(videos)
    revealed = entry.get("revealed")
    if revealed is None:
        revealed = entry.get("revealed_at")
    seed = str(entry.get("seed") or "")
    mode = str(entry.get("mode") or "")
    dev = entry.get("dev")
    if dev is None:  # bench.py's own flag when present; else the same rule it uses
        dev = bool(seed and seed != "test") or bool(mode and mode != "sampled")
    return {
        "name": str(entry.get("name") or ""),
        "arms": labels,
        "videos": n_videos,
        "judged": _as_int(entry.get("judged")),
        "total": _as_int(entry.get("total")),
        "revealed": bool(revealed),
        "seed": seed,
        "mode": mode,
        "dev": bool(dev),
        "stopped": bool(entry.get("stopped")),
        "created_at": str(entry.get("created_at") or "")[:10],
    }


def item_view(item: dict | None) -> dict | None:
    """What the page may know about the next item: its id, whether it is a
    pair or a solo clip, and its sides. Deliberately nothing else — not the
    kind of pair (an "arms" pair vs a "ceiling" pair against your own cut),
    the arms, the video, or any score — even if bench.py ever sends more."""
    if not isinstance(item, dict):
        return None
    item_id = str(item.get("item_id") or item.get("id") or "")
    if not valid_item_id(item_id):
        log.error("Clip bench item has an unusable id %r; skipping it", item_id)
        return None
    solo = str(item.get("kind") or item.get("mode") or "") == "solo"
    sides = [s for s in dict.fromkeys(item.get("sides") or []) if s in SIDES]
    if not sides or (solo and len(sides) != 1) or (not solo and len(sides) != 2):
        sides = ["a"] if solo else ["a", "b"]
    return {"item_id": item_id, "solo": solo, "sides": sides}


def safe_judge_payload(result: Any) -> dict:
    """``{"ok": True}`` plus progress counts from bench.judge's answer (nested
    under ``progress`` or at the top level), and nothing else."""
    out: dict = {"ok": True}
    if isinstance(result, dict):
        counts = result.get("progress") if isinstance(result.get("progress"), dict) else result
        for key in _PROGRESS_KEYS:
            value = counts.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                out[key] = value
    return jsonable_encoder(out)


def contained_file(raw: Any, run_dir: Path, root: Path) -> Path | None:
    """``raw`` as an existing file inside ``root`` (relative paths resolve
    against the run directory), else None — so a bad or hostile pairs entry
    can never make the route serve a file from outside the bench tree."""
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = run_dir / path
    try:
        path = path.resolve()
        root = root.resolve()
    except OSError:
        return None
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


# --- bench access --------------------------------------------------------------

def _bench():
    from ..clip_engine import bench

    return bench


def _find_run(bench, name: str) -> dict | None:
    if not valid_run_name(name):
        return None
    for entry in bench.list_runs() or []:
        if isinstance(entry, dict) and str(entry.get("name") or "") == name:
            return run_view(entry)
    return None


def _media_path(bench, name: str, item_id: str, side: str) -> Path | None:
    # bench.media_path is the one (item_id, side) -> render lookup; the
    # containment check here is the page's own guard against a crafted path.
    from ..clip_engine import bench_dir

    root = bench_dir()
    return contained_file(bench.media_path(name, item_id, side), root / name, root)


def _render(request: Request, context: dict, status_code: int = 200):
    templates = getattr(request.app.state, "templates", None)
    if templates is None:  # main.py sets it before including this router
        raise RuntimeError("app.state.templates is not set; the clip bench page "
                           "needs main.py's template environment")
    base = {"runs": [], "selected": None, "item": None, "report": "", "error": "",
            "reasons": REASONS, "msg": "", "active": "analytics"}
    base.update(context)
    return templates.TemplateResponse(request, "clip_bench.html", base,
                                      status_code=status_code)


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


# --- routes ------------------------------------------------------------------------

@router.get("/clip-bench", response_class=HTMLResponse)
def clip_bench_page(request: Request, run: str = "", msg: str = ""):
    """Runs list; or, for ``?run=<name>``, that run's next item (before the
    reveal) or its report (after)."""
    try:
        bench = _bench()
        runs = [run_view(r) for r in (bench.list_runs() or []) if isinstance(r, dict)]
    except Exception as exc:
        log.exception("Clip bench page failed to list runs")
        return _render(request, {"msg": msg, "error": f"The clip bench couldn't load: {exc}"},
                       status_code=500)
    if not run:
        return _render(request, {"runs": runs, "msg": msg})

    selected = next((r for r in runs if r["name"] == run), None) if valid_run_name(run) else None
    if selected is None:
        return _render(request, {"runs": runs, "msg": msg,
                                 "error": f"There is no bench run named “{run}”."},
                       status_code=404)
    context: dict = {"selected": selected, "msg": msg}
    try:
        if selected["revealed"]:
            context["report"] = str(bench.report(run) or "")
        else:
            raw = bench.next_item(run)
            context["item"] = item_view(raw)
            if raw is not None and context["item"] is None:
                context["error"] = ("The next item is malformed, so it can't be shown "
                                    "(see the dashboard log).")
                return _render(request, context, status_code=500)
    except Exception as exc:
        log.exception("Clip bench page failed for run %s", run)
        context["error"] = f"Couldn't read this run: {exc}"
        return _render(request, context, status_code=500)
    return _render(request, context)


@router.get("/clip-bench/{name}/media/{item_id}/{side}")
def clip_bench_media(name: str, item_id: str, side: str):
    """One side of one item. No filename and nothing arm-specific in the URL or
    the headers; ``no-cache`` because a re-queued item keeps its id but gets a
    new render (the ETag changes with it)."""
    if not (valid_run_name(name) and valid_item_id(item_id) and side in SIDES):
        return _error("Not found", 404)
    try:
        path = _media_path(_bench(), name, item_id, side)
    except Exception:
        log.exception("Clip bench media lookup failed for %s/%s/%s", name, item_id, side)
        path = None
    if path is None:
        return _error("No render for this item", 404)
    return FileResponse(str(path), media_type="video/mp4",
                        headers={"Cache-Control": "no-cache"})


@router.post("/clip-bench/{name}/judge")
def clip_bench_judge(name: str, item_id: str = Form(""), choice: str = Form(""),
                     recognised: str = Form(""), reason: str = Form("")):
    choice = choice.strip().lower()
    reason = reason.strip().lower()
    if not valid_run_name(name):
        return _error("Unknown bench run", 404)
    if not valid_item_id(item_id):
        return _error("Unknown item", 400)
    if choice not in PAIR_CHOICES + SOLO_CHOICES:
        return _error(f"Unknown choice {choice!r}", 400)
    if reason and reason not in _REASON_KEYS:
        return _error(f"Unknown reason {reason!r}", 400)
    try:
        bench = _bench()
        run = _find_run(bench, name)
        if run is None:
            return _error("Unknown bench run", 404)
        if run["revealed"]:
            return _error("This run's results are revealed, so judging is closed.", 409)
        result = bench.judge(name, item_id, choice, recognised=truthy(recognised),
                             reason=reason)
    except FileNotFoundError as exc:
        return _error(f"Not found: {exc}", 404)
    except (ValueError, LookupError) as exc:
        return _error(str(exc) or "That judgment was refused", 400)
    except Exception:
        log.exception("Clip bench judge failed for %s/%s", name, item_id)
        return _error("Couldn't save the judgment (see the dashboard log)", 500)
    return JSONResponse(safe_judge_payload(result))


@router.post("/clip-bench/{name}/reveal")
def clip_bench_reveal(name: str):
    if not valid_run_name(name):
        return _error("Unknown bench run", 404)
    try:
        bench = _bench()
        if _find_run(bench, name) is None:
            return _error("Unknown bench run", 404)
        result = bench.reveal(name)
    except FileNotFoundError as exc:
        return _error(f"Not found: {exc}", 404)
    except (ValueError, LookupError) as exc:
        return _error(str(exc) or "The reveal was refused", 400)
    except Exception:
        log.exception("Clip bench reveal failed for %s", name)
        return _error("Couldn't reveal the results (see the dashboard log)", 500)
    payload = {"ok": True}
    if isinstance(result, dict):
        payload.update(result)
        payload["ok"] = True
    return JSONResponse(jsonable_encoder(payload))
