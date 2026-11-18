"""The Clip handbook page: what the operator's past edits have in common.

``run.py clip-engine handbook build`` studies his exported cuts and writes a
report (config/clip_handbook.yaml): the candidate house rules with how many
edits support them, the exceptions, example cuts, and a validation on
held-out videos. The candidates themselves wait in the SAME queue as rules
drawn from his notes, on the Brand & audience page, where he adds or
dismisses them; an added one is an ordinary brand rule. So this page is a
report only — it shows each candidate's evidence and where it stands
(pending, added, dismissed, not proposed) and links to the Brand page for
the decisions. No writes, no database.

Like the clip bench page it lives in its own router, imports the engine
(``app.clip_engine.handbook``) inside the handler only, and renders through
``request.app.state.templates`` (base.html's globals live on main.py's
environment).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

log = logging.getLogger("clip_engine.handbook_page")

router = APIRouter()

STATUS_LABELS = {
    "pending": ("Waiting for you", "badge"),
    "added": ("Added as a rule", "badge score-hi"),
    "dismissed": ("Dismissed", "badge score-lo"),
    "not proposed": ("Not proposed", "badge"),
}


def percent(value: Any) -> str:
    """0.433 -> "43%"; None or junk -> "–"."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "–"
    if number != number:  # NaN
        return "–"
    return f"{number * 100:.0f}%"


def candidate_view(c: dict) -> dict:
    """One candidate as the table shows it."""
    def whole(v) -> int:
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0

    support, exceptions = whole(c.get("support")), whole(c.get("exceptions"))
    total = support + exceptions
    status = str(c.get("status") or "not proposed")
    label, badge = STATUS_LABELS.get(status, (status.capitalize(), "badge"))
    examples = [e for e in c.get("examples") or [] if e is not None]
    return {
        "id": str(c.get("id") or ""), "text": str(c.get("text") or ""),
        "when": str(c.get("when") or "Always"), "why": str(c.get("why") or ""),
        "support": support, "exceptions": exceptions,
        "share": percent(support / total) if total else "–",
        "strength": "rule" if c.get("strength") == "rule" else "tendency",
        "examples": examples, "examples_count": len(examples),
        "status": status, "status_label": label, "status_badge": badge,
        "formatted": str(c.get("formatted") or ""),
    }


def validation_view(v: Any) -> dict | None:
    """The stored validation summary, flattened for the template."""
    if not isinstance(v, dict) or not v:
        return None

    def arm(name: str) -> dict:
        data = v.get(name) if isinstance(v.get(name), dict) else {}
        return {"n": data.get("n", 0), "moment1": percent(data.get("moment_hit1")),
                "moment3": percent(data.get("moment_hit3")), "hit1": percent(data.get("hit1")),
                "hit3": percent(data.get("hit3")), "errors": data.get("errors", 0),
                "empty": data.get("empty", 0)}

    return {"at": str(v.get("at") or "")[:16].replace("T", " "),
            "videos": v.get("videos", 0), "rules_tested": v.get("rules_tested", 0),
            "setup": v.get("setup") or "", "cost": v.get("cost_usd"),
            "stopped": str(v.get("stopped") or ""),
            "arms": [("Current rules", arm("current")),
                     ("Current + candidates", arm("candidates"))]}


def report_view(report: dict | None, candidates: list[dict]) -> dict | None:
    if not report:
        return None
    rows = [candidate_view(c) for c in candidates]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "generated_at": str(report.get("generated_at") or "")[:16].replace("T", " "),
        "model": report.get("model") or "", "summary": report.get("summary") or "",
        "examples_used": report.get("examples_used") or 0,
        "holdout": report.get("holdout") or 0, "holdout_videos": report.get("holdout_videos") or 0,
        "failed": report.get("failed") or 0, "stopped": report.get("stopped") or "",
        "cost": report.get("cost_usd"),
        "rules": [r for r in rows if r["strength"] == "rule"],
        "tendencies": [r for r in rows if r["strength"] != "rule"],
        "counts": counts, "pending": counts.get("pending", 0),
        "validation": validation_view(report.get("validation")),
    }


def _render(request: Request, context: dict, status_code: int = 200):
    templates = getattr(request.app.state, "templates", None)
    if templates is None:  # main.py sets it before including this router
        raise RuntimeError("app.state.templates is not set; the clip handbook page "
                           "needs main.py's template environment")
    base = {"report": None, "error": "", "active": "analytics"}
    base.update(context)
    return templates.TemplateResponse(request, "clip_handbook.html", base,
                                      status_code=status_code)


@router.get("/clip-handbook", response_class=HTMLResponse)
def clip_handbook_page(request: Request):
    try:
        from ..clip_engine import handbook

        report = handbook.load()
        candidates = handbook.statuses(report) if report else []
        view = report_view(report, candidates)
    except Exception as exc:
        log.exception("Clip handbook page failed")
        return _render(request, {"error": f"The handbook report couldn't load: {exc}"},
                       status_code=500)
    return _render(request, {"report": view})
