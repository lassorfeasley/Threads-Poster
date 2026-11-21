"""Privacy policy and terms of use: every version, in the platform project's
``legal_documents`` table (schema in legal_schema.sql).

The admin site edits drafts and publishes them here, through Supabase's REST
API with the service key. Published versions are frozen by a database trigger,
so the table is the record of what was in force when. The public pages on
continuity.marketing read published versions with the anon key
(demo/api/legal.js).
"""
from __future__ import annotations

import datetime as dt
import html
import os
import re
from pathlib import Path

import httpx

KINDS = {"privacy": "Privacy policy", "terms": "Terms of use"}
PUBLIC_PATHS = {"privacy": "/privacy", "terms": "/terms"}
STARTERS = Path(__file__).parent / "legal_starters"


class LegalError(ValueError):
    """A problem the admin should see (bad input, a published version)."""


def _base() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/") + "/rest/v1/legal_documents"


def _headers(**extra) -> dict:
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    return {"Authorization": f"Bearer {key}", "apikey": key, **extra}


def _check(r: httpx.Response) -> httpx.Response:
    if r.status_code >= 400:
        try:
            detail = r.json().get("message") or r.text
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail[:200]}")
    return r


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise LegalError(f"Unknown document: {kind}")
    return kind


# --- reading ---------------------------------------------------------------------------

def versions(kind: str) -> list[dict]:
    """Every version of one document, newest first."""
    r = httpx.get(_base(), headers=_headers(), timeout=15, params={
        "kind": f"eq.{_kind(kind)}", "order": "version.desc",
        "select": "id,kind,version,status,title,summary,effective_date,created_at,"
                  "updated_at,published_at,published_by"})
    return _check(r).json()


def get(kind: str, version: int) -> dict | None:
    r = httpx.get(_base(), headers=_headers(), timeout=15, params={
        "kind": f"eq.{_kind(kind)}", "version": f"eq.{int(version)}", "select": "*"})
    rows = _check(r).json()
    return rows[0] if rows else None


def current(rows: list[dict], today: dt.date | None = None) -> dict | None:
    """The version in force today: the newest published one already effective."""
    today = today or dt.date.today()
    live = [v for v in rows if v["status"] == "published" and v.get("effective_date")
            and dt.date.fromisoformat(v["effective_date"]) <= today]
    return max(live, key=lambda v: (v["effective_date"], v["version"]), default=None)


# --- writing ---------------------------------------------------------------------------

def _starter(kind: str) -> str:
    path = STARTERS / f"{kind}.md"
    return path.read_text() if path.exists() else ""


def new_draft(kind: str) -> dict:
    """Start the next version: a copy of the newest one, or the starter text.
    There is only ever one draft per document; if one exists it is returned."""
    rows = versions(kind)
    draft = next((v for v in rows if v["status"] == "draft"), None)
    if draft:
        return draft
    if rows:
        base = get(kind, rows[0]["version"]) or {}
        body, title = base.get("body_md", ""), base.get("title") or KINDS[kind]
    else:
        body, title = _starter(kind), KINDS[kind]
    row = {"kind": kind, "version": (rows[0]["version"] + 1) if rows else 1, "status": "draft",
           "title": title, "summary": "" if rows else "First version.",
           "body_md": body, "body_html": render(body)}
    r = httpx.post(_base(), headers=_headers(Prefer="return=representation"), timeout=15, json=row)
    return _check(r).json()[0]


def _draft(kind: str, version: int) -> dict:
    row = get(kind, version)
    if row is None:
        raise LegalError("That version doesn't exist.")
    if row["status"] != "draft":
        raise LegalError("That version is published and can't be changed. Start a new version instead.")
    return row


def save_draft(kind: str, version: int, *, title: str, summary: str, body_md: str) -> None:
    _draft(kind, version)
    title = title.strip() or KINDS[kind]
    r = httpx.patch(_base(), headers=_headers(), timeout=15,
                    params={"kind": f"eq.{kind}", "version": f"eq.{int(version)}", "status": "eq.draft"},
                    json={"title": title[:200], "summary": summary.strip()[:1000],
                          "body_md": body_md, "body_html": render(body_md)})
    _check(r)


def publish(kind: str, version: int, *, effective: dt.date, by: str) -> None:
    row = _draft(kind, version)
    if not row["body_md"].strip():
        raise LegalError("The document is empty.")
    if not row["summary"].strip():
        raise LegalError("Say what changed in this version before publishing it.")
    if re.search(r"\[[A-Z][A-Z ]+\]", row["body_md"]):
        raise LegalError("The text still has a [PLACEHOLDER] in capitals. Fill it in first.")
    r = httpx.patch(_base(), headers=_headers(), timeout=15,
                    params={"kind": f"eq.{kind}", "version": f"eq.{int(version)}", "status": "eq.draft"},
                    json={"status": "published", "effective_date": effective.isoformat(),
                          "published_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                          "published_by": by[:200]})
    _check(r)


def delete_draft(kind: str, version: int) -> None:
    _draft(kind, version)
    r = httpx.delete(_base(), headers=_headers(), timeout=15,
                     params={"kind": f"eq.{kind}", "version": f"eq.{int(version)}", "status": "eq.draft"})
    _check(r)


# --- markdown --------------------------------------------------------------------------
# A small subset, enough for legal text: #/##/### headings, paragraphs, - and
# 1. lists, **bold**, *italic*, [links](https://...). Everything is escaped
# first, so the stored HTML can't carry markup the admin didn't mean.

_LINK = re.compile(r"\[([^\]]+)\]\(((?:https?://|mailto:|/)[^)\s]+)\)")


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    # The text is escaped already; only quotes are left to close off in the href.
    out = _LINK.sub(lambda m: f'<a href="{m.group(2).replace(chr(34), "&quot;")}">{m.group(1)}</a>', out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", out)
    return out


def render(md: str) -> str:
    blocks: list[str] = []
    para: list[str] = []
    items: list[str] = []
    list_tag = ""

    def flush_para():
        if para:
            blocks.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def flush_list():
        nonlocal list_tag
        if items:
            blocks.append(f"<{list_tag}>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + f"</{list_tag}>")
            items.clear()
        list_tag = ""

    for raw in md.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        bullet = re.match(r"^[-*]\s+(.*)$", line)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", line)
        if not line:
            flush_para(); flush_list()
        elif heading:
            flush_para(); flush_list()
            level = len(heading.group(1)) + 1          # the page title is the h1
            blocks.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif bullet or numbered:
            flush_para()
            tag = "ul" if bullet else "ol"
            if list_tag and list_tag != tag:
                flush_list()
            list_tag = tag
            items.append((bullet or numbered).group(1))
        elif items and raw[:1] in (" ", "\t"):
            items[-1] += " " + line                   # an indented continuation
        else:
            flush_list()
            para.append(line)
    flush_para(); flush_list()
    return "\n".join(blocks)
