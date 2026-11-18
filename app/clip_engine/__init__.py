"""The clip engine: clip suggestions from a model that watches the video.

The original suggester (``clip_proposals.propose`` -> ``llm.suggest_clips``)
reads a transcript and guesses. It can't see the picture, can't hear the
delivery, and never looks at what it proposed. This package replaces that with
a pipeline that does what an editor does:

1. log the source video (a video model watches and listens, or Claude reads
   sampled frames plus the Whisper transcript) — ``shotlog``
2. draft many candidate clips through different lenses — ``editor``
3. snap every boundary onto word edges, shot changes and pauses — ``snap``
4. render each draft — ``media``
5. watch every render cold and rank them head-to-head against this account's
   own hits and flops — ``critic`` / ``taste`` / ``playbook``
6. revise the best one once and keep the better cut — ``engine``

``jobs`` runs it in the background of the dashboard; ``bench`` replays it over
past videos so it can be judged against what the operator actually shipped.
See docs/clip-engine.md.

Nothing here touches the database except ``jobs`` (reads inputs, writes the
proposals) and ``bench`` / ``taste`` / ``playbook`` (read-only), and nothing
the headless scheduler imports may import this package at module level.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

log = logging.getLogger("clip_engine")

# Identifies this process. A job record written by an earlier boot that never
# finished was interrupted by a restart (uvicorn --reload restarts on every save).
BOOT_ID = uuid.uuid4().hex[:8]

_MODEL_DEFAULTS = {"gemini": "gemini-3.8-flash", "claude": "claude-opus-5-5"}

SETUPS = ("transcript", "gemini", "hybrid", "claude")

ROLES = {
    "gemini": {"perceiver": "gemini", "editor": "gemini", "critic": "gemini"},
    "hybrid": {"perceiver": "gemini", "editor": "claude", "critic": "claude"},
    "claude": {"perceiver": "claude", "editor": "claude", "critic": "claude"},
}


def configured_setup(settings) -> str:
    """The setup named in settings (``clip_engine.setup``); unknown or blank
    values mean the original transcript suggester."""
    raw = str(settings.get("clip_engine.setup", "transcript") or "transcript")
    value = raw.strip().lower()
    return value if value in SETUPS else "transcript"


def effective_setup(settings, requested: str | None = None) -> tuple[str, str]:
    """``(setup, reason)``: the setup that can actually run right now.

    A setup whose provider key is missing falls back to ``transcript`` rather
    than failing, so a missing ``GEMINI_API_KEY`` degrades to today's behaviour
    instead of leaving a video with no suggestions. ``reason`` says why when it
    fell back, and is "" otherwise.
    """
    setup = (requested or configured_setup(settings)).strip().lower()
    if setup not in SETUPS:
        return "transcript", f"unknown clip engine setup {setup!r}"
    if setup == "transcript":
        return "transcript", ""
    from ..config import env

    roles = ROLES[setup]
    if "gemini" in roles.values():
        from . import gemini
        if not gemini.is_configured():
            return "transcript", "GEMINI_API_KEY is not set"
    if "claude" in roles.values() and not env("ANTHROPIC_API_KEY", "").strip():
        return "transcript", "ANTHROPIC_API_KEY is not set"
    return setup, ""


def opt(settings, key: str, default):
    """``settings.get`` that treats a blank or null YAML value as unset.

    ``Settings.get`` returns whatever is stored when the key exists, so
    ``editor_model: ""`` or a bare ``editor_model:`` would otherwise come back
    as ``""``/``None`` instead of the default."""
    value = settings.get(key, default)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value


def model_for(settings, provider: str, role: str) -> str:
    """The model id for one role ("perceiver", "editor", "critic", "judge").

    ``clip_engine.<provider>.<role>_model`` wins; else the provider-wide
    ``clip_engine.<provider>.model``; else the code default. The judge role
    (head-to-head ranking and revision checks) falls back to the critic's."""
    keys = [f"clip_engine.{provider}.{role}_model"]
    if role == "judge":
        keys.append(f"clip_engine.{provider}.critic_model")
    keys.append(f"clip_engine.{provider}.model")
    for key in keys:
        value = settings.get(key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _MODEL_DEFAULTS[provider]


def brand() -> dict:
    """brand.yaml (name, topic, mission, audience, ...). Never raises: blank
    or missing fields fall back to the prompts' own generic wording."""
    try:
        from ..config import load_brand

        return dict(load_brand() or {})
    except Exception as exc:  # pragma: no cover - a broken brand.yaml must not stop a run
        log.warning("brand.yaml unreadable: %s", exc)
        return {}


def engine_dir() -> Path:
    """Where engine artifacts live: shot logs, shot maps, renders, sidecars,
    job records. ``CLIP_ENGINE_DIR`` overrides it (tests point it at a temp dir)."""
    override = os.environ.get("CLIP_ENGINE_DIR", "").strip()
    if override:
        path = Path(override)
    else:
        from ..config import DATA_DIR

        path = DATA_DIR / "clip_engine"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bench_dir() -> Path:
    """Where replay-benchmark runs live. ``CLIPBENCH_DIR`` overrides it."""
    override = os.environ.get("CLIPBENCH_DIR", "").strip()
    if override:
        path = Path(override)
    else:
        from ..config import DATA_DIR

        path = DATA_DIR / "clipbench"
    path.mkdir(parents=True, exist_ok=True)
    return path
