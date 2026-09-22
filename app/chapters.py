"""A video's table of contents: what it covers, and where.

Deliberately not part of the ``clip_proposals`` ledger. That module scores
every proposal against what actually shipped, because a proposed clip IS the
product and being wrong about it matters. A chapter is never published and
never scored — it exists to get the operator to the right minute of a
45-minute source, and being five seconds out is not being wrong.

Chapters tile the whole video: no gaps, no overlaps, first starts at 0, last
ends at the end. The trim editor leans on that, drawing them as one continuous
strip and zooming its waveform by them.
"""
from __future__ import annotations

import json
import logging

from .llm import segment_chapters

log = logging.getLogger("chapters")


def load(candidate) -> list[dict]:
    """The video's stored chapters, or [] when it has none."""
    try:
        data = json.loads(candidate.chapters or "[]")
    except (ValueError, TypeError):
        return []
    return data if isinstance(data, list) else []


def generate(candidate, settings, transcript_segments: list[dict]) -> list[dict]:
    """Run one chapter pass over a video and store the result on it.

    An empty result is stored as empty rather than left alone: "asked, and this
    video is one subject" is an answer, and re-running it on every page load
    would be a bill for the same no.

    Model failures propagate so the caller can choose between a logged warning
    and an error response.
    """
    model = settings.get("chapters.model",
                         settings.get("matching.model", "claude-haiku-4-5"))
    found = segment_chapters(
        model, candidate.title, transcript_segments,
        max_chapters=int(settings.get("chapters.max", 12)),
        min_seconds=float(settings.get("chapters.min_seconds", 30)),
        description=candidate.description or "",
    )
    candidate.chapters = json.dumps(found)
    log.info("Chapters for %s: %d", candidate.video_id, len(found))
    return found
