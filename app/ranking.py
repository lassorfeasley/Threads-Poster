"""Candidate ranking: climate relevance, nudged by learned trait verdicts.

Trait verdicts come from published-clip performance (see
``analytics.learn_trait_weights``). Influence is capped and only ACTIVE
verdicts count — collecting/provisional traits never move the queue.
"""
from __future__ import annotations

from sqlalchemy import func, select

from .config import Settings
from .models import Candidate, TraitWeight


def order_expr(settings: Settings):
    """SQL ordering key: relevance only (visual scores retired)."""
    del settings
    return func.coalesce(Candidate.relevance_score, 0.0)


def load_trait_weights(session, metric: str = "views") -> dict[str, dict]:
    """Return {trait: {"lift", "n_posts", "status"}} from the learned table."""
    rows = session.execute(
        select(TraitWeight).where(TraitWeight.metric == metric)
    ).scalars().all()
    return {
        r.trait: {"lift": r.lift or 0.0, "n_posts": r.n_posts or 0,
                  "status": r.status or TraitWeight.STATUS_COLLECTING}
        for r in rows
    }


def trait_multiplier(traits_csv: str, weights: dict[str, dict], settings: Settings) -> float:
    """Multiplier (~1.0) from the mean lift of ACTIVE traits on this candidate.
    Storyboard tags are only a soft prior once learning unlocks; until then
    this always returns 1.0."""
    influence = float(settings.get("ranking.trait_influence", 0.3))
    if influence <= 0:
        return 1.0
    traits = [t.strip() for t in (traits_csv or "").split(",") if t.strip()]
    lifts = []
    for t in traits:
        w = weights.get(t)
        if w and w.get("status") == TraitWeight.STATUS_ACTIVE and w["lift"] is not None:
            lifts.append(max(-1.0, min(1.0, w["lift"])))
    if not lifts:
        return 1.0
    return 1.0 + (sum(lifts) / len(lifts)) * influence


def blended_score(candidate: Candidate, weights: dict[str, dict], settings: Settings) -> float:
    """Rank key: relevance × active-trait multiplier (no visual score)."""
    base = candidate.relevance_score if candidate.relevance_score is not None else 0.0
    return base * trait_multiplier(candidate.visual_traits, weights, settings)


def sort_candidates(candidates: list[Candidate], weights: dict[str, dict],
                    settings: Settings) -> list[Candidate]:
    """Stable sort by blended score, highest first."""
    return sorted(
        candidates,
        key=lambda c: (
            blended_score(c, weights, settings),
            c.published_at or c.created_at,
        ),
        reverse=True,
    )


def trait_guidance_text(weights: dict[str, dict], settings: Settings, limit: int = 6) -> str:
    """Summary of ACTIVE verdicts (unused by tag-only vision; kept for digests)."""
    del settings
    usable = [(t, w) for t, w in weights.items()
              if w.get("status") == TraitWeight.STATUS_ACTIVE and w["lift"] is not None]
    usable.sort(key=lambda kv: kv[1]["lift"], reverse=True)
    lines = []
    for t, w in usable[:limit]:
        pct = round(w["lift"] * 100)
        direction = "outperform" if pct >= 0 else "underperform"
        lines.append(f"- clips showing '{t}' {direction} by ~{abs(pct)}% "
                     f"(n={w['n_posts']})")
    return "\n".join(lines)
