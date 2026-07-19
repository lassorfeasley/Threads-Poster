"""Blended candidate ranking: relevance + visual appeal, nudged by learned
trait weights.

The trait weights are the self-improvement loop — they come from the operator's
own posts' performance (see ``analytics.learn_trait_weights``), so ranking
drifts toward whatever footage traits actually correlate with more views. All
correlational; influence is capped and gated on sample size so a few lucky posts
can't dominate.
"""
from __future__ import annotations

from sqlalchemy import func, select

from .config import Settings
from .models import Candidate, TraitWeight


def order_expr(settings: Settings):
    """SQL ordering key mirroring `blended_score`'s base (before trait nudge),
    so the DB can rank + cap candidates by combined relevance+visual appeal.

    An unscored visual falls back to relevance (coalesce), so candidates that
    haven't been vision-scored yet aren't penalized — they simply rank on
    relevance alone until scored.
    """
    w_rel = float(settings.get("ranking.relevance_weight", 0.5))
    w_vis = float(settings.get("ranking.visual_weight", 0.5))
    rel = func.coalesce(Candidate.relevance_score, 0.0)
    vis = func.coalesce(Candidate.visual_score, Candidate.relevance_score, 0.0)
    return w_rel * rel + w_vis * vis


def load_trait_weights(session, metric: str = "views") -> dict[str, dict]:
    """Return {trait: {"lift": float, "n_posts": int}} from the learned table."""
    rows = session.execute(
        select(TraitWeight).where(TraitWeight.metric == metric)
    ).scalars().all()
    return {r.trait: {"lift": r.lift or 0.0, "n_posts": r.n_posts or 0} for r in rows}


def trait_multiplier(traits_csv: str, weights: dict[str, dict], settings: Settings) -> float:
    """Multiplier (~1.0) applied to a candidate's base score, from the mean
    learned lift of its detected traits. Only traits with enough posts count."""
    influence = float(settings.get("ranking.trait_influence", 0.3))
    min_posts = int(settings.get("ranking.trait_min_posts", 8))
    if influence <= 0:
        return 1.0
    traits = [t.strip() for t in (traits_csv or "").split(",") if t.strip()]
    lifts = []
    for t in traits:
        w = weights.get(t)
        if w and w["n_posts"] >= min_posts and w["lift"] is not None:
            # Clamp each lift to [-1, 1] so one outlier trait can't swing wildly.
            lifts.append(max(-1.0, min(1.0, w["lift"])))
    if not lifts:
        return 1.0
    return 1.0 + (sum(lifts) / len(lifts)) * influence


def blended_score(candidate: Candidate, weights: dict[str, dict], settings: Settings) -> float:
    """Rank key for a candidate. Combines relevance and visual scores per the
    configured weights, then applies the learned trait multiplier."""
    w_rel = float(settings.get("ranking.relevance_weight", 0.5))
    w_vis = float(settings.get("ranking.visual_weight", 0.5))
    rel = candidate.relevance_score
    vis = candidate.visual_score

    if rel is None and vis is None:
        base = 0.0
    elif vis is None:
        base = rel  # not vision-scored yet: rank on relevance alone
    elif rel is None:
        base = vis
    else:
        total = w_rel + w_vis or 1.0
        base = (w_rel * rel + w_vis * vis) / total

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
    """One-line-per-trait summary of learned performance, fed back into the
    vision prompt as a soft prior. Empty until enough posts accumulate."""
    min_posts = int(settings.get("ranking.trait_min_posts", 8))
    usable = [(t, w) for t, w in weights.items()
              if w["n_posts"] >= min_posts and w["lift"] is not None]
    usable.sort(key=lambda kv: kv[1]["lift"], reverse=True)
    lines = []
    for t, w in usable[:limit]:
        pct = round(w["lift"] * 100)
        direction = "outperform" if pct >= 0 else "underperform"
        lines.append(f"- clips showing '{t}' {direction} by ~{abs(pct)}% "
                     f"(n={w['n_posts']})")
    return "\n".join(lines)
