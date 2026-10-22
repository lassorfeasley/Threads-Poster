"""Predicts the operator's triage call (approve vs reject) from their own past
decisions, so the queue can say how likely a video is to be approved and —
once the track record holds up — take the step over.

The training record is ``TriageDecision``: every approve/reject, latest call
per video, undone calls excluded. Features are the words of the title and
description (TF-IDF unigrams + bigrams) plus the relevance score and length.
Words carry nearly all of the signal: on the first 1,285 decisions a
time-ordered backtest scored 0.81 AUC with them against 0.62 for the relevance
score alone. Channel and category one-hots were tried and made it worse.

Pure Python on purpose. The monitor that discovers candidates runs on GitHub
Actions with the headless dependency set, and new candidates should arrive
already scored. A logistic regression over a few thousand short documents
trains in about a second without numpy.

Probabilities are trained unweighted so they read as real rates: of the videos
it calls 10-20%, roughly that share get approved. Every decision also records
the probability that was on screen when it was made (``predicted_prob``), which
is what the live scorecard grades — never the model's opinion after the fact.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import threading
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import case, func, select, update

from .config import Settings, load_settings
from .models import Candidate, STATUS_NEW, TriageDecision, TriageModel, utcnow

log = logging.getLogger("triage_model")

# Bump when the features or training change shape: a stored model with another
# version is retrained before it scores anything.
FEATURE_VERSION = 1

DESCRIPTION_CHARS = 1500
MIN_DF = 2
L2 = 1e-4
EPOCHS = 15
LEARNING_RATE = 0.1
HOLDOUT_SHARE = 0.25
# Stored params are ~0.5 MB of JSON; keep them on the newest few models only.
# The metrics stay on every row as the model's history.
KEEP_PARAMS = 3

# Calibration buckets and hand-off thresholds the scorecard reports.
BUCKETS = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0)
HANDOFF_THRESHOLDS = (0.02, 0.05, 0.10, 0.15)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9']+")
_train_lock = threading.Lock()
_cache: TriagePredictor | None = None


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _term_counts(title: str, description: str) -> Counter:
    words = _TOKEN.findall(f"{title}\n{(description or '')[:DESCRIPTION_CHARS]}".lower())
    return Counter(words + [f"{a}_{b}" for a, b in zip(words, words[1:])])


def _vector(terms: Counter, idf: dict[str, float], relevance: float | None,
            duration: int | None) -> dict[str, float]:
    v = {t: (1.0 + math.log(n)) * idf[t] for t, n in terms.items() if t in idf}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    v = {t: x / norm for t, x in v.items()}
    # Leading underscores can't collide with a token (tokens start [a-z0-9]).
    v["_bias"] = 1.0
    v["_relevance"] = relevance or 0.0
    v["_duration"] = math.log1p(duration or 0) / 8.0
    return v


@dataclass
class TriagePredictor:
    model_pk: int
    idf: dict[str, float]
    weights: dict[str, float]

    def predict(self, title: str, description: str, relevance: float | None,
                duration: int | None) -> float:
        x = _vector(_term_counts(title, description), self.idf, relevance, duration)
        return _sigmoid(sum(self.weights.get(k, 0.0) * v for k, v in x.items()))


# --- Training ---------------------------------------------------------------------

@dataclass
class _Example:
    approved: bool
    terms: Counter
    relevance: float | None
    duration: int | None


def _load_examples(session) -> list[_Example]:
    """Latest non-undone decision per video, oldest first."""
    rows = session.execute(
        select(TriageDecision.candidate_pk, TriageDecision.action,
               Candidate.title, func.substr(Candidate.description, 1, DESCRIPTION_CHARS),
               Candidate.relevance_score, Candidate.duration_seconds)
        .join(Candidate, Candidate.id == TriageDecision.candidate_pk)
        .where(TriageDecision.undone.is_(False))
        .order_by(TriageDecision.decided_at, TriageDecision.id)
    ).all()
    latest: dict[int, _Example] = {}
    for pk, action, title, desc, rel, dur in rows:
        # Re-inserting moves a re-decided video to its newest position, so the
        # holdout really is the most recent decisions.
        latest.pop(pk, None)
        latest[pk] = _Example(action == "approve", _term_counts(title or "", desc or ""),
                              rel, dur)
    return list(latest.values())


def _fit_idf(examples: list[_Example]) -> dict[str, float]:
    df: Counter = Counter()
    for ex in examples:
        df.update(ex.terms.keys())
    n = len(examples)
    return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items() if c >= MIN_DF}


def _fit(examples: list[_Example], idf: dict[str, float], seed: int = 0) -> dict[str, float]:
    """L2-regularised logistic regression by AdaGrad SGD."""
    xs = [_vector(ex.terms, idf, ex.relevance, ex.duration) for ex in examples]
    ys = [1.0 if ex.approved else 0.0 for ex in examples]
    w: dict[str, float] = {}
    g2: dict[str, float] = {}
    order = list(range(len(xs)))
    rnd = random.Random(seed)
    for _ in range(EPOCHS):
        rnd.shuffle(order)
        for i in order:
            x = xs[i]
            err = _sigmoid(sum(w.get(k, 0.0) * v for k, v in x.items())) - ys[i]
            for k, v in x.items():
                wk = w.get(k, 0.0)
                g = err * v + L2 * wk
                g2[k] = g2.get(k, 0.0) + g * g
                w[k] = wk - LEARNING_RATE * g / (math.sqrt(g2[k]) + 1e-8)
    return w


def _predict_examples(examples: list[_Example], idf: dict[str, float],
                      w: dict[str, float]) -> list[float]:
    out = []
    for ex in examples:
        x = _vector(ex.terms, idf, ex.relevance, ex.duration)
        out.append(_sigmoid(sum(w.get(k, 0.0) * v for k, v in x.items())))
    return out


# --- Grading ----------------------------------------------------------------------

def _auc(labels: list[bool], probs: list[float]) -> float | None:
    """Probability a random approval outranks a random rejection (ties = half)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    pairs = sorted(zip(probs, labels))
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        mid_rank = (i + 1 + j) / 2
        rank_sum += mid_rank * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(labels: list[bool], probs: list[float]) -> dict:
    """How the predictions held up against what the operator actually did.

    ``buckets``: within each predicted range, the share actually approved —
    honest probabilities land inside their own range. ``handoff``: had every
    video under the threshold been rejected automatically, how much of the
    queue that takes away and how many real approvals it would have thrown out.
    """
    buckets = []
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        sel = [y for y, p in zip(labels, probs)
               if lo <= p < hi or (hi == BUCKETS[-1] and p == hi)]
        buckets.append({"lo": lo, "hi": hi, "n": len(sel), "approved": sum(sel)})
    handoff = []
    for t in HANDOFF_THRESHOLDS:
        below = [y for y, p in zip(labels, probs) if p < t]
        handoff.append({"threshold": t, "below": len(below), "approvals_lost": sum(below)})
    return {"n": len(labels), "approved": sum(labels), "auc": _auc(labels, probs),
            "buckets": buckets, "handoff": handoff}


# --- Model lifecycle --------------------------------------------------------------

def _settings_get(settings: Settings | None, key: str, default):
    return (settings or load_settings()).get(f"triage_model.{key}", default)


def enabled(settings: Settings | None = None) -> bool:
    return bool(_settings_get(settings, "enabled", True))


def train(session, settings: Settings | None = None) -> TriageModel | None:
    """Backtest on the newest decisions, then fit on all of them and store it.

    Returns None while there aren't yet enough of both outcomes to learn from.
    """
    examples = _load_examples(session)
    approved = sum(ex.approved for ex in examples)
    rejected = len(examples) - approved
    min_each = int(_settings_get(settings, "min_decisions_each", 30))
    if approved < min_each or rejected < min_each:
        log.info("Triage model: %d approvals / %d rejections, need %d of each",
                 approved, rejected, min_each)
        return None

    cut = int(len(examples) * (1 - HOLDOUT_SHARE))
    head, tail = examples[:cut], examples[cut:]
    idf = _fit_idf(head)
    backtest = evaluate([ex.approved for ex in tail],
                        _predict_examples(tail, idf, _fit(head, idf)))

    idf = _fit_idf(examples)
    w = _fit(examples, idf)
    w = {k: round(v, 6) for k, v in w.items() if abs(v) >= 1e-6}
    idf = {t: round(v, 4) for t, v in idf.items() if t in w}

    row = TriageModel(
        feature_version=FEATURE_VERSION,
        n_decisions=len(examples), n_approved=approved,
        backtest=json.dumps(backtest),
        params=json.dumps({"idf": idf, "weights": w}),
    )
    session.add(row)
    session.flush()
    stale = session.execute(
        select(TriageModel).where(TriageModel.params != "")
        .order_by(TriageModel.id.desc()).offset(KEEP_PARAMS)
    ).scalars().all()
    for old in stale:
        old.params = ""
    log.info("Triage model #%s trained on %d decisions (%d approved); backtest AUC %s",
             row.id, len(examples), approved,
             f"{backtest['auc']:.3f}" if backtest["auc"] is not None else "n/a")
    return row


def latest_model(session) -> TriageModel | None:
    return session.execute(
        select(TriageModel).where(TriageModel.feature_version == FEATURE_VERSION)
        .order_by(TriageModel.id.desc()).limit(1)
    ).scalar_one_or_none()


def load_predictor(session) -> TriagePredictor | None:
    """The newest usable model, cached in-process until a newer one exists."""
    global _cache
    pk = session.execute(
        select(TriageModel.id).where(TriageModel.feature_version == FEATURE_VERSION,
                                     TriageModel.params != "")
        .order_by(TriageModel.id.desc()).limit(1)
    ).scalar_one_or_none()
    if pk is None:
        return None
    if _cache is not None and _cache.model_pk == pk:
        return _cache
    params = json.loads(session.get(TriageModel, pk).params)
    _cache = TriagePredictor(pk, params["idf"], params["weights"])
    return _cache


def score_new_candidates(session, predictor: TriagePredictor | None = None,
                         only_missing: bool = False) -> int:
    """Write ``approve_prob`` for every candidate still awaiting triage.

    ``only_missing`` skips rows the current model already scored — what the
    monitor wants after a pass. A retrain rescores everything.
    """
    predictor = predictor or load_predictor(session)
    if predictor is None:
        return 0
    query = select(Candidate.id, Candidate.title,
                   func.substr(Candidate.description, 1, DESCRIPTION_CHARS),
                   Candidate.relevance_score, Candidate.duration_seconds
                   ).where(Candidate.status == STATUS_NEW)
    if only_missing:
        query = query.where((Candidate.approve_model_pk.is_(None))
                            | (Candidate.approve_model_pk != predictor.model_pk))
    scores = {pk: round(predictor.predict(title or "", desc or "", rel, dur), 4)
              for pk, title, desc, rel, dur in session.execute(query).all()}
    # One UPDATE per chunk rather than per row: against the remote database a
    # full rescore would otherwise be thousands of round trips.
    ids = list(scores)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        session.execute(
            update(Candidate).where(Candidate.id.in_(chunk)).values(
                approve_prob=case({pk: scores[pk] for pk in chunk}, value=Candidate.id),
                approve_model_pk=predictor.model_pk,
            ).execution_options(synchronize_session=False)
        )
    return len(scores)


def maybe_retrain(force: bool = False) -> bool:
    """Retrain and rescore when enough new decisions have come in since the
    last model (``triage_model.retrain_every``). Returns True if it retrained.

    Safe to call after every decision: the check is one count query, and a
    second caller while a training run is underway returns immediately.
    """
    from .db import session_scope

    settings = load_settings()
    if not enabled(settings) and not force:
        return False
    if not _train_lock.acquire(blocking=False):
        return False
    try:
        with session_scope() as session:
            if not force:
                current = latest_model(session)
                if current is not None:
                    fresh = session.execute(
                        select(func.count(TriageDecision.id)).where(
                            TriageDecision.undone.is_(False),
                            TriageDecision.decided_at > current.trained_at)
                    ).scalar_one()
                    if fresh < int(_settings_get(settings, "retrain_every", 20)):
                        return False
            row = train(session, settings)
            if row is None:
                return False
            session.commit()
            n = score_new_candidates(session, load_predictor(session))
            log.info("Triage model #%s scored %d candidates awaiting triage", row.id, n)
        return True
    finally:
        _train_lock.release()


def scorecard(session, limit: int = 300) -> dict:
    """The current model's backtest, plus the live grade: the probability that
    was on screen at each of the last ``limit`` decisions against the call made."""
    model = latest_model(session)
    rows = session.execute(
        select(TriageDecision.action, TriageDecision.predicted_prob)
        .where(TriageDecision.undone.is_(False), TriageDecision.predicted_prob.is_not(None))
        .order_by(TriageDecision.decided_at.desc()).limit(limit)
    ).all()
    live = evaluate([a == "approve" for a, _ in rows], [p for _, p in rows])
    return {
        "model": None if model is None else {
            "id": model.id,
            "trained_at": model.trained_at.isoformat() if model.trained_at else None,
            "n_decisions": model.n_decisions,
            "n_approved": model.n_approved,
            "backtest": json.loads(model.backtest or "{}"),
        },
        "live": live,
        "handoff_thresholds": list(HANDOFF_THRESHOLDS),
    }
