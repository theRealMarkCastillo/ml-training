"""Metrics with explicit undefined values and validation-only thresholds."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def threshold_at_fpr(y, scores, budget: float = .05) -> float:
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes")
    fpr, tpr, thresholds = roc_curve(y, scores)
    allowed = np.flatnonzero(fpr <= budget)
    best = allowed[np.argmax(tpr[allowed])]
    # A threshold above 1 explicitly represents 'predict no positives'.
    return float(thresholds[best]) if np.isfinite(thresholds[best]) else 1.000001


def binary_metrics(y, scores, threshold: float) -> dict:
    y = np.asarray(y)
    scores = np.asarray(scores)
    predicted = scores >= threshold
    positive = y == 1
    negative = ~positive
    both = bool(positive.any() and negative.any())
    return {
        "rows": len(y), "positives": int(positive.sum()), "prevalence": float(y.mean()),
        "pr_auc_average_precision": float(average_precision_score(y, scores)) if positive.any() else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "brier": float(brier_score_loss(y, scores)), "threshold": threshold,
        "recall": float(predicted[positive].mean()) if positive.any() else None,
        "false_positive_rate": float(predicted[negative].mean()) if negative.any() else None,
    }


def bootstrap_ap(y, scores, seed: int, repetitions: int = 100) -> dict:
    """Row bootstrap: descriptive only; correlated entities reduce independence."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) == 2:
            values.append(average_precision_score(y[ids], scores[ids]))
    return {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5)),
            "repetitions": repetitions, "method": "row bootstrap; ignores within-entity correlation"} if values else {"lower": None, "upper": None}
