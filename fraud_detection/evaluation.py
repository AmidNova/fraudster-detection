"""Alerting, cost and drift helpers used to evaluate the fraud model."""
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import precision_score, recall_score

PSI_BINS = 10
PSI_EPSILON = 1e-6


def alert_stats(y_true: ArrayLike, flagged: ArrayLike) -> dict[str, float]:
    """Size and quality of an alert list: how many users flagged, how many real fraudsters among them."""
    y_true, flagged = np.asarray(y_true), np.asarray(flagged, dtype=bool)
    return {"n_flagged": int(flagged.sum()), "fraud_caught": int((flagged & (y_true == 1)).sum()),
            "precision": precision_score(y_true, flagged, zero_division=0), "recall": recall_score(y_true, flagged)}


def expected_cost(y_true: ArrayLike, scores: ArrayLike, threshold: float, ratio: float) -> float:
    """False alarms cost 1, missed fraudsters cost `ratio`."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold
    return (flagged & (y_true == 0)).sum() + ratio * (~flagged & (y_true == 1)).sum()


def cost_optimal_threshold(y_true: ArrayLike, scores: ArrayLike, ratio: float, grid: Sequence[float]) -> float:
    """Threshold of `grid` with the lowest expected cost."""
    costs = [expected_cost(y_true, scores, t, ratio) for t in grid]
    return grid[int(np.argmin(costs))]


def population_stability_index(reference: ArrayLike, current: ArrayLike, bins: int = PSI_BINS) -> float:
    """PSI between two samples of one feature, using quantile bins of the reference."""
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    inner_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))[1:-1]
    if len(inner_edges) == 0:                # binary / constant feature: split just above the mean
        inner_edges = [reference.mean() + 1e-9]
    edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    ref_pct = np.histogram(reference, edges)[0] / len(reference) + PSI_EPSILON
    cur_pct = np.histogram(current, edges)[0] / len(current) + PSI_EPSILON
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
