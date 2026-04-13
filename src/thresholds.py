"""Seuils sur scores : courbe PR et coût métier C_FN*FN + C_FP*FP."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def best_threshold_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> tuple[float, float, dict]:
    """
    Grille de seuils sur [0,1] ; minimise cost_fn * FN + cost_fp * FP.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    best_t = 0.5
    best_cost = float("inf")

    for t in np.linspace(0.01, 0.99, 199):
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        c = cost_fn * fn + cost_fp * fp
        if c < best_cost:
            best_cost = c
            best_t = float(t)

    return best_t, best_cost, {"cost_fn": cost_fn, "cost_fp": cost_fp}


def cost_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> tuple[int, int, int, int, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    c = cost_fn * fn + cost_fp * fp
    return int(tn), int(fp), int(fn), int(tp), float(c)
