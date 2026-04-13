"""Indicateurs simples de dérive train vs test (KS, PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ks_drift(train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Test KS deux échantillons ; retourne (statistic, p-value)."""
    train = np.asarray(train).ravel()
    test = np.asarray(test).ravel()
    train = train[~np.isnan(train)]
    test = test[~np.isnan(test)]
    if len(train) < 2 or len(test) < 2:
        return float("nan"), float("nan")
    return stats.ks_2samp(train, test)


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (approximatif par quantiles sur expected).
    PSI < 0.1 souvent considéré comme stable.
    """
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < buckets or len(actual) < buckets:
        return float("nan")
    qs = np.linspace(0, 1, buckets + 1)
    breaks = np.unique(np.quantile(expected, qs))
    if len(breaks) < 2:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=breaks)
    a_counts, _ = np.histogram(actual, bins=breaks)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    e_pct = np.clip(e_pct, 1e-6, 1.0)
    a_pct = np.clip(a_pct, 1e-6, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in X_train.columns:
            continue
        ks_stat, ks_p = ks_drift(X_train[col].values, X_test[col].values)
        p = psi(X_train[col].values, X_test[col].values)
        rows.append(
            {
                "feature": col,
                "KS_stat": ks_stat,
                "KS_pvalue": ks_p,
                "PSI": p,
            }
        )
    return pd.DataFrame(rows)
