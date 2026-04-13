"""Découpages train / test : stratifié ou temporel (ordre Time)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_train_test_indices(
    df: pd.DataFrame, time_col: str, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Indices lignes : train = premières (1-test_size) fraction du temps trié, test = fin."""
    if time_col not in df.columns:
        raise ValueError(f"Colonne {time_col} absente")
    order = df[time_col].values.argsort(kind="mergesort")
    n = len(order)
    cut = int(n * (1.0 - test_size))
    train_idx = order[:cut]
    test_idx = order[cut:]
    return train_idx, test_idx


def apply_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def temporal_xy_split_ordered(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train = premières lignes, test = dernières (dataframe déjà trié par temps croissant).
    """
    n = len(X)
    cut = max(1, int(n * (1.0 - test_size)))
    return (
        X.iloc[:cut].copy(),
        X.iloc[cut:].copy(),
        y.iloc[:cut].copy(),
        y.iloc[cut:].copy(),
    )
