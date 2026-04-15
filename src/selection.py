"""Sélection de variables : top-K coefficients L1 (régression logistique sur échantillon)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def select_features_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 25,
    sample_rows: int = 40_000,
    random_state: int = 42,
) -> list[str]:
    """
    Régression L1 (sparse) sur sous-échantillon stratifié, puis conservation des K plus
    grandes valeurs absolues de coefficients (classe fraude).
    """
    cols = list(X_train.columns)
    n = min(sample_rows, len(X_train))
    if n < len(X_train):
        X_s, _, y_s, _ = train_test_split(
            X_train,
            y_train,
            train_size=n,
            stratify=y_train,
            random_state=random_state,
        )
    else:
        X_s, y_s = X_train, y_train

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_s)

    # sklearn ≥ 1.8 : préférer l1_ratio=1 (L1 pur) à penalty="l1" (déprécié)
    lr = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=0.1,
        max_iter=3000,
        class_weight="balanced",
        random_state=random_state,
    )
    lr.fit(Xz, y_s)

    coef = np.abs(lr.coef_).ravel()
    k = min(max_features, len(cols))
    order = np.argsort(coef)[::-1][:k]
    selected = [cols[i] for i in sorted(order, key=lambda i: cols[i])]
    if len(selected) < 5:
        return cols
    return selected
