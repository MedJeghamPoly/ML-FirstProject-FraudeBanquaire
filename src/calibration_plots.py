"""Courbe de calibration (fiabilité des probabilités)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def plot_calibration_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration curve",
) -> plt.Figure:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Modèle")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Parfaitement calibré")
    ax.set_xlabel("Probabilité prédite (moyenne par bin)")
    ax.set_ylabel("Fraction de positifs (réel)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
