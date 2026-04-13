"""Résumé SHAP optionnel pour modèles à arbres."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_shap_summary(
    clf,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    try:
        import shap
    except ImportError:
        return

    explainer = shap.TreeExplainer(clf, data=X_background)
    sv = explainer.shap_values(X_explain)
    if isinstance(sv, list):
        sv = sv[1]
    shap.summary_plot(
        sv,
        X_explain,
        feature_names=feature_names,
        show=False,
        max_display=min(20, len(feature_names)),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
