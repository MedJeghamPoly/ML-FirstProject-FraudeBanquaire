"""Résumé SHAP optionnel pour modèles à arbres."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def estimator_supports_tree_shap(est) -> bool:
    """TreeExplainer SHAP : modèles à arbres / boosting tabulaire compatibles."""
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier

    tree_like = (
        DecisionTreeClassifier,
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
    )
    if isinstance(est, tree_like):
        return True
    try:
        from xgboost import XGBClassifier

        if isinstance(est, XGBClassifier):
            return True
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        if isinstance(est, LGBMClassifier):
            return True
    except ImportError:
        pass
    return False


def estimator_supports_linear_shap(est) -> bool:
    """LinearExplainer SHAP : régression logistique sur espace déjà mis à l'échelle."""
    from sklearn.linear_model import LogisticRegression

    return isinstance(est, LogisticRegression)


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
