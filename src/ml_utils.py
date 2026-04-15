"""Pipelines SMOTE + modèles, métriques, validation croisée."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from .paths import RANDOM_STATE


def _dataframe_to_float_array(X):
    """Entrée pipeline en ndarray : évite les avertissements LightGBM / sklearn (noms de colonnes)."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def make_fraud_pipeline(classifier) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("to_array", FunctionTransformer(_dataframe_to_float_array, validate=False)),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", classifier),
        ]
    )


def stratified_train_subset(
    X: pd.DataFrame, y: pd.Series, n_samples: int, random_state: int
):
    n_samples = min(n_samples, len(X))
    if n_samples == len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state,
    )
    return X_sub, y_sub


def clf_params_from_grid(best_params: dict) -> dict:
    return {k.split("__", 1)[1]: v for k, v in best_params.items() if k.startswith("clf__")}


def evaluate_pipeline(pipe: ImbPipeline, X_test, y_test, name: str) -> dict:
    y_pred = pipe.predict(X_test)
    proba = None
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        proba = pipe.decision_function(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": np.nan if proba is None else roc_auc_score(y_test, proba),
    }


def cross_val_scores_for_model(
    fitted_pipe: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    cv_splitter,
    max_samples: int,
    random_state: int,
) -> dict:
    X_cv, y_cv = stratified_train_subset(X_train, y_train, max_samples, random_state)
    sk = clone(fitted_pipe)
    scores = cross_validate(
        sk,
        X_cv,
        y_cv,
        cv=cv_splitter,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        n_jobs=-1,
    )
    return {
        "Model": model_name,
        "CV_F1_mean": float(np.mean(scores["test_f1"])),
        "CV_F1_std": float(np.std(scores["test_f1"])),
        "CV_ROC_AUC_mean": float(np.mean(scores["test_roc_auc"])),
        "CV_ROC_AUC_std": float(np.std(scores["test_roc_auc"])),
    }
