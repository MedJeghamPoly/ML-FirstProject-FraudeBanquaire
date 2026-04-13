"""Régénère notebooks/02_Modeling.ipynb (contenu enrichi)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_Modeling.ipynb"

cells = []

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in s.strip().split("\n")]}

def code(s):
    return {"cell_type": "code", "metadata": {}, "source": [line + "\n" for line in s.strip().split("\n")], "execution_count": None, "outputs": []}

cells.append(md("""
# 02 — Prétraitement et modélisation (version enrichie)

- **Validation temporelle** (option) : données triées par `Time` — train = premières transactions, test = dernières (plus réaliste qu’un tirage stratifié aléatoire).
- **Features** : `log1p(Amount)`, ratios avec moyenne mobile **causale** (passé uniquement).
- **Sélection de variables** : L1 sparse sur échantillon train (CdC : *feature selection*).
- **Modèles** : LR, arbre, RF (GridSearch F1), XGBoost / HistGB, **LightGBM** si installé.
- **Après coup** : dérive train/test (KS, PSI), **courbe PR + coût** \\(C_{FN} \\cdot FN + C_{FP} \\cdot FP\\), **calibration**, **SHAP** (échantillon).
"""))

cells.append(code(r"""
import sys
from pathlib import Path

_root = Path.cwd().resolve()
if _root.name == "notebooks":
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.paths import (
    CV_FOLDS,
    CV_SCORING_FOLDS,
    CV_SCORING_MAX_SAMPLES,
    DATA_PATH,
    GRIDSEARCH_TRAIN_SAMPLES,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.config import (
    COST_FN,
    COST_FP,
    ENABLE_CALIBRATION_PLOT,
    ENABLE_DRIFT_KS,
    ENABLE_FEATURE_SELECTION,
    ENABLE_PR_COST_CURVE,
    ENABLE_SHAP,
    L1_MAX_FEATURES,
    L1_SAMPLE_ROWS,
    SHAP_MAX_BACKGROUND,
    SHAP_MAX_EXPLAIN,
    TIME_COL,
    USE_TEMPORAL_SPLIT,
)
from src.feature_engineering import add_engineered_features
from src.splits import temporal_xy_split_ordered
from src.selection import select_features_l1
from src.drift import drift_report
from src.thresholds import best_threshold_cost, cost_at_threshold
from src.calibration_plots import plot_calibration_reliability
from src.eda_plotting import COLOR_NEUTRAL, plot_smote_effect, save_fig, setup_plot_style
from src.ml_utils import (
    clf_params_from_grid,
    cross_val_scores_for_model,
    evaluate_pipeline,
    make_fraud_pipeline,
    stratified_train_subset,
)

setup_plot_style()
TARGET = "Class"

if not DATA_PATH.exists():
    raise FileNotFoundError("data/creditcard.csv introuvable.")
"""))

cells.append(code(r"""
# Chargement, feature engineering, split (temporel ou stratifié)
df = pd.read_csv(DATA_PATH)
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
df[TARGET] = df[TARGET].astype(int)

df = add_engineered_features(df, time_col=TIME_COL, amount_col="Amount")

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].copy()
y = df[TARGET].copy()

if X.isnull().any().any():
    X = X.fillna(X.median(numeric_only=True))

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
else:
    print("Aucune variable catégorielle à encoder.")

if USE_TEMPORAL_SPLIT:
    X_train, X_test, y_train, y_test = temporal_xy_split_ordered(X, y, test_size=TEST_SIZE)
    print("Découpage **temporel** : train = premiers instants, test = derniers.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Découpage **stratifié aléatoire**.")

print(f"Train : {len(X_train):,} | Test : {len(X_test):,}")
print(f"Taux fraude train : {y_train.mean():.5f} | test : {y_test.mean():.5f}")

if ENABLE_FEATURE_SELECTION:
    selected = select_features_l1(
        X_train, y_train, max_features=L1_MAX_FEATURES, sample_rows=L1_SAMPLE_ROWS
    )
    print(f"Sélection L1 : {len(selected)} variables retenues (sur {X.shape[1]}).")
    X_train = X_train[selected]
    X_test = X_test[selected]

FEATURE_NAMES = list(X_train.columns)

_sc = StandardScaler().fit(X_train)
X_train_s = _sc.transform(X_train)
_sm = SMOTE(random_state=RANDOM_STATE)
_, y_smote_demo = _sm.fit_resample(X_train_s, y_train)
plot_smote_effect(y_train, y_smote_demo, OUTPUT_DIR / "smote_class_distribution.png")
"""))

cells.append(md("## Entraînement (GridSearch RF + autres modèles)"))

cells.append(code(r"""
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
X_tune, y_tune = stratified_train_subset(
    X_train, y_train, GRIDSEARCH_TRAIN_SAMPLES, RANDOM_STATE
)
print(f"GridSearch RF : {len(X_tune):,} lignes (train : {len(X_train):,})")

rf_pipe_search = make_fraud_pipeline(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
)
param_grid_rf = {"clf__n_estimators": [100, 200], "clf__max_depth": [25, None]}
grid_rf = GridSearchCV(
    estimator=rf_pipe_search,
    param_grid=param_grid_rf,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
)
grid_rf.fit(X_tune, y_tune)
print("Meilleurs paramètres RF :", grid_rf.best_params_)

rf_best_kwargs = clf_params_from_grid(grid_rf.best_params_)
pipe_rf = make_fraud_pipeline(
    RandomForestClassifier(**rf_best_kwargs, random_state=RANDOM_STATE, n_jobs=-1)
)
pipe_rf.fit(X_train, y_train)

pipe_lr = make_fraud_pipeline(
    LogisticRegression(max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE)
)
pipe_lr.fit(X_train, y_train)

pipe_dt = make_fraud_pipeline(
    DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
)
pipe_dt.fit(X_train, y_train)

if HAS_XGB:
    gb_est = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    gb_label = "Gradient Boosting (XGBoost)"
else:
    gb_est = HistGradientBoostingClassifier(
        max_iter=150, max_depth=7, learning_rate=0.08, random_state=RANDOM_STATE
    )
    gb_label = "Gradient Boosting (HistGradientBoosting)"

pipe_gb = make_fraud_pipeline(gb_est)
pipe_gb.fit(X_train, y_train)

fitted_pipelines = {
    "Logistic Regression": pipe_lr,
    "Decision Tree": pipe_dt,
    "Random Forest (tuned)": pipe_rf,
    gb_label: pipe_gb,
}

if HAS_LGBM:
    pipe_lgbm = make_fraud_pipeline(
        LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.9,
            random_state=RANDOM_STATE,
            verbose=-1,
            force_row_wise=True,
        )
    )
    pipe_lgbm.fit(X_train, y_train)
    fitted_pipelines["LightGBM"] = pipe_lgbm

print("Modèles ajustés :", list(fitted_pipelines.keys()))
"""))

cells.append(md("## Métriques (CV train + test)"))

cells.append(code(r"""
cv_splitter = StratifiedKFold(
    n_splits=CV_SCORING_FOLDS, shuffle=True, random_state=RANDOM_STATE
)
cv_rows = [
    cross_val_scores_for_model(
        pipe, X_train, y_train, name, cv_splitter,
        CV_SCORING_MAX_SAMPLES, RANDOM_STATE,
    )
    for name, pipe in fitted_pipelines.items()
]
cv_df = pd.DataFrame(cv_rows)
display(cv_df.round(4))

rows = [evaluate_pipeline(pipe, X_test, y_test, name) for name, pipe in fitted_pipelines.items()]
results_df = pd.DataFrame(rows).merge(cv_df, on="Model", how="left")
col_order = [
    "Model", "CV_F1_mean", "CV_F1_std", "CV_ROC_AUC_mean", "CV_ROC_AUC_std",
    "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
]
results_df = results_df[[c for c in col_order if c in results_df.columns]]
results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
display(results_df.round(4))
sorted_f1 = results_df.sort_values(["F1", "ROC-AUC"], ascending=False).reset_index(drop=True)
"""))

cells.append(md("## Dérive train / test (KS, PSI)"))

cells.append(code(r"""
if ENABLE_DRIFT_KS:
    cols_drift = [c for c in ["Amount", "Amount_log", "Time", "V14", "V12"] if c in X_train.columns]
    rep = drift_report(X_train, X_test, cols_drift)
    display(rep.round(4))
    rep.to_csv(OUTPUT_DIR / "drift_report.csv", index=False)
else:
    print("Analyse de dérive désactivée (config).")
"""))

cells.append(md("## Courbe précision–rappel et seuil coût métier"))

cells.append(code(r"""
best_name = sorted_f1.iloc[0]["Model"]
best_pipe = fitted_pipelines[best_name]
y_score = best_pipe.predict_proba(X_test)[:, 1]

if ENABLE_PR_COST_CURVE:
    ap = average_precision_score(y_test, y_score)
    prec, rec, thr = precision_recall_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, color=COLOR_NEUTRAL, label=f"PR (AP={ap:.4f})")
    ax.set_xlabel("Rappel (fraude)")
    ax.set_ylabel("Précision")
    ax.set_title(f"Courbe précision–rappel — {best_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(OUTPUT_DIR / "precision_recall_curve.png")

    t_star, cost_star, info = best_threshold_cost(
        y_test.values, y_score, cost_fn=COST_FN, cost_fp=COST_FP
    )
    print(f"Coûts relatifs : FN×{COST_FN} + FP×{COST_FP}")
    print(f"Seuil approx. minimisant le coût sur le test : {t_star:.4f} (coût agrégé {cost_star:.2f})")
    tn, fp, fn, tp, c05 = cost_at_threshold(y_test.values, y_score, 0.5, COST_FN, COST_FP)
    tn2, fp2, fn2, tp2, ct = cost_at_threshold(y_test.values, y_score, t_star, COST_FN, COST_FP)
    print(f"À seuil 0.5 : FP={fp}, FN={fn}, coût={c05:.2f}")
    print(f"À seuil {t_star:.4f} : FP={fp2}, FN={fn2}, coût={ct:.2f}")
else:
    print("Courbe PR / coût désactivée.")
"""))

cells.append(md("## Meilleur modèle — figures, calibration, SHAP"))

cells.append(code(r"""
print("Meilleur modèle (F1 test) :", best_name)
joblib.dump(best_pipe, OUTPUT_DIR / "best_model.pkl")

y_pred = best_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=ax, display_labels=["Legitimate", "Fraud"], cmap="Blues", colorbar=True
)
ax.set_title(f"Matrice de confusion — {best_name}", fontweight="semibold")
plt.tight_layout()
save_fig(OUTPUT_DIR / "confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, y_score)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(fpr, tpr, lw=2.5, color=COLOR_NEUTRAL, label=f"AUC = {auc(fpr, tpr):.4f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("Faux positifs"); ax.set_ylabel("Vrais positifs")
ax.set_title(f"ROC — {best_name}", fontweight="semibold")
ax.legend()
plt.tight_layout()
save_fig(OUTPUT_DIR / "roc_curve.png")

clf = best_pipe.named_steps["clf"]
if hasattr(clf, "feature_importances_"):
    imp = clf.feature_importances_
    names = np.array(FEATURE_NAMES)
    k = min(25, len(imp))
    order = np.argsort(imp)[::-1][:k]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=imp[order], y=names[order], ax=ax, orient="h", color=COLOR_NEUTRAL)
    ax.set_title(f"Importance — {best_name}")
    plt.tight_layout()
    save_fig(OUTPUT_DIR / "feature_importance.png")

if ENABLE_CALIBRATION_PLOT:
    fig = plot_calibration_reliability(y_test.values, y_score, title=f"Calibration — {best_name}")
    fig.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if ENABLE_SHAP:
    try:
        import shap
        scaler = best_pipe.named_steps["scaler"]
        n_bg = min(SHAP_MAX_BACKGROUND, len(X_train))
        n_ex = min(SHAP_MAX_EXPLAIN, len(X_test))
        idx_bg = np.random.RandomState(RANDOM_STATE).choice(len(X_train), n_bg, replace=False)
        idx_ex = np.random.RandomState(RANDOM_STATE + 1).choice(len(X_test), n_ex, replace=False)
        X_bg = scaler.transform(X_train.iloc[idx_bg])
        X_explain = scaler.transform(X_test.iloc[idx_ex])
        explainer = shap.TreeExplainer(clf, data=X_bg)
        sv = explainer.shap_values(X_explain)
        if isinstance(sv, list):
            sv = sv[1]
        shap.summary_plot(sv, X_explain, feature_names=FEATURE_NAMES, show=False, max_display=15)
        plt.tight_layout()
        save_fig(OUTPUT_DIR / "shap_summary.png")
    except Exception as e:
        print("SHAP (TreeExplainer) indisponible ou non applicable :", e)

print("Sorties dans :", OUTPUT_DIR.resolve())
"""))

cells.append(md("""
**Lecture :** la validation temporelle peut écarter les taux de fraude train/test ; le seuil optimal par coût est indicatif (évalué sur le test — en pratique utiliser une validation dédiée). SHAP requiert `pip install shap` et un estimateur compatible (`TreeExplainer` pour les arbres).
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT)
