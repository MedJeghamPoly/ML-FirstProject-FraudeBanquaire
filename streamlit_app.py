"""
Interface Streamlit — détection de fraude (probabilité + seuil).

Lancer depuis la racine du projet :
    streamlit run streamlit_app.py

Prérequis : exécuter notebooks/02_Modeling.ipynb pour générer
    outputs/best_model.pkl et outputs/feature_columns.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import TIME_COL
from src.feature_engineering import add_engineered_features
from src.paths import DATA_PATH, OUTPUT_DIR

MODEL_PATH = OUTPUT_DIR / "best_model.pkl"
FEATURE_JSON = OUTPUT_DIR / "feature_columns.json"
RESULTS_CSV = OUTPUT_DIR / "model_results.csv"
TARGET = "Class"
# Limite pour tenir en RAM dans le navigateur / machine étudiante
MAX_ROWS_ENGINEER = 350_000


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_columns() -> list[str] | None:
    if not FEATURE_JSON.is_file():
        return None
    with open(FEATURE_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_raw_data() -> pd.DataFrame | None:
    if not DATA_PATH.is_file():
        return None
    return pd.read_csv(DATA_PATH)


def main() -> None:
    st.set_page_config(
        page_title="Fraude bancaire — ML",
        page_icon="💳",
        layout="wide",
    )

    st.title("Détection de fraude par carte bancaire")
    st.caption("Probabilité de fraude avec le modèle entraîné (`outputs/best_model.pkl`).")

    pipe = load_model()
    feature_cols = load_feature_columns()
    df_raw = load_raw_data()

    if pipe is None:
        st.error(
            "Aucun modèle trouvé. Exécutez d’abord **`notebooks/02_Modeling.ipynb`** "
            "pour produire `outputs/best_model.pkl` et `outputs/feature_columns.json`."
        )
        st.stop()

    if feature_cols is None:
        st.warning(
            "`feature_columns.json` est absent. Relancez le notebook de modélisation "
            "(version récente) pour l’exporter."
        )
        st.stop()

    if df_raw is None:
        st.error(
            f"Jeu de données introuvable : `{DATA_PATH}`. Placez **creditcard.csv** dans **data/**."
        )
        st.stop()

    with st.sidebar:
        st.header("Options")
        seuil = st.slider(
            "Seuil de décision (probabilité ≥ seuil → fraude)", 0.01, 0.99, 0.5, 0.01
        )
        n_rows = st.number_input(
            "Nombre de lignes à scorer (fenêtre contiguë après feature engineering)",
            100,
            50_000,
            2000,
            100,
        )
        seed = st.number_input("Graine (position aléatoire de la fenêtre)", 0, 2**31 - 1, 42)

    tab1, tab2, tab3 = st.tabs(["Échantillon du jeu", "Métriques exportées", "À propos"])

    with tab1:
        st.subheader("Prédictions sur un sous-échantillon")
        st.info(
            "Les ratios **Amount** sont calculés sur tout le sous-jeu chargé (tri par `Time`), "
            "puis une fenêtre de lignes est extraite — cohérent avec le notebook."
        )

        df = df_raw.copy()
        if TARGET not in df.columns:
            st.error("La colonne `Class` est requise dans le CSV.")
            st.stop()
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        df = df.iloc[: min(len(df), MAX_ROWS_ENGINEER)].copy()

        df_eng = add_engineered_features(df, time_col=TIME_COL, amount_col="Amount")

        missing = [c for c in feature_cols if c not in df_eng.columns]
        if missing:
            st.error(f"Colonnes attendues par le modèle mais absentes : {missing[:12]}…")
            st.stop()

        n_win = min(int(n_rows), len(df_eng))
        rng = np.random.RandomState(int(seed))
        i0 = rng.randint(0, max(1, len(df_eng) - n_win + 1))
        chunk = df_eng.iloc[i0 : i0 + n_win].copy()

        X = chunk[feature_cols]
        y_true = chunk[TARGET].astype(int).values

        proba = pipe.predict_proba(X)[:, 1]
        pred_default = pipe.predict(X)
        pred_seuil = (proba >= seuil).astype(int)

        out = pd.DataFrame(
            {
                "proba_fraude": proba,
                "pred_seuil": pred_seuil,
                "pred_0_5": pred_default,
                "class_reelle": y_true,
            }
        )

        st.dataframe(out.head(80), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Probabilité moyenne", f"{float(proba.mean()):.4f}")
        with c2:
            st.metric("Fraude prédite (seuil)", f"{100 * pred_seuil.mean():.2f} %")
        with c3:
            st.metric("Fraude prédite (0,5)", f"{100 * pred_default.mean():.2f} %")
        with c4:
            st.metric("Fraude réelle (échantillon)", f"{100 * y_true.mean():.2f} %")

        st.subheader("Histogramme des probabilités")
        hist_df = pd.DataFrame({"proba_fraude": proba})
        st.bar_chart(hist_df)

    with tab2:
        st.subheader("Derniers résultats (`model_results.csv`)")
        if RESULTS_CSV.is_file():
            st.dataframe(pd.read_csv(RESULTS_CSV), use_container_width=True)
        else:
            st.info("Fichier absent — exécutez le notebook de modélisation.")

    with tab3:
        st.markdown(
            """
            - **Données** : jeu ULB (Kaggle), fichier `data/creditcard.csv`.
            - **Pipeline** : standardisation + SMOTE (à l’entraînement) + classifieur.
            - **Seuil** : le seuil 0,5 n’est pas obligatoire ; ajustez selon le coût FN/FP (`src/config.py`).
            - **Code** : `README.md`, `Explication_Code_Projet.txt`, notebooks `01_EDA` / `02_Modeling`.
            """
        )


if __name__ == "__main__":
    main()
