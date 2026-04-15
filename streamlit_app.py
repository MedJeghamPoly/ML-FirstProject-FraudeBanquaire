"""
Interface Streamlit — détection de fraude (probabilité + seuil).

Lancer depuis la racine du projet :
    streamlit run streamlit_app.py

Prérequis : exécuter notebooks/02_Modeling.ipynb pour générer
    outputs/best_model.pkl et outputs/feature_columns.json
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import COST_FN, COST_FP, TIME_COL
from src.feature_engineering import add_engineered_features
from src.paths import DATA_PATH, OUTPUT_DIR

MODEL_PATH = OUTPUT_DIR / "best_model.pkl"
FEATURE_JSON = OUTPUT_DIR / "feature_columns.json"
RESULTS_CSV = OUTPUT_DIR / "model_results.csv"
TARGET = "Class"
MAX_ROWS_ENGINEER = 350_000
# Lignes d’historique du jeu réel pour moyennes mobiles causales sur données importées
HISTORY_ROWS_FOR_UPLOAD = 12_000
V_COLS = [f"V{i}" for i in range(1, 29)]
BASE_REQUIRED = [TIME_COL, "Amount", *V_COLS]


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=3600, show_spinner="Chargement des colonnes du modèle…")
def load_feature_columns() -> list[str] | None:
    if not FEATURE_JSON.is_file():
        return None
    with open(FEATURE_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner="Chargement du CSV (peut prendre un moment)…")
def load_raw_data() -> pd.DataFrame | None:
    if not DATA_PATH.is_file():
        return None
    return pd.read_csv(DATA_PATH)


def confusion_and_cost(
    y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float, cost_fp: float
) -> tuple[int, int, int, int, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    cost = float(cost_fn * fn + cost_fp * fp)
    return tn, fp, fn, tp, cost


def pick_window_start(
    n_total: int, n_win: int, mode: str, seed: int, i0_fixed: int
) -> tuple[int, dict]:
    """Retourne l’indice de début i0 et des métadonnées pour la reproductibilité."""
    n_win = min(int(n_win), n_total)
    if n_total <= 0 or n_win <= 0:
        return 0, {"i0": 0, "n_win": 0, "mode": mode, "seed": seed, "i0_fixed": i0_fixed}
    max_start = max(0, n_total - n_win)
    if mode == "Depuis le début":
        i0 = 0
    elif mode == "Décalage fixe":
        i0 = min(max(0, int(i0_fixed)), max_start)
    else:
        rng = np.random.RandomState(int(seed))
        i0 = int(rng.randint(0, max_start + 1))
    meta = {
        "i0": i0,
        "n_win": n_win,
        "mode": mode,
        "seed": int(seed),
        "i0_fixed": int(i0_fixed),
        "n_total_engineered": n_total,
    }
    return i0, meta


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
        st.caption(f"Coûts relatifs (config) : FN×{COST_FN:g} + FP×{COST_FP:g}")
        n_rows = st.number_input(
            "Nombre de lignes à scorer (fenêtre contiguë après feature engineering)",
            100,
            50_000,
            2000,
            100,
        )
        window_mode = st.radio(
            "Mode de sélection de la fenêtre",
            ("Aléatoire (graine)", "Depuis le début", "Décalage fixe"),
            index=0,
        )
        seed = st.number_input(
            "Graine (fenêtre aléatoire)", 0, 2**31 - 1, 42, disabled=(window_mode != "Aléatoire (graine)")
        )
        _n_eff = min(len(df_raw), MAX_ROWS_ENGINEER)
        max_i0 = max(0, _n_eff - min(int(n_rows), _n_eff))
        i0_fixed = st.number_input(
            "Indice de début i0 (décalage fixe)",
            0,
            max(0, max_i0),
            0,
            disabled=(window_mode != "Décalage fixe"),
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Échantillon du jeu", "Tester mes données (CSV)", "Métriques exportées", "À propos"]
    )

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

        with st.spinner("Feature engineering sur le sous-jeu…"):
            df_eng = add_engineered_features(df, time_col=TIME_COL, amount_col="Amount")

        missing = [c for c in feature_cols if c not in df_eng.columns]
        if missing:
            st.error(f"Colonnes attendues par le modèle mais absentes : {missing[:12]}…")
            st.stop()

        n_win = min(int(n_rows), len(df_eng))
        i0, slice_meta = pick_window_start(
            len(df_eng),
            n_win,
            window_mode,
            int(seed),
            int(i0_fixed),
        )
        chunk = df_eng.iloc[i0 : i0 + slice_meta["n_win"]].copy()

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

        with st.expander("Reproductibilité — paramètres de la fenêtre", expanded=False):
            st.json(slice_meta)

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                label="Télécharger les prédictions (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions_echantillon.csv",
                mime="text/csv",
            )
        with c_dl2:
            st.caption("Le fichier inclut probabilités, prédictions et étiquette réelle.")

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

        tn, fp, fn, tp, cost_t = confusion_and_cost(y_true, pred_seuil, COST_FN, COST_FP)
        st.markdown("**À seuil courant (barre latérale)**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("TN", tn)
        m2.metric("FP", fp)
        m3.metric("FN", fn)
        m4.metric("TP", tp)
        m5.metric("Coût agrégé", f"{cost_t:.1f}")

        st.subheader("Distribution des probabilités (par classe réelle)")
        plot_df = out.assign(
            classe=out["class_reelle"].map({0: "Légitime", 1: "Fraude"}),
        )
        fig = px.histogram(
            plot_df,
            x="proba_fraude",
            color="classe",
            nbins=50,
            barmode="overlay",
            opacity=0.65,
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        fig.update_layout(
            xaxis_title="Probabilité de fraude",
            yaxis_title="Effectif",
            legend_title_text="Classe réelle",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Importer un CSV pour scorer")
        st.info(
            "Colonnes requises : `Time`, `Amount`, `V1` … `V28`. "
            "La colonne **`Class` est optionnelle** (pour comparer aux prédictions si présente). "
            f"Pour des moyennes mobiles **causales**, vos lignes sont concaténées après les "
            f"**{HISTORY_ROWS_FOR_UPLOAD:,}** dernières transactions du jeu (tri par `Time`), "
            "puis le feature engineering est appliqué sur l’ensemble."
        )
        uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="upload_csv")

        if uploaded is None:
            st.caption("Choisissez un fichier pour afficher les probabilités et décisions.")
        else:
            try:
                raw_upload = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            except Exception as e:
                st.error(f"Lecture CSV impossible : {e}")
                st.stop()

            miss = [c for c in BASE_REQUIRED if c not in raw_upload.columns]
            if miss:
                st.error(f"Colonnes manquantes : {miss[:15]}{'…' if len(miss) > 15 else ''}")
                st.stop()

            for c in BASE_REQUIRED:
                raw_upload[c] = pd.to_numeric(raw_upload[c], errors="coerce")
            raw_upload = raw_upload.dropna(subset=BASE_REQUIRED).reset_index(drop=True)
            if len(raw_upload) == 0:
                st.error("Aucune ligne valide après conversion numérique.")
                st.stop()

            has_class = TARGET in raw_upload.columns
            if has_class:
                raw_upload[TARGET] = pd.to_numeric(raw_upload[TARGET], errors="coerce")

            upload_df = raw_upload[BASE_REQUIRED].copy()
            if has_class:
                upload_df[TARGET] = raw_upload[TARGET]
            else:
                upload_df[TARGET] = np.nan

            upload_df["_src"] = "import"
            upload_df["__upload_order__"] = np.arange(len(upload_df), dtype=np.int64)

            ref = (
                df_raw.sort_values(TIME_COL)
                .tail(HISTORY_ROWS_FOR_UPLOAD)
                .reset_index(drop=True)
            )
            ref_sub = ref[BASE_REQUIRED + ([TARGET] if TARGET in ref.columns else [])].copy()
            if TARGET not in ref_sub.columns:
                ref_sub[TARGET] = np.nan
            ref_sub["_src"] = "historique"
            ref_sub["__upload_order__"] = np.nan

            combined = pd.concat([ref_sub, upload_df], ignore_index=True)
            combined = combined.sort_values(TIME_COL).reset_index(drop=True)

            with st.spinner("Feature engineering (historique + vos lignes)…"):
                eng = add_engineered_features(combined, time_col=TIME_COL, amount_col="Amount")

            miss_f = [c for c in feature_cols if c not in eng.columns]
            if miss_f:
                st.error(f"Colonnes modèle absentes après engineering : {miss_f[:10]}…")
                st.stop()

            scored = (
                eng.loc[eng["_src"] == "import"]
                .sort_values("__upload_order__")
                .reset_index(drop=True)
            )
            X_new = scored[feature_cols]
            proba_u = pipe.predict_proba(X_new)[:, 1]
            pred_def_u = pipe.predict(X_new)
            pred_seuil_u = (proba_u >= seuil).astype(int)

            res = raw_upload.copy()
            res["proba_fraude"] = proba_u
            res["pred_seuil"] = pred_seuil_u
            res["pred_0_5"] = pred_def_u

            st.success(f"{len(res)} ligne(s) scorée(s). Seuil courant : {seuil:.2f}.")
            st.dataframe(
                res.head(min(200, len(res))),
                use_container_width=True,
            )

            st.download_button(
                label="Télécharger le résultat (CSV)",
                data=res.to_csv(index=False).encode("utf-8"),
                file_name="predictions_import.csv",
                mime="text/csv",
                key="dl_upload",
            )

            if has_class and res[TARGET].notna().any():
                y_u = res[TARGET].fillna(0).astype(int).values
                mask_eval = res[TARGET].notna()
                if mask_eval.sum() > 0:
                    y_e = y_u[mask_eval]
                    p_e = pred_seuil_u[mask_eval]
                    tn_u, fp_u, fn_u, tp_u, cost_u = confusion_and_cost(y_e, p_e, COST_FN, COST_FP)
                    st.markdown("**Évaluation (lignes où `Class` est renseignée)**")
                    u1, u2, u3, u4, u5 = st.columns(5)
                    u1.metric("TN", tn_u)
                    u2.metric("FP", fp_u)
                    u3.metric("FN", fn_u)
                    u4.metric("TP", tp_u)
                    u5.metric("Coût agrégé", f"{cost_u:.1f}")

                plot_u = pd.DataFrame({"proba_fraude": proba_u, "class_reelle": res[TARGET]})
                plot_u = plot_u.dropna(subset=["class_reelle"])
                if len(plot_u) > 0:
                    plot_u["classe"] = plot_u["class_reelle"].astype(int).map(
                        {0: "Légitime", 1: "Fraude"}
                    )
                    fig_u = px.histogram(
                        plot_u,
                        x="proba_fraude",
                        color="classe",
                        nbins=40,
                        barmode="overlay",
                        opacity=0.65,
                    )
                    fig_u.update_layout(xaxis_title="Probabilité", yaxis_title="Effectif")
                    st.plotly_chart(fig_u, use_container_width=True)

    with tab3:
        st.subheader("Derniers résultats (`model_results.csv`)")
        if RESULTS_CSV.is_file():
            st.dataframe(pd.read_csv(RESULTS_CSV), use_container_width=True)
        else:
            st.info("Fichier absent — exécutez le notebook de modélisation.")

    with tab4:
        st.markdown(
            """
            - **Données** : jeu ULB (Kaggle), fichier `data/creditcard.csv`.
            - **Pipeline** : standardisation + SMOTE (à l’entraînement) + classifieur.
            - **Seuil** : ajustez selon le coût FN/FP dans `src/config.py` (`COST_FN`, `COST_FP`).
            - **Import CSV** : les moyennes mobiles utilisent un historique récent du jeu + vos lignes (tri par `Time`).
            - **Code** : `README.md`, `Explication_Code_Projet.txt`, notebooks `01_EDA` / `02_Modeling`.
            """
        )


if __name__ == "__main__":
    main()
