Ce dossier est **généré** par `notebooks/02_Modeling.ipynb` (et partiellement par `01_EDA.ipynb` pour les figures d’exploration).

Fichiers typiques :

- **EDA** : `class_distribution.png`, `correlation_heatmap.png`, `feature_distributions.png`, `smote_class_distribution.png` (selon le notebook).
- **Modèle** : `model_results.csv`, `best_model.pkl`, `feature_columns.json` (liste des colonnes pour Streamlit / inférence), `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`.
- **Seuils & calibration** : `precision_recall_curve.png`, `calibration_curve.png` (noms exacts selon le notebook).
- **Dérive** : `drift_report.csv` (KS / PSI sur variables choisies).
- **Explicabilité** : `shap_summary.png` (si SHAP s’exécute correctement pour le modèle retenu).

Les noms exacts des PNG peuvent varier légèrement ; consulter les appels `savefig` / `outputs/` dans `02_Modeling.ipynb`.
