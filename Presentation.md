---
title: Détection de fraude par carte bancaire
subtitle: Projet Machine Learning — soutenance
author: "[Votre nom]"
date: "Avril 2026"
---

# Contexte

- Fraude par carte : enjeux financiers, conformité, confiance client.
- Volume transactionnel : filtrage automatique indispensable.
- Données : `creditcard.csv` — ~285 k transactions, variables `Time`, `Amount`, `V1`–`V28` (PCA), cible `Class`.

---

# Problème formalisé

- **Classification binaire** supervisée.
- **Déséquilibre sévère** : la majorité des lignes sont « légitimes ».
- **Métriques** : F1, ROC-AUC, précision/rappel sur la fraude — pas seulement l’exactitude.

---

# Chaîne de traitement

1. EDA : distributions, corrélations, valeurs manquantes.
2. Tri par `Time` puis **validation temporelle** : train = début de période, test = fin (évite le mélange temporel irréaliste).
3. **Feature engineering** : `log1p(Amount)`, ratios vs moyenne mobile causale (sans fuite).
4. **Sélection de variables** : L1 (logistique) sur échantillon train, top coefficients.
5. Prétraitement : standardisation, **SMOTE** sur le train uniquement (pipeline `imblearn`).
6. Modèles : régression logistique, **Random Forest** (GridSearch F1), **XGBoost**, **LightGBM**.
7. **Seuil** : courbe précision–rappel + minimisation du coût \(C_{\mathrm{FN}} \cdot \mathrm{FN} + C_{\mathrm{FP}} \cdot \mathrm{FP}\).
8. **Calibration** des probabilités (courbe de fiabilité).
9. **Dérive** train vs test (KS, PSI) sur `Amount` et quelques `V*`.
10. **SHAP** (échantillon) sur le meilleur modèle arborescent pour l’interprétation.

---

# Résultats (figures — après exécution du notebook)

Les fichiers suivants sont produits dans **`outputs/`** (exemples de noms) :

- Matrice de confusion : `confusion_matrix.png` (ou équivalent généré par le notebook).
- Courbe ROC : `roc_curve.png`.
- Précision–rappel : `precision_recall_curve.png`.
- Calibration : `calibration_curve.png`.
- Synthèse SHAP : `shap_summary.png`.

Tableau des métriques : **`outputs/model_results.csv`**. Dérive : **`outputs/drift_report.csv`**.

---

# Limites & perspectives

- Données datées (2013) : **dérive** et généralisation limitées ; le monitoring (PSI, KS) reste une piste opérationnelle.
- Variables anonymisées (PCA) : interprétation métier indirecte ; SHAP aide sur les variables du modèle.
- Pistes : **CatBoost** / **TabNet**, déploiement, réglage des coûts avec les équipes métier.

---

# Conclusion

- Pipeline reproductible : `requirements.txt`, `Makefile` / `tasks.ps1`, tests `pytest`, CI GitHub Actions.
- Le modèle est un **outil d’aide à la décision** ; le déploiement réel impose gouvernance, seuils et suivi de la dérive.
