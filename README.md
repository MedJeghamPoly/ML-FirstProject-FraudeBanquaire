# Détection de fraude par carte bancaire (Machine Learning)

**Auteur :** [Votre nom]  
**Cours :** Machine Learning — Polytech (Semestre 4)

## Problématique

Classer les transactions en **légitimes** (`Class = 0`) ou **frauduleuses** (`Class = 1`) à partir du jeu `creditcard.csv` (transactions anonymisées, composantes PCA `V1`–`V28`, variables `Time` et `Amount`). Le jeu est **fortement déséquilibré** ; les métriques adaptées (F1, ROC-AUC, précision/rappel sur la fraude) priment sur l’exactitude globale.

## Méthode et validation (résumé)

- **Découpage temporel** : les données sont triées par `Time` ; le train correspond au **début de la période**, le test à la **fin** (plus réaliste qu’un split aléatoire pour la fraude). Paramètre : `USE_TEMPORAL_SPLIT` dans `src/config.py`.
- **Seuils métier** : courbe précision–rappel et coût total \(C_{\mathrm{FN}} \cdot \mathrm{FN} + C_{\mathrm{FP}} \cdot \mathrm{FP}\) pour choisir un seuil sur les probabilités (pas seulement 0,5). Constantes : `COST_FN` / `COST_FP` dans `src/config.py`.
- **Calibration** : courbe de fiabilité des probabilités (sortie dans `outputs/` après exécution du notebook de modélisation).
- **Feature engineering** : `log1p(Amount)`, ratios par rapport à une moyenne mobile **causale** (fenêtre passée uniquement, sans fuite d’information) — voir `src/feature_engineering.py`.
- **Sélection de variables** : pénalisation L1 sur un sous-échantillon train (coefficients non nuls, top-K) — `src/selection.py`.
- **Modèles** : régression logistique, forêt aléatoire, XGBoost, **LightGBM** (si installé). CatBoost / TabNet peuvent être ajoutés comme extensions.
- **Dérive** : comparaison train vs test (KS, PSI approximatif) sur `Amount` et quelques `V*` — `src/drift.py`, export CSV dans `outputs/`.
- **Explicabilité** : **SHAP** (TreeExplainer) sur un échantillon du meilleur modèle lorsque le classifieur est un arbre / ensemble compatible — figure `outputs/shap_summary.png` si l’exécution réussit.

## Source des données

Jeu **Credit Card Fraud Detection** (référence courante : [Kaggle — ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)), également cité sur le dépôt UCI.  
**Justification (hors portails prioritaires du cahier des charges) :** benchmark académique standard pour le déséquilibre de classes et la détection d’anomalies supervisée ; volume suffisant (~285 k lignes) et nombreuses variables numériques.

Placez `creditcard.csv` dans le dossier **`data/`** (voir `data/README.md`). Si une copie existe encore à la racine du projet, vous pouvez la supprimer pour éviter la duplication (~150 Mo).

## Installation

```bash
cd ProjetFraudeBanquaire
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Avec Conda (optionnel) : `conda env create -f environment.yml` puis `conda activate fraude-ml`.

## Tests et intégration continue

```bash
pytest tests -v
```

Sur GitHub : workflow **`.github/workflows/ci.yml`** (Python 3.11, installation des dépendances, exécution de `pytest`).

## Automatisation (Makefile / PowerShell)

| Commande | Rôle |
|----------|------|
| `make install` / `.\tasks.ps1 install` | Installer les dépendances |
| `make test` / `.\tasks.ps1 test` | Lancer les tests unitaires |
| `make notebooks` / `.\tasks.ps1 notebooks` | Exécuter `01_EDA.ipynb` puis `02_Modeling.ipynb` (nbconvert) |
| `make pdf` / `.\tasks.ps1 pdf` | Générer `Presentation.pdf` |
| `.\tasks.ps1 emit02` | Régénérer `notebooks/02_Modeling.ipynb` depuis `scripts/emit_02_notebook.py` |
| `.\tasks.ps1 streamlit` | Lancer l’interface Streamlit (`streamlit run streamlit_app.py`) |

Sous Linux/macOS, `make notebooks` suppose que `jupyter` est dans le `PATH`.

## Lancer les notebooks (manuel)

1. Depuis la racine du projet :  
   `jupyter notebook notebooks/01_EDA.ipynb`  
   puis **`notebooks/02_Modeling.ipynb`**.

2. Le noyau Python doit avoir la **racine du projet** accessible : les notebooks ajoutent automatiquement le répertoire parent à `sys.path` pour importer le package `src/`.

3. Les figures, `model_results.csv`, `drift_report.csv`, matrices de confusion, courbes ROC/PR, calibration, et `best_model.pkl` sont écrits dans **`outputs/`**.

4. Le notebook enregistre aussi **`outputs/feature_columns.json`** (noms des variables attendues par le modèle), utilisé par l’interface Streamlit.

## Interface Streamlit (démo)

Après avoir exécuté **`02_Modeling.ipynb`** (pour `outputs/best_model.pkl` et `feature_columns.json`) et avec **`data/creditcard.csv`** présent :

```bash
streamlit run streamlit_app.py
```

Sous Windows : **`.\tasks.ps1 streamlit`**

L’application affiche des prédictions sur une fenêtre de transactions (probabilité de fraude, seuil réglable, comparaison avec la classe réelle sur l’échantillon) et le tableau `model_results.csv` si disponible.

## Structure du dépôt

```
├── data/                 # creditcard.csv (+ script d’aide au téléchargement)
├── notebooks/
│   ├── 01_EDA.ipynb      # Analyse exploratoire
│   └── 02_Modeling.ipynb # Pipeline complet (split temporel, SMOTE, modèles, seuils, drift, SHAP)
├── src/                  # Config, chemins, features, splits, seuils, dérive, sélection, etc.
├── tests/                # pytest (forme des prédictions, chemins, pas de NaN)
├── outputs/              # Résultats générés (CSV, PNG, modèle)
├── draft/                # Brouillons et anciennes versions de notebooks
├── requirements.txt
├── environment.yml
├── Makefile
├── tasks.ps1
├── Presentation.md       # Support de soutenance (source)
├── Presentation.pdf      # Généré via scripts/build_presentation_pdf.py (ou Pandoc)
├── scripts/
│   ├── build_presentation_pdf.py
│   └── emit_02_notebook.py
└── Rapport_Detection_Fraude_ML.md
```

## Résumé des résultats (indicatif)

Après exécution de `02_Modeling.ipynb`, consulter **`outputs/model_results.csv`**. Les métriques exactes dépendent du jeu de données et du découpage (`RANDOM_STATE` dans `src/paths.py`, autres paramètres dans `src/config.py`). Le meilleur modèle au sens F1 (ou grille) est documenté dans le notebook et dans le CSV.

## Rapport et soutenance

- Rapport long : `Rapport_Detection_Fraude_ML.md`  
- Slides : **`Presentation.pdf`** (généré par `py scripts/build_presentation_pdf.py`, ou éditer `Presentation.md` puis exporter avec Pandoc / Word).  
  Le fichier `Presentation.md` référence les figures sous **`outputs/`** (matrice de confusion, ROC, précision–rappel, calibration, SHAP) : exécutez d’abord le notebook de modélisation pour les générer.

## Licence du jeu de données

Respecter les conditions d’usage du fournisseur (Kaggle / auteurs du dataset ULB).
