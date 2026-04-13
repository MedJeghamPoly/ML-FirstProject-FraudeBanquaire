"""Paramètres d'expérience (validation temporelle, coûts métier, options analyse)."""

# True = découpage selon Time (80 % premiers instants / 20 % derniers)
USE_TEMPORAL_SPLIT = True
TIME_COL = "Time"

# Coûts relatifs pour optimisation de seuil (PR + coût total)
COST_FN = 10.0  # fraude manquée
COST_FP = 1.0  # fausse alerte

# Sélection de variables (L1 sur échantillon train)
ENABLE_FEATURE_SELECTION = True
L1_MAX_FEATURES = 25
L1_SAMPLE_ROWS = 40_000

# Analyses post-modèle
ENABLE_DRIFT_KS = True
ENABLE_PR_COST_CURVE = True
ENABLE_CALIBRATION_PLOT = True
ENABLE_CALIBRATED_MODEL = False  # True = refit CalibratedClassifierCV (plus long)
ENABLE_SHAP = True
SHAP_MAX_BACKGROUND = 2000
SHAP_MAX_EXPLAIN = 500
