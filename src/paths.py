"""Chemins du projet : fonctionne depuis la racine ou depuis notebooks/."""

from pathlib import Path


def get_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "notebooks":
        return cwd.parent
    if (cwd / "notebooks").is_dir() and (cwd / "data").is_dir():
        return cwd
    # pytest / IDE peut avoir cwd = tests/
    p = cwd.parent
    if (p / "notebooks").is_dir() and (p / "data").is_dir():
        return p
    return cwd


PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
# Sorties : figures, CSV, modèle sauvegardé
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
GRIDSEARCH_TRAIN_SAMPLES = 25_000
CV_FOLDS = 2
CV_SCORING_FOLDS = 3
CV_SCORING_MAX_SAMPLES = 35_000

FIG_KW = {"dpi": 150, "bbox_inches": "tight"}
