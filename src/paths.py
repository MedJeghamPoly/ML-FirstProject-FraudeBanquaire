"""Chemins du projet : fonctionne depuis la racine ou depuis notebooks/."""

from pathlib import Path


def get_project_root() -> Path:
    """Racine du dépôt (dossier contenant ``src/paths.py``), même si cwd = notebooks/ ou tests/."""
    cwd = Path.cwd().resolve()
    marker = Path("src") / "paths.py"
    for base in (cwd, cwd.parent, cwd.parent.parent):
        if (base / marker).is_file():
            return base
    if cwd.name == "notebooks":
        return cwd.parent
    if (cwd / "notebooks").is_dir() and (cwd / "src").is_dir():
        return cwd
    p = cwd.parent
    if (p / "notebooks").is_dir() and (p / "src").is_dir():
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
