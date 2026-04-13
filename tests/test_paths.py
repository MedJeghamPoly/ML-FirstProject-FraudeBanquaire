from pathlib import Path

from src.paths import DATA_PATH, OUTPUT_DIR, get_project_root


def test_paths_exist_types():
    assert isinstance(DATA_PATH, Path)
    assert isinstance(OUTPUT_DIR, Path)
    root = get_project_root()
    assert (root / "src").is_dir() or root.name == "src"
