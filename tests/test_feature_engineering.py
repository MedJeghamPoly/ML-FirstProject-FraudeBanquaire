import pandas as pd

from src.feature_engineering import add_engineered_features


def test_add_log_column():
    df = pd.DataFrame(
        {
            "Time": [0, 10, 20],
            "Amount": [1.0, 100.0, 50.0],
            "Class": [0, 1, 0],
        }
    )
    out = add_engineered_features(df)
    assert "Amount_log" in out.columns
    assert "Amount_to_roll_ratio" in out.columns or "Amount_roll_mean_past" in out.columns
