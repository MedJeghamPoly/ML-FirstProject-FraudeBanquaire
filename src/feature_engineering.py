"""Features dérivées : log(Amount), moyenne mobile causale (sans regarder le futur)."""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame, time_col: str = "Time", amount_col: str = "Amount") -> pd.DataFrame:
    """
    - Amount_log = log1p(Amount)
    - Après tri par temps : moyenne mobile d'Amount sur le passé uniquement (shift(1)),
      puis ratio Amount / (roll + eps).
    """
    out = df.copy()
    if amount_col in out.columns:
        out["Amount_log"] = np.log1p(out[amount_col].clip(lower=0))

    if time_col in out.columns and amount_col in out.columns:
        out = out.sort_values(time_col).reset_index(drop=True)
        w = 5000
        roll = out[amount_col].rolling(window=w, min_periods=1).mean().shift(1)
        roll = roll.fillna(out[amount_col].expanding().mean().shift(1))
        eps = 1e-6
        out["Amount_roll_mean_past"] = roll
        out["Amount_to_roll_ratio"] = out[amount_col] / (roll.abs() + eps)

    return out
