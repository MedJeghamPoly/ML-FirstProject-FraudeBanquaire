import numpy as np
import pandas as pd

from src.splits import temporal_xy_split_ordered


def test_temporal_split_preserves_order():
    n = 100
    X = pd.DataFrame({"a": np.arange(n), "b": np.random.randn(n)})
    y = pd.Series([0] * 90 + [1] * 10)
    Xt, Xv, yt, yv = temporal_xy_split_ordered(X, y, test_size=0.2)
    assert len(Xt) == 80 and len(Xv) == 20
    assert Xt["a"].iloc[0] == 0
    assert Xv["a"].iloc[0] == 80
