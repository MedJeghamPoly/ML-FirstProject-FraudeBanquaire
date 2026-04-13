import numpy as np

from src.thresholds import best_threshold_cost, cost_at_threshold


def test_cost_threshold_extremes():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    t, c, _ = best_threshold_cost(y_true, y_score, cost_fn=1.0, cost_fp=1.0)
    assert 0 <= t <= 1
    tn, fp, fn, tp, _ = cost_at_threshold(y_true, y_score, t, 1.0, 1.0)
    assert tn + fp + fn + tp == 4
