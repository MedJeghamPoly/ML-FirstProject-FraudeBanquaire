from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.shap_utils import estimator_supports_linear_shap, estimator_supports_tree_shap


def test_tree_shap_support():
    assert estimator_supports_tree_shap(DecisionTreeClassifier()) is True
    assert estimator_supports_tree_shap(RandomForestClassifier(n_estimators=2)) is True
    assert estimator_supports_tree_shap(LogisticRegression()) is False


def test_linear_shap_support():
    assert estimator_supports_linear_shap(LogisticRegression()) is True
    assert estimator_supports_linear_shap(RandomForestClassifier(n_estimators=2)) is False
