import numpy as np

from hcc_mvi_ith.metrics import classification_metrics, prevalence_adjusted_ppv_npv, wilson_ci


def test_wilson_ci_is_bounded():
    lo, hi = wilson_ci(5, 10)
    assert 0 <= lo <= hi <= 1


def test_classification_metrics_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    out = classification_metrics(y, p, 0.5)
    assert out["tp"] == 2
    assert out["tn"] == 2
    assert out["auc"] == 1.0


def test_prevalence_projection_shape():
    rows = prevalence_adjusted_ppv_npv(0.876, 0.855, [0.2, 0.5])
    assert len(rows) == 2
    assert rows[0]["npv"] > rows[0]["ppv"]
