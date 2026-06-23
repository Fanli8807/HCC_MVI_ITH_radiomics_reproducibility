import numpy as np
import pytest

from hcc_mvi_ith.metrics import delong_test
from hcc_mvi_ith.survival import concordance_index, cumulative_dynamic_auc


def test_delong_identical_models_give_p_close_to_one():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + 0.4 * y
    z, p = delong_test(y, s, s)
    assert abs(z) < 1e-6
    assert p > 0.99


def test_delong_returns_finite_p_for_two_models():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 300)
    a = rng.normal(size=300) + 0.7 * y
    b = rng.normal(size=300) + 0.2 * y
    z, p = delong_test(y, a, b)
    assert np.isfinite(z)
    assert 0.0 <= p <= 1.0


def test_concordance_index_perfect_ordering():
    time = np.array([10, 20, 30, 40, 50], dtype=float)
    event = np.array([1, 1, 1, 1, 1])
    score = np.array([5, 4, 3, 2, 1], dtype=float)
    assert concordance_index(time, event, score) == pytest.approx(1.0)


def test_fixed_time_auc_has_two_valid_horizons():
    rng = np.random.default_rng(42)
    n = 200
    score = rng.uniform(0, 1, n)
    time = (1 - score) * 60 + rng.normal(0, 5, n)
    event = (rng.uniform(size=n) < 0.8).astype(int)
    assert np.isfinite(cumulative_dynamic_auc(time, event, score, 24.0))
    assert np.isfinite(cumulative_dynamic_auc(time, event, score, 60.0))
