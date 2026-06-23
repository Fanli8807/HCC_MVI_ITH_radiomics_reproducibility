import numpy as np

from hcc_mvi_ith.habitat import ith_score


def test_ith_score_zero_for_unfragmented_labels():
    labels = np.zeros((3, 3, 3), dtype=np.int16)
    mask = np.zeros_like(labels, dtype=bool)
    labels[0, 0, 0] = 1
    labels[0, 0, 1] = 1
    labels[2, 2, 2] = 2
    labels[2, 2, 1] = 2
    mask = labels > 0
    assert ith_score(labels, mask, n_labels=2) == 0.0


def test_ith_score_increases_with_fragmentation():
    labels = np.zeros((5, 5, 5), dtype=np.int16)
    labels[0, 0, 0] = 1
    labels[4, 4, 4] = 1
    labels[2, 2, 2] = 2
    mask = labels > 0
    expected = 1.0 - ((1 / 2) + 1) / 3
    assert abs(ith_score(labels, mask, n_labels=2) - expected) < 1e-9
