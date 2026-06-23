from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def concordance_index(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    """Harrell-style C-index with higher score indicating higher risk."""
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                concordant += 1 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0
            elif event[j] == 1 and time[j] < time[i]:
                comparable += 1
                concordant += 1 if score[j] > score[i] else 0.5 if score[i] == score[j] else 0
    return float(concordant / comparable) if comparable else float("nan")


def cumulative_dynamic_auc(time: np.ndarray, event: np.ndarray, score: np.ndarray, horizon: float) -> float:
    """Simple cumulative/dynamic AUC at a fixed time horizon."""
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    valid = cases | controls
    if len(np.unique(cases[valid].astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(cases[valid].astype(int), score[valid]))
