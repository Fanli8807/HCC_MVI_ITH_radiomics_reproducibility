from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, brier_score_loss


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, prob)
    idx = int(np.argmax(tpr - fpr))
    return float(thr[idx])


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else math.nan
    rows = {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else math.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else math.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else math.nan,
        "npv": tn / (tn + fn) if (tn + fn) else math.nan,
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "brier": float(brier_score_loss(y_true, prob)),
    }
    intervals = {
        "accuracy_ci": wilson_ci(int((pred == y_true).sum()), len(y_true)),
        "sensitivity_ci": wilson_ci(int(tp), int(tp + fn)),
        "specificity_ci": wilson_ci(int(tn), int(tn + fp)),
        "ppv_ci": wilson_ci(int(tp), int(tp + fp)),
        "npv_ci": wilson_ci(int(tn), int(tn + fn)),
    }
    rows.update({k: [float(v[0]), float(v[1])] for k, v in intervals.items()})
    return rows


def hosmer_lemeshow(y_true: np.ndarray, prob: np.ndarray, groups: int = 10) -> dict:
    order = np.argsort(prob)
    y = y_true[order]
    p = prob[order]
    bins = np.array_split(np.arange(len(y)), min(groups, len(y)))
    chi2 = 0.0
    used = 0
    for idx in bins:
        obs = y[idx].sum()
        exp = p[idx].sum()
        n = len(idx)
        denom = exp * (1 - exp / n)
        if denom > 0:
            chi2 += (obs - exp) ** 2 / denom
            used += 1
    df = max(used - 2, 1)
    return {"chi2": float(chi2), "df": int(df), "p_value": float(1 - stats.chi2.cdf(chi2, df))}


def decision_curve(y_true: np.ndarray, prob: np.ndarray, thresholds: Iterable[float]) -> list[dict]:
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    rows = []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            continue
        pred = prob >= pt
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        net = tp / n - fp / n * (pt / (1 - pt))
        treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({"threshold": float(pt), "model": float(net), "treat_all": float(treat_all), "treat_none": 0.0})
    return rows


def prevalence_adjusted_ppv_npv(sensitivity: float, specificity: float, prevalences: Iterable[float]) -> list[dict]:
    rows = []
    for prev in prevalences:
        ppv = (sensitivity * prev) / ((sensitivity * prev) + (1 - specificity) * (1 - prev))
        npv = (specificity * (1 - prev)) / (((1 - sensitivity) * prev) + specificity * (1 - prev))
        rows.append({"prevalence": float(prev), "ppv": float(ppv), "npv": float(npv)})
    return rows


def bootstrap_youden_ci(y_true: np.ndarray, prob: np.ndarray, n_boot: int = 1000, random_state: int = 42) -> dict:
    rng = np.random.default_rng(random_state)
    thresholds = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        thresholds.append(youden_threshold(y_true[idx], prob[idx]))
    if not thresholds:
        return {"threshold_ci": [math.nan, math.nan], "n_bootstrap_used": 0}
    lo, hi = np.percentile(thresholds, [2.5, 97.5])
    return {"threshold_ci": [float(lo), float(hi)], "n_bootstrap_used": int(len(thresholds))}



def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks used by the non-parametric DeLong paired AUC test."""
    x = np.asarray(x, dtype=float)
    sort_idx = np.argsort(x, kind="mergesort")
    x_sorted = x[sort_idx]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[sort_idx] = ranks
    return out


def delong_test(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray) -> tuple[float, float]:
    """Paired non-parametric DeLong test for two correlated ROC AUCs.

    Returns
    -------
    z_statistic, two_sided_p_value
    """
    y_true = np.asarray(y_true).astype(int)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_score_a) & np.isfinite(y_score_b)
    y_true, y_score_a, y_score_b = y_true[valid], y_score_a[valid], y_score_b[valid]
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return 0.0, 1.0

    def _components(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        pos = scores[y_true == 1]
        neg = scores[y_true == 0]
        all_scores = np.concatenate([pos, neg])
        all_ranks = _midrank(all_scores)
        pos_ranks = _midrank(pos)
        neg_ranks = _midrank(neg)
        m, n = len(pos), len(neg)
        v10 = (all_ranks[:m] - pos_ranks) / n
        v01 = 1.0 - (all_ranks[m:] - neg_ranks) / m
        return float(v10.mean()), v10, v01

    auc_a, v10_a, v01_a = _components(y_score_a)
    auc_b, v10_b, v01_b = _components(y_score_b)
    var_a = np.var(v10_a, ddof=1) / n_pos + np.var(v01_a, ddof=1) / n_neg
    var_b = np.var(v10_b, ddof=1) / n_pos + np.var(v01_b, ddof=1) / n_neg
    cov = (np.cov(v10_a, v10_b, ddof=1)[0, 1] / n_pos) + (np.cov(v01_a, v01_b, ddof=1)[0, 1] / n_neg)
    var_diff = var_a + var_b - 2.0 * cov
    if not np.isfinite(var_diff) or var_diff <= 0:
        return 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)
