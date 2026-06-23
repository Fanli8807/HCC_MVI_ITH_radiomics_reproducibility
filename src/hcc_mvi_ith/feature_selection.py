from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV


@dataclass
class FeatureImputer:
    columns: list[str]
    medians: pd.Series


def clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in x.columns:
        median = x[col].median()
        x[col] = x[col].fillna(0.0 if pd.isna(median) else median)
    return x


def fit_feature_imputer(df: pd.DataFrame) -> FeatureImputer:
    x = df.copy().replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True).reindex(x.columns).fillna(0.0)
    return FeatureImputer(columns=list(x.columns), medians=medians)


def apply_feature_imputer(df: pd.DataFrame, imputer: FeatureImputer) -> pd.DataFrame:
    x = df.reindex(columns=imputer.columns).copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.fillna(imputer.medians).fillna(0.0)


def univariable_t_filter(x: pd.DataFrame, y: np.ndarray, p_threshold: float) -> list[str]:
    if len(np.unique(y)) < 2:
        return list(x.columns)
    keep = []
    for col in x.columns:
        a = x.loc[y == 0, col].values
        b = x.loc[y == 1, col].values
        if np.nanstd(a) == 0 and np.nanstd(b) == 0:
            continue
        try:
            p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p = 1.0
        if np.isfinite(p) and p < p_threshold:
            keep.append(col)
    return keep or list(x.columns)


def correlation_prune(x: pd.DataFrame, threshold: float) -> list[str]:
    if x.shape[1] <= 1:
        return list(x.columns)
    corr = x.corr().abs().fillna(0.0)
    keep = []
    removed = set()
    for col in corr.columns:
        if col in removed:
            continue
        keep.append(col)
        high = corr.index[(corr[col] > threshold) & (corr.index != col)]
        removed.update(high)
    return keep


def lasso_select(
    x: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 10,
    random_state: int = 42,
    max_demo_features: int = 12,
) -> list[str]:
    counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(counts.min())
    if min_class < 2 or x.shape[0] < 10:
        scores = {}
        for col in x.columns:
            scores[col] = abs(float(x.loc[y == 1, col].mean() - x.loc[y == 0, col].mean()))
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[: min(max_demo_features, len(ranked))]
    cv = min(cv_folds, min_class)
    model = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        max_iter=5000,
        random_state=random_state,
        n_jobs=None,
    )
    model.fit(x.values, y)
    coef = np.abs(model.coef_[0])
    selected = [col for col, c in zip(x.columns, coef) if c > 1e-8]
    return selected or list(x.columns)


def select_features(
    x: pd.DataFrame,
    y: np.ndarray,
    p_threshold: float = 0.05,
    corr_threshold: float = 0.90,
    lasso_cv_folds: int = 10,
    random_state: int = 42,
    max_demo_features: int = 12,
) -> tuple[pd.DataFrame, list[str]]:
    x = clean_feature_frame(x)
    cols = univariable_t_filter(x, y, p_threshold)
    x = x[cols]
    cols = correlation_prune(x, corr_threshold)
    x = x[cols]
    cols = lasso_select(x, y, lasso_cv_folds, random_state, max_demo_features)
    return x[cols], cols
