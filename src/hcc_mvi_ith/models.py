from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class FittedModel:
    estimator: BaseEstimator
    features: list[str]
    train_auc: float | None
    selected_classifier: str


def classifier_grids(random_state: int = 42) -> Dict[str, tuple[BaseEstimator, dict]]:
    grids: Dict[str, tuple[BaseEstimator, dict]] = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, random_state=random_state),
            {
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l1", "l2"],
                "clf__solver": ["liblinear"],
            },
        ),
        "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}),
        "Support Vector Machine": (
            SVC(probability=True, random_state=random_state),
            {"clf__C": [0.1, 1, 10, 100], "clf__kernel": ["linear", "rbf", "poly"], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"clf__max_depth": [3, 5, 7, 10, None], "clf__min_samples_split": [2, 5, 10], "clf__criterion": ["gini", "entropy"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=random_state),
            {"clf__n_estimators": [100, 300, 500], "clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 3, 5]},
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=random_state),
            {"clf__n_estimators": [100, 300, 500], "clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 3, 5]},
        ),
        "MLP": (
            MLPClassifier(max_iter=1000, random_state=random_state),
            {
                "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                "clf__activation": ["relu", "tanh"],
                "clf__alpha": [0.0001, 0.001, 0.01],
                "clf__learning_rate_init": [0.001, 0.01],
            },
        ),
    }
    try:
        from xgboost import XGBClassifier

        grids["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=random_state),
            {
                "clf__n_estimators": [100, 300, 500],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        grids["LightGBM"] = (
            LGBMClassifier(random_state=random_state),
            {
                "clf__num_leaves": [15, 31, 63],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__n_estimators": [100, 300, 500],
                "clf__min_child_samples": [10, 20, 30],
            },
        )
    except Exception:
        pass
    return grids


def _can_cross_validate(y: np.ndarray, requested_folds: int) -> bool:
    counts = np.bincount(y.astype(int), minlength=2)
    return int(counts.min()) >= 2 and len(y) >= requested_folds


def fit_binary_model(
    x: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
    prefer: str | None = None,
) -> FittedModel:
    features = list(x.columns)
    if not features:
        raise ValueError("No features supplied")
    if not _can_cross_validate(y, cv_folds):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="liblinear", random_state=random_state)),
            ]
        )
        estimator.fit(x.values, y)
        prob = estimator.predict_proba(x.values)[:, 1]
        auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else None
        return FittedModel(estimator=estimator, features=features, train_auc=auc, selected_classifier="Logistic Regression (demo)")

    cv = min(cv_folds, int(np.bincount(y.astype(int), minlength=2).min()))
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grids = classifier_grids(random_state)
    if prefer and prefer in grids:
        grids = {prefer: grids[prefer], **{k: v for k, v in grids.items() if k != prefer}}

    best = None
    best_name = None
    best_score = -np.inf
    for name, (clf, grid) in grids.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=splitter, n_jobs=-1, error_score="raise")
        try:
            search.fit(x.values, y)
        except Exception:
            continue
        if search.best_score_ > best_score:
            best = search.best_estimator_
            best_name = name
            best_score = float(search.best_score_)
    if best is None:
        return fit_binary_model(x, y, cv_folds=1, random_state=random_state)
    best.fit(x.values, y)
    return FittedModel(estimator=best, features=features, train_auc=best_score, selected_classifier=best_name or "unknown")


def predict_probability(model: FittedModel, frame: pd.DataFrame) -> np.ndarray:
    x = frame[model.features].values
    return model.estimator.predict_proba(x)[:, 1]
