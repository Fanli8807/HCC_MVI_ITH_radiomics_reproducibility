from __future__ import annotations

import numpy as np


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def compute_shap_summary(fitted_model, x: np.ndarray, feature_names: list[str], max_features: int = 20) -> list[dict]:
    """Return a compact mean(|SHAP|) table for tree-based Fusion models.

    The helper deliberately avoids slow KernelExplainer calculations by default so that
    optional interpretability checks cannot hang on small synthetic demos. For non-tree models,
    it returns an empty list and the caller records that SHAP was unavailable for that
    estimator type. Full cohort SHAP figures can be generated from tree-based locked
    model artifacts in an environment with `shap` installed.
    """
    if not shap_available():
        return []
    try:
        import shap
        estimator = fitted_model.estimator
        if hasattr(estimator, "named_steps"):
            scaler = estimator.named_steps.get("scaler")
            clf = estimator.named_steps.get("clf", estimator)
            x_in = scaler.transform(x) if scaler is not None else x
        else:
            clf = estimator
            x_in = x
        clf_name = type(clf).__name__.lower()
        tree_like = any(t in clf_name for t in ("xgb", "lgbm", "forest", "tree"))
        if not tree_like:
            return []
        explainer = shap.TreeExplainer(clf)
        vals = explainer.shap_values(x_in)
        if isinstance(vals, list):
            vals = vals[-1]
        mean_abs = np.abs(np.asarray(vals)).mean(axis=0)
    except Exception:
        return []
    order = np.argsort(-mean_abs)[:max_features]
    return [
        {"rank": int(i + 1), "feature": str(feature_names[j]), "mean_abs_shap": float(mean_abs[j])}
        for i, j in enumerate(order)
    ]
