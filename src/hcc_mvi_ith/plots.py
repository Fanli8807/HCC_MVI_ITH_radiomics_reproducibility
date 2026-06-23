from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

from .metrics import decision_curve


def save_basic_figures(y_true: np.ndarray, prob: np.ndarray, out_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, prob)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "roc_fusion.png")
    plt.close(fig)

    frac, mean = calibration_curve(y_true, prob, n_bins=min(5, len(y_true)), strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(mean, frac, marker="o")
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event fraction")
    fig.tight_layout()
    fig.savefig(out / "calibration_fusion.png")
    plt.close(fig)

    dca = pd.DataFrame(decision_curve(y_true, prob, np.linspace(0.05, 0.95, 19)))
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(dca["threshold"], dca["model"], label="Fusion")
    ax.plot(dca["threshold"], dca["treat_all"], label="Treat all", ls="--")
    ax.plot(dca["threshold"], dca["treat_none"], label="Treat none", ls=":")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decision_curve_fusion.png")
    plt.close(fig)
