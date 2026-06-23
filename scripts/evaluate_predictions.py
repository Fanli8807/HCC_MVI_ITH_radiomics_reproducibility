from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_mvi_ith.io import save_json
from hcc_mvi_ith.metrics import (
    bootstrap_youden_ci,
    classification_metrics,
    hosmer_lemeshow,
    prevalence_adjusted_ppv_npv,
    youden_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate locked model probabilities.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--label-col", default="MVI_label")
    parser.add_argument("--prob-col", default="Fusion_probability")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    y = df[args.label_col].astype(int).values
    p = df[args.prob_col].astype(float).values
    threshold = args.threshold if args.threshold is not None else youden_threshold(y, p)
    metrics = {
        "threshold": float(threshold),
        "classification": classification_metrics(y, p, threshold),
        "hosmer_lemeshow": hosmer_lemeshow(y, p),
        "threshold_bootstrap": bootstrap_youden_ci(y, p, n_boot=1000),
    }
    sens = metrics["classification"]["sensitivity"]
    spec = metrics["classification"]["specificity"]
    if np.isfinite(sens) and np.isfinite(spec):
        metrics["prevalence_shift_ppv_npv"] = prevalence_adjusted_ppv_npv(sens, spec, [0.20, 0.30, 0.40, 0.50])
    save_json(metrics, args.output)


if __name__ == "__main__":
    main()
