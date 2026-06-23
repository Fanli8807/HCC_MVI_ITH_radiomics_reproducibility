from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_mvi_ith.interpretability import compute_shap_summary, shap_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional SHAP summary for the fitted Fusion model.")
    parser.add_argument("--artifacts", required=True, type=Path, help="model_artifacts.joblib from the pipeline.")
    parser.add_argument("--features", required=True, type=Path, help="features.csv from the pipeline.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = joblib.load(args.artifacts)
    model = artifacts.get("models", {}).get("Fusion")
    if model is None:
        raise SystemExit("Fusion model not found in artifact file.")
    df = pd.read_csv(args.features)
    x = df[model.features].values
    summary = compute_shap_summary(model, x, model.features, max_features=20)
    out = {"shap_available": shap_available(), "n_features": len(model.features), "top_features": summary, "note": "Tree-based SHAP summary is returned when the fitted Fusion classifier supports TreeExplainer; otherwise the list is intentionally empty to avoid slow optional KernelExplainer runs."}
    with open(args.output / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"SHAP summary written to {args.output / 'shap_summary.json'}")


if __name__ == "__main__":
    main()
