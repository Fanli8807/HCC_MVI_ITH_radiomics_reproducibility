from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_mvi_ith.survival import concordance_index, cumulative_dynamic_auc


def _km_table(duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    order = np.argsort(duration)
    times, events = duration[order], event[order]
    at_risk = len(times)
    survival = 1.0
    rows = []
    for t in np.unique(times):
        at_t = times == t
        d = int(events[at_t].sum())
        c = int(at_t.sum() - d)
        if at_risk > 0 and d > 0:
            survival *= 1.0 - d / at_risk
        rows.append({"time_months": float(t), "events": d, "censored": c, "at_risk": int(at_risk), "survival": float(survival)})
        at_risk -= int(at_t.sum())
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RFS analysis for locked Fusion probabilities.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--score-col", default="Fusion_probability")
    parser.add_argument("--duration-col", default="RFS_months")
    parser.add_argument("--event-col", default="RFS_event")
    parser.add_argument("--threshold", type=float, default=0.473)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    for col in [args.score_col, args.duration_col, args.event_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")
    args.output.mkdir(parents=True, exist_ok=True)
    score = df[args.score_col].astype(float).values
    duration = df[args.duration_col].astype(float).values
    event = df[args.event_col].astype(int).values
    high = score >= args.threshold
    summary = {
        "n_cases": int(len(df)),
        "threshold": float(args.threshold),
        "n_high_risk": int(high.sum()),
        "n_low_risk": int((~high).sum()),
        "c_index": concordance_index(duration, event, score),
        "auc_24_months": cumulative_dynamic_auc(duration, event, score, 24.0),
        "auc_60_months": cumulative_dynamic_auc(duration, event, score, 60.0),
    }
    _km_table(duration[high], event[high]).to_csv(args.output / "km_high_risk.csv", index=False)
    _km_table(duration[~high], event[~high]).to_csv(args.output / "km_low_risk.csv", index=False)
    with open(args.output / "survival_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Survival outputs written to {args.output}")


if __name__ == "__main__":
    main()
