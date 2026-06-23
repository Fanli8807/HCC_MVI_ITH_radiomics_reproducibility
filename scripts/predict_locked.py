from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the published locked Fusion threshold to probability outputs.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prob-col", default="Fusion_probability")
    parser.add_argument("--threshold", type=float, default=0.473)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    df["fusion_risk_group_locked"] = df[args.prob_col].ge(args.threshold).map({True: "high", False: "low"})
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
