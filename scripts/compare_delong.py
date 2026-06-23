from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_mvi_ith.io import save_json
from hcc_mvi_ith.metrics import delong_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired DeLong comparison of two probability columns.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--label-col", default="MVI_label")
    parser.add_argument("--model-a", required=True, help="Probability column for model A, e.g. Fusion_probability")
    parser.add_argument("--model-b", required=True, help="Probability column for model B, e.g. Clinical_probability")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    y = df[args.label_col].astype(int).values
    a = df[args.model_a].astype(float).values
    b = df[args.model_b].astype(float).values
    z, p = delong_test(y, a, b)
    save_json({"model_a": args.model_a, "model_b": args.model_b, "z": z, "p_value": p}, args.output)


if __name__ == "__main__":
    main()
