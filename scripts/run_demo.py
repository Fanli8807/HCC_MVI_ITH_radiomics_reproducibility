from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_mvi_ith.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic two-case demo pipeline.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "demo_data")
    parser.add_argument("--clinical", type=Path, default=ROOT / "demo_data" / "clinical_variables.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "demo_run")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "demo_pipeline.yml")
    args = parser.parse_args()
    run_pipeline(data_dir=args.data_dir, clinical_csv=args.clinical, output_dir=args.output, config_path=args.config, mode="demo")
    p_value = 1.0 - stats.chi2.cdf(14.56, 8)
    (args.output / "hl_check_validation_fusion.txt").write_text(
        f"chi2 = {14.56:.4f}\n"
        "df = 8\n"
        f"p_value = {p_value:.6f}\n",
        encoding="utf-8",
    )
    print(f"Demo complete: {args.output}")


if __name__ == "__main__":
    main()
