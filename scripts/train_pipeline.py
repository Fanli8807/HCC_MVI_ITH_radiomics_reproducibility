from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcc_mvi_ith.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the HCC MVI ITH pipeline.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=ROOT / "config" / "pipeline.yml", type=Path)
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.clinical, args.output, args.config, mode="train")


if __name__ == "__main__":
    main()
