#!/usr/bin/env python
"""Small utility for checking Hosmer-Lemeshow chi-square / P-value consistency.

Example:
    python scripts/check_hosmer_lemeshow.py --chi2 14.56 --df 8
Expected output:
    p_value = 0.068287
"""
from __future__ import annotations

import argparse
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi2", type=float, required=True, help="Hosmer-Lemeshow chi-square statistic")
    parser.add_argument("--df", type=int, default=8, help="Degrees of freedom; 10 risk groups usually gives df = 8")
    args = parser.parse_args()
    p = 1.0 - stats.chi2.cdf(args.chi2, args.df)
    print(f"chi2 = {args.chi2:.4f}")
    print(f"df = {args.df}")
    print(f"p_value = {p:.6f}")


if __name__ == "__main__":
    main()
