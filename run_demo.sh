#!/usr/bin/env bash
set -euo pipefail
python scripts/run_demo.py
python scripts/check_hosmer_lemeshow.py --chi2 14.56 --df 8 > results/demo_run/hl_check_validation_fusion.txt
PYTHONPATH=src pytest -q tests
