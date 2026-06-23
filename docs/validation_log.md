# Local validation log

This package was checked after reproducibility hardening.

- Static compilation: passed for `src/`, `scripts/`, and `tests/`.
- Core test functions: 9 direct assertions passed for metrics, ITH score, DeLong, and survival helpers.
- Reproducibility guard tests: passed for training-only feature imputation and demo-only `t1c.nii.gz` aliasing.
- Synthetic demo demo run: `python scripts/run_demo.py --output results/demo_run_fixed` completed and generated `features.csv`, `predictions.csv`, `metrics.json`, habitat maps, figures, and `hl_check_validation_fusion.txt`.
- Hosmer-Lemeshow consistency helper: `chi2 = 14.56`, `df = 8`, `p_value = 0.068287`.
- Train-mode phase safeguard: `scripts/train_pipeline.py` rejects the bundled demo folders in train mode because `ap.nii.gz`, `pvp.nii.gz`, and `dp.nii.gz` are absent; demo-only `t1c.nii.gz` aliasing is not used for full-cohort runs.

The validation used only the bundled synthetic demo data and did not include any patient-level data.
