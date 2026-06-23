# HCC MVI ITH Radiomics Reproducibility Package

This repository provides reproducibility code for the study:

**Preoperative MRI-based intratumoral heterogeneity radiomics for predicting microvascular invasion and stratifying recurrence-free survival in hepatocellular carcinoma**

The package is designed for methodological verification under privacy constraints. It runs end-to-end on fully synthetic demonstration data and uses the same entry points for institutional patient-level data when such data are available under appropriate ethics and data-use approvals.

## Important scope statement

Patient-level MRI, clinical tables, original fitted coefficients, and institution-specific model artifacts are **not bundled**. This is intentional: the study data are restricted by privacy, institutional governance, and ethics approvals. The included `demo_data/` contains two synthetic, format-compatible cases only. The demo verifies executability, data flow, leakage-control logic, and output structure; it is **not** intended to reproduce the reported AUC, C-index, survival curves, or subgroup estimates.

## What this package verifies

- training-set-only fitting of scaling, GMM, feature-selection, model, and threshold parameters;
- locked validation and external testing without using held-out data for feature selection, classifier benchmarking, hyperparameter tuning, or Youden-threshold derivation;
- tumor + 5-mm peritumoral GTR construction;
- 24-dimensional voxel-wise local feature matrix for GMM habitat assignment;
- k = 3 GMM habitat model and locked habitat assignment rules;
- closed-form 3D ITH fragmentation score using 26-connectivity;
- GTR, ITH, GTR-ITH, Clinical, and Fusion model blocks;
- diagnostic metrics, calibration, decision-curve analysis, DeLong comparison helper, threshold bootstrap, and prevalence-adjusted PPV/NPV;
- optional RFS and SHAP helper scripts for full-cohort analyses.

## Repository structure

```text
config/                    analysis and demo pipeline settings
scripts/                   executable entry points
docs/                      data dictionary, methods crosswalk, reproducibility checklist
src/hcc_mvi_ith/           implementation modules
tests/                     unit tests for key statistics and ITH score
demo_data/                 fully synthetic two-case NIfTI demo data
```

## Quick start

```bash
conda env create -f environment.yml
conda activate hcc-mvi-ith
bash run_demo.sh
```

or, with an existing Python environment:

```bash
pip install -r requirements.txt
python scripts/run_demo.py
PYTHONPATH=src pytest -q tests
```

On Windows, `python scripts/run_demo.py` is the preferred cross-platform demo entry
point. It now also writes the Hosmer-Lemeshow consistency check that `run_demo.sh`
creates on Unix-like systems.

Expected outputs are generated under `results/demo_run/` after running the demo:

```text
features.csv
predictions.csv
metrics.json
gmm_k_search.csv
habitat_maps/
figures/
model_artifacts.joblib
hl_check_validation_fusion.txt
```

The Hosmer-Lemeshow helper can also be run independently:

```bash
python scripts/check_hosmer_lemeshow.py --chi2 14.56 --df 8
# p_value = 0.068287
```

## Full cohort run

Full-cohort training requires the complete environment in `environment.yml`,
including SimpleITK and PyRadiomics. In `train` mode the pipeline uses
PyRadiomics with `config/pyradiomics.yml`; the lightweight fallback extractor is
restricted to the synthetic demo run and is not used for full-cohort
feature extraction.

Prepare a case-folder structure and clinical table according to `docs/data_dictionary.md`. If the clinical CSV includes a `split` column with values `train`, `validation`, and `external_test`, the training split is the only split used for GMM fitting, feature selection, classifier tuning, model fitting, and threshold derivation.

```bash
python scripts/train_pipeline.py \
  --data-dir /path/to/case_folders \
  --clinical /path/to/clinical_variables.csv \
  --output results/full_run \
  --config config/pipeline.yml \
  --mode train
```

When `split` is supplied, `metrics.json` reports Fusion performance separately
for `train`, `validation`, and `external_test` at both the training-derived
Youden threshold and the published locked threshold. These split-specific
metrics should be used for reported performance tables rather than pooled all-case
performance.

## Optional analysis helpers

```bash
python scripts/compare_delong.py \
  --predictions results/full_run/predictions.csv \
  --label-col MVI_label \
  --model-a Fusion_probability \
  --model-b Clinical_probability \
  --output results/full_run/delong_fusion_vs_clinical.json

python scripts/survival_analysis.py \
  --input results/full_run/predictions_with_rfs.csv \
  --score-col Fusion_probability \
  --duration-col RFS_months \
  --event-col RFS_event \
  --threshold 0.473 \
  --output results/full_run/survival

python scripts/shap_interpretability.py \
  --artifacts results/full_run/model_artifacts.joblib \
  --features results/full_run/features.csv \
  --output results/full_run/shap
```

## Method constants

- Random seed: 42.
- GTR ROI: tumor mask + 5-mm peritumoral margin.
- GMM habitat number: k = 3.
- GMM input: six local first-order statistics in a 3 × 3 × 3 neighborhood across DWI, arterial, portal venous, and delayed phases.
- ITH score: `1 - (1 / S_total) * sum_i(S_i,max / n_i)`, using 26-connectivity.
- PyRadiomics bin width: 25.
- Clinical covariates: HBV infection, maximum tumor diameter, non-smooth tumor margin, and pseudocapsule.
- Published Fusion threshold: 0.473.

## Reproducibility safeguards

- The external test set is never used during scaling, GMM fitting, feature selection, classifier benchmarking, hyperparameter tuning, model fitting, or threshold derivation.
- Scaling and GMM parameters are fitted on the training set and then frozen.
- Validation and external test sets are used only for performance evaluation.
- No postoperative pathological variable enters the predictive models; histologic differentiation is handled only as a baseline/subgroup variable.
- The demo deliberately reports that it is synthetic and underpowered, preventing over-claiming of clinical performance from the bundled data.

See `docs/methods_crosswalk.md` and `docs/reproducibility_checklist.md` for analysis-to-code mapping.
