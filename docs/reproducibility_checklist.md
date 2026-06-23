# Reproducibility Checklist

This checklist summarizes the safeguards most relevant to the study.

## Data-leakage safeguards

- Cohort splits are defined before model development.
- Training cases only are used to fit normalization/scaling parameters.
- Training cases only are used to fit GMM habitat centroids.
- Training cases only are used for univariable filtering, correlation pruning, and LASSO selection.
- Training cases only are used for classifier benchmarking, hyperparameter tuning, model fitting, and Youden-threshold derivation.
- Training cases only are used to estimate feature-imputation medians before applying the locked values to validation and external test cases.
- Validation and external test cases are evaluated after parameter locking.
- No postoperative pathological variable is used as a model predictor.

## Study constants represented in code

- GTR dilation: 5 mm.
- Habitat clusters: k = 3.
- GMM feature vector: 24 dimensions.
- ITH score: 3D connected-component fragmentation index.
- Connectivity: 26-connected 3D neighborhoods.
- PyRadiomics bin width: 25.
- Published Fusion threshold: 0.473.

## What the synthetic demo can and cannot show

The synthetic demo can verify that the pipeline runs, produces features, trains placeholder models, exports predictions, creates habitat maps, and calculates metrics. It cannot reproduce reported AUCs, C-indices, survival curves, or subgroup estimates because it contains only two artificial cases.

The synthetic demo intentionally uses a lightweight feature extractor. Full-cohort
`train` mode requires PyRadiomics/SimpleITK and applies the PyRadiomics settings
in `config/pyradiomics.yml`.

## Privacy and artifact-scope safeguards

- The release contains only synthetic demo data.
- No patient-level MRI, clinical table, original fitted coefficients, or institution-specific model artifact is bundled.
- Demo outputs are generated locally by `run_demo.sh`; they should not be interpreted as reported performance.
