# Methods Crosswalk

This file maps the study methods and supplementary material to the code. It is intended to help users verify that the released implementation reflects the documented workflow rather than a simplified post-hoc analysis.

| Study / supplement item | Code implementation |
|---|---|
| Prespecified split and leakage prevention | `pipeline.py` reads a `split` column when supplied. GMM fitting, feature selection, classifier benchmarking, model fitting, and threshold derivation use the training cases only. |
| Center 4 as held-out external test set | Supported through `split = external_test` in the clinical CSV; external cases are evaluated after model fitting and are not used to estimate model parameters. |
| GTR ROI = tumor + 5-mm peritumoral margin | `roi.construct_gtr()` performs 3D dilation using image spacing. Optional liver, vessel, and bile-duct masks can constrain the ROI if available. |
| Macroscopic imaging descriptors distinct from GTR radiomics | Clinical variables are listed separately in `config/pipeline.yml`; GTR radiomics are extracted from imaging ROI masks. |
| 24-dimensional voxel-wise GMM input | `local_features.extract_local_feature_matrix()` computes local mean, standard deviation, energy, entropy, skewness, and kurtosis for DWI, A, V, and D. |
| GMM habitat model with k = 3 | `config/pipeline.yml` sets `n_clusters: 3`; `habitat.fit_habitat_model()` fits the training-set GMM; `predict_habitat_labels()` applies frozen centroids. |
| k search over candidate cluster numbers | `habitat.evaluate_k_range()` exports `gmm_k_search.csv` with Calinski-Harabasz and silhouette scores. |
| ITH closed-form fragmentation score | `habitat.ith_score()` implements `1 - (1 / S_total) * sum_i(S_i,max / n_i)` using 26-connectivity. |
| PyRadiomics settings | `config/pyradiomics.yml` records bin width 25, original plus derived image types, and the reported feature classes. In `train` mode, the pipeline calls PyRadiomics for GTR and ITH habitat masks. A lightweight fallback extractor is restricted to the synthetic demo run. |
| Clinical covariates | `config/pipeline.yml` lists HBV infection, maximum tumor diameter, non-smooth tumor margin, and pseudocapsule. |
| Feature selection | `feature_selection.py` implements univariable t-test screening, Pearson correlation pruning, and LASSO logistic regression cross-validation. |
| Classifier benchmarking | `models.py` encodes the nine classifier families and Supplementary Table S5 hyperparameter search spaces, with optional XGBoost and LightGBM. |
| Locked Fusion threshold | `config/pipeline.yml` records the published threshold 0.473; `predict_locked.py` applies it to exported prediction tables. |
| Diagnostic metrics | `metrics.py` computes AUC, accuracy, sensitivity, specificity, PPV, NPV, Wilson CIs, Brier score, Hosmer-Lemeshow, DCA, bootstrap Youden CI, and prevalence-shift PPV/NPV. When a `split` column is supplied, `metrics.json` reports split-specific validation and external-test performance rather than relying on pooled all-case metrics. |
| Hosmer-Lemeshow table consistency check | `scripts/check_hosmer_lemeshow.py` verifies chi-square / P-value consistency, e.g., Chi2 = 14.56 with df = 8 gives P = 0.068287. |
| Survival extension | `survival.py` provides C-index and fixed-time AUC helpers for RFS analyses. |
| SHAP / interpretability | Top features and plain-language interpretations are documented in the supplement; model artifacts exported by the full run can be used for SHAP analysis in environments with `shap` installed. |
| Paired AUC comparison / DeLong testing | `metrics.delong_test()` and `scripts/compare_delong.py` provide a paired DeLong helper for probability columns exported by the pipeline. |
| Optional RFS analysis | `scripts/survival_analysis.py` uses Fusion probabilities, RFS months, event status, and the locked threshold to export C-index, 24-/60-month AUCs, and KM tables. |
| Optional SHAP summary | `scripts/shap_interpretability.py` uses the saved Fusion model artifact and `features.csv`; it writes a compact JSON summary when SHAP is installed and exits safely otherwise. |
