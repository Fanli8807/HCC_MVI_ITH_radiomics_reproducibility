from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_selection import apply_feature_imputer, fit_feature_imputer, select_features
from .habitat import evaluate_k_range, fit_habitat_model, habitat_region_features, predict_habitat_labels
from .io import (
    discover_case_dirs,
    find_phase_paths,
    load_clinical_table,
    load_yaml,
    optional_mask,
    read_nifti,
    require_columns,
    save_json,
    write_nifti_like,
)
from .local_features import extract_local_feature_matrix
from .metrics import (
    bootstrap_youden_ci,
    classification_metrics,
    hosmer_lemeshow,
    prevalence_adjusted_ppv_npv,
    youden_threshold,
)
from .models import fit_binary_model, predict_probability
from .plots import save_basic_figures
from .preprocessing import align_channels_to_mask, align_mask_to_reference, robust_clip
from .radiomics_features import extract_fallback_features, extract_pyradiomics_from_files, pyradiomics_available
from .roi import construct_gtr

LOGGER = logging.getLogger(__name__)


def _case_id_from_dir(path: Path) -> str:
    return path.name


def _prepare_case(case_dir: Path, config: dict, mode: str) -> dict:
    phase_paths, phase_warnings = find_phase_paths(case_dir, config, allow_demo_alias=(mode == "demo"))
    for msg in phase_warnings:
        LOGGER.info(msg)
    mask_img = read_nifti(case_dir / config["data"]["mask_file"])
    mask = mask_img.data > 0.5
    channels = {}
    for phase, path in phase_paths.items():
        img = read_nifti(path)
        channels[phase] = robust_clip(img.data)
    channels = align_channels_to_mask(channels, mask)

    def _load_optional(name: str):
        p = optional_mask(case_dir, config["data"].get(name))
        if p is None:
            return None
        return align_mask_to_reference(read_nifti(p).data, mask.shape)

    gtr = construct_gtr(
        mask,
        spacing=mask_img.spacing[:3],
        dilation_mm=config["roi"]["gtr_dilation_mm"],
        liver_mask=_load_optional("optional_liver_mask_file"),
        vessel_mask=_load_optional("optional_vessel_mask_file"),
        bile_duct_mask=_load_optional("optional_bile_duct_mask_file"),
    )
    local_matrix, local_names, bbox = extract_local_feature_matrix(
        channels,
        mask,
        window_size=config["preprocessing"]["local_window_voxels"],
        entropy_bins=config["preprocessing"]["local_entropy_bins"],
    )
    return {
        "case_id": _case_id_from_dir(case_dir),
        "case_dir": case_dir,
        "phase_paths": phase_paths,
        "mask_path": case_dir / config["data"]["mask_file"],
        "mask_img": mask_img,
        "mask": mask,
        "gtr": gtr,
        "channels": channels,
        "local_matrix": local_matrix,
        "local_names": local_names,
        "bbox": bbox,
    }


def _feature_groups(frame: pd.DataFrame, clinical_cols: list[str]) -> dict[str, list[str]]:
    cols = list(frame.columns)
    gtr = [c for c in cols if "_gtr_" in c]
    ith = [c for c in cols if c.startswith("ITH_") or "_ITH_" in c]
    return {
        "Clinical": [c for c in clinical_cols if c in cols],
        "GTR": gtr,
        "ITH": ith,
        "GTR_ITH": gtr + ith,
        "Fusion": gtr + ith + [c for c in clinical_cols if c in cols],
    }


def _fit_component(name: str, x: pd.DataFrame, y: np.ndarray, cfg: dict):
    fs = cfg["feature_selection"]
    selected_x, selected = select_features(
        x,
        y,
        p_threshold=fs["univariable_p_threshold"],
        corr_threshold=fs["correlation_threshold"],
        lasso_cv_folds=fs["lasso_cv_folds"],
        random_state=cfg["random_seed"],
        max_demo_features=fs["max_demo_features"],
    )
    model = fit_binary_model(
        selected_x,
        y,
        cv_folds=cfg["modeling"]["classifier_cv_folds"],
        random_state=cfg["random_seed"],
    )
    model.features = selected
    return model


def _resolve_pyradiomics_params(config_path: Path, cfg: dict) -> Path:
    configured = cfg.get("radiomics", {}).get("pyradiomics_params", "pyradiomics.yml")
    path = Path(configured)
    if path.is_absolute():
        return path
    candidate = config_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def _prefix_radiomics_features(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _extract_pyradiomics_features(case: dict, labels: np.ndarray, cfg: dict, output_dir: Path, params_path: Path) -> dict:
    mask_dir = output_dir / "derived_masks" / case["case_id"]
    mask_dir.mkdir(parents=True, exist_ok=True)
    feats = {}

    gtr_path = mask_dir / "gtr_mask.nii.gz"
    write_nifti_like(case["gtr"].astype(np.int16), case["mask_img"], gtr_path)

    habitat_paths = {}
    for label in range(1, cfg["habitat"]["n_clusters"] + 1):
        habitat_path = mask_dir / f"habitat_{label}_mask.nii.gz"
        write_nifti_like(((labels == label) & (case["mask"] > 0)).astype(np.int16), case["mask_img"], habitat_path)
        habitat_paths[label] = habitat_path

    for phase, image_path in case["phase_paths"].items():
        feats.update(
            _prefix_radiomics_features(
                f"{phase}_gtr",
                extract_pyradiomics_from_files(image_path, gtr_path, params_path),
            )
        )
        for label, habitat_path in habitat_paths.items():
            feats.update(
                _prefix_radiomics_features(
                    f"{phase}_ITH_habitat_{label}",
                    extract_pyradiomics_from_files(image_path, habitat_path, params_path),
                )
            )
    return feats


def _extract_case_features(case: dict, labels: np.ndarray, cfg: dict, output_dir: Path, mode: str, params_path: Path) -> dict:
    if mode == "demo":
        return extract_fallback_features(case["channels"], case["mask"], case["gtr"], labels, cfg["habitat"]["n_clusters"])
    if not pyradiomics_available():
        raise RuntimeError(
            "PyRadiomics and SimpleITK are required in train mode. "
            "Install the environment from environment.yml, or run scripts/run_demo.py for the synthetic demo run."
        )
    return _extract_pyradiomics_features(case, labels, cfg, output_dir, params_path)


def _safe_classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    out = {"n": int(len(y_true)), "threshold": float(threshold)}
    if len(y_true) == 0:
        out["status"] = "empty"
        return out
    if len(np.unique(y_true)) < 2:
        out["status"] = "single_class"
        out["class"] = int(y_true[0])
        return out
    out.update(classification_metrics(y_true, prob, threshold))
    return out


def _metrics_by_split(split_values: pd.Series, y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    rows = {}
    for split in ["train", "validation", "external_test"]:
        mask = split_values.eq(split).values
        if mask.any():
            rows[split] = _safe_classification_metrics(y[mask], prob[mask], threshold)
    for split in sorted(set(split_values) - set(rows)):
        mask = split_values.eq(split).values
        rows[split] = _safe_classification_metrics(y[mask], prob[mask], threshold)
    return rows


def run_pipeline(data_dir: Path, clinical_csv: Path, output_dir: Path, config_path: Path, mode: str = "demo") -> None:
    cfg = load_yaml(config_path)
    params_path = _resolve_pyradiomics_params(config_path, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "habitat_maps").mkdir(exist_ok=True)
    shutil.copy2(config_path, output_dir / "pipeline.yml")

    clinical = load_clinical_table(clinical_csv, cfg["data"]["case_id_column"])
    clinical_cols = cfg["data"]["clinical_columns"]
    label_col = cfg["data"]["label_column"]
    require_columns(clinical, [cfg["data"]["case_id_column"], *clinical_cols])
    has_label = label_col in clinical.columns

    case_dirs = discover_case_dirs(data_dir, cfg["data"]["mask_file"])
    prepared = [_prepare_case(p, cfg, mode) for p in case_dirs]
    local_names = prepared[0]["local_names"]

    by_id = clinical.set_index(cfg["data"]["case_id_column"])
    split_col = cfg["data"].get("split_column")
    if split_col in clinical.columns and mode != "demo":
        split_series = clinical.set_index(cfg["data"]["case_id_column"])[split_col].astype(str).str.lower()
        train_case_ids = set(split_series[split_series.eq("train")].index)
    else:
        train_case_ids = {c["case_id"] for c in prepared}
    matrices_for_fit = [c["local_matrix"] for c in prepared if c["case_id"] in train_case_ids]
    if not matrices_for_fit:
        raise ValueError("No training cases found for fitting the GMM habitat model")
    habitat_model = fit_habitat_model(
        matrices_for_fit,
        local_names,
        n_clusters=cfg["habitat"]["n_clusters"],
        random_state=cfg["random_seed"],
        max_voxels_per_case=cfg["habitat"]["max_fit_voxels_per_case"],
    )
    k_rows = evaluate_k_range(
        matrices_for_fit,
        range(cfg["habitat"]["search_k_min"], cfg["habitat"]["search_k_max"] + 1),
        random_state=cfg["random_seed"],
        max_metric_voxels=cfg["habitat"]["max_metric_voxels"],
    )
    pd.DataFrame(k_rows).to_csv(output_dir / "gmm_k_search.csv", index=False)

    feature_rows = []
    for case in prepared:
        labels = predict_habitat_labels(case["local_matrix"], case["mask"], case["bbox"], habitat_model)
        if cfg["outputs"].get("save_habitat_maps", True):
            write_nifti_like(labels, case["mask_img"], output_dir / "habitat_maps" / f"{case['case_id']}_habitat.nii.gz")
        feats = {"case_id": case["case_id"]}
        feats.update(habitat_region_features(labels, case["mask"], cfg["habitat"]["n_clusters"]))
        feats.update(_extract_case_features(case, labels, cfg, output_dir, mode, params_path))
        if case["case_id"] in by_id.index:
            for col in clinical_cols:
                feats[col] = by_id.loc[case["case_id"], col]
            if has_label:
                feats[label_col] = by_id.loc[case["case_id"], label_col]
        feature_rows.append(feats)

    features = pd.DataFrame(feature_rows)
    features.to_csv(output_dir / "features.csv", index=False)
    if not has_label:
        joblib.dump({"habitat_model": habitat_model}, output_dir / "habitat_model.joblib")
        return

    y = features[label_col].astype(int).values
    groups = _feature_groups(features.drop(columns=["case_id", label_col]), clinical_cols)
    fitted = {}
    imputers = {}
    predictions = features[["case_id", label_col]].copy()
    train_mask = np.ones(len(features), dtype=bool)
    split_values = pd.Series(["train"] * len(features), index=features.index)
    if split_col in clinical.columns and mode != "demo":
        split_map = clinical.set_index(cfg["data"]["case_id_column"])[split_col]
        split_values = features["case_id"].map(split_map).fillna("train").str.lower()
        train_mask = split_values.eq("train").values
        predictions["split"] = split_values

    for model_name, cols in groups.items():
        if not cols:
            continue
        imputer = fit_feature_imputer(features.loc[train_mask, cols])
        x_all = apply_feature_imputer(features[cols], imputer)
        model = _fit_component(model_name, x_all.loc[train_mask], y[train_mask], cfg)
        fitted[model_name] = model
        imputers[model_name] = imputer
        predictions[f"{model_name}_probability"] = predict_probability(model, x_all)

    fusion_prob = predictions["Fusion_probability"].values if "Fusion_probability" in predictions else predictions.iloc[:, -1].values
    train_prob = fusion_prob[train_mask]
    threshold = youden_threshold(y[train_mask], train_prob) if len(np.unique(y[train_mask])) == 2 else 0.5
    predictions["fusion_risk_group_demo_threshold"] = np.where(fusion_prob >= threshold, "high", "low")
    published_thr = float(cfg["modeling"]["published_fusion_threshold"])
    predictions["fusion_risk_group_published_0_473"] = np.where(fusion_prob >= published_thr, "high", "low")
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    metrics = {
        "mode": mode,
        "n_cases": int(len(features)),
        "demo_or_training_youden_threshold": float(threshold),
        "published_fusion_threshold": published_thr,
        "models": {name: {"selected_features": m.features, "selected_classifier": m.selected_classifier, "train_auc": m.train_auc} for name, m in fitted.items()},
    }
    if len(np.unique(y)) == 2:
        if mode != "demo" and split_col in clinical.columns:
            metrics["fusion_metrics_by_split_training_threshold"] = _metrics_by_split(split_values, y, fusion_prob, threshold)
            metrics["fusion_metrics_by_split_published_threshold"] = _metrics_by_split(split_values, y, fusion_prob, published_thr)
            metrics["fusion_hosmer_lemeshow_by_split"] = {
                split: hosmer_lemeshow(y[split_values.eq(split).values], fusion_prob[split_values.eq(split).values], groups=10)
                for split in sorted(split_values.unique())
                if len(np.unique(y[split_values.eq(split).values])) == 2
            }
        else:
            metrics["fusion_metrics_demo_threshold" if mode == "demo" else "fusion_metrics_all_cases_training_threshold"] = classification_metrics(y, fusion_prob, threshold)
            metrics["fusion_hosmer_lemeshow"] = hosmer_lemeshow(y, fusion_prob, groups=10)
        metrics["fusion_threshold_bootstrap"] = bootstrap_youden_ci(
            y[train_mask],
            train_prob,
            n_boot=min(int(cfg["modeling"]["bootstrap_iterations"]), 200 if mode == "demo" else int(cfg["modeling"]["bootstrap_iterations"])),
            random_state=cfg["random_seed"],
        )
        if mode != "demo" and split_col in clinical.columns:
            split_metrics = metrics["fusion_metrics_by_split_training_threshold"]
            reference_metrics = split_metrics.get("validation") or split_metrics.get("external_test") or split_metrics.get("train")
        else:
            reference_metrics = metrics["fusion_metrics_demo_threshold" if mode == "demo" else "fusion_metrics_all_cases_training_threshold"]
        sens = reference_metrics.get("sensitivity") if reference_metrics else None
        spec = reference_metrics.get("specificity") if reference_metrics else None
        if sens is not None and spec is not None and np.isfinite(sens) and np.isfinite(spec):
            metrics["prevalence_shift_ppv_npv"] = prevalence_adjusted_ppv_npv(sens, spec, [0.20, 0.30, 0.40, 0.50])
        if cfg["outputs"].get("save_figures", True):
            save_basic_figures(y, fusion_prob, output_dir / "figures")
    save_json(metrics, output_dir / "metrics.json")
    joblib.dump({"habitat_model": habitat_model, "models": fitted, "imputers": imputers, "config": cfg, "features": features.columns.tolist()}, output_dir / "model_artifacts.joblib")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HCC MVI ITH radiomics pipeline.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=Path("config/pipeline.yml"), type=Path)
    parser.add_argument("--mode", choices=["demo", "train"], default="demo")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args.data_dir, args.clinical, args.output, args.config, args.mode)


if __name__ == "__main__":
    main()
