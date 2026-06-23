from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import stats


def pyradiomics_available() -> bool:
    try:
        import radiomics  # noqa: F401
        import SimpleITK  # noqa: F401

        return True
    except Exception:
        return False


def extract_pyradiomics_from_files(image_path: str | Path, mask_path: str | Path, params_path: str | Path) -> dict:
    """Extract PyRadiomics features from image and mask files."""
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_path))
    result = extractor.execute(str(image_path), str(mask_path))
    clean = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            clean[key] = float(value)
        except Exception:
            continue
    return clean


def _first_order(values: np.ndarray, prefix: str) -> dict:
    vals = values[np.isfinite(values)].astype(np.float64)
    if vals.size == 0:
        return {f"{prefix}_{name}": np.nan for name in ["Mean", "Std", "Energy", "Entropy", "Skewness", "Kurtosis"]}
    hist, _ = np.histogram(vals, bins=32, density=False)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    positive = p > 0
    entropy = -np.sum(p[positive] * np.log2(p[positive]))
    return {
        f"{prefix}_Mean": float(np.mean(vals)),
        f"{prefix}_Std": float(np.std(vals)),
        f"{prefix}_Energy": float(np.sum(vals**2)),
        f"{prefix}_Entropy": float(entropy),
        f"{prefix}_Skewness": float(stats.skew(vals, bias=False)) if vals.size > 2 else 0.0,
        f"{prefix}_Kurtosis": float(stats.kurtosis(vals, bias=False)) if vals.size > 3 else 0.0,
        f"{prefix}_P10": float(np.percentile(vals, 10)),
        f"{prefix}_P90": float(np.percentile(vals, 90)),
    }


def extract_fallback_features(
    channels: Mapping[str, np.ndarray],
    tumor_mask: np.ndarray,
    gtr_mask: np.ndarray,
    habitat_labels: np.ndarray | None = None,
    n_habitats: int = 3,
) -> dict:
    """Dependency-light features used by the synthetic demo run."""
    feats = {}
    masks = {"tumor": tumor_mask > 0, "gtr": gtr_mask > 0}
    for phase, image in channels.items():
        for roi_name, roi_mask in masks.items():
            prefix = f"{phase}_{roi_name}_firstorder"
            feats.update(_first_order(image[roi_mask], prefix))
            feats[f"{phase}_{roi_name}_shape_VoxelVolume"] = float(roi_mask.sum())
        if habitat_labels is not None:
            for label in range(1, n_habitats + 1):
                hmask = (habitat_labels == label) & (tumor_mask > 0)
                prefix = f"{phase}_ITH_habitat_{label}_firstorder"
                feats.update(_first_order(image[hmask], prefix))
    return feats
