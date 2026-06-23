from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from .preprocessing import apply_minmax, fit_minmax


@dataclass
class HabitatModel:
    minv: np.ndarray
    scale: np.ndarray
    gmm: GaussianMixture
    feature_names: list[str]


def _sample_rows(x: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= max_rows:
        return x
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx]


def fit_habitat_model(
    matrices: Sequence[np.ndarray],
    feature_names: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
    max_voxels_per_case: int = 50000,
) -> HabitatModel:
    rng = np.random.default_rng(random_state)
    sampled = [_sample_rows(m, max_voxels_per_case, rng) for m in matrices if m.size]
    x = np.vstack(sampled)
    minv, scale = fit_minmax(x)
    x_scaled = apply_minmax(x, minv, scale)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state)
    gmm.fit(x_scaled)
    return HabitatModel(minv=minv, scale=scale, gmm=gmm, feature_names=feature_names)


def evaluate_k_range(
    matrices: Sequence[np.ndarray],
    k_values: Iterable[int],
    random_state: int = 42,
    max_metric_voxels: int = 10000,
) -> list[dict]:
    rng = np.random.default_rng(random_state)
    x = np.vstack([_sample_rows(m, max_metric_voxels // max(len(matrices), 1), rng) for m in matrices if m.size])
    x = _sample_rows(x, max_metric_voxels, rng)
    minv, scale = fit_minmax(x)
    x = apply_minmax(x, minv, scale)
    rows = []
    for k in k_values:
        if x.shape[0] <= k:
            continue
        labels = GaussianMixture(n_components=k, random_state=random_state).fit_predict(x)
        if len(np.unique(labels)) < 2:
            continue
        rows.append(
            {
                "k": int(k),
                "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
                "silhouette": float(silhouette_score(x, labels)),
            }
        )
    return rows


def predict_habitat_labels(
    matrix: np.ndarray,
    tumor_mask: np.ndarray,
    bbox: tuple[slice, ...],
    model: HabitatModel,
) -> np.ndarray:
    labels = np.zeros(tumor_mask.shape, dtype=np.int16)
    scaled = apply_minmax(matrix, model.minv, model.scale)
    pred = model.gmm.predict(scaled).astype(np.int16) + 1
    mask_crop = tumor_mask[bbox] > 0
    crop_labels = np.zeros(mask_crop.shape, dtype=np.int16)
    crop_labels[mask_crop] = pred
    labels[bbox] = crop_labels
    return labels


def ith_score(labels: np.ndarray, tumor_mask: np.ndarray, n_labels: int | None = None) -> float:
    """Closed-form ITH score: 1 - (1 / S_total) * sum_i(S_i,max / n_i)."""
    mask = tumor_mask > 0
    s_total = int(mask.sum())
    if s_total == 0:
        raise ValueError("Tumor mask is empty")
    if n_labels is None:
        n_labels = int(labels[mask].max())
    structure = np.ones((3, 3, 3), dtype=bool)
    contribution = 0.0
    for i in range(1, n_labels + 1):
        habitat = (labels == i) & mask
        if not habitat.any():
            continue
        cc, n_i = ndimage.label(habitat, structure=structure)
        if n_i == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        s_i_max = int(sizes.max()) if sizes.size else 0
        contribution += s_i_max / float(n_i)
    return float(1.0 - contribution / float(s_total))


def habitat_region_features(labels: np.ndarray, tumor_mask: np.ndarray, n_labels: int) -> dict:
    mask = tumor_mask > 0
    total = max(int(mask.sum()), 1)
    feats = {"ITH_score": ith_score(labels, mask, n_labels)}
    structure = np.ones((3, 3, 3), dtype=bool)
    for i in range(1, n_labels + 1):
        habitat = (labels == i) & mask
        cc, n_i = ndimage.label(habitat, structure=structure)
        sizes = np.bincount(cc.ravel())[1:] if n_i else np.array([])
        feats[f"ITH_habitat_{i}_volume_ratio"] = float(habitat.sum() / total)
        feats[f"ITH_habitat_{i}_n_components"] = int(n_i)
        feats[f"ITH_habitat_{i}_largest_component_ratio"] = float((sizes.max() if sizes.size else 0) / total)
    return feats
