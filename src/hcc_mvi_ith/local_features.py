from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import ndimage

from .preprocessing import bounding_box


def _local_moments(volume: np.ndarray, size: int) -> tuple[np.ndarray, ...]:
    vol = np.nan_to_num(volume.astype(np.float32), copy=False)
    m1 = ndimage.uniform_filter(vol, size=size, mode="nearest")
    m2 = ndimage.uniform_filter(vol**2, size=size, mode="nearest")
    m3 = ndimage.uniform_filter(vol**3, size=size, mode="nearest")
    m4 = ndimage.uniform_filter(vol**4, size=size, mode="nearest")
    var = np.maximum(m2 - m1**2, 0.0)
    std = np.sqrt(var)
    eps = 1e-6
    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3
    mu4 = m4 - 4 * m1 * m3 + 6 * (m1**2) * m2 - 3 * m1**4
    skew = np.where(std > eps, mu3 / (std**3 + eps), 0.0)
    kurt = np.where(std > eps, mu4 / (std**4 + eps), 0.0)
    energy = m2 * (size**volume.ndim)
    return m1, std, energy, skew, kurt


def _local_entropy(volume: np.ndarray, size: int, bins: int) -> np.ndarray:
    vals = volume[np.isfinite(volume)]
    if vals.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    edges = np.linspace(lo, hi, bins + 1)
    digitized = np.clip(np.digitize(volume, edges) - 1, 0, bins - 1)
    entropy = np.zeros_like(volume, dtype=np.float32)
    for b in range(bins):
        p = ndimage.uniform_filter((digitized == b).astype(np.float32), size=size, mode="nearest")
        positive = p > 0
        entropy[positive] -= p[positive] * np.log2(p[positive])
    return entropy


def extract_local_feature_matrix(
    channels: Mapping[str, np.ndarray],
    tumor_mask: np.ndarray,
    window_size: int = 3,
    entropy_bins: int = 16,
) -> tuple[np.ndarray, list[str], tuple[slice, ...]]:
    """Build the 24-dimensional per-voxel GMM feature matrix described in the paper."""
    bbox = bounding_box(tumor_mask, pad=window_size)
    mask_crop = tumor_mask[bbox] > 0
    matrices = []
    names = []
    for phase in ["DWI", "A", "V", "D"]:
        if phase not in channels:
            continue
        vol = channels[phase][bbox].astype(np.float32)
        mean, std, energy, skew, kurt = _local_moments(vol, window_size)
        entropy = _local_entropy(vol, window_size, entropy_bins)
        feature_maps = {
            "local_mean": mean,
            "local_std": std,
            "local_energy": energy,
            "local_entropy": entropy,
            "local_skewness": skew,
            "local_kurtosis": kurt,
        }
        for fname, fmap in feature_maps.items():
            matrices.append(fmap[mask_crop])
            names.append(f"{phase}_{fname}")
    matrix = np.vstack(matrices).T.astype(np.float32)
    return matrix, names, bbox
