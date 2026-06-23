from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage


def resample_to_shape(array: np.ndarray, target_shape: Sequence[int], order: int) -> np.ndarray:
    if tuple(array.shape) == tuple(target_shape):
        return array
    zoom = [t / s for t, s in zip(target_shape, array.shape)]
    return ndimage.zoom(array, zoom=zoom, order=order)


def align_channels_to_mask(channels: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: resample_to_shape(img, mask.shape, order=1).astype(np.float32) for name, img in channels.items()}


def align_mask_to_reference(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    return resample_to_shape(mask.astype(np.float32), target_shape, order=0) > 0.5


def robust_clip(image: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    vals = image[np.isfinite(image)]
    if vals.size == 0:
        return image.astype(np.float32)
    lo, hi = np.percentile(vals, [lower, upper])
    if hi <= lo:
        return image.astype(np.float32)
    return np.clip(image, lo, hi).astype(np.float32)


def n4_bias_correct_if_available(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Apply SimpleITK N4 correction when available; otherwise return the image."""
    try:
        import SimpleITK as sitk
    except Exception:
        warnings.warn("SimpleITK is not installed; N4 bias-field correction was skipped.", RuntimeWarning)
        return image.astype(np.float32)

    sitk_img = sitk.GetImageFromArray(image.astype(np.float32))
    if mask is not None:
        sitk_mask = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
    else:
        sitk_mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    corrected = sitk.N4BiasFieldCorrection(sitk_img, sitk_mask)
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def fit_minmax(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minv = np.nanmin(matrix, axis=0)
    maxv = np.nanmax(matrix, axis=0)
    scale = np.where(maxv > minv, maxv - minv, 1.0)
    return minv, scale


def apply_minmax(matrix: np.ndarray, minv: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = (matrix - minv) / scale
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def bounding_box(mask: np.ndarray, pad: int = 2) -> tuple[slice, ...]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty")
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
