from __future__ import annotations

import numpy as np
from scipy import ndimage


def construct_gtr(
    tumor_mask: np.ndarray,
    spacing: tuple[float, float, float],
    dilation_mm: float = 5.0,
    liver_mask: np.ndarray | None = None,
    vessel_mask: np.ndarray | None = None,
    bile_duct_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the Global Tumor Region: tumor plus a 5 mm peritumoral margin."""
    tumor = tumor_mask > 0
    if not tumor.any():
        raise ValueError("Tumor mask is empty")
    outside_distance = ndimage.distance_transform_edt(~tumor, sampling=spacing[: tumor.ndim])
    gtr = tumor | (outside_distance <= float(dilation_mm))
    if liver_mask is not None:
        gtr &= liver_mask > 0
    if vessel_mask is not None:
        gtr &= ~(vessel_mask > 0)
    if bile_duct_mask is not None:
        gtr &= ~(bile_duct_mask > 0)
    return gtr


def peritumoral_ring(tumor_mask: np.ndarray, gtr_mask: np.ndarray) -> np.ndarray:
    return (gtr_mask > 0) & ~(tumor_mask > 0)
