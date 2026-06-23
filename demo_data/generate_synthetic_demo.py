"""
Synthetic Demo Dataset Generator for HCC-MVI-ITH-Radiomics Pipeline
====================================================================

This script generates a fully synthetic, format-compatible demonstration
dataset for end-to-end pipeline execution. No patient data are used or
referenced.

Each synthetic case includes:
  - DWI volume (NIfTI)
  - T1WI+C volume (NIfTI)
  - Tumor mask (NIfTI)
  - Clinical variables (in shared CSV)

Output structure:
  demo_data/
    case_001/
      dwi.nii.gz
      t1c.nii.gz
      mask.nii.gz
    case_002/
      ...
    clinical_variables.csv
    README_demo_data.md
"""

import os
import numpy as np
import nibabel as nib
import csv

OUTPUT_DIR = "demo_data"
N_CASES = 2
N_MVI_POS = 1  # case_001 = MVI+, case_002 = MVI-
np.random.seed(42)

# Volume dimensions (axial liver MRI typical)
DWI_SHAPE = (192, 192, 24)       # DWI: lower resolution, fewer slices
T1C_SHAPE = (256, 256, 56)       # T1WI+C: higher resolution, more slices
VOXEL_SIZE_DWI = (1.875, 1.875, 6.0)   # mm
VOXEL_SIZE_T1C = (1.367, 1.367, 3.0)   # mm

# ============================================================
# Helper: generate a single 3D volume with embedded tumor
# ============================================================
def generate_volume(shape, voxel_size, tumor_center, tumor_radius,
                    bg_mean=200, bg_std=40, tumor_intensity_offset=80,
                    mvi_positive=False):
    """
    Generate a synthetic MRI volume with a Gaussian-blob tumor.
    
    For MVI-positive cases: tumor has more spatial heterogeneity
    (multiple sub-regions of varying intensity) and irregular boundary,
    mimicking the imaging signatures we wish to detect.
    """
    nx, ny, nz = shape
    
    # Smooth liver background (low-frequency variation + noise)
    xx, yy, zz = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1, 1, ny),
        np.linspace(-1, 1, nz),
        indexing='ij'
    )
    # Slow gradient (anatomical variation)
    background = bg_mean + 30 * (xx * 0.3 + yy * 0.2 + zz * 0.1)
    # Gaussian noise (acquisition noise)
    background += np.random.normal(0, bg_std * 0.3, shape)
    
    # Insert tumor
    cx, cy, cz = tumor_center
    distance = np.sqrt(
        ((xx + 1) * nx / 2 - cx) ** 2 +
        ((yy + 1) * ny / 2 - cy) ** 2 +
        ((zz + 1) * nz / 2 - cz) ** 2 * (voxel_size[2] / voxel_size[0]) ** 2
    )
    tumor_profile = np.exp(-(distance / tumor_radius) ** 2)
    
    volume = background + tumor_intensity_offset * tumor_profile
    
    # For MVI-positive cases, add intratumoral heterogeneity
    # (this mimics the habitat fragmentation captured by ITH-score)
    if mvi_positive:
        # Add 2-3 random hyper/hypo-intense sub-regions inside tumor
        for _ in range(np.random.randint(2, 4)):
            sub_center_offset = np.random.normal(0, tumor_radius * 0.4, 3)
            sub_cx = cx + sub_center_offset[0]
            sub_cy = cy + sub_center_offset[1]
            sub_cz = cz + sub_center_offset[2] / 4
            sub_radius = tumor_radius * np.random.uniform(0.2, 0.4)
            sub_intensity = np.random.choice([-50, 50]) * np.random.uniform(0.6, 1.0)
            sub_distance = np.sqrt(
                ((xx + 1) * nx / 2 - sub_cx) ** 2 +
                ((yy + 1) * ny / 2 - sub_cy) ** 2 +
                ((zz + 1) * nz / 2 - sub_cz) ** 2 * (voxel_size[2] / voxel_size[0]) ** 2
            )
            sub_profile = np.exp(-(sub_distance / sub_radius) ** 2)
            volume += sub_intensity * sub_profile * tumor_profile
        
        # Add edge irregularity (non-smooth margin)
        edge_noise = np.random.normal(0, 20, shape) * (np.abs(tumor_profile - 0.4) < 0.15)
        volume += edge_noise
    
    # Final small noise
    volume += np.random.normal(0, bg_std * 0.15, shape)
    
    # Clip to typical MRI intensity range (uint16-ish)
    volume = np.clip(volume, 0, 1000)
    
    return volume.astype(np.float32)


def generate_mask(shape, voxel_size, tumor_center, tumor_radius, irregular=False):
    """Generate tumor mask. Irregular=True for MVI+ (non-smooth boundary)."""
    nx, ny, nz = shape
    xx, yy, zz = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1, 1, ny),
        np.linspace(-1, 1, nz),
        indexing='ij'
    )
    cx, cy, cz = tumor_center
    distance = np.sqrt(
        ((xx + 1) * nx / 2 - cx) ** 2 +
        ((yy + 1) * ny / 2 - cy) ** 2 +
        ((zz + 1) * nz / 2 - cz) ** 2 * (voxel_size[2] / voxel_size[0]) ** 2
    )
    
    if irregular:
        # Wobble the radius slightly per voxel (irregular boundary)
        radius_noise = np.random.normal(1.0, 0.06, shape)
        effective_radius = tumor_radius * radius_noise
        mask = (distance < effective_radius).astype(np.uint8)
    else:
        mask = (distance < tumor_radius).astype(np.uint8)
    
    return mask


def make_affine(voxel_size):
    """Build an affine matrix matching the voxel spacing."""
    aff = np.eye(4)
    aff[0, 0] = voxel_size[0]
    aff[1, 1] = voxel_size[1]
    aff[2, 2] = voxel_size[2]
    return aff


# ============================================================
# Generate cases
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

clinical_records = []

print("Generating synthetic demo dataset...")
for i in range(N_CASES):
    case_id = f"case_{i + 1:03d}"
    case_dir = os.path.join(OUTPUT_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)
    
    is_mvi_pos = i < N_MVI_POS
    
    # Tumor placement: random in liver region (right side of axial)
    # DWI tumor center
    dwi_cx = np.random.uniform(70, 130)
    dwi_cy = np.random.uniform(70, 130)
    dwi_cz = np.random.uniform(8, 16)
    # Tumor radius: 18-30 voxels (≈ 3-6 cm in DWI voxel size)
    dwi_tumor_r = np.random.uniform(15, 25)
    
    # T1WI+C tumor center (same anatomical location, different voxel grid)
    t1c_cx = dwi_cx * (T1C_SHAPE[0] / DWI_SHAPE[0])
    t1c_cy = dwi_cy * (T1C_SHAPE[1] / DWI_SHAPE[1])
    t1c_cz = dwi_cz * (T1C_SHAPE[2] / DWI_SHAPE[2])
    t1c_tumor_r = dwi_tumor_r * (T1C_SHAPE[0] / DWI_SHAPE[0])
    
    # Generate DWI volume
    dwi = generate_volume(
        DWI_SHAPE, VOXEL_SIZE_DWI,
        (dwi_cx, dwi_cy, dwi_cz), dwi_tumor_r,
        bg_mean=150, bg_std=30, tumor_intensity_offset=120,
        mvi_positive=is_mvi_pos
    )
    
    # Generate T1WI+C volume
    t1c = generate_volume(
        T1C_SHAPE, VOXEL_SIZE_T1C,
        (t1c_cx, t1c_cy, t1c_cz), t1c_tumor_r,
        bg_mean=400, bg_std=60, tumor_intensity_offset=200,
        mvi_positive=is_mvi_pos
    )
    
    # Generate mask (define on T1WI+C grid, irregular if MVI+)
    mask = generate_mask(
        T1C_SHAPE, VOXEL_SIZE_T1C,
        (t1c_cx, t1c_cy, t1c_cz), t1c_tumor_r,
        irregular=is_mvi_pos
    )
    
    # Save NIfTIs
    nib.save(nib.Nifti1Image(dwi, make_affine(VOXEL_SIZE_DWI)),
             os.path.join(case_dir, "dwi.nii.gz"))
    nib.save(nib.Nifti1Image(t1c, make_affine(VOXEL_SIZE_T1C)),
             os.path.join(case_dir, "t1c.nii.gz"))
    nib.save(nib.Nifti1Image(mask, make_affine(VOXEL_SIZE_T1C)),
             os.path.join(case_dir, "mask.nii.gz"))
    
    # Generate clinical variables (correlated with MVI status, per Table S3 ORs)
    # HBV (OR 1.85 for MVI+): higher prevalence in MVI+
    hbv = int(np.random.random() < (0.85 if is_mvi_pos else 0.60))
    # Maximum diameter (OR 2.87 per Table S3): larger for MVI+
    max_diameter = np.round(np.random.uniform(3.5, 6.0) if is_mvi_pos else np.random.uniform(2.0, 4.0), 1)
    # Non-smooth margin (OR 3.35): more common in MVI+
    non_smooth_margin = int(np.random.random() < (0.85 if is_mvi_pos else 0.40))
    # Pseudocapsule (OR 0.76, protective): less common in MVI+
    pseudocapsule = int(np.random.random() < (0.25 if is_mvi_pos else 0.50))
    
    clinical_records.append({
        "case_id": case_id,
        "HBV": hbv,
        "max_diameter_cm": max_diameter,
        "non_smooth_margin": non_smooth_margin,
        "pseudocapsule": pseudocapsule,
        "MVI_label": int(is_mvi_pos),
    })
    
    print(f"  {case_id}: MVI={'+' if is_mvi_pos else '-'}, "
          f"DWI={dwi.shape}, T1C={t1c.shape}, mask_voxels={int(mask.sum())}")

# Save clinical CSV
clinical_csv = os.path.join(OUTPUT_DIR, "clinical_variables.csv")
with open(clinical_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(clinical_records[0].keys()))
    writer.writeheader()
    writer.writerows(clinical_records)

print(f"\nClinical variables saved to: {clinical_csv}")
print(f"Total synthetic cases: {N_CASES} ({N_MVI_POS} MVI+, {N_CASES - N_MVI_POS} MVI-)")
print(f"Output directory: {OUTPUT_DIR}/")
