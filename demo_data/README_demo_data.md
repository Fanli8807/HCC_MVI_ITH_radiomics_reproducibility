# Synthetic Demonstration Dataset (2 cases)

This is a **fully synthetic, format-compatible demonstration dataset** that allows the HCC-MVI-ITH-Radiomics pipeline to be executed end-to-end from raw image/mask input to final prediction output. **No patient data are used or referenced.**

## Purpose

Demonstrate pipeline executability. Two cases are provided as a minimal end-to-end example (one MVI-positive, one MVI-negative). The dataset is **not** intended to reproduce or validate the model's discriminative performance; for that, see the full study external test cohort.

## Contents

```
demo_data/
├── case_001/                            # Example MVI-positive case
│   ├── dwi.nii.gz                       # DWI (192×192×24, 1.875×1.875×6.0 mm)
│   ├── t1c.nii.gz                       # Contrast-enhanced T1WI (256×256×56, 1.367×1.367×3.0 mm)
│   └── mask.nii.gz                      # Tumor mask on T1WI+C grid
├── case_002/                            # Example MVI-negative case
│   └── ...
├── clinical_variables.csv               # Four Clinical-model variables for each case
└── README_demo_data.md                  # This file
```

## Generation

Generated using NumPy random number generation with seed=42. Synthetic tumors are Gaussian-blob structures with anatomically representative voxel resolution. The MVI-positive case carries a larger, less smoothly bounded tumor with internal heterogeneity, in keeping with the imaging signatures discussed in the study.

## Clinical variables

The accompanying `clinical_variables.csv` provides the four Clinical-model inputs for each synthetic case:

| Column | Description | Encoding |
|---|---|---|
| `case_id` | Synthetic case identifier | string |
| `HBV` | HBV infection | 0 = no, 1 = yes |
| `max_diameter_cm` | Maximum tumor diameter | continuous (cm) |
| `non_smooth_margin` | Non-smooth tumor margin on MRI | 0 = smooth, 1 = non-smooth |
| `pseudocapsule` | Pseudocapsule on MRI | 0 = absent, 1 = present |
| `MVI_label` | Ground-truth MVI label (synthetic) | 0 = MVI-, 1 = MVI+ |

## Usage

```bash
python run_pipeline.py --input demo_data/case_001 \
                      --clinical demo_data/clinical_variables.csv \
                      --case-id case_001 \
                      --output results/case_001/
```

## Real-data access

De-identified patient-level data may be available from the corresponding author upon reasonable request, subject to approval by the institutional review board and a formal data-sharing agreement (see study Data Availability statement).

## License

Released under the same license as the code (see `LICENSE`).
