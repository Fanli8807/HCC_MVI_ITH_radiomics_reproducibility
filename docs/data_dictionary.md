# Data Dictionary

## Folder layout

Each patient/case is represented by one folder:

```text
case_001/
  dwi.nii.gz
  ap.nii.gz
  pvp.nii.gz
  dp.nii.gz
  mask.nii.gz
  liver_mask.nii.gz        # optional
  vessel_mask.nii.gz       # optional
  bile_duct_mask.nii.gz    # optional
```

The synthetic demo contains `t1c.nii.gz` rather than separate AP/PVP/DP files. In demo mode this file is intentionally reused as A, V, and D so the example can run the complete pipeline without patient data.

## Clinical CSV

Required columns:

| Column | Meaning | Encoding |
|---|---|---|
| `case_id` | Folder/case identifier | string |
| `HBV` | HBV infection | 0 no, 1 yes |
| `max_diameter_cm` | Maximum tumor diameter on MRI | numeric |
| `non_smooth_margin` | Non-smooth tumor margin | 0 smooth, 1 non-smooth |
| `pseudocapsule` | MRI pseudocapsule | 0 absent, 1 present |
| `MVI_label` | Pathology MVI label | 0 negative, 1 positive |
| `split` | Optional modeling split | train, validation, external_test |

Histologic grade and postoperative pathology variables are not model inputs. They can be used only for post-hoc subgroup analyses, matching the study design.
