from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class NiftiImage:
    data: np.ndarray
    spacing: tuple
    affine: np.ndarray
    header: bytes
    path: Path


_NIFTI_DTYPES = {
    2: np.dtype("uint8"),
    4: np.dtype("<i2"),
    8: np.dtype("<i4"),
    16: np.dtype("<f4"),
    64: np.dtype("<f8"),
    256: np.dtype("int8"),
    512: np.dtype("<u2"),
    768: np.dtype("<u4"),
}


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Mapping, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_nifti(path: str | Path) -> NiftiImage:
    """Read NIfTI using nibabel when present, with a small NIfTI-1 fallback."""
    path = Path(path)
    try:
        import nibabel as nib

        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj).astype(np.float32)
        spacing = tuple(float(v) for v in img.header.get_zooms()[: data.ndim])
        return NiftiImage(data=data, spacing=spacing, affine=img.affine, header=bytes(img.header.binaryblock), path=path)
    except Exception:
        return _read_nifti1_fallback(path)


def _read_nifti1_fallback(path: Path) -> NiftiImage:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()

    header = raw[:348]
    sizeof_hdr = struct.unpack("<i", header[:4])[0]
    if sizeof_hdr != 348:
        raise ValueError(f"{path} does not look like a little-endian NIfTI-1 file")

    dim = struct.unpack("<8h", header[40:56])
    ndim = int(dim[0])
    shape = tuple(int(v) for v in dim[1 : 1 + ndim])
    datatype = struct.unpack("<h", header[70:72])[0]
    dtype = _NIFTI_DTYPES.get(datatype)
    if dtype is None:
        raise ValueError(f"Unsupported NIfTI datatype code {datatype} in {path}")

    pixdim = struct.unpack("<8f", header[76:108])
    spacing = tuple(float(v) for v in pixdim[1 : 1 + ndim])
    vox_offset = int(struct.unpack("<f", header[108:112])[0])
    slope = struct.unpack("<f", header[112:116])[0]
    inter = struct.unpack("<f", header[116:120])[0]
    slope = 1.0 if slope == 0 or not np.isfinite(slope) else float(slope)
    inter = 0.0 if not np.isfinite(inter) else float(inter)

    n_values = int(np.prod(shape))
    data = np.frombuffer(raw, dtype=dtype, count=n_values, offset=vox_offset).copy()
    data = data.reshape(shape, order="F").astype(np.float32) * slope + inter
    affine = np.diag([*(spacing[:3] if len(spacing) >= 3 else spacing), 1.0])
    return NiftiImage(data=data, spacing=spacing, affine=affine, header=header, path=path)


def write_nifti_like(data: np.ndarray, reference: NiftiImage, path: str | Path) -> None:
    """Write a label map, preferring nibabel and falling back to a simple NIfTI-1 writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data)
    try:
        import nibabel as nib

        nib.save(nib.Nifti1Image(arr.astype(np.int16), reference.affine), str(path))
        return
    except Exception:
        pass

    header = bytearray(reference.header[:348])
    shape = arr.shape
    struct.pack_into("<8h", header, 40, len(shape), *shape, *([1] * (7 - len(shape))))
    struct.pack_into("<h", header, 70, 4)
    struct.pack_into("<h", header, 72, 16)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    payload = arr.astype("<i2").ravel(order="F").tobytes()
    blob = bytes(header) + b"\0\0\0\0" + payload
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wb") as f:
        f.write(blob)


def load_clinical_table(path: str | Path, case_id_col: str = "case_id") -> pd.DataFrame:
    df = pd.read_csv(path)
    if case_id_col not in df.columns:
        raise ValueError(f"Clinical table must contain a '{case_id_col}' column")
    return df


def discover_case_dirs(data_dir: str | Path, mask_file: str = "mask.nii.gz") -> list[Path]:
    root = Path(data_dir)
    cases = sorted(p for p in root.iterdir() if p.is_dir() and (p / mask_file).exists())
    if not cases:
        raise FileNotFoundError(f"No case folders with {mask_file} found under {root}")
    return cases


def find_phase_paths(case_dir: str | Path, config: Mapping, allow_demo_alias: bool = False) -> tuple[Dict[str, Path], list[str]]:
    case_dir = Path(case_dir)
    phase_files = config["data"]["phase_files"]
    alias = config["data"].get("demo_t1c_alias", "t1c.nii.gz")
    warnings = []
    paths: Dict[str, Path] = {}
    for phase, filename in phase_files.items():
        candidate = case_dir / filename
        if candidate.exists():
            paths[phase] = candidate
        elif allow_demo_alias and phase in {"A", "V", "D"} and (case_dir / alias).exists():
            paths[phase] = case_dir / alias
            warnings.append(f"{case_dir.name}: using {alias} as {phase} phase for synthetic demo compatibility")
        else:
            raise FileNotFoundError(f"Missing {phase} image for {case_dir}: expected {filename}")
    return paths, warnings


def optional_mask(case_dir: str | Path, filename: Optional[str]) -> Optional[Path]:
    if not filename:
        return None
    path = Path(case_dir) / filename
    return path if path.exists() else None


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
