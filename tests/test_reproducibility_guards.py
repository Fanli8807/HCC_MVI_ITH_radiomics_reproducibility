from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcc_mvi_ith.feature_selection import apply_feature_imputer, fit_feature_imputer
from hcc_mvi_ith.io import find_phase_paths, load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_feature_imputer_uses_training_medians_only():
    train = pd.DataFrame({"a": [1.0, 3.0], "b": [np.nan, 10.0]})
    full = pd.DataFrame({"a": [np.nan, 100.0], "b": [5.0, np.nan]})
    imputer = fit_feature_imputer(train)
    out = apply_feature_imputer(full, imputer)
    assert out.loc[0, "a"] == 2.0
    assert out.loc[1, "b"] == 10.0


def test_t1c_alias_is_demo_only():
    cfg = load_yaml(ROOT / "config" / "demo_pipeline.yml")
    case_dir = ROOT / "demo_data" / "case_001"
    phase_paths, warnings = find_phase_paths(case_dir, cfg, allow_demo_alias=True)
    assert {"DWI", "A", "V", "D"} <= set(phase_paths)
    assert warnings
    with pytest.raises(FileNotFoundError):
        find_phase_paths(case_dir, cfg, allow_demo_alias=False)
