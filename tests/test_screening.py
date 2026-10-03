"""
Unit tests for virtual screening and AF3 conformation rectification modules.
"""

import numpy as np
import pandas as pd
from dynamic_esm.screening.virtual_screening import evaluate_virtual_screening_targets
from dynamic_esm.screening.af3_correction import evaluate_af3_conformation_rectification


def test_virtual_screening():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(4.0, 10.0, 200)
    y_pred = y_true + rng.normal(0, 0.3, 200)

    df_results, detailed_dict = evaluate_virtual_screening_targets(y_true, y_pred)
    assert isinstance(df_results, pd.DataFrame)
    assert len(df_results) == 5  # 5 benchmark targets
    assert "Target Class" in df_results.columns
    assert "ROC-AUC" in df_results.columns
    assert "BEDROC (a=16.1)" in df_results.columns
    assert "EF 1%" in df_results.columns
    assert "EF 5%" in df_results.columns


def test_af3_conformation_rectification():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(4.0, 10.0, 100)
    y_pred = y_true + rng.normal(0, 0.4, 100)

    df_rect, thermo_data = evaluate_af3_conformation_rectification(y_true, y_pred)
    assert isinstance(df_rect, pd.DataFrame)
    assert len(df_rect) == 2
    assert "Conformation Source" in df_rect.columns
    assert "RMSE" in df_rect.columns
    assert "delta_g_exp" in thermo_data
    assert "delta_g_pred" in thermo_data
    assert len(thermo_data["delta_g_exp"]) == 100
