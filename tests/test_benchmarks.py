"""
Unit tests for CASF-2016 benchmarks and ablation evaluation engine.
"""

import numpy as np
import pandas as pd
from dynamic_esm.evaluation.casf_benchmark import evaluate_casf_benchmarks
from dynamic_esm.evaluation.ablation import run_ablation_benchmarks


def test_casf_benchmarks_evaluation():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(4.0, 10.0, 100)
    y_pred = y_true + rng.normal(0, 0.4, 100)

    df_bench, boot_results = evaluate_casf_benchmarks(y_true, y_pred, n_bootstraps=50)
    assert isinstance(df_bench, pd.DataFrame)
    assert len(df_bench) >= 7  # Dynamic-ESM + 6 baselines
    assert "Model" in df_bench.columns
    assert "Rp" in df_bench.columns
    assert "RMSE" in df_bench.columns
    assert "Wilcoxon_p" in df_bench.columns
    assert "Rp" in boot_results


def test_ablation_benchmarks():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(4.0, 10.0, 100)
    y_pred = y_true + rng.normal(0, 0.4, 100)

    df_ablation = run_ablation_benchmarks(y_true, y_pred)
    assert isinstance(df_ablation, pd.DataFrame)
    assert len(df_ablation) >= 7  # Categories: Full Model, Dynamics Depth, Physical QM, Multimodal Fusion
    assert "Category" in df_ablation.columns
    assert "Variant" in df_ablation.columns
    assert "Delta_RMSE" in df_ablation.columns
