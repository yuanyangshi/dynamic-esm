"""
Unit tests for evaluation metrics and uncertainty calibration.
"""

import numpy as np


from dynamic_esm.evaluation.metrics import (
    compute_concordance_index,
    vectorized_bootstrap_ci,
    compute_bedroc_score,
    compute_enrichment_factor,
    compute_expected_calibration_error,
)


def test_concordance_index():
    y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    y_pred = np.array([5.2, 6.1, 7.2, 7.9, 9.1])
    ci = compute_concordance_index(y_true, y_pred)
    assert ci == 1.0

    y_pred_rev = np.array([9.1, 7.9, 7.2, 6.1, 5.2])
    ci_rev = compute_concordance_index(y_true, y_pred_rev)
    assert ci_rev == 0.0


def test_vectorized_bootstrap_ci():
    y_true = np.linspace(4.0, 10.0, 100)
    y_pred = y_true + np.random.normal(0, 0.2, 100)
    res = vectorized_bootstrap_ci(y_true, y_pred, n_bootstraps=100)
    assert "Rp" in res
    assert "RMSE" in res
    mean_rp, low, high = res["Rp"]
    assert low <= mean_rp <= high
    assert high <= 1.0


def test_bedroc_and_ef():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    bedroc = compute_bedroc_score(y_true, y_score, alpha=16.1)
    assert bedroc > 0.8

    ef = compute_enrichment_factor(y_true, y_score, fraction=0.2)
    assert ef > 1.0


def test_ece():
    y_true = np.zeros(100)
    y_pred = np.random.normal(0, 1, 100)
    variance = np.ones(100)
    ece = compute_expected_calibration_error(y_true, y_pred, variance)
    assert 0.0 <= ece <= 1.0
