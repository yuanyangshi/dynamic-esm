"""
Evaluation, calibration, and benchmarking subpackage for Dynamic-ESM.
"""

from dynamic_esm.evaluation.metrics import (
    compute_concordance_index,
    vectorized_bootstrap_ci,
    compute_roc_curve_and_auc,
    compute_bedroc_score,
    compute_enrichment_factor,
    compute_expected_calibration_error,
)
from dynamic_esm.evaluation.casf_benchmark import evaluate_casf_benchmarks
from dynamic_esm.evaluation.ablation import run_ablation_benchmarks

__all__ = [
    "compute_concordance_index",
    "vectorized_bootstrap_ci",
    "compute_roc_curve_and_auc",
    "compute_bedroc_score",
    "compute_enrichment_factor",
    "compute_expected_calibration_error",
    "evaluate_casf_benchmarks",
    "run_ablation_benchmarks",
]
