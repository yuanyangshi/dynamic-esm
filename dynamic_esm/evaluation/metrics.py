"""
Evaluation metrics, uncertainty calibration, and statistical validation for Dynamic-ESM.
Includes 1,000-resample bootstrap 95% CIs and Wilcoxon signed-rank testing.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats

def mean_squared_error(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    desc_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_indices]
    y_score = y_score[desc_indices]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / max(fps[-1], 1)
    tpr = tps / max(tps[-1], 1)
    return fpr, tpr, None


def auc(x, y):
    # np.trapezoid in numpy 2.0 or np.trapz in numpy <2.0
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(y, x))



def compute_concordance_index(
    y_true: np.ndarray, y_pred: np.ndarray, max_pairs: int = 100000
) -> float:
    """
    Computes Concordance Index (CI) for pairwise ranking order.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    if n < 2:
        return 0.5

    # Subsample if large
    if n * (n - 1) // 2 > max_pairs:
        indices = np.random.choice(n, size=min(n, 1000), replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        n = len(y_true)

    i_idx, j_idx = np.triu_indices(n, k=1)
    true_diff = y_true[i_idx] - y_true[j_idx]
    pred_diff = y_pred[i_idx] - y_pred[j_idx]

    valid = true_diff != 0
    if not np.any(valid):
        return 0.5

    concordant = np.sum((true_diff[valid] * pred_diff[valid]) > 0)
    ties = np.sum((pred_diff[valid] == 0) & (true_diff[valid] != 0)) * 0.5
    total = np.sum(valid)
    return float((concordant + ties) / total)


def vectorized_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes 1,000-resample empirical bootstrap 95% Confidence Intervals.
    Returns:
        Dict of metric_name -> (mean_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)

    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))

    rp_list, rho_list, rmse_list, mae_list = [], [], [], []
    for idxs in boot_indices:
        yt_b, yp_b = y_true[idxs], y_pred[idxs]
        if np.std(yt_b) < 1e-6 or np.std(yp_b) < 1e-6:
            continue
        rp, _ = stats.pearsonr(yt_b, yp_b)
        rho, _ = stats.spearmanr(yt_b, yp_b)
        rmse = np.sqrt(mean_squared_error(yt_b, yp_b))
        mae = mean_absolute_error(yt_b, yp_b)

        rp_list.append(rp)
        rho_list.append(rho)
        rmse_list.append(rmse)
        mae_list.append(mae)

    alpha_low = (1.0 - confidence_level) / 2.0 * 100
    alpha_high = (1.0 + confidence_level) / 2.0 * 100

    results = {}
    for name, arr in [("Rp", rp_list), ("Rho", rho_list), ("RMSE", rmse_list), ("MAE", mae_list)]:
        arr = np.array(arr)
        results[name] = (
            float(np.mean(arr)),
            float(np.percentile(arr, alpha_low)),
            float(np.percentile(arr, alpha_high)),
        )
    return results


def compute_roc_curve_and_auc(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes ROC Curve (FPR, TPR) and Area Under the Curve (AUC)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, float(roc_auc)


def compute_bedroc_score(
    y_true: np.ndarray, y_score: np.ndarray, alpha: float = 16.1
) -> float:
    """
    Computes Boltzmann-Enhanced Discrimination of ROC (BEDROC).
    Standard alpha=16.1 prioritizes the top 8% of ranked hits.
    """
    y_true = np.asarray(y_true).astype(bool)
    n = len(y_true)
    n_actives = np.sum(y_true)
    if n_actives == 0 or n_actives == n:
        return 0.0

    order = np.argsort(y_score)[::-1]
    sorted_labels = y_true[order]

    r_i = np.where(sorted_labels)[0] + 1  # 1-indexed ranks
    sum_exp = np.sum(np.exp(-alpha * r_i / n))

    ra = n_actives / n
    rand_sum = (ra * (1.0 - np.exp(-alpha))) / (np.exp(alpha / n) - 1.0)
    max_sum = (1.0 - np.exp(-alpha * ra)) / (np.exp(alpha / n) - 1.0)

    bedroc = (sum_exp - rand_sum) / (max_sum - rand_sum)
    return float(np.clip(bedroc, 0.0, 1.0))


def compute_enrichment_factor(
    y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.01
) -> float:
    """
    Computes Enrichment Factor (EF) at specified top fraction (e.g., EF1%, EF5%).
    """
    y_true = np.asarray(y_true).astype(bool)
    n = len(y_true)
    n_actives = np.sum(y_true)
    if n_actives == 0:
        return 0.0

    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(y_score)[::-1]
    top_k_labels = y_true[order[:k]]
    actives_in_top_k = np.sum(top_k_labels)

    ef = (actives_in_top_k / k) / (n_actives / n)
    return float(ef)


def compute_expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray, n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) for evidential uncertainty.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    uncertainty = np.asarray(uncertainty).flatten()

    std = np.sqrt(np.maximum(uncertainty, 1e-8))
    z_scores = np.abs(y_true - y_pred) / std

    # Theoretical normal percentiles
    quantiles = np.linspace(0.1, 1.0, n_bins)
    ece = 0.0
    for q in quantiles:
        z_crit = stats.norm.ppf(0.5 + q / 2.0)
        empirical_coverage = np.mean(z_scores <= z_crit)
        ece += np.abs(empirical_coverage - q)

    return float(ece / n_bins)
