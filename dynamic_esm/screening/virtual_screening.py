"""
Virtual Screening Engine & CASF-2016 Screening Power Evaluation.
Evaluates hit enrichment factors (EF1%, EF5%) and early recognition (BEDROC).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dynamic_esm.evaluation.metrics import (
    compute_roc_curve_and_auc,
    compute_bedroc_score,
    compute_enrichment_factor,
)
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.screening")


DEFAULT_HIERARCHICAL_POOLS = [
    ("Sub-Nanomolar Hits (pKd >= 8.5)", 8.5),
    ("Nanomolar Hits (pKd >= 8.0)", 8.0),
    ("High-Affinity Binders (pKd >= 7.5)", 7.5),
    ("Active Binders (pKd >= 7.0)", 7.0),
    ("Overall (Median Split)", None),
]


def evaluate_virtual_screening_targets(
    y_true: np.ndarray, y_score: np.ndarray, targets: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluates virtual screening performance across hierarchical affinity thresholds or target pools.
    """
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    records = []
    roc_dict = {}

    if targets is None:
        pool_specs = DEFAULT_HIERARCHICAL_POOLS
    else:
        pool_specs = [(t, None) for t in targets]

    for tgt, thresh in pool_specs:
        if thresh is not None and np.any(y_true >= thresh) and np.any(y_true < thresh):
            y_binary = (y_true >= thresh).astype(int)
        else:
            y_binary = (y_true > np.median(y_true)).astype(int)

        n_actives = int(np.sum(y_binary))
        n_total = len(y_binary)
        active_ratio = (n_actives / max(n_total, 1)) * 100.0

        fpr, tpr, roc_auc = compute_roc_curve_and_auc(y_binary, y_score)
        bedroc_16 = compute_bedroc_score(y_binary, y_score, alpha=16.1)
        bedroc_80 = compute_bedroc_score(y_binary, y_score, alpha=80.5)
        ef_1 = compute_enrichment_factor(y_binary, y_score, fraction=0.01)
        ef_5 = compute_enrichment_factor(y_binary, y_score, fraction=0.05)

        records.append({
            "Target Class": tgt,
            "Active Hits": n_actives,
            "Active Ratio (%)": round(active_ratio, 2),
            "ROC-AUC": roc_auc,
            "EF 1%": ef_1,
            "EF 5%": ef_5,
            "BEDROC (a=80.5)": bedroc_80,
            "BEDROC (a=16.1)": bedroc_16,
        })
        roc_dict[tgt] = (fpr, tpr, roc_auc)

    df_results = pd.DataFrame(records)
    logger.info("Virtual screening benchmark evaluation completed.")
    return df_results, roc_dict
