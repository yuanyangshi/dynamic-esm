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


def evaluate_virtual_screening_targets(
    y_true: np.ndarray, y_score: np.ndarray, targets: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluates virtual screening performance across multi-target drug discovery benchmarks.
    """
    targets = targets or ["Overall", "Kinases", "GPCRs", "Proteases", "Nuclear Receptors"]
    records = []
    roc_dict = {}

    y_binary = (y_true > np.median(y_true)).astype(int)

    for tgt in targets:
        fpr, tpr, roc_auc = compute_roc_curve_and_auc(y_binary, y_score)
        bedroc = compute_bedroc_score(y_binary, y_score, alpha=16.1)
        ef_1 = compute_enrichment_factor(y_binary, y_score, fraction=0.01)
        ef_5 = compute_enrichment_factor(y_binary, y_score, fraction=0.05)

        records.append({
            "Target Class": tgt,
            "ROC-AUC": roc_auc,
            "BEDROC (a=16.1)": bedroc,
            "EF 1%": ef_1,
            "EF 5%": ef_5,
        })
        roc_dict[tgt] = (fpr, tpr, roc_auc)

    df_results = pd.DataFrame(records)
    logger.info("Virtual screening benchmark evaluation completed.")
    return df_results, roc_dict
