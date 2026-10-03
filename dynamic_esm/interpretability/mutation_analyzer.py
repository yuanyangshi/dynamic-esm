"""
Clinical Drug Resistance Mutation Analysis & Hotspot Verification.
Benchmarks Dynamic-ESM attention distributions against known therapeutic mutation sites.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats

from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.mutation")

CLINICAL_BENCHMARK_CASES = {
    "1IEP": {
        "target": "Abl Kinase",
        "drug": "Imatinib",
        "key_mutations": ["T315I", "E255K", "M351T"],
        "critical_residues": [315, 381, 255, 351],
        "mechanism": "Gatekeeper resistance and DFG-out conformation stabilization",
    },
    "2JIT": {
        "target": "EGFR Kinase",
        "drug": "Gefitinib",
        "key_mutations": ["T790M", "L858R", "G719S"],
        "critical_residues": [790, 858, 719, 745],
        "mechanism": "Gatekeeper steric clash and catalytic loop activation",
    },
    "3QRJ": {
        "target": "Abl T315I",
        "drug": "Ponatinib",
        "key_mutations": ["T315I"],
        "critical_residues": [315, 286, 381],
        "mechanism": "Overcoming bulky isoleucine gatekeeper mutation via ethynyl linker",
    },
    "6LUD": {
        "target": "EGFR T790M/C797S",
        "drug": "Osimertinib",
        "key_mutations": ["T790M", "C797S"],
        "critical_residues": [790, 797, 745],
        "mechanism": "Third-generation covalent resistance and cysteine covalent decoupling",
    },
    "6OIM": {
        "target": "KRAS G12C",
        "drug": "Sotorasib",
        "key_mutations": ["G12C"],
        "critical_residues": [12, 61, 95],
        "mechanism": "Switch-II pocket covalent engagement and allosteric locking",
    },
    "7VH8": {
        "target": "SARS-CoV-2 Mpro",
        "drug": "Nirmatrelvir",
        "key_mutations": ["H41Y", "C145A", "E166V"],
        "critical_residues": [41, 145, 166, 189],
        "mechanism": "Catalytic dyad covalent inhibition and oxyanion hole interaction",
    },
}


def analyze_mutation_hotspots(
    case_pdb: str, residue_numbers: List[int], attention_scores: np.ndarray
) -> Dict:
    """
    Evaluates whether clinically verified mutation sites coincide with top model attention peaks.
    """
    info = CLINICAL_BENCHMARK_CASES.get(case_pdb, {})
    crit_res = set(info.get("critical_residues", []))

    is_hotspot = np.array([r in crit_res for r in residue_numbers])
    hotspot_scores = attention_scores[is_hotspot] if np.any(is_hotspot) else np.array([])
    background_scores = attention_scores[~is_hotspot] if np.any(~is_hotspot) else np.array([])

    mean_hotspot = float(np.mean(hotspot_scores)) if len(hotspot_scores) > 0 else 0.0
    mean_bg = float(np.mean(background_scores)) if len(background_scores) > 0 else 0.0

    # Mann-Whitney U test
    if len(hotspot_scores) > 0 and len(background_scores) > 0:
        _, p_val = stats.mannwhitneyu(hotspot_scores, background_scores, alternative="greater")
    else:
        p_val = 1.0

    return {
        "pdb_id": case_pdb,
        "target": info.get("target", "Target"),
        "drug": info.get("drug", "Ligand"),
        "mean_hotspot_attention": mean_hotspot,
        "mean_background_attention": mean_bg,
        "enrichment_ratio": mean_hotspot / max(mean_bg, 1e-6),
        "p_value": float(p_val),
    }
