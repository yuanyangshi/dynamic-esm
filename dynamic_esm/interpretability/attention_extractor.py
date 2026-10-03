"""
Cross-Attention Extraction & Residue Hotspot Mapping for Dynamic-ESM.
Projects multi-modal attention weights to 3D atomic coordinates and residue indices.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.interpretability")


def extract_pocket_attention(
    model: nn.Module,
    x: torch.Tensor,
    pos: torch.Tensor,
    seq_embeds: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs model forward pass and extracts residue-level attention weights.
    Returns:
        pocket_importance: (N_pocket,) normalized importance scores
        cross_attention_matrix: (N_pocket, N_seq)
    """
    model.eval()
    with torch.no_grad():
        _, _, _, _, attn_weights = model(x, pos, seq_embeds=seq_embeds)

    # attn_weights shape: (B, N_pocket, N_seq)
    if attn_weights is not None:
        attn_mat = attn_weights[0].cpu().numpy()
        # Sum over sequence dimension to obtain pocket residue importance
        pocket_importance = np.mean(attn_mat, axis=-1)
        # Normalize to [0, 1]
        p_min, p_max = np.min(pocket_importance), np.max(pocket_importance)
        if p_max > p_min:
            pocket_importance = (pocket_importance - p_min) / (p_max - p_min)
        else:
            pocket_importance = np.ones_like(pocket_importance)
    else:
        pocket_importance = np.ones(x.size(0))
        attn_mat = np.ones((x.size(0), 100))

    return pocket_importance, attn_mat
