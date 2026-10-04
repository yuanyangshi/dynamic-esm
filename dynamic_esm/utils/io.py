"""
Safe checkpoint loading and weight serialization utilities.
"""

import os
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.io")


def safe_save_checkpoint(
    state: Dict,
    save_path: str,
    is_best: bool = False,
    best_path: Optional[str] = None
) -> None:
    """Save PyTorch checkpoint safely with optional best model mirroring."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(state, save_path)
    logger.info(f"Checkpoint saved: {save_path}")
    if is_best and best_path:
        os.makedirs(os.path.dirname(os.path.abspath(best_path)), exist_ok=True)
        torch.save(state, best_path)
        logger.info(f"Best model updated: {best_path}")


def safe_load_checkpoint(
    model: nn.Module,
    checkpoint_candidates: Union[str, List[str]],
    device: Union[str, torch.device] = "cpu",
    strict: bool = False
) -> bool:
    """
    Search and load the first available checkpoint from a list of candidates.
    Handles both raw state_dicts and PyTorch Lightning checkpoint structures.
    """
    if isinstance(checkpoint_candidates, str):
        checkpoint_candidates = [checkpoint_candidates]

    resolved_path = None
    for candidate in checkpoint_candidates:
        if candidate and os.path.exists(candidate) and os.path.isfile(candidate):
            resolved_path = candidate
            break

    if not resolved_path:
        logger.warning(
            f"No valid checkpoint found among candidates: {checkpoint_candidates}"
        )
        return False

    logger.info(f"Loading checkpoint from: {resolved_path}")
    try:
        try:
            from dynamic_esm.config import Config
            import sys
            if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "Config"):
                sys.modules["__main__"].Config = Config
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([Config])
        except Exception:
            pass

        try:
            ckpt = torch.load(resolved_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resolved_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)

        # Clean prefix mismatches (e.g. from Lightning wrapper 'model.', 'module.')
        cleaned_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            for prefix in ["model.", "module."]:
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            cleaned_dict[clean_k] = v

        # If loading into a standalone backbone, strip est_gnn. or backbone. prefix if present
        if not hasattr(model, "est_gnn") and any(k.startswith("est_gnn.") for k in cleaned_dict):
            cleaned_dict = {
                (k[8:] if k.startswith("est_gnn.") else k): v
                for k, v in cleaned_dict.items()
            }
        elif not hasattr(model, "backbone") and any(k.startswith("backbone.") for k in cleaned_dict):
            cleaned_dict = {
                (k[9:] if k.startswith("backbone.") else k): v
                for k, v in cleaned_dict.items()
            }

        # Dynamically attach tensor_input_adapter if present in state_dict for standalone backbone
        if hasattr(model, "tensor_input_adapter") and model.tensor_input_adapter is None:
            if "tensor_input_adapter.weight" in cleaned_dict or "est_gnn.tensor_input_adapter.weight" in cleaned_dict:
                model.tensor_input_adapter = nn.Linear(model.hidden_dim, model.hidden_dim).to(device)

        missing, unexpected = model.load_state_dict(cleaned_dict, strict=strict)
        logger.info(
            f"Checkpoint loaded successfully from {resolved_path}. (Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)})"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {resolved_path}: {e}")
        return False
