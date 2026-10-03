"""
End-to-End Fine-Tuning Engine for Dynamic-ESM.
Optimizes Macro-Micro Multimodal Fusion via Evidential Deep Learning (EDL).
"""

import os
import time
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from dynamic_esm.config import Config
from dynamic_esm.data.dataset import DynamicESMDataset, custom_collate_fn
from dynamic_esm.models.dynamic_esm_model import DynamicESM
from dynamic_esm.models.evidential import dynamic_esm_criterion, EDLHead
from dynamic_esm.utils.hardware import setup_hardware, set_seed
from dynamic_esm.utils.io import safe_load_checkpoint, safe_save_checkpoint
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.finetune")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluates Dynamic-ESM model on validation/test set.
    Returns:
        all_y: ground-truth affinities
        all_gamma: predicted affinities
        aleatoric: aleatoric uncertainties
        epistemic: epistemic uncertainties
        total_uncert: combined predictive uncertainties
        val_loss: mean evaluation loss
    """
    model.eval()
    all_y, all_gamma = [], []
    all_v, all_alpha, all_beta = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            x = batch.x.to(device)
            pos = batch.pos.to(device)
            y = getattr(batch, "y", torch.zeros(1, 1)).to(device)
            if y.dim() == 1:
                y = y.unsqueeze(-1)

            gamma, v, alpha, beta, _ = model(x, pos)
            all_y.append(y.cpu().numpy())
            all_gamma.append(gamma.cpu().numpy())
            all_v.append(v.cpu().numpy())
            all_alpha.append(alpha.cpu().numpy())
            all_beta.append(beta.cpu().numpy())

    if not all_y:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 0.0

    all_y = np.concatenate(all_y, axis=0).flatten()
    all_gamma = np.concatenate(all_gamma, axis=0).flatten()
    all_v = np.concatenate(all_v, axis=0).flatten()
    all_alpha = np.concatenate(all_alpha, axis=0).flatten()
    all_beta = np.concatenate(all_beta, axis=0).flatten()

    denom = np.maximum(all_alpha - 1.0, 1e-6)
    aleatoric = all_beta / denom
    epistemic = all_beta / (np.maximum(all_v, 1e-6) * denom)
    total_uncert = aleatoric + epistemic

    rmse = np.sqrt(np.mean((all_y - all_gamma) ** 2)) if len(all_y) > 0 else 0.0
    return all_y, all_gamma, aleatoric, epistemic, total_uncert, float(rmse)


def run_finetuning(cfg: Config) -> str:
    """
    Main training execution for Dynamic-ESM.
    Returns path to the best fine-tuned checkpoint.
    """
    set_seed(cfg.SEED)
    device = setup_hardware()
    cfg.make_dirs()

    logger.info("Initializing Fine-tuning DataLoaders...")
    train_dataset = DynamicESMDataset(cfg.DATA_DIR, split="train")
    val_dataset = DynamicESMDataset(cfg.DATA_DIR, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )

    logger.info("Instantiating Dynamic-ESM Model...")
    model = DynamicESM(
        esm_model_name=cfg.ESM_MODEL_NAME,
        node_in_dim=494,
        gnn_hidden_dim=cfg.GNN_HIDDEN_DIM,
        cross_attn_dim=cfg.CROSS_ATTN_DIM,
        num_heads=cfg.NUM_CROSS_HEADS,
        num_cross_layers=cfg.NUM_CROSS_LAYERS,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
    ).to(device)

    # Load pre-trained backbone if available
    backbone_ckpt = os.path.join(cfg.PRETRAIN_DIR, "backbone_pretrain_best.ckpt")
    if os.path.exists(backbone_ckpt):
        safe_load_checkpoint(model.backbone, backbone_ckpt, device=device)

    # Optimizer with differential learning rates
    params = [
        {"params": model.backbone.parameters(), "lr": cfg.BACKBONE_LR},
        {"params": model.cross_attention.parameters(), "lr": cfg.LEARNING_RATE},
        {"params": model.edl_head.parameters(), "lr": cfg.LEARNING_RATE},
    ]
    if model.esm_model is not None:
        lora_params = [p for p in model.esm_model.parameters() if p.requires_grad]
        params.append({"params": lora_params, "lr": cfg.ESM_LORA_LR})

    optimizer = torch.optim.AdamW(params, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    scaler = GradScaler(enabled=cfg.USE_AMP)

    best_rmse = float("inf")
    best_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "dynamic_esm_main_best.ckpt")
    history_records = []

    logger.info("Starting Fine-tuning Loop...")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        total_loss, total_steps = 0.0, 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue
            x = batch.x.to(device)
            pos = batch.pos.to(device)
            y = getattr(batch, "y", torch.zeros(x.size(0), 1)).to(device)
            if y.dim() == 1:
                y = y.unsqueeze(-1)

            with autocast(enabled=cfg.USE_AMP):
                gamma, v, alpha, beta, _ = model(x, pos)
                loss, loss_dict = dynamic_esm_criterion(
                    y, gamma, v, alpha, beta,
                    lambda_reg=cfg.LAMBDA_REG,
                    lambda_mse=cfg.LAMBDA_MSE,
                    lambda_rank=cfg.LAMBDA_RANK,
                    margin=cfg.RANKING_MARGIN,
                )
                loss = loss / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg.GRAD_ACCUM_STEPS
            total_steps += 1

        scheduler.step()
        train_loss = total_loss / max(total_steps, 1)

        # Validation
        y_val, gamma_val, aleatoric_val, epistemic_val, total_uncert_val, val_rmse = evaluate_model(
            model, val_loader, device
        )
        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:02d}/{cfg.NUM_EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Time: {elapsed:.1f}s"
        )

        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse": val_rmse,
        })

        if val_rmse < best_rmse and len(y_val) > 0:
            best_rmse = val_rmse
            safe_save_checkpoint(
                {"state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch, "best_rmse": best_rmse},
                best_ckpt_path,
            )
            # Save validation predictions for downstream calibration & benchmark
            np.savez_compressed(
                os.path.join(cfg.CHECKPOINT_DIR, "val_empirical_predictions.npz"),
                y_true=y_val,
                y_pred=gamma_val,
                aleatoric=aleatoric_val,
                epistemic=epistemic_val,
                total_uncert=total_uncert_val,
            )

    # Save training history
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(os.path.join(cfg.WORKING_DIR, "finetune_history.csv"), index=False)

    logger.info(f"Fine-tuning complete. Best Val RMSE: {best_rmse:.4f}. Model: {best_ckpt_path}")
    return best_ckpt_path
