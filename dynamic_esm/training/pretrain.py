"""
Backbone Pretraining Engine for Dynamic-ESM.
Executes self-supervised 4D-QM physical restoration and spatiotemporal representation learning.
"""

import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dynamic_esm.config import Config
from dynamic_esm.data.dataset import DynamicESMDataset, custom_collate_fn
from dynamic_esm.models.dynamic_esm_model import DynamicESMPretrainModule
from dynamic_esm.utils.hardware import setup_hardware, set_seed
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.pretrain")


def run_backbone_pretraining(cfg: Config) -> str:
    """
    Run self-supervised backbone pretraining on MISATO MD trajectories.
    Returns path to the best pre-trained checkpoint.
    """
    set_seed(cfg.SEED)
    device = setup_hardware()
    cfg.make_dirs()

    logger.info("Initializing Pretraining DataLoaders...")
    train_dataset = DynamicESMDataset(cfg.DATA_DIR, split="train")
    val_dataset = DynamicESMDataset(cfg.DATA_DIR, split="val")

    if len(train_dataset) == 0:
        logger.warning(
            f"No training data found in {cfg.DATA_DIR}. Please check DATA_DIR."
        )

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

    logger.info("Instantiating DynamicESMPretrainModule...")
    module = DynamicESMPretrainModule(cfg=cfg)

    ckpt_callback = ModelCheckpoint(
        dirpath=cfg.CHECKPOINT_DIR,
        filename="backbone_pretrain_best",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.NUM_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_callback],
        precision="16-mixed" if (torch.cuda.is_available() and cfg.USE_AMP) else "32",
        accumulate_grad_batches=cfg.GRAD_ACCUM_STEPS,
    )

    logger.info("Starting Backbone Pretraining...")
    trainer.fit(module, train_dataloaders=train_loader)

    best_path = ckpt_callback.best_model_path
    logger.info(f"Backbone pretraining completed. Best checkpoint: {best_path}")
    return best_path
