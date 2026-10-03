#!/usr/bin/env python3
"""
CLI Script: Dynamic-ESM 4D-QM Backbone Pretraining Pipeline.
Executes self-supervised representation learning on molecular dynamics trajectories.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from dynamic_esm.config import Config
from dynamic_esm.training.pretrain import run_backbone_pretraining
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.cli.pretrain")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain 4D-QM EST-GNN spatial-temporal backbone.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory with processed graph datasets.")
    parser.add_argument("--checkpoint-dir", type=str, default="./output/checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of pretraining epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.DATA_DIR = args.data_dir
    cfg.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.NUM_EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr

    logger.info("Launching Backbone Pretraining Pipeline...")
    best_path = run_backbone_pretraining(cfg)
    logger.info(f"Done. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
