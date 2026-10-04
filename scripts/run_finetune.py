#!/usr/bin/env python3
"""
CLI Script: Dynamic-ESM End-to-End Fine-Tuning Pipeline.
Optimizes multimodal fusion (ESM-2 LoRA + EST-GNN + Bi-SDPA) via Evidential Deep Learning.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from dynamic_esm.config import Config
from dynamic_esm.training.finetune import run_finetuning
from dynamic_esm.visualization.plot_fig1 import generate_figure_1
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.cli.finetune")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Dynamic-ESM on binding affinity prediction.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory with processed graph datasets.")
    parser.add_argument("--pretrain-dir", type=str, default="./output/checkpoints", help="Directory with pre-trained backbone.")
    parser.add_argument("--checkpoint-dir", type=str, default="./output/checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--figures-dir", type=str, default="./output/figures", help="Directory to save publication figures.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.DATA_DIR = args.data_dir
    cfg.PRETRAIN_DIR = args.pretrain_dir
    cfg.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.FIGURES_DIR = args.figures_dir
    cfg.NUM_EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr

    logger.info("Launching Fine-tuning Pipeline...")
    best_path = run_finetuning(cfg)
    logger.info(f"Fine-tuning complete. Best model: {best_path}")

    logger.info("Generating Publication Figure 1...")
    generate_figure_1(out_dir=cfg.FIGURES_DIR)
    logger.info("Figure 1 generated successfully.")


if __name__ == "__main__":
    main()
