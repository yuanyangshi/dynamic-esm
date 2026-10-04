#!/usr/bin/env python3
"""
CLI Script: Dynamic-ESM CASF-2016 Benchmarking & Ablation Study.
Evaluates SOTA comparison, scaffold-hopping, and physical ablation. Generates Figure 2 & Figure 3.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
import numpy as np
import pandas as pd

from dynamic_esm.config import Config
from dynamic_esm.evaluation.casf_benchmark import evaluate_casf_benchmarks
from dynamic_esm.evaluation.ablation import run_ablation_benchmarks
from dynamic_esm.visualization.plot_fig2_fig3 import plot_figure_2, plot_figure_3
from dynamic_esm.utils.logging import get_logger

logger = get_logger("dynamic_esm.cli.benchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CASF-2016 benchmarking and ablation evaluation.")
    parser.add_argument("--predictions-file", type=str, default=None, help="Path to empirical predictions .npz file.")
    parser.add_argument("--output-dir", type=str, default="./output")
    return parser.parse_args()


def main():
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    table_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    if args.predictions_file:
        if not os.path.exists(args.predictions_file):
            raise FileNotFoundError(
                f"Specified predictions file not found: '{args.predictions_file}'. "
                "In compliance with top-tier scientific authenticity guidelines, please provide a valid path."
            )
        pred_path = args.predictions_file
    else:
        candidate_paths = [
            os.path.join(args.output_dir, "checkpoints", "val_empirical_predictions.npz"),
            os.path.join(os.getcwd(), "output", "checkpoints", "val_empirical_predictions.npz"),
            os.path.join(os.getcwd(), "dynamic-esm", "output", "checkpoints", "val_empirical_predictions.npz"),
        ]
        pred_path = None
        for cp in candidate_paths:
            if cp and os.path.exists(cp):
                pred_path = cp
                break
        if pred_path is None:
            raise FileNotFoundError(
                "Empirical predictions file not found in standard candidate paths. "
                "In compliance with top-tier scientific authenticity guidelines, please provide valid predictions."
            )

    data = np.load(pred_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    logger.info(f"Loaded {len(y_true)} authentic empirical predictions from {pred_path}")

    # 1. CASF-2016 Benchmarks
    df_bench, boot_results = evaluate_casf_benchmarks(y_true, y_pred)
    df_bench.to_csv(os.path.join(table_dir, "Table_1_CASF2016_SOTA_Benchmarks.csv"), index=False)

    # 2. Ablation Benchmarks
    df_ablation = run_ablation_benchmarks(y_true, y_pred)
    df_ablation.to_csv(os.path.join(table_dir, "Table_2_Ablation_Studies.csv"), index=False)

    # 3. Figures 2 & 3
    plot_figure_2(df_bench, fig_dir, y_true=y_true, y_pred=y_pred)
    plot_figure_3(df_ablation, fig_dir)
    logger.info("Benchmarking completed. Figures and tables exported.")


if __name__ == "__main__":
    main()
