"""
Unit tests for publication-grade visualization and figure generation scripts.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Force headless non-GUI backend

from dynamic_esm.visualization.plot_fig1 import generate_figure_1
from dynamic_esm.visualization.plot_fig2_fig3 import plot_figure_2, plot_figure_3
from dynamic_esm.visualization.plot_fig4 import plot_figure_4
from dynamic_esm.visualization.plot_fig5 import plot_figure_5
from dynamic_esm.evaluation.casf_benchmark import evaluate_casf_benchmarks
from dynamic_esm.evaluation.ablation import run_ablation_benchmarks


def test_figure_generation_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generate empirical test inputs
        rng = np.random.default_rng(42)
        y_true = rng.uniform(4.0, 10.0, 100)
        y_pred = y_true + rng.normal(0, 0.4, 100)

        # 1. Figure 1
        generate_figure_1(out_dir=tmp_dir)
        fig1_png = os.path.join(tmp_dir, "Figure_1_Master_Architecture_Calibration.png")
        fig1_pdf = os.path.join(tmp_dir, "Figure_1_Master_Architecture_Calibration.pdf")
        assert os.path.exists(fig1_png) and os.path.getsize(fig1_png) > 1000
        assert os.path.exists(fig1_pdf) and os.path.getsize(fig1_pdf) > 1000

        # 2. Figure 2 & 3
        df_bench, _ = evaluate_casf_benchmarks(y_true, y_pred, n_bootstraps=20)
        plot_figure_2(df_bench, out_dir=tmp_dir, y_true=y_true, y_pred=y_pred)
        fig2_png = os.path.join(tmp_dir, "Figure_2_CASF2016_SOTA_ScaffoldHopping.png")
        assert os.path.exists(fig2_png) and os.path.getsize(fig2_png) > 1000

        df_ablation = run_ablation_benchmarks(y_true, y_pred)
        plot_figure_3(df_ablation, out_dir=tmp_dir)
        fig3_png = os.path.join(tmp_dir, "Figure_3_Physical_Dynamics_Ablation.png")
        assert os.path.exists(fig3_png) and os.path.getsize(fig3_png) > 1000

        # 3. Figure 4
        plot_figure_4({}, out_dir=tmp_dir)
        fig4_png = os.path.join(tmp_dir, "Figure_4_Mechanistic_Interpretability.png")
        assert os.path.exists(fig4_png) and os.path.getsize(fig4_png) > 1000

        # 4. Figure 5
        df_screen = pd.DataFrame([
            {"Target Class": "Kinases", "ROC-AUC": 0.92, "BEDROC (a=16.1)": 0.88, "EF 1%": 18.5, "EF 5%": 8.2},
            {"Target Class": "GPCRs", "ROC-AUC": 0.89, "BEDROC (a=16.1)": 0.84, "EF 1%": 15.2, "EF 5%": 7.1},
        ])
        df_rect = pd.DataFrame([
            {"Conformation Source": "Raw AlphaFold 3 (Static)", "Pearson Rp": 0.76, "RMSE": 1.35, "Delta_RMSE": 0.24},
            {"Conformation Source": "Dynamic-ESM Rectified", "Pearson Rp": 0.84, "RMSE": 1.11, "Delta_RMSE": 0.0},
        ])
        thermo_data = {
            "delta_g_exp": np.linspace(-12.0, -4.0, 50),
            "delta_g_pred": np.linspace(-12.0, -4.0, 50) + np.random.normal(0, 0.3, 50),
        }
        plot_figure_5(df_screen, df_rect, thermo_data, out_dir=tmp_dir)
        fig5_png = os.path.join(tmp_dir, "Figure_5_Virtual_Screening_AF3_Rectification.png")
        assert os.path.exists(fig5_png) and os.path.getsize(fig5_png) > 1000
