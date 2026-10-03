"""
Unit tests for mechanistic interpretability and PyMOL script exporter.
"""

import os
import tempfile
import numpy as np
import torch
from dynamic_esm.models.dynamic_esm_model import DynamicESM
from dynamic_esm.interpretability.attention_extractor import extract_pocket_attention
from dynamic_esm.interpretability.mutation_analyzer import analyze_mutation_hotspots
from dynamic_esm.interpretability.pymol_visualizer import (
    export_pdb_with_attention,
    generate_pymol_script,
)


def test_analyze_mutation_hotspots():
    res_nums = [310, 315, 320, 381, 400]
    # Give high attention to 315 and 381
    attn = np.array([0.1, 0.95, 0.2, 0.90, 0.05])
    res = analyze_mutation_hotspots("1IEP", res_nums, attn)

    assert res["pdb_id"] == "1IEP"
    assert res["target"] == "Abl Kinase"
    assert res["mean_hotspot_attention"] > res["mean_background_attention"]
    assert res["enrichment_ratio"] > 1.0


def test_pymol_script_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_pdb = os.path.join(tmp_dir, "test.pdb")
        with open(dummy_pdb, "w") as f:
            f.write("ATOM      1  N   MET A   1      20.154  14.288  10.543  1.00 20.00           N\n")
            f.write("ATOM      2  CA  MET A   1      21.234  15.123  11.234  1.00 20.00           C\n")

        annotated_pdb = os.path.join(tmp_dir, "test_attn.pdb")
        out_pml = os.path.join(tmp_dir, "test_render.pml")

        export_pdb_with_attention(dummy_pdb, annotated_pdb, {1: 0.85})
        assert os.path.exists(annotated_pdb)

        generate_pymol_script(annotated_pdb, out_pml, critical_residues=[1])
        assert os.path.exists(out_pml)
        with open(out_pml, "r") as f:
            pml_content = f.read()
        assert "spectrum b, blue_white_red" in pml_content
        assert "ray 2400, 1800" in pml_content


def test_extract_pocket_attention():
    model = DynamicESM(
        esm_model_name=None,
        node_in_dim=32,
        gnn_hidden_dim=64,
        cross_attn_dim=128,
        num_heads=4,
        num_cross_layers=1,
    )
    x = torch.randn(20, 32)
    pos = torch.randn(5, 20, 3)
    seq_embeds = torch.randn(1, 50, 480)

    importance, attn_mat = extract_pocket_attention(model, x, pos, seq_embeds=seq_embeds)
    assert importance.shape == (20,)
    assert attn_mat.shape == (20, 50)
    assert np.all(importance >= 0.0)
    assert np.all(importance <= 1.0)

