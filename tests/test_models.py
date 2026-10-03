"""
Unit tests for Dynamic-ESM neural network architectures and tensor dimensions.
"""

import torch
import torch.nn as nn


from dynamic_esm.models.est_gnn import (
    GaussianRBF,
    EquivariantMessagePassingLayer,
    TemporalAttentionFusion,
    DynamicESMBackbone,
)
from dynamic_esm.models.cross_attention import (
    BiCrossAttentionLayer,
    DeepBiCrossAttention,
)
from dynamic_esm.models.evidential import EDLHead, evidential_nll_loss
from dynamic_esm.models.lora import LoRALinear
from dynamic_esm.models.dynamic_esm_model import DynamicESM, DynamicESMPretrainModule
from torch_geometric.data import Data


def test_gaussian_rbf():
    rbf = GaussianRBF(num_rbf=32, cutoff=12.0)
    dist = torch.tensor([[1.0], [5.0], [10.0]])
    out = rbf(dist)
    assert out.shape == (3, 32)
    assert not torch.isnan(out).any()


def test_equivariant_message_passing():
    layer = EquivariantMessagePassingLayer(hidden_dim=64, edge_dim=32)
    h = torch.randn(10, 64)
    pos = torch.randn(10, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_rbf = torch.randn(4, 32)

    h_out, pos_out = layer(h, pos, edge_index, edge_rbf)
    assert h_out.shape == (10, 64)
    assert pos_out.shape == (10, 3)
    assert not torch.isnan(h_out).any()
    assert not torch.isnan(pos_out).any()


def test_temporal_attention_fusion():
    fusion = TemporalAttentionFusion(hidden_dim=64, num_heads=4)
    h_frames = torch.randn(10, 5, 64)  # 10 atoms, 5 frames, 64 dim
    fused_h, attn_w = fusion(h_frames)
    assert fused_h.shape == (10, 64)
    assert attn_w.shape == (10, 5, 5)


def test_backbone():
    backbone = DynamicESMBackbone(node_in_dim=32, hidden_dim=64, num_layers=2)
    x = torch.randn(15, 32)
    pos_traj = torch.randn(5, 15, 3)  # 5 frames, 15 atoms, 3 coords

    fused_h, attn_w = backbone(x, pos_traj)
    assert fused_h.shape == (15, 64)


def test_cross_attention():
    cross_attn = DeepBiCrossAttention(
        pocket_in_dim=64, seq_in_dim=128, embed_dim=128, num_heads=4, num_layers=1
    )
    h_p = torch.randn(2, 20, 64)   # batch=2, 20 pocket atoms
    h_s = torch.randn(2, 50, 128)  # batch=2, 50 residues

    fused, attn_weights = cross_attn(h_p, h_s)
    assert fused.shape == (2, 128)
    assert attn_weights.shape == (2, 20, 50)


def test_edl_head():
    head = EDLHead(in_dim=128, hidden_dim=64)
    x = torch.randn(4, 128)
    gamma, v, alpha, beta = head(x)

    assert gamma.shape == (4, 1)
    assert v.shape == (4, 1)
    assert alpha.shape == (4, 1)
    assert beta.shape == (4, 1)

    assert (v > 0).all()
    assert (alpha > 1).all()
    assert (beta > 0).all()

    aleatoric, epistemic, total = head.compute_uncertainty(v, alpha, beta)
    assert aleatoric.shape == (4, 1)
    assert (aleatoric > 0).all()
    assert (epistemic > 0).all()


def test_lora_linear():
    base = nn.Linear(64, 64)
    lora = LoRALinear(base, rank=8, alpha=16.0)
    x = torch.randn(4, 64)
    out = lora(x)
    assert out.shape == (4, 64)
    assert not base.weight.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_dynamic_esm_model():
    model = DynamicESM(
        esm_model_name=None,
        node_in_dim=32,
        gnn_hidden_dim=64,
        cross_attn_dim=128,
        num_heads=4,
        num_cross_layers=1,
    )
    x = torch.randn(15, 32)
    pos_traj = torch.randn(5, 15, 3)
    seq_embeds = torch.randn(1, 40, 480)
    gamma, v, alpha, beta, attn = model(x, pos_traj, seq_embeds=seq_embeds)
    assert gamma.shape == (1, 1)
    assert v.shape == (1, 1)
    assert alpha.shape == (1, 1)
    assert beta.shape == (1, 1)
    assert attn.shape == (1, 15, 40)


def test_dynamic_esm_pretrain_module():
    module = DynamicESMPretrainModule()
    batch = Data(x=torch.randn(10, 494), pos=torch.randn(5, 10, 3))
    loss = module.training_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0


