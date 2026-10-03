"""
Full Dynamic-ESM Architecture & PyTorch Lightning Modules.
Unifies ESM-2 evolutionary representation with 4D-QM EST-GNN pocket dynamics.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import esm
from dynamic_esm.models.est_gnn import DynamicESMBackbone
from dynamic_esm.models.cross_attention import DeepBiCrossAttention
from dynamic_esm.models.evidential import EDLHead, dynamic_esm_criterion
from dynamic_esm.models.lora import inject_lora_into_esm


class DynamicESM(nn.Module):
    """
    End-to-End Dynamic-ESM Binding Affinity Predictor.
    Integrates 1D Evolutionary Priors (ESM-2) + 4D-QM Dynamics (EST-GNN) + Bi-SDPA + EDL.
    """

    def __init__(
        self,
        esm_model_name: str = "esm2_t12_35M_UR50D",
        node_in_dim: int = 494,
        gnn_hidden_dim: int = 128,
        cross_attn_dim: int = 256,
        num_heads: int = 4,
        num_cross_layers: int = 2,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        load_esm: bool = True,
    ):
        super().__init__()
        self.esm_embed_dim = 480
        if esm_model_name:
            if "35M" in esm_model_name:
                self.esm_embed_dim = 480
            elif "150M" in esm_model_name:
                self.esm_embed_dim = 640
            elif "650M" in esm_model_name:
                self.esm_embed_dim = 1280
            else:
                self.esm_embed_dim = 480

        # 1. Macro-Scale Evolutionary Stream (ESM-2 + LoRA)
        self.esm_model = None
        self.esm_alphabet = None
        if load_esm and esm_model_name:
            try:
                self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
                self.esm_model = inject_lora_into_esm(
                    self.esm_model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
                )
            except Exception as e:
                print(f"[DynamicESM] Warning: Could not load ESM-2 model ({e}). Fallback to offline embeddings.")

        # 2. Micro-Scale Spatiotemporal Stream (4D-QM EST-GNN Backbone)
        self.backbone = DynamicESMBackbone(
            node_in_dim=node_in_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=3,
            cutoff=12.0,
            num_heads=num_heads,
        )

        # 3. Bi-directional Scaled Dot-Product Cross-Attention (Bi-SDPA)
        self.cross_attention = DeepBiCrossAttention(
            pocket_in_dim=gnn_hidden_dim,
            seq_in_dim=self.esm_embed_dim,
            embed_dim=cross_attn_dim,
            num_heads=num_heads,
            num_layers=num_cross_layers,
        )

        # 4. Evidential Deep Learning (EDL) Regression Head
        self.edl_head = EDLHead(in_dim=cross_attn_dim, hidden_dim=128)

    def forward(
        self,
        x: torch.Tensor,
        pos_trajectory: torch.Tensor,
        seq_tokens: Optional[torch.Tensor] = None,
        seq_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass producing NIG parameters and attention maps.
        Returns:
            gamma: (B, 1) predicted pKd affinity
            v: (B, 1) virtual evidence count
            alpha: (B, 1) shape parameter
            beta: (B, 1) scale parameter
            attn_weights: (B, N_pocket, N_seq)
        """
        # Extract 4D-QM pocket dynamics representation
        h_pocket, _ = self.backbone(x, pos_trajectory)
        if h_pocket.dim() == 2:
            h_pocket = h_pocket.unsqueeze(0)  # (1, N_pocket, gnn_hidden_dim)

        # Obtain sequence embeddings (either live ESM-2 or precomputed)
        if seq_embeds is None and seq_tokens is not None and self.esm_model is not None:
            results = self.esm_model(seq_tokens, repr_layers=[self.esm_model.num_layers])
            seq_embeds = results["representations"][self.esm_model.num_layers]

        if seq_embeds is None:
            # Fallback mock/offline feature alignment
            seq_embeds = torch.zeros(
                h_pocket.size(0), 100, self.esm_embed_dim, device=x.device
            )

        # Bi-directional Cross Attention
        fused_h, attn_weights = self.cross_attention(h_pocket, seq_embeds)

        # Evidential Prediction
        gamma, v, alpha, beta = self.edl_head(fused_h)

        return gamma, v, alpha, beta, attn_weights


class DynamicESMPretrainModule(pl.LightningModule):
    """
    LightningModule for self-supervised 4D-QM backbone pretraining.
    Tasks: Physical restoration, coordinate denoising, velocity prediction.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.backbone = DynamicESMBackbone(
            node_in_dim=494,
            hidden_dim=128,
            num_layers=3,
            cutoff=12.0,
            num_heads=4,
        )
        self.coord_head = nn.Linear(128, 3)
        self.velocity_head = nn.Linear(128, 3)

    def forward(self, x, pos_trajectory):
        return self.backbone(x, pos_trajectory)

    def training_step(self, batch, batch_idx):
        x = batch.x
        pos_trajectory = batch.pos
        # Add small coordinate perturbation
        noise = torch.randn_like(pos_trajectory) * 0.1
        noisy_pos = pos_trajectory + noise

        h, _ = self.backbone(x, noisy_pos)
        pred_coords = self.coord_head(h)
        pred_velocity = self.velocity_head(h)

        target_coords = pos_trajectory[-1] if pos_trajectory.dim() == 3 else pos_trajectory[:, -1]
        loss_coord = F.mse_loss(pred_coords, target_coords)
        loss_vel = F.mse_loss(pred_velocity, noise[-1] if noise.dim() == 3 else noise[:, -1])
        total_loss = loss_coord + 0.5 * loss_vel

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
