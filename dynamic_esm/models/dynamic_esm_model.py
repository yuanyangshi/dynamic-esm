"""
Full Dynamic-ESM Architecture & PyTorch Lightning Modules.
Unifies ESM-2 evolutionary representation with 4D-QM EST-GNN pocket dynamics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
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
        esm_model_name: Optional[str] = "esm2_t12_35M_UR50D",
        node_in_dim: int = 654,
        gnn_hidden_dim: int = 256,
        cross_attn_dim: int = 512,
        num_heads: int = 8,
        num_cross_layers: int = 3,
        num_gnn_layers: int = 6,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        load_esm: bool = True,
        dropout: float = 0.1,
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
        self.esm = None
        self.alphabet = None
        self.batch_converter = None
        self._esm_mock = True
        if load_esm and esm_model_name:
            try:
                import esm
                esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.esm = inject_lora_into_esm(
                    esm_model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
                )
                self._esm_mock = False
            except Exception:
                self.esm = nn.Embedding(33, self.esm_embed_dim)
        else:
            self.esm = nn.Embedding(33, self.esm_embed_dim)

        # 2. Micro-Scale Spatiotemporal Stream (4D-QM EST-GNN Backbone)
        self.est_gnn = DynamicESMBackbone(
            node_in_dim=node_in_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            cutoff=6.5,
            num_heads=num_heads,
            dropout=dropout,
            include_adapter=True,
        )

        # 3. Bi-directional Scaled Dot-Product Cross-Attention (Bi-SDPA)
        self.fusion = DeepBiCrossAttention(
            seq_dim=self.esm_embed_dim,
            geo_dim=gnn_hidden_dim,
            hidden_dim=cross_attn_dim,
            num_heads=num_heads,
            num_layers=num_cross_layers,
            dropout=dropout,
        )

        # 4. Evidential Deep Learning (EDL) Regression Head
        self.head = EDLHead(in_dim=cross_attn_dim)

    # Aliases for backward compatibility
    @property
    def esm_model(self) -> Optional[nn.Module]:
        return self.esm

    @property
    def backbone(self) -> DynamicESMBackbone:
        return self.est_gnn

    @property
    def cross_attention(self) -> DeepBiCrossAttention:
        return self.fusion

    @property
    def edl_head(self) -> EDLHead:
        return self.head

    def forward(
        self,
        seq_tokens_or_x: Union[torch.Tensor, Dict[str, Any]],
        geo_features_or_pos: Optional[torch.Tensor] = None,
        seq_tokens: Optional[torch.Tensor] = None,
        seq_embeds: Optional[torch.Tensor] = None,
        seq_padding_mask: Optional[torch.Tensor] = None,
        geo_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ]:
        """
        Polymorphic forward pass producing NIG parameters and attention maps.
        Supports:
        1. Coordinate trajectory mode: model(x, pos_trajectory, seq_embeds=seq_embeds)
        2. Sequence and 4D pocket tensor mode: model(seq_tokens, geo_features)
        3. Batched dictionary input: model(batch)
        """
        if isinstance(seq_tokens_or_x, dict):
            batch = seq_tokens_or_x
            seq_tokens_or_x = batch.get("seq_tokens", None)
            geo_features_or_pos = batch.get("geo_features", None)
            seq_padding_mask = batch.get("seq_padding_mask", None)
            geo_padding_mask = batch.get("geo_padding_mask", None)

        # Mode 1: Coordinate trajectory input (x: atom features, pos: coordinate trajectory)
        if geo_features_or_pos is not None and geo_features_or_pos.size(-1) == 3:
            x = seq_tokens_or_x
            pos = geo_features_or_pos
            h_pocket, _ = self.est_gnn(x, pos)
            if h_pocket.dim() == 2:
                bsz = seq_embeds.size(0) if seq_embeds is not None else (seq_tokens.size(0) if seq_tokens is not None else 1)
                h_pocket = h_pocket.unsqueeze(0)
                if bsz > 1:
                    h_pocket = h_pocket.expand(bsz, -1, -1)

            if seq_embeds is None:
                if seq_tokens is not None and not self._esm_mock and hasattr(self.esm, "num_layers"):
                    res = self.esm(seq_tokens, repr_layers=[self.esm.num_layers])
                    seq_embeds = res["representations"][self.esm.num_layers]
                else:
                    seq_embeds = torch.zeros(h_pocket.size(0), 100, self.esm_embed_dim, device=x.device)

            fused_h, attn_w = self.fusion(seq_embeds, h_pocket, return_attn=return_attn)
            gamma, v, alpha, beta = self.head(fused_h)
            if return_attn:
                return gamma, v, alpha, beta, attn_w
            return gamma, v, alpha, beta

        # Mode 2: Macro-Micro tensor fusion mode
        seq_input = seq_tokens_or_x
        geo_input = geo_features_or_pos

        if isinstance(seq_input, torch.Tensor) and seq_input.dtype in [torch.float32, torch.float16, torch.bfloat16] and seq_input.dim() >= 3:
            H_seq = seq_input
        elif not self._esm_mock and hasattr(self.esm, "num_layers") and seq_input is not None:
            res = self.esm(seq_input, repr_layers=[self.esm.num_layers])
            H_seq = res["representations"][self.esm.num_layers]
        elif self.esm is not None and seq_input is not None:
            H_seq = self.esm(seq_input)
        else:
            H_seq = torch.zeros(1, 100, self.esm_embed_dim, device=next(self.parameters()).device)

        H_geo, _ = self.est_gnn(geo_input, geo_padding_mask=geo_padding_mask)
        fused_h, attn_w = self.fusion(H_seq, H_geo, key_padding_mask=seq_padding_mask, return_attn=return_attn)
        gamma, v, alpha, beta = self.head(fused_h)
        if return_attn:
            return gamma, v, alpha, beta, attn_w
        return gamma, v, alpha, beta


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

        if getattr(self, "_trainer", None) is not None:
            self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
