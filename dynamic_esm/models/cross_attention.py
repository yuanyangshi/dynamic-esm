"""
Bi-directional Cross-Attention (Bi-SDPA) Network.
Bridges 1D macro evolutionary priors (ESM-2) and 4D-QM micro spatiotemporal dynamics.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiCrossAttentionLayer(nn.Module):
    """
    Bi-directional Scaled Dot-Product Cross-Attention Layer.
    - Stream 1: Seq -> Geo (H_geo queries H_seq for evolutionary conservation context)
    - Stream 2: Geo -> Seq (H_seq queries H_geo for pocket induced-fit transitions)
    - Multi-Head Gated / FFN Residual Connections
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.seq_to_geo_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.geo_to_seq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm1_seq = nn.LayerNorm(hidden_dim)
        self.norm1_geo = nn.LayerNorm(hidden_dim)

        self.ffn_seq = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_geo = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm2_seq = nn.LayerNorm(hidden_dim)
        self.norm2_geo = nn.LayerNorm(hidden_dim)

    # Aliases for backward compatibility
    @property
    def embed_dim(self) -> int:
        return self.hidden_dim

    @property
    def pocket_attn(self) -> nn.MultiheadAttention:
        return self.seq_to_geo_attn

    @property
    def seq_attn(self) -> nn.MultiheadAttention:
        return self.geo_to_seq_attn

    def forward(
        self,
        H_seq: torch.Tensor,
        H_geo: torch.Tensor,
        seq_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        out_geo, attn_weights = self.seq_to_geo_attn(
            query=H_geo,
            key=H_seq,
            value=H_seq,
            key_padding_mask=seq_padding_mask,
            need_weights=return_attn,
        )
        H_geo = self.norm1_geo(H_geo + out_geo)

        out_seq, _ = self.geo_to_seq_attn(
            query=H_seq,
            key=H_geo,
            value=H_geo,
            need_weights=False,
        )
        H_seq = self.norm1_seq(H_seq + out_seq)

        H_seq = self.norm2_seq(H_seq + self.ffn_seq(H_seq))
        H_geo = self.norm2_geo(H_geo + self.ffn_geo(H_geo))

        if return_attn:
            return H_seq, H_geo, attn_weights
        return H_seq, H_geo, None


class DeepBiCrossAttention(nn.Module):
    """
    Stacked Deep Bi-directional Cross-Attention with multi-scale pooling and fusion.
    """

    def __init__(
        self,
        seq_dim: int = 480,
        geo_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        pocket_in_dim: Optional[int] = None,
        seq_in_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        if pocket_in_dim is not None:
            geo_dim = pocket_in_dim
        if seq_in_dim is not None:
            seq_dim = seq_in_dim
        if embed_dim is not None:
            hidden_dim = embed_dim

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.geo_proj = nn.Linear(geo_dim, hidden_dim)

        self.layers = nn.ModuleList([
            BiCrossAttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # Backward compatibility aliases
    @property
    def proj_seq(self) -> nn.Linear:
        return self.seq_proj

    @property
    def proj_pocket(self) -> nn.Linear:
        return self.geo_proj

    @property
    def fusion(self) -> nn.Sequential:
        return self.fusion_ffn

    def _forward_impl(
        self,
        H_seq: torch.Tensor,
        H_geo: torch.Tensor,
        seq_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        H_seq_h = self.seq_proj(H_seq)
        H_geo_h = self.geo_proj(H_geo)

        last_weights = None
        for layer in self.layers:
            H_seq_h, H_geo_h, last_weights = layer(
                H_seq_h, H_geo_h, seq_padding_mask=seq_padding_mask, return_attn=return_attn
            )

        if seq_padding_mask is not None:
            valid_mask = (~seq_padding_mask).unsqueeze(-1).float()
            z_seq = (H_seq_h * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        else:
            z_seq = H_seq_h.mean(dim=1)

        z_geo = H_geo_h.mean(dim=1)
        z_fused = self.fusion_ffn(torch.cat([z_seq, z_geo], dim=-1))
        return z_fused, last_weights

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = True,
        use_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Polymorphic forward supporting both:
        - (h_pocket, h_seq) [from coordinate trajectory mode]
        - (h_seq, h_geo) [from multimodal tensor fusion mode]
        """
        if h1.size(-1) == self.seq_proj.in_features:
            H_seq, H_geo = h1, h2
        elif h2.size(-1) == self.seq_proj.in_features:
            H_geo, H_seq = h1, h2
        else:
            if h1.size(-1) > h2.size(-1):
                H_seq, H_geo = h1, h2
            else:
                H_geo, H_seq = h1, h2

        z_fused, last_weights = self._forward_impl(
            H_seq, H_geo, seq_padding_mask=key_padding_mask, return_attn=return_attn
        )
        return z_fused, last_weights
