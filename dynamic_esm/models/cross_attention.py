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
    Single Bi-directional Cross-Attention Layer.
    - Stream 1: Pocket -> Sequence (Local pocket queries global evolutionary context)
    - Stream 2: Sequence -> Pocket (Global sequence queries local dynamic pocket)
    - Gated Cross-Modal Residual Fusion
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Stream 1: Pocket queries Sequence
        self.pocket_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_pocket = nn.LayerNorm(embed_dim)

        # Stream 2: Sequence queries Pocket
        self.seq_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_seq = nn.LayerNorm(embed_dim)

        # Feed-forward networks
        self.ffn_pocket = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_ffn_pocket = nn.LayerNorm(embed_dim)

        self.ffn_seq = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_ffn_seq = nn.LayerNorm(embed_dim)

        # Gated fusion gate
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_pocket: torch.Tensor,
        h_seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h_pocket: (B, N_p, embed_dim)
        h_seq: (B, N_s, embed_dim)
        """
        # Stream 1: Pocket queries Seq
        p2s_out, p2s_weights = self.pocket_attn(
            query=h_pocket,
            key=h_seq,
            value=h_seq,
            key_padding_mask=key_padding_mask,
        )
        h_pocket_mid = self.norm_pocket(h_pocket + p2s_out)
        h_pocket_out = self.norm_ffn_pocket(h_pocket_mid + self.ffn_pocket(h_pocket_mid))

        # Stream 2: Seq queries Pocket
        s2p_out, s2p_weights = self.seq_attn(
            query=h_seq,
            key=h_pocket,
            value=h_pocket,
        )
        h_seq_mid = self.norm_seq(h_seq + s2p_out)
        h_seq_out = self.norm_ffn_seq(h_seq_mid + self.ffn_seq(h_seq_mid))

        return h_pocket_out, h_seq_out, p2s_weights


class DeepBiCrossAttention(nn.Module):
    """
    Stacked Deep Bi-directional Cross-Attention with cross-scale pooling.
    """

    def __init__(
        self,
        pocket_in_dim: int = 128,
        seq_in_dim: int = 480,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_pocket = nn.Linear(pocket_in_dim, embed_dim)
        self.proj_seq = nn.Linear(seq_in_dim, embed_dim)

        self.layers = nn.ModuleList([
            BiCrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pocket_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1),
        )

        self.seq_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        h_pocket: torch.Tensor,
        h_seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_pocket: (B, N_p, pocket_in_dim)
        h_seq: (B, N_s, seq_in_dim)
        Returns:
            fused_representation: (B, embed_dim)
            last_attention_weights: (B, N_p, N_s)
        """
        h_p = self.proj_pocket(h_pocket)
        h_s = self.proj_seq(h_seq)

        last_weights = None
        for layer in self.layers:
            h_p, h_s, last_weights = layer(h_p, h_s, key_padding_mask=key_padding_mask)

        # Attention-weighted pooling
        w_p = self.pocket_pool(h_p)  # (B, N_p, 1)
        p_vec = torch.sum(h_p * w_p, dim=1)  # (B, embed_dim)

        w_s = self.seq_pool(h_s)  # (B, N_s, 1)
        s_vec = torch.sum(h_s * w_s, dim=1)  # (B, embed_dim)

        # Multimodal fusion
        fused = self.fusion(torch.cat([p_vec, s_vec], dim=-1))
        return fused, last_weights
