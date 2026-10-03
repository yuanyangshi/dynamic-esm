"""
Equivariant Spatiotemporal Graph Neural Network (EST-GNN) for 4D-QM Pocket Dynamics.
Guarantees SE(3) geometric equivariance and captures nanosecond molecular fluctuations.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):
    """Gaussian Radial Basis Functions for interatomic distance expansion."""

    def __init__(self, num_rbf: int = 32, cutoff: float = 12.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.gamma = (num_rbf / cutoff) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (E, 1)
        return torch.exp(-self.gamma * (dist - self.centers.view(1, -1)) ** 2)


class EquivariantMessagePassingLayer(nn.Module):
    """
    SE(3)-equivariant Message Passing Layer.
    Propagates scalar invariant features and equivariant 3D coordinate vector updates.
    """

    def __init__(self, hidden_dim: int = 128, edge_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: (N, hidden_dim)
        pos: (N, 3)
        edge_index: (2, E)
        edge_rbf: (E, edge_dim)
        """
        row, col = edge_index[0], edge_index[1]

        # Calculate relative displacement vector: pos_j - pos_i
        rel_pos = pos[col] - pos[row]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
        unit_vec = rel_pos / dist

        # Edge message
        msg_in = torch.cat([h[row], h[col], edge_rbf], dim=-1)
        msg = self.msg_mlp(msg_in)  # (E, hidden_dim)

        # Coordinate update (SE(3) equivariant)
        coord_weights = self.coord_mlp(msg)  # (E, 1)
        coord_msg = coord_weights * unit_vec  # (E, 3)

        # Aggregate messages
        num_nodes = h.size(0)
        h_agg = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        h_agg.index_add_(0, row, msg)

        pos_agg = torch.zeros(num_nodes, 3, device=pos.device)
        pos_agg.index_add_(0, row, coord_msg)

        # Update node representations
        h_out = self.norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))
        pos_out = pos + pos_agg

        return h_out, pos_out


class TemporalAttentionFusion(nn.Module):
    """
    Temporal Attention Fusion across MD trajectory frames.
    Captures dynamic entropy and conformational transitions over time.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_frames: (N, T, hidden_dim)
        Returns:
            fused_h: (N, hidden_dim)
            attn_weights: (N, T, T)
        """
        attn_out, attn_weights = self.mha(h_frames, h_frames, h_frames)
        h_res = self.norm(h_frames + attn_out)
        h_out = self.norm2(h_res + self.mlp(h_res))
        fused_h = h_out.mean(dim=1)  # Temporal mean pooling
        return fused_h, attn_weights


class DynamicESMBackbone(nn.Module):
    """
    4D-QM Spatiotemporal Backbone Network.
    Integrates atomic coordinates, QM electrostatic charges/dipoles, and MD trajectories.
    """

    def __init__(
        self,
        node_in_dim: int = 494,  # e.g., 10 (atom one-hot) + 1 (charge) + 3 (dipole) + 480 (ESM-2)
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rbf: int = 32,
        cutoff: float = 12.0,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)

        self.layers = nn.ModuleList([
            EquivariantMessagePassingLayer(hidden_dim=hidden_dim, edge_dim=num_rbf)
            for _ in range(num_layers)
        ])

        self.temporal_fusion = TemporalAttentionFusion(
            hidden_dim=hidden_dim, num_heads=num_heads
        )

    def _build_radius_graph(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct distance-based radius graph within cutoff."""
        # Pairwise distance matrix
        dist_mat = torch.cdist(pos, pos)
        mask = (dist_mat < self.cutoff) & (dist_mat > 1e-4)
        edge_index = mask.nonzero().t()  # (2, E)
        distances = dist_mat[mask].unsqueeze(-1)  # (E, 1)
        edge_rbf = self.rbf(distances)
        return edge_index, edge_rbf

    def forward(
        self,
        x: torch.Tensor,
        pos_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, node_in_dim)
        pos_trajectory: (N, T, 3) or (T, N, 3)
        """
        if pos_trajectory.dim() == 4:
            # (Batch, T, N, 3) -> squeeze batch if unbatched
            pos_trajectory = pos_trajectory.squeeze(0)

        # Standardize shape to (T, N, 3)
        if pos_trajectory.size(1) < pos_trajectory.size(0) and pos_trajectory.size(1) <= 10:
            # Shape is (N, T, 3) -> permute to (T, N, 3)
            pos_trajectory = pos_trajectory.permute(1, 0, 2)

        T, N, _ = pos_trajectory.shape
        h_initial = self.node_embed(x)  # (N, hidden_dim)

        frame_embeddings = []
        for t in range(T):
            pos_t = pos_trajectory[t]
            edge_index, edge_rbf = self._build_radius_graph(pos_t)

            h_t = h_initial
            for layer in self.layers:
                h_t, pos_t = layer(h_t, pos_t, edge_index, edge_rbf)
            frame_embeddings.append(h_t)

        # Stack over time: (N, T, hidden_dim)
        h_frames = torch.stack(frame_embeddings, dim=1)
        fused_h, temporal_attn = self.temporal_fusion(h_frames)

        return fused_h, temporal_attn
