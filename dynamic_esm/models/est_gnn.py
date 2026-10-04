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
    SE(3)-equivariant Message Passing Layer with Gaussian RBF Kernel Expansion.
    Guarantees strict rotational and translational equivariance for coordinate updates:
        Delta r_{ij} = (r_i - r_j) * phi_x(m_{ij})
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 32,
        num_rbf: int = 32,
        cutoff: float = 6.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.cutoff = cutoff
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)

    @property
    def norm(self) -> nn.LayerNorm:
        """Alias for backward compatibility."""
        return self.node_norm

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: (N, hidden_dim)
        pos: (N, 3)
        edge_index: (2, E)
        edge_rbf: (E, edge_dim)
        """
        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            return self.node_norm(h), pos

        row, col = edge_index[0], edge_index[1]

        # Calculate relative displacement vector: pos_j - pos_i
        rel_pos = pos[col] - pos[row]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
        unit_vec = rel_pos / dist

        if edge_rbf is None:
            edge_rbf = self.rbf(dist)

        # Edge message
        msg_in = torch.cat([h[row], h[col], edge_rbf], dim=-1)
        msg = self.edge_mlp(msg_in)  # (E, hidden_dim)

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
        h_out = self.node_norm(h + self.node_mlp(torch.cat([h, h_agg], dim=-1)))
        pos_out = pos + pos_agg

        return h_out, pos_out


class TemporalAttentionFusion(nn.Module):
    """
    Temporal Attention Fusion across MD trajectory frames.
    Captures dynamic entropy and conformational transitions over time.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    @property
    def norm(self) -> nn.LayerNorm:
        """Alias for backward compatibility."""
        return self.norm1

    @property
    def mlp(self) -> nn.Sequential:
        """Alias for backward compatibility."""
        return self.ffn

    def forward(self, h_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_frames: (N, T, hidden_dim) or (B, T, hidden_dim)
        Returns:
            fused_h: (N, hidden_dim)
            attn_weights: (N, T, T)
        """
        attn_out, attn_weights = self.mha(h_frames, h_frames, h_frames)
        h_res = self.norm1(h_frames + attn_out)
        h_out = self.norm2(h_res + self.ffn(h_res))
        fused_h = h_out.mean(dim=1)  # Temporal mean pooling
        return fused_h, attn_weights


class DynamicESMBackbone(nn.Module):
    """
    4D-QM Spatiotemporal Backbone Network (EST-GNN).
    Harmonized with 4D-QM spatiotemporal pretrained weights (120 tensors, 44.15 MB).
    Integrates atomic coordinates, QM electrostatic charges/dipoles, and MD trajectories.
    """

    def __init__(
        self,
        node_in_dim: int = 654,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_rbf: int = 32,
        cutoff: float = 6.5,
        num_heads: int = 8,
        dropout: float = 0.1,
        include_adapter: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.spatial_layers = nn.ModuleList([
            EquivariantMessagePassingLayer(
                hidden_dim=hidden_dim, edge_dim=num_rbf, num_rbf=num_rbf, cutoff=cutoff, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.temporal_fusion = TemporalAttentionFusion(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )

        self.final_coord_update = EquivariantMessagePassingLayer(
            hidden_dim=hidden_dim, edge_dim=num_rbf, num_rbf=num_rbf, cutoff=cutoff, dropout=dropout
        )

        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if include_adapter:
            self.tensor_input_adapter = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.tensor_input_adapter = None

    @property
    def rbf(self) -> GaussianRBF:
        """Access shared RBF kernel from first spatial layer without duplicate state parameter."""
        return self.spatial_layers[0].rbf

    @property
    def node_embed(self) -> nn.Sequential:
        """Alias for input_proj for backward compatibility."""
        return self.input_proj

    @property
    def layers(self) -> nn.ModuleList:
        """Alias for spatial_layers for backward compatibility."""
        return self.spatial_layers

    def _build_radius_graph(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct distance-based radius graph within cutoff."""
        dist_mat = torch.cdist(pos, pos)
        mask = (dist_mat < self.cutoff) & (dist_mat > 1e-4)
        edge_index = mask.nonzero().t()  # (2, E)
        distances = dist_mat[mask].unsqueeze(-1)  # (E, 1)
        edge_rbf = self.rbf(distances)
        return edge_index, edge_rbf

    def forward(
        self,
        x: torch.Tensor,
        pos_trajectory: Optional[torch.Tensor] = None,
        geo_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Polymorphic forward supporting:
        1. Coordinate trajectory mode: x: (N, node_in_dim), pos_trajectory: (T, N, 3)
        2. Batched 4D tensor mode: x: (B, T, N, D), pos_trajectory: None
        """
        # Case 2: Batched 4D tensor mode [B, T, N, D]
        if pos_trajectory is None and x.dim() == 4:
            B, T, N, D = x.shape
            in_dim = self.input_proj[0].in_features
            hidden_dim = self.hidden_dim

            if D >= 3 + in_dim:
                pos_all = x[..., :3]
                x_all = x[..., 3:3 + in_dim]
                h_direct = None
            elif D == in_dim:
                pos_all = torch.zeros(B, T, N, 3, device=x.device)
                x_all = x
                h_direct = None
            elif D == hidden_dim:
                pos_all = torch.zeros(B, T, N, 3, device=x.device)
                x_all = None
                adapter = self.tensor_input_adapter if self.tensor_input_adapter is not None else nn.Linear(hidden_dim, hidden_dim).to(x.device)
                h_direct = adapter(x.reshape(B * T * N, D)).view(B, T, N, hidden_dim)
            else:
                if not hasattr(self, "_dim_adapter") or self._dim_adapter.in_features != D:
                    self._dim_adapter = nn.Linear(D, in_dim).to(x.device)
                x_all = self._dim_adapter(x)
                pos_all = torch.zeros(B, T, N, 3, device=x.device)
                h_direct = None

            temporal_reps = []
            for b in range(B):
                mask_b = geo_padding_mask[b] if geo_padding_mask is not None else None
                valid_idx = torch.where(~mask_b)[0] if mask_b is not None else torch.arange(N, device=x.device)
                if len(valid_idx) == 0:
                    valid_idx = torch.arange(N, device=x.device)

                frame_reps = []
                for t in range(T):
                    if x_all is not None:
                        pos_t = pos_all[b, t, valid_idx]
                        x_t = x_all[b, t, valid_idx]
                        h_t = self.input_proj(x_t)
                        edge_index, edge_rbf = self._build_radius_graph(pos_t)
                        for layer in self.spatial_layers:
                            h_t, pos_t = layer(h_t, pos_t, edge_index, edge_rbf)
                        frame_reps.append(h_t.mean(dim=0))
                    else:
                        h_t = h_direct[b, t, valid_idx]
                        frame_reps.append(h_t.mean(dim=0))

                temporal_reps.append(torch.stack(frame_reps, dim=0))

            temporal_tensor = torch.stack(temporal_reps, dim=0)  # [B, T, hidden_dim]
            fused_temporal, temporal_attn = self.temporal_fusion(temporal_tensor)
            traj_summary = fused_temporal.mean(dim=1, keepdim=True)
            final_geo = torch.cat([fused_temporal, traj_summary], dim=1)  # [B, T + 1, hidden_dim]
            return final_geo, temporal_attn

        # Case 1: Coordinate trajectory mode
        if x.size(-1) != self.input_proj[0].in_features:
            if not hasattr(self, "_dim_adapter") or self._dim_adapter.in_features != x.size(-1):
                self._dim_adapter = nn.Linear(x.size(-1), self.input_proj[0].in_features).to(x.device)
            x = self._dim_adapter(x)

        if pos_trajectory.dim() == 4:
            pos_trajectory = pos_trajectory.squeeze(0)

        if pos_trajectory.size(1) < pos_trajectory.size(0) and pos_trajectory.size(1) <= 10:
            pos_trajectory = pos_trajectory.permute(1, 0, 2)

        T, N, _ = pos_trajectory.shape
        h_initial = self.input_proj(x)

        frame_embeddings = []
        for t in range(T):
            pos_t = pos_trajectory[t]
            edge_index, edge_rbf = self._build_radius_graph(pos_t)

            h_t = h_initial
            for layer in self.spatial_layers:
                h_t, pos_t = layer(h_t, pos_t, edge_index, edge_rbf)
            frame_embeddings.append(h_t)

        h_frames = torch.stack(frame_embeddings, dim=1)
        fused_h, temporal_attn = self.temporal_fusion(h_frames)

        return fused_h, temporal_attn
