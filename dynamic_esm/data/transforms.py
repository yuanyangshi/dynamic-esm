"""
Data transformations and coordinate normalization for Dynamic-ESM graphs.
"""

import torch
from torch_geometric.data import Data


class CenterTrajectory:
    """Zero-centers all MD trajectory frames around pocket center-of-mass."""

    def __call__(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            # pos shape: (T, N, 3)
            center = data.pos[0].mean(dim=0, keepdim=True)  # (1, 3)
            data.pos = data.pos - center
        return data


class AddGaussianNoise:
    """Adds random coordinate perturbation to simulate thermal jitter for pretraining."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            noise = torch.randn_like(data.pos) * self.std
            data.noisy_pos = data.pos + noise
        return data
