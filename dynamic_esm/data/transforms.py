"""
Data transformations and coordinate normalization for Dynamic-ESM graphs.
"""

from typing import Union
import torch
from torch_geometric.data import Data


def center_trajectory(data_or_pos: Union[Data, torch.Tensor]) -> Union[Data, torch.Tensor]:
    """Zero-centers MD trajectory frames around pocket center-of-mass (frame 0)."""
    if isinstance(data_or_pos, Data):
        if hasattr(data_or_pos, "pos") and data_or_pos.pos is not None:
            center = data_or_pos.pos[0].mean(dim=0, keepdim=True)
            data_or_pos.pos = data_or_pos.pos - center
        return data_or_pos
    elif isinstance(data_or_pos, torch.Tensor):
        if data_or_pos.dim() == 3:
            center = data_or_pos[0].mean(dim=0, keepdim=True)
            return data_or_pos - center
        elif data_or_pos.dim() == 2:
            center = data_or_pos.mean(dim=0, keepdim=True)
            return data_or_pos - center
        return data_or_pos
    return data_or_pos


class CenterTrajectory:
    """Zero-centers all MD trajectory frames around pocket center-of-mass."""

    def __call__(self, data: Union[Data, torch.Tensor]) -> Union[Data, torch.Tensor]:
        return center_trajectory(data)


class AddGaussianNoise:
    """Adds random coordinate perturbation to simulate thermal jitter for pretraining."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            noise = torch.randn_like(data.pos) * self.std
            data.noisy_pos = data.pos + noise
        return data
