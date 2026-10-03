"""
Unit tests for data transformations and coordinate processing.
"""

import torch
from torch_geometric.data import Data

from dynamic_esm.data.transforms import CenterTrajectory, AddGaussianNoise


def test_center_trajectory():
    pos = torch.tensor([
        [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]],
        [[12.0, 12.0, 12.0], [22.0, 22.0, 22.0]]
    ])
    data = Data(pos=pos)
    transform = CenterTrajectory()
    out = transform(data)

    # Frame 0 mean should be centered to ~0
    f0_mean = out.pos[0].mean(dim=0)
    assert torch.allclose(f0_mean, torch.zeros(3), atol=1e-5)


def test_add_gaussian_noise():
    pos = torch.zeros(2, 5, 3)
    data = Data(pos=pos)
    transform = AddGaussianNoise(std=0.05)
    out = transform(data)

    assert hasattr(out, "noisy_pos")
    assert out.noisy_pos.shape == pos.shape
    assert not torch.allclose(out.noisy_pos, pos)
