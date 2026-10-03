"""
Unit tests for DynamicESMDataset loader and batch collation.
"""

import os
import tempfile
import torch
from torch_geometric.data import Data
from dynamic_esm.data.dataset import DynamicESMDataset, custom_collate_fn


def test_dynamic_esm_dataset_and_collate():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy sample .pt files
        d1 = Data(
            x=torch.randn(10, 32),
            pos=torch.randn(5, 10, 3),
            y=torch.tensor([5.5]),
        )
        d2 = Data(
            x=torch.randn(15, 32),
            pos=torch.randn(5, 15, 3),
            y=torch.tensor([7.2]),
        )
        torch.save(d1, os.path.join(tmp_dir, "train_001.pt"))
        torch.save(d2, os.path.join(tmp_dir, "train_002.pt"))

        dataset = DynamicESMDataset(data_dir=tmp_dir, split="train")
        assert len(dataset) == 2

        item0 = dataset[0]
        assert item0.x.shape == (10, 32)
        assert item0.pos.shape == (5, 10, 3)

        batch = custom_collate_fn([dataset[0], dataset[1]])
        assert batch is not None
        assert batch.num_graphs == 2
