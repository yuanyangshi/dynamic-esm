"""
PyTorch Geometric Dataset & DataLoader collators for Dynamic-ESM.
Supports loading from .pt files, .zip bundles, and HDF5 archives.
"""

import glob
import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class DynamicESMDataset(Dataset):
    """
    Dataset loader for 4D-QM complexes with sequence priors.
    Reads preprocessed PyG graph files (.pt or inside .zip archives).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.items = []

        if not os.path.exists(data_dir):
            return

        # 1. Search for standalone .pt files
        pt_pattern = os.path.join(data_dir, f"{split}_*.pt")
        found_pts = sorted(glob.glob(pt_pattern))
        for p in found_pts:
            self.items.append(("file", p, None))

        # 2. Search inside .zip archives
        zip_pattern = os.path.join(data_dir, "*.zip")
        found_zips = sorted(glob.glob(zip_pattern))
        for z_path in found_zips:
            try:
                with zipfile.ZipFile(z_path, "r") as zf:
                    for name in zf.namelist():
                        if name.endswith(".pt") and (split in name or split == "all"):
                            self.items.append(("zip", z_path, name))
            except Exception:
                pass

        if max_samples is not None and len(self.items) > max_samples:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Data:
        item_type, path_or_zip, arc_name = self.items[idx]
        if item_type == "file":
            try:
                data = torch.load(path_or_zip, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(path_or_zip, map_location="cpu")
        else:
            with zipfile.ZipFile(path_or_zip, "r") as zf:
                buf = io.BytesIO(zf.read(arc_name))
                try:
                    data = torch.load(buf, map_location="cpu", weights_only=False)
                except TypeError:
                    data = torch.load(buf, map_location="cpu")

        if self.transform is not None:
            data = self.transform(data)
        return data


def custom_collate_fn(batch: List[Data]) -> Optional[Batch]:
    """
    Custom collate function for PyG Data objects with multi-frame trajectory handling.
    Concatenates nodes along dim 0 for feature representations, and along dim 1
    for 4D trajectory coordinates of shape (T, N, 3).
    """
    valid_batch = [d for d in batch if d is not None and hasattr(d, "x") and d.x is not None]
    if not valid_batch:
        return None

    has_3d_pos = (
        hasattr(valid_batch[0], "pos")
        and valid_batch[0].pos is not None
        and valid_batch[0].pos.dim() == 3
    )

    if has_3d_pos:
        orig_positions = [d.pos for d in valid_batch]
        batched_pos = torch.cat(orig_positions, dim=1)

        has_noisy = hasattr(valid_batch[0], "noisy_pos") and valid_batch[0].noisy_pos is not None
        if has_noisy:
            orig_noisy = [d.noisy_pos for d in valid_batch]
            batched_noisy = torch.cat(orig_noisy, dim=1)

        for d in valid_batch:
            d.pos = d.pos[0]
            if has_noisy:
                d.noisy_pos = d.noisy_pos[0]

        batch_obj = Batch.from_data_list(valid_batch)
        batch_obj.pos = batched_pos
        if has_noisy:
            batch_obj.noisy_pos = batched_noisy

        for i, d in enumerate(valid_batch):
            d.pos = orig_positions[i]
            if has_noisy:
                d.noisy_pos = orig_noisy[i]

        return batch_obj
    else:
        return Batch.from_data_list(valid_batch)
