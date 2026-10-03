"""
Hardware acceleration, CUDA memory management, and seed initialization.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_hardware() -> torch.device:
    """Configure CUDA memory allocator and return optimal execution device."""
    # Mitigate CUDA memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[HARDWARE] GPU Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[HARDWARE] CUDA unavailable, falling back to CPU execution.")
        
    return device
