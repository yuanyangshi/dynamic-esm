"""
Unit tests for configuration and hardware utilities.
"""

import os
import torch
from dynamic_esm.config import Config
from dynamic_esm.utils.hardware import set_seed, setup_hardware
from dynamic_esm.utils.logging import get_logger


def test_config_initialization():
    cfg = Config()
    assert cfg.ESM_MODEL_NAME == "esm2_t12_35M_UR50D"
    assert cfg.GNN_HIDDEN_DIM == 128
    assert cfg.CROSS_ATTN_DIM == 256
    assert cfg.BATCH_SIZE == 16
    assert cfg.LAMBDA_REG == 0.2
    assert cfg.LAMBDA_MSE == 1.0


def test_config_yaml_serialization(tmp_path):
    cfg = Config()
    cfg.BATCH_SIZE = 64
    cfg.LEARNING_RATE = 5e-4
    yaml_file = str(tmp_path / "test_config.yaml")
    
    cfg.to_yaml(yaml_file)
    assert os.path.exists(yaml_file)
    
    loaded_cfg = Config.from_yaml(yaml_file)
    assert loaded_cfg.BATCH_SIZE == 64
    assert loaded_cfg.LEARNING_RATE == 5e-4
    assert loaded_cfg.ESM_MODEL_NAME == cfg.ESM_MODEL_NAME


def test_hardware_seed_and_device():
    set_seed(123)
    t1 = torch.rand(5)
    set_seed(123)
    t2 = torch.rand(5)
    assert torch.allclose(t1, t2)

    device = setup_hardware()
    assert isinstance(device, torch.device)


def test_logger():
    logger = get_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"
