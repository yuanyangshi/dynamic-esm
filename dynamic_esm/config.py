"""
Global Configuration & Environment Settings for Dynamic-ESM.
Supports environment variables, YAML config files, and CLI overrides.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # -------------------------------------------------------------
    # Data & Paths Configuration
    # -------------------------------------------------------------
    DATA_DIR: str = field(
        default_factory=lambda: os.environ.get(
            "DYNAMIC_ESM_DATA_DIR", "./data"
        )
    )
    PRETRAIN_DIR: str = field(
        default_factory=lambda: os.environ.get(
            "DYNAMIC_ESM_PRETRAIN_DIR", "./checkpoints"
        )
    )
    WORKING_DIR: str = field(
        default_factory=lambda: os.environ.get(
            "DYNAMIC_ESM_WORKING_DIR", "./output"
        )
    )
    CHECKPOINT_DIR: str = field(
        default_factory=lambda: os.environ.get(
            "DYNAMIC_ESM_CKPT_DIR", "./output/checkpoints"
        )
    )
    FIGURES_DIR: str = field(
        default_factory=lambda: os.environ.get(
            "DYNAMIC_ESM_FIGURES_DIR", "./output/figures"
        )
    )
    TABLES_DIR: str = field(
        default_factory=lambda: os.environ.get(
            "DYNAMIC_ESM_TABLES_DIR", "./output/tables"
        )
    )

    # -------------------------------------------------------------
    # Model Architecture Parameters
    # -------------------------------------------------------------
    ESM_MODEL_NAME: str = "esm2_t12_35M_UR50D"
    ESM_EMBED_DIM: int = 480
    GNN_HIDDEN_DIM: int = 128
    CROSS_ATTN_DIM: int = 256
    NUM_CROSS_HEADS: int = 4
    NUM_CROSS_LAYERS: int = 2
    DROPOUT: float = 0.1
    LORA_RANK: int = 16
    LORA_ALPHA: float = 16.0
    LORA_DROPOUT: float = 0.05
    NUM_FRAMES: int = 5
    POCKET_RADIUS: float = 12.0
    MAX_SEQ_LEN: int = 1024

    # -------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------
    BATCH_SIZE: int = 16
    VAL_BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    BACKBONE_LR: float = 5e-5
    ESM_LORA_LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 30
    PATIENCE: int = 7
    GRAD_ACCUM_STEPS: int = 2
    USE_AMP: bool = True
    SEED: int = 42

    # -------------------------------------------------------------
    # Evidential Deep Learning (EDL) Loss Weights
    # -------------------------------------------------------------
    LAMBDA_REG: float = 0.2
    LAMBDA_MSE: float = 1.0
    LAMBDA_RANK: float = 0.5
    RANKING_MARGIN: float = 0.5

    # -------------------------------------------------------------
    # Remote / Cloud Sync (e.g., Kaggle, Hugging Face)
    # -------------------------------------------------------------
    PUSH_TO_REMOTE: bool = False
    DATASET_REF: Optional[str] = field(
        default_factory=lambda: os.environ.get("DATASET_REF", None)
    )
    MAX_RUNTIME_SEC: float = 11.5 * 3600

    def make_dirs(self):
        """Create output directories if they do not exist."""
        for d in [self.WORKING_DIR, self.CHECKPOINT_DIR, self.FIGURES_DIR, self.TABLES_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration settings from a YAML file.
        
        Args:
            yaml_path: Absolute or relative path to the YAML configuration file.
            
        Returns:
            Config: An instance populated with settings from the YAML file.
        """
        import yaml

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        cfg = cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Flatten nested sections (data, model, training, loss)
        for section, values in data.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    attr_name = k.upper()
                    if hasattr(cfg, attr_name):
                        setattr(cfg, attr_name, v)
            else:
                attr_name = section.upper()
                if hasattr(cfg, attr_name):
                    setattr(cfg, attr_name, values)
        return cfg

    def to_yaml(self, yaml_path: str):
        """Export current configuration to a YAML file.
        
        Args:
            yaml_path: Path where the YAML file will be saved.
        """
        import yaml

        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        data = {
            "data": {
                "data_dir": self.DATA_DIR,
                "pretrain_dir": self.PRETRAIN_DIR,
                "working_dir": self.WORKING_DIR,
                "num_frames": self.NUM_FRAMES,
                "pocket_radius": self.POCKET_RADIUS,
                "max_seq_len": self.MAX_SEQ_LEN,
            },
            "model": {
                "esm_model_name": self.ESM_MODEL_NAME,
                "esm_embed_dim": self.ESM_EMBED_DIM,
                "gnn_hidden_dim": self.GNN_HIDDEN_DIM,
                "cross_attn_dim": self.CROSS_ATTN_DIM,
                "num_cross_heads": self.NUM_CROSS_HEADS,
                "num_cross_layers": self.NUM_CROSS_LAYERS,
                "lora_rank": self.LORA_RANK,
                "lora_alpha": self.LORA_ALPHA,
                "lora_dropout": self.LORA_DROPOUT,
            },
            "training": {
                "batch_size": self.BATCH_SIZE,
                "val_batch_size": self.VAL_BATCH_SIZE,
                "learning_rate": self.LEARNING_RATE,
                "backbone_lr": self.BACKBONE_LR,
                "esm_lora_lr": self.ESM_LORA_LR,
                "weight_decay": self.WEIGHT_DECAY,
                "num_epochs": self.NUM_EPOCHS,
                "grad_accum_steps": self.GRAD_ACCUM_STEPS,
                "use_amp": self.USE_AMP,
                "seed": self.SEED,
            },
            "loss": {
                "lambda_reg": self.LAMBDA_REG,
                "lambda_mse": self.LAMBDA_MSE,
                "lambda_rank": self.LAMBDA_RANK,
                "ranking_margin": self.RANKING_MARGIN,
            },
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
