"""
Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) for ESM-2.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for linear transformations.
    Injects trainable rank-decomposition matrices into frozen backbone weights.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.linear = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Freeze original linear weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    @property
    def base_layer(self) -> nn.Linear:
        """Alias for linear attribute for backward compatibility."""
        return self.linear

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.t().to(x.dtype)) @ self.lora_B.t().to(x.dtype)
        return orig_out + lora_out * self.scaling


def inject_lora_into_esm(
    esm_model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
) -> nn.Module:
    """
    Freeze all ESM-2 parameters and inject LoRALinear adapters into attention projection layers.
    Targets q_proj and v_proj by default to match production weights.
    """
    # Freeze entire ESM-2 model
    for param in esm_model.parameters():
        param.requires_grad = False

    # Inject LoRA into multihead attention layers
    for layer in esm_model.layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            for mod_name in target_modules:
                if hasattr(attn, mod_name):
                    orig_mod = getattr(attn, mod_name)
                    if isinstance(orig_mod, nn.Linear):
                        setattr(
                            attn,
                            mod_name,
                            LoRALinear(orig_mod, rank=rank, alpha=alpha, dropout=dropout),
                        )
    return esm_model
