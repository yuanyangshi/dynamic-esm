"""
Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) for ESM-2.
"""

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
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Freeze original linear weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.base_layer(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return orig_out + lora_out * self.scaling


def inject_lora_into_esm(
    esm_model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Freeze all ESM-2 parameters and inject LoRALinear adapters into query and key projections.
    """
    # Freeze entire ESM-2 model
    for param in esm_model.parameters():
        param.requires_grad = False

    # Inject LoRA into multihead attention layers
    for layer in esm_model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.q_proj = LoRALinear(
                layer.self_attn.q_proj, rank=rank, alpha=alpha, dropout=dropout
            )
            layer.self_attn.k_proj = LoRALinear(
                layer.self_attn.k_proj, rank=rank, alpha=alpha, dropout=dropout
            )
    return esm_model
