"""
Evidential Deep Learning (EDL) Head & Multi-Component Loss Functions.
Models epistemic and aleatoric uncertainty via Normal-Inverse-Gamma (NIG) distributions.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLHead(nn.Module):
    """
    Evidential Regression Head parameterized by a Normal-Inverse-Gamma (NIG) distribution.
    Outputs:
        gamma: mean prediction (pKd / pKi / deltaG)
        v: degrees of freedom / virtual observation count (> 0)
        alpha: shape parameter (> 1)
        beta: scale parameter (> 0)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        mid_dim = hidden_dim if hidden_dim is not None else in_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, 4),
        )
        # Physics-regularized initialization preventing evidence collapse
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)
        nn.init.xavier_uniform_(self.fc[2].weight)
        # softplus(0.54) ~ 1.0, initializing v ~ 1.1, alpha ~ 2.2, beta ~ 1.05
        self.fc[2].bias.data = torch.tensor([0.0, 0.54, 0.54, 0.54])

    @property
    def mlp(self) -> nn.Sequential:
        """Alias for fc sequence for backward compatibility."""
        return self.fc

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.fc(x)
        gamma = out[:, 0:1]
        v = F.softplus(out[:, 1:2]) + 0.1
        alpha = F.softplus(out[:, 2:3]) + 1.2
        beta = F.softplus(out[:, 3:4]) + 0.05
        return gamma, v, alpha, beta

    @staticmethod
    def compute_uncertainty(
        v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute aleatoric, epistemic, and total predictive uncertainty.
        Aleatoric: beta / (alpha - 1)
        Epistemic: beta / (v * (alpha - 1))
        Total: Aleatoric + Epistemic
        """
        denom = torch.clamp(alpha - 1.0, min=1e-6)
        aleatoric = beta / denom
        epistemic = beta / (torch.clamp(v, min=1e-6) * denom)
        total = aleatoric + epistemic
        return aleatoric, epistemic, total


def evidential_nll_loss(
    y_true: torch.Tensor,
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lambda_reg: float = 0.2,
    lambda_mse: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes NIG Negative Log-Likelihood (Student-t marginal) with error-scaled evidence regularizer.
    """
    # 2 * beta * (1 + v)
    omega = 2.0 * beta * (1.0 + v)

    # NLL term
    nll = 0.5 * torch.log(torch.pi / v) \
        - alpha * torch.log(omega) \
        + (alpha + 0.5) * torch.log((y_true - gamma) ** 2 * v + omega) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    nll_loss = torch.mean(nll)

    # Evidence regularizer penalizing overconfident incorrect predictions
    error = torch.abs(y_true - gamma)
    reg = error * (2.0 * v + alpha)
    reg_loss = torch.mean(reg)

    # Auxiliary MSE loss for smooth gradient convergence
    mse_loss = F.mse_loss(gamma, y_true)

    total_loss = nll_loss + lambda_reg * reg_loss + lambda_mse * mse_loss
    return total_loss, nll_loss, reg_loss


def pairwise_ranking_loss(
    y_true: torch.Tensor,
    gamma: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss to optimize Spearman correlation and virtual screening power.
    """
    if y_true.size(0) < 2:
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)

    y_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)  # (N, N)
    pred_diff = gamma.unsqueeze(1) - gamma.unsqueeze(0)  # (N, N)

    # Target sign: 1 if y_i > y_j, -1 if y_i < y_j, 0 if equal
    target = torch.sign(y_diff)
    mask = target != 0

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)

    loss = F.relu(-target[mask] * pred_diff[mask] + margin)
    return torch.mean(loss)


def dynamic_esm_criterion(
    y_true: torch.Tensor,
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    lambda_reg: float = 0.2,
    lambda_mse: float = 1.0,
    lambda_rank: float = 0.5,
    margin: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Combined Multi-Objective Loss Function."""
    edl_loss, nll, reg = evidential_nll_loss(
        y_true, gamma, v, alpha, beta, lambda_reg=lambda_reg, lambda_mse=lambda_mse
    )
    rank_loss = pairwise_ranking_loss(y_true, gamma, margin=margin)
    total_loss = edl_loss + lambda_rank * rank_loss

    loss_dict = {
        "loss": total_loss.item(),
        "nll": nll.item(),
        "reg": reg.item(),
        "rank": rank_loss.item(),
    }
    return total_loss, loss_dict
