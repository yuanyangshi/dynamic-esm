"""
Unit tests for Evidential Deep Learning (EDL) loss functions and regularization.
"""

import torch
from dynamic_esm.training.losses import (
    evidential_nll_loss,
    pairwise_ranking_loss,
    dynamic_esm_criterion,
)


def test_evidential_nll_loss():
    y_true = torch.tensor([[5.0], [6.0], [7.0]])
    gamma = torch.tensor([[5.1], [5.9], [7.2]])
    v = torch.tensor([[2.0], [2.5], [1.8]])
    alpha = torch.tensor([[3.0], [3.5], [4.0]])
    beta = torch.tensor([[1.0], [1.2], [0.9]])

    total_loss, nll_loss, reg_loss = evidential_nll_loss(y_true, gamma, v, alpha, beta)
    assert total_loss.dim() == 0
    assert nll_loss.dim() == 0
    assert reg_loss.dim() == 0
    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)


def test_pairwise_ranking_loss():
    y_true = torch.tensor([[4.0], [6.0], [8.0]])
    gamma = torch.tensor([[4.2], [5.8], [8.1]])

    loss = pairwise_ranking_loss(y_true, gamma, margin=0.5)
    assert loss.dim() == 0
    assert loss >= 0.0
    assert not torch.isnan(loss)


def test_dynamic_esm_criterion():
    y_true = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
    gamma = torch.tensor([[5.1], [6.2], [6.9], [7.8]])
    v = torch.tensor([[2.0], [2.0], [2.0], [2.0]])
    alpha = torch.tensor([[3.0], [3.0], [3.0], [3.0]])
    beta = torch.tensor([[1.0], [1.0], [1.0], [1.0]])

    total_loss, loss_dict = dynamic_esm_criterion(
        y_true, gamma, v, alpha, beta,
        lambda_reg=0.2, lambda_mse=1.0, lambda_rank=0.5, margin=0.5
    )
    assert total_loss.dim() == 0
    assert not torch.isnan(total_loss)
    assert "loss" in loss_dict
    assert "nll" in loss_dict
    assert "reg" in loss_dict
    assert "rank" in loss_dict
