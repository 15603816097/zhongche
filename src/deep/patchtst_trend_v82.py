"""V8.2 PatchTST trend-aware utilities.

Adds trend-aware loss and delta feature generation without changing V8 API.
"""
from __future__ import annotations

import torch


TREND_LOSS_WEIGHT = 0.3


def trend_difference(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:, :] - x[..., :-1, :]


def trend_aware_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    trend_weight: float = TREND_LOSS_WEIGHT,
) -> torch.Tensor:
    """RMSE objective aligned with competition trend consistency."""
    mse = torch.mean((prediction - target) ** 2)
    pred_delta = prediction[:, 1:, :] - prediction[:, :-1, :]
    true_delta = target[:, 1:, :] - target[:, :-1, :]
    trend = torch.mean((pred_delta - true_delta) ** 2)
    return mse + float(trend_weight) * trend


def add_delta_features(history):
    """Append first-order differences along time dimension.

    Input shape: [B,T,C] or numpy-compatible array handled by caller.
    """
    import numpy as np

    x = np.asarray(history)
    delta = np.zeros_like(x)
    delta[1:] = x[1:] - x[:-1]
    return np.concatenate([x, delta], axis=-1)
