from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TCNConfig:
    input_channels: int = 12  # 6 normalized signals + 6 repeated modality-mask channels
    output_channels: int = 6
    horizon: int = 96
    hidden_channels: int = 96
    num_blocks: int = 5
    kernel_size: int = 3
    dropout: float = 0.10


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.GroupNorm(1, channels),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.GroupNorm(1, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MaskedTCNForecaster(nn.Module):
    """TCN forecaster for the heterogeneous 512->96 pretraining corpus.

    Input values are normalized with history-only statistics by the corpus builder.
    Missing modalities are zero-filled, while the modality mask is repeated over time
    and concatenated to the six signal channels.

    The model predicts a residual around the persistence forecast. This stabilizes
    pretraining on heterogeneous machines and makes the validation comparison against
    persistence directly meaningful.
    """

    def __init__(self, config: TCNConfig | None = None):
        super().__init__()
        self.config = config or TCNConfig()
        c = self.config

        self.input_proj = nn.Conv1d(c.input_channels, c.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=c.hidden_channels,
                    kernel_size=c.kernel_size,
                    dilation=2**i,
                    dropout=c.dropout,
                )
                for i in range(c.num_blocks)
            ]
        )
        self.final_norm = nn.GroupNorm(1, c.hidden_channels)

        # Last-state + global mean retains both recent-state and long-context information.
        self.head = nn.Sequential(
            nn.Linear(c.hidden_channels * 2, c.hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_channels * 2, c.horizon * c.output_channels),
        )

        # Start close to persistence rather than a random long-horizon forecast.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 512, 6] normalized history.
            modality_mask: [B, 6], 1 for a real source-domain modality, 0 if missing.
        Returns:
            [B, 96, 6] normalized future forecast.
        """
        if x.ndim != 3 or x.shape[-1] != self.config.output_channels:
            raise ValueError(f"unexpected x shape: {tuple(x.shape)}")
        if modality_mask.ndim != 2 or modality_mask.shape[-1] != self.config.output_channels:
            raise ValueError(f"unexpected modality_mask shape: {tuple(modality_mask.shape)}")

        repeated_mask = modality_mask[:, None, :].expand(-1, x.shape[1], -1)
        inp = torch.cat([x, repeated_mask], dim=-1).transpose(1, 2)

        h = self.input_proj(inp)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)

        context = torch.cat([h[:, :, -1], h.mean(dim=-1)], dim=1)
        residual = self.head(context).view(
            x.shape[0], self.config.horizon, self.config.output_channels
        )

        persistence = x[:, -1:, :].expand(-1, self.config.horizon, -1)
        return persistence + residual
