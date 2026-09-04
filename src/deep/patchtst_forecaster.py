from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PatchTSTConfig:
    input_length: int = 512
    output_channels: int = 6
    horizon: int = 96
    patch_length: int = 32
    stride: int = 16
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.10

    @property
    def num_patches(self) -> int:
        if self.input_length < self.patch_length:
            raise ValueError("input_length must be >= patch_length")
        return 1 + (self.input_length - self.patch_length) // self.stride


class MaskedPatchTSTForecaster(nn.Module):
    """Channel-independent patch transformer for the masked 512->96 corpus.

    Each sensor channel is patch-encoded independently with shared temporal weights,
    while a learned channel embedding and modality-presence embedding keep sensor
    identity/missingness explicit. The output is a residual around persistence.

    This intentionally follows the PatchTST idea (patching + channel-independent
    temporal encoding) without depending on any external PatchTST package.
    """

    def __init__(self, config: PatchTSTConfig | None = None):
        super().__init__()
        self.config = config or PatchTSTConfig()
        c = self.config

        self.patch_embed = nn.Linear(c.patch_length, c.d_model)
        self.position = nn.Parameter(torch.zeros(1, c.num_patches, c.d_model))
        self.channel_embedding = nn.Embedding(c.output_channels, c.d_model)
        self.mask_embedding = nn.Embedding(2, c.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=c.num_layers,
            norm=nn.LayerNorm(c.d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(c.d_model * 2, c.d_model * 2),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model * 2, c.horizon),
        )

        nn.init.normal_(self.position, mean=0.0, std=0.02)
        # Start exactly at persistence, matching the TCN V1 evaluation protocol.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        c = self.config
        if x.ndim != 3 or x.shape[1] != c.input_length or x.shape[2] != c.output_channels:
            raise ValueError(f"unexpected x shape: {tuple(x.shape)}")
        if modality_mask.ndim != 2 or modality_mask.shape != (x.shape[0], c.output_channels):
            raise ValueError(f"unexpected modality_mask shape: {tuple(modality_mask.shape)}")

        # [B,T,C] -> [B,C,T] -> [B,C,N,P]
        patches = x.transpose(1, 2).unfold(
            dimension=2,
            size=c.patch_length,
            step=c.stride,
        )
        if patches.shape[2] != c.num_patches:
            raise RuntimeError(
                f"unexpected patch count: {patches.shape[2]} vs {c.num_patches}"
            )

        b, channels, n_patches, patch_len = patches.shape
        tokens = self.patch_embed(patches.reshape(b * channels, n_patches, patch_len))
        tokens = tokens + self.position

        channel_ids = torch.arange(channels, device=x.device)
        channel_ids = channel_ids.unsqueeze(0).expand(b, -1).reshape(-1)
        tokens = tokens + self.channel_embedding(channel_ids)[:, None, :]

        mask_ids = (modality_mask > 0.5).to(torch.long).reshape(-1)
        tokens = tokens + self.mask_embedding(mask_ids)[:, None, :]

        encoded = self.encoder(tokens)
        context = torch.cat([encoded[:, -1, :], encoded.mean(dim=1)], dim=1)
        residual = self.head(context).view(b, channels, c.horizon).transpose(1, 2)

        # A missing modality must remain exactly on the zero-filled persistence branch.
        residual = residual * modality_mask[:, None, :]
        persistence = x[:, -1:, :].expand(-1, c.horizon, -1)
        return persistence + residual
