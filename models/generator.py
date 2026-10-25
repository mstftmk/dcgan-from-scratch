"""
Generator backbone for a DCGAN-style 32x32 image synthesis.
"""

from __future__ import annotations

import torch
from torch import nn


def dcgan_weights_init(module: nn.Module) -> None:
    """
    Initialize weights following DCGAN recommendations.
    """
    if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0.0)


class Generator(nn.Module):
    """
    Maps a latent vector z -> RGB image in [-1, 1].
    """

    def __init__(
        self,
        z_dim: int = 128,
        base_channels: int = 64,
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        # Shape flow: [B, z_dim, 1, 1] -> [B, 4*base, 4, 4] -> ... -> [B, 3, 32, 32]
        # ConvTranspose upsamples while learning, which improves gradient flow.
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base_channels * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # Output in [-1, 1] matches dataset normalization for stability.
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        # z shape is [B, z_dim, 1, 1]. Memory footprint scales with batch size.
        return self.net(z)
