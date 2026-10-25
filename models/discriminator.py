"""
Discriminator backbone for a DCGAN-style 32x32 image classifier.
"""

from __future__ import annotations

import torch
from torch import nn


def dcgan_weights_init(module: nn.Module) -> None:
    """
    Same initializer as Generator for consistency.
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0.0)


class Discriminator(nn.Module):
    """
    Maps image -> logit (real vs fake).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        # Note: No sigmoid here; we use BCEWithLogitsLoss for numerical stability.
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # BatchNorm can be unstable in D early layers; keep it optional.
        if use_batchnorm:
            layers += [
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            layers += [
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        # x shape is [B, 3, 32, 32]; output shape [B, 1, 1, 1].
        logits = self.net(x)
        return logits.view(-1)  # Flatten to [B] for BCEWithLogitsLoss.
