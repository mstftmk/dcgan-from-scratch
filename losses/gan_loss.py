"""
GAN loss helpers based on BCEWithLogits for stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossOutput:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor


class GANLoss:
    """
    Encapsulates discriminator and generator losses.
    """

    def __init__(self) -> None:
        # BCEWithLogitsLoss combines sigmoid + BCE, improving numerical stability.
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> LossOutput:
        """
        Compute D loss from real and fake logits.
        """
        # Real targets in (0,1]; smoothing pushes them slightly below 1.
        real_targets = torch.full_like(real_logits, 1.0 - label_smoothing)
        fake_targets = torch.zeros_like(fake_logits)

        real_loss = self.criterion(real_logits, real_targets)
        fake_loss = self.criterion(fake_logits, fake_targets)

        total = real_loss + fake_loss
        return LossOutput(total=total, real=real_loss, fake=fake_loss)

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute G loss by trying to fool D into predicting real.
        """
        targets = torch.ones_like(fake_logits)
        return self.criterion(fake_logits, targets)
