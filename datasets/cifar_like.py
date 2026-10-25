"""
Dataset helpers for CIFAR10-like 32x32 RGB images.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_cifar10_dataloader(
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
    subset_size: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader that yields images normalized to [-1, 1].

    The output tensor shape is [B, 3, 32, 32].
    """

    # Transform pipeline keeps shape explicit and matches Tanh output range.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1] float, shape [C, H, W].
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # -> [-1, 1] range for numerical stability with Tanh generator.
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    if subset_size is not None:
        # Deterministic subset for small, realistic training.
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g)[:subset_size].tolist()
        dataset = Subset(dataset, indices)

    # drop_last keeps batch size consistent, which stabilizes BatchNorm stats.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster H2D copies on CUDA; small memory overhead.
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return loader
