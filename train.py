"""
Entry point for training a small DCGAN on CIFAR10-like data.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from datasets.cifar_like import build_cifar10_dataloader
from losses.gan_loss import GANLoss
from models.discriminator import Discriminator, dcgan_weights_init as d_init
from models.generator import Generator, dcgan_weights_init as g_init
from trainer.gan_trainer import GANTrainer, TrainConfig


def set_seed(seed: int) -> None:
    """
    Basic reproducibility helper.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small DCGAN on CIFAR10.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 2 on CUDA/CPU, 0 on Apple MPS)")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint to continue from (writes into the same run folder)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Faster, but less deterministic.
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU.
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    # Worker processes can stall training on macOS + MPS, so default to in-process loading there.
    if args.num_workers is None:
        args.num_workers = 0 if device.type == "mps" else 2

    loader = build_cifar10_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    generator = Generator(z_dim=args.z_dim).to(device)
    discriminator = Discriminator().to(device)

    # Weight init helps stabilize early training.
    generator.apply(g_init)
    discriminator.apply(d_init)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    loss_fn = GANLoss()

    if args.resume:
        ckpt_dir = os.path.dirname(os.path.abspath(args.resume))  # runs/<run>/checkpoints
        log_dir = os.path.dirname(ckpt_dir)
    else:
        log_dir = os.path.join("./runs", time.strftime("dcgan_%Y%m%d_%H%M%S"))
        ckpt_dir = os.path.join(log_dir, "checkpoints")

    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        loss_fn=loss_fn,
        device=device,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        fixed_noise=fixed_noise,
    )

    config = TrainConfig(
        epochs=args.epochs,
        z_dim=args.z_dim,
        label_smoothing=args.label_smoothing,
        log_every=args.log_every,
        sample_every=args.sample_every,
        checkpoint_every=args.ckpt_every,
    )

    start_epoch = 1
    if args.resume:
        # Same seed and creation order as a fresh run, so fixed_noise (and the sample grids) match.
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"resumed from {args.resume}: continuing at epoch {start_epoch}")

    trainer.train(loader, config, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
