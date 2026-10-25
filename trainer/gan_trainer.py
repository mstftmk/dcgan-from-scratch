"""
Training loop, logging, and checkpointing for GANs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from losses.gan_loss import GANLoss


@dataclass
class TrainConfig:
    epochs: int
    z_dim: int
    label_smoothing: float = 0.0
    log_every: int = 100
    sample_every: int = 500
    checkpoint_every: int = 1


class GANTrainer:
    """
    Orchestrates GAN training with clean logging and checkpoints.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        loss_fn: GANLoss,
        device: torch.device,
        log_dir: str,
        ckpt_dir: str,
        fixed_noise: torch.Tensor,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.fixed_noise = fixed_noise

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.sample_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save training state for resume/debugging.
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"gan_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "g_optimizer": self.g_optimizer.state_dict(),
                "d_optimizer": self.d_optimizer.state_dict(),
            },
            ckpt_path,
        )

    def load_checkpoint(self, ckpt_path: str) -> int:
        """
        Restore models, optimizers and the step counter; return the last finished epoch.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.g_optimizer.load_state_dict(ckpt["g_optimizer"])
        self.d_optimizer.load_state_dict(ckpt["d_optimizer"])
        self.global_step = ckpt["global_step"]
        return ckpt["epoch"]

    def _log_samples(self, epoch: int) -> None:
        """
        Generate a fixed grid of samples for visual tracking.
        """
        self.generator.eval()
        with torch.no_grad():
            fake = self.generator(self.fixed_noise)
        self.generator.train()

        # Make a grid; normalize to [0,1] for image saving.
        grid = make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
        save_image(grid, os.path.join(self.sample_dir, f"epoch_{epoch:03d}.png"))
        self.writer.add_image("samples", grid, global_step=epoch)

    def train(self, dataloader: DataLoader, config: TrainConfig, start_epoch: int = 1) -> None:
        """
        Main training loop. start_epoch > 1 continues a resumed run.
        """
        start_time = time.time()
        self.generator.train()
        self.discriminator.train()

        for epoch in range(start_epoch, config.epochs + 1):
            for real, _ in dataloader:
                self.global_step += 1

                # ----------------------------
                # 1) Train Discriminator
                # ----------------------------
                real = real.to(self.device, non_blocking=True)
                batch_size = real.size(0)

                # Noise shape [B, z_dim, 1, 1]; memory scales with B and z_dim.
                noise = torch.randn(batch_size, config.z_dim, 1, 1, device=self.device)
                fake = self.generator(noise)

                real_logits = self.discriminator(real)
                fake_logits = self.discriminator(fake.detach())
                # detach() blocks gradient flow into G for D update.

                d_losses = self.loss_fn.discriminator_loss(
                    real_logits=real_logits,
                    fake_logits=fake_logits,
                    label_smoothing=config.label_smoothing,
                )

                self.d_optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory.
                d_losses.total.backward()
                self.d_optimizer.step()

                # ----------------------------
                # 2) Train Generator
                # ----------------------------
                noise = torch.randn(batch_size, config.z_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                fake_logits = self.discriminator(fake)
                # Gradients flow through D into G, improving generator updates.

                g_loss = self.loss_fn.generator_loss(fake_logits)

                self.g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                self.g_optimizer.step()

                # ----------------------------
                # Logging
                # ----------------------------
                if self.global_step % config.log_every == 0:
                    elapsed = time.time() - start_time
                    self.writer.add_scalar("loss/d_total", d_losses.total.item(), self.global_step)
                    self.writer.add_scalar("loss/d_real", d_losses.real.item(), self.global_step)
                    self.writer.add_scalar("loss/d_fake", d_losses.fake.item(), self.global_step)
                    self.writer.add_scalar("loss/g", g_loss.item(), self.global_step)
                    print(
                        f"[epoch {epoch:03d}] step {self.global_step:06d} "
                        f"D: {d_losses.total.item():.4f} (R {d_losses.real.item():.4f}, F {d_losses.fake.item():.4f}) "
                        f"G: {g_loss.item():.4f} time: {elapsed:.1f}s"
                    )

                if self.global_step % config.sample_every == 0:
                    self._log_samples(epoch)

            if epoch % config.checkpoint_every == 0:
                self.save_checkpoint(epoch)
