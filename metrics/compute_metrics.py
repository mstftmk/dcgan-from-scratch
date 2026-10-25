"""
Compute GAN evaluation metrics (FID, Inception Score) for CIFAR10-like images.

One Inception-v3 pass per image yields both the 2048-d pool features (for FID) and the
unbiased logits (for Inception Score), so each image is processed once. Statistics of the
real images are cached on disk, so later evaluations only process generated images.

Same Inception weights and preprocessing as torchmetrics / torch-fidelity
("inception-v3-compat"), so numbers are comparable with those libraries.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterator, Optional

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.generator import Generator

try:
    from torchmetrics.image.fid import NoTrainInceptionV3  # Wraps torch-fidelity's Inception-v3.

    TORCHMETRICS_AVAILABLE = True
except Exception:
    TORCHMETRICS_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID and Inception Score for a GAN.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint with generator weights.")
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gen", type=int, default=10000, help="Number of generated images.")
    parser.add_argument("--real-subset", type=int, default=10000, help="Number of real images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="./metrics/cache", help="Where real-image statistics are stored.")
    # Inception runs on CPU: on Apple MPS it measured slower than CPU (some ops fall back), and the
    # float64 statistics are not supported there.
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor to uint8 in [0, 255]. Accepts [0, 1] or [-1, 1] floats.
    """
    if images.dtype.is_floating_point:
        if images.min() < 0:
            images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]
        images = (images.clamp(0.0, 1.0) * 255.0).round()
    return images.to(torch.uint8)


class Progress:
    """Prints done/total, images per second and time left about every 10 seconds."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total, self.done = label, total, 0
        self.start = self.last = time.time()

    def update(self, n: int) -> None:
        self.done += n
        now = time.time()
        if now - self.last >= 10 or self.done >= self.total:
            rate = self.done / max(now - self.start, 1e-9)
            left = (self.total - self.done) / max(rate, 1e-9)
            print(f"[{self.label}] {self.done}/{self.total} images | {rate:.1f} img/s | ~{left / 60:.1f} min left", flush=True)
            self.last = now


@torch.no_grad()
def extract(net: torch.nn.Module, batches: Iterator[torch.Tensor], total: int, label: str, device: str):
    """Return (pool features [N, 2048], unbiased logits [N, 1008]) as float64 numpy arrays."""
    feats, logits, progress = [], [], Progress(label, total)
    for batch in batches:
        f, l = net._torch_fidelity_forward(to_uint8(batch).to(device))
        feats.append(f.double().cpu())
        logits.append(l.double().cpu())
        progress.update(batch.size(0))
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2))."""
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():  # Regularize near-singular products.
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def inception_score(logits: np.ndarray, splits: int = 10) -> tuple[float, float]:
    """IS = exp(E_x KL(p(y|x) || p(y))), averaged over `splits` chunks (as in torchmetrics)."""
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    scores = []
    for chunk in np.array_split(probs, splits):
        p_y = chunk.mean(axis=0, keepdims=True)
        kl = (chunk * (np.log(chunk + 1e-12) - np.log(p_y + 1e-12))).sum(axis=1).mean()
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def real_batches(data_root: str, batch_size: int, num_workers: int, subset_size: Optional[int], seed: int):
    dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transforms.ToTensor())
    if subset_size is not None:
        g = torch.Generator().manual_seed(seed)
        dataset = Subset(dataset, torch.randperm(len(dataset), generator=g)[:subset_size].tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return (images for images, _ in loader), len(dataset)


def fake_batches(generator: Generator, num: int, batch_size: int, z_dim: int, device: str):
    done = 0
    while done < num:
        b = min(batch_size, num - done)
        yield generator(torch.randn(b, z_dim, 1, 1, device=device)).cpu()
        done += b


def load_generator(ckpt_path: str, z_dim: int, device: str) -> Generator:
    generator = Generator(z_dim=z_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    generator.load_state_dict(ckpt["generator"] if "generator" in ckpt else ckpt)
    return generator.eval()


def main() -> None:
    args = parse_args()
    if not TORCHMETRICS_AVAILABLE:
        raise RuntimeError("torchmetrics is not installed. Install with: pip install torchmetrics torch-fidelity")

    torch.manual_seed(args.seed)
    net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048", "logits_unbiased"]).to(args.device).eval()

    # 1) Real-image statistics, cached by subset size and seed.
    os.makedirs(args.cache_dir, exist_ok=True)
    cache = os.path.join(args.cache_dir, f"cifar10_train_n{args.real_subset}_seed{args.seed}.npz")
    if os.path.exists(cache):
        stats = np.load(cache)
        mu_r, sigma_r = stats["mu"], stats["sigma"]
        print(f"loaded real statistics from {cache}")
    else:
        batches, n_real = real_batches(args.data_root, args.batch_size, args.num_workers, args.real_subset, args.seed)
        feats_r, _ = extract(net, batches, n_real, "real", args.device)
        mu_r, sigma_r = feats_r.mean(axis=0), np.cov(feats_r, rowvar=False)
        np.savez(cache, mu=mu_r, sigma=sigma_r)

    # 2) Generated images: one pass gives FID features and IS logits.
    generator = load_generator(args.ckpt, args.z_dim, args.device)
    with torch.no_grad():
        feats_g, logits_g = extract(net, fake_batches(generator, args.num_gen, args.batch_size, args.z_dim, args.device),
                                    args.num_gen, "generated", args.device)

    fid = frechet_distance(feats_g.mean(axis=0), np.cov(feats_g, rowvar=False), mu_r, sigma_r)
    is_mean, is_std = inception_score(logits_g, splits=10)

    print("=== Metrics ===")
    print(f"real images: {args.real_subset} | generated images: {args.num_gen}")
    print(f"FID: {fid:.4f}")
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")


if __name__ == "__main__":
    main()
