"""
Evaluation metrics: PSNR, L1, LPIPS, and distributed MetricAccumulator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------

def compute_psnr(recon: np.ndarray, gt: np.ndarray) -> float:
    """Compute PSNR between reconstructed and ground truth uint8 images.

    Args:
        recon, gt: [N, H, W, C] uint8 [0, 255].

    Returns:
        Average PSNR in dB.
    """
    assert recon.shape == gt.shape and recon.dtype == np.uint8 and gt.dtype == np.uint8
    recon_f = recon.astype(np.float64)
    gt_f = gt.astype(np.float64)
    mse = np.mean((recon_f - gt_f) ** 2, axis=(1, 2, 3))
    mse = np.maximum(mse, 1e-10)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return float(np.mean(psnr))


def compute_l1(recon: np.ndarray, gt: np.ndarray) -> float:
    """Compute normalized L1 error between uint8 images.

    Returns:
        Average L1 in [0, 1] range (divided by 255).
    """
    assert recon.shape == gt.shape and recon.dtype == np.uint8 and gt.dtype == np.uint8
    return float(np.mean(np.abs(recon.astype(np.float64) - gt.astype(np.float64))) / 255.0)


# ---------------------------------------------------------------------------
# Perceptual metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_lpips_batch(
    recon: torch.Tensor,
    gt: torch.Tensor,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> float:
    """Compute LPIPS for a batch of [0,1] float images.

    Args:
        recon, gt: [N, C, H, W] float [0, 1].
        lpips_model: Pre-loaded LPIPS model.

    Returns:
        Average LPIPS score (lower = more similar).
    """
    return float(lpips_model(recon.to(device), gt.to(device)).mean().item())


def load_lpips_model(device: torch.device) -> torch.nn.Module:
    """Load LPIPS model for evaluation."""
    from modules.autoencoding_utils.lpips import LPIPS
    return LPIPS().to(device).eval()


# ---------------------------------------------------------------------------
# Distributed metric aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics_across_ranks(
    local_sum: float,
    local_count: int,
    device: torch.device,
) -> float:
    """Aggregate metric sums/counts across all ranks. Returns global average."""
    if not dist.is_initialized():
        return local_sum / max(local_count, 1)
    sum_t = torch.tensor([local_sum], dtype=torch.float64, device=device)
    count_t = torch.tensor([local_count], dtype=torch.float64, device=device)
    dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    return sum_t.item() / max(count_t.item(), 1)


class MetricAccumulator:
    """Running metric accumulator with distributed aggregation support.

    Usage::

        acc = MetricAccumulator()
        for batch in loader:
            acc.update(psnr_value, count=batch_size)
        final = acc.compute(device)  # aggregates across ranks
    """

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1):
        """Add value * count to running sum."""
        self.sum += value * count
        self.count += count

    def update_sum(self, value_sum: float, count: int):
        """Add pre-computed sum."""
        self.sum += value_sum
        self.count += count

    def compute(self, device: torch.device = None) -> float:
        """Compute final average, aggregating across ranks if distributed."""
        if device is not None and dist.is_initialized():
            return aggregate_metrics_across_ranks(self.sum, self.count, device)
        return self.sum / max(self.count, 1)

    def reset(self):
        self.sum = 0.0
        self.count = 0
