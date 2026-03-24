"""
FID and Inception Score computation utilities.

Supports two modes:
- Online: Extract Inception features in-memory and compute FID from .npz reference stats.
- Disk-based: Save images to disk and use torch_fidelity.calculate_metrics.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:
    import torch_fidelity
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False


# ---------------------------------------------------------------------------
# Inception model
# ---------------------------------------------------------------------------

def get_inception_model(device: torch.device) -> torch.nn.Module:
    """Load InceptionV3 feature extractor from torch_fidelity.

    Returns model that outputs pool3 features (2048-d for FID)
    and logits (1008-d for IS).
    """
    if not HAS_TORCH_FIDELITY:
        raise ImportError("torch_fidelity is required. Install with: pip install torch-fidelity")

    weights_path = os.environ.get("INCEPTION_WEIGHTS")
    if not weights_path:
        raise RuntimeError("INCEPTION_WEIGHTS environment variable not set. Point it to the Inception .pth file.")
    
    model = FeatureExtractorInceptionV3(
        name="inception-v3-compat",
        features_list=["2048", "logits_unbiased"],
        feature_extractor_weights_path=weights_path,
    ).to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Online feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_inception_features(
    images: torch.Tensor,
    inception_model: torch.nn.Module,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract InceptionV3 features from images.

    Args:
        images: (N, C, H, W) float in [0, 1].
        inception_model: From get_inception_model().
        batch_size: Extraction batch size.

    Returns:
        pool3_features: (N, 2048) for FID.
        logits: (N, 1008) for IS.
    """
    device = next(inception_model.parameters()).device

    # torch_fidelity expects uint8 [0, 255]; the model handles resizing internally
    images = (images * 255).clamp(0, 255).to(torch.uint8)

    all_pool3, all_logits = [], []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        features = inception_model(batch)
        all_pool3.append(features[0].cpu())
        all_logits.append(features[1].cpu())

    return torch.cat(all_pool3, dim=0), torch.cat(all_logits, dim=0)


# ---------------------------------------------------------------------------
# FID from pre-extracted features
# ---------------------------------------------------------------------------

def compute_fid_from_features(
    features: torch.Tensor,
    fid_statistics_file: str,
) -> float:
    """Compute FID from pool3 features and a reference .npz statistics file.

    Uses the same formula as torch_fidelity (Frechet distance).
    """
    import scipy.linalg

    ref = np.load(fid_statistics_file)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]

    features_np = features.numpy().astype(np.float64)
    mu_gen = np.mean(features_np, axis=0)
    sigma_gen = np.cov(features_np, rowvar=False)

    diff = mu_gen - mu_ref
    covmean, _ = scipy.linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_gen.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma_gen + offset) @ (sigma_ref + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_gen) + np.trace(sigma_ref) - 2 * np.trace(covmean))
    return fid


# ---------------------------------------------------------------------------
# Inception Score
# ---------------------------------------------------------------------------

def compute_inception_score(
    logits: torch.Tensor,
    num_splits: int = 10,
) -> Tuple[float, float]:
    """Compute Inception Score from logits.

    Returns: (is_mean, is_std).
    """
    probs = F.softmax(logits[:, :1000].float(), dim=1).numpy()
    split_size = probs.shape[0] // num_splits

    scores = []
    for i in range(num_splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = np.sum(part * (np.log(part + 1e-10) - np.log(py + 1e-10)), axis=1)
        scores.append(np.exp(np.mean(kl)))

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Disk-based FID (torch_fidelity)
# ---------------------------------------------------------------------------

def fid_calc(
    image_path: str,
    fid_statistics_file: Optional[str] = None,
    num_expected: int = 50000,
    gt_image_path: Optional[str] = None,
    image_size: int = 256,
    cuda: bool = True,
    verbose: bool = False,
) -> Tuple[float, Optional[float]]:
    """Calculate FID using torch_fidelity (disk-based).

    Either fid_statistics_file or gt_image_path must be provided (not both).

    Returns: (fid, is_mean_or_None).
    """
    if not HAS_TORCH_FIDELITY:
        raise ImportError("torch_fidelity is required. Install with: pip install torch-fidelity")

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Verify image count on rank 0
    if rank == 0:
        flat = glob.glob(os.path.join(image_path, "*.png"))
        nested = glob.glob(os.path.join(image_path, "**", "*.png"), recursive=True)
        num_found = len(flat) if flat else len(nested)
        if num_found < num_expected:
            raise RuntimeError(f"[FID] Expected {num_expected} images, found {num_found}")

    fid, is_mean = None, None
    if rank == 0:
        if gt_image_path is None:
            assert fid_statistics_file is not None
            metrics = torch_fidelity.calculate_metrics(
                input1=image_path,
                input2=None,
                fid_statistics_file=fid_statistics_file,
                cuda=cuda,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=verbose,
                num_workers=0,
            )
        else:
            assert fid_statistics_file is None
            metrics = torch_fidelity.calculate_metrics(
                input1=image_path,
                input2=gt_image_path,
                fid_statistics_file=None,
                cuda=cuda,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=verbose,
                samples_find_deep=True,
                num_workers=0,
                batch_size=128,
            )
        fid = metrics["frechet_inception_distance"]
        is_mean = metrics.get("inception_score_mean")
        logger.info(f"[FID] FID={fid:.4f}" + (f", IS={is_mean:.4f}" if is_mean else ""))

    # Broadcast to all ranks
    if dist.is_initialized():
        dist.barrier()
        device = "cuda" if cuda else "cpu"
        fid_t = torch.tensor([fid or 0.0], device=device)
        is_t = torch.tensor([is_mean if is_mean is not None else -1.0], device=device)
        dist.broadcast(fid_t, src=0)
        dist.broadcast(is_t, src=0)
        fid = fid_t.item()
        is_mean = None if is_t.item() < 0 else is_t.item()
        dist.barrier()

    return fid, is_mean