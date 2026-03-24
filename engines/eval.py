"""
Evaluation engine: FID, R-FID, adaptive CFG search.

Simplified from v3/engines/utils.py:
- Removed: dual EMA, S3 upload (commented out in v3)
- Kept: evaluate_fid (online FID), evaluate_rfid, adaptive_cfg_search
- All evaluation is online-only (in-memory Inception features, no disk I/O for FID)
"""

from __future__ import annotations

import os
import sys
import time
import logging
import shutil
from typing import Any, Dict, List, Optional, Tuple
import tqdm as tqdm_module
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from utils.crop import CenterCropTransform
from utils import wandb_utils
from utils.distributed import is_main_process
from utils.fid import (
    get_inception_model,
    extract_inception_features,
    compute_fid_from_features,
    compute_inception_score,
    fid_calc
)
from utils.eval_metrics import (
    compute_psnr, compute_l1, compute_lpips_batch, load_lpips_model, MetricAccumulator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wandb key formatting
# ---------------------------------------------------------------------------

def _format_cfg_key(
    cfg_scale: float,
    cfg_interval: Tuple[float, float],
    ema_decay: Optional[float] = None,
    metric_type: str = "score",
    cfg_norm_order: str = "cfg_first",
) -> str:
    """Format a wandb metric key.

    metric_type: "score" → FID_..., "is_score" → IS_..., "samples" → fid_..._samples.
    """
    low, high = cfg_interval
    cfg_str = f"{cfg_scale:.1f}"
    interval_str = f"int-{low}-{high}"

    if metric_type in ("score", "is_score"):
        prefix_map = {"cfg_first": "FID", "norm_first": "FIDpre", "both": "FIDboth"}
        if metric_type == "is_score":
            prefix_map = {"cfg_first": "IS", "norm_first": "ISpre", "both": "ISboth"}
        prefix = prefix_map.get(cfg_norm_order, "FID")
        ema_str = "noema" if ema_decay is None else f"ema{ema_decay:.4f}"
        return f"{prefix}_{cfg_str}_{interval_str}_{ema_str}"
    else:
        prefix_map = {"cfg_first": "fid", "norm_first": "fid_pre", "both": "fid_both"}
        prefix = prefix_map.get(cfg_norm_order, "fid")
        return f"{prefix}_{cfg_str}_{interval_str}_{metric_type}"


# ---------------------------------------------------------------------------
# FID reference paths
# ---------------------------------------------------------------------------

def _resolve_fid_stats(image_size: int = 256, num_classes: int = 1000) -> str:
    """Find FID statistics file on disk."""
    if image_size != 256:
        raise NotImplementedError(f"FID stats not available for image_size={image_size}")
    stats_path = os.environ.get("IN256_FID_STATS")
    if not stats_path:
        raise RuntimeError("IN256_FID_STATS environment variable not set. Point it to the .npz reference stats file.")
    return stats_path


# ---------------------------------------------------------------------------
# evaluate_fid (online, no disk I/O)
# ---------------------------------------------------------------------------

def evaluate_fid(
    ema_model: torch.nn.Module,
    device: torch.device,
    global_step: int,
    ema_decay: Optional[float] = None,
    image_size: int = 256,
    cfg_scales: Optional[List[float]] = None,
    cfg_intervals: Optional[List[Tuple[float, float]]] = None,
    cfg_norm_orders: Optional[List[str]] = None,
    num_images: Optional[int] = None,
    num_classes: int = 1000,
    batch_size: int = 50,
    log_to_wandb: bool = True,
    log_fid_best: bool = True,
    current_best_cfg_scale: float = 2.4,
) -> Tuple[int, float, float]:
    """Evaluate FID by generating images and computing FID online.

    Uses Inception feature extraction in memory — no disk I/O.

    Returns: (best_cfg_scale, best_fid).
    """

    if cfg_scales is None:
        cfg_scales = [2.4]
    if cfg_intervals is None:
        cfg_intervals = [(0.1, 1.0)]
    if cfg_norm_orders is None:
        cfg_norm_orders = ["norm_first"]

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if num_images is None:
        num_images = 5000 if num_classes == 100 else 50000

    ref_path = _resolve_fid_stats(image_size, num_classes=num_classes)

    if is_main_process():
        n_combos = len(cfg_norm_orders) * len(cfg_scales) * len(cfg_intervals)
        logger.info(
            f"[FID] Evaluating {n_combos} combinations, {num_images} images, "
            f"num_classes={num_classes}"
        )

    # Load Inception model once
    inception_model = get_inception_model(device)

    # Balanced labels
    if num_images % num_classes != 0 and is_main_process():
        logger.warning(
            f"[FID] num_images={num_images} is not divisible by num_classes={num_classes}; "
            f"truncating to balanced subset."
        )
    images_per_class = num_images // num_classes
    num_images = images_per_class * num_classes
    all_labels = np.tile(np.arange(num_classes), images_per_class)
    chunks = np.array_split(np.arange(num_images), world_size)
    my_indices = list(chunks[rank])
    my_labels = all_labels[my_indices]

    # Visualization labels (fixed seed)
    vis_rng = np.random.RandomState(seed=42)
    vis_labels = vis_rng.randint(0, num_classes, size=36)

    best_fid = float("inf")
    best_cfg_scale = current_best_cfg_scale
    overall_best_fid = float("inf")

    try:
        for cfg_norm_order in cfg_norm_orders:
            for cfg_scale in cfg_scales:
                for cfg_interval in cfg_intervals:
                    # Generate and extract features
                    local_pool3, local_logits = [], []
                    num_batches = (len(my_indices) + batch_size - 1) // batch_size

                    pbar = range(num_batches)
                    if is_main_process():
                        pbar = tqdm_module.tqdm(pbar, desc=f"FID cfg={cfg_scale}")

                    with torch.no_grad():
                        for batch_idx in pbar:
                            s = batch_idx * batch_size
                            e = min(s + batch_size, len(my_indices))
                            y = torch.tensor(my_labels[s:e], dtype=torch.long, device=device)

                            samples = ema_model.diffusion_generate(
                                device=device,
                                num_visuals=len(y),
                                cfg_scale=cfg_scale,
                                cfg_interval=cfg_interval,
                                cfg_norm_order=cfg_norm_order,
                                y_given=y,
                            ).clamp(0.0, 1.0)

                            pool3, logits = extract_inception_features(
                                samples, inception_model, batch_size=len(y)
                            )
                            local_pool3.append(pool3)
                            local_logits.append(logits)

                            if batch_idx % 20 == 0:
                                torch.cuda.empty_cache()

                    # Gather features across ranks
                    local_pool3 = torch.cat(local_pool3, dim=0) if local_pool3 else torch.empty(0, 2048)
                    local_logits = torch.cat(local_logits, dim=0) if local_logits else torch.empty(0, 1008)

                    local_count = torch.tensor([local_pool3.shape[0]], dtype=torch.long, device=device)
                    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
                    if dist.is_initialized():
                        dist.all_gather(all_counts, local_count)
                    else:
                        all_counts = [local_count]
                    all_counts = [c.item() for c in all_counts]
                    max_count = max(all_counts) if all_counts else 0

                    padded_pool3 = torch.zeros(max_count, 2048, device=device)
                    padded_pool3[: local_pool3.shape[0]] = local_pool3.to(device)
                    padded_logits = torch.zeros(max_count, 1008, device=device)
                    padded_logits[: local_logits.shape[0]] = local_logits.to(device)

                    if dist.is_initialized():
                        gathered_pool3 = [torch.zeros_like(padded_pool3) for _ in range(world_size)]
                        gathered_logits = [torch.zeros_like(padded_logits) for _ in range(world_size)]
                        dist.all_gather(gathered_pool3, padded_pool3)
                        dist.all_gather(gathered_logits, padded_logits)
                    else:
                        gathered_pool3 = [padded_pool3]
                        gathered_logits = [padded_logits]

                    # Compute FID/IS on rank 0
                    fid_val, is_val = 0.0, 0.0
                    if rank == 0:
                        all_p = torch.cat([g[:c].cpu() for g, c in zip(gathered_pool3, all_counts)])
                        all_l = torch.cat([g[:c].cpu() for g, c in zip(gathered_logits, all_counts)])
                        fid_val = compute_fid_from_features(all_p, ref_path)
                        is_val, _ = compute_inception_score(all_l)
                        logger.info(f"[FID] FID={fid_val:.4f}, IS={is_val:.4f} "
                                    f"(cfg={cfg_scale}, order={cfg_norm_order})")

                    # Broadcast
                    fid_t = torch.tensor([fid_val], device=device)
                    is_t = torch.tensor([is_val], device=device)
                    if dist.is_initialized():
                        dist.broadcast(fid_t, src=0)
                        dist.broadcast(is_t, src=0)
                    fid_val = fid_t.item()
                    is_val = is_t.item()

                    # Track bests
                    if fid_val < overall_best_fid:
                        overall_best_fid = fid_val
                    if cfg_norm_order == "norm_first" and fid_val < best_fid:
                        best_fid = fid_val
                        best_cfg_scale = cfg_scale

                    # Wandb logging
                    if log_to_wandb and rank == 0:
                        with torch.no_grad():
                            vis_y = torch.tensor(vis_labels, dtype=torch.long, device=device)
                            vis = ema_model.diffusion_generate(
                                device=device,
                                num_visuals=36,
                                cfg_scale=cfg_scale,
                                cfg_interval=cfg_interval,
                                cfg_norm_order=cfg_norm_order,
                                y_given=vis_y,
                            ).clamp(0.0, 1.0).cpu()

                        try:
                            import wandb
                            from einops import rearrange
                            vis_img = rearrange(vis[:36], "(r c) ch h w -> ch (r h) (c w)", r=6, c=6)
                            fid_key = _format_cfg_key(cfg_scale, cfg_interval, ema_decay, "score", cfg_norm_order)
                            is_key = _format_cfg_key(cfg_scale, cfg_interval, ema_decay, "is_score", cfg_norm_order)
                            vis_key = _format_cfg_key(cfg_scale, cfg_interval, None, "samples", cfg_norm_order)
                            wandb_utils.log({
                                fid_key: fid_val,
                                is_key: is_val,
                                vis_key: [wandb.Image(vis_img)],
                            }, step=global_step)
                        except ImportError:
                            pass

                    # Cleanup GPU memory
                    del local_pool3, local_logits, padded_pool3, padded_logits
                    del gathered_pool3, gathered_logits
                    torch.cuda.empty_cache()

                    if dist.is_initialized():
                        dist.barrier()

    finally:
        del inception_model
        torch.cuda.empty_cache()

    best_cfg_scale = round(best_cfg_scale, 1)

    if log_to_wandb and log_fid_best and is_main_process():
        wandb_utils.log({"fid-best": overall_best_fid, "fid/best_cfg_scale": best_cfg_scale}, step=global_step)

    return best_cfg_scale, overall_best_fid


# ---------------------------------------------------------------------------
# Adaptive CFG search
# ---------------------------------------------------------------------------

def adaptive_cfg_search(
    ema_model: torch.nn.Module,
    device: torch.device,
    global_step: int,
    ema_decay: Optional[float] = None,
    image_size: int = 256,
    current_best_cfg_scale: float = 2.4,
    cfg_interval: Tuple[float, float] = (0.1, 1.0),
    cfg_norm_order: str = "norm_first",
    batch_size: int = 50,
    num_images: Optional[int] = None,
    num_classes: int = 1000,
    max_iterations: int = 7,
) -> Tuple[int, float, float]:
    """Bidirectional adaptive CFG search.

    1. Try X and X-0.2
    2. Search in the better direction until FID stops improving.

    Returns: (best_cfg_scale, best_fid).
    """
    X = round(current_best_cfg_scale, 1)

    if is_main_process():
        logger.info(f"[Adaptive CFG] Starting from X={X}")

    overall_best = float("inf")

    # Step 1: try X and X-0.2
    initial = [s for s in [X, round(X - 0.2, 1)] if s > 0]
    best_cfg, best_fid = evaluate_fid(
        ema_model, device, global_step, ema_decay, image_size,
        cfg_scales=initial, cfg_intervals=[cfg_interval], cfg_norm_orders=[cfg_norm_order],
        batch_size=batch_size, num_images=num_images, num_classes=num_classes,
        log_fid_best=False, current_best_cfg_scale=X,
    )
    if best_fid < overall_best:
        overall_best = best_fid

    # Determine direction
    direction = -0.2 if best_cfg < X else 0.2
    prev_fid = best_fid

    for _ in range(max_iterations):
        next_scale = round(best_cfg + direction, 1)
        if next_scale <= 0 or next_scale > 10.0:
            break
        try:
            _, new_fid = evaluate_fid(
                ema_model, device, global_step, ema_decay, image_size,
                cfg_scales=[next_scale], cfg_intervals=[cfg_interval],
                cfg_norm_orders=[cfg_norm_order], batch_size=batch_size,
                num_images=num_images, num_classes=num_classes,
                log_fid_best=False, current_best_cfg_scale=best_cfg,
            )
            if new_fid < overall_best:
                overall_best = new_fid
            if new_fid >= prev_fid:
                break
            best_cfg = next_scale
            prev_fid = new_fid
        except Exception as e:
            if is_main_process():
                logger.warning(f"[Adaptive CFG] Error at {next_scale}: {e}")
            break

    best_cfg = round(best_cfg, 1)

    if is_main_process():
        wandb_utils.log({"fid-best": overall_best, "fid/best_cfg_scale": best_cfg}, step=global_step)
        logger.info(f"[Adaptive CFG] Done: cfg={best_cfg}, fid-best={overall_best:.3f}")

    return best_cfg, overall_best


# ---------------------------------------------------------------------------
# R-FID (Reconstruction FID)
# ---------------------------------------------------------------------------

def evaluate_rfid(
    ema_model: torch.nn.Module,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    global_step: int = 0,
    image_size: int = 256,
) -> bool:
    """Evaluate R-FID on ImageNet validation set.

    Reconstructs 50K images, computes R-FID-val (vs GT), PSNR, L1, LPIPS.

    Returns True if evaluation ran, False if skipped.
    """

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Find validation directory
    train_dir = os.environ.get("TRAIN_DATA_DIR", "")
    if not train_dir or not train_dir.endswith("/train"):
        if is_main_process():
            logger.info("[R-FID] Skipping: TRAIN_DATA_DIR not set or doesn't end with /train")
        return False

    val_dir = train_dir.replace("/train", "/val")
    if not os.path.isdir(val_dir):
        if is_main_process():
            logger.info(f"[R-FID] Skipping: val dir not found: {val_dir}")
        return False

    TOTAL = 50000
    BATCH_SIZE = 64

    val_transform = transforms.Compose([
        CenterCropTransform(image_size),
        transforms.ToTensor(),
    ])

    try:
        val_dataset = ImageFolder(val_dir, transform=val_transform)
    except Exception as e:
        if is_main_process():
            logger.warning(f"[R-FID] Failed to load validation dataset: {e}")
        return False

    if len(val_dataset) < TOTAL:
        TOTAL = len(val_dataset)
    else:
        val_dataset = Subset(val_dataset, list(range(TOTAL)))

    all_indices = list(range(TOTAL))
    chunks = np.array_split(all_indices, world_size)
    my_indices = list(chunks[rank])

    # Create temp directories
    if rank == 0:
        ts = int(time.time())
        for base in ["/tmp"]:
            if os.path.exists(base):
                base_dir = f"{base}/temp-rfid-{ts}"
                break
        else:
            base_dir = f"/tmp/temp-rfid-{ts}"
    else:
        base_dir = None

    base_dir_list = [base_dir]
    if dist.is_initialized():
        dist.broadcast_object_list(base_dir_list, src=0)
    base_dir = base_dir_list[0]

    gt_dir = f"{base_dir}/gt"
    cleanup_done = False

    def _cleanup():
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        if rank == 0:
            try:
                shutil.rmtree(base_dir)
            except Exception:
                pass

    try:
        os.makedirs(gt_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        lpips_model = load_lpips_model(device)

        models_to_eval = [("EMA", ema_model, "")]
        if model is not None:
            raw = model.module if hasattr(model, "module") else model
            models_to_eval.append(("non-EMA", raw, "_nonema"))

        # Save GT images
        with torch.no_grad():
            my_subset = Subset(val_dataset, my_indices)
            loader = DataLoader(my_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            for batch_idx, (imgs, _labels) in enumerate(loader):
                gt_np = (imgs.numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
                for i in range(gt_np.shape[0]):
                    local_idx = batch_idx * BATCH_SIZE + i
                    if local_idx >= len(my_indices):
                        break
                    Image.fromarray(gt_np[i]).save(f"{gt_dir}/{my_indices[local_idx]:06d}.png", compress_level=0)

        if dist.is_initialized():
            dist.barrier()

        # Check ref path
        try:
            ref_path = _resolve_fid_stats(image_size)
        except FileNotFoundError:
            ref_path = None

        all_stats = {}

        for model_name, eval_model, suffix in models_to_eval:
            recon_dir = f"{base_dir}/recon_{model_name.lower().replace('-', '_')}"
            os.makedirs(recon_dir, exist_ok=True)
            if dist.is_initialized():
                dist.barrier()

            eval_model.eval()
            psnr_acc = MetricAccumulator()
            l1_acc = MetricAccumulator()
            lpips_acc = MetricAccumulator()

            with torch.no_grad():
                my_subset = Subset(val_dataset, my_indices)
                loader = DataLoader(my_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                pbar = loader
                if is_main_process():
                    pbar = tqdm_module.tqdm(loader, desc=f"R-FID ({model_name})")

                for batch_idx, (imgs, labels) in enumerate(pbar):
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    recon = eval_model.reconstruct_batch(imgs, labels=labels).clamp(0.0, 1.0)
                    lpips_val = compute_lpips_batch(recon, imgs, lpips_model, device)
                    lpips_acc.update(lpips_val, count=recon.shape[0])

                    recon_np = (recon.cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
                    gt_np = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)

                    psnr_acc.update(compute_psnr(recon_np, gt_np), count=recon_np.shape[0])
                    l1_acc.update(compute_l1(recon_np, gt_np), count=recon_np.shape[0])

                    for i in range(recon_np.shape[0]):
                        local_idx = batch_idx * BATCH_SIZE + i
                        if local_idx >= len(my_indices):
                            break
                        Image.fromarray(recon_np[i]).save(
                            f"{recon_dir}/{my_indices[local_idx]:06d}.png", compress_level=0
                        )

                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()

            if dist.is_initialized():
                dist.barrier()

            psnr_final = psnr_acc.compute(device)
            l1_final = l1_acc.compute(device)
            lpips_final = lpips_acc.compute(device)

            # R-FID-val (vs JIT-preprocessed GT)
            rfid_val, is_val = fid_calc(
                recon_dir, gt_image_path=gt_dir, num_expected=TOTAL, image_size=image_size, verbose=True,
            )

            rfid_val_t = torch.tensor([rfid_val or 0.0], device=device)
            is_val_t = torch.tensor([float(is_val if is_val is not None else 0.0)], device=device)
            if dist.is_initialized():
                dist.broadcast(rfid_val_t, src=0)
                dist.broadcast(is_val_t, src=0)

            # v3-compatible aliases: non-val and val names both present
            all_stats[f"rfid/score{suffix}"] = rfid_val_t.item()
            all_stats[f"rfid-val/score{suffix}"] = rfid_val_t.item()
            all_stats[f"rinception/score{suffix}"] = is_val_t.item()
            all_stats[f"rinception-val/score{suffix}"] = is_val_t.item()
            all_stats[f"rpsnr/score{suffix}"] = psnr_final
            all_stats[f"rl1/score{suffix}"] = l1_final
            all_stats[f"rlpips/score{suffix}"] = lpips_final

            if rank == 0:
                try:
                    shutil.rmtree(recon_dir)
                except Exception:
                    pass

            if dist.is_initialized():
                dist.barrier()

        # Log
        if rank == 0:
            wandb_utils.log(all_stats, step=global_step)
            logger.info(f"[R-FID] Results: {all_stats}")

        _cleanup()
        if dist.is_initialized():
            dist.barrier()
        return True

    finally:
        _cleanup()




def on_epoch_end_fid(
    ema, ema_decay, 
    device, global_step, epoch, 
    best_cfg, is_sweep, is_normal,
    image_size, fid_num_classes, fid_num_images, eval_batch_size):

    # Epoch 0: toy FID run (5K images) to verify pipeline
    if epoch == 0:
        if is_main_process() == 0:
            logger.info("[FID] Epoch 0 toy run: 5K images, CFG=3.4, norm_first")
        _, _ = evaluate_fid(
            ema, device, global_step, ema_decay, image_size,
            cfg_scales=[3.4], cfg_intervals=[(0.1, 1.0)], cfg_norm_orders=["norm_first"],
            num_images=5000, num_classes=fid_num_classes,
            log_to_wandb=False, batch_size=eval_batch_size,
            current_best_cfg_scale=best_cfg,
        )
        return best_cfg


    if not (is_sweep or is_normal):
        return best_cfg

    if is_normal:
        if is_main_process() == 0:
            logger.info(f"[FID] Normal evaluation at epoch {epoch}: adaptive search from {best_cfg}")
        best_cfg, best_fid = adaptive_cfg_search(
            ema, device, global_step, ema_decay,
            image_size, best_cfg, batch_size=eval_batch_size,
            num_images=fid_num_images, num_classes=fid_num_classes,
        )
        if is_main_process() == 0:
            logger.info(f"[FID] Updated best_cfg to {best_cfg:.2f}, best_fid={best_fid:.3f}")

    else:  # is_sweep
        interval_tests = [(0.0, 1.0), (0.05, 1.0), (0.15, 1.0), (0.2, 1.0)]
        overall_best_fid = float('inf')

        # Step 1: Adaptive CFG search
        if is_main_process() == 0:
            logger.info(f"[FID] Sweep Step 1: EMA Adaptive CFG search from {best_cfg}")
        best_cfg, best_fid_s1 = adaptive_cfg_search(
            ema, device, global_step, ema_decay,
            image_size, best_cfg, batch_size=eval_batch_size,
            num_images=fid_num_images, num_classes=fid_num_classes,
        )
        overall_best_fid = min(overall_best_fid, best_fid_s1)
        if is_main_process() == 0:
            logger.info(f"[FID] Step 1 done: best_cfg={best_cfg:.1f}, fid={best_fid_s1:.3f}")

        # Step 2: Interval sweep with best scale, norm_first
        if is_main_process() == 0:
            logger.info(f"[FID] Sweep Step 2: interval sweep {interval_tests} with cfg={best_cfg:.1f}")
        _, best_fid_s2 = evaluate_fid(
            ema, device, global_step, ema_decay, image_size,
            cfg_scales=[best_cfg], cfg_intervals=interval_tests,
            cfg_norm_orders=["norm_first"], log_fid_best=False,
            batch_size=eval_batch_size, num_images=fid_num_images, num_classes=fid_num_classes,
            current_best_cfg_scale=best_cfg,
        )
        overall_best_fid = min(overall_best_fid, best_fid_s2)

        # Step 3: cfg_first with best scale
        if is_main_process() == 0:
            logger.info(f"[FID] Sweep Step 3: cfg_first with cfg={best_cfg:.1f}")
        _, best_fid_s3 = evaluate_fid(
            ema, device, global_step, ema_decay, image_size,
            cfg_scales=[best_cfg], cfg_intervals=[(0.1, 1.0)],
            cfg_norm_orders=["cfg_first"], log_fid_best=False,
            batch_size=eval_batch_size, num_images=fid_num_images, num_classes=fid_num_classes,
            current_best_cfg_scale=best_cfg,
        )
        overall_best_fid = min(overall_best_fid, best_fid_s3)

        if is_main_process() == 0:
            logger.info(f"[FID] Sweep complete: fid-best={overall_best_fid:.3f}, best_cfg={best_cfg:.1f}")

    return best_cfg


def on_epoch_end_rfid(ema, raw_model, device, global_step, epoch):
    if is_main_process() == 0:
        logger.info(f"[R-FID] Running reconstruction FID at epoch {epoch + 1}")
    evaluate_rfid(ema, model=raw_model, device=device, global_step=global_step)


