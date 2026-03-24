#!/usr/bin/env python
"""Training entry point for UNITE.

DDP training with AdamW + Muon, two-stage LR schedule, gradient accumulation,
torch.compile, and adaptive CFG FID evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wandb
import logging
from copy import deepcopy
from pathlib import Path

import yaml
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import wandb_utils
from utils.logging import create_logger
from utils.distributed import setup_distributed, cleanup_distributed
from utils.checkpoint import load_checkpoint
from utils.data import prepare_dataloader, get_transform_name
from utils.optim import build_scheduler, build_optimizer, build_optimizer_muon, scale_learning_rates
from models.unite import UNITE
from engines.train import train_one_epoch
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
# Enable TF32: set both new and legacy APIs to keep torch.compile/Inductor happy
# (Inductor checks the legacy flag internally and errors if it disagrees with the new one)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "tf32"
if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
    torch.backends.cudnn.conv.fp32_precision = "tf32"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNITE model.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--data-path", type=Path, required=True, help="ImageFolder dataset path.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path to resume from.")
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    rank, world_size, device = setup_distributed()

    # Load config
    with open(args.config, "r") as f:
        full_cfg = yaml.safe_load(f)

    training_cfg = dict(full_cfg.get("training", {}))
    gen_tok_cfg = dict(full_cfg.get("gen_tok", {}))
    if not gen_tok_cfg:
        raise ValueError("Config must define a 'gen_tok' section.")

    # Batch size + LR scaling
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    batch_size = int(training_cfg.get("batch_size", 16))
    total_batch_size = training_cfg.get("total_batch_size")
    if total_batch_size is not None:
        total_batch_size = int(total_batch_size)
        batch_size = total_batch_size // (world_size * grad_accum_steps)
    else:
        total_batch_size = batch_size * world_size * grad_accum_steps

    lr_factor = max(1.0, total_batch_size / 384.0)
    scale_learning_rates(training_cfg, lr_factor)
    if rank == 0:
        print(f"Total batch size: {total_batch_size}, per-GPU batch: {batch_size}, world size: {world_size}, grad_accum: {grad_accum_steps}")
        print(f"LR batch size factor: {lr_factor:.4f} (scaling all LRs by total_batch_size/384.0)")

    if batch_size == 96 and torch.cuda.is_available() and "H200" not in torch.cuda.get_device_name(0).upper():
        batch_size = 48
        total_batch_size = batch_size * world_size
        if rank == 0:
            print("Reduced batch size from 96 to 48 (not on H200 GPUs)")

    # Training config
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None and float(clip_grad_val) > 0 else None
    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 1000))
    max_keep_checkpoints = int(training_cfg.get("max_keep_checkpoints", 0))
    fid_epoch = int(training_cfg.get("fid_epoch", 20))
    fid_sweep_epoch = int(training_cfg.get("fid_sweep_epoch", 100))
    fid_start_epoch = int(training_cfg.get("fid_start_epoch", 0))
    rfid_epoch = int(training_cfg.get("rfid_epoch", 20))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    torch_compile_decoder = bool(training_cfg.get("torch_compile_decoder", False))
    transform_type = int(training_cfg.get("transform_type", 0))
    rrc_scale_min = float(training_cfg.get("rrc_scale_min", 0.8))
    rrc_scale_max = float(training_cfg.get("rrc_scale_max", 1.0))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 50))
    precision = str(training_cfg.get("precision", "fp32"))

    current_best_cfg_scale = 3.4
    fid_num_classes, fid_num_images = 1000, 50000 # this is set for imagenet

    if rank == 0:
        print(f"[Data] transform_type={transform_type} ({get_transform_name(transform_type)})")
        if transform_type == 2:
            print(f"[Data] RandomResizedCrop scale=({rrc_scale_min}, {rrc_scale_max})")
        print(f"[Optimization] precision={precision}")


    # Seed
    seed = int(training_cfg.get("global_seed", 0))
    seed = seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Directories
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_dir = os.path.join(args.results_dir, args.experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log = create_logger(experiment_dir)
        log.info(f"Experiment directory created at {experiment_dir}")
    else:
        experiment_dir = checkpoint_dir = None
        log = create_logger(None)

    checkpoint_dir = checkpoint_dir or os.path.join(args.results_dir, args.experiment_name, "checkpoints")

    # WANDB
    if rank==0:
        wandb.init(
            project="UNITE", 
            group="imagenet", 
            name=args.experiment_name,
            config={
                    "args": vars(args),
                    "yaml_config": full_cfg,
                    "computed": {
                        "lr_bsz_factor": lr_factor,
                        "total_batch_size": total_batch_size,
                        "batch_size": batch_size,
                        "world_size": world_size,
                        "seed": seed,
                        "num_workers": num_workers,
                        "clip_grad": clip_grad,
                    },
                },
        )


    # ===== Model =====
    num_classes = 1000
    model = UNITE(gen_tok_cfg, num_classes=num_classes).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if torch_compile_decoder:
        # Use "default" mode instead of "reduce-overhead" to avoid CUDAGraph
        # buffer conflicts with gradient accumulation (multiple fwd before bwd).
        compile_mode = "reduce-overhead" if grad_accum_steps <= 1 else "default"
        model.decoder = torch.compile(model.decoder, mode=compile_mode)

    ema_model = deepcopy(model).to(device).eval()
    ema_model.requires_grad_(False)

    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)
    model_raw = ddp_model.module

    if rank == 0:
        n_enc = sum(p.numel() for p in model_raw.encoder.parameters() if p.requires_grad)
        n_dec = sum(p.numel() for p in model_raw.decoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model_raw.parameters() if p.requires_grad)
        log.info(
            "Param counts (trainable): "
            + ", ".join([
                f"encoder={n_enc / 1e6:.2f}M",
                f"decoder={n_dec / 1e6:.2f}M",
                f"total={n_total / 1e6:.2f}M",
            ])
        )
        wandb_utils.log({"params/encoder": n_enc, "params/decoder": n_dec, "params/total": n_total}, step=0)



    # ===== Optimizer =====
    optimizer = {}
    non_muon = model_raw.get_learnable_params(non_muon=True)
    muon = model_raw.get_learnable_params(muon=True)

    optim_msgs = []
    optimizer["adamw"], msg = build_optimizer(non_muon, training_cfg, prefix="adamw_")
    optim_msgs.append(msg)
    optimizer["muon"], msg = build_optimizer_muon(muon, training_cfg, prefix="muon_")
    optim_msgs.append(msg)
    optim_msg = " | ".join(optim_msgs) if optim_msgs else "No optimizer"

    # ===== Scaler =====
    if precision == "bf16":
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
        if rank == 0:
            log.info("[Precision] Using BF16 mixed precision (no gradient scaler needed)")
    elif precision == "fp32":
        autocast_kwargs = dict(enabled=False)


    # ===== Dataloader =====
    loader, sampler = prepare_dataloader(
        args.data_path, args.image_size, batch_size, num_workers, rank, world_size,
        transform_type=transform_type, rrc_scale_min=rrc_scale_min, rrc_scale_max=rrc_scale_max,
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned 0 batches.")


    # ===== Scheduler =====
    sched_msgs = []
    scheduler = {}
    scheduler["adamw"], msg = build_scheduler(optimizer["adamw"], steps_per_epoch, training_cfg, prefix="adamw_")
    sched_msgs.append(msg)
    scheduler["muon"], msg = build_scheduler(optimizer["muon"], steps_per_epoch, training_cfg, prefix="muon_")
    sched_msgs.append(msg)
    sched_msg = " | ".join(sched_msgs) if sched_msgs else None


    # ===== Load checkpoint =====
    start_epoch = 0
    global_step = 0

    if args.ckpt:
        if rank == 0:
            log.info(f"[Checkpoint] Loading from {args.ckpt}")
        ckpt_path = Path(args.ckpt)
        if ckpt_path.exists():
            start_epoch, global_step, loaded_cfg = load_checkpoint(
                str(ckpt_path), ddp_model, ema_model, optimizer, scheduler,
            )
            current_best_cfg_scale = loaded_cfg
            if rank == 0:
                log.info(f"[Checkpoint] Resumed: epoch={start_epoch}, step={global_step}, cfg={current_best_cfg_scale:.2f}")
        else:
            if rank == 0:
                raise FileNotFoundError(f"[Checkpoint] File not found: {args.ckpt}")

    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        log.info(f"UNITE trainable parameters: {num_params/1e6:.2f}M")
        if clip_grad is not None:
            log.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            log.info("Not clipping gradients.")
        if optim_msg:
            log.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        log.info(f"LR batch size scaling factor: {lr_factor:.4f} (all LRs scaled by total_batch_size/384.0)")
        log.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
        log.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        log.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    # ===== Training loop =====
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        sampler.set_epoch(epoch)

        is_fid_sweep_epoch = fid_sweep_epoch > 0 and (epoch + 1) % fid_sweep_epoch == 0
        is_fid_normal_epoch = ((epoch + 1) >= fid_start_epoch and fid_epoch > 0 and (epoch + 1) % fid_epoch == 0)
        is_fid_normal_epoch = is_fid_normal_epoch and not is_fid_sweep_epoch
        is_fid_normal_epoch = is_fid_normal_epoch or (fid_start_epoch > 0 and (epoch + 1) == fid_start_epoch)
        is_rfid_epoch = rfid_epoch > 0 and (epoch + 1) % rfid_epoch == 0

        global_step, epoch_metrics, current_best_cfg_scale = train_one_epoch(
            ddp_model, ema_model, ema_decay,
            loader, optimizer, scheduler,
            clip_grad, autocast_kwargs,
            epoch, global_step, device,
            log_interval=log_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            max_keep_checkpoints=max_keep_checkpoints,
            current_best_cfg_scale=current_best_cfg_scale,
            grad_accum_steps=grad_accum_steps,
            is_fid_sweep_epoch=is_fid_sweep_epoch, is_fid_normal_epoch=is_fid_normal_epoch, is_rfid_epoch=is_rfid_epoch,
            image_size=args.image_size, fid_num_classes=fid_num_classes, fid_num_images=fid_num_images, eval_batch_size=eval_batch_size
        )


        if rank == 0:
            nb = epoch_metrics.get("num_batches", 1)
            avg_total = (epoch_metrics.get("total", torch.zeros(1)).item()) / nb
            avg_recon = (epoch_metrics.get("recon", torch.zeros(1)).item()) / nb
            avg_lpips = (epoch_metrics.get("lpips", torch.zeros(1)).item()) / nb
            avg_flow = (epoch_metrics.get("flow", torch.zeros(1)).item()) / nb
            elapsed = time.time() - epoch_start
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_flow": avg_flow,
                "perf/epoch_time_sec": elapsed,
                "perf/epoch_time_min": elapsed / 60.0,
                "perf/epoch_avg_step_sec": elapsed / max(1, nb),
            }
            log.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
            wandb_utils.log(epoch_stats, step=global_step)

    cleanup_distributed()


if __name__ == "__main__":
    main()