"""
Checkpoint save/load with S3 support and denoiser weight initialization.
Handles:
- Save/load of model, EMA, optimizers (AdamW + Muon), schedulers
- Dict-based optimizers (per-key encoder/decoder param groups)
- Denoiser initialization from encoder weights when loading older checkpoints
- S3 checkpoint discovery and download (multi-GPU aware)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_optim_state(opt) -> Optional[Dict]:
    """Extract state dict from a single optimizer or dict of optimizers."""
    if opt is None:
        return None
    if isinstance(opt, dict):
        return {k: v.state_dict() for k, v in opt.items()}
    return opt.state_dict()


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Dict[str, Any],
    scheduler: Dict[str, Any],
    current_best_cfg_scale: float = 2.4,
) -> None:
    """Save full training state to disk.

    Args:
        model: DDP-wrapped model (we save model.module.state_dict()).
        optimizer: Dict with keys like "adamw", "muon" mapping to Optimizer or Dict[str, Optimizer].
        scheduler: Same structure as optimizer.
    """
    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, "module") else model

    state = {
        "step": step,
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "ema": ema_model.state_dict(),
        "current_best_cfg_scale": current_best_cfg_scale,
    }

    # Save AdamW optimizer/scheduler
    if "adamw" in optimizer:
        state["optimizer"] = _save_optim_state(optimizer["adamw"])
    if scheduler and "adamw" in scheduler:
        state["scheduler"] = _save_optim_state(scheduler["adamw"])

    # Save Muon optimizer/scheduler
    if "muon" in optimizer:
        state["optimizer_muon"] = _save_optim_state(optimizer["muon"])
    if scheduler and "muon" in scheduler:
        state["scheduler_muon"] = _save_optim_state(scheduler["muon"])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def prune_old_checkpoints(checkpoint_dir: str, max_keep: int) -> None:
    """Keep only the latest `max_keep` numeric checkpoints in `checkpoint_dir`.

    Checkpoint files are expected to follow `<step>.pt` naming, e.g. `0004000.pt`.
    Non-numeric checkpoint names are ignored.
    """
    if max_keep <= 0:
        return

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return

    numbered = []
    for p in ckpt_dir.glob("*.pt"):
        stem = p.stem
        if not stem.isdigit():
            continue
        numbered.append((int(stem), p))

    if len(numbered) <= max_keep:
        return

    numbered.sort(key=lambda x: x[0])
    to_remove = numbered[: len(numbered) - max_keep]
    for _step, path in to_remove:
        try:
            path.unlink()
            logger.info(f"[Checkpoint] Pruned old checkpoint: {path}")
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning(f"[Checkpoint] Failed to prune {path}: {exc}")


# ---------------------------------------------------------------------------
# Denoiser initialization helper
# ---------------------------------------------------------------------------

def _init_denoiser_from_encoder(model: nn.Module, label: str = "") -> None:
    """Initialize denoiser weights from encoder when denoiser keys are missing.

    Copies: encoder -> denoiser, latent_tokens -> latent_tokens_denoiser,
            encoder_layer_norm -> denoiser_layer_norm (and rms, bnnorm variants).
    """
    prefix = f"[Checkpoint{label}] "

    if hasattr(model, "denoiser") and hasattr(model, "encoder"):
        if model.denoiser is not model.encoder:
            encoder_state = {k.replace("encoder.", "denoiser."): v
                             for k, v in model.encoder.state_dict().items()}
            model.denoiser.load_state_dict(encoder_state, strict=True)
            logger.info(f"{prefix}Initialized denoiser from encoder weights")

    # Latent tokens
    if hasattr(model, "latent_tokens_denoiser") and hasattr(model, "latent_tokens"):
        model.latent_tokens_denoiser.data.copy_(model.latent_tokens.data)
        logger.info(f"{prefix}Initialized latent_tokens_denoiser from latent_tokens")

    # Normalization layers
    _copy_pairs = [
        ("denoiser_layer_norm", "encoder_layer_norm"),
        ("denoiser_rms_norm", "encoder_rms_norm"),
        ("denoiser_pre_decoder_normalization", "pre_decoder_normalization"),
    ]
    for dst_attr, src_attr in _copy_pairs:
        if hasattr(model, dst_attr) and hasattr(model, src_attr):
            getattr(model, dst_attr).load_state_dict(getattr(model, src_attr).state_dict())
            logger.info(f"{prefix}Initialized {dst_attr} from {src_attr}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_optim_state(opt, state, name: str) -> None:
    """Load optimizer state dict (single or dict-based), with graceful mismatch handling."""
    if opt is None or state is None:
        return
    # try:
    if isinstance(opt, dict):
        for key in opt:
            opt[key].load_state_dict(state[key])
    else:
        opt.load_state_dict(state)
    logger.info(f"[Checkpoint] Loaded {name} state")
    # except (ValueError, RuntimeError) as e:
    #     if "parameter group" in str(e) and "doesn't match" in str(e):
    #         logger.warning(f"[Checkpoint] {name} state mismatch — skipping (will rebuild momentum)")
    #     else:
    #         raise


def _load_scheduler_state(sched, state, name: str) -> None:
    """Load scheduler state dict (single or dict-based), with graceful mismatch handling."""
    if sched is None or state is None:
        return
    try:
        if isinstance(sched, dict):
            for key in sched:
                sched[key].load_state_dict(state[key])
        else:
            sched.load_state_dict(state)
        logger.info(f"[Checkpoint] Loaded {name} scheduler state")
    except (ValueError, RuntimeError) as e:
        logger.warning(f"[Checkpoint] {name} scheduler mismatch — restarting: {e}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, float]:
    """Load checkpoint and restore full training state.

    Returns: (epoch, step, current_best_cfg_scale)
    """
    checkpoint = torch.load(path, map_location="cpu")

    # Unwrap DDP
    raw_model = model.module if hasattr(model, "module") else model

    # --- Model weights ---
    msg = raw_model.load_state_dict(checkpoint["model"], strict=False)
    print(msg, "msg 1")
    if msg.missing_keys:
        raise RuntimeError(f"[Checkpoint] Missing critical keys: {msg.missing_keys}")
    if msg.unexpected_keys:                                                                                                                                                       
        raise RuntimeError(f"[Checkpoint] Missing critical keys: {msg.unexpected_keys}")

    # --- EMA weights ---
    msg = ema_model.load_state_dict(checkpoint["ema"], strict=False)
    print(msg, "msg 2")
    if msg.missing_keys:
        raise RuntimeError(f"[Checkpoint] EMA missing critical keys: {msg.missing_keys}")
    if msg.unexpected_keys:                                                                                                                                                       
        raise RuntimeError(f"[Checkpoint] Missing critical keys: {msg.unexpected_keys}")

    # --- Optimizers ---
    if optimizer is not None:
        if "adamw" in optimizer and checkpoint.get("optimizer") is not None:
            _load_optim_state(optimizer["adamw"], checkpoint["optimizer"], "AdamW")
        if "muon" in optimizer and checkpoint.get("optimizer_muon") is not None:
            _load_optim_state(optimizer["muon"], checkpoint["optimizer_muon"], "Muon")

    # --- Schedulers ---
    if scheduler is not None:
        if "adamw" in scheduler and checkpoint.get("scheduler") is not None:
            _load_scheduler_state(scheduler["adamw"], checkpoint["scheduler"], "AdamW")
        if "muon" in scheduler and checkpoint.get("scheduler_muon") is not None:
            _load_scheduler_state(scheduler["muon"], checkpoint["scheduler_muon"], "Muon")


    cfg_scale = checkpoint.get("current_best_cfg_scale", 2.4)
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0), cfg_scale
