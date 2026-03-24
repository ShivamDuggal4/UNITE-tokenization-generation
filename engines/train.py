import logging
from collections import defaultdict
from contextlib import nullcontext
import torch
from torch.amp import autocast
from utils import wandb_utils
from utils.checkpoint import save_checkpoint, prune_old_checkpoints
from utils.ema import update_ema, copy_bn_buffers
from utils.distributed import is_main_process
from engines.eval import on_epoch_end_fid, on_epoch_end_rfid

logger = logging.getLogger(__name__)

def train_one_epoch(
    model, ema, ema_decay, 
    loader, optimizer, scheduler, clip_grad, autocast_kwargs,
    epoch, global_step, device,
    log_interval = 100,
    checkpoint_dir = "",
    checkpoint_interval = 1000,
    max_keep_checkpoints = 0,
    current_best_cfg_scale = 3.4,
    grad_accum_steps = 1,
    # Epoch-end evaluation hooks (callables, called if not None)
    is_fid_sweep_epoch=False,
    is_fid_normal_epoch=False,
    is_rfid_epoch=False,
    image_size=None, fid_num_classes=1000, fid_num_images=50000, eval_batch_size=None
):
    """
    Run one epoch of training.
    Returns: (global_step, epoch_metrics, current_best_cfg_scale)
    Epoch-end hooks:
        on_epoch_end_fid for gFID evaluation
        on_epoch_end_rfid for rFID evaluation
    """
    model.train()
    ema.eval()
    num_batches = 0
    steps_per_epoch = len(loader)
    epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
    raw_model = model.module if hasattr(model, "module") else model

    # ===== Per-batch loop =====
    # Gradient accumulation: accumulate over grad_accum_steps micro-batches,
    # then clip/step/EMA. DDP sync is skipped during accumulation via no_sync().
    use_no_sync = hasattr(model, "no_sync")  # DDP wrapper

    for step, (images, labels) in enumerate(loader):
        is_accum_step = (step + 1) % grad_accum_steps != 0  # True = accumulating, skip optimizer step

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if step % grad_accum_steps == 0:
            for optim_i in optimizer.values(): optim_i.zero_grad()

        # For accumulation: skip DDP all-reduce on intermediate micro-batches
        ctx = model.no_sync() if (use_no_sync and is_accum_step) else nullcontext()

        with ctx:
            # Forward
            with autocast("cuda", **autocast_kwargs):
                total_loss, output_dict = model(images, labels=labels)
                if grad_accum_steps > 1:
                    total_loss = total_loss / grad_accum_steps

            total_loss.backward()

        # Accumulate epoch metrics (every micro-batch)
        epoch_metrics["recon"] += output_dict["rec_loss"].detach()
        epoch_metrics["lpips"] += output_dict["lpips_loss"].detach()
        epoch_metrics["total"] += total_loss.detach()
        epoch_metrics["flow"] += output_dict["flow/flow_loss"].detach()
        num_batches += 1
        epoch_metrics["num_batches"] = num_batches

        # Only step optimizer after accumulating grad_accum_steps micro-batches
        if is_accum_step:
            continue

        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        
        for optim_i in optimizer.values(): optim_i.step()
        if scheduler is not None:
            for sched_i in scheduler.values(): sched_i.step()

        # EMA update
        update_ema(ema, raw_model, ema_decay)
        copy_bn_buffers(raw_model, ema, broadcast_from_rank0=True)
        
        # Logging
        if log_interval > 0 and global_step % log_interval == 0 and is_main_process():
            stats = {
                "loss/total": total_loss.detach().item(),
                "loss/recon": output_dict["rec_loss"].detach().item(),
                "loss/lpips": output_dict["lpips_loss"].detach().item(),
                "epoch": epoch + (step / steps_per_epoch),
                "loss/flow-flow": output_dict["flow/flow_loss"].detach().item()
            }
            logger.info(
                f"[Epoch {epoch} | Step {global_step}] "
                + ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in stats.items())
            )
            wandb_utils.log(stats, step=global_step)


        if checkpoint_interval > 0 and global_step % checkpoint_interval == 0 and is_main_process():
            ckpt_path = f"{checkpoint_dir}/{global_step:07d}.pt"
            save_checkpoint(
                ckpt_path, global_step, epoch, model, ema,
                optimizer, scheduler,
                current_best_cfg_scale=current_best_cfg_scale,
            )
            logger.info(f"[Checkpoint] Saved: {ckpt_path}")
            if max_keep_checkpoints > 0:
                prune_old_checkpoints(checkpoint_dir, max_keep_checkpoints)

        global_step += 1

    if epoch==0 or is_fid_sweep_epoch or is_fid_normal_epoch:
        current_best_cfg_scale = on_epoch_end_fid(
            ema, ema_decay,
            device, global_step, epoch, 
            current_best_cfg_scale, is_fid_sweep_epoch, is_fid_normal_epoch,
            image_size, fid_num_classes, fid_num_images, eval_batch_size
        )

    if is_rfid_epoch:
        on_epoch_end_rfid(ema, raw_model, device, global_step, epoch)

    return global_step, dict(epoch_metrics), current_best_cfg_scale
