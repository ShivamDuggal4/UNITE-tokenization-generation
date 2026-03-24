
import torch
import torch.nn as nn
import torch.distributed as dist


@torch.no_grad()
def update_ema(
    ema_model,
    current_model,
    decay,
) -> None:
    """Update EMA weights from current model.
    Args:
        decay: Default EMA decay rate.
    """
    ema_params = dict(ema_model.named_parameters())
    for name, param in current_model.named_parameters():
        if name not in ema_params:
            continue
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def copy_bn_buffers(
    src_model,
    dst_model,
    *,
    broadcast_from_rank0=False,
    src_rank=0,
) -> None:
    """Copy BatchNorm running stats from src to dst, with optional distributed broadcast."""
    def _is_bn(m):
        return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))

    src_modules = dict(src_model.named_modules())
    for name, m_dst in dst_model.named_modules():
        m_src = src_modules.get(name)
        if m_src is None or not _is_bn(m_src) or not _is_bn(m_dst):
            continue
        m_dst.running_mean.copy_(m_src.running_mean)
        m_dst.running_var.copy_(m_src.running_var)
        if hasattr(m_src, "num_batches_tracked") and hasattr(m_dst, "num_batches_tracked"):
            m_dst.num_batches_tracked.copy_(m_src.num_batches_tracked)

    if broadcast_from_rank0 and dist.is_initialized():
        for _name, m_dst in dst_model.named_modules():
            if _is_bn(m_dst):
                dist.broadcast(m_dst.running_mean, src=src_rank)
                dist.broadcast(m_dst.running_var, src=src_rank)
                if hasattr(m_dst, "num_batches_tracked"):
                    dist.broadcast(m_dst.num_batches_tracked, src=src_rank)
