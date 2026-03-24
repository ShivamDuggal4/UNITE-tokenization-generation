"""
Optimizer and scheduler construction utilities.
Supports AdamW and Muon optimizers with per-key (encoder/decoder) parameter groups.
Two-stage cosine/linear LR schedule with optional linear warmup.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import inf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Training config helpers
# ---------------------------------------------------------------------------

def scale_learning_rates(training_cfg: Dict[str, Any], lr_factor: float) -> None:
    """Scale training-config LR values in place (v3-compatible behavior)."""
    opt = training_cfg.get("optimizer", {})
    if "lr" in opt:
        opt["lr"] = float(opt["lr"]) * lr_factor
    if "base_lr_1" in training_cfg:
        training_cfg["base_lr_1"] = float(training_cfg["base_lr_1"]) * lr_factor
    sched = training_cfg.get("scheduler", {})
    for key in ("base_lr_1", "base_lr_2", "final_lr"):
        if key in sched:
            sched[key] = float(sched[key]) * lr_factor


# ---------------------------------------------------------------------------
# Optimizer builders
# ---------------------------------------------------------------------------
def _as_tuple(values: Any, length: int = 2) -> tuple:
    """Convert scalar or sequence to a fixed-length tuple of floats."""
    if isinstance(values, (list, tuple)):
        if len(values) != length:
            raise ValueError(f"Expected sequence of length {length}, got {len(values)}.")
        return tuple(float(v) for v in values)
    return tuple(float(values) for _ in range(length))


def build_optimizer(
    parameters: Union[Iterable, Dict[str, list]],
    training_cfg: Dict[str, Any],
    prefix: str = "",
):

    opt_cfg = dict(training_cfg.get("optimizer", {}))
    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer '{opt_type}'. Only AdamW is supported.")

    betas = _as_tuple(opt_cfg.get("betas", opt_cfg.get("beta", (0.9, 0.95))))
    eps = float(opt_cfg.get("eps", 1e-6))

    def _make_adamw(params, lr, wd):
        params = list(params)  # Materialize generator for retry
        try:
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd, eps=eps, fused=True)
        except RuntimeError:
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd, eps=eps)

    base_lr = float(opt_cfg.get(f"{prefix}lr", opt_cfg.get("lr",
                        training_cfg.get(f"{prefix}base_lr_1", 2e-4))))
    wd = float(opt_cfg.get("weight_decay", opt_cfg.get("wd", 0.0)))
    optimizer = _make_adamw(parameters, base_lr, wd)
    msg = f"AdamW lr={base_lr}, betas={betas}, wd={wd}, eps={eps}"
    return optimizer, msg



def build_optimizer_muon(
    parameters: Union[Iterable, Dict[str, list]],
    training_cfg: Dict[str, Any],
    prefix: str = "",
):
    opt_cfg = dict(training_cfg.get("optimizer", {}))
    opt_type = opt_cfg.get("type", "muon").lower()
    if opt_type != "muon":
        raise ValueError(f"Unsupported optimizer '{opt_type}'. Only Muon is supported.")
    momentum = float(opt_cfg.get("momentum", 0.95))
    adjust_lr_fn = opt_cfg.get("muon_adjust_lr_fn", "match_rms_adamw")
    eps = float(opt_cfg.get("eps", 1e-6))
    assert adjust_lr_fn == "match_rms_adamw", f"Expected match_rms_adamw, got {adjust_lr_fn}"
    assert momentum == 0.95, f"Expected momentum=0.95, got {momentum}"

    if not hasattr(torch.optim, "Muon"):
        raise AttributeError(
            "torch.optim.Muon is unavailable in this environment. "
            "For exact v3 anchor behavior, install/use the same Muon-enabled torch build."
        )

    base_lr = float(opt_cfg.get(f"{prefix}lr", opt_cfg.get("lr",
                        training_cfg.get(f"{prefix}base_lr_1", 2e-4))))
    wd = float(opt_cfg.get("muon_weight_decay", opt_cfg.get("muon_wd", 0.0)))
    print("weight_decay_muon: {} | adjust_lr_fn: {}".format(wd, adjust_lr_fn))
    optimizer = torch.optim.Muon(parameters, lr=base_lr, momentum=momentum,
                                    weight_decay=wd, eps=eps,
                                    adjust_lr_fn=adjust_lr_fn)
    msg = f"Optimizer: Muon with lr={base_lr}, momentum={momentum}, weight_decay={wd}, eps={eps}"
    return optimizer, msg



# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: Union[Optimizer, Dict[str, Optimizer]],
    steps_per_epoch: int,
    training_cfg: Dict[str, Any],
    state_dict: Optional[Dict[str, Any]] = None,
    prefix: str = "",
):
    """Two-stage LR schedule with optional linear warmup.

    Schedule stages:
        [0, warmup):           linear 0 -> base_lr_1
        [warmup, warmup_1):    flat at base_lr_1
        [warmup_1, decay_end_1): decay base_lr_1 -> base_lr_2
        [decay_end_1, warmup_2): flat at base_lr_2
        [warmup_2, decay_end_2): decay base_lr_2 -> final_lr
        [decay_end_2, inf):    flat at final_lr

    Supports both single Optimizer and Dict[str, Optimizer] for per-key scheduling.
    """
    cfg = dict(training_cfg.get("scheduler", {}))
    schedule_type = cfg.get("type", "cosine").lower()
    if schedule_type not in {"cosine", "linear"}:
        raise ValueError(f"Unsupported scheduler '{schedule_type}' (use 'linear' or 'cosine').")

    def interp(progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        if schedule_type == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def decay_between(step: int, s0: int, s1: int, lr_start: float, lr_end: float) -> float:
        if s1 <= s0:
            return lr_end
        p = (step - s0) / (s1 - s0)
        return lr_end + (lr_start - lr_end) * interp(p)

    def make_lr_lambda(warmup, w1, e1, w2, e2, base_lr_1, base_lr_2, final_lr):
        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                lr = base_lr_1 * (step / warmup)
            elif step < w1:
                lr = base_lr_1
            elif step < e1:
                lr = decay_between(step, w1, e1, base_lr_1, base_lr_2)
            elif step < w2:
                lr = base_lr_2
            elif step < e2:
                lr = decay_between(step, w2, e2, base_lr_2, final_lr)
            else:
                lr = final_lr
            return lr / base_lr_1
        return lr_lambda

    def to_steps(key: Optional[str], stage: str) -> int:
        # Priority: key_{stage}_steps > {stage}_steps > key_{stage}_epochs > {stage}_epochs
        if key is not None:
            ks = f"{key}_{stage}_steps"
            if ks in cfg:
                return max(int(cfg[ks]), 0)
        gs = f"{stage}_steps"
        if gs in cfg:
            return max(int(cfg[gs]), 0)
        if key is not None:
            ke = f"{key}_{stage}_epochs"
            if ke in cfg:
                return max(int(round(float(cfg[ke]) * steps_per_epoch)), 0)
        ge = f"{stage}_epochs"
        return max(int(round(float(cfg.get(ge, 0.0)) * steps_per_epoch)), 0)

    def get_lrs(key: Optional[str], opt: Optimizer):
        def pick(name: str, default: float) -> float:
            if key is not None:
                for k in [f"{key}_{prefix}{name}", f"{key}_{name}"]:
                    if k in cfg:
                        return float(cfg[k])
            for k in [f"{prefix}{name}", name]:
                if k in cfg:
                    return float(cfg[k])
            return default
        current = float(opt.param_groups[0]["lr"])
        base_lr_1 = pick("base_lr_1", current)
        base_lr_2 = pick("base_lr_2", base_lr_1)
        final_lr = pick("final_lr", base_lr_2)
        return base_lr_1, base_lr_2, final_lr

    def build_one(key, opt, sd):
        base_lr_1, base_lr_2, final_lr = get_lrs(key, opt)
        warmup = to_steps(key, "warmup")
        w1 = to_steps(key, "warmup_1")
        e1 = max(to_steps(key, "decay_end_1"), w1)
        w2 = max(to_steps(key, "warmup_2"), e1)
        e2 = max(to_steps(key, "decay_end_2"), w2)

        for g in opt.param_groups:
            g["lr"] = base_lr_1

        sched = LambdaLR(opt, lr_lambda=make_lr_lambda(warmup, w1, e1, w2, e2,
                                                         base_lr_1, base_lr_2, final_lr))
        if sd is not None:
            sched.load_state_dict(sd)

        warmup_str = f"warmup={warmup}, " if warmup > 0 else ""
        msg = f"{key or 'single'}: {warmup_str}w1={w1}, e1={e1}, w2={w2}, e2={e2}, " \
              f"base1={base_lr_1}, base2={base_lr_2}, final={final_lr}"
        return sched, msg

    # Dispatch
    if isinstance(optimizer, dict):
        schedulers = {}
        msgs = []
        for k, opt in optimizer.items():
            sd_k = None
            if state_dict is not None and isinstance(state_dict, dict) and k in state_dict:
                sd_k = state_dict[k] if isinstance(state_dict[k], dict) else None
            sched, msg_k = build_one(k, opt, sd_k)
            schedulers[k] = sched
            msgs.append(msg_k)
        return schedulers, f"Two-stage {schedule_type} LR (per-key) | " + " | ".join(msgs)

    sd = state_dict if isinstance(state_dict, dict) else None
    sched, msg = build_one(None, optimizer, sd)
    return sched, f"Two-stage {schedule_type} LR | {msg}"


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def get_grad_norm_(parameters, norm_type: float = 2.0, device=None) -> torch.Tensor:
    """Compute gradient norm across all parameters with gradients."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(0.0, device=device)
    if device is None:
        device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                         for p in parameters]),
            norm_type,
        )
    return total_norm
