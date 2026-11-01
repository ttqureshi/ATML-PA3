from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_param_groups(model, weight_decay: float = 0.01):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude biases and parameters with fewer than two dimensions.
        if param.ndim < 2 or name.endswith(".bias") or name.split(".")[-1] == "bias":
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model, lr: float, weight_decay: float = 0.01, betas=(0.9, 0.999), eps=1e-8):
    groups = build_param_groups(model, weight_decay=weight_decay)
    return AdamW(groups, lr=lr, betas=betas, eps=eps)


def build_cosine_scheduler(optimizer, num_updates: int, warmup_updates: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < warmup_updates:
            return float(current_step) / float(max(1, warmup_updates))
        progress = float(current_step - warmup_updates) / float(
            max(1, num_updates - warmup_updates)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def collect_starting_point(model) -> dict[str, torch.Tensor]:
    """Detach CPU copies of parameters for L2-SP (same dtype as model; keeps RAM down)."""
    ref = {}
    for name, param in model.named_parameters():
        # Keep model dtype (FP16 for full FT) — casting everything to FP32 doubles RAM.
        ref[name] = param.detach().cpu().clone()
    return ref


def l2_sp_penalty(model, theta0: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Ω = 1/2 Σ_j (θ_j - θ0_j)^2 over all unique parameters. Not averaged."""
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in model.named_parameters():
        if name not in theta0:
            continue
        # One parameter at a time to avoid materializing a full second model on GPU.
        ref = theta0[name].to(device=device, dtype=param.dtype, non_blocking=True)
        diff = param - ref
        total = total + 0.5 * torch.sum(diff * diff)
        del ref
    return total


def full_model_displacement(model, theta0: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name not in theta0:
            continue
        diff = param.detach().float().cpu() - theta0[name]
        total += float(torch.sum(diff * diff).item())
    return float(math.sqrt(total))
