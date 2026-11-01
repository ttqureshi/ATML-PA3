from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.oplora import OPLoRALinear


def leading_bases(weight: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().float().cpu()
    U, _, Vh = torch.linalg.svd(w, full_matrices=False)
    k_eff = min(k, U.size(1), Vh.size(0))
    Uk = U[:, :k_eff]
    Vk = Vh[:k_eff, :].transpose(0, 1)
    return Uk, Vk


def subspace_interference_ratio(delta_w: torch.Tensor, Uk: torch.Tensor) -> float:
    d = delta_w.detach().float().cpu()
    proj = Uk @ (Uk.transpose(0, 1) @ d)
    num = torch.linalg.norm(proj, ord="fro") ** 2
    den = torch.linalg.norm(d, ord="fro") ** 2
    if den.item() == 0:
        return float("nan")
    return float((num / den).item())


def relative_update_magnitude(delta_w: torch.Tensor, w0: torch.Tensor) -> float:
    num = torch.linalg.norm(delta_w.detach().float().cpu(), ord="fro")
    den = torch.linalg.norm(w0.detach().float().cpu(), ord="fro")
    return float((num / den).item())


def get_linear_module(model, dotted_suffix: str) -> nn.Linear | OPLoRALinear:
    for name, module in model.named_modules():
        if name.endswith(dotted_suffix):
            return module
    raise KeyError(dotted_suffix)


def extract_delta_and_base(
    trained_module: nn.Module,
    base_module: nn.Module,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    w0 = base_module.weight.detach()
    if method in {"full_ft", "full_ft_l2sp", "baseline"}:
        delta = trained_module.weight.detach() - w0.to(trained_module.weight.device)
        return delta.cpu(), w0.cpu()
    if isinstance(trained_module, OPLoRALinear):
        return trained_module.delta_weight().detach().cpu(), w0.cpu()
    # PEFT LoRA linear: lora_B @ lora_A * scaling
    if hasattr(trained_module, "lora_A") and hasattr(trained_module, "lora_B"):
        # peft stores lora_A/B as ModuleDict keyed by adapter name
        a = trained_module.lora_A["default"].weight.detach().float()
        b = trained_module.lora_B["default"].weight.detach().float()
        scaling = trained_module.scaling["default"]
        delta = (scaling * (b @ a)).cpu()
        return delta, w0.cpu()
    # Merged model: compare to base
    delta = trained_module.weight.detach() - w0.to(trained_module.weight.device)
    return delta.cpu(), w0.cpu()


def analyze_down_proj_layers(
    *,
    base_model,
    trained_model,
    method: str,
    layers: list[int],
    k: int = 128,
) -> list[dict[str, Any]]:
    rows = []
    for layer in layers:
        suffix = f"layers.{layer}.mlp.down_proj"
        base_mod = get_linear_module(base_model, suffix)
        tr_mod = get_linear_module(trained_model, suffix)
        delta, w0 = extract_delta_and_base(tr_mod, base_mod, method)
        Uk, _ = leading_bases(w0, k)
        rows.append(
            {
                "layer": layer,
                "module": suffix,
                "rho_k": subspace_interference_ratio(delta, Uk),
                "eta": relative_update_magnitude(delta, w0),
                "delta_fro": float(torch.linalg.norm(delta.float(), ord="fro").item()),
                "w0_fro": float(torch.linalg.norm(w0.float(), ord="fro").item()),
                "k": int(Uk.size(1)),
            }
        )
    return rows
