from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .load_qwen import count_trainable_parameters, enable_input_require_grads, freeze_backbone
from .lora import DEFAULT_TARGETS


@dataclass
class OPLoRABasisInfo:
    module_name: str
    k: int
    preprocess_seconds: float
    u_bytes: int
    v_bytes: int


class OPLoRALinear(nn.Module):
    """Linear layer with orthogonally protected low-rank adapters.

    Trainable factors A, B are unconstrained. The effective factors used in the
    forward/export are:
        B_tilde = B - Uk (Uk^T B)
        A_tilde = A - (A Vk) Vk^T
        delta W = (alpha / r) B_tilde A_tilde
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int,
        k: int,
        dropout: float = 0.05,
        Uk: torch.Tensor | None = None,
        Vk: torch.Tensor | None = None,
    ):
        super().__init__()
        if base.weight.ndim != 2:
            raise ValueError("OPLoRALinear expects a 2D weight")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.k = k
        self.scaling = alpha / r

        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # A: r x in, B: out x r  (matches assignment notation)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if Uk is None or Vk is None:
            Uk, Vk = self._compute_bases(self.weight.detach().float(), k)
        # Freeze bases
        self.register_buffer("Uk", Uk.contiguous(), persistent=True)
        self.register_buffer("Vk", Vk.contiguous(), persistent=True)

    @staticmethod
    def _compute_bases(weight_fp32: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # W0 = U Σ V^T with W shaped (out, in)
        U, _, Vh = torch.linalg.svd(weight_fp32, full_matrices=False)
        k_eff = min(k, U.size(1), Vh.size(0))
        Uk = U[:, :k_eff].contiguous()
        Vk = Vh[:k_eff, :].transpose(0, 1).contiguous()  # (in, k)
        return Uk, Vk

    def projected_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        # B_tilde = B - Uk (Uk^T B)
        B = self.lora_B
        A = self.lora_A
        Uk = self.Uk
        Vk = self.Vk
        B_tilde = B - Uk @ (Uk.transpose(0, 1) @ B)
        A_tilde = A - (A @ Vk) @ Vk.transpose(0, 1)
        return A_tilde, B_tilde

    def delta_weight(self) -> torch.Tensor:
        A_tilde, B_tilde = self.projected_factors()
        return (self.scaling * (B_tilde @ A_tilde)).to(dtype=self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        A_tilde, B_tilde = self.projected_factors()
        x_d = self.dropout(x)
        # Compute in FP32 for adapter path, cast back
        x32 = x_d.float()
        update = (x32 @ A_tilde.transpose(0, 1)) @ B_tilde.transpose(0, 1)
        update = (self.scaling * update).to(dtype=base_out.dtype)
        return base_out + update

    def merge_into_weight_(self) -> None:
        with torch.no_grad():
            self.weight.data = self.weight.data + self.delta_weight()
            self.lora_A.zero_()
            self.lora_B.zero_()


def _replace_module(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def attach_oplora(
    model,
    r: int = 16,
    alpha: int = 32,
    k: int = 128,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    freeze_backbone(model)
    enable_input_require_grads(model)
    targets = set(target_modules or DEFAULT_TARGETS)

    basis_infos: list[OPLoRABasisInfo] = []
    replaced = 0
    t0 = time.perf_counter()

    # Materialize named modules first to avoid mutation during iteration issues
    linear_targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            linear_targets.append((name, module))

    for name, module in linear_targets:
        w = module.weight.detach().float().cpu()
        t_mod = time.perf_counter()
        Uk, Vk = OPLoRALinear._compute_bases(w, k)
        preprocess_s = time.perf_counter() - t_mod
        new_mod = OPLoRALinear(
            base=module,
            r=r,
            alpha=alpha,
            k=min(k, Uk.size(1)),
            dropout=dropout,
            Uk=Uk,
            Vk=Vk,
        )
        # Keep adapters on same device as base weight
        device = module.weight.device
        new_mod.to(device)
        _replace_module(model, name, new_mod)
        basis_infos.append(
            OPLoRABasisInfo(
                module_name=name,
                k=int(new_mod.Uk.size(1)),
                preprocess_seconds=preprocess_s,
                u_bytes=int(new_mod.Uk.numel() * new_mod.Uk.element_size()),
                v_bytes=int(new_mod.Vk.numel() * new_mod.Vk.element_size()),
            )
        )
        replaced += 1

    total_preprocess = time.perf_counter() - t0
    meta = {
        "replaced_modules": replaced,
        "r": r,
        "alpha": alpha,
        "k": k,
        "dropout": dropout,
        "target_modules": sorted(targets),
        "basis_preprocess_seconds_total": total_preprocess,
        "basis_storage_bytes": int(sum(b.u_bytes + b.v_bytes for b in basis_infos)),
        "basis_infos": [b.__dict__ for b in basis_infos],
    }
    return model, meta


def export_projected_adapter_state(model) -> dict[str, torch.Tensor]:
    """Export projected A_tilde/B_tilde factors for inference/diagnostics."""
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, OPLoRALinear):
            A_tilde, B_tilde = module.projected_factors()
            state[f"{name}.lora_A"] = A_tilde.detach().cpu()
            state[f"{name}.lora_B"] = B_tilde.detach().cpu()
            state[f"{name}.scaling"] = torch.tensor(module.scaling)
            state[f"{name}.Uk"] = module.Uk.detach().cpu()
            state[f"{name}.Vk"] = module.Vk.detach().cpu()
    return state


def merge_oplora_into_fresh_backbone(model, model_name: str, torch_dtype=torch.float16):
    """Create a fresh backbone copy and bake projected ΔW into weights."""
    deltas: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, OPLoRALinear):
            deltas[name] = module.delta_weight().detach().cpu()

    fresh = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    name_to_module = dict(fresh.named_modules())
    with torch.no_grad():
        for name, delta in deltas.items():
            mod = name_to_module[name]
            mod.weight.data = mod.weight.data + delta.to(
                device=mod.weight.device, dtype=mod.weight.dtype
            )
    return fresh


def verify_oplora_constraints(model, atol: float = 1e-4) -> dict[str, Any]:
    """Numerically verify Uk^T B_tilde ≈ 0 and A_tilde Vk ≈ 0."""
    report = {"modules_checked": 0, "max_abs_U_violation": 0.0, "max_abs_V_violation": 0.0}
    for module in model.modules():
        if not isinstance(module, OPLoRALinear):
            continue
        A_t, B_t = module.projected_factors()
        u_viol = (module.Uk.transpose(0, 1) @ B_t).abs().max().item()
        v_viol = (A_t @ module.Vk).abs().max().item()
        report["modules_checked"] += 1
        report["max_abs_U_violation"] = max(report["max_abs_U_violation"], u_viol)
        report["max_abs_V_violation"] = max(report["max_abs_V_violation"], v_viol)
    report["ok"] = (
        report["max_abs_U_violation"] < atol and report["max_abs_V_violation"] < atol
    )
    return report


def oplora_parameter_report(model) -> dict[str, int]:
    report = count_trainable_parameters(model)
    adapter = 0
    for module in model.modules():
        if isinstance(module, OPLoRALinear):
            adapter += module.lora_A.numel() + module.lora_B.numel()
    report["adapter"] = int(adapter)
    return report
