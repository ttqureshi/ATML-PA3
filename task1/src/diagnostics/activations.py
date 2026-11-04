from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.diagnostics.weight_space import extract_delta_and_base, get_linear_module


def _tokenize_prompt(tokenizer, text: str, use_chat: bool) -> torch.Tensor:
    if use_chat:
        messages = [{"role": "user", "content": text}]
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        # Newer transformers may return str / BatchEncoding instead of a Tensor.
        if isinstance(ids, str):
            ids = tokenizer(ids, return_tensors="pt").input_ids
        elif hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif not torch.is_tensor(ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def cache_down_proj_inputs(
    base_model,
    tokenizer,
    prompts: list[dict[str, Any]],
    layers: list[int],
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    """Cache base-model down_proj inputs z at last non-padding prompt token."""
    caches: dict[str, dict[int, torch.Tensor]] = {}
    handles = []
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_id: int):
        def hook(_module, inputs, _output):
            # inputs[0]: (batch, seq, hidden)
            captured[layer_id] = inputs[0].detach()

        return hook

    for layer in layers:
        mod = get_linear_module(base_model, f"layers.{layer}.mlp.down_proj")
        handles.append(mod.register_forward_hook(make_hook(layer)))

    base_model.eval()
    for item in prompts:
        key = item["key"]
        ids = _tokenize_prompt(tokenizer, item["text"], use_chat=item["use_chat"]).to(device)
        captured.clear()
        _ = base_model(input_ids=ids)
        # last non-padding token index
        last = ids.size(1) - 1
        layer_map = {}
        for layer in layers:
            z = captured[layer][0, last, :].float().cpu()
            layer_map[layer] = z
        caches[key] = layer_map

    for h in handles:
        h.remove()
    return caches


def activation_correction_ratio(
    delta_w: torch.Tensor,
    w0: torch.Tensor,
    z: torch.Tensor,
) -> float:
    # R(z) = ||ΔW z|| / (||W0 z|| + 1e-8)
    z = z.float()
    dw = delta_w.float()
    w = w0.float()
    num = torch.linalg.norm(dw @ z)
    den = torch.linalg.norm(w @ z) + 1e-8
    return float((num / den).item())


def measure_activation_corrections(
    *,
    base_model,
    trained_model,
    method: str,
    caches: dict[str, dict[int, torch.Tensor]],
    prompt_meta: list[dict[str, Any]],
    layers: list[int],
) -> list[dict[str, Any]]:
    rows = []
    key_to_domain = {p["key"]: p["domain"] for p in prompt_meta}
    for layer in layers:
        base_mod = get_linear_module(base_model, f"layers.{layer}.mlp.down_proj")
        tr_mod = get_linear_module(trained_model, f"layers.{layer}.mlp.down_proj")
        delta, w0 = extract_delta_and_base(tr_mod, base_mod, method)
        for key, layer_map in caches.items():
            z = layer_map[layer]
            rows.append(
                {
                    "layer": layer,
                    "key": key,
                    "domain": key_to_domain[key],
                    "R": activation_correction_ratio(delta, w0, z),
                }
            )
    return rows
