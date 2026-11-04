from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.models.load_qwen import load_base_model, load_tokenizer
from src.models.lora import _neutralize_torchao_check
from src.models.oplora import OPLoRALinear, attach_oplora, merge_oplora_into_fresh_backbone

_neutralize_torchao_check()


def load_eval_model(
    *,
    model_name: str,
    method: str,
    checkpoint_dir: Path | None,
    device: torch.device,
    merge_adapters: bool = True,
) -> tuple[Any, Any]:
    """Load a model for evaluation in FP16 with dropout disabled."""
    tokenizer = load_tokenizer(model_name)

    if checkpoint_dir is None or method == "baseline":
        model = load_base_model(
            model_name,
            torch_dtype=torch.float16,
            gradient_checkpointing=False,
            device=device,
        )
        model.eval()
        return model, tokenizer

    model_dir = checkpoint_dir / "model"
    if method in {"full_ft", "full_ft_l2sp"}:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
    elif method == "lora":
        base = load_base_model(
            model_name, torch_dtype=torch.float16, gradient_checkpointing=False
        )
        peft_model = PeftModel.from_pretrained(base, model_dir)
        if merge_adapters:
            model = peft_model.merge_and_unload()
        else:
            model = peft_model
        model = model.to(device)
    elif method == "oplora":
        projected_path = model_dir / "oplora_projected_factors.pt"
        adapters_path = model_dir / "oplora_adapters.pt"
        legacy_path = model_dir / "oplora_state.pt"
        if merge_adapters and projected_path.exists():
            model = _merge_from_projected_factors(
                model_name, projected_path, device=device
            )
        else:
            model = load_base_model(
                model_name, torch_dtype=torch.float16, gradient_checkpointing=False
            )
            model, _ = attach_oplora(model)
            if adapters_path.exists():
                blob = torch.load(adapters_path, map_location="cpu", weights_only=False)
                name_to_mod = dict(model.named_modules())
                for key, tensor in blob.items():
                    if key.endswith(".lora_A"):
                        name_to_mod[key[: -len(".lora_A")]].lora_A.data.copy_(tensor)
                    elif key.endswith(".lora_B"):
                        name_to_mod[key[: -len(".lora_B")]].lora_B.data.copy_(tensor)
            elif legacy_path.exists():
                state = torch.load(legacy_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state, strict=False)
            if merge_adapters:
                model = merge_oplora_into_fresh_backbone(model, model_name)
            model = model.to(device)
    else:
        raise ValueError(method)

    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return model, tokenizer


def _merge_from_projected_factors(model_name: str, projected_path: Path, device):
    projected = torch.load(projected_path, map_location="cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    # Keys like "{module}.lora_A", "{module}.lora_B", "{module}.scaling"
    modules = {}
    for key, tensor in projected.items():
        if key.endswith(".lora_A"):
            mod = key[: -len(".lora_A")]
            modules.setdefault(mod, {})["A"] = tensor
        elif key.endswith(".lora_B"):
            mod = key[: -len(".lora_B")]
            modules.setdefault(mod, {})["B"] = tensor
        elif key.endswith(".scaling"):
            mod = key[: -len(".scaling")]
            modules.setdefault(mod, {})["scaling"] = float(tensor)

    name_to_module = dict(model.named_modules())
    with torch.no_grad():
        for mod_name, parts in modules.items():
            linear = name_to_module[mod_name]
            A = parts["A"].float()
            B = parts["B"].float()
            scaling = parts["scaling"]
            delta = (scaling * (B @ A)).to(dtype=linear.weight.dtype)
            linear.weight.data.add_(delta.to(linear.weight.device))
    return model.to(device)


def compare_merged_unmerged_logits(
    model_name: str,
    method: str,
    checkpoint_dir: Path,
    tokenizer,
    device: torch.device,
    sample_text: str = "What is 2+2?",
) -> dict:
    """Check merged vs unmerged logits on a small batch for adapter methods."""
    if method not in {"lora", "oplora"}:
        return {"skipped": True, "reason": "not an adapter method"}

    merged, _ = load_eval_model(
        model_name=model_name,
        method=method,
        checkpoint_dir=checkpoint_dir,
        device=device,
        merge_adapters=True,
    )
    unmerged, _ = load_eval_model(
        model_name=model_name,
        method=method,
        checkpoint_dir=checkpoint_dir,
        device=device,
        merge_adapters=False,
    )

    messages = [{"role": "user", "content": sample_text}]
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
    ids = ids.to(device)

    with torch.no_grad():
        logits_m = merged(ids).logits.float()
        logits_u = unmerged(ids).logits.float()
    max_abs = (logits_m - logits_u).abs().max().item()
    mean_abs = (logits_m - logits_u).abs().mean().item()
    del merged, unmerged
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "skipped": False,
        "max_abs_logit_diff": max_abs,
        "mean_abs_logit_diff": mean_abs,
        "ok": max_abs < 1e-2,
    }
