from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.float16,
    gradient_checkpointing: bool = True,
    device: str | torch.device | None = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device is not None:
        model.to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
    return model


def count_trainable_parameters(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def freeze_backbone(model) -> None:
    for p in model.parameters():
        p.requires_grad = False


def enable_input_require_grads(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)


def get_named_matrix(model, module_suffix: str):
    """Return a module matching a dotted suffix, e.g. model.layers.0.mlp.down_proj."""
    for name, module in model.named_modules():
        if name.endswith(module_suffix):
            return name, module
    raise KeyError(module_suffix)


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "base_model") and hasattr(model, "peft_config"):
        return model
    return model


def build_generation_kwargs(enable_thinking: bool = False) -> dict[str, Any]:
    return {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "enable_thinking": enable_thinking,
    }
