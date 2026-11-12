from __future__ import annotations

from typing import Any

import torch

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _neutralize_torchao_check() -> None:
    """Kaggle's torchao build makes recent PEFT raise instead of skipping it."""
    try:
        import peft.import_utils as import_utils

        import_utils.is_torchao_available = lambda: False  # type: ignore[assignment]
    except Exception:
        return


def enable_input_require_grads(model) -> None:
    """Allow gradient checkpointing to backprop into inputs when the backbone is frozen."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    embeddings = base.get_input_embeddings()

    def _hook(_module, _inputs, output):
        if torch.is_tensor(output) and not output.requires_grad:
            output.requires_grad_(True)
        return output

    embeddings.register_forward_hook(_hook)


def freeze_backbone(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def attach_lora(model, r: int, alpha: int, dropout: float = 0.05, target_modules: list[str] | None = None) -> Any:
    from peft import LoraConfig, get_peft_model

    freeze_backbone(model)
    enable_input_require_grads(model)
    _neutralize_torchao_check()
    config = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules or DEFAULT_TARGETS),
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, config)
    for _name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return peft_model


def is_peft_model(model) -> bool:
    try:
        from peft import PeftModel
    except ImportError:
        return hasattr(model, "peft_config")
    return isinstance(model, PeftModel) or hasattr(model, "peft_config")


def trainable_parameter_report(lm, connector) -> dict[str, int]:
    connector_params = sum(p.numel() for p in connector.parameters() if p.requires_grad)
    lora_params = 0
    other = 0
    frozen = 0
    for name, param in lm.named_parameters():
        if not param.requires_grad:
            frozen += param.numel()
            continue
        if "lora_" in name:
            lora_params += param.numel()
        else:
            other += param.numel()
    return {
        "connector": int(connector_params),
        "lora": int(lora_params),
        "other_trainable": int(other),
        "frozen_lm": int(frozen),
        "trainable_total": int(connector_params + lora_params + other),
    }


def assert_only_lora_trainable(lm) -> None:
    for name, param in lm.named_parameters():
        if param.requires_grad and "lora_" not in name:
            raise RuntimeError(f"Non-LoRA LM parameter is trainable: {name}")
