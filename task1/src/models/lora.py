from __future__ import annotations

from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model

from .load_qwen import count_trainable_parameters, enable_input_require_grads, freeze_backbone


DEFAULT_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _neutralize_torchao_check() -> None:
    """Kaggle ships torchao 0.10; recent peft raises ImportError instead of skipping it."""
    try:
        import peft.import_utils as iu

        iu.is_torchao_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass


_neutralize_torchao_check()


def attach_lora(
    model,
    r: int,
    alpha: int,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    freeze_backbone(model)
    enable_input_require_grads(model)
    _neutralize_torchao_check()
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or DEFAULT_TARGETS,
        init_lora_weights=True,  # Gaussian A, zeros B
    )
    peft_model = get_peft_model(model, cfg)
    # Keep adapter weights in FP32 as required.
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return peft_model


def merge_lora_into_base(peft_model) -> Any:
    """Return a fresh merged transformers model (adapters baked into weights)."""
    merged = peft_model.merge_and_unload()
    return merged


def lora_parameter_report(model) -> dict[str, int]:
    report = count_trainable_parameters(model)
    adapter_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_" in name or "lora_A" in name or "lora_B" in name):
            adapter_params += p.numel()
    report["adapter"] = int(adapter_params)
    return report


def is_peft_model(model) -> bool:
    return isinstance(model, PeftModel) or hasattr(model, "peft_config")
