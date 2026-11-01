from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from src.common.seed import get_rng_state, set_rng_state
from src.models.lora import is_peft_model
from src.models.oplora import OPLoRALinear, export_projected_adapter_state


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def prune_old_trainer_states(variant_ckpt_root: Path, keep_dir: Path) -> None:
    """Delete bulky optimizer states from older checkpoints to stay within Kaggle disk limits."""
    if not variant_ckpt_root.exists():
        return
    for d in variant_ckpt_root.glob("step_*"):
        if d.resolve() == keep_dir.resolve():
            continue
        for name in ("trainer_state.pt",):
            p = d / name
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


def save_training_checkpoint(
    out_dir: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    examples_seen: int,
    variant_name: str,
    method: str,
    model_name: str,
    theta0_path: Path | None = None,
    oplora_meta: dict | None = None,
    extra: dict | None = None,
    save_optimizer: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    model_dir = out_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_peft_model(model):
        model.save_pretrained(model_dir)
        (model_dir / ".peft_marker").write_text("peft\n", encoding="utf-8")
    elif any(isinstance(m, OPLoRALinear) for m in model.modules()):
        # Only adapter factors + frozen SVD bases (not the full backbone).
        adapter_state = {}
        for name, module in model.named_modules():
            if isinstance(module, OPLoRALinear):
                adapter_state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
                adapter_state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
                adapter_state[f"{name}.Uk"] = module.Uk.detach().cpu()
                adapter_state[f"{name}.Vk"] = module.Vk.detach().cpu()
                adapter_state[f"{name}.scaling"] = torch.tensor(module.scaling)
        torch.save(adapter_state, model_dir / "oplora_adapters.pt")
        projected = export_projected_adapter_state(model)
        torch.save(projected, model_dir / "oplora_projected_factors.pt")
        if oplora_meta:
            with open(model_dir / "oplora_meta.json", "w", encoding="utf-8") as f:
                json.dump(oplora_meta, f, indent=2)
    else:
        # Ensure FP16 serialization for full FT (avoids ~2.4GB FP32 dumps).
        was_training = model.training
        model.eval()
        try:
            for p in model.parameters():
                if p.is_floating_point() and p.dtype != torch.float16:
                    p.data = p.data.half()
            model.save_pretrained(model_dir, safe_serialization=True)
        finally:
            if was_training:
                model.train()

    if save_optimizer:
        # RNG state can be large/problematic to pickle across versions; store lightly.
        rng = get_rng_state()
        light_rng = {
            "torch": rng.get("torch"),
            "cuda": rng.get("cuda"),
            # skip python/numpy deep states if they bloat; still restore torch
        }
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "step": step,
                "examples_seen": examples_seen,
                "variant_name": variant_name,
                "method": method,
                "model_name": model_name,
                "rng": light_rng,
                "theta0_path": str(theta0_path) if theta0_path else None,
                "extra": extra or {},
            },
            out_dir / "trainer_state.pt",
        )
        prune_old_trainer_states(out_dir.parent, out_dir)

    meta = {
        "step": step,
        "examples_seen": examples_seen,
        "variant_name": variant_name,
        "method": method,
        "model_name": model_name,
        "checkpoint_dir": str(out_dir),
        "size_bytes": _dir_size_bytes(out_dir),
        "save_seconds": time.perf_counter() - t0,
        "saved_optimizer": bool(save_optimizer),
    }
    with open(out_dir / "checkpoint_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    return torch.load(checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False)


def restore_rng_from_checkpoint(checkpoint_dir: Path) -> None:
    state = load_trainer_state(checkpoint_dir)
    if "rng" in state and state["rng"]:
        set_rng_state({
            **get_rng_state(),
            **{k: v for k, v in state["rng"].items() if v is not None},
        })
