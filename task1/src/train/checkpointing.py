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
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Model / adapter weights
    model_dir = out_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_peft_model(model):
        model.save_pretrained(model_dir)
        torch.save(None, model_dir / ".peft_marker")
    elif any(isinstance(m, OPLoRALinear) for m in model.modules()):
        # Save full OPLoRA module state (includes trainable A/B and frozen bases)
        torch.save(model.state_dict(), model_dir / "oplora_state.pt")
        projected = export_projected_adapter_state(model)
        torch.save(projected, model_dir / "oplora_projected_factors.pt")
        if oplora_meta:
            with open(model_dir / "oplora_meta.json", "w", encoding="utf-8") as f:
                json.dump(oplora_meta, f, indent=2)
    else:
        model.save_pretrained(model_dir)

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
            "rng": get_rng_state(),
            "theta0_path": str(theta0_path) if theta0_path else None,
            "extra": extra or {},
        },
        out_dir / "trainer_state.pt",
    )

    meta = {
        "step": step,
        "examples_seen": examples_seen,
        "variant_name": variant_name,
        "method": method,
        "model_name": model_name,
        "checkpoint_dir": str(out_dir),
        "size_bytes": _dir_size_bytes(out_dir),
        "save_seconds": time.perf_counter() - t0,
    }
    with open(out_dir / "checkpoint_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    return torch.load(checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False)


def restore_rng_from_checkpoint(checkpoint_dir: Path) -> None:
    state = load_trainer_state(checkpoint_dir)
    if "rng" in state:
        set_rng_state(state["rng"])
