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


def prune_unkept_checkpoints(variant_ckpt_root: Path, keep_steps: set[int]) -> None:
    """Remove whole step_* dirs not in keep_steps (Kaggle output quota / downloadability)."""
    if not variant_ckpt_root.exists():
        return
    for d in list(variant_ckpt_root.glob("step_*")):
        try:
            step = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if step in keep_steps:
            continue
        try:
            shutil.rmtree(d)
            print(f"PRUNED_CKPT {d}", flush=True)
        except OSError as e:
            print(f"PRUNE_FAIL {d} {e}", flush=True)


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
        # Full-FT path: never mutate live training tensors in-place (half()/float()
        # permanently truncates FP32 masters and previously risked dtype corruption).
        # Build a CPU FP16 state_dict copy and hand it to HF save_pretrained.
        was_training = model.training
        model.eval()
        try:
            cpu_half = {}
            for k, v in model.state_dict().items():
                if torch.is_tensor(v) and v.is_floating_point():
                    t = v.detach()
                    if not torch.isfinite(t).all():
                        raise RuntimeError(
                            f"Refusing to save non-finite full-FT weights at key={k}"
                        )
                    cpu_half[k] = t.to(dtype=torch.float16, device="cpu").contiguous()
                elif torch.is_tensor(v):
                    cpu_half[k] = v.detach().to("cpu").contiguous()
                else:
                    cpu_half[k] = v
            model.save_pretrained(
                model_dir,
                state_dict=cpu_half,
                safe_serialization=True,
            )
            del cpu_half
            # HF copies live-model config (FP32 masters). Weights on disk are FP16.
            cfg_path = model_dir / "config.json"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("dtype") not in {"float16", "fp16"}:
                    cfg["dtype"] = "float16"
                    cfg["torch_dtype"] = "float16"
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
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
