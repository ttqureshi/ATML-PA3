from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from src.common.logging_utils import write_json
from src.common.paths import checkpoint_dir
from src.common.seed import get_rng_state, set_rng_state
from src.models.lora import is_peft_model


def save_checkpoint(
    out_dir: Path,
    *,
    connector,
    lm,
    optimizer,
    scheduler,
    scaler,
    progress: dict[str, Any],
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(connector.state_dict(), out_dir / "connector.pt")
    if is_peft_model(lm):
        lm.save_pretrained(out_dir / "lora")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "scheduler_last_epoch": int(scheduler.last_epoch),
            "progress": progress,
            "rng": get_rng_state(),
        },
        out_dir / "trainer_state.pt",
    )
    write_json(out_dir / "progress.json", progress)


def save_stage_weights(out_dir: Path, *, connector, lm, progress: dict[str, Any]) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(connector.state_dict(), out_dir / "connector.pt")
    if is_peft_model(lm):
        lm.save_pretrained(out_dir / "lora")
    write_json(out_dir / "progress.json", progress)


def load_checkpoint_blob(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path / "trainer_state.pt", map_location="cpu")


def load_connector_weights(connector, path: Path) -> None:
    weight_path = path / "connector.pt"
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    connector.load_state_dict(state)


def find_latest_checkpoint(root: Path) -> Path | None:
    best: Path | None = None
    best_step = -1
    if not root.exists():
        return None
    for path in root.glob("step_*"):
        if not (path / "trainer_state.pt").exists():
            continue
        try:
            step = int(path.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if step > best_step:
            best, best_step = path, step
    return best


def resolve_final_weights(root: Path, total_updates: int) -> Path:
    stage_end = root / "stage3_end"
    if (stage_end / "connector.pt").exists():
        return stage_end
    latest = find_latest_checkpoint(root)
    if latest is None:
        raise FileNotFoundError(f"No Task 2 checkpoint under {root}")
    step = int(latest.name.split("_")[1])
    if step < total_updates:
        raise RuntimeError(f"Final eval needs update {total_updates}; latest checkpoint is {step}")
    return latest


def set_schedule_position(optimizer, scheduler, multiplier_fn, epoch: int) -> None:
    """Set LambdaLR so the next optimizer step uses ``multiplier_fn(epoch)``."""
    scheduler.last_epoch = int(epoch)
    multiplier = float(multiplier_fn(epoch))
    last_lrs = []
    for group in optimizer.param_groups:
        base = float(group.get("initial_lr", group["lr"]))
        group["initial_lr"] = base
        group["lr"] = base * multiplier
        last_lrs.append(group["lr"])
    scheduler._last_lr = last_lrs
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = int(epoch) + 1


def restore_rng(blob: dict[str, Any]) -> None:
    if blob.get("rng"):
        set_rng_state(blob["rng"])


def read_dev_marker(root: Path) -> int:
    path = root / "dev_eval_marker.json"
    if not path.exists():
        return -1
    return int(json.loads(path.read_text(encoding="utf-8")).get("update", -1))


def write_dev_marker(root: Path, update: int) -> None:
    write_json(root / "dev_eval_marker.json", {"update": int(update)})
