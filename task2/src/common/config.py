from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

from src.common.logging_utils import write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_shared_config(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    return load_yaml(root / "configs" / "shared.yaml")


def save_config_snapshot(cfg: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def smoke_enabled(args: argparse.Namespace | None = None) -> bool:
    if args is not None and bool(getattr(args, "smoke", False)):
        return True
    return os.environ.get("ATML_SMOKE", "").strip().lower() in {"1", "true", "yes", "y"}


def apply_runtime_overrides(
    cfg: dict[str, Any],
    *,
    smoke: bool,
    smoke_steps: int | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["smoke_active"] = bool(smoke)
    if out.get("enable_thinking", False):
        raise RuntimeError("enable_thinking must stay false for Task 2")
    out["enable_thinking"] = False
    if not smoke:
        _assert_full_budget(out)
        return out
    sm = out["smoke"]
    steps = int(smoke_steps if smoke_steps is not None else sm["updates_per_stage"])
    if steps < 1:
        raise ValueError("smoke updates per stage must be >= 1")
    data = out["data"]
    data["caption_train_images"] = int(sm["caption_train_examples"])
    data["captions_per_train_image"] = int(sm["captions_per_train_image"])
    data["caption_dev_images"] = int(sm["caption_dev_images"])
    data["grounding_train_pool_cap"] = dict(sm["grounding_train_pool_cap"])
    data["grounding_dev_pairs"] = dict(sm["grounding_dev_pairs"])
    data["grounding_test_pairs"] = dict(sm["grounding_test_pairs"])
    for key in ("natural_train_pairs", "natural_dev_pairs", "natural_test_pairs"):
        data[key] = dict(sm["natural_pairs"])
    if "feature_batch_size" in sm:
        data["feature_batch_size"] = int(sm["feature_batch_size"])
    out["evaluation"]["bootstrap_resamples"] = int(sm["bootstrap_resamples"])
    for stage in out["training"]["stages"]:
        stage["updates"] = steps
    out["smoke_steps"] = steps
    return out


def _assert_full_budget(cfg: dict[str, Any]) -> None:
    train = cfg["training"]
    stages = train["stages"]
    updates = [int(s["updates"]) for s in stages]
    if updates != [2250, 1500, 750]:
        raise RuntimeError(f"Full-run stage updates must be 2250/1500/750, got {updates}")
    if int(train["microbatch_size"]) * int(train["grad_accum"]) != int(train["effective_batch"]):
        raise RuntimeError("microbatch * accum must equal effective batch")
    if int(train["effective_batch"]) != 16:
        raise RuntimeError("Full-run effective batch must be 16")
    if sum(updates) != 4500:
        raise RuntimeError("Full run must be 4500 updates")
    if sum(updates) * 16 != 72000:
        raise RuntimeError("Full run must present 72000 examples")
    data = cfg["data"]
    if int(data["caption_train_images"]) * int(data["captions_per_train_image"]) != 18000:
        raise RuntimeError("Caption train set must be 6000 images x 3 captions")
    if int(data["caption_dev_images"]) != 96:
        raise RuntimeError("Caption dev set must be 96 images")
    if int(cfg["model"]["num_visual_tokens"]) not in {49, 196}:
        raise RuntimeError("num_visual_tokens must be 196 or 49")


def stage_lengths(cfg: dict[str, Any]) -> list[int]:
    return [int(stage["updates"]) for stage in cfg["training"]["stages"]]


def total_updates(cfg: dict[str, Any]) -> int:
    return sum(stage_lengths(cfg))


def cumulative_stage_ends(cfg: dict[str, Any]) -> list[int]:
    ends: list[int] = []
    running = 0
    for length in stage_lengths(cfg):
        running += length
        ends.append(running)
    return ends


def dev_eval_points(cfg: dict[str, Any]) -> set[int]:
    """Update counts AFTER which dev eval runs. Step 0 is handled by the caller."""
    points = set(cumulative_stage_ends(cfg))
    every = int(cfg["evaluation"]["dev_every"])
    total = total_updates(cfg)
    step = every
    while step <= total:
        points.add(step)
        step += every
    return points


def checkpoint_points(cfg: dict[str, Any]) -> set[int]:
    points = set(cumulative_stage_ends(cfg))
    every = int(cfg["training"]["checkpoint_every"])
    total = total_updates(cfg)
    step = every
    while step <= total:
        points.add(step)
        step += every
    if cfg.get("smoke_active"):
        points.add(1)
    return {p for p in points if 1 <= p <= total}


def locate_completed(completed_updates: int, lengths: list[int]) -> tuple[int, int]:
    """1-based stage index and within-stage step of the last completed update.

    completed_updates == 0 returns stage 1, step 0 (nothing finished yet).
    """
    if completed_updates <= 0:
        return 1, 0
    remaining = completed_updates
    for index, length in enumerate(lengths, start=1):
        if remaining <= length:
            return index, remaining
        remaining -= length
    return len(lengths), lengths[-1]


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    return parser


def prepare_runtime(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    root = find_repo_root()
    cfg = load_shared_config(root)
    cfg = apply_runtime_overrides(cfg, smoke=smoke_enabled(args), smoke_steps=args.smoke_steps)
    artifacts = resolve_artifacts_root(cfg, args.artifacts_root)
    dirs = ensure_result_dirs(artifacts)
    save_config_snapshot(cfg, dirs["configs_used"] / "resolved.yaml")
    write_json(
        dirs["configs_used"] / "runtime.json",
        {
            "smoke_active": bool(cfg.get("smoke_active")),
            "seed": cfg["seed"],
            "num_visual_tokens": cfg["model"]["num_visual_tokens"],
            "artifacts_root": str(artifacts),
        },
    )
    return root, cfg, dirs
