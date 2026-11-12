from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

from src.common.logging_utils import write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root, resolve_data_root


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_config(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    return load_yaml(root / "configs" / "default.yaml")


def smoke_enabled(args: argparse.Namespace | None = None) -> bool:
    if args is not None and bool(getattr(args, "smoke", False)):
        return True
    return os.environ.get("ATML_SMOKE", "").strip().lower() in {"1", "true", "yes", "y"}


def apply_smoke_overrides(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["smoke_active"] = bool(smoke)
    if not smoke:
        return out
    sm = out.get("smoke", {})
    live = out["live"]
    for key in (
        "single_max_new_tokens",
        "parallel_max_new_tokens",
        "refine_initial_max_new_tokens",
        "refine_revision_max_new_tokens",
    ):
        if key in sm:
            live[key] = int(sm[key])
    out["live_n_questions_override"] = int(sm.get("n_live_questions", 2))
    return out


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def prepare_runtime(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Path], Path]:
    root = find_repo_root()
    cfg = load_config(root)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    cfg = apply_smoke_overrides(cfg, smoke=smoke_enabled(args))
    artifacts = resolve_artifacts_root(cfg, args.artifacts_root)
    dirs = ensure_result_dirs(artifacts)
    data_root = resolve_data_root(cfg, args.data_root)
    write_json(
        dirs["configs_used"] / "runtime.json",
        {
            "smoke_active": bool(cfg.get("smoke_active")),
            "seed": cfg["seed"],
            "artifacts_root": str(artifacts),
            "data_root": str(data_root),
            "data_exists": data_root.exists(),
        },
    )
    with open(dirs["configs_used"] / "resolved.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return root, cfg, dirs, data_root
