from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import find_repo_root


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_shared_config(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    return load_yaml(root / "configs" / "shared.yaml")


def load_variant_config(name: str, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    path = root / "configs" / "variants" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown variant config: {path}")
    return load_yaml(path)


def merge_configs(shared: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    out = dict(shared)
    out["variant"] = variant
    return out


def save_config_snapshot(cfg: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
