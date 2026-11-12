from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Task 2 package root (directory that contains configs/ and src/)."""
    here = Path(__file__).resolve().parents[2]
    if (here / "configs" / "shared.yaml").is_file() and (here / "src" / "run_stage.py").is_file():
        return here
    cur = Path(start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "src" / "models" / "connector.py").is_file() and (candidate / "configs").is_dir():
            return candidate
        task2 = candidate / "task2"
        if (task2 / "src" / "models" / "connector.py").is_file():
            return task2
    for candidate in (Path("/kaggle/working/task2"), Path("/kaggle/working")):
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return here


def resolve_artifacts_root(shared_cfg: dict, override: str | None = None) -> Path:
    if override:
        root = Path(override)
    elif os.environ.get("ATML_ARTIFACTS_ROOT"):
        root = Path(os.environ["ATML_ARTIFACTS_ROOT"])
    elif Path("/kaggle/working").exists() and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        root = Path("/kaggle/working/results")
    else:
        root = find_repo_root() / shared_cfg.get("paths", {}).get("artifacts_root", "results")
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_result_dirs(artifacts_root: Path) -> dict[str, Path]:
    mapping = {
        "root": artifacts_root,
        "data": artifacts_root / "raw" / "data",
        "annotations": artifacts_root / "raw" / "data" / "annotations",
        "images": artifacts_root / "raw" / "data" / "images",
        "features": artifacts_root / "raw" / "features",
        "checkpoints": artifacts_root / "checkpoints",
        "logs": artifacts_root / "logs",
        "eval": artifacts_root / "raw" / "eval",
        "tables": artifacts_root / "tables",
        "figures": artifacts_root / "figures",
        "qualitative": artifacts_root / "qualitative",
        "configs_used": artifacts_root / "configs_used",
        "resources": artifacts_root / "resources",
    }
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping


def checkpoint_dir(dirs: dict[str, Path], smoke: bool) -> Path:
    path = dirs["checkpoints"] / ("smoke" if smoke else "vlm")
    path.mkdir(parents=True, exist_ok=True)
    return path
