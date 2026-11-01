from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Locate the task1 package root (directory containing configs/ and src/)."""
    cur = Path(start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
        if (candidate / "task1" / "configs").is_dir():
            return candidate / "task1"
    # Kaggle code-dataset layouts
    for candidate in [
        Path("/kaggle/input/atml-pa3-task1-code"),
        Path("/kaggle/working/task1"),
        Path("/kaggle/working"),
    ]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return cur


def resolve_artifacts_root(shared_cfg: dict, override: str | None = None) -> Path:
    if override:
        root = Path(override)
    elif os.environ.get("ATML_ARTIFACTS_ROOT"):
        root = Path(os.environ["ATML_ARTIFACTS_ROOT"])
    elif Path("/kaggle/working").exists():
        root = Path("/kaggle/working/results")
    else:
        root = find_repo_root() / shared_cfg["paths"]["artifacts_root"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_result_dirs(artifacts_root: Path) -> dict[str, Path]:
    mapping = {
        "root": artifacts_root,
        "data": artifacts_root / "raw" / "data",
        "checkpoints": artifacts_root / "checkpoints",
        "logs": artifacts_root / "logs",
        "eval": artifacts_root / "raw" / "eval",
        "diagnostics": artifacts_root / "diagnostics",
        "qualitative": artifacts_root / "qualitative",
        "tables": artifacts_root / "tables",
        "figures": artifacts_root / "figures",
        "configs_used": artifacts_root / "configs_used",
        "resources": artifacts_root / "resources",
        "probe": artifacts_root / "raw" / "probe",
    }
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping
