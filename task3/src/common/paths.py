from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    here = Path(__file__).resolve().parents[2]
    if (here / "configs" / "default.yaml").is_file() and (here / "src").is_dir():
        return here
    cur = Path(start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs" / "default.yaml").is_file() and (candidate / "src" / "run_live.py").is_file():
            return candidate
        task3 = candidate / "task3"
        if (task3 / "configs" / "default.yaml").is_file():
            return task3
    for candidate in (Path("/kaggle/working/task3"), Path("/kaggle/working")):
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return here


def resolve_data_root(cfg: dict, override: str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("ATML_PA3_DATA", "").strip()
    if env:
        return Path(env)
    kaggle = Path("/kaggle/input/atml-pa3-data")
    if kaggle.exists():
        # Prefer nested ATML-PA3-Data if present.
        nested = kaggle / "ATML-PA3-Data"
        return nested if nested.exists() else kaggle
    return find_repo_root() / cfg.get("paths", {}).get("data_root", "data/ATML-PA3-Data")


def resolve_artifacts_root(cfg: dict, override: str | None = None) -> Path:
    if override:
        root = Path(override)
    elif os.environ.get("ATML_ARTIFACTS_ROOT"):
        root = Path(os.environ["ATML_ARTIFACTS_ROOT"])
    elif Path("/kaggle/working").exists() and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        root = Path("/kaggle/working/results")
    else:
        root = find_repo_root() / cfg.get("paths", {}).get("artifacts_root", "results")
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_result_dirs(artifacts_root: Path) -> dict[str, Path]:
    mapping = {
        "root": artifacts_root,
        "splits": artifacts_root / "splits",
        "tables": artifacts_root / "tables",
        "figures": artifacts_root / "figures",
        "live": artifacts_root / "live",
        "adaptive": artifacts_root / "adaptive",
        "analysis": artifacts_root / "analysis",
        "logs": artifacts_root / "logs",
        "configs_used": artifacts_root / "configs_used",
        "resources": artifacts_root / "resources",
        "per_question": artifacts_root / "per_question",
    }
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping
