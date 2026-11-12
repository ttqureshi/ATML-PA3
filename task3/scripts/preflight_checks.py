"""CPU-only preflight before spending Kaggle GPU time."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # Ensure kaggle identity is mtayyabtahir (do NOT restore access_token).
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access = Path.home() / ".kaggle" / "access_token"
    print("kaggle.json exists", kaggle_json.exists())
    print("access_token exists", access.exists(), "(should be False / unused)")
    if kaggle_json.exists():
        import json

        meta = json.loads(kaggle_json.read_text(encoding="utf-8"))
        print("kaggle_user", meta.get("username"))
        if meta.get("username") != "mtayyabtahir":
            print("WARNING: expected mtayyabtahir")
    rc = subprocess.call([sys.executable, str(ROOT / "tests" / "test_core.py")], cwd=str(ROOT))
    if rc != 0:
        return rc
    print("ALL_PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
