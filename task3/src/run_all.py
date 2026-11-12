from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # Offline first (may exit 2 if no data), then live.
    offline = subprocess.call([sys.executable, "-m", "src.run_offline", *sys.argv[1:]], cwd=str(ROOT))
    live = subprocess.call([sys.executable, "-m", "src.run_live", *sys.argv[1:]], cwd=str(ROOT))
    if live != 0:
        return live
    return 0 if offline in {0, 2} else offline


if __name__ == "__main__":
    raise SystemExit(main())
