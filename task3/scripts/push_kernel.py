from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNELS = {
    "smoke": ROOT / "kaggle" / "kernels" / "atml-pa3-t3-smoke",
    "live": ROOT / "kaggle" / "kernels" / "atml-pa3-t3-live",
    "offline": ROOT / "kaggle" / "kernels" / "atml-pa3-t3-offline",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["smoke", "live", "offline", "all"],
        default="live",
        help="Which kernel(s) to push. Default is live.",
    )
    args = parser.parse_args()
    names = list(KERNELS) if args.only == "all" else [args.only]
    for name in names:
        kernel = KERNELS[name]
        if not (kernel / "kernel-metadata.json").exists():
            raise SystemExit(f"Missing kernel metadata: {kernel}. Run scripts/make_notebook.py first.")
        cmd = ["kaggle", "kernels", "push", "-p", str(kernel)]
        print(">>", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
