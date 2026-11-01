from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def status(slug: str) -> str:
    out = subprocess.check_output(["kaggle", "kernels", "status", f"mtayyabtahir/{slug}"], text=True)
    for label in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED", "PENDING"):
        if label in out:
            return label
    return out.strip()


def main() -> None:
    subprocess.check_call(["python", str(ROOT / "scripts" / "make_notebook.py")])
    print("Waiting for a free GPU slot to re-push Full FT...", flush=True)
    while True:
        prep = status("atml-pa3-t1-prep-baseline")
        l2 = status("atml-pa3-t1-train-full-ft-l2sp")
        busy = sum(1 for s in (prep, l2) if s in {"RUNNING", "QUEUED", "PENDING"})
        print(time.strftime("%H:%M:%S"), "prep", prep, "l2sp", l2, "busy", busy, flush=True)
        if busy < 2:
            folder = ROOT / "kaggle" / "kernels" / "atml-pa3-t1-train-full-ft"
            subprocess.check_call(["kaggle", "kernels", "push", "-p", str(folder)])
            print("Re-pushed Full FT", flush=True)
            print(status("atml-pa3-t1-train-fullft"), flush=True)
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
