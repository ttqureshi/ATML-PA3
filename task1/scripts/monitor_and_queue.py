from __future__ import annotations

"""Monitor the smoke kernel; on success, push/queue the remaining Task 1 kernels."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "mtayyabtahir"

QUEUE_AFTER_SMOKE = [
    "atml-pa3-t1-pipeline-prep",
    "atml-pa3-t1-train-full-ft",
    "atml-pa3-t1-train-full-ft-l2sp",
    "atml-pa3-t1-train-lora-r4",
    "atml-pa3-t1-train-lora-r16",
    "atml-pa3-t1-train-oplora",
]


def status(kernel_id: str) -> str:
    out = subprocess.check_output(["kaggle", "kernels", "status", kernel_id], text=True)
    # e.g. user/slug has status "KernelWorkerStatus.COMPLETE"
    if "COMPLETE" in out:
        return "COMPLETE"
    if "ERROR" in out or "CANCELLED" in out:
        return "ERROR"
    if "RUNNING" in out:
        return "RUNNING"
    if "QUEUED" in out or "PENDING" in out:
        return "QUEUED"
    return out.strip()


def push(folder: Path) -> None:
    print("PUSH", folder.name, flush=True)
    subprocess.check_call(["kaggle", "kernels", "push", "-p", str(folder)])


def main() -> None:
    smoke = f"{USER}/atml-pa3-t1-smoke"
    print("Monitoring", smoke, flush=True)
    while True:
        st = status(smoke)
        print(time.strftime("%H:%M:%S"), st, flush=True)
        if st == "COMPLETE":
            break
        if st == "ERROR":
            raise SystemExit("Smoke failed; not queueing training kernels")
        time.sleep(60)

    # Ensure notebooks exist
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_notebook.py")])
    for name in QUEUE_AFTER_SMOKE:
        folder = ROOT / "kaggle" / "kernels" / name
        push(folder)
        time.sleep(2)
        print(name, status(f"{USER}/{name}"), flush=True)

    print("Queued training/prep kernels. pipeline-post should be pushed after trainings finish.", flush=True)


if __name__ == "__main__":
    main()
