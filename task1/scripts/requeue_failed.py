from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# folder on disk -> live Kaggle slug
JOBS = [
    ("atml-pa3-t1-pipeline-prep", "atml-pa3-t1-prep-baseline"),
    ("atml-pa3-t1-train-lora-r4", "atml-pa3-t1-train-lora-r4"),
    ("atml-pa3-t1-train-lora-r16", "atml-pa3-t1-train-lora-r16"),
    ("atml-pa3-t1-train-oplora", "atml-pa3-t1-train-oplora"),
]

WATCH = [
    "atml-pa3-t1-prep-baseline",
    "atml-pa3-t1-train-fullft",
    "atml-pa3-t1-train-full-ft-l2sp",
    "atml-pa3-t1-train-lora-r4",
    "atml-pa3-t1-train-lora-r16",
    "atml-pa3-t1-train-oplora",
]


def status(slug: str) -> str:
    try:
        out = subprocess.check_output(
            ["kaggle", "kernels", "status", f"mtayyabtahir/{slug}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return "MISSING"
    for label in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED", "PENDING"):
        if label in out:
            return label
    return "UNKNOWN"


def busy_count() -> int:
    return sum(1 for s in WATCH if status(s) in {"RUNNING", "QUEUED", "PENDING"})


def main() -> None:
    subprocess.check_call(["python", str(ROOT / "scripts" / "make_notebook.py")])
    pending = list(JOBS)
    while pending:
        busy = busy_count()
        print(time.strftime("%H:%M:%S"), "busy", busy, "pending", [p[1] for p in pending], flush=True)
        for slug in WATCH:
            print(" ", slug, status(slug), flush=True)
        if busy < 2:
            folder, slug = pending[0]
            # skip if already running/complete
            st = status(slug)
            if st in {"RUNNING", "QUEUED", "PENDING", "COMPLETE"}:
                print("skip", slug, st, flush=True)
                pending.pop(0)
                continue
            try:
                subprocess.check_call(
                    ["kaggle", "kernels", "push", "-p", str(ROOT / "kaggle" / "kernels" / folder)]
                )
                time.sleep(8)
                st = status(slug)
                print("pushed", slug, "->", st, flush=True)
                if st in {"RUNNING", "QUEUED", "PENDING"}:
                    pending.pop(0)
                else:
                    time.sleep(90)
            except subprocess.CalledProcessError:
                print("push rejected; waiting", flush=True)
                time.sleep(90)
        else:
            time.sleep(120)
    print("Failed jobs re-queued.", flush=True)


if __name__ == "__main__":
    main()
