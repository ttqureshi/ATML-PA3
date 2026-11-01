from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WATCH = [
    "atml-pa3-t1-prep-baseline",
    "atml-pa3-t1-train-lora-r4",
    "atml-pa3-t1-train-lora-r16",
    "atml-pa3-t1-train-oplora",
    "atml-pa3-t1-train-fullft",
    "atml-pa3-t1-train-full-ft-l2sp",
]

TO_PUSH = [
    ("atml-pa3-t1-train-full-ft", "atml-pa3-t1-train-fullft"),
    ("atml-pa3-t1-train-full-ft-l2sp", "atml-pa3-t1-train-full-ft-l2sp"),
]


def status(slug: str) -> str:
    try:
        out = subprocess.check_output(
            ["kaggle", "kernels", "status", f"mtayyabtahir/{slug}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return f"ERR:{e}"
    for label in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED", "PENDING"):
        if label in out:
            return label
    return "UNKNOWN"


def busy_count() -> int:
    n = 0
    for slug in WATCH:
        st = status(slug)
        if st in {"RUNNING", "QUEUED", "PENDING"}:
            n += 1
    return n


def main() -> None:
    subprocess.check_call(["python", str(ROOT / "scripts" / "make_notebook.py")])
    pending = list(TO_PUSH)
    while pending:
        busy = busy_count()
        print(time.strftime("%H:%M:%S"), "busy", busy, "pending", [p[1] for p in pending], flush=True)
        for slug in WATCH:
            print(" ", slug, status(slug), flush=True)
        if busy < 2:
            folder, slug = pending[0]
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
                    # push rejected or immediately failed; wait and retry
                    time.sleep(90)
            except subprocess.CalledProcessError:
                print("push rejected (likely GPU cap); waiting", flush=True)
                time.sleep(90)
        else:
            time.sleep(120)
    print("Full FT and L2-SP re-queued.", flush=True)


if __name__ == "__main__":
    main()
