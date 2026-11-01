from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "mtayyabtahir"

# Prefer the actual live slugs observed after the first push, then fall back.
LIVE = {
    "pipeline-prep": ["atml-pa3-t1-prep-baseline", "atml-pa3-t1-pipeline-prep"],
    "train-full-ft": ["atml-pa3-t1-train-fullft", "atml-pa3-t1-train-full-ft"],
}

PUSH_ORDER = [
    ("atml-pa3-t1-train-full-ft-l2sp", "atml-pa3-t1-train-full-ft-l2sp"),
    ("atml-pa3-t1-train-lora-r4", "atml-pa3-t1-train-lora-r4"),
    ("atml-pa3-t1-train-lora-r16", "atml-pa3-t1-train-lora-r16"),
    ("atml-pa3-t1-train-oplora", "atml-pa3-t1-train-oplora"),
]


def status(kernel_id: str) -> str:
    try:
        out = subprocess.check_output(["kaggle", "kernels", "status", f"{USER}/{kernel_id}"], text=True)
    except subprocess.CalledProcessError as e:
        return f"MISSING:{e}"
    for label in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED", "PENDING"):
        if label in out:
            return label
    return out.strip()


def any_status(candidates: list[str]) -> tuple[str, str]:
    for c in candidates:
        st = status(c)
        if not st.startswith("MISSING"):
            return c, st
    return candidates[0], "MISSING"


def gpu_busy_count() -> int:
    n = 0
    watched = []
    for v in LIVE.values():
        watched.extend(v)
    watched += [k for k, _ in PUSH_ORDER]
    seen = set()
    for kid in watched:
        if kid in seen:
            continue
        seen.add(kid)
        st = status(kid)
        if st in {"RUNNING", "QUEUED", "PENDING"}:
            n += 1
    return n


def push(folder_name: str) -> None:
    folder = ROOT / "kaggle" / "kernels" / folder_name
    print("PUSH", folder, flush=True)
    subprocess.check_call(["kaggle", "kernels", "push", "-p", str(folder)])


def main() -> None:
    subprocess.check_call(["python", str(ROOT / "scripts" / "make_notebook.py")])
    pending = list(PUSH_ORDER)
    while pending:
        # show live jobs
        for label, cands in LIVE.items():
            kid, st = any_status(cands)
            print(time.strftime("%H:%M:%S"), label, kid, st, flush=True)
        busy = gpu_busy_count()
        print(time.strftime("%H:%M:%S"), "gpu_busy", busy, "pending", [p[0] for p in pending], flush=True)
        if busy < 2:
            folder, _ = pending.pop(0)
            try:
                push(folder)
            except subprocess.CalledProcessError as e:
                print("push_failed", folder, e, flush=True)
                pending.insert(0, (folder, _))
                time.sleep(90)
                continue
            time.sleep(5)
        else:
            time.sleep(120)
    print("All remaining train kernels pushed.", flush=True)


if __name__ == "__main__":
    main()
