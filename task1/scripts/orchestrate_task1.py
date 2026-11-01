from __future__ import annotations

"""Drive Task 1 to completion on Kaggle (2 GPU slots)."""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "mtayyabtahir"

# folder -> live slug
JOBS = [
    ("atml-pa3-t1-smoke", "atml-pa3-t1-smoke", True),  # must pass before trains
    ("atml-pa3-t1-pipeline-prep", "atml-pa3-t1-prep-baseline", False),
    ("atml-pa3-t1-train-lora-r4", "atml-pa3-t1-train-lora-r4", False),
    ("atml-pa3-t1-train-lora-r16", "atml-pa3-t1-train-lora-r16", False),
    ("atml-pa3-t1-train-oplora", "atml-pa3-t1-train-oplora", False),
    ("atml-pa3-t1-train-full-ft", "atml-pa3-t1-train-fullft", False),
    ("atml-pa3-t1-train-full-ft-l2sp", "atml-pa3-t1-train-full-ft-l2sp", False),
    ("atml-pa3-t1-pipeline-post", "atml-pa3-t1-pipeline-post", False),
]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def status(slug: str) -> str:
    try:
        out = subprocess.check_output(
            ["kaggle", "kernels", "status", f"{USER}/{slug}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return "MISSING"
    for lab in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED", "PENDING"):
        if lab in out:
            return lab
    return "UNKNOWN"


def busy_slugs() -> list[str]:
    busy = []
    for _, slug, _ in JOBS:
        st = status(slug)
        if st in {"RUNNING", "QUEUED", "PENDING"}:
            busy.append(slug)
    return busy


def push_folder(folder: str) -> None:
    run(["kaggle", "kernels", "push", "-p", str(ROOT / "kaggle" / "kernels" / folder)])


def wait_until_done(slug: str, timeout_sec: int = 60 * 60 * 12) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        st = status(slug)
        print(time.strftime("%H:%M:%S"), slug, st, flush=True)
        if st in {"COMPLETE", "ERROR", "CANCELLED", "MISSING"}:
            return st
        time.sleep(90)
    return "TIMEOUT"


def pull_small(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.call(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(dest)])
    # drop bulky files
    for p in dest.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".safetensors", ".pt", ".bin", ".jsonl"} or p.stat().st_size > 500_000:
            if p.name in {"failure.txt", "subprocess_last.log", "stage_status.json"}:
                continue
            if "exception" in p.name or p.suffix in {".log", ".txt"} and p.stat().st_size < 2_000_000:
                continue
            try:
                p.unlink()
            except OSError:
                pass


def main() -> None:
    run(["python", str(ROOT / "scripts" / "make_notebook.py")])
    state_path = ROOT / "kaggle_pulls" / "orchestrate_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if state_path.exists():
        done = set(json.loads(state_path.read_text(encoding="utf-8")).get("done", []))

    for folder, slug, is_gate in JOBS:
        if slug in done:
            print("skip done", slug, flush=True)
            continue

        # wait for a free slot
        while True:
            busy = busy_slugs()
            # if this slug already running, just wait
            if slug in busy:
                break
            if len(busy) < 2:
                break
            print(time.strftime("%H:%M:%S"), "waiting slot; busy=", busy, flush=True)
            time.sleep(90)

        print("PUSH", folder, "->", slug, flush=True)
        try:
            push_folder(folder)
        except subprocess.CalledProcessError:
            print("push failed, retry later", flush=True)
            time.sleep(120)
            push_folder(folder)

        st = wait_until_done(slug)
        pull_dir = ROOT / "kaggle_pulls" / slug.replace("/", "_")
        try:
            pull_small(slug, pull_dir)
        except Exception as e:
            print("pull failed", e, flush=True)

        if st != "COMPLETE":
            # show failure artifacts if present
            for name in ("results/failure.txt", "results/subprocess_last.log", "results/stage_status.json"):
                p = pull_dir / name
                if p.exists():
                    print("====", name, "====", flush=True)
                    print(p.read_text(encoding="utf-8", errors="replace")[:8000], flush=True)
            if is_gate:
                raise SystemExit(f"Gate job failed: {slug} status={st}")
            print("NON-GATE FAILURE", slug, st, "continuing", flush=True)
            continue

        done.add(slug)
        state_path.write_text(json.dumps({"done": sorted(done)}, indent=2), encoding="utf-8")
        print("DONE", slug, flush=True)

    print("Task 1 orchestration finished.", flush=True)


if __name__ == "__main__":
    main()
