from __future__ import annotations

"""Drive Task 1 to completion on Kaggle.

Kaggle allows at most 2 concurrent GPU kernels. This orchestrator keeps both
slots filled: push the next pending job whenever a slot frees, instead of
waiting serially on one job at a time.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
USER = "mtayyabtahir"
MAX_SLOTS = 2

# folder, live slug, is_gate
JOBS = [
    ("atml-pa3-t1-smoke", "atml-pa3-t1-smoke", True),
    ("atml-pa3-t1-train-lora-r4", "atml-pa3-t1-train-lora-r4", False),
    ("atml-pa3-t1-train-lora-r16", "atml-pa3-t1-train-lora-r16", False),
    ("atml-pa3-t1-train-oplora", "atml-pa3-t1-train-oplora", False),
    ("atml-pa3-t1-train-full-ft", "atml-pa3-t1-train-fullft", False),
    ("atml-pa3-t1-train-full-ft-l2sp", "atml-pa3-t1-train-full-ft-l2sp", False),
    ("atml-pa3-t1-pipeline-prep", "atml-pa3-t1-prep-baseline", False),
    ("atml-pa3-t1-pipeline-post", "atml-pa3-t1-pipeline-post", False),
]

# Jobs that must wait until all prior train variants are done.
POST_DEPS = {
    "atml-pa3-t1-pipeline-post": {
        "atml-pa3-t1-train-lora-r4",
        "atml-pa3-t1-train-lora-r16",
        "atml-pa3-t1-train-oplora",
        "atml-pa3-t1-train-fullft",
        "atml-pa3-t1-train-full-ft-l2sp",
    },
}


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


def push_folder(folder: str) -> None:
    run(["kaggle", "kernels", "push", "-p", str(ROOT / "kaggle" / "kernels" / folder)])


def pull_small(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # Prefer not to hang forever on multi-GB checkpoints.
    subprocess.call(
        ["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(dest)],
        timeout=180,
    )
    for p in dest.rglob("*"):
        if not p.is_file():
            continue
        keep_names = {"failure.txt", "subprocess_last.log", "stage_status.json", "code_fingerprint.json"}
        if p.name in keep_names:
            continue
        if p.suffix in {".safetensors", ".pt", ".bin", ".jsonl"} or p.stat().st_size > 500_000:
            if "exception" in p.name or (p.suffix in {".log", ".txt"} and p.stat().st_size < 2_000_000):
                continue
            try:
                p.unlink()
            except OSError:
                pass


def save_state(
    path: Path,
    done: set[str],
    failed: set[str],
    in_flight: dict[str, str],
    retries: dict[str, int] | None = None,
) -> None:
    blob: dict[str, Any] = {
        "done": sorted(done),
        "failed": sorted(failed),
        "in_flight": in_flight,
    }
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            blob["retries"] = prev.get("retries", {})
        except Exception:
            blob["retries"] = {}
    if retries is not None:
        blob["retries"] = retries
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def deps_satisfied(slug: str, done: set[str]) -> bool:
    need = POST_DEPS.get(slug)
    if not need:
        return True
    return need.issubset(done)


def main() -> None:
    run(["python", str(ROOT / "scripts" / "make_notebook.py")])
    state_path = ROOT / "kaggle_pulls" / "orchestrate_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    failed: set[str] = set()
    if state_path.exists():
        blob = json.loads(state_path.read_text(encoding="utf-8"))
        done = set(blob.get("done", []))
        failed = set(blob.get("failed", []))

    # Adopt already-COMPLETE train kernels so we don't waste a multi-hour re-run.
    for folder, slug, is_gate in JOBS:
        if slug in done or is_gate:
            continue
        if status(slug) == "COMPLETE":
            print("adopt existing COMPLETE", slug, flush=True)
            done.add(slug)

    # folder/slug/gate still pending
    pending = [(f, s, g) for f, s, g in JOBS if s not in done]
    # slug -> (folder, is_gate)
    in_flight: dict[str, tuple[str, bool]] = {}

    # Re-attach anything already RUNNING/QUEUED that we still need.
    for folder, slug, is_gate in pending:
        st = status(slug)
        if st in {"RUNNING", "QUEUED", "PENDING"}:
            print("attach in-flight", slug, st, flush=True)
            in_flight[slug] = (folder, is_gate)

    print(
        "start parallel orch; pending=",
        [s for _, s, _ in pending],
        "in_flight=",
        list(in_flight),
        "done=",
        sorted(done),
        flush=True,
    )

    while pending or in_flight:
        # Harvest finished in-flight jobs.
        for slug in list(in_flight):
            st = status(slug)
            print(time.strftime("%H:%M:%S"), "watch", slug, st, flush=True)
            if st in {"RUNNING", "QUEUED", "PENDING", "UNKNOWN"}:
                continue
            folder, is_gate = in_flight.pop(slug)
            pull_dir = ROOT / "kaggle_pulls" / slug.replace("/", "_")
            try:
                pull_small(slug, pull_dir)
            except Exception as e:
                print("pull failed/timeout", slug, e, flush=True)

            if st == "COMPLETE":
                done.add(slug)
                failed.discard(slug)
                print("DONE", slug, flush=True)
            else:
                failed.add(slug)
                for name in (
                    "results/failure.txt",
                    "results/subprocess_last.log",
                    "results/stage_status.json",
                ):
                    p = pull_dir / name
                    if p.exists():
                        print("====", name, "====", flush=True)
                        print(p.read_text(encoding="utf-8", errors="replace")[:4000], flush=True)
                if is_gate:
                    save_state(
                        state_path,
                        done,
                        failed,
                        {k: v[0] for k, v in in_flight.items()},
                    )
                    raise SystemExit(f"Gate job failed: {slug} status={st}")
                # Do NOT auto-retry train failures: burning Kaggle GPU quota on the same
                # OOM/bug is worse than waiting for a real fix + manual/controlled requeue.
                print("NON-GATE FAILURE", slug, st, "— not auto-retrying (save GPU quota)", flush=True)

            # Drop from pending if still listed
            pending = [(f, s, g) for f, s, g in pending if s != slug]
            save_state(state_path, done, failed, {k: v[0] for k, v in in_flight.items()})

        # Fill free GPU slots.
        while len(in_flight) < MAX_SLOTS:
            next_job = None
            for i, (folder, slug, is_gate) in enumerate(pending):
                if slug in in_flight or slug in done:
                    continue
                if not deps_satisfied(slug, done):
                    print("defer", slug, "waiting deps", flush=True)
                    continue
                next_job = (i, folder, slug, is_gate)
                break
            if next_job is None:
                break
            i, folder, slug, is_gate = next_job
            print("PUSH", folder, "->", slug, "slots", len(in_flight) + 1, "/", MAX_SLOTS, flush=True)
            try:
                push_folder(folder)
            except subprocess.CalledProcessError:
                print("push failed, retry in 60s", flush=True)
                time.sleep(60)
                try:
                    push_folder(folder)
                except subprocess.CalledProcessError as e:
                    print("push failed again", e, flush=True)
                    break
            in_flight[slug] = (folder, is_gate)
            pending = [(f, s, g) for j, (f, s, g) in enumerate(pending) if not (s == slug and j == i)]
            # Re-build pending without this slug
            pending = [(f, s, g) for f, s, g in JOBS if s not in done and s not in in_flight]
            save_state(state_path, done, failed, {k: v[0] for k, v in in_flight.items()})

        if not pending and not in_flight:
            break
        if pending and not in_flight:
            # Only deferred deps left (e.g. post waiting on failed trains)
            blocked = [s for _, s, _ in pending if not deps_satisfied(s, done)]
            if len(blocked) == len(pending):
                print("blocked on failed/missing deps; done=", sorted(done), "failed=", sorted(failed), flush=True)
                break
        time.sleep(60)

    print(
        "Task 1 orchestration finished.",
        "done=",
        sorted(done),
        "failed=",
        sorted(failed),
        flush=True,
    )
    if failed:
        raise SystemExit(f"Completed with failures: {sorted(failed)}")


if __name__ == "__main__":
    main()
