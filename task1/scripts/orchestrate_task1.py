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


def current_kaggle_user() -> str:
    import os

    for key in ("ATML_KAGGLE_USER", "KAGGLE_USERNAME"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    cred = Path.home() / ".kaggle" / "credentials.json"
    if cred.exists():
        try:
            return str(json.loads(cred.read_text(encoding="utf-8")).get("username") or "").strip()
        except Exception:
            pass
    return "mtayyabtahir"


USER = current_kaggle_user()
MAX_SLOTS = 2
# Bump when a real training fix must replace a failed kernel once (not blind OOM loops).
CODE_FIX_ID = "gsm8k-thinking-disable-v2"

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


def push_cpu_kill(slug: str, folder: str) -> None:
    """Cancel a wasted GPU session for an already-done train."""
    path = ROOT / "kaggle" / "kernels" / folder
    path.mkdir(parents=True, exist_ok=True)
    code_file = f"{folder}.ipynb"
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [f"print('ORCH_KILL: {slug} already done; releasing GPU')\n"],
            }
        ],
    }
    (path / code_file).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    push_folder(folder)
    print("ALERT done_job_was_running — pushed CPU kill for", slug, flush=True)


def pull_small(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # Drop stale status files so a timed-out pull cannot fake SUCCESS/FAIL.
    for stale in (
        dest / "results" / "stage_status.json",
        dest / "results" / "failure.txt",
    ):
        try:
            stale.unlink()
        except OSError:
            pass
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
    force_rerun: set[str] | None = None,
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
            blob["fix_applied"] = prev.get("fix_applied", {})
            blob["force_rerun"] = prev.get("force_rerun", [])
            if prev.get("note"):
                blob["note"] = prev["note"]
        except Exception:
            blob["retries"] = {}
    if retries is not None:
        blob["retries"] = retries
    if force_rerun is not None:
        blob["force_rerun"] = sorted(force_rerun)
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
    force_rerun: set[str] = set()
    if state_path.exists():
        blob = json.loads(state_path.read_text(encoding="utf-8"))
        done = set(blob.get("done", []))
        failed = set(blob.get("failed", []))
        force_rerun = set(blob.get("force_rerun", []))
        # Never treat a forced re-run as already finished.
        done -= force_rerun

    # Adopt already-COMPLETE train kernels so we don't waste a multi-hour re-run.
    # Skip force_rerun: COMPLETE may be a CPU stub / wiped partial, not a real train.
    for folder, slug, is_gate in JOBS:
        if slug in done or is_gate or slug in force_rerun:
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
        # Safety: a DONE train must never burn GPU again.
        for folder, slug, _is_gate in JOBS:
            if slug not in done:
                continue
            if "train" not in slug:
                continue
            st = status(slug)
            if st in {"RUNNING", "QUEUED", "PENDING"}:
                print(
                    "ALERT unexpected re-run of DONE job",
                    slug,
                    st,
                    "— killing with CPU stub",
                    flush=True,
                )
                # Wake the Cursor agent watching this terminal.
                print(
                    'AGENT_LOOP_TICK_task1_watch '
                    + json.dumps(
                        {
                            "prompt": (
                                f"Task1 orch ALERT: done train {slug} is {st} again. "
                                "CPU-stub kill attempted. Verify no wasted GPU and do not "
                                "re-push completed trains."
                            )
                        }
                    ),
                    flush=True,
                )
                try:
                    push_cpu_kill(slug, folder)
                except Exception as e:
                    print("kill failed", slug, e, flush=True)

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
                # Kaggle COMPLETE ≠ training success. Require stage_status returncode==0
                # when that file exists (skip for CPU-stub kills / missing pulls).
                stage_ok = True
                stage_path = pull_dir / "results" / "stage_status.json"
                if "train" in slug or "prep" in slug:
                    if not stage_path.exists():
                        stage_ok = False
                        print(
                            "COMPLETE but missing stage_status after pull",
                            slug,
                            "— not marking done",
                            flush=True,
                        )
                if stage_path.exists():
                    try:
                        stage = json.loads(stage_path.read_text(encoding="utf-8"))
                        if int(stage.get("returncode", 1)) != 0:
                            stage_ok = False
                            print(
                                "COMPLETE but stage_status failed",
                                slug,
                                stage,
                                flush=True,
                            )
                    except Exception as e:
                        stage_ok = False
                        print("stage_status unreadable", slug, e, flush=True)
                if stage_ok:
                    done.add(slug)
                    failed.discard(slug)
                    force_rerun.discard(slug)
                    print("DONE", slug, flush=True)
                else:
                    failed.add(slug)
                    print(
                        'AGENT_LOOP_TICK_task1_watch '
                        + json.dumps(
                            {
                                "prompt": (
                                    f"Task1 orch ALERT: {slug} kernel COMPLETE but "
                                    "stage_status returncode!=0 — not marking done; "
                                    "weights/results may be missing. Investigate."
                                )
                            }
                        ),
                        flush=True,
                    )
                    st = "ERROR"

            if st != "COMPLETE":
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
                        force_rerun=force_rerun,
                    )
                    raise SystemExit(f"Gate job failed: {slug} status={st}")
                # At most one requeue per CODE_FIX_ID (deploy a real fix, don't loop OOMs).
                blob = {}
                if state_path.exists():
                    try:
                        blob = json.loads(state_path.read_text(encoding="utf-8"))
                    except Exception:
                        blob = {}
                applied = dict(blob.get("fix_applied", {}))
                if applied.get(slug) != CODE_FIX_ID and "pipeline-post" not in slug:
                    print(
                        "NON-GATE FAILURE",
                        slug,
                        st,
                        f"— requeue once for code fix {CODE_FIX_ID}",
                        flush=True,
                    )
                    failed.discard(slug)
                    pending.append((folder, slug, is_gate))
                    applied[slug] = CODE_FIX_ID
                    blob["fix_applied"] = applied
                    blob["done"] = sorted(done)
                    blob["failed"] = sorted(failed)
                    blob["in_flight"] = {k: v[0] for k, v in in_flight.items()}
                    blob["force_rerun"] = sorted(force_rerun)
                    state_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
                    continue
                print(
                    "NON-GATE FAILURE",
                    slug,
                    st,
                    "— not retrying; fix already applied for",
                    CODE_FIX_ID,
                    flush=True,
                )

            # Drop from pending if still listed
            pending = [(f, s, g) for f, s, g in pending if s != slug]
            save_state(
                state_path,
                done,
                failed,
                {k: v[0] for k, v in in_flight.items()},
                force_rerun=force_rerun,
            )

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
            if slug in done:
                print("REFUSE push of DONE job", slug, flush=True)
                pending = [(f, s, g) for f, s, g in pending if s != slug]
                continue
            # Never restart a train that Kaggle already marks COMPLETE,
            # unless force_rerun says that COMPLETE was a stub/bad run.
            if "train" in slug and slug not in force_rerun and status(slug) == "COMPLETE":
                print("adopt COMPLETE before push", slug, flush=True)
                done.add(slug)
                pending = [(f, s, g) for f, s, g in pending if s != slug]
                save_state(
                    state_path,
                    done,
                    failed,
                    {k: v[0] for k, v in in_flight.items()},
                    force_rerun=force_rerun,
                )
                continue
            if slug in force_rerun:
                print("FORCE RERUN", slug, "(ignoring Kaggle COMPLETE)", flush=True)
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
            save_state(
                state_path,
                done,
                failed,
                {k: v[0] for k, v in in_flight.items()},
                force_rerun=force_rerun,
            )

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
