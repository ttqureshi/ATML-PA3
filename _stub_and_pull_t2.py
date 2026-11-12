"""Stub huge Task2 raw annotation outputs, then pull remaining via kaggle CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEST = Path(r"g:\My Drive\MS AI\Semester_3\ATML\PAs\ATML-PA3\task2\results_pipeline_kaggle")
USER_SLUG = "mtayyabtahir/atml-pa3-t2-pipeline"


def list_all_files() -> list[str]:
    names: list[str] = []
    token = None
    for page in range(100):
        cmd = ["kaggle", "kernels", "files", USER_SLUG, "--csv"]
        # older CLI may not support page token via flag; use default pagination print
        # Fall back: parse normal output pages by calling API through CLI repeatedly is hard.
        # Use --page-size if available.
        proc = subprocess.run(
            ["kaggle", "kernels", "files", USER_SLUG, "-v"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        break
    return names


def stub_raw_blobs() -> None:
    raw = DEST / "results" / "raw"
    # Known large annotation artifacts from prior download logs / assignment data prep
    stubs = [
        "results/raw/data/annotations/annotations_train.zip",
        "results/raw/data/annotations/annotations_trainval2014.zip",
        "results/raw/data/annotations/annotations_val.zip",
        "results/raw/data/annotations/pairs_train.zip",
        "results/raw/data/annotations/pairs_val.zip",
        "results/raw/data/annotations/questions_train.zip",
        "results/raw/data/annotations/questions_val.zip",
        "results/raw/data/annotations/coco/annotations/captions_train2014.json",
        "results/raw/data/annotations/coco/annotations/captions_val2014.json",
        "results/raw/data/annotations/coco/annotations/instances_train2014.json",
        "results/raw/data/annotations/coco/annotations/instances_val2014.json",
        "results/raw/data/annotations/coco/annotations/person_keypoints_train2014.json",
        "results/raw/data/annotations/coco/annotations/person_keypoints_val2014.json",
        "results/raw/data/annotation_paths.json",
    ]
    # Also stub any existing partial huge files under raw (>50MB incomplete risk)
    if raw.exists():
        for p in raw.rglob("*"):
            if p.is_file() and p.stat().st_size > 50 * 1024 * 1024:
                print(f"touch existing large {p}", flush=True)
    for rel in stubs:
        path = DEST / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            path.write_bytes(b"STUB_SKIP_DOWNLOAD\n")
            print(f"stubbed {rel}", flush=True)
        else:
            # bump mtime so kaggle prefers local
            path.touch()
            print(f"touched {rel} size={path.stat().st_size}", flush=True)


def main() -> None:
    stub_raw_blobs()
    print("Running kaggle kernels output...", flush=True)
    proc = subprocess.run(
        ["kaggle", "kernels", "output", USER_SLUG, "-p", str(DEST)],
        capture_output=False,
    )
    print("exit", proc.returncode, flush=True)
    found = list(DEST.rglob("*final_results*"))
    print("final_results:", found, flush=True)
    tables = DEST / "results" / "tables"
    if tables.exists():
        print("tables:", list(tables.iterdir()), flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
