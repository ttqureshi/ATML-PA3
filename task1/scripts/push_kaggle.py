from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USER = "mtayyabtahir"
DATASET_SLUG = "atml-pa3-task1-code"


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def package_code_dataset(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    include = [
        "src",
        "configs",
        "assets",
        "requirements.txt",
        "README.md",
    ]
    for name in include:
        src = ROOT / name
        dst = staging / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"),
            )
        elif src.exists():
            shutil.copy2(src, dst)

    # Ensure results placeholder exists so path helpers behave offline
    (staging / "results" / ".gitkeep").parent.mkdir(parents=True, exist_ok=True)
    (staging / "results" / ".gitkeep").write_text("")

    meta = {
        "title": "ATML PA3 Task1 Code",
        "id": f"{USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def push_dataset(staging: Path, create: bool) -> None:
    # --dir-mode zip uploads folders; Kaggle serves them as a normal directory tree.
    if create:
        run(["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "zip"])
    else:
        run([
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(staging),
            "-m",
            "ATML PA3 Task1 code update",
            "--dir-mode",
            "zip",
        ])


def ensure_notebooks() -> None:
    run(["python", str(ROOT / "scripts" / "make_notebook.py")])


def push_kernel(folder: Path, push_only: bool = False) -> None:
    cmd = ["kaggle", "kernels", "push", "-p", str(folder)]
    run(cmd)
    if not push_only:
        # status check
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        kid = meta["id"]
        run(["kaggle", "kernels", "status", kid])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-dataset", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=["smoke"],
        help="Kernel slug suffixes under kaggle/kernels (e.g. smoke train-full-ft)",
    )
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--push-only", action="store_true")
    args = parser.parse_args()

    ensure_notebooks()

    if not args.skip_dataset:
        staging = Path(tempfile.mkdtemp(prefix="atml_pa3_code_"))
        try:
            package_code_dataset(staging)
            # Detect whether dataset already exists
            create = args.create_dataset
            if not create:
                ls = subprocess.run(
                    ["kaggle", "datasets", "list", "-m", "--search", DATASET_SLUG],
                    capture_output=True,
                    text=True,
                )
                create = DATASET_SLUG not in (ls.stdout or "")
            push_dataset(staging, create=create)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    kernel_names = list(args.kernels)
    if args.all_train:
        kernel_names = [
            "smoke",
            "pipeline-prep",
            "train-full-ft",
            "train-full-ft-l2sp",
            "train-lora-r4",
            "train-lora-r16",
            "train-oplora",
            "pipeline-post",
        ]

    for name in kernel_names:
        # folders are named atml-pa3-t1-<slug>
        slug = name if name.startswith("atml-pa3-t1-") else f"atml-pa3-t1-{name}"
        folder = ROOT / "kaggle" / "kernels" / slug
        if not folder.exists():
            # try with generated naming from make_notebook
            candidates = list((ROOT / "kaggle" / "kernels").glob(f"*{name}*"))
            if not candidates:
                raise FileNotFoundError(folder)
            folder = candidates[0]
        print("Pushing kernel", folder)
        push_kernel(folder, push_only=args.push_only)


if __name__ == "__main__":
    main()
