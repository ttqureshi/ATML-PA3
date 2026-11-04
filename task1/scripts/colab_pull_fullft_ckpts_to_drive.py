"""Run in Colab (with Drive mounted) to pull healthy Full-FT ckpts onto Drive.

Uses Kaggle API under salmanrifa. Prefers the lean CPU export kernel output;
falls back to raw train kernels.

Drive target (edit if needed):
  /content/drive/MyDrive/MS AI/Semester_3/ATML/PAs/ATML-PA3/task1/colab_data/checkpoints
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# Colab: from google.colab import drive; drive.mount('/content/drive')
DRIVE_CKPT = Path(
    "/content/drive/MyDrive/MS AI/Semester_3/ATML/PAs/ATML-PA3/task1/colab_data/checkpoints"
)
PULL = Path("/content/kaggle_pull_fullft")
EXPORT_SLUG = "salmanrifa/atml-pa3-t1-export-ckpts"
TRAIN_SLUGS = {
    "full_ft": "salmanrifa/atml-pa3-t1-train-fullft",
    "full_ft_l2sp": "salmanrifa/atml-pa3-t1-train-full-ft-l2sp",
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def kaggle_output(slug: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    run(["kaggle", "kernels", "output", slug, "-p", str(dest), "--force"])


def finite_ok(model_dir: Path, n: int = 8) -> bool:
    from safetensors import safe_open
    import torch

    path = model_dir / "model.safetensors"
    if not path.exists() or path.stat().st_size < 100_000_000:
        return False
    checked = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if not t.is_floating_point():
                continue
            if not torch.isfinite(t).all():
                print("nonfinite", key)
                return False
            checked += 1
            if checked >= n:
                break
    return checked > 0


def install_step(src_step: Path, variant: str) -> Path:
    dest = DRIVE_CKPT / variant / src_step.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        bak = DRIVE_CKPT / variant / f"{src_step.name}__OLD_BAD"
        if bak.exists():
            shutil.rmtree(bak)
        dest.rename(bak)
        print("backed_up", bak)
    dest.mkdir(parents=True)
    shutil.copytree(src_step / "model", dest / "model")
    meta = src_step / "checkpoint_meta.json"
    if meta.exists():
        shutil.copy2(meta, dest / "checkpoint_meta.json")
    assert finite_ok(dest / "model")
    print("installed", dest)
    return dest


def main() -> None:
    assert os.path.exists("/content/drive"), "Mount Drive first"
    DRIVE_CKPT.mkdir(parents=True, exist_ok=True)
    PULL.mkdir(parents=True, exist_ok=True)

    # Prefer lean export kernel
    export_dir = PULL / "export"
    try:
        kaggle_output(EXPORT_SLUG, export_dir)
        root = export_dir / "export_ckpts"
        if not root.exists():
            # sometimes files land at export_dir/results/... or directly
            hits = list(export_dir.rglob("export_ckpts"))
            root = hits[0] if hits else export_dir
    except Exception as e:
        print("export kernel pull failed, falling back to train kernels:", e)
        root = None

    installed = []
    if root and root.exists():
        for variant in ("full_ft", "full_ft_l2sp"):
            vdir = root / variant
            if not vdir.exists():
                continue
            for step_dir in sorted(vdir.glob("step_*")):
                if finite_ok(step_dir / "model"):
                    installed.append(str(install_step(step_dir, variant)))
    else:
        for variant, slug in TRAIN_SLUGS.items():
            dest = PULL / variant
            kaggle_output(slug, dest)
            ck = next(dest.rglob(f"checkpoints/{variant}"))
            want = {625, 1250} if variant == "full_ft" else {1250}
            for step_dir in sorted(ck.glob("step_*")):
                step = int(step_dir.name.split("_")[1])
                if step not in want:
                    continue
                if finite_ok(step_dir / "model"):
                    installed.append(str(install_step(step_dir, variant)))

    report = {"installed": installed}
    out = DRIVE_CKPT.parent.parent / "results_colab" / "FULLFT_DRIVE_INSTALL.json"
    # parent.parent from colab_data -> task1
    out = Path(
        "/content/drive/MyDrive/MS AI/Semester_3/ATML/PAs/ATML-PA3/task1/results_colab/FULLFT_DRIVE_INSTALL.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DONE", report)


if __name__ == "__main__":
    main()
