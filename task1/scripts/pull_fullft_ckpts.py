#!/usr/bin/env python3
"""Pull healthy full_ft / full_ft_l2sp checkpoints from salmanrifa Kaggle outputs to Drive.

Gates: train_summary skipped_updates small, metrics finite, safetensors finite sample.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

USER = "salmanrifa"
SLUGS = {
    "full_ft": f"{USER}/atml-pa3-t1-train-fullft",
    "full_ft_l2sp": f"{USER}/atml-pa3-t1-train-full-ft-l2sp",
}
DRIVE_CKPT = Path(r"g:\My Drive\MS AI\Semester_3\ATML\PAs\ATML-PA3\task1\colab_data\checkpoints")
LOCAL_PULL = Path(r"C:\Users\PMLS\kaggle_pulls")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def status(slug: str) -> str:
    p = run(["kaggle", "kernels", "status", slug])
    return ((p.stdout or "") + (p.stderr or "")).strip()


def download(slug: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        p = run(["kaggle", "kernels", "output", slug, "-p", str(dest)])
        print("download", slug, "attempt", i + 1, "rc", p.returncode)
        print(((p.stdout or "") + (p.stderr or ""))[-400:])
        # Enough if we have train summary or checkpoints
        if any(dest.rglob("train_summary_*.json")) or any(dest.rglob("model.safetensors")):
            return
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"Failed to download usable output for {slug}")


def check_metrics(pull_root: Path, variant: str) -> dict:
    metrics = pull_root / "results" / "logs" / f"train_metrics_{variant}.jsonl"
    summary = pull_root / "results" / "resources" / f"train_summary_{variant}.json"
    rows = []
    if metrics.exists():
        rows = [json.loads(l) for l in metrics.read_text(encoding="utf-8").splitlines() if l.strip()]
    summ = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    skips = int(summ.get("skipped_updates", sum(1 for r in rows if r.get("skipped"))))
    bad = sum(1 for r in rows if not math.isfinite(float(r.get("sft_loss", float("nan")))))
    tail = rows[-50:] if rows else []
    tail_skips = sum(1 for r in tail if r.get("skipped"))
    ok = bool(rows) and bad == 0 and skips < max(50, len(rows) // 10) and tail_skips == 0
    return {
        "ok": ok,
        "steps": len(rows),
        "skipped_updates": skips,
        "tail_skips": tail_skips,
        "bad_loss": bad,
        "last_loss": rows[-1].get("sft_loss") if rows else None,
        "peak_mem_gb": summ.get("peak_mem_gb"),
    }


def check_weights_finite(model_dir: Path, max_tensors: int = 8) -> bool:
    from safetensors import safe_open
    import torch

    path = model_dir / "model.safetensors"
    if not path.exists():
        return False
    checked = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if t.is_floating_point():
                if not torch.isfinite(t).all():
                    print("nonfinite", key)
                    return False
                checked += 1
                if checked >= max_tensors:
                    break
    return checked > 0


def copy_variant(pull_root: Path, variant: str, needed_steps: list[int]) -> list[str]:
    src = pull_root / "results" / "checkpoints" / variant
    if not src.exists():
        # sometimes nested differently
        matches = list(pull_root.rglob(f"checkpoints/{variant}"))
        if not matches:
            raise FileNotFoundError(f"No checkpoints for {variant} under {pull_root}")
        src = matches[0]
    dst_root = DRIVE_CKPT / variant
    copied = []
    for step in needed_steps:
        matches = sorted(src.glob(f"step_{step:04d}_*"))
        if not matches:
            raise FileNotFoundError(f"Missing step {step} under {src}")
        step_dir = matches[-1]
        model_dir = step_dir / "model"
        if not check_weights_finite(model_dir):
            raise RuntimeError(f"Non-finite or missing weights at {model_dir}")
        dest = dst_root / step_dir.name
        backup = None
        if dest.exists():
            backup = dst_root / f"{step_dir.name}__NAN_BACKUP"
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            dest.rename(backup)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Copy model/ + meta only (skip bulky trainer_state if present)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, dest / "model")
        meta = step_dir / "checkpoint_meta.json"
        if meta.exists():
            shutil.copy2(meta, dest / "checkpoint_meta.json")
        copied.append(str(dest))
        print("copied", dest, "backup", backup)
    return copied


def main() -> int:
    assert "salmanrifa" in run(["kaggle", "config", "view"]).stdout
    LOCAL_PULL.mkdir(parents=True, exist_ok=True)
    report = {}
    for variant, slug in SLUGS.items():
        st = status(slug)
        print(variant, st)
        if "COMPLETE" not in st:
            print(f"{variant} not COMPLETE yet")
            return 2
        dest = LOCAL_PULL / slug.split("/")[-1]
        download(slug, dest)
        gate = check_metrics(dest, variant)
        report[variant] = gate
        print("gate", variant, gate)
        if not gate["ok"]:
            print("GATE FAILED", variant)
            return 3
        steps = [625, 1250] if variant == "full_ft" else [1250]
        report[variant]["copied"] = copy_variant(dest, variant, steps)
    out = DRIVE_CKPT.parent.parent / "results_colab" / "FULLFT_RETRAIN_GATE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
