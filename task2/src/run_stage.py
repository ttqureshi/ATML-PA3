from __future__ import annotations

"""Unified entrypoint for local and Kaggle stage execution.

Stages:
  prepare_data    - download COCO/VQAv2 annotations, select splits, download images
  cache_features  - SigLIP FP16 patch cache, then free the vision encoder
  train           - continued 3-stage run with dev eval
  eval_final      - greedy final-test evaluation, four image conditions
  analyze         - tables, figures, qualitative pairs, STATUS
  pipeline        - prepare_data + cache_features + train + eval_final + analyze
  smoke           - the same pipeline on the tiny smoke subset
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(module: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    print(">>", " ".join(cmd), flush=True)
    art = Path(os.environ.get("ATML_ARTIFACTS_ROOT", "results"))
    art.mkdir(parents=True, exist_ok=True)
    log_path = art / "subprocess_last.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)
    except Exception:
        import traceback

        detail = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        (art / "failure.txt").write_text(
            "CMD: " + " ".join(cmd) + "\n\n" + traceback.format_exc() + "\n\n--- subprocess log ---\n" + detail,
            encoding="utf-8",
        )
        raise


def _child_args(args: argparse.Namespace, *, smoke: bool) -> list[str]:
    extra = ["--artifacts-root", args.artifacts_root]
    if smoke:
        extra.append("--smoke")
    if args.smoke_steps is not None:
        extra.extend(["--smoke-steps", str(args.smoke_steps)])
    if args.fresh:
        extra.append("--fresh")
    return extra


def _assert_smoke(root: Path) -> None:
    required = [
        "logs/train_metrics.jsonl",
        "logs/dev_metrics.jsonl",
        "raw/features/features_index.json",
        "raw/data/selected_ids.json",
        "raw/data/splits.json",
        "configs_used/connector_calibration.json",
        "raw/eval/final/summary.json",
        "raw/eval/final/per_example.jsonl",
        "raw/eval/final/generations.jsonl",
        "tables/final_results.md",
        "tables/final_results.csv",
        "figures/learning_curves.png",
        "figures/final_scores.png",
        "qualitative/complementary_examples.json",
        "resources/mixture_counts.json",
        "resources/train_resources.json",
        "resources/feature_cache.json",
        "checkpoints/smoke/stage3_end/connector.pt",
        "STATUS.md",
    ]
    missing = [rel for rel in required if not (root / rel).exists()]
    if missing:
        raise SystemExit("smoke missing artifacts: " + ", ".join(missing))
    train_rows = [
        json.loads(line)
        for line in (root / "logs/train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not train_rows or any(not math.isfinite(float(row["loss"])) for row in train_rows):
        raise SystemExit("smoke train loss missing or non-finite")
    dev_rows = [
        json.loads(line)
        for line in (root / "logs/dev_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updates = {int(row["update"]) for row in dev_rows}
    if 0 not in updates:
        raise SystemExit(f"smoke dev eval missing the pre-training point; saw {sorted(updates)}")
    print("SMOKE OK", {"train_rows": len(train_rows), "dev_rows": len(dev_rows)}, flush=True)


def _write_fingerprint(artifacts: Path) -> None:
    def _text(rel: str) -> str:
        return (ROOT / rel).read_text(encoding="utf-8")

    vlm = _text("src/models/vlm.py")
    siglip = _text("src/models/siglip.py")
    trainer = _text("src/train/trainer.py")
    forward_body = vlm.split("def run_lm_forward", 1)[-1].split("\ndef ", 1)[0]
    payload = {
        "enable_thinking_false": "enable_thinking=False" in vlm or '"enable_thinking": False' in vlm,
        "run_lm_forward": "def run_lm_forward" in vlm,
        "no_grad_not_wrapping_lm": "with torch.no_grad" not in forward_body and "torch.no_grad()" not in forward_body,
        "output_scale": "output_scale" in _text("src/models/connector.py"),
        "layernorm_once": "exactly once" in siglip,
        "example_weighted_loss": "def example_weighted_loss" in vlm,
        "unscale_before_clip": "unscale_" in trainer and "clip_grad_norm_" in trainer,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "code_fingerprint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("CODE_FINGERPRINT", payload, flush=True)
    if not all(payload.values()):
        raise SystemExit(f"Code fingerprint failed: {payload}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--artifacts-root", default=os.environ.get("ATML_ARTIFACTS_ROOT", str(ROOT / "results")))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    stage = args.stage
    smoke = stage == "smoke" or args.smoke or os.environ.get("ATML_SMOKE", "").strip().lower() in {"1", "true", "yes"}
    if stage == "smoke":
        smoke = True
        os.environ["ATML_SMOKE"] = "1"
    os.environ["ATML_ARTIFACTS_ROOT"] = args.artifacts_root
    _write_fingerprint(Path(args.artifacts_root))
    extra = _child_args(args, smoke=smoke)
    if stage in {"prepare_data", "pipeline", "smoke"}:
        _run("src.data.prepare_splits", extra)
    if stage in {"cache_features", "pipeline", "smoke"}:
        _run("src.data.feature_cache", extra)
    if stage in {"train", "pipeline", "smoke"}:
        _run("src.train.run_train", extra)
    if stage in {"eval_final", "pipeline", "smoke"}:
        _run("src.eval.run_eval", extra)
    if stage in {"analyze", "pipeline", "smoke"}:
        _run("src.analysis.run_analysis", extra)
    if stage == "smoke":
        _assert_smoke(Path(args.artifacts_root))
    elif stage not in {"prepare_data", "cache_features", "train", "eval_final", "analyze", "pipeline"}:
        raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
