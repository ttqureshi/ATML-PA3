from __future__ import annotations

"""Unified entrypoint for Kaggle / local stage execution.

Stages:
  smoke              - tiny end-to-end train+sanity for full_ft and lora_r4
  prepare_data       - MetaMathQA 40k filter + probe set
  train:<variant>    - one of full_ft, full_ft_l2sp, lora_r4, lora_r16, oplora
  eval:<condition>   - one evaluation condition or 'all_required'
  diagnostics        - weight/activation diagnostics
  qualitative        - 168 greedy responses
  analyze            - tables + figures for the report
  pipeline_prep      - prepare_data + baseline eval + probe
  pipeline_post      - eval all available + diagnostics + qualitative + analyze
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(module: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module] + (extra or [])
    print(">>", " ".join(cmd), flush=True)
    from pathlib import Path as _P
    art = _P(os.environ.get("ATML_ARTIFACTS_ROOT", "/kaggle/working/results"))
    art.mkdir(parents=True, exist_ok=True)
    log_path = art / "subprocess_last.log"
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        # Also echo the log to notebook stdout for Kaggle visibility
        try:
            print(log_path.read_text(encoding="utf-8", errors="replace")[-20000:], flush=True)
        except Exception:
            pass
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except Exception:
        import traceback
        detail = ""
        if log_path.exists():
            detail = log_path.read_text(encoding="utf-8", errors="replace")
        (art / "failure.txt").write_text(
            "CMD: " + " ".join(cmd) + "\n\n" + traceback.format_exc() + "\n\n--- subprocess log ---\n" + detail,
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--artifacts-root", default=os.environ.get("ATML_ARTIFACTS_ROOT", "/kaggle/working/results"))
    parser.add_argument("--smoke-steps", type=int, default=4)
    args = parser.parse_args()

    os.environ["ATML_ARTIFACTS_ROOT"] = args.artifacts_root
    art = ["--artifacts-root", args.artifacts_root]
    stage = args.stage

    if stage == "prepare_data":
        _run("src.data.prepare_metamath", art)
        _run("src.diagnostics.probe_set", art)
    elif stage.startswith("train:"):
        variant = stage.split(":", 1)[1]
        data_file = Path(args.artifacts_root) / "raw" / "data" / "metamath_40k.jsonl"
        if not data_file.exists():
            print("Prepared data missing; running prepare_data first", flush=True)
            _run("src.data.prepare_metamath", art)
            _run("src.diagnostics.probe_set", art)
        _run("src.train.run_train", ["--variant", variant] + art)
    elif stage.startswith("eval:"):
        cond = stage.split(":", 1)[1]
        _run("src.eval.run_eval", ["--condition", cond] + art)
    elif stage == "diagnostics":
        _run("src.diagnostics.run_diagnostics", art)
    elif stage == "qualitative":
        _run("src.qualitative.run_qualitative", art)
    elif stage == "analyze":
        _run("src.analysis.run_analysis", art)
    elif stage == "smoke":
        # Tiny data slice so the GPU session validates the training stack quickly.
        smoke_n = max(64, int(args.smoke_steps) * 32)
        _run("src.data.prepare_metamath", art + ["--num-examples", str(smoke_n)])
        # LoRA-only smoke keeps artifacts small enough to download; Full FT is validated
        # by the dedicated train:full_ft kernel.
        _run(
            "src.train.run_train",
            ["--variant", "lora_r4", "--smoke-steps", str(args.smoke_steps)] + art,
        )
        # Drop bulky smoke checkpoints; keep logs/metrics only.
        import shutil
        from pathlib import Path as _P
        ckpt = _P(args.artifacts_root) / "checkpoints"
        if ckpt.exists():
            shutil.rmtree(ckpt, ignore_errors=True)
        print("SMOKE OK", flush=True)
    elif stage == "pipeline_prep":
        _run("src.data.prepare_metamath", art)
        _run("src.diagnostics.probe_set", art)
        _run("src.eval.run_eval", ["--condition", "baseline"] + art)
        _run("src.analysis.run_analysis", art)
    elif stage == "pipeline_post":
        _run("src.eval.run_eval", ["--condition", "all_required"] + art)
        _run("src.diagnostics.run_diagnostics", art)
        _run("src.qualitative.run_qualitative", art)
        _run("src.analysis.run_analysis", art)
    else:
        raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
