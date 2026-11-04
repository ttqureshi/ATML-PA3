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
        # Stream child output live to Kaggle logs AND to subprocess_last.log.
        # (Previously we redirected only to a file, so Logs tab looked frozen for hours.)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with open(log_path, "w", encoding="utf-8") as logf:
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
                logf.write(line)
                logf.flush()
                print(line, end="", flush=True)
            rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
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

    # Fingerprint so we can verify Kaggle ran the intended code revision.
    try:
        from pathlib import Path as _P
        import json as _json
        trainer_txt = (_P(ROOT) / "src" / "train" / "trainer.py").read_text(encoding="utf-8")
        lora_txt = (_P(ROOT) / "src" / "models" / "lora.py").read_text(encoding="utf-8")
        fp = {
            "stage": stage,
            "full_ft_uses_scaler": 'self.use_scaler = self.device.type == "cuda"' in trainer_txt,
            "full_ft_fp32_masters": "torch_dtype=torch.float32" in trainer_txt,
            "has_skip_abort": "max_consecutive_skips" in trainer_txt,
            "has_torchao_neutralize": "_neutralize_torchao_check" in lora_txt,
            "has_smoke_full_ft": "smoke_full_ft"
            in (_P(ROOT) / "src" / "run_stage.py").read_text(encoding="utf-8"),
        }
        out = _P(args.artifacts_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "code_fingerprint.json").write_text(_json.dumps(fp, indent=2), encoding="utf-8")
        print("CODE_FINGERPRINT", fp, flush=True)
    except Exception as e:
        print("fingerprint_failed", e, flush=True)

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
        # by the dedicated smoke_full_ft / train:full_ft kernels.
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
    elif stage == "smoke_full_ft":
        # Required gate before burning ~3h×2 on full_ft / full_ft_l2sp retrains.
        steps = max(20, int(args.smoke_steps))
        smoke_n = max(64, steps * 32)
        _run("src.data.prepare_metamath", art + ["--num-examples", str(smoke_n)])
        _run(
            "src.train.run_train",
            ["--variant", "full_ft", "--smoke-steps", str(steps)] + art,
        )
        # Assert finite loss / no skips from the metrics jsonl.
        import json as _json
        from pathlib import Path as _P
        metrics = _P(args.artifacts_root) / "logs" / "train_metrics_full_ft.jsonl"
        summary = _P(args.artifacts_root) / "resources" / "train_summary_full_ft.json"
        if not metrics.exists():
            raise SystemExit(f"smoke_full_ft missing metrics: {metrics}")
        rows = [
            _json.loads(line)
            for line in metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise SystemExit("smoke_full_ft empty metrics")
        import math as _math
        bad_loss = [
            r for r in rows if not _math.isfinite(float(r.get("sft_loss", float("nan"))))
        ]
        # GradScaler may skip a few early steps while adjusting scale; require the
        # tail of the run to be clean and overall skip rate to stay small.
        tail = rows[-max(10, len(rows) // 2) :]
        tail_skips = sum(1 for r in tail if r.get("skipped"))
        skips = sum(1 for r in rows if r.get("skipped"))
        skipped_total = 0
        if summary.exists():
            skipped_total = int(
                _json.loads(summary.read_text(encoding="utf-8")).get("skipped_updates", 0)
            )
        print(
            "SMOKE_FULL_FT_CHECK",
            {
                "steps": len(rows),
                "skips": skips,
                "tail_skips": tail_skips,
                "skipped_updates": skipped_total,
                "bad_loss": len(bad_loss),
                "last_loss": rows[-1].get("sft_loss"),
            },
            flush=True,
        )
        if bad_loss or tail_skips > 0 or skips > max(5, len(rows) // 5):
            raise SystemExit(
                f"smoke_full_ft FAILED: skips={skips} tail_skips={tail_skips} "
                f"skipped_updates={skipped_total} nonfinite_loss_rows={len(bad_loss)}"
            )
        # Drop bulky smoke checkpoints; keep logs/metrics only.
        import shutil
        ckpt = _P(args.artifacts_root) / "checkpoints"
        if ckpt.exists():
            shutil.rmtree(ckpt, ignore_errors=True)
        print("SMOKE_FULL_FT OK", flush=True)
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
    elif stage == "post_analyze":
        # After per-condition eval kernels: diagnostics + qualitative + tables.
        # Probe set is required by diagnostics; recreate if eval-only kernels didn't ship it.
        from pathlib import Path as _P
        probe = _P(args.artifacts_root) / "raw" / "probe" / "activation_probe_set.json"
        if not probe.exists():
            print("probe set missing; generating", flush=True)
            _run("src.diagnostics.probe_set", art)
        _run("src.diagnostics.run_diagnostics", art)
        _run("src.qualitative.run_qualitative", art)
        _run("src.analysis.run_analysis", art)
    else:
        raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
