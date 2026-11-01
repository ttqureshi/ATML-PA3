from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_shared_config, save_config_snapshot
from src.common.logging_utils import setup_logger, write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root
from src.common.seed import set_seed
from src.eval.model_loader import compare_merged_unmerged_logits, load_eval_model


EVAL_CONDITIONS = {
    "baseline": {"variant": None, "method": "baseline", "step": 0},
    "full_ft_midpoint": {"variant": "full_ft", "method": "full_ft", "step": 625},
    "full_ft_final": {"variant": "full_ft", "method": "full_ft", "step": 1250},
    "full_ft_l2sp_final": {"variant": "full_ft_l2sp", "method": "full_ft_l2sp", "step": 1250},
    "lora_r4_final": {"variant": "lora_r4", "method": "lora", "step": 1250},
    "lora_r16_final": {"variant": "lora_r16", "method": "lora", "step": 1250},
    "oplora_final": {"variant": "oplora", "method": "oplora", "step": 1250},
}


def find_checkpoint(dirs: dict[str, Path], variant: str, step: int) -> Path:
    root = dirs["checkpoints"] / variant
    if not root.exists():
        raise FileNotFoundError(f"No checkpoints for {variant} under {root}")
    matches = sorted(root.glob(f"step_{step:04d}_*"))
    if not matches:
        matches = sorted(root.glob("step_*"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint step={step} for {variant}")
    return matches[-1]


def materialize_hf_model_dir(
    *,
    model_name: str,
    method: str,
    checkpoint_dir: Path | None,
    device: torch.device,
    out_dir: Path,
) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_eval_model(
        model_name=model_name,
        method=method,
        checkpoint_dir=checkpoint_dir,
        device=device,
        merge_adapters=True,
    )
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def _persist_eval_blob(out_dir: Path, blob: dict[str, Any] | None, combined_results: dict, samples: dict) -> None:
    if not blob:
        return
    combined_results.update(blob.get("results", {}) or {})
    if blob.get("samples"):
        samples.update(blob["samples"])
    write_json(out_dir / "metrics.json", combined_results)
    for task, rows in samples.items():
        write_json(out_dir / f"samples_{task}.json", rows)
        if "gsm8k" in task:
            slim = []
            for i, row in enumerate(rows):
                slim.append(
                    {
                        "idx": i,
                        "doc_id": row.get("doc_id"),
                        "target": row.get("target"),
                        "filtered_resps": row.get("filtered_resps"),
                        "exact_match": row.get("exact_match"),
                        "truncated": bool(row.get("truncated", False)),
                    }
                )
            write_json(out_dir / "gsm8k_extractions.json", slim)


def run_lm_eval(model_path: str, out_dir: Path, batch_size: str | int = "auto") -> dict[str, Any]:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_results: dict[str, Any] = {}
    samples: dict[str, Any] = {}

    lm_like = HFLM(
        pretrained=model_path,
        dtype="float16",
        trust_remote_code=True,
        batch_size=batch_size,
    )
    like_results = evaluator.simple_evaluate(
        model=lm_like,
        tasks=["hellaswag", "boolq", "arc_easy"],
        num_fewshot=0,
        batch_size=batch_size,
        log_samples=True,
    )
    _persist_eval_blob(out_dir, like_results, combined_results, samples)
    del lm_like
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Do NOT pass max_new_tokens into HFLM(...): recent transformers+lm-eval stacks
    # forward it into Qwen3ForCausalLM.__init__ and crash. Put generation limits in gen_kwargs.
    lm_gen = HFLM(
        pretrained=model_path,
        dtype="float16",
        trust_remote_code=True,
        batch_size=1,
        apply_chat_template=True,
    )
    gsm_results = evaluator.simple_evaluate(
        model=lm_gen,
        tasks=["gsm8k_cot"],
        num_fewshot=0,
        batch_size=1,
        log_samples=True,
        gen_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 1024},
    )
    _persist_eval_blob(out_dir, gsm_results, combined_results, samples)
    del lm_gen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return combined_results


def evaluate_condition(
    condition: str,
    dirs: dict[str, Path],
    shared: dict,
    device: torch.device,
    keep_merged_dir: bool = False,
) -> dict[str, Any]:
    spec = EVAL_CONDITIONS[condition]
    model_name = shared["model_name"]
    ckpt = None
    if spec["variant"] is not None:
        ckpt = find_checkpoint(dirs, spec["variant"], spec["step"])

    out_dir = dirs["eval"] / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec["method"] in {"lora", "oplora"} and ckpt is not None:
        _, tokenizer = load_eval_model(
            model_name=model_name,
            method="baseline",
            checkpoint_dir=None,
            device=device,
        )
        merge_report = compare_merged_unmerged_logits(
            model_name=model_name,
            method=spec["method"],
            checkpoint_dir=ckpt,
            tokenizer=tokenizer,
            device=device,
        )
        write_json(out_dir / "merge_check.json", merge_report)

    merged_dir = out_dir / "hf_model"
    materialize_hf_model_dir(
        model_name=model_name,
        method=spec["method"],
        checkpoint_dir=ckpt,
        device=device,
        out_dir=merged_dir,
    )
    metrics = run_lm_eval(str(merged_dir), out_dir)
    payload = {"condition": condition, "checkpoint": str(ckpt) if ckpt else None, "metrics": metrics}
    write_json(out_dir / "summary.json", payload)
    if not keep_merged_dir and merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ATML PA3 Task1 evaluation")
    parser.add_argument("--condition", required=True, choices=list(EVAL_CONDITIONS.keys()) + ["all_required"])
    parser.add_argument("--artifacts-root", type=str, default=None)
    parser.add_argument("--keep-merged-dir", action="store_true")
    args = parser.parse_args()

    repo_root = find_repo_root()
    shared = load_shared_config(repo_root)
    set_seed(shared["seed"])
    artifacts = resolve_artifacts_root(shared, args.artifacts_root)
    dirs = ensure_result_dirs(artifacts)
    logger = setup_logger("eval", dirs["logs"] / "eval.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conditions = list(EVAL_CONDITIONS.keys()) if args.condition == "all_required" else [args.condition]
    all_results = {}
    for cond in conditions:
        logger.info("Evaluating condition=%s", cond)
        try:
            all_results[cond] = evaluate_condition(
                cond, dirs, shared, device, keep_merged_dir=args.keep_merged_dir
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", cond, e)
            all_results[cond] = {"error": str(e)}

    write_json(dirs["eval"] / "all_conditions_summary.json", all_results)
    save_config_snapshot({"shared": shared, "conditions": conditions}, dirs["configs_used"] / "eval.yaml")
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
