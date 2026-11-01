from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_shared_config
from src.common.logging_utils import read_json, setup_logger, write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root
from src.common.seed import set_seed
from src.diagnostics.activations import cache_down_proj_inputs, measure_activation_corrections
from src.diagnostics.weight_space import analyze_down_proj_layers
from src.eval.model_loader import load_eval_model
from src.eval.run_eval import EVAL_CONDITIONS, find_checkpoint


DIAG_CONDITIONS = [
    "full_ft_midpoint",
    "full_ft_final",
    "full_ft_l2sp_final",
    "lora_r4_final",
    "lora_r16_final",
    "oplora_final",
]


def build_prompt_meta(probe: dict) -> list[dict]:
    prompts = []
    for ex in probe["gsm8k"]:
        prompts.append(
            {
                "key": f"gsm8k_{ex['idx']}",
                "text": ex["question"],
                "domain": "math",
                "use_chat": True,
            }
        )
    for ex in probe["hellaswag"]:
        prompts.append(
            {
                "key": f"hellaswag_{ex['idx']}",
                "text": ex["ctx"],
                "domain": "non_math",
                "use_chat": False,
            }
        )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight-space and activation diagnostics")
    parser.add_argument("--artifacts-root", type=str, default=None)
    parser.add_argument("--condition", type=str, default="all")
    args = parser.parse_args()

    repo = find_repo_root()
    shared = load_shared_config(repo)
    set_seed(shared["seed"])
    dirs = ensure_result_dirs(resolve_artifacts_root(shared, args.artifacts_root))
    logger = setup_logger("diagnostics", dirs["logs"] / "diagnostics.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_path = dirs["probe"] / "activation_probe_set.json"
    if not probe_path.exists():
        raise FileNotFoundError("Probe set missing; run probe_set.py first")
    probe = read_json(probe_path)
    prompt_meta = build_prompt_meta(probe)
    layers = list(shared["evaluation"]["layers_down_proj"])
    k = int(shared["evaluation"]["protection_rank_k"])

    base_model, tokenizer = load_eval_model(
        model_name=shared["model_name"],
        method="baseline",
        checkpoint_dir=None,
        device=device,
    )
    logger.info("Caching base-model down_proj activations on probe set")
    caches = cache_down_proj_inputs(base_model, tokenizer, prompt_meta, layers, device)
    # Persist compact cache stats only (full tensors can be large)
    write_json(
        dirs["diagnostics"] / "probe_cache_meta.json",
        {
            "num_prompts": len(caches),
            "layers": layers,
            "hidden": int(next(iter(caches.values()))[layers[0]].numel()),
        },
    )
    # Save tensors
    torch.save(caches, dirs["diagnostics"] / "base_down_proj_input_cache.pt")

    conditions = DIAG_CONDITIONS if args.condition == "all" else [args.condition]
    all_weight = {}
    all_act = {}

    for cond in conditions:
        spec = EVAL_CONDITIONS[cond]
        try:
            ckpt = find_checkpoint(dirs, spec["variant"], spec["step"])
        except FileNotFoundError as e:
            logger.warning("%s", e)
            continue
        logger.info("Diagnostics for %s (%s)", cond, ckpt)
        # For adapters, load unmerged so delta comes from adapter factors / OPLoRA
        merge = spec["method"] in {"full_ft", "full_ft_l2sp"}
        trained, _ = load_eval_model(
            model_name=shared["model_name"],
            method=spec["method"],
            checkpoint_dir=ckpt,
            device=device,
            merge_adapters=merge,
        )
        weight_rows = analyze_down_proj_layers(
            base_model=base_model,
            trained_model=trained,
            method=spec["method"],
            layers=layers,
            k=k,
        )
        act_rows = measure_activation_corrections(
            base_model=base_model,
            trained_model=trained,
            method=spec["method"],
            caches=caches,
            prompt_meta=prompt_meta,
            layers=layers,
        )
        # Aggregate activation R by domain/layer
        summary = {}
        for row in act_rows:
            key = (row["layer"], row["domain"])
            summary.setdefault(key, []).append(row["R"])
        act_summary = [
            {
                "layer": layer,
                "domain": domain,
                "R_mean": float(sum(vals) / len(vals)),
                "R_median": float(sorted(vals)[len(vals) // 2]),
            }
            for (layer, domain), vals in sorted(summary.items())
        ]
        write_json(dirs["diagnostics"] / f"weight_space_{cond}.json", weight_rows)
        write_json(dirs["diagnostics"] / f"activations_{cond}.json", {"rows": act_rows, "summary": act_summary})
        all_weight[cond] = weight_rows
        all_act[cond] = act_summary
        del trained
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(dirs["diagnostics"] / "weight_space_all.json", all_weight)
    write_json(dirs["diagnostics"] / "activations_summary_all.json", all_act)
    logger.info("Diagnostics complete")


if __name__ == "__main__":
    main()
