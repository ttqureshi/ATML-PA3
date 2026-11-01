from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_shared_config
from src.common.logging_utils import setup_logger, write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root
from src.common.seed import set_seed


def build_probe_set(seed: int = 6304, gsm8k_n: int = 32, hellaswag_n: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")

    gsm_idx = rng.choice(len(gsm), size=gsm8k_n, replace=False)
    hs_idx = rng.choice(len(hs), size=hellaswag_n, replace=False)
    gsm_idx = sorted(int(i) for i in gsm_idx)
    hs_idx = sorted(int(i) for i in hs_idx)

    gsm_examples = []
    for i in gsm_idx:
        ex = gsm[int(i)]
        gsm_examples.append({"idx": int(i), "question": ex["question"], "answer": ex["answer"]})

    hs_examples = []
    for i in hs_idx:
        ex = hs[int(i)]
        ctx = (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip()
        if not ctx:
            ctx = ex.get("ctx", "")
        hs_examples.append(
            {
                "idx": int(i),
                "ctx": ctx,
                "endings": ex.get("endings"),
                "label": ex.get("label"),
            }
        )

    return {
        "seed": seed,
        "gsm8k": gsm_examples,
        "hellaswag": hs_examples,
        "note": "Sampled before inspecting model results; reuse for every model.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=str, default=None)
    args = parser.parse_args()
    repo = find_repo_root()
    shared = load_shared_config(repo)
    set_seed(shared["seed"])
    dirs = ensure_result_dirs(resolve_artifacts_root(shared, args.artifacts_root))
    logger = setup_logger("probe_set", dirs["logs"] / "probe_set.log")
    probe_cfg = shared["evaluation"]["probe"]
    probe = build_probe_set(
        seed=int(probe_cfg["seed"]),
        gsm8k_n=int(probe_cfg["gsm8k_n"]),
        hellaswag_n=int(probe_cfg["hellaswag_n"]),
    )
    write_json(dirs["probe"] / "activation_probe_set.json", probe)
    write_json(
        dirs["probe"] / "activation_probe_ids.json",
        {
            "gsm8k_ids": [e["idx"] for e in probe["gsm8k"]],
            "hellaswag_ids": [e["idx"] for e in probe["hellaswag"]],
        },
    )
    logger.info("Saved probe set with %d GSM8K and %d HellaSwag examples", len(probe["gsm8k"]), len(probe["hellaswag"]))


if __name__ == "__main__":
    main()
