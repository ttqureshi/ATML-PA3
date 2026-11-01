from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m src.train.run_train` from task1 root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import (
    load_shared_config,
    load_variant_config,
    merge_configs,
    save_config_snapshot,
)
from src.common.logging_utils import setup_logger
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root
from src.common.seed import set_seed
from src.train.trainer import SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="ATML PA3 Task1 training")
    parser.add_argument("--variant", required=True, choices=[
        "full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora"
    ])
    parser.add_argument("--artifacts-root", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--smoke-steps", type=int, default=None, help="Override updates for smoke tests")
    args = parser.parse_args()

    repo_root = find_repo_root()
    shared = load_shared_config(repo_root)
    variant = load_variant_config(args.variant, repo_root)
    if args.smoke_steps is not None:
        shared = dict(shared)
        shared["training"] = dict(shared["training"])
        shared["training"]["num_updates"] = int(args.smoke_steps)
        # Avoid writing multi-GB checkpoints during smoke unless the budget is large.
        if int(args.smoke_steps) < 50:
            shared["training"]["checkpoint_update_counts"] = []
        else:
            shared["training"]["checkpoint_update_counts"] = [
                u for u in shared["training"]["checkpoint_update_counts"] if u <= int(args.smoke_steps)
            ] or [int(args.smoke_steps)]

    set_seed(shared["seed"])
    artifacts = resolve_artifacts_root(shared, args.artifacts_root)
    dirs = ensure_result_dirs(artifacts)
    logger = setup_logger(
        f"train.{args.variant}",
        dirs["logs"] / f"train_{args.variant}.log",
    )

    cfg = merge_configs(shared, variant)
    save_config_snapshot(cfg, dirs["configs_used"] / f"train_{args.variant}.yaml")
    logger.info("Starting variant=%s method=%s", variant["name"], variant["method"])

    try:
        logger.info("Constructing trainer...")
        trainer = SFTTrainer(
            shared_cfg=shared,
            variant_cfg=variant,
            artifacts_dirs=dirs,
            resume_from=Path(args.resume_from) if args.resume_from else None,
        )
        logger.info("Trainer ready; beginning optimization loop")
        summary = trainer.train()
    except Exception:
        import traceback
        err = dirs["logs"] / f"train_exception_{args.variant}.txt"
        err.write_text(traceback.format_exc(), encoding="utf-8")
        logger.exception("Training failed; wrote %s", err)
        raise
    logger.info("Finished: %s", summary)


if __name__ == "__main__":
    main()
