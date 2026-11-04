from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_shared_config
from src.common.logging_utils import setup_logger, write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root
from src.common.seed import set_seed
from src.eval.model_loader import load_eval_model
from src.eval.run_eval import EVAL_CONDITIONS, find_checkpoint


def load_prompts(yaml_path: Path) -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        items = yaml.safe_load(f)
    prompts = []
    for item in items:
        qid = item["id"]
        if qid.endswith("_json"):
            fmt = "json"
            base = qid[: -len("_json")]
        elif qid.endswith("_template"):
            fmt = "template"
            base = qid[: -len("_template")]
        else:
            fmt = "plain"
            base = qid
        text = item["paraphrases"][0]
        system = item.get("system")
        prompts.append(
            {
                "id": qid,
                "base_id": base,
                "format": fmt,
                "user": text,
                "system": system,
            }
        )
    return prompts


@torch.no_grad()
def generate_one(model, tokenizer, prompt: dict, max_new_tokens: int, device) -> dict:
    messages = []
    if prompt.get("system"):
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["user"]})
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    if isinstance(input_ids, str):
        input_ids = tokenizer(input_ids, return_tensors="pt").input_ids
    elif hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    elif not torch.is_tensor(input_ids):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, input_ids.size(1) :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    truncated = int(gen.size(0) >= max_new_tokens)
    return {
        "prompt_id": prompt["id"],
        "base_id": prompt["base_id"],
        "format": prompt["format"],
        "user": prompt["user"],
        "system": prompt.get("system"),
        "response": text,
        "n_new_tokens": int(gen.size(0)),
        "truncated": bool(truncated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=str, default=None)
    parser.add_argument("--condition", type=str, default="all")
    args = parser.parse_args()

    repo = find_repo_root()
    shared = load_shared_config(repo)
    set_seed(shared["seed"])
    dirs = ensure_result_dirs(resolve_artifacts_root(shared, args.artifacts_root))
    logger = setup_logger("qualitative", dirs["logs"] / "qualitative.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts_path = repo / shared["evaluation"]["qualitative"]["prompts_file"]
    prompts = load_prompts(prompts_path)
    write_json(dirs["qualitative"] / "prompts_used.json", prompts)
    max_new = int(shared["evaluation"]["qualitative"]["max_new_tokens"])

    conditions = list(EVAL_CONDITIONS.keys()) if args.condition == "all" else [args.condition]
    all_outputs = {}

    for cond in conditions:
        spec = EVAL_CONDITIONS[cond]
        ckpt = None
        if spec["variant"] is not None:
            try:
                ckpt = find_checkpoint(dirs, spec["variant"], spec["step"])
            except FileNotFoundError as e:
                logger.warning("%s", e)
                continue
        logger.info("Generating qualitative responses for %s", cond)
        model, tokenizer = load_eval_model(
            model_name=shared["model_name"],
            method=spec["method"],
            checkpoint_dir=ckpt,
            device=device,
            merge_adapters=True,
        )
        rows = []
        for p in prompts:
            rows.append(generate_one(model, tokenizer, p, max_new, device))
        write_json(dirs["qualitative"] / f"responses_{cond}.json", rows)
        all_outputs[cond] = rows
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Group by question/format for report prep
    grouped = {}
    for cond, rows in all_outputs.items():
        for row in rows:
            key = f"{row['base_id']}::{row['format']}"
            grouped.setdefault(key, {})[cond] = {
                "response": row["response"],
                "truncated": row["truncated"],
                "n_new_tokens": row["n_new_tokens"],
            }
    write_json(dirs["qualitative"] / "responses_grouped.json", grouped)

    # Identify annotation subset: first and last plain-question base ids
    plain_bases = []
    for p in prompts:
        if p["format"] == "plain" and p["base_id"] not in plain_bases:
            plain_bases.append(p["base_id"])
    annotate_bases = [plain_bases[0], plain_bases[-1]] if plain_bases else []
    annotation = {}
    for base in annotate_bases:
        for fmt in ("plain", "json", "template"):
            key = f"{base}::{fmt}"
            if key in grouped:
                annotation[key] = grouped[key]
    write_json(
        dirs["qualitative"] / "annotation_subset_42.json",
        {
            "bases": annotate_bases,
            "n_expected": 2 * 3 * 7,
            "n_present": sum(len(v) for v in annotation.values()),
            "items": annotation,
        },
    )
    logger.info("Qualitative generation complete")


if __name__ == "__main__":
    main()
