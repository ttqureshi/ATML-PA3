from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.common.config import load_shared_config, save_config_snapshot
from src.common.logging_utils import setup_logger, write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root
from src.common.seed import set_seed
from src.common.text_norm import normalize_question_text


def build_gsm8k_test_set() -> set[str]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return {normalize_question_text(ex["question"]) for ex in ds}


def format_chat_length(
    tokenizer,
    question: str,
    response: str,
    enable_thinking: bool = False,
) -> int:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    # Qwen3 supports enable_thinking in chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def select_metamath_examples(
    tokenizer,
    num_examples: int = 40000,
    max_seq_len: int = 2048,
    enable_thinking: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gsm8k_test = build_gsm8k_test_set()
    raw = load_dataset("meta-math/MetaMathQA", split="train")

    kept: list[dict[str, Any]] = []
    stats = {
        "source_rows_scanned": 0,
        "excluded_empty": 0,
        "excluded_overlength": 0,
        "excluded_gsm8k_overlap": 0,
        "kept": 0,
        "normalization": "nfkc+lower+strip+collapse_ws+strip_quotes",
        "max_seq_len": max_seq_len,
        "num_examples_requested": num_examples,
    }

    for idx, ex in enumerate(tqdm(raw, desc="Filtering MetaMathQA")):
        stats["source_rows_scanned"] = idx + 1
        question = (ex.get("query") or "").strip()
        response = (ex.get("response") or "").strip()
        original_q = (ex.get("original_question") or "").strip()

        if not question or not response:
            stats["excluded_empty"] += 1
            continue

        norm_query = normalize_question_text(question)
        norm_original = normalize_question_text(original_q) if original_q else ""
        if norm_query in gsm8k_test or (norm_original and norm_original in gsm8k_test):
            stats["excluded_gsm8k_overlap"] += 1
            continue

        try:
            tok_len = format_chat_length(
                tokenizer, question, response, enable_thinking=enable_thinking
            )
        except Exception:
            stats["excluded_overlength"] += 1
            continue

        if tok_len > max_seq_len:
            stats["excluded_overlength"] += 1
            continue

        kept.append(
            {
                "source_index": idx,
                "query": question,
                "response": response,
                "original_question": original_q,
                "type": ex.get("type"),
                "formatted_length": tok_len,
            }
        )
        if len(kept) >= num_examples:
            break

    stats["kept"] = len(kept)
    if len(kept) < num_examples:
        raise RuntimeError(
            f"Only collected {len(kept)} eligible examples; needed {num_examples}."
        )
    return kept, stats


def save_prepared_data(examples: list[dict[str, Any]], stats: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "metamath_40k.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    ids = [ex["source_index"] for ex in examples]
    write_json(out_dir / "example_ids.json", ids)
    write_json(out_dir / "filter_stats.json", stats)
    write_json(
        out_dir / "data_manifest.json",
        {
            "dataset": "meta-math/MetaMathQA",
            "num_examples": len(examples),
            "order": "source_order_no_shuffle",
            "truncation": "none_on_answers",
            "files": {
                "jsonl": str(jsonl_path.name),
                "ids": "example_ids.json",
                "stats": "filter_stats.json",
            },
            "stats": stats,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MetaMathQA 40k training subset")
    parser.add_argument("--artifacts-root", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num-examples", type=int, default=None, help="Override example count (smoke tests)")
    args = parser.parse_args()

    repo_root = find_repo_root()
    shared = load_shared_config(repo_root)
    set_seed(shared["seed"])
    artifacts = resolve_artifacts_root(shared, args.artifacts_root)
    dirs = ensure_result_dirs(artifacts)
    logger = setup_logger("prepare_metamath", dirs["logs"] / "prepare_metamath.log")

    model_name = args.model_name or shared["model_name"]
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    n_examples = int(args.num_examples or shared["data"]["num_examples"])
    examples, stats = select_metamath_examples(
        tokenizer=tokenizer,
        num_examples=n_examples,
        max_seq_len=shared["data"]["max_seq_len"],
        enable_thinking=shared.get("enable_thinking", False),
    )
    save_prepared_data(examples, stats, dirs["data"])
    save_config_snapshot(
        {"shared": shared, "stats": stats},
        dirs["configs_used"] / "prepare_metamath.yaml",
    )
    logger.info("Saved %d examples to %s", len(examples), dirs["data"])
    logger.info("Filter stats: %s", json.dumps(stats))


if __name__ == "__main__":
    main()
