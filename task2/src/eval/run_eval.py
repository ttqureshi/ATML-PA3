from __future__ import annotations

import time
from pathlib import Path

import torch

from src.common.config import parse_common_args, prepare_runtime, total_updates
from src.common.logging_utils import JsonlLogger, read_json, write_json
from src.common.paths import checkpoint_dir
from src.common.seed import set_seed
from src.data.feature_cache import load_feature_store
from src.data.samplers import caption_example, pair_to_examples
from src.eval.scoring import score_prediction, summarize_prediction_rows
from src.models.lora import is_peft_model
from src.models.vlm import (
    collate_examples,
    discover_prompt_split,
    end_of_turn_id,
    greedy_generate,
    load_language_model,
    load_tokenizer,
    per_example_nll,
    prompt_embeds_for_generation,
    run_lm_forward,
)
from src.train.checkpointing import load_connector_weights, resolve_final_weights


def _prompt_split(tokenizer):
    cached = getattr(tokenizer, "_atml_prompt_split", None)
    if cached is None:
        cached = discover_prompt_split(tokenizer)
        tokenizer._atml_prompt_split = cached
    return cached


def _batches(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _with_condition(example: dict, condition: str, mismatch_id: int | None = None) -> dict:
    view = dict(example)
    view["condition"] = condition
    if condition == "correct":
        return view
    if condition == "complementary":
        if view.get("partner_image_id") is None:
            raise RuntimeError(f"{view['example_id']} has no complementary image")
        view["image_id"] = int(view["partner_image_id"])
        return view
    if condition == "no_image":
        view["image_id"] = None
        return view
    if condition == "mismatched":
        if mismatch_id is None:
            raise RuntimeError("mismatched condition requires a donor image")
        view["image_id"] = int(mismatch_id)
        return view
    raise ValueError(condition)


def fixed_mismatch_map(examples: list[dict]) -> dict[str, int]:
    """Cyclic donor: next pair_id in sorted order, its lower image id, excluding the pair itself."""
    pair_images: dict[str, set[int]] = {}
    for example in examples:
        pair_images.setdefault(example["pair_id"], set()).add(int(example["image_id"]))
    pair_ids = sorted(pair_images)
    if len(pair_ids) < 2:
        raise RuntimeError("Mismatched-image mapping needs at least two pairs")
    mapping: dict[str, int] = {}
    for index, pair_id in enumerate(pair_ids):
        donor = pair_ids[(index + 1) % len(pair_ids)]
        own = pair_images[pair_id]
        donor_choices = [img for img in sorted(pair_images[donor]) if img not in own]
        if not donor_choices:
            raise RuntimeError(f"No mismatched image for pair {pair_id}")
        donor_img = donor_choices[0]
        for example in examples:
            if example["pair_id"] == pair_id:
                mapping[example["example_id"]] = donor_img
    return mapping


def _eval_examples_from_pairs(pairs: list[dict], image_split: str, suite: str) -> list[dict]:
    rows = []
    for pair in pairs:
        sides = pair_to_examples(pair, image_split=image_split)
        for index, side in enumerate(sides):
            row = dict(side)
            row["example_index"] = index
            row["suite"] = suite
            row["question"] = row["user_text"]
            rows.append(row)
    return rows


def dev_sets(splits: dict, cfg: dict) -> dict[str, list[dict]]:
    prompt = cfg["data"]["caption_prompt"]
    return {
        "caption": [caption_example(row, prompt) for row in splits["captions_dev"]],
        "grounding": _eval_examples_from_pairs(splits["grounding_dev"], "val2014", "grounding"),
        "natural": _eval_examples_from_pairs(splits["natural_dev"], "val2014", "natural"),
    }


def final_sets(splits: dict) -> dict[str, list[dict]]:
    return {
        "grounding": _eval_examples_from_pairs(splits["grounding_test"], "val2014", "grounding"),
        "natural": _eval_examples_from_pairs(splits["natural_test"], "val2014", "natural"),
    }


def _collate_kwargs(cfg: dict, tokenizer, eot_id: int, include_eot: bool, device):
    prefix, suffix = _prompt_split(tokenizer)
    train = cfg["training"]
    return dict(
        prefix_ids=prefix,
        suffix_ids=suffix,
        eot_id=eot_id,
        question_limit=int(train["question_tokens"]),
        caption_limit=int(train["caption_tokens"]),
        answer_limit=int(train["short_answer_tokens"]),
        include_eot=include_eot,
        device=device,
    )


def reference_nlls(lm, connector, tokenizer, feature_store, examples, cfg, device, *, include_eot: bool) -> list[float]:
    """Example-weighted teacher-forced NLL. ``include_eot`` is true for caption loss only."""
    if not examples:
        return []
    eot_id = end_of_turn_id(tokenizer)
    kwargs = _collate_kwargs(cfg, tokenizer, eot_id, include_eot, device)
    batch_size = int(cfg["evaluation"]["gen_batch_size"])
    values: list[float] = []
    was_training = lm.training
    connector_training = connector.training
    lm.eval()
    connector.eval()
    try:
        for chunk in _batches(examples, batch_size):
            with torch.inference_mode():
                batch = collate_examples(
                    lm, connector, tokenizer, feature_store, chunk, **kwargs
                )
                outputs = run_lm_forward(
                    lm,
                    batch.inputs_embeds,
                    batch.attention_mask,
                    batch.position_ids,
                    train_mode=False,
                )
                values.extend(per_example_nll(outputs.logits, batch.labels))
    finally:
        if was_training:
            lm.train()
        if connector_training:
            connector.train()
    if len(values) != len(examples):
        raise RuntimeError("NLL count does not match the example count")
    return values


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _paired_delta(correct: dict[str, float], other: dict[str, float]) -> float:
    keys = sorted(set(correct) & set(other))
    if not keys:
        return float("nan")
    return _mean([other[key] - correct[key] for key in keys])


def evaluate_dev(lm, connector, tokenizer, feature_store, splits, cfg, device, update: int, stage_name: str) -> dict:
    """Caption reference loss plus answer NLL under correct, complementary, and no-image."""
    suites = dev_sets(splits, cfg)
    caption_losses = reference_nlls(
        lm, connector, tokenizer, feature_store, suites["caption"], cfg, device, include_eot=True
    )
    record: dict = {
        "update": int(update),
        "stage": stage_name,
        "caption_loss": _mean(caption_losses),
        "caption_examples": len(caption_losses),
        "suites": {},
    }
    for suite_name in ("grounding", "natural"):
        base = suites[suite_name]
        per_condition = {}
        for condition in ("correct", "complementary", "no_image"):
            views = [_with_condition(example, condition) for example in base]
            values = reference_nlls(
                lm, connector, tokenizer, feature_store, views, cfg, device, include_eot=False
            )
            per_condition[condition] = {
                example["example_id"]: value for example, value in zip(base, values)
            }
        correct = per_condition["correct"]
        suite_record = {
            "n_examples": len(base),
            "nll": {condition: _mean(list(values.values())) for condition, values in per_condition.items()},
            "delta_nll": _paired_delta(correct, per_condition["complementary"]),
            "delta_nll_no_image": _paired_delta(correct, per_condition["no_image"]),
            "by_answer_type": {},
        }
        by_type: dict[str, list[dict]] = {}
        for example in base:
            by_type.setdefault(str(example.get("answer_type")), []).append(example)
        for answer_type, rows in sorted(by_type.items()):
            ids = [row["example_id"] for row in rows]
            suite_record["by_answer_type"][answer_type] = {
                "nll": {
                    condition: _mean([per_condition[condition][example_id] for example_id in ids])
                    for condition in per_condition
                },
                "delta_nll": _mean(
                    [
                        per_condition["complementary"][example_id] - per_condition["correct"][example_id]
                        for example_id in ids
                    ]
                ),
            }
        record["suites"][suite_name] = suite_record
    return record


def _load_eval_model(cfg: dict, dirs: dict[str, Path], device: torch.device):
    from peft import PeftModel

    from src.models.connector import Connector

    tokenizer = load_tokenizer(cfg["model"]["language_model"])
    lm = load_language_model(cfg["model"]["language_model"], device)
    connector = Connector(
        in_dim=int(cfg["model"]["vision_dim"]),
        hidden_dim=int(cfg["model"]["connector_hidden"]),
        out_dim=int(cfg["model"]["lm_hidden"]),
        num_visual_tokens=int(cfg["model"]["num_visual_tokens"]),
    )
    root = checkpoint_dir(dirs, bool(cfg.get("smoke_active")))
    weights = resolve_final_weights(root, total_updates(cfg))
    load_connector_weights(connector, weights)
    lora_dir = weights / "lora"
    if lora_dir.exists():
        lm = PeftModel.from_pretrained(lm, lora_dir)
        lm.to(device)
    connector.to(device=device, dtype=torch.float32)
    lm.eval()
    connector.eval()
    print("EVAL_WEIGHTS", weights, "lora", lora_dir.exists(), "peft", is_peft_model(lm), flush=True)
    return lm, connector, tokenizer, weights


def run_final(cfg: dict, dirs: dict[str, Path], device: torch.device) -> dict:
    set_seed(int(cfg["seed"]))
    splits = read_json(dirs["data"] / "splits.json")
    feature_store = load_feature_store(dirs)
    lm, connector, tokenizer, weights = _load_eval_model(cfg, dirs, device)
    eot_id = end_of_turn_id(tokenizer)
    eos_ids = {eot_id}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    out_dir = dirs["eval"] / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Truncate previous final files so a rerun does not append.
    for name in ("generations.jsonl", "per_example.jsonl"):
        path = out_dir / name
        if path.exists():
            path.unlink()
    gen_logger = JsonlLogger(out_dir / "generations.jsonl")
    metric_logger = JsonlLogger(out_dir / "per_example.jsonl")
    rows: list[dict] = []
    mismatch_maps = {}
    batch_size = int(cfg["evaluation"]["gen_batch_size"])
    max_new = int(cfg["evaluation"]["max_new_tokens"])
    nll_kwargs = _collate_kwargs(cfg, tokenizer, eot_id, False, device)
    for suite_name, base_examples in final_sets(splits).items():
        mismatch = fixed_mismatch_map(base_examples)
        mismatch_maps[suite_name] = mismatch
        for condition in ("correct", "complementary", "mismatched", "no_image"):
            views = [
                _with_condition(
                    example,
                    condition,
                    mismatch.get(example["example_id"]) if condition == "mismatched" else None,
                )
                for example in base_examples
            ]
            print(f"FINAL {suite_name} {condition} n={len(views)}", flush=True)
            nlls: list[float] = []
            for chunk in _batches(views, batch_size):
                with torch.inference_mode():
                    batch = collate_examples(lm, connector, tokenizer, feature_store, chunk, **nll_kwargs)
                    outputs = run_lm_forward(
                        lm, batch.inputs_embeds, batch.attention_mask, batch.position_ids, train_mode=False
                    )
                    nlls.extend(per_example_nll(outputs.logits, batch.labels))
            cursor = 0
            for chunk in _batches(views, batch_size):
                with torch.inference_mode():
                    prompts = prompt_embeds_for_generation(
                        lm,
                        connector,
                        tokenizer,
                        feature_store,
                        chunk,
                        prefix_ids=nll_kwargs["prefix_ids"],
                        suffix_ids=nll_kwargs["suffix_ids"],
                        question_limit=nll_kwargs["question_limit"],
                        device=device,
                    )
                    started = time.perf_counter()
                    generated = greedy_generate(lm, prompts, max_new_tokens=max_new, eos_ids=eos_ids)
                    elapsed = time.perf_counter() - started
                per_time = elapsed / max(1, len(chunk))
                for example, token_ids in zip(chunk, generated):
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    scored = score_prediction(suite_name, text, example)
                    original = base_examples[cursor]
                    row = {
                        "suite": suite_name,
                        "condition": condition,
                        "pair_id": original["pair_id"],
                        "example_id": original["example_id"],
                        "example_index": original["example_index"],
                        "question": original["user_text"],
                        "answer_type": original.get("answer_type"),
                        "source": original.get("source"),
                        "image_id": example.get("image_id"),
                        "partner_image_id": original.get("partner_image_id"),
                        "target": original["target_text"],
                        "canonical": original.get("canonical"),
                        "partner_canonical": original.get("partner_canonical"),
                        "answers": original.get("answers"),
                        "prediction": text,
                        "prediction_norm": scored["prediction_norm"],
                        "empty": scored["empty"],
                        "exact": scored["exact"],
                        "soft": scored["soft"],
                        "primary": scored["primary"],
                        "follow_partner": scored["follow_partner"],
                        "follow_original": scored["follow_original"],
                        "nll": nlls[cursor],
                        "gen_time_sec": per_time,
                    }
                    rows.append(row)
                    gen_logger.log(
                        {
                            "suite": suite_name,
                            "condition": condition,
                            "pair_id": original["pair_id"],
                            "example_id": original["example_id"],
                            "prediction": text,
                            "target": original["target_text"],
                        }
                    )
                    metric_logger.log(row)
                    cursor += 1
            if cursor != len(views):
                raise RuntimeError("Generation count drifted from the eval set")
    summary = summarize_prediction_rows(
        rows,
        int(cfg["evaluation"]["bootstrap_resamples"]),
        int(cfg["evaluation"]["bootstrap_seed"]),
    )
    summary["weights"] = str(weights)
    summary["max_new_tokens"] = max_new
    summary["gen_batch_size"] = batch_size
    summary["decoding"] = "greedy"
    summary["enable_thinking"] = False
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "mismatched_map.json", mismatch_maps)
    gen_times = [row["gen_time_sec"] for row in rows]
    write_json(
        dirs["resources"] / "generation.json",
        {
            "answers": len(rows),
            "mean_gen_time_sec": _mean(gen_times),
            "max_new_tokens": max_new,
            "batch_size": batch_size,
            "kv_cache": False,
        },
    )
    print("FINAL_EVAL done", "rows", len(rows), flush=True)
    return summary


def main() -> None:
    parser = parse_common_args("Greedy final-test evaluation under four image conditions.")
    args = parser.parse_args()
    _root, cfg, dirs = prepare_runtime(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_final(cfg, dirs, device)


if __name__ == "__main__":
    main()
