from __future__ import annotations

import random
from typing import Any


def build_stage1_order(n_examples: int, passes: int, seed: int) -> list[int]:
    """Two (or ``passes``) shuffled passes over caption example indices."""
    if n_examples <= 0:
        raise ValueError("caption pool is empty")
    rng = random.Random(seed)
    order: list[int] = []
    for _ in range(max(1, int(passes))):
        chunk = list(range(n_examples))
        rng.shuffle(chunk)
        order.extend(chunk)
    return order


def build_mixture_draws(
    n_draws: int,
    mixture: dict[str, float],
    pool_sizes: dict[str, int],
    seed: int,
) -> list[tuple]:
    """Seeded draws. Each draw is one pair or two caption indices.

    Caption draws are ``("caption", i, j)``. Other draws are ``(source, index)``.
    Sampling is with replacement. Quotas are not enforced.
    """
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    names = list(mixture.keys())
    weights = [float(mixture[name]) for name in names]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"mixture weights must sum to 1, got {sum(weights)}")
    for name in names:
        if name != "caption" and int(pool_sizes.get(name, 0)) <= 0:
            raise ValueError(f"mixture source {name} has an empty pool")
        if name == "caption" and int(pool_sizes.get("caption", 0)) <= 0:
            raise ValueError("caption replay pool is empty")
    rng = random.Random(seed)
    draws: list[tuple] = []
    for _ in range(n_draws):
        source = rng.choices(names, weights=weights, k=1)[0]
        if source == "caption":
            n_cap = int(pool_sizes["caption"])
            draws.append(("caption", rng.randrange(n_cap), rng.randrange(n_cap)))
        else:
            draws.append((source, rng.randrange(int(pool_sizes[source]))))
    return draws


def flatten_draws(draws: list[tuple], pools: dict[str, list[dict]]) -> list[dict]:
    """Expand draws into example dicts. Each draw contributes exactly two examples."""
    examples: list[dict] = []
    for draw in draws:
        source = draw[0]
        if source == "caption":
            examples.append(pools["caption"][draw[1]])
            examples.append(pools["caption"][draw[2]])
            continue
        pair = pools[source][draw[1]]
        pair_examples = pair["examples"]
        if len(pair_examples) != 2:
            raise RuntimeError(f"{source} pair {pair.get('pair_id')} does not have 2 examples")
        examples.extend(pair_examples)
    return examples


class ExampleSampler:
    """Cursor over a deterministic example list. Position is the resume cursor."""

    def __init__(self, examples: list[dict], position: int = 0):
        if not examples:
            raise ValueError("sampler example list is empty")
        self.examples = examples
        self.pos = int(position)

    def take(self, n: int) -> list[dict]:
        if n < 0 or self.pos + n > len(self.examples):
            raise RuntimeError(
                f"sampler exhausted: pos={self.pos} take={n} length={len(self.examples)}"
            )
        batch = self.examples[self.pos : self.pos + n]
        self.pos += n
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"position": self.pos, "length": len(self.examples)}


def pair_to_examples(pair: dict[str, Any], *, image_split: str) -> list[dict]:
    """Two question-image examples that share a question and point at each other."""
    left, right = pair["examples"]
    return [
        _example_from_side(pair, left, right, image_split),
        _example_from_side(pair, right, left, image_split),
    ]


def _example_from_side(pair: dict, side: dict, partner: dict, image_split: str) -> dict:
    source = pair.get("source") or pair.get("task")
    answer_type = pair.get("answer_type") or pair.get("task")
    return {
        "example_id": f"{pair['pair_id']}::{int(side['image_id'])}",
        "pair_id": pair["pair_id"],
        "source": source,
        "image_split": image_split,
        "image_id": int(side["image_id"]),
        "partner_image_id": int(partner["image_id"]),
        "user_text": pair["question"],
        "target_text": str(side["target"]),
        "target_kind": "short",
        "answer_type": answer_type,
        "canonical": side.get("canonical", str(side["target"])),
        "partner_canonical": partner.get("canonical", str(partner["target"])),
        "answers": side.get("answers"),
        "question_id": side.get("question_id"),
    }


def caption_example(row: dict, prompt: str) -> dict:
    return {
        "example_id": row["example_id"],
        "pair_id": None,
        "source": "caption",
        "image_split": row["image_split"],
        "image_id": int(row["image_id"]),
        "partner_image_id": None,
        "user_text": prompt,
        "target_text": row["caption"],
        "target_kind": "caption",
        "answer_type": "caption",
        "canonical": None,
        "partner_canonical": None,
        "answers": None,
        "question_id": row.get("caption_id"),
    }


def load_training_pools(splits: dict, prompt: str) -> dict[str, Any]:
    """Materialize caption rows and pair-example lists used by the samplers."""
    captions = [caption_example(row, prompt) for row in splits["captions_train"]]
    pools = {
        "caption": captions,
        "presence": [],
        "counting": [],
        "natural": [],
    }
    for pair in splits["grounding_train_presence"]:
        pools["presence"].append({"pair_id": pair["pair_id"], "examples": pair_to_examples(pair, image_split="train2014")})
    for pair in splits["grounding_train_counting"]:
        pools["counting"].append({"pair_id": pair["pair_id"], "examples": pair_to_examples(pair, image_split="train2014")})
    for pair in splits["natural_train"]:
        pools["natural"].append({"pair_id": pair["pair_id"], "examples": pair_to_examples(pair, image_split="train2014")})
    return pools


def build_stage_examples(cfg: dict, splits: dict, stage: dict, seed: int) -> list[dict]:
    prompt = cfg["data"]["caption_prompt"]
    pools = load_training_pools(splits, prompt)
    name = stage["name"]
    updates = int(stage["updates"])
    batch = int(cfg["training"]["effective_batch"])
    stage_names = [row["name"] for row in cfg["training"]["stages"]]
    stage_seed = int(seed) + 1000 * (stage_names.index(name) + 1)
    if name == "caption_alignment":
        n_cap = len(pools["caption"])
        passes = int(cfg["data"].get("caption_passes", 2))
        order = build_stage1_order(n_cap, passes, stage_seed)
        needed = updates * batch
        if len(order) < needed:
            raise RuntimeError(
                f"Stage 1 plan has {len(order)} captions but {needed} presentations are required"
            )
        return [pools["caption"][index] for index in order[:needed]]
    n_draws = updates * batch // 2
    if updates * batch % 2 != 0:
        raise RuntimeError("effective batch must be even so each draw stays intact")
    mixture = dict(stage["mixture"])
    pool_sizes = {key: len(value) if key != "caption" else len(pools["caption"]) for key, value in pools.items()}
    # Stage 2 has no natural source. Drop missing keys from the size check by only passing mixture keys.
    sizes = {key: pool_sizes[key] for key in mixture}
    if "caption" in mixture:
        sizes["caption"] = pool_sizes["caption"]
    draws = build_mixture_draws(n_draws, mixture, sizes, stage_seed)
    flat_pools = {
        "caption": pools["caption"],
        "presence": pools["presence"],
        "counting": pools["counting"],
        "natural": pools["natural"],
    }
    examples = flatten_draws(draws, flat_pools)
    if len(examples) != updates * batch:
        raise RuntimeError(f"{name} produced {len(examples)} examples, expected {updates * batch}")
    return examples
