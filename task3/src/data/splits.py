from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def make_question_splits(
    sample_ids: list[str],
    seed: int = 6304,
    n_dev: int = 200,
    n_eval: int = 800,
    n_live: int = 50,
) -> dict[str, Any]:
    """Sort IDs lexicographically, permute with default_rng(seed), split.

    First n_dev = development; remaining = evaluation; first n_live of eval for live gen.
    """
    ids = sorted(str(x) for x in sample_ids)
    if len(ids) < n_dev + n_eval:
        raise ValueError(f"Need at least {n_dev + n_eval} question IDs, got {len(ids)}")
    if len(ids) > n_dev + n_eval:
        # Keep a fixed 1000-question working set: first 1000 after lexical sort.
        ids = ids[: n_dev + n_eval]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    ordered = [ids[i] for i in perm.tolist()]
    dev_ids = ordered[:n_dev]
    eval_ids = ordered[n_dev : n_dev + n_eval]
    live_ids = eval_ids[:n_live]
    return {
        "seed": seed,
        "n_dev": n_dev,
        "n_eval": n_eval,
        "n_live": n_live,
        "sorted_ids": ids,
        "permuted_ids": ordered,
        "dev_ids": dev_ids,
        "eval_ids": eval_ids,
        "live_ids": live_ids,
    }


def make_rollout_orders(
    n_rollouts: int = 32,
    n_orders: int = 10,
    seed: int = 6304,
    sample_ids: list[str] | None = None,
) -> dict[str, list[list[int]]]:
    """Ten seeded random permutations of rollout IDs per question.

    Uses a per-question derived seed so orders are reusable and deterministic.
    """
    base = np.random.default_rng(seed)
    # Global shared orders (same across questions) — assignment: generate ten seeded
    # permutations of the 32 rollout IDs per question. We derive per-question streams
    # from (seed, sample_id) so they are stable across methods.
    if sample_ids is None:
        orders = []
        for i in range(n_orders):
            rng = np.random.default_rng(int(base.integers(0, 2**31 - 1)))
            orders.append(rng.permutation(n_rollouts).tolist())
        return {"__shared__": orders}

    out: dict[str, list[list[int]]] = {}
    for sid in sample_ids:
        # Stable 32-bit seed from master seed + sample id.
        sid_seed = int(np.random.default_rng([seed, abs(hash(str(sid))) % (2**31)]).integers(0, 2**31 - 1))
        rng = np.random.default_rng(sid_seed)
        out[str(sid)] = [rng.permutation(n_rollouts).tolist() for _ in range(n_orders)]
    return out


def save_splits(splits: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
