"""CPU-only preflight checks before spending Kaggle GPU time."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_normalize() -> None:
    from src.data.vqa_normalize import normalize_prediction, normalize_vqa_answer

    assert normalize_vqa_answer("A dog.") == "dog"
    assert normalize_vqa_answer("two") == "2"
    assert normalize_prediction("<think>x</think>yes") == "yes"
    print("OK normalize")


def test_soft_score() -> None:
    from src.eval.scoring import soft_vqa_accuracy

    refs = ["yes"] * 8 + ["no"] * 2
    score = soft_vqa_accuracy("yes", refs)
    assert 0.0 < score <= 1.0
    assert soft_vqa_accuracy("maybe", refs) == 0.0
    print("OK soft_score", score)


def test_schedule() -> None:
    from src.common.config import apply_runtime_overrides, load_shared_config
    from src.train.trainer import lr_multiplier, warmup_steps

    cfg = apply_runtime_overrides(load_shared_config(ROOT), smoke=False)
    updates = [int(s["updates"]) for s in cfg["training"]["stages"]]
    assert updates == [2250, 1500, 750]
    assert sum(updates) * 16 == 72000
    warm = warmup_steps(2250, 0.05)
    assert warm == 113  # round(0.05 * 2250) via floor(x + 0.5)
    assert abs(lr_multiplier(0, warm, 2250, 0.1) - 1.0 / warm) < 1e-9
    assert abs(lr_multiplier(warm - 1, warm, 2250, 0.1) - 1.0) < 1e-9
    assert abs(lr_multiplier(2249, warm, 2250, 0.1) - 0.1) < 1e-6
    print("OK schedule", warm)


def test_pool_and_connector_grad() -> None:
    from src.models.connector import Connector, pool_2x2_spatial

    feats = torch.randn(2, 196, 768)
    pooled = pool_2x2_spatial(feats)
    assert pooled.shape == (2, 49, 768)
    connector = Connector(num_visual_tokens=196)
    embed = nn.Embedding(32, 1024)
    for p in embed.parameters():
        p.requires_grad = False
    ids = torch.randint(0, 32, (2, 5))
    prefix = embed(ids)
    visual = connector(feats)
    visual = visual.to(dtype=prefix.dtype)
    joined = torch.cat([prefix, visual], dim=1)
    assert joined.requires_grad
    loss = joined.float().pow(2).mean()
    loss.backward()
    assert connector.fc1.weight.grad is not None
    assert torch.isfinite(connector.fc1.weight.grad).all()
    assert float(connector.fc1.weight.grad.abs().sum()) > 0
    print("OK connector_grad", float(connector.fc1.weight.grad.abs().sum()))


def test_samplers() -> None:
    from src.data.samplers import build_mixture_draws, build_stage1_order, flatten_draws

    order = build_stage1_order(32, 2, 6304)
    assert len(order) == 64
    draws = build_mixture_draws(
        8,
        {"presence": 0.5, "counting": 0.25, "caption": 0.25},
        {"presence": 4, "counting": 4, "caption": 10},
        6304,
    )
    pools = {
        "caption": [{"example_id": f"c{i}"} for i in range(10)],
        "presence": [
            {"pair_id": f"p{i}", "examples": [{"example_id": f"p{i}a"}, {"example_id": f"p{i}b"}]}
            for i in range(4)
        ],
        "counting": [
            {"pair_id": f"c{i}", "examples": [{"example_id": f"c{i}a"}, {"example_id": f"c{i}b"}]}
            for i in range(4)
        ],
    }
    flat = flatten_draws(draws, pools)
    assert len(flat) == 16
    print("OK samplers", len(draws), len(flat))


def test_fingerprint() -> None:
    from src.run_stage import _write_fingerprint

    out = ROOT / "results_preflight"
    out.mkdir(parents=True, exist_ok=True)
    _write_fingerprint(out)
    print("OK fingerprint")


def test_coerce_token_ids() -> None:
    from src.models.vlm import _coerce_token_ids

    assert _coerce_token_ids([1, 2, 3]) == [1, 2, 3]
    assert _coerce_token_ids({"input_ids": [4, 5]}) == [4, 5]
    assert _coerce_token_ids([[6, 7]]) == [6, 7]
    assert _coerce_token_ids(torch.tensor([8, 9])) == [8, 9]
    print("OK coerce_token_ids")


def main() -> None:
    test_normalize()
    test_soft_score()
    test_schedule()
    test_pool_and_connector_grad()
    test_samplers()
    test_fingerprint()
    test_coerce_token_ids()
    print("ALL_PREFLIGHT_OK")



if __name__ == "__main__":
    main()
