from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_boxed():
    from src.data.boxed import extract_last_boxed

    assert extract_last_boxed(r"foo \boxed{42} bar") == "42"
    assert extract_last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_last_boxed(r"\boxed{a} then \boxed{b}") == "b"
    assert extract_last_boxed("no box") is None
    # Nested braces
    assert extract_last_boxed(r"\boxed{{a}+{b}}") == "{a}+{b}"
    print("OK boxed")


def test_splits():
    from src.data.splits import make_question_splits

    ids = [f"q{i:04d}" for i in range(1000)]
    splits = make_question_splits(ids, seed=6304)
    assert len(splits["dev_ids"]) == 200
    assert len(splits["eval_ids"]) == 800
    assert len(splits["live_ids"]) == 50
    assert splits["live_ids"] == splits["eval_ids"][:50]
    # Deterministic
    splits2 = make_question_splits(ids, seed=6304)
    assert splits["permuted_ids"] == splits2["permuted_ids"]
    print("OK splits", splits["live_ids"][:3])


def test_rwc_and_ranks():
    from src.metrics.scores import average_ranks, rwc_weights
    from src.select.rules import rank_weighted_consensus, self_consistency

    ranks = average_ranks([0.5, 0.9, 0.9, 0.1])
    # Two tied at top -> average ranks 1.5, 1.5
    assert sorted(ranks)[:2] == [1.5, 1.5] or set(ranks) >= {1.5}
    w = rwc_weights([1, 2, 3])
    assert abs(w[0] - 1.0) < 1e-9
    assert abs(w[2] - 0.0) < 1e-9

    answers = ["A", "A", "B", "B"]
    scores = [0.1, 0.9, 0.8, 0.2]
    order = [0, 1, 2, 3]
    rwc = rank_weighted_consensus(answers, order, scores, dataset="mathvista", metadata={"answer_type": "multi_choice"})
    assert rwc["answer"] in {"A", "B"}
    sc = self_consistency(answers, order, dataset="mathvista", metadata={"answer_type": "multi_choice"})
    assert sc["answer"] in {"A", "B"}
    # N=1
    assert self_consistency(["Z"], [0], "math")["answer"] == "Z"
    print("OK rwc", rwc["margin"])


def test_visconf():
    from src.metrics.scores import visconf_scores

    kl = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.5, 0.5, 10.0, 10.0]),
    ]
    vis = [0.2, 0.8]
    out = visconf_scores(kl, vis, kl)  # reuse kl shape as fake vi tokens
    assert len(out["Z"]) == 2
    assert all(0 <= c <= 1 for c in out["C"])
    # Recompute on subset (prefix calibration changes)
    out2 = visconf_scores(kl[:1], vis[:1])
    assert len(out2["Z"]) == 1
    print("OK visconf", out["Z"], out2["Z"])


def test_adaptive():
    from src.adaptive.policies import agreement_should_stop, quality_aware_should_stop, select_thresholds, simulate_adaptive

    answers = ["1", "1", "1", "2", "2", "2", "2", "2"] + ["1"] * 24
    order = list(range(32))
    ok, info = agreement_should_stop(answers, order[:8], "math", tau=0.5)
    assert isinstance(ok, bool)
    scores = [float(i) for i in range(32)]
    ok2, info2 = quality_aware_should_stop(answers, order[:8], "math", scores, tau=0.5, gamma=0.0)
    assert isinstance(ok2, bool)

    def scores_fn(prefix):
        return scores

    sim = simulate_adaptive(
        answers,
        order,
        "math",
        [10] * 32,
        scores_fn,
        policy="agreement",
        tau=0.9,
        check_ns=[4, 8, 16],
        max_n=32,
    )
    assert sim["stop_n"] in {4, 8, 16, 32}
    assert sim["token_cost"] == 10 * sim["stop_n"]

    selected = select_thresholds(
        [
            {"policy": "always32", "tau": 1.0, "gamma": 0.0, "accuracy": 0.80, "mean_token_cost": 3200},
            {"policy": "agreement", "tau": 0.75, "gamma": 0.0, "accuracy": 0.795, "mean_token_cost": 1200},
            {"policy": "agreement", "tau": 0.5, "gamma": 0.0, "accuracy": 0.70, "mean_token_cost": 400},
        ],
        fixed32_accuracy=0.80,
        accuracy_tolerance_pp=1.0,
    )
    assert selected["tau"] == 0.75
    print("OK adaptive", sim["stop_n"], selected)


def test_synthetic_offline_smoke():
    from src.data.splits import make_question_splits, make_rollout_orders
    from src.eval.protocol import evaluate_offline_dataset

    rows = []
    for qi in range(12):
        sid = f"s{qi:03d}"
        gold = "42"
        for rid in range(8):
            ans = "42" if rid % 3 == 0 else ("7" if rid % 3 == 1 else None)
            rows.append(
                {
                    "dataset": "math",
                    "sample_id": sid,
                    "rollout_id": rid,
                    "extracted_answer": ans,
                    "extracted_answer_used": ans if ans else None,
                    "gold_answer": gold,
                    "gold_answer_used": gold,
                    "is_correct": ans == gold,
                    "logp": [ -0.1 * (rid + 1)] * 5,
                    "kl_u_p": [0.5 + 0.1 * rid] * 5,
                    "num_generated_tokens": 5 + rid,
                }
            )
    df = pd.DataFrame(rows)
    # Attach correctness properly for missing
    df["is_correct"] = [a == "42" for a in df["extracted_answer_used"].tolist()]
    ids = sorted(df["sample_id"].unique().tolist())
    # Tiny split override
    splits = {
        "dev_ids": ids[:4],
        "eval_ids": ids[4:],
        "live_ids": ids[4:6],
        "seed": 6304,
    }
    orders = make_rollout_orders(n_rollouts=8, n_orders=3, seed=6304, sample_ids=ids)
    ev = evaluate_offline_dataset(df, splits, orders, dataset="math", budgets=[1, 2, 4, 8], split_name="eval")
    assert ev["summary"]["n_questions"] == len(splits["eval_ids"])
    assert 8 in ev["summary"]["budgets"]
    print("OK synthetic_offline", ev["summary"]["budgets"][8]["methods"]["SelfConsistency"])


def main():
    test_boxed()
    test_splits()
    test_rwc_and_ranks()
    test_visconf()
    test_adaptive()
    test_synthetic_offline_smoke()
    print("ALL_TESTS_OK")


if __name__ == "__main__":
    main()
