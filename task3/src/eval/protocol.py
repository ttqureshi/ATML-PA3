from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.answer_equiv import NO_ANSWER, form_answer_groups, grade_against_gold, is_missing_extraction
from src.data.load_supplied import token_count_for_cost
from src.metrics.scores import mean_logp, self_certainty, visconf_scores, visual_engagement
from src.select.rules import first_ans, highest_score, rank_weighted_consensus, self_consistency


def paired_bootstrap_ci(
    diffs: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 6304,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Question-level paired bootstrap 95% CI on mean difference."""
    arr = np.asarray(list(diffs), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    means = []
    n = arr.size
    for _ in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        means.append(float(np.mean(sample)))
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": float(np.mean(arr)), "low": low, "high": high, "n": int(n)}


def _question_frame(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    sub = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    if "rollout_id" in sub.columns:
        sub = sub.sort_values("rollout_id")
    return sub.reset_index(drop=True)


def _scores_for_prefix(
    sub: pd.DataFrame,
    prefix: Sequence[int],
    score_name: str,
    dataset: str,
) -> list[float]:
    """Return a full-length score list (indexed by local row index). Recompute VisConf on prefix."""
    n = len(sub)
    scores = [float("-inf")] * n
    if score_name == "mean_logp":
        for i in prefix:
            scores[i] = mean_logp(sub.loc[i, "logp"] if "logp" in sub.columns else None)
        return scores
    if score_name == "self_certainty":
        for i in prefix:
            scores[i] = self_certainty(sub.loc[i, "kl_u_p"] if "kl_u_p" in sub.columns else None)
        return scores
    if score_name in {"vision", "visconf", "visconf_token"}:
        vis = []
        kl = []
        vi_tok = []
        for i in prefix:
            ai = sub.loc[i].get("attn_image", sub.loc[i].get("A_image"))
            ap = sub.loc[i].get("attn_prompt", sub.loc[i].get("A_prompt"))
            ag = sub.loc[i].get("attn_generated", sub.loc[i].get("A_generated"))
            Vi, vi_t = visual_engagement(ai, ap, ag)
            vis.append(Vi)
            vi_tok.append(vi_t)
            kl.append(sub.loc[i, "kl_u_p"] if "kl_u_p" in sub.columns else [])
        if score_name == "vision":
            for j, i in enumerate(prefix):
                scores[i] = vis[j]
            return scores
        fused = visconf_scores(kl, vis, vi_tok)
        key = "Ztoken" if score_name == "visconf_token" else "Z"
        for j, i in enumerate(prefix):
            scores[i] = fused[key][j]
        return scores
    raise ValueError(score_name)


def evaluate_question_budget(
    sub: pd.DataFrame,
    order: Sequence[int],
    n: int,
    dataset: str,
    methods: Sequence[tuple[str, str | None, str]],
) -> dict[str, Any]:
    """Evaluate selection methods on a nested prefix of length n.

    methods: list of (name, score_name|None, aggregator) where aggregator in
    {first, sc, highest, rwc}.
    """
    prefix = list(order[:n])
    answers = [
        sub.loc[i, "extracted_answer_used"] if "extracted_answer_used" in sub.columns else sub.loc[i].get("extracted_answer")
        for i in range(len(sub))
    ]
    gold = sub.loc[0, "gold_answer_used"] if "gold_answer_used" in sub.columns else sub.loc[0].get("gold_answer")
    correct_flags = sub["is_correct"].tolist() if "is_correct" in sub.columns else [
        grade_against_gold(answers[i], gold, dataset) for i in range(len(sub))
    ]
    token_costs = [token_count_for_cost(sub.loc[i]) for i in range(len(sub))]
    meta = {}
    for c in ("answer_type", "precision", "choices", "options", "question_type"):
        if c in sub.columns and pd.notna(sub.loc[0, c]):
            meta[c] = sub.loc[0, c]

    oracle = any(correct_flags[i] for i in prefix)
    groups = form_answer_groups(answers, dataset, meta, order=prefix)
    n_distinct = len(groups)

    # FirstAns cost: only first candidate.
    first_cost = int(token_costs[prefix[0]])
    full_prefix_cost = int(sum(token_costs[i] for i in prefix))

    out_methods = {}
    for name, score_name, agg in methods:
        if agg == "first":
            sel = first_ans(answers, prefix)
            cost = first_cost
        elif agg == "sc":
            sel = self_consistency(answers, prefix, dataset, meta)
            cost = full_prefix_cost
        else:
            assert score_name is not None
            scores = _scores_for_prefix(sub, prefix, score_name, dataset)
            if agg == "highest":
                sel = highest_score(answers, prefix, scores)
            elif agg == "rwc":
                sel = rank_weighted_consensus(answers, prefix, scores, dataset, meta)
            else:
                raise ValueError(agg)
            cost = full_prefix_cost
        pred = sel["answer"]
        is_corr = bool(grade_against_gold(pred, gold, dataset, meta))
        out_methods[name] = {
            "answer": pred if not is_missing_extraction(pred) else NO_ANSWER,
            "correct": is_corr,
            "index": sel.get("index"),
            "token_cost": cost,
        }

    # Sampling vs selection failure at this budget for SC (or any):
    any_correct = oracle
    return {
        "oracle": oracle,
        "n_distinct_groups": n_distinct,
        "methods": out_methods,
        "prefix": prefix,
        "any_correct_in_prefix": any_correct,
        "first_cost": first_cost,
        "prefix_cost": full_prefix_cost,
    }


def default_methods_for_dataset(dataset: str) -> list[tuple[str, str | None, str]]:
    methods = [
        ("FirstAns", None, "first"),
        ("SelfConsistency", None, "sc"),
        ("HighestMeanLogp", "mean_logp", "highest"),
        ("HighestSelfCertainty", "self_certainty", "highest"),
        ("SelfCertaintyRWC", "self_certainty", "rwc"),
    ]
    if dataset.lower() in {"mathvista", "vlm", "qwen2.5-vl-3b-instruct"}:
        methods.extend(
            [
                ("HighestVision", "vision", "highest"),
                ("VisionRWC", "vision", "rwc"),
                ("HighestVisConf", "visconf", "highest"),
                ("VisConfRWC", "visconf", "rwc"),
            ]
        )
    return methods


def evaluate_offline_dataset(
    df: pd.DataFrame,
    splits: dict[str, Any],
    orders: dict[str, list[list[int]]],
    dataset: str,
    budgets: Sequence[int] = (1, 2, 4, 8, 16, 32),
    split_name: str = "eval",
) -> dict[str, Any]:
    """Average over orders within question, then across questions."""
    qids = splits["eval_ids"] if split_name == "eval" else splits["dev_ids"]
    methods = default_methods_for_dataset(dataset)
    # Accumulators: budget -> method -> list of per-question means
    acc: dict[int, dict[str, list[float]]] = {b: {m[0]: [] for m in methods} for b in budgets}
    oracle: dict[int, list[float]] = {b: [] for b in budgets}
    distinct: dict[int, list[float]] = {b: [] for b in budgets}
    cost: dict[int, dict[str, list[float]]] = {b: {m[0]: [] for m in methods} for b in budgets}
    # Failure decomposition at N=32 for SelfConsistency
    sampling_fail = 0
    selection_fail = 0
    per_question = []

    for sid in qids:
        sub = _question_frame(df, sid)
        if len(sub) == 0:
            continue
        # Map rollout_id -> local index
        if "rollout_id" in sub.columns:
            id_to_idx = {int(r): i for i, r in enumerate(sub["rollout_id"].tolist())}
            # Orders are permutations of rollout IDs 0..31 ideally.
            q_orders_raw = orders.get(str(sid), orders.get(sid, []))
            q_orders = []
            for perm in q_orders_raw:
                mapped = []
                for rid in perm:
                    if int(rid) in id_to_idx:
                        mapped.append(id_to_idx[int(rid)])
                    elif int(rid) < len(sub):
                        mapped.append(int(rid))
                # Ensure full coverage
                if len(mapped) < len(sub):
                    missing = [i for i in range(len(sub)) if i not in mapped]
                    mapped.extend(missing)
                q_orders.append(mapped[: len(sub)])
        else:
            q_orders = orders.get(str(sid), [[i for i in range(len(sub))]])

        q_rec = {"sample_id": sid, "budgets": {}}
        for b in budgets:
            meth_acc = {m[0]: [] for m in methods}
            meth_cost = {m[0]: [] for m in methods}
            orc = []
            dist = []
            for order in q_orders:
                ev = evaluate_question_budget(sub, order, min(b, len(order)), dataset, methods)
                orc.append(1.0 if ev["oracle"] else 0.0)
                dist.append(ev["n_distinct_groups"])
                for mname, mres in ev["methods"].items():
                    meth_acc[mname].append(1.0 if mres["correct"] else 0.0)
                    meth_cost[mname].append(float(mres["token_cost"]))
            oracle[b].append(float(np.mean(orc)))
            distinct[b].append(float(np.mean(dist)))
            q_budget = {}
            for mname in meth_acc:
                mean_a = float(np.mean(meth_acc[mname]))
                mean_c = float(np.mean(meth_cost[mname]))
                acc[b][mname].append(mean_a)
                cost[b][mname].append(mean_c)
                q_budget[mname] = {"accuracy": mean_a, "token_cost": mean_c}
            q_budget["oracle"] = float(np.mean(orc))
            q_budget["n_distinct_groups"] = float(np.mean(dist))
            q_rec["budgets"][b] = q_budget

            if b == 32:
                # Decompose using mean over orders: if oracle mean==0 -> sampling fail;
                # elif SC accuracy < oracle -> selection issues (count question if SC mean < 1 and oracle>0)
                sc_mean = float(np.mean(meth_acc["SelfConsistency"]))
                orc_mean = float(np.mean(orc))
                if orc_mean <= 0:
                    sampling_fail += 1
                elif sc_mean < orc_mean - 1e-12:
                    selection_fail += 1
        per_question.append(q_rec)

    summary = {
        "dataset": dataset,
        "split": split_name,
        "n_questions": len(per_question),
        "budgets": {},
        "failure_decomposition_n32": {
            "sampling_failures": sampling_fail,
            "selection_failures": selection_fail,
            "note": "sampling=no correct candidate in prefixes; selection=correct available but SC missed (avg over orders)",
        },
    }
    for b in budgets:
        summary["budgets"][b] = {
            "oracle_coverage": float(np.mean(oracle[b])) if oracle[b] else float("nan"),
            "mean_distinct_groups": float(np.mean(distinct[b])) if distinct[b] else float("nan"),
            "methods": {
                m: {
                    "accuracy": float(np.mean(acc[b][m])) if acc[b][m] else float("nan"),
                    "mean_token_cost": float(np.mean(cost[b][m])) if cost[b][m] else float("nan"),
                }
                for m in acc[b]
            },
        }
    return {"summary": summary, "per_question": per_question}
