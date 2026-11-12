from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from src.data.answer_equiv import NO_ANSWER, form_answer_groups, is_missing_extraction
from src.metrics.scores import average_ranks, rwc_weights


def first_ans(answers: Sequence[Any], order: Sequence[int]) -> dict[str, Any]:
    """Return the only / first candidate in the current order."""
    if not order:
        return {"answer": NO_ANSWER, "index": None, "method": "FirstAns"}
    idx = int(order[0])
    ans = answers[idx]
    if is_missing_extraction(ans):
        ans = NO_ANSWER
    return {"answer": ans, "index": idx, "method": "FirstAns", "group_key": None}


def self_consistency(
    answers: Sequence[Any],
    order: Sequence[int],
    dataset: str,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """One vote per rollout; ties -> earliest supporting rollout in current order."""
    if len(order) == 1:
        return first_ans(answers, order) | {"method": "SelfConsistency"}
    groups = form_answer_groups(list(answers), dataset, metadata, order=list(order))
    # Count votes; pick max; tie-break by earliest supporter in order.
    best_key = None
    best_count = -1
    best_first_pos = 10**9
    order_pos = {idx: p for p, idx in enumerate(order)}
    for key, members in groups.items():
        count = len(members)
        first_pos = min(order_pos[m] for m in members)
        if count > best_count or (count == best_count and first_pos < best_first_pos):
            best_key = key
            best_count = count
            best_first_pos = first_pos
    assert best_key is not None
    winner_idx = min(groups[best_key], key=lambda m: order_pos[m])
    ans = answers[winner_idx]
    if best_key == NO_ANSWER or is_missing_extraction(ans):
        ans = NO_ANSWER
    return {
        "answer": ans,
        "index": winner_idx,
        "method": "SelfConsistency",
        "group_key": best_key,
        "votes": {k: len(v) for k, v in groups.items()},
    }


def highest_score(
    answers: Sequence[Any],
    order: Sequence[int],
    scores: Sequence[float],
) -> dict[str, Any]:
    """Select highest score; ties -> earliest in current order. N=1 returns only candidate."""
    if len(order) == 1:
        return first_ans(answers, order) | {"method": "HighestScore"}
    best_idx = None
    best_score = float("-inf")
    best_pos = 10**9
    for pos, idx in enumerate(order):
        sc = float(scores[idx])
        if sc > best_score or (sc == best_score and pos < best_pos):
            best_score = sc
            best_idx = int(idx)
            best_pos = pos
    assert best_idx is not None
    ans = answers[best_idx]
    if is_missing_extraction(ans):
        ans = NO_ANSWER
    return {
        "answer": ans,
        "index": best_idx,
        "method": "HighestScore",
        "score": best_score,
    }


def rank_weighted_consensus(
    answers: Sequence[Any],
    order: Sequence[int],
    scores: Sequence[float],
    dataset: str,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """RWC: wi=(N-ri)/(N-1); vote by answer groups; ties -> earliest supporter."""
    if len(order) == 1:
        return first_ans(answers, order) | {"method": "RWC"}
    # Ranks among the N candidates in `order` only.
    local_scores = [float(scores[i]) for i in order]
    local_ranks = average_ranks(local_scores, higher_is_better=True)
    weights_local = rwc_weights(local_ranks)
    weight_by_idx = {int(order[j]): weights_local[j] for j in range(len(order))}
    groups = form_answer_groups(list(answers), dataset, metadata, order=list(order))
    order_pos = {idx: p for p, idx in enumerate(order)}
    best_key = None
    best_w = float("-inf")
    best_first = 10**9
    vote_weights: dict[str, float] = {}
    for key, members in groups.items():
        w = sum(weight_by_idx[m] for m in members)
        vote_weights[key] = w
        first_pos = min(order_pos[m] for m in members)
        if w > best_w or (w == best_w and first_pos < best_first):
            best_w = w
            best_key = key
            best_first = first_pos
    assert best_key is not None
    winner_idx = min(groups[best_key], key=lambda m: order_pos[m])
    ans = answers[winner_idx]
    if best_key == NO_ANSWER or is_missing_extraction(ans):
        ans = NO_ANSWER
    # Margin: (top - second) / total
    weights_sorted = sorted(vote_weights.values(), reverse=True)
    total = sum(weights_sorted) if weights_sorted else 0.0
    top = weights_sorted[0] if weights_sorted else 0.0
    second = weights_sorted[1] if len(weights_sorted) > 1 else 0.0
    margin = ((top - second) / total) if total > 0 else 0.0
    return {
        "answer": ans,
        "index": winner_idx,
        "method": "RWC",
        "group_key": best_key,
        "vote_weights": vote_weights,
        "margin": margin,
        "ranks": {int(order[j]): local_ranks[j] for j in range(len(order))},
        "weights": weight_by_idx,
    }


def select_answer(
    method: str,
    answers: Sequence[Any],
    order: Sequence[int],
    dataset: str,
    scores: Sequence[float] | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    method = method.lower()
    if method in {"firstans", "first", "first_ans"}:
        return first_ans(answers, order)
    if method in {"sc", "self_consistency", "self-consistency"}:
        return self_consistency(answers, order, dataset, metadata)
    if method in {"highest", "highest_score", "bon", "best_of_n"}:
        assert scores is not None
        return highest_score(answers, order, scores)
    if method in {"rwc", "rank_weighted", "rank-weighted"}:
        assert scores is not None
        return rank_weighted_consensus(answers, order, scores, dataset, metadata)
    raise ValueError(f"Unknown selection method: {method}")
