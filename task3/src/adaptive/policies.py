from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from src.data.answer_equiv import NO_ANSWER, form_answer_groups, is_missing_extraction
from src.select.rules import rank_weighted_consensus, self_consistency


def agreement_should_stop(
    answers: Sequence[Any],
    order_prefix: Sequence[int],
    dataset: str,
    tau: float,
    metadata: dict | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Stop if unique non-NO_ANSWER leading answer has >= tau fraction of n candidates."""
    n = len(order_prefix)
    if n == 0:
        return False, {"reason": "empty"}
    groups = form_answer_groups(list(answers), dataset, metadata, order=list(order_prefix))
    # Leading by vote count; require uniqueness.
    counts = {k: len(v) for k, v in groups.items()}
    if not counts:
        return False, {"reason": "no_groups"}
    max_count = max(counts.values())
    leaders = [k for k, c in counts.items() if c == max_count]
    if len(leaders) != 1:
        return False, {"reason": "tied_leaders", "counts": counts}
    leader = leaders[0]
    if leader == NO_ANSWER:
        return False, {"reason": "leader_no_answer", "counts": counts}
    frac = max_count / float(n)
    ok = frac >= tau
    return ok, {"reason": "agreement" if ok else "frac_below_tau", "leader": leader, "frac": frac, "counts": counts}


def quality_aware_should_stop(
    answers: Sequence[Any],
    order_prefix: Sequence[int],
    dataset: str,
    scores: Sequence[float],
    tau: float,
    gamma: float,
    metadata: dict | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Agreement + SC/RWC winners agree + RWC margin >= gamma."""
    ag_ok, ag_info = agreement_should_stop(answers, order_prefix, dataset, tau, metadata)
    if not ag_ok:
        return False, {"reason": "agreement_failed", "agreement": ag_info}
    sc = self_consistency(answers, order_prefix, dataset, metadata)
    rwc = rank_weighted_consensus(answers, order_prefix, scores, dataset, metadata)
    # Compare by group keys / equivalence.
    sc_key = sc.get("group_key")
    rwc_key = rwc.get("group_key")
    if sc_key != rwc_key:
        # Fall back to answer equivalence.
        from src.data.answer_equiv import answers_equivalent

        if not answers_equivalent(sc["answer"], rwc["answer"], dataset, metadata):
            return False, {
                "reason": "sc_rwc_disagree",
                "sc": sc_key,
                "rwc": rwc_key,
                "agreement": ag_info,
            }
    margin = float(rwc.get("margin", 0.0))
    if margin < gamma:
        return False, {
            "reason": "margin_below_gamma",
            "margin": margin,
            "gamma": gamma,
            "agreement": ag_info,
        }
    return True, {
        "reason": "quality_aware",
        "margin": margin,
        "agreement": ag_info,
        "sc_key": sc_key,
        "rwc_key": rwc_key,
    }


def simulate_adaptive(
    answers: Sequence[Any],
    full_order: Sequence[int],
    dataset: str,
    token_costs: Sequence[int],
    scores_fn: Callable[[Sequence[int]], Sequence[float]],
    policy: str,
    tau: float = 0.75,
    gamma: float = 0.0,
    check_ns: Sequence[int] = (4, 8, 16),
    max_n: int = 32,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Reveal rollouts in order; check stop after n in check_ns else stop at max_n.

    Always returns current self-consistency answer at stopping time.
    scores_fn(prefix_indices) -> score list aligned to full answer list (unused indices ignored).
    """
    stop_n = max_n
    stop_info: dict[str, Any] = {"reason": "max_n"}
    for n in list(check_ns) + ([max_n] if max_n not in check_ns else []):
        if n > len(full_order):
            continue
        prefix = list(full_order[:n])
        if n == max_n and n not in check_ns:
            stop_n = n
            stop_info = {"reason": "max_n"}
            break
        if policy == "always32" or policy == "always_max":
            continue
        scores = scores_fn(prefix)
        if policy == "agreement":
            ok, info = agreement_should_stop(answers, prefix, dataset, tau, metadata)
        elif policy == "quality":
            ok, info = quality_aware_should_stop(answers, prefix, dataset, scores, tau, gamma, metadata)
        else:
            raise ValueError(policy)
        if ok and n in check_ns:
            stop_n = n
            stop_info = info
            break
        if n == max_n:
            stop_n = n
            stop_info = {"reason": "max_n", "last": info if "info" in dir() else {}}
    prefix = list(full_order[:stop_n])
    sc = self_consistency(answers, prefix, dataset, metadata)
    cost = int(sum(int(token_costs[i]) for i in prefix))
    return {
        "stop_n": stop_n,
        "answer": sc["answer"],
        "index": sc["index"],
        "group_key": sc.get("group_key"),
        "token_cost": cost,
        "stop_info": stop_info,
        "policy": policy,
        "tau": tau,
        "gamma": gamma,
    }


def analyze_early_stop_recovery(
    answers: Sequence[Any],
    full_order: Sequence[int],
    dataset: str,
    token_costs: Sequence[int],
    scores_fn: Callable[[Sequence[int]], Sequence[float]],
    policy: str,
    tau: float,
    gamma: float,
    gold: Any,
    check_ns: Sequence[int] = (4, 8, 16),
    max_n: int = 32,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Compare adaptive stop vs fixed-max SC for one order (retrospective).

    Returns whether the early stop was incorrect and whether fixed-max SC
    would have recovered the correct answer.
    """
    from src.data.answer_equiv import grade_against_gold

    sim = simulate_adaptive(
        answers,
        full_order,
        dataset,
        token_costs,
        scores_fn,
        policy=policy,
        tau=tau,
        gamma=gamma,
        check_ns=check_ns,
        max_n=max_n,
        metadata=metadata,
    )
    fixed = simulate_adaptive(
        answers,
        full_order,
        dataset,
        token_costs,
        scores_fn,
        policy="always32",
        tau=0.0,
        gamma=0.0,
        check_ns=check_ns,
        max_n=max_n,
        metadata=metadata,
    )
    early_correct = bool(grade_against_gold(sim["answer"], gold, dataset, metadata))
    fixed_correct = bool(grade_against_gold(fixed["answer"], gold, dataset, metadata))
    early_stop = int(sim["stop_n"]) < int(max_n)
    incorrect_early_stop = bool(early_stop and (not early_correct))
    recovered_by_fixed32 = bool(incorrect_early_stop and fixed_correct)
    return {
        "stop_n": int(sim["stop_n"]),
        "early_stop": early_stop,
        "early_correct": early_correct,
        "fixed32_correct": fixed_correct,
        "incorrect_early_stop": incorrect_early_stop,
        "recovered_by_fixed32": recovered_by_fixed32,
        "token_cost": int(sim["token_cost"]),
        "fixed32_token_cost": int(fixed["token_cost"]),
    }


def select_thresholds(
    results: list[dict[str, Any]],
    fixed32_accuracy: float,
    accuracy_tolerance_pp: float = 1.0,
) -> dict[str, Any]:
    """Pick lowest mean token cost with dev accuracy <= 1pp below fixed-32 SC.

    Ties: higher accuracy, then smaller tau, then smaller gamma.
    `results` entries need: tau, gamma (optional), accuracy, mean_token_cost.
    """
    floor = fixed32_accuracy - (accuracy_tolerance_pp / 100.0)
    eligible = [r for r in results if r["accuracy"] >= floor - 1e-12]
    if not eligible:
        # Fallback: always-32 style entry if present, else best accuracy.
        always = [r for r in results if r.get("policy") in {"always32", "always_max"}]
        if always:
            return always[0]
        return sorted(results, key=lambda r: (-r["accuracy"], r["mean_token_cost"]))[0]

    def sort_key(r: dict[str, Any]) -> tuple:
        return (
            r["mean_token_cost"],
            -r["accuracy"],
            float(r.get("tau", 0.0)),
            float(r.get("gamma", 0.0)),
        )

    return sorted(eligible, key=sort_key)[0]
