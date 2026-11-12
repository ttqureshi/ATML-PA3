from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.data.vqa_normalize import normalize_prediction, normalize_vqa_answer
from src.eval.bootstrap import bootstrap_metrics


def soft_vqa_accuracy(prediction_norm: str, reference_norms: list[str]) -> float:
    """Local VQA soft consensus against all ten annotators.

    ``s = (1/10) * sum_i min(1, (1/3) * |{j != i : a_j = prediction}| )``
    """
    if len(reference_norms) != 10:
        raise ValueError(f"VQA soft score expects 10 references, got {len(reference_norms)}")
    total = 0.0
    for index in range(10):
        matches = 0
        for other, answer in enumerate(reference_norms):
            if other != index and answer == prediction_norm:
                matches += 1
        total += min(1.0, matches / 3.0)
    return total / 10.0


def exact_match(prediction_norm: str, gold_norm: str) -> int:
    return int(prediction_norm == gold_norm)


def score_prediction(task_family: str, prediction: str, row: dict) -> dict:
    """Score one generation. Empties stay in the denominator as zeros."""
    pred_norm = normalize_prediction(prediction)
    empty = int(pred_norm == "")
    canonical = "" if row.get("canonical") is None else str(row["canonical"])
    if task_family == "natural":
        refs = [normalize_vqa_answer(ans) for ans in row["answers"]]
        soft = soft_vqa_accuracy(pred_norm, refs)
        exact = exact_match(pred_norm, canonical)
        primary = soft
    else:
        soft = None
        exact = exact_match(pred_norm, canonical)
        primary = float(exact)
    partner = row.get("partner_canonical")
    follow_partner = int(partner is not None and pred_norm == str(partner) and pred_norm != canonical)
    follow_original = exact
    return {
        "prediction_norm": pred_norm,
        "empty": empty,
        "exact": exact,
        "soft": soft,
        "primary": primary,
        "follow_partner": follow_partner,
        "follow_original": follow_original,
    }


def pair_records_from_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Group per-example rows into complete pairs, one list per suite."""
    grouped: dict[tuple, dict] = {}
    for row in rows:
        key = (row["suite"], row["pair_id"])
        bucket = grouped.setdefault(
            key,
            {
                "suite": row["suite"],
                "pair_id": row["pair_id"],
                "subtype": row.get("answer_type") or row.get("source"),
                "question": row.get("question") or row.get("user_text"),
                "examples": {},
            },
        )
        condition = row["condition"]
        cond_map = bucket["examples"].setdefault(condition, {})
        cond_map[int(row["example_index"])] = row
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for (suite, _pair_id), bucket in grouped.items():
        conditions = sorted(bucket["examples"])
        if not conditions:
            continue
        indexes = sorted(bucket["examples"][conditions[0]])
        if indexes != [0, 1]:
            raise RuntimeError(f"Pair {bucket['pair_id']} is missing an example: {indexes}")
        record = {
            "suite": suite,
            "pair_id": bucket["pair_id"],
            "subtype": bucket["subtype"],
            "question": bucket["question"],
            "conditions": {},
        }
        for condition in conditions:
            sides = [bucket["examples"][condition][0], bucket["examples"][condition][1]]
            record["conditions"][condition] = {
                "primary": [float(side["primary"]) for side in sides],
                "exact": [int(side["exact"]) for side in sides],
                "empty": [int(side["empty"]) for side in sides],
                "nll": [float(side["nll"]) for side in sides],
                "pred_norm": [side["prediction_norm"] for side in sides],
                "canonical": [side.get("canonical") for side in sides],
                "partner_canonical": [side.get("partner_canonical") for side in sides],
                "gen_time": [float(side.get("gen_time_sec") or 0.0) for side in sides],
                "prediction": [side.get("prediction") for side in sides],
                "image_id": [side.get("image_id") for side in sides],
                "follow_partner": [int(side.get("follow_partner") or 0) for side in sides],
                "answers": sides[0].get("answers"),
            }
        by_suite[suite].append(record)
    for suite in by_suite:
        by_suite[suite].sort(key=lambda rec: rec["pair_id"])
    return dict(by_suite)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def condition_accuracy(records: list[dict], condition: str) -> float:
    values = [score for rec in records for score in rec["conditions"][condition]["primary"]]
    return _mean(values)


def condition_nll(records: list[dict], condition: str) -> float:
    values = [score for rec in records for score in rec["conditions"][condition]["nll"]]
    return _mean(values)


def pair_accuracy(records: list[dict], condition: str) -> float:
    if not records:
        return float("nan")
    hits = 0
    for rec in records:
        exact = rec["conditions"][condition]["exact"]
        hits += int(exact[0] == 1 and exact[1] == 1)
    return hits / len(records)


def identical_answer_rate(records: list[dict], condition: str) -> float:
    if not records:
        return float("nan")
    hits = 0
    for rec in records:
        preds = rec["conditions"][condition]["pred_norm"]
        hits += int(preds[0] == preds[1])
    return hits / len(records)


def empty_answer_rate(records: list[dict], condition: str) -> float:
    values = [flag for rec in records for flag in rec["conditions"][condition]["empty"]]
    return _mean([float(v) for v in values])


def mean_gen_time(records: list[dict], condition: str) -> float:
    values = [t for rec in records for t in rec["conditions"][condition]["gen_time"]]
    return _mean(values)


def follow_partner_rate(records: list[dict], condition: str = "complementary") -> float:
    values = [flag for rec in records for flag in rec["conditions"][condition]["follow_partner"]]
    return _mean([float(v) for v in values])


def suite_stat_fns(conditions: list[str]) -> dict[str, callable]:
    fns = {}
    for condition in conditions:
        fns[f"accuracy/{condition}"] = lambda recs, c=condition: condition_accuracy(recs, c)
        fns[f"nll/{condition}"] = lambda recs, c=condition: condition_nll(recs, c)
        fns[f"pair_accuracy/{condition}"] = lambda recs, c=condition: pair_accuracy(recs, c)
        fns[f"identical_answer_rate/{condition}"] = lambda recs, c=condition: identical_answer_rate(recs, c)
        fns[f"empty_answer_rate/{condition}"] = lambda recs, c=condition: empty_answer_rate(recs, c)
        fns[f"gen_time_sec/{condition}"] = lambda recs, c=condition: mean_gen_time(recs, c)
    if "correct" in conditions and "complementary" in conditions:
        fns["advantage/accuracy"] = lambda recs: condition_accuracy(recs, "correct") - condition_accuracy(
            recs, "complementary"
        )
        fns["advantage/nll"] = lambda recs: condition_nll(recs, "complementary") - condition_nll(recs, "correct")
        fns["follow_partner_rate"] = lambda recs: follow_partner_rate(recs, "complementary")
    return fns


def subtype_names(records: list[dict]) -> list[str]:
    return sorted({str(rec["subtype"]) for rec in records})


def summarize_prediction_rows(rows: list[dict], n_resamples: int, seed: int) -> dict:
    """Bootstrap complete pairs. The same resample is used for every condition."""
    by_suite = pair_records_from_rows(rows)
    summary: dict = {"bootstrap_resamples": int(n_resamples), "bootstrap_seed": int(seed), "suites": {}}
    preferred_order = ("correct", "complementary", "mismatched", "no_image")
    for suite, records in by_suite.items():
        present = set(records[0]["conditions"]) if records else set()
        conditions = [name for name in preferred_order if name in present]
        summary["suites"][suite] = {
            "n_pairs": len(records),
            "n_examples": len(records) * 2,
            "overall": bootstrap_metrics(records, suite_stat_fns(conditions), n_resamples, seed),
            "by_subtype": {},
        }
        for subtype in subtype_names(records):
            subset = [rec for rec in records if str(rec["subtype"]) == subtype]
            summary["suites"][suite]["by_subtype"][subtype] = bootstrap_metrics(
                subset, suite_stat_fns(conditions), n_resamples, seed
            )
    return summary
