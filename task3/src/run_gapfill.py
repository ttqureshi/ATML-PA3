"""Fill Task 3 report gaps: tokenwise VisConf ablation + early-stop recovery.

Reuses existing adaptive thresholds from results when present; otherwise selects
them. Writes into the artifacts root (intended: merge into offline pack).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive.policies import analyze_early_stop_recovery, select_thresholds, simulate_adaptive
from src.analysis.plots import save_accuracy_curves
from src.common.config import parse_common_args, prepare_runtime
from src.common.logging_utils import append_status, write_json
from src.common.seed import set_seed
from src.data.answer_equiv import grade_against_gold, scorer_versions
from src.data.load_supplied import load_all_supplied, token_count_for_cost
from src.eval.protocol import evaluate_offline_dataset, paired_bootstrap_ci
from src.metrics.scores import self_certainty, visconf_scores, visual_engagement
from src.run_offline import _score_fn_factory, run_adaptive_for_dataset


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _early_stop_for_dataset(payload: dict, cfg: dict, dataset: str, selected: dict) -> dict:
    df = payload["frame"]
    splits = payload["splits"]
    orders = payload["orders"]
    ad_cfg = cfg["adaptive"]
    score_name = "self_certainty" if dataset == "math" else "visconf"
    out = {}
    for pname, sel in selected.items():
        n_orders = 0
        n_early = 0
        n_incorrect_early = 0
        n_recovered = 0
        n_incorrect_not_recovered = 0
        for sid in splits["eval_ids"]:
            sub = df[df["sample_id"].astype(str) == str(sid)].copy()
            if "rollout_id" in sub.columns:
                sub = sub.sort_values("rollout_id").reset_index(drop=True)
            else:
                sub = sub.reset_index(drop=True)
            answers = (
                sub["extracted_answer_used"].tolist()
                if "extracted_answer_used" in sub.columns
                else sub.get("extracted_answer").tolist()
            )
            gold = sub.loc[0, "gold_answer_used"] if "gold_answer_used" in sub.columns else sub.loc[0].get("gold_answer")
            token_costs = [token_count_for_cost(sub.loc[i]) for i in range(len(sub))]
            scores_fn = _score_fn_factory(sub, dataset, score_name)
            q_orders = orders.get(str(sid), [])
            for perm in q_orders:
                if "rollout_id" in sub.columns:
                    id_to_idx = {int(r): i for i, r in enumerate(sub["rollout_id"].tolist())}
                    order = [id_to_idx.get(int(r), int(r) if int(r) < len(sub) else 0) for r in perm]
                else:
                    order = list(perm)
                rec = analyze_early_stop_recovery(
                    answers,
                    order,
                    dataset,
                    token_costs,
                    scores_fn,
                    policy=pname,
                    tau=float(sel.get("tau", 1.0)),
                    gamma=float(sel.get("gamma", 0.0)),
                    gold=gold,
                    check_ns=ad_cfg["check_ns"],
                    max_n=ad_cfg["max_n"],
                )
                n_orders += 1
                if rec["early_stop"]:
                    n_early += 1
                if rec["incorrect_early_stop"]:
                    n_incorrect_early += 1
                    if rec["recovered_by_fixed32"]:
                        n_recovered += 1
                    else:
                        n_incorrect_not_recovered += 1
        out[pname] = {
            "tau": float(sel.get("tau", 1.0)),
            "gamma": float(sel.get("gamma", 0.0)),
            "n_question_orders": n_orders,
            "n_early_stops": n_early,
            "n_incorrect_early_stops": n_incorrect_early,
            "n_recovered_by_fixed32_sc": n_recovered,
            "n_incorrect_early_not_recovered": n_incorrect_not_recovered,
            "frac_orders_early_stop": (n_early / n_orders) if n_orders else float("nan"),
            "frac_orders_incorrect_early_stop": (n_incorrect_early / n_orders) if n_orders else float("nan"),
            "frac_incorrect_early_recovered_by_fixed32": (n_recovered / n_incorrect_early)
            if n_incorrect_early
            else float("nan"),
            "note": (
                "Counts are over held-out questions × 10 nested orders. "
                "incorrect_early_stop = stopped before 32 and SC answer wrong; "
                "recovered_by_fixed32_sc = that order's fixed-32 SC would have been correct."
            ),
        }
    return out


def main() -> int:
    parser = parse_common_args("ATML PA3 Task 3 — gap fill (tokenwise + early-stop recovery)")
    args = parser.parse_args()
    root, cfg, dirs, data_root = prepare_runtime(args)
    set_seed(int(cfg["seed"]))

    status_path = dirs["root"] / "STATUS.md"
    append_status(status_path, "# Task 3 gap-fill run")
    append_status(status_path, f"- seed: {cfg['seed']}")
    append_status(status_path, f"- scorer: {scorer_versions()}")

    supplied = load_all_supplied(data_root, cfg, dirs)
    if not supplied.get("available"):
        append_status(status_path, f"- BLOCKED: {supplied.get('reason')}")
        print("GAPFILL_BLOCKED", supplied.get("reason"))
        return 2

    # 1) Tokenwise VisConf ablation on MathVista (full method table includes new methods).
    if "mathvista" in supplied["datasets"]:
        payload = supplied["datasets"]["mathvista"]
        append_status(status_path, f"- mathvista rows={payload['n_rows']} questions={payload['n_questions']}")
        ev = evaluate_offline_dataset(
            payload["frame"],
            payload["splits"],
            payload["orders"],
            dataset="mathvista",
            budgets=cfg["budgets"],
            split_name="eval",
        )
        write_json(dirs["tables"] / "mathvista_eval_summary.json", ev["summary"])
        write_json(dirs["per_question"] / "mathvista_eval_per_question.json", ev["per_question"])
        save_accuracy_curves(ev["summary"], dirs["figures"] / "mathvista_accuracy_curves.png")

        pq = ev["per_question"]
        sc32 = [q["budgets"][32]["SelfConsistency"]["accuracy"] for q in pq]
        boot = {}
        for mname in ev["summary"]["budgets"][32]["methods"]:
            if mname == "SelfConsistency":
                continue
            other = [q["budgets"][32][mname]["accuracy"] for q in pq]
            diffs = [a - b for a, b in zip(other, sc32)]
            boot[mname] = paired_bootstrap_ci(
                diffs, n_resamples=int(cfg["bootstrap"]["n_resamples"]), seed=int(cfg["seed"])
            )
        write_json(dirs["tables"] / "mathvista_bootstrap_vs_sc_n32.json", boot)

        tokenwise = {
            "N": 32,
            "HighestVisConf": ev["summary"]["budgets"][32]["methods"].get("HighestVisConf"),
            "VisConfRWC": ev["summary"]["budgets"][32]["methods"].get("VisConfRWC"),
            "HighestVisConfToken": ev["summary"]["budgets"][32]["methods"].get("HighestVisConfToken"),
            "VisConfTokenRWC": ev["summary"]["budgets"][32]["methods"].get("VisConfTokenRWC"),
            "bootstrap_vs_sc": {
                k: boot[k]
                for k in ("HighestVisConf", "VisConfRWC", "HighestVisConfToken", "VisConfTokenRWC")
                if k in boot
            },
            "note": "Tokenwise fusion Ztoken=mean_t(c_t v_t) vs summary Z=V*C; same prefix calibration.",
        }
        write_json(dirs["tables"] / "mathvista_tokenwise_visconf.json", tokenwise)
        append_status(
            status_path,
            f"- tokenwise N=32 HighestVisConfToken={tokenwise['HighestVisConfToken']} "
            f"VisConfTokenRWC={tokenwise['VisConfTokenRWC']}",
        )
    else:
        append_status(status_path, "- mathvista missing; skipped tokenwise ablation")

    # Frozen thresholds from the COMPLETE offline run (dev-selected). Used when
    # artifacts are missing on a fresh Kaggle working directory.
    frozen_thresholds = {
        "math": {
            "agreement": {"policy": "agreement", "tau": 0.625, "gamma": 0.0},
            "quality": {"policy": "quality", "tau": 0.625, "gamma": 0.0},
        },
        "mathvista": {
            "agreement": {"policy": "agreement", "tau": 0.625, "gamma": 0.0},
            "quality": {"policy": "quality", "tau": 0.5, "gamma": 0.1},
        },
    }

    # 2) Early-stop recovery for both datasets (reuse frozen thresholds when available).
    for ds_name, payload in supplied["datasets"].items():
        existing = _load_json(dirs["adaptive"] / f"{ds_name}_adaptive.json") or {
            "dataset": ds_name,
            "score_name": "self_certainty" if ds_name == "math" else "visconf",
            "note": "Thresholds reused from prior offline COMPLETE; early_stop_analysis added by gapfill.",
        }
        if "selected_agreement" in existing and "selected_quality" in existing:
            selected = {
                "agreement": existing["selected_agreement"],
                "quality": existing["selected_quality"],
            }
        else:
            selected = frozen_thresholds[ds_name]
            existing["selected_agreement"] = selected["agreement"]
            existing["selected_quality"] = selected["quality"]
            existing["thresholds_source"] = "frozen_from_offline_complete"
            append_status(status_path, f"- using frozen thresholds for {ds_name}: {selected}")

        analysis = _early_stop_for_dataset(payload, cfg, ds_name, selected)
        existing["early_stop_analysis"] = analysis
        write_json(dirs["adaptive"] / f"{ds_name}_adaptive.json", existing)
        append_status(
            status_path,
            f"- early_stop {ds_name} agreement_incorrect={analysis['agreement']['n_incorrect_early_stops']} "
            f"recovered={analysis['agreement']['n_recovered_by_fixed32_sc']}; "
            f"quality_incorrect={analysis['quality']['n_incorrect_early_stops']} "
            f"recovered={analysis['quality']['n_recovered_by_fixed32_sc']}",
        )

    append_status(status_path, "- GAPFILL_COMPLETE")
    print("GAPFILL_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
