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
from src.analysis.signals import analyze_signals_n32
from src.common.config import parse_common_args, prepare_runtime
from src.common.logging_utils import append_status, write_json
from src.common.seed import set_seed
from src.data.answer_equiv import scorer_versions
from src.data.load_supplied import load_all_supplied, token_count_for_cost
from src.eval.protocol import evaluate_offline_dataset, paired_bootstrap_ci
from src.metrics.scores import mean_logp, self_certainty, visconf_scores, visual_engagement


def _score_fn_factory(sub: pd.DataFrame, dataset: str, score_name: str):
    def scores_fn(prefix):
        n = len(sub)
        out = [float("-inf")] * n
        if score_name == "self_certainty":
            for i in prefix:
                out[i] = self_certainty(sub.loc[i].get("kl_u_p"))
            return out
        # VisConf recomputed on revealed set
        vis = []
        kl = []
        for i in prefix:
            row = sub.loc[i]
            Vi, _ = visual_engagement(row.get("attn_image", row.get("A_image")), row.get("attn_prompt", row.get("A_prompt")), row.get("attn_generated", row.get("A_generated")))
            vis.append(Vi)
            kl.append(row.get("kl_u_p"))
        fused = visconf_scores(kl, vis)
        for j, i in enumerate(prefix):
            out[i] = fused["Z"][j]
        return out

    return scores_fn


def run_adaptive_for_dataset(payload: dict, cfg: dict, dirs: dict, dataset: str) -> dict:
    df = payload["frame"]
    splits = payload["splits"]
    orders = payload["orders"]
    ad_cfg = cfg["adaptive"]
    score_name = "self_certainty" if dataset == "math" else "visconf"

    def eval_policy(qids, policy, tau=0.0, gamma=0.0):
        accs = []
        costs = []
        stop_ns = []
        for sid in qids:
            sub = df[df["sample_id"].astype(str) == str(sid)].copy()
            if "rollout_id" in sub.columns:
                sub = sub.sort_values("rollout_id").reset_index(drop=True)
            else:
                sub = sub.reset_index(drop=True)
            answers = sub["extracted_answer_used"].tolist() if "extracted_answer_used" in sub.columns else sub.get("extracted_answer").tolist()
            gold = sub.loc[0, "gold_answer_used"] if "gold_answer_used" in sub.columns else sub.loc[0].get("gold_answer")
            token_costs = [token_count_for_cost(sub.loc[i]) for i in range(len(sub))]
            q_orders = orders.get(str(sid), [])
            order_acc = []
            order_cost = []
            order_stop = []
            for perm in q_orders:
                # map rollout ids
                if "rollout_id" in sub.columns:
                    id_to_idx = {int(r): i for i, r in enumerate(sub["rollout_id"].tolist())}
                    order = [id_to_idx.get(int(r), int(r) if int(r) < len(sub) else 0) for r in perm]
                else:
                    order = list(perm)
                sim = simulate_adaptive(
                    answers,
                    order,
                    dataset,
                    token_costs,
                    _score_fn_factory(sub, dataset, score_name),
                    policy=policy,
                    tau=tau,
                    gamma=gamma,
                    check_ns=ad_cfg["check_ns"],
                    max_n=ad_cfg["max_n"],
                )
                from src.data.answer_equiv import grade_against_gold

                order_acc.append(1.0 if grade_against_gold(sim["answer"], gold, dataset) else 0.0)
                order_cost.append(sim["token_cost"])
                order_stop.append(sim["stop_n"])
            accs.append(float(np.mean(order_acc)))
            costs.append(float(np.mean(order_cost)))
            stop_ns.append(float(np.mean(order_stop)))
        return {
            "policy": policy,
            "tau": tau,
            "gamma": gamma,
            "accuracy": float(np.mean(accs)) if accs else float("nan"),
            "mean_token_cost": float(np.mean(costs)) if costs else float("nan"),
            "mean_rollouts": float(np.mean(stop_ns)) if stop_ns else float("nan"),
            "stop_ns_mean_per_q": stop_ns,
        }

    # Fixed-32 SC baseline on dev
    fixed = eval_policy(splits["dev_ids"], "always32")
    fixed32_acc = fixed["accuracy"]

    grid = []
    # Always-32 fallback
    grid.append(fixed)
    for tau in ad_cfg["tau_grid"]:
        grid.append(eval_policy(splits["dev_ids"], "agreement", tau=float(tau)))
    for tau in ad_cfg["tau_grid"]:
        for gamma in ad_cfg["gamma_grid"]:
            grid.append(eval_policy(splits["dev_ids"], "quality", tau=float(tau), gamma=float(gamma)))

    # Select per family
    agreement_grid = [r for r in grid if r["policy"] in {"agreement", "always32"}]
    quality_grid = [r for r in grid if r["policy"] in {"quality", "always32"}]
    sel_agr = select_thresholds(agreement_grid, fixed32_acc, ad_cfg["accuracy_tolerance_pp"])
    sel_qual = select_thresholds(quality_grid, fixed32_acc, ad_cfg["accuracy_tolerance_pp"])

    # Held-out eval
    held = {
        "fixed32": eval_policy(splits["eval_ids"], "always32"),
        "agreement": eval_policy(splits["eval_ids"], "agreement", tau=float(sel_agr.get("tau", 1.0))),
        "quality": eval_policy(
            splits["eval_ids"],
            "quality",
            tau=float(sel_qual.get("tau", 1.0)),
            gamma=float(sel_qual.get("gamma", 0.0)),
        ),
    }

    # Retrospective: incorrect early stops and fixed-32 recovery (held-out).
    early_stop_analysis = {}
    for pname, sel in (("agreement", sel_agr), ("quality", sel_qual)):
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
        early_stop_analysis[pname] = {
            "tau": float(sel.get("tau", 1.0)),
            "gamma": float(sel.get("gamma", 0.0)),
            "n_question_orders": n_orders,
            "n_early_stops": n_early,
            "n_incorrect_early_stops": n_incorrect_early,
            "n_recovered_by_fixed32_sc": n_recovered,
            "n_incorrect_early_not_recovered": n_incorrect_not_recovered,
            "frac_orders_early_stop": (n_early / n_orders) if n_orders else float("nan"),
            "frac_orders_incorrect_early_stop": (n_incorrect_early / n_orders) if n_orders else float("nan"),
            "frac_incorrect_early_recovered_by_fixed32": (n_recovered / n_incorrect_early) if n_incorrect_early else float("nan"),
            "note": (
                "Counts are over held-out questions × 10 nested orders. "
                "incorrect_early_stop = stopped before 32 and SC answer wrong; "
                "recovered_by_fixed32_sc = that order's fixed-32 SC would have been correct."
            ),
        }

    out = {
        "dataset": dataset,
        "score_name": score_name,
        "fixed32_dev_accuracy": fixed32_acc,
        "selected_agreement": sel_agr,
        "selected_quality": sel_qual,
        "dev_grid_size": len(grid),
        "held_out": held,
        "early_stop_analysis": early_stop_analysis,
    }
    write_json(dirs["adaptive"] / f"{dataset}_adaptive.json", out)
    return out


def main() -> int:
    parser = parse_common_args("ATML PA3 Task 3 — offline Steps 1–3,5–6")
    args = parser.parse_args()
    root, cfg, dirs, data_root = prepare_runtime(args)
    set_seed(int(cfg["seed"]))

    status_path = dirs["root"] / "STATUS.md"
    append_status(status_path, "# Task 3 offline run")
    append_status(status_path, f"- seed: {cfg['seed']}")
    append_status(status_path, f"- scorer: {scorer_versions()}")
    write_json(dirs["configs_used"] / "scorer_versions.json", scorer_versions())

    supplied = load_all_supplied(data_root, cfg, dirs)
    write_json(dirs["tables"] / "data_availability.json", {k: v for k, v in supplied.items() if k != "datasets"})
    if not supplied.get("available"):
        append_status(
            status_path,
            f"- BLOCKED: official data missing at {data_root}. Offline Steps 1–3/5–6 require ATML-PA3-Data.",
        )
        append_status(status_path, f"- reason: {supplied.get('reason')}")
        print("OFFLINE_BLOCKED_NO_DATA", supplied.get("reason"))
        return 2

    for ds_name, payload in supplied["datasets"].items():
        append_status(status_path, f"- loaded {ds_name}: rows={payload['n_rows']} questions={payload['n_questions']}")
        append_status(status_path, f"  alignment: {payload['alignment']}")
        # Step 1–2 eval
        ev = evaluate_offline_dataset(
            payload["frame"],
            payload["splits"],
            payload["orders"],
            dataset=ds_name,
            budgets=cfg["budgets"],
            split_name="eval",
        )
        write_json(dirs["tables"] / f"{ds_name}_eval_summary.json", ev["summary"])
        write_json(dirs["per_question"] / f"{ds_name}_eval_per_question.json", ev["per_question"])
        save_accuracy_curves(ev["summary"], dirs["figures"] / f"{ds_name}_accuracy_curves.png")

        # Bootstrap main accuracy differences vs SelfConsistency at N=32
        pq = ev["per_question"]
        sc32 = [q["budgets"][32]["SelfConsistency"]["accuracy"] for q in pq]
        boot = {}
        for mname in ev["summary"]["budgets"][32]["methods"]:
            if mname == "SelfConsistency":
                continue
            other = [q["budgets"][32][mname]["accuracy"] for q in pq]
            diffs = [a - b for a, b in zip(other, sc32)]
            boot[mname] = paired_bootstrap_ci(diffs, n_resamples=int(cfg["bootstrap"]["n_resamples"]), seed=int(cfg["seed"]))
        write_json(dirs["tables"] / f"{ds_name}_bootstrap_vs_sc_n32.json", boot)

        # Step 3 signals
        sig = analyze_signals_n32(payload["frame"], payload["splits"]["eval_ids"], ds_name)
        write_json(dirs["analysis"] / f"{ds_name}_signals_n32.json", sig)

        # Step 5 adaptive
        ad = run_adaptive_for_dataset(payload, cfg, dirs, ds_name)
        append_status(status_path, f"- adaptive {ds_name} selected agreement={ad['selected_agreement']}")
        append_status(status_path, f"- adaptive {ds_name} selected quality={ad['selected_quality']}")

    append_status(status_path, "- OFFLINE_COMPLETE")
    print("OFFLINE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
