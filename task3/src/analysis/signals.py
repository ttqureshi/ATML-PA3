from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics.scores import mean_logp, self_certainty, visconf_scores, visual_engagement


def _quartile_correctness_diff(scores: list[float], correct: list[bool]) -> float:
    n = len(scores)
    if n < 4:
        return float("nan")
    order = np.argsort(scores)
    q = max(1, n // 4)
    bottom = [correct[i] for i in order[:q]]
    top = [correct[i] for i in order[-q:]]
    return float(np.mean(top) - np.mean(bottom))


def analyze_signals_n32(df: pd.DataFrame, sample_ids: list[str], dataset: str) -> dict[str, Any]:
    """Within-question associations at N=32."""
    length_corr = []
    logp_diff = []
    sc_diff = []
    vision_diff = []
    visconf_diff = []
    shortest_acc = []
    conf_vision_corrs = []
    ztoken_minus_z = []

    for sid in sample_ids:
        sub = df[df["sample_id"].astype(str) == str(sid)].copy()
        if "rollout_id" in sub.columns:
            sub = sub.sort_values("rollout_id")
        sub = sub.reset_index(drop=True)
        if len(sub) == 0:
            continue
        lengths = []
        logps = []
        scs = []
        visions = []
        correct = sub["is_correct"].astype(bool).tolist() if "is_correct" in sub.columns else [False] * len(sub)
        kl_list = []
        vi_tok_list = []
        for i in range(len(sub)):
            row = sub.loc[i]
            length = int(row.get("num_generated_tokens") or row.get("gen_length") or 0)
            lengths.append(length)
            logps.append(mean_logp(row.get("logp")))
            scs.append(self_certainty(row.get("kl_u_p")))
            ai = row.get("attn_image", row.get("A_image"))
            ap = row.get("attn_prompt", row.get("A_prompt"))
            ag = row.get("attn_generated", row.get("A_generated"))
            Vi, vi_t = visual_engagement(ai, ap, ag)
            visions.append(Vi)
            kl_list.append(row.get("kl_u_p"))
            vi_tok_list.append(vi_t)
            if vi_t.size and len(_as(row.get("kl_u_p"))) == vi_t.size:
                conf_vision_corrs.append(float(np.corrcoef(_as(row.get("kl_u_p")), vi_t)[0, 1]))

        if np.std(lengths) > 0:
            length_corr.append(float(np.corrcoef(lengths, np.asarray(correct, dtype=float))[0, 1]))
        logp_diff.append(_quartile_correctness_diff(logps, correct))
        sc_diff.append(_quartile_correctness_diff(scs, correct))
        # Shortest-rollout control
        shortest_idx = int(np.argmin(lengths))
        shortest_acc.append(1.0 if correct[shortest_idx] else 0.0)

        if dataset.lower() in {"mathvista", "vlm"}:
            vision_diff.append(_quartile_correctness_diff(visions, correct))
            fused = visconf_scores(kl_list, visions, vi_tok_list)
            visconf_diff.append(_quartile_correctness_diff(fused["Z"], correct))
            for i in range(len(sub)):
                ztoken_minus_z.append(fused["Ztoken"][i] - fused["Z"][i])

    out = {
        "dataset": dataset,
        "mean_length_correctness_corr": float(np.nanmean(length_corr)) if length_corr else float("nan"),
        "mean_logp_top_bottom_correctness_diff": float(np.nanmean(logp_diff)) if logp_diff else float("nan"),
        "self_certainty_top_bottom_correctness_diff": float(np.nanmean(sc_diff)) if sc_diff else float("nan"),
        "shortest_rollout_accuracy": float(np.mean(shortest_acc)) if shortest_acc else float("nan"),
    }
    if dataset.lower() in {"mathvista", "vlm"}:
        out.update(
            {
                "vision_top_bottom_correctness_diff": float(np.nanmean(vision_diff)) if vision_diff else float("nan"),
                "visconf_top_bottom_correctness_diff": float(np.nanmean(visconf_diff)) if visconf_diff else float("nan"),
                "mean_conf_vision_corr": float(np.nanmean(conf_vision_corrs)) if conf_vision_corrs else float("nan"),
                "mean_Ztoken_minus_Z": float(np.mean(ztoken_minus_z)) if ztoken_minus_z else float("nan"),
                "note_Ztoken": "Ztoken_i - Z_i = Cov_t(c_i,t, v_i,t)",
            }
        )
    return out


def _as(val) -> np.ndarray:
    from src.metrics.scores import _as_float_array

    return _as_float_array(val)
