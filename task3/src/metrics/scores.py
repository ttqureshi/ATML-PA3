from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _as_float_array(val: Any) -> np.ndarray:
    if val is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(val, np.ndarray):
        return val.astype(np.float64, copy=False).reshape(-1)
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float64).reshape(-1)
    if isinstance(val, str):
        try:
            import ast

            return np.asarray(ast.literal_eval(val), dtype=np.float64).reshape(-1)
        except Exception:
            return np.asarray([], dtype=np.float64)
    try:
        if hasattr(val, "tolist"):
            return np.asarray(val.tolist(), dtype=np.float64).reshape(-1)
    except Exception:
        pass
    return np.asarray([], dtype=np.float64)


def mean_logp(logp_seq: Any) -> float:
    """Li = (1/Ti) sum_t log p_i,t(y_i,t)."""
    arr = _as_float_array(logp_seq)
    if arr.size == 0:
        return float("-inf")
    return float(np.mean(arr))


def self_certainty(kl_u_p_seq: Any) -> float:
    """Ki = (1/Ti) sum_t KL(U || p_i,t)."""
    arr = _as_float_array(kl_u_p_seq)
    if arr.size == 0:
        return float("-inf")
    return float(np.mean(arr))


def visual_engagement(
    a_image: Any,
    a_prompt: Any,
    a_generated: Any,
) -> tuple[float, np.ndarray]:
    """Vi = mean_t  Aimage / (Aimage + Aprompt + Agenerated)."""
    ai = _as_float_array(a_image)
    ap = _as_float_array(a_prompt)
    ag = _as_float_array(a_generated)
    n = min(ai.size, ap.size, ag.size)
    if n == 0:
        return 0.0, np.asarray([], dtype=np.float64)
    ai, ap, ag = ai[:n], ap[:n], ag[:n]
    denom = ai + ap + ag
    with np.errstate(divide="ignore", invalid="ignore"):
        vi_t = np.where(denom > 0, ai / denom, 0.0)
    return float(np.mean(vi_t)), vi_t


def _percentile_linear(values: np.ndarray, q: float) -> float:
    """Linear-interpolation percentile (NumPy default method='linear')."""
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q, method="linear"))


def calibrate_kl_tokenwise(kl_values_by_rollout: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Pool token KL across current candidate set G; calibrate with 5th/95th percentiles."""
    if not kl_values_by_rollout:
        return []
    pooled = np.concatenate([_as_float_array(x) for x in kl_values_by_rollout if _as_float_array(x).size])
    if pooled.size == 0:
        return [np.asarray([], dtype=np.float64) for _ in kl_values_by_rollout]
    q05 = _percentile_linear(pooled, 5)
    q95 = _percentile_linear(pooled, 95)
    denom = max(q95 - q05, 1e-8)
    out = []
    for seq in kl_values_by_rollout:
        arr = _as_float_array(seq)
        if arr.size == 0:
            out.append(arr)
            continue
        ci = (arr - q05) / denom
        ci = np.clip(ci, 0.0, 1.0)
        out.append(ci)
    return out


def visconf_scores(
    kl_values_by_rollout: Sequence[Any],
    visual_engagements: Sequence[float],
    vi_token_seqs: Sequence[Any] | None = None,
) -> dict[str, list[float]]:
    """Compute Ci, Zi, and tokenwise fusion Ztoken for the current candidate set."""
    kl_arrays = [_as_float_array(x) for x in kl_values_by_rollout]
    calibrated = calibrate_kl_tokenwise(kl_arrays)
    C: list[float] = []
    Z: list[float] = []
    Ztoken: list[float] = []
    for i, ci in enumerate(calibrated):
        if ci.size == 0:
            Ci = 0.0
        else:
            Ci = float(np.mean(ci))
        Vi = float(visual_engagements[i]) if i < len(visual_engagements) else 0.0
        C.append(Ci)
        Z.append(Vi * Ci)
        if vi_token_seqs is not None and i < len(vi_token_seqs):
            vi_t = _as_float_array(vi_token_seqs[i])
            n = min(ci.size, vi_t.size)
            if n == 0:
                Ztoken.append(0.0)
            else:
                Ztoken.append(float(np.mean(ci[:n] * vi_t[:n])))
        else:
            Ztoken.append(Vi * Ci)
    return {"C": C, "Z": Z, "Ztoken": Ztoken}


def average_ranks(scores: Sequence[float], higher_is_better: bool = True) -> list[float]:
    """Average ranks for ties. Rank 1 is best."""
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i], reverse=higher_is_better)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # ranks i..j (0-based positions) map to rank values (i+1)..(j+1)
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = float(avg)
        i = j + 1
    return ranks


def rwc_weights(ranks: Sequence[float]) -> list[float]:
    """wi = (N - ri) / (N - 1) for N > 1; for N=1 return [1]."""
    n = len(ranks)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    return [(n - float(r)) / (n - 1) for r in ranks]
