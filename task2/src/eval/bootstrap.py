from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_metrics(
    pair_records: list[dict],
    stat_fns: dict[str, Callable[[list[dict]], float]],
    n_resamples: int,
    seed: int,
) -> dict[str, dict]:
    """Percentile 95% CIs from resamples of complete pairs.

    One index draw is reused for every statistic, so differences compare the
    same pair IDs across image conditions.
    """
    point = {name: float(fn(pair_records)) for name, fn in stat_fns.items()}
    n_pairs = len(pair_records)
    if n_pairs == 0 or n_resamples <= 0:
        return {
            name: {"point": value, "lo": value, "hi": value, "n_resamples": 0, "n_pairs": n_pairs}
            for name, value in point.items()
        }
    rng = np.random.default_rng(int(seed))
    accum = {name: np.empty(n_resamples, dtype=np.float64) for name in stat_fns}
    for draw in range(n_resamples):
        indices = rng.integers(0, n_pairs, size=n_pairs)
        sample = [pair_records[int(index)] for index in indices]
        for name, fn in stat_fns.items():
            accum[name][draw] = float(fn(sample))
    out: dict[str, dict] = {}
    for name, samples in accum.items():
        lo, hi = np.quantile(samples, [0.025, 0.975])
        out[name] = {
            "point": point[name],
            "lo": float(lo),
            "hi": float(hi),
            "n_resamples": int(n_resamples),
            "n_pairs": n_pairs,
        }
    return out
