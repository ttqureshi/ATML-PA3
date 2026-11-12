# ATML PA3 — Task 3: Test-Time Compute

Implements assignment §3 (sampling, selection, VisConf/RWC, live parallel vs sequential, adaptive stopping). Seed **6304** everywhere.

## Layout

- `src/data` — parquet join, splits, boxed extraction, MATH / MathVista equivalence
- `src/metrics` — mean logp, self-certainty, visual engagement, VisConf, ranks
- `src/select` — FirstAns, self-consistency, highest-score, RWC
- `src/adaptive` — agreement / quality-aware stopping + threshold selection
- `src/live` — Qwen2.5-3B-Instruct live Step 4 (single / parallel / sequential)
- `src/eval` — budgets, nested orders, bootstrap CIs
- `src/analysis` — signal analysis + plots
- `scripts/` — `make_notebook.py`, `push_kernel.py`, `monitor_kernel.py`, `preflight_checks.py`

## Data

Place official **ATML-PA3-Data** at:

- `task3/data/ATML-PA3-Data/`, or
- env `ATML_PA3_DATA`, or
- `/kaggle/input/atml-pa3-data` on Kaggle

Expected per-model folders with `generations.parquet`, `probability_metrics.parquet` (+ `attention_metrics.parquet` for VLM).

If official data is missing, **offline Steps 1–3/5–6 are blocked**. Live Step 4 falls back to a 1000-question HF MATH test subset (`DigitalLearningGmbH/MATH-lighteval`), split with seed 6304; live IDs may need re-alignment once official data arrives.

## Local commands

```bash
python scripts/preflight_checks.py
python -m src.run_offline --artifacts-root results
python -m src.run_live --smoke --artifacts-root results_smoke
python -m src.run_live --artifacts-root results   # full 50-q live (GPU)
```

## Kaggle (account `mtayyabtahir`)

Uses `~/.kaggle/kaggle.json` only — **do not** restore `~/.kaggle/access_token`.

```bash
python scripts/make_notebook.py
python scripts/push_kernel.py --only live
kaggle kernels status mtayyabtahir/atml-pa3-t3-live
python scripts/monitor_kernel.py --ref mtayyabtahir/atml-pa3-t3-live
```

- Live URL: https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t3-live
- Smoke URL: https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t3-smoke
- GPU: NvidiaTeslaT4, internet on, private

## Scorer

MATH: `math-verify`. MathVista: local answer-type / choice / precision rules. Missing extractions → single `NO_ANSWER` group (always incorrect, kept in voting/cost/accuracy).
