# Task 3 status

Implementation of ATML-PA3.pdf §3 (Test-Time Compute) lives in this folder.

## Covered

- Offline protocol scaffolding: join keys, splits (seed 6304), 10 nested rollout orders, budgets `{1,2,4,8,16,32}`
- Answer groups + correctness via `math-verify` (MATH) and MathVista-local rules; `NO_ANSWER` handling
- Quality scores: mean logp \(L_i\), self-certainty \(K_i\), visual engagement \(V_i\), VisConf \(Z_i=V_i C_i\) with per-prefix 5th/95th calibration
- Aggregators: FirstAns, self-consistency, highest-score, RWC with average ranks for ties
- Adaptive agreement / quality-aware controllers + dev threshold selection
- Live Step 4: Qwen2.5-3B-Instruct FP16, single 3072 / parallel 3×1024 / sequential 1024+2048, shared initial, timing + peak GPU memory
- Kaggle self-contained notebooks with embedded zip (mirrors Task 2 pattern)

## Data status

- Local copy at `task3/data/ATML-PA3-Data/` (`qwen25-math`, `qwen25vl-mathvista`).
- Kaggle dataset: https://www.kaggle.com/datasets/mtayyabtahir/atml-pa3-data (private zip of official parquets).
- Live Step 4 already ran with HF MATH fallback; offline uses the official dataset.

## Execution log

- 2026-11-10: Implemented Task 3 tree; live kernel COMPLETE (acc 0.76 all strategies; HF fallback).
- 2026-11-11: Local full in-memory offline load froze the machine (~400 MB parquet → multi-GB RAM). Moved offline to Kaggle T4.
- 2026-11-11: Uploaded `mtayyabtahir/atml-pa3-data`; pushed **`mtayyabtahir/atml-pa3-t3-offline` v1** → ERROR (0-byte MathVista `probability_metrics.parquet` on dataset mount).
- 2026-11-11: Re-uploaded dataset v2 with valid parquets; rebuilt notebook (parquet size checks + zip extract); pushed **v2** → RUNNING. Monitor: `task3/results_offline_kaggle/monitor.log`.

## Monitor

```bash
kaggle kernels status mtayyabtahir/atml-pa3-t3-offline
python scripts/monitor_kernel.py --ref mtayyabtahir/atml-pa3-t3-offline --interval 90 --max-minutes 480
# After COMPLETE:
kaggle kernels output mtayyabtahir/atml-pa3-t3-offline -p task3/results_offline_kaggle
```
