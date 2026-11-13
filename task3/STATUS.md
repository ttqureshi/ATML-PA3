# Task 3 status

**COMPLETE** (offline + gap-fill + official live). Report writeup: `README.md` and `../report/TASK3_GAP_FILL_STATUS.md`.

## Covered

- Offline protocol: join keys, splits (seed 6304), 10 nested rollout orders, budgets `{1,2,4,8,16,32}`
- Answer groups + correctness via `math-verify` (MATH) and MathVista-local rules; `NO_ANSWER` handling
- Quality scores: mean logp \(L_i\), self-certainty \(K_i\), visual engagement \(V_i\), VisConf \(Z_i=V_i C_i\)
- Tokenwise VisConf ablation: `HighestVisConfToken` / `VisConfTokenRWC`
- Aggregators: FirstAns, self-consistency, highest-score, RWC
- Adaptive agreement / quality-aware + `early_stop_analysis` (incorrect early stops vs fixed-32 recovery)
- Live Step 4 on **official** first-50 held-out MATH IDs: single 0.66 / parallel 0.58 / sequential 0.62

## Pointers

- Offline (+ gap-fill merges): `results_offline_kaggle/results/`
- Official live: `results_live_official_kaggle/results/` (`live_source: supplied`)
- HF-fallback archive only: `results_live_kaggle/` — do **not** cite as primary Step 4
- Kernels: `atml-pa3-t3-offline`, `atml-pa3-t3-gapfill`, `atml-pa3-t3-live` on `mtayyabtahir`
