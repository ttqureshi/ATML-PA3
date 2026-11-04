# ATML PA3 Task 1 — execution status

**Updated:** 2026-11-04  
**State:** **COMPLETE** (`report_bundle/meta/POST_COMPLETE.json`, host `colab-t4-resume`)

## Done

| Piece | Status |
|---|---|
| Prep + baseline eval | DONE (GSM8K flex ≈ 0.431) |
| LoRA r4 / r16 / OPLoRA train + eval | DONE |
| Full FT mid / final (healthy retrain) | DONE (GSM8K ≈ 0.560 / 0.643) |
| Full FT + L2-SP final | DONE (GSM8K ≈ 0.660) |
| Diagnostics + qualitative + analyze | DONE |
| Writeup evidence pack | `task1/report_bundle/` committed for report agents |

## Authoritative numbers

Use `report_bundle/tables/benchmark_table.md` and `report_bundle/metrics/*/summary.json`.  
Ignore Drive folders named `*__OLD_BAD*` and any Full-FT train summary with `skipped_updates: 1249`.

## Known gaps for the PDF

- Combined SFT-loss figure not present in synced artifacts (`sft_loss: null` in analysis index).  
- Healthy Full-FT wall-time / peak-mem summaries not re-exported (NaN-run summaries must not be cited).  
- Checkpoint sizes for healthy Full FT: ~1.50GB model-only (see `report_bundle/checkpoint_meta/`).

## Do not redo

- Retrain / re-eval any of the seven conditions unless regenerating missing loss curves or resource logs.
