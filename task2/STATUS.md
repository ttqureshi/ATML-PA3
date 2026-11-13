# Task 2 status

**COMPLETE** for the 196-token main recipe. Report writeup should start from `README.md` and `results_t2_export_kaggle/results/`.

## Covered

- Frozen `google/siglip-base-patch16-224` patch cache (196 tokens, FP16, final layer norm once) and removal of the vision encoder before Qwen training
- Connector `Linear(768→1024) → GELU → Linear(1024→1024)` with a recorded `output_scale`
- `Qwen/Qwen3-0.6B`, original weights frozen, `enable_thinking=False`
- LoRA only in stages 2–3: r=16, alpha=32, dropout 0.05, no bias, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`
- One continued run: 2250 / 1500 / 750 updates, microbatch 8, accumulation 2, AdamW weight decay 0.01, clip 1.0, per-stage 5% warmup then cosine to 10% of peak LR
- Stage learning rates 1e-3, then 1e-4 / 2e-5, then 5e-5 / 1e-5
- Stage mixtures: captions; then presence 50% / counting 25% / caption 25%; then natural 70% / presence 10% / counting 10% / caption 10%
- Example-weighted causal CE; FP16 autocast + GradScaler; seed 6304
- Final greedy eval, four image conditions, 4000 pair-bootstrap CIs, learning curves, qualitative pairs, resources

## Deferred

- Optional full 49-token retraining run (`model.num_visual_tokens: 49`, separate artifacts directory).

## Execution log (short)

- Smoke v3 COMPLETE on Kaggle.
- Pipeline v1 ERROR (FP16 GradScaler skips at stage-1 update 1).
- Pipeline after GradScaler/autocast fix COMPLETE; artifacts in `results_pipeline_kaggle/` and slim `results_t2_export_kaggle/`.
- Primary correct-image accuracy: grounding 0.6875; natural 0.5917 (see export `STATUS.md`).

## Interpretations recorded in config

- Dev/test grounding pairs split evenly presence/counting (12/12 and 32/32).
- Training presence/counting pools capped at 2048 then sampled with replacement.
- Mismatched images: fixed cyclic map over pair ids.
