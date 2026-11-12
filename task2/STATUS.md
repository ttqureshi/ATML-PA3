# Task 2 status

Implementation of ATML-PA3.pdf §2 is in this folder. No full 4500-update training run has been executed here.

## Covered

- Frozen `google/siglip-base-patch16-224` patch cache (196 tokens, FP16, final layer norm once) and removal of the vision encoder before Qwen training
- Connector `Linear(768→1024) → GELU → Linear(1024→1024)` with a recorded `output_scale`
- `Qwen/Qwen3-0.6B`, original weights frozen, `enable_thinking=False`
- LoRA only in stages 2–3: r=16, alpha=32, dropout 0.05, no bias, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`
- One continued run: 2250 / 1500 / 750 updates, microbatch 8, accumulation 2, AdamW weight decay 0.01, clip 1.0, per-stage 5% warmup then cosine to 10% of peak LR
- Stage learning rates 1e-3, then 1e-4 / 2e-5, then 5e-5 / 1e-5
- Stage mixtures: captions; then presence 50% / counting 25% / caption 25%; then natural 70% / presence 10% / counting 10% / caption 10%
- Each stage-2/3 draw contributes two examples (a full pair or two captions)
- Example-weighted causal CE on answer/caption tokens plus end-of-turn; visual tokens, prompts, and padding masked
- FP16 autocast, dynamic GradScaler, FP16 frozen backbone, FP32 trainable parameters, unscale before clipping, gradient checkpointing, no quantization
- Token caps 80 / 64 / 16
- COCO 2014 captions and instances plus VQAv2 questions, annotations, and complementary pairs downloaded from the official URLs; only selected image ids are fetched
- Split sizes, seed 6304, saved image/question/pair ids, eval images image-disjoint and absent from train/replay
- Natural pairs require normalized consensus answers that differ, each with at least 8/10 agreement; the training target is an original string from the consensus group; all 10 answers are kept
- Dev NLL on correct / complementary / no-image, caption reference loss, ΔNLL, schedule before training, every 500 updates, and at 2250 / 3750 / 4500
- Final greedy eval, 16 new tokens, batch 8, four image conditions, grounding exact match, natural soft consensus, pair accuracy, identical-answer rate, empty-answer rate, mean reference NLL, generation time, correct-image advantages, 4000 pair-bootstrap 95% CIs
- Resumable checkpoints every 250 updates and stage-end connector/adapter weights, including optimizer, scheduler, scaler, sampler position, stage progress, and RNG
- Smoke path (`ATML_SMOKE=1` or `--stage smoke`)
- Kaggle notebook builder for `mtayyabtahir`, T4, internet on

## Deferred

- The optional full 49-token retraining run. Set `model.num_visual_tokens: 49` and use a separate artifacts directory. The 2×2 average is implemented and the cache remains 196 tokens.

## Execution log

- 2026-11-10: Code review fixes — FP16/FP32 visual-token cast before LM `torch.cat`; refuse to detach connector graphs in `run_lm_forward`; per-stage sampler seeds.
- 2026-11-10: CPU preflight (`scripts/preflight_checks.py`) passed.
- 2026-11-10: Smoke **v2 ERROR** — `discover_prompt_split` crashed because Qwen `apply_chat_template(tokenize=True)` returned a BatchEncoding (`input_ids` dict). Fixed in `src/models/vlm.py` via `_coerce_token_ids`.
- 2026-11-10: Pushed smoke **version 3** with that fix. Versions 1–2 are stale.
- Data prep on the failed run succeeded (COCO/VQAv2 annotation paths were written).
- 2026-11-10: Smoke **v3 COMPLETE**. Analysis wrote `STATUS.md`; stage1/2/3-end checkpoints + LoRA present. Train loop ~25s on smoke; peak train GPU ~3.7 GB.
- 2026-11-10: Pushed full pipeline `mtayyabtahir/atml-pa3-t2-pipeline` **version 1** (4500 updates). Monitor until COMPLETE (~5–12h expected).
- 2026-11-10: Pipeline **v1 ERROR** after ~12 min — `GradScaler skipped 5 attempts at update 1` (stage1 caption). Calibration/`DEV update=0` succeeded (`output_scale≈0.055`). Smoke had passed; full batches overflowed FP16 grads.
- 2026-11-10: Fix — collate/connector outside autocast; GradScaler `init_scale=2**12`; more skip retries; FP32 fallback after repeated skips. Rebuild+push pipeline **v2**.
- Smoke URL: https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t2-smoke
- Pipeline URL: https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t2-pipeline
- Monitor: `python scripts/monitor_kernel.py --ref mtayyabtahir/atml-pa3-t2-pipeline --max-minutes 720`
- After COMPLETE: `kaggle kernels output mtayyabtahir/atml-pa3-t2-pipeline -p task2/results_pipeline_kaggle`

## Interpretations recorded in config

- Dev/test grounding pairs are split evenly across presence and counting (12/12 and 32/32). The PDF states the combined totals only.
- Training presence and counting pools are capped at 2048 unique-image pairs each, then drawn with replacement. That cap is not a PDF quota.
- Smoke captions are 32 examples with one caption per image so the count stays exact. The full run uses 6000 images × 3 captions.
- Mismatched images are a fixed cyclic map over pair ids. Generation recomputes the prefix each new token instead of using a KV cache; the decoded tokens are the greedy tokens either way.
- Pair accuracy uses normalized exact match to the canonical answer. Two empty predictions count as identical. Empties stay in accuracy denominators.
