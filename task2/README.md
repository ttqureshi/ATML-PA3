# ATML PA3 — Task 2: Vision-Language Models and Visual Grounding

SigLIP (`google/siglip-base-patch16-224`, frozen) is connected to Qwen (`Qwen/Qwen3-0.6B`) by a two-layer MLP. One continued run has three stages: caption alignment, visual instruction tuning with LoRA, and natural VQA. Thinking is disabled (`enable_thinking=False`). Seed is **6304**.

The full recipe is in `configs/shared.yaml` and matches ATML-PA3.pdf §2: 4500 updates, effective batch 16, and the stage learning rates, mixtures, and evaluation controls specified there.

## Layout

- `src/run_stage.py` — `prepare_data`, `cache_features`, `train`, `eval_final`, `analyze`, `pipeline`, `smoke`
- `src/data/` — VQA normalization, split selection, image download, feature cache, samplers
- `src/models/` — connector, SigLIP patch features, Qwen forward, LoRA
- `src/train/` — example-weighted trainer and resumable checkpoints
- `src/eval/` — soft/exact scoring, greedy final eval, pair bootstrap
- `src/analysis/` — learning curves, tables, qualitative pairs
- `kaggle/kernels/` — T4 notebooks with internet on, user `mtayyabtahir`

Artifacts land in `results/` (or `ATML_ARTIFACTS_ROOT`).

## Smoke locally

Smoke uses 32 caption examples, 8 pairs in each training pool, 2 updates per stage, and 50 bootstrap resamples. It still runs prepare → cache → train → eval → analyze, so it downloads the annotation zips, the selected JPEGs, SigLIP, and Qwen. That is a short GPU job, not a CPU unit test.

PowerShell, from this directory:

```powershell
$env:ATML_SMOKE = "1"
python -m src.run_stage --stage smoke --artifacts-root results_smoke --fresh
```

Or without the environment variable:

```powershell
python -m src.run_stage --stage smoke --smoke-steps 2 --artifacts-root results_smoke --fresh
```

`--smoke-steps` is the update count of each stage (2 by default, 4 is also within the smoke caption plan).

## Full run

```powershell
python -m src.run_stage --stage pipeline --artifacts-root results
```

Stages can be launched on their own: `--stage prepare_data`, `cache_features`, `train`, `eval_final`, `analyze`. Training auto-resumes from the latest `checkpoints/vlm/step_*` directory. Pass `--fresh` to ignore it.

Kaggle notebooks:

```powershell
python scripts/make_notebook.py
python scripts/push_kernel.py
```

Both kernels use a single Nvidia T4 and enable internet. `atml-pa3-t2-smoke` runs the smoke pipeline. `atml-pa3-t2-pipeline` runs the 4500-update recipe.

## What the trainer is doing

Visual tokens are the 196 SigLIP patch features in 14×14 order after the encoder's final layer norm **once**. They are cached in FP16, then the vision tower is removed before Qwen is loaded. The connector is `Linear(768→1024) → GELU → Linear(1024→1024)`. A buffer `output_scale` matches the mean L2 of connector outputs to the mean L2 of training-caption token embeddings (`results/configs_used/connector_calibration.json`).

The language-model forward is **not** wrapped in `torch.no_grad()`. Frozen weights use `requires_grad=False`. `enable_input_require_grads()` is called so checkpointing still feeds gradients to the connector and, from stage 2, to LoRA. Loss is causal cross-entropy on the caption or answer tokens plus the end-of-turn token. Visual tokens, the prompt, and padding are masked. The loss averages token NLL inside each example, then averages examples in the effective batch.

Dev evaluation runs before training, every 500 updates, and at the stage ends. It records caption reference loss and answer NLL (termination excluded) under the correct image, the complementary image, and no image. Final eval is greedy, 16 new tokens, batch 8, with a fixed mismatched-image map as the fourth condition.

## Notes that are easy to miss

- Annotation-derived dev/test totals are 24 and 64 pairs. This repo splits those evenly into presence and counting (12/12 and 32/32) so both types can be scored.
- Training grounding pools are capped (2048 pairs per type) and then sampled with replacement. The PDF does not require a fixed unique training-pair count.
- `model.num_visual_tokens: 196` can be set to `49`. That averages each 2×2 block before the connector. The feature cache stays 196-wide. A 49-token run needs its own artifacts directory and is not trained here.
- Pair accuracy is exact match to the canonical answer on both images. Natural-question accuracy is the VQA soft score against all 10 annotators. Empty generations stay in the denominators.
