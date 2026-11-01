# ATML PA3 — Task 1: LLM Fine-Tuning

Controlled comparison of Full FT, Full FT + L2-SP, LoRA (r=4, r=16), and OPLoRA (r=16, k=128) on `Qwen/Qwen3-0.6B` using MetaMathQA (40k), evaluated on GSM8K / HellaSwag / BoolQ / ARC-Easy.

## Layout

- `configs/` — shared recipe + per-variant configs
- `src/` — data, training, eval, diagnostics, qualitative, analysis
- `assets/first_plot_questions.yaml` — qualitative prompts
- `results/` — all durable artifacts (logs, checkpoints, metrics, figures, tables)
- `kaggle/` — dataset metadata and kernel push folders

## Local / Kaggle stage runner

```bash
export ATML_ARTIFACTS_ROOT=/kaggle/working/results
python -m src.run_stage --stage prepare_data
python -m src.run_stage --stage train:full_ft
python -m src.run_stage --stage eval:baseline
python -m src.run_stage --stage pipeline_post
python -m src.run_stage --stage smoke --smoke-steps 2
```

## Artifact contract (for the report)

| Path | Contents |
|------|----------|
| `results/raw/data/` | MetaMath 40k JSONL, IDs, filter stats |
| `results/checkpoints/<variant>/` | Resumable mid/final checkpoints |
| `results/logs/` | JSONL learning curves, resource events |
| `results/raw/eval/<condition>/` | lm-eval metrics + per-example outputs |
| `results/diagnostics/` | ρ_k, η, activation R(z) |
| `results/qualitative/` | 168 responses + annotation subset |
| `results/tables/` | CSV/MD benchmark, resource, diagnostic tables |
| `results/figures/` | Loss curves, benchmark bars, diagnostic plots |
| `results/analysis_index.json` | Index of produced tables/figures |

Seed: `6304`. No old-task replay. Thinking disabled (`enable_thinking=False`).
