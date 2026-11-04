# ATML PA3 — Task 1: LLM Fine-Tuning

**Status: COMPLETE** (benchmarks + diagnostics + qualitative + analysis artifacts ready for the report).  
**Fresh-session purpose of this file:** enable another agent (or human) to read results and produce the Task 1 portion of the 8-page NeurIPS-style PDF against `ATML-PA3.pdf` §1 / Required Evidence / Research Questions.

> **Course AI policy (critical):** generative AI may assist with code, but **must not write the PDF report’s language, interpretation, or analysis**. Use this README + `report_bundle/` to gather facts, tables, figure paths, and an evidence checklist. The student must write the final report prose.

Public repo: https://github.com/ttqureshi/ATML-PA3  
Canonical full artifact tree (Drive; too large for git):  
`G:\My Drive\MS AI\Semester_3\ATML\PAs\ATML-PA3\task1\results_colab\`  
Assignment PDF: `../ATML-PA3.pdf` (or Drive sibling of this folder).

---

## 1. What Task 1 requires (from the PDF)

Controlled comparison on **`Qwen/Qwen3-0.6B`** (thinking off), seed **`6304`**, MetaMathQA first **40k** eligible examples (no GSM8K-test overlap; no shuffle/truncation of answers), **1 epoch / 1250 updates**, shared recipe in `configs/shared.yaml`.

| Variant | Key settings |
|---------|----------------|
| Full FT | peak LR `2e-5`, λ=0 |
| Full FT + L2-SP | same + λ=`1e-3` |
| LoRA r4 | `(r,α)=(4,8)`, peak LR `2e-4` |
| LoRA r16 | `(16,32)`, peak LR `2e-4` |
| OPLoRA | r=16, α=32, protection k=128 |

**Seven evaluation conditions** (all four benchmarks; FP16; GSM8K = `gsm8k_cot` 0-shot greedy, max 1024 new tokens; primary GSM8K metric = **flexible-extract** exact match):

1. `baseline`
2. `full_ft_midpoint` (step 625 / 20k examples)
3. `full_ft_final` (step 1250)
4. `full_ft_l2sp_final`
5. `lora_r4_final`
6. `lora_r16_final`
7. `oplora_final`

**Required evidence in the report (PDF “Required Evidence”):**

1. **Benchmark table** — 7 conditions × 4 benchmarks + Δpp vs baseline  
2. **Learning / regularization** — combined SFT-loss plot; Full FT vs L2-SP penalty / displacement discussion  
3. **Resources** — trainable params, train time, peak GPU mem, checkpoint sizes  
4. **Diagnostics** — ρ_k, η, activation R(z) math vs non-math (layers 0/13/27 `down_proj`)  
5. **Qualitative** — annotate fixed **42**-response subset; short aligned excerpts  

**Research questions to answer (PDF):** specialization/retention; effect of restrictions (L2-SP / rank / OPLoRA); parameter geometry vs behavior; performance vs cost; behavior beyond scores.

**Controlled comparisons the PDF asks for:** LoRA-4 vs LoRA-16; Full FT vs L2-SP; LoRA-16 vs OPLoRA; additionally Full FT vs LoRA-16; Full FT midpoint→final progression.

---

## 2. Start here (5-minute orientation)

| Priority | Path | Why |
|----------|------|-----|
| 1 | `report_bundle/meta/POST_COMPLETE.json` | Machine-readable “Task1 done” marker (`host: colab-t4-resume`, 2026-11-04) |
| 2 | `report_bundle/tables/benchmark_table.md` | Primary scores + Δpp (cite these numbers) |
| 3 | `report_bundle/tables/benchmark_primary.csv` / `benchmark_delta_pp.csv` | Same data, spreadsheet-friendly |
| 4 | `report_bundle/figures/benchmark_scores.png` | Bar figure for the report |
| 5 | `report_bundle/figures/weight_space_diagnostics.png` | ρ_k / η figure |
| 6 | `report_bundle/figures/activation_diagnostics.png` | R(z) figure |
| 7 | `report_bundle/diagnostics/weight_space_all.json` + `tables/weight_space.csv` | Exact ρ_k, η |
| 8 | `report_bundle/diagnostics/activations_summary_all.json` | Math vs non-math R mean/median |
| 9 | `report_bundle/qualitative/annotation_subset_42.json` | Fixed 42 responses to annotate |
| 10 | `report_bundle/metrics/<condition>/summary.json` | Per-condition lm-eval aggregates |
| 11 | `STATUS.md` | Execution history / caveats |
| 12 | `configs/shared.yaml` + `configs/variants/*.yaml` | Hyperparameters to quote |

If `report_bundle/` is missing in a checkout, rebuild from Drive `results_colab/` (same relative layout under `tables/`, `figures/`, `diagnostics/`, `qualitative/`, `raw/eval/*/`).

---

## 3. Final benchmark numbers (GSM8K = flexible-extract)

Source: `report_bundle/tables/benchmark_table.md` (do not use `*__OLD_BAD*` folders on Drive).

| condition | gsm8k | hellaswag_acc_norm | boolq_acc | arc_easy_acc_norm | Δgsm8k (pp) |
|-----------|------:|-------------------:|----------:|------------------:|------------:|
| baseline | 0.431 | 0.473 | 0.644 | 0.560 | 0 |
| full_ft_midpoint | 0.560 | 0.456 | 0.639 | 0.624 | +12.9 |
| full_ft_final | 0.643 | 0.461 | 0.630 | 0.638 | +21.2 |
| full_ft_l2sp_final | 0.660 | 0.464 | 0.664 | 0.634 | +22.9 |
| lora_r4_final | 0.594 | 0.474 | 0.683 | 0.618 | +16.3 |
| lora_r16_final | 0.614 | 0.471 | 0.714 | 0.641 | +18.3 |
| oplora_final | 0.623 | 0.474 | 0.714 | 0.641 | +19.2 |

**Quick reading for analysis (observations only — write interpretations yourself):**

- All methods beat baseline on GSM8K; **L2-SP final is strongest on math** among the five finals.  
- Full FT mid→final: large GSM8K gain; mild HellaSwag/BoolQ dips vs baseline; ARC-Easy rises.  
- LoRA/OPLoRA: smaller GSM8K gains than Full FT / L2-SP, but **BoolQ improves** substantially (esp. r16 / OPLoRA); HellaSwag ≈ baseline.  
- OPLoRA vs LoRA-16: similar GSM8K/ARC; near-identical BoolQ; weight-space ρ_k ≈ 0 for OPLoRA (constraint working) while LoRA-16 ρ_k ~0.10–0.15.  
- **Strict-match GSM8K is often ~0** while flexible-extract is healthy — discuss extractability vs correctness (PDF “What to Watch For”).

---

## 4. Diagnostics cheat-sheet

### Weight space (`down_proj` layers 0, 13, 27; k=128)

- Full FT / L2-SP / LoRA: ρ_k roughly **0.10–0.15**; η ~ **0.008–0.018**.  
- L2-SP η generally **smaller** than Full FT final at the same layers (smaller relative updates).  
- **OPLoRA ρ_k ≈ 0** (numerical confirmation of orthogonal protection); η still comparable to LoRA-16.

### Activations R(z)

See `activations_summary_all.json`. Typical pattern for Full FT: math-domain R often higher than non-math at early/mid layers (e.g. layer 0/13); do not over-claim causation (PDF warns local R(z) ≠ causal proof of behavior).

### Qualitative

- Prompts: `assets/first_plot_questions.yaml` (24 prompts × 7 conditions = 168 gens).  
- Annotation subset: first & last underlying questions × 3 formats × 7 conditions → `report_bundle/qualitative/annotation_subset_42.json`.  
- Full 168 responses live on Drive under `results_colab/qualitative/responses_*.json` and `responses_grouped.json` (not all committed — size).  
- Annotate: format compliance, coherence, repetition, unnecessary math framing, inappropriate content; flag truncation; use short excerpts.

---

## 5. Evidence gaps / caveats (do not invent)

| PDF ask | Status | Notes |
|---------|--------|-------|
| 7-condition benchmark table + Δpp | **Ready** | `report_bundle/tables/` |
| Benchmark figure | **Ready** | `figures/benchmark_scores.png` |
| Combined SFT-loss plot (5 runs) | **Missing in artifacts** | `analysis_index.json` has `"sft_loss": null`. Train JSONL curves were not synced into `results_colab`. Do **not** fabricate curves. Options: recover from Kaggle kernel outputs if still available, or discuss losses qualitatively / omit figure and state limitation. |
| Full FT vs L2-SP Ω / ‖θ−θ₀‖₂ | **Partial** | Use weight-space η / δ_fro as proxy for relative update size; full-model displacement scalars may be absent from synced logs. |
| Resource table | **Partial** | LoRA/OPLoRA train summaries in `report_bundle/resources/` are usable. **Do not cite** `resources/INVALID_nan_fullft_do_not_cite/` — those are from the **failed NaN Full-FT runs** (`skipped_updates: 1249`). Healthy Full-FT ckpts are model-only ~1.50GB (`report_bundle/checkpoint_meta/full_ft_*.json`). Peak mem / wall-time for the **healthy** Full-FT retrain were not re-exported into this bundle. |
| Diagnostics figures + tables | **Ready** | |
| Qualitative 42-subset | **Ready** | Annotation is a writeup task |
| Ignore | Drive `results_colab/raw/eval/*__OLD_BAD*` | Collapsed Full-FT (`!!!!` gens); superseded |

**History (for honesty if asked):** early Full-FT used FP16 without a working GradScaler path → NaN weights. Fixed to FP32 masters + AMP/GradScaler; model-only mid/final saves; healthy weights installed under Drive `colab_data/checkpoints/`; re-eval + post_analyze on Colab T4 → current `POST_COMPLETE`.

---

## 6. Suggested report outline (Task 1 section of the 8-page PDF)

Keep Task 1 tight — the PDF is shared across three tasks.

1. Setup (model, data filter, shared recipe, seed, thinking off) — cite `configs/`.  
2. Benchmark table + short progression / method comparison paragraphs.  
3. Resource comparison (what exists; note Full-FT gap).  
4. Diagnostics: ρ_k / η / R(z) linked carefully to scores.  
5. Qualitative: 42-subset summary + 2–4 aligned excerpts.  
6. Answers to research questions 1–5 in compact form.

Put GitHub URL at end of abstract (assignment rule).

---

## 7. Repo layout & reproduction

```text
task1/
  configs/           # shared + variants
  src/               # prepare / train / eval / diagnostics / qualitative / analysis
  assets/            # first_plot_questions.yaml
  kaggle/kernels/    # notebook wrappers for Kaggle
  scripts/           # push/queue/pull helpers
  report_bundle/     # committed slim results for writeup (THIS is what graders/agents need)
  results/           # local/Kaggle run root (large raw outputs gitignored)
  STATUS.md          # execution status
```

Stage runner:

```bash
cd task1
pip install -r requirements.txt
python -m src.run_stage --stage prepare_data --artifacts-root ./results
python -m src.run_stage --stage train:<variant> --artifacts-root ./results
python -m src.run_stage --stage eval:<condition> --artifacts-root ./results
python -m src.run_stage --stage pipeline_post --artifacts-root ./results
```

Healthy checkpoints (Drive, not in git): `task1/colab_data/checkpoints/{full_ft,full_ft_l2sp,lora_r4,lora_r16,oplora}/...`

---

## 8. Attribution / reuse

- Hugging Face `transformers`, `peft`, `datasets`, `accelerate`  
- `lm-evaluation-harness`  
- Qualitative prompts from [emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) (`first_plot_questions.yaml`)  
- OPLoRA formulation per Xiong & Xie (assignment reading)

Do not commit raw MetaMathQA/GSM8K downloads or full model weights.
