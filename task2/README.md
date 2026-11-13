# ATML PA3 — Task 2: Vision-Language Models and Visual Grounding

**Status: COMPLETE** (196-token SigLIP–Qwen curriculum trained on Kaggle T4; final eval, bootstrap CIs, learning curves, qualitative pairs, and resource tables ready for the report).  
**Fresh-session purpose of this file:** enable another agent (or human) to read results and produce the Task 2 portion of the 8-page NeurIPS-style PDF against `ATML-PA3.pdf` §2 / Required Evidence / Research Questions — without inventing numbers or citing “lecture notes.”

Public repo: https://github.com/ttqureshi/ATML-PA3  
Assignment PDF: `../ATML-PA3.pdf`  
Lecture notes for **notation / theory inspiration only** (never name them in the report prose): `../lecture_notes_for_context/VLM Notes _2_.pdf`  
Canonical report packs (local Drive / this checkout):

| Pack | Path | Role |
|------|------|------|
| Slim export (prefer for writeup) | `results_t2_export_kaggle/` | Tables, figures, qualitative, configs, STATUS |
| Full pipeline pull | `results_pipeline_kaggle/results/` | Same content + larger raw eval dumps |

Cite numbers from `results_t2_export_kaggle/results/` unless a file is missing there, then fall back to `results_pipeline_kaggle/results/`.

---

## 0. Hard rules for the report agent (AI usage)

From `ATML-PA3.pdf` Instructions / AI Usage Policy and §4 Synthesis:

1. Generative AI may write/compile the PDF, but the **tone must be human**, not AI-slop (no robotic symmetry, no fluff adjectives, no “delve / leverage / robust suite / comprehensive analysis” filler).
2. The report must be **self-contained**. Write equations and short theory in-line when needed.
3. **Forbidden phrasing:** anything like “as we have studied in the lecture notes…”, “as discussed in class…”, “from the notes…”. Use the notes/PDF only to get symbols and definitions right; then state them as part of the paper.
4. Every number must trace to a saved artifact below (or a reproducible command). Do **not** invent CIs, NLL values, or qualitative examples.
5. Prefer concrete comparisons over abstract claims. Explain what a table/figure shows before interpreting it.
6. Task 2 is one subsection of an 8-page main PDF shared with Tasks 1 and 3 — stay compact; put long dumps in an appendix if needed.
7. Put the GitHub URL at the end of the abstract (assignment rule).

**Suggested drafting model for human-like prose:** Claude Opus (latest available Opus / Opus-thinking in the session). Instruct it explicitly: short sentences, uneven rhythm, no buzzwords, define symbols once, interpret numbers with hedging where CIs overlap. After drafting, a human pass should still cut anything that sounds like a blog or a marketing summary. GPT-class models are fine for checklisting evidence; Opus tends to sound less polished-empty when constrained this way.

---

## 1. What Task 2 requires (from the PDF)

Connect frozen **`google/siglip-base-patch16-224`** to **`Qwen/Qwen3-0.6B`** via a trainable MLP connector; keep original LM weights frozen; enable LoRA only in Stages 2–3. Thinking off (`enable_thinking=False`). Seed **`6304`**. One continued run, **4500** updates, effective batch **16**.

| Stage | Updates | Trainable | Peak LRs | Mixture |
|-------|--------:|-----------|----------|---------|
| 1 Caption alignment | 2250 | connector only | connector `1e-3` | 18k captions (2 passes) |
| 2 Visual instruction | 1500 | connector + LoRA | `1e-4` / `2e-5` | presence 50% / counting 25% / caption 25% |
| 3 Natural VQA | 750 | connector + LoRA | `5e-5` / `1e-5` | natural 70% / presence 10% / counting 10% / caption 10% |

**Architecture (as implemented):**

- Visual tokens: **196** SigLIP patch features (14×14), final LN applied **once**, cached FP16; vision tower unloaded before Qwen training.
- Connector: `Linear(768→1024) → GELU → Linear(1024→1024)` + recorded `output_scale`.
- LoRA: `r=16`, `α=32`, dropout `0.05`, no bias; targets `q_proj,k_proj,v_proj,o_proj`.
- Loss (example-weighted causal CE on caption/answer + EOT; visuals/prompts/pad masked):

\[
L = -\frac{1}{|B|}\sum_{i\in B}\frac{1}{|R_i|}\sum_{t\in R_i}\log p(y_{i,t}\mid I_i,q_i,y_{i,<t}).
\]

- Image-swap penalty on teacher-forced answer NLL:

\[
\Delta\mathrm{NLL} = \mathrm{NLL}(y\mid q,I_{\mathrm{partner}}) - \mathrm{NLL}(y\mid q,I_{\mathrm{correct}}).
\]

**Final eval conditions (fixed question + references):** correct image, complementary partner, mismatched unrelated image, no image. Greedy, 16 new tokens, batch 8. Grounding = normalized exact match; natural = local VQA soft consensus over 10 annotators. Pair-bootstrap **4000** resamples for 95% CIs.

**Required evidence (PDF):**

1. Learning curves (train loss, caption loss, answer NLL, ΔNLL) with stage boundaries  
2. Final tables for both suites × 4 image conditions (+ subtypes, pair diagnostics, NLL, advantages, CIs)  
3. Qualitative complementary examples + selection rule  
4. Resources (train/gen time, peak GPU, trainable params by stage, feature-cache cost)  
5. Optional 49-token compression — **not run** (state as omitted)

**Research questions (organize analysis around these; do not answer as a numbered FAQ):**

1. Learning a visual interface (curriculum / stage carry-over)  
2. Grounding and generalization (annotation-derived vs natural)  
3. Measuring image use (likelihood vs generation vs pair metrics vs qualitative)  
4. Capabilities and cost  

---

## 2. Start here (5-minute orientation)

| Priority | Path under `results_t2_export_kaggle/results/` | Why |
|----------|------------------------------------------------|-----|
| 1 | `STATUS.md` | Primary accuracies + resources one-pager |
| 2 | `tables/final_results.md` | **Cite these** final numbers + CIs |
| 3 | `tables/final_advantages.csv` | Correct−complementary accuracy / NLL advantages |
| 4 | `tables/dev_metrics.md` / `.csv` | Stage-wise NLL / ΔNLL for learning-curve discussion |
| 5 | `figures/learning_curves.png` | Required learning-curve figure |
| 6 | `figures/final_scores.png` | Final accuracy bars |
| 7 | `figures/qualitative_pairs.png` | Visual for qualitative section |
| 8 | `qualitative/complementary_examples.json` | Questions, refs, preds + **selection_rule** string |
| 9 | `tables/resources.md` / `resources/*.json` | Params, wall time, peak mem |
| 10 | `configs_used/resolved.yaml` + `connector_calibration.json` | Hyperparams + `output_scale≈0.05523` |
| 11 | `raw/eval/final/summary.json` | Machine-readable final aggregates |
| 12 | `../STATUS.md` (task2 root) | Execution history / deviations |

Also useful: `tables/dev_metrics.csv`, `logs/train_metrics.jsonl`, `logs/dev_metrics.jsonl`.

---

## 3. Final numbers to cite (do not round past what you need)

Source: `tables/final_results.md` / `STATUS.md`. Brackets = 95% pair-bootstrap percentile intervals.

### Correct-image primary accuracy

| Suite | Accuracy | Adv. vs complementary | Notes |
|-------|---------:|----------------------:|-------|
| Grounding overall | **0.6875** [0.5938, 0.7734] | **+0.5312** | Presence strong; counting weaker |
| Natural overall | **0.5917** [0.5297, 0.6526] | **+0.3365** | Soft VQA score |

### Grounding by type (correct image)

| Slice | Acc | Pair acc | Identical-answer rate | Mean NLL |
|-------|----:|---------:|----------------------:|---------:|
| presence | 0.9062 | 0.8125 | 0.1875 | 0.321 |
| counting | 0.4688 | 0.2188 | 0.3438 | 1.272 |
| overall | 0.6875 | 0.5156 | 0.2656 | 0.797 |

### Natural by type (correct image)

| Slice | Acc | Pair acc | Identical-answer rate | Mean NLL |
|-------|----:|---------:|----------------------:|---------:|
| yes/no | 0.7111 | 0.3889 | 0.5278 | 0.627 |
| number | 0.4271 | 0.0833 | 0.4167 | 1.701 |
| other | 0.5819 | 0.3056 | 0.2778 | 1.477 |
| overall | 0.5917 | 0.2812 | 0.4062 | 1.214 |

### Image controls (overall suites) — interpret carefully

| Suite | Condition | Acc | Pair acc | Identical | Mean NLL |
|-------|-----------|----:|---------:|----------:|---------:|
| grounding | complementary | 0.1562 | 0.0156 | 0.2656 | 3.235 |
| grounding | mismatched | 0.3594 | 0.0000 | **1.0000** | 1.990 |
| grounding | no_image | 0.0000 | 0.0000 | 1.0000 | 5.848 |
| natural | complementary | 0.2552 | 0.0312 | 0.4062 | 2.558 |
| natural | mismatched | 0.3177 | 0.0000 | **1.0000** | 2.227 |
| natural | no_image | 0.0000 | 0.0000 | 1.0000 | 8.902 |

Empty-answer rate is **0** across reported cells. Mean gen time ≈ **0.089 s/answer**.

**Quick reading (observations only — write the interpretation yourself):**

- Correct ≫ complementary on both suites → answers move with the image, not only the question text.  
- **Presence** grounding is near-solved; **counting** is the weak slice (pair accuracy 0.22).  
- **No-image** accuracy collapses to 0 with much higher NLL → visual tokens are necessary under this eval (also a shorter prefix; PDF warns about that confound).  
- **Mismatched** identical-answer rate = 1.0 → both sides of a pair often emit the *same* string under a shared unrelated image (language prior / template collapse), while pair accuracy is 0.  
- Natural yes/no retains more complementary accuracy (0.41) than other types → language priors remain visible.  
- Stage-2 instruction tuning is where ΔNLL and correct-image NLL jump (see §4).

---

## 4. Learning / curriculum cheat-sheet

Source: `tables/dev_metrics.md` (dev teacher-forced metrics).

| Update | Stage | Caption loss | Ground ΔNLL | Natural ΔNLL | Ground NLL correct |
|-------:|-------|-------------:|------------:|-------------:|-------------------:|
| 0 | init | 5.52 | 0.05 | −0.15 | 12.13 |
| 2250 | end caption | 2.36 | 0.09 | 0.34 | 6.24 |
| 3750 | end instruction | 2.37 | **4.34** | 1.62 | **0.67** |
| 4500 | end natural | 2.39 | 3.25 | **1.99** | 0.75 |

**Figure:** `figures/learning_curves.png` — mark stage boundaries at 2250 / 3750 / 4500.

Calibration (training captions only): `output_scale = mean_L2(text)/mean_L2(connector) ≈ 0.05523` (`configs_used/connector_calibration.json`).

---

## 5. Resources

Source: `tables/resources.md` / `STATUS.md`.

| Item | Value |
|------|------:|
| Feature-cache time | ~181 s |
| Feature-cache peak GPU | ~315 MB |
| Train loop time | ~8343 s (~2.3 h) |
| Train peak GPU | ~4070 MB |
| Examples seen | 72,000 |
| Updates | 4,500 |
| Stage-1 trainable | 1,837,056 (connector) |
| Stage-2/3 trainable | 6,424,576 (connector 1.84M + LoRA 4.59M) |
| Mean gen time / answer | ~0.089 s |

Optional 49-token rerun: **not executed**. Code supports `model.num_visual_tokens: 49` with a separate artifacts root; do not invent a comparison.

---

## 6. Qualitative guidance

File: `qualitative/complementary_examples.json`.

- Embedded `selection_rule`: ranks pairs by correct-image exact matches, partner-follow under complementary, prediction diversity, then mean ΔNLL; keeps best of each subtype then fills to four pairs/suite.  
- For the report: pick 2–4 aligned examples (presence + counting + one natural). Show shared question, both images’ refs, preds under correct vs complementary.  
- Discuss: does complementary follow the partner answer, stay stuck on the original, or invent a third string?  
- Figure: `figures/qualitative_pairs.png`.

---

## 7. Evidence gaps / caveats (do not invent)

| PDF ask | Status | Notes |
|---------|--------|-------|
| Learning curves | **Ready** | `figures/learning_curves.png` + `dev_metrics.*` |
| Final 4-condition tables + CIs | **Ready** | `tables/final_results.md` |
| Qualitative pairs + selection rule | **Ready** | JSON + PNG |
| Resources by stage | **Ready** | |
| 49-token optional | **Not run** | State omission; no fabricated comparison |
| Full VQAv2/COCO server scores | **N/A** | Local subset only (PDF) |
| Ignoring smoke / failed pipeline v1 | Ignore `results_smoke_*`, early ERROR kernels | Final = pipeline after GradScaler fix |

**Honesty notes for Methods:**

- Dev/test grounding totals (24 / 64 pairs) are split **evenly** presence/counting (12/12, 32/32) so subtypes can be scored; PDF only states combined totals.  
- Training grounding pools capped at 2048 unique pairs/type then sampled with replacement (not a PDF quota).  
- Mismatched map is a fixed cyclic map over pair IDs.  
- Early full pipeline v1 hit FP16 GradScaler skips; fixed (collate/connector outside autocast, lower init scale, FP32 fallback). Final reported run is the successful COMPLETE kernel.

---

## 8. Suggested report outline (Task 2 section)

Keep tight for the shared 8-page limit.

1. **Setup** — SigLIP frozen, connector, Qwen frozen + LoRA, seed, data/splits, loss, four image conditions (cite `configs/shared.yaml`). Define ΔNLL and soft VQA score briefly.  
2. **Curriculum** — learning curves; what Stage 1 vs 2 vs 3 changes (caption loss vs ΔNLL).  
3. **Final grounding table** — conditions × presence/counting; advantages; what complementary vs mismatched vs no-image show.  
4. **Natural VQA table** — yes/no vs number vs other; priors.  
5. **Qualitative** — 2–4 pairs; selection rule one sentence.  
6. **Cost** — params / time / mem.  
7. Fold answers to RQs 1–4 into the narrative (not a separate Q&A list).

---

## 9. Repo layout & reproduction

```text
task2/
  configs/shared.yaml
  src/                 # prepare_data, cache_features, train, eval_final, analyze
  kaggle/kernels/      # smoke, pipeline, export
  scripts/             # make_notebook, push, monitor
  results_t2_export_kaggle/   # WRITEUP PACK (prefer)
  results_pipeline_kaggle/    # full Kaggle pull
  STATUS.md
```

```powershell
cd task2
pip install -r requirements.txt
python -m src.run_stage --stage pipeline --artifacts-root results
# or stages: prepare_data | cache_features | train | eval_final | analyze
```

Kaggle (account `mtayyabtahir`, T4, internet on):  
https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t2-pipeline  

---

## 10. Attribution / reuse

- Hugging Face `transformers`, `peft`, `datasets`, SigLIP & Qwen checkpoints  
- COCO 2014 captions/instances; VQAv2 questions/annotations/complementary pairs  
- VQA answer normalization / soft scoring (local)  
- Visual instruction tuning / connector pattern: Liu et al. (cite in References)  
- Complementary-pair motivation: Goyal et al. (optional cite)

Do not commit raw COCO/VQAv2 dumps, JPEG caches, or multi-GB checkpoints.
