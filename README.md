# ATML PA3 — Advanced Topics in Machine Learning (Fall 2026)

Public code repository for Programming Assignment 3.

**Course:** EE-5102 / CS-6304  
**Author:** Muhammad Tayyab Tahir Qureshi  

## Repository layout

| Path | Contents |
|------|----------|
| `task1/` | Task 1 — LLM fine-tuning (Full FT, L2-SP, LoRA, OPLoRA) |
| `task1/configs/` | Shared training recipe + per-variant configs |
| `task1/src/` | Data prep, trainer, eval, diagnostics, qualitative, analysis |
| `task1/kaggle/` | Kaggle kernel metadata + notebooks |
| `task1/scripts/` | Dataset packaging and Kaggle push/queue helpers |
| `task1/assets/` | Qualitative prompts (`first_plot_questions.yaml`) |

Tasks 2–3 will be added in later commits as they are implemented.

## Task 1 quick start

```bash
cd task1
pip install -r requirements.txt

# Prepare MetaMathQA 40k subset + activation probe IDs
python -m src.run_stage --stage prepare_data --artifacts-root ./results

# Train one variant
python -m src.run_stage --stage train:full_ft --artifacts-root ./results
python -m src.run_stage --stage train:full_ft_l2sp --artifacts-root ./results
python -m src.run_stage --stage train:lora_r4 --artifacts-root ./results
python -m src.run_stage --stage train:lora_r16 --artifacts-root ./results
python -m src.run_stage --stage train:oplora --artifacts-root ./results

# Evaluate / diagnose / plot
python -m src.run_stage --stage eval:baseline --artifacts-root ./results
python -m src.run_stage --stage pipeline_post --artifacts-root ./results
```

Kaggle execution uses the notebooks under `task1/kaggle/kernels/` with the stage runner `python -m src.run_stage --stage ...`.

Seed: `6304`. Model: `Qwen/Qwen3-0.6B` with thinking disabled.

## Reproducibility notes

- Random seeds, configs, and filter manifests are written under `results/configs_used/` and `results/raw/data/` during a run.
- Do not commit raw MetaMathQA / GSM8K downloads or large checkpoints; pull Kaggle kernel outputs separately if needed.
- External code reused: Hugging Face `transformers` / `peft` / `datasets`, `lm-evaluation-harness`, and qualitative prompts from [emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) (`first_plot_questions.yaml`).

## License

Course assignment code for academic use.
