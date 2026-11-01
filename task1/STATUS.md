# ATML PA3 Task 1 — execution status

Canonical code: `C:\Users\PMLS\ATML-PA3\task1`  
(Google Drive workspace was out of quota; pointer file left in the Drive folder.)

## Implementation

Complete local package under `task1/`:

- Data filter for MetaMathQA 40k
- Shared SFT trainer
- Five variants: Full FT, Full FT+L2-SP, LoRA-4, LoRA-16, OPLoRA
- lm-eval harness evaluation
- Weight/activation diagnostics
- Qualitative generations (`first_plot_questions.yaml`)
- Tables + figures writers under `results/`

## Kaggle (kicked off)

| Kernel (URL slug) | Role | Notes |
|---|---|---|
| [atml-pa3-t1-smoke](https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t1-smoke) | Smoke test | Used to validate the stack |
| [atml-pa3-t1-prep-baseline](https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t1-prep-baseline) | Data + probe + baseline eval | RUNNING |
| [atml-pa3-t1-train-fullft](https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t1-train-fullft) | Full FT 1250 updates | ERROR on first try (likely huge theta0 dump); fixed, re-push queued |
| [atml-pa3-t1-train-full-ft-l2sp](https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t1-train-full-ft-l2sp) | Full FT + L2-SP | RUNNING |
| atml-pa3-t1-train-full-ft-l2sp | Full FT + L2-SP | Queued when a GPU slot frees (limit 2) |
| atml-pa3-t1-train-lora-r4 | LoRA r=4 | Queued by `scripts/queue_remaining.py` |
| atml-pa3-t1-train-lora-r16 | LoRA r=16 | Queued by monitor |
| atml-pa3-t1-train-oplora | OPLoRA | Queued by monitor |
| atml-pa3-t1-pipeline-post | Eval + diagnostics + plots | Push after all trains finish |

Code dataset: https://www.kaggle.com/datasets/mtayyabtahir/atml-pa3-task1-code

Kaggle allows **2 concurrent GPU sessions**. A background queue script pushes the remaining train kernels as slots open.

## Artifacts

On Kaggle: `/kaggle/working/results/...`  
Pull later with `kaggle kernels output <slug> -p ...` (avoid pulling HF caches; notebooks keep cache under `/tmp`).

## Next (not in this step)

After all five trainings + prep finish: push `pipeline_post`, pull metrics, then compile the Task 1 report PDF when you ask.
