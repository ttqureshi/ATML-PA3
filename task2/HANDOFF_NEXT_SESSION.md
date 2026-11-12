# Task 2 handoff

## Where we are

- Full Task 2 codebase is under `task2/` (SigLIP→connector→Qwen3-0.6B, 3-stage curriculum, grounding eval, analysis).
- Kaggle CLI is authenticated as **mtayyabtahir** via `~/.kaggle/kaggle.json` (LEGACY_API_KEY). Do **not** restore the old `access_token` file or auth flips back to another account.
- GPU quota: 30.00h remaining (as of smoke push).
- Smoke kernel pushed: https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t2-smoke (version 2 embeds the dtype-cast + connector-grad fixes).
- Full pipeline notebook is built locally but **not** pushed yet: `task2/kaggle/kernels/atml-pa3-t2-pipeline/`.

## Next actions

1. Wait for smoke to leave `QUEUED` → `RUNNING` → `COMPLETE`.
   ```powershell
   $kj = Get-Content "$env:USERPROFILE\.kaggle\kaggle.json" -Raw | ConvertFrom-Json
   $env:KAGGLE_USERNAME=$kj.username; $env:KAGGLE_KEY=$kj.key
   python task2/scripts/monitor_kernel.py --ref mtayyabtahir/atml-pa3-t2-smoke
   ```
2. On COMPLETE, download output and confirm `SMOKE OK` / required artifacts:
   ```powershell
   kaggle kernels output mtayyabtahir/atml-pa3-t2-smoke -p task2/results_smoke_kaggle
   ```
3. Only then push the full 4500-update run:
   ```powershell
   cd task2
   python scripts/make_notebook.py   # refresh embedded zip if code changed
   python scripts/push_kernel.py --only pipeline
   ```
4. After pipeline completes, pull `/kaggle/working/results` for report tables/figures.

## Do not

- Do not launch the full pipeline before smoke COMPLETE.
- Do not cite Task 1 NaN Full-FT artifacts.
- Optional 49-token compression is implemented as a config flag but not scheduled.
