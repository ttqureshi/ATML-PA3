# ATML PA3 — Task 3: Test-Time Compute

**Status: COMPLETE** (offline Steps 1–3/5–6 on official ATML-PA3-Data via Kaggle; live Step 4 COMPLETE on 50 MATH questions).  
**Fresh-session purpose of this file:** enable another agent (or human) to read results and produce the Task 3 portion of the 8-page NeurIPS-style PDF against `ATML-PA3.pdf` §3 / Required Evidence / Research Questions — without inventing numbers, without using gold labels for selection/stopping, and without citing “lecture notes.”

Public repo: https://github.com/ttqureshi/ATML-PA3  
Assignment PDF: `../ATML-PA3.pdf`  
Lecture notes for **notation / theory inspiration only** (never name them in the report prose): `../lecture_notes_for_context/Test Time Scaling Notes.pdf`  

Canonical result packs:

| Pack | Path | Role |
|------|------|------|
| Offline | `results_offline_kaggle/results/` | Curves, method tables, signals, adaptive, splits, STATUS |
| Live | `results_live_kaggle/results/` | 50-q single / parallel / sequential + `live_summary.json` |

Marker: offline `STATUS.md` ends with `OFFLINE_COMPLETE`; live with `LIVE_COMPLETE`.

---

## 0. Hard rules for the report agent (AI usage)

From `ATML-PA3.pdf` Instructions / AI Usage Policy, §3 What to Watch For, and §4 Synthesis:

1. Generative AI may write/compile the PDF; tone must be **human**, not AI-slop. Prefer plain academic prose over polished emptiness.
2. Report must be **self-contained**. Define \(L_i\), \(K_i\), \(V_i\), \(Z_i\), RWC weights, etc. in the paper when used — do **not** write “as in the lecture notes / as discussed in class.”
3. Selection, stopping, and VisConf calibration must use **only revealed candidates**. Gold answers and future rollouts are for retrospective analysis only (failure decomp, oracle coverage).
4. Every reported number → saved JSON/MD/PNG below. Do not invent transition counts or CIs.
5. Offline **token cost is a proxy** (no prefill/attention/scoring wall time in the export). Say so once.
6. Live Step 4 used an **HF MATH fallback** for the 50 IDs because official data was not mounted on that kernel — state this deviation clearly (see §7).
7. Shared 8-page PDF: keep Task 3 tight; appendix for long tables.

**Suggested drafting model for human-like prose:** Claude Opus (latest Opus / Opus-thinking available in the session), with an explicit anti-slop brief (short clauses, define symbols once, interpret CIs honestly, no “robust/comprehensive/delve”). Use a GPT-class model if you want a second pass that stress-checks every claim against the JSON paths in this README.

---

## 1. What Task 3 requires (from the PDF)

**Offline** (supplied rollouts; no new generation except live Step 4):

| Model | Dataset | Questions | Rollouts |
|-------|---------|----------:|---------:|
| Qwen2.5-3B-Instruct | MATH test subset | 1000 | 32 each |
| Qwen2.5-VL-3B-Instruct | MathVista testmini | 1000 | 32 each |

Generation settings in the export: temp 0.7, top-p 0.8, top-k 20, rep-penalty 1.0, 2048 new-token cap.

**Splits (seed 6304):** sort question IDs lexicographically; `default_rng(6304)` permute; first **200** = development, remaining **800** = held-out eval. All rollouts of a question stay together. First **50** MATH eval IDs in that permutation = live subset (intended). Saved under `splits/*_splits.json` and `*_orders.*`.

**Budgets:** \(N\in\{1,2,4,8,16,32\}\). Ten nested random rollout orders per question; average over orders then questions. Ties: earliest supporting rollout in current order; RWC uses average ranks.

**Scores (define in report when used):**

\[
L_i=\frac1{T_i}\sum_t \log p_{i,t}(y_{i,t}),\qquad
K_i=\frac1{T_i}\sum_t k_{i,t},\quad k_{i,t}=D_{\mathrm{KL}}(U\|p_{i,t}).
\]

MathVista visual engagement and VisConf:

\[
v_{i,t}=\frac{A^{\mathrm{image}}_{i,t}}{A^{\mathrm{image}}+A^{\mathrm{prompt}}+A^{\mathrm{generated}}},\quad
V_i=\tfrac1{T_i}\sum_t v_{i,t},
\]

\[
c_{i,t}=\mathrm{clip}_{[0,1]}\!\left(\frac{k_{i,t}-Q_{.05}(G)}{\max(Q_{.95}(G)-Q_{.05}(G),10^{-8})}\right),\quad
C_i=\tfrac1{T_i}\sum_t c_{i,t},\quad Z_i=V_i C_i.
\]

Calibration \(G\) = **current candidate prefix only**. RWC weights with descending average rank \(r_i\) (rank 1 best):

\[
w_i=\frac{N-r_i}{N-1},\quad W(a)=\sum_{i:a_i=a}w_i.
\]

**Methods to report:**

- Both datasets: FirstAns, self-consistency, highest \(L\), highest \(K\), self-certainty+RWC  
- MathVista also: vision-only and VisConf, each highest-score and RWC  

**Adaptive:** check stop at \(n\in\{4,8,16\}\) else 32; return current SC answer. Dev-select \(\tau,\gamma\) with ≤1 pp drop vs fixed-32 SC, minimize mean token cost; freeze before held-out.

**Live Step 4** (same 3072-token total output allowance):

| Strategy | Protocol |
|----------|----------|
| Single | one response ≤3072 |
| Parallel | 3× ≤1024; SC over answers |
| Sequential | shared first parallel as init (≤1024) + critique/revise ≤2048; return revised |

**Required evidence:** sampling/selection curves + failure decomp; signal analysis; live accuracy/cost/runtime/transitions; adaptive thresholds + held-out cost–accuracy; qualitative cases.

**Research questions (weave into analysis, don’t FAQ them):** exploration vs selection; evidence & consensus; breadth vs refinement; adaptive allocation; compute & reliability.

---

## 2. Start here (5-minute orientation)

### Offline (`results_offline_kaggle/results/`)

| Priority | Path | Why |
|----------|------|-----|
| 1 | `STATUS.md` | OFFLINE_COMPLETE + selected adaptive policies |
| 2 | `tables/math_eval_summary.json` | MATH accuracy / oracle / cost by \(N\) + failure decomp |
| 3 | `tables/mathvista_eval_summary.json` | Same + VisConf / vision methods |
| 4 | `figures/math_accuracy_curves.png` | Required MATH curves |
| 5 | `figures/mathvista_accuracy_curves.png` | Required MathVista curves |
| 6 | `tables/math_bootstrap_vs_sc_n32.json` | Paired Δacc vs SC at \(N=32\) (95% intervals) |
| 7 | `tables/mathvista_bootstrap_vs_sc_n32.json` | Same for VLM methods |
| 8 | `analysis/math_signals_n32.json` | Length/score–correctness associations |
| 9 | `analysis/mathvista_signals_n32.json` | + vision / VisConf / \(Z^{\mathrm{token}}-Z\) |
| 10 | `adaptive/math_adaptive.json` | Dev-selected τ/γ + held-out adaptive |
| 11 | `adaptive/mathvista_adaptive.json` | Same (VisConf score) |
| 12 | `configs_used/scorer_versions.json` | Scorer freeze record |
| 13 | `splits/*` | Reproducible IDs / orders |
| 14 | `per_question/*_eval_per_question.json` | Case studies / appendix |

### Live (`results_live_kaggle/results/`)

| Priority | Path | Why |
|----------|------|-----|
| 1 | `STATUS.md` / `live/live_summary.json` | Accuracies, tokens, runtime, transitions |
| 2 | `live/live_results.json` | Per-question generations |
| 3 | `live/live_questions.json` (+ `_full`) | Prompt / ID list |
| 4 | `configs_used/resolved.yaml` | Live settings |

---

## 3. Offline numbers to cite

All held-out **800** questions unless noted. Accuracies averaged over 10 orders.

### MATH (Qwen2.5-3B) — key budgets

| \(N\) | Oracle | FirstAns | SC | Highest \(L\) | Highest \(K\) | \(K\)+RWC | Mean token cost (SC) |
|------:|-------:|---------:|---:|--------------:|------------:|----------:|---------------------:|
| 1 | 0.5815 | 0.5815 | 0.5815 | 0.5815 | 0.5815 | 0.5815 | 665 |
| 4 | 0.7197 | 0.5815 | 0.6291 | 0.6056 | 0.6254 | 0.6324 | 2667 |
| 8 | 0.7655 | 0.5815 | 0.6539 | 0.6084 | 0.6339 | 0.6620 | 5335 |
| 16 | 0.8028 | 0.5815 | 0.6605 | 0.5933 | 0.6388 | 0.6720 | 10667 |
| **32** | **0.8338** | **0.5815** | **0.6655** | **0.5788** | **0.6388** | **0.6791** | **21347** |

Failure decomp at \(N=32\) (avg over orders): **sampling** 133 / **selection** 139  
(`failure_decomposition_n32` in `math_eval_summary.json`).

Bootstrap Δacc vs SC at \(N=32\) (`math_bootstrap_vs_sc_n32.json`):  
FirstAns −0.084 [−0.100, −0.068]; Highest \(L\) −0.087 [−0.111, −0.061]; Highest \(K\) −0.027 [−0.048, −0.004]; **\(K\)+RWC +0.014 [+0.003, +0.024]** (only RWC clearly above SC).

### MathVista (Qwen2.5-VL-3B) — key budgets

| \(N\) | Oracle | FirstAns | SC | Highest \(L\) | Highest \(K\) | \(K\)+RWC | Highest \(V\) | \(V\)+RWC | Highest \(Z\) | \(Z\)+RWC |
|------:|-------:|---------:|---:|--------------:|------------:|----------:|--------------:|----------:|--------------:|----------:|
| 1 | 0.5295 | 0.5295 | 0.5295 | 0.5295 | 0.5295 | 0.5295 | 0.5295 | 0.5295 | 0.5295 | 0.5295 |
| 8 | 0.8209 | 0.5295 | 0.5975 | 0.5296 | 0.5441 | 0.5965 | 0.5324 | 0.5973 | 0.5561 | 0.6011 |
| 16 | 0.8624 | 0.5295 | 0.6150 | 0.5056 | 0.5356 | 0.6186 | 0.5330 | 0.6153 | 0.5599 | 0.6271 |
| **32** | **0.8900** | **0.5295** | **0.6221** | **0.4763** | **0.5375** | **0.6296** | **0.5300** | **0.6288** | **0.5687** | **0.6375** |

Mean SC token cost at \(N=32\): **8898**.  
Failures \(N=32\): sampling **88**, selection **218** (selection gap dominates).

Bootstrap Δacc vs SC @32: Highest \(L\) −0.146; Highest \(K\) −0.085; Highest \(V\) −0.092; Highest \(Z\) −0.053;  
\(K\)+RWC +0.008 (CI crosses 0); \(V\)+RWC +0.007 (crosses 0); **\(Z\)+RWC +0.015 [+0.001, +0.031]** (small but positive).

**Quick reading (observations only):**

- Oracle rises much faster than any selector → repeated sampling creates coverage; selection lags.  
- Mean log-prob as BoN is **harmful** at large \(N\) (esp. MathVista).  
- Pure highest-score confidence/vision underperforms SC; **RWC** recovers / slightly beats SC.  
- MathVista selection failures ≫ sampling failures at \(N=32\).

---

## 4. Signal analysis (\(N=32\))

Sources: `analysis/math_signals_n32.json`, `mathvista_signals_n32.json`.

| Metric | MATH | MathVista |
|--------|-----:|----------:|
| Mean length–correctness corr | −0.093 | −0.083 |
| Top−bottom correctness: mean logp | +0.070 | +0.017 |
| Top−bottom: self-certainty | +0.110 | +0.036 |
| Top−bottom: vision | — | +0.038 |
| Top−bottom: VisConf | — | +0.075 |
| Shortest-rollout accuracy | 0.560 | 0.529 |
| Mean conf–vision corr (within rollout) | — | **−0.209** |
| Mean \(Z^{\mathrm{token}}-Z\) \(=\mathrm{Cov}_t(c,v)\) | — | −0.0022 |

Interpretation hooks (write carefully): self-certainty separates correct/incorrect rollouts better than mean logp on MATH; on MathVista VisConf’s top−bottom gap > vision alone; confidence and vision are **negatively** correlated on average inside a rollout — so product fusion is not “two votes for the same thing.” Tokenwise vs summary fusion difference is tiny here.

---

## 5. Adaptive inference

Dev selects lowest mean token cost with accuracy ≥ fixed-32 SC − 1 pp. Freeze then evaluate held-out.

### MATH (`adaptive/math_adaptive.json`) — score = self-certainty

| Policy | τ | γ | Split | Acc | Mean tokens | Mean rollouts |
|--------|--:|--:|-------|----:|------------:|--------------:|
| Fixed-32 SC | — | — | held-out | 0.6655 | 21347 | 32.0 |
| Agreement (selected) | **0.625** | 0 | held-out | 0.6651 | **10417** | 12.44 |
| Quality-aware (selected) | **0.625** | **0** | held-out | 0.6653 | 10472 | 12.52 |

Dev fixed-32 SC = 0.75; selected agreement/quality ≈ 0.744 on dev.

### MathVista (`adaptive/mathvista_adaptive.json`) — score = VisConf

| Policy | τ | γ | Split | Acc | Mean tokens | Mean rollouts |
|--------|--:|--:|-------|----:|------------:|--------------:|
| Fixed-32 SC | — | — | held-out | 0.6221 | 8898 | 32.0 |
| Agreement (selected) | **0.625** | 0 | held-out | 0.6203 | **4595** | 13.60 |
| Quality-aware (selected) | **0.5** | **0.1** | held-out | 0.6165 | **3723** | 11.28 |

**Quick reading:** adaptive roughly **halves** (MATH) or more (MathVista quality) token cost with ≤~0.6 pp held-out drop vs always-32 SC. On MATH, quality-aware collapsed to γ=0 (same family as agreement). Use `stop_ns_mean_per_q` only for distributional discussion / appendix — do not dump the full arrays in the main PDF.

---

## 6. Live Step 4 numbers

Source: `results_live_kaggle/results/live/live_summary.json`.

| Strategy | Acc | Mean out tokens | Mean in tokens | Trunc | Extract | Mean sec / q |
|----------|----:|----------------:|---------------:|------:|--------:|-------------:|
| Single (≤3072) | **0.76** | 474.3 | 105.0 | 0.00 | 1.00 | 21.05 |
| Parallel 3×1024 + SC | **0.76** | 1327.2 | 315.1 | 0.04 | 0.973 | 58.56 |
| Sequential 1024+2048 | **0.76** | 761.9 | 670.3 | 0.00 | 1.00 | 34.21 |

- Inference wall time (excl. load): **4758.9 s**; peak GPU ≈ **6.03 GiB**; model load ≈ 51 s.  
- Model: `Qwen/Qwen2.5-3B-Instruct`, FP16, batch 1, temp 0.7 / top-p 0.8 / top-k 20.  
- Sequential transitions vs shared initial: **wrong→correct = 0**, **correct→wrong = 0**, correct→correct 38, wrong→wrong 12. Revision did not change graded answers on this set.  
- Cost note in JSON: charge shared initial to both parallel and sequential when comparing.

**Deviation (must state):** this live kernel did **not** see `ATML-PA3-Data`; questions came from **HF MATH fallback** (`STATUS.md`: `live_source: huggingface_fallback`, `n_live_questions: 50`). Offline used the official dataset. Do not claim the live 50 IDs are bit-identical to the offline “first 50 eval” list unless you re-verify against `splits/math_splits.json`.

---

## 7. Evidence gaps / caveats (do not invent)

| PDF ask | Status | Notes |
|---------|--------|-------|
| Accuracy / oracle curves both datasets | **Ready** | figures + `*_eval_summary.json` |
| Method comparison + failure decomp | **Ready** | |
| Signal analysis + \(Z^{\mathrm{token}}\) ablation summary | **Ready** | scalars in analysis JSON; pick illustrative trajectories from `per_question/` if needed |
| Live 3 strategies + costs/transitions | **Ready** | with HF-fallback caveat |
| Adaptive thresholds + held-out | **Ready** | |
| Qualitative aligned examples | **Partial** | No pre-baked “pretty” case card; select from `per_question/*.json` + live results with an explicit selection rule |
| Scorer versions | **Partial** | `math-verify==unknown` string in export — still report what was used (`scorer_versions.json`) |
| Official live ID alignment | **Deviation** | HF fallback |

**Integrity reminders:** NO_ANSWER group always incorrect but kept in votes/cost/accuracy. Missing extractions not dropped. Bootstrap questions, not rollouts.

---

## 8. Suggested report outline (Task 3 section)

1. **Setup** — datasets, splits, budgets, scorers, definitions of \(L,K,V,Z\), RWC, cost proxy.  
2. **Sampling vs selection** — oracle vs FirstAns/SC curves; failure decomp; MATH vs MathVista.  
3. **Scores & consensus** — BoN vs SC vs RWC; bootstrap deltas; why mean-logp fails.  
4. **Signals** — table from §4; conf–vision anticorrelation; keep causal claims humble.  
5. **Adaptive** — selected (τ,γ); held-out accuracy–cost vs fixed-32.  
6. **Live breadth vs refinement** — equal accuracy, different tokens/time; null revision transitions.  
7. Weave RQs 1–5 into the above; end with reliability / limitation paragraph.

---

## 9. Repo layout & reproduction

```text
task3/
  configs/
  src/           # data, metrics, select, adaptive, live, eval, analysis
  scripts/       # make_notebook, push_kernel, monitor, preflight
  data/          # ATML-PA3-Data (gitignored blobs)
  results_offline_kaggle/
  results_live_kaggle/
  STATUS.md
```

```powershell
cd task3
pip install -r requirements.txt
python scripts/preflight_checks.py
python -m src.run_offline --artifacts-root results
python -m src.run_live --artifacts-root results_live
```

Kaggle:

- Offline: https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t3-offline  
- Live: https://www.kaggle.com/code/mtayyabtahir/atml-pa3-t3-live  
- Dataset: https://www.kaggle.com/datasets/mtayyabtahir/atml-pa3-data  

Local full offline previously OOM’d on a laptop; prefer the Kaggle COMPLETE artifacts already pulled.

---

## 10. Attribution / reuse

- Supplied ATML-PA3-Data rollouts (cite assignment data README)  
- `math-verify` for MATH equivalence; MathVista-local answer-type / choice / precision rules  
- Self-certainty / VisConf / RWC formulations: Kang et al.; Usman et al. (cite in References)  
- Self-consistency: Wang et al.; sequential refinement inspiration: Madaan et al.  
- Hugging Face `transformers`, Qwen2.5(-VL) checkpoints  

Do not commit the multi-GB parquet trees or model weights.
