from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.data.answer_equiv import grade_against_gold, scorer_versions
from src.data.boxed import extract_last_boxed
from src.select.rules import self_consistency


SYSTEM_DEFAULT = "You are a helpful assistant."
USER_SUFFIX_DEFAULT = (
    "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
)
REVIEW_DEFAULT = (
    "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
    "Give your final answer inside \\boxed{}."
)


def _peak_gpu_memory_bytes() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    live = cfg["live"]
    model_id = live["model_id"]
    revision = live.get("model_revision")
    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": None,
    }
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer


def build_user_content(problem: str, suffix: str) -> str:
    return f"{problem.strip()}\n\n{suffix.strip()}"


def apply_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_once(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = apply_chat(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    seq = out[0]
    new_tokens = seq[input_len:]
    n_new = int(new_tokens.numel())
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    truncated = n_new >= int(max_new_tokens)
    extracted = extract_last_boxed(text)
    return {
        "prompt": prompt,
        "messages": messages,
        "response": text,
        "extracted_answer": extracted,
        "input_tokens": input_len,
        "output_tokens": n_new,
        "truncated": truncated,
        "elapsed_sec": elapsed,
        "seed": seed,
        "max_new_tokens": int(max_new_tokens),
    }


def run_question_strategies(
    model,
    tokenizer,
    question: dict[str, Any],
    cfg: dict,
    base_seed: int,
) -> dict[str, Any]:
    live = cfg["live"]
    system = live.get("system_message", SYSTEM_DEFAULT)
    suffix = live.get("user_suffix", USER_SUFFIX_DEFAULT)
    review = live.get("review_prompt", REVIEW_DEFAULT)
    problem = question["problem"]
    user = build_user_content(problem, suffix)

    # --- Strategy A: single longer response ---
    single_msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    single = generate_once(
        model,
        tokenizer,
        single_msgs,
        max_new_tokens=int(live["single_max_new_tokens"]),
        temperature=float(live["temperature"]),
        top_p=float(live["top_p"]),
        top_k=int(live["top_k"]),
        repetition_penalty=float(live["repetition_penalty"]),
        seed=base_seed + 1,
    )

    # --- Strategy B: parallel sampling (3 x 1024) ---
    parallel = []
    for i in range(int(live.get("n_parallel", 3))):
        parallel.append(
            generate_once(
                model,
                tokenizer,
                single_msgs,
                max_new_tokens=int(live["parallel_max_new_tokens"]),
                temperature=float(live["temperature"]),
                top_p=float(live["top_p"]),
                top_k=int(live["top_k"]),
                repetition_penalty=float(live["repetition_penalty"]),
                seed=base_seed + 100 + i,
            )
        )

    # Self-consistency over parallel extracted answers.
    answers = [p["extracted_answer"] for p in parallel]
    sc = self_consistency(answers, list(range(len(answers))), dataset="math")

    # --- Strategy C: sequential refinement; reuse first parallel as initial ---
    initial = parallel[0]
    refine_msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": initial["response"]},
        {"role": "user", "content": review},
    ]
    revision = generate_once(
        model,
        tokenizer,
        refine_msgs,
        max_new_tokens=int(live["refine_revision_max_new_tokens"]),
        temperature=float(live["temperature"]),
        top_p=float(live["top_p"]),
        top_k=int(live["top_k"]),
        repetition_penalty=float(live["repetition_penalty"]),
        seed=base_seed + 200,
    )

    gold = question.get("gold_answer")
    single_correct = bool(grade_against_gold(single["extracted_answer"], gold, "math")) if gold else None
    parallel_correct = bool(grade_against_gold(sc["answer"], gold, "math")) if gold else None
    initial_correct = bool(grade_against_gold(initial["extracted_answer"], gold, "math")) if gold else None
    revised_correct = bool(grade_against_gold(revision["extracted_answer"], gold, "math")) if gold else None

    transition = None
    if initial_correct is not None and revised_correct is not None:
        if (not initial_correct) and revised_correct:
            transition = "wrong_to_correct"
        elif initial_correct and (not revised_correct):
            transition = "correct_to_wrong"
        elif initial_correct and revised_correct:
            transition = "correct_to_correct"
        else:
            transition = "wrong_to_wrong"

    # Costs: charge shared initial to both parallel and sequential.
    parallel_output_tokens = sum(p["output_tokens"] for p in parallel)
    parallel_input_tokens = sum(p["input_tokens"] for p in parallel)
    sequential_output_tokens = initial["output_tokens"] + revision["output_tokens"]
    sequential_input_tokens = initial["input_tokens"] + revision["input_tokens"]

    return {
        "sample_id": question["sample_id"],
        "gold_answer": gold,
        "single": {
            **single,
            "final_answer": single["extracted_answer"],
            "correct": single_correct,
            "total_output_tokens": single["output_tokens"],
            "total_input_tokens": single["input_tokens"],
        },
        "parallel": {
            "responses": parallel,
            "final_answer": sc["answer"],
            "selected_index": sc["index"],
            "correct": parallel_correct,
            "total_output_tokens": parallel_output_tokens,
            "total_input_tokens": parallel_input_tokens,
            "shared_initial_output_tokens": initial["output_tokens"],
        },
        "sequential": {
            "initial": {
                "extracted_answer": initial["extracted_answer"],
                "output_tokens": initial["output_tokens"],
                "input_tokens": initial["input_tokens"],
                "truncated": initial["truncated"],
                "correct": initial_correct,
                "response": initial["response"],
            },
            "revision": revision,
            "final_answer": revision["extracted_answer"],
            "correct": revised_correct,
            "transition": transition,
            "total_output_tokens": sequential_output_tokens,
            "total_input_tokens": sequential_input_tokens,
            # Shared initial charged when comparing with parallel.
            "shared_initial_output_tokens": initial["output_tokens"],
            "critique_output_tokens": revision["output_tokens"],
            "revision_prompt_input_tokens": revision["input_tokens"],
        },
    }


def summarize_live(results: list[dict[str, Any]], inference_sec: float, peak_mem: Optional[int]) -> dict[str, Any]:
    def rate(key_path: list[str], pred) -> float:
        vals = []
        for r in results:
            obj = r
            for k in key_path:
                obj = obj[k]
            vals.append(pred(obj, r))
        return float(np.mean(vals)) if vals else float("nan")

    def acc(strategy: str) -> float:
        vals = [r[strategy]["correct"] for r in results if r[strategy]["correct"] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_tokens(strategy: str, field: str) -> float:
        return float(np.mean([r[strategy][field] for r in results])) if results else float("nan")

    transitions = [r["sequential"]["transition"] for r in results]
    trans_counts = {k: transitions.count(k) for k in sorted(set(transitions))}

    return {
        "n_questions": len(results),
        "inference_sec_excluding_load": inference_sec,
        "peak_gpu_memory_bytes": peak_mem,
        "peak_gpu_memory_gib": (peak_mem / (1024**3)) if peak_mem else None,
        "scorer": scorer_versions(),
        "single": {
            "accuracy": acc("single"),
            "mean_output_tokens": mean_tokens("single", "total_output_tokens"),
            "mean_input_tokens": mean_tokens("single", "total_input_tokens"),
            "truncation_rate": float(np.mean([r["single"]["truncated"] for r in results])),
            "extraction_rate": float(np.mean([r["single"]["extracted_answer"] is not None for r in results])),
            "mean_elapsed_sec": float(np.mean([r["single"]["elapsed_sec"] for r in results])),
        },
        "parallel": {
            "accuracy": acc("parallel"),
            "mean_output_tokens": mean_tokens("parallel", "total_output_tokens"),
            "mean_input_tokens": mean_tokens("parallel", "total_input_tokens"),
            "truncation_rate": float(
                np.mean([any(p["truncated"] for p in r["parallel"]["responses"]) for r in results])
            ),
            "extraction_rate": float(
                np.mean(
                    [
                        np.mean([p["extracted_answer"] is not None for p in r["parallel"]["responses"]])
                        for r in results
                    ]
                )
            ),
            "mean_elapsed_sec": float(
                np.mean([sum(p["elapsed_sec"] for p in r["parallel"]["responses"]) for r in results])
            ),
        },
        "sequential": {
            "accuracy": acc("sequential"),
            "mean_output_tokens": mean_tokens("sequential", "total_output_tokens"),
            "mean_input_tokens": mean_tokens("sequential", "total_input_tokens"),
            "truncation_rate": float(np.mean([r["sequential"]["revision"]["truncated"] for r in results])),
            "extraction_rate": float(
                np.mean([r["sequential"]["revision"]["extracted_answer"] is not None for r in results])
            ),
            "mean_elapsed_sec": float(
                np.mean(
                    [
                        r["sequential"]["initial"]["output_tokens"] * 0  # placeholder replaced below
                        for r in results
                    ]
                )
            ),
            "transitions": trans_counts,
            "wrong_to_correct": trans_counts.get("wrong_to_correct", 0),
            "correct_to_wrong": trans_counts.get("correct_to_wrong", 0),
        },
        "cost_note": (
            "Shared initial parallel response is charged to both parallel and sequential "
            "when comparing costs; critique tokens and longer revision prompt are counted."
        ),
    }


def run_live_experiment(
    cfg: dict,
    questions: list[dict[str, Any]],
    artifacts_dirs: dict[str, Path],
) -> dict[str, Any]:
    seed = int(cfg["seed"])
    print(f"[live] loading model {cfg['live']['model_id']} ...", flush=True)
    load_t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(cfg)
    load_sec = time.perf_counter() - load_t0
    print(f"[live] model loaded in {load_sec:.1f}s; n_questions={len(questions)}", flush=True)

    _reset_peak_memory()
    if True:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    infer_t0 = time.perf_counter()
    results = []
    for qi, q in enumerate(questions):
        print(f"[live] question {qi+1}/{len(questions)} id={q['sample_id']}", flush=True)
        q_seed = seed + 1000 * (qi + 1)
        result = run_question_strategies(model, tokenizer, q, cfg, base_seed=q_seed)
        results.append(result)
        # Incremental save for crash recovery.
        (artifacts_dirs["live"] / "live_results_partial.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    if True:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    inference_sec = time.perf_counter() - infer_t0
    peak_mem = _peak_gpu_memory_bytes()

    summary = summarize_live(results, inference_sec, peak_mem)
    # Fix sequential mean elapsed properly.
    summary["sequential"]["mean_elapsed_sec"] = float(
        np.mean(
            [
                # initial time is already in parallel[0]; add revision time only for incremental,
                # but report end-to-end sequential wall as initial+revision generation times.
                r["parallel"]["responses"][0]["elapsed_sec"] + r["sequential"]["revision"]["elapsed_sec"]
                for r in results
            ]
        )
    ) if results else float("nan")
    summary["model_load_sec"] = load_sec
    summary["model_id"] = cfg["live"]["model_id"]
    summary["model_revision"] = cfg["live"].get("model_revision")
    summary["generation_settings"] = {
        "temperature": cfg["live"]["temperature"],
        "top_p": cfg["live"]["top_p"],
        "top_k": cfg["live"]["top_k"],
        "repetition_penalty": cfg["live"]["repetition_penalty"],
        "dtype": "float16",
        "batch_size": 1,
    }

    (artifacts_dirs["live"] / "live_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    (artifacts_dirs["live"] / "live_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("[live] done", json.dumps({k: summary[k] for k in ("n_questions", "inference_sec_excluding_load", "single", "parallel", "sequential") if k in summary}, default=str)[:500], flush=True)
    return {"results": results, "summary": summary}
