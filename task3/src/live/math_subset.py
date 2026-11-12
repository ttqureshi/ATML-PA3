from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.splits import make_question_splits, save_splits


def _problem_text(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt", "input"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"No problem text in row keys={list(row.keys())}")


def _gold_text(row: dict[str, Any]) -> str:
    for key in ("solution", "answer", "gold", "target"):
        if key in row and row[key] is not None:
            text = str(row[key])
            # MATH solutions often contain \boxed{}; keep full solution as gold source.
            return text
    return ""


def _sample_id_for_row(row: dict[str, Any], index: int) -> str:
    for key in ("unique_id", "id", "problem_id", "sample_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    level = str(row.get("level", "unknown"))
    typ = str(row.get("type", row.get("subject", "unknown")))
    return f"{level}/{typ}/{index:05d}"


def load_hf_math_test(dataset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    rows = []
    for i, ex in enumerate(ds):
        row = dict(ex)
        sid = _sample_id_for_row(row, i)
        rows.append(
            {
                "sample_id": sid,
                "problem": _problem_text(row),
                "gold_raw": _gold_text(row),
                "level": row.get("level"),
                "type": row.get("type", row.get("subject")),
                "source_index": i,
            }
        )
    return rows


def extract_gold_answer(gold_raw: str) -> str | None:
    from src.data.boxed import extract_last_boxed

    boxed = extract_last_boxed(gold_raw)
    if boxed is not None:
        return boxed
    # Some MATH exports put the final answer on the last line.
    lines = [ln.strip() for ln in str(gold_raw).strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def load_live_math_questions(
    cfg: dict,
    artifacts_dirs: dict[str, Path] | None = None,
    supplied_live_ids: list[str] | None = None,
    supplied_problems: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Build the 50-question live MATH subset.

    Prefer official supplied IDs/problems when available. Otherwise:
    load HF MATH test, take a 1000-question subset (first 1000 after lexical sort of IDs),
    permute with seed 6304, use first 50 of the 800 eval questions.
    """
    seed = int(cfg["seed"])
    split_cfg = cfg["splits"]
    live_cfg = cfg["live"]
    n_override = cfg.get("live_n_questions_override")

    if supplied_live_ids is not None and supplied_problems is not None:
        live_ids = list(supplied_live_ids)
        if n_override is not None:
            live_ids = live_ids[: int(n_override)]
        questions = []
        for sid in live_ids:
            item = supplied_problems[sid]
            questions.append(
                {
                    "sample_id": sid,
                    "problem": item["problem"],
                    "gold_answer": item.get("gold_answer"),
                    "source": "supplied",
                }
            )
        payload = {
            "source": "supplied",
            "live_ids": live_ids,
            "questions": questions,
            "note": "Using official ATML-PA3-Data live IDs",
        }
    else:
        rows = load_hf_math_test(live_cfg["math_hf_dataset"], live_cfg["math_hf_split"])
        all_ids = [r["sample_id"] for r in rows]
        # Form 1000-question subset: first 1000 after lexical sort.
        sorted_ids = sorted(all_ids)
        subset_ids = sorted_ids[: int(split_cfg["n_questions"])]
        by_id = {r["sample_id"]: r for r in rows}
        splits = make_question_splits(
            subset_ids,
            seed=seed,
            n_dev=int(split_cfg["n_dev"]),
            n_eval=int(split_cfg["n_eval"]),
            n_live=int(split_cfg["n_live"]),
        )
        live_ids = list(splits["live_ids"])
        if n_override is not None:
            live_ids = live_ids[: int(n_override)]
        questions = []
        for sid in live_ids:
            r = by_id[sid]
            questions.append(
                {
                    "sample_id": sid,
                    "problem": r["problem"],
                    "gold_answer": extract_gold_answer(r["gold_raw"]),
                    "gold_raw": r["gold_raw"],
                    "level": r.get("level"),
                    "type": r.get("type"),
                    "source": "huggingface_fallback",
                }
            )
        payload = {
            "source": "huggingface_fallback",
            "dataset": live_cfg["math_hf_dataset"],
            "split": live_cfg["math_hf_split"],
            "splits": splits,
            "live_ids": live_ids,
            "questions": questions,
            "note": (
                "Official ATML-PA3-Data not found. Live IDs come from a 1000-question "
                "HF MATH test subset (first 1000 after lexical sort), split with seed 6304. "
                "IDs may need re-alignment once official data arrives."
            ),
        }
        if artifacts_dirs is not None:
            save_splits(splits, artifacts_dirs["splits"] / "math_hf_fallback_splits.json")

    if artifacts_dirs is not None:
        out = artifacts_dirs["live"] / "live_questions.json"
        out.write_text(json.dumps({k: v for k, v in payload.items() if k != "questions"}, indent=2) + "\n", encoding="utf-8")
        # Also save questions (may be large but 50 is fine).
        (artifacts_dirs["live"] / "live_questions_full.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return payload
