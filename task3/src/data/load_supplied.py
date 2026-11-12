from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.answer_equiv import grade_against_gold, is_missing_extraction
from src.data.splits import make_question_splits, make_rollout_orders, save_splits


JOIN_KEYS = ["dataset", "sample_id", "rollout_id"]

# Official ATML-PA3-Data column names → names used by scoring / selection code.
COLUMN_ALIASES = {
    "img_attn_total": "attn_image",
    "prompt_text_attn_total": "attn_prompt",
    "generated_text_attn_total": "attn_generated",
    "n_generated_text_tokens": "num_generated_tokens",
    "generated_token_ids": "token_ids",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for src, dst in COLUMN_ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def discover_model_dirs(data_root: Path, cfg: dict) -> dict[str, Path]:
    """Locate MATH (LLM) and MathVista (VLM) folders under ATML-PA3-Data."""
    paths = cfg.get("paths", {})
    candidates = {
        "math": [
            data_root / paths.get("math_model_folder", "Qwen2.5-3B-Instruct"),
            data_root / "qwen25-math",
            data_root / "math",
            data_root / "LLM",
            data_root / "Qwen2.5-3B-Instruct",
        ],
        "mathvista": [
            data_root / paths.get("mathvista_model_folder", "Qwen2.5-VL-3B-Instruct"),
            data_root / "qwen25vl-mathvista",
            data_root / "mathvista",
            data_root / "VLM",
            data_root / "Qwen2.5-VL-3B-Instruct",
        ],
    }
    found: dict[str, Path] = {}
    for name, opts in candidates.items():
        for opt in opts:
            if (opt / "generations.parquet").exists():
                found[name] = opt
                break
    return found


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _as_list(val: Any) -> list:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    if isinstance(val, str):
        text = val.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
        return [val]
    return [val]


def verify_token_alignment(row: pd.Series) -> dict[str, Any]:
    """Check that token-level lists align with generated token length."""
    token_ids = _as_list(row.get("token_ids", row.get("generated_token_ids")))
    logp = _as_list(row.get("logp"))
    kl = _as_list(row.get("kl_u_p"))
    length = row.get(
        "num_generated_tokens",
        row.get("n_generated_text_tokens", row.get("gen_length", row.get("length"))),
    )
    try:
        length_i = int(length) if length is not None and not (isinstance(length, float) and np.isnan(length)) else None
    except Exception:
        length_i = None
    n_ids = len(token_ids)
    n_logp = len(logp)
    n_kl = len(kl)
    ok = True
    notes = []
    if n_logp and n_kl and n_logp != n_kl:
        ok = False
        notes.append(f"logp({n_logp})!=kl_u_p({n_kl})")
    if length_i is not None and n_logp and length_i != n_logp:
        # Termination tokens excluded from measurements — length may be n_logp or n_logp+1.
        if abs(length_i - n_logp) > 1:
            ok = False
            notes.append(f"length({length_i})!~logp({n_logp})")
    if n_ids and n_logp and abs(n_ids - n_logp) > 1:
        ok = False
        notes.append(f"token_ids({n_ids})!~logp({n_logp})")
    return {
        "ok": ok,
        "n_token_ids": n_ids,
        "n_logp": n_logp,
        "n_kl": n_kl,
        "length": length_i,
        "notes": notes,
    }


def load_model_tables(model_dir: Path, dataset_name: str) -> pd.DataFrame:
    gens = _read_parquet(model_dir / "generations.parquet")
    probs = _read_parquet(model_dir / "probability_metrics.parquet")
    for key in JOIN_KEYS:
        if key not in gens.columns and key == "dataset":
            gens["dataset"] = dataset_name
        if key not in probs.columns and key == "dataset":
            probs["dataset"] = dataset_name
    df = gens.merge(probs, on=JOIN_KEYS, how="inner", suffixes=("", "_prob"))
    attn_path = model_dir / "attention_metrics.parquet"
    if attn_path.exists():
        attn = _read_parquet(attn_path)
        if "dataset" not in attn.columns:
            attn["dataset"] = dataset_name
        df = df.merge(attn, on=JOIN_KEYS, how="left", suffixes=("", "_attn"))
    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    return _normalize_columns(df)


def attach_correctness(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    extracted_col = _first_present(out, ["extracted_answer", "pred_answer", "answer_extracted"])
    gold_col = _first_present(out, ["gold_answer", "answer", "reference", "gt_answer"])
    meta_cols = [c for c in out.columns if c in {"answer_type", "precision", "choices", "options", "question_type", "metadata"}]

    correct = []
    for _, row in out.iterrows():
        meta = {c: row[c] for c in meta_cols if c in row and pd.notna(row[c])}
        if "metadata" in meta and isinstance(meta["metadata"], str):
            try:
                meta.update(json.loads(meta["metadata"]))
            except Exception:
                pass
        ans = row[extracted_col] if extracted_col else None
        gold = row[gold_col] if gold_col else None
        correct.append(bool(grade_against_gold(ans, gold, dataset, meta)))
    out["is_correct"] = correct
    out["extracted_answer_used"] = out[extracted_col] if extracted_col else None
    out["gold_answer_used"] = out[gold_col] if gold_col else None
    out["is_missing_extraction"] = [
        is_missing_extraction(a) for a in (out["extracted_answer_used"].tolist() if extracted_col else [None] * len(out))
    ]
    return out


def _first_present(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def load_all_supplied(data_root: Path, cfg: dict, artifacts_dirs: dict[str, Path] | None = None) -> dict[str, Any]:
    if not data_root.exists():
        return {
            "available": False,
            "data_root": str(data_root),
            "reason": "ATML-PA3-Data not found",
        }
    model_dirs = discover_model_dirs(data_root, cfg)
    if not model_dirs:
        return {
            "available": False,
            "data_root": str(data_root),
            "reason": "No generations.parquet found under data root",
        }

    seed = int(cfg["seed"])
    split_cfg = cfg["splits"]
    payloads: dict[str, Any] = {"available": True, "data_root": str(data_root), "datasets": {}}

    for ds_name, model_dir in model_dirs.items():
        df = load_model_tables(model_dir, dataset_name=ds_name)
        df = attach_correctness(df, dataset=ds_name)
        # Alignment audit on a sample of rows.
        align_ok = 0
        align_bad = 0
        notes: list[str] = []
        sample = df.head(min(64, len(df)))
        for _, row in sample.iterrows():
            info = verify_token_alignment(row)
            if info["ok"]:
                align_ok += 1
            else:
                align_bad += 1
                notes.extend(info["notes"])
        sample_ids = sorted(df["sample_id"].astype(str).unique().tolist())
        splits = make_question_splits(
            sample_ids,
            seed=seed,
            n_dev=int(split_cfg["n_dev"]),
            n_eval=int(split_cfg["n_eval"]),
            n_live=int(split_cfg["n_live"]),
        )
        orders = make_rollout_orders(
            n_rollouts=int(split_cfg["n_rollouts"]),
            n_orders=int(split_cfg["n_orders"]),
            seed=seed,
            sample_ids=splits["permuted_ids"],
        )
        if artifacts_dirs is not None:
            save_splits(splits, artifacts_dirs["splits"] / f"{ds_name}_splits.json")
            (artifacts_dirs["splits"] / f"{ds_name}_orders.json").write_text(
                json.dumps({k: v for k, v in list(orders.items())[:5]}, indent=2),
                encoding="utf-8",
            )
            # Full orders are large; store compressed via parquet sidecar of question->order index.
            order_rows = []
            for sid, perms in orders.items():
                for oi, perm in enumerate(perms):
                    order_rows.append({"sample_id": sid, "order_id": oi, "rollout_order": perm})
            pd.DataFrame(order_rows).to_parquet(artifacts_dirs["splits"] / f"{ds_name}_orders.parquet", index=False)

        payloads["datasets"][ds_name] = {
            "model_dir": str(model_dir),
            "frame": df,
            "splits": splits,
            "orders": orders,
            "alignment": {"ok": align_ok, "bad": align_bad, "notes": notes[:20]},
            "n_rows": int(len(df)),
            "n_questions": int(df["sample_id"].nunique()),
        }
    return payloads


def token_count_for_cost(row: pd.Series) -> int:
    for key in (
        "num_generated_tokens",
        "n_generated_text_tokens",
        "gen_length",
        "n_tokens",
        "length",
    ):
        if key in row.index and pd.notna(row[key]):
            try:
                return int(row[key])
            except Exception:
                pass
    logp = row.get("logp")
    if isinstance(logp, (list, tuple, np.ndarray)):
        return int(len(logp))
    return 0
