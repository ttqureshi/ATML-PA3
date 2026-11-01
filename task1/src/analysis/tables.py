from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PRIMARY_KEYS = {
    "gsm8k_cot": ["exact_match,flexible-extract", "exact_match"],
    "hellaswag": ["acc_norm,none", "acc_norm", "acc,none", "acc"],
    "boolq": ["acc,none", "acc"],
    "arc_easy": ["acc_norm,none", "acc_norm", "acc,none", "acc"],
}


def _pick_metric(task_metrics: dict[str, Any], candidates: list[str]) -> float | None:
    for c in candidates:
        if c in task_metrics:
            return float(task_metrics[c])
    # fallback: first numeric
    for k, v in task_metrics.items():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def build_benchmark_table(eval_dir: Path, tables_dir: Path) -> Path | None:
    rows = []
    baseline_scores = {}
    conditions = [
        "baseline",
        "full_ft_midpoint",
        "full_ft_final",
        "full_ft_l2sp_final",
        "lora_r4_final",
        "lora_r16_final",
        "oplora_final",
    ]
    for cond in conditions:
        metrics_path = eval_dir / cond / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"condition": cond}
        for task, keys in PRIMARY_KEYS.items():
            task_blob = metrics.get(task, {})
            val = _pick_metric(task_blob, keys) if isinstance(task_blob, dict) else None
            col = {
                "gsm8k_cot": "gsm8k",
                "hellaswag": "hellaswag_acc_norm",
                "boolq": "boolq_acc",
                "arc_easy": "arc_easy_acc_norm",
            }[task]
            row[col] = val
            if cond == "baseline" and val is not None:
                baseline_scores[col] = val
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "benchmark_primary.csv", index=False)

    # percentage-point deltas vs baseline
    delta_rows = []
    for _, r in df.iterrows():
        d = {"condition": r["condition"]}
        for col in baseline_scores:
            if pd.isna(r[col]):
                d[f"delta_{col}"] = None
            else:
                d[f"delta_{col}"] = 100.0 * (float(r[col]) - float(baseline_scores[col]))
        delta_rows.append(d)
    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(tables_dir / "benchmark_delta_pp.csv", index=False)

    # combined markdown for report drafting
    merged = df.merge(df_delta, on="condition")
    md_path = tables_dir / "benchmark_table.md"
    try:
        merged.to_markdown(md_path, index=False)
    except Exception:
        md_path.write_text(merged.to_csv(index=False), encoding="utf-8")
    return tables_dir / "benchmark_primary.csv"


def build_resource_table(resources_dir: Path, tables_dir: Path) -> Path | None:
    rows = []
    for path in sorted(resources_dir.glob("train_summary_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": data.get("variant"),
                "method": data.get("method"),
                "train_seconds": data.get("train_seconds"),
                "peak_mem_gb": data.get("peak_mem_gb"),
                "trainable_params": (data.get("trainable_params") or {}).get("trainable"),
                "adapter_params": (data.get("trainable_params") or {}).get("adapter"),
                "skipped_updates": data.get("skipped_updates"),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "resources.csv"
    df.to_csv(out, index=False)
    try:
        df.to_markdown(tables_dir / "resources.md", index=False)
    except Exception:
        (tables_dir / "resources.md").write_text(df.to_csv(index=False), encoding="utf-8")
    return out


def build_diagnostics_table(diagnostics_dir: Path, tables_dir: Path) -> Path | None:
    weight_path = diagnostics_dir / "weight_space_all.json"
    if not weight_path.exists():
        return None
    data = json.loads(weight_path.read_text(encoding="utf-8"))
    rows = []
    for cond, items in data.items():
        for it in items:
            rows.append({"condition": cond, **it})
    df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "weight_space.csv"
    df.to_csv(out, index=False)
    return out
