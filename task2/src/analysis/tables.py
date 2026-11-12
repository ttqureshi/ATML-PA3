from __future__ import annotations

import csv
from pathlib import Path


def _fmt(stat: dict | None, digits: int = 4) -> str:
    if not stat:
        return ""
    return f"{float(stat['point']):.{digits}f} [{float(stat['lo']):.{digits}f}, {float(stat['hi']):.{digits}f}]"


def _write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _markdown(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def write_final_tables(summary: dict, tables_dir: Path) -> str:
    headers = [
        "suite",
        "slice",
        "condition",
        "accuracy",
        "pair_accuracy",
        "identical_answer_rate",
        "empty_answer_rate",
        "mean_nll",
        "gen_time_sec",
    ]
    rows: list[list] = []
    advantage_rows: list[list] = []
    conditions = ("correct", "complementary", "mismatched", "no_image")
    for suite, payload in summary.get("suites", {}).items():
        blocks = [("overall", payload["overall"])]
        blocks.extend(payload.get("by_subtype", {}).items())
        for slice_name, metrics in blocks:
            for condition in conditions:
                if f"accuracy/{condition}" not in metrics:
                    continue
                rows.append(
                    [
                        suite,
                        slice_name,
                        condition,
                        _fmt(metrics.get(f"accuracy/{condition}")),
                        _fmt(metrics.get(f"pair_accuracy/{condition}")),
                        _fmt(metrics.get(f"identical_answer_rate/{condition}")),
                        _fmt(metrics.get(f"empty_answer_rate/{condition}")),
                        _fmt(metrics.get(f"nll/{condition}")),
                        _fmt(metrics.get(f"gen_time_sec/{condition}")),
                    ]
                )
            if "advantage/accuracy" in metrics:
                advantage_rows.append(
                    [
                        suite,
                        slice_name,
                        _fmt(metrics.get("advantage/accuracy")),
                        _fmt(metrics.get("advantage/nll")),
                        _fmt(metrics.get("follow_partner_rate")),
                    ]
                )
    _write_csv(tables_dir / "final_results.csv", headers, rows)
    _write_csv(
        tables_dir / "final_advantages.csv",
        ["suite", "slice", "accuracy_advantage", "nll_advantage", "follow_partner_rate"],
        advantage_rows,
    )
    body = [
        "# Final test results",
        "",
        "Primary accuracy is normalized exact match for grounding and VQA soft consensus for natural questions.",
        "Pair accuracy is exact match to the canonical answer on both images. Empty outputs stay in every denominator.",
        "Brackets are 95% percentile intervals from bootstrap resamples of complete pairs.",
        "Accuracy advantage is correct minus complementary. NLL advantage is complementary minus correct.",
        "",
        _markdown(headers, rows),
        "",
        "## Correct-image advantages",
        "",
        _markdown(
            ["suite", "slice", "accuracy_advantage", "nll_advantage", "follow_partner_rate"],
            advantage_rows,
        ),
    ]
    text = "\n".join(body)
    (tables_dir / "final_results.md").write_text(text, encoding="utf-8")
    return text


def write_dev_table(dev_rows: list[dict], tables_dir: Path) -> None:
    headers = [
        "update",
        "stage",
        "caption_loss",
        "ground_nll_correct",
        "ground_nll_comp",
        "ground_nll_none",
        "ground_delta_nll",
        "natural_nll_correct",
        "natural_nll_comp",
        "natural_nll_none",
        "natural_delta_nll",
    ]
    rows = []
    for row in dev_rows:
        ground = row["suites"]["grounding"]
        natural = row["suites"]["natural"]
        rows.append(
            [
                row["update"],
                row["stage"],
                f"{row['caption_loss']:.6f}",
                f"{ground['nll']['correct']:.6f}",
                f"{ground['nll']['complementary']:.6f}",
                f"{ground['nll']['no_image']:.6f}",
                f"{ground['delta_nll']:.6f}",
                f"{natural['nll']['correct']:.6f}",
                f"{natural['nll']['complementary']:.6f}",
                f"{natural['nll']['no_image']:.6f}",
                f"{natural['delta_nll']:.6f}",
            ]
        )
    _write_csv(tables_dir / "dev_metrics.csv", headers, rows)
    (tables_dir / "dev_metrics.md").write_text(_markdown(headers, rows), encoding="utf-8")


def write_resources_table(resources: dict, tables_dir: Path) -> None:
    rows = []
    cache = resources.get("feature_cache") or {}
    rows.append(["feature_cache_sec", cache.get("wall_time_sec", "")])
    rows.append(["feature_cache_peak_gpu_mb", cache.get("peak_gpu_mem_mb", "")])
    rows.append(["train_loop_sec", resources.get("train_loop_sec", "")])
    rows.append(["train_peak_gpu_mb", resources.get("peak_gpu_mem_mb", "")])
    rows.append(["model_load_sec", resources.get("model_load_sec", "")])
    rows.append(["examples_seen", resources.get("examples_seen", "")])
    rows.append(["updates", resources.get("updates", "")])
    for name, report in (resources.get("trainable_by_stage") or {}).items():
        rows.append([f"{name}_trainable", report.get("trainable_total", "")])
        rows.append([f"{name}_connector", report.get("connector", "")])
        rows.append([f"{name}_lora", report.get("lora", "")])
    gen = resources.get("generation") or {}
    rows.append(["mean_gen_time_sec", gen.get("mean_gen_time_sec", "")])
    text = _markdown(["item", "value"], rows)
    (tables_dir / "resources.md").write_text(text, encoding="utf-8")
    _write_csv(tables_dir / "resources.csv", ["item", "value"], rows)
