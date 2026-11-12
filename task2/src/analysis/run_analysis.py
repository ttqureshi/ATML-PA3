from __future__ import annotations

from src.common.config import parse_common_args, prepare_runtime
from src.common.logging_utils import read_json, read_jsonl, write_json
from src.eval.scoring import summarize_prediction_rows
from src.analysis.plots import plot_final_accuracy, plot_learning_curves
from src.analysis.qualitative import build_qualitative, save_contact_sheet
from src.analysis.tables import write_dev_table, write_final_tables, write_resources_table


def run_analysis(cfg: dict, dirs: dict) -> None:
    train_rows = read_jsonl(dirs["logs"] / "train_metrics.jsonl")
    dev_rows = read_jsonl(dirs["logs"] / "dev_metrics.jsonl")
    per_example_path = dirs["eval"] / "final" / "per_example.jsonl"
    if not per_example_path.exists():
        raise FileNotFoundError(f"Final per-example metrics are missing: {per_example_path}")
    if not train_rows or not dev_rows:
        raise FileNotFoundError("train_metrics.jsonl and dev_metrics.jsonl are required")
    final_rows = read_jsonl(per_example_path)
    summary = summarize_prediction_rows(
        final_rows,
        int(cfg["evaluation"]["bootstrap_resamples"]),
        int(cfg["evaluation"]["bootstrap_seed"]),
    )
    write_json(dirs["eval"] / "final" / "summary_recomputed.json", summary)
    plot_learning_curves(train_rows, dev_rows, dirs["figures"] / "learning_curves.png")
    plot_final_accuracy(summary, dirs["figures"] / "final_scores.png")
    write_dev_table(dev_rows, dirs["tables"])
    write_final_tables(summary, dirs["tables"])
    resources = {}
    resource_path = dirs["resources"] / "train_resources.json"
    if resource_path.exists():
        resources.update(read_json(resource_path))
    gen_path = dirs["resources"] / "generation.json"
    if gen_path.exists():
        resources["generation"] = read_json(gen_path)
    cache_path = dirs["resources"] / "feature_cache.json"
    if cache_path.exists() and "feature_cache" not in resources:
        resources["feature_cache"] = read_json(cache_path)
    write_resources_table(resources, dirs["tables"])
    qualitative = build_qualitative(final_rows, dirs["images"])
    write_json(dirs["qualitative"] / "complementary_examples.json", qualitative)
    save_contact_sheet(qualitative, dirs["images"], dirs["figures"] / "qualitative_pairs.png")
    status = _status_text(cfg, summary, resources, len(qualitative["examples"]))
    (dirs["root"] / "STATUS.md").write_text(status, encoding="utf-8")
    print("ANALYSIS wrote tables, figures, qualitative examples, and STATUS.md", flush=True)


def _status_text(cfg: dict, summary: dict, resources: dict, n_qual: int) -> str:
    lines = [
        "# Task 2 run summary",
        "",
        f"Smoke run: {bool(cfg.get('smoke_active'))}.",
        f"Visual tokens: {cfg['model']['num_visual_tokens']}.",
        f"Bootstrap resamples: {summary.get('bootstrap_resamples')}.",
        f"Qualitative pairs saved: {n_qual}.",
        "",
        "## Correct-image primary accuracy",
        "",
    ]
    for suite, payload in summary.get("suites", {}).items():
        stat = payload["overall"].get("accuracy/correct", {})
        adv = payload["overall"].get("advantage/accuracy", {})
        lines.append(
            f"- {suite}: {stat.get('point')} (95% CI {stat.get('lo')} to {stat.get('hi')}); "
            f"advantage vs complementary {adv.get('point')}"
        )
    lines.extend(
        [
            "",
            "## Resources",
            "",
            f"- Train loop seconds: {resources.get('train_loop_sec')}",
            f"- Train peak GPU MB: {resources.get('peak_gpu_mem_mb')}",
            f"- Feature-cache peak GPU MB: {(resources.get('feature_cache') or {}).get('peak_gpu_mem_mb')}",
            f"- Mean generation seconds per answer: {(resources.get('generation') or {}).get('mean_gen_time_sec')}",
            "",
            "Final numbers are in `tables/final_results.md`. Learning curves are in `figures/learning_curves.png`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = parse_common_args("Build Task 2 tables, figures, and qualitative examples.")
    args = parser.parse_args()
    _root, cfg, dirs = prepare_runtime(args)
    run_analysis(cfg, dirs)


if __name__ == "__main__":
    main()
