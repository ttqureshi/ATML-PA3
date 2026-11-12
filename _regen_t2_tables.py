"""Regenerate Task2 tables from local logs + summary.json if present."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"g:\My Drive\MS AI\Semester_3\ATML\PAs\ATML-PA3\task2")
sys.path.insert(0, str(ROOT))

from src.analysis.tables import write_dev_table, write_final_tables, write_resources_table
from src.common.logging_utils import read_json, read_jsonl


ART = ROOT / "results_pipeline_kaggle" / "results"


def main() -> None:
    tables = ART / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    logs = ART / "logs"
    resources_dir = ART / "resources"
    eval_final = ART / "raw" / "eval" / "final"

    # Dev table from local logs (always available).
    dev_path = logs / "dev_metrics.jsonl"
    if dev_path.exists():
        write_dev_table(read_jsonl(dev_path), tables)
        print("wrote dev_metrics", flush=True)
    else:
        print("MISSING", dev_path, flush=True)

    # Resources table.
    resources = {}
    for name in ("train_resources.json", "generation.json", "feature_cache.json"):
        p = resources_dir / name
        if p.exists():
            payload = read_json(p)
            if name == "generation.json":
                resources["generation"] = payload
            elif name == "feature_cache.json":
                resources["feature_cache"] = payload
            else:
                resources.update(payload)
    status = (ART / "STATUS.md").read_text(encoding="utf-8") if (ART / "STATUS.md").exists() else ""
    if not resources and status:
        # Parse STATUS resource lines as fallback.
        import re

        def grab(label: str):
            m = re.search(rf"- {re.escape(label)}: (.+)", status)
            return m.group(1).strip() if m else ""

        resources = {
            "train_loop_sec": grab("Train loop seconds"),
            "peak_gpu_mem_mb": grab("Train peak GPU MB"),
            "feature_cache": {"peak_gpu_mem_mb": grab("Feature-cache peak GPU MB")},
            "generation": {"mean_gen_time_sec": grab("Mean generation seconds per answer")},
        }
    if resources:
        write_resources_table(resources, tables)
        print("wrote resources", flush=True)

    # Final tables from summary.json / summary_recomputed.json.
    summary = None
    for name in ("summary_recomputed.json", "summary.json"):
        p = eval_final / name
        if p.exists():
            summary = read_json(p)
            print("loaded", p, flush=True)
            break
    if summary is None:
        # Minimal STATUS-derived table so the report isn't empty.
        import re

        rows_md = ["# Final test results", "", "Reconstructed from STATUS.md (full per-condition table pending summary.json).", ""]
        rows_md.append("| suite | slice | condition | accuracy |")
        rows_md.append("| --- | --- | --- | --- |")
        for m in re.finditer(
            r"- (\w+): ([0-9.]+) \(95% CI ([0-9.]+) to ([0-9.]+)\); advantage vs complementary ([0-9.]+)",
            status,
        ):
            suite, point, lo, hi, adv = m.groups()
            rows_md.append(f"| {suite} | overall | correct | {float(point):.4f} [{float(lo):.4f}, {float(hi):.4f}] |")
            rows_md.append(f"| {suite} | overall | advantage | {float(adv):.4f} | |")
        (tables / "final_results.md").write_text("\n".join(rows_md) + "\n", encoding="utf-8")
        print("wrote STATUS-derived final_results.md", flush=True)
        return

    text = write_final_tables(summary, tables)
    print("wrote final_results from summary", len(text), "chars", flush=True)


if __name__ == "__main__":
    main()
