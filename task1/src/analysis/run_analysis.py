from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.plots import (
    plot_benchmark_bars,
    plot_diagnostics,
    plot_l2sp_penalty,
    plot_sft_losses,
)
from src.analysis.tables import (
    build_benchmark_table,
    build_diagnostics_table,
    build_resource_table,
)
from src.common.config import load_shared_config
from src.common.logging_utils import setup_logger, write_json
from src.common.paths import ensure_result_dirs, find_repo_root, resolve_artifacts_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile tables and figures for Task 1")
    parser.add_argument("--artifacts-root", type=str, default=None)
    args = parser.parse_args()

    shared = load_shared_config(find_repo_root())
    dirs = ensure_result_dirs(resolve_artifacts_root(shared, args.artifacts_root))
    logger = setup_logger("analysis", dirs["logs"] / "analysis.log")

    artifacts = {
        "tables": {},
        "figures": {},
    }
    artifacts["tables"]["benchmark"] = str(build_benchmark_table(dirs["eval"], dirs["tables"]))
    artifacts["tables"]["resources"] = str(build_resource_table(dirs["resources"], dirs["tables"]))
    artifacts["tables"]["weight_space"] = str(
        build_diagnostics_table(dirs["diagnostics"], dirs["tables"])
    )

    artifacts["figures"]["sft_loss"] = str(plot_sft_losses(dirs["logs"], dirs["figures"]))
    artifacts["figures"]["l2sp"] = str(plot_l2sp_penalty(dirs["logs"], dirs["figures"]))
    artifacts["figures"]["benchmark"] = str(plot_benchmark_bars(dirs["tables"], dirs["figures"]))
    artifacts["figures"]["diagnostics"] = [
        str(p) for p in plot_diagnostics(dirs["diagnostics"], dirs["figures"])
    ]

    write_json(dirs["root"] / "analysis_index.json", artifacts)
    logger.info("Analysis artifacts written to %s", dirs["root"])


if __name__ == "__main__":
    main()
