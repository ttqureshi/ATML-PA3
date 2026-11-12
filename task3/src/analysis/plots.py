from __future__ import annotations

from pathlib import Path
from typing import Any


def save_accuracy_curves(summary: dict[str, Any], out_path: Path) -> None:
    """Save accuracy / oracle curves vs budget N."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        out_path.write_text("matplotlib unavailable\n", encoding="utf-8")
        return

    budgets = sorted(int(b) for b in summary["budgets"].keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(budgets, [summary["budgets"][b]["oracle_coverage"] for b in budgets], "k--", label="Oracle")
    method_names = list(summary["budgets"][budgets[0]]["methods"].keys())
    for name in method_names:
        ys = [summary["budgets"][b]["methods"][name]["accuracy"] for b in budgets]
        ax.plot(budgets, ys, marker="o", label=name)
    ax.set_xlabel("Candidate budget N")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{summary.get('dataset', '')} ({summary.get('split', '')})")
    ax.legend(fontsize=7, loc="best")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
