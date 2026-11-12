from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _stage_boundaries(rows: list[dict]) -> list[int]:
    bounds: list[int] = []
    previous_stage = None
    previous_update = None
    for row in rows:
        stage = row.get("stage")
        if previous_stage is not None and stage != previous_stage and previous_update is not None:
            bounds.append(int(previous_update))
        previous_stage = stage
        previous_update = row.get("update")
    return bounds


def _vlines(axis, bounds: list[int]) -> None:
    for index, bound in enumerate(bounds):
        axis.axvline(bound, color="0.4", linestyle="--", linewidth=1, label="stage end" if index == 0 else None)


def plot_learning_curves(train_rows: list[dict], dev_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = _stage_boundaries(train_rows)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0][0]
    if train_rows:
        ax.plot([r["update"] for r in train_rows], [r["loss"] for r in train_rows], color="#1f4b99", linewidth=1.2)
    ax.set_title("Training loss")
    ax.set_xlabel("Update")
    ax.set_ylabel("Example-weighted CE")
    _vlines(ax, bounds)

    ax = axes[0][1]
    if dev_rows:
        ax.plot([r["update"] for r in dev_rows], [r["caption_loss"] for r in dev_rows], marker="o", color="#0b6e4f")
    ax.set_title("Dev caption reference loss")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    _vlines(ax, bounds)

    ax = axes[1][0]
    colors = {"correct": "#1f4b99", "complementary": "#b45309", "no_image": "#6b7280"}
    for suite, style in (("grounding", "-"), ("natural", "--")):
        for condition, color in colors.items():
            xs, ys = [], []
            for row in dev_rows:
                suite_row = row.get("suites", {}).get(suite)
                if not suite_row:
                    continue
                xs.append(row["update"])
                ys.append(suite_row["nll"][condition])
            if xs:
                ax.plot(xs, ys, style, color=color, marker="o", label=f"{suite[:1]}/{condition}")
    ax.set_title("Dev answer NLL")
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean token NLL")
    ax.legend(fontsize=7, ncol=2)
    _vlines(ax, bounds)

    ax = axes[1][1]
    for suite, color in (("grounding", "#1f4b99"), ("natural", "#9f1239")):
        xs, ys = [], []
        for row in dev_rows:
            suite_row = row.get("suites", {}).get(suite)
            if not suite_row:
                continue
            xs.append(row["update"])
            ys.append(suite_row["delta_nll"])
        if xs:
            ax.plot(xs, ys, marker="o", color=color, label=suite)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_title("Dev ΔNLL (partner − correct)")
    ax.set_xlabel("Update")
    ax.set_ylabel("ΔNLL")
    ax.legend(fontsize=8)
    _vlines(ax, bounds)

    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def plot_final_accuracy(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suites = list(summary.get("suites", {}))
    if not suites:
        return
    conditions = ["correct", "complementary", "mismatched", "no_image"]
    figure, axes = plt.subplots(1, len(suites), figsize=(5.2 * len(suites), 4.2), squeeze=False)
    for axis, suite in zip(axes[0], suites):
        overall = summary["suites"][suite]["overall"]
        xs = list(range(len(conditions)))
        heights, yerr_lo, yerr_hi = [], [], []
        for condition in conditions:
            stat = overall.get(f"accuracy/{condition}")
            if not stat:
                heights.append(0.0)
                yerr_lo.append(0.0)
                yerr_hi.append(0.0)
                continue
            point = float(stat["point"])
            lo = min(float(stat["lo"]), point)
            hi = max(float(stat["hi"]), point)
            heights.append(point)
            yerr_lo.append(point - lo)
            yerr_hi.append(hi - point)
        axis.bar(xs, heights, color=["#1f4b99", "#b45309", "#7c3aed", "#6b7280"], yerr=[yerr_lo, yerr_hi], capsize=3)
        axis.set_xticks(xs, conditions, rotation=20)
        axis.set_ylim(0, 1)
        axis.set_title(suite)
        axis.set_ylabel("Primary accuracy")
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
