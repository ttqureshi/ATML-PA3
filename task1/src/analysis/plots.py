from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def plot_sft_losses(logs_dir: Path, figures_dir: Path) -> Path | None:
    variants = ["full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for v in variants:
        rows = _read_jsonl(logs_dir / f"train_metrics_{v}.jsonl")
        if not rows:
            continue
        any_data = True
        steps = [r["step"] for r in rows]
        losses = [r["sft_loss"] for r in rows]
        ax.plot(steps, losses, label=v, linewidth=1.5)
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("SFT loss")
    ax.set_title("Task 1: SFT loss vs update")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "sft_loss_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(figures_dir / "sft_loss_curves.pdf")
    plt.close(fig)
    return out


def plot_l2sp_penalty(logs_dir: Path, figures_dir: Path) -> Path | None:
    rows = _read_jsonl(logs_dir / "train_metrics_full_ft_l2sp.jsonl")
    if not rows or "omega" not in rows[0]:
        return None
    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, [r["omega"] for r in rows], label="Ω")
    ax.plot(steps, [r["lambda_omega"] for r in rows], label="λΩ")
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Penalty")
    ax.set_title("L2-SP penalty during training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = figures_dir / "l2sp_penalty.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_benchmark_bars(tables_dir: Path, figures_dir: Path) -> Path | None:
    path = tables_dir / "benchmark_primary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    metrics = [c for c in df.columns if c not in {"condition"}]
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        ax.bar(df["condition"], df[m])
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Task 1 benchmark scores")
    fig.tight_layout()
    out = figures_dir / "benchmark_scores.png"
    fig.savefig(out, dpi=200)
    fig.savefig(figures_dir / "benchmark_scores.pdf")
    plt.close(fig)
    return out


def plot_diagnostics(diagnostics_dir: Path, figures_dir: Path) -> list[Path]:
    outs = []
    weight_path = diagnostics_dir / "weight_space_all.json"
    if weight_path.exists():
        data = json.loads(weight_path.read_text(encoding="utf-8"))
        rows = []
        for cond, items in data.items():
            for it in items:
                rows.append({"condition": cond, **it})
        if rows:
            df = pd.DataFrame(rows)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for layer, g in df.groupby("layer"):
                axes[0].plot(g["condition"], g["rho_k"], marker="o", label=f"L{layer}")
                axes[1].plot(g["condition"], g["eta"], marker="o", label=f"L{layer}")
            axes[0].set_title(r"Subspace interference $\rho_k$")
            axes[1].set_title(r"Relative update $\eta$")
            for ax in axes:
                ax.tick_params(axis="x", rotation=75, labelsize=7)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
            fig.tight_layout()
            out = figures_dir / "weight_space_diagnostics.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            outs.append(out)

    act_path = diagnostics_dir / "activations_summary_all.json"
    if act_path.exists():
        data = json.loads(act_path.read_text(encoding="utf-8"))
        rows = []
        for cond, items in data.items():
            for it in items:
                rows.append({"condition": cond, **it})
        if rows:
            df = pd.DataFrame(rows)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for (layer, domain), g in df.groupby(["layer", "domain"]):
                ax.plot(g["condition"], g["R_mean"], marker="o", label=f"L{layer}-{domain}")
            ax.set_ylabel("mean R(z)")
            ax.set_title("Activation correction ratios")
            ax.tick_params(axis="x", rotation=75, labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = figures_dir / "activation_diagnostics.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            outs.append(out)
    return outs
