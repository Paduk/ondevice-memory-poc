#!/usr/bin/env python3
"""Plot matched runtime and accuracy responses under increasing update load."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_training.scripts.plot_compaction_interval_ablation_two_panel import (
    DELTA_METHODS,
    METHOD_COLORS,
    METHODS,
    MODELS,
    WORKLOAD_LABELS,
    WORKLOADS,
    load_rows,
    load_stress_composites,
)


OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"
OUTPUT_STEM = OUTPUT_DIR / "delta-v3-on-device-pareto-under-update-load"
MARKERS = {"Patch": "X", "k=2": "o", "k=5": "s", "k=10": "D"}


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values)


def save(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUTPUT_STEM.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)


def main() -> None:
    runtime_rows = load_rows()
    stress_composites = load_stress_composites()
    runtime_index = {
        (row["model"], row["method"], row["workload"]): row
        for row in runtime_rows
    }

    latency: dict[tuple[str, str], list[float]] = defaultdict(list)
    accuracy: dict[tuple[str, str], list[float]] = defaultdict(list)
    for model in MODELS:
        for workload in WORKLOADS:
            for method in DELTA_METHODS:
                latency[(method, workload)].append(
                    runtime_index[(model, method, workload)]["relative_latency_percent"]
                )
            for method in METHODS:
                accuracy[(method, workload)].append(
                    stress_composites[(model, method, workload)]
                )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_latency, ax_accuracy) = plt.subplots(1, 2, figsize=(12.4, 4.5))
    x = list(range(len(WORKLOADS)))

    # (a) Runtime is normalized to the matched Patch within each model/load.
    for method in DELTA_METHODS:
        stats = [mean_and_sd(latency[(method, workload)]) for workload in WORKLOADS]
        means = [value for value, _ in stats]
        spreads = [value for _, value in stats]
        lower = [value - spread for value, spread in zip(means, spreads)]
        upper = [value + spread for value, spread in zip(means, spreads)]
        ax_latency.fill_between(
            x,
            lower,
            upper,
            color=METHOD_COLORS[method],
            alpha=0.10,
            linewidth=0,
        )
        ax_latency.plot(
            x,
            means,
            color=METHOD_COLORS[method],
            marker=MARKERS[method],
            markersize=5.6,
            linewidth=2.1,
            zorder=3,
        )
    ax_latency.axhline(100, color=METHOD_COLORS["Patch"], linestyle="--", linewidth=1.8)
    ax_latency.set_ylim(20, 106)
    ax_latency.set_ylabel("Update-cycle latency relative\nto matched Patch (%) ↓")
    ax_latency.set_title("(a) Projected on-device latency")

    # (b) Accuracy is measured on the same workload and scenario subset.
    for method in METHODS:
        stats = [mean_and_sd(accuracy[(method, workload)]) for workload in WORKLOADS]
        means = [value for value, _ in stats]
        spreads = [value for _, value in stats]
        lower = [value - spread for value, spread in zip(means, spreads)]
        upper = [value + spread for value, spread in zip(means, spreads)]
        ax_accuracy.fill_between(
            x,
            lower,
            upper,
            color=METHOD_COLORS[method],
            alpha=0.08,
            linewidth=0,
        )
        ax_accuracy.plot(
            x,
            means,
            color=METHOD_COLORS[method],
            linestyle="--" if method == "Patch" else "-",
            marker=MARKERS[method],
            markersize=5.6,
            linewidth=2.1,
            zorder=3,
        )
    ax_accuracy.set_ylim(0, 80)
    ax_accuracy.set_ylabel("Composite (%) ↑")
    ax_accuracy.set_title("(b) Accuracy under update load")

    for ax in (ax_latency, ax_accuracy):
        ax.set_xticks(x, WORKLOAD_LABELS)
        ax.set_xlabel("Update workload")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle="--" if method == "Patch" else "-",
            marker=MARKERS[method],
            linewidth=2,
            markersize=5.5,
            label=method,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.text(
        0.5,
        0.005,
        "Held-out S86–S90 stress subset; curves and bands denote mean ± SD across six model configurations.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.18, wspace=0.27)
    save(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
