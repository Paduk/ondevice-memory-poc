#!/usr/bin/env python3
"""Render the original absolute-value Section 6.2 figure for six models."""

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
    load_rows,
)
from memory_training.scripts.plot_on_device_pareto_test24_partial import (
    ACCURACY_MODELS,
    ACCURACY_WORKLOAD_LABELS,
    ACCURACY_WORKLOADS,
    MARKERS,
    load_test24_accuracy,
)


OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"
OUTPUT_STEM = OUTPUT_DIR / "section6-2-compaction-interval-ablation-absolute-six-model"


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
    runtime_index = {
        (row["model"], row["method"], row["workload"]): row
        for row in runtime_rows
    }
    test24_accuracy = load_test24_accuracy(runtime_index)

    latency: dict[tuple[str, str], list[float]] = defaultdict(list)
    accuracy: dict[tuple[str, str], list[float]] = defaultdict(list)
    for model in ACCURACY_MODELS:
        for method in DELTA_METHODS:
            for workload in ACCURACY_WORKLOADS:
                latency[(method, workload)].append(
                    runtime_index[(model, method, workload)][
                        "relative_latency_percent"
                    ]
                )
        for method in METHODS:
            for workload in ACCURACY_WORKLOADS:
                accuracy[(method, workload)].append(
                    test24_accuracy[(model, method, workload)]
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
    x = list(range(len(ACCURACY_WORKLOADS)))

    for method in DELTA_METHODS:
        stats = [
            mean_and_sd(latency[(method, workload)])
            for workload in ACCURACY_WORKLOADS
        ]
        means = [value for value, _ in stats]
        spreads = [value for _, value in stats]
        ax_latency.fill_between(
            x,
            [value - spread for value, spread in zip(means, spreads)],
            [value + spread for value, spread in zip(means, spreads)],
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
    ax_latency.axhline(
        100,
        color=METHOD_COLORS["Patch"],
        linestyle="--",
        linewidth=1.8,
    )
    ax_latency.set_xticks(x, ACCURACY_WORKLOAD_LABELS)
    ax_latency.set_ylim(20, 106)
    ax_latency.set_ylabel(
        "Cache-ON update-cycle latency\nrelative to matched Patch (%) ↓"
    )
    ax_latency.set_title("(a) Latency across compaction intervals")

    for method in METHODS:
        stats = [
            mean_and_sd(accuracy[(method, workload)])
            for workload in ACCURACY_WORKLOADS
        ]
        means = [value for value, _ in stats]
        spreads = [value for _, value in stats]
        ax_accuracy.fill_between(
            x,
            [value - spread for value, spread in zip(means, spreads)],
            [value + spread for value, spread in zip(means, spreads)],
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
    ax_accuracy.set_xticks(x, ACCURACY_WORKLOAD_LABELS)
    ax_accuracy.set_ylim(0, 80)
    ax_accuracy.set_ylabel("Composite (%) ↑")
    ax_accuracy.set_title("(b) Accuracy across compaction intervals")

    for axis in (ax_latency, ax_accuracy):
        axis.set_xlabel("Update workload")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9CA3AF")
        axis.spines["bottom"].set_color("#9CA3AF")

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
        "Lines and shaded regions show mean ± SD across six models.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.18, wspace=0.27)
    save(fig)
    plt.close(fig)

    print("Mean relative latency (Base / U40 / U60 / U80):")
    for method in DELTA_METHODS:
        values = [mean(latency[(method, workload)]) for workload in ACCURACY_WORKLOADS]
        print(f"  {method}: " + " / ".join(f"{value:.2f}" for value in values))
    print("Mean Composite (Base / U40 / U60 / U80):")
    for method in METHODS:
        values = [mean(accuracy[(method, workload)]) for workload in ACCURACY_WORKLOADS]
        print(f"  {method}: " + " / ".join(f"{value:.2f}" for value in values))


if __name__ == "__main__":
    main()
