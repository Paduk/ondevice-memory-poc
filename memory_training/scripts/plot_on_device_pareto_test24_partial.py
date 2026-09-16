#!/usr/bin/env python3
"""Plot the interim Section 6.2 compaction-interval ablation.

Both panels use the same currently completed model set and report paired
changes from the matched Patch result. Panel (a) reports cache-ON latency
reduction. Panel (b) reports the Composite change on the same 24 scenarios.
"""

from __future__ import annotations

import json
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
    METHOD_SLUGS,
    METHODS,
    composite,
    load_rows,
)


BENCHMARK_ROOT = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training/benchmarks")
OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"
OUTPUT_STEM = OUTPUT_DIR / "section6-2-compaction-interval-ablation-paired-six-model"
MARKERS = {"Patch": "X", "k=2": "o", "k=5": "s", "k=10": "D"}

ACCURACY_MODELS = (
    "Granite 350M",
    "Qwen 0.8B",
    "Llama 1B",
    "Granite 1B",
    "Qwen 2B",
    "Llama 3B",
)
MODEL_SLUGS = {
    "Granite 350M": "granite4-350m",
    "Qwen 0.8B": "qwen3.5-0.8b",
    "Llama 1B": "llama3.2-1b",
    "Granite 1B": "granite4-1b",
    "Qwen 2B": "qwen3.5-2b",
    "Llama 3B": "llama3.2-3b",
}
ACCURACY_WORKLOADS = ("base", "40", "60", "80")
ACCURACY_WORKLOAD_LABELS = ("Base", "U40", "U60", "U80")


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values)


def load_test24_accuracy(runtime_index: dict) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    for model in ACCURACY_MODELS:
        for method in METHODS:
            values[(model, method, "base")] = runtime_index[
                (model, method, "base")
            ]["test_composite"]
            run = (
                f"{MODEL_SLUGS[model]}-{METHOD_SLUGS[method]}"
                "-test24-update-stress-20260915-v1"
            )
            for workload in ACCURACY_WORKLOADS[1:]:
                path = (
                    BENCHMARK_ROOT
                    / run
                    / "composite"
                    / f"update{workload}"
                    / "summary.json"
                )
                summary = json.loads(path.read_text())
                if not summary.get("complete") or len(summary.get("completed_scenarios", [])) != 24:
                    raise ValueError(f"Incomplete Test24 result: {path}")
                values[(model, method, workload)] = composite(summary)
    return values


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

    latency_saving: dict[tuple[str, str], list[float]] = defaultdict(list)
    composite_change: dict[tuple[str, str], list[float]] = defaultdict(list)
    # Use the same completed models and workloads in both panels so that the
    # latency and accuracy curves describe matched operating points.
    for model in ACCURACY_MODELS:
        for method in DELTA_METHODS:
            for workload in ACCURACY_WORKLOADS:
                latency_saving[(method, workload)].append(
                    100.0
                    - runtime_index[(model, method, workload)][
                        "relative_latency_percent"
                    ]
                )
    for model in ACCURACY_MODELS:
        for method in DELTA_METHODS:
            for workload in ACCURACY_WORKLOADS:
                composite_change[(method, workload)].append(
                    test24_accuracy[(model, method, workload)]
                    - test24_accuracy[(model, "Patch", workload)]
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

    # (a) Paired cache-ON latency saving from the matched Patch.
    runtime_x = list(range(len(ACCURACY_WORKLOADS)))
    for method in DELTA_METHODS:
        stats = [
            mean_and_sd(latency_saving[(method, workload)])
            for workload in ACCURACY_WORKLOADS
        ]
        means = [value for value, _ in stats]
        spreads = [value for _, value in stats]
        ax_latency.fill_between(
            runtime_x,
            [0.0] * len(runtime_x),
            means,
            color=METHOD_COLORS[method],
            alpha=0.055,
            linewidth=0,
            zorder=1,
        )
        ax_latency.errorbar(
            runtime_x,
            means,
            yerr=spreads,
            color=METHOD_COLORS[method],
            marker=MARKERS[method],
            markersize=5.6,
            linewidth=2.1,
            elinewidth=1.0,
            capsize=2.8,
            alpha=0.96,
            zorder=3,
        )
    ax_latency.axhline(0, color=METHOD_COLORS["Patch"], linestyle="--", linewidth=1.6)
    ax_latency.set_xticks(runtime_x, ACCURACY_WORKLOAD_LABELS)
    ax_latency.set_xlim(-0.15, 3.42)
    ax_latency.set_ylim(-5, 105)
    ax_latency.set_ylabel("Latency reduction from Patch (%) ↑")
    ax_latency.set_title("(a) Efficiency gain")

    # (b) Paired Composite change from the matched Patch.
    accuracy_x = list(range(len(ACCURACY_WORKLOADS)))
    for method in DELTA_METHODS:
        stats = [
            mean_and_sd(composite_change[(method, workload)])
            for workload in ACCURACY_WORKLOADS
        ]
        means = [value for value, _ in stats]
        spreads = [value for _, value in stats]
        ax_accuracy.fill_between(
            accuracy_x,
            [0.0] * len(accuracy_x),
            means,
            color=METHOD_COLORS[method],
            alpha=0.055,
            linewidth=0,
            zorder=1,
        )
        ax_accuracy.errorbar(
            accuracy_x,
            means,
            yerr=spreads,
            color=METHOD_COLORS[method],
            marker=MARKERS[method],
            markersize=5.6,
            linewidth=2.1,
            elinewidth=1.0,
            capsize=2.8,
            alpha=0.96,
            zorder=3,
        )
    ax_accuracy.axhline(0, color=METHOD_COLORS["Patch"], linestyle="--", linewidth=1.6)
    ax_accuracy.set_xticks(accuracy_x, ACCURACY_WORKLOAD_LABELS)
    ax_accuracy.set_xlim(-0.15, 3.42)
    ax_accuracy.set_ylim(-30, 10)
    ax_accuracy.set_ylabel("Composite change from Patch (pp) ↑")
    ax_accuracy.set_title("(b) Accuracy change")

    # Label only the high-load endpoints to keep the main-paper figure sparse.
    for method in DELTA_METHODS:
        latency_u80 = mean(latency_saving[(method, "80")])
        composite_u80 = mean(composite_change[(method, "80")])
        ax_latency.annotate(
            f"{latency_u80:.0f}%",
            (3, latency_u80),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            color=METHOD_COLORS[method],
            fontsize=8.5,
        )
        ax_accuracy.annotate(
            f"{composite_u80:.1f}",
            (3, composite_u80),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            color=METHOD_COLORS[method],
            fontsize=8.5,
        )

    for ax in (ax_latency, ax_accuracy):
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
            marker=None if method == "Patch" else MARKERS[method],
            linewidth=2,
            markersize=5.5,
            label="Patch baseline" if method == "Patch" else method,
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
        "Paired change from Patch; points and error bars show mean ± SD across six models.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.18, wspace=0.27)
    save(fig)
    plt.close(fig)

    print("Mean cache-ON latency reduction from Patch (Base / U40 / U60 / U80):")
    for method in DELTA_METHODS:
        values = [
            mean(latency_saving[(method, workload)])
            for workload in ACCURACY_WORKLOADS
        ]
        print(f"  {method}: " + " / ".join(f"{value:.2f}" for value in values))

    print("Mean Composite change from Patch (Base / U40 / U60 / U80):")
    for method in DELTA_METHODS:
        values = [
            mean(composite_change[(method, workload)])
            for workload in ACCURACY_WORKLOADS
        ]
        print(f"  {method}: " + " / ".join(f"{value:.2f}" for value in values))


if __name__ == "__main__":
    main()
