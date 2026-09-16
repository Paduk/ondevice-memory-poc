#!/usr/bin/env python3
"""Plot base-test accuracy, stress-runtime scaling, and stress accuracy.

Panel (a) reports the fixed 24-scenario Test Composite for Patch and each
Delta-v3 compaction interval. Panel (b) reports cache-ON update-cycle latency
under Base/U20/U40/U60/U80, normalized to the matched Patch for every model and
workload. Panel (c) reports Composite measured on the same held-out stress
subset from Base through U80. The panels deliberately keep standard-Test quality,
stress runtime, and stress-set quality distinct.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/delta-v3-six-model-workload-response.csv"
FOUR_MODEL_STRESS_INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.csv"
OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"
STRESS_WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training/benchmarks")

MODELS = (
    "Granite 350M",
    "Qwen 0.8B",
    "Llama 1B",
    "Granite 1B",
    "Qwen 2B",
    "Llama 3B",
)
FOUR_MODEL_STRESS_MODELS = (
    "Granite 350M",
    "Qwen 0.8B",
    "Granite 1B",
    "Qwen 2B",
)
METHODS = ("Patch", "k=2", "k=5", "k=10")
DELTA_METHODS = ("k=2", "k=5", "k=10")
WORKLOADS = ("base", "20", "40", "60", "80")
WORKLOAD_LABELS = ("Base", "U20", "U40", "U60", "U80")

METHOD_COLORS = {
    "Patch": "#6B7280",
    "k=2": "#0072B2",
    "k=5": "#E69F00",
    "k=10": "#009E73",
}
METHOD_MARKERS = {"k=2": "o", "k=5": "s", "k=10": "D"}
METHOD_SLUGS = {
    "Patch": "patch",
    "k=2": "delta_v3_compact_k2",
    "k=5": "delta_v3_compact_k5",
    "k=10": "delta_v3_compact_k10",
}
LLAMA_STRESS_RUNS = {
    "Llama 1B": "llama32-1b-stress-test-once-20260908-v1",
    "Llama 3B": "llama32-3b-stress-test-once-20260908-v1",
}


def load_rows() -> list[dict]:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["test_composite"] = float(row["test_composite"])
        row["relative_latency_percent"] = float(row["relative_latency_percent"])
    return rows


def composite(summary: dict) -> float:
    return 100.0 * (
        0.60 * float(summary["closed_loop_quiz"]["esm"])
        + 0.25 * float(summary["memory"]["final_state_f1"])
        + 0.15 * float(summary["memory"]["update_f1"])
    )


def load_stress_composites() -> dict[tuple[str, str, str], float]:
    """Load same-subset Composite for all workloads across all six models."""
    values: dict[tuple[str, str, str], float] = {}
    with FOUR_MODEL_STRESS_INPUT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in FOUR_MODEL_STRESS_MODELS or row["method"] not in METHODS:
                continue
            if row["workload"] not in WORKLOADS:
                continue
            values[(row["model"], row["method"], row["workload"])] = float(row["composite"])

    for model, run_name in LLAMA_STRESS_RUNS.items():
        for method in METHODS:
            for workload in WORKLOADS:
                path = (
                    STRESS_WORKSPACE
                    / run_name
                    / "composite"
                    / workload
                    / METHOD_SLUGS[method]
                    / "summary.json"
                )
                summary = json.loads(path.read_text())
                if not summary.get("complete") or len(summary.get("completed_scenarios", [])) != 5:
                    raise ValueError(f"Incomplete stress Composite: {path}")
                values[(model, method, workload)] = composite(summary)

    expected = len(MODELS) * len(METHODS) * len(WORKLOADS)
    if len(values) != expected:
        raise ValueError(f"Expected {expected} stress Composite values, got {len(values)}")
    return values


def save(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "delta-v3-compaction-interval-ablation-stacked"
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)


def main() -> None:
    rows = load_rows()
    stress_composites = load_stress_composites()
    index = {
        (row["model"], row["method"], row["workload"]): row
        for row in rows
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(12.8, 7.0))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(0.92, 1.35),
        height_ratios=(1, 1),
        hspace=0.14,
        wspace=0.28,
    )
    ax_accuracy = fig.add_subplot(grid[:, 0])
    ax_latency = fig.add_subplot(grid[0, 1])
    ax_stress_accuracy = fig.add_subplot(grid[1, 1], sharex=ax_latency)

    # Panel (a): fixed standard-Test score. Bars start at zero so the modest
    # differences among methods are not visually exaggerated.
    x_methods = list(range(len(METHODS)))
    method_values = {
        method: [index[(model, method, "base")]["test_composite"] for model in MODELS]
        for method in METHODS
    }
    method_means = [mean(method_values[method]) for method in METHODS]
    method_stds = [stdev(method_values[method]) for method in METHODS]
    ax_accuracy.bar(
        x_methods,
        method_means,
        yerr=method_stds,
        width=0.68,
        capsize=3.5,
        color=[METHOD_COLORS[method] for method in METHODS],
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "ecolor": "#374151"},
        zorder=2,
    )
    point_offsets = (-0.20, -0.12, -0.04, 0.04, 0.12, 0.20)
    for method_index, method in enumerate(METHODS):
        ax_accuracy.scatter(
            [method_index + offset for offset in point_offsets],
            method_values[method],
            s=15,
            color="#111827",
            alpha=0.48,
            linewidth=0,
            zorder=3,
        )
    ax_accuracy.set_xticks(x_methods, METHODS)
    ax_accuracy.set_ylabel("Composite (%)")
    ax_accuracy.set_title("(a) Accuracy on the standard Test")
    ax_accuracy.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax_accuracy.set_ylim(0, 80)

    # Panel (b): matched-Patch-relative runtime. Lines show the six-model mean;
    # bands show one sample standard deviation across model configurations.
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for model in MODELS:
        for method in DELTA_METHODS:
            for workload in WORKLOADS:
                grouped[(method, workload)].append(
                    index[(model, method, workload)]["relative_latency_percent"]
                )

    x_workloads = list(range(len(WORKLOADS)))
    for method in DELTA_METHODS:
        means = [mean(grouped[(method, workload)]) for workload in WORKLOADS]
        stds = [stdev(grouped[(method, workload)]) for workload in WORKLOADS]
        lower = [value - spread for value, spread in zip(means, stds)]
        upper = [value + spread for value, spread in zip(means, stds)]
        ax_latency.fill_between(
            x_workloads,
            lower,
            upper,
            color=METHOD_COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
        ax_latency.plot(
            x_workloads,
            means,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5.2,
            linewidth=2.0,
            label=method,
        )
    ax_latency.axhline(
        100,
        color="#4B5563",
        linestyle="--",
        linewidth=1.5,
        label="Patch",
    )
    ax_latency.set_xticks(x_workloads)
    ax_latency.tick_params(axis="x", labelbottom=False)
    ax_latency.set_ylabel("Cache-ON update-cycle latency\nrelative to matched Patch (%)")
    ax_latency.set_title("(b) Runtime scaling under update load")
    ax_latency.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    ax_latency.legend(ncol=4, frameon=False, loc="upper right")
    ax_latency.set_ylim(bottom=20, top=106)

    # Panel (c): accuracy measured on the same held-out subset at every load.
    stress_markers = {"Patch": "X", "k=2": "o", "k=5": "s", "k=10": "D"}
    for method in METHODS:
        values_by_workload = [
            [stress_composites[(model, method, workload)] for model in MODELS]
            for workload in WORKLOADS
        ]
        averages = [mean(values) for values in values_by_workload]
        spreads = [stdev(values) for values in values_by_workload]
        lower = [value - spread for value, spread in zip(averages, spreads)]
        upper = [value + spread for value, spread in zip(averages, spreads)]
        ax_stress_accuracy.fill_between(
            x_workloads,
            lower,
            upper,
            color=METHOD_COLORS[method],
            alpha=0.10,
            linewidth=0,
        )
        ax_stress_accuracy.plot(
            x_workloads,
            averages,
            color=METHOD_COLORS[method],
            linestyle="--" if method == "Patch" else "-",
            marker=stress_markers[method],
            markersize=5.0,
            linewidth=1.8,
            label=method,
            zorder=2,
        )
    ax_stress_accuracy.set_xticks(x_workloads, WORKLOAD_LABELS)
    ax_stress_accuracy.set_ylabel("Composite (%)")
    ax_stress_accuracy.set_title("(c) Accuracy on the stress subset")
    ax_stress_accuracy.set_ylim(0, 80)
    ax_stress_accuracy.grid(axis="y", color="#E5E7EB", linewidth=0.75)

    for ax in (ax_accuracy, ax_latency, ax_stress_accuracy):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")

    fig.text(
        0.5,
        -0.015,
        "(a) 24-scenario V2 Test; (b,c) held-out S86–S90 stress subset. "
        "Bars/error bars and lines/bands denote mean ± SD across six model configurations.",
        ha="center",
        fontsize=8.3,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.11)
    save(fig)
    plt.close(fig)

    print("Mean Base Composite:")
    for method, value in zip(METHODS, method_means):
        print(f"  {method}: {value:.2f}")
    print("Mean Patch-relative latency at Base / U80:")
    for method in DELTA_METHODS:
        base = mean(grouped[(method, "base")])
        u80 = mean(grouped[(method, "80")])
        print(f"  {method}: {base:.1f}% / {u80:.1f}%")
    print("Mean stress-subset Composite by workload:")
    for method in METHODS:
        values = [
            mean(stress_composites[(model, method, workload)] for model in MODELS)
            for workload in WORKLOADS
        ]
        print(f"  {method}: " + " / ".join(f"{value:.2f}" for value in values))


if __name__ == "__main__":
    main()
