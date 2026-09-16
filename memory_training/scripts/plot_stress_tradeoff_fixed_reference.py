#!/usr/bin/env python3
"""Plot Composite vs update-cycle latency normalized to Patch/Base/Cache-OFF."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"
MODELS = ("Granite 350M", "Qwen 0.8B", "Granite 1B", "Qwen 2B")
METHODS = ("Patch", "Summary", "k=2", "k=5", "k=10")
WORKLOADS = ("base", "20", "40", "60", "80")
COLORS = {
    "Patch": "#6B7280",
    "Summary": "#CC79A7",
    "k=2": "#0072B2",
    "k=5": "#E69F00",
    "k=10": "#009E73",
}
MARKERS = {"base": "o", "20": "s", "40": "^", "60": "D", "80": "P"}


def save(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "stress-tradeoff-fixed-patch-base-normalized"
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)


def main() -> None:
    data = json.loads(INPUT.read_text())["models"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), sharex=True, sharey=True)
    for row_index, cache in enumerate(("OFF", "ON")):
        latency_key = f"cache_{cache.lower()}_seconds_mean"
        for col_index, model in enumerate(MODELS):
            ax = axes[row_index, col_index]
            methods = data[model]["methods"]
            reference = methods["Patch"]["base"]["update_cycle"]["cache_off_seconds_mean"]
            for method in METHODS:
                x = [methods[method][w]["update_cycle"][latency_key] / reference for w in WORKLOADS]
                y = [methods[method][w]["composite"] for w in WORKLOADS]
                ax.plot(x, y, color=COLORS[method], linewidth=1.55, alpha=0.9)
                for workload, xv, yv in zip(WORKLOADS, x, y):
                    ax.scatter(
                        xv,
                        yv,
                        marker=MARKERS[workload],
                        s=48,
                        color=COLORS[method],
                        edgecolor="#111827",
                        linewidth=0.5,
                        zorder=3,
                    )
            ax.axvline(1.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.55)
            ax.grid(True, color="#e5e7eb", linewidth=0.65)
            ax.set_title(model)
            if col_index == 0:
                ax.set_ylabel(f"Cache {cache}\nAbsolute Composite")
            if row_index == 1:
                ax.set_xlabel("Normalized update-cycle latency")
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")

    method_handles = [
        Line2D([0], [0], color=COLORS[method], linewidth=2, label=method) for method in METHODS
    ]
    workload_handles = [
        Line2D(
            [0], [0], color="#111827", marker=MARKERS[w], linestyle="none", markersize=6,
            label="Base" if w == "base" else f"U{w}",
        )
        for w in WORKLOADS
    ]
    fig.legend(
        handles=method_handles + workload_handles,
        loc="upper center",
        ncol=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        "Composite vs fixed-reference update-cycle latency",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Reference within each model: Patch · Base · Cache OFF = 1.0. Left and up are better.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.85, bottom=0.10, hspace=0.20, wspace=0.10)
    save(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
