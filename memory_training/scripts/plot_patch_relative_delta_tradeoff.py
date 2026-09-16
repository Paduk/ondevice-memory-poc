#!/usr/bin/env python3
"""Plot Patch-relative update-cycle latency ratio against Composite."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"
MODELS = ("Granite 350M", "Qwen 0.8B", "Granite 1B", "Qwen 2B")
METHODS = ("Patch", "k=2", "k=5", "k=10")
WORKLOADS = ("base", "20", "40", "60", "80")
WORKLOAD_COLORS = {
    "base": "#6B7280",
    "20": "#56B4E9",
    "40": "#0072B2",
    "60": "#D55E00",
    "80": "#CC79A7",
}
METHOD_MARKERS = {"Patch": "o", "k=2": "s", "k=5": "D", "k=10": "P"}


def latency_ratio(delta: float, patch: float) -> float:
    return 100 * delta / patch


def save(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "patch-relative-delta-update-cycle-tradeoff"
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
    all_ratios: list[float] = []
    for cache in ("OFF", "ON"):
        metric = f"cache_{cache.lower()}_seconds_mean"
        for model in MODELS:
            methods = data[model]["methods"]
            for workload in WORKLOADS:
                patch = methods["Patch"][workload]["update_cycle"][metric]
                all_ratios.extend(
                    latency_ratio(methods[method][workload]["update_cycle"][metric], patch)
                    for method in METHODS
                )
    x_min = min(all_ratios) - 4
    x_max = max(all_ratios) + 4

    for row_index, cache in enumerate(("OFF", "ON")):
        metric = f"cache_{cache.lower()}_seconds_mean"
        for col_index, model in enumerate(MODELS):
            ax = axes[row_index, col_index]
            methods = data[model]["methods"]
            for workload in WORKLOADS:
                x, y = [], []
                for method in METHODS:
                    patch = methods["Patch"][workload]["update_cycle"][metric]
                    latency = methods[method][workload]["update_cycle"][metric]
                    x.append(latency_ratio(latency, patch))
                    y.append(methods[method][workload]["composite"])
                ax.plot(x, y, color=WORKLOAD_COLORS[workload], linewidth=1.5, alpha=0.82)
                for method, xv, yv in zip(METHODS, x, y):
                    ax.scatter(
                        xv,
                        yv,
                        marker=METHOD_MARKERS[method],
                        s=50,
                        color=WORKLOAD_COLORS[workload],
                        edgecolor="#111827",
                        linewidth=0.5,
                        zorder=3,
                    )
            ax.axvline(100, color="#111827", linestyle="--", linewidth=0.9, alpha=0.6)
            ax.set_xlim(x_min, x_max)
            ax.grid(True, color="#e5e7eb", linewidth=0.65)
            ax.set_title(model)
            if col_index == 0:
                ax.set_ylabel(f"Cache {cache}\nAbsolute Composite")
            if row_index == 1:
                ax.set_xlabel("Latency relative to Patch (%)")
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")

    workload_handles = [
        Line2D(
            [0], [0], color=WORKLOAD_COLORS[w], linewidth=2,
            label="Base" if w == "base" else f"U{w}",
        )
        for w in WORKLOADS
    ]
    method_handles = [
        Line2D(
            [0], [0], color="#111827", marker=METHOD_MARKERS[method], linestyle="none",
            markersize=6, label=method,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=workload_handles + method_handles,
        loc="upper center",
        ncol=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        "Patch-to-Delta k-sweeps: latency–accuracy trade-off",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Each line holds workload fixed and connects Patch → k2 → k5 → k10. Patch = 100%; left and up are better.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.85, bottom=0.10, hspace=0.20, wspace=0.10)
    save(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
