#!/usr/bin/env python3
"""Create two readable Cache-ON Patch-vs-Delta trade-off alternatives."""

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
DELTAS = ("k=2", "k=5", "k=10")
WORKLOADS = ("base", "20", "40", "60", "80")
WORKLOAD_COLORS = {
    "base": "#6B7280",
    "20": "#56B4E9",
    "40": "#0072B2",
    "60": "#D55E00",
    "80": "#CC79A7",
}
METHOD_COLORS = {"k=2": "#0072B2", "k=5": "#E69F00", "k=10": "#009E73"}
METHOD_MARKERS = {"Patch": "o", "k=2": "s", "k=5": "D", "k=10": "P"}
WORKLOAD_MARKERS = {"base": "o", "20": "s", "40": "^", "60": "D", "80": "P"}


def ratio(delta: float, patch: float) -> float:
    return 100 * delta / patch


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {})):
        path = OUTPUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")


def plot_absolute_k_sweeps(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        methods = data[model]["methods"]
        for workload in WORKLOADS:
            patch_latency = methods["Patch"][workload]["update_cycle"]["cache_on_seconds_mean"]
            xs = [
                ratio(methods[m][workload]["update_cycle"]["cache_on_seconds_mean"], patch_latency)
                for m in METHODS
            ]
            ys = [methods[m][workload]["composite"] for m in METHODS]
            ax.plot(xs, ys, color=WORKLOAD_COLORS[workload], linewidth=1.7, alpha=0.85)
            for method, x, y in zip(METHODS, xs, ys):
                ax.scatter(
                    x, y, marker=METHOD_MARKERS[method], s=65,
                    color=WORKLOAD_COLORS[workload], edgecolor="#111827", linewidth=0.55, zorder=3,
                )
        ax.axvline(100, color="#111827", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_title(model)
        ax.set_xlabel("Latency relative to Patch (%)")
        ax.set_ylabel("Absolute Composite")
        style_axis(ax)
    workload_handles = [
        Line2D([0], [0], color=WORKLOAD_COLORS[w], linewidth=2.2,
               label="Base" if w == "base" else f"U{w}") for w in WORKLOADS
    ]
    method_handles = [
        Line2D([0], [0], color="#111827", marker=METHOD_MARKERS[m], linestyle="none",
               markersize=6.5, label=m) for m in METHODS
    ]
    fig.legend(handles=workload_handles + method_handles, loc="upper center", ncol=9,
               frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Cache ON: workload-matched Patch-to-Delta k-sweeps",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012,
             "Each line holds workload fixed. Patch = 100%; left and up are better.",
             ha="center", fontsize=9.5, color="#4b5563")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.09, hspace=0.23, wspace=0.14)
    save(fig, "cache-on-patch-delta-absolute-composite-k-sweeps")


def plot_patch_delta_quadrants(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        methods = data[model]["methods"]
        for method in DELTAS:
            for workload in WORKLOADS:
                patch = methods["Patch"][workload]
                delta = methods[method][workload]
                x = ratio(
                    delta["update_cycle"]["cache_on_seconds_mean"],
                    patch["update_cycle"]["cache_on_seconds_mean"],
                )
                y = delta["composite"] - patch["composite"]
                ax.scatter(
                    x, y, marker=WORKLOAD_MARKERS[workload], s=72,
                    color=METHOD_COLORS[method], edgecolor="#111827", linewidth=0.55, zorder=3,
                )
        ax.axvline(100, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.axhline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.set_title(model)
        ax.set_xlabel("Latency relative to Patch (%)")
        ax.set_ylabel("Composite difference vs Patch (%p)")
        style_axis(ax)
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker="o", linestyle="none",
               markersize=7, label=m) for m in DELTAS
    ]
    workload_handles = [
        Line2D([0], [0], color="#111827", marker=WORKLOAD_MARKERS[w], linestyle="none",
               markersize=6.5, label="Base" if w == "base" else f"U{w}") for w in WORKLOADS
    ]
    fig.legend(handles=method_handles + workload_handles, loc="upper center", ncol=8,
               frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Cache ON: Delta operating points relative to matched Patch",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012,
             "Upper-left beats Patch on both metrics; lower-left trades accuracy for latency.",
             ha="center", fontsize=9.5, color="#4b5563")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.09, hspace=0.23, wspace=0.14)
    save(fig, "cache-on-patch-relative-delta-quadrants")


def main() -> None:
    setup_style()
    data = json.loads(INPUT.read_text())["models"]
    plot_absolute_k_sweeps(data)
    plot_patch_delta_quadrants(data)


if __name__ == "__main__":
    main()
