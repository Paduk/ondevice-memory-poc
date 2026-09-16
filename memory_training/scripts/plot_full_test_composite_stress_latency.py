#!/usr/bin/env python3
"""Plot 24-scenario Test Composite against 5-scenario stress Cache-ON latency."""

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
METHOD_COLORS = {"k=2": "#0072B2", "k=5": "#E69F00", "k=10": "#009E73"}
WORKLOAD_MARKERS = {"base": "o", "20": "s", "40": "^", "60": "D", "80": "P"}


def ratio(value: float, reference: float) -> float:
    return 100 * value / reference


def save(fig: plt.Figure, name: str = "full-test-composite-cache-on-stress-latency") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / name
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(fig)


def plot_relative_quadrants(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        methods = data[model]["methods"]
        patch_composite = methods["Patch"]["full_test"]["composite"]
        for method in ("k=2", "k=5", "k=10"):
            composite_delta = methods[method]["full_test"]["composite"] - patch_composite
            for workload in WORKLOADS:
                patch_latency = methods["Patch"][workload]["update_cycle"]["cache_on_seconds_mean"]
                method_latency = methods[method][workload]["update_cycle"]["cache_on_seconds_mean"]
                ax.scatter(
                    ratio(method_latency, patch_latency), composite_delta,
                    marker=WORKLOAD_MARKERS[workload], s=76,
                    color=METHOD_COLORS[method], edgecolor="#111827", linewidth=0.55, zorder=3,
                )
        ax.axvline(100, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.axhline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.grid(True, color="#e5e7eb", linewidth=0.7)
        ax.set_title(model)
        ax.set_xlabel("Cache-ON latency relative to matched Patch (%)")
        ax.set_ylabel("Full-Test Composite difference vs Patch (%p)")
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker="o", linestyle="none",
               markersize=7, label=m) for m in ("k=2", "k=5", "k=10")
    ]
    workload_handles = [
        Line2D([0], [0], color="#111827", marker=WORKLOAD_MARKERS[w], linestyle="none",
               markersize=6.5, label="Base" if w == "base" else f"U{w}") for w in WORKLOADS
    ]
    fig.legend(handles=method_handles + workload_handles, loc="upper center", ncol=8,
               frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Full-Test Delta operating points vs matched Patch",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.012,
        "Y: natural Test Composite (24 scenarios; Qwen Patch: seed 46). X: held-out stress latency (S86–S90). Upper-left dominates Patch.",
        ha="center", fontsize=9.3, color="#4b5563",
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.86, bottom=0.10, hspace=0.23, wspace=0.15)
    save(fig, "full-test-composite-delta-cache-on-quadrants")


def main() -> None:
    data = json.loads(INPUT.read_text())["models"]
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
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        methods = data[model]["methods"]
        for workload in WORKLOADS:
            patch_latency = methods["Patch"][workload]["update_cycle"]["cache_on_seconds_mean"]
            xs = [
                ratio(methods[m][workload]["update_cycle"]["cache_on_seconds_mean"], patch_latency)
                for m in METHODS
            ]
            ys = [methods[m]["full_test"]["composite"] for m in METHODS]
            ax.plot(xs, ys, color=WORKLOAD_COLORS[workload], linewidth=1.7, alpha=0.85)
            for method, x, y in zip(METHODS, xs, ys):
                ax.scatter(
                    x, y, marker=METHOD_MARKERS[method], s=67,
                    color=WORKLOAD_COLORS[workload], edgecolor="#111827", linewidth=0.55, zorder=3,
                )
        ax.axvline(100, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.grid(True, color="#e5e7eb", linewidth=0.7)
        ax.set_title(model)
        ax.set_xlabel("Cache-ON latency relative to matched Patch (%)")
        ax.set_ylabel("Full-Test Composite (24 scenarios)")
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")

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
    fig.suptitle("Full-Test accuracy vs Cache-ON stress latency",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.012,
        "Y: natural Test Composite (24 scenarios; Qwen Patch: seed 46). X: held-out stress latency (S86–S90). Patch = 100%; left and up are better.",
        ha="center", fontsize=9.3, color="#4b5563",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.10, hspace=0.23, wspace=0.15)
    save(fig)
    plot_relative_quadrants(data)


if __name__ == "__main__":
    main()
