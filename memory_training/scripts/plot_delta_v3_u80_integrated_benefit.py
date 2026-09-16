#!/usr/bin/env python3
"""Plot one integrated U80 Patch-relative benefit panel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"
OUTPUT_STEM = OUTPUT_DIR / "delta-v3-u80-integrated-benefit"

MODELS = ("Granite 350M", "Qwen 0.8B", "Granite 1B", "Qwen 2B")
METHODS = ("k=2", "k=5", "k=10")
MODEL_COLORS = {
    "Granite 350M": "#0072B2",
    "Qwen 0.8B": "#E69F00",
    "Granite 1B": "#009E73",
    "Qwen 2B": "#CC79A7",
}
METHOD_MARKERS = {"k=2": "s", "k=5": "D", "k=10": "P"}


def main() -> None:
    data = json.loads(INPUT.read_text())["models"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )

    fig, ax = plt.subplots(figsize=(9.4, 6.7))
    for model in MODELS:
        methods = data[model]["methods"]
        patch_latency = methods["Patch"]["80"]["update_cycle"]["cache_on_seconds_mean"]
        patch_composite = methods["Patch"]["80"]["composite"]
        for method in METHODS:
            latency = methods[method]["80"]["update_cycle"]["cache_on_seconds_mean"]
            saving = 100.0 * (1.0 - latency / patch_latency)
            composite_delta = methods[method]["80"]["composite"] - patch_composite
            ax.scatter(
                saving,
                composite_delta,
                s=112,
                marker=METHOD_MARKERS[method],
                color=MODEL_COLORS[model],
                edgecolor="#111827",
                linewidth=0.75,
                zorder=3,
            )

    ax.scatter(0, 0, s=105, marker="o", color="#111827", zorder=4)
    ax.annotate("Patch", (0, 0), xytext=(7, 7), textcoords="offset points", fontsize=9.5)
    ax.axhline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.axvline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_xlim(-3, 72)
    ax.set_ylim(-32, 6)
    ax.set_xlabel("U80 Cache-ON latency saving vs matched Patch (%)")
    ax.set_ylabel("U80 Stress Composite difference vs matched Patch (%p)")
    ax.set_title("Integrated U80 Patch-relative benefit", fontweight="bold", pad=15)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")

    model_handles = [
        Line2D([0], [0], linestyle="none", marker="o", markersize=7,
               markerfacecolor=MODEL_COLORS[model], markeredgecolor="#111827", label=model)
        for model in MODELS
    ]
    method_handles = [
        Line2D([0], [0], linestyle="none", marker=METHOD_MARKERS[method], markersize=7,
               markerfacecolor="#9ca3af", markeredgecolor="#111827", label=method)
        for method in METHODS
    ]
    fig.legend(
        handles=model_handles + method_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
    )
    fig.text(
        0.5,
        0.018,
        "Both axes use the matched U80 S86–S90 stress result. Upper-right dominates Patch; relative latency hides absolute model speed.",
        ha="center",
        fontsize=9.2,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.77, bottom=0.13)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {})):
        output = OUTPUT_STEM.with_suffix(f".{suffix}")
        fig.savefig(output, facecolor="white", bbox_inches="tight", **options)
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
