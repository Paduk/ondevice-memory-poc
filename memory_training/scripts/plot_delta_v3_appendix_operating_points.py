#!/usr/bin/env python3
"""Plot all 4 models x 5 workloads as appendix operating-point panels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"

MODELS = ("Granite 350M", "Qwen 0.8B", "Granite 1B", "Qwen 2B")
METHODS = ("Patch", "k=2", "k=5", "k=10")
WORKLOADS = ("base", "20", "40", "60", "80")
METHOD_COLORS = {
    "Patch": "#111827",
    "k=2": "#0072B2",
    "k=5": "#E69F00",
    "k=10": "#009E73",
}
METHOD_MARKERS = {"Patch": "o", "k=2": "s", "k=5": "D", "k=10": "P"}
POINT_LABELS = {"Patch": "P", "k=2": "2", "k=5": "5", "k=10": "10"}
LABEL_OFFSETS = {
    "Patch": (-7, 7, "right"),
    "k=2": (6, 7, "left"),
    "k=5": (6, -13, "left"),
    "k=10": (-7, -13, "right"),
}
GLOBAL_Y_LIMITS = (55, 74)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def pareto_front(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    frontier = []
    for label, x, y in points:
        dominated = any(
            other_x <= x and other_y >= y and (other_x < x or other_y > y)
            for other_label, other_x, other_y in points
            if other_label != label
        )
        if not dominated:
            frontier.append((label, x, y))
    return sorted(frontier, key=lambda row: row[1])


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {})):
        path = (OUTPUT_DIR / stem).with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(fig)


def plot_model(model: str, model_data: dict) -> None:
    methods = model_data["methods"]
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 4.15), sharex=True, sharey=True)
    for ax, workload in zip(axes, WORKLOADS):
        patch_seconds = methods["Patch"][workload]["update_cycle"]["cache_on_seconds_mean"]
        points = []
        for method in METHODS:
            seconds = methods[method][workload]["update_cycle"]["cache_on_seconds_mean"]
            x = 100.0 * seconds / patch_seconds
            y = methods[method]["full_test"]["composite"]
            points.append((method, x, y))

        frontier = pareto_front(points)
        frontier_names = {row[0] for row in frontier}
        for method, x, y in points:
            ax.scatter(
                x,
                y,
                s=92 if method in frontier_names else 68,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                edgecolor="#111827",
                linewidth=1.8 if method in frontier_names else 0.4,
                zorder=3,
            )
            dx, dy, alignment = LABEL_OFFSETS[method]
            ax.annotate(
                POINT_LABELS[method],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=alignment,
                va="center",
                fontsize=8.5,
                fontweight="bold" if method in frontier_names else "normal",
                color=METHOD_COLORS[method],
                zorder=4,
            )

        label = "Base" if workload == "base" else f"U{workload}"
        ax.set_title(f"{label}\nPatch = {patch_seconds:.1f} s/cycle")
        ax.axvline(100, color="#111827", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.axhline(methods["Patch"]["full_test"]["composite"], color="#9ca3af",
                   linestyle=":", linewidth=0.8)
        ax.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
        ax.set_xlim(25, 104)
        ax.set_ylim(*GLOBAL_Y_LIMITS)
        ax.set_xlabel("← Faster · Latency vs Patch (%)")
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")

    axes[0].set_ylabel("Full-Test Composite (24 scenarios) · Higher ↑")
    handles = [
        Line2D(
            [0], [0], linestyle="none", marker=METHOD_MARKERS[method], markersize=7,
            markerfacecolor=METHOD_COLORS[method], markeredgecolor="#111827", label=method,
        )
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f"{model}: Patch vs Delta-v3 across all update workloads",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.015,
        "X: matched stress Cache-ON update-cycle latency. Y: one fixed 24-scenario Test Composite per method. Bold outline: non-dominated point.",
        ha="center",
        fontsize=9.2,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.77, bottom=0.19, wspace=0.10)
    save(fig, f"delta-v3-appendix-operating-points-{slugify(model)}")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    data = json.loads(INPUT.read_text())["models"]
    for model in MODELS:
        plot_model(model, data[model])


if __name__ == "__main__":
    main()
