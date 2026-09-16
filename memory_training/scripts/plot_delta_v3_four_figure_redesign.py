#!/usr/bin/env python3
"""Create four compact Patch-vs-Delta figures from the consolidated results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT_DIR = REPO_ROOT / "docs/engineering/figures"

MODELS = ("Granite 350M", "Qwen 0.8B", "Granite 1B", "Qwen 2B")
METHODS = ("k=2", "k=5", "k=10")
WORKLOADS = ("base", "20", "40", "60", "80")
WORKLOAD_LABELS = ("Base", "U20", "U40", "U60", "U80")
METHOD_COLORS = {"k=2": "#0072B2", "k=5": "#E69F00", "k=10": "#009E73"}
METHOD_MARKERS = {"k=2": "s", "k=5": "D", "k=10": "P"}


def load_data() -> dict:
    return json.loads(INPUT.read_text())["models"]


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {})):
        path = (OUTPUT_DIR / stem).with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(fig)


def latency_ratio(methods: dict, method: str, workload: str) -> float:
    patch = methods["Patch"][workload]["update_cycle"]["cache_on_seconds_mean"]
    value = methods[method][workload]["update_cycle"]["cache_on_seconds_mean"]
    return 100.0 * value / patch


def composite_delta(methods: dict, method: str) -> float:
    return methods[method]["full_test"]["composite"] - methods["Patch"]["full_test"]["composite"]


def setup_axes(axes) -> None:
    for ax in axes.flat:
        ax.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.012, text, ha="center", fontsize=9.1, color="#4b5563")


def plot_stress_response(data: dict) -> None:
    """Aligned latency and matched stress-quality scaling from Base to U80."""
    fig = plt.figure(figsize=(12.0, 11.8))
    grid = fig.add_gridspec(5, 2, height_ratios=(1.0, 0.76, 0.23, 1.0, 0.76))
    latency_axes = []
    quality_axes = []
    all_methods = ("Patch", *METHODS)
    colors = {"Patch": "#111827", **METHOD_COLORS}
    markers = {"Patch": "o", **METHOD_MARKERS}
    x = np.arange(len(WORKLOADS))

    for index, model in enumerate(MODELS):
        col = index % 2
        row = 0 if index < 2 else 3
        latency_ax = fig.add_subplot(grid[row, col])
        quality_ax = fig.add_subplot(grid[row + 1, col], sharex=latency_ax)
        latency_axes.append(latency_ax)
        quality_axes.append(quality_ax)
        methods = data[model]["methods"]
        patch_base = methods["Patch"]["base"]["update_cycle"]["cache_on_seconds_mean"]
        for method in all_methods:
            latency = [
                methods[method][workload]["update_cycle"]["cache_on_seconds_mean"] / patch_base
                for workload in WORKLOADS
            ]
            quality = [methods[method][workload]["composite"] for workload in WORKLOADS]
            style = {
                "color": colors[method],
                "marker": markers[method],
                "linewidth": 2.1,
                "markersize": 6.2,
                "markeredgecolor": "#111827",
                "markeredgewidth": 0.45,
            }
            latency_ax.plot(x, latency, **style)
            quality_ax.plot(x, quality, **style)
        latency_ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.9)
        latency_ax.set_title(model, fontweight="bold")
        latency_ax.set_ylabel("Latency vs Patch Base (×)")
        latency_ax.tick_params(axis="x", labelbottom=False)
        quality_ax.set_ylabel("Stress Composite")
        if index >= 2:
            quality_ax.set_xlabel("UPDATE workload")
        quality_ax.set_xticks(x, WORKLOAD_LABELS)

    setup_axes(np.array(latency_axes, dtype=object).reshape(2, 2))
    setup_axes(np.array(quality_axes, dtype=object).reshape(2, 2))
    for ax in latency_axes:
        ax.set_ylim(0.55, 3.75)
    for ax in quality_axes:
        ax.set_ylim(25, 72)
    handles = [
        Line2D([0], [0], color=colors[m], marker=markers[m], linewidth=2,
               markersize=7, label=m) for m in all_methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.952))
    fig.suptitle("Cache-ON latency and accuracy under update stress",
                 fontsize=15, fontweight="bold", y=0.995)
    footer(fig, "Top: mean update-cycle latency normalized to each model's Patch Base. Bottom: matched S86–S90 stress Composite; lines connect observations only.")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.075,
                        hspace=0.10, wspace=0.16)
    save(fig, "delta-v3-stress-response-arrows")


def pareto_front(points: list[tuple[str, float, float]]) -> set[str]:
    keep = set()
    for label, x, y in points:
        dominated = any(
            other_x <= x and other_y >= y and (other_x < x or other_y > y)
            for other_label, other_x, other_y in points if other_label != label
        )
        if not dominated:
            keep.add(label)
    return keep


def plot_u80_pareto(data: dict) -> None:
    """U80 cost versus full-Test quality, with the empirical frontier highlighted."""
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    setup_axes(axes)
    for ax, model in zip(axes.flat, MODELS):
        methods = data[model]["methods"]
        points = [("Patch", 100.0, 0.0)] + [
            (m, latency_ratio(methods, m, "80"), composite_delta(methods, m))
            for m in METHODS
        ]
        front = pareto_front(points)
        front_points = sorted((x, y) for label, x, y in points if label in front)
        ax.plot([x for x, _ in front_points], [y for _, y in front_points],
                color="#111827", linewidth=2.2, zorder=1)
        for label, x, y in points:
            if label == "Patch":
                color, marker = "#111827", "o"
            else:
                color, marker = METHOD_COLORS[label], METHOD_MARKERS[label]
            ax.scatter(x, y, s=105 if label in front else 75, marker=marker, color=color,
                       edgecolor="#111827", linewidth=1.1 if label in front else 0.45,
                       zorder=3)
            ax.annotate(label, (x, y), xytext=(-7 if x > 90 else 6, 7),
                        textcoords="offset points", ha="right" if x > 90 else "left",
                        fontsize=9, fontweight="bold" if label in front else "normal")
        ax.axvline(100, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.axhline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.set_title(model)
        ax.set_xlabel("U80 Cache-ON latency vs Patch (%)")
        ax.set_ylabel("Full-Test Composite difference vs Patch (%p)")
    fig.suptitle("U80 quality–latency operating points",
                 fontsize=15, fontweight="bold", y=0.995)
    footer(fig, "The black polyline is the empirical non-dominated frontier. Upper-left is better; accuracy and latency use different evaluation scopes.")
    fig.subplots_adjust(left=0.095, right=0.99, top=0.93, bottom=0.105,
                        hspace=0.24, wspace=0.16)
    save(fig, "delta-v3-u80-pareto-by-model")


def budget_envelope(methods: dict) -> list[tuple[float, float, str]]:
    candidates = [(0.0, 0.0, "Patch")]
    for method in METHODS:
        loss = max(0.0, -composite_delta(methods, method))
        saving = 100.0 - latency_ratio(methods, method, "80")
        candidates.append((loss, saving, method))
    thresholds = sorted({0.0, 8.0, *(loss for loss, _, _ in candidates)})
    result = []
    for threshold in thresholds:
        eligible = [row for row in candidates if row[0] <= threshold + 1e-9]
        loss, saving, method = max(eligible, key=lambda row: row[1])
        result.append((threshold, saving, method))
    return result


def plot_accuracy_budget(data: dict) -> None:
    """Empirical best U80 saving allowed by a Full-Test accuracy-loss budget."""
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    setup_axes(axes)
    for ax, model in zip(axes.flat, MODELS):
        methods = data[model]["methods"]
        envelope = budget_envelope(methods)
        xs = [row[0] for row in envelope]
        ys = [row[1] for row in envelope]
        ax.step(xs, ys, where="post", color="#0072B2", linewidth=2.4, zorder=2)
        last_method = None
        for x, y, method in envelope[:-1]:
            if method != last_method:
                color = "#111827" if method == "Patch" else METHOD_COLORS[method]
                marker = "o" if method == "Patch" else METHOD_MARKERS[method]
                ax.scatter(x, y, s=78, color=color, marker=marker,
                           edgecolor="#111827", linewidth=0.55, zorder=3)
                ax.annotate(method, (x, y), xytext=(6, 6), textcoords="offset points",
                            fontsize=9, color=color)
                last_method = method
        ax.set_title(model)
        ax.set_xlabel("Allowed Full-Test Composite loss (%p)")
        ax.set_ylabel("Best available U80 latency saving vs Patch (%)")
        ax.set_xlim(-0.15, 8.0)
        ax.set_ylim(-2, 71)
    fig.suptitle("Empirical U80 latency saving under an accuracy budget",
                 fontsize=15, fontweight="bold", y=0.995)
    footer(fig, "Step envelope over the observed Patch/k=2/k=5/k=10 points only; no interpolation or statistical confidence is implied.")
    fig.subplots_adjust(left=0.095, right=0.99, top=0.93, bottom=0.105,
                        hspace=0.24, wspace=0.16)
    save(fig, "delta-v3-u80-accuracy-budget-envelope")


def plot_decision_matrix(data: dict) -> None:
    """Compact model-by-k view: saving as text, quality delta as color."""
    quality = np.array([
        [composite_delta(data[model]["methods"], method) for method in METHODS]
        for model in MODELS
    ])
    saving = np.array([
        [100.0 - latency_ratio(data[model]["methods"], method, "80") for method in METHODS]
        for model in MODELS
    ])
    bound = max(abs(float(quality.min())), abs(float(quality.max())))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    image = ax.imshow(quality, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_xlabel("Delta-v3 compaction interval")
    ax.set_ylabel("Model")
    ax.tick_params(length=0)
    for row, model in enumerate(MODELS):
        methods = data[model]["methods"]
        points = [("Patch", 100.0, 0.0)] + [
            (m, latency_ratio(methods, m, "80"), composite_delta(methods, m))
            for m in METHODS
        ]
        front = pareto_front(points)
        for col, method in enumerate(METHODS):
            value = quality[row, col]
            text_color = "white" if abs(value) > 0.62 * bound else "#111827"
            ax.text(col, row - 0.08, f"{saving[row, col]:.0f}% faster",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color=text_color)
            ax.text(col, row + 0.22, f"Composite {value:+.2f}%p",
                    ha="center", va="center", fontsize=9.5, color=text_color)
            if method in front:
                ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="#111827", linewidth=2.2))
    for x in np.arange(-0.5, len(METHODS), 1):
        ax.axvline(x, color="white", linewidth=2)
    for y in np.arange(-0.5, len(MODELS), 1):
        ax.axhline(y, color="white", linewidth=2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Full-Test Composite difference vs Patch (%p)")
    ax.set_title("U80 Delta-v3 decision matrix", fontsize=15, fontweight="bold", pad=18)
    footer(fig, "Cell text: U80 Cache-ON latency saving. Cell color: 24-scenario Full-Test quality difference. Dark outline: non-dominated at U80.")
    fig.subplots_adjust(left=0.19, right=0.90, top=0.88, bottom=0.14)
    save(fig, "delta-v3-u80-decision-matrix")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    data = load_data()
    plot_stress_response(data)
    plot_u80_pareto(data)
    plot_accuracy_budget(data)
    plot_decision_matrix(data)


if __name__ == "__main__":
    main()
