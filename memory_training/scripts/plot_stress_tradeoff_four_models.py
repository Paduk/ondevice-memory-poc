#!/usr/bin/env python3
"""Plot four-model stress scaling and projected on-device Pareto views."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INPUT = Path("/home/hj153lee/PalmClaw/docs/engineering/results/stress-tradeoff-four-models.csv")
OUTPUT_DIR = Path("/home/hj153lee/PalmClaw/docs/engineering/figures")
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
MODEL_MARKERS = {"Granite 350M": "o", "Qwen 0.8B": "s", "Granite 1B": "^", "Qwen 2B": "D"}
WORKLOAD_MARKERS = {"base": "o", "20": "s", "40": "^", "60": "D", "80": "P"}
MODEL_SHORT = {"Granite 350M": "G350", "Qwen 0.8B": "Q0.8", "Granite 1B": "G1", "Qwen 2B": "Q2"}


def load_rows() -> list[dict]:
    with INPUT.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in tuple(row.items()):
            if key not in ("model", "method", "workload"):
                row[key] = float(value)
    return rows


def save(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        path = OUTPUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(fig)


def pareto_front(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        dominated = any(
            other["composite"] >= row["composite"]
            and other["all_projected_latency_seconds_total"] <= row["all_projected_latency_seconds_total"]
            and (
                other["composite"] > row["composite"]
                or other["all_projected_latency_seconds_total"] < row["all_projected_latency_seconds_total"]
            )
            for other in rows
            if other is not row
        )
        if not dominated:
            result.append(row)
    return sorted(result, key=lambda row: row["composite"])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def plot_model_trajectories(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for ax, model in zip(axes.flat, MODELS):
        model_rows = [row for row in rows if row["model"] == model]
        for method in METHODS:
            points = sorted(
                (row for row in model_rows if row["method"] == method),
                key=lambda row: WORKLOADS.index(row["workload"]),
            )
            x = [row["composite"] for row in points]
            y = [row["all_projected_latency_seconds_total"] / 5 / 60 for row in points]
            ax.plot(x, y, color=COLORS[method], linewidth=1.5, alpha=0.85)
            for row, xv, yv in zip(points, x, y):
                ax.scatter(xv, yv, marker=WORKLOAD_MARKERS[row["workload"]], s=57,
                           color=COLORS[method], edgecolor="#111827", linewidth=0.55, zorder=3)
        ax.set_title(model)
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#e5e7eb", linewidth=0.65)
        ax.set_xlabel("Absolute Composite")
        ax.set_ylabel("Projected latency per scenario (min, log scale)")
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
    method_handles = [Line2D([0], [0], color=COLORS[m], linewidth=2, label=m) for m in METHODS]
    workload_handles = [Line2D([0], [0], color="#111827", marker=WORKLOAD_MARKERS[w], linestyle="none",
                               markersize=6, label="Base" if w == "base" else f"U{w}") for w in WORKLOADS]
    fig.legend(handles=method_handles + workload_handles, loc="upper center", ncol=10,
               frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Accuracy–latency trajectories under increasing update load", fontsize=15,
                 fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, "Right and down are better. Latency uses model-specific prefill/decode throughput assumptions.",
             ha="center", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.09, hspace=0.28, wspace=0.20)
    save(fig, "stress-tradeoff-model-pareto-trajectories")


def plot_scaling(rows: list[dict]) -> None:
    metrics = (
        ("composite", "Composite", 1.0),
        ("all_evaluated_prefill_tokens_total", "Prefill tokens / scenario (K)", 1 / 5000),
        ("all_decode_tokens_total", "Decode tokens / scenario (K)", 1 / 5000),
        ("all_projected_latency_seconds_total", "Projected latency / scenario (min)", 1 / 300),
    )
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True)
    for row_index, model in enumerate(MODELS):
        model_rows = [row for row in rows if row["model"] == model]
        for col_index, (key, title, scale) in enumerate(metrics):
            ax = axes[row_index, col_index]
            for method in METHODS:
                points = sorted(
                    (row for row in model_rows if row["method"] == method),
                    key=lambda row: row["updates_per_scenario"],
                )
                ax.plot(
                    [point["updates_per_scenario"] for point in points],
                    [point[key] * scale for point in points],
                    color=COLORS[method], marker="o", linewidth=1.55, markersize=4.5,
                )
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(model)
            if row_index == len(MODELS) - 1:
                ax.set_xlabel("Updates per scenario")
            ax.set_xticks((13.4, 20, 40, 60, 80), ("Base\n13.4", "20", "40", "60", "80"))
            ax.grid(True, color="#e5e7eb", linewidth=0.65)
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")
    handles = [Line2D([0], [0], color=COLORS[m], marker="o", linewidth=2, markersize=5, label=m) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Stress-test scaling: quality, token cost, and projected latency", fontsize=15,
                 fontweight="bold", y=0.998)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.07, hspace=0.22, wspace=0.18)
    save(fig, "stress-tradeoff-four-model-scaling")


def plot_global_frontiers(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.3), sharex=True, sharey=True)
    for ax, workload in zip(axes, WORKLOADS):
        points = [row for row in rows if row["workload"] == workload]
        frontier = pareto_front(points)
        ax.plot(
            [row["composite"] for row in frontier],
            [row["all_projected_latency_seconds_total"] / 5 / 60 for row in frontier],
            color="#111827", linewidth=1.4, zorder=1,
        )
        for row in points:
            on_frontier = row in frontier
            ax.scatter(
                row["composite"], row["all_projected_latency_seconds_total"] / 5 / 60,
                color=COLORS[row["method"]], marker=MODEL_MARKERS[row["model"]], s=61,
                edgecolor="#111827" if on_frontier else "none", linewidth=0.8,
                alpha=1.0 if on_frontier else 0.22, zorder=3,
            )
        for row in frontier:
            ax.annotate(
                f"{MODEL_SHORT[row['model']]} {row['method']}",
                (row["composite"], row["all_projected_latency_seconds_total"] / 5 / 60),
                xytext=(3, 4), textcoords="offset points", fontsize=6.8, color="#111827",
            )
        ax.set_title("Base (13.4)" if workload == "base" else f"{workload} updates")
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#e5e7eb", linewidth=0.65)
        ax.set_xlabel("Composite")
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
    axes[0].set_ylabel("Projected latency / scenario (min, log scale)")
    method_handles = [Line2D([0], [0], color=COLORS[m], marker="o", linestyle="none", markersize=6, label=m) for m in METHODS]
    model_handles = [Line2D([0], [0], color="#111827", marker=MODEL_MARKERS[m], linestyle="none",
                            markersize=6, label=m) for m in MODELS]
    fig.legend(handles=method_handles + model_handles, loc="upper center", ncol=9,
               frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Global projected on-device Pareto frontiers", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, 0.015, "Frontiers compare all model–method choices within the same workload; right and down are better.",
             ha="center", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.06, right=0.995, top=0.76, bottom=0.20, wspace=0.10)
    save(fig, "stress-tradeoff-global-pareto-frontiers")


def main() -> None:
    setup_style()
    rows = load_rows()
    plot_model_trajectories(rows)
    plot_scaling(rows)
    plot_global_frontiers(rows)


if __name__ == "__main__":
    main()
