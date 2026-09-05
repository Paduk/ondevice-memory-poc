#!/usr/bin/env python3
"""Create three complementary paper-figure views of the current stress results."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_delta_v3_stress_pareto import COLORS, DATA, MARKERS, pareto_front


OUTPUT_DIR = Path("/home/hj153lee/PalmClaw/docs/engineering/figures")
MODELS = tuple(DATA)
UPDATES = (20, 40, 60, 80)
METHODS = ("Patch", "k=2", "k=5", "k=10")


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def save(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    print(png)
    print(pdf)


def plot_overlay():
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), sharex=True, sharey=True)
    model_styles = {
        "Qwen3.5 0.8B": {"linestyle": "-", "filled": True},
        "Granite 4 1B": {"linestyle": "--", "filled": False},
    }
    for ax, update in zip(axes, UPDATES):
        for model in MODELS:
            points = DATA[model][update]
            frontier = pareto_front(points)
            frontier_xy = [points[label] for label in frontier]
            ax.plot(
                [point[0] for point in frontier_xy],
                [point[1] for point in frontier_xy],
                color="#111827",
                linestyle=model_styles[model]["linestyle"],
                linewidth=1.4,
                zorder=1,
            )
            for method, (composite, saving) in points.items():
                on_frontier = method in frontier
                face = COLORS[method] if model_styles[model]["filled"] else "white"
                ax.scatter(
                    composite,
                    saving,
                    s=67,
                    marker=MARKERS[method],
                    facecolor=face,
                    edgecolor=COLORS[method],
                    linewidth=1.8,
                    alpha=1.0 if on_frontier else 0.25,
                    zorder=3,
                )
        ax.set_title(f"{update} updates")
        ax.set_xlim(37, 71)
        ax.set_ylim(-5, 78)
        ax.set_xticks((40, 50, 60, 70))
        ax.set_yticks((0, 20, 40, 60))
        ax.grid(True, color="#e5e7eb", linewidth=0.65)
        ax.axhline(0, color="#9ca3af", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Absolute Composite")
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
    axes[0].set_ylabel("Prefill token saving vs Patch (%)")

    method_handles = [
        Line2D([0], [0], marker=MARKERS[m], color="none", markerfacecolor=COLORS[m],
               markeredgecolor=COLORS[m], markersize=7, label=m)
        for m in METHODS
    ]
    model_handles = [
        Line2D([0], [0], color="#111827", linestyle="-", marker="o",
               markerfacecolor="#111827", markersize=6, label="Qwen3.5 0.8B"),
        Line2D([0], [0], color="#111827", linestyle="--", marker="o",
               markerfacecolor="white", markersize=6, label="Granite 4 1B"),
    ]
    fig.legend(handles=method_handles + model_handles, loc="upper center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Cross-model Pareto view", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, 0.015, "Top-right is better; faded points are dominated within the same model and workload.",
             ha="center", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.75, bottom=0.19, wspace=0.12)
    save(fig, "delta-v3-pareto-cross-model-overlay")


def best_under_budget(points, budget):
    patch_composite = points["Patch"][0]
    eligible = [
        (method, composite, saving)
        for method, (composite, saving) in points.items()
        if composite >= patch_composite - budget
    ]
    return max(eligible, key=lambda row: (row[2], row[1]))


def plot_accuracy_budget():
    budgets = (0, 5, 10, 15)
    budget_colors = {0: "#0072B2", 5: "#009E73", 10: "#E69F00", 15: "#CC79A7"}
    offsets = {0: -0.18, 5: -0.06, 10: 0.06, 15: 0.18}
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1), sharey=True)
    for ax, update in zip(axes, UPDATES):
        for model_index, model in enumerate(MODELS):
            patch_selected = False
            for budget in budgets:
                method, _, saving = best_under_budget(DATA[model][update], budget)
                patch_selected = patch_selected or method == "Patch"
                x_value = model_index + offsets[budget]
                ax.vlines(x_value, 0, saving, color=budget_colors[budget], linewidth=1.0, alpha=0.6)
                ax.scatter(x_value, saving, s=58, color=budget_colors[budget], edgecolor="#111827",
                           linewidth=0.7, zorder=3)
                if method != "Patch":
                    ax.annotate(method.replace("=", ""), (x_value, saving), xytext=(0, 5),
                                textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
            if patch_selected:
                ax.annotate("Patch", (model_index, 0), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)
        ax.set_title(f"{update} updates")
        ax.set_xticks((0, 1), ("Qwen\n0.8B", "Granite\n1B"))
        ax.set_xlim(-0.43, 1.43)
        ax.set_ylim(-4, 78)
        ax.set_yticks((0, 20, 40, 60))
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.65)
        for spine in ax.spines.values():
            spine.set_color("#9ca3af")
    axes[0].set_ylabel("Maximum prefill saving (%)")
    handles = [Line2D([0], [0], marker="o", color=budget_colors[b], markersize=6,
                      label=f"Composite loss ≤ {b} pp") for b in budgets]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Accuracy-budget operating points", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, 0.015, "For each model, choose the largest saving that stays inside the allowed Composite loss.",
             ha="center", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.75, bottom=0.18, wspace=0.14)
    save(fig, "delta-v3-accuracy-budget-by-model")


def plot_update_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col", sharey="row")
    for col, model in enumerate(MODELS):
        for method in METHODS:
            composites = [DATA[model][u][method][0] for u in UPDATES]
            savings = [DATA[model][u][method][1] for u in UPDATES]
            axes[0, col].plot(UPDATES, composites, marker=MARKERS[method], color=COLORS[method],
                              linewidth=1.7, markersize=6, label=method)
            axes[1, col].plot(UPDATES, savings, marker=MARKERS[method], color=COLORS[method],
                              linewidth=1.7, markersize=6, label=method)
        axes[0, col].set_title(model)
        axes[1, col].set_xlabel("Updates per scenario")
        axes[1, col].set_xticks(UPDATES)
        for row in (0, 1):
            axes[row, col].grid(True, color="#e5e7eb", linewidth=0.65)
            for spine in axes[row, col].spines.values():
                spine.set_color("#9ca3af")
    axes[0, 0].set_ylabel("Absolute Composite")
    axes[1, 0].set_ylabel("Prefill token saving vs Patch (%)")
    axes[0, 0].set_ylim(37, 72)
    axes[1, 0].set_ylim(-5, 78)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("Update-load response", fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.09, hspace=0.15, wspace=0.12)
    save(fig, "delta-v3-update-scaling-quality-efficiency")


def main():
    setup_style()
    plot_overlay()
    plot_accuracy_budget()
    plot_update_scaling()


if __name__ == "__main__":
    main()
