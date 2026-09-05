#!/usr/bin/env python3
"""Plot the normalized latency-quality Pareto frontier for the stress Test."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT_DIR = Path("/home/hj153lee/PalmClaw/docs/engineering/figures")

DATA = {
    "Qwen3.5 0.8B": {
        20: {"Patch": (57.29, 0.00), "k=2": (61.11, 34.28), "k=5": (53.17, 35.42), "k=10": (57.00, 5.94)},
        40: {"Patch": (64.71, 0.00), "k=2": (62.36, 40.95), "k=5": (54.89, 53.56), "k=10": (59.36, 38.81)},
        60: {"Patch": (66.54, 0.00), "k=2": (58.25, 43.57), "k=5": (49.81, 61.55), "k=10": (61.96, 54.39)},
        80: {"Patch": (63.20, 0.00), "k=2": (66.94, 45.05), "k=5": (48.52, 65.70), "k=10": (58.13, 62.35)},
    },
    "Granite 4 1B": {
        20: {"Patch": (59.66, 0.00), "k=2": (50.88, 16.79), "k=5": (58.29, 31.78), "k=10": (48.52, 39.68)},
        40: {"Patch": (64.51, 0.00), "k=2": (53.92, 23.26), "k=5": (54.08, 49.36), "k=10": (51.87, 59.92)},
        60: {"Patch": (68.19, 0.00), "k=2": (51.16, 25.99), "k=5": (63.65, 56.85), "k=10": (41.52, 68.32)},
        80: {"Patch": (68.78, 0.00), "k=2": (55.46, 27.63), "k=5": (58.63, 60.61), "k=10": (39.62, 72.61)},
    },
}

COLORS = {
    "Patch": "#6b7280",
    "k=2": "#0072B2",
    "k=5": "#E69F00",
    "k=10": "#009E73",
}
MARKERS = {"Patch": "s", "k=2": "o", "k=5": "^", "k=10": "D"}


def pareto_front(points):
    """Return non-dominated labels: higher x and higher y are better."""
    frontier = []
    for label, (x_value, y_value) in points.items():
        dominated = any(
            other_x >= x_value
            and other_y >= y_value
            and (other_x > x_value or other_y > y_value)
            for other_label, (other_x, other_y) in points.items()
            if other_label != label
        )
        if not dominated:
            frontier.append(label)
    return sorted(frontier, key=lambda label: points[label][0])


def main():
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
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True, sharey=True)

    for row, (model, by_update) in enumerate(DATA.items()):
        for col, update_count in enumerate((20, 40, 60, 80)):
            ax = axes[row, col]
            points = by_update[update_count]
            frontier = pareto_front(points)

            frontier_xy = [points[label] for label in frontier]
            ax.plot(
                [value[0] for value in frontier_xy],
                [value[1] for value in frontier_xy],
                color="#111827",
                linewidth=1.5,
                zorder=2,
            )

            for label, (x_value, y_value) in points.items():
                is_frontier = label in frontier
                ax.scatter(
                    x_value,
                    y_value,
                    s=74 if is_frontier else 60,
                    marker=MARKERS[label],
                    color=COLORS[label],
                    edgecolor="#111827" if is_frontier else "none",
                    linewidth=1.2,
                    alpha=1.0 if is_frontier else 0.34,
                    zorder=3,
                )
                x_offset = 4 if x_value < 65 else -4
                alignment = "left" if x_value < 65 else "right"
                ax.annotate(
                    label,
                    (x_value, y_value),
                    xytext=(x_offset, 5 if y_value < 67 else -12),
                    textcoords="offset points",
                    ha=alignment,
                    va="bottom",
                    fontsize=8.5,
                    color="#111827",
                    alpha=1.0 if is_frontier else 0.55,
                )

            ax.axhline(0, color="#d1d5db", linewidth=0.8, linestyle=":", zorder=0)
            ax.grid(True, color="#e5e7eb", linewidth=0.65, alpha=0.8)
            ax.set_xlim(37, 71)
            ax.set_ylim(-5, 78)
            ax.set_xticks((40, 50, 60, 70))
            ax.set_yticks((0, 20, 40, 60))
            ax.set_title(f"{update_count} updates")
            if col == 0:
                ax.set_ylabel(f"{model}\nPrefill token saving vs Patch (%)")
            if row == 1:
                ax.set_xlabel("Absolute Composite")
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")
                spine.set_linewidth(0.75)

    axes[0, 0].annotate(
        "better",
        xy=(69, 74),
        xytext=(62, 51),
        arrowprops={"arrowstyle": "->", "color": "#374151", "linewidth": 1.0},
        ha="center",
        color="#374151",
        fontsize=9,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[label],
            color="none",
            markerfacecolor=COLORS[label],
            markeredgecolor="#111827",
            markersize=7.5,
            label=label,
        )
        for label in ("Patch", "k=2", "k=5", "k=10")
    ]
    handles.append(Line2D([0], [0], color="#111827", linewidth=1.5, label="Pareto frontier"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(
        "Delta-v3 stress-test Pareto frontiers",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Higher Composite and greater prefill-token saving are better. Dominated points are faded.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.105, hspace=0.28, wspace=0.14)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "delta-v3-stress-pareto-qwen-granite.png"
    pdf_path = OUTPUT_DIR / "delta-v3-stress-pareto-qwen-granite.pdf"
    fig.savefig(png_path, dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
