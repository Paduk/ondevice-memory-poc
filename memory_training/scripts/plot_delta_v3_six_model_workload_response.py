#!/usr/bin/env python3
"""Plot six-model quality vs Patch-relative update-cycle latency.

The y-axis is one fixed 24-scenario Test Composite per trained method. The
x-axis changes with the stress workload and is normalized to Patch under the
same model and workload. Consequently, each Delta line is a workload-response
trajectory, not a Pareto frontier across workloads.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch as LegendPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_training.scripts.aggregate_stress_tradeoff import aggregate_cache


WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")
FOUR_MODEL_INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.csv"
DATA_OUTPUT = REPO_ROOT / "docs/engineering/results/delta-v3-six-model-workload-response.csv"
FIGURE_DIR = REPO_ROOT / "docs/engineering/figures"

MODELS = (
    "Granite 350M",
    "Qwen 0.8B",
    "Llama 1B",
    "Granite 1B",
    "Qwen 2B",
    "Llama 3B",
)
METHODS = ("Patch", "k=2", "k=5", "k=10")
WORKLOADS = ("base", "20", "40", "60", "80")
MAIN_WORKLOADS = ("base", "40", "80")
WORKLOAD_LABELS = {"base": "Base", "20": "U20", "40": "U40", "60": "U60", "80": "U80"}
EPOCHS = {
    "Granite 350M": {"Patch": 4, "k=2": 3, "k=5": 3, "k=10": 4},
    "Qwen 0.8B": {"Patch": 3, "k=2": 3, "k=5": 3, "k=10": 3},
    "Llama 1B": {"Patch": 3, "k=2": 3, "k=5": 4, "k=10": 2},
    "Granite 1B": {"Patch": 3, "k=2": 3, "k=5": 4, "k=10": 4},
    "Qwen 2B": {"Patch": 3, "k=2": 3, "k=5": 4, "k=10": 4},
    "Llama 3B": {"Patch": 4, "k=2": 3, "k=5": 4, "k=10": 4},
}
THROUGHPUT = {
    "Granite 350M": (137.7, 51.8),
    "Qwen 0.8B": (68.0, 27.6),
    "Llama 1B": (35.7, 19.8),
    "Granite 1B": (25.7, 16.4),
    "Qwen 2B": (24.7, 14.5),
    "Llama 3B": (11.1, 7.3),
}
METHOD_COLORS = {
    "Patch": "#4B5563",
    "k=2": "#0072B2",
    "k=5": "#E69F00",
    "k=10": "#009E73",
}
WORKLOAD_MARKERS = {"base": "o", "20": "s", "40": "^", "60": "D", "80": "P"}
METHOD_LABEL_OFFSETS = {
    "Granite 350M": {"k=2": (4, 7), "k=5": (4, 7), "k=10": (4, -9)},
    "Qwen 0.8B": {"k=2": (4, 8), "k=5": (4, -9), "k=10": (4, 9)},
    "Llama 1B": {"k=2": (4, 8), "k=5": (4, 0), "k=10": (4, -8)},
    "Granite 1B": {"k=2": (4, 8), "k=5": (4, 0), "k=10": (4, -8)},
    "Qwen 2B": {"k=2": (4, -8), "k=5": (4, -9), "k=10": (4, 9)},
    "Llama 3B": {"k=2": (4, 8), "k=5": (4, 0), "k=10": (4, -8)},
}

LLAMA_RUNS = {
    "Patch": (
        "llama3.2-1b-patch-multitask-noop5-e4-b8-trainseed45-evalfixed-noop5-r2",
        "eval-fixed-test-best-epoch-03",
        "patch",
    ),
    "k=2": (
        "llama3.2-1b-delta_v3_compact_k2-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1",
        "eval-fixed-test-best-epoch-03",
        "delta_v3_compact_k2",
    ),
    "k=5": (
        "llama3.2-1b-delta_v3_compact_k5-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r2",
        "eval-fixed-test-best-epoch-04",
        "delta_v3_compact_k5",
    ),
    "k=10": (
        "llama3.2-1b-delta_v3_compact_k10-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1",
        "eval-fixed-test-best-epoch-02",
        "delta_v3_compact_k10",
    ),
}

LLAMA_3B_RUNS = {
    "Patch": (
        "llama3.2-3b-patch-multitask-noop5-e4-b4-trainseed45-evalfixed-noop5-r1",
        "eval-fixed-test-best-epoch-04",
        "patch",
    ),
    "k=2": (
        "llama3.2-3b-delta_v3_compact_k2-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1",
        "eval-fixed-test-best-epoch-03",
        "delta_v3_compact_k2",
    ),
    "k=5": (
        "llama3.2-3b-delta_v3_compact_k5-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1",
        "eval-fixed-test-best-epoch-04",
        "delta_v3_compact_k5",
    ),
    "k=10": (
        "llama3.2-3b-delta_v3_compact_k10-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1",
        "eval-fixed-test-best-epoch-04",
        "delta_v3_compact_k10",
    ),
}


def composite(summary: dict) -> float:
    return 100.0 * (
        0.60 * float(summary["closed_loop_quiz"]["esm"])
        + 0.25 * float(summary["memory"]["final_state_f1"])
        + 0.15 * float(summary["memory"]["update_f1"])
    )


def load_four_model_rows() -> list[dict]:
    rows = []
    with FOUR_MODEL_INPUT.open() as handle:
        for source in csv.DictReader(handle):
            if source["method"] not in METHODS:
                continue
            rows.append(
                {
                    "model": source["model"],
                    "method": source["method"],
                    "epoch": EPOCHS[source["model"]][source["method"]],
                    "workload": source["workload"],
                    "test_composite": float(source["full_test_composite"]),
                    "update_cycle_latency_seconds": float(source["update_cycle_cache_on_seconds_mean"]),
                    "prefill_tps": float(source["prefill_tps"]),
                    "decode_tps": float(source["decode_tps"]),
                }
            )
    return rows


def load_llama_rows(model: str, stress_name: str, runs: dict) -> list[dict]:
    stress_root = WORKSPACE / "benchmarks" / stress_name / "cache"
    prefill_tps, decode_tps = THROUGHPUT[model]
    rows = []
    for method, (run_name, test_name, method_slug) in runs.items():
        summary_path = WORKSPACE / "runs" / run_name / test_name / "summary.json"
        summary = json.loads(summary_path.read_text())
        if not summary.get("complete") or len(summary.get("completed_scenarios", [])) != 24:
            raise ValueError(f"Incomplete {model} Test result: {summary_path}")
        test_composite = composite(summary)
        for workload in WORKLOADS:
            turns_path = stress_root / workload / method_slug / "on/turns.jsonl"
            update_cycle = aggregate_cache(turns_path, prefill_tps, decode_tps)["update_cycle"]
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "epoch": EPOCHS[model][method],
                    "workload": workload,
                    "test_composite": test_composite,
                    "update_cycle_latency_seconds": update_cycle["cache_on_seconds_mean"],
                    "prefill_tps": prefill_tps,
                    "decode_tps": decode_tps,
                }
            )
    return rows


def build_rows() -> list[dict]:
    rows = (
        load_four_model_rows()
        + load_llama_rows(
            "Llama 1B",
            "llama32-1b-stress-test-once-20260908-v1",
            LLAMA_RUNS,
        )
        + load_llama_rows(
            "Llama 3B",
            "llama32-3b-stress-test-once-20260908-v1",
            LLAMA_3B_RUNS,
        )
    )
    index = {(row["model"], row["method"], row["workload"]): row for row in rows}
    expected = len(MODELS) * len(METHODS) * len(WORKLOADS)
    if len(index) != expected:
        raise ValueError(f"Expected {expected} unique operating points, got {len(index)}")

    for row in rows:
        patch = index[(row["model"], "Patch", row["workload"])]["update_cycle_latency_seconds"]
        row["relative_latency_percent"] = 100.0 * row["update_cycle_latency_seconds"] / patch

    rows.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            METHODS.index(row["method"]),
            WORKLOADS.index(row["workload"]),
        )
    )
    return rows


def write_rows(rows: list[dict]) -> None:
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DATA_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(DATA_OUTPUT)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = (FIGURE_DIR / stem).with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(fig)


def add_figure_legends(fig: plt.Figure, workloads: tuple[str, ...]) -> None:
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2.2, label=method)
        for method in METHODS
    ]
    workload_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=WORKLOAD_MARKERS[workload],
            markersize=7,
            markerfacecolor="#FFFFFF",
            markeredgecolor="#111827",
            label=WORKLOAD_LABELS[workload],
        )
        for workload in workloads
    ]
    budget_handles = [
        LegendPatch(facecolor="#DBEAFE", edgecolor="none", label="≥ Patch"),
        LegendPatch(facecolor="#BBF7D0", edgecolor="none", label="≤1 pp loss"),
        LegendPatch(facecolor="#DCFCE7", edgecolor="none", label="≤3 pp loss"),
    ]
    legend1 = fig.legend(
        handles=method_handles,
        title="Method",
        loc="lower center",
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.20, 0.002),
    )
    fig.add_artist(legend1)
    legend2 = fig.legend(
        handles=workload_handles,
        title="Update load",
        loc="lower center",
        frameon=False,
        ncol=len(workloads),
        bbox_to_anchor=(0.53, 0.002),
    )
    fig.add_artist(legend2)
    fig.legend(
        handles=budget_handles,
        title="Accuracy budget",
        loc="lower center",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.83, 0.002),
    )


def plot(rows: list[dict], workloads: tuple[str, ...], stem: str, subtitle: str) -> None:
    selected = [row for row in rows if row["workload"] in workloads]
    min_relative = min(row["relative_latency_percent"] for row in selected)
    x_min = max(0.0, 5.0 * math.floor((min_relative - 6.0) / 5.0))
    x_max = 104.0
    patch_bands = [
        row["test_composite"] - 3.0 for row in selected if row["method"] == "Patch"
    ]
    y_min = math.floor(min(min(row["test_composite"] for row in selected), min(patch_bands)) - 0.5)
    y_max = math.ceil(max(row["test_composite"] for row in selected) + 0.8)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.4), sharex=True, sharey=True)
    axes_flat = axes.flat
    for panel_index, model in enumerate(MODELS):
        ax = axes_flat[panel_index]
        model_rows = [row for row in selected if row["model"] == model]
        index = {(row["method"], row["workload"]): row for row in model_rows}
        patch_score = index[("Patch", workloads[0])]["test_composite"]

        ax.axhspan(patch_score, y_max, color="#DBEAFE", alpha=0.65, zorder=0)
        ax.axhspan(patch_score - 1.0, patch_score, color="#BBF7D0", alpha=0.72, zorder=0)
        ax.axhspan(patch_score - 3.0, patch_score - 1.0, color="#DCFCE7", alpha=0.72, zorder=0)
        ax.axhline(patch_score, color="#4B5563", linestyle=":", linewidth=1.0, zorder=1)
        ax.axvline(100.0, color="#6B7280", linestyle="--", linewidth=0.9, alpha=0.8, zorder=1)

        ax.scatter(
            100.0,
            patch_score,
            marker="*",
            s=155,
            color=METHOD_COLORS["Patch"],
            edgecolor="#111827",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            "Patch",
            (100.0, patch_score),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=METHOD_COLORS["Patch"],
            fontweight="bold",
        )

        for method in METHODS[1:]:
            points = [index[(method, workload)] for workload in workloads]
            xs = [point["relative_latency_percent"] for point in points]
            y = points[0]["test_composite"]
            ax.annotate(
                "",
                xy=(xs[-1], y),
                xytext=(xs[0], y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": METHOD_COLORS[method],
                    "linewidth": 1.5,
                    "alpha": 0.78,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=2,
            )
            for point in points:
                ax.scatter(
                    point["relative_latency_percent"],
                    point["test_composite"],
                    marker=WORKLOAD_MARKERS[point["workload"]],
                    s=68,
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
            dx, dy = METHOD_LABEL_OFFSETS[model][method]
            label = ax.annotate(
                method,
                (xs[-1], y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.2,
                color=METHOD_COLORS[method],
                fontweight="bold",
                zorder=6,
            )
            label.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white")])

        ax.set_title(model, pad=7, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([tick for tick in (25, 40, 60, 80, 100) if x_min <= tick <= x_max])
        ax.grid(True, color="#D1D5DB", linewidth=0.55, alpha=0.75, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#9CA3AF")
            spine.set_linewidth(0.8)
        if panel_index in (0, 3):
            ax.set_ylabel("Test Composite · Higher ↑")
        if panel_index >= 3:
            ax.set_xlabel("← Faster · Latency vs. Patch (%)")

    add_figure_legends(fig, workloads)
    fig.suptitle(
        "Quality–latency trade-off under update load",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        subtitle,
        ha="center",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.875, bottom=0.17, hspace=0.27, wspace=0.16)
    save(fig, stem)


def main() -> None:
    setup_style()
    rows = build_rows()
    write_rows(rows)
    plot(
        rows,
        MAIN_WORKLOADS,
        "delta-v3-six-model-workload-response-main",
        "Base · U40 · U80",
    )
    plot(
        rows,
        WORKLOADS,
        "delta-v3-six-model-workload-response-all-workloads",
        "Base · U20 · U40 · U60 · U80",
    )


if __name__ == "__main__":
    main()
