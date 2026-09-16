#!/usr/bin/env python3
"""Plot the Section 6.1 prefix-cache intervention for Patch and Delta-v3 k=5.

The current artifact is built from the existing controlled-replay traces.  The
cache-OFF cost evaluates the complete logical prompt, while cache-ON evaluates
only the suffix not covered by the reusable prefix.  Absolute device timings
can replace these inputs without changing the figure layout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")
BENCHMARKS = WORKSPACE / "benchmarks"
OUTPUT_CSV = REPO_ROOT / "docs/engineering/results/prefix-cache-reuse-ablation.csv"
OUTPUT_STEM = REPO_ROOT / "docs/engineering/figures/prefix-cache-reuse-ablation"

WORKLOADS = ("base", "40", "60", "80")
WORKLOAD_LABELS = ("Base", "U40", "U60", "U80")
EXPECTED_CYCLES = {"base": 67, "40": 200, "60": 300, "80": 400}
METHODS = {"Patch": "patch", "Delta-v3": "delta_v3_compact_k5"}

MODELS = {
    "Granite 350M": {
        "prefill_tps": 137.7,
        "decode_tps": 51.8,
        "natural": "granite4-350m-natural-s86-s90-cache-once-20260907-v1",
        "stress": "granite4-350m-stress-test-once-20260905-v1",
        "stress_prefix": "update",
    },
    "Qwen 0.8B": {
        "prefill_tps": 68.0,
        "decode_tps": 27.6,
        "natural": "qwen35-0.8b-natural-s86-s90-cache-once-20260905-v1",
        "stress": "qwen35-0.8b-stress-test-once-20260904-v1",
        "stress_prefix": "update",
    },
    "Llama 1B": {
        "prefill_tps": 35.7,
        "decode_tps": 19.8,
        "stress": "llama32-1b-stress-test-once-20260908-v1",
        "stress_prefix": "",
    },
    "Granite 1B": {
        "prefill_tps": 25.7,
        "decode_tps": 16.4,
        "natural": "granite4-1b-natural-s86-s90-cache-once-20260905-v1",
        "stress": "granite4-1b-stress-test-once-20260904-v1",
        "stress_prefix": "update",
    },
    "Qwen 2B": {
        "prefill_tps": 24.7,
        "decode_tps": 14.5,
        "natural": "qwen35-2b-natural-s86-s90-cache-once-20260907-v1",
        "stress": "qwen35-2b-stress-test-once-20260905-v1",
        "stress_prefix": "update",
    },
    "Llama 3B": {
        "prefill_tps": 11.1,
        "decode_tps": 7.3,
        "stress": "llama32-3b-stress-test-once-20260908-v1",
        "stress_prefix": "",
    },
}

PATCH_COLOR = "#6B7280"
DELTA_COLOR = "#0072B2"


def turns_path(config: dict, method: str, workload: str) -> Path:
    if workload == "base" and "natural" in config:
        return BENCHMARKS / config["natural"] / "cache" / method / "on/turns.jsonl"
    workload_dir = f"{config['stress_prefix']}{workload}"
    return (
        BENCHMARKS
        / config["stress"]
        / "cache"
        / workload_dir
        / method
        / "on/turns.jsonl"
    )


def update_cycles(path: Path, *, prefill_tps: float, decode_tps: float) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    cycles = [
        (previous, current)
        for previous, current in zip(records, records[1:])
        if previous["scenario_index"] == current["scenario_index"]
        and previous["repetition"] == current["repetition"]
        and previous["gold_decision"] == "UPDATE"
    ]
    logical = [float(current["logical_prompt_tokens"]) for _, current in cycles]
    evaluated = [float(current["evaluated_prefill_tokens"]) for _, current in cycles]
    decoded = [float(previous["decode_tokens"]) for previous, _ in cycles]
    cache_off = [
        decode / decode_tps + prompt / prefill_tps
        for decode, prompt in zip(decoded, logical)
    ]
    cache_on = [
        decode / decode_tps + prompt / prefill_tps
        for decode, prompt in zip(decoded, evaluated)
    ]
    return {
        "cycles": len(cycles),
        "cache_off_seconds": mean(cache_off),
        "cache_on_seconds": mean(cache_on),
        "logical_prefill_tokens": mean(logical),
        "evaluated_prefill_tokens": mean(evaluated),
        "prefix_reuse_percent": 100.0 * (1.0 - sum(evaluated) / sum(logical)),
    }


def build_rows() -> list[dict]:
    rows = []
    for model, config in MODELS.items():
        for workload in WORKLOADS:
            metrics = {
                method: update_cycles(
                    turns_path(config, slug, workload),
                    prefill_tps=float(config["prefill_tps"]),
                    decode_tps=float(config["decode_tps"]),
                )
                for method, slug in METHODS.items()
            }
            cycle_counts = {value["cycles"] for value in metrics.values()}
            if cycle_counts != {EXPECTED_CYCLES[workload]}:
                raise ValueError(
                    f"Unexpected cycle count for {model}/{workload}: {cycle_counts}"
                )
            patch = metrics["Patch"]
            delta = metrics["Delta-v3"]
            reference = patch["cache_off_seconds"]
            rows.append(
                {
                    "model": model,
                    "workload": workload,
                    "cycles": patch["cycles"],
                    "patch_cache_off_seconds": patch["cache_off_seconds"],
                    "delta_cache_off_seconds": delta["cache_off_seconds"],
                    "patch_cache_on_seconds": patch["cache_on_seconds"],
                    "delta_cache_on_seconds": delta["cache_on_seconds"],
                    "patch_cache_off_relative": 100.0,
                    "delta_cache_off_relative": 100.0
                    * delta["cache_off_seconds"]
                    / reference,
                    "patch_cache_on_relative": 100.0
                    * patch["cache_on_seconds"]
                    / reference,
                    "delta_cache_on_relative": 100.0
                    * delta["cache_on_seconds"]
                    / reference,
                    "patch_prefix_reuse_percent": patch["prefix_reuse_percent"],
                    "delta_prefix_reuse_percent": delta["prefix_reuse_percent"],
                    "patch_logical_prefill_tokens": patch["logical_prefill_tokens"],
                    "delta_logical_prefill_tokens": delta["logical_prefill_tokens"],
                    "patch_evaluated_prefill_tokens": patch[
                        "evaluated_prefill_tokens"
                    ],
                    "delta_evaluated_prefill_tokens": delta[
                        "evaluated_prefill_tokens"
                    ],
                }
            )
    return rows


def write_rows(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT_CSV)


def values(rows: list[dict], field: str, workload: str) -> list[float]:
    indexed = {
        (row["model"], row["workload"]): float(row[field]) for row in rows
    }
    return [indexed[(model, workload)] for model in MODELS]


def mean_and_sd(rows: list[dict], field: str) -> tuple[list[float], list[float]]:
    grouped = [values(rows, field, workload) for workload in WORKLOADS]
    return [mean(group) for group in grouped], [stdev(group) for group in grouped]


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA3AF")
    axis.spines["bottom"].set_color("#9CA3AF")


def plot(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (latency_axis, reuse_axis) = plt.subplots(
        1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": (1.12, 1.0)}
    )
    x = list(range(len(WORKLOADS)))

    latency_series = (
        ("Patch, cache OFF", "patch_cache_off_relative", PATCH_COLOR, "o", "--", "white"),
        ("Delta-v3, cache OFF", "delta_cache_off_relative", DELTA_COLOR, "s", "--", "white"),
        ("Patch, cache ON", "patch_cache_on_relative", PATCH_COLOR, "o", "-", PATCH_COLOR),
        ("Delta-v3, cache ON", "delta_cache_on_relative", DELTA_COLOR, "s", "-", DELTA_COLOR),
    )
    for label, field, color, marker, linestyle, facecolor in latency_series:
        averages, spreads = mean_and_sd(rows, field)
        latency_axis.errorbar(
            x,
            averages,
            yerr=spreads,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=6.0,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=1.1,
            capsize=2.7,
            elinewidth=0.9,
            alpha=0.98,
            label=label,
            zorder=3,
        )
    latency_axis.axhline(100, color="#111827", linewidth=0.8, alpha=0.35)
    latency_axis.set_xticks(x, WORKLOAD_LABELS)
    latency_axis.set_ylim(20, 112)
    latency_axis.set_ylabel("Normalized update-cycle latency (%)\n(Patch, cache OFF = 100)")
    latency_axis.set_xlabel("Update workload")
    latency_axis.set_title("(a) Cache intervention")
    latency_axis.legend(frameon=False, ncol=2, loc="lower left")
    latency_axis.text(
        0.02,
        0.97,
        "Lower is better",
        transform=latency_axis.transAxes,
        ha="left",
        va="top",
        color="#4B5563",
        fontsize=8.2,
    )
    style_axis(latency_axis)

    reuse_series = (
        ("Patch", "patch_prefix_reuse_percent", PATCH_COLOR, "o"),
        ("Delta-v3 (k=5)", "delta_prefix_reuse_percent", DELTA_COLOR, "s"),
    )
    for label, field, color, marker in reuse_series:
        averages, spreads = mean_and_sd(rows, field)
        reuse_axis.errorbar(
            x,
            averages,
            yerr=spreads,
            color=color,
            linewidth=2.1,
            marker=marker,
            markersize=6.2,
            capsize=2.7,
            elinewidth=0.9,
            label=label,
            zorder=3,
        )
    reuse_axis.set_xticks(x, WORKLOAD_LABELS)
    reuse_axis.set_ylim(0, 88)
    reuse_axis.set_ylabel("Post-UPDATE prefix reuse (%)")
    reuse_axis.set_xlabel("Update workload")
    reuse_axis.set_title("(b) Reused prefix after UPDATE")
    reuse_axis.legend(frameon=False, loc="center right")
    reuse_axis.text(
        0.02,
        0.97,
        "Higher is better",
        transform=reuse_axis.transAxes,
        ha="left",
        va="top",
        color="#4B5563",
        fontsize=8.2,
    )
    style_axis(reuse_axis)

    figure.text(
        0.5,
        0.012,
        "Mean ± SD across six model configurations; Delta-v3 uses k=5.",
        ha="center",
        color="#4B5563",
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.19, wspace=0.28)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
        path = OUTPUT_STEM.with_suffix(f".{suffix}")
        figure.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(figure)


def main() -> None:
    rows = build_rows()
    write_rows(rows)
    plot(rows)


if __name__ == "__main__":
    main()
