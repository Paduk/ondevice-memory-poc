#!/usr/bin/env python3
"""Export Figure 6.2 raw data and appendix-ready tables."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_training.scripts.plot_compaction_interval_ablation_two_panel import (
    METHODS,
    load_rows,
)
from memory_training.scripts.plot_on_device_pareto_test24_partial import (
    ACCURACY_MODELS,
    ACCURACY_WORKLOAD_LABELS,
    ACCURACY_WORKLOADS,
    load_test24_accuracy,
)


RESULTS_DIR = REPO_ROOT / "docs/engineering/results"
RAW_OUTPUT = RESULTS_DIR / "section6-2-compaction-interval-six-model-raw.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "section6-2-compaction-interval-six-model-summary.csv"
APPENDIX_OUTPUT = REPO_ROOT / "docs/engineering/section6-2-compaction-interval-appendix.md"

METHOD_LABELS = {
    "Patch": "Patch",
    "k=2": r"$k=2$",
    "k=5": r"$k=5$",
    "k=10": r"$k=10$",
}


def sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runtime_rows = load_rows()
    runtime_index = {
        (row["model"], row["method"], row["workload"]): row
        for row in runtime_rows
    }
    accuracy = load_test24_accuracy(runtime_index)

    raw_rows: list[dict[str, str | int | float]] = []
    for model in ACCURACY_MODELS:
        for workload, workload_label in zip(
            ACCURACY_WORKLOADS, ACCURACY_WORKLOAD_LABELS
        ):
            patch_runtime = runtime_index[(model, "Patch", workload)]
            patch_composite = accuracy[(model, "Patch", workload)]
            for method in METHODS:
                runtime = runtime_index[(model, method, workload)]
                relative_latency = (
                    100.0
                    if method == "Patch"
                    else float(runtime["relative_latency_percent"])
                )
                method_composite = accuracy[(model, method, workload)]
                latency_reduction = (
                    0.0 if method == "Patch" else 100.0 - relative_latency
                )
                composite_change = (
                    0.0 if method == "Patch" else method_composite - patch_composite
                )
                raw_rows.append(
                    {
                        "model": model,
                        "workload": workload_label,
                        "method": method,
                        "cache_mode": "ON",
                        "accuracy_scenarios": 24,
                        "projected_update_cycle_latency_s": float(
                            runtime["update_cycle_latency_seconds"]
                        ),
                        "patch_projected_update_cycle_latency_s": float(
                            patch_runtime["update_cycle_latency_seconds"]
                        ),
                        "relative_latency_percent": relative_latency,
                        "latency_reduction_percent": latency_reduction,
                        "composite_percent": method_composite,
                        "patch_composite_percent": patch_composite,
                        "composite_change_pp": composite_change,
                        "prefill_tokens_per_s": float(runtime["prefill_tps"]),
                        "decode_tokens_per_s": float(runtime["decode_tps"]),
                    }
                )

    raw_fields = list(raw_rows[0])
    with RAW_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["workload"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, str | int | float]] = []
    for workload in ACCURACY_WORKLOAD_LABELS:
        for method in METHODS:
            rows = grouped[(workload, method)]
            metrics = {
                "relative_latency_percent": [
                    float(row["relative_latency_percent"]) for row in rows
                ],
                "latency_reduction_percent": [
                    float(row["latency_reduction_percent"]) for row in rows
                ],
                "composite_percent": [float(row["composite_percent"]) for row in rows],
                "composite_change_pp": [
                    float(row["composite_change_pp"]) for row in rows
                ],
            }
            summary: dict[str, str | int | float] = {
                "workload": workload,
                "method": method,
                "n_models": len(rows),
            }
            for metric, values in metrics.items():
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_sd"] = sample_sd(values)
            summary_rows.append(summary)

    summary_fields = list(summary_rows[0])
    with SUMMARY_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_index = {
        (str(row["workload"]), str(row["method"])): row for row in summary_rows
    }
    aggregate_rows: list[list[str]] = []
    for workload in ACCURACY_WORKLOAD_LABELS:
        for method in METHODS:
            row = summary_index[(workload, method)]
            aggregate_rows.append(
                [
                    workload,
                    METHOD_LABELS[method],
                    f'{float(row["relative_latency_percent_mean"]):.1f} ± '
                    f'{float(row["relative_latency_percent_sd"]):.1f}',
                    f'{float(row["latency_reduction_percent_mean"]):.1f} ± '
                    f'{float(row["latency_reduction_percent_sd"]):.1f}',
                    f'{float(row["composite_percent_mean"]):.1f} ± '
                    f'{float(row["composite_percent_sd"]):.1f}',
                    f'{float(row["composite_change_pp_mean"]):+.1f} ± '
                    f'{float(row["composite_change_pp_sd"]):.1f}',
                ]
            )

    raw_index = {
        (str(row["model"]), str(row["workload"]), str(row["method"])): row
        for row in raw_rows
    }
    u80_rows: list[list[str]] = []
    for model in ACCURACY_MODELS:
        cells = [model]
        for method in ("k=2", "k=5", "k=10"):
            row = raw_index[(model, "U80", method)]
            cells.append(
                f'{float(row["latency_reduction_percent"]):.1f}% / '
                f'{float(row["composite_change_pp"]):+.1f} pp'
            )
        u80_rows.append(cells)

    latency_rows: list[list[str]] = []
    composite_rows: list[list[str]] = []
    for model in ACCURACY_MODELS:
        for workload in ACCURACY_WORKLOAD_LABELS:
            latency_cells = [model, workload]
            composite_cells = [model, workload]
            for method in METHODS:
                row = raw_index[(model, workload, method)]
                latency_cells.append(f'{float(row["relative_latency_percent"]):.1f}')
                composite_cells.append(f'{float(row["composite_percent"]):.1f}')
            latency_rows.append(latency_cells)
            composite_rows.append(composite_cells)

    appendix = f"""# Appendix X. Compaction-Interval Results

Figure X summarizes the effect of the Delta-v3 compaction interval. This
appendix reports the corresponding aggregate and model-level results. Latency
is normalized within each model and workload to the matched Patch result.
Composite changes are also paired against the matched Patch result before
aggregation. Reported standard deviations therefore describe variation across
the six model configurations, rather than repeated-run uncertainty.

U40--U80 jointly increase update frequency and memory growth without
workload-specific adaptation. They are treated as stress operating points, not
as estimates of expected in-distribution performance.

## Aggregate results

{markdown_table(
    [
        "Workload",
        "Method",
        "Relative latency (%)",
        "Latency reduction (%)",
        "Composite (%)",
        "Composite change (pp)",
    ],
    aggregate_rows,
)}

Values are mean ± standard deviation across six models. Relative latency is
lower-is-better; latency reduction and Composite are higher-is-better.

## U80 model-level operating points

{markdown_table(
    ["Model", "$k=2$", "$k=5$", "$k=10$"],
    u80_rows,
)}

Each cell reports `latency reduction / Composite change` relative to the
matched Patch result. This table exposes the model dependence hidden by the
cross-model mean, particularly for the aggressive $k=10$ setting.

## Full model-level relative latency

{markdown_table(
    ["Model", "Workload", "Patch", "$k=2$", "$k=5$", "$k=10$"],
    latency_rows,
)}

Values are cache-ON update-cycle latency as a percentage of the matched Patch
latency; lower is better.

## Full model-level Composite

{markdown_table(
    ["Model", "Workload", "Patch", "$k=2$", "$k=5$", "$k=10$"],
    composite_rows,
)}

Values are Composite percentages on the same 24 scenarios.

## Data provenance for the current draft

The exact Figure inputs are provided in
`docs/engineering/results/section6-2-compaction-interval-six-model-raw.csv`.
The aggregate values are provided in
`docs/engineering/results/section6-2-compaction-interval-six-model-summary.csv`.
The current latency values are device-calibrated projections obtained by
applying Galaxy Z Fold7 prefill/decode throughput to controlled update-cycle
token traces. They should be relabeled if replaced by direct end-to-end device
measurements in the final manuscript.
"""
    APPENDIX_OUTPUT.write_text(appendix)

    print(RAW_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(APPENDIX_OUTPUT)


if __name__ == "__main__":
    main()
