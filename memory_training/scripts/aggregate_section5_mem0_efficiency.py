#!/usr/bin/env python3
"""Combine Section 5 cache traces with Mem0 one- and two-stage costs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .aggregate_section5_update_cycle_efficiency import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_DIR,
    MODEL_SOURCES,
    aggregate_cycles,
    aggregate_turns,
    applied_update_cycles,
    build_rows,
    read_jsonl,
    verify_reference,
)

DEFAULT_MEM0_CAMPAIGN = (
    DEFAULT_BENCHMARK_ROOT / "mem0-efficiency-controlled-s86-s90-20260918-v1"
)
MODEL_SLUGS = {
    "Granite 350M": "granite4-350m",
    "Qwen 0.8B": "qwen35-0.8b",
    "Granite 1B": "granite4-1b",
    "Llama 3.2 1B": "llama3.2-1b",
    "Llama 3.2 3B": "llama3.2-3b",
    "Qwen 2B": "qwen35-2b",
}
MEM0_METHODS = (
    ("Mem0 One-pass", "mem0_one_pass"),
    ("Mem0 Two-stage", "mem0_two_stage"),
)
METHOD_ORDER = {
    method: index
    for index, method in enumerate(
        (
            "Summary",
            "Patch",
            "Delta-v3 (k=5)",
            "Mem0 One-pass",
            "Mem0 Two-stage",
        )
    )
}


def build_combined_rows(
    *,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    mem0_campaign: Path = DEFAULT_MEM0_CAMPAIGN,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = build_rows(benchmark_root)
    verify_reference(rows)
    missing: list[str] = []
    summary_by_model = {
        str(row["model"]): row for row in rows if row["method"] == "Summary"
    }
    for source in MODEL_SOURCES:
        slug = MODEL_SLUGS[source.model]
        for method, method_slug in MEM0_METHODS:
            turns_path = mem0_campaign / slug / "cache" / method_slug / "on/turns.jsonl"
            if not turns_path.is_file():
                missing.append(str(turns_path))
                continue
            records = read_jsonl(turns_path)
            if method_slug == "mem0_two_stage":
                _validate_two_stage_records(records, turns_path)
            cycles = aggregate_cycles(
                applied_update_cycles(records),
                prefill_tokens_per_second=source.prefill_tokens_per_second,
                decode_tokens_per_second=source.decode_tokens_per_second,
            )
            all_turns = aggregate_turns(
                records,
                prefill_tokens_per_second=source.prefill_tokens_per_second,
                decode_tokens_per_second=source.decode_tokens_per_second,
            )
            summary = summary_by_model[source.model]
            rows.append(
                {
                    "model": source.model,
                    "method": method,
                    "method_slug": method_slug,
                    "trace_path": str(turns_path),
                    "prefill_tokens_per_second": source.prefill_tokens_per_second,
                    "decode_tokens_per_second": source.decode_tokens_per_second,
                    **cycles,
                    **all_turns,
                    "relative_latency_summary_100": (
                        100.0
                        * float(cycles["projected_cache_on_latency_seconds_mean"])
                        / float(summary["projected_cache_on_latency_seconds_mean"])
                    ),
                }
            )
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing)} Mem0 traces; first missing: {missing[0]}"
        )
    for row in rows:
        summary = summary_by_model[str(row["model"])]
        row["all_turn_relative_latency_summary_100"] = (
            100.0
            * float(row["all_turn_projected_latency_seconds_mean"])
            / float(summary["all_turn_projected_latency_seconds_mean"])
        )
    model_order = {source.model: index for index, source in enumerate(MODEL_SOURCES)}
    rows.sort(key=lambda row: (model_order[row["model"]], METHOD_ORDER[row["method"]]))
    return rows, missing


def _validate_two_stage_records(
    records: list[dict[str, Any]], turns_path: Path
) -> None:
    for index, row in enumerate(records):
        expected = {
            "logical_prompt_tokens": (
                float(row.get("extract_logical_prompt_tokens", 0))
                + float(row.get("manager_logical_prompt_tokens", 0))
            ),
            "evaluated_prefill_tokens": (
                float(row.get("extract_evaluated_prefill_tokens", 0))
                + float(row.get("manager_evaluated_prefill_tokens", 0))
            ),
            "decode_tokens": (
                float(row.get("extract_decode_tokens", 0))
                + float(row.get("manager_decode_tokens", 0))
            ),
        }
        for metric, value in expected.items():
            if float(row[metric]) != value:
                raise ValueError(
                    f"Invalid two-stage sum in {turns_path}:{index + 1}: {metric}"
                )


def write_outputs(
    rows: list[dict[str, Any]], missing: list[str], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "section5-mem0-efficiency"
    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "notes": {
                    "projected_latency": (
                        "LLM prefill/decode tokens projected with Galaxy Z Fold7 "
                        "Q4_K_M throughput"
                    ),
                    "retrieval_latency": (
                        "Separately measured during the HF/A100 trace; not added "
                        "to projected on-device latency"
                    ),
                },
                "missing_traces": missing,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Section 5 efficiency including Mem0",
        "",
        "| Model | Method | Cycles | UPDATE-cycle (s) | All-turn tokens | All-turn (s) | All-turn relative (Summary=100) | Manager calls (%) | Retrieval (ms/turn) | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['cycles']} | "
            f"{float(row['projected_cache_on_latency_seconds_mean']):.2f} | "
            f"{float(row['all_turn_model_tokens_mean']):.2f} | "
            f"{float(row['all_turn_projected_latency_seconds_mean']):.2f} | "
            f"{float(row['all_turn_relative_latency_summary_100']):.1f} | "
            f"{100.0 * float(row['manager_invocation_rate']):.1f} | "
            f"{1000.0 * float(row['all_turn_retrieval_seconds_mean']):.2f} | "
            f"{row['generation_errors']} |"
        )
    if missing:
        lines.extend(
            [
                "",
                f"> Partial result: {len(missing)} Mem0 traces are not complete.",
            ]
        )
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--mem0-campaign", type=Path, default=DEFAULT_MEM0_CAMPAIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows, missing = build_combined_rows(
        benchmark_root=args.benchmark_root.resolve(),
        mem0_campaign=args.mem0_campaign.resolve(),
        allow_partial=args.allow_partial,
    )
    write_outputs(rows, missing, args.output_dir.resolve())
    print(f"Wrote {len(rows)} rows; missing Mem0 traces: {len(missing)}")


if __name__ == "__main__":
    main()
