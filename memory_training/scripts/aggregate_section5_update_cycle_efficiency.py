#!/usr/bin/env python3
"""Reproduce the Section 5 UPDATE-cycle efficiency table from cache traces.

The Section 5 metric is conditional on an update that the model actually
applied.  For every such turn, it combines the update turn's decode cost with
the evaluated (cache-on) prefill cost of the immediately following turn:

    update decode tokens / decode tokens-per-second
      + next-turn evaluated prefill tokens / prefill tokens-per-second

Only adjacent records from the same scenario and repetition form a cycle.
This deliberately differs from the stress-test aggregation, which selects
cycles using the gold decision.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/on-device-memory-training/benchmarks"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs/engineering/results"


@dataclass(frozen=True)
class ModelSource:
    model: str
    benchmark: str
    turns_template: str
    prefill_tokens_per_second: float
    decode_tokens_per_second: float

    def turns_path(self, benchmark_root: Path, method_slug: str) -> Path:
        return (
            benchmark_root
            / self.benchmark
            / self.turns_template.format(method=method_slug)
        )


MODEL_SOURCES = (
    ModelSource(
        "Granite 350M",
        "granite4-350m-natural-s86-s90-cache-once-20260907-v1",
        "cache/{method}/on/turns.jsonl",
        137.7,
        51.8,
    ),
    ModelSource(
        "Qwen 0.8B",
        "qwen35-0.8b-natural-s86-s90-cache-once-20260905-v1",
        "cache/{method}/on/turns.jsonl",
        68.0,
        27.6,
    ),
    ModelSource(
        "Granite 1B",
        "granite4-1b-natural-s86-s90-cache-once-20260905-v1",
        "cache/{method}/on/turns.jsonl",
        25.7,
        16.4,
    ),
    ModelSource(
        "Llama 3.2 1B",
        "llama32-1b-stress-test-once-20260908-v1",
        "cache/base/{method}/on/turns.jsonl",
        35.7,
        19.8,
    ),
    ModelSource(
        "Llama 3.2 3B",
        "llama32-3b-stress-test-once-20260908-v1",
        "cache/base/{method}/on/turns.jsonl",
        11.1,
        7.3,
    ),
    ModelSource(
        "Qwen 2B",
        "qwen35-2b-natural-s86-s90-cache-once-20260907-v1",
        "cache/{method}/on/turns.jsonl",
        24.7,
        14.5,
    ),
)

METHODS = (
    ("Summary", "summary"),
    ("Patch", "patch"),
    ("Delta-v3 (k=5)", "delta_v3_compact_k5"),
)

# Values already reported in the Section 5 draft.  Keeping them here makes a
# change in trace selection or source provenance fail loudly.
REFERENCE = {
    ("Granite 350M", "Summary"): (33, 6.54),
    ("Granite 350M", "Patch"): (48, 2.96),
    ("Granite 350M", "Delta-v3 (k=5)"): (38, 2.67),
    ("Qwen 0.8B", "Summary"): (39, 15.28),
    ("Qwen 0.8B", "Patch"): (64, 8.38),
    ("Qwen 0.8B", "Delta-v3 (k=5)"): (32, 7.62),
    ("Granite 1B", "Summary"): (59, 23.36),
    ("Granite 1B", "Patch"): (59, 12.47),
    ("Granite 1B", "Delta-v3 (k=5)"): (41, 11.08),
    ("Llama 3.2 1B", "Summary"): (34, 18.00),
    ("Llama 3.2 1B", "Patch"): (55, 9.31),
    ("Llama 3.2 1B", "Delta-v3 (k=5)"): (35, 8.55),
    ("Llama 3.2 3B", "Summary"): (27, 47.63),
    ("Llama 3.2 3B", "Patch"): (45, 28.95),
    ("Llama 3.2 3B", "Delta-v3 (k=5)"): (33, 24.82),
    ("Qwen 2B", "Summary"): (28, 36.38),
    ("Qwen 2B", "Patch"): (53, 20.46),
    ("Qwen 2B", "Delta-v3 (k=5)"): (42, 17.32),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def applied_update_cycles(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return applied UPDATE records paired with their true next turn."""

    return [
        (current, following)
        for current, following in pairwise(records)
        if bool(current.get("applied_update"))
        and current.get("scenario_index") == following.get("scenario_index")
        and current.get("repetition") == following.get("repetition")
    ]


def aggregate_cycles(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    prefill_tokens_per_second: float,
    decode_tokens_per_second: float,
) -> dict[str, float | int]:
    if prefill_tokens_per_second <= 0 or decode_tokens_per_second <= 0:
        raise ValueError("Token throughput must be positive")
    if not pairs:
        raise ValueError("No applied UPDATE cycles were found")

    update_decode = [float(current["decode_tokens"]) for current, _ in pairs]
    next_logical_prefill = [
        float(following["logical_prompt_tokens"]) for _, following in pairs
    ]
    next_evaluated_prefill = [
        float(following["evaluated_prefill_tokens"]) for _, following in pairs
    ]
    cache_on_latency = [
        decode / decode_tokens_per_second + evaluated / prefill_tokens_per_second
        for decode, evaluated in zip(update_decode, next_evaluated_prefill)
    ]
    cache_off_latency = [
        decode / decode_tokens_per_second + logical / prefill_tokens_per_second
        for decode, logical in zip(update_decode, next_logical_prefill)
    ]
    return {
        "cycles": len(pairs),
        "update_decode_tokens_mean": mean(update_decode),
        "next_logical_prefill_tokens_mean": mean(next_logical_prefill),
        "next_evaluated_prefill_tokens_mean": mean(next_evaluated_prefill),
        "projected_cache_on_latency_seconds_mean": mean(cache_on_latency),
        "projected_cache_off_latency_seconds_mean": mean(cache_off_latency),
    }


def aggregate_turns(
    records: Sequence[Mapping[str, Any]],
    *,
    prefill_tokens_per_second: float,
    decode_tokens_per_second: float,
) -> dict[str, float | int]:
    """Aggregate the per-request cost that is paid on every evaluated turn."""

    if prefill_tokens_per_second <= 0 or decode_tokens_per_second <= 0:
        raise ValueError("Token throughput must be positive")
    if not records:
        raise ValueError("No turn records were found")
    logical_prefill = [float(row["logical_prompt_tokens"]) for row in records]
    evaluated_prefill = [float(row["evaluated_prefill_tokens"]) for row in records]
    decode = [float(row["decode_tokens"]) for row in records]
    projected = [
        prefill / prefill_tokens_per_second + output / decode_tokens_per_second
        for prefill, output in zip(evaluated_prefill, decode, strict=True)
    ]
    retrieval = [float(row.get("retrieval_seconds", 0.0)) for row in records]
    manager_invocations = sum(bool(row.get("manager_invoked")) for row in records)
    return {
        "all_turns": len(records),
        "all_turn_logical_prefill_tokens_mean": mean(logical_prefill),
        "all_turn_evaluated_prefill_tokens_mean": mean(evaluated_prefill),
        "all_turn_decode_tokens_mean": mean(decode),
        "all_turn_model_tokens_mean": mean(
            [
                prefill + output
                for prefill, output in zip(evaluated_prefill, decode, strict=True)
            ]
        ),
        "all_turn_projected_latency_seconds_mean": mean(projected),
        "all_turn_retrieval_seconds_mean": mean(retrieval),
        "generation_errors": sum(row.get("error") is not None for row in records),
        "manager_invocations": manager_invocations,
        "manager_invocation_rate": manager_invocations / len(records),
    }


def build_rows(benchmark_root: Path = DEFAULT_BENCHMARK_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in MODEL_SOURCES:
        model_rows: list[dict[str, Any]] = []
        for method, method_slug in METHODS:
            turns_path = source.turns_path(benchmark_root, method_slug)
            if not turns_path.is_file():
                raise FileNotFoundError(f"Missing cache trace: {turns_path}")
            records = read_jsonl(turns_path)
            metrics = aggregate_cycles(
                applied_update_cycles(records),
                prefill_tokens_per_second=source.prefill_tokens_per_second,
                decode_tokens_per_second=source.decode_tokens_per_second,
            )
            all_turn_metrics = aggregate_turns(
                records,
                prefill_tokens_per_second=source.prefill_tokens_per_second,
                decode_tokens_per_second=source.decode_tokens_per_second,
            )
            model_rows.append(
                {
                    "model": source.model,
                    "method": method,
                    "method_slug": method_slug,
                    "trace_path": str(turns_path),
                    "prefill_tokens_per_second": source.prefill_tokens_per_second,
                    "decode_tokens_per_second": source.decode_tokens_per_second,
                    **metrics,
                    **all_turn_metrics,
                }
            )
        summary_latency = float(
            next(
                row["projected_cache_on_latency_seconds_mean"]
                for row in model_rows
                if row["method"] == "Summary"
            )
        )
        for row in model_rows:
            row["relative_latency_summary_100"] = (
                100.0
                * float(row["projected_cache_on_latency_seconds_mean"])
                / summary_latency
            )
        rows.extend(model_rows)
    return rows


def verify_reference(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["model"]), str(row["method"]))
        if key not in REFERENCE:
            raise AssertionError(f"Unexpected operating point: {key}")
        expected_cycles, expected_latency = REFERENCE[key]
        actual_cycles = int(row["cycles"])
        actual_latency = round(float(row["projected_cache_on_latency_seconds_mean"]), 2)
        if (actual_cycles, actual_latency) != (expected_cycles, expected_latency):
            raise AssertionError(
                f"{key}: expected cycles/latency "
                f"{expected_cycles}/{expected_latency:.2f}, got "
                f"{actual_cycles}/{actual_latency:.2f}"
            )
        seen.add(key)
    missing = set(REFERENCE) - seen
    if missing:
        raise AssertionError(f"Missing operating points: {sorted(missing)}")


def write_outputs(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "section5-update-cycle-efficiency"

    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "metric": {
            "selector": "applied_update == true",
            "pairing": "immediately following turn in the same scenario and repetition",
            "formula": (
                "update_decode_tokens / decode_tokens_per_second + "
                "next_evaluated_prefill_tokens / prefill_tokens_per_second"
            ),
            "latency_kind": "projected_from_Galaxy_Z_Fold7_Q4_K_M_throughput",
        },
        "sources": [asdict(source) for source in MODEL_SOURCES],
        "rows": list(rows),
    }
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Section 5 UPDATE-cycle efficiency",
        "",
        "`applied_update=true`인 턴의 decode와 동일 시나리오의 바로 다음 턴에서",
        "실제로 계산한 prefill을 결합한 재현 결과다.",
        "",
        "| Model | Method | Cycles | UPDATE decode tokens | Next evaluated prefill tokens | UPDATE-cycle latency (s) | Relative (Summary=100) | All-turn model tokens | All-turn latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['cycles']} | "
            f"{float(row['update_decode_tokens_mean']):.2f} | "
            f"{float(row['next_evaluated_prefill_tokens_mean']):.2f} | "
            f"{float(row['projected_cache_on_latency_seconds_mean']):.2f} | "
            f"{float(row['relative_latency_summary_100']):.1f} | "
            f"{float(row['all_turn_model_tokens_mean']):.2f} | "
            f"{float(row['all_turn_projected_latency_seconds_mean']):.2f} |"
        )
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="Write outputs without checking against the currently reported table.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_rows(args.benchmark_root.resolve())
    if not args.skip_reference_check:
        verify_reference(rows)
    write_outputs(rows, args.output_dir.resolve())
    print(f"Verified and wrote {len(rows)} operating points to {args.output_dir}")


if __name__ == "__main__":
    main()
