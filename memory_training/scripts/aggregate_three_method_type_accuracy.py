#!/usr/bin/env python3
"""Aggregate fixed-Test quiz accuracy by type for Summary, Delta-v3, and Patch."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHODS = ("Summary", "Delta-v3", "Patch")
MODEL_ARTIFACTS: Mapping[str, Mapping[str, str]] = {
    "Granite 350M": {
        "Summary": "granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/eval-fixed-v2-test-best-epoch-04/summary.json",
        "Delta-v3": "granite4-350m-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/eval-fixed-test-best-epoch-03/summary.json",
        "Patch": "granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1/eval-fixed-v2-test-best-epoch-04/summary.json",
    },
    "Qwen 0.8B": {
        "Summary": "qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2/eval-fixed-test-best-epoch-03/summary.json",
        "Delta-v3": "qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/eval-fixed-test-best-epoch-03/summary.json",
        "Patch": "qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed46-evalfixed-noop5-r1/eval-fixed-composite-test-epoch-03/summary.json",
    },
    "Granite 1B": {
        "Summary": "granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/eval-fixed-v2-test-best-epoch-03/summary.json",
        "Delta-v3": "granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/eval-fixed-test-best-epoch-04/summary.json",
        "Patch": "granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/eval-fixed-v2-test-best-epoch-03/summary.json",
    },
    "Llama 3.2 1B": {
        "Summary": "llama3.2-1b-summary-multitask-noop5-e4-b4-trainseed45-evalfixed-noop5-r1/eval-fixed-test-best-epoch-04/summary.json",
        "Delta-v3": "llama3.2-1b-delta_v3_compact_k5-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r2/eval-fixed-test-best-epoch-04/summary.json",
        "Patch": "llama3.2-1b-patch-multitask-noop5-e4-b8-trainseed45-evalfixed-noop5-r2/eval-fixed-test-best-epoch-03/summary.json",
    },
    "Qwen 2B": {
        "Summary": "qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1/eval-fixed-v2-test-best-epoch-04/summary.json",
        "Delta-v3": "qwen35-2b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/eval-fixed-test-best-epoch-04/summary.json",
        "Patch": "qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-trainseed46-r2/eval-fixed-v2-test-best-epoch-03/summary.json",
    },
    "Llama 3.2 3B": {
        "Summary": "llama3.2-3b-summary-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1/eval-fixed-test-best-epoch-03/summary.json",
        "Delta-v3": "llama3.2-3b-delta_v3_compact_k5-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1/eval-fixed-test-best-epoch-04/summary.json",
        "Patch": "llama3.2-3b-patch-multitask-noop5-e4-b4-trainseed45-evalfixed-noop5-r1/eval-fixed-test-best-epoch-04/summary.json",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    count = len(records)
    if not count:
        return {"count": 0, "esm": 0.0, "tool_f1": 0.0, "arg_exact": 0.0}
    return {
        "count": count,
        "esm": sum(float(row["exact_state_match"]) for row in records) / count,
        "tool_f1": sum(float(row["tool_f1"]) for row in records) / count,
        "arg_exact": sum(float(row["arg_exact"]) for row in records) / count,
    }


def _group(
    records: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], str]
) -> dict[str, dict[str, int | float]]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return {name: _metrics(rows) for name, rows in sorted(groups.items())}


def _mcnemar_exact_p(patch_only: int, delta_only: int) -> float:
    discordant = patch_only + delta_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(patch_only, delta_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _paired_metrics(
    patch: Sequence[Mapping[str, Any]], delta: Sequence[Mapping[str, Any]]
) -> dict[str, int | float]:
    patch_by_id = {str(row["sample_id"]): row for row in patch}
    delta_by_id = {str(row["sample_id"]): row for row in delta}
    if patch_by_id.keys() != delta_by_id.keys():
        raise ValueError("Patch and Delta-v3 paired cohorts differ")
    patch_only = delta_only = 0
    for sample_id, patch_row in patch_by_id.items():
        patch_correct = bool(patch_row["exact_state_match"])
        delta_correct = bool(delta_by_id[sample_id]["exact_state_match"])
        patch_only += int(patch_correct and not delta_correct)
        delta_only += int(delta_correct and not patch_correct)
    patch_metrics = _metrics(patch)
    delta_metrics = _metrics(delta)
    return {
        "count": len(patch),
        "patch_esm": float(patch_metrics["esm"]),
        "delta_esm": float(delta_metrics["esm"]),
        "patch_minus_delta_percentage_points": 100
        * (float(patch_metrics["esm"]) - float(delta_metrics["esm"])),
        "patch_only_correct": patch_only,
        "delta_only_correct": delta_only,
        "mcnemar_exact_p": _mcnemar_exact_p(patch_only, delta_only),
    }


def _paired_group(
    patch: Sequence[Mapping[str, Any]],
    delta: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
) -> dict[str, dict[str, int | float]]:
    patch_groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    delta_groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in patch:
        patch_groups[key(row)].append(row)
    for row in delta:
        delta_groups[key(row)].append(row)
    if patch_groups.keys() != delta_groups.keys():
        raise ValueError("Patch and Delta-v3 paired group labels differ")
    return {
        name: _paired_metrics(patch_groups[name], delta_groups[name])
        for name in sorted(patch_groups)
    }


def aggregate(run_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    patch_delta_comparisons: dict[str, Any] = {}
    canonical_ids: set[str] | None = None
    canonical_metadata: dict[str, tuple[str, str]] | None = None
    for model, artifacts in MODEL_ARTIFACTS.items():
        model_result: dict[str, Any] = {}
        records_by_method: dict[str, list[Mapping[str, Any]]] = {}
        for method in METHODS:
            path = run_root / artifacts[method]
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload["closed_loop_quiz"]["records"]
            if len(records) != 960:
                raise ValueError(f"Expected 960 quiz records: {path}")
            ids = {str(row["sample_id"]) for row in records}
            metadata = {
                str(row["sample_id"]): (
                    str(row["quiz_type"]),
                    str(row["reasoning_type"]),
                )
                for row in records
            }
            if len(ids) != len(records):
                raise ValueError(f"Duplicate quiz sample IDs: {path}")
            if canonical_ids is None:
                canonical_ids = ids
                canonical_metadata = metadata
            elif ids != canonical_ids or metadata != canonical_metadata:
                raise ValueError(f"Quiz cohort or type metadata differs: {path}")
            records_by_method[method] = records
            model_result[method] = {
                "artifact": str(path),
                "overall": _metrics(records),
                "by_quiz_type": _group(records, lambda row: str(row["quiz_type"])),
                "by_reasoning_type": _group(
                    records, lambda row: str(row["reasoning_type"])
                ),
                "by_quiz_and_reasoning_type": _group(
                    records,
                    lambda row: f"{row['quiz_type']}::{row['reasoning_type']}",
                ),
            }
        patch_records = records_by_method["Patch"]
        delta_records = records_by_method["Delta-v3"]
        patch_delta_comparisons[model] = {
            "overall": _paired_metrics(patch_records, delta_records),
            "by_quiz_type": _paired_group(
                patch_records, delta_records, lambda row: str(row["quiz_type"])
            ),
            "by_reasoning_type": _paired_group(
                patch_records,
                delta_records,
                lambda row: str(row["reasoning_type"]),
            ),
        }
        results[model] = model_result
    return {
        "schema_version": "palmclaw-three-method-quiz-type-accuracy-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "exact_state_match",
        "methods": list(METHODS),
        "models": list(MODEL_ARTIFACTS),
        "quiz_cohort": {
            "records": len(canonical_ids or ()),
            "identical_sample_ids_and_types_across_all_artifacts": True,
        },
        "results": results,
        "comparisons": {"Patch-vs-Delta-v3": patch_delta_comparisons},
    }


def main() -> None:
    args = build_parser().parse_args()
    result = aggregate(args.run_root.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "models": len(result["models"])}))


if __name__ == "__main__":
    main()
