#!/usr/bin/env python3
"""Prepare paired Patch-vs-Delta disagreement cases for causal error coding."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_training.scripts.aggregate_three_method_type_accuracy import (
    MODEL_ARTIFACTS,
)


METHODS = ("Summary", "Delta-v3", "Patch")
DIRECTIONS = ("PATCH_ONLY_CORRECT", "DELTA_ONLY_CORRECT")
PRIMARY_CAUSES = (
    "MISSING_FACT",
    "STALE_VALUE",
    "CONFLICTING_FACTS",
    "WRONG_OWNER",
    "CONDITION_LOSS",
    "PENDING_READ_FAILURE",
    "EVIDENCE_IGNORED",
)
OUTPUT_SYMPTOMS = ("WRONG_TOOL", "WRONG_ARGUMENT", "OVER_ACTION")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--quiz-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_quiz_source(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["closed_loop_quiz"]["records"]
    result = {str(row["sample_id"]): row for row in records}
    if len(records) != 960 or len(result) != 960:
        raise ValueError(f"Expected 960 unique quiz records: {path}")
    return result


def _canonical_call(call: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(call["name"]),
        json.dumps(
            call.get("arguments", {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _output_symptoms(
    predicted: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]
) -> list[str]:
    predicted_names = Counter(str(call["name"]) for call in predicted)
    gold_names = Counter(str(call["name"]) for call in gold)
    symptoms: list[str] = []
    if predicted_names != gold_names:
        symptoms.append("WRONG_TOOL")
    if len(predicted) > len(gold):
        symptoms.append("OVER_ACTION")
    predicted_by_name: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    gold_by_name: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for call in predicted:
        predicted_by_name[str(call["name"])].append(_canonical_call(call))
    for call in gold:
        gold_by_name[str(call["name"])].append(_canonical_call(call))
    if any(
        sorted(predicted_by_name[name]) != sorted(gold_by_name[name])
        for name in predicted_by_name.keys() & gold_by_name.keys()
    ):
        symptoms.append("WRONG_ARGUMENT")
    if not symptoms and Counter(map(_canonical_call, predicted)) != Counter(
        map(_canonical_call, gold)
    ):
        symptoms.append("WRONG_ARGUMENT")
    return symptoms


def _gold_calls(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(call) for call in row["target"]["tool_calls"]]


def _gold_context(row: Mapping[str, Any]) -> tuple[str, str]:
    user = next(
        str(message["content"])
        for message in row["messages"]
        if message["role"] == "user"
    )
    marker = "\n\n[Current request]\n"
    if marker not in user or not user.startswith("[Memory]\n"):
        raise ValueError(f"Unexpected quiz prompt format: {row['sample_id']}")
    memory, request = user[len("[Memory]\n") :].split(marker, 1)
    return memory, request


def _method_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exact_state_match": int(bool(record["exact_state_match"])),
        "tool_f1": float(record["tool_f1"]),
        "arg_exact": float(record["arg_exact"]),
        "parse_success": int(bool(record["parse_success"])),
        "execution_success": int(bool(record["execution_success"])),
        "predicted_calls": record["predicted_calls"],
        "execution_errors": record.get("execution_errors", []),
        "output": record.get("output", ""),
    }


def prepare(run_root: Path, quiz_source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _load_quiz_source(quiz_source)
    cases: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}
    for model, artifacts in MODEL_ARTIFACTS.items():
        records: dict[str, dict[str, dict[str, Any]]] = {}
        sources[model] = {}
        for method in METHODS:
            path = run_root / artifacts[method]
            sources[model][method] = str(path)
            records[method] = _load_records(path)
        cohort = set(records["Patch"])
        if any(set(records[method]) != cohort for method in METHODS):
            raise ValueError(f"Quiz cohort differs across methods for {model}")
        for sample_id in sorted(cohort):
            patch_correct = bool(records["Patch"][sample_id]["exact_state_match"])
            delta_correct = bool(records["Delta-v3"][sample_id]["exact_state_match"])
            if patch_correct == delta_correct:
                continue
            direction = (
                "PATCH_ONLY_CORRECT" if patch_correct else "DELTA_ONLY_CORRECT"
            )
            loser = "Delta-v3" if patch_correct else "Patch"
            source_row = source[sample_id]
            gold_memory, current_request = _gold_context(source_row)
            gold_calls = _gold_calls(source_row)
            loser_calls = records[loser][sample_id]["predicted_calls"]
            memory_ref = source_row["memory_ref"]
            cases.append(
                {
                    "case_id": f"{model}::{sample_id}",
                    "model": model,
                    "sample_id": sample_id,
                    "scenario_index": int(source_row["scenario_index"]),
                    "global_turn_index": int(memory_ref["global_turn_index"]),
                    "quiz_type": str(source_row["quiz_type"]),
                    "reasoning_type": str(source_row["reasoning_type"]),
                    "direction": direction,
                    "winner": "Patch" if patch_correct else "Delta-v3",
                    "loser": loser,
                    "gold_memory": gold_memory,
                    "current_request": current_request,
                    "gold_calls": gold_calls,
                    "methods": {
                        method: _method_record(records[method][sample_id])
                        for method in METHODS
                    },
                    "coding": {
                        "primary_cause": None,
                        "primary_cause_status": "NEEDS_PREDICTED_MEMORY_SNAPSHOT",
                        "output_symptoms": _output_symptoms(loser_calls, gold_calls),
                        "review_required": None,
                        "review_notes": "",
                    },
                }
            )
    if len(cases) != 1121:
        raise ValueError(f"Expected 1,121 discordant model×quiz cases, got {len(cases)}")
    counts = Counter(case["direction"] for case in cases)
    by_model = {
        model: dict(Counter(case["direction"] for case in cases if case["model"] == model))
        for model in MODEL_ARTIFACTS
    }
    by_reasoning = {
        reason: dict(
            Counter(
                case["direction"]
                for case in cases
                if case["reasoning_type"] == reason
            )
        )
        for reason in sorted({case["reasoning_type"] for case in cases})
    }
    symptom_counts = Counter(
        symptom for case in cases for symptom in case["coding"]["output_symptoms"]
    )
    summary = {
        "schema_version": "palmclaw-patch-delta-disagreement-preparation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "unit": "model_x_quiz",
        "methods": list(METHODS),
        "directions": list(DIRECTIONS),
        "coding_contract": {
            "primary_causes_exclusive": list(PRIMARY_CAUSES),
            "output_symptoms_multilabel": list(OUTPUT_SYMPTOMS),
            "human_or_llm_review_primary_causes": ["CONDITION_LOSS"],
            "primary_precedence": [
                "WRONG_OWNER",
                "CONFLICTING_FACTS",
                "STALE_VALUE",
                "MISSING_FACT",
                "CONDITION_LOSS",
                "PENDING_READ_FAILURE",
                "EVIDENCE_IGNORED",
            ],
        },
        "cases": len(cases),
        "direction_counts": dict(counts),
        "by_model": by_model,
        "by_reasoning_type": by_reasoning,
        "output_symptom_counts_multilabel": dict(symptom_counts),
        "primary_cause_status": "awaiting predicted memory snapshots",
        "sources": sources,
    }
    return cases, summary


def _snapshot_request_manifest(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_ARTIFACTS:
        selected = [case for case in cases if case["model"] == model]
        by_model[model] = [
            {
                "sample_id": case["sample_id"],
                "scenario_index": case["scenario_index"],
                "global_turn_index": case["global_turn_index"],
                "direction": case["direction"],
            }
            for case in selected
        ]
    return {
        "schema_version": "palmclaw-disagreement-snapshot-request-v1",
        "required_methods": ["Patch", "Delta-v3"],
        "optional_triangulation_method": "Summary",
        "required_snapshot_fields": [
            "materialized_memory",
            "base_memory",
            "pending_deltas",
        ],
        "by_model": by_model,
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_review_csv(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "case_id",
        "model",
        "sample_id",
        "reasoning_type",
        "direction",
        "loser",
        "output_symptoms",
        "primary_cause",
        "review_required",
        "review_notes",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            coding = case["coding"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "model": case["model"],
                    "sample_id": case["sample_id"],
                    "reasoning_type": case["reasoning_type"],
                    "direction": case["direction"],
                    "loser": case["loser"],
                    "output_symptoms": ";".join(coding["output_symptoms"]),
                    "primary_cause": "",
                    "review_required": "",
                    "review_notes": "",
                }
            )
    os.replace(temporary, path)


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, summary = prepare(args.run_root.resolve(), args.quiz_source.resolve())
    _write_jsonl(output / "disagreement-cases.jsonl", cases)
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "snapshot-request-manifest.json",
        _snapshot_request_manifest(cases),
    )
    _write_review_csv(output / "review-template.csv", cases)
    print(json.dumps({"output": str(output), "cases": len(cases)}))


if __name__ == "__main__":
    main()
