"""Prepare an isolated V2-compatible evaluation root for an authored pilot.

The authored source remains excluded from benchmark reporting.  This adapter
assigns it an evaluation-only canonical scenario alias so the existing trusted
closed-loop evaluator can be reused without weakening its split checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from memory_training.dataset import DatasetCatalog, IndexedMemoryDataset, build_catalog
from memory_training.methods.delta_v2 import (
    compact_operations,
    delta_v2_input_from_state,
)
from memory_training.methods.delta_v3 import DeltaV3Method
from memory_training.methods.operations import normalize_memory
from memory_training.quiz_sft import (
    TOOL_SCHEMA_VERSION,
    _convert_row,
    _load_tool_modules,
    _load_tool_schemas,
    _sha256,
    _tool_validators,
)

SUMMARY_SCHEMA = "vehiclemembench-v2-grouped-summary-sft-v2"
PATCH_SCHEMA = "vehiclemembench-v2-grouped-patch-sft-v2"
DELTA_SCHEMA = "vehiclemembench-v2-grouped-delta-v2-sft-v2"
MANIFEST_SCHEMA = "vehiclemembench-human-authored-eval-adapter-manifest-v1"
EVALUATION_SCENARIO = 120
EVALUATION_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluation-scenario",
        type=int,
        default=EVALUATION_SCENARIO,
        help="Evaluation-only scenario alias assigned inside the isolated root.",
    )
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def memory_sha256(memory: str) -> str:
    return hashlib.sha256(normalize_memory(memory).encode("utf-8")).hexdigest()


def canonical_v2_memory(memory: str) -> str:
    """Remove annotation-only speaker IDs from grouped V2 memory headings."""
    return normalize_memory(
        re.sub(r"^### ([^|\n]+?) \| [^\n]+$", r"### \1", memory, flags=re.MULTILINE)
    )


def canonical_v2_operations(operations: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "op": str(operation["op"]),
            "target": canonical_v2_memory(str(operation.get("target", ""))),
            "content": canonical_v2_memory(str(operation.get("content", ""))),
        }
        for operation in operations
    ]


def source_provenance(
    row: dict[str, Any], source_scenario: int, evaluation_scenario: int
) -> dict[str, Any]:
    provenance = deepcopy(row.get("provenance", {}))
    provenance.update(
        {
            "source_scenario_index": source_scenario,
            "source_split": row.get("split"),
            "evaluation_alias_scenario_index": evaluation_scenario,
            "evaluation_only": True,
            "eligible_for_human_test": False,
        }
    )
    return provenance


def remap_common(
    row: dict[str, Any], *, view: str, source_scenario: int, evaluation_scenario: int
) -> dict[str, Any]:
    result = deepcopy(row)
    suffix = int(result["global_turn_index"])
    result.update(
        {
            "sample_id": f"s{evaluation_scenario:03d}:{view}:{suffix:05d}",
            "scenario_index": evaluation_scenario,
            "split": EVALUATION_SPLIT,
            "run_id": (
                f"external-adapted-pilot-s{source_scenario}-"
                f"eval-alias-s{evaluation_scenario}-v1"
            ),
            "train_eligible": False,
            "provenance": source_provenance(
                row, source_scenario, evaluation_scenario
            ),
        }
    )
    return result


def build_memory_views(
    pilot_root: Path,
    evaluation_scenario: int = EVALUATION_SCENARIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    patches = load_jsonl(pilot_root / "patch.jsonl")
    snapshots = load_jsonl(pilot_root / "memory_snapshots.jsonl")
    if len(patches) != len(snapshots) or not patches:
        raise ValueError("Pilot Patch rows and snapshots must be non-empty and aligned")
    source_scenario = int(patches[0]["scenario_index"])
    summaries: list[dict[str, Any]] = []
    remapped_patches: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    delta_method = DeltaV3Method()
    delta_state = delta_method.initial_state()

    for patch, snapshot in zip(patches, snapshots, strict=True):
        if int(patch["scenario_index"]) != source_scenario:
            raise ValueError("Pilot contains more than one source scenario")
        if patch["turn_id"] != snapshot["turn_id"]:
            raise ValueError("Pilot Patch/snapshot turn mismatch")
        before = canonical_v2_memory(str(patch["input"]["previous_memory"]))
        after = canonical_v2_memory(str(snapshot["memory"]))
        if delta_method.materialize_memory(delta_state) != before:
            raise ValueError(f"Delta replay diverged before {patch['turn_id']}")

        summary = remap_common(
            patch,
            view="summary",
            source_scenario=source_scenario,
            evaluation_scenario=evaluation_scenario,
        )
        summary["schema_version"] = SUMMARY_SCHEMA
        summary["input"] = {"previous_memory": before}
        summary["target"] = {
            "decision": patch["target"]["decision"],
            "reason_code": patch["target"]["reason_code"],
            "reason": patch["target"]["reason"],
            "next_memory": after,
        }
        summaries.append(summary)

        patch_row = remap_common(
            patch,
            view="patch",
            source_scenario=source_scenario,
            evaluation_scenario=evaluation_scenario,
        )
        patch_row["schema_version"] = PATCH_SCHEMA
        patch_row["input"] = {"previous_memory": before}
        patch_row["target"]["operations"] = canonical_v2_operations(
            patch["target"].get("operations", [])
        )
        for row in (summary, patch_row):
            row["provenance"]["before_memory_sha256"] = memory_sha256(before)
            row["provenance"]["after_memory_sha256"] = memory_sha256(after)
        remapped_patches.append(patch_row)

        delta = remap_common(
            patch,
            view="delta",
            source_scenario=source_scenario,
            evaluation_scenario=evaluation_scenario,
        )
        delta["schema_version"] = DELTA_SCHEMA
        delta["input"] = delta_v2_input_from_state(delta_state)
        delta["provenance"]["before_memory_sha256"] = memory_sha256(before)
        delta["provenance"]["after_memory_sha256"] = memory_sha256(after)
        operations = patch_row["target"].get("operations", [])
        compact = (
            compact_operations(operations)
            if patch["target"]["decision"] == "UPDATE"
            else []
        )
        delta["target"] = {
            "decision": patch["target"]["decision"],
            "reason_code": patch["target"]["reason_code"],
            "reason": patch["target"]["reason"],
            "operations": compact,
        }
        parsed = delta_method.parse_output(
            json.dumps(
                {"decision": "NO_OP"}
                if patch["target"]["decision"] == "NO_OP"
                else {
                    "decision": "UPDATE",
                    "operations": compact,
                },
                ensure_ascii=False,
            )
        )
        delta_state = delta_method.apply_output(delta_state, parsed)
        if delta_method.materialize_memory(delta_state) != after:
            raise ValueError(f"Delta replay diverged after {patch['turn_id']}")
        deltas.append(delta)

    return summaries, remapped_patches, deltas


def remap_quizzes(
    pilot_root: Path,
    source_scenario: int,
    memory_hashes: dict[int, str],
    evaluation_scenario: int = EVALUATION_SCENARIO,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for filename, kind in (
        ("turn_quiz.jsonl", "turn_quiz"),
        ("final_quiz.jsonl", "final_quiz"),
    ):
        rows = []
        for source in load_jsonl(pilot_root / filename):
            row = deepcopy(source)
            turn = int(row["memory_ref"]["global_turn_index"])
            quiz_id = str(row["quiz_id"])
            row.update(
                {
                    "sample_id": f"s{evaluation_scenario:03d}:{kind}:{quiz_id}",
                    "scenario_index": evaluation_scenario,
                    "split": EVALUATION_SPLIT,
                    "run_id": (
                        f"external-adapted-pilot-s{source_scenario}-"
                        f"eval-alias-s{evaluation_scenario}-v1"
                    ),
                    "source_chain_id": f"s{source_scenario}-evaluation-alias",
                    "provenance": source_provenance(
                        source, source_scenario, evaluation_scenario
                    ),
                }
            )
            row["memory_ref"].update(
                {
                    "checkpoint_id": f"s{evaluation_scenario:03d}:memory:{turn:05d}",
                    "summary_sample_id": f"s{evaluation_scenario:03d}:summary:{turn:05d}",
                    "patch_sample_id": f"s{evaluation_scenario:03d}:patch:{turn:05d}",
                    "delta_sample_id": f"s{evaluation_scenario:03d}:delta:{turn:05d}",
                    "memory_snapshot_sha256": memory_hashes[turn],
                }
            )
            row["provenance"]["grouped_memory_sha256"] = memory_hashes[turn]
            rows.append(row)
        result[filename] = rows
    return result


def build_quiz_rows(output_dir: Path, benchmark_root: Path) -> list[dict[str, Any]]:
    tools, source_tool_sha256 = _load_tool_schemas(benchmark_root)
    validators = _tool_validators(tools)
    tool_modules, module_tools = _load_tool_modules(benchmark_root, tools)
    tool_payload = {
        "schema_version": TOOL_SCHEMA_VERSION,
        "source": str(benchmark_root / "evaluation" / "functions_schema.json"),
        "source_sha256": source_tool_sha256,
        "tools": tools,
        "modules": module_tools,
    }
    write_json(output_dir / "vehicle_tools.json", tool_payload)
    tools_sha256 = _sha256(output_dir / "vehicle_tools.json")
    build_catalog(output_dir, output_dir / "catalog.sqlite")
    summaries = IndexedMemoryDataset(
        DatasetCatalog(output_dir / "catalog.sqlite", output_dir),
        "summary",
    )
    quiz_rows = []
    for source_name in ("turn_quiz.jsonl", "final_quiz.jsonl"):
        for line_number, source in enumerate(
            load_jsonl(output_dir / source_name), start=1
        ):
            quiz_rows.append(
                _convert_row(
                    source,
                    source_name=source_name,
                    line_number=line_number,
                    summaries=summaries,
                    validators=validators,
                    tool_modules=tool_modules,
                    module_tools=module_tools,
                    tools_sha256=tools_sha256,
                )
            )
    summaries.close()
    return quiz_rows


def build(
    pilot_root: Path,
    output_dir: Path,
    benchmark_root: Path,
    evaluation_scenario: int = EVALUATION_SCENARIO,
) -> dict[str, Any]:
    pilot_root = pilot_root.resolve(strict=True)
    benchmark_root = benchmark_root.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(
        (pilot_root / "manifest.json").read_text(encoding="utf-8")
    )
    source_scenario = int(source_manifest["scenario_index"])
    summaries, patches, deltas = build_memory_views(
        pilot_root, evaluation_scenario=evaluation_scenario
    )
    memory_hashes = {
        int(row["global_turn_index"]): memory_sha256(row["target"]["next_memory"])
        for row in summaries
    }
    quizzes = remap_quizzes(
        pilot_root,
        source_scenario,
        memory_hashes,
        evaluation_scenario=evaluation_scenario,
    )
    write_jsonl(output_dir / "summary.jsonl", summaries)
    write_jsonl(output_dir / "patch.jsonl", patches)
    write_jsonl(output_dir / "delta.jsonl", deltas)
    for filename, rows in quizzes.items():
        write_jsonl(output_dir / filename, rows)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "purpose": "evaluation-only V2 compatibility adapter",
        "source_root": str(pilot_root),
        "source_scenario_index": source_scenario,
        "evaluation_alias_scenario_index": evaluation_scenario,
        "split": EVALUATION_SPLIT,
        "eligible_for_training": False,
        "eligible_for_human_test": False,
        "model_authored_pilot": "external_adaptation" not in source_manifest,
        "external_adapted_pilot": "external_adaptation" in source_manifest,
        "counts": {
            "turns": len(summaries),
            "updates": sum(row["target"]["decision"] == "UPDATE" for row in summaries),
            "noops": sum(row["target"]["decision"] == "NO_OP" for row in summaries),
            "turn_quizzes": len(quizzes["turn_quiz.jsonl"]),
            "final_quizzes": len(quizzes["final_quiz.jsonl"]),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "quiz_manifest.json",
        {
            "schema_version": "vehiclemembench-human-authored-eval-quiz-manifest-v1",
            "source_scenario_index": source_scenario,
            "evaluation_alias_scenario_index": evaluation_scenario,
            "counts": manifest["counts"],
        },
    )
    quiz_rows = build_quiz_rows(output_dir, benchmark_root)
    write_jsonl(output_dir / "quiz_sft.jsonl", quiz_rows)
    write_json(
        output_dir / "quiz_sft_manifest.json",
        {
            "schema_version": "vehiclemembench-human-authored-eval-quiz-sft-manifest-v1",
            "row_count": len(quiz_rows),
            "tool_schema_sha256": _sha256(output_dir / "vehicle_tools.json"),
        },
    )
    manifest["quiz_sft_rows"] = len(quiz_rows)
    write_json(output_dir / "manifest.json", manifest)
    # The final manifest participates in the catalog fingerprint.
    build_catalog(output_dir, output_dir / "catalog.sqlite")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build(
        args.pilot_root,
        args.output_dir,
        args.vehiclemembench_root,
        evaluation_scenario=args.evaluation_scenario,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
