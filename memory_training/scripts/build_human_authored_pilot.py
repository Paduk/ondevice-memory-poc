"""Validate and export a human-authored VehicleMemBench transfer scenario.

The source format is deliberately easier for human annotators than the runtime
JSONL.  This script deterministically derives turn-wise Patch rows, full memory
snapshots, and Quiz rows while validating every Tool call against both the
official JSON schemas and the VehicleWorld simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime

from memory_training.methods.operations import apply_operations, normalize_memory

SOURCE_SCHEMA_VERSION = "vehiclemembench-human-authored-scenario-source-v1"
PATCH_SCHEMA_VERSION = "vehiclemembench-v2-grouped-patch-sft-v1"
TURN_QUIZ_SCHEMA_VERSION = "vehiclemembench-v2-turn-quiz-row-v1"
FINAL_QUIZ_SCHEMA_VERSION = "vehiclemembench-v2-final-quiz-row-v1"
SNAPSHOT_SCHEMA_VERSION = "vehiclemembench-human-authored-memory-snapshot-v1"
MANIFEST_SCHEMA_VERSION = "vehiclemembench-human-authored-pilot-manifest-v1"
VALID_REASONING_TYPES = {
    "conditional_constraint",
    "coreference_resolution",
    "error_correction",
    "preference_conflict",
    "state_shift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_source(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(source, dict), "scenario source must be an object")
    require(
        source.get("schema_version") == SOURCE_SCHEMA_VERSION,
        f"unsupported source schema: {source.get('schema_version')}",
    )
    require(source.get("split") == "pilot_excluded", "pilot must be excluded")
    authorship = source.get("authorship", {})
    require(
        authorship.get("eligible_for_human_test") is False,
        "model-authored pilot must be marked ineligible for the human test",
    )
    return source


def load_tool_schemas(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "evaluation" / "functions_schema.json"
    schemas = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(schemas, list) and schemas, "Tool schema must be a non-empty list"
    )
    result = {str(schema["name"]): schema for schema in schemas}
    require(len(result) == len(schemas), "Tool schema contains duplicate names")
    return result


def speaker_maps(
    source: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_short_id: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    for speaker in source.get("speakers", []):
        short_id = str(speaker["short_id"])
        speaker_id = str(speaker["speaker_id"])
        require(short_id not in by_short_id, f"duplicate speaker short_id: {short_id}")
        require(speaker_id not in names, f"duplicate speaker_id: {speaker_id}")
        by_short_id[short_id] = speaker
        names[speaker_id] = str(speaker["speaker_name"])
    require(len(by_short_id) >= 2, "scenario needs at least two human speakers")
    names["shared-vehicle"] = "shared-vehicle"
    return by_short_id, names


def render_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_memory_line(timestamp: str, update: dict[str, Any]) -> str:
    stamp = timestamp[:16].replace("T", " ")
    line = (
        f"- [{stamp}] {update['tool_name']}.{update['argument_name']}; "
        f"value={render_value(update['value'])}"
    )
    context = update.get("context_arguments", {})
    if context:
        arguments = ", ".join(
            f"{key}={render_value(context[key])}" for key in sorted(context)
        )
        line += f"; context=({arguments})"
    condition = str(update.get("condition", "")).strip()
    if condition:
        line += f"; condition={condition}"
    return line


def owner_header(owner_id: str, speaker_names: dict[str, str]) -> str:
    if owner_id == "shared-vehicle":
        return "### shared-vehicle"
    return f"### {speaker_names[owner_id]} | {owner_id}"


def target_for_new_line(
    active_slots: OrderedDict[str, dict[str, str]], owner_id: str
) -> str:
    owned = [
        record["line"]
        for record in active_slots.values()
        if record["owner_id"] == owner_id
    ]
    return owned[-1] if owned else ""


def validate_tool_arguments(
    tool_schemas: dict[str, dict[str, Any]],
    runtime: VehicleWorldRuntime,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    schema = tool_schemas.get(name)
    require(schema is not None, f"unknown Tool: {name}")
    parameters = deepcopy(schema["parameters"])
    parameters.setdefault("additionalProperties", False)
    validator = Draft202012Validator(parameters)
    errors = sorted(
        validator.iter_errors(arguments), key=lambda error: list(error.path)
    )
    require(
        not errors,
        f"invalid arguments for {name}: {errors[0].message if errors else ''}",
    )
    world = runtime.create_world()
    result = runtime.execute(world, name, arguments)
    require(
        isinstance(result, dict) and result.get("success") is True,
        f"simulator rejected {name}{arguments}: {result}",
    )
    return result


def execute_quiz(
    runtime: VehicleWorldRuntime,
    tool_schemas: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    require(calls, "Quiz must contain at least one Gold Tool call")
    world = runtime.create_world()
    initial_state = runtime.state(world)
    for call in calls:
        name = str(call["name"])
        arguments = dict(call["arguments"])
        schema = tool_schemas.get(name)
        require(schema is not None, f"Quiz references unknown Tool: {name}")
        parameters = deepcopy(schema["parameters"])
        parameters.setdefault("additionalProperties", False)
        errors = list(Draft202012Validator(parameters).iter_errors(arguments))
        require(
            not errors,
            f"invalid Quiz arguments for {name}: {errors[0].message if errors else ''}",
        )
        result = runtime.execute(world, name, arguments)
        require(
            isinstance(result, dict) and result.get("success") is True,
            f"simulator rejected Quiz call {name}{arguments}: {result}",
        )
    target_state = runtime.state(world)
    require(
        target_state != initial_state, "Quiz Tool calls must change simulator state"
    )
    return target_state, text_sha256(canonical_json(target_state))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def build(source_path: Path, output_dir: Path, benchmark_root: Path) -> dict[str, Any]:
    source_path = source_path.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_source(source_path)
    tool_schemas = load_tool_schemas(benchmark_root.resolve(strict=True))
    runtime = VehicleWorldRuntime(benchmark_root)
    speakers, speaker_names = speaker_maps(source)

    scenario_id = str(source["scenario_id"])
    scenario_index = int(source["scenario_index"])
    is_external_adaptation = (
        source.get("authorship", {}).get("kind")
        == "external_adapted_model_assembled_pilot"
    )
    provenance_kind = (
        "external-adapted-pilot-template"
        if is_external_adaptation
        else "human-authored-pilot-template"
    )
    run_prefix = (
        "external-adapted-pilot" if is_external_adaptation else "human-authored-pilot"
    )
    run_id = f"{run_prefix}-{scenario_id.lower()}-v1"
    source_hash = file_sha256(source_path)

    patch_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    dialogue_rows: list[dict[str, Any]] = []
    active_slots: OrderedDict[str, dict[str, str]] = OrderedDict()
    snapshots_by_turn: dict[str, dict[str, Any]] = {}
    active_lines_by_turn: dict[str, dict[str, str]] = {}
    seen_turn_ids: set[str] = set()
    seen_session_ids: set[str] = set()
    previous_memory = ""
    operation_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    tool_names: set[str] = set()
    source_dataset_counts: Counter[str] = Counter()
    source_origin_counts: Counter[str] = Counter()
    reuse_mode_counts: Counter[str] = Counter()
    global_turn_index = 0
    previous_timestamp = ""

    for session in source.get("sessions", []):
        session_id = str(session["session_id"])
        require(
            session_id not in seen_session_ids, f"duplicate session_id: {session_id}"
        )
        seen_session_ids.add(session_id)
        timestamp = str(session["timestamp"])
        require(
            timestamp > previous_timestamp, "sessions must be strictly chronological"
        )
        previous_timestamp = timestamp
        turns = session.get("turns", [])
        require(turns, f"session has no turns: {session_id}")

        for event_turn_index, turn in enumerate(turns):
            turn_id = str(turn["turn_id"])
            require(turn_id not in seen_turn_ids, f"duplicate turn_id: {turn_id}")
            seen_turn_ids.add(turn_id)
            speaker_short_id = str(turn["speaker"])
            require(
                speaker_short_id in speakers, f"unknown speaker: {speaker_short_id}"
            )
            speaker = speakers[speaker_short_id]
            text = str(turn["text"]).strip()
            require(text, f"empty turn text: {turn_id}")
            source_trace = deepcopy(turn.get("source_trace"))
            if source_trace is not None:
                require(
                    isinstance(source_trace, dict),
                    f"source_trace must be an object: {turn_id}",
                )
                dataset = str(source_trace.get("dataset", "")).strip()
                origin = str(source_trace.get("origin", "")).strip()
                reuse_mode = str(source_trace.get("reuse_mode", "")).strip()
                original_text = str(source_trace.get("original_text", "")).strip()
                require(dataset, f"source_trace dataset is empty: {turn_id}")
                require(
                    origin in {"human_authored", "synthetic"},
                    f"invalid source origin: {turn_id}",
                )
                require(
                    reuse_mode
                    in {
                        "verbatim_text_role_normalized",
                        "minimal_persistence_adaptation",
                        "vehicle_domain_rewrite",
                        "vehicle_domain_rewrite_with_persistence",
                        "source_inspired_single_intent_rewrite",
                        "explicit_nonvehicle_calibration_wrapper",
                    },
                    f"invalid reuse_mode: {turn_id}",
                )
                require(original_text, f"source original_text is empty: {turn_id}")
                if reuse_mode == "verbatim_text_role_normalized":
                    require(
                        text == original_text,
                        f"verbatim source text changed: {turn_id}",
                    )
                else:
                    require(
                        text != original_text,
                        f"adapted source text is unchanged: {turn_id}",
                    )
                source_dataset_counts[dataset] += 1
                source_origin_counts[origin] += 1
                reuse_mode_counts[reuse_mode] += 1
            before_memory = previous_memory
            update = turn.get("update")

            if update is None:
                decision = "NO_OP"
                reason_code = "NO_NEW_VEHICLE_FACT"
                reason = "This turn does not commit a new or changed durable vehicle-memory fact."
                operations: list[dict[str, str]] = []
            else:
                require(
                    isinstance(update, dict), f"update must be an object: {turn_id}"
                )
                op = str(update["op"])
                require(op in {"add", "replace"}, f"unsupported pilot operation: {op}")
                slot_id = str(update["slot_id"])
                owner_id = str(update.get("owner", speaker["speaker_id"]))
                require(owner_id in speaker_names, f"unknown memory owner: {owner_id}")
                full_arguments = {
                    str(update["argument_name"]): update["value"],
                    **dict(update.get("context_arguments", {})),
                }
                validate_tool_arguments(
                    tool_schemas,
                    runtime,
                    str(update["tool_name"]),
                    full_arguments,
                )
                tool_names.add(str(update["tool_name"]))
                new_line = render_memory_line(timestamp, update)

                if op == "add":
                    require(
                        slot_id not in active_slots,
                        f"ADD reuses active slot: {slot_id}",
                    )
                    target = target_for_new_line(active_slots, owner_id)
                    content = (
                        new_line
                        if target
                        else f"{owner_header(owner_id, speaker_names)}\n{new_line}"
                    )
                else:
                    require(
                        slot_id in active_slots,
                        f"REPLACE has no active slot: {slot_id}",
                    )
                    old_record = active_slots[slot_id]
                    require(
                        old_record["owner_id"] == owner_id,
                        f"REPLACE changes owner for {slot_id}",
                    )
                    require(
                        old_record["tool_name"] == str(update["tool_name"])
                        and old_record["argument_name"] == str(update["argument_name"]),
                        f"REPLACE changes Tool attribute for {slot_id}",
                    )
                    target = old_record["line"]
                    content = new_line

                operations = [{"op": op, "target": target, "content": content}]
                previous_memory, _ = apply_operations(before_memory, operations)
                active_slots[slot_id] = {
                    "owner_id": owner_id,
                    "line": new_line,
                    "tool_name": str(update["tool_name"]),
                    "argument_name": str(update["argument_name"]),
                    "source_turn_id": turn_id,
                }
                decision = "UPDATE"
                reason_code = (
                    "NEW_VEHICLE_MEMORY" if op == "add" else "UPDATED_VEHICLE_MEMORY"
                )
                reason = str(update["reason"]).strip()
                require(reason, f"empty update reason: {turn_id}")
                operation_counts[op] += 1

            after_memory = previous_memory
            if decision == "NO_OP":
                require(
                    after_memory == before_memory, f"NO_OP changed memory: {turn_id}"
                )
            decision_counts[decision] += 1
            before_hash = text_sha256(normalize_memory(before_memory))
            after_hash = text_sha256(normalize_memory(after_memory))
            sample_suffix = f"{global_turn_index:05d}"

            patch_row = {
                "schema_version": PATCH_SCHEMA_VERSION,
                "sample_id": f"{scenario_id.lower()}:patch:{sample_suffix}",
                "scenario_index": scenario_index,
                "split": "pilot_excluded",
                "run_id": run_id,
                "global_turn_index": global_turn_index,
                "event_turn_index": event_turn_index,
                "turn_id": turn_id,
                "event_id": session_id,
                "timestamp": timestamp,
                "current_turn": {
                    "speaker_id": speaker["speaker_id"],
                    "speaker_name": speaker["speaker_name"],
                    "text": text,
                },
                "train_eligible": False,
                "input": {"previous_memory": before_memory},
                "target": {
                    "decision": decision,
                    "reason_code": reason_code,
                    "reason": reason,
                    "operations": operations,
                },
                "provenance": {
                    "source_kind": provenance_kind,
                    "source_scenario_sha256": source_hash,
                    "evidence": [
                        {
                            "source_event_id": session_id,
                            "event_turn_index": event_turn_index,
                            "turn_id": turn_id,
                            "quote": text,
                        }
                    ]
                    if decision == "UPDATE"
                    else [],
                    "before_memory_sha256": before_hash,
                    "after_memory_sha256": after_hash,
                    "source_before_memory_sha256": before_hash,
                    "source_after_memory_sha256": after_hash,
                    "grouping": "speaker_id_first_appearance",
                    **({"external_source_trace": source_trace} if source_trace else {}),
                },
            }
            snapshot = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "checkpoint_id": f"{scenario_id.lower()}:memory:{sample_suffix}",
                "scenario_index": scenario_index,
                "split": "pilot_excluded",
                "global_turn_index": global_turn_index,
                "event_turn_index": event_turn_index,
                "event_id": session_id,
                "turn_id": turn_id,
                "timestamp": timestamp,
                "decision": decision,
                "memory": after_memory,
                "memory_snapshot_sha256": after_hash,
                "active_slot_ids": list(active_slots),
            }
            dialogue_row = {
                "scenario_id": scenario_id,
                "session_id": session_id,
                "session_context": session["context"],
                "global_turn_index": global_turn_index,
                "event_turn_index": event_turn_index,
                "turn_id": turn_id,
                "timestamp": timestamp,
                "speaker_id": speaker["speaker_id"],
                "speaker_name": speaker["speaker_name"],
                "text": text,
                **({"source_trace": source_trace} if source_trace else {}),
            }

            patch_rows.append(patch_row)
            snapshot_rows.append(snapshot)
            dialogue_rows.append(dialogue_row)
            snapshots_by_turn[turn_id] = snapshot
            active_lines_by_turn[turn_id] = {
                slot_id: record["line"] for slot_id, record in active_slots.items()
            }
            global_turn_index += 1

    require(decision_counts["UPDATE"] >= 1, "scenario contains no UPDATE")
    require(operation_counts["replace"] >= 1, "scenario contains no REPLACE")

    turn_quiz_rows: list[dict[str, Any]] = []
    final_quiz_rows: list[dict[str, Any]] = []
    seen_quiz_ids: set[str] = set()
    for quiz in source.get("quizzes", []):
        quiz_id = str(quiz["quiz_id"])
        require(quiz_id not in seen_quiz_ids, f"duplicate quiz_id: {quiz_id}")
        seen_quiz_ids.add(quiz_id)
        quiz_type = str(quiz["quiz_type"])
        require(quiz_type in {"TURN", "FINAL"}, f"invalid quiz_type: {quiz_type}")
        cutoff_turn_id = str(quiz["cutoff_turn_id"])
        require(
            cutoff_turn_id in snapshots_by_turn,
            f"unknown Quiz cutoff: {cutoff_turn_id}",
        )
        snapshot = snapshots_by_turn[cutoff_turn_id]
        active_lines = active_lines_by_turn[cutoff_turn_id]
        evidence_ids = [str(value) for value in quiz["evidence_slot_ids"]]
        missing = sorted(set(evidence_ids) - set(active_lines))
        require(not missing, f"Quiz {quiz_id} references inactive slots: {missing}")
        evidence_lines = [active_lines[slot_id] for slot_id in evidence_ids]
        for line in evidence_lines:
            require(
                line in snapshot["memory"],
                f"Quiz evidence missing from memory: {quiz_id}",
            )
        reasoning_type = str(quiz["reasoning_type"])
        require(
            reasoning_type in VALID_REASONING_TYPES,
            f"invalid reasoning_type for {quiz_id}: {reasoning_type}",
        )
        calls = [dict(call) for call in quiz["gold_calls"]]
        target_state, target_state_hash = execute_quiz(runtime, tool_schemas, calls)
        checkpoint_suffix = f"{snapshot['global_turn_index']:05d}"
        row = {
            "schema_version": TURN_QUIZ_SCHEMA_VERSION
            if quiz_type == "TURN"
            else FINAL_QUIZ_SCHEMA_VERSION,
            "sample_id": f"{scenario_id.lower()}:{quiz_type.lower()}_quiz:{quiz_id}",
            "scenario_index": scenario_index,
            "split": "pilot_excluded",
            "run_id": run_id,
            "quiz_type": quiz_type,
            "quiz_id": quiz_id,
            "source_chain_id": scenario_id.lower(),
            "memory_ref": {
                "checkpoint_id": snapshot["checkpoint_id"],
                "position": "AFTER_TURN",
                "global_turn_index": snapshot["global_turn_index"],
                "turn_id": cutoff_turn_id,
                "memory_snapshot_sha256": snapshot["memory_snapshot_sha256"],
                "summary_sample_id": f"{scenario_id.lower()}:summary:{checkpoint_suffix}",
                "patch_sample_id": f"{scenario_id.lower()}:patch:{checkpoint_suffix}",
                "delta_sample_id": f"{scenario_id.lower()}:delta:{checkpoint_suffix}",
            },
            "input": {
                "query": str(quiz["query"]).strip(),
                "reasoning_type": reasoning_type,
            },
            "target": {
                "gold_calls": calls,
                "target_state": target_state,
                "target_state_sha256": target_state_hash,
            },
            "evidence": {"memory_evidence_lines": evidence_lines}
            if quiz_type == "TURN"
            else {"gold_memory": "\n".join(evidence_lines)},
            "simulator": {
                "call_count": len(calls),
                "passed": True,
            },
            "provenance": {
                "source_kind": provenance_kind,
                "source_scenario_sha256": source_hash,
                "grouped_memory_sha256": snapshot["memory_snapshot_sha256"],
            },
        }
        if quiz_type == "TURN":
            turn_quiz_rows.append(row)
        else:
            final_quiz_rows.append(row)

    require(turn_quiz_rows and final_quiz_rows, "scenario needs Turn and Final quizzes")
    final_snapshot = snapshot_rows[-1]
    require(
        all(
            row["memory_ref"]["turn_id"] == final_snapshot["turn_id"]
            for row in final_quiz_rows
        ),
        "all Final quizzes must reference the final turn",
    )

    outputs = {
        "dialogue.jsonl": dialogue_rows,
        "patch.jsonl": patch_rows,
        "memory_snapshots.jsonl": snapshot_rows,
        "turn_quiz.jsonl": turn_quiz_rows,
        "final_quiz.jsonl": final_quiz_rows,
    }
    for filename, rows in outputs.items():
        write_jsonl(output_dir / filename, rows)

    files = {
        filename: {
            "row_count": len(rows),
            "sha256": file_sha256(output_dir / filename),
        }
        for filename, rows in outputs.items()
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_index": scenario_index,
        "split": "pilot_excluded",
        "eligible_for_human_test": False,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "statistics": {
            "speaker_count": len(speakers),
            "session_count": len(seen_session_ids),
            "turn_count": len(patch_rows),
            "update_count": decision_counts["UPDATE"],
            "noop_count": decision_counts["NO_OP"],
            "add_count": operation_counts["add"],
            "replace_count": operation_counts["replace"],
            "active_memory_slot_count": len(active_slots),
            "distinct_memory_tool_count": len(tool_names),
            "turn_quiz_count": len(turn_quiz_rows),
            "final_quiz_count": len(final_quiz_rows),
            "quiz_tool_call_count": sum(
                len(row["target"]["gold_calls"])
                for row in [*turn_quiz_rows, *final_quiz_rows]
            ),
        },
        "final_memory": {
            "turn_id": final_snapshot["turn_id"],
            "sha256": final_snapshot["memory_snapshot_sha256"],
            "active_slot_ids": final_snapshot["active_slot_ids"],
        },
        "validation": {
            "source_contract": "PASS",
            "chronology_and_unique_ids": "PASS",
            "patch_replay_all_turns": "PASS",
            "noop_memory_identity": "PASS",
            "quiz_evidence_active_at_cutoff": "PASS",
            "official_tool_json_schema": "PASS",
            "official_vehicleworld_simulator": "PASS",
            "final_quiz_uses_final_snapshot": "PASS",
        },
        "files": files,
    }
    if source_dataset_counts:
        require(
            sum(source_dataset_counts.values()) == len(patch_rows),
            "all turns must carry source_trace when external adaptation is used",
        )
        manifest["external_adaptation"] = {
            "externally_sourced_turn_count": sum(source_dataset_counts.values()),
            "externally_sourced_turn_rate": sum(source_dataset_counts.values())
            / len(patch_rows),
            "dataset_turn_counts": dict(sorted(source_dataset_counts.items())),
            "origin_turn_counts": dict(sorted(source_origin_counts.items())),
            "reuse_mode_turn_counts": dict(sorted(reuse_mode_counts.items())),
            "source_catalog": deepcopy(source.get("external_sources", [])),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build(args.source, args.output_dir, args.vehiclemembench_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
