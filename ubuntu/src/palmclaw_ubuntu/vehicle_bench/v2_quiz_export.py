"""Deterministic Turn/Final Quiz JSONL export for Hybrid V2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.v1_reproduction import canonical_json_sha256
from palmclaw_ubuntu.vehicle_bench.v2_training_export import (
    canonical_hybrid_run_root,
    scenario_split,
)

V2_QUIZ_EXPORT_VERSION = "vehiclemembench-v2-hybrid-quiz-export-v1"
V2_TURN_QUIZ_ROW_VERSION = "vehiclemembench-v2-turn-quiz-row-v1"
V2_FINAL_QUIZ_ROW_VERSION = "vehiclemembench-v2-final-quiz-row-v1"


def export_quiz_views(
    *,
    base_root: Path,
    output_root: Path,
    scenarios: Iterable[int] = range(1, 101),
    train_end: int = 80,
    validation_end: int = 90,
    training_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Export normalized Quiz rows and validate every memory reference."""
    output_root.mkdir(parents=True, exist_ok=True)
    turn_path = output_root / "turn_quiz.jsonl"
    final_path = output_root / "final_quiz.jsonl"
    turn_tmp = turn_path.with_suffix(".jsonl.tmp")
    final_tmp = final_path.with_suffix(".jsonl.tmp")
    scenario_rows = []
    counts = Counter()
    split_counts = Counter()

    try:
        with turn_tmp.open("w", encoding="utf-8") as turn_handle, final_tmp.open(
            "w", encoding="utf-8"
        ) as final_handle:
            for scenario_index in tuple(scenarios):
                run_root = canonical_hybrid_run_root(base_root, scenario_index)
                split = scenario_split(
                    scenario_index,
                    train_end=train_end,
                    validation_end=validation_end,
                )
                scenario_result = _export_scenario_quizzes(
                    scenario_index=scenario_index,
                    split=split,
                    run_root=run_root,
                    turn_handle=turn_handle,
                    final_handle=final_handle,
                )
                scenario_rows.append(scenario_result)
                counts.update(
                    turn_quizzes=scenario_result["turn_quiz_count"],
                    final_quizzes=scenario_result["final_quiz_count"],
                    total_quizzes=(
                        scenario_result["turn_quiz_count"]
                        + scenario_result["final_quiz_count"]
                    ),
                )
                split_counts[split] += 1
        turn_tmp.replace(turn_path)
        final_tmp.replace(final_path)
    finally:
        turn_tmp.unlink(missing_ok=True)
        final_tmp.unlink(missing_ok=True)

    training_manifest_record = None
    if training_manifest_path is not None:
        resolved = training_manifest_path.resolve(strict=True)
        training_manifest_record = {
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
        }
    manifest = {
        "schema_version": V2_QUIZ_EXPORT_VERSION,
        "source_base_root": str(base_root),
        "scenario_count": len(scenario_rows),
        "split_scenario_counts": dict(sorted(split_counts.items())),
        "counts": dict(counts),
        "training_manifest": training_manifest_record,
        "files": {
            "turn_quiz": _file_record(turn_path),
            "final_quiz": _file_record(final_path),
        },
        "scenarios": scenario_rows,
    }
    manifest_path = output_root / "quiz_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _export_scenario_quizzes(
    *,
    scenario_index: int,
    split: str,
    run_root: Path,
    turn_handle: Any,
    final_handle: Any,
) -> dict[str, Any]:
    hybrid = _read_json(run_root / "hybrid.json")
    turn_artifact = _read_json(
        run_root / "turn-quiz-30-v1" / "turn-quizzes.json"
    )
    final_artifact = _read_json(run_root / "final-v1" / "stage3.json")
    labels, cutoff_checkpoints, last_turn = _hybrid_indexes(hybrid)
    labels_by_global_turn = {
        int(global_turn_index): label
        for (global_turn_index, _turn_id), label in labels.items()
    }
    run_id = run_root.name

    turn_rows = []
    for raw_quiz in turn_artifact["quizzes"]:
        raw_quiz = dict(raw_quiz)
        global_turn_index = int(raw_quiz["global_turn_index"])
        if not raw_quiz.get("turn_id"):
            label_at_turn = labels_by_global_turn.get(global_turn_index)
            if label_at_turn is None:
                raise ValueError(
                    f"Turn Quiz has no memory label: S{scenario_index} "
                    f"global_turn={global_turn_index}"
                )
            raw_quiz["turn_id"] = label_at_turn["turn_id"]
        key = (global_turn_index, raw_quiz["turn_id"])
        label = labels.get(key)
        if label is None:
            raise ValueError(f"Turn Quiz has no memory label: S{scenario_index} {key}")
        if label["after_memory_sha256"] != raw_quiz["memory_snapshot_sha256"]:
            raise ValueError(f"Turn Quiz memory hash mismatch: S{scenario_index} {key}")
        cutoff_checkpoint = cutoff_checkpoints[key]
        turn_rows.append(
            _turn_quiz_row(
                scenario_index=scenario_index,
                split=split,
                run_id=run_id,
                source_artifact_sha256=turn_artifact["artifact_sha256"],
                raw_quiz=raw_quiz,
                cutoff_checkpoint=cutoff_checkpoint,
            )
        )

    final_rows = []
    final_memory_sha256 = hybrid["final_memory_sha256"]
    for raw_quiz in final_artifact["scenario"]["final_quizzes"]:
        final_rows.append(
            _final_quiz_row(
                scenario_index=scenario_index,
                split=split,
                run_id=run_id,
                source_artifact_sha256=final_artifact["artifact_sha256"],
                raw_quiz=raw_quiz,
                last_turn_index=last_turn[0],
                last_turn_id=last_turn[1],
                final_memory_sha256=final_memory_sha256,
            )
        )

    _validate_quiz_ids(turn_rows, final_rows, scenario_index=scenario_index)
    _write_rows(turn_handle, turn_rows)
    _write_rows(final_handle, final_rows)
    return {
        "scenario_index": scenario_index,
        "split": split,
        "run_id": run_id,
        "source_hybrid_sha256": hybrid["artifact_sha256"],
        "source_turn_quiz_sha256": turn_artifact["artifact_sha256"],
        "source_final_quiz_sha256": final_artifact["artifact_sha256"],
        "turn_quiz_count": len(turn_rows),
        "final_quiz_count": len(final_rows),
        "final_memory_sha256": final_memory_sha256,
    }


def _turn_quiz_row(
    *,
    scenario_index: int,
    split: str,
    run_id: str,
    source_artifact_sha256: str,
    raw_quiz: Mapping[str, Any],
    cutoff_checkpoint: Mapping[str, str],
) -> dict[str, Any]:
    target_state = raw_quiz["target_state"]
    if canonical_json_sha256(target_state) != raw_quiz["target_state_sha256"]:
        raise ValueError(f"Turn Quiz target-state hash mismatch: {raw_quiz['quiz_id']}")
    gold_calls = _normalize_gold_calls(raw_quiz["gold_calls"])
    memory_sha256 = raw_quiz["memory_snapshot_sha256"]
    global_turn_index = raw_quiz["global_turn_index"]
    return {
        "schema_version": V2_TURN_QUIZ_ROW_VERSION,
        "sample_id": f"s{scenario_index:03d}:turn_quiz:{raw_quiz['quiz_id']}",
        "scenario_index": scenario_index,
        "split": split,
        "run_id": run_id,
        "quiz_type": "TURN",
        "quiz_id": raw_quiz["quiz_id"],
        "source_chain_id": raw_quiz["source_chain_id"],
        "memory_ref": _memory_ref(
            scenario_index=scenario_index,
            global_turn_index=global_turn_index,
            turn_id=raw_quiz["turn_id"],
            memory_sha256=memory_sha256,
        ),
        "input": {
            "query": raw_quiz["query"],
            "reasoning_type": raw_quiz["reasoning_type"],
        },
        "target": {
            "gold_calls": gold_calls,
            "target_state": target_state,
            "target_state_sha256": raw_quiz["target_state_sha256"],
        },
        "evidence": {
            "memory_evidence_lines": raw_quiz["memory_evidence_lines"],
        },
        "simulator": {
            "call_count": raw_quiz["simulator_call_count"],
            "state_changed": raw_quiz["state_changed"],
        },
        "provenance": {
            "source_artifact_sha256": source_artifact_sha256,
            "source_quiz_sha256": raw_quiz["quiz_sha256"],
            "source_checkpoint_sha256": raw_quiz["source_checkpoint_sha256"],
            "source_event_id": raw_quiz["source_event_id"],
            "cutoff_checkpoint_sha256": cutoff_checkpoint["checkpoint_sha256"],
            "cutoff_event_id": cutoff_checkpoint["event_id"],
            "event_turn_index": raw_quiz["event_turn_index"],
        },
    }


def _final_quiz_row(
    *,
    scenario_index: int,
    split: str,
    run_id: str,
    source_artifact_sha256: str,
    raw_quiz: Mapping[str, Any],
    last_turn_index: int,
    last_turn_id: str,
    final_memory_sha256: str,
) -> dict[str, Any]:
    target_state = raw_quiz["target_state"]
    if canonical_json_sha256(target_state) != raw_quiz["target_state_sha256"]:
        raise ValueError(
            f"Final Quiz target-state hash mismatch: {raw_quiz['quiz_id']}"
        )
    return {
        "schema_version": V2_FINAL_QUIZ_ROW_VERSION,
        "sample_id": f"s{scenario_index:03d}:final_quiz:{raw_quiz['quiz_id']}",
        "scenario_index": scenario_index,
        "split": split,
        "run_id": run_id,
        "quiz_type": "FINAL",
        "quiz_id": raw_quiz["quiz_id"],
        "source_chain_id": raw_quiz["source_chain_id"],
        "memory_ref": _memory_ref(
            scenario_index=scenario_index,
            global_turn_index=last_turn_index,
            turn_id=last_turn_id,
            memory_sha256=final_memory_sha256,
        ),
        "input": {
            "query": raw_quiz["query"],
            "reasoning_type": raw_quiz["reasoning_type"],
        },
        "target": {
            "gold_calls": _normalize_gold_calls(raw_quiz["gold_calls"]),
            "target_state": target_state,
            "target_state_sha256": raw_quiz["target_state_sha256"],
        },
        "evidence": {"gold_memory": raw_quiz["gold_memory"]},
        "provenance": {
            "source_artifact_sha256": source_artifact_sha256,
            "source_quiz_sha256": canonical_json_sha256(raw_quiz),
        },
    }


def _memory_ref(
    *,
    scenario_index: int,
    global_turn_index: int,
    turn_id: str | None,
    memory_sha256: str,
) -> dict[str, Any]:
    prefix = f"s{scenario_index:03d}"
    index = f"{global_turn_index:05d}"
    return {
        "checkpoint_id": f"{prefix}:memory:{index}",
        "position": "AFTER_TURN",
        "global_turn_index": global_turn_index,
        "turn_id": turn_id,
        "memory_snapshot_sha256": memory_sha256,
        "summary_sample_id": f"{prefix}:summary:{index}",
        "patch_sample_id": f"{prefix}:patch:{index}",
        "delta_sample_id": f"{prefix}:delta:{index}",
    }


def _normalize_gold_calls(
    raw_calls: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    calls = []
    for raw_call in raw_calls:
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")
        if not isinstance(name, str) or not name.startswith("carcontrol_"):
            raise ValueError("Quiz contains an invalid Tool name")
        if isinstance(arguments, Sequence) and not isinstance(arguments, (str, bytes)):
            arguments = {
                argument["name"]: argument["value"] for argument in arguments
            }
        if not isinstance(arguments, Mapping):
            raise ValueError(f"Quiz contains invalid arguments for {name}")
        calls.append({"name": name, "arguments": dict(arguments)})
    if not calls:
        raise ValueError("Quiz must contain at least one gold Tool call")
    return tuple(calls)


def _hybrid_indexes(
    hybrid: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, str], Mapping[str, Any]],
    dict[tuple[int, str], dict[str, str]],
    tuple[int, str],
]:
    labels = {}
    cutoff_checkpoints = {}
    last_turn_index = -1
    last_turn_id = ""
    for checkpoint in hybrid["event_checkpoints"]:
        for label in checkpoint["turn_labels"]:
            key = (label["global_turn_index"], label["turn_id"])
            if key in labels:
                raise ValueError(f"Duplicate Hybrid turn label: {key}")
            labels[key] = label
            cutoff_checkpoints[key] = {
                "event_id": checkpoint["event_id"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            }
            if label["global_turn_index"] > last_turn_index:
                last_turn_index = label["global_turn_index"]
                last_turn_id = label["turn_id"]
    if last_turn_index < 0:
        raise ValueError("Hybrid artifact contains no Turn label")
    return labels, cutoff_checkpoints, (last_turn_index, last_turn_id)


def _validate_quiz_ids(
    turn_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    *,
    scenario_index: int,
) -> None:
    sample_ids = [row["sample_id"] for row in (*turn_rows, *final_rows)]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate Quiz sample ID in S{scenario_index}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_rows(handle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    line_count = 0
    with path.open("rb") as handle:
        for _ in handle:
            line_count += 1
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "line_count": line_count,
        "bytes": path.stat().st_size,
    }
