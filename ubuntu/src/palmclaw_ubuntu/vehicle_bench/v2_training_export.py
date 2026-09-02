"""Deterministic Summary and appended-Delta training views for Hybrid V2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.providers import apply_recursive_summary_patch
from palmclaw_ubuntu.vehicle_bench.v1_generation import V1Stage2Artifact
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1GeneratedEventDialogue
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridArtifact,
    event_dialogue_turns,
    text_sha256,
)

V2_TRAINING_EXPORT_VERSION = "vehiclemembench-v2-hybrid-training-export-v2"
V2_SUMMARY_VIEW_VERSION = "vehiclemembench-v2-summary-sft-v1"
V2_PATCH_VIEW_VERSION = "vehiclemembench-v2-patch-sft-v1"
V2_DELTA_VIEW_VERSION = "vehiclemembench-v2-appended-delta-sft-v1"
V2_COMPACTION_VIEW_VERSION = "vehiclemembench-v2-5-update-compaction-sft-v1"
V2_BATCH_TRAINING_EXPORT_VERSION = (
    "vehiclemembench-v2-hybrid-batch-training-export-v1"
)
V2_SUMMARY_BATCH_VIEW_VERSION = "vehiclemembench-v2-summary-batch-sft-v1"
V2_PATCH_BATCH_VIEW_VERSION = "vehiclemembench-v2-patch-batch-sft-v1"


@dataclass(frozen=True)
class CanonicalTrainingTurn:
    global_turn_index: int
    event_turn_index: int
    turn_id: str
    event_id: str
    timestamp: str
    speaker_id: str
    speaker_name: str
    text: str
    decision: str
    reason_code: str
    reason: str
    operations: tuple[dict[str, str], ...]
    evidence: tuple[dict[str, Any], ...]
    source_update_indexes: tuple[int, ...]
    before_memory_sha256: str
    after_memory_sha256: str
    after_memory: str | None


def canonical_hybrid_run_root(base_root: Path, scenario_index: int) -> Path:
    if not 1 <= scenario_index <= 100:
        raise ValueError("Hybrid training export supports scenarios 1..100")
    if scenario_index == 1:
        name = "hybrid-full-terra-s1-r2"
    elif scenario_index in {2, 4, 5}:
        name = f"hybrid-pilot-terra-s{scenario_index:02d}-r2"
    else:
        name = f"hybrid-pilot-terra-s{scenario_index:02d}-r1"
    return base_root / name


def scenario_split(
    scenario_index: int,
    *,
    train_end: int = 80,
    validation_end: int = 90,
) -> str:
    if scenario_index <= train_end:
        return "train"
    if scenario_index <= validation_end:
        return "validation"
    return "test"


def load_canonical_training_turns(
    run_root: Path,
) -> tuple[V2HybridArtifact, tuple[CanonicalTrainingTurn, ...]]:
    hybrid = V2HybridArtifact.model_validate_json(
        (run_root / "hybrid.json").read_text(encoding="utf-8")
    )
    source_root = None
    stage2 = None
    candidates = [run_root]
    if run_root.name.endswith("-r2"):
        candidates.append(run_root.with_name(run_root.name[:-1] + "1"))
    for candidate in candidates:
        stage2_path = candidate / "stage2-v2-anchored.json"
        if not stage2_path.exists():
            continue
        candidate_stage2 = V1Stage2Artifact.model_validate_json(
            stage2_path.read_text(encoding="utf-8")
        )
        if candidate_stage2.artifact_sha256 == hybrid.source_stage2_sha256:
            stage2 = candidate_stage2
            source_root = candidate
            break
    if stage2 is None or source_root is None:
        raise ValueError(f"No matching Stage2 source for Hybrid run: {run_root}")

    turns = []
    for checkpoint in hybrid.event_checkpoints:
        dialogue = V1GeneratedEventDialogue.model_validate_json(
            (source_root / "dialogues" / f"{checkpoint.event_id}.json").read_text(
                encoding="utf-8"
            )
        )
        materialized = event_dialogue_turns(stage2, dialogue)
        by_turn_id = {turn.turn_id: turn for turn in materialized}
        for label in checkpoint.turn_labels:
            turn = by_turn_id.get(label.turn_id)
            if turn is None:
                raise ValueError(f"Missing materialized turn {label.turn_id}")
            turns.append(
                CanonicalTrainingTurn(
                    global_turn_index=label.global_turn_index,
                    event_turn_index=label.event_turn_index,
                    turn_id=label.turn_id,
                    event_id=checkpoint.event_id,
                    timestamp=turn.timestamp,
                    speaker_id=turn.speaker_id,
                    speaker_name=turn.speaker_name,
                    text=turn.text,
                    decision=label.decision,
                    reason_code=label.reason_code,
                    reason=label.reason,
                    operations=tuple(
                        operation.model_dump(mode="json")
                        for operation in label.operations
                    ),
                    evidence=tuple(
                        evidence.model_dump(mode="json")
                        for evidence in label.evidence
                    ),
                    source_update_indexes=label.source_update_indexes,
                    before_memory_sha256=label.before_memory_sha256,
                    after_memory_sha256=label.after_memory_sha256,
                    after_memory=label.after_memory,
                )
            )
    return hybrid, tuple(turns)


def build_training_views(
    *,
    scenario_index: int,
    run_id: str,
    source_hybrid_sha256: str,
    turns: Sequence[CanonicalTrainingTurn],
    split: str,
    compaction_interval: int = 5,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    if compaction_interval < 1:
        raise ValueError("Compaction interval must be positive")

    summary_rows = []
    patch_rows = []
    delta_rows = []
    compaction_rows = []
    canonical_memory = ""
    base_summary = ""
    pending_deltas: list[dict[str, Any]] = []
    updates_since_compaction = 0
    previous_global_index = -1

    for turn in turns:
        if turn.global_turn_index != previous_global_index + 1:
            raise ValueError("Training turns must have contiguous global indexes")
        previous_global_index = turn.global_turn_index
        if text_sha256(canonical_memory) != turn.before_memory_sha256:
            raise ValueError(f"Before-memory mismatch at {turn.turn_id}")
        replay_before = _apply_delta_batches(base_summary, pending_deltas)
        if replay_before != canonical_memory:
            raise ValueError(f"Pending Delta replay mismatch at {turn.turn_id}")

        common = {
            "scenario_index": scenario_index,
            "split": split,
            "run_id": run_id,
            "source_hybrid_sha256": source_hybrid_sha256,
            "global_turn_index": turn.global_turn_index,
            "event_turn_index": turn.event_turn_index,
            "turn_id": turn.turn_id,
            "event_id": turn.event_id,
            "timestamp": turn.timestamp,
            "current_turn": {
                "speaker_id": turn.speaker_id,
                "speaker_name": turn.speaker_name,
                "text": turn.text,
            },
        }
        target_common = {
            "decision": turn.decision,
            "reason_code": turn.reason_code,
            "reason": turn.reason,
        }

        if turn.decision == "UPDATE":
            canonical_after, _ = apply_recursive_summary_patch(
                canonical_memory,
                list(turn.operations),
            )
            if turn.after_memory is None or canonical_after != turn.after_memory:
                raise ValueError(f"Patch/after-memory mismatch at {turn.turn_id}")
        elif turn.decision == "NO_OP":
            if turn.operations or turn.after_memory is not None:
                raise ValueError(f"Invalid NO_OP payload at {turn.turn_id}")
            canonical_after = canonical_memory
        else:
            raise ValueError(f"Unknown decision at {turn.turn_id}: {turn.decision}")
        if text_sha256(canonical_after) != turn.after_memory_sha256:
            raise ValueError(f"After-memory hash mismatch at {turn.turn_id}")

        summary_rows.append(
            {
                "schema_version": V2_SUMMARY_VIEW_VERSION,
                "sample_id": (
                    f"s{scenario_index:03d}:summary:"
                    f"{turn.global_turn_index:05d}"
                ),
                **common,
                "input": {"previous_memory": canonical_memory},
                "target": {**target_common, "next_memory": canonical_after},
                "provenance": _provenance(turn),
            }
        )
        patch_rows.append(
            {
                "schema_version": V2_PATCH_VIEW_VERSION,
                "sample_id": (
                    f"s{scenario_index:03d}:patch:"
                    f"{turn.global_turn_index:05d}"
                ),
                **common,
                "input": {"previous_memory": canonical_memory},
                "target": {
                    **target_common,
                    "operations": turn.operations,
                },
                "provenance": _provenance(turn),
            }
        )
        delta_rows.append(
            {
                "schema_version": V2_DELTA_VIEW_VERSION,
                "sample_id": (
                    f"s{scenario_index:03d}:delta:"
                    f"{turn.global_turn_index:05d}"
                ),
                **common,
                "input": {
                    "base_summary": base_summary,
                    "pending_deltas": tuple(pending_deltas),
                    "updates_since_compaction": updates_since_compaction,
                },
                "target": {
                    **target_common,
                    "operations": turn.operations,
                },
                "provenance": _provenance(turn),
            }
        )

        canonical_memory = canonical_after
        if turn.decision == "UPDATE":
            pending_deltas.append(
                {
                    "turn_id": turn.turn_id,
                    "operations": turn.operations,
                }
            )
            updates_since_compaction += 1
            if updates_since_compaction == compaction_interval:
                compaction_rows.append(
                    _compaction_row(
                        scenario_index=scenario_index,
                        split=split,
                        run_id=run_id,
                        source_hybrid_sha256=source_hybrid_sha256,
                        turn=turn,
                        base_summary=base_summary,
                        pending_deltas=pending_deltas,
                        next_summary=canonical_memory,
                        trigger="UPDATE_INTERVAL",
                    )
                )
                base_summary = canonical_memory
                pending_deltas = []
                updates_since_compaction = 0

    if pending_deltas:
        turn = turns[-1]
        compaction_rows.append(
            _compaction_row(
                scenario_index=scenario_index,
                split=split,
                run_id=run_id,
                source_hybrid_sha256=source_hybrid_sha256,
                turn=turn,
                base_summary=base_summary,
                pending_deltas=pending_deltas,
                next_summary=canonical_memory,
                trigger="FINAL_FLUSH",
            )
        )
        base_summary = canonical_memory
        pending_deltas = []

    if _apply_delta_batches(base_summary, pending_deltas) != canonical_memory:
        raise ValueError("Final compacted memory differs from canonical memory")
    return (
        tuple(summary_rows),
        tuple(patch_rows),
        tuple(delta_rows),
        tuple(compaction_rows),
    )


def export_training_views(
    *,
    base_root: Path,
    output_root: Path,
    scenarios: Iterable[int] = range(1, 101),
    compaction_interval: int = 5,
    train_end: int = 80,
    validation_end: int = 90,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.jsonl"
    patch_path = output_root / "patch.jsonl"
    delta_path = output_root / "delta.jsonl"
    compaction_path = output_root / "compaction.jsonl"
    scenario_rows = []
    counts = Counter()
    split_counts = Counter()

    with (
        summary_path.open("w", encoding="utf-8") as summary_handle,
        patch_path.open("w", encoding="utf-8") as patch_handle,
        delta_path.open("w", encoding="utf-8") as delta_handle,
        compaction_path.open("w", encoding="utf-8") as compaction_handle,
    ):
        for scenario_index in tuple(scenarios):
            run_root = canonical_hybrid_run_root(base_root, scenario_index)
            hybrid, turns = load_canonical_training_turns(run_root)
            split = scenario_split(
                scenario_index,
                train_end=train_end,
                validation_end=validation_end,
            )
            (
                summary_rows,
                patch_rows,
                delta_rows,
                compaction_rows,
            ) = build_training_views(
                scenario_index=scenario_index,
                run_id=run_root.name,
                source_hybrid_sha256=hybrid.artifact_sha256,
                turns=turns,
                split=split,
                compaction_interval=compaction_interval,
            )
            _write_rows(summary_handle, summary_rows)
            _write_rows(patch_handle, patch_rows)
            _write_rows(delta_handle, delta_rows)
            _write_rows(compaction_handle, compaction_rows)
            updates = sum(turn.decision == "UPDATE" for turn in turns)
            operations = sum(len(turn.operations) for turn in turns)
            scenario_rows.append(
                {
                    "scenario_index": scenario_index,
                    "split": split,
                    "run_id": run_root.name,
                    "source_hybrid_sha256": hybrid.artifact_sha256,
                    "turn_count": len(turns),
                    "update_count": updates,
                    "operation_count": operations,
                    "compaction_count": len(compaction_rows),
                    "final_memory_sha256": hybrid.final_memory_sha256,
                }
            )
            counts.update(
                turns=len(turns),
                updates=updates,
                no_ops=len(turns) - updates,
                operations=operations,
                compactions=len(compaction_rows),
            )
            split_counts[split] += 1

    manifest = {
        "schema_version": V2_TRAINING_EXPORT_VERSION,
        "source_base_root": str(base_root),
        "scenario_count": len(scenario_rows),
        "split_scenario_counts": dict(sorted(split_counts.items())),
        "compaction_interval_updates": compaction_interval,
        "counts": dict(counts),
        "files": {
            "summary": _file_record(summary_path),
            "patch": _file_record(patch_path),
            "delta": _file_record(delta_path),
            "compaction": _file_record(compaction_path),
        },
        "scenarios": scenario_rows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_batch_training_views(
    *,
    scenario_index: int,
    run_id: str,
    source_hybrid_sha256: str,
    turns: Sequence[CanonicalTrainingTurn],
    split: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Build lossless end-of-day Summary and ordered-Patch training views."""

    summary_rows = []
    patch_rows = []
    canonical_memory = ""
    previous_global_index = -1
    previous_date = ""

    for batch_index, batch_turns in enumerate(_turns_by_date(turns)):
        batch_date = _calendar_date(batch_turns[0].timestamp)
        if previous_date and batch_date < previous_date:
            raise ValueError("Training turn dates must be non-decreasing")
        previous_date = batch_date
        before_memory = canonical_memory
        before_memory_sha256 = text_sha256(before_memory)
        operations: list[dict[str, str]] = []
        update_sources = []

        for turn in batch_turns:
            if turn.global_turn_index != previous_global_index + 1:
                raise ValueError("Training turns must have contiguous global indexes")
            previous_global_index = turn.global_turn_index
            if text_sha256(canonical_memory) != turn.before_memory_sha256:
                raise ValueError(f"Before-memory mismatch at {turn.turn_id}")

            if turn.decision == "UPDATE":
                operation_offset = len(operations)
                canonical_after, _ = apply_recursive_summary_patch(
                    canonical_memory,
                    list(turn.operations),
                )
                if turn.after_memory is None or canonical_after != turn.after_memory:
                    raise ValueError(f"Patch/after-memory mismatch at {turn.turn_id}")
                operations.extend(dict(operation) for operation in turn.operations)
                update_sources.append(
                    {
                        "global_turn_index": turn.global_turn_index,
                        "turn_id": turn.turn_id,
                        "reason_code": turn.reason_code,
                        "reason": turn.reason,
                        "operation_start_index": operation_offset,
                        "operation_count": len(turn.operations),
                        "source_update_indexes": turn.source_update_indexes,
                        "evidence": turn.evidence,
                    }
                )
            elif turn.decision == "NO_OP":
                if turn.operations or turn.after_memory is not None:
                    raise ValueError(f"Invalid NO_OP payload at {turn.turn_id}")
                canonical_after = canonical_memory
            else:
                raise ValueError(
                    f"Unknown decision at {turn.turn_id}: {turn.decision}"
                )
            if text_sha256(canonical_after) != turn.after_memory_sha256:
                raise ValueError(f"After-memory hash mismatch at {turn.turn_id}")
            canonical_memory = canonical_after

        if operations:
            replayed, _ = apply_recursive_summary_patch(before_memory, operations)
        else:
            replayed = before_memory
        if replayed != canonical_memory:
            raise ValueError(f"Batch Patch replay mismatch on {batch_date}")

        update_count = len(update_sources)
        decision = "UPDATE" if update_count else "NO_OP"
        turns_payload = tuple(
            {
                "global_turn_index": turn.global_turn_index,
                "event_turn_index": turn.event_turn_index,
                "turn_id": turn.turn_id,
                "event_id": turn.event_id,
                "timestamp": turn.timestamp,
                "speaker_id": turn.speaker_id,
                "speaker_name": turn.speaker_name,
                "text": turn.text,
            }
            for turn in batch_turns
        )
        common = {
            "scenario_index": scenario_index,
            "split": split,
            "run_id": run_id,
            "source_hybrid_sha256": source_hybrid_sha256,
            # The training catalog expects one monotonically increasing identity
            # per sample.  For the batch views, one sample is one calendar date.
            "global_turn_index": batch_index,
            "turn_id": f"batch:{batch_date}",
            "timestamp": batch_turns[-1].timestamp,
            "batch_index": batch_index,
            "batch_date": batch_date,
            "start_global_turn_index": batch_turns[0].global_turn_index,
            "end_global_turn_index": batch_turns[-1].global_turn_index,
            "start_turn_id": batch_turns[0].turn_id,
            "end_turn_id": batch_turns[-1].turn_id,
            "input": {
                "previous_memory": before_memory,
                "turns": turns_payload,
            },
        }
        target_common = {
            "decision": decision,
            "update_count": update_count,
            "no_op_count": len(batch_turns) - update_count,
        }
        provenance = {
            "source_turn_count": len(batch_turns),
            "source_summary_sample_range": {
                "start": (
                    f"s{scenario_index:03d}:summary:"
                    f"{batch_turns[0].global_turn_index:05d}"
                ),
                "end": (
                    f"s{scenario_index:03d}:summary:"
                    f"{batch_turns[-1].global_turn_index:05d}"
                ),
            },
            "source_patch_sample_range": {
                "start": (
                    f"s{scenario_index:03d}:patch:"
                    f"{batch_turns[0].global_turn_index:05d}"
                ),
                "end": (
                    f"s{scenario_index:03d}:patch:"
                    f"{batch_turns[-1].global_turn_index:05d}"
                ),
            },
            "update_sources": tuple(update_sources),
            "before_memory_sha256": before_memory_sha256,
            "after_memory_sha256": text_sha256(canonical_memory),
        }
        sample_suffix = f"{batch_index:03d}:{batch_date}"
        summary_rows.append(
            {
                "schema_version": V2_SUMMARY_BATCH_VIEW_VERSION,
                "sample_id": f"s{scenario_index:03d}:summary_batch:{sample_suffix}",
                **common,
                "target": {
                    **target_common,
                    "next_memory": canonical_memory,
                },
                "provenance": provenance,
            }
        )
        patch_rows.append(
            {
                "schema_version": V2_PATCH_BATCH_VIEW_VERSION,
                "sample_id": f"s{scenario_index:03d}:patch_batch:{sample_suffix}",
                **common,
                "target": {
                    **target_common,
                    "operations": tuple(operations),
                },
                "provenance": provenance,
            }
        )

    return tuple(summary_rows), tuple(patch_rows)


def export_batch_training_views(
    *,
    base_root: Path,
    output_root: Path,
    scenarios: Iterable[int] = range(1, 101),
    train_end: int = 80,
    validation_end: int = 90,
) -> dict[str, Any]:
    """Export date-batched views without modifying the turn-wise files."""

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary_batch.jsonl"
    patch_path = output_root / "patch_batch.jsonl"
    summary_temp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    patch_temp = patch_path.with_suffix(patch_path.suffix + ".tmp")
    scenario_rows = []
    counts = Counter()
    split_counts = Counter()

    try:
        with (
            summary_temp.open("w", encoding="utf-8") as summary_handle,
            patch_temp.open("w", encoding="utf-8") as patch_handle,
        ):
            for scenario_index in tuple(scenarios):
                run_root = canonical_hybrid_run_root(base_root, scenario_index)
                hybrid, turns = load_canonical_training_turns(run_root)
                split = scenario_split(
                    scenario_index,
                    train_end=train_end,
                    validation_end=validation_end,
                )
                summary_rows, patch_rows = build_batch_training_views(
                    scenario_index=scenario_index,
                    run_id=run_root.name,
                    source_hybrid_sha256=hybrid.artifact_sha256,
                    turns=turns,
                    split=split,
                )
                if not summary_rows or not patch_rows:
                    raise ValueError(f"Scenario {scenario_index} has no batches")
                if (
                    summary_rows[-1]["provenance"]["after_memory_sha256"]
                    != hybrid.final_memory_sha256
                ):
                    raise ValueError(
                        f"Final batch memory mismatch for scenario {scenario_index}"
                    )
                _write_rows(summary_handle, summary_rows)
                _write_rows(patch_handle, patch_rows)

                update_batches = sum(
                    row["target"]["decision"] == "UPDATE"
                    for row in patch_rows
                )
                source_updates = sum(
                    row["target"]["update_count"] for row in patch_rows
                )
                operations = sum(
                    len(row["target"]["operations"]) for row in patch_rows
                )
                multi_update_batches = sum(
                    row["target"]["update_count"] > 1 for row in patch_rows
                )
                scenario_rows.append(
                    {
                        "scenario_index": scenario_index,
                        "split": split,
                        "run_id": run_root.name,
                        "source_hybrid_sha256": hybrid.artifact_sha256,
                        "turn_count": len(turns),
                        "batch_count": len(patch_rows),
                        "update_batch_count": update_batches,
                        "no_op_batch_count": len(patch_rows) - update_batches,
                        "source_update_turn_count": source_updates,
                        "operation_count": operations,
                        "multi_update_batch_count": multi_update_batches,
                        "max_updates_in_batch": max(
                            row["target"]["update_count"] for row in patch_rows
                        ),
                        "final_memory_sha256": hybrid.final_memory_sha256,
                    }
                )
                counts.update(
                    turns=len(turns),
                    batches=len(patch_rows),
                    update_batches=update_batches,
                    no_op_batches=len(patch_rows) - update_batches,
                    source_update_turns=source_updates,
                    source_no_op_turns=len(turns) - source_updates,
                    operations=operations,
                    multi_update_batches=multi_update_batches,
                )
                split_counts[split] += 1
        summary_temp.replace(summary_path)
        patch_temp.replace(patch_path)
    finally:
        summary_temp.unlink(missing_ok=True)
        patch_temp.unlink(missing_ok=True)

    source_files = {}
    for name in ("summary.jsonl", "patch.jsonl", "manifest.json"):
        path = output_root / name
        if path.exists():
            source_files[name] = _file_record(path)
    multitask_files = {}
    for name in ("quiz_sft.jsonl", "quiz_sft_manifest.json", "vehicle_tools.json"):
        path = output_root / name
        if path.exists():
            multitask_files[name] = _file_record(path)
    quiz_sft_counts = None
    quiz_sft_manifest_path = output_root / "quiz_sft_manifest.json"
    if quiz_sft_manifest_path.exists():
        quiz_sft_counts = json.loads(
            quiz_sft_manifest_path.read_text(encoding="utf-8")
        ).get("counts")
    manifest = {
        "schema_version": V2_BATCH_TRAINING_EXPORT_VERSION,
        "source_base_root": str(base_root),
        "batch_key": "calendar_date",
        "scenario_count": len(scenario_rows),
        "split_scenario_counts": dict(sorted(split_counts.items())),
        "counts": dict(counts),
        "source_turnwise_files": source_files,
        "multitask": {
            "strategy": "shared_quiz_tool_call_sft",
            "training_target": "gold_tool_calls",
            "evaluation_metrics": ["ESM", "Tool F1", "Argument Exact"],
            "memory_train_scenarios": "S15-S80",
            "quiz_train_scenarios": "S15-S80",
            "validation_scenarios": "S81-S85",
            "test_scenarios": "S86-S100",
            "quiz_sft_counts": quiz_sft_counts,
            "files": multitask_files,
        },
        "files": {
            "summary_batch": _file_record(summary_path),
            "patch_batch": _file_record(patch_path),
        },
        "scenarios": scenario_rows,
    }
    manifest_path = output_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _turns_by_date(
    turns: Sequence[CanonicalTrainingTurn],
) -> tuple[tuple[CanonicalTrainingTurn, ...], ...]:
    batches: list[tuple[CanonicalTrainingTurn, ...]] = []
    current_date = None
    current_turns: list[CanonicalTrainingTurn] = []
    for turn in turns:
        turn_date = _calendar_date(turn.timestamp)
        if current_date is None or turn_date == current_date:
            current_date = turn_date
            current_turns.append(turn)
            continue
        batches.append(tuple(current_turns))
        current_date = turn_date
        current_turns = [turn]
    if current_turns:
        batches.append(tuple(current_turns))
    return tuple(batches)


def _calendar_date(timestamp: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", timestamp):
        raise ValueError(f"Invalid ISO timestamp: {timestamp}")
    return timestamp[:10]


def _provenance(turn: CanonicalTrainingTurn) -> dict[str, Any]:
    return {
        "source_update_indexes": turn.source_update_indexes,
        "evidence": turn.evidence,
        "before_memory_sha256": turn.before_memory_sha256,
        "after_memory_sha256": turn.after_memory_sha256,
    }


def _compaction_row(
    *,
    scenario_index: int,
    split: str,
    run_id: str,
    source_hybrid_sha256: str,
    turn: CanonicalTrainingTurn,
    base_summary: str,
    pending_deltas: Sequence[Mapping[str, Any]],
    next_summary: str,
    trigger: str,
) -> dict[str, Any]:
    replayed = _apply_delta_batches(base_summary, pending_deltas)
    if replayed != next_summary:
        raise ValueError(f"Compaction replay mismatch at {turn.turn_id}")
    return {
        "schema_version": V2_COMPACTION_VIEW_VERSION,
        "sample_id": (
            f"s{scenario_index:03d}:compact:{turn.global_turn_index:05d}:"
            f"{trigger.lower()}"
        ),
        "scenario_index": scenario_index,
        "split": split,
        "run_id": run_id,
        "source_hybrid_sha256": source_hybrid_sha256,
        "global_turn_index": turn.global_turn_index,
        "turn_id": turn.turn_id,
        "event_id": turn.event_id,
        "trigger": trigger,
        "input": {
            "base_summary": base_summary,
            "pending_deltas": tuple(pending_deltas),
        },
        "target": {
            "next_summary": next_summary,
            "next_summary_sha256": text_sha256(next_summary),
        },
    }


def _apply_delta_batches(
    base_summary: str,
    pending_deltas: Sequence[Mapping[str, Any]],
) -> str:
    memory = base_summary
    for batch in pending_deltas:
        memory, _ = apply_recursive_summary_patch(
            memory,
            list(batch["operations"]),
        )
    return memory


def _write_rows(handle, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    line_count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            line_count += 1
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "line_count": line_count,
        "bytes": path.stat().st_size,
    }
