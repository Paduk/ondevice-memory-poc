"""End-to-end validation for Temporal Stage 2 and Hybrid turn labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.v1_generation import V1Stage2Artifact
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1GeneratedEventDialogue
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridArtifact,
    event_dialogue_turns,
    hybrid_memory_identity,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_scenario import (
    load_temporal_plan,
)


def exact_transition_cue(transition: dict[str, Any], text: str) -> str:
    """Return the gold plan cue only when the label turn preserves it exactly."""

    cue = str(transition.get("temporal_cue", "")).strip()
    return cue if cue and cue.casefold() in text.casefold() else ""


def validate_temporal_hybrid_run(
    *,
    stage2: V1Stage2Artifact,
    hybrid: V2HybridArtifact,
    plan_path: Path,
    dialogue_root: Path,
) -> dict[str, Any]:
    """Verify action-to-Patch semantics, cues, and baseline restoration."""

    if hybrid.source_stage2_sha256 != stage2.artifact_sha256:
        raise ValueError("Temporal Hybrid artifact belongs to another Stage 2")
    transitions = load_temporal_plan(
        plan_path,
        source_stage2_sha256=stage2.artifact_sha256,
    )
    events = {
        item.event.event_id: item.event for item in stage2.interleaved_timeline
    }
    checkpoints = {item.event_id: item for item in hybrid.event_checkpoints}
    counts: Counter[str] = Counter()
    cue_count = 0
    baseline_checks = 0
    for event_id, by_update in transitions.items():
        event = events.get(event_id)
        checkpoint = checkpoints.get(event_id)
        if event is None or checkpoint is None:
            raise ValueError(f"Temporal event is absent from Hybrid: {event_id}")
        dialogue = V1GeneratedEventDialogue.model_validate_json(
            (dialogue_root / f"{event_id}.json").read_text(encoding="utf-8")
        )
        turns = {
            turn.turn_id: turn for turn in event_dialogue_turns(stage2, dialogue)
        }
        for update_index, transition in by_update.items():
            labels = [
                label
                for label in checkpoint.turn_labels
                if update_index in label.source_update_indexes
            ]
            if len(labels) != 1:
                raise ValueError(
                    f"Temporal update {event_id}/{update_index} has "
                    f"{len(labels)} aligned labels"
                )
            label = labels[0]
            action = str(transition["temporal_action"])
            operation_kinds = [operation.op for operation in label.operations]
            if action == "temporary_override" and operation_kinds != ["add"]:
                raise ValueError("temporary_override must preserve baseline via ADD")
            if action == "end_temporary" and operation_kinds != ["delete"]:
                raise ValueError("end_temporary must only delete the override")
            if action in {
                "durable_upsert",
                "current_upsert",
                "conditional_upsert",
            } and (not operation_kinds or "delete" in operation_kinds):
                raise ValueError(f"{action} emitted an invalid Patch shape")
            turn = turns.get(label.turn_id)
            if turn is None:
                raise ValueError(f"Temporal label turn is missing: {label.turn_id}")
            cue = exact_transition_cue(transition, turn.text)
            if action in {"temporary_override", "end_temporary"}:
                if not cue:
                    raise ValueError(f"Temporal label lacks exact {action} cue")
                cue_count += 1
                baseline_event = str(transition["baseline_event_id"])
                baseline_index = int(transition["baseline_source_update_index"])
                baseline_update = events[baseline_event].preference_updates[
                    baseline_index
                ]
                baseline_identity = hybrid_memory_identity(baseline_update)
                identities = {
                    entry.identity_key for entry in checkpoint.memory_entries_after
                }
                if baseline_identity not in identities:
                    raise ValueError(f"{action} lost its durable baseline")
                if action == "end_temporary" and any(
                    entry.source_event_id
                    == event.preference_updates[update_index].supersedes_event_id
                    for entry in checkpoint.memory_entries_after
                ):
                    raise ValueError("end_temporary left its override active")
                baseline_checks += 1
            counts[action] += 1
    return {
        "transition_count": sum(counts.values()),
        "action_counts": dict(sorted(counts.items())),
        "exact_dialogue_cue_count": cue_count,
        "baseline_check_count": baseline_checks,
        "passed": True,
    }
