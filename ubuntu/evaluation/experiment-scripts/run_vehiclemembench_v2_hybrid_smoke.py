#!/usr/bin/env python3
"""Run resumable event-batched/turn-labeled VehicleMemBench V2 Hybrid generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    V1_DIALOGUE_PROMPT_VERSION,
    OpenAIV1DialogueGenerationModel,
    V1GeneratedEventDialogue,
    build_dialogue_generation_contexts,
    dialogue_turn_target,
    validate_dialogue_payload,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2_HYBRID_ALIGNMENT_PROMPT_VERSION,
    OpenAIV2HybridEventAlignmentModel,
    V2HybridEventCheckpoint,
    build_hybrid_artifact,
    build_hybrid_event_checkpoint,
    deterministic_empty_alignment,
    event_dialogue_turns,
    event_sha256,
    write_hybrid_artifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_anchor import (
    build_deterministic_anchor_alignment,
    load_temporal_anchor_plan,
    validate_dialogue_contains_anchors,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_hybrid_validation import (
    validate_temporal_hybrid_run,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_scenario import (
    V2_TEMPORAL_ALIGNMENT_INSTRUCTIONS,
    V2_TEMPORAL_ALIGNMENT_PROMPT_VERSION,
    V2_TEMPORAL_DIALOGUE_INSTRUCTIONS,
    V2_TEMPORAL_DIALOGUE_PROMPT_VERSION,
    V2_TEMPORAL_EVENT_PROMPT_VERSIONS,
    load_temporal_plan,
    normalize_temporal_hybrid_alignment,
    validate_temporal_dialogue_cues,
)

DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v4/stage2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/"
    "hybrid-smoke-terra-s1-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dialogue-root",
        type=Path,
        help="Optional shared Stage 3 dialogue checkpoint directory.",
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--event-limit",
        type=int,
        help="Process only the first N events while preserving causal checkpoints.",
    )
    selection.add_argument(
        "--single-event-index",
        type=int,
        help=(
            "Run one isolated fresh event smoke. Allowed only when all earlier "
            "events contain no preference update."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    stage2 = V1Stage2Artifact.model_validate_json(
        args.stage2_path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    contexts = build_dialogue_generation_contexts(stage2)
    reason_profile = (
        "state_evolution"
        if stage2.event_chains.prompt_version
        in {
            V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION,
            *V2_TEMPORAL_EVENT_PROMPT_VERSIONS,
        }
        else "legacy"
    )
    if args.event_limit is not None and args.event_limit < 1:
        raise ValueError("--event-limit must be positive")
    if args.single_event_index is not None:
        return _run_single_event_smoke(
            args,
            stage2=stage2,
            contexts=contexts,
        )
    selected = contexts[: args.event_limit] if args.event_limit else contexts
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dialogue_root = (
        args.dialogue_root.expanduser().resolve()
        if args.dialogue_root is not None
        else output_root / "dialogues"
    )
    dialogue_root.mkdir(parents=True, exist_ok=True)
    event_root = output_root / "events"
    event_root.mkdir(parents=True, exist_ok=True)
    temporal_transitions = load_temporal_plan(
        output_root / "temporal-plan.json",
        source_stage2_sha256=stage2.artifact_sha256,
        required=False,
    )
    temporal_anchor_artifact = None
    temporal_anchors = {}
    if temporal_transitions:
        temporal_plan = json.loads(
            (output_root / "temporal-plan.json").read_text(encoding="utf-8")
        )
        temporal_anchor_artifact, temporal_anchors = load_temporal_anchor_plan(
            output_root / "temporal-anchor-plan.json",
            source_stage2_sha256=stage2.artifact_sha256,
            source_temporal_plan_sha256=str(temporal_plan["artifact_sha256"]),
        )

    dialogue_model = OpenAIV1DialogueGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
        additional_instructions=(
            V2_TEMPORAL_DIALOGUE_INSTRUCTIONS if temporal_transitions else None
        ),
        prompt_version=(
            V2_TEMPORAL_DIALOGUE_PROMPT_VERSION
            if temporal_transitions
            else V1_DIALOGUE_PROMPT_VERSION
        ),
    )
    alignment_model = OpenAIV2HybridEventAlignmentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
        additional_instructions=(
            V2_TEMPORAL_ALIGNMENT_INSTRUCTIONS if temporal_transitions else None
        ),
        prompt_version=(
            V2_TEMPORAL_ALIGNMENT_PROMPT_VERSION
            if temporal_transitions
            else V2_HYBRID_ALIGNMENT_PROMPT_VERSION
        ),
        normalize_earlier_confirmation=not bool(temporal_transitions),
    )
    checkpoints: list[V2HybridEventCheckpoint] = []
    previous_entries = ()
    global_turn_offset = 0
    dialogues: list[V1GeneratedEventDialogue] = []

    for context in selected:
        dialogue_path = dialogue_root / f"{context.event.event_id}.json"
        dialogue = _load_or_generate_dialogue(
            dialogue_path,
            stage2=stage2,
            context=context,
            model=dialogue_model,
            max_attempts=args.max_attempts,
            temporal_transitions=temporal_transitions.get(
                context.event.event_id
            ),
            temporal_anchors=temporal_anchors.get(context.event.event_id, ()),
        )
        dialogues.append(dialogue)
        turns = event_dialogue_turns(stage2, dialogue)
        checkpoint_path = event_root / (
            f"{context.timeline_index:03d}-{context.event.event_id}.json"
        )
        if checkpoint_path.exists():
            checkpoint = V2HybridEventCheckpoint.model_validate_json(
                checkpoint_path.read_text(encoding="utf-8")
            )
            rebuilt = build_hybrid_event_checkpoint(
                event=context.event,
                timeline_index=context.timeline_index,
                dialogue_turns=turns,
                generated_alignment=checkpoint.generated_alignment,
                previous_entries=previous_entries,
                personas=stage2.persona_group.payload.personas,
                global_turn_offset=global_turn_offset,
                chain_kind=context.chain_kind,
                reason_profile=reason_profile,
                temporal_transitions_by_update=temporal_transitions.get(
                    context.event.event_id
                ),
            )
            if rebuilt != checkpoint:
                raise ValueError(f"stale Hybrid event checkpoint: {checkpoint_path}")
        else:
            previous_memory = (
                checkpoints[-1].after_memory if checkpoints else ""
            )
            if temporal_anchor_artifact is not None:
                generated_alignment = build_deterministic_anchor_alignment(
                    context.event,
                    turns,
                    temporal_anchors.get(context.event.event_id, ()),
                    previous_memory=previous_memory,
                )
            else:
                generated_alignment = _generate_alignment(
                    context.event,
                    turns,
                    previous_memory=previous_memory,
                    model=alignment_model,
                    max_attempts=args.max_attempts,
                    temporal_transitions=temporal_transitions.get(
                        context.event.event_id
                    ),
                )
            checkpoint = build_hybrid_event_checkpoint(
                event=context.event,
                timeline_index=context.timeline_index,
                dialogue_turns=turns,
                generated_alignment=generated_alignment,
                previous_entries=previous_entries,
                personas=stage2.persona_group.payload.personas,
                global_turn_offset=global_turn_offset,
                chain_kind=context.chain_kind,
                reason_profile=reason_profile,
                temporal_transitions_by_update=temporal_transitions.get(
                    context.event.event_id
                ),
            )
            _write_json(checkpoint_path, checkpoint.model_dump(mode="json"))
        checkpoints.append(checkpoint)
        previous_entries = checkpoint.memory_entries_after
        global_turn_offset += len(turns)

    artifact = build_hybrid_artifact(stage2, checkpoints)
    artifact_path = write_hybrid_artifact(output_root, artifact)
    temporal_audit = None
    if temporal_transitions and artifact.audit.completed:
        temporal_audit = validate_temporal_hybrid_run(
            stage2=stage2,
            hybrid=artifact,
            plan_path=output_root / "temporal-plan.json",
            dialogue_root=dialogue_root,
        )
        _write_json(
            output_root / "temporal-hybrid-audit.json",
            {
                "schema_version": "vehiclemembench-v2-temporal-hybrid-audit-v1",
                "source_stage2_sha256": stage2.artifact_sha256,
                "source_hybrid_sha256": artifact.artifact_sha256,
                "audit": temporal_audit,
            },
        )
    state_evolution_audit = None
    if reason_profile == "state_evolution" and artifact.audit.completed:
        state_evolution_audit = _write_state_evolution_label_audit(
            output_root,
            artifact,
            minimums=(
                {"add": 10, "replace": 1, "delete": 1}
                if temporal_transitions
                else None
            ),
        )
    return {
        "status": "completed" if artifact.audit.completed else "partial",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact.artifact_sha256,
        "audit": artifact.audit.model_dump(mode="json"),
        "state_evolution_label_audit": state_evolution_audit,
        "temporal_hybrid_audit": temporal_audit,
        "usage": {
            "dialogue": _sum_usage(dialogues),
            "alignment": _sum_usage(
                [checkpoint.generated_alignment for checkpoint in checkpoints]
            ),
        },
    }


def _write_state_evolution_label_audit(
    output_root: Path,
    artifact,
    minimums: dict[str, int] | None = None,
) -> dict[str, Any]:
    labels = [
        label
        for checkpoint in artifact.event_checkpoints
        for label in checkpoint.turn_labels
    ]
    operations = Counter(
        operation.op for label in labels for operation in label.operations
    )
    decisions = Counter(label.decision for label in labels)
    reasons = Counter(label.reason_code for label in labels)
    minimums = minimums or {"add": 10, "replace": 2, "delete": 1}
    passed = all(operations[key] >= value for key, value in minimums.items())
    if not passed:
        raise ValueError(
            "Hybrid state-evolution operations fell below the approved Stage 2 "
            f"coverage: {dict(operations)}"
        )
    audit = {
        "operation_counts": dict(sorted(operations.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "reason_code_counts": dict(sorted(reasons.items())),
        "minimum_operation_counts": minimums,
        "passed": True,
    }
    _write_json(
        output_root / "state-evolution-label-audit.json",
        {
            "schema_version": "vehiclemembench-v2-state-evolution-label-audit-v1",
            "source_stage2_sha256": artifact.source_stage2_sha256,
            "source_hybrid_sha256": artifact.artifact_sha256,
            "audit": audit,
        },
    )
    return audit


def _run_single_event_smoke(
    args: argparse.Namespace,
    *,
    stage2: V1Stage2Artifact,
    contexts,
) -> dict[str, Any]:
    index = args.single_event_index
    if index is None or not 0 <= index < len(contexts):
        raise ValueError("--single-event-index is outside the Stage 2 timeline")
    if any(context.event.preference_updates for context in contexts[:index]):
        raise ValueError(
            "isolated event smoke cannot skip an earlier preference update"
        )
    context = contexts[index]
    output_root = args.output_root.expanduser().resolve()
    dialogue_root = output_root / "dialogues"
    event_root = output_root / "events"
    dialogue_root.mkdir(parents=True, exist_ok=True)
    event_root.mkdir(parents=True, exist_ok=True)
    temporal_transitions = load_temporal_plan(
        output_root / "temporal-plan.json",
        source_stage2_sha256=stage2.artifact_sha256,
        required=False,
    )
    temporal_anchor_artifact = None
    temporal_anchors = {}
    if temporal_transitions:
        temporal_plan = json.loads(
            (output_root / "temporal-plan.json").read_text(encoding="utf-8")
        )
        temporal_anchor_artifact, temporal_anchors = load_temporal_anchor_plan(
            output_root / "temporal-anchor-plan.json",
            source_stage2_sha256=stage2.artifact_sha256,
            source_temporal_plan_sha256=str(temporal_plan["artifact_sha256"]),
        )
    dialogue_model = OpenAIV1DialogueGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
        additional_instructions=(
            V2_TEMPORAL_DIALOGUE_INSTRUCTIONS if temporal_transitions else None
        ),
        prompt_version=(
            V2_TEMPORAL_DIALOGUE_PROMPT_VERSION
            if temporal_transitions
            else V1_DIALOGUE_PROMPT_VERSION
        ),
    )
    alignment_model = OpenAIV2HybridEventAlignmentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
        additional_instructions=(
            V2_TEMPORAL_ALIGNMENT_INSTRUCTIONS if temporal_transitions else None
        ),
        prompt_version=(
            V2_TEMPORAL_ALIGNMENT_PROMPT_VERSION
            if temporal_transitions
            else V2_HYBRID_ALIGNMENT_PROMPT_VERSION
        ),
        normalize_earlier_confirmation=not bool(temporal_transitions),
    )
    dialogue_path = dialogue_root / f"{context.event.event_id}.json"
    dialogue = _load_or_generate_dialogue(
        dialogue_path,
        stage2=stage2,
        context=context,
        model=dialogue_model,
        max_attempts=args.max_attempts,
        temporal_transitions=temporal_transitions.get(context.event.event_id),
        temporal_anchors=temporal_anchors.get(context.event.event_id, ()),
    )
    turns = event_dialogue_turns(stage2, dialogue)
    if temporal_anchor_artifact is not None:
        generated_alignment = build_deterministic_anchor_alignment(
            context.event,
            turns,
            temporal_anchors.get(context.event.event_id, ()),
            previous_memory="",
        )
    else:
        generated_alignment = _generate_alignment(
            context.event,
            turns,
            previous_memory="",
            model=alignment_model,
            max_attempts=args.max_attempts,
            temporal_transitions=temporal_transitions.get(context.event.event_id),
        )
    checkpoint = build_hybrid_event_checkpoint(
        event=context.event,
        timeline_index=context.timeline_index,
        dialogue_turns=turns,
        generated_alignment=generated_alignment,
        previous_entries=(),
        personas=stage2.persona_group.payload.personas,
        global_turn_offset=0,
        chain_kind=context.chain_kind,
        reason_profile=(
            "state_evolution"
            if stage2.event_chains.prompt_version
            in {
                V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION,
                *V2_TEMPORAL_EVENT_PROMPT_VERSIONS,
            }
            else "legacy"
        ),
        temporal_transitions_by_update=temporal_transitions.get(
            context.event.event_id
        ),
    )
    checkpoint_path = event_root / (
        f"{context.timeline_index:03d}-{context.event.event_id}.json"
    )
    _write_json(checkpoint_path, checkpoint.model_dump(mode="json"))
    update_labels = [
        label for label in checkpoint.turn_labels if label.decision == "UPDATE"
    ]
    return {
        "status": "single_event_smoke_completed",
        "standalone_global_turn_indexes": True,
        "checkpoint_path": str(checkpoint_path),
        "event": {
            "timeline_index": context.timeline_index,
            "event_id": context.event.event_id,
            "dialogue_turns": len(turns),
            "source_updates": len(context.event.preference_updates),
            "update_labels": len(update_labels),
            "patch_operations": sum(len(label.operations) for label in update_labels),
        },
        "final_memory": checkpoint.after_memory,
        "usage": {
            "dialogue": dict(dialogue.usage),
            "alignment": dict(generated_alignment.usage),
        },
    }


def _load_or_generate_dialogue(
    path: Path,
    *,
    stage2: V1Stage2Artifact,
    context,
    model: OpenAIV1DialogueGenerationModel,
    max_attempts: int,
    temporal_transitions=None,
    temporal_anchors=(),
) -> V1GeneratedEventDialogue:
    if path.exists():
        checkpoint = V1GeneratedEventDialogue.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if checkpoint.source_event_sha256 != event_sha256(context.event):
            raise ValueError(f"stale dialogue checkpoint: {path}")
        validate_dialogue_payload(
            context,
            checkpoint.payload,
            requested_turn_count=dialogue_turn_target(context),
        )
        validate_temporal_dialogue_cues(
            temporal_transitions,
            [line.text for line in checkpoint.payload.turns],
            event=context.event,
            speaker_ids=[line.speaker_id for line in checkpoint.payload.turns],
        )
        validate_dialogue_contains_anchors(
            temporal_anchors,
            checkpoint.payload.turns,
        )
        return checkpoint
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            checkpoint = model.generate(
                stage2.persona_group.payload.personas,
                context,
                requested_turn_count=dialogue_turn_target(context),
                frozen_anchors=[
                    item.model_dump(mode="json") for item in temporal_anchors
                ],
            )
            checkpoint = checkpoint.model_copy(
                update={
                    "usage": {
                        **checkpoint.usage,
                        "generation_attempts": attempt,
                    }
                }
            )
            validate_temporal_dialogue_cues(
                temporal_transitions,
                [line.text for line in checkpoint.payload.turns],
                event=context.event,
                speaker_ids=[line.speaker_id for line in checkpoint.payload.turns],
            )
            validate_dialogue_contains_anchors(
                temporal_anchors,
                checkpoint.payload.turns,
            )
            _write_json(path, checkpoint.model_dump(mode="json"))
            return checkpoint
        except Exception as exc:  # bounded provider/schema retry
            last_error = exc
    assert last_error is not None
    raise last_error


def _generate_alignment(
    event,
    turns,
    *,
    previous_memory: str,
    model: OpenAIV2HybridEventAlignmentModel,
    max_attempts: int,
    temporal_transitions=None,
):
    if not event.preference_updates:
        return deterministic_empty_alignment(
            event,
            turns,
            previous_memory=previous_memory,
        )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            generated = model.generate(
                event,
                turns,
                previous_memory=previous_memory,
            )
            generated = normalize_temporal_hybrid_alignment(
                event,
                turns,
                generated,
                temporal_transitions,
            )
            return generated.model_copy(
                update={
                    "usage": {
                        **generated.usage,
                        "generation_attempts": attempt,
                    }
                }
            )
        except Exception as exc:  # bounded provider/schema/evidence retry
            last_error = exc
    assert last_error is not None
    raise last_error


def _sum_usage(records) -> dict[str, int]:
    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "latency_ms",
    )
    return {
        key: sum(int(record.usage.get(key, 0)) for record in records)
        for key in keys
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
