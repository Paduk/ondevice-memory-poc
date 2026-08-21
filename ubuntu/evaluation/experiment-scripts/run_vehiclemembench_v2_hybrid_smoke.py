#!/usr/bin/env python3
"""Run resumable event-batched/turn-labeled VehicleMemBench V2 Hybrid generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    OpenAIV1DialogueGenerationModel,
    V1GeneratedEventDialogue,
    build_dialogue_generation_contexts,
    dialogue_turn_target,
    validate_dialogue_payload,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    OpenAIV2HybridEventAlignmentModel,
    V2HybridEventCheckpoint,
    build_hybrid_artifact,
    build_hybrid_event_checkpoint,
    deterministic_empty_alignment,
    event_dialogue_turns,
    event_sha256,
    write_hybrid_artifact,
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

    dialogue_model = OpenAIV1DialogueGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    alignment_model = OpenAIV2HybridEventAlignmentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
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
            )
            if rebuilt != checkpoint:
                raise ValueError(f"stale Hybrid event checkpoint: {checkpoint_path}")
        else:
            previous_memory = (
                checkpoints[-1].after_memory if checkpoints else ""
            )
            generated_alignment = _generate_alignment(
                context.event,
                turns,
                previous_memory=previous_memory,
                model=alignment_model,
                max_attempts=args.max_attempts,
            )
            checkpoint = build_hybrid_event_checkpoint(
                event=context.event,
                timeline_index=context.timeline_index,
                dialogue_turns=turns,
                generated_alignment=generated_alignment,
                previous_entries=previous_entries,
                personas=stage2.persona_group.payload.personas,
                global_turn_offset=global_turn_offset,
            )
            _write_json(checkpoint_path, checkpoint.model_dump(mode="json"))
        checkpoints.append(checkpoint)
        previous_entries = checkpoint.memory_entries_after
        global_turn_offset += len(turns)

    artifact = build_hybrid_artifact(stage2, checkpoints)
    artifact_path = write_hybrid_artifact(output_root, artifact)
    return {
        "status": "completed" if artifact.audit.completed else "partial",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact.artifact_sha256,
        "audit": artifact.audit.model_dump(mode="json"),
        "usage": {
            "dialogue": _sum_usage(dialogues),
            "alignment": _sum_usage(
                [checkpoint.generated_alignment for checkpoint in checkpoints]
            ),
        },
    }


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
    dialogue_model = OpenAIV1DialogueGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    alignment_model = OpenAIV2HybridEventAlignmentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    dialogue_path = dialogue_root / f"{context.event.event_id}.json"
    dialogue = _load_or_generate_dialogue(
        dialogue_path,
        stage2=stage2,
        context=context,
        model=dialogue_model,
        max_attempts=args.max_attempts,
    )
    turns = event_dialogue_turns(stage2, dialogue)
    generated_alignment = _generate_alignment(
        context.event,
        turns,
        previous_memory="",
        model=alignment_model,
        max_attempts=args.max_attempts,
    )
    checkpoint = build_hybrid_event_checkpoint(
        event=context.event,
        timeline_index=context.timeline_index,
        dialogue_turns=turns,
        generated_alignment=generated_alignment,
        previous_entries=(),
        personas=stage2.persona_group.payload.personas,
        global_turn_offset=0,
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
        return checkpoint
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            checkpoint = model.generate(
                stage2.persona_group.payload.personas,
                context,
                requested_turn_count=dialogue_turn_target(context),
            )
            checkpoint = checkpoint.model_copy(
                update={
                    "usage": {
                        **checkpoint.usage,
                        "generation_attempts": attempt,
                    }
                }
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
