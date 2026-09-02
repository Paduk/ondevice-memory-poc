#!/usr/bin/env python3
"""Expand one Hybrid or Native scenario to 30 common V2 Turn Quizzes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime
from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1GeneratedEventDialogue
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import V2HybridArtifact
from palmclaw_ubuntu.vehicle_bench.v2_native_turnwise import (
    V2NativeScenarioArtifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz import (
    V2_TURN_QUIZ_PROMPT_VERSION,
    OpenAIV2TurnQuizGenerationModel,
    V2TurnQuizArtifact,
    V2TurnQuizRecord,
    build_turn_quiz_artifact,
    build_turn_quiz_contexts,
    finalize_turn_quiz,
    validate_turn_quiz_query,
    write_turn_quiz_artifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz_expansion import (
    V2_COMPOSITE_QUIZ_PLAN_PROMPT_VERSION,
    OpenAIV2CompositeQuizPlanModel,
    V2GeneratedCompositeQuizPlan,
    build_supplemental_turn_quiz_contexts,
    build_turn_quiz_expansion_plan,
    build_turn_quiz_facts,
    repair_composite_quiz_plan_for_checkpoints,
    select_delayed_context_indexes,
    validate_composite_quiz_plan,
)

DEFAULT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/"
    "hybrid-full-terra-s1-r2"
)
DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/"
    "hybrid-full-terra-s1-r1/stage2-v2-anchored.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument(
        "--source-kind", choices=("hybrid", "native"), default="hybrid"
    )
    parser.add_argument(
        "--hybrid-artifact", type=Path, default=DEFAULT_ROOT / "hybrid.json"
    )
    parser.add_argument("--native-artifact", type=Path)
    parser.add_argument(
        "--dialogue-root",
        type=Path,
        help="Optional external dialogue checkpoints; Native embeds its dialogues.",
    )
    parser.add_argument(
        "--immediate-artifact",
        type=Path,
        help="Optional existing immediate-Quiz artifact; otherwise generate/resume it.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT / "turn-quiz-30-v1",
    )
    parser.add_argument("--target-quiz-count", type=int, default=30)
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("Quiz workers and attempts must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    runtime = VehicleWorldRuntime(dataset.root)
    stage2 = V1Stage2Artifact.model_validate_json(
        args.stage2_path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    checkpoints, dialogues = _load_source(args, stage2)
    immediate_contexts = build_turn_quiz_contexts(stage2, checkpoints, dialogues)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "quizzes"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    query_model = OpenAIV2TurnQuizGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    chain_by_id = {
        chain.chain_id: chain for chain in stage2.event_chains.payload.vehicle_chains
    }
    personas = tuple(
        item.model_dump(mode="json") for item in stage2.persona_group.payload.personas
    )
    immediate_records = _load_or_generate_immediate_records(
        args,
        stage2=stage2,
        contexts=immediate_contexts,
        checkpoint_root=checkpoint_root,
        query_model=query_model,
        chain_by_id=chain_by_id,
        personas=personas,
        tool_schemas=dataset.tool_schemas,
        runtime=runtime,
    )
    immediate_by_id = {item.quiz_id: item for item in immediate_records}
    facts = build_turn_quiz_facts(
        immediate_contexts,
        evidence_by_quiz_id={
            quiz_id: item.memory_evidence_lines
            for quiz_id, item in immediate_by_id.items()
        },
    )
    immediate_count = len(immediate_contexts)
    supplemental_capacity = args.target_quiz_count - immediate_count
    if supplemental_capacity < 2:
        raise ValueError("Target Quiz count cannot retain delayed and composite coverage")
    # Preserve the S1-S20 mix when possible, but reserve four composite quizzes
    # when S21+ state evolution increases the number of UPDATE checkpoints.
    minimum_composite_count = min(4, supplemental_capacity - 1)
    delayed_count = min(
        immediate_count,
        supplemental_capacity - minimum_composite_count,
    )
    composite_count = supplemental_capacity - delayed_count
    delayed_context_indexes = select_delayed_context_indexes(
        immediate_count,
        delayed_count,
    )

    plan_checkpoint = output_root / "composite-plan.json"
    if plan_checkpoint.exists():
        generated_plan = V2GeneratedCompositeQuizPlan.model_validate_json(
            plan_checkpoint.read_text(encoding="utf-8")
        )
        if generated_plan.prompt_version != V2_COMPOSITE_QUIZ_PLAN_PROMPT_VERSION:
            raise ValueError("Stale composite Quiz planning prompt")
        validate_composite_quiz_plan(
            facts,
            generated_plan.payload,
            group_count=composite_count,
        )
    else:
        plan_model = OpenAIV2CompositeQuizPlanModel(
            args.model,
            timeout_seconds=args.timeout_seconds,
        )
        plan_error: Exception | None = None
        for _ in range(args.max_attempts):
            try:
                generated_plan = plan_model.generate(
                    facts,
                    group_count=composite_count,
                )
                break
            except Exception as exc:
                plan_error = exc
        else:
            assert plan_error is not None
            raise plan_error
        _write_json(plan_checkpoint, generated_plan.model_dump(mode="json"))

    repaired_plan = repair_composite_quiz_plan_for_checkpoints(
        facts,
        generated_plan,
        immediate_contexts,
        checkpoints,
    )
    if repaired_plan != generated_plan:
        generated_plan = repaired_plan
        _write_json(plan_checkpoint, generated_plan.model_dump(mode="json"))

    contexts, specs, rebuilt_facts = build_supplemental_turn_quiz_contexts(
        stage2,
        checkpoints,
        dialogues,
        immediate_contexts,
        generated_plan,
        facts=facts,
        delayed_context_indexes=delayed_context_indexes,
    )
    if (
        rebuilt_facts != facts
        or len(immediate_contexts) + len(contexts) != args.target_quiz_count
    ):
        raise ValueError("Turn Quiz expansion did not produce the requested total")
    spec_by_id = {item.quiz_id: item for item in specs}
    expansion = build_turn_quiz_expansion_plan(
        stage2,
        facts,
        generated_plan,
        specs,
        immediate_quiz_count=len(immediate_contexts),
        target_quiz_count=args.target_quiz_count,
    )
    _write_json(output_root / "expansion-plan.json", expansion.model_dump(mode="json"))

    supplemental_records = _generate_records(
        contexts,
        start_index=len(immediate_contexts) + 1,
        descriptions={
            quiz_id: "\n".join(spec.source_event_descriptions)
            for quiz_id, spec in spec_by_id.items()
        },
        checkpoint_root=checkpoint_root,
        query_model=query_model,
        chain_by_id=chain_by_id,
        personas=personas,
        tool_schemas=dataset.tool_schemas,
        runtime=runtime,
        workers=args.workers,
        max_attempts=args.max_attempts,
    )
    all_contexts = (*immediate_contexts, *contexts)
    all_records = (*immediate_records, *supplemental_records)
    artifact = build_turn_quiz_artifact(
        stage2,
        all_contexts,
        all_records,
        update_checkpoint_count=len(immediate_contexts),
    )
    artifact_path = write_turn_quiz_artifact(output_root, artifact)
    return {
        "status": "completed",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact.artifact_sha256,
        "audit": artifact.audit.model_dump(mode="json"),
        "expansion_plan_sha256": expansion.artifact_sha256,
        "quiz_kind_counts": {
            "immediate": len(immediate_contexts),
            "delayed": delayed_count,
            "composite": composite_count,
        },
        "usage": _sum_usage(all_records),
    }


def _load_source(args, stage2):
    if args.source_kind == "native":
        if args.native_artifact is None:
            raise ValueError("--native-artifact is required for a Native source")
        source = V2NativeScenarioArtifact.model_validate_json(
            args.native_artifact.expanduser().resolve(strict=True).read_text(
                encoding="utf-8"
            )
        )
        if not source.audit.completed:
            raise ValueError("Native Turn Quiz expansion requires a complete scenario")
        checkpoints = source.event_artifacts
        embedded_dialogues = tuple(
            item.generated_dialogue for item in source.event_artifacts
        )
    else:
        source = V2HybridArtifact.model_validate_json(
            args.hybrid_artifact.expanduser().resolve(strict=True).read_text(
                encoding="utf-8"
            )
        )
        checkpoints = source.event_checkpoints
        embedded_dialogues = None
    if source.source_stage2_sha256 != stage2.artifact_sha256:
        raise ValueError("Quiz source and Stage 2 artifact do not match")
    if args.dialogue_root is not None:
        dialogue_root = args.dialogue_root.expanduser().resolve(strict=True)
        dialogues = tuple(
            V1GeneratedEventDialogue.model_validate_json(
                (dialogue_root / f"{checkpoint.event_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            for checkpoint in checkpoints
        )
    elif embedded_dialogues is not None:
        dialogues = embedded_dialogues
    else:
        dialogue_root = (
            DEFAULT_ROOT.parent / "hybrid-full-terra-s1-r1" / "dialogues"
        ).resolve(strict=True)
        dialogues = tuple(
            V1GeneratedEventDialogue.model_validate_json(
                (dialogue_root / f"{checkpoint.event_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            for checkpoint in checkpoints
        )
    if len(checkpoints) != len(stage2.interleaved_timeline):
        raise ValueError("Turn Quiz expansion requires every scenario event")
    return checkpoints, dialogues


def _load_or_generate_immediate_records(
    args,
    *,
    stage2,
    contexts,
    checkpoint_root,
    query_model,
    chain_by_id,
    personas,
    tool_schemas,
    runtime,
):
    immediate_artifact = args.immediate_artifact
    legacy_immediate = DEFAULT_ROOT / "turn-quiz-v3" / "turn-quizzes.json"
    if (
        immediate_artifact is None
        and args.source_kind == "hybrid"
        and legacy_immediate.exists()
    ):
        immediate_artifact = legacy_immediate
    if immediate_artifact is not None:
        artifact = V2TurnQuizArtifact.model_validate_json(
            immediate_artifact.expanduser().resolve(strict=True).read_text(
                encoding="utf-8"
            )
        )
        by_id = {item.quiz_id: item for item in artifact.quizzes}
        if set(by_id) != {item.quiz_id for item in contexts}:
            raise ValueError("Immediate Quiz artifact does not match checkpoints")
        records = tuple(by_id[item.quiz_id] for item in contexts)
        for record, context in zip(records, contexts, strict=True):
            _validate_record(record, context)
        return records
    descriptions = {
        f"turn-quiz-{item.event.event_id}-": item.event.description
        for item in stage2.interleaved_timeline
    }
    by_quiz_id = {
        context.quiz_id: next(
            description
            for prefix, description in descriptions.items()
            if context.quiz_id.startswith(prefix)
        )
        for context in contexts
    }
    return _generate_records(
        contexts,
        start_index=1,
        descriptions=by_quiz_id,
        checkpoint_root=checkpoint_root,
        query_model=query_model,
        chain_by_id=chain_by_id,
        personas=personas,
        tool_schemas=tool_schemas,
        runtime=runtime,
        workers=args.workers,
        max_attempts=args.max_attempts,
    )


def _generate_records(
    contexts,
    *,
    start_index,
    descriptions,
    checkpoint_root,
    query_model,
    chain_by_id,
    personas,
    tool_schemas,
    runtime,
    workers,
    max_attempts,
):
    results = {}
    missing = []
    for index, context in enumerate(contexts, start=start_index):
        path = checkpoint_root / f"{context.quiz_id}.json"
        if path.exists():
            record = V2TurnQuizRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            _validate_record(record, context)
            results[context.quiz_id] = record
        else:
            missing.append((index, context, path))

    def generate(index, context):
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                generated = query_model.generate(
                    context,
                    personas=personas,
                    source_event_description=descriptions[context.quiz_id],
                    retry_feedback=str(last_error) if last_error is not None else None,
                )
                generated = generated.model_copy(
                    update={
                        "usage": {
                            **generated.usage,
                            "generation_attempts": attempt,
                        }
                    }
                )
                return finalize_turn_quiz(
                    context,
                    generated,
                    chain=chain_by_id[context.source_chain_id],
                    quiz_index=index,
                    tool_schemas=tool_schemas,
                    runtime=runtime,
                )
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    failures = []
    with ThreadPoolExecutor(max_workers=min(workers, len(missing) or 1)) as pool:
        futures = {
            pool.submit(generate, index, context): (context, path)
            for index, context, path in missing
        }
        for future in as_completed(futures):
            context, path = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                failures.append((context.quiz_id, exc))
                continue
            _write_json(path, record.model_dump(mode="json"))
            results[context.quiz_id] = record
    if failures:
        detail = "; ".join(f"{quiz_id}: {error}" for quiz_id, error in failures)
        raise RuntimeError(f"Turn Quiz failures: {detail}")
    return tuple(results[item.quiz_id] for item in contexts)


def _validate_record(record, context) -> None:
    if (
        record.quiz_id != context.quiz_id
        or record.source_checkpoint_sha256 != context.source_checkpoint_sha256
        or record.memory_snapshot_sha256 != context.current_memory_sha256
    ):
        raise ValueError(f"Stale supplemental Quiz checkpoint: {record.quiz_id}")
    if record.generated_query.prompt_version != V2_TURN_QUIZ_PROMPT_VERSION:
        raise ValueError(f"Stale supplemental Quiz prompt: {record.quiz_id}")
    validate_turn_quiz_query(context, record.generated_query.payload)


def _sum_usage(records) -> dict[str, int]:
    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "latency_ms",
    )
    return {
        key: sum(int(item.generated_query.usage.get(key, 0)) for item in records)
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
