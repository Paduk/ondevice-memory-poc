#!/usr/bin/env python3
"""Generate resumable V1-style Turn Quizzes from Hybrid UPDATE checkpoints."""

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
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridArtifact,
    V2HybridEventCheckpoint,
)
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz import (
    V2_TURN_QUIZ_PROMPT_VERSION,
    OpenAIV2TurnQuizGenerationModel,
    V2TurnQuizRecord,
    build_turn_quiz_artifact,
    build_turn_quiz_contexts,
    finalize_turn_quiz,
    validate_turn_quiz_query,
    write_turn_quiz_artifact,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v4/stage2.json"
)
DEFAULT_HYBRID_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/"
    "hybrid-update-smoke-terra-s1-v3"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hybrid-artifact", type=Path)
    source.add_argument("--event-checkpoint", type=Path)
    parser.add_argument(
        "--dialogue-root",
        type=Path,
        default=DEFAULT_HYBRID_ROOT / "dialogues",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_HYBRID_ROOT / "turn-quiz-v1",
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("Turn Quiz workers and attempts must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    runtime = VehicleWorldRuntime(dataset.root)
    stage2 = V1Stage2Artifact.model_validate_json(
        args.stage2_path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    checkpoints = _load_source_checkpoints(args)
    dialogue_root = args.dialogue_root.expanduser().resolve(strict=True)
    dialogues = tuple(
        V1GeneratedEventDialogue.model_validate_json(
            (dialogue_root / f"{checkpoint.event_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for checkpoint in checkpoints
    )
    contexts = build_turn_quiz_contexts(stage2, checkpoints, dialogues)
    if not contexts:
        raise ValueError("Turn Quiz source contains no UPDATE checkpoint")
    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = output_root / "quizzes"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model = OpenAIV2TurnQuizGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    chain_by_id = {
        chain.chain_id: chain
        for chain in stage2.event_chains.payload.vehicle_chains
    }
    event_by_id = {
        event.event_id: event
        for chain in stage2.event_chains.payload.all_chains
        for event in chain.events
    }
    personas = tuple(
        persona.model_dump(mode="json")
        for persona in stage2.persona_group.payload.personas
    )
    results: dict[str, V2TurnQuizRecord] = {}
    missing = []
    for index, context in enumerate(contexts, start=1):
        path = checkpoint_root / f"{context.quiz_id}.json"
        if path.exists():
            record = V2TurnQuizRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            _validate_record_source(record, context)
            results[context.quiz_id] = record
        else:
            missing.append((index, context, path))

    def generate(index, context):
        chain = chain_by_id[context.source_chain_id]
        event = event_by_id[context.source_event_id]
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                generated = model.generate(
                    context,
                    personas=personas,
                    source_event_description=event.description,
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
                    chain=chain,
                    quiz_index=index,
                    tool_schemas=dataset.tool_schemas,
                    runtime=runtime,
                )
            except Exception as exc:  # bounded schema/evidence/simulator retry
                last_error = exc
        assert last_error is not None
        raise last_error

    failures = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(missing) or 1)) as pool:
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
        raise RuntimeError(f"Turn Quiz generation failures: {detail}")

    quizzes = tuple(results[context.quiz_id] for context in contexts)
    artifact = build_turn_quiz_artifact(stage2, contexts, quizzes)
    artifact_path = write_turn_quiz_artifact(output_root, artifact)
    return {
        "status": "completed",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact.artifact_sha256,
        "audit": artifact.audit.model_dump(mode="json"),
        "usage": _sum_usage(quizzes),
    }


def _load_source_checkpoints(
    args: argparse.Namespace,
) -> tuple[V2HybridEventCheckpoint, ...]:
    if args.hybrid_artifact is not None:
        artifact = V2HybridArtifact.model_validate_json(
            args.hybrid_artifact.expanduser()
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
        return artifact.event_checkpoints
    path = (
        args.event_checkpoint
        if args.event_checkpoint is not None
        else DEFAULT_HYBRID_ROOT / "events" / "051-v01-e4.json"
    )
    return (
        V2HybridEventCheckpoint.model_validate_json(
            path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
        ),
    )


def _validate_record_source(record, context) -> None:
    if (
        record.quiz_id != context.quiz_id
        or record.source_checkpoint_sha256 != context.source_checkpoint_sha256
        or record.memory_snapshot_sha256 != context.current_memory_sha256
    ):
        raise ValueError(f"stale Turn Quiz checkpoint: {record.quiz_id}")
    if record.generated_query.prompt_version != V2_TURN_QUIZ_PROMPT_VERSION:
        raise ValueError(f"stale Turn Quiz prompt version: {record.quiz_id}")
    validate_turn_quiz_query(context, record.generated_query.payload)


def _sum_usage(quizzes) -> dict[str, int]:
    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "latency_ms",
    )
    return {
        key: sum(int(quiz.generated_query.usage.get(key, 0)) for quiz in quizzes)
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
