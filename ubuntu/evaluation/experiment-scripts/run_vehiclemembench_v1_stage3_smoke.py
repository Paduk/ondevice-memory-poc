#!/usr/bin/env python3
"""Build one resumable Terra dialogue/final-QA V1 reproduction canary."""

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
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    OpenAIV1DialogueGenerationModel,
    OpenAIV1FinalQuizGenerationModel,
    V1GeneratedEventDialogue,
    V1GeneratedFinalQuiz,
    build_dialogue_generation_contexts,
    build_stage3_artifact,
    dialogue_turn_target,
    finalize_generated_quiz,
    materialize_dialogue_turns,
    relevant_tool_schemas_for_chain,
    validate_dialogue_payload,
    write_stage3_artifact,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v3/stage2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage3-smoke-terra-s1-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dialogue-root",
        type=Path,
        help="Optional shared dialogue checkpoint directory to reuse.",
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--scenario-id", default="scenario-reproduction-001")
    parser.add_argument("--public-scenario-index", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument(
        "--event-limit",
        type=int,
        help="Generate only the first N timeline dialogues, preserving checkpoints.",
    )
    parser.add_argument("--dialogue-only", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    runtime = VehicleWorldRuntime(dataset.root)
    stage2 = V1Stage2Artifact.model_validate_json(
        args.stage2_path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contexts = build_dialogue_generation_contexts(stage2)
    selected_contexts = (
        contexts[: args.event_limit]
        if args.event_limit is not None
        else contexts
    )
    if not selected_contexts:
        raise ValueError("--event-limit must be positive")
    dialogue_root = (
        args.dialogue_root.expanduser().resolve()
        if args.dialogue_root is not None
        else output_root / "dialogues"
    )
    dialogue_root.mkdir(parents=True, exist_ok=True)
    dialogue_model = OpenAIV1DialogueGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    dialogues = _generate_dialogues(
        stage2,
        selected_contexts,
        dialogue_root,
        dialogue_model,
        workers=args.workers,
        max_attempts=args.max_attempts,
    )
    if len(selected_contexts) < len(contexts):
        return {
            "status": "dialogue_partial",
            "completed_dialogue_events": len(dialogues),
            "total_dialogue_events": len(contexts),
            "usage": _sum_usage(dialogues),
        }
    dialogues = tuple(
        _load_dialogue(dialogue_root / f"{context.event.event_id}.json")
        for context in contexts
    )
    if args.dialogue_only:
        return {
            "status": "dialogue_completed",
            "completed_dialogue_events": len(dialogues),
            "dialogue_turns": sum(len(item.payload.turns) for item in dialogues),
            "usage": _sum_usage(dialogues),
        }

    dialogue_turns = materialize_dialogue_turns(stage2, dialogues)
    quiz_root = output_root / "quizzes"
    quiz_root.mkdir(parents=True, exist_ok=True)
    quiz_model = OpenAIV1FinalQuizGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    quizzes = _generate_quizzes(
        stage2,
        dialogue_turns,
        dataset.tool_schemas,
        runtime,
        quiz_root,
        quiz_model,
        workers=args.workers,
        max_attempts=args.max_attempts,
    )
    artifact = build_stage3_artifact(
        stage2,
        dialogues,
        quizzes,
        scenario_id=args.scenario_id,
        tool_schemas=dataset.tool_schemas,
        runtime=runtime,
    )
    paths = write_stage3_artifact(
        output_root,
        artifact,
        public_scenario_index=args.public_scenario_index,
    )
    return {
        "status": "completed",
        "paths": {name: str(path) for name, path in paths.items()},
        "artifact_sha256": artifact.artifact_sha256,
        "audit": artifact.audit.model_dump(mode="json"),
        "usage": {
            "dialogue": _sum_usage(dialogues),
            "quiz": _sum_usage(quizzes),
        },
    }


def _generate_dialogues(
    stage2: V1Stage2Artifact,
    contexts,
    checkpoint_root: Path,
    model: OpenAIV1DialogueGenerationModel,
    *,
    workers: int,
    max_attempts: int,
) -> tuple[V1GeneratedEventDialogue, ...]:
    personas = stage2.persona_group.payload.personas
    results: dict[str, V1GeneratedEventDialogue] = {}
    missing = []
    for context in contexts:
        path = checkpoint_root / f"{context.event.event_id}.json"
        if path.exists():
            checkpoint = _load_dialogue(path)
            if checkpoint.source_event_sha256 != _event_hash(context.event):
                raise ValueError(f"stale dialogue checkpoint: {path}")
            validate_dialogue_payload(
                context,
                checkpoint.payload,
                requested_turn_count=dialogue_turn_target(context),
            )
            results[context.event.event_id] = checkpoint
        else:
            missing.append(context)

    def generate(context):
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                checkpoint = model.generate(
                    personas,
                    context,
                    requested_turn_count=dialogue_turn_target(context),
                )
                return checkpoint.model_copy(
                    update={
                        "usage": {
                            **checkpoint.usage,
                            "generation_attempts": attempt,
                        }
                    }
                )
            except Exception as exc:  # bounded provider/schema retry
                last_error = exc
        assert last_error is not None
        raise last_error

    with ThreadPoolExecutor(max_workers=min(workers, len(missing) or 1)) as pool:
        futures = {pool.submit(generate, context): context for context in missing}
        failures: list[tuple[str, Exception]] = []
        for future in as_completed(futures):
            context = futures[future]
            try:
                checkpoint = future.result()
            except Exception as exc:
                failures.append((context.event.event_id, exc))
                continue
            _write_json(
                checkpoint_root / f"{context.event.event_id}.json",
                checkpoint.model_dump(mode="json"),
            )
            results[context.event.event_id] = checkpoint
    if failures:
        details = "; ".join(
            f"{event_id}: {error}" for event_id, error in failures
        )
        raise RuntimeError(f"dialogue generation failures: {details}")
    return tuple(results[context.event.event_id] for context in contexts)


def _generate_quizzes(
    stage2: V1Stage2Artifact,
    dialogue_turns,
    tool_schemas,
    runtime: VehicleWorldRuntime,
    checkpoint_root: Path,
    model: OpenAIV1FinalQuizGenerationModel,
    *,
    workers: int,
    max_attempts: int,
) -> tuple[V1GeneratedFinalQuiz, ...]:
    personas = stage2.persona_group.payload.personas
    chains = stage2.event_chains.payload.vehicle_chains
    results: dict[str, V1GeneratedFinalQuiz] = {}
    missing = []
    for index, chain in enumerate(chains, start=1):
        path = checkpoint_root / f"{chain.chain_id}.json"
        checkpoint = _load_valid_quiz(
            path,
            chain,
            index=index,
            tool_schemas=tool_schemas,
            runtime=runtime,
        )
        if checkpoint is None:
            missing.append((index, chain, path))
        else:
            results[chain.chain_id] = checkpoint

    def generate(index, chain):
        selected_schemas = relevant_tool_schemas_for_chain(chain, tool_schemas)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                checkpoint = model.generate(
                    personas,
                    chain,
                    dialogue_turns,
                    selected_schemas,
                )
                finalize_generated_quiz(
                    checkpoint,
                    chain,
                    quiz_index=index,
                    tool_schemas=tool_schemas,
                    runtime=runtime,
                    require_state_change=False,
                )
                return checkpoint.model_copy(
                    update={
                        "usage": {
                            **checkpoint.usage,
                            "generation_attempts": attempt,
                        }
                    }
                )
            except Exception as exc:  # bounded provider/simulator repair retry
                last_error = exc
        assert last_error is not None
        raise last_error

    with ThreadPoolExecutor(max_workers=min(workers, len(missing) or 1)) as pool:
        futures = {
            pool.submit(generate, index, chain): (chain, path)
            for index, chain, path in missing
        }
        failures: list[tuple[str, Exception]] = []
        for future in as_completed(futures):
            chain, path = futures[future]
            try:
                checkpoint = future.result()
            except Exception as exc:
                failures.append((chain.chain_id, exc))
                continue
            _write_json(path, checkpoint.model_dump(mode="json"))
            results[chain.chain_id] = checkpoint
    if failures:
        details = "; ".join(
            f"{chain_id}: {error}" for chain_id, error in failures
        )
        raise RuntimeError(f"Quiz generation failures: {details}")
    return tuple(results[chain.chain_id] for chain in chains)


def _load_valid_quiz(
    path: Path,
    chain,
    *,
    index: int,
    tool_schemas,
    runtime: VehicleWorldRuntime,
) -> V1GeneratedFinalQuiz | None:
    if not path.exists():
        return None
    checkpoint = V1GeneratedFinalQuiz.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    try:
        finalize_generated_quiz(
            checkpoint,
            chain,
            quiz_index=index,
            tool_schemas=tool_schemas,
            runtime=runtime,
            require_state_change=False,
        )
    except ValueError:
        invalid_path = path.with_name(
            f"{path.stem}.invalid-{checkpoint.input_sha256[:8]}.json"
        )
        path.replace(invalid_path)
        return None
    return checkpoint


def _load_dialogue(path: Path) -> V1GeneratedEventDialogue:
    return V1GeneratedEventDialogue.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _event_hash(event) -> str:
    from palmclaw_ubuntu.vehicle_bench.v1_reproduction import canonical_json_sha256

    return canonical_json_sha256(event.model_dump(mode="json"))


def _sum_usage(records) -> dict[str, int]:
    keys = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens")
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
