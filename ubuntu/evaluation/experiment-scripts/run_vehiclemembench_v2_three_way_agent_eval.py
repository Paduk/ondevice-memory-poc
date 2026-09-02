#!/usr/bin/env python3
"""Run/resume memory-only Agent evaluation for ready V2 quality artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    GoldToolCall,
    VehicleTask,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.memory import VehicleMemoryContext
from palmclaw_ubuntu.vehicle_bench.runner import _run_agent_task
from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime
from palmclaw_ubuntu.vehicle_bench.tools import vehicle_tool_definitions
from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    QualityArtifactPaths,
    QualityQuiz,
    default_quality_artifact_paths,
    load_quality_quizzes,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT / "vehiclemembench-v2-three-way-evaluation"
)
AGENT_PROFILE = "cloud_recursive_summary"
AGENT_EVALUATION_VERSION = "vehiclemembench-v2-memory-only-agent-eval-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("post_hoc", "hybrid", "native_turnwise"),
    )
    parser.add_argument("--scenarios", nargs="+", type=int)
    parser.add_argument("--quiz-limit-per-artifact", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--task-attempts", type=int, default=2)
    parser.add_argument("--max-tool-rounds", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from palmclaw_ubuntu.providers import OpenAIResponsesAgentModel

    if min(args.repeats, args.workers, args.task_attempts) < 1:
        raise ValueError("repeats, workers, and task attempts must be positive")
    if args.quiz_limit_per_artifact is not None and args.quiz_limit_per_artifact < 1:
        raise ValueError("quiz limit must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    output_root = args.output_root.expanduser().resolve() / "agent"
    output_root.mkdir(parents=True, exist_ok=True)
    ready: list[tuple[QualityArtifactPaths, tuple[QualityQuiz, ...]]] = []
    scenario_indices = tuple(args.scenarios) if args.scenarios else None
    for paths in default_quality_artifact_paths(
        args.evaluation_root,
        scenario_indices=scenario_indices,
    ):
        if args.methods and paths.method not in args.methods:
            continue
        if args.scenarios and paths.scenario_index not in args.scenarios:
            continue
        required = (
            paths.memory_artifact,
            paths.turn_quiz_artifact,
            paths.final_quiz_artifact,
        )
        if not all(path.is_file() for path in required):
            continue
        quizzes = load_quality_quizzes(paths)
        if args.quiz_limit_per_artifact is not None:
            quizzes = quizzes[: args.quiz_limit_per_artifact]
        ready.append((paths, quizzes))
    if not ready:
        raise ValueError("No complete V2 quality artifacts are ready")

    agent_model = OpenAIResponsesAgentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
        reasoning_effort="low",
        redact_pii=False,
    )
    work: list[tuple[QualityQuiz, int, Path, str]] = []
    records: list[dict[str, Any]] = []
    for _, quizzes in ready:
        for repeat in range(1, args.repeats + 1):
            for quiz in quizzes:
                checkpoint = (
                    output_root
                    / quiz.method
                    / f"s{quiz.scenario_index:02d}"
                    / f"r{repeat}"
                    / "tasks"
                    / f"{quiz.quiz_id}.json"
                )
                signature = _evaluation_signature(quiz, repeat, args.model)
                if checkpoint.is_file():
                    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if stored.get("evaluation_signature") != signature:
                        raise ValueError(f"Stale Agent checkpoint: {checkpoint}")
                    if stored.get("status") == "completed":
                        records.append(stored)
                        continue
                work.append((quiz, repeat, checkpoint, signature))

    def evaluate(item: tuple[QualityQuiz, int, Path, str]) -> dict[str, Any]:
        quiz, repeat, checkpoint, signature = item
        last_record: dict[str, Any] | None = None
        for attempt in range(1, args.task_attempts + 1):
            runtime = VehicleWorldRuntime(dataset.root)
            definitions = vehicle_tool_definitions(
                dataset.tool_schemas,
                timeout_seconds=5,
            )
            task = VehicleTask(
                id=f"{quiz.method}-s{quiz.scenario_index:02d}-{quiz.quiz_id}-r{repeat}",
                scenario_index=quiz.scenario_index,
                event_index=0,
                query=quiz.query,
                gold_memory="",
                reasoning_type=quiz.reasoning_type,
                gold_calls=tuple(
                    GoldToolCall(
                        name=call.name,
                        arguments=dict(call.arguments),
                        source="v2-quality-gold",
                    )
                    for call in quiz.gold_calls
                ),
            )

            def memory_resolver(profile: str, query: str) -> VehicleMemoryContext:
                del profile, query
                return VehicleMemoryContext(
                    content=quiz.memory,
                    metadata={
                        "method": quiz.method,
                        "scenario_index": quiz.scenario_index,
                        "quiz_id": quiz.quiz_id,
                        "memory_sha256": quiz.memory_sha256,
                    },
                    trace={},
                )

            record = _run_agent_task(
                runtime=runtime,
                definitions=definitions,
                agent_model=agent_model,
                profile=AGENT_PROFILE,
                task=task,
                max_tool_rounds=args.max_tool_rounds,
                max_tool_result_chars=20_000,
                memory_resolver=memory_resolver,
                oracle_retrieval_annotations=None,
                oracle_gate_annotations=None,
                oracle_stage_fact_annotations=None,
            )
            record.update(
                {
                    "evaluation_version": AGENT_EVALUATION_VERSION,
                    "evaluation_signature": signature,
                    "method": quiz.method,
                    "quality_scenario_index": quiz.scenario_index,
                    "quiz_id": quiz.quiz_id,
                    "quiz_kind": quiz.quiz_kind,
                    "repeat": repeat,
                    "source_stage2_sha256": quiz.source_stage2_sha256,
                    "source_quiz_sha256": quiz.source_quiz_sha256,
                    "memory_sha256": quiz.memory_sha256,
                    "evaluation_attempt": attempt,
                    "agent_model": args.model,
                }
            )
            last_record = record
            if record.get("status") == "completed":
                break
        assert last_record is not None
        _write_json(checkpoint, last_record)
        return last_record

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(work) or 1)) as pool:
        futures = {pool.submit(evaluate, item): item for item in work}
        for future in as_completed(futures):
            quiz, repeat, _, _ = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(
                    f"{quiz.method}/S{quiz.scenario_index}/{quiz.quiz_id}/R{repeat}: "
                    f"{type(exc).__name__}: {exc}"
                )
    summary = _aggregate(records, failures=failures, model=args.model)
    _write_json(output_root / "agent-summary.json", summary)
    if failures:
        raise RuntimeError(f"Agent evaluation failures: {'; '.join(failures)}")
    return summary


def _aggregate(
    records: list[dict[str, Any]],
    *,
    failures: list[str],
    model: str,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["method"]), "all")].append(record)
        groups[(str(record["method"]), str(record["quiz_kind"]))].append(record)
    metrics = {}
    for (method, kind), selected in sorted(groups.items()):
        key = f"{method}/{kind}"
        metrics[key] = {
            "tasks": len(selected),
            "completion_rate": _mean(r.get("status") == "completed" for r in selected),
            "exact_state_match": _mean(
                bool(r.get("score", {}).get("exact_state_match")) for r in selected
            ),
            "tool_f1": _mean(
                float(r.get("score", {}).get("tool_score", {}).get("f1", 0))
                for r in selected
            ),
            "argument_exact_match": _mean(
                bool(r.get("argument_exact_match")) for r in selected
            ),
            "input_tokens": sum(
                int(r.get("usage", {}).get("input_tokens", 0)) for r in selected
            ),
            "output_tokens": sum(
                int(r.get("usage", {}).get("output_tokens", 0)) for r in selected
            ),
            "latency_ms": sum(
                int(r.get("usage", {}).get("model_latency_ms", 0))
                for r in selected
            ),
        }
    return {
        "version": AGENT_EVALUATION_VERSION,
        "status": "COMPLETED" if not failures else "PARTIAL",
        "model": model,
        "record_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "metrics": metrics,
    }


def _evaluation_signature(quiz: QualityQuiz, repeat: int, model: str) -> str:
    body = {
        "version": AGENT_EVALUATION_VERSION,
        "model": model,
        "repeat": repeat,
        "source_quiz_sha256": quiz.source_quiz_sha256,
        "memory_sha256": quiz.memory_sha256,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mean(values: Any) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
