#!/usr/bin/env python3
"""Evaluate Luna over generated Hybrid-dialogue memory snapshots."""

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
    default_quality_artifact_paths,
    load_quality_quizzes,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT / "vehiclemembench-v2-hybrid-cloud-memory-gap"
)
ARMS = ("turnwise_summary", "turnwise_combined")
EVALUATION_VERSION = "vehiclemembench-v2-hybrid-cloud-gap-agent-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--scenarios", nargs="+", type=int)
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
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    selected_arms = tuple(args.arms or ARMS)
    selected_scenarios = set(args.scenarios or range(1, 6))
    quality_paths = {
        paths.scenario_index: paths
        for paths in default_quality_artifact_paths(
            args.evaluation_root,
            sorted(selected_scenarios),
        )
        if paths.method == "hybrid" and paths.scenario_index in selected_scenarios
    }
    if set(selected_scenarios) != set(quality_paths):
        raise ValueError("One or more frozen Hybrid quality scenarios are unavailable")

    tasks: list[tuple[str, Any, str, str]] = []
    for scenario in sorted(selected_scenarios):
        quizzes = load_quality_quizzes(quality_paths[scenario])
        quality_by_id = {quiz.quiz_id: quiz for quiz in quizzes}
        for arm in selected_arms:
            artifact_path = (
                args.output_root / arm / f"s{scenario:02d}" / "memory-replay.json"
            )
            if not artifact_path.is_file():
                raise ValueError(f"Memory replay is unavailable: {artifact_path}")
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact.get("status") != "COMPLETED":
                raise ValueError(f"Memory replay is incomplete: {artifact_path}")
            snapshots = {item["quiz_id"]: item for item in artifact["quiz_snapshots"]}
            if set(snapshots) != set(quality_by_id):
                raise ValueError(f"Memory replay Quiz set mismatch: {artifact_path}")
            for quiz_id, quality in quality_by_id.items():
                snapshot = snapshots[quiz_id]
                if snapshot["source_quiz_sha256"] != quality.source_quiz_sha256:
                    raise ValueError(
                        f"Stale Quiz snapshot: {arm}/S{scenario}/{quiz_id}"
                    )
                tasks.append(
                    (
                        arm,
                        quality,
                        str(snapshot["memory"]),
                        str(artifact["artifact_sha256"]),
                    )
                )

    agent_model = OpenAIResponsesAgentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
        reasoning_effort="low",
        redact_pii=False,
    )
    output_dir = args.output_root / "quiz-agent"
    work: list[tuple[str, Any, str, str, int, Path, str]] = []
    records: list[dict[str, Any]] = []
    for arm, quiz, memory, artifact_sha256 in tasks:
        for repeat in range(1, args.repeats + 1):
            checkpoint = (
                output_dir
                / arm
                / f"s{quiz.scenario_index:02d}"
                / f"r{repeat}"
                / "tasks"
                / f"{quiz.quiz_id}.json"
            )
            signature = _signature(
                arm=arm,
                quiz_sha256=quiz.source_quiz_sha256,
                memory=memory,
                memory_artifact_sha256=artifact_sha256,
                repeat=repeat,
                model=args.model,
            )
            if checkpoint.is_file():
                stored = json.loads(checkpoint.read_text(encoding="utf-8"))
                if stored.get("evaluation_signature") != signature:
                    raise ValueError(f"Stale Agent checkpoint: {checkpoint}")
                if stored.get("status") == "completed":
                    records.append(stored)
                    continue
            work.append(
                (arm, quiz, memory, artifact_sha256, repeat, checkpoint, signature)
            )

    def evaluate(item: tuple[str, Any, str, str, int, Path, str]) -> dict[str, Any]:
        arm, quiz, memory, artifact_sha256, repeat, checkpoint, signature = item
        last: dict[str, Any] | None = None
        for attempt in range(1, args.task_attempts + 1):
            runtime = VehicleWorldRuntime(dataset.root)
            definitions = vehicle_tool_definitions(
                dataset.tool_schemas,
                timeout_seconds=5,
            )
            task = VehicleTask(
                id=f"{arm}-s{quiz.scenario_index:02d}-{quiz.quiz_id}-r{repeat}",
                scenario_index=quiz.scenario_index,
                event_index=0,
                query=quiz.query,
                gold_memory="",
                reasoning_type=quiz.reasoning_type,
                gold_calls=tuple(
                    GoldToolCall(
                        name=call.name,
                        arguments=dict(call.arguments),
                        source="v2-hybrid-quality-gold",
                    )
                    for call in quiz.gold_calls
                ),
            )

            def resolver(profile: str, query: str) -> VehicleMemoryContext:
                del profile, query
                return VehicleMemoryContext(
                    content=memory,
                    metadata={"arm": arm, "quiz_id": quiz.quiz_id},
                    trace={},
                )

            last = _run_agent_task(
                runtime=runtime,
                definitions=definitions,
                agent_model=agent_model,
                profile="cloud_recursive_summary",
                task=task,
                max_tool_rounds=args.max_tool_rounds,
                max_tool_result_chars=20_000,
                memory_resolver=resolver,
                oracle_retrieval_annotations=None,
                oracle_gate_annotations=None,
                oracle_stage_fact_annotations=None,
            )
            last.update(
                {
                    "evaluation_version": EVALUATION_VERSION,
                    "evaluation_signature": signature,
                    "arm": arm,
                    "quality_scenario_index": quiz.scenario_index,
                    "quiz_id": quiz.quiz_id,
                    "quiz_kind": quiz.quiz_kind,
                    "repeat": repeat,
                    "source_quiz_sha256": quiz.source_quiz_sha256,
                    "memory_sha256": hashlib.sha256(memory.encode()).hexdigest(),
                    "memory_artifact_sha256": artifact_sha256,
                    "evaluation_attempt": attempt,
                    "agent_model": args.model,
                }
            )
            if last.get("status") == "completed":
                break
        assert last is not None
        _write_json(checkpoint, last)
        return last

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(work) or 1)) as pool:
        futures = {pool.submit(evaluate, item): item for item in work}
        for future in as_completed(futures):
            arm, quiz, _, _, repeat, _, _ = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(
                    f"{arm}/S{quiz.scenario_index}/{quiz.quiz_id}/R{repeat}: "
                    f"{type(exc).__name__}: {exc}"
                )
    summary = _aggregate(records, failures=failures, model=args.model)
    _write_json(output_dir / "agent-summary.json", summary)
    if failures:
        raise RuntimeError(f"Agent evaluation failures: {'; '.join(failures)}")
    return summary


def _aggregate(
    records: list[dict[str, Any]], *, failures: list[str], model: str
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["arm"]), "all")].append(record)
        groups[(str(record["arm"]), str(record["quiz_kind"]))].append(record)
    metrics = {}
    for (arm, kind), selected in sorted(groups.items()):
        metrics[f"{arm}/{kind}"] = {
            "tasks": len(selected),
            "completion_rate": _mean(
                item.get("status") == "completed" for item in selected
            ),
            "exact_state_match": _mean(
                bool(item.get("score", {}).get("exact_state_match"))
                for item in selected
            ),
            "tool_f1": _mean(
                float(item.get("score", {}).get("tool_score", {}).get("f1", 0))
                for item in selected
            ),
            "argument_exact_match": _mean(
                bool(item.get("argument_exact_match")) for item in selected
            ),
            "input_tokens": sum(
                int(item.get("usage", {}).get("input_tokens", 0))
                for item in selected
            ),
            "output_tokens": sum(
                int(item.get("usage", {}).get("output_tokens", 0))
                for item in selected
            ),
            "latency_ms": sum(
                int(item.get("usage", {}).get("model_latency_ms", 0))
                for item in selected
            ),
        }
    return {
        "version": EVALUATION_VERSION,
        "status": "COMPLETED" if not failures else "PARTIAL",
        "model": model,
        "record_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "metrics": metrics,
    }


def _signature(
    *,
    arm: str,
    quiz_sha256: str,
    memory: str,
    memory_artifact_sha256: str,
    repeat: int,
    model: str,
) -> str:
    body = {
        "version": EVALUATION_VERSION,
        "arm": arm,
        "quiz_sha256": quiz_sha256,
        "memory_sha256": hashlib.sha256(memory.encode()).hexdigest(),
        "memory_artifact_sha256": memory_artifact_sha256,
        "repeat": repeat,
        "model": model,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mean(values) -> float:
    items = list(values)
    return sum(float(value) for value in items) / len(items) if items else 0.0


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
