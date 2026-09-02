#!/usr/bin/env python3
"""Evaluate V1 final quizzes from an on-device memory replay artifact."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from memory_training.ollama_client import OllamaAgentModel, OllamaClient

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.memory import VehicleMemoryContext
from palmclaw_ubuntu.vehicle_bench.runner import (
    run_agent_evaluation,
    vehicle_agent_metrics,
)

PROFILE = "cloud_recursive_summary"


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    client = OllamaClient(args.ollama_url, timeout_seconds=args.timeout_seconds)
    agent = OllamaAgentModel(
        client,
        args.model,
        context_length=args.context_length,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    )
    reports = []
    records = []
    for scenario in args.scenarios:
        replay_path = args.memory_replay_root / f"s{scenario:03d}" / "summary.json"
        replay = _read_json(replay_path)
        memory = str(replay.get("final_memory") or "")
        source = str(replay_path)

        def resolver(
            profile: str,
            query: str,
            *,
            content: str = memory,
            source_path: str = source,
        ) -> VehicleMemoryContext:
            del profile, query
            return VehicleMemoryContext(
                content=content,
                metadata={"source": source_path, "label": args.label},
                trace={},
            )

        result = run_agent_evaluation(
            dataset,
            agent_model=agent,
            profile_names=(PROFILE,),
            scenario_index=scenario,
            task_limit=10,
            max_tool_rounds=args.max_tool_rounds,
            output_root=args.output_root / "runs",
            new_run_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"palmclaw:v1-final-quiz:{args.label}:s{scenario:03d}",
                )
            ),
            memory_resolver=resolver,
            memory_manifest={
                "kind": "ondevice-v1-final-memory",
                "label": args.label,
                "replay_path": str(replay_path),
                "memory_sha256": replay.get("final_memory_sha256"),
            },
        )
        reports.append(
            {
                "scenario_index": scenario,
                "run_id": result.run_id,
                "status": result.status,
                "metrics": result.metrics,
            }
        )
        records.extend(result.tasks)
    aggregate = vehicle_agent_metrics(records, (PROFILE,))
    report = {
        "schema_version": "palmclaw-ondevice-v1-final-quiz-v1",
        "label": args.label,
        "agent_model": args.model,
        "ollama_url": args.ollama_url,
        "scenarios": args.scenarios,
        "aggregate": aggregate,
        "scenario_reports": reports,
    }
    _write_json(args.output_root / "summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench")
    )
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--max-tool-rounds", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
