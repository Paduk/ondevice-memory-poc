#!/usr/bin/env python3
"""Run VehicleMemBench quizzes against a completed V2 critic memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.providers import OpenAIResponsesAgentModel
from palmclaw_ubuntu.vehicle_bench import (
    OFFICIAL_UPSTREAM_COMMIT,
    VehicleMemoryContext,
    load_vehicle_benchmark,
    run_agent_evaluation,
)

_PROFILES = {
    "summary": "cloud_turnwise_recursive_summary",
    "combined": "cloud_turnwise_recursive_summary_patch_temporal_compact",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_final_memory(critic_dir: Path) -> tuple[str, dict[str, Any]]:
    if not (critic_dir / "COMPLETED").is_file():
        raise RuntimeError(f"Critic run is not completed: {critic_dir}")
    summary = _read_json(critic_dir / "run_summary.json")
    final_label: dict[str, Any] | None = None
    with (critic_dir / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Critic label must be a JSON object")
                final_label = item
    if final_label is None:
        raise RuntimeError(f"Critic labels are empty: {critic_dir}")
    memory = str(final_label["critic"]["result"]["approved_memory_after"])
    digest = hashlib.sha256(memory.encode("utf-8")).hexdigest()
    expected = str(summary["final_memory_sha256"])
    if digest != expected:
        raise RuntimeError(f"Final critic memory hash mismatch: {digest} != {expected}")
    return memory, summary


def _load_consensus_final_memory(
    consensus_root: Path,
    scenario_index: int,
) -> tuple[str, dict[str, Any]]:
    scenario_dir = consensus_root / f"scenario-{scenario_index:02d}"
    summary = _read_json(scenario_dir / "run_summary.json")
    if summary.get("status") != "COMPLETED":
        raise RuntimeError(f"Consensus run is not completed: {scenario_dir}")
    memory = str(summary.get("final_memory") or "")
    if not memory:
        raise RuntimeError(f"Consensus final memory is empty: {scenario_dir}")
    digest = hashlib.sha256(memory.encode("utf-8")).hexdigest()
    expected = str(summary["final_memory_sha256"])
    if digest != expected:
        raise RuntimeError(
            f"Final consensus memory hash mismatch: {digest} != {expected}"
        )
    return memory, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(_PROFILES), required=True)
    parser.add_argument("--scenario", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--critic-root", type=Path)
    source.add_argument("--consensus-root", type=Path)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--task-limit", type=int, default=10)
    parser.add_argument("--max-tool-rounds", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    if args.consensus_root is not None:
        consensus_root = args.consensus_root.expanduser().resolve()
        memory, critic_summary = _load_consensus_final_memory(
            consensus_root,
            args.scenario,
        )
        source_name = "vehiclemembench_v2_consensus_final_memory"
        source_run = consensus_root.name
    else:
        critic_dir = (
            args.critic_root
            / args.method
            / f"scenario-{args.scenario:02d}"
        ).expanduser().resolve()
        memory, critic_summary = _load_final_memory(critic_dir)
        source_name = "vehiclemembench_v2_critic_final_memory"
        source_run = args.critic_root.expanduser().resolve().name
    dataset = load_vehicle_benchmark(
        args.benchmark_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
    )
    profile = _PROFILES[args.method]

    def resolve_memory(
        requested_profile: str,
        query: str,
        *unused_args: Any,
        **unused_kwargs: Any,
    ) -> VehicleMemoryContext:
        if requested_profile != profile:
            raise ValueError(
                f"Unexpected profile: {requested_profile}; expected {profile}"
            )
        return VehicleMemoryContext(
            content=memory,
            metadata={
                "source": source_name,
                "source_run": source_run,
                "method": args.method,
                "scenario_index": args.scenario,
                "memory_sha256": critic_summary["final_memory_sha256"],
                "query_dependent": False,
            },
            trace={},
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = OpenAIResponsesAgentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens=2048,
        reasoning_effort="low",
        redact_pii=True,
    )
    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=(profile,),
        scenario_index=args.scenario,
        task_limit=args.task_limit,
        max_tool_rounds=args.max_tool_rounds,
        output_root=output_dir,
        memory_resolver=resolve_memory,
        memory_manifest={
            "status": "ready",
            "source": source_name,
            "source_run": source_run,
            "method": args.method,
            "scenario_index": args.scenario,
            "cache_key": critic_summary["final_memory_sha256"],
            "memory_sha256": critic_summary["final_memory_sha256"],
            "memory_characters": len(memory),
        },
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return int(result.status != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
