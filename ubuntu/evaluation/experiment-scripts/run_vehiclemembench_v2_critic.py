"""Review one VehicleMemBench R2 turn-wise memory trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.critic import (
    OpenAIVehicleMemoryAdjudicatorModel,
    OpenAIVehicleMemoryCriticModel,
)
from palmclaw_ubuntu.vehicle_bench.critic_metrics import build_critic_metrics
from palmclaw_ubuntu.vehicle_bench.critic_pipeline import (
    CriticCheckpointStore,
    VehicleMemoryCriticPipeline,
)
from palmclaw_ubuntu.vehicle_bench.critic_trace import (
    CriticMethod,
    CriticSourceTurn,
    CriticTraceSource,
    discover_r2_trace_source,
    extract_source_turns,
)

DEFAULT_ARTIFACT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/"
    "turnwise-memory-critic-r2"
)
RUNNER_VERSION = "vehiclemembench-v2-turnwise-memory-critic-runner-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("summary", "combined"), required=True)
    parser.add_argument("--scenario", type=int, choices=range(1, 51), required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--luna-model", default="gpt-5.6-luna")
    parser.add_argument("--terra-model", default="gpt-5.6-terra")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-output-tokens", type=int, default=2_048)
    parser.add_argument("--max-semantic-retries", type=int, default=2)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--retrieval-neighbor-window", type=int, default=1)
    parser.add_argument("--max-rollbacks", type=int, default=16)
    parser.add_argument(
        "--max-commits",
        type=int,
        help="Stop after this many commits in this invocation (resume-safe smoke).",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be positive")
    if args.max_semantic_retries < 0:
        raise ValueError("--max-semantic-retries cannot be negative")
    if args.retrieval_top_k < 1:
        raise ValueError("--retrieval-top-k must be positive")
    if args.retrieval_neighbor_window < 0:
        raise ValueError("--retrieval-neighbor-window cannot be negative")
    if args.max_rollbacks < 0:
        raise ValueError("--max-rollbacks cannot be negative")
    if args.max_commits is not None and args.max_commits < 1:
        raise ValueError("--max-commits must be positive")

    method: CriticMethod = args.method
    source = discover_r2_trace_source(
        args.artifact_root,
        method=method,
        scenario_index=args.scenario,
    )
    turns = extract_source_turns(source)
    output_dir = (
        args.output_root.expanduser().resolve()
        / method
        / f"scenario-{args.scenario:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "source_manifest.json",
        _source_manifest(source, turns),
    )
    _write_running_marker(output_dir, method, args.scenario)

    store = CriticCheckpointStore(
        output_dir / "critic_state.sqlite",
        turns=turns,
    )
    luna = OpenAIVehicleMemoryCriticModel(
        args.luna_model,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        max_semantic_retries=args.max_semantic_retries,
    )
    terra = OpenAIVehicleMemoryAdjudicatorModel(
        args.terra_model,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        max_semantic_retries=args.max_semantic_retries,
    )
    pipeline = VehicleMemoryCriticPipeline(
        turns=turns,
        store=store,
        luna=luna,
        terra=terra,
        max_memory_chars=source.max_memory_chars,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_neighbor_window=args.retrieval_neighbor_window,
        max_rollbacks=args.max_rollbacks,
    )
    if store.state().status == "UNRESOLVED":
        pipeline.resume_unresolved_as_defer_keep()
    result = pipeline.run(max_commits=args.max_commits)

    _write_enriched_labels(output_dir / "labels.jsonl", store, turns)
    store.export_review_queue(output_dir / "review_queue.jsonl")
    metrics = build_critic_metrics(store, turns=turns)
    _write_json(output_dir / "metrics.json", metrics)
    summary = {
        "runner_version": RUNNER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "method": result.method,
        "scenario_index": result.scenario_index,
        "status": result.status,
        "processed_turns": result.processed_turns,
        "turn_count": result.turn_count,
        "replay_generation": result.replay_generation,
        "final_memory_sha256": result.final_memory_sha256,
        "attempt_count": result.attempt_count,
        "rollback_count": result.rollback_count,
        "unresolved_count": len(store.unresolved_events()),
        "estimated_cost_usd": metrics["total"]["estimated_cost_usd"],
        "provider_latency_seconds": metrics["total"]["latency_seconds"],
        "full_run_extrapolation": metrics["full_run_extrapolation"],
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_status_marker(output_dir, result.status, summary)
    return summary


def _source_manifest(
    source: CriticTraceSource,
    turns: tuple[CriticSourceTurn, ...],
) -> dict[str, Any]:
    source_data = asdict(source)
    for key in ("experiment_root", "database_path", "cache_manifest_path"):
        source_data[key] = str(source_data[key])
    return {
        "runner_version": RUNNER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source_data,
        "database_sha256": _file_sha256(source.database_path),
        "turn_count": len(turns),
        "update_count": sum(turn.original_decision == "UPDATE" for turn in turns),
        "noop_count": sum(turn.original_decision == "NO_OP" for turn in turns),
        "first_message_id": turns[0].message_id,
        "last_message_id": turns[-1].message_id,
    }


def _write_enriched_labels(
    path: Path,
    store: CriticCheckpointStore,
    turns: tuple[CriticSourceTurn, ...],
) -> None:
    source_by_index = {turn.turn_index: turn for turn in turns}
    attempts_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for attempt in store.attempts():
        key = (int(attempt["turn_index"]), int(attempt["replay_generation"]))
        attempts_by_key.setdefault(key, []).append(dict(attempt))
    records = []
    for item in store.active_results():
        turn = source_by_index[int(item["turn_index"])]
        key = (turn.turn_index, int(item["replay_generation"]))
        records.append(
            {
                "method": turn.method,
                "scenario_index": turn.scenario_index,
                "turn_index": turn.turn_index,
                "message_id": turn.message_id,
                "date": turn.date,
                "source": {
                    "consolidation_run_id": turn.consolidation_run_id,
                    "model_call_id": turn.model_call_id,
                    "original_decision": turn.original_decision,
                    "original_before_sha256": turn.original_before_sha256,
                    "original_after_sha256": turn.original_after_sha256,
                    "model_id": turn.model_id,
                    "prompt_version": turn.prompt_version,
                    "schema_version": turn.schema_version,
                },
                "critic": item,
                "model_attempts": attempts_by_key.get(key, []),
            }
        )
    _write_jsonl(path, records)


def _write_status_marker(
    output_dir: Path,
    status: str,
    summary: dict[str, Any],
) -> None:
    marker = "PARTIAL" if status == "RUNNING" else status
    for name in ("RUNNING", "PARTIAL", "COMPLETED", "UNRESOLVED", "FAILED"):
        candidate = output_dir / name
        if candidate.exists():
            candidate.unlink()
    _write_json(output_dir / marker, summary)


def _write_running_marker(
    output_dir: Path,
    method: CriticMethod,
    scenario_index: int,
) -> None:
    for name in ("RUNNING", "PARTIAL", "COMPLETED", "UNRESOLVED", "FAILED"):
        candidate = output_dir / name
        if candidate.exists():
            candidate.unlink()
    _write_json(
        output_dir / "RUNNING",
        {
            "runner_version": RUNNER_VERSION,
            "started_at": datetime.now(UTC).isoformat(),
            "method": method,
            "scenario_index": scenario_index,
        },
    )


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in records
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = (
        args.output_root.expanduser().resolve()
        / args.method
        / f"scenario-{args.scenario:02d}"
    )
    try:
        summary = run(args)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("RUNNING", "PARTIAL", "COMPLETED", "UNRESOLVED", "FAILED"):
            candidate = output_dir / name
            if candidate.exists():
                candidate.unlink()
        failure = {
            "runner_version": RUNNER_VERSION,
            "failed_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(output_dir / "FAILED", failure)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
