#!/usr/bin/env python3
"""Run one checkpointed VehicleMemBench V2 consensus-Gold scenario."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.consensus_gold import (
    CONSENSUS_GOLD_PATCH_R2_PROMPT_VERSION,
    CONSENSUS_GOLD_PROMPT_VERSION,
    CONSENSUS_GOLD_SCHEMA_VERSION,
    V2_PATCH_R2_COMPACTION_OUTPUT_INSTRUCTIONS,
    V2_PATCH_R2_OUTPUT_INSTRUCTIONS,
    OpenAIV2CandidateResolverModel,
    OpenAIV2CombinedCandidateModel,
    OpenAIV2CompactionCandidateModel,
    OpenAIV2CompactionResolverModel,
    build_compaction_input,
    build_gold_generation_input,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.consensus_gold_pipeline import (
    GOLD_CHECKPOINT_SCHEMA_VERSION,
    ConsensusGoldPipeline,
    GoldCheckpointStore,
)
from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark
from palmclaw_ubuntu.vehicle_bench.human_review import (
    HumanReviewQueue,
    build_reviewed_candidate,
    build_reviewed_compaction_candidate,
)
from palmclaw_ubuntu.vehicle_bench.memory import (
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_INSTRUCTIONS,
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_PATCH_INSTRUCTIONS,
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_COMPACTION_INSTRUCTIONS,
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_PATCH_INSTRUCTIONS,
    parse_vehicle_history,
)
from palmclaw_ubuntu.vehicle_bench.qa_coverage import (
    OpenAIV2FinalMemoryCoverageModel,
    QACoverageReport,
    qa_coverage_recommendation,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/consensus-gold-v2"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--review-queue",
        type=Path,
        help=(
            "Central Human Review SQLite path "
            "(default: OUTPUT_ROOT/review_queue.sqlite)."
        ),
    )
    parser.add_argument("--scenario", type=int, choices=range(1, 51), default=1)
    parser.add_argument("--luna-model", default="gpt-5.6-luna")
    parser.add_argument("--terra-model", default="gpt-5.6-terra")
    parser.add_argument("--sol-model", default="gpt-5.6-sol")
    parser.add_argument("--qa-coverage-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--memory-policy",
        choices=("temporal", "patch-r2"),
        default="temporal",
        help=(
            "Memory-writing policy inside the consensus pipeline. patch-r2 keeps "
            "the original non-temporal Turn-wise Patch representation."
        ),
    )
    parser.add_argument(
        "--qa-failure-threshold",
        type=int,
        default=3,
        help=(
            "Maximum allowed MISSING tasks; with 10 tasks, regeneration is "
            "recommended when at least four tasks are MISSING."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-candidate-attempts", type=int, default=2)
    parser.add_argument("--max-memory-chars", type=int, default=8_192)
    parser.add_argument("--compaction-add-threshold", type=int, default=30)
    parser.add_argument("--compaction-target-ratio", type=float, default=0.70)
    parser.add_argument(
        "--max-commits",
        type=int,
        default=1,
        help="Commit at most this many turns (default: one-turn canary).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run every remaining S1 turn instead of --max-commits.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Permit model calls and checkpoint writes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_vehicle_benchmark(args.dataset_root, strict=True)
    scenario = dataset.scenario(args.scenario)
    entries = tuple(
        sorted(
            parse_vehicle_history(scenario.history_path),
            key=lambda entry: (entry.timestamp, entry.line_number),
        )
    )
    output_dir = (
        args.output_root.expanduser().resolve() / f"scenario-{args.scenario:02d}"
    )
    review_queue_path = (
        args.review_queue.expanduser().resolve()
        if args.review_queue is not None
        else args.output_root.expanduser().resolve() / "review_queue.sqlite"
    )
    preview = {
        "scenario_index": args.scenario,
        "turn_count": len(entries),
        "dataset_sha256": dataset.manifest.dataset_sha256,
        "output_dir": str(output_dir),
        "max_commits": None if args.run_all else args.max_commits,
        "execute": args.execute,
        "review_queue": str(review_queue_path),
        "memory_policy": args.memory_policy,
    }
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required with --execute")

    output_dir.mkdir(parents=True, exist_ok=True)
    store = GoldCheckpointStore(
        output_dir / "checkpoint.sqlite",
        scenario_index=args.scenario,
        entries=entries,
    )
    if args.memory_policy == "patch-r2":
        candidate_instructions = VEHICLE_TURNWISE_RECURSIVE_SUMMARY_PATCH_INSTRUCTIONS
        candidate_output_instructions = V2_PATCH_R2_OUTPUT_INSTRUCTIONS
        compaction_instructions = VEHICLE_TURNWISE_RECURSIVE_SUMMARY_INSTRUCTIONS
        compaction_output_instructions = (
            V2_PATCH_R2_COMPACTION_OUTPUT_INSTRUCTIONS
        )
        prompt_version = CONSENSUS_GOLD_PATCH_R2_PROMPT_VERSION
    else:
        candidate_instructions = (
            VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_PATCH_INSTRUCTIONS
        )
        candidate_output_instructions = None
        compaction_instructions = (
            VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_COMPACTION_INSTRUCTIONS
        )
        compaction_output_instructions = None
        prompt_version = CONSENSUS_GOLD_PROMPT_VERSION

    candidate_kwargs: dict[str, Any] = {
        "timeout_seconds": args.timeout_seconds,
        "base_instructions": candidate_instructions,
        "prompt_version": prompt_version,
    }
    if candidate_output_instructions is not None:
        candidate_kwargs["output_instructions"] = candidate_output_instructions
    luna = OpenAIV2CombinedCandidateModel(
        args.luna_model,
        **candidate_kwargs,
    )
    terra = OpenAIV2CandidateResolverModel(
        args.terra_model,
        stage="TERRA",
        timeout_seconds=args.timeout_seconds,
    )
    sol = OpenAIV2CandidateResolverModel(
        args.sol_model,
        stage="SOL",
        timeout_seconds=args.timeout_seconds,
    )
    compaction_kwargs: dict[str, Any] = {
        "timeout_seconds": args.timeout_seconds,
        "base_instructions": compaction_instructions,
        "prompt_version": prompt_version,
    }
    if compaction_output_instructions is not None:
        compaction_kwargs["output_instructions"] = compaction_output_instructions
    compactor_luna = OpenAIV2CompactionCandidateModel(
        args.luna_model,
        **compaction_kwargs,
    )
    compactor_terra = OpenAIV2CompactionResolverModel(
        args.terra_model,
        stage="TERRA",
        timeout_seconds=args.timeout_seconds,
    )
    compactor_sol = OpenAIV2CompactionResolverModel(
        args.sol_model,
        stage="SOL",
        timeout_seconds=args.timeout_seconds,
    )
    pipeline = ConsensusGoldPipeline(
        scenario_index=args.scenario,
        entries=entries,
        store=store,
        luna=luna,
        terra=terra,
        sol=sol,
        compactor_luna=compactor_luna,
        compactor_terra=compactor_terra,
        compactor_sol=compactor_sol,
        max_candidate_attempts=args.max_candidate_attempts,
        max_memory_chars=args.max_memory_chars,
        compaction_add_threshold=args.compaction_add_threshold,
        compaction_target_ratio=args.compaction_target_ratio,
    )
    review_queue = HumanReviewQueue(review_queue_path)
    paused_state = store.state()
    if paused_state.status == "PAUSED_REVIEW":
        review_queue.register_paused(
            checkpoint_path=store.path,
            state=paused_state,
            entry=entries[paused_state.next_turn_index],
        )
        submitted = review_queue.submitted_for(
            scenario_index=args.scenario,
            turn_index=paused_state.next_turn_index,
            checkpoint_path=store.path,
        )
        if submitted is not None:
            try:
                pending_payload = paused_state.pending_review["payload"]
                if pending_payload.get("review_kind") == "COMPACTION":
                    compaction_input = build_compaction_input(
                        scenario_index=args.scenario,
                        turn_index=paused_state.next_turn_index,
                        post_patch_memory=pending_payload["post_patch_memory"],
                        target_ratio=args.compaction_target_ratio,
                    )
                    reviewed_compaction = build_reviewed_compaction_candidate(
                        submitted,
                        compaction_input,
                    )
                    reviewed_state = pipeline.resume_human_compaction(
                        reviewed_compaction
                    )
                else:
                    generation_input = build_gold_generation_input(
                        scenario_index=args.scenario,
                        turn_index=paused_state.next_turn_index,
                        entry=entries[paused_state.next_turn_index],
                        previous_memory=paused_state.approved_memory,
                    )
                    reviewed_candidate = build_reviewed_candidate(
                        submitted,
                        generation_input,
                    )
                    reviewed_state = pipeline.resume_human_candidate(
                        reviewed_candidate
                    )
                transitioned_to_compaction = (
                    reviewed_state.status == "PAUSED_REVIEW"
                    and reviewed_state.pending_review is not None
                    and reviewed_state.pending_review.get("payload", {}).get(
                        "review_kind"
                    )
                    == "COMPACTION"
                )
                if (
                    reviewed_state.next_turn_index <= paused_state.next_turn_index
                    and not transitioned_to_compaction
                ):
                    raise RuntimeError("Human Review did not advance the paused turn")
                review_queue.mark_applied(submitted.review_id)
            except (RuntimeError, ValueError) as error:
                review_queue.mark_apply_error(submitted.review_id, error)
                raise
    result = pipeline.run(max_commits=None if args.run_all else args.max_commits)
    if result.status == "PAUSED_REVIEW":
        current_state = store.state()
        review_queue.register_paused(
            checkpoint_path=store.path,
            state=current_state,
            entry=entries[current_state.next_turn_index],
        )
    manifest = {
        **preview,
        "execute": True,
        "schema_version": CONSENSUS_GOLD_SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "checkpoint_schema_version": GOLD_CHECKPOINT_SCHEMA_VERSION,
        "models": {
            "luna": args.luna_model,
            "terra": args.terra_model,
            "sol": args.sol_model,
        },
        "reasoning_effort": {"luna": "medium", "terra": "high", "sol": "high"},
        "compaction": {
            "add_threshold": args.compaction_add_threshold,
            "target_ratio": args.compaction_target_ratio,
            "escalation": "sol_direct_then_human",
        },
        "final_qa_coverage": {
            "model": args.qa_coverage_model,
            "failure_threshold": args.qa_failure_threshold,
            "evidence_source": "final_memory_only",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_labels(output_dir / "labels.jsonl", store)
    state = store.state()
    summary = {
        **result.__dict__,
        "committed_turns": state.next_turn_index,
        "pending_review": state.pending_review,
    }
    if result.status == "COMPLETED":
        coverage = _load_or_run_qa_coverage(
            path=output_dir / "qa_coverage_validation.json",
            scenario_index=args.scenario,
            final_memory=result.final_memory,
            tasks=scenario.tasks,
            model_id=args.qa_coverage_model,
            failure_threshold=args.qa_failure_threshold,
            timeout_seconds=args.timeout_seconds,
        )
        summary["qa_coverage"] = {
            "recommendation": coverage.recommendation,
            "counts": coverage.counts,
            "failed_task_count": coverage.failed_task_count,
            "artifact": str(output_dir / "qa_coverage_validation.json"),
        }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return int(result.status in {"PAUSED_REVIEW", "FAILED"})


def _write_labels(path: Path, store: GoldCheckpointStore) -> None:
    content = "".join(
        json.dumps(label.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        + "\n"
        for label in store.active_labels()
    )
    _atomic_write(path, content)


def _load_or_run_qa_coverage(
    *,
    path: Path,
    scenario_index: int,
    final_memory: str,
    tasks: tuple,
    model_id: str,
    failure_threshold: int,
    timeout_seconds: float,
) -> QACoverageReport:
    if path.is_file():
        report = QACoverageReport.model_validate_json(path.read_text(encoding="utf-8"))
        if report.model_id != model_id:
            raise ValueError("Existing QA coverage model does not match")
        if report.failure_threshold != failure_threshold:
            raise ValueError("Existing QA coverage threshold does not match")
        if report.final_memory_sha256 != text_sha256(final_memory):
            raise ValueError("Existing QA coverage final-memory hash does not match")
        recommendation = qa_coverage_recommendation(
            report.counts,
            failure_threshold=failure_threshold,
        )
        missing_count = int(report.counts.get("MISSING", 0))
        if (
            report.recommendation != recommendation
            or report.failed_task_count != missing_count
        ):
            report = report.model_copy(
                update={
                    "recommendation": recommendation,
                    "failed_task_count": missing_count,
                }
            )
            _write_json(path, report.model_dump(mode="json"))
        return report
    model = OpenAIV2FinalMemoryCoverageModel(
        model_id,
        timeout_seconds=timeout_seconds,
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            report = model.audit(
                scenario_index=scenario_index,
                final_memory=final_memory,
                tasks=tasks,
                failure_threshold=failure_threshold,
            )
            _write_json(path, report.model_dump(mode="json"))
            return report
        except (RuntimeError, ValueError) as error:
            last_error = error
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("QA coverage failed") from last_error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
