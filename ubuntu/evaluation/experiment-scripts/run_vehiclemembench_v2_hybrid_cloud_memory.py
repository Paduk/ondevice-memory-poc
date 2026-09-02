#!/usr/bin/env python3
"""Generate Turn-wise Summary/Combined memories over frozen Hybrid dialogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.memory import (
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_INSTRUCTIONS,
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_PROMPT_VERSION,
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_COMPACT_PROMPT_VERSION,
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_COMPACTION_INSTRUCTIONS,
    VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_PATCH_INSTRUCTIONS,
)
from palmclaw_ubuntu.vehicle_bench.v2_cloud_memory_gap import (
    execute_cloud_memory_replay,
    load_hybrid_cloud_replay_plan,
)
from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    default_quality_artifact_paths,
)

DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT / "vehiclemembench-v2-hybrid-cloud-memory-gap"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, required=True, choices=range(1, 101))
    parser.add_argument(
        "--arm",
        required=True,
        choices=("turnwise_summary", "turnwise_combined"),
    )
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--instruction-mode",
        choices=("legacy", "training_aligned"),
        default="legacy",
        help=(
            "Use the historical Combined instructions or an adapter derived "
            "from the exact on-device Patch SFT system prompt"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-output-tokens", type=int, default=2_048)
    parser.add_argument("--max-memory-chars", type=int, default=8_192)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--batch-token-limit", type=int, default=8_000)
    parser.add_argument("--compaction-adds", type=int, default=64)
    parser.add_argument("--compaction-tokens", type=int, default=1_000)
    parser.add_argument("--compaction-target-ratio", type=float, default=0.70)
    parser.add_argument("--max-compaction-attempts", type=int, default=4)
    parser.add_argument(
        "--redact-pii",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    # Import after vehicle_bench package initialization to avoid the established
    # providers <-> vehicle_fact_ontology package cycle.
    from palmclaw_ubuntu.providers import (
        OpenAICompactingTemporalAwareRecursiveSummaryPatchMemoryModel,
        OpenAIRecursiveSummaryMemoryModel,
    )

    selected = next(
        (
            paths
            for paths in default_quality_artifact_paths(
                args.evaluation_root,
                (args.scenario,),
            )
            if paths.method == "hybrid" and paths.scenario_index == args.scenario
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"Hybrid scenario is unavailable: S{args.scenario}")
    plan = load_hybrid_cloud_replay_plan(
        selected,
        max_batch_tokens=args.batch_token_limit,
    )
    common = {
        "model_id": args.model,
        "timeout_seconds": args.timeout_seconds,
        "max_output_tokens": args.max_output_tokens,
        "max_memory_chars": args.max_memory_chars,
        "update_cadence": "history_entry",
        "reasoning_effort": "low",
        "redact_pii": args.redact_pii,
    }
    if args.arm == "turnwise_summary":
        model = OpenAIRecursiveSummaryMemoryModel(
            **common,
            instructions=VEHICLE_TURNWISE_RECURSIVE_SUMMARY_INSTRUCTIONS,
            prompt_version=VEHICLE_TURNWISE_RECURSIVE_SUMMARY_PROMPT_VERSION,
        )
    else:
        instructions = VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_PATCH_INSTRUCTIONS
        compaction_instructions = (
            VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_COMPACTION_INSTRUCTIONS
        )
        prompt_version = (
            VEHICLE_TURNWISE_RECURSIVE_SUMMARY_TEMPORAL_COMPACT_PROMPT_VERSION
        )
        if args.instruction_mode == "training_aligned":
            instructions = _training_aligned_combined_instructions()
            compaction_instructions = _training_aligned_compaction_instructions()
            prompt_version = (
                "vehicle-turnwise-combined-training-aligned-patch-sft-v1"
            )
        model = OpenAICompactingTemporalAwareRecursiveSummaryPatchMemoryModel(
            **common,
            instructions=instructions,
            prompt_version=prompt_version,
            compaction_add_threshold=args.compaction_adds,
            compaction_token_threshold=args.compaction_tokens,
            compaction_target_ratio=args.compaction_target_ratio,
            max_compaction_attempts=args.max_compaction_attempts,
            compaction_instructions=compaction_instructions,
        )
    destination = args.output_root / args.arm / f"s{args.scenario:02d}"
    result = execute_cloud_memory_replay(
        plan,
        arm=args.arm,
        model=model,
        output_dir=destination,
        max_attempts=args.max_attempts,
    )
    return {
        key: result[key]
        for key in (
            "status",
            "arm",
            "scenario_index",
            "dialogue_turn_count",
            "batch_count",
            "update_count",
            "noop_count",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "latency_ms",
            "final_memory_sha256",
            "artifact_sha256",
        )
    }


def _training_aligned_combined_instructions() -> str:
    """Adapt the exact Patch SFT instruction only at the cloud Tool boundary."""
    from memory_training.methods.patch import PATCH_SYSTEM_PROMPT

    instruction = PATCH_SYSTEM_PROMPT.replace(
        "Given previous_memory and the current turn, return exactly one JSON object.",
        "Given previous_memory and the current turn, decide whether memory changes.",
    ).replace(
        'If nothing changes, return {"decision":"NO_OP"}.',
        "If nothing changes, do not call any tool.",
    ).replace(
        'Otherwise return {"decision":"UPDATE","operations":[...]}.',
        "Otherwise call memory_patch exactly once with operations.",
    ).replace(
        "Each operation has exactly op, target, content. "
        "op is add, replace, or delete.",
        "Each operation uses op, target, content. op is add, replace, or delete.",
    )
    return instruction + """

Cloud Tool adapter: memory_patch additionally requires identity_key,
temporal_action, and temporal_cue on every operation. Use a stable
subject.setting identity_key. Use durable_upsert for stable preferences,
current_upsert for an observed current state, temporary_override only for an
explicitly temporary state, end_temporary only for an explicit end,
conditional_upsert for a reusable condition, and non_temporal otherwise.
temporal_cue must be the shortest exact source phrase for temporary_override
or end_temporary and empty otherwise. Preserve a durable baseline when adding
or ending a temporary override. Do not infer elapsed-time endings.
""".strip()


def _training_aligned_compaction_instructions() -> str:
    """Use the Summary SFT preservation contract for Combined compaction."""
    from memory_training.methods.summary import SUMMARY_SYSTEM_PROMPT

    return SUMMARY_SYSTEM_PROMPT + """

Cloud Tool adapter: this is a maintenance rewrite of materialized memory, not
a new conversation turn. Call memory_update exactly once with the complete
compacted memory. Preserve every still-valid fact and keep durable baselines
separate from current, temporary, and conditional states. Do not infer that a
temporary state ended merely because time elapsed.
""".strip()


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
