#!/usr/bin/env python3
"""Run one resumable Native Turn-wise VehicleMemBench event canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    build_dialogue_generation_contexts,
    dialogue_turn_target,
)
from palmclaw_ubuntu.vehicle_bench.v2_native_turnwise import (
    OpenAIV2NativeAlignmentModel,
    OpenAIV2NativeTurnGenerationModel,
    run_native_event,
)

DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v4/stage2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-native/"
    "native-turnwise-terra-s1-event-canary-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--event-index",
        type=int,
        default=51,
        help="Isolated event index; all earlier events must contain no updates.",
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    stage2 = V1Stage2Artifact.model_validate_json(
        args.stage2_path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    contexts = build_dialogue_generation_contexts(stage2)
    if not 0 <= args.event_index < len(contexts):
        raise ValueError("--event-index is outside the Stage 2 timeline")
    if any(
        context.event.preference_updates for context in contexts[: args.event_index]
    ):
        raise ValueError(
            "isolated Native canary cannot skip an earlier preference update"
        )
    context = contexts[args.event_index]
    turn_model = OpenAIV2NativeTurnGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    alignment_model = OpenAIV2NativeAlignmentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    artifact = run_native_event(
        stage2.persona_group.payload.personas,
        context,
        turn_model,
        alignment_model,
        output_dir=args.output_root,
        target_turn_count=dialogue_turn_target(context),
        previous_entries=(),
        global_turn_offset=0,
        previous_checkpoint_sha256=None,
        max_attempts=args.max_attempts,
    )
    return {
        "status": "completed",
        "event_id": artifact.event_id,
        "artifact_sha256": artifact.artifact_sha256,
        "output_path": str(args.output_root.expanduser().resolve() / "event.json"),
        "audit": artifact.audit.model_dump(mode="json"),
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
