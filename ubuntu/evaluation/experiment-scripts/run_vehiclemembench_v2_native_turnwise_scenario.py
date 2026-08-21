#!/usr/bin/env python3
"""Run one resumable full Native Turn-wise VehicleMemBench V2 scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_native_turnwise import (
    OpenAIV2NativeAlignmentModel,
    OpenAIV2NativeTurnGenerationModel,
    run_native_scenario,
)

DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v4/stage2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-native/"
    "native-turnwise-terra-s1-r1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--event-limit",
        type=int,
        help="Optional causal-prefix smoke; omit to process all 80 events.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    stage2 = V1Stage2Artifact.model_validate_json(
        args.stage2_path.expanduser().resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    turn_model = OpenAIV2NativeTurnGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    alignment_model = OpenAIV2NativeAlignmentModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    artifact = run_native_scenario(
        stage2,
        turn_model,
        alignment_model,
        output_dir=args.output_root,
        event_limit=args.event_limit,
        max_attempts=args.max_attempts,
    )
    return {
        "status": "completed" if artifact.audit.completed else "partial",
        "artifact_path": str(args.output_root.expanduser().resolve() / "native.json"),
        "artifact_sha256": artifact.artifact_sha256,
        "audit": artifact.audit.model_dump(mode="json"),
    }


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
