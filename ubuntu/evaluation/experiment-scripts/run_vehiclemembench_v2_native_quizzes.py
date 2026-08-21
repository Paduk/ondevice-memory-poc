#!/usr/bin/env python3
"""Generate/resume the common 30 Turn Quizzes and 10 final V1 Quizzes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_generation import V1_DEFAULT_GENERATION_MODEL

DEFAULT_NATIVE_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-native/"
    "native-turnwise-terra-s1-r1"
)
DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v4/stage2.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <native-root>/quiz-30-final-10-v1.",
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--scenario-id", default="scenario-native-turnwise-001")
    parser.add_argument("--public-scenario-index", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("Quiz workers and attempts must be positive")
    native_root = args.native_root.expanduser().resolve(strict=True)
    native_artifact = (native_root / "native.json").resolve(strict=True)
    dialogue_root = (native_root / "dialogues").resolve(strict=True)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else native_root / "quiz-30-final-10-v1"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    script_root = Path(__file__).resolve().parent
    common = [
        "--dataset-root",
        str(args.dataset_root.expanduser().resolve(strict=True)),
        "--stage2-path",
        str(args.stage2_path.expanduser().resolve(strict=True)),
        "--model",
        args.model,
        "--workers",
        str(args.workers),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-attempts",
        str(args.max_attempts),
    ]
    turn_result = _run_json_command(
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v2_turn_quiz_expansion.py"),
            *common,
            "--source-kind",
            "native",
            "--native-artifact",
            str(native_artifact),
            "--output-root",
            str(output_root / "turn-quiz-30"),
            "--target-quiz-count",
            "30",
        ]
    )
    final_result = _run_json_command(
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v1_stage3_smoke.py"),
            *common,
            "--dialogue-root",
            str(dialogue_root),
            "--output-root",
            str(output_root / "final-v1"),
            "--scenario-id",
            args.scenario_id,
            "--public-scenario-index",
            str(args.public_scenario_index),
        ]
    )
    result = {
        "status": "completed",
        "native_artifact": str(native_artifact),
        "output_root": str(output_root),
        "turn_quiz": turn_result,
        "final_quiz": final_result,
    }
    _write_json(output_root / "quiz-run.json", result)
    return result


def _run_json_command(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Quiz subprocess did not return a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
