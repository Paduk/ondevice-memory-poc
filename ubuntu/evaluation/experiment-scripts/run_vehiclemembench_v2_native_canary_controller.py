#!/usr/bin/env python3
"""Run a full Native scenario and its common 30+10 Quiz workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_generation import V1_DEFAULT_GENERATION_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--scenario-id", default="scenario-native-turnwise-001")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage2_path = args.stage2_path.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    script_root = Path(__file__).resolve().parent
    common = [
        "--stage2-path",
        str(stage2_path),
        "--model",
        args.model,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-attempts",
        str(args.max_attempts),
    ]
    scenario = _run_logged_json(
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v2_native_turnwise_scenario.py"),
            *common,
            "--output-root",
            str(output_root),
        ],
        output_root / "scenario-run.log",
    )
    quizzes = _run_logged_json(
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v2_native_quizzes.py"),
            *common,
            "--dataset-root",
            str(dataset_root),
            "--native-root",
            str(output_root),
            "--workers",
            str(args.workers),
            "--scenario-id",
            args.scenario_id,
        ],
        output_root / "quiz-run.log",
    )
    result = {
        "status": "completed",
        "scenario": scenario,
        "quizzes": quizzes,
    }
    _write_json(output_root / "canary-run.json", result)
    return result


def _run_logged_json(command: list[str], log_path: Path) -> dict[str, object]:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Native canary subprocess failed: {log_path}")
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Native canary subprocess returned invalid JSON: {log_path}")
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
