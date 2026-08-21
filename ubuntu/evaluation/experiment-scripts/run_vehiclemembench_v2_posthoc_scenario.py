#!/usr/bin/env python3
"""Run one resumable Post-hoc VehicleMemBench V2 scenario from frozen Stage 2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_generation import V1_DEFAULT_GENERATION_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--stage2-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--public-scenario-index", type=int, required=True)
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("workers and attempts must be positive")
    if args.public_scenario_index < 1:
        raise ValueError("public scenario index must be positive")

    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    stage2_path = args.stage2_path.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dialogue_root = output_root / "dialogues"
    script_root = Path(__file__).resolve().parent

    common = [
        "--dataset-root",
        str(dataset_root),
        "--stage2-path",
        str(stage2_path),
        "--model",
        args.model,
        "--workers",
        str(args.workers),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-attempts",
        str(args.max_attempts),
    ]

    dialogue_result = _run_json_command(
        "complete V1 dialogues",
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v1_stage3_smoke.py"),
            *common,
            "--output-root",
            str(output_root),
            "--dialogue-root",
            str(dialogue_root),
            "--dialogue-only",
        ],
    )

    # Only after every V1 dialogue exists, recover causal turn labels and memory.
    alignment_result = _run_json_command(
        "recover Post-hoc turn labels",
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v2_hybrid_smoke.py"),
            "--stage2-path",
            str(stage2_path),
            "--output-root",
            str(output_root),
            "--dialogue-root",
            str(dialogue_root),
            "--model",
            args.model,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--max-attempts",
            str(args.max_attempts),
        ],
    )

    immediate_root = output_root / "turn-quiz-immediate"
    immediate_result = _run_json_command(
        "generate immediate Turn Quizzes",
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v2_turn_quiz_smoke.py"),
            *common,
            "--hybrid-artifact",
            str(output_root / "hybrid.json"),
            "--dialogue-root",
            str(dialogue_root),
            "--output-root",
            str(immediate_root),
        ],
    )

    turn_quiz_result = _run_json_command(
        "expand to 30 Turn Quizzes",
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v2_turn_quiz_expansion.py"),
            *common,
            "--source-kind",
            "hybrid",
            "--hybrid-artifact",
            str(output_root / "hybrid.json"),
            "--dialogue-root",
            str(dialogue_root),
            "--immediate-artifact",
            str(immediate_root / "turn-quizzes.json"),
            "--output-root",
            str(output_root / "turn-quiz-30"),
            "--target-quiz-count",
            "30",
        ],
    )

    final_result = _run_json_command(
        "generate 10 final V1 Quizzes",
        [
            sys.executable,
            str(script_root / "run_vehiclemembench_v1_stage3_smoke.py"),
            *common,
            "--output-root",
            str(output_root / "final-v1"),
            "--dialogue-root",
            str(dialogue_root),
            "--scenario-id",
            args.scenario_id,
            "--public-scenario-index",
            str(args.public_scenario_index),
        ],
    )

    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "vehiclemembench-v2-posthoc-run-v1",
        "method": "post_hoc",
        "status": "completed",
        "scenario_id": args.scenario_id,
        "public_scenario_index": args.public_scenario_index,
        "source_stage2_path": str(stage2_path),
        "source_stage2_sha256": stage2["artifact_sha256"],
        "output_root": str(output_root),
        "dialogue": dialogue_result,
        "alignment": alignment_result,
        "immediate_turn_quiz": immediate_result,
        "turn_quiz": turn_quiz_result,
        "final_quiz": final_result,
    }
    _write_json(output_root / "posthoc-run.json", result)
    return result


def _run_json_command(label: str, command: list[str]) -> dict[str, object]:
    print(f"[post_hoc] {label}", file=sys.stderr, flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4_000:]
        raise RuntimeError(f"{label} failed:\n{detail}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} did not return a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
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
