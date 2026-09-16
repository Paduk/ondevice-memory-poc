"""Prepare one isolated evaluation root for an external-adapted HVP suite."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from memory_training.dataset import build_catalog
from memory_training.scripts.prepare_human_authored_pilot_evaluation import (
    build as build_single,
)
from memory_training.scripts.prepare_human_authored_pilot_evaluation import (
    load_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-base-root",
        type=Path,
        default=Path("evaluation/human-authored-vehicle-memory"),
    )
    parser.add_argument(
        "--scenario-derived-root",
        type=Path,
        help=(
            "Optional directory containing HVP01/.../HVP20 derived datasets. "
            "When set, these replace the legacy pilot-hv*/derived inputs."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--scenario-count", type=int, default=20)
    parser.add_argument("--scenario-prefix", default="HVP")
    parser.add_argument("--evaluation-start", type=int, default=901)
    return parser.parse_args()


def pilot_root(base: Path, index: int) -> Path:
    prefix = "pilot-hv" if index <= 4 else "pilot-hvp"
    return base / f"{prefix}{index:02d}" / "derived"


def build_suite(
    base: Path,
    output: Path,
    benchmark: Path,
    scenario_derived_root: Path | None = None,
    scenario_count: int = 20,
    scenario_prefix: str = "HVP",
    evaluation_start: int = 901,
) -> dict[str, Any]:
    base = base.resolve(strict=True)
    benchmark = benchmark.resolve(strict=True)
    if scenario_derived_root is not None:
        scenario_derived_root = scenario_derived_root.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "summary.jsonl",
        "patch.jsonl",
        "delta.jsonl",
        "turn_quiz.jsonl",
        "final_quiz.jsonl",
        "quiz_sft.jsonl",
    )
    combined: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    scenario_manifests = []
    vehicle_tools: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="hvp-suite-") as temporary:
        temporary_root = Path(temporary)
        for index in range(1, scenario_count + 1):
            scenario = evaluation_start + index - 1
            stage = temporary_root / f"{scenario_prefix}{index:02d}"
            source_root = (
                scenario_derived_root / f"{scenario_prefix}{index:02d}"
                if scenario_derived_root is not None
                else pilot_root(base, index)
            )
            manifest = build_single(
                source_root,
                stage,
                benchmark,
                evaluation_scenario=scenario,
            )
            scenario_manifests.append(manifest)
            for name in names:
                combined[name].extend(load_jsonl(stage / name))
            candidate_tools = json.loads(
                (stage / "vehicle_tools.json").read_text(encoding="utf-8")
            )
            if vehicle_tools is None:
                vehicle_tools = candidate_tools
            elif candidate_tools != vehicle_tools:
                raise ValueError("Vehicle Tool schema differs across pilot adapters")

    assert vehicle_tools is not None
    for name, rows in combined.items():
        write_jsonl(output / name, rows)
    write_json(output / "vehicle_tools.json", vehicle_tools)

    manifest = {
        "schema_version": "vehiclemembench-external-adapted-suite-eval-v1",
        "purpose": (
            f"evaluation-only {scenario_prefix}01-{scenario_prefix}{scenario_count:02d} "
            "closed-loop transfer suite"
        ),
        "source_root": str(scenario_derived_root or base),
        "evaluation_scenarios": list(
            range(evaluation_start, evaluation_start + scenario_count)
        ),
        "split": "test",
        "eligible_for_training": False,
        "eligible_for_human_test": False,
        "counts": {
            "scenarios": scenario_count,
            "turns": len(combined["summary.jsonl"]),
            "updates": sum(
                row["target"]["decision"] == "UPDATE"
                for row in combined["summary.jsonl"]
            ),
            "noops": sum(
                row["target"]["decision"] == "NO_OP"
                for row in combined["summary.jsonl"]
            ),
            "turn_quizzes": len(combined["turn_quiz.jsonl"]),
            "final_quizzes": len(combined["final_quiz.jsonl"]),
            "quiz_sft_rows": len(combined["quiz_sft.jsonl"]),
        },
        "scenario_manifests": scenario_manifests,
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "quiz_manifest.json",
        {
            "schema_version": "vehiclemembench-external-adapted-suite-quiz-v1",
            "evaluation_scenarios": list(
                range(evaluation_start, evaluation_start + scenario_count)
            ),
            "turn_quizzes": len(combined["turn_quiz.jsonl"]),
            "final_quizzes": len(combined["final_quiz.jsonl"]),
        },
    )
    write_json(
        output / "quiz_sft_manifest.json",
        {
            "schema_version": "vehiclemembench-external-adapted-suite-quiz-sft-v1",
            "row_count": len(combined["quiz_sft.jsonl"]),
        },
    )
    build_catalog(output, output / "catalog.sqlite")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_suite(
        args.pilot_base_root,
        args.output_dir,
        args.vehiclemembench_root,
        args.scenario_derived_root,
        args.scenario_count,
        args.scenario_prefix,
        args.evaluation_start,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
