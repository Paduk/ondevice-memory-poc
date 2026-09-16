"""Aggregate paired Delta-v3 pending-depth ablation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT

CONDITIONS = ("unstratified", "depth-weighted")
DEFAULT_UNIFORM_REFERENCE_RUN = (
    "granite4-1b-delta_v3_compact_k5-multitask-noop5-"
    "uniform-depth-e4-b2-trainseed45-r1"
)
METRICS = (
    "test_composite",
    "closed_loop_quiz_esm",
    "closed_loop_final_state_f1",
    "closed_loop_update_f1",
    "closed_loop_false_update_rate",
    "diagnostic_update_recall",
    "diagnostic_false_update_rate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(45, 46, 47))
    parser.add_argument("--run-tag", default="r1")
    parser.add_argument(
        "--uniform-reference-run-id",
        default=DEFAULT_UNIFORM_REFERENCE_RUN,
        help="Existing uniform-depth run reported as an unpaired reference.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    reports: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        reports[condition] = [
            _load_run(workspace, condition, seed, args.run_tag)
            for seed in args.seeds
        ]
    result = aggregate_reports(reports)
    result["uniform_reference"] = _load_uniform_reference(
        workspace, args.uniform_reference_run_id
    )
    result.update(
        {
            "schema_version": "palmclaw-delta-pending-ablation-summary-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seeds": list(args.seeds),
            "run_tag": args.run_tag,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_render_markdown(result), encoding="utf-8")
    return result


def aggregate_reports(
    reports: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    for condition in CONDITIONS:
        if condition not in reports:
            raise ValueError(f"Missing condition: {condition}")
    seeds_by_condition = {
        condition: [int(report["seed"]) for report in reports[condition]]
        for condition in CONDITIONS
    }
    if seeds_by_condition[CONDITIONS[0]] != seeds_by_condition[CONDITIONS[1]]:
        raise ValueError("Conditions must contain the same seeds in the same order")
    summaries = {
        condition: {
            metric: _summary([float(report[metric]) for report in condition_reports])
            for metric in METRICS
        }
        for condition, condition_reports in reports.items()
    }
    paired = []
    for unstratified, weighted in zip(
        reports["unstratified"], reports["depth-weighted"], strict=True
    ):
        paired.append(
            {
                "seed": int(unstratified["seed"]),
                "depth_weighted_minus_unstratified": {
                    metric: float(weighted[metric]) - float(unstratified[metric])
                    for metric in METRICS
                },
            }
        )
    paired_summary = {
        metric: _summary(
            [
                row["depth_weighted_minus_unstratified"][metric]
                for row in paired
            ]
        )
        for metric in METRICS
    }
    return {
        "conditions": list(CONDITIONS),
        "runs": reports,
        "condition_summaries": summaries,
        "paired_differences": paired,
        "paired_difference_summary": paired_summary,
    }


def _load_run(
    workspace: Path,
    condition: str,
    seed: int,
    run_tag: str,
) -> dict[str, Any]:
    run_id = (
        "granite4-1b-delta_v3_compact_k5-pending-ablation-"
        f"{condition}-e4-b2-seed{seed}-{run_tag}"
    )
    run_dir = workspace / "runs" / run_id
    selection = _read_json(run_dir / "eval-fixed-best-checkpoint.json")
    winner = selection["winner"]
    epoch = int(winner["epoch"])
    epoch_label = f"epoch-{epoch:02d}"
    test_path = run_dir / f"eval-fixed-test-best-{epoch_label}" / "summary.json"
    diagnostic_path = run_dir / f"pending-depth-test-{epoch_label}.json"
    test = _read_json(test_path)
    diagnostic = _read_json(diagnostic_path)
    quiz_esm = float(test["closed_loop_quiz"]["esm"])
    final_state_f1 = float(test["memory"]["final_state_f1"])
    update_f1 = float(test["memory"]["update_f1"])
    return {
        "condition": condition,
        "seed": seed,
        "run_id": run_id,
        "selected_epoch": epoch,
        "validation_composite": float(winner["composite_score"]),
        "test_composite": 0.60 * quiz_esm + 0.25 * final_state_f1 + 0.15 * update_f1,
        "closed_loop_quiz_esm": quiz_esm,
        "closed_loop_final_state_f1": final_state_f1,
        "closed_loop_update_f1": update_f1,
        "closed_loop_false_update_rate": float(test["memory"]["false_update_rate"]),
        "diagnostic_update_recall": float(diagnostic["aggregate"]["update_recall"]),
        "diagnostic_false_update_rate": float(
            diagnostic["aggregate"]["false_update_rate"]
        ),
        "by_pending_depth": diagnostic["by_pending_depth"],
        "artifacts": {
            "selection": str(run_dir / "eval-fixed-best-checkpoint.json"),
            "test": str(test_path),
            "diagnostic": str(diagnostic_path),
        },
    }


def _load_uniform_reference(workspace: Path, run_id: str) -> dict[str, Any]:
    run_dir = workspace / "runs" / run_id
    selection = _read_json(run_dir / "eval-fixed-best-checkpoint.json")
    epoch = int(selection["winner"]["epoch"])
    summary_path = (
        run_dir / f"eval-fixed-test-best-epoch-{epoch:02d}" / "summary.json"
    )
    summary = _read_json(summary_path)
    quiz_esm = float(summary["closed_loop_quiz"]["esm"])
    final_state_f1 = float(summary["memory"]["final_state_f1"])
    update_f1 = float(summary["memory"]["update_f1"])
    return {
        "run_id": run_id,
        "seed_count": 1,
        "paired": False,
        "selected_epoch": epoch,
        "test_composite": 0.60 * quiz_esm + 0.25 * final_state_f1 + 0.15 * update_f1,
        "closed_loop_quiz_esm": quiz_esm,
        "closed_loop_final_state_f1": final_state_f1,
        "closed_loop_update_f1": update_f1,
        "closed_loop_false_update_rate": float(
            summary["memory"]["false_update_rate"]
        ),
        "artifact": str(summary_path),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty metric")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required ablation artifact is missing: {path}") from exc
    if not isinstance(result, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Delta-v3 pending-depth sampling ablation",
        "",
        "All values are mean ± sample standard deviation across paired seeds.",
        "",
        "| Metric | Unstratified random | Weighted 40/30/15/10/5 | Paired Δ |",
        "|---|---:|---:|---:|",
    ]
    summaries = result["condition_summaries"]
    paired = result["paired_difference_summary"]
    for metric in METRICS:
        random = summaries["unstratified"][metric]
        weighted = summaries["depth-weighted"][metric]
        difference = paired[metric]
        lines.append(
            f"| {metric} | {random['mean']:.4f} ± {random['sample_std']:.4f} | "
            f"{weighted['mean']:.4f} ± {weighted['sample_std']:.4f} | "
            f"{difference['mean']:+.4f} ± {difference['sample_std']:.4f} |"
        )
    reference = result["uniform_reference"]
    lines.extend(
        (
            "",
            "Paired Δ is weighted minus unstratified for the same seed.",
            "",
            "## Existing uniform-depth reference",
            "",
            (
                "The current `1/1/1/1/1` result is a single unpaired reference, "
                "not a third condition in this ablation."
            ),
            "",
            "| Test composite | Quiz ESM | Final-memory F1 | UPDATE F1 | False-update rate |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {reference['test_composite']:.4f} | "
                f"{reference['closed_loop_quiz_esm']:.4f} | "
                f"{reference['closed_loop_final_state_f1']:.4f} | "
                f"{reference['closed_loop_update_f1']:.4f} | "
                f"{reference['closed_loop_false_update_rate']:.4f} |"
            ),
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result["condition_summaries"], indent=2))


if __name__ == "__main__":
    main()
