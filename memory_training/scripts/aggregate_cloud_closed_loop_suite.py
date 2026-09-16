"""Aggregate per-scenario Cloud closed-loop summaries with micro metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def weighted(summaries: list[dict[str, Any]], section: str, metric: str) -> float:
    numerator = sum(
        float(summary[section][metric]) * int(summary[section]["tasks"])
        for summary in summaries
    )
    denominator = sum(int(summary[section]["tasks"]) for summary in summaries)
    return safe_ratio(numerator, denominator)


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob("s*/summary.json"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not summaries or not all(summary.get("complete") for summary in summaries):
        raise ValueError("all per-scenario summaries must exist and be complete")

    decisions: Counter[str] = Counter()
    for summary in summaries:
        decisions.update(summary["memory"]["decision_counts"])
    true_update = decisions["UPDATE->UPDATE"]
    false_update = decisions["NO_OP->UPDATE"]
    missed_update = decisions["UPDATE->NO_OP"] + decisions["UPDATE->INVALID"]
    precision = safe_ratio(true_update, true_update + false_update)
    recall = safe_ratio(true_update, true_update + missed_update)
    update_f1 = safe_ratio(2 * precision * recall, precision + recall)
    turns = sum(int(summary["memory"]["turns"]) for summary in summaries)
    state_f1 = safe_ratio(
        sum(float(summary["memory"]["state_f1"]) * int(summary["memory"]["turns"]) for summary in summaries),
        turns,
    )
    final_state_f1 = sum(float(summary["memory"]["final_state_f1"]) for summary in summaries) / len(summaries)
    quiz_esm = weighted(summaries, "closed_loop_quiz", "esm")
    quiz_tool_f1 = weighted(summaries, "closed_loop_quiz", "tool_f1")
    quiz_arg_exact = weighted(summaries, "closed_loop_quiz", "arg_exact")
    quiz_tasks = sum(int(summary["closed_loop_quiz"]["tasks"]) for summary in summaries)
    by_quiz_type = {}
    for quiz_type in ("TURN", "FINAL"):
        entries = [summary["closed_loop_quiz"]["by_quiz_type"][quiz_type] for summary in summaries]
        tasks = sum(int(entry["tasks"]) for entry in entries)
        by_quiz_type[quiz_type] = {
            "tasks": tasks,
            "esm": safe_ratio(sum(float(entry["esm"]) * int(entry["tasks"]) for entry in entries), tasks),
            "tool_f1": safe_ratio(sum(float(entry["tool_f1"]) * int(entry["tasks"]) for entry in entries), tasks),
            "arg_exact": safe_ratio(sum(float(entry["arg_exact"]) * int(entry["tasks"]) for entry in entries), tasks),
        }

    result = {
        "schema_version": "palmclaw-cloud-closed-loop-suite-summary-v1",
        "complete": True,
        "scenario_count": len(summaries),
        "scenarios": [int(summary["scenario"]) for summary in summaries],
        "memory": {
            "turns": turns,
            "decision_counts": dict(decisions),
            "update_precision": precision,
            "update_recall": recall,
            "update_f1": update_f1,
            "state_f1": state_f1,
            "final_state_f1": final_state_f1,
            "invalid_outputs": sum(int(summary["memory"]["invalid_outputs"]) for summary in summaries),
            "cost_usd": sum(float(summary["memory"]["cost_usd"]) for summary in summaries),
        },
        "closed_loop_quiz": {
            "tasks": quiz_tasks,
            "esm": quiz_esm,
            "tool_f1": quiz_tool_f1,
            "arg_exact": quiz_arg_exact,
            "by_quiz_type": by_quiz_type,
            "cost_usd": sum(float(summary["closed_loop_quiz"]["cost_usd"]) for summary in summaries),
        },
        "composite": 0.60 * quiz_esm + 0.25 * final_state_f1 + 0.15 * update_f1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
