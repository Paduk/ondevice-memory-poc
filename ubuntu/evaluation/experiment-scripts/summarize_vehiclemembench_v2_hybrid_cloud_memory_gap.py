#!/usr/bin/env python3
"""Summarize Hybrid Gold vs generated Turn-wise cloud memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT / "vehiclemembench-v2-hybrid-cloud-memory-gap"
)
DEFAULT_GOLD_AGENT_SUMMARY = (
    DEFAULT_EVALUATION_ROOT
    / "vehiclemembench-v2-three-way-evaluation"
    / "agent"
    / "agent-summary.json"
)
ARMS = ("turnwise_summary", "turnwise_combined")
KINDS = ("immediate", "delayed", "composite", "final")
LUNA_INPUT_PER_MILLION = 0.20
LUNA_CACHED_INPUT_PER_MILLION = 0.02
LUNA_OUTPUT_PER_MILLION = 1.20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gold-agent-summary",
        type=Path,
        default=DEFAULT_GOLD_AGENT_SUMMARY,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.expanduser().resolve()
    generated = _read_json(root / "quiz-agent" / "agent-summary.json")
    gold = _read_json(args.gold_agent_summary.expanduser().resolve())
    rows: dict[str, Any] = {
        "hybrid_gold": {
            "memory": None,
            "quiz": _quiz_metrics(gold["metrics"], "hybrid"),
        }
    }
    for arm in ARMS:
        memory_records = [
            _read_json(root / arm / f"s{scenario:02d}" / "memory-replay.json")
            for scenario in range(1, 6)
        ]
        if any(record.get("status") != "COMPLETED" for record in memory_records):
            raise ValueError(f"Incomplete memory replay: {arm}")
        memory = {
            "scenario_count": len(memory_records),
            "dialogue_turn_count": sum(
                int(record["dialogue_turn_count"]) for record in memory_records
            ),
            "call_count": sum(int(record["batch_count"]) for record in memory_records),
            "update_count": sum(
                int(record["update_count"]) for record in memory_records
            ),
            "noop_count": sum(int(record["noop_count"]) for record in memory_records),
            "input_tokens": sum(
                int(record["input_tokens"]) for record in memory_records
            ),
            "output_tokens": sum(
                int(record["output_tokens"]) for record in memory_records
            ),
            "cached_tokens": sum(
                int(record["cached_tokens"]) for record in memory_records
            ),
            "latency_ms": sum(int(record["latency_ms"]) for record in memory_records),
            "patch_operation_count": sum(
                int(record["patch_operation_count"]) for record in memory_records
            ),
            "compaction_count": sum(
                int(record["compaction_count"]) for record in memory_records
            ),
        }
        memory["estimated_cost_usd"] = _cost(memory)
        quiz = _quiz_metrics(generated["metrics"], arm)
        rows[arm] = {"memory": memory, "quiz": quiz}

    gold_esm = rows["hybrid_gold"]["quiz"]["all"]["exact_state_match"]
    for arm in ARMS:
        rows[arm]["gold_gap"] = {
            kind: (
                rows[arm]["quiz"][kind]["exact_state_match"]
                - rows["hybrid_gold"]["quiz"][kind]["exact_state_match"]
            )
            for kind in ("all", "turn", "final")
        }
        assert rows[arm]["gold_gap"]["all"] == (
            rows[arm]["quiz"]["all"]["exact_state_match"] - gold_esm
        )
    result = {
        "version": "vehiclemembench-v2-hybrid-cloud-memory-gap-summary-v1",
        "status": "COMPLETED",
        "model": "gpt-5.6-luna",
        "scenario_count": 5,
        "turn_quizzes_per_scenario": 30,
        "final_quizzes_per_scenario": 10,
        "quiz_repeats": 2,
        "methods": rows,
    }
    _write_json(root / "summary.json", result)
    (root / "summary.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _quiz_metrics(metrics: dict[str, Any], method: str) -> dict[str, Any]:
    values = {kind: dict(metrics[f"{method}/{kind}"]) for kind in ("all", *KINDS)}
    values["turn"] = _weighted_metrics(
        [values[kind] for kind in ("immediate", "delayed", "composite")]
    )
    for item in values.values():
        item["estimated_cost_usd"] = _cost(item)
    return values


def _weighted_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sum(int(item["tasks"]) for item in items)
    weighted = {}
    for key in (
        "completion_rate",
        "exact_state_match",
        "tool_f1",
        "argument_exact_match",
    ):
        weighted[key] = (
            sum(float(item[key]) * int(item["tasks"]) for item in items) / tasks
        )
    return {
        "tasks": tasks,
        **weighted,
        "input_tokens": sum(int(item["input_tokens"]) for item in items),
        "output_tokens": sum(int(item["output_tokens"]) for item in items),
        "latency_ms": sum(int(item["latency_ms"]) for item in items),
        "cached_tokens": sum(int(item.get("cached_tokens", 0)) for item in items),
    }


def _cost(usage: dict[str, Any]) -> float:
    input_tokens = int(usage.get("input_tokens", 0))
    cached_tokens = int(usage.get("cached_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    return (
        (input_tokens - cached_tokens) * LUNA_INPUT_PER_MILLION
        + cached_tokens * LUNA_CACHED_INPUT_PER_MILLION
        + output_tokens * LUNA_OUTPUT_PER_MILLION
    ) / 1_000_000


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Hybrid Gold vs Turn-wise Cloud Memory",
        "",
        "| Method | Turn ESM | Final ESM | Overall ESM | Gold gap | "
        "Tool F1 | Arg Exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "hybrid_gold": "Hybrid Gold Memory",
        "turnwise_summary": "Turn-wise Summary",
        "turnwise_combined": "Turn-wise Combined",
    }
    for method, label in labels.items():
        record = result["methods"][method]
        quiz = record["quiz"]
        gap = record.get("gold_gap", {}).get("all", 0.0)
        lines.append(
            f"| {label} | {quiz['turn']['exact_state_match']:.3f} | "
            f"{quiz['final']['exact_state_match']:.3f} | "
            f"{quiz['all']['exact_state_match']:.3f} | {gap:+.3f} | "
            f"{quiz['all']['tool_f1']:.3f} | "
            f"{quiz['all']['argument_exact_match']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Method | Memory calls | Memory in/out | Memory latency | "
            "Memory cost | Quiz in/out | Quiz latency | Quiz cost |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ARMS:
        record = result["methods"][method]
        memory = record["memory"]
        quiz = record["quiz"]["all"]
        lines.append(
            f"| {labels[method]} | {memory['call_count']:,} | "
            f"{memory['input_tokens']:,}/{memory['output_tokens']:,} | "
            f"{memory['latency_ms'] / 60000:.1f}m | "
            f"${memory['estimated_cost_usd']:.2f} | "
            f"{quiz['input_tokens']:,}/{quiz['output_tokens']:,} | "
            f"{quiz['latency_ms'] / 60000:.1f}m | "
            f"${quiz['estimated_cost_usd']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
