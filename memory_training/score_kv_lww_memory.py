"""Score deterministic KV/LWW memory trajectories and compute Composite."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate_quiz_baseline import (
    TEST_SCENARIOS,
    _load_turns,
    _materialize_kv_store,
    _replay_kv_lww,
    _slug,
)
from .validation import memory_scores


MODEL_KEYS = (
    "granite4-350m",
    "qwen3.5-0.8b",
    "granite4-1b",
    "llama3.2-1b",
    "qwen3.5-2b",
    "llama3.2-3b",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--decision-eval-root",
        type=Path,
        help=(
            "Optional fixed-ratio evaluation root. When supplied, UPDATE/NO_OP "
            "decisions are scored only at the turns listed in selection.jsonl, "
            "while the KV state is still replayed over the complete dialogue."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, default=TEST_SCENARIOS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = tuple(dict.fromkeys(args.scenarios))
    if any(scenario not in TEST_SCENARIOS for scenario in scenarios):
        raise ValueError("Only fixed Test scenarios are supported")
    data_root = args.data_root.resolve()
    turns = _load_turns(data_root / "patch.jsonl", scenarios)
    gold = _load_gold(data_root / "summary.jsonl", scenarios)
    selected_turns = (
        _load_selected_turns(
            args.decision_eval_root.resolve() / "selection.jsonl", scenarios
        )
        if args.decision_eval_root
        else {scenario: None for scenario in scenarios}
    )

    counts: defaultdict[str, int] = defaultdict(int)
    reports = []
    structured_reports = []
    for scenario in scenarios:
        rows = turns[scenario]
        decisions = gold[scenario]["decisions"]
        final_store: Mapping[str, Mapping[str, Any]] = {}
        for row, predicted_update, store in _replay_kv_lww(rows):
            turn = int(row["global_turn_index"])
            if selected_turns[scenario] is not None and turn not in selected_turns[scenario]:
                final_store = store
                continue
            gold_update = decisions[turn] == "UPDATE"
            if gold_update and predicted_update:
                counts["true_update"] += 1
            elif gold_update:
                counts["missed_update"] += 1
            elif predicted_update:
                counts["false_update"] += 1
            else:
                counts["true_noop"] += 1
            final_store = store
        score = memory_scores(
            _materialize_kv_store(final_store), str(gold[scenario]["final_memory"])
        )
        reports.append({"scenario_index": scenario, **score})
        structured = _set_scores(
            _predicted_structured_facts(final_store),
            _gold_structured_facts(str(gold[scenario]["final_memory"])),
        )
        structured_reports.append({"scenario_index": scenario, **structured})

    update_precision = _ratio(
        counts["true_update"], counts["true_update"] + counts["false_update"]
    )
    update_recall = _ratio(
        counts["true_update"], counts["true_update"] + counts["missed_update"]
    )
    update_f1 = (
        2 * update_precision * update_recall / (update_precision + update_recall)
        if update_precision + update_recall
        else 0.0
    )
    final_state_f1 = sum(float(row["f1"]) for row in reports) / len(reports)
    final_state_exact = sum(float(row["exact"]) for row in reports) / len(reports)
    structured_final_state_f1 = sum(
        float(row["f1"]) for row in structured_reports
    ) / len(structured_reports)
    structured_final_state_exact = sum(
        float(row["exact"]) for row in structured_reports
    ) / len(structured_reports)
    memory = {
        "scoring_contract": "canonical non-empty line-set exact match (same as memory_training.validation.memory_scores)",
        "turns": sum(
            len(selected_turns[scenario])
            if selected_turns[scenario] is not None
            else len(turns[scenario])
            for scenario in scenarios
        ),
        "replayed_turns": sum(len(turns[scenario]) for scenario in scenarios),
        "decision_eval_root": (
            str(args.decision_eval_root.resolve()) if args.decision_eval_root else None
        ),
        "decision_counts": dict(counts),
        "update_precision": update_precision,
        "update_recall": update_recall,
        "update_f1": update_f1,
        "final_state_f1": final_state_f1,
        "final_state_exact": final_state_exact,
        "structured_scoring_contract": "exact set match over (owner, tool.attribute, value), ignoring timestamp/condition prose",
        "structured_final_state_f1": structured_final_state_f1,
        "structured_final_state_exact": structured_final_state_exact,
        "scenario_reports": reports,
        "structured_scenario_reports": structured_reports,
    }

    by_model = {}
    evaluation_root = args.evaluation_root.resolve()
    for model in MODEL_KEYS:
        summary_path = evaluation_root / model / "kv_lww" / "summary.json"
        quiz = json.loads(summary_path.read_text(encoding="utf-8"))
        esm = float(quiz["esm"])
        by_model[model] = {
            "quiz_esm": esm,
            "final_state_f1": final_state_f1,
            "update_f1": update_f1,
            "composite": 0.60 * esm + 0.25 * final_state_f1 + 0.15 * update_f1,
            "structured_composite_diagnostic": (
                0.60 * esm
                + 0.25 * structured_final_state_f1
                + 0.15 * update_f1
            ),
        }

    result = {
        "schema_version": "quiz-kv-lww-memory-composite-v1",
        "method": "kv_lww",
        "scenarios": list(scenarios),
        "memory": memory,
        "by_model": by_model,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.output.resolve(), result)
    return result


def _load_gold(path: Path, scenarios: Sequence[int]) -> dict[int, dict[str, Any]]:
    selected = set(scenarios)
    result: dict[int, dict[str, Any]] = {
        scenario: {"decisions": {}, "final_turn": -1, "final_memory": ""}
        for scenario in scenarios
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scenario = int(row["scenario_index"])
            if scenario not in selected:
                continue
            turn = int(row["global_turn_index"])
            target = row["target"]
            result[scenario]["decisions"][turn] = str(target["decision"])
            if turn > int(result[scenario]["final_turn"]):
                result[scenario]["final_turn"] = turn
                result[scenario]["final_memory"] = str(target["next_memory"])
    for scenario, payload in result.items():
        decisions = payload["decisions"]
        expected = list(range(int(payload["final_turn"]) + 1))
        if sorted(decisions) != expected:
            raise ValueError(f"S{scenario} Gold turn sequence is incomplete")
    return result


def _load_selected_turns(
    path: Path, scenarios: Sequence[int]
) -> dict[int, set[int]]:
    selected_scenarios = set(scenarios)
    result = {scenario: set() for scenario in scenarios}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scenario = int(row["scenario_index"])
            if scenario in selected_scenarios:
                result[scenario].add(int(row["global_turn_index"]))
    missing = [scenario for scenario, turns in result.items() if not turns]
    if missing:
        raise ValueError(f"No selected turns found for scenarios: {missing}")
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _predicted_structured_facts(
    store: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    return {
        (
            _slug(str(record["speaker"])),
            str(record["field"]),
            _canonical_value(record["value"]),
        )
        for record in store.values()
    }


def _gold_structured_facts(memory: str) -> set[tuple[str, str, str]]:
    owner = ""
    facts: set[tuple[str, str, str]] = set()
    pattern = re.compile(r"^- \[[^]]+\] ([^;]+); value=([^;]+)")
    for raw in memory.splitlines():
        line = raw.strip()
        if line.startswith("### "):
            owner = _slug(line[4:].strip())
            continue
        match = pattern.match(line)
        if not owner or match is None:
            continue
        facts.add((owner, match.group(1).strip(), _canonical_raw_value(match.group(2))))
    return facts


def _canonical_raw_value(raw: str) -> str:
    value = raw.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return _canonical_value(parsed)


def _canonical_value(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                number = float(value)
            except ValueError:
                pass
            else:
                value = int(number) if number.is_integer() else number
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _set_scores(
    predicted: set[tuple[str, str, str]], gold: set[tuple[str, str, str]]
) -> dict[str, float | int]:
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else float(not gold)
    recall = overlap / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_facts": len(predicted),
        "gold_facts": len(gold),
        "matched_facts": overlap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(predicted == gold),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({"memory": result["memory"], "by_model": result["by_model"]}))


if __name__ == "__main__":
    main()
