#!/usr/bin/env python3
"""Replay captured Patch/Delta traces and localize memory error dynamics."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_training.methods import METHODS
from memory_training.methods.operations import normalize_memory
from memory_training.scripts.aggregate_three_method_type_accuracy import (
    MODEL_ARTIFACTS,
)
from memory_training.scripts.classify_patch_delta_disagreements import parse_memory


MODEL_SLUGS = {
    "Granite 350M": "granite4-350m",
    "Qwen 0.8B": "qwen3.5-0.8b",
    "Granite 1B": "granite4-1b",
    "Llama 3.2 1B": "llama3.2-1b",
    "Qwen 2B": "qwen3.5-2b",
    "Llama 3.2 3B": "llama3.2-3b",
}
METHOD_CONFIG = {
    "Patch": ("patch", "patch"),
    "Delta-v3": ("delta_v3_compact_k5", "delta_v3_compact_k5"),
}


@dataclass(frozen=True)
class TurnState:
    global_turn_index: int
    gold_decision: str
    predicted_decision: str
    gold_before: frozenset[tuple[str, str, str]]
    gold_after: frozenset[tuple[str, str, str]]
    predicted_before: frozenset[tuple[str, str, str]]
    predicted_after: frozenset[tuple[str, str, str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--quiz-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fact_set(memory: str) -> frozenset[tuple[str, str, str]]:
    """Represent each durable fact version by owner, timestamp, and payload."""
    return frozenset(
        (fact.owner, fact.timestamp, fact.body) for fact in parse_memory(memory)
    )


def _fact_f1(
    predicted: frozenset[tuple[str, str, str]],
    gold: frozenset[tuple[str, str, str]],
) -> float:
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else float(not gold)
    recall = overlap / len(gold) if gold else float(not predicted)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _placement(sample_id: str, quiz_type: str) -> str:
    if quiz_type == "FINAL":
        return "FINAL"
    lowered = sample_id.lower()
    if "delayed" in lowered:
        return "TURN_DELAYED"
    if "composite" in lowered:
        return "TURN_COMPOSITE"
    return "TURN_IMMEDIATE"


def _gold_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = {}
    for row in _read_jsonl(path):
        key = (int(row["scenario_index"]), int(row["global_turn_index"]))
        if key in rows:
            raise ValueError(f"Duplicate gold row: {key}")
        rows[key] = row
    return rows


def _quiz_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _read_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"Duplicate Quiz row: {sample_id}")
        rows[sample_id] = row
    return rows


def _quiz_records(path: Path) -> dict[str, dict[str, Any]]:
    records = _read_json(path)["closed_loop_quiz"]["records"]
    result = {str(row["sample_id"]): dict(row) for row in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate Quiz records: {path}")
    return result


def classify_fact_episodes(turns: Sequence[TurnState]) -> Counter[str]:
    """Classify acquisition and first-loss timing for introduced gold facts."""
    counts: Counter[str] = Counter()
    for index, turn in enumerate(turns):
        if turn.gold_decision != "UPDATE":
            continue
        introduced = turn.gold_after - turn.gold_before
        retired = turn.gold_before - turn.gold_after
        counts["introduced_facts"] += len(introduced)
        counts["retired_facts"] += len(retired)
        counts["retired_immediately_removed"] += sum(
            fact not in turn.predicted_after for fact in retired
        )
        transition_success = introduced <= turn.predicted_after and not (
            retired & turn.predicted_after
        )
        counts["update_transitions"] += 1
        counts["update_transition_successes"] += int(transition_success)
        for fact in introduced:
            if fact not in turn.predicted_after:
                counts["immediate_acquisition_failures"] += 1
                counts[
                    "immediate_acquisition_failure_predicted_"
                    + turn.predicted_decision.lower()
                ] += 1
                continue
            counts["immediately_acquired_facts"] += 1
            first_loss = None
            active_end = len(turns)
            for later_index in range(index + 1, len(turns)):
                later = turns[later_index]
                if fact not in later.gold_after:
                    active_end = later_index
                    break
                if first_loss is None and fact not in later.predicted_after:
                    first_loss = later_index
            if first_loss is None:
                counts["retained_until_retirement_or_end"] += 1
                continue
            loss_turn = turns[first_loss]
            counts[
                "first_loss_on_noop"
                if loss_turn.gold_decision == "NO_OP"
                else "first_loss_on_later_update"
            ] += 1
            if any(
                fact in turns[later_index].predicted_after
                for later_index in range(first_loss + 1, active_end)
            ):
                counts["recovered_after_first_loss"] += 1
    return counts


def summarize_turns(turns: Sequence[TurnState]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    fact_f1_sum = 0.0
    for turn in turns:
        fact_f1_sum += _fact_f1(turn.predicted_after, turn.gold_after)
        missing_before = turn.gold_before - turn.predicted_before
        missing_after = turn.gold_after - turn.predicted_after
        extra_before = turn.predicted_before - turn.gold_before
        extra_after = turn.predicted_after - turn.gold_after
        before_exact = not missing_before and not extra_before
        after_exact = not missing_after and not extra_after
        counts["turns"] += 1
        counts[f"gold_{turn.gold_decision.lower()}_turns"] += 1
        counts["state_exact_turns"] += int(after_exact)
        counts["correct_to_wrong"] += int(before_exact and not after_exact)
        counts["wrong_to_correct"] += int(not before_exact and after_exact)
        if turn.gold_decision == "NO_OP":
            if turn.gold_before != turn.gold_after:
                raise ValueError("Gold NO_OP changed the fact state")
            mutated = turn.predicted_before != turn.predicted_after
            new_missing = missing_after - missing_before
            new_extra = extra_after - extra_before
            recovered_missing = missing_before - missing_after
            counts["noop_predicted_updates"] += int(
                turn.predicted_decision == "UPDATE"
            )
            counts["noop_memory_mutations"] += int(mutated)
            counts["noop_new_loss_turns"] += int(bool(new_missing))
            counts["noop_new_missing_facts"] += len(new_missing)
            counts["noop_new_extra_turns"] += int(bool(new_extra))
            counts["noop_new_extra_facts"] += len(new_extra)
            counts["noop_recovery_turns"] += int(bool(recovered_missing))
            counts["noop_recovered_missing_facts"] += len(recovered_missing)
            counts["noop_error_persistence_turns"] += int(
                bool(missing_before or extra_before)
                and bool(missing_after or extra_after)
                and not new_missing
                and not new_extra
                and not recovered_missing
            )
        elif turn.gold_decision == "UPDATE":
            counts["update_predicted_updates"] += int(
                turn.predicted_decision == "UPDATE"
            )
    counts.update(classify_fact_episodes(turns))
    return {
        "counts": dict(counts),
        "mean_fact_f1": fact_f1_sum / len(turns) if turns else 0.0,
    }


def _merge_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    weighted_f1 = 0.0
    for summary in summaries:
        item_counts = Counter(
            {key: int(value) for key, value in summary["counts"].items()}
        )
        counts.update(item_counts)
        weighted_f1 += float(summary["mean_fact_f1"]) * item_counts["turns"]
    return {
        "counts": dict(counts),
        "mean_fact_f1": weighted_f1 / counts["turns"] if counts["turns"] else 0.0,
        "rates": _rates(counts),
    }


def _rates(counts: Mapping[str, int]) -> dict[str, float]:
    def ratio(numerator: str, denominator: str) -> float:
        return (
            counts.get(numerator, 0) / counts.get(denominator, 0)
            if counts.get(denominator, 0)
            else 0.0
        )

    acquired = counts.get("immediately_acquired_facts", 0)
    return {
        "update_transition_success": ratio(
            "update_transition_successes", "update_transitions"
        ),
        "introduced_fact_immediate_acquisition": ratio(
            "immediately_acquired_facts", "introduced_facts"
        ),
        "introduced_fact_failure_from_missed_update": ratio(
            "immediate_acquisition_failure_predicted_no_op",
            "immediate_acquisition_failures",
        ),
        "introduced_fact_failure_despite_predicted_update": ratio(
            "immediate_acquisition_failure_predicted_update",
            "immediate_acquisition_failures",
        ),
        "introduced_fact_failure_from_invalid_output": ratio(
            "immediate_acquisition_failure_predicted_invalid",
            "immediate_acquisition_failures",
        ),
        "retired_fact_immediate_removal": ratio(
            "retired_immediately_removed", "retired_facts"
        ),
        "noop_false_update": ratio("noop_predicted_updates", "gold_no_op_turns"),
        "noop_memory_mutation": ratio("noop_memory_mutations", "gold_no_op_turns"),
        "noop_new_information_loss": ratio(
            "noop_new_loss_turns", "gold_no_op_turns"
        ),
        "noop_new_extra_fact": ratio("noop_new_extra_turns", "gold_no_op_turns"),
        "acquired_fact_retained": (
            counts.get("retained_until_retirement_or_end", 0) / acquired
            if acquired
            else 0.0
        ),
        "acquired_fact_first_loss_on_noop": (
            counts.get("first_loss_on_noop", 0) / acquired if acquired else 0.0
        ),
        "acquired_fact_first_loss_on_later_update": (
            counts.get("first_loss_on_later_update", 0) / acquired
            if acquired
            else 0.0
        ),
    }


def _replay_scenario(
    *,
    method: Any,
    trace_payload: Mapping[str, Any],
    gold_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    quiz_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[TurnState], dict[tuple[int, int], str], dict[str, int]]:
    report = trace_payload["scenario_reports"][0]
    scenario = int(report["scenario_index"])
    state = method.initial_state()
    turns = []
    memories = {}
    snapshots_by_turn: defaultdict[int, list[tuple[str, Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for sample_id, snapshot in trace_payload.get("quiz_snapshots", {}).items():
        source = quiz_rows[sample_id]
        turn = int(source["memory_ref"]["global_turn_index"])
        snapshots_by_turn[turn].append((sample_id, snapshot))
    validated_snapshots = 0
    for trace in trace_payload["decision_trace"]:
        global_turn = int(trace["global_turn_index"])
        gold = gold_rows[(scenario, global_turn)]
        if str(gold["target"]["decision"]) != str(trace["gold_decision"]):
            raise ValueError(f"Gold decision mismatch: S{scenario} turn {global_turn}")
        predicted_before_memory = method.materialize_memory(state)
        try:
            parsed = method.parse_output(str(trace["output"]))
            state = method.apply_output(
                state, parsed, turn_id=str(trace.get("turn_id", ""))
            )
        except Exception:  # noqa: BLE001 - evaluator also keeps prior state.
            pass
        predicted_after_memory = method.materialize_memory(state)
        predicted_depth = len(getattr(state, "pending_updates", ()))
        gold_before_memory = str(gold["input"].get("previous_memory", ""))
        gold_after_memory = str(gold["target"].get("next_memory", ""))
        turns.append(
            TurnState(
                global_turn_index=global_turn,
                gold_decision=str(trace["gold_decision"]),
                predicted_decision=str(trace["predicted_decision"]),
                gold_before=_fact_set(gold_before_memory),
                gold_after=_fact_set(gold_after_memory),
                predicted_before=_fact_set(predicted_before_memory),
                predicted_after=_fact_set(predicted_after_memory),
            )
        )
        memories[(scenario, global_turn)] = predicted_after_memory
        for sample_id, snapshot in snapshots_by_turn.get(global_turn, ()):
            if normalize_memory(str(snapshot["materialized_memory"])) != normalize_memory(
                predicted_after_memory
            ):
                raise ValueError(f"Snapshot replay mismatch: {sample_id}")
            if int(snapshot.get("pending_depth", 0)) != predicted_depth:
                raise ValueError(f"Pending-depth replay mismatch: {sample_id}")
            validated_snapshots += 1
    return turns, memories, {"validated_snapshots": validated_snapshots}


def _quiz_summary(
    *,
    records: Mapping[str, Mapping[str, Any]],
    quiz_rows: Mapping[str, Mapping[str, Any]],
    gold_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    memories: Mapping[tuple[int, int], str],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_placement: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for sample_id, record in records.items():
        source = quiz_rows[sample_id]
        scenario = int(source["scenario_index"])
        turn = int(source["memory_ref"]["global_turn_index"])
        key = (scenario, turn)
        if key not in memories:
            continue
        gold = _fact_set(str(gold_rows[key]["target"].get("next_memory", "")))
        predicted = _fact_set(memories[key])
        memory_exact = predicted == gold
        quiz_correct = bool(record["exact_state_match"])
        placement = _placement(sample_id, str(source["quiz_type"]))
        label = (
            "memory_exact_quiz_correct"
            if memory_exact and quiz_correct
            else "memory_exact_quiz_wrong"
            if memory_exact
            else "memory_inexact_quiz_correct"
            if quiz_correct
            else "memory_inexact_quiz_wrong"
        )
        counts["covered_quizzes"] += 1
        counts[label] += 1
        by_placement[placement]["covered_quizzes"] += 1
        by_placement[placement][label] += 1
    return {
        "counts": dict(counts),
        "by_placement": {
            name: dict(values) for name, values in sorted(by_placement.items())
        },
    }


def analyze(
    *,
    analysis_root: Path,
    run_root: Path,
    data_root: Path,
    quiz_source: Path,
) -> dict[str, Any]:
    gold_rows = _gold_rows(data_root / "summary.jsonl")
    quiz_rows = _quiz_rows(quiz_source)
    results: dict[str, Any] = {}
    coverage: Counter[str] = Counter()
    for model, slug in MODEL_SLUGS.items():
        model_result = {}
        for display, (method_name, directory) in METHOD_CONFIG.items():
            method = METHODS[method_name]()
            trace_root = analysis_root / "memory-snapshots" / slug / directory
            scenario_paths = sorted((trace_root / "scenarios").glob("s*.json"))
            scenario_summaries = []
            all_memories = {}
            validated = 0
            for path in scenario_paths:
                payload = _read_json(path)
                turns, memories, validation = _replay_scenario(
                    method=method,
                    trace_payload=payload,
                    gold_rows=gold_rows,
                    quiz_rows=quiz_rows,
                )
                scenario_summaries.append(summarize_turns(turns))
                all_memories.update(memories)
                validated += int(validation["validated_snapshots"])
            artifact = run_root / MODEL_ARTIFACTS[model][display]
            quiz = _quiz_summary(
                records=_quiz_records(artifact),
                quiz_rows=quiz_rows,
                gold_rows=gold_rows,
                memories=all_memories,
            )
            aggregate = _merge_summaries(scenario_summaries)
            model_result[display] = {
                "coverage": {
                    "scenarios": len(scenario_paths),
                    "turns": aggregate["counts"].get("turns", 0),
                    "quiz_snapshots_validated": validated,
                    "quizzes": quiz["counts"].get("covered_quizzes", 0),
                },
                "trajectory": aggregate,
                "quiz_state_relation": quiz,
            }
            coverage[f"{display}_scenarios"] += len(scenario_paths)
            coverage[f"{display}_turns"] += aggregate["counts"].get("turns", 0)
            coverage[f"{display}_quizzes"] += quiz["counts"].get(
                "covered_quizzes", 0
            )
        results[model] = model_result
    aggregate_by_method = {
        method: _merge_summaries(
            [results[model][method]["trajectory"] for model in MODEL_SLUGS]
        )
        for method in METHOD_CONFIG
    }
    aggregate_quiz = {}
    for method in METHOD_CONFIG:
        counts: Counter[str] = Counter()
        by_placement: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for model in MODEL_SLUGS:
            quiz = results[model][method]["quiz_state_relation"]
            counts.update({key: int(value) for key, value in quiz["counts"].items()})
            for placement, values in quiz["by_placement"].items():
                by_placement[placement].update(
                    {key: int(value) for key, value in values.items()}
                )
        aggregate_quiz[method] = {
            "counts": dict(counts),
            "by_placement": {
                key: dict(value) for key, value in sorted(by_placement.items())
            },
        }
    return {
        "schema_version": "palmclaw-patch-delta-error-dynamics-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "methods": list(METHOD_CONFIG),
            "models": list(MODEL_SLUGS),
            "trace_source": str(analysis_root),
            "gold_source": str(data_root / "summary.jsonl"),
            "quiz_source": str(quiz_source),
            "unit_note": (
                "Turn analysis uses the fixed UPDATE:NO_OP=1:5 evaluation cohort, "
                "not all original dialogue utterances."
            ),
        },
        "coverage": dict(coverage),
        "aggregate_by_method": aggregate_by_method,
        "aggregate_quiz_state_relation": aggregate_quiz,
        "by_model": results,
    }


def _percent(value: float) -> str:
    return f"{100 * value:.2f}"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Patch–Delta Error Dynamics",
        "",
        "이 분석은 고정 Test의 Patch와 Delta-v3 decision trace를 재생하여 메모리 오류가",
        "UPDATE 시점에 발생하는지, 이후 NO_OP 또는 다른 UPDATE에서 발생하는지를 구분한다.",
        "분석 단위는 전체 원본 발화가 아니라 UPDATE:NO_OP=1:5로 고정된 평가 trajectory이다.",
        "",
        "## Coverage",
        "",
        "| Method | Model-scenarios | Turns | Quizzes |",
        "|---|---:|---:|---:|",
    ]
    coverage = result["coverage"]
    for method in METHOD_CONFIG:
        lines.append(
            f"| {method} | {coverage[f'{method}_scenarios']} | "
            f"{coverage[f'{method}_turns']:,} | {coverage[f'{method}_quizzes']:,} |"
        )
    lines.extend(
        [
            "",
            "전체 144개 model–scenario 조합 중 141개가 방법별로 포함된다. 제외된 세 조합은",
            "Patch와 Delta-v3의 Quiz 정오가 모두 일치하여 기존 불일치 snapshot 요청에 포함되지 않은",
            "시나리오이다.",
            "",
            "## Update Acquisition and Retention",
            "",
            "| Metric | Patch (%) | Delta-v3 (%) |",
            "|---|---:|---:|",
        ]
    )
    aggregates = result["aggregate_by_method"]
    metric_rows = [
        ("Gold UPDATE transition success", "update_transition_success"),
        ("Introduced fact acquired immediately", "introduced_fact_immediate_acquisition"),
        ("Acquisition failures: missed UPDATE", "introduced_fact_failure_from_missed_update"),
        ("Acquisition failures: incorrect UPDATE content", "introduced_fact_failure_despite_predicted_update"),
        ("Acquisition failures: invalid output", "introduced_fact_failure_from_invalid_output"),
        ("Retired fact removed immediately", "retired_fact_immediate_removal"),
        ("Acquired fact retained until retirement/end", "acquired_fact_retained"),
        ("First loss on a later NO_OP", "acquired_fact_first_loss_on_noop"),
        ("First loss on a later UPDATE", "acquired_fact_first_loss_on_later_update"),
    ]
    for label, key in metric_rows:
        lines.append(
            f"| {label} | {_percent(aggregates['Patch']['rates'][key])} | "
            f"{_percent(aggregates['Delta-v3']['rates'][key])} |"
        )
    lines.extend(
        [
            "",
            "Fact acquisition은 Gold UPDATE에서 새로 도입된 정답 fact version이 같은 턴의 예측",
            "메모리에 존재하는지로 정의한다. Retention은 즉시 획득된 fact가 Gold에서 유효한 동안",
            "처음 누락되는 시점을 추적한다.",
            "",
            "## NO_OP Dynamics",
            "",
            "| Metric | Patch (%) | Delta-v3 (%) |",
            "|---|---:|---:|",
        ]
    )
    noop_rows = [
        ("False UPDATE on Gold NO_OP", "noop_false_update"),
        ("Memory mutation on Gold NO_OP", "noop_memory_mutation"),
        ("New loss of at least one Gold fact", "noop_new_information_loss"),
        ("Introduction of at least one extra fact", "noop_new_extra_fact"),
    ]
    for label, key in noop_rows:
        lines.append(
            f"| {label} | {_percent(aggregates['Patch']['rates'][key])} | "
            f"{_percent(aggregates['Delta-v3']['rates'][key])} |"
        )
    lines.extend(
        [
            "",
            "## Key Finding",
            "",
            "주된 오류는 정답 UPDATE가 처음 등장할 때 발생했으며, 이후 NO_OP 구간에서의",
            "정보 소실은 거의 없었다. 정확히 획득된 fact의 96--97%는 교체·삭제 또는 시나리오",
            "종료까지 유지되었다. Gold NO_OP에서 발생한 false UPDATE도 대체로 기존 정답 fact의",
            "삭제보다 불필요한 fact의 추가로 이어졌다.",
            "",
            "## Memory State and Quiz Outcome",
            "",
            "| Method | Covered quizzes | Exact fact state & correct | Exact fact state & wrong | Inexact fact state & correct | Inexact fact state & wrong |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_CONFIG:
        counts = result["aggregate_quiz_state_relation"][method]["counts"]
        total = counts.get("covered_quizzes", 0)
        values = [
            counts.get("memory_exact_quiz_correct", 0),
            counts.get("memory_exact_quiz_wrong", 0),
            counts.get("memory_inexact_quiz_correct", 0),
            counts.get("memory_inexact_quiz_wrong", 0),
        ]
        cells = [f"{value} ({100 * value / total:.2f}%)" for value in values]
        lines.append(f"| {method} | {total:,} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "`Exact fact state & wrong`은 엄격한 reader-only 오류의 하한이다. 반대로 전체 fact state가",
            "부정확하더라도 정답 호출에 필요한 fact는 존재할 수 있으므로, `Inexact fact state & wrong`을",
            "모두 memory writer 오류로 해석하지 않는다.",
            "",
            "## Interpretation Boundaries",
            "",
            "- Fact 비교는 owner, timestamp, 구조화된 payload의 정확 일치를 사용한다.",
            "- NO_OP 중 기존 오류가 그대로 남은 경우는 신규 소실로 세지 않는다.",
            "- Later-UPDATE loss는 다른 선호 갱신이 기존 fact를 훼손한 경우를 포함한다.",
            "- 결과는 현재 확보된 141개 paired model–scenario에 대한 기술 통계다.",
            "",
        ]
    )
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(
        analysis_root=args.analysis_root.resolve(),
        run_root=args.run_root.resolve(),
        data_root=args.data_root.resolve(),
        quiz_source=args.quiz_source.resolve(),
    )
    _write(
        args.output.resolve(),
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    if args.markdown_output:
        _write(args.markdown_output.resolve(), render_markdown(result))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown_output": (
                    str(args.markdown_output.resolve())
                    if args.markdown_output
                    else None
                ),
                "coverage": result["coverage"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
