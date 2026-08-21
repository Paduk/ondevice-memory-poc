#!/usr/bin/env python3
"""Summarize the paired S1-S5 VehicleMemBench V2 three-way evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    default_quality_artifact_paths,
)

DEFAULT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-three-way-evaluation"
)
METHODS = ("post_hoc", "hybrid", "native_turnwise")
PRICES = {
    "gpt-5.6-luna": {"input": 0.20, "cached": 0.02, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gpt-5.6-sol": {"input": 5.00, "cached": 0.50, "output": 30.00},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def run(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    agent = _read_json(root / "agent" / "agent-summary.json")
    answerability = _read_json(root / "answerability" / "answerability-summary.json")
    manifest = _read_json(root / "update-audit" / "manifest.json")
    audit = _audit_metrics(root, manifest)
    final_audit_by_case = audit.pop("_final_by_case")
    cross_errors = _cross_error_metrics(root, final_audit_by_case)
    generation = _generation_metrics(root.parent)
    scenario_metrics = _scenario_metrics(root, final_audit_by_case)
    methods = {}
    for method in METHODS:
        agent_metrics = agent["metrics"][f"{method}/all"]
        answer_metrics = answerability["metrics"][f"{method}/all"]
        answer_usage = _answerability_usage(root, method)
        agent_usage = {
            "input_tokens": agent_metrics["input_tokens"],
            "cached_tokens": 0,
            "output_tokens": agent_metrics["output_tokens"],
            "total_tokens": (
                agent_metrics["input_tokens"] + agent_metrics["output_tokens"]
            ),
        }
        methods[method] = {
            "quiz_agent": {
                key: agent_metrics[key]
                for key in (
                    "tasks",
                    "exact_state_match",
                    "tool_f1",
                    "argument_exact_match",
                    "input_tokens",
                    "output_tokens",
                    "latency_ms",
                )
            },
            "answerability": {
                **answer_metrics,
                "usage": answer_usage,
            },
            "update_audit_core": audit[method],
            "cross_errors": cross_errors[method],
            "generation": generation[method],
            "evaluation_cost_usd": {
                "quiz_agent": _cost(agent_usage, "gpt-5.6-luna"),
                "answerability": _cost(answer_usage, "gpt-5.6-terra"),
                "update_audit": audit[method]["estimated_cost_usd"],
            },
        }
        costs = methods[method]["evaluation_cost_usd"]
        costs["total"] = sum(costs.values())
    result = {
        "schema_version": "vehiclemembench-v2-three-way-summary-v1",
        "scope": {
            "scenarios": [1, 2, 3, 4, 5],
            "quiz_repeats": 2,
            "quiz_tasks_per_method": 400,
            "answerability_quizzes_per_method": 200,
            "audit_core_cases_per_method": 86,
            "expanded_no_op_cases_excluded_from_primary_comparison": True,
        },
        "status": "PROVISIONAL" if audit["unresolved"] else "COMPLETED",
        "unresolved_audit_case_ids": audit["unresolved"],
        "methods": methods,
        "scenario_metrics": scenario_metrics,
    }
    _write_json(root / "three-way-summary.json", result)
    (root / "three-way-summary.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _audit_metrics(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    core_ids: dict[str, set[str]] = defaultdict(set)
    sampled_no_ops: dict[str, set[str]] = defaultdict(set)
    for artifact in manifest["artifacts"]:
        method = artifact["method"]
        scenario = int(artifact["scenario_index"])
        prefix = f"{method}:s{scenario:02d}:"
        core_ids[method].update(
            prefix + event_id for event_id in artifact["judge_event_ids"]
        )
        sampled_no_ops[method].update(
            prefix + event_id
            for event_ids in artifact["sampled_no_op_events_by_stratum"].values()
            for event_id in event_ids
        )
    consensus_by_id = {}
    sol_by_id = {}
    audit_root = root / "update-audit" / "judge"
    for path in audit_root.rglob("consensus.json"):
        report = _read_json(path)
        consensus_by_id[report["case_id"]] = report
        sol_path = path.with_name("sol.json")
        if sol_path.is_file():
            sol_by_id[report["case_id"]] = _read_json(sol_path)

    unresolved = []
    final_by_case: dict[str, dict[str, Any] | None] = {}
    output: dict[str, Any] = {}
    for method in METHODS:
        decisions = []
        role_usage: dict[str, Counter[str]] = {
            role: Counter() for role in ("luna", "terra", "sol")
        }
        method_unresolved = []
        for case_id in sorted(core_ids[method]):
            consensus = consensus_by_id[case_id]
            for role in ("luna", "terra"):
                role_usage[role].update(
                    consensus["judge_reports"][role].get("usage", {})
                )
            if consensus["status"] == "AGREED":
                decision = consensus["adopted_decision"]
            else:
                sol = sol_by_id[case_id]
                role_usage["sol"].update(sol.get("usage", {}))
                decision = (
                    sol.get("decision") if sol["resolution"] == "RESOLVE" else None
                )
            if decision is None:
                method_unresolved.append(case_id)
                unresolved.append(case_id)
            else:
                decisions.append((case_id, decision))
            final_by_case[case_id] = decision
        verdicts = Counter(decision["verdict"] for _, decision in decisions)
        memory = Counter(decision["memory_result"] for _, decision in decisions)
        expected = Counter(
            item["status"]
            for _, decision in decisions
            for item in decision["expected_updates"]
        )
        candidates = Counter(
            item["status"]
            for _, decision in decisions
            for item in decision["candidate_updates"]
        )
        supported_expected = (
            sum(expected.values()) - expected["UNSUPPORTED_BY_DIALOGUE"]
        )
        no_op_decisions = [
            decision
            for case_id, decision in decisions
            if case_id in sampled_no_ops[method]
        ]
        usages = {
            role: _normalized_usage(values) for role, values in role_usage.items()
        }
        output[method] = {
            "case_count": len(core_ids[method]),
            "resolved_count": len(decisions),
            "unresolved_count": len(method_unresolved),
            "verdicts": dict(sorted(verdicts.items())),
            "pass_rate": verdicts["PASS"] / len(decisions),
            "memory_results": dict(sorted(memory.items())),
            "memory_faithful_rate": memory["FAITHFUL"] / len(decisions),
            "expected_update_statuses": dict(sorted(expected.items())),
            "expected_update_recall": (
                expected["FOUND"] / supported_expected if supported_expected else None
            ),
            "candidate_update_statuses": dict(sorted(candidates.items())),
            "candidate_update_precision": (
                candidates["VALID"] / sum(candidates.values()) if candidates else None
            ),
            "sampled_no_op_count": len(no_op_decisions),
            "sampled_no_op_accuracy": sum(
                decision["verdict"] == "PASS"
                and decision["memory_result"] == "FAITHFUL"
                for decision in no_op_decisions
            )
            / len(no_op_decisions),
            "judge_usage": usages,
            "estimated_cost_usd": sum(
                _cost(usages[role], model)
                for role, model in (
                    ("luna", "gpt-5.6-luna"),
                    ("terra", "gpt-5.6-terra"),
                    ("sol", "gpt-5.6-sol"),
                )
            ),
        }
    output["unresolved"] = unresolved
    output["_final_by_case"] = final_by_case
    return output


def _cross_error_metrics(
    root: Path,
    final_audit_by_case: dict[str, dict[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    answer_verdicts: dict[tuple[str, str], str] = {}
    for method in METHODS:
        for path in (root / "answerability" / method).rglob("*.json"):
            report = _read_json(path)
            for result in report["results"]:
                answer_verdicts[(method, result["quiz_id"])] = result["verdict"]
    output = {}
    for method in METHODS:
        counts: Counter[str] = Counter()
        mapped_quizzes = set()
        for path in (root / "agent" / method).rglob("tasks/*.json"):
            record = _read_json(path)
            quiz_id = str(record["quiz_id"])
            event_id = _turn_quiz_event_id(quiz_id)
            if event_id is None:
                continue
            scenario = int(record["quality_scenario_index"])
            audit = final_audit_by_case.get(f"{method}:s{scenario:02d}:{event_id}")
            if audit is None:
                continue
            answer = answer_verdicts[(method, quiz_id)]
            audit_status = (
                "AUDIT_PASS"
                if audit["verdict"] == "PASS" and audit["memory_result"] == "FAITHFUL"
                else "AUDIT_ERROR"
            )
            answer_status = "SUPPORTED" if answer == "FULL_SUPPORT" else "UNSUPPORTED"
            agent_status = (
                "CORRECT"
                if bool(record.get("score", {}).get("exact_state_match"))
                else "WRONG"
            )
            counts[f"{audit_status}/{answer_status}/{agent_status}"] += 1
            mapped_quizzes.add(quiz_id)
        output[method] = {
            "scope": (
                "turn quizzes with a resolved core audit event; Agent repeats retained"
            ),
            "mapped_unique_quizzes": len(mapped_quizzes),
            "mapped_agent_tasks": sum(counts.values()),
            "counts": dict(sorted(counts.items())),
        }
    return output


def _scenario_metrics(
    root: Path,
    final_audit_by_case: dict[str, dict[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    agent_records: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    answer_results: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    audit_decisions: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for method in METHODS:
        for path in (root / "agent" / method).rglob("tasks/*.json"):
            record = _read_json(path)
            agent_records[(method, int(record["quality_scenario_index"]))].append(
                record
            )
        for path in (root / "answerability" / method).rglob("*.json"):
            report = _read_json(path)
            answer_results[(method, int(report["scenario_index"]))].extend(
                report["results"]
            )
    for case_id, decision in final_audit_by_case.items():
        if decision is None:
            continue
        method, scenario_label, _ = case_id.split(":", 2)
        audit_decisions[(method, int(scenario_label.removeprefix("s")))].append(
            decision
        )
    output = {}
    for method in METHODS:
        output[method] = {}
        for scenario in range(1, 6):
            agents = agent_records[(method, scenario)]
            answers = answer_results[(method, scenario)]
            audits = audit_decisions[(method, scenario)]
            output[method][f"s{scenario:02d}"] = {
                "quiz_tasks": len(agents),
                "exact_state_match": _mean(
                    bool(item.get("score", {}).get("exact_state_match"))
                    for item in agents
                ),
                "tool_f1": _mean(
                    float(item.get("score", {}).get("tool_score", {}).get("f1", 0))
                    for item in agents
                ),
                "argument_exact_match": _mean(
                    bool(item.get("argument_exact_match")) for item in agents
                ),
                "answerability_quizzes": len(answers),
                "full_support_rate": _mean(
                    item["verdict"] == "FULL_SUPPORT" for item in answers
                ),
                "audit_resolved_cases": len(audits),
                "audit_pass_rate": _mean(
                    item["verdict"] == "PASS" and item["memory_result"] == "FAITHFUL"
                    for item in audits
                ),
            }
    return output


def _turn_quiz_event_id(quiz_id: str) -> str | None:
    if not quiz_id.startswith("turn-quiz-"):
        return None
    body = quiz_id.removeprefix("turn-quiz-")
    event_id, separator, turn_index = body.rpartition("-")
    if not separator or not turn_index.isdigit():
        return None
    return event_id


def _answerability_usage(root: Path, method: str) -> dict[str, int]:
    usage: Counter[str] = Counter()
    for path in (root / "answerability" / method).rglob("*.json"):
        usage.update(_read_json(path).get("usage", {}))
    return _normalized_usage(usage)


def _generation_metrics(evaluation_root: Path) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Counter[str]]] = {
        method: defaultdict(Counter) for method in METHODS
    }
    for paths in default_quality_artifact_paths(evaluation_root):
        memory = _read_json(paths.memory_artifact)
        final_quiz = _read_json(paths.final_quiz_artifact)
        turn_quiz = _read_json(paths.turn_quiz_artifact)
        if paths.method == "native_turnwise":
            _update_usage(
                stages[paths.method]["dialogue"],
                memory["audit"]["dialogue_usage"],
            )
            _update_usage(
                stages[paths.method]["memory_alignment"],
                memory["audit"]["alignment_usage"],
            )
        else:
            for generated in final_quiz["generated_dialogues"]:
                _update_usage(stages[paths.method]["dialogue"], generated["usage"])
            for checkpoint in memory["event_checkpoints"]:
                _update_usage(
                    stages[paths.method]["memory_alignment"],
                    checkpoint["generated_alignment"]["usage"],
                )
        for quiz in turn_quiz["quizzes"]:
            _update_usage(
                stages[paths.method]["turn_quiz"],
                quiz["generated_query"]["usage"],
            )
        for quiz in final_quiz["generated_quizzes"]:
            _update_usage(
                stages[paths.method]["final_quiz"],
                quiz["usage"],
            )
    output = {}
    for method in METHODS:
        normalized = {
            stage: _normalized_generation_usage(usage)
            for stage, usage in stages[method].items()
        }
        total: Counter[str] = Counter()
        for usage in normalized.values():
            total.update(usage)
        normalized_total = _normalized_generation_usage(total)
        output[method] = {
            "model": "gpt-5.6-terra",
            "stages": normalized,
            "total": normalized_total,
            "estimated_cost_usd": _cost(
                normalized_total,
                "gpt-5.6-terra",
            ),
        }
    return output


def _update_usage(target: Counter[str], usage: dict[str, Any]) -> None:
    for key in (
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
    ):
        target[key] += int(usage.get(key, 0))


def _normalized_generation_usage(values: Counter[str]) -> dict[str, int]:
    return {
        key: int(values[key])
        for key in (
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "total_tokens",
            "latency_ms",
        )
    }


def _normalized_usage(values: Counter[str]) -> dict[str, int]:
    return {
        key: int(values[key])
        for key in ("input_tokens", "cached_tokens", "output_tokens", "total_tokens")
    }


def _cost(usage: dict[str, int], model: str) -> float:
    price = PRICES[model]
    cached = usage.get("cached_tokens", 0)
    regular = max(0, usage.get("input_tokens", 0) - cached)
    return (
        regular * price["input"]
        + cached * price["cached"]
        + usage.get("output_tokens", 0) * price["output"]
    ) / 1_000_000


def _mean(values) -> float:
    selected = list(values)
    return sum(selected) / len(selected) if selected else 0.0


def _markdown(result: dict[str, Any]) -> str:
    rows = []
    generation_rows = []
    cross_rows = []
    scenario_rows = []
    for method in METHODS:
        item = result["methods"][method]
        agent = item["quiz_agent"]
        answer = item["answerability"]
        audit = item["update_audit_core"]
        rows.append(
            "| "
            + " | ".join(
                (
                    method,
                    f"{agent['exact_state_match']:.3f}",
                    f"{agent['tool_f1']:.3f}",
                    f"{agent['argument_exact_match']:.3f}",
                    f"{answer['full_support_rate']:.3f}",
                    f"{audit['pass_rate']:.3f}",
                    f"{audit['expected_update_recall']:.3f}",
                    f"{audit['candidate_update_precision']:.3f}",
                    f"{audit['sampled_no_op_accuracy']:.3f}",
                    f"{agent['input_tokens'] / 1e6:.3f}M",
                    f"{agent['output_tokens'] / 1e6:.3f}M",
                    f"{agent['latency_ms'] / 60000:.1f}m",
                    f"${item['evaluation_cost_usd']['total']:.2f}",
                )
            )
            + " |"
        )
        cross_rows.append(
            f"| {method} | {item['cross_errors']['mapped_unique_quizzes']} | "
            f"{item['cross_errors']['mapped_agent_tasks']} | "
            f"`{json.dumps(item['cross_errors']['counts'], sort_keys=True)}` |"
        )
        generation = item["generation"]
        generation_usage = generation["total"]
        generation_rows.append(
            f"| {method} | {generation_usage['input_tokens'] / 1e6:.3f}M | "
            f"{generation_usage['output_tokens'] / 1e6:.3f}M | "
            f"{generation_usage['latency_ms'] / 60000:.1f}m | "
            f"${generation['estimated_cost_usd']:.2f} |"
        )
        for scenario, metrics in result["scenario_metrics"][method].items():
            scenario_rows.append(
                f"| {method} | {scenario.upper()} | "
                f"{metrics['exact_state_match']:.3f} | "
                f"{metrics['tool_f1']:.3f} | "
                f"{metrics['argument_exact_match']:.3f} | "
                f"{metrics['full_support_rate']:.3f} | "
                f"{metrics['audit_pass_rate']:.3f} | "
                f"{metrics['audit_resolved_cases']}/"
                f"{17 if scenario != 's05' else 18} |"
            )
    return "\n".join(
        (
            "# VehicleMemBench V2 three-way S1–S5 results",
            "",
            "Primary audit comparison uses the same 86 core cases per method; ",
            "automatically expanded NO_OP strata are excluded from the main rate.",
            "",
            "| Method | ESM | Tool F1 | Arg Exact | Answer support | Audit PASS | "
            "Expected recall | Candidate precision | NO_OP acc. | Quiz input | "
            "Quiz output | Quiz latency | Eval cost |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Dataset generation efficiency",
            "",
            "| Method | Input | Output | Cumulative latency | Estimated cost |",
            "|---|---:|---:|---:|---:|",
            *generation_rows,
            "",
            "## Scenario-level paired results",
            "",
            "| Method | Scenario | ESM | Tool F1 | Arg Exact | Answer support | "
            "Audit PASS | Audit resolved |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *scenario_rows,
            "",
            "## Cross-error counts",
            "",
            "| Method | Unique turn quizzes | Agent tasks | "
            "Audit/Support/Agent counts |",
            "|---|---:|---:|---|",
            *cross_rows,
            "",
            f"Status: {result['status']}. Unresolved audit cases: "
            f"{', '.join(result['unresolved_audit_case_ids']) or 'none'}.",
            "",
        )
    )


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
    print(json.dumps(run(build_parser().parse_args().root), indent=2))


if __name__ == "__main__":
    main()
