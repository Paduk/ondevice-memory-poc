"""Run once-only Ollama closed-loop memory and VehicleMemBench Quiz tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT
from .dataset import IndexedMemoryDataset, default_catalog_path, ensure_catalog
from .methods import METHODS
from .ollama_client import OllamaAgentModel, OllamaClient
from .ubuntu_bridge import enable_ubuntu_runtime
from .validation import (
    memory_scores,
    row_with_runtime_state,
    state_from_input,
    state_to_input,
)


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    config = _read_json(run_dir / "config.json")
    method_name = args.method or str(config.get("method", ""))
    if method_name not in METHODS:
        raise ValueError(f"Unknown memory method: {method_name}")
    method = METHODS[method_name]()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (run_dir / "ollama-test").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(args.ollama_url, timeout_seconds=args.timeout_seconds)
    if not args.memory_only and not args.quiz_model:
        raise ValueError("--quiz-model is required unless --memory-only is used")
    environment = _environment(client, args.memory_model, args.quiz_model)
    catalog = ensure_catalog(args.data_root, default_catalog_path(args.workspace))
    memory_modes = list(dict.fromkeys(args.memory_modes))
    train_diagnostic_scenarios = list(
        dict.fromkeys(args.train_diagnostic_scenarios or [])
    )
    all_memory_scenarios = list(
        dict.fromkeys([*args.scenarios, *train_diagnostic_scenarios])
    )
    signature = _signature(
        {
            "schema": "palmclaw-ollama-test-v2",
            "dataset": catalog.metadata().get("source_fingerprint"),
            "method": method_name,
            "memory_model": environment["memory_model"],
            "quiz_model": environment["quiz_model"],
            "scenarios": args.scenarios,
            "train_diagnostic_scenarios": train_diagnostic_scenarios,
            "memory_modes": memory_modes,
            "seed": args.seed,
            "memory_context_length": args.memory_context_length,
            "quiz_context_length": args.quiz_context_length,
            "memory_max_tokens": args.memory_max_tokens,
            "quiz_max_tokens": args.quiz_max_tokens,
        }
    )
    quizzes = load_quizzes(args.data_root, args.scenarios)
    memory_summaries: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in memory_modes
    }
    if not args.quiz_only:
        for mode in memory_modes:
            for scenario in all_memory_scenarios:
                memory_summaries[mode].append(
                    run_memory_scenario(
                        client,
                        method,
                        catalog,
                        scenario,
                        quizzes.get(scenario, []),
                        output_dir=output_dir,
                        model=args.memory_model,
                        signature=signature,
                        seed=args.seed,
                        context_length=args.memory_context_length,
                        max_new_tokens=args.memory_max_tokens,
                        checkpoint_interval=args.checkpoint_interval,
                        turn_limit=args.turn_limit,
                        force=args.force,
                        evaluation_mode=mode,
                    )
                )
    else:
        for mode in memory_modes:
            memory_summaries[mode] = [
                _read_json(
                    output_dir
                    / _memory_directory(mode)
                    / f"s{scenario:03d}"
                    / "summary.json"
                )
                for scenario in all_memory_scenarios
            ]
    quiz_summaries: dict[str, dict[str, Any]] = {}
    if not args.memory_only:
        for mode in memory_modes:
            quiz_summaries[mode] = run_quizzes(
                client,
                quizzes,
                output_dir=output_dir,
                dataset_root=args.dataset_root,
                model=str(args.quiz_model),
                signature=signature,
                seed=args.seed,
                context_length=args.quiz_context_length,
                max_new_tokens=args.quiz_max_tokens,
                max_tool_rounds=args.max_tool_rounds,
                quiz_limit=args.quiz_limit,
                force=args.force,
                evaluation_mode=mode,
            )
    test_summaries = {
        mode: _select_scenario_summaries(summaries, args.scenarios)
        for mode, summaries in memory_summaries.items()
    }
    train_summaries = {
        mode: _select_scenario_summaries(summaries, train_diagnostic_scenarios)
        for mode, summaries in memory_summaries.items()
    }
    test_closed = _aggregate_mode(test_summaries, "closed_loop")
    test_teacher_forced = _aggregate_mode(test_summaries, "teacher_forced")
    train_closed = _aggregate_mode(train_summaries, "closed_loop")
    train_teacher_forced = _aggregate_mode(train_summaries, "teacher_forced")
    report = {
        "schema_version": "palmclaw-ollama-test-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "run_id": run_dir.name,
        "method": method_name,
        "environment": environment,
        "scenarios": args.scenarios,
        "train_diagnostic_scenarios": train_diagnostic_scenarios,
        "memory_modes": memory_modes,
        "memory": test_closed,
        "memory_teacher_forced": test_teacher_forced,
        "train_diagnostic": {
            "scenarios": train_diagnostic_scenarios,
            "memory": train_closed,
            "memory_teacher_forced": train_teacher_forced,
            "memory_mode_comparison": compare_memory_modes(
                train_closed, train_teacher_forced
            ),
        }
        if train_diagnostic_scenarios
        else None,
        # Keep the original field as the closed-loop Quiz result for backward
        # compatibility, and expose the gold-previous-memory diagnostic beside it.
        "quiz": quiz_summaries.get("closed_loop"),
        "quiz_teacher_forced": quiz_summaries.get("teacher_forced"),
        "output_dir": str(output_dir),
    }
    report["memory_mode_comparison"] = compare_memory_modes(
        report["memory"], report["memory_teacher_forced"]
    )
    report["quiz_mode_comparison"] = compare_quiz_modes(
        report["quiz"], report["quiz_teacher_forced"]
    )
    _write_json(output_dir / "summary.json", report)
    _write_json(run_dir / "ollama-test-summary.json", report)
    client.close()
    return report


def run_memory_scenario(
    client: OllamaClient,
    method: Any,
    catalog: Any,
    scenario: int,
    quizzes: list[dict[str, Any]],
    *,
    output_dir: Path,
    model: str,
    signature: str,
    seed: int,
    context_length: int,
    max_new_tokens: int,
    checkpoint_interval: int,
    turn_limit: int | None,
    force: bool,
    evaluation_mode: str = "closed_loop",
) -> dict[str, Any]:
    scenario_dir = output_dir / _memory_directory(evaluation_mode) / f"s{scenario:03d}"
    summary_path = scenario_dir / "summary.json"
    if summary_path.is_file() and not force:
        stored = _read_json(summary_path)
        if stored.get("signature") != signature:
            raise ValueError(f"Stale completed memory test: {summary_path}")
        return stored
    scenario_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = scenario_dir / "checkpoint.json"
    turns_path = scenario_dir / "turns.jsonl"
    row_ids = catalog.scenario_row_ids(scenario)
    if turn_limit is not None:
        row_ids = row_ids[:turn_limit]
    source = IndexedMemoryDataset(catalog, method.source_view, row_ids=row_ids)
    gold = IndexedMemoryDataset(catalog, "summary", row_ids=row_ids)
    requested_snapshots = defaultdict(list)
    for quiz in quizzes:
        requested_snapshots[int(quiz["memory_ref"]["global_turn_index"])].append(
            quiz["sample_id"]
        )
    state = method.initial_state()
    start_position = 0
    snapshots: dict[str, str] = {}
    if checkpoint_path.is_file() and not force:
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("signature") != signature:
            raise ValueError(f"Stale memory checkpoint: {checkpoint_path}")
        start_position = int(checkpoint["next_position"])
        state = _deserialize_state(method, checkpoint["state"])
        snapshots = dict(checkpoint.get("quiz_snapshots", {}))
        _trim_turns(turns_path, start_position)
    elif force:
        turns_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)

    for position in range(start_position, len(source)):
        canonical = _single(source, position)
        gold_row = _single(gold, position)
        if evaluation_mode == "teacher_forced":
            state = state_from_input(method, canonical["input"])
        runtime_row = row_with_runtime_state(canonical, method, state)
        result = client.chat(
            model=model,
            messages=(
                {"role": "system", "content": method.system_prompt},
                {"role": "user", "content": method.format_input(runtime_row)},
            ),
            json_mode=True,
            think=False,
            temperature=0,
            seed=seed,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
        parse_ok = apply_ok = False
        predicted_decision = "INVALID"
        error = None
        try:
            parsed = method.parse_output(result.content)
            parse_ok = True
            predicted_decision = parsed.decision
            next_state = method.apply_output(
                state, parsed, turn_id=str(canonical.get("turn_id", ""))
            )
            apply_ok = True
            if evaluation_mode == "closed_loop":
                state = next_state
            predicted_memory = method.materialize_memory(next_state)
        except Exception as exc:  # noqa: BLE001 - invalid model output is a metric.
            error = f"{type(exc).__name__}: {exc}"
            predicted_memory = method.materialize_memory(state)
        gold_memory = str(gold_row["target"].get("next_memory", ""))
        score = memory_scores(predicted_memory, gold_memory)
        global_turn = int(canonical["global_turn_index"])
        for sample_id in requested_snapshots.get(global_turn, []):
            snapshots[sample_id] = predicted_memory
        record = {
            "position": position,
            "evaluation_mode": evaluation_mode,
            "global_turn_index": global_turn,
            "turn_id": canonical.get("turn_id"),
            "gold_decision": canonical["target"]["decision"],
            "predicted_decision": predicted_decision,
            "parse_ok": parse_ok,
            "apply_ok": apply_ok,
            "state": score,
            "output": result.content,
            "error": error,
            "usage": _usage(result),
            "predicted_memory_sha256": _text_sha256(predicted_memory),
            "gold_memory_sha256": _text_sha256(gold_memory),
        }
        _append_jsonl(turns_path, record)
        next_position = position + 1
        if next_position % checkpoint_interval == 0 or next_position == len(source):
            _write_json(
                checkpoint_path,
                {
                    "signature": signature,
                    "next_position": next_position,
                    "state": _serialize_state(method, state),
                    "quiz_snapshots": snapshots,
                },
            )
    records = _deduplicated_turns(turns_path)
    missing_snapshots = sorted(
        quiz["sample_id"] for quiz in quizzes if quiz["sample_id"] not in snapshots
    )
    final_score = records[-1]["state"] if records else {"exact": 0.0, "f1": 0.0}
    summary = {
        "signature": signature,
        "scenario_index": scenario,
        "evaluation_mode": evaluation_mode,
        "status": "COMPLETED",
        "turns": len(records),
        "metrics": _memory_metrics(records),
        "final_state": final_score,
        "quiz_snapshots": snapshots,
        "missing_quiz_snapshots": missing_snapshots,
    }
    _write_json(summary_path, summary)
    return summary


def run_quizzes(
    client: OllamaClient,
    quizzes: dict[int, list[dict[str, Any]]],
    *,
    output_dir: Path,
    dataset_root: Path,
    model: str,
    signature: str,
    seed: int,
    context_length: int,
    max_new_tokens: int,
    max_tool_rounds: int,
    quiz_limit: int | None,
    force: bool,
    evaluation_mode: str = "closed_loop",
) -> dict[str, Any]:
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.vehicle_bench.dataset import (
        OFFICIAL_UPSTREAM_COMMIT,
        GoldToolCall,
        VehicleTask,
        load_vehicle_benchmark,
    )
    from palmclaw_ubuntu.vehicle_bench.memory import VehicleMemoryContext
    from palmclaw_ubuntu.vehicle_bench.runner import _run_agent_task
    from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime
    from palmclaw_ubuntu.vehicle_bench.tools import vehicle_tool_definitions

    dataset = load_vehicle_benchmark(
        dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    agent = OllamaAgentModel(
        client,
        model,
        context_length=context_length,
        max_output_tokens=max_new_tokens,
        seed=seed,
    )
    records = []
    selected = [quiz for scenario in sorted(quizzes) for quiz in quizzes[scenario]]
    if quiz_limit is not None:
        selected = selected[:quiz_limit]
    for quiz in selected:
        scenario = int(quiz["scenario_index"])
        memory_summary = _read_json(
            output_dir
            / _memory_directory(evaluation_mode)
            / f"s{scenario:03d}"
            / "summary.json"
        )
        memory_signature = memory_summary.get("signature")
        if not memory_signature:
            raise ValueError(f"Unsigned memory snapshot for S{scenario}")
        memory = memory_summary["quiz_snapshots"].get(quiz["sample_id"])
        if memory is None:
            raise ValueError(f"Missing predicted memory: {quiz['sample_id']}")
        checkpoint = (
            output_dir
            / _quiz_directory(evaluation_mode)
            / f"s{scenario:03d}"
            / (_safe_name(quiz["sample_id"]) + ".json")
        )
        task_signature = _signature(
            {
                "run": signature,
                "memory_run": memory_signature,
                "memory_evaluation_mode": evaluation_mode,
                "quiz": quiz["sample_id"],
                "memory": _text_sha256(memory),
            }
        )
        if checkpoint.is_file() and not force:
            stored = _read_json(checkpoint)
            if stored.get("evaluation_signature") != task_signature:
                raise ValueError(f"Stale Quiz result: {checkpoint}")
            records.append(stored)
            continue
        runtime = VehicleWorldRuntime(dataset.root)
        definitions = vehicle_tool_definitions(dataset.tool_schemas, timeout_seconds=5)
        task = VehicleTask(
            id=quiz["sample_id"],
            scenario_index=scenario,
            event_index=0,
            query=str(quiz["input"]["query"]),
            gold_memory="",
            reasoning_type=str(quiz["input"].get("reasoning_type", "unknown")),
            gold_calls=tuple(
                GoldToolCall(
                    name=str(call["name"]),
                    arguments=dict(call["arguments"]),
                    source="v2-on-device-gold",
                )
                for call in quiz["target"]["gold_calls"]
            ),
        )

        def memory_resolver(
            profile: str,
            query: str,
            *,
            content: str = memory,
            scenario_index: int = scenario,
            quiz_id: str = str(quiz["quiz_id"]),
        ) -> Any:
            del profile, query
            return VehicleMemoryContext(
                content=content,
                metadata={
                    "method": "on_device_turnwise",
                    "scenario_index": scenario_index,
                    "quiz_id": quiz_id,
                    "memory_sha256": _text_sha256(content),
                },
                trace={},
            )

        record = _run_agent_task(
            runtime=runtime,
            definitions=definitions,
            agent_model=agent,
            profile="cloud_recursive_summary",
            task=task,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=20_000,
            memory_resolver=memory_resolver,
            oracle_retrieval_annotations=None,
            oracle_gate_annotations=None,
            oracle_stage_fact_annotations=None,
        )
        record.update(
            {
                "evaluation_signature": task_signature,
                "quiz_type": quiz["quiz_type"],
                "quiz_id": quiz["quiz_id"],
                "memory_ref": quiz["memory_ref"],
                "agent_model": model,
                "memory_evaluation_mode": evaluation_mode,
            }
        )
        _write_json(checkpoint, record)
        records.append(record)
    return aggregate_quizzes(records)


def load_quizzes(
    data_root: Path, scenarios: list[int]
) -> dict[int, list[dict[str, Any]]]:
    selected = set(scenarios)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for filename in ("turn_quiz.jsonl", "final_quiz.jsonl"):
        with (data_root / filename).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                scenario = int(row["scenario_index"])
                if scenario in selected:
                    grouped[scenario].append(row)
    return dict(grouped)


def aggregate_memory(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [summary["metrics"] for summary in summaries]
    total_turns = sum(int(summary["turns"]) for summary in summaries)
    update_precision = _ratio_sum(metrics, "update_tp", ("update_tp", "false_update"))
    update_recall = _ratio_sum(metrics, "update_tp", ("update_tp", "missed_update"))
    update_f1 = (
        2 * update_precision * update_recall / (update_precision + update_recall)
        if update_precision + update_recall
        else 0.0
    )
    return {
        "scenarios": len(summaries),
        "turns": total_turns,
        "state_exact": _weighted(metrics, "state_exact", "turns"),
        "state_f1": _weighted(metrics, "state_f1", "turns"),
        "final_state_exact": _mean(
            summary["final_state"]["exact"] for summary in summaries
        ),
        "final_state_f1": _mean(summary["final_state"]["f1"] for summary in summaries),
        "update_precision": update_precision,
        "update_recall": update_recall,
        "update_f1": update_f1,
        "noop_specificity": _ratio_sum(
            metrics, "noop_correct", ("noop_correct", "false_update", "invalid_noop")
        ),
        "parse_success_rate": _weighted(metrics, "parse_success_rate", "turns"),
        "apply_success_rate": _weighted(metrics, "apply_success_rate", "turns"),
        "prefill_tokens": sum(metric["prefill_tokens"] for metric in metrics),
        "decode_tokens": sum(metric["decode_tokens"] for metric in metrics),
        "load_seconds": sum(metric["load_seconds"] for metric in metrics),
        "prefill_seconds": sum(metric["prefill_seconds"] for metric in metrics),
        "decode_seconds": sum(metric["decode_seconds"] for metric in metrics),
        "latency_seconds": sum(metric["latency_seconds"] for metric in metrics),
        "scenario_summaries": summaries,
    }


def _select_scenario_summaries(
    summaries: list[dict[str, Any]], scenarios: list[int]
) -> list[dict[str, Any]]:
    selected = set(scenarios)
    return [
        summary for summary in summaries if int(summary["scenario_index"]) in selected
    ]


def _aggregate_mode(
    summaries: dict[str, list[dict[str, Any]]], mode: str
) -> dict[str, Any] | None:
    selected = summaries.get(mode) or []
    return aggregate_memory(selected) if selected else None


def compare_memory_modes(
    closed_loop: dict[str, Any] | None,
    teacher_forced: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Report the degradation caused by feeding predictions back across turns."""
    if closed_loop is None or teacher_forced is None:
        return None
    return {
        "teacher_forced_minus_closed_loop": {
            key: float(teacher_forced[key]) - float(closed_loop[key])
            for key in ("state_f1", "final_state_f1", "update_f1")
        },
        "closed_loop_over_teacher_forced_latency": (
            float(closed_loop["latency_seconds"])
            / float(teacher_forced["latency_seconds"])
            if teacher_forced["latency_seconds"]
            else None
        ),
    }


def compare_quiz_modes(
    closed_loop: dict[str, Any] | None,
    teacher_forced: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compare downstream Quiz scores from predicted versus gold previous memory."""
    if closed_loop is None or teacher_forced is None:
        return None
    score_keys = (
        "exact_state_match",
        "state_f1",
        "tool_f1",
        "argument_exact_match",
    )
    return {
        group: {
            "teacher_forced_minus_closed_loop": {
                key: float(teacher_forced[group][key]) - float(closed_loop[group][key])
                for key in score_keys
            }
        }
        for group in sorted(set(closed_loop) & set(teacher_forced))
    }


def aggregate_quizzes(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[str(record["quiz_type"]).upper()].append(record)
        groups[f"reasoning:{record['reasoning_type']}"].append(record)
    return {name: _quiz_metrics(selected) for name, selected in sorted(groups.items())}


def _quiz_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    timing = _quiz_ollama_timing(records)
    return {
        "tasks": len(records),
        "completion_rate": _mean(
            record.get("status") == "completed" for record in records
        ),
        "exact_state_match": _mean(
            bool((record.get("score") or {}).get("exact_state_match"))
            for record in records
        ),
        "state_f1": _mean(
            float(
                (record.get("score") or {}).get("state_score", {}).get("f1_positive", 0)
            )
            for record in records
        ),
        "tool_f1": _mean(
            float((record.get("score") or {}).get("tool_score", {}).get("f1", 0))
            for record in records
        ),
        "argument_exact_match": _mean(
            bool(record.get("argument_exact_match")) for record in records
        ),
        "input_tokens": sum(
            int(record.get("usage", {}).get("input_tokens", 0)) for record in records
        ),
        "output_tokens": sum(
            int(record.get("usage", {}).get("output_tokens", 0)) for record in records
        ),
        "latency_seconds": sum(
            float(record.get("usage", {}).get("model_latency_ms", 0))
            for record in records
        )
        / 1000,
        **timing,
    }


def _quiz_ollama_timing(records: list[dict[str, Any]]) -> dict[str, float]:
    totals = {
        "load_seconds": 0.0,
        "prefill_seconds": 0.0,
        "decode_seconds": 0.0,
    }
    mapping = {
        "load_duration_ns": "load_seconds",
        "prompt_duration_ns": "prefill_seconds",
        "output_duration_ns": "decode_seconds",
    }
    for record in records:
        for call in record.get("model_trace") or []:
            metadata = call.get("metadata") or {}
            for source, target in mapping.items():
                totals[target] += float(metadata.get(source) or 0) / 1_000_000_000
    return totals


def _memory_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    update_tp = false_update = missed_update = noop_correct = invalid_noop = 0
    for record in records:
        gold = record["gold_decision"]
        predicted = record["predicted_decision"]
        if gold == "UPDATE" and predicted == "UPDATE":
            update_tp += 1
        elif gold == "NO_OP" and predicted == "UPDATE":
            false_update += 1
        elif gold == "UPDATE":
            missed_update += 1
        elif predicted == "NO_OP":
            noop_correct += 1
        else:
            invalid_noop += 1
    return {
        "turns": len(records),
        "state_exact": _mean(record["state"]["exact"] for record in records),
        "state_f1": _mean(record["state"]["f1"] for record in records),
        "parse_success_rate": _mean(record["parse_ok"] for record in records),
        "apply_success_rate": _mean(record["apply_ok"] for record in records),
        "update_tp": update_tp,
        "false_update": false_update,
        "missed_update": missed_update,
        "noop_correct": noop_correct,
        "invalid_noop": invalid_noop,
        "prefill_tokens": sum(record["usage"]["prefill_tokens"] for record in records),
        "decode_tokens": sum(record["usage"]["decode_tokens"] for record in records),
        "load_seconds": sum(record["usage"]["load_seconds"] for record in records),
        "prefill_seconds": sum(
            record["usage"]["prefill_seconds"] for record in records
        ),
        "decode_seconds": sum(record["usage"]["decode_seconds"] for record in records),
        "latency_seconds": sum(
            record["usage"]["latency_seconds"] for record in records
        ),
    }


def _environment(
    client: OllamaClient, memory_model: str, quiz_model: str | None
) -> dict[str, Any]:
    tags = {str(item.get("name") or item.get("model")): item for item in client.tags()}
    requested = [model for model in (memory_model, quiz_model) if model]
    missing = [model for model in requested if model not in tags]
    if missing:
        raise ValueError(f"Ollama model tag is not installed: {missing}")
    return {
        "ollama_version": client.version(),
        "memory_model": {"tag": memory_model, **tags[memory_model]},
        "quiz_model": ({"tag": quiz_model, **tags[quiz_model]} if quiz_model else None),
    }


def _usage(result: Any) -> dict[str, Any]:
    return {
        "prefill_tokens": result.prompt_tokens,
        "decode_tokens": result.output_tokens,
        "latency_seconds": result.latency_seconds,
        "load_seconds": result.load_duration_ns / 1_000_000_000,
        "prefill_seconds": result.prompt_duration_ns / 1_000_000_000,
        "decode_seconds": result.output_duration_ns / 1_000_000_000,
    }


def _serialize_state(method: Any, state: Any) -> dict[str, Any]:
    return state_to_input(method, state)


def _memory_directory(evaluation_mode: str) -> str:
    if evaluation_mode == "closed_loop":
        return "memory"
    if evaluation_mode == "teacher_forced":
        return "memory_teacher_forced"
    raise ValueError(f"Unknown memory evaluation mode: {evaluation_mode}")


def _quiz_directory(evaluation_mode: str) -> str:
    if evaluation_mode == "closed_loop":
        return "quiz"
    if evaluation_mode != "teacher_forced":
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
    return "quiz_teacher_forced"


def _deserialize_state(method: Any, value: Mapping[str, Any]) -> Any:
    return state_from_input(method, value)


def _trim_turns(path: Path, next_position: int) -> None:
    if not path.is_file():
        return
    records = [
        record
        for record in _read_jsonl(path)
        if int(record.get("position", -1)) < next_position
    ]
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _deduplicated_turns(path: Path) -> list[dict[str, Any]]:
    by_position = {int(record["position"]): record for record in _read_jsonl(path)}
    return [by_position[position] for position in sorted(by_position)]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except FileNotFoundError:
        pass
    return records


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _single(dataset: IndexedMemoryDataset, position: int) -> dict[str, Any]:
    row = dataset[position]
    if not isinstance(row, dict):
        raise TypeError("Expected one row")
    return copy.deepcopy(row)


def _safe_name(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _signature(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _mean(values: Iterable[Any]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _weighted(metrics: list[dict[str, Any]], value: str, weight: str) -> float:
    denominator = sum(float(metric[weight]) for metric in metrics)
    return (
        sum(float(metric[value]) * float(metric[weight]) for metric in metrics)
        / denominator
        if denominator
        else 0.0
    )


def _ratio_sum(
    metrics: list[dict[str, Any]], numerator: str, denominator_parts: tuple[str, ...]
) -> float:
    top = sum(int(metric[numerator]) for metric in metrics)
    bottom = sum(
        sum(int(metric[key]) for key in denominator_parts) for metric in metrics
    )
    return top / bottom if bottom else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-model", required=True)
    parser.add_argument("--quiz-model")
    parser.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--scenarios", nargs="+", type=int, default=list(range(91, 101))
    )
    parser.add_argument(
        "--train-diagnostic-scenarios",
        nargs="+",
        type=int,
        help="Optional S1-S80 retention diagnostics, reported separately from Test.",
    )
    parser.add_argument(
        "--memory-modes",
        nargs="+",
        choices=("closed_loop", "teacher_forced"),
        default=["closed_loop"],
        help="Memory evaluation axes; use both for error-accumulation diagnosis.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-context-length", type=int, default=8192)
    parser.add_argument("--quiz-context-length", type=int, default=8192)
    parser.add_argument("--memory-max-tokens", type=int, default=768)
    parser.add_argument("--quiz-max-tokens", type=int, default=1024)
    parser.add_argument("--max-tool-rounds", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--turn-limit", type=int)
    parser.add_argument("--quiz-limit", type=int)
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--quiz-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.memory_only and args.quiz_only:
        raise ValueError("--memory-only and --quiz-only are mutually exclusive")
    if any(scenario < 91 or scenario > 100 for scenario in args.scenarios):
        raise ValueError("Test scenarios must be S91-S100")
    train_diagnostics = args.train_diagnostic_scenarios or []
    if any(scenario < 1 or scenario > 80 for scenario in train_diagnostics):
        raise ValueError("Train diagnostic scenarios must be S1-S80")
    result = run_test(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
