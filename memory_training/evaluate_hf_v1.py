"""Evaluate an HF/PEFT memory model on the original VehicleMemBench V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .methods import METHODS
from .quiz_sft import (
    SYSTEM_PROMPT,
    VehicleToolSchemaStore,
    _candidate_tools,
    _user_message,
)
from .training_data import ChatExampleEncoder
from .ubuntu_bridge import enable_ubuntu_runtime
from .v1_memory_replay import parse_history
from .validation import (
    evaluate_quiz_tool_calling,
    generate_outputs_batch,
    state_from_input,
    state_to_input,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument(
        "--method",
        choices=("summary", "patch", "temporal_patch", "delta_v2", "delta_v3"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, required=True)
    parser.add_argument(
        "--scenario-variant",
        action="append",
        default=[],
        metavar="EVAL_INDEX:BASE_INDEX:HISTORY_PATH:QA_PATH",
        help=(
            "Register a rewritten V1 scenario under a distinct evaluation index. "
            "Tool-call gold labels are inherited from BASE_INDEX while query and "
            "gold-memory text are read from QA_PATH. May be repeated."
        ),
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vehicle-tools-path", type=Path)
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--quiz-max-new-tokens", type=int, default=256)
    parser.add_argument("--scenario-batch-size", type=int, default=8)
    parser.add_argument("--quiz-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = tuple(dict.fromkeys(args.scenarios))
    variant_specs = _parse_scenario_variants(args.scenario_variant)
    variant_indices = {item[0] for item in variant_specs}
    if not scenarios or any(
        value < 1 or (value > 50 and value not in variant_indices)
        for value in scenarios
    ):
        raise ValueError(
            "V1 scenarios must be S1-S50 or registered --scenario-variant indexes"
        )
    if min(
        args.scenario_batch_size,
        args.quiz_batch_size,
        args.checkpoint_interval,
    ) < 1:
        raise ValueError("Batch sizes and checkpoint interval must be positive")

    checkpoint = args.checkpoint.resolve()
    adapter = checkpoint / "adapter"
    if not adapter.is_dir():
        raise FileNotFoundError(f"Checkpoint adapter not found: {adapter}")
    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    dataset_root = args.vehiclemembench_root.resolve()
    output_dir = args.output_dir.resolve()
    scenario_dir = output_dir / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    _configure_runtime_paths(workspace)
    enable_ubuntu_runtime()

    from accelerate import Accelerator
    from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark

    from .models import load_peft_bundle

    benchmark = load_vehicle_benchmark(dataset_root, strict=True)
    selected_by_index = {
        value: benchmark.scenario(value) for value in scenarios if value <= 50
    }
    selected_by_index.update(_load_scenario_variants(variant_specs, benchmark))
    selected = [selected_by_index[value] for value in scenarios]
    dataset_sha256 = _variant_dataset_sha256(
        benchmark.manifest.dataset_sha256, variant_specs
    )
    tools_path = (args.vehicle_tools_path or data_root / "vehicle_tools.json").resolve()
    tools = VehicleToolSchemaStore(tools_path)
    quiz_rows = _build_v1_quiz_rows(selected, tools_path, tools.sha256)
    quiz_indices = list(range(len(quiz_rows)))
    histories = {
        scenario.index: parse_history(scenario.history_path) for scenario in selected
    }
    total_turns = sum(len(value) for value in histories.values())
    signature = _signature(
        args,
        checkpoint,
        scenarios,
        dataset_sha256,
        tools.sha256,
    )
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    started_at = _prepare_manifest(
        manifest_path,
        signature=signature,
        args=args,
        checkpoint=checkpoint,
        scenarios=scenarios,
        tasks=len(quiz_rows),
        turns=total_turns,
        dataset_sha256=dataset_sha256,
    )
    total_work = total_turns + 2 * len(quiz_rows)
    _progress(
        progress_path,
        started_at=started_at,
        status="INITIALIZING",
        phase="model_loading",
        scenarios=scenarios,
        completed_scenarios=_completed_scenarios(scenario_dir, signature),
        item=0,
        items=total_work,
        phase_item=0,
        phase_items=len(quiz_rows),
    )

    accelerator = Accelerator()
    bundle = load_peft_bundle(
        args.model,
        adapter_path=adapter,
        gradient_checkpointing=False,
        cache_dir=workspace / "cache" / "huggingface" / "hub",
    )
    model = accelerator.prepare(bundle.model)
    model.eval()
    method = METHODS[args.method]()

    gold_path = output_dir / "gold-memory-quiz.json"
    gold_quiz = _load_signed_result(gold_path, signature)
    if gold_quiz is None:
        print(f"Gold-memory Quiz started: {len(quiz_rows)} tasks", flush=True)

        def gold_progress(payload: Mapping[str, Any]) -> None:
            position = int(payload.get("item", 0))
            _progress(
                progress_path,
                started_at=started_at,
                status="RUNNING",
                phase="gold_memory_quiz",
                scenarios=scenarios,
                completed_scenarios=(),
                item=position,
                items=total_work,
                phase_item=position,
                phase_items=len(quiz_rows),
            )

        gold_quiz = evaluate_quiz_tool_calling(
            model,
            bundle.tokenizer,
            quiz_rows,
            tools,
            quiz_indices,
            dataset_root=dataset_root,
            accelerator=accelerator,
            max_length=args.max_length,
            max_new_tokens=args.quiz_max_new_tokens,
            batch_size=args.quiz_batch_size,
            progress_callback=gold_progress,
            allow_gold_execution_failure=True,
        )
        _write_json(gold_path, {"signature": signature, **gold_quiz})
        print(f"Gold-memory Quiz complete: ESM={gold_quiz['esm']:.4f}", flush=True)
    else:
        print(f"Gold-memory Quiz restored: ESM={gold_quiz['esm']:.4f}", flush=True)

    reports = _replay_histories(
        model=model,
        tokenizer=bundle.tokenizer,
        method=method,
        accelerator=accelerator,
        histories=histories,
        scenarios=scenarios,
        scenario_dir=scenario_dir,
        signature=signature,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.scenario_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        progress_path=progress_path,
        started_at=started_at,
        work_offset=len(quiz_rows),
        total_work=total_work,
    )

    predicted_path = output_dir / "predicted-memory-quiz.json"
    predicted_quiz = _load_signed_result(predicted_path, signature)
    if predicted_quiz is None:
        final_memory = {
            int(report["scenario_index"]): str(report["final_memory"])
            for report in reports
        }
        overrides = {
            str(row["sample_id"]): final_memory[int(row["scenario_index"])]
            for row in quiz_rows
        }
        print(f"Predicted-memory Quiz started: {len(quiz_rows)} tasks", flush=True)

        def predicted_progress(payload: Mapping[str, Any]) -> None:
            position = int(payload.get("item", 0))
            _progress(
                progress_path,
                started_at=started_at,
                status="RUNNING",
                phase="predicted_memory_quiz",
                scenarios=scenarios,
                completed_scenarios=scenarios,
                item=len(quiz_rows) + total_turns + position,
                items=total_work,
                phase_item=position,
                phase_items=len(quiz_rows),
            )

        predicted_quiz = evaluate_quiz_tool_calling(
            model,
            bundle.tokenizer,
            quiz_rows,
            tools,
            quiz_indices,
            dataset_root=dataset_root,
            accelerator=accelerator,
            max_length=args.max_length,
            max_new_tokens=args.quiz_max_new_tokens,
            memory_overrides=overrides,
            batch_size=args.quiz_batch_size,
            progress_callback=predicted_progress,
            allow_gold_execution_failure=True,
        )
        _write_json(predicted_path, {"signature": signature, **predicted_quiz})
        print(
            f"Predicted-memory Quiz complete: ESM={predicted_quiz['esm']:.4f}",
            flush=True,
        )
    else:
        print(
            f"Predicted-memory Quiz restored: ESM={predicted_quiz['esm']:.4f}",
            flush=True,
        )

    summary = _aggregate(
        scenarios=scenarios,
        signature=signature,
        reports=reports,
        gold_quiz=gold_quiz,
        predicted_quiz=predicted_quiz,
    )
    _write_json(output_dir / "summary.json", summary)
    _progress(
        progress_path,
        started_at=started_at,
        status="COMPLETED",
        phase="complete",
        scenarios=scenarios,
        completed_scenarios=scenarios,
        item=total_work,
        items=total_work,
        phase_item=len(quiz_rows),
        phase_items=len(quiz_rows),
        esm=float(predicted_quiz["esm"]),
    )
    return summary


def _parse_scenario_variants(
    values: Sequence[str],
) -> tuple[tuple[int, int, Path, Path], ...]:
    parsed = []
    seen = set()
    for value in values:
        fields = value.split(":", 3)
        if len(fields) != 4:
            raise ValueError(
                "--scenario-variant must be "
                "EVAL_INDEX:BASE_INDEX:HISTORY_PATH:QA_PATH"
            )
        eval_index, base_index = int(fields[0]), int(fields[1])
        if eval_index <= 50 or base_index < 1 or base_index > 50:
            raise ValueError(
                "Variant EVAL_INDEX must be >50 and BASE_INDEX must be S1-S50"
            )
        if eval_index in seen:
            raise ValueError(f"Duplicate scenario variant index: {eval_index}")
        history_path = Path(fields[2]).expanduser().resolve(strict=True)
        qa_path = Path(fields[3]).expanduser().resolve(strict=True)
        if not history_path.is_file() or not qa_path.is_file():
            raise ValueError(f"Variant paths must be files: {value}")
        parsed.append((eval_index, base_index, history_path, qa_path))
        seen.add(eval_index)
    return tuple(parsed)


def _load_scenario_variants(
    specs: Sequence[tuple[int, int, Path, Path]], benchmark: Any
) -> dict[int, Any]:
    result = {}
    for eval_index, base_index, history_path, qa_path in specs:
        base = benchmark.scenario(base_index)
        base_payload = _read_json(base.qa_path)
        variant_payload = _read_json(qa_path)
        key = "related_to_vehicle_preference"
        base_rows = base_payload.get(key)
        variant_rows = variant_payload.get(key)
        if not isinstance(base_rows, list) or not isinstance(variant_rows, list):
            raise TypeError(f"Malformed V1 variant QA payload: {qa_path}")
        if len(variant_rows) != len(base.tasks) or len(base_rows) != len(base.tasks):
            raise ValueError(f"Variant task count differs from S{base_index}: {qa_path}")
        tasks = []
        for event_index, (task, source_row, row) in enumerate(
            zip(base.tasks, base_rows, variant_rows, strict=True)
        ):
            if row.get("reasoning_type") != task.reasoning_type:
                raise ValueError(
                    f"Variant reasoning type differs at item {event_index}: {qa_path}"
                )
            if row.get("new_answer") != source_row.get("new_answer"):
                raise ValueError(
                    f"Variant Tool/Arguments differ at item {event_index}: {qa_path}"
                )
            tasks.append(
                replace(
                    task,
                    id=f"v1-variant-s{eval_index:03d}-q{event_index + 1:02d}",
                    scenario_index=eval_index,
                    event_index=event_index,
                    query=str(row["query"]),
                    gold_memory=str(row["gold_memory"]),
                )
            )
        result[eval_index] = replace(
            base,
            index=eval_index,
            history_path=history_path,
            qa_path=qa_path,
            tasks=tuple(tasks),
        )
    return result


def _variant_dataset_sha256(
    base_sha256: str, specs: Sequence[tuple[int, int, Path, Path]]
) -> str:
    digest = hashlib.sha256(base_sha256.encode("ascii"))
    for eval_index, base_index, history_path, qa_path in specs:
        digest.update(f"{eval_index}:{base_index}".encode("ascii"))
        digest.update(history_path.read_bytes())
        digest.update(qa_path.read_bytes())
    return digest.hexdigest()


def _build_v1_quiz_rows(
    scenarios: Sequence[Any], tools_path: Path, tools_sha256: str
) -> list[dict[str, Any]]:
    payload = json.loads(tools_path.read_text(encoding="utf-8"))
    modules = payload.get("modules")
    if not isinstance(modules, dict):
        raise TypeError("vehicle_tools.json has no module mapping")
    module_tools = {
        str(module): [str(value) for value in values]
        for module, values in modules.items()
    }
    tool_modules = {
        tool: module for module, values in module_tools.items() for tool in values
    }
    rows = []
    for scenario in scenarios:
        for task in scenario.tasks:
            calls = [
                {"name": call.name, "arguments": dict(call.arguments)}
                for call in task.gold_calls
            ]
            sample_id = f"v1:s{scenario.index:03d}:quiz:{task.event_index + 1:02d}"
            candidate_names, candidate_modules = _candidate_tools(
                sample_id=sample_id,
                gold_names={call["name"] for call in calls},
                tool_modules=tool_modules,
                module_tools=module_tools,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "scenario_index": scenario.index,
                    "quiz_type": "FINAL",
                    "quiz_id": task.id,
                    "reasoning_type": task.reasoning_type,
                    "tool_schema_ref": {
                        "path": tools_path.name,
                        "sha256": tools_sha256,
                        "selection_policy": (
                            "gold_plus_same_module_plus_global_negative"
                        ),
                        "module_names": candidate_modules,
                        "tool_names": candidate_names,
                    },
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _user_message(task.gold_memory, task.query),
                        },
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": call["name"],
                                        "arguments": call["arguments"],
                                    },
                                }
                                for call in calls
                            ],
                        },
                    ],
                    "target": {"tool_calls": calls},
                }
            )
    return rows


def _replay_histories(
    *,
    model: Any,
    tokenizer: Any,
    method: Any,
    accelerator: Any,
    histories: Mapping[int, list[dict[str, str]]],
    scenarios: Sequence[int],
    scenario_dir: Path,
    signature: str,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    checkpoint_interval: int,
    progress_path: Path,
    started_at: str,
    work_offset: int,
    total_work: int,
) -> list[dict[str, Any]]:
    encoder = ChatExampleEncoder(tokenizer, max_length=max_length)
    streams = []
    reports: dict[int, dict[str, Any]] = {}
    for scenario in scenarios:
        summary_path = scenario_dir / f"s{scenario:03d}.json"
        if summary_path.is_file():
            report = _read_json(summary_path)
            if report.get("signature") != signature:
                raise ValueError(f"Stale V1 scenario result: {summary_path}")
            reports[scenario] = report
            continue
        checkpoint_path = scenario_dir / f"s{scenario:03d}.checkpoint.json"
        state = method.initial_state()
        position = 0
        counters: Counter[str] = Counter()
        usage = {"prefill_tokens": 0, "decode_tokens": 0, "latency_seconds": 0.0}
        failures: list[dict[str, Any]] = []
        if checkpoint_path.is_file():
            saved = _read_json(checkpoint_path)
            if saved.get("signature") != signature:
                raise ValueError(f"Stale V1 checkpoint: {checkpoint_path}")
            state = (
                state_from_input(method, saved["state"])
                if "state" in saved
                else str(saved["memory"])
            )
            position = int(saved["next_position"])
            counters.update(saved.get("decision_counts") or {})
            usage.update(saved.get("usage") or {})
            failures = list(saved.get("failures") or [])
        streams.append(
            {
                "scenario": scenario,
                "state": state,
                "position": position,
                "decision_counts": counters,
                "usage": usage,
                "failures": failures,
                "checkpoint_path": checkpoint_path,
                "summary_path": summary_path,
            }
        )

    restored_turns = sum(
        len(histories[scenario]) for scenario in reports
    ) + sum(int(stream["position"]) for stream in streams)
    pending = list(streams)
    active = pending[:batch_size]
    del pending[:batch_size]
    processed = restored_turns
    completed = sorted(reports)
    while active:
        rows = []
        for stream in active:
            scenario = int(stream["scenario"])
            position = int(stream["position"])
            turn = histories[scenario][position]
            rows.append(
                {
                    "turn_id": f"v1-s{scenario:03d}-turn-{position + 1:05d}",
                    "scenario_index": scenario,
                    "timestamp": turn["timestamp"],
                    "current_turn": {
                        "speaker_id": turn["speaker"],
                        "speaker_name": turn["speaker"],
                        "text": turn["text"],
                    },
                    "input": state_to_input(method, stream["state"]),
                }
            )
        generated = generate_outputs_batch(
            model,
            tokenizer,
            method,
            rows,
            encoder=encoder,
            accelerator=accelerator,
            max_new_tokens=max_new_tokens,
        )
        finished = []
        for stream, row, result in zip(active, rows, generated, strict=True):
            output, prefill, decode, latency = result
            decision = "INVALID"
            try:
                parsed = method.parse_output(output)
                stream["state"] = method.apply_output(
                    stream["state"], parsed, turn_id=str(row["turn_id"])
                )
                decision = str(parsed.decision)
            except Exception as exc:  # noqa: BLE001 - retain invalid generations.
                if len(stream["failures"]) < 50:
                    stream["failures"].append(
                        {
                            "position": int(stream["position"]),
                            "turn_id": row["turn_id"],
                            "error": f"{type(exc).__name__}: {exc}",
                            "output": output,
                        }
                    )
            stream["decision_counts"][decision] += 1
            stream["usage"]["prefill_tokens"] += prefill
            stream["usage"]["decode_tokens"] += decode
            stream["usage"]["latency_seconds"] += latency
            stream["position"] += 1
            processed += 1
            scenario = int(stream["scenario"])
            scenario_turns = len(histories[scenario])
            if (
                int(stream["position"]) % checkpoint_interval == 0
                or int(stream["position"]) == scenario_turns
            ):
                _save_stream(stream, method, signature)
            if int(stream["position"]) == scenario_turns:
                report = {
                    "schema_version": "palmclaw-hf-v1-scenario-v1",
                    "signature": signature,
                    "scenario_index": scenario,
                    "turns": scenario_turns,
                    "decision_counts": dict(stream["decision_counts"]),
                    "usage": dict(stream["usage"]),
                    "failures": list(stream["failures"]),
                    "final_memory": method.materialize_memory(stream["state"]),
                    "final_memory_sha256": _text_sha256(
                        method.materialize_memory(stream["state"])
                    ),
                }
                _write_json(stream["summary_path"], report)
                stream["checkpoint_path"].unlink(missing_ok=True)
                reports[scenario] = report
                completed = sorted(reports)
                finished.append(stream)
                print(
                    f"[{len(completed)}/{len(scenarios)}] S{scenario} memory complete",
                    flush=True,
                )
        for stream in finished:
            active.remove(stream)
        while pending and len(active) < batch_size:
            active.append(pending.pop(0))
        if processed % max(batch_size * 5, 1) < len(generated) or finished:
            _progress(
                progress_path,
                started_at=started_at,
                status="RUNNING",
                phase="closed_loop_memory",
                scenarios=scenarios,
                completed_scenarios=completed,
                item=work_offset + processed,
                items=total_work,
                phase_item=processed,
                phase_items=sum(len(value) for value in histories.values()),
                current_scenarios=[int(value["scenario"]) for value in active],
            )
    return [reports[value] for value in scenarios]


def _save_stream(stream: Mapping[str, Any], method: Any, signature: str) -> None:
    _write_json(
        stream["checkpoint_path"],
        {
            "schema_version": "palmclaw-hf-v1-checkpoint-v2",
            "signature": signature,
            "scenario_index": int(stream["scenario"]),
            "next_position": int(stream["position"]),
            "state": state_to_input(method, stream["state"]),
            "memory": method.materialize_memory(stream["state"]),
            "decision_counts": dict(stream["decision_counts"]),
            "usage": dict(stream["usage"]),
            "failures": list(stream["failures"]),
        },
    )


def _aggregate(
    *,
    scenarios: Sequence[int],
    signature: str,
    reports: Sequence[Mapping[str, Any]],
    gold_quiz: Mapping[str, Any],
    predicted_quiz: Mapping[str, Any],
) -> dict[str, Any]:
    decisions: Counter[str] = Counter()
    prefill = decode = 0
    latency = 0.0
    for report in reports:
        decisions.update(report.get("decision_counts") or {})
        usage = report.get("usage") or {}
        prefill += int(usage.get("prefill_tokens", 0))
        decode += int(usage.get("decode_tokens", 0))
        latency += float(usage.get("latency_seconds", 0.0))
    return {
        "schema_version": "palmclaw-hf-v1-test-summary-v1",
        "signature": signature,
        "expected_scenarios": list(scenarios),
        "completed_scenarios": list(scenarios),
        "complete": True,
        "evaluation_order": [
            "gold_memory_quiz",
            "closed_loop_memory",
            "predicted_final_memory_quiz",
        ],
        "memory": {
            "turns": sum(int(report["turns"]) for report in reports),
            "decision_counts": dict(decisions),
            "prefill_tokens": prefill,
            "decode_tokens": decode,
            "total_tokens": prefill + decode,
            "latency_seconds": latency,
            "scenario_reports": [
                {
                    key: report[key]
                    for key in (
                        "scenario_index",
                        "turns",
                        "decision_counts",
                        "usage",
                        "final_memory_sha256",
                    )
                }
                for report in reports
            ],
        },
        "gold_memory_quiz": dict(gold_quiz),
        "closed_loop_quiz": dict(predicted_quiz),
    }


def _prepare_manifest(
    path: Path,
    *,
    signature: str,
    args: argparse.Namespace,
    checkpoint: Path,
    scenarios: Sequence[int],
    tasks: int,
    turns: int,
    dataset_sha256: str,
) -> str:
    if path.is_file():
        value = _read_json(path)
        if value.get("signature") != signature:
            raise ValueError(f"Stale V1 output directory: {path.parent}")
        return str(value["created_at"])
    started_at = _utc_now()
    _write_json(
        path,
        {
            "schema_version": "palmclaw-hf-v1-test-manifest-v1",
            "signature": signature,
            "created_at": started_at,
            "model": args.model,
            "method": args.method,
            "checkpoint": str(checkpoint),
            "scenarios": list(scenarios),
            "tasks": tasks,
            "turns": turns,
            "dataset_sha256": dataset_sha256,
            "evaluation_order": [
                "gold_memory_quiz",
                "closed_loop_memory",
                "predicted_final_memory_quiz",
            ],
        },
    )
    return started_at


def _signature(
    args: argparse.Namespace,
    checkpoint: Path,
    scenarios: Sequence[int],
    dataset_sha256: str,
    tools_sha256: str,
) -> str:
    weights = next((checkpoint / "adapter").glob("adapter_model.*"), None)
    payload = {
        "schema": "palmclaw-hf-v1-test-v1",
        "model": args.model,
        "method": args.method,
        "checkpoint": str(checkpoint),
        "adapter_size": weights.stat().st_size if weights else None,
        "adapter_mtime_ns": weights.stat().st_mtime_ns if weights else None,
        "scenarios": list(scenarios),
        "dataset_sha256": dataset_sha256,
        "tools_sha256": tools_sha256,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "quiz_max_new_tokens": args.quiz_max_new_tokens,
        "scenario_batch_size": args.scenario_batch_size,
        "quiz_batch_size": args.quiz_batch_size,
    }
    return _text_sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _completed_scenarios(path: Path, signature: str) -> list[int]:
    completed = []
    for result_path in path.glob("s[0-9][0-9][0-9].json"):
        result = _read_json(result_path)
        if result.get("signature") != signature:
            raise ValueError(f"Stale V1 scenario result: {result_path}")
        completed.append(int(result_path.stem[1:]))
    return sorted(completed)


def _load_signed_result(path: Path, signature: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = _read_json(path)
    if result.get("signature") != signature:
        raise ValueError(f"Stale V1 evaluation result: {path}")
    result.pop("signature", None)
    return result


def _configure_runtime_paths(workspace: Path) -> None:
    paths = {
        "XDG_CACHE_HOME": workspace / "cache",
        "HF_HOME": workspace / "cache" / "huggingface",
        "HF_HUB_CACHE": workspace / "cache" / "huggingface" / "hub",
        "HF_XET_CACHE": workspace / "cache" / "huggingface" / "xet",
        "HF_DATASETS_CACHE": workspace / "cache" / "huggingface" / "datasets",
        "TORCH_HOME": workspace / "cache" / "torch",
        "TMPDIR": workspace / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def _progress(
    path: Path,
    *,
    started_at: str,
    status: str,
    phase: str,
    scenarios: Sequence[int],
    completed_scenarios: Sequence[int],
    item: int,
    items: int,
    phase_item: int,
    phase_items: int,
    current_scenarios: Sequence[int] = (),
    esm: float | None = None,
) -> None:
    elapsed = max(
        0.0,
        (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ).total_seconds(),
    )
    fraction = min(max(item / items if items else 0.0, 0.0), 1.0)
    _write_json(
        path,
        {
            "schema_version": "palmclaw-evaluation-progress-v1",
            "updated_at": _utc_now(),
            "started_at": started_at,
            "status": status,
            "job_type": "TEST_V1",
            "phase": phase,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expected_scenarios": list(scenarios),
            "completed_scenarios": list(completed_scenarios),
            "current_scenario": (
                int(current_scenarios[0]) if current_scenarios else None
            ),
            "current_scenarios": list(current_scenarios),
            "item": item,
            "items": items,
            "phase_item": phase_item,
            "phase_items": phase_items,
            "progress": fraction,
            "elapsed_seconds": elapsed,
            "eta_seconds": elapsed / fraction * (1 - fraction) if fraction else None,
            "esm": esm,
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
