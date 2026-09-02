"""Evaluate the on-device Memory/Quiz contract with an OpenAI Responses model.

The memory system prompts, compact JSON turn inputs, method parsers, deterministic
state transitions, Quiz prompts, candidate tools, and VehicleMemBench scorer are
shared with the HF closed-loop evaluator.  Only model generation is replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import IndexedMemoryDataset, default_catalog_path, ensure_catalog
from .methods import METHODS, DeltaV2Method, DeltaV2State, SummaryMethod
from .quiz_sft import (
    IndexedQuizSFTDataset,
    VehicleToolSchemaStore,
    sft_split_for_scenario,
)
from .ubuntu_bridge import enable_ubuntu_runtime
from .validation import (
    _aggregate_quiz_records,
    _argument_exact_match,
    _quiz_row_with_memory,
    closed_loop_quiz_snapshot_requests,
    memory_scores,
    quiz_indices_for_scenarios,
    row_with_runtime_state,
    state_from_input,
    state_to_input,
)

DEFAULT_DATA_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2"
)
DEFAULT_WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")
DEFAULT_VEHICLEMEMBENCH = Path("/home/hj153lee/VehicleMemBench")

VEHICLE_MEMORY_FILTER = (
    "Update memory only for explicit, durable in-vehicle preferences, settings, "
    "constraints, conditions, or corrections tied to a specific user. Return "
    "NO_OP for general vehicle discussion, work, plans, industry information, "
    "route commentary, assistant claims, tool output, and one-time requests "
    "without durable implications."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("summary", "patch", "delta_v3"), required=True)
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--vehicle-tools-path", type=Path)
    parser.add_argument("--quiz-sft-path", type=Path)
    parser.add_argument("--vehiclemembench-root", type=Path, default=DEFAULT_VEHICLEMEMBENCH)
    parser.add_argument("--memory-max-output-tokens", type=int, default=768)
    parser.add_argument("--quiz-max-output-tokens", type=int, default=256)
    parser.add_argument("--quiz-workers", type=int, default=8)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--summary-memory-token-budget", type=int)
    parser.add_argument("--vehicle-memory-filter", action="store_true")
    parser.add_argument("--semantic-compaction-every", type=int)
    parser.add_argument("--semantic-compaction-token-budget", type=int, default=600)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--limit-turns", type=int)
    parser.add_argument("--skip-quiz", action="store_true")
    parser.add_argument("--input-price-per-million", type=float, default=0.20)
    parser.add_argument("--cached-input-price-per-million", type=float, default=0.02)
    parser.add_argument("--output-price-per-million", type=float, default=1.20)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    details = getattr(usage, "input_tokens_details", None)
    if isinstance(details, Mapping):
        cached_tokens = int(details.get("cached_tokens", 0) or 0)
    else:
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": max(0, input_tokens - cached_tokens),
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _cost(usage: Mapping[str, int], args: argparse.Namespace) -> float:
    return (
        int(usage["uncached_input_tokens"]) * args.input_price_per_million
        + int(usage["cached_input_tokens"]) * args.cached_input_price_per_million
        + int(usage["output_tokens"]) * args.output_price_per_million
    ) / 1_000_000


def _merge_usage(*values: Mapping[str, int]) -> dict[str, int]:
    keys = (
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "total_tokens",
    )
    return {key: sum(int(value.get(key, 0)) for value in values) for key in keys}


def _response_create(client: Any, *, attempts: int = 4, **kwargs: Any) -> tuple[Any, float, int]:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            response = client.responses.create(**kwargs)
            status = str(getattr(response, "status", "completed") or "completed")
            # An incomplete response is a model-generation failure (usually the
            # same max-token truncation the HF evaluator sees), not a transport
            # failure. Return its partial output so the method parser records an
            # INVALID turn and closed-loop evaluation can continue.
            if status in {"completed", "in_progress", "incomplete"}:
                return response, time.monotonic() - started, attempt
            if status not in {"completed", "in_progress"}:
                raise RuntimeError(f"provider response status={status}")
        except BaseException as exc:  # SDK retries first; retain resumability after that.
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(30.0, 2.0**attempt))
    assert last_error is not None
    raise last_error


def _sum_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "total_tokens",
    )
    return {key: sum(int(row["usage"].get(key, 0)) for row in records) for key in keys}


def _decision_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        f'{row["gold_decision"]}->{row["predicted_decision"]}' for row in records
    )
    tp = counts["UPDATE->UPDATE"]
    fp = counts["NO_OP->UPDATE"]
    fn = counts["UPDATE->NO_OP"] + counts["UPDATE->INVALID"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "update_precision": precision,
        "update_recall": recall,
        "update_f1": f1,
        "decision_accuracy": sum(
            int(row["gold_decision"] == row["predicted_decision"]) for row in records
        )
        / len(records),
        "decision_counts": dict(sorted(counts.items())),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_quiz(args: argparse.Namespace, data_root: Path) -> tuple[Any, list[int], Any]:
    split = sft_split_for_scenario(args.scenario)
    source = IndexedQuizSFTDataset(args.quiz_sft_path or data_root / "quiz_sft.jsonl", split=split)
    indices = quiz_indices_for_scenarios(source, [args.scenario])
    if not indices:
        raise ValueError(f"No Quiz rows for scenario {args.scenario}")
    tools = VehicleToolSchemaStore(args.vehicle_tools_path or data_root / "vehicle_tools.json")
    return source, indices, tools


def _semantic_compaction(
    args: argparse.Namespace,
    client: Any,
    memory: str,
) -> tuple[str | None, dict[str, Any]]:
    """Rewrite a materialized Patch/Delta state with the Summary contract."""
    summary_method = SummaryMethod()
    system_prompt = (
        summary_method.system_prompt
        + "\n"
        + VEHICLE_MEMORY_FILTER
        + "\nKeep next_memory under "
        + f"{args.semantic_compaction_token_budget} tokens. Retain only durable, "
        "vehicle-relevant facts while preserving all still-valid users, values, "
        "conditions, and temporal scopes. This is a mandatory maintenance rewrite "
        "of existing memory only: return decision=UPDATE with the complete compacted "
        "next_memory, and do not infer or add any new facts."
    )
    payload = json.dumps(
        {
            "memory_state": {"previous_memory": memory},
            "current_turn": {
                "turn_id": "semantic_compaction",
                "timestamp": None,
                "speaker_id": "system",
                "speaker_name": "system",
                "text": "Maintenance compaction only; no new conversation facts.",
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    response, elapsed, attempts = _response_create(
        client,
        model=args.model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
        max_output_tokens=max(768, args.semantic_compaction_token_budget + 128),
        reasoning={"effort": args.reasoning_effort},
        store=False,
    )
    generated = str(getattr(response, "output_text", "") or "").strip()
    usage = _usage(response)
    result = {
        "attempted": True,
        "success": False,
        "output": generated,
        "error": None,
        "usage": usage,
        "cost_usd": _cost(usage, args),
        "latency_seconds": elapsed,
        "transport_attempts": attempts,
        "response_id": getattr(response, "id", None),
        "memory_chars_before": len(memory),
        "memory_chars_after": len(memory),
    }
    try:
        parsed = summary_method.parse_output(generated)
        if parsed.decision != "UPDATE":
            raise ValueError("semantic compaction must return UPDATE")
        compacted = summary_method.apply_output(memory, parsed)
        result.update({"success": True, "memory_chars_after": len(compacted)})
        return compacted, result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return None, result


def _memory_phase(
    args: argparse.Namespace,
    client: Any,
    catalog: Any,
    quiz_source: Any,
    quiz_indices: Sequence[int],
) -> tuple[dict[str, Any], dict[str, str]]:
    method = METHODS[args.method]()
    system_prompt = method.system_prompt
    if args.method == "summary" and args.summary_memory_token_budget is not None:
        if args.summary_memory_token_budget < 1:
            raise ValueError("summary-memory-token-budget must be positive")
        system_prompt += (
            "\nKeep next_memory under "
            f"{args.summary_memory_token_budget} tokens. Retain only durable, "
            "vehicle-relevant facts while preserving all still-valid users, "
            "values, conditions, and temporal scopes."
        )
    if args.vehicle_memory_filter:
        if args.method not in {"summary", "patch", "delta_v3"}:
            raise ValueError("vehicle-memory-filter supports the smoke-test methods")
        system_prompt += "\n" + VEHICLE_MEMORY_FILTER
    row_ids = catalog.scenario_row_ids(args.scenario)
    if args.limit_turns is not None:
        row_ids = row_ids[: args.limit_turns]
    source = IndexedMemoryDataset(catalog, method.source_view, row_ids=row_ids)
    gold = IndexedMemoryDataset(catalog, "summary", row_ids=row_ids)
    snapshot_requests = closed_loop_quiz_snapshot_requests(quiz_source, quiz_indices)
    steps_dir = args.output_dir / "memory_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    state = method.initial_state()
    records: list[dict[str, Any]] = []
    snapshots: dict[str, str] = {}
    updates_since_semantic_compaction = 0
    started = time.monotonic()
    for position, row_id in enumerate(row_ids):
        canonical = source[position]
        gold_row = gold[position]
        turn = int(canonical["global_turn_index"])
        step_path = steps_dir / f"{turn:05d}.json"
        if step_path.is_file():
            record = json.loads(step_path.read_text(encoding="utf-8"))
            state = state_from_input(method, record["state_after"])
            updates_since_semantic_compaction = int(
                record.get("updates_since_semantic_compaction", 0)
            )
        else:
            row = row_with_runtime_state(canonical, method, state)
            response, elapsed, attempts = _response_create(
                client,
                model=args.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": method.format_input(row)},
                ],
                max_output_tokens=args.memory_max_output_tokens,
                reasoning={"effort": args.reasoning_effort},
                store=False,
            )
            generated = str(getattr(response, "output_text", "") or "").strip()
            predicted = "INVALID"
            error = None
            applied_update = False
            try:
                parsed = method.parse_output(generated)
                predicted = parsed.decision
                state = method.apply_output(
                    state, parsed, turn_id=str(canonical.get("turn_id", ""))
                )
                applied_update = predicted == "UPDATE"
            except Exception as exc:  # Invalid output leaves memory unchanged.
                error = f"{type(exc).__name__}: {exc}"
            response_usage = _usage(response)
            semantic_compaction: dict[str, Any] = {"attempted": False}
            if applied_update and args.semantic_compaction_every is not None:
                updates_since_semantic_compaction += 1
                if updates_since_semantic_compaction >= args.semantic_compaction_every:
                    current_memory = method.materialize_memory(state)
                    compacted, semantic_compaction = _semantic_compaction(
                        args, client, current_memory
                    )
                    # Avoid retrying a malformed maintenance response on every
                    # subsequent UPDATE; the next five updates trigger a new try.
                    updates_since_semantic_compaction = 0
                    if compacted is not None:
                        if isinstance(method, DeltaV2Method):
                            state = DeltaV2State(base_summary=compacted)
                        else:
                            state = compacted
            predicted_memory = method.materialize_memory(state)
            score = memory_scores(
                predicted_memory, str(gold_row["target"].get("next_memory", ""))
            )
            total_usage = _merge_usage(
                response_usage,
                semantic_compaction.get("usage", {}),
            )
            record = {
                "row_id": row_id,
                "global_turn_index": turn,
                "turn_id": canonical.get("turn_id"),
                "gold_decision": str(canonical["target"]["decision"]),
                "predicted_decision": predicted,
                "output": generated,
                "error": error,
                "state_after": state_to_input(method, state),
                "state_score": score,
                "usage": total_usage,
                "cost_usd": _cost(response_usage, args)
                + float(semantic_compaction.get("cost_usd", 0.0)),
                "latency_seconds": elapsed
                + float(semantic_compaction.get("latency_seconds", 0.0)),
                "transport_attempts": attempts,
                "api_calls": 1 + int(bool(semantic_compaction.get("attempted"))),
                "response_id": getattr(response, "id", None),
                "semantic_compaction": semantic_compaction,
                "updates_since_semantic_compaction": updates_since_semantic_compaction,
            }
            _atomic_json(step_path, record)
        memory = method.materialize_memory(state)
        for sample_id in snapshot_requests.get(args.scenario, {}).get(turn, ()):
            snapshots[str(sample_id)] = memory
        records.append(record)
        _atomic_json(
            args.output_dir / "progress.json",
            {
                "status": "RUNNING",
                "phase": "memory",
                "method": args.method,
                "scenario": args.scenario,
                "item": position + 1,
                "items": len(row_ids),
                "progress": (position + 1) / len(row_ids),
                "elapsed_seconds": time.monotonic() - started,
                "updated_at": _utc_now(),
            },
        )
    usage = _sum_usage(records)
    final = records[-1]["state_score"] if records else {"exact": 0.0, "f1": 0.0}
    result = {
        **_decision_metrics(records),
        "turns": len(records),
        "state_exact": _mean([float(row["state_score"]["exact"]) for row in records]),
        "state_f1": _mean([float(row["state_score"]["f1"]) for row in records]),
        "final_state_exact": float(final["exact"]),
        "final_state_f1": float(final["f1"]),
        "invalid_outputs": sum(row["predicted_decision"] == "INVALID" for row in records),
        "semantic_compaction_attempts": sum(
            bool(row.get("semantic_compaction", {}).get("attempted")) for row in records
        ),
        "semantic_compaction_successes": sum(
            bool(row.get("semantic_compaction", {}).get("success")) for row in records
        ),
        "api_calls": sum(int(row.get("api_calls", 1)) for row in records),
        "usage": usage,
        "cost_usd": sum(float(row.get("cost_usd", 0.0)) for row in records),
        "latency_seconds": sum(float(row["latency_seconds"]) for row in records),
        "records_path": str(steps_dir),
    }
    return result, snapshots


def _flatten_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for tool in tools:
        function = tool["function"]
        flattened.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"],
                # VehicleMemBench has optional tool arguments. OpenAI strict
                # schemas require every property to be listed in required, so
                # preserve the benchmark schema with non-strict tool parsing.
                "strict": False,
            }
        )
    return flattened


def _quiz_one(
    args: argparse.Namespace,
    client: Any,
    row: dict[str, Any],
    candidate_tools: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    response, elapsed, attempts = _response_create(
        client,
        model=args.model,
        input=[dict(message) for message in row["messages"][:-1]],
        tools=_flatten_tools(candidate_tools),
        max_output_tokens=args.quiz_max_output_tokens,
        reasoning={"effort": args.reasoning_effort},
        store=False,
    )
    calls = []
    for item in getattr(response, "output", ()) or ():
        if getattr(item, "type", None) != "function_call":
            continue
        raw = str(getattr(item, "arguments", "{}") or "{}")
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            arguments = {"_raw": raw}
        calls.append({"name": str(item.name), "arguments": arguments})
    response_usage = _usage(response)
    return {
        "output": str(getattr(response, "output_text", "") or ""),
        "predicted_calls": calls,
        "usage": response_usage,
        "cost_usd": _cost(response_usage, args),
        "latency_seconds": elapsed,
        "transport_attempts": attempts,
        "response_id": getattr(response, "id", None),
    }


def _score_quiz_record(args: argparse.Namespace, row: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.vehicle_bench.dataset import GoldToolCall
    from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime

    runtime = VehicleWorldRuntime(args.vehiclemembench_root)
    predicted_calls = tuple(
        GoldToolCall(name=call["name"], arguments=call["arguments"], source="cloud")
        for call in generated["predicted_calls"]
    )
    reference_calls = tuple(
        GoldToolCall(name=call["name"], arguments=call["arguments"], source="gold")
        for call in row["target"]["tool_calls"]
    )
    reference_world = runtime.create_world()
    predicted_world = runtime.create_world()
    initial_state = runtime.state(reference_world)
    reference_errors = []
    for call in reference_calls:
        try:
            runtime.execute(reference_world, call.name, call.arguments)
        except Exception as exc:
            reference_errors.append(str(exc))
    execution_errors = []
    for call in predicted_calls:
        try:
            result = runtime.execute(predicted_world, call.name, call.arguments)
            if isinstance(result, Mapping) and result.get("success") is False:
                execution_errors.append(f"{call.name}: {result}")
        except Exception as exc:
            execution_errors.append(str(exc))
    score = runtime.score(
        initial_state=initial_state,
        reference_state=runtime.state(reference_world),
        predicted_state=runtime.state(predicted_world),
        predicted_calls=predicted_calls,
        reference_calls=reference_calls,
    )
    return {
        "sample_id": row["sample_id"],
        "scenario_index": row["scenario_index"],
        "quiz_type": row["quiz_type"],
        "reasoning_type": row["reasoning_type"],
        "exact_state_match": float(score.exact_state_match),
        "tool_f1": float(score.tool_score.get("f1", 0.0)),
        "arg_exact": float(_argument_exact_match(predicted_calls, reference_calls)),
        "parse_success": float(bool(predicted_calls)),
        "execution_success": float(not execution_errors),
        "reference_execution_errors": reference_errors,
        "execution_errors": execution_errors,
        "memory_mode": "closed_loop_predicted",
        **generated,
    }


def _quiz_phase(
    args: argparse.Namespace,
    client: Any,
    source: Any,
    indices: Sequence[int],
    tools: Any,
    snapshots: Mapping[str, str],
) -> dict[str, Any]:
    quiz_dir = args.output_dir / "quiz_records"
    quiz_dir.mkdir(parents=True, exist_ok=True)
    records_by_id: dict[str, dict[str, Any]] = {}
    pending = []
    for index in indices:
        row = source[index]
        sample_id = str(row["sample_id"])
        path = quiz_dir / f"{sample_id}.json"
        if path.is_file():
            records_by_id[sample_id] = json.loads(path.read_text(encoding="utf-8"))
            continue
        if sample_id not in snapshots:
            raise ValueError(f"No closed-loop memory snapshot for {sample_id}")
        updated = _quiz_row_with_memory(row, snapshots[sample_id])
        pending.append((path, updated, tools.tools_for(updated)))
    started = time.monotonic()
    completed = len(records_by_id)
    with ThreadPoolExecutor(max_workers=args.quiz_workers) as executor:
        futures = {
            executor.submit(_quiz_one, args, client, row, candidates): (path, row)
            for path, row, candidates in pending
        }
        for future in as_completed(futures):
            path, row = futures[future]
            record = _score_quiz_record(args, row, future.result())
            _atomic_json(path, record)
            records_by_id[str(row["sample_id"])] = record
            completed += 1
            _atomic_json(
                args.output_dir / "progress.json",
                {
                    "status": "RUNNING",
                    "phase": "quiz",
                    "method": args.method,
                    "scenario": args.scenario,
                    "item": completed,
                    "items": len(indices),
                    "progress": completed / len(indices),
                    "elapsed_seconds": time.monotonic() - started,
                    "updated_at": _utc_now(),
                },
            )
    records = [records_by_id[str(source[index]["sample_id"])] for index in indices]
    result = _aggregate_quiz_records(records)
    usage = _sum_usage(records)
    result.update(
        {
            "usage": usage,
            "cost_usd": sum(float(row["cost_usd"]) for row in records),
            "latency_seconds": sum(float(row["latency_seconds"]) for row in records),
            "records": records,
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    if args.quiz_workers < 1:
        raise ValueError("quiz-workers must be positive")
    if args.semantic_compaction_every is not None:
        if args.method not in {"patch", "delta_v3"}:
            raise ValueError("semantic compaction supports patch and delta_v3 only")
        if args.semantic_compaction_every < 1:
            raise ValueError("semantic-compaction-every must be positive")
        if args.semantic_compaction_token_budget < 1:
            raise ValueError("semantic-compaction-token-budget must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()
    catalog_path = (args.catalog_path or default_catalog_path(args.workspace)).resolve()
    catalog = ensure_catalog(data_root, catalog_path)
    if not catalog.scenario_row_ids(args.scenario):
        raise ValueError(f"Memory rows missing for scenario {args.scenario}")
    quiz_source, quiz_indices, tools = _load_quiz(args, data_root)
    from openai import OpenAI

    client = OpenAI(timeout=args.timeout_seconds, max_retries=5)
    manifest = {
        "schema_version": "palmclaw-cloud-ondevice-prompt-closed-loop-v1",
        "model": args.model,
        "method": args.method,
        "scenario": args.scenario,
        "scenario_label": f"T{args.scenario - 100}" if args.scenario > 100 else f"S{args.scenario}",
        "data_root": str(data_root),
        "prompt_source": f"memory_training.methods.{args.method}",
        "summary_memory_token_budget": args.summary_memory_token_budget,
        "vehicle_memory_filter": args.vehicle_memory_filter,
        "semantic_compaction_every": args.semantic_compaction_every,
        "semantic_compaction_token_budget": args.semantic_compaction_token_budget,
        "memory_mode": "closed_loop_predicted_only",
        "pricing_usd_per_million": {
            "input": args.input_price_per_million,
            "cached_input": args.cached_input_price_per_million,
            "output": args.output_price_per_million,
        },
        "created_at": _utc_now(),
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    memory, snapshots = _memory_phase(args, client, catalog, quiz_source, quiz_indices)
    quiz = None if args.skip_quiz or args.limit_turns is not None else _quiz_phase(
        args, client, quiz_source, quiz_indices, tools, snapshots
    )
    total_cost = memory["cost_usd"] + (quiz["cost_usd"] if quiz else 0.0)
    total_latency = memory["latency_seconds"] + (quiz["latency_seconds"] if quiz else 0.0)
    summary = {
        **manifest,
        "completed_at": _utc_now(),
        "complete": quiz is not None or args.skip_quiz or args.limit_turns is not None,
        "memory": memory,
        "closed_loop_quiz": quiz,
        "total": {
            "logical_calls": memory["api_calls"] + (quiz["tasks"] if quiz else 0),
            "cost_usd": total_cost,
            "latency_seconds": total_latency,
        },
    }
    if quiz:
        summary["total"]["usage"] = {
            key: memory["usage"][key] + quiz["usage"][key] for key in memory["usage"]
        }
    else:
        summary["total"]["usage"] = memory["usage"]
    _atomic_json(args.output_dir / "summary.json", summary)
    _atomic_json(
        args.output_dir / "progress.json",
        {
            "status": "COMPLETED",
            "phase": "complete",
            "method": args.method,
            "scenario": args.scenario,
            "item": 1,
            "items": 1,
            "progress": 1.0,
            "esm": quiz["esm"] if quiz else None,
            "update_f1": memory["update_f1"],
            "updated_at": _utc_now(),
        },
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
