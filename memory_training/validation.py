"""Teacher-forced, one-step, and sequential closed-loop validation."""

from __future__ import annotations

import copy
import json
import math
import random
import re
import shlex
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import DatasetCatalog, IndexedMemoryDataset
from .methods import DeltaMethod, DeltaState, DeltaV2Method, MemoryMethod
from .methods.delta import DeltaBatch
from .methods.delta_v2 import delta_v2_input_from_state, delta_v2_state_from_input
from .methods.operations import normalize_memory, validate_operations
from .quiz_sft import IndexedQuizSFTDataset, VehicleToolSchemaStore
from .training_data import ChatExampleEncoder
from .ubuntu_bridge import enable_ubuntu_runtime


def evaluate_teacher_forced_loss(
    model: Any,
    loader: DataLoader,
    *,
    accelerator: Any,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    token_count = torch.tensor(0, device=accelerator.device)
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and batch_count >= max_batches:
                break
            metadata = batch.pop("metadata", None)
            del metadata
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            output = model(**batch)
            tokens = (batch["labels"] != -100).sum()
            loss_sum += output.loss.detach() * tokens
            token_count += tokens
            batch_count += 1
    gathered_loss = accelerator.reduce(loss_sum, reduction="sum")
    gathered_tokens = accelerator.reduce(token_count, reduction="sum")
    mean_loss = float((gathered_loss / gathered_tokens.clamp_min(1)).cpu())
    if was_training:
        model.train()
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "target_tokens": int(gathered_tokens.cpu()),
        "batches": batch_count,
    }


def evaluate_one_step(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    source: IndexedMemoryDataset,
    gold_summary: IndexedMemoryDataset,
    row_ids: Sequence[int],
    *,
    accelerator: Any,
    max_length: int,
    max_new_tokens: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    if len(source) != len(gold_summary):
        raise ValueError("Source and gold Summary views are not aligned")
    if batch_size < 1:
        raise ValueError("One-step batch_size must be positive")
    encoder = ChatExampleEncoder(tokenizer, max_length=max_length)
    counts = _DecisionCounts()
    decision_usage = _DecisionUsage()
    state_scores = []
    parse_success = 0
    apply_success = 0
    prefill_tokens = 0
    generated_tokens = 0
    latency = 0.0
    reason_correct = 0
    reason_total = 0
    failures = []
    for batch_start in range(0, len(row_ids), batch_size):
        batch_row_ids = row_ids[batch_start : batch_start + batch_size]
        prepared = []
        for row_id in batch_row_ids:
            position = source.position_for_row_id(row_id)
            row = _single(source, position)
            prepared.append(
                (
                    row_id,
                    row,
                    _single(gold_summary, position),
                    state_from_input(method, row["input"]),
                )
            )
        generated_batch = generate_outputs_batch(
            model,
            tokenizer,
            method,
            [row for _, row, _, _ in prepared],
            encoder=encoder,
            accelerator=accelerator,
            max_new_tokens=max_new_tokens,
        )
        for (row_id, row, gold_row, state), generated_result in zip(
            prepared, generated_batch, strict=True
        ):
            generated, prompt_count, token_count, elapsed = generated_result
            prefill_tokens += prompt_count
            generated_tokens += token_count
            latency += elapsed
            gold_decision = str(row["target"]["decision"])
            predicted_decision = "INVALID"
            try:
                parsed = method.parse_output(generated)
                predicted_decision = str(parsed.decision)
                parse_success += 1
                if parsed.reason_code is not None:
                    reason_total += 1
                    reason_correct += int(
                        parsed.reason_code == row["target"].get("reason_code")
                    )
                next_state = method.apply_output(
                    state, parsed, turn_id=str(row.get("turn_id", ""))
                )
                apply_success += 1
                predicted_memory = method.materialize_memory(next_state)
                counts.add(gold_decision, predicted_decision)
            except Exception as exc:  # noqa: BLE001 - captures model failures.
                predicted_decision = "INVALID"
                counts.add(gold_decision, "INVALID")
                predicted_memory = method.materialize_memory(state)
                if len(failures) < 20:
                    failures.append(
                        {"row_id": row_id, "error": str(exc), "output": generated}
                    )
            decision_usage.add(
                gold_decision,
                predicted_decision,
                prefill_tokens=prompt_count,
                decode_tokens=token_count,
                attributed_latency_seconds=elapsed,
            )
            gold_memory = str(gold_row["target"].get("next_memory", ""))
            state_scores.append(memory_scores(predicted_memory, gold_memory))
    return {
        **counts.metrics(),
        "rows": len(row_ids),
        "schema_success_rate": parse_success / len(row_ids) if row_ids else 0.0,
        "apply_success_rate": apply_success / len(row_ids) if row_ids else 0.0,
        "state_exact": _mean(score["exact"] for score in state_scores),
        "state_f1": _mean(score["f1"] for score in state_scores),
        "reason_code_accuracy": reason_correct / reason_total if reason_total else None,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": generated_tokens,
        "total_tokens": prefill_tokens + generated_tokens,
        "latency_seconds": latency,
        "tokens_per_second": generated_tokens / latency if latency else 0.0,
        **decision_usage.as_dict(),
        "failures": failures,
    }


def aggregate_closed_loop_results(
    results: Sequence[Mapping[str, Any]],
    *,
    split: str = "validation",
) -> dict[str, Any]:
    """Merge metrics from independently evaluated closed-loop scenario groups."""
    counts: defaultdict[str, int] = defaultdict(int)
    reports = []
    turns = prefill = decode = original_turns = evaluated_turns = 0
    state_exact_sum = state_f1_sum = latency = 0.0
    full_scenarios = []
    sparse_scenarios = []
    sparse_fraction = 0.0
    for result in results:
        for key, value in result.get("decision_counts", {}).items():
            counts[key] += int(value)
        result_turns = int(result.get("turns", 0))
        turns += result_turns
        state_exact_sum += float(result.get("state_exact", 0.0)) * result_turns
        state_f1_sum += float(result.get("state_f1", 0.0)) * result_turns
        prefill += int(result.get("prefill_tokens", 0))
        decode += int(result.get("decode_tokens", 0))
        latency += float(result.get("latency_seconds", 0.0))
        reports.extend(result.get("scenario_reports", []))
        sampling = result.get("sampling", {})
        full_scenarios.extend(sampling.get("full_scenarios", []))
        sparse_scenarios.extend(sampling.get("sparse_scenarios", []))
        sparse_fraction = max(
            sparse_fraction,
            float(sampling.get("sparse_noop_keep_fraction", 0.0)),
        )
        original_turns += int(sampling.get("original_turns", 0))
        evaluated_turns += int(sampling.get("evaluated_turns", 0))
    tp, fp = counts["true_update"], counts["false_update"]
    fn, tn = counts["missed_update"], counts["true_noop"]
    invalid_noop = counts["invalid_noop"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "update_precision": precision,
        "update_recall": recall,
        "update_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "false_update_rate": (
            fp / (tn + fp + invalid_noop) if tn + fp + invalid_noop else 0.0
        ),
        "noop_specificity": (
            tn / (tn + fp + invalid_noop) if tn + fp + invalid_noop else 0.0
        ),
        "invalid_count": counts["invalid"],
        "decision_counts": dict(counts),
        "split": split,
        "scenarios": len(reports),
        "turns": turns,
        "state_exact": state_exact_sum / turns if turns else 0.0,
        "state_f1": state_f1_sum / turns if turns else 0.0,
        "final_state_exact": _mean(
            float(report.get("final_state_exact", 0.0)) for report in reports
        ),
        "final_state_f1": _mean(
            float(report.get("final_state_f1", 0.0)) for report in reports
        ),
        "prefill_tokens": prefill,
        "decode_tokens": decode,
        "total_tokens": prefill + decode,
        "latency_seconds": latency,
        **merge_decision_usage(results),
        "scenario_reports": reports,
        "sampling": {
            "full_scenarios": full_scenarios,
            "sparse_scenarios": sparse_scenarios,
            "sparse_noop_keep_fraction": sparse_fraction,
            "original_turns": original_turns,
            "evaluated_turns": evaluated_turns,
        },
    }


def evaluate_closed_loop(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    catalog: DatasetCatalog,
    *,
    accelerator: Any,
    split: str = "validation",
    scenario_min: int | None = None,
    scenario_max: int | None = None,
    scenario_limit: int | None = None,
    turn_limit_per_scenario: int | None = None,
    full_scenarios: Sequence[int] | None = None,
    sparse_scenarios: Sequence[int] | None = None,
    sparse_noop_keep_fraction: float = 0.10,
    sparse_seed: int = 42,
    snapshot_requests: Mapping[int, Mapping[int, Sequence[str]]] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval: int = 25,
    max_length: int = 4096,
    max_new_tokens: int = 768,
) -> dict[str, Any]:
    available_scenarios = catalog.scenarios(split=split)
    requested = [*(full_scenarios or ()), *(sparse_scenarios or ())]
    if requested:
        invalid = sorted(set(requested) - set(available_scenarios))
        if invalid:
            raise ValueError(f"Scenarios are outside the {split} split: {invalid}")
        scenarios = requested
    else:
        scenarios = [
            scenario
            for scenario in available_scenarios
            if (scenario_min is None or scenario >= scenario_min)
            and (scenario_max is None or scenario <= scenario_max)
        ]
    if scenario_limit is not None:
        scenarios = scenarios[:scenario_limit]
    source = IndexedMemoryDataset(catalog, method.source_view, split=split)
    gold_summary = IndexedMemoryDataset(catalog, "summary", split=split)
    encoder = ChatExampleEncoder(tokenizer, max_length=max_length)
    counts = _DecisionCounts()
    decision_usage = _DecisionUsage()
    all_scores = []
    scenario_reports = []
    total_tokens = 0
    total_prefill_tokens = 0
    total_latency = 0.0
    quiz_snapshots: dict[str, str] = {}
    for scenario_position, scenario in enumerate(scenarios, start=1):
        row_ids = catalog.scenario_row_ids(scenario)
        row_ids = [row_id for row_id in row_ids if source.contains_row_id(row_id)]
        if turn_limit_per_scenario is not None:
            row_ids = row_ids[:turn_limit_per_scenario]
        original_turns = len(row_ids)
        mode = "sparse" if scenario in set(sparse_scenarios or ()) else "full"
        if mode == "sparse":
            row_ids = sparse_closed_loop_row_ids(
                source,
                row_ids,
                noop_keep_fraction=sparse_noop_keep_fraction,
                seed=sparse_seed + scenario,
            )
        state = method.initial_state()
        first_error = None
        recovery_count = 0
        was_wrong = False
        scenario_scores = []
        failures = 0
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "closed_loop_memory",
                    "scenario_index": scenario,
                    "scenario_position": scenario_position,
                    "scenario_total": len(scenarios),
                    "item": 0,
                    "items": len(row_ids),
                }
            )
        for sequence_index, row_id in enumerate(row_ids):
            position = source.position_for_row_id(row_id)
            canonical = _single(source, position)
            row = row_with_runtime_state(canonical, method, state)
            gold = _single(gold_summary, position)
            gold_decision = str(canonical["target"]["decision"])
            generated, prompt_tokens, generated_tokens, elapsed = generate_output(
                model,
                tokenizer,
                method,
                row,
                encoder=encoder,
                accelerator=accelerator,
                max_new_tokens=max_new_tokens,
            )
            total_prefill_tokens += prompt_tokens
            total_tokens += generated_tokens
            total_latency += elapsed
            predicted_decision = "INVALID"
            try:
                parsed = method.parse_output(generated)
                predicted_decision = str(parsed.decision)
                state = method.apply_output(
                    state, parsed, turn_id=str(row.get("turn_id", ""))
                )
                counts.add(gold_decision, predicted_decision)
            except Exception:  # noqa: BLE001 - invalid generation keeps prior state.
                predicted_decision = "INVALID"
                counts.add(gold_decision, "INVALID")
                failures += 1
            decision_usage.add(
                gold_decision,
                predicted_decision,
                prefill_tokens=prompt_tokens,
                decode_tokens=generated_tokens,
                attributed_latency_seconds=elapsed,
            )
            score = memory_scores(
                method.materialize_memory(state),
                str(gold["target"].get("next_memory", "")),
            )
            global_turn = int(canonical["global_turn_index"])
            for sample_id in (
                (snapshot_requests or {}).get(scenario, {}).get(global_turn, ())
            ):
                quiz_snapshots[str(sample_id)] = method.materialize_memory(state)
            scenario_scores.append(score)
            all_scores.append(score)
            wrong = not bool(score["exact"])
            if wrong and first_error is None:
                first_error = sequence_index
            if was_wrong and not wrong:
                recovery_count += 1
            was_wrong = wrong
            if progress_callback is not None and (
                (sequence_index + 1) % progress_interval == 0
                or sequence_index + 1 == len(row_ids)
            ):
                progress_callback(
                    {
                        "phase": "closed_loop_memory",
                        "scenario_index": scenario,
                        "scenario_position": scenario_position,
                        "scenario_total": len(scenarios),
                        "item": sequence_index + 1,
                        "items": len(row_ids),
                    }
                )
        final = scenario_scores[-1] if scenario_scores else {"exact": 0.0, "f1": 0.0}
        scenario_reports.append(
            {
                "scenario_index": scenario,
                "sampling_mode": mode,
                "original_turns": original_turns,
                "turns": len(row_ids),
                "state_exact": _mean(score["exact"] for score in scenario_scores),
                "state_f1": _mean(score["f1"] for score in scenario_scores),
                "final_state_exact": final["exact"],
                "final_state_f1": final["f1"],
                "first_error_turn": first_error,
                "recovery_count": recovery_count,
                "invalid_outputs": failures,
            }
        )
    result = {
        **counts.metrics(),
        "decision_counts": counts.raw_counts(),
        "split": split,
        "scenarios": len(scenario_reports),
        "turns": len(all_scores),
        "state_exact": _mean(score["exact"] for score in all_scores),
        "state_f1": _mean(score["f1"] for score in all_scores),
        "final_state_exact": _mean(
            report["final_state_exact"] for report in scenario_reports
        ),
        "final_state_f1": _mean(
            report["final_state_f1"] for report in scenario_reports
        ),
        "prefill_tokens": total_prefill_tokens,
        "decode_tokens": total_tokens,
        "total_tokens": total_prefill_tokens + total_tokens,
        "latency_seconds": total_latency,
        **decision_usage.as_dict(),
        "scenario_reports": scenario_reports,
        "sampling": {
            "full_scenarios": list(full_scenarios or ()),
            "sparse_scenarios": list(sparse_scenarios or ()),
            "sparse_noop_keep_fraction": sparse_noop_keep_fraction,
            "original_turns": sum(
                report["original_turns"] for report in scenario_reports
            ),
            "evaluated_turns": sum(report["turns"] for report in scenario_reports),
        },
    }
    if snapshot_requests is not None:
        requested = {
            str(sample_id)
            for by_turn in snapshot_requests.values()
            for sample_ids in by_turn.values()
            for sample_id in sample_ids
        }
        missing = sorted(requested - set(quiz_snapshots))
        if missing:
            raise ValueError(
                f"Closed-loop Validation missed {len(missing)} Quiz snapshots"
            )
        result["quiz_snapshots"] = quiz_snapshots
    return result


@dataclass
class _BatchedClosedLoopStream:
    scenario: int
    row_ids: list[int]
    original_turns: int
    sampled_updates: int
    sampled_noops: int
    state: Any
    counts: Any
    decision_usage: Any
    next_index: int = 0
    scores: list[dict[str, float]] = field(default_factory=list)
    snapshots: dict[str, str] = field(default_factory=dict)
    first_error: int | None = None
    recovery_count: int = 0
    was_wrong: bool = False
    failures: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    latency_seconds: float = 0.0


def evaluate_closed_loop_batched(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    catalog: DatasetCatalog,
    scenarios: Sequence[int],
    *,
    accelerator: Any,
    split: str = "test",
    batch_size: int = 8,
    noop_per_update: float | None = None,
    sampling_seed: int = 42,
    snapshot_requests: Mapping[int, Mapping[int, Sequence[str]]] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval: int = 100,
    max_length: int = 4096,
    max_new_tokens: int = 768,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict[int, dict[str, Any]]:
    """Evaluate several dependent trajectories through one cross-scenario batch.

    Each scenario remains strictly sequential. Only the next turn of independent
    scenarios is grouped into the same ``generate`` call.
    """
    if batch_size < 1:
        raise ValueError("Closed-loop batch_size must be positive")
    if noop_per_update is not None and noop_per_update < 0:
        raise ValueError("Closed-loop noop_per_update must be non-negative")
    selected = list(dict.fromkeys(int(value) for value in scenarios))
    if not selected:
        return {}
    available = set(catalog.scenarios(split=split))
    invalid = sorted(set(selected) - available)
    if invalid:
        raise ValueError(f"Scenarios are outside the {split} split: {invalid}")
    source = IndexedMemoryDataset(catalog, method.source_view, split=split)
    gold_summary = IndexedMemoryDataset(catalog, "summary", split=split)
    encoder = ChatExampleEncoder(tokenizer, max_length=max_length)
    streams = []
    for scenario in selected:
        original_row_ids = [
            row_id
            for row_id in catalog.scenario_row_ids(scenario)
            if source.contains_row_id(row_id)
        ]
        row_ids = original_row_ids
        if noop_per_update is not None:
            required_row_ids = {
                catalog.row_id_for_turn(scenario, int(turn))
                for turn in (snapshot_requests or {}).get(scenario, {})
            }
            row_ids = ratio_closed_loop_row_ids(
                source,
                original_row_ids,
                noop_per_update=noop_per_update,
                seed=sampling_seed + scenario,
                required_row_ids=required_row_ids,
            )
        sampled_decisions = [
            str(
                _single(source, source.position_for_row_id(row_id))["target"][
                    "decision"
                ]
            )
            for row_id in row_ids
        ]
        streams.append(
            _BatchedClosedLoopStream(
                scenario=scenario,
                row_ids=row_ids,
                original_turns=len(original_row_ids),
                sampled_updates=sampled_decisions.count("UPDATE"),
                sampled_noops=sampled_decisions.count("NO_OP"),
                state=method.initial_state(),
                counts=_DecisionCounts(),
                decision_usage=_DecisionUsage(),
            )
        )
    total_turns = sum(len(stream.row_ids) for stream in streams)
    processed_turns = 0
    last_reported = 0
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "closed_loop_memory",
                "scenario_indices": selected,
                "item": 0,
                "items": total_turns,
            }
        )
    while True:
        active = [
            stream for stream in streams if stream.next_index < len(stream.row_ids)
        ]
        if not active:
            break
        for batch_start in range(0, len(active), batch_size):
            batch_streams = active[batch_start : batch_start + batch_size]
            prepared = []
            for stream in batch_streams:
                row_id = stream.row_ids[stream.next_index]
                position = source.position_for_row_id(row_id)
                canonical = _single(source, position)
                prepared.append(
                    (
                        stream,
                        canonical,
                        _single(gold_summary, position),
                        row_with_runtime_state(canonical, method, stream.state),
                    )
                )
            generated_batch = generate_outputs_batch(
                model,
                tokenizer,
                method,
                [row for _, _, _, row in prepared],
                encoder=encoder,
                accelerator=accelerator,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            for (stream, canonical, gold, row), generated_result in zip(
                prepared, generated_batch, strict=True
            ):
                generated, prompt_tokens, generated_tokens, elapsed = generated_result
                stream.prefill_tokens += prompt_tokens
                stream.decode_tokens += generated_tokens
                stream.latency_seconds += elapsed
                gold_decision = str(canonical["target"]["decision"])
                predicted_decision = "INVALID"
                try:
                    parsed = method.parse_output(generated)
                    predicted_decision = str(parsed.decision)
                    stream.state = method.apply_output(
                        stream.state,
                        parsed,
                        turn_id=str(row.get("turn_id", "")),
                    )
                    stream.counts.add(gold_decision, predicted_decision)
                except Exception:  # noqa: BLE001 - invalid generation keeps state.
                    predicted_decision = "INVALID"
                    stream.counts.add(gold_decision, "INVALID")
                    stream.failures += 1
                stream.decision_usage.add(
                    gold_decision,
                    predicted_decision,
                    prefill_tokens=prompt_tokens,
                    decode_tokens=generated_tokens,
                    attributed_latency_seconds=elapsed,
                )
                memory = method.materialize_memory(stream.state)
                score = memory_scores(
                    memory, str(gold["target"].get("next_memory", ""))
                )
                global_turn = int(canonical["global_turn_index"])
                for sample_id in (
                    (snapshot_requests or {})
                    .get(stream.scenario, {})
                    .get(global_turn, ())
                ):
                    stream.snapshots[str(sample_id)] = memory
                stream.scores.append(score)
                wrong = not bool(score["exact"])
                if wrong and stream.first_error is None:
                    stream.first_error = stream.next_index
                if stream.was_wrong and not wrong:
                    stream.recovery_count += 1
                stream.was_wrong = wrong
                stream.next_index += 1
                processed_turns += 1
            if progress_callback is not None and (
                processed_turns - last_reported >= progress_interval
                or processed_turns == total_turns
            ):
                progress_callback(
                    {
                        "phase": "closed_loop_memory",
                        "scenario_indices": [
                            stream.scenario for stream in batch_streams
                        ],
                        "item": processed_turns,
                        "items": total_turns,
                    }
                )
                last_reported = processed_turns
    results = {}
    for stream in streams:
        final = stream.scores[-1] if stream.scores else {"exact": 0.0, "f1": 0.0}
        sampling_mode = "noop_ratio" if noop_per_update is not None else "full"
        report = {
            "scenario_index": stream.scenario,
            "sampling_mode": sampling_mode,
            "original_turns": stream.original_turns,
            "turns": len(stream.row_ids),
            "state_exact": _mean(score["exact"] for score in stream.scores),
            "state_f1": _mean(score["f1"] for score in stream.scores),
            "final_state_exact": final["exact"],
            "final_state_f1": final["f1"],
            "first_error_turn": stream.first_error,
            "recovery_count": stream.recovery_count,
            "invalid_outputs": stream.failures,
        }
        result = {
            **stream.counts.metrics(),
            "decision_counts": stream.counts.raw_counts(),
            "split": split,
            "scenarios": 1,
            "turns": len(stream.row_ids),
            "state_exact": report["state_exact"],
            "state_f1": report["state_f1"],
            "final_state_exact": report["final_state_exact"],
            "final_state_f1": report["final_state_f1"],
            "prefill_tokens": stream.prefill_tokens,
            "decode_tokens": stream.decode_tokens,
            "total_tokens": stream.prefill_tokens + stream.decode_tokens,
            "latency_seconds": stream.latency_seconds,
            **stream.decision_usage.as_dict(),
            "scenario_reports": [report],
            "sampling": {
                "full_scenarios": (
                    [stream.scenario] if noop_per_update is None else []
                ),
                "sparse_scenarios": (
                    [] if noop_per_update is None else [stream.scenario]
                ),
                "sparse_noop_keep_fraction": 0.0,
                "noop_per_update": noop_per_update,
                "sampled_updates": stream.sampled_updates,
                "sampled_noops": stream.sampled_noops,
                "original_turns": stream.original_turns,
                "evaluated_turns": len(stream.row_ids),
            },
        }
        if snapshot_requests is not None:
            requested = {
                str(sample_id)
                for sample_ids in snapshot_requests.get(stream.scenario, {}).values()
                for sample_id in sample_ids
            }
            missing = sorted(requested - set(stream.snapshots))
            if missing:
                raise ValueError(
                    f"S{stream.scenario} closed-loop missed "
                    f"{len(missing)} Quiz snapshots"
                )
            result["quiz_snapshots"] = stream.snapshots
        results[stream.scenario] = result
    return results


def ratio_closed_loop_row_ids(
    source: IndexedMemoryDataset,
    row_ids: Sequence[int],
    *,
    noop_per_update: float,
    seed: int,
    required_row_ids: set[int] | None = None,
) -> list[int]:
    """Keep all UPDATEs and a deterministic, ordered NO_OP ratio subset.

    Quiz-anchor rows are mandatory, followed by immediate NO_OP neighbors of
    UPDATE turns. The remaining budget is sampled from other NO_OPs, and the
    final row IDs preserve the original trajectory order.
    """
    if noop_per_update < 0:
        raise ValueError("noop_per_update must be non-negative")
    decisions = [
        str(_single(source, source.position_for_row_id(row_id))["target"]["decision"])
        for row_id in row_ids
    ]
    updates = {
        index for index, decision in enumerate(decisions) if decision == "UPDATE"
    }
    noops = [index for index, decision in enumerate(decisions) if decision == "NO_OP"]
    required = set(required_row_ids or ())
    unknown_required = required - set(row_ids)
    if unknown_required:
        raise ValueError(
            f"Required rows are outside the trajectory: {unknown_required}"
        )
    required_noops = {
        index
        for index, row_id in enumerate(row_ids)
        if row_id in required and decisions[index] == "NO_OP"
    }
    target_noops = min(
        len(noops), max(round(len(updates) * noop_per_update), len(required_noops))
    )
    adjacent = sorted(
        {
            neighbor
            for index in updates
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(row_ids) and decisions[neighbor] == "NO_OP"
        }
    )
    generator = random.Random(seed)
    selected_set = set(required_noops)
    adjacent_candidates = [index for index in adjacent if index not in selected_set]
    adjacent_budget = target_noops - len(selected_set)
    selected_set.update(
        adjacent_candidates
        if len(adjacent_candidates) <= adjacent_budget
        else generator.sample(adjacent_candidates, adjacent_budget)
    )
    remaining = target_noops - len(selected_set)
    candidates = [index for index in noops if index not in selected_set]
    selected_noops = selected_set | set(generator.sample(candidates, remaining))
    selected = updates | selected_noops
    return [row_id for index, row_id in enumerate(row_ids) if index in selected]


def sparse_closed_loop_row_ids(
    source: IndexedMemoryDataset,
    row_ids: Sequence[int],
    *,
    noop_keep_fraction: float,
    seed: int,
) -> list[int]:
    """Keep every UPDATE, hard adjacent NO_OPs, and a fixed NO_OP subsample."""
    if not 0 <= noop_keep_fraction <= 1:
        raise ValueError("noop_keep_fraction must be in [0, 1]")
    decisions = []
    for row_id in row_ids:
        row = _single(source, source.position_for_row_id(row_id))
        decisions.append(str(row["target"]["decision"]))
    updates = {
        index for index, decision in enumerate(decisions) if decision == "UPDATE"
    }
    noops = [index for index, decision in enumerate(decisions) if decision == "NO_OP"]
    adjacent = {
        neighbor
        for index in updates
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < len(row_ids) and decisions[neighbor] == "NO_OP"
    }
    target_noops = round(len(noops) * noop_keep_fraction)
    random_budget = max(0, target_noops - len(adjacent))
    candidates = [index for index in noops if index not in adjacent]
    sampled = set(
        random.Random(seed).sample(candidates, min(random_budget, len(candidates)))
    )
    selected = updates | adjacent | sampled
    return [row_id for index, row_id in enumerate(row_ids) if index in selected]


def stratified_one_step_row_ids(
    catalog: DatasetCatalog,
    *,
    split: str = "validation",
    noop_ratio: float = 1.0,
    seed: int = 42,
    max_rows: int | None = None,
    scenario_min: int | None = None,
    scenario_max: int | None = None,
    scenarios: Sequence[int] | None = None,
) -> list[int]:
    updates = _scenario_filtered_row_ids(
        catalog,
        catalog.row_ids(split=split, decision="UPDATE"),
        scenario_min=scenario_min,
        scenario_max=scenario_max,
        scenarios=scenarios,
    )
    noops = _scenario_filtered_row_ids(
        catalog,
        catalog.row_ids(split=split, decision="NO_OP"),
        scenario_min=scenario_min,
        scenario_max=scenario_max,
        scenarios=scenarios,
    )
    target_noops = min(len(noops), round(len(updates) * noop_ratio))
    selected_noops = random.Random(seed).sample(noops, target_noops)
    selected = sorted([*updates, *selected_noops])
    if max_rows is not None and len(selected) > max_rows:
        selected = random.Random(seed).sample(selected, max_rows)
        selected.sort()
    return selected


def stratified_teacher_forced_row_ids(
    catalog: DatasetCatalog,
    *,
    split: str = "validation",
    max_rows: int = 512,
    adjacent_noop_fraction: float = 0.5,
    adjacent_max_distance: int = 2,
    seed: int = 42,
    scenario_min: int | None = None,
    scenario_max: int | None = None,
    scenarios: Sequence[int] | None = None,
) -> list[int]:
    """Build a fixed, UPDATE-preserving teacher-forced Validation subset.

    UPDATE rows are always retained. The remaining budget is split between NO_OP
    rows nearest to an UPDATE and deterministic random NO_OP rows. Keeping the row
    IDs fixed across checkpoints makes intermediate loss directly comparable while
    avoiding a prefix-biased slice of the Validation JSONL.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if not 0 <= adjacent_noop_fraction <= 1:
        raise ValueError("adjacent_noop_fraction must be in [0, 1]")
    if adjacent_max_distance < 1:
        raise ValueError("adjacent_max_distance must be positive")

    updates = _scenario_filtered_row_ids(
        catalog,
        catalog.row_ids(split=split, decision="UPDATE"),
        scenario_min=scenario_min,
        scenario_max=scenario_max,
        scenarios=scenarios,
    )
    if len(updates) > max_rows:
        raise ValueError(
            f"Validation subset size {max_rows} cannot retain all "
            f"{len(updates)} UPDATE rows"
        )
    noops = _scenario_filtered_row_ids(
        catalog,
        catalog.row_ids(split=split, decision="NO_OP"),
        scenario_min=scenario_min,
        scenario_max=scenario_max,
        scenarios=scenarios,
    )
    noop_budget = min(len(noops), max_rows - len(updates))
    adjacent_target = round(noop_budget * adjacent_noop_fraction)
    generator = random.Random(seed)

    tiers = catalog.adjacent_row_ids(
        split=split,
        max_distance=adjacent_max_distance,
    )
    adjacent_candidates = {
        row_id
        for candidates in tiers.values()
        for row_id in _scenario_filtered_row_ids(
            catalog,
            candidates,
            scenario_min=scenario_min,
            scenario_max=scenario_max,
            scenarios=scenarios,
        )
    }
    selected_adjacent: list[int] = []
    for distance in range(1, adjacent_max_distance + 1):
        remaining = adjacent_target - len(selected_adjacent)
        if remaining <= 0:
            break
        candidates = _scenario_filtered_row_ids(
            catalog,
            tiers.get(distance, []),
            scenario_min=scenario_min,
            scenario_max=scenario_max,
            scenarios=scenarios,
        )
        selected_adjacent.extend(
            candidates
            if len(candidates) <= remaining
            else generator.sample(candidates, remaining)
        )

    random_target = noop_budget - len(selected_adjacent)
    selected_set = set(selected_adjacent)
    random_candidates = [
        row_id for row_id in noops if row_id not in adjacent_candidates
    ]
    if len(random_candidates) < random_target:
        random_candidates.extend(
            row_id
            for row_id in noops
            if row_id in adjacent_candidates and row_id not in selected_set
        )
    selected_random = generator.sample(random_candidates, random_target)
    return sorted([*updates, *selected_adjacent, *selected_random])


def _scenario_filtered_row_ids(
    catalog: DatasetCatalog,
    row_ids: Sequence[int],
    *,
    scenario_min: int | None,
    scenario_max: int | None,
    scenarios: Sequence[int] | None = None,
) -> list[int]:
    if scenarios is None and scenario_min is None and scenario_max is None:
        return list(row_ids)
    selected = set(scenarios) if scenarios is not None else None
    allowed = {
        row_id
        for scenario in catalog.scenarios()
        if selected is None or scenario in selected
        if (scenario_min is None or scenario >= scenario_min)
        and (scenario_max is None or scenario <= scenario_max)
        for row_id in catalog.scenario_row_ids(scenario)
    }
    return [row_id for row_id in row_ids if row_id in allowed]


def generate_output(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    row: dict[str, Any],
    *,
    encoder: ChatExampleEncoder,
    accelerator: Any,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[str, int, int, float]:
    inputs = {
        key: value.to(accelerator.device)
        for key, value in encoder.generation_inputs(row, method).items()
    }
    was_training = model.training
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    elapsed = time.perf_counter() - start
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output[0, prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if was_training:
        model.train()
    return text, int(prompt_length), int(generated.numel()), elapsed


def generate_outputs_batch(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    rows: Sequence[dict[str, Any]],
    *,
    encoder: ChatExampleEncoder,
    accelerator: Any,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[tuple[str, int, int, float]]:
    """Greedily generate independent memory turns with one model forward batch."""
    if not rows:
        return []
    if len(rows) == 1:
        return [
            generate_output(
                model,
                tokenizer,
                method,
                rows[0],
                encoder=encoder,
                accelerator=accelerator,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        ]
    prompts = [encoder.generation_prompt(row, method) for row in rows]
    inputs = _batch_generation_inputs(
        tokenizer,
        prompts,
        max_length=encoder.max_length,
        device=accelerator.device,
    )
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    elapsed = time.perf_counter() - started
    input_width = int(inputs["input_ids"].shape[-1])
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    elapsed_share = elapsed / len(rows)
    results = []
    for index, prompt_length in enumerate(prompt_lengths):
        generated = _effective_generated_tokens(
            output[index, input_width:], tokenizer.eos_token_id
        )
        results.append(
            (
                tokenizer.decode(generated, skip_special_tokens=True).strip(),
                int(prompt_length),
                int(generated.numel()),
                elapsed_share,
            )
        )
    if was_training:
        model.train()
    return results


def stratified_quiz_validation_indices(
    source: IndexedQuizSFTDataset,
    *,
    rows_per_scenario: int = 10,
    seed: int = 42,
) -> list[int]:
    """Select a fixed TURN/FINAL and reasoning-aware subset per scenario."""
    if rows_per_scenario < 2:
        raise ValueError("Quiz Validation needs at least two rows per scenario")
    grouped: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"TURN": [], "FINAL": []}
    )
    for index in range(len(source)):
        row = source[index]
        if not isinstance(row, dict):
            raise TypeError("Expected one Quiz SFT row")
        grouped[int(row["scenario_index"])][str(row["quiz_type"])].append(index)
    selected: list[int] = []
    final_budget = max(1, round(rows_per_scenario * 0.25))
    turn_budget = rows_per_scenario - final_budget
    for scenario, by_type in sorted(grouped.items()):
        selected.extend(
            _reasoning_balanced_sample(
                source,
                by_type["TURN"],
                count=turn_budget,
                seed=seed + scenario * 101,
            )
        )
        selected.extend(
            _reasoning_balanced_sample(
                source,
                by_type["FINAL"],
                count=final_budget,
                seed=seed + scenario * 101 + 1,
            )
        )
    return sorted(selected)


def random_quiz_validation_indices(
    source: IndexedQuizSFTDataset,
    *,
    total_rows: int,
    excluded_scenarios: Sequence[int] = (),
    seed: int = 42,
) -> list[int]:
    """Select a fixed random Gold-memory Quiz subset without scenario overlap."""
    if total_rows < 1:
        raise ValueError("Gold-memory Quiz Validation rows must be positive")
    excluded = set(excluded_scenarios)
    candidates = [
        index
        for index in range(len(source))
        if int(source[index]["scenario_index"]) not in excluded
    ]
    if len(candidates) < total_rows:
        raise ValueError(
            f"Gold-memory Quiz pool has only {len(candidates)}/{total_rows} rows"
        )
    return sorted(random.Random(seed).sample(candidates, total_rows))


def quiz_indices_for_scenarios(
    source: IndexedQuizSFTDataset,
    scenarios: Sequence[int],
) -> list[int]:
    """Return every Quiz row for a fixed scenario set."""
    selected = set(scenarios)
    return [
        index
        for index in range(len(source))
        if int(source[index]["scenario_index"]) in selected
    ]


def closed_loop_quiz_snapshot_requests(
    source: IndexedQuizSFTDataset,
    indices: Sequence[int],
) -> dict[int, dict[int, list[str]]]:
    """Map scenario/turn checkpoints to Quiz sample IDs needing a snapshot."""
    requests: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for index in indices:
        row = source[index]
        scenario = int(row["scenario_index"])
        turn = int(row["memory_ref"]["global_turn_index"])
        requests[scenario][turn].append(str(row["sample_id"]))
    return {
        scenario: {turn: list(sample_ids) for turn, sample_ids in by_turn.items()}
        for scenario, by_turn in requests.items()
    }


def _reasoning_balanced_sample(
    source: IndexedQuizSFTDataset,
    indices: Sequence[int],
    *,
    count: int,
    seed: int,
) -> list[int]:
    if len(indices) < count:
        raise ValueError(f"Quiz stratum has only {len(indices)}/{count} rows")
    generator = random.Random(seed)
    by_reason: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        row = source[index]
        if not isinstance(row, dict):
            raise TypeError("Expected one Quiz SFT row")
        by_reason[str(row["reasoning_type"])].append(index)
    for values in by_reason.values():
        generator.shuffle(values)
    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for reason in sorted(by_reason):
            values = by_reason[reason]
            if values and len(selected) < count:
                selected.append(values.pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError("Could not fill the Quiz Validation stratum")
    return selected


def evaluate_quiz_tool_calling(
    model: Any,
    tokenizer: Any,
    source: IndexedQuizSFTDataset,
    tools: VehicleToolSchemaStore,
    indices: Sequence[int],
    *,
    dataset_root: Path,
    accelerator: Any,
    max_length: int = 4096,
    max_new_tokens: int = 256,
    memory_overrides: Mapping[str, str] | None = None,
    batch_size: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval: int = 5,
    allow_gold_execution_failure: bool = False,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """Evaluate Tool Calling with Gold or supplied predicted memory."""
    if batch_size < 1:
        raise ValueError("Quiz batch_size must be positive")
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.vehicle_bench.dataset import GoldToolCall
    from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime

    runtime = VehicleWorldRuntime(dataset_root)
    was_training = model.training
    model.eval()
    records = []
    prefill_tokens = 0
    decode_tokens = 0
    latency = 0.0
    prepared = []
    for index in indices:
        row = source[index]
        if not isinstance(row, dict):
            raise TypeError("Expected one Quiz SFT row")
        if memory_overrides is not None:
            sample_id = str(row["sample_id"])
            if sample_id not in memory_overrides:
                raise ValueError(f"Missing predicted memory for {sample_id}")
            row = _quiz_row_with_memory(row, memory_overrides[sample_id])
        prepared.append((row, tools.tools_for(row)))
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        generated_batch = generate_quiz_outputs_batch(
            model,
            tokenizer,
            [row for row, _ in batch],
            [candidates for _, candidates in batch],
            accelerator=accelerator,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        for offset, ((row, candidates), generated_result) in enumerate(
            zip(batch, generated_batch, strict=True), start=1
        ):
            generated, prompt_count, output_count, elapsed = generated_result
            position = batch_start + offset
            prefill_tokens += prompt_count
            decode_tokens += output_count
            latency += elapsed
            parsed = parse_tool_calls(generated, candidates)
            predicted_calls = tuple(
                GoldToolCall(
                    name=str(call["name"]),
                    arguments=dict(call["arguments"]),
                    source="hf-quiz-validation",
                )
                for call in parsed
            )
            reference_calls = tuple(
                GoldToolCall(
                    name=str(call["name"]),
                    arguments=dict(call["arguments"]),
                    source="quiz-sft-gold",
                )
                for call in row["target"]["tool_calls"]
            )
            reference_world = runtime.create_world()
            predicted_world = runtime.create_world()
            initial_state = runtime.state(reference_world)
            reference_execution_errors = []
            for call in reference_calls:
                try:
                    result = runtime.execute(reference_world, call.name, call.arguments)
                    if isinstance(result, Mapping) and result.get("success") is False:
                        raise ValueError(f"{call.name}: {result}")
                except Exception as exc:
                    if not allow_gold_execution_failure:
                        raise ValueError(
                            f"Gold Tool execution failed: {call.name}: {exc}"
                        ) from exc
                    reference_execution_errors.append(str(exc))
            execution_errors = []
            for call in predicted_calls:
                try:
                    result = runtime.execute(predicted_world, call.name, call.arguments)
                    if isinstance(result, Mapping) and result.get("success") is False:
                        execution_errors.append(f"{call.name}: {result}")
                except Exception as exc:  # noqa: BLE001 - metric captures failures.
                    execution_errors.append(f"{call.name}: {exc}")
            score = runtime.score(
                initial_state=initial_state,
                reference_state=runtime.state(reference_world),
                predicted_state=runtime.state(predicted_world),
                predicted_calls=predicted_calls,
                reference_calls=reference_calls,
            )
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "scenario_index": row["scenario_index"],
                    "quiz_type": row["quiz_type"],
                    "reasoning_type": row["reasoning_type"],
                    "exact_state_match": float(score.exact_state_match),
                    "tool_f1": float(score.tool_score.get("f1", 0.0)),
                    "arg_exact": float(
                        _argument_exact_match(predicted_calls, reference_calls)
                    ),
                    "parse_success": float(bool(parsed)),
                    "execution_success": float(not execution_errors),
                    "predicted_calls": parsed,
                    "reference_execution_errors": reference_execution_errors,
                    "execution_errors": execution_errors,
                    "output": generated,
                    "prefill_tokens": prompt_count,
                    "decode_tokens": output_count,
                    "latency_seconds": elapsed,
                    "memory_mode": (
                        "closed_loop_predicted"
                        if memory_overrides is not None
                        else "gold"
                    ),
                }
            )
            if progress_callback is not None and (
                position % progress_interval == 0 or position == len(indices)
            ):
                progress_callback(
                    {
                        "phase": "closed_loop_quiz"
                        if memory_overrides is not None
                        else "gold_memory_quiz",
                        "item": position,
                        "items": len(indices),
                    }
                )
    if was_training:
        model.train()
    result = _aggregate_quiz_records(records)
    result.update(
        {
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "total_tokens": prefill_tokens + decode_tokens,
            "latency_seconds": latency,
            "records": records,
        }
    )
    return result


def _quiz_row_with_memory(row: Mapping[str, Any], memory: str) -> dict[str, Any]:
    updated = copy.deepcopy(dict(row))
    messages = updated.get("messages")
    if not isinstance(messages, list):
        raise TypeError("Quiz row has no messages")
    user = next(
        (message for message in messages if message.get("role") == "user"),
        None,
    )
    if not isinstance(user, dict) or not isinstance(user.get("content"), str):
        raise TypeError("Quiz row has no user message")
    marker = "\n\n[Current request]\n"
    content = str(user["content"])
    if not content.startswith("[Memory]\n") or marker not in content:
        raise ValueError("Quiz user message does not match the memory contract")
    request = content.split(marker, 1)[1]
    user["content"] = f"[Memory]\n{normalize_memory(memory)}{marker}{request}"
    return updated


def generate_quiz_output(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    tools: list[dict[str, Any]],
    *,
    accelerator: Any,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[str, int, int, float]:
    messages = row["messages"][:-1]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "tools": tools,
    }
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except (TypeError, ValueError):
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {
        key: value.to(accelerator.device)
        for key, value in encoded.items()
        if key != "token_type_ids"
    }
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    elapsed = time.perf_counter() - started
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output[0, prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, int(prompt_length), int(generated.numel()), elapsed


def generate_quiz_outputs_batch(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    tools_by_row: Sequence[list[dict[str, Any]]],
    *,
    accelerator: Any,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[tuple[str, int, int, float]]:
    """Greedily generate independent Tool Calls with one model forward batch."""
    if len(rows) != len(tools_by_row):
        raise ValueError("Quiz rows and Tool schemas are not aligned")
    if not rows:
        return []
    if len(rows) == 1:
        return [
            generate_quiz_output(
                model,
                tokenizer,
                rows[0],
                tools_by_row[0],
                accelerator=accelerator,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        ]
    prompts = [
        _quiz_generation_prompt(tokenizer, row, tools)
        for row, tools in zip(rows, tools_by_row, strict=True)
    ]
    inputs = _batch_generation_inputs(
        tokenizer,
        prompts,
        max_length=max_length,
        device=accelerator.device,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    elapsed = time.perf_counter() - started
    input_width = int(inputs["input_ids"].shape[-1])
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    elapsed_share = elapsed / len(rows)
    results = []
    for index, prompt_length in enumerate(prompt_lengths):
        generated = _effective_generated_tokens(
            output[index, input_width:], tokenizer.eos_token_id
        )
        results.append(
            (
                tokenizer.decode(generated, skip_special_tokens=True).strip(),
                int(prompt_length),
                int(generated.numel()),
                elapsed_share,
            )
        )
    return results


def _quiz_generation_prompt(
    tokenizer: Any,
    row: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
) -> str:
    messages = row["messages"][:-1]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "tools": tools,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, **kwargs)


def _batch_generation_inputs(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    device: Any,
) -> dict[str, torch.Tensor]:
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            list(prompts),
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if key != "token_type_ids"
    }


def _effective_generated_tokens(
    generated: torch.Tensor, eos_token_id: int | Sequence[int] | None
) -> torch.Tensor:
    if eos_token_id is None:
        return generated
    eos_ids = (
        {int(eos_token_id)}
        if isinstance(eos_token_id, int)
        else {int(value) for value in eos_token_id}
    )
    for position, token in enumerate(generated.tolist()):
        if int(token) in eos_ids:
            return generated[: position + 1]
    return generated


_JSON_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_XML_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\n]+)>(?P<body>.*?)"
    r"</function>\s*</tool_call>",
    re.DOTALL,
)
_XML_PARAMETER = re.compile(
    r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_calls(
    text: str, tools: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Parse Granite JSON or Qwen3.5 XML Tool Call generations."""
    schemas = {
        str(tool.get("function", tool).get("name")): tool.get("function", tool)
        for tool in tools
        if isinstance(tool.get("function", tool), Mapping)
    }
    calls = []
    for match in _JSON_TOOL_CALL.finditer(text):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and isinstance(value.get("name"), str):
            arguments = value.get("arguments")
            if isinstance(arguments, Mapping):
                calls.append({"name": str(value["name"]), "arguments": dict(arguments)})
    if calls:
        return calls
    for match in _XML_TOOL_CALL.finditer(text):
        name = match.group("name").strip()
        raw_arguments = {
            parameter.group("key").strip(): parameter.group("value").strip()
            for parameter in _XML_PARAMETER.finditer(match.group("body"))
        }
        calls.append(
            {
                "name": name,
                "arguments": _coerce_arguments(raw_arguments, schemas.get(name, {})),
            }
        )
    return calls


def _coerce_arguments(
    raw: Mapping[str, str], schema: Mapping[str, Any]
) -> dict[str, Any]:
    parameters = schema.get("parameters")
    properties = (
        parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
    )
    return {
        key: _coerce_argument(value, properties.get(key, {}))
        for key, value in raw.items()
    }


def _coerce_argument(raw: str, schema: Mapping[str, Any]) -> Any:
    type_name = schema.get("type")
    try:
        if type_name == "integer":
            return int(raw)
        if type_name == "number":
            value = float(raw)
            return int(value) if value.is_integer() else value
        if type_name == "boolean" and raw.lower() in {"true", "false", "1", "0"}:
            return raw.lower() in {"true", "1"}
        if type_name in {"object", "array"}:
            return json.loads(raw)
        if type_name == "null" and raw.lower() in {"null", "none"}:
            return None
        if type_name == "string" and raw.startswith('"'):
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw
    return raw


def _argument_exact_match(
    predicted_calls: Sequence[Any], reference_calls: Sequence[Any]
) -> bool:
    if len(predicted_calls) != len(reference_calls):
        return False
    return all(
        predicted.name == reference.name
        and _canonical_arguments(predicted.arguments)
        == _canonical_arguments(reference.arguments)
        for predicted, reference in zip(predicted_calls, reference_calls, strict=True)
    )


def _canonical_arguments(arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(arguments), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _aggregate_quiz_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "tasks": len(selected),
            "esm": _mean(float(row["exact_state_match"]) for row in selected),
            "tool_f1": _mean(float(row["tool_f1"]) for row in selected),
            "arg_exact": _mean(float(row["arg_exact"]) for row in selected),
            "parse_success_rate": _mean(
                float(row["parse_success"]) for row in selected
            ),
            "execution_success_rate": _mean(
                float(row["execution_success"]) for row in selected
            ),
        }

    by_type = {
        quiz_type: metrics([row for row in records if row["quiz_type"] == quiz_type])
        for quiz_type in sorted({str(row["quiz_type"]) for row in records})
    }
    by_reason = {
        reason: metrics([row for row in records if row["reasoning_type"] == reason])
        for reason in sorted({str(row["reasoning_type"]) for row in records})
    }
    return {**metrics(records), "by_quiz_type": by_type, "by_reasoning_type": by_reason}


def state_from_input(method: MemoryMethod[Any], value: Mapping[str, Any]) -> Any:
    if isinstance(method, DeltaMethod):
        pending = tuple(
            DeltaBatch(
                str(batch["turn_id"]),
                validate_operations(batch["operations"]),
            )
            for batch in value["pending_deltas"]
        )
        return DeltaState(
            base_summary=normalize_memory(str(value["base_summary"])),
            pending_deltas=pending,
            updates_since_compaction=int(value["updates_since_compaction"]),
        )
    if isinstance(method, DeltaV2Method):
        return delta_v2_state_from_input(
            value, compaction_interval=method.compaction_interval
        )
    return normalize_memory(str(value["previous_memory"]))


def row_with_runtime_state(
    canonical: dict[str, Any], method: MemoryMethod[Any], state: Any
) -> dict[str, Any]:
    row = copy.deepcopy(canonical)
    row["input"] = state_to_input(method, state)
    if not isinstance(method, (DeltaMethod, DeltaV2Method)):
        canonical_input = canonical.get("input")
        if isinstance(canonical_input, Mapping) and "turns" in canonical_input:
            # Batch methods still need the immutable conversation batch while
            # replacing only the evolving memory state during closed-loop replay.
            row["input"]["turns"] = copy.deepcopy(canonical_input["turns"])
    return row


def state_to_input(method: MemoryMethod[Any], state: Any) -> dict[str, Any]:
    """Serialize a method runtime state into its next-turn/checkpoint contract."""
    if isinstance(method, DeltaMethod):
        return {
            "base_summary": state.base_summary,
            "pending_deltas": [asdict(batch) for batch in state.pending_deltas],
            "updates_since_compaction": state.updates_since_compaction,
        }
    if isinstance(method, DeltaV2Method):
        return delta_v2_input_from_state(
            state, compaction_interval=method.compaction_interval
        )
    return {"previous_memory": method.materialize_memory(state)}


def memory_scores(predicted: str, gold: str) -> dict[str, float]:
    predicted = normalize_memory(predicted)
    gold = normalize_memory(gold)
    predicted_lines = {line.strip() for line in predicted.splitlines() if line.strip()}
    gold_lines = {line.strip() for line in gold.splitlines() if line.strip()}
    overlap = len(predicted_lines & gold_lines)
    precision = (
        overlap / len(predicted_lines) if predicted_lines else float(not gold_lines)
    )
    recall = overlap / len(gold_lines) if gold_lines else float(not predicted_lines)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact": float(predicted == gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_quiz_hook(
    command: str, context: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Run an optional adapter-to-Quiz bridge; it must write one JSON result."""
    context_path = output_path.with_suffix(".context.json")
    context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    rendered = command.format(
        context=shlex.quote(str(context_path)), output=shlex.quote(str(output_path))
    )
    subprocess.run(shlex.split(rendered), check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


@dataclass
class _UsageTotals:
    turns: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    attributed_latency_seconds: float = 0.0

    def add(
        self,
        *,
        prefill_tokens: int,
        decode_tokens: int,
        attributed_latency_seconds: float,
        turns: int = 1,
    ) -> None:
        self.turns += int(turns)
        self.prefill_tokens += int(prefill_tokens)
        self.decode_tokens += int(decode_tokens)
        self.attributed_latency_seconds += float(attributed_latency_seconds)

    def merge(self, value: Mapping[str, Any]) -> None:
        self.add(
            turns=int(value.get("turns", 0)),
            prefill_tokens=int(value.get("prefill_tokens", 0)),
            decode_tokens=int(value.get("decode_tokens", 0)),
            attributed_latency_seconds=float(
                value.get("attributed_latency_seconds", 0.0)
            ),
        )

    def as_dict(self) -> dict[str, int | float]:
        total_tokens = self.prefill_tokens + self.decode_tokens
        turns = self.turns
        return {
            "turns": turns,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "total_tokens": total_tokens,
            "attributed_latency_seconds": self.attributed_latency_seconds,
            "average_prefill_tokens": self.prefill_tokens / turns if turns else 0.0,
            "average_decode_tokens": self.decode_tokens / turns if turns else 0.0,
            "average_total_tokens": total_tokens / turns if turns else 0.0,
            "average_attributed_latency_seconds": (
                self.attributed_latency_seconds / turns if turns else 0.0
            ),
        }


class _DecisionUsage:
    """Attribute generation usage by gold, predicted, and paired decisions."""

    def __init__(self) -> None:
        self.by_gold = {name: _UsageTotals() for name in ("UPDATE", "NO_OP")}
        self.by_predicted = {
            name: _UsageTotals() for name in ("UPDATE", "NO_OP", "INVALID")
        }
        self.by_pair = {
            f"{gold}->{predicted}": _UsageTotals()
            for gold in ("UPDATE", "NO_OP")
            for predicted in ("UPDATE", "NO_OP", "INVALID")
        }

    def add(
        self,
        gold: str,
        predicted: str,
        *,
        prefill_tokens: int,
        decode_tokens: int,
        attributed_latency_seconds: float,
    ) -> None:
        normalized_gold = gold if gold in self.by_gold else "NO_OP"
        normalized_predicted = (
            predicted if predicted in self.by_predicted else "INVALID"
        )
        values = {
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "attributed_latency_seconds": attributed_latency_seconds,
        }
        self.by_gold[normalized_gold].add(**values)
        self.by_predicted[normalized_predicted].add(**values)
        self.by_pair[f"{normalized_gold}->{normalized_predicted}"].add(**values)

    def merge(self, value: Mapping[str, Any]) -> None:
        # The pair table is the canonical source. Rebuilding the marginal tables
        # avoids double-counting serialized by-gold and by-predicted totals.
        for pair, totals in value.get("usage_by_decision_pair", {}).items():
            if pair not in self.by_pair or not isinstance(totals, Mapping):
                continue
            gold, predicted = pair.split("->", 1)
            turns = int(totals.get("turns", 0))
            raw = {
                "turns": turns,
                "prefill_tokens": int(totals.get("prefill_tokens", 0)),
                "decode_tokens": int(totals.get("decode_tokens", 0)),
                "attributed_latency_seconds": float(
                    totals.get("attributed_latency_seconds", 0.0)
                ),
            }
            self.by_gold[gold].merge(raw)
            self.by_predicted[predicted].merge(raw)
            self.by_pair[pair].merge(raw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "usage_by_gold_decision": {
                key: value.as_dict() for key, value in self.by_gold.items()
            },
            "usage_by_predicted_decision": {
                key: value.as_dict() for key, value in self.by_predicted.items()
            },
            "usage_by_decision_pair": {
                key: value.as_dict() for key, value in self.by_pair.items()
            },
            "decision_usage_latency_kind": "attributed_wall_clock",
        }


def merge_decision_usage(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge additive decision-usage reports from independent scenarios."""
    usage = _DecisionUsage()
    for result in results:
        usage.merge(result)
    return usage.as_dict()


class _DecisionCounts:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = self.invalid = 0
        self.invalid_noop = 0

    def add(self, gold: str, predicted: str) -> None:
        if predicted == "INVALID":
            self.invalid += 1
            if gold == "UPDATE":
                self.fn += 1
            else:
                self.invalid_noop += 1
            return
        if gold == "UPDATE" and predicted == "UPDATE":
            self.tp += 1
        elif gold == "NO_OP" and predicted == "UPDATE":
            self.fp += 1
        elif gold == "UPDATE":
            self.fn += 1
        else:
            self.tn += 1

    def metrics(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return {
            "update_precision": precision,
            "update_recall": recall,
            "update_f1": f1,
            "noop_specificity": self.tn / (self.tn + self.fp + self.invalid_noop)
            if self.tn + self.fp + self.invalid_noop
            else 0.0,
            "false_update_rate": self.fp / (self.tn + self.fp + self.invalid_noop)
            if self.tn + self.fp + self.invalid_noop
            else 0.0,
            "invalid_count": self.invalid,
        }

    def raw_counts(self) -> dict[str, int]:
        return {
            "true_update": self.tp,
            "false_update": self.fp,
            "missed_update": self.fn,
            "true_noop": self.tn,
            "invalid": self.invalid,
            "invalid_noop": self.invalid_noop,
        }


def _single(dataset: IndexedMemoryDataset, position: int) -> dict[str, Any]:
    row = dataset[position]
    if not isinstance(row, dict):
        raise TypeError("Expected one row")
    return row


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0
