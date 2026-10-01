"""Benchmark batch-1 HF memory generation with optional cross-turn KV reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache_benchmark_manifest import validate_manifest
from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .dataset import IndexedMemoryDataset, default_catalog_path, ensure_catalog
from .delta_append_runtime import DeltaAppendPromptSession
from .hf_prefix_cache import HFPrefixCacheGenerator, PrefixGenerationResult
from .methods import (
    METHODS,
    DeltaV2Method,
    DeltaV3AppendMethod,
    Mem0TwoStageMethod,
)
from .validation import row_with_runtime_state, state_from_input

SCHEMA_VERSION = "palmclaw-hf-prefix-cache-benchmark-v3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument(
        "--method",
        choices=(
            "summary",
            "patch",
            "mem0_one_pass",
            "mem0_two_stage",
            "delta_v3",
            "delta_v3_append",
            "delta_v3_compact_k2",
            "delta_v3_compact_k5",
            "delta_v3_compact_k10",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--turn-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--cache-mode", choices=("off", "on"), required=True)
    parser.add_argument(
        "--replay-mode", choices=("controlled", "predicted"), required=True
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--warmup-turns", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--delta-v3-append-max-turns",
        type=int,
        default=32,
        help="Maximum turns retained in one append-only cache epoch.",
    )
    parser.add_argument(
        "--append-allow-turn-gaps",
        action="store_true",
        help=(
            "Treat non-consecutive manifest rows as consecutive requests. "
            "By default a gap resets the append-only epoch because omitted "
            "turns may have changed memory."
        ),
    )
    parser.add_argument(
        "--background-prefill",
        action="store_true",
        help=(
            "After a state-changing UPDATE, prefill the stable memory prefix "
            "outside the next turn's foreground latency."
        ),
    )
    parser.add_argument(
        "--reference-turns",
        type=Path,
        help="Cache-OFF turns.jsonl whose generated outputs must match exactly.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_length < 1 or args.max_new_tokens < 1:
        raise ValueError("Generation lengths must be positive")
    if args.warmup_turns < 0 or args.repetitions < 1:
        raise ValueError("Warmup must be non-negative and repetitions positive")
    if args.delta_v3_append_max_turns < 1:
        raise ValueError("Delta-v3 append max turns must be positive")
    if args.background_prefill and args.cache_mode != "on":
        raise ValueError("Background prefill requires --cache-mode on")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "adapter").is_dir():
        raise FileNotFoundError(f"Checkpoint adapter not found: {checkpoint}")
    output_dir = args.output_dir.resolve()
    turns_path = output_dir / "turns.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.force and (turns_path.exists() or summary_path.exists()):
        raise FileExistsError(f"Benchmark output already exists: {output_dir}")

    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    _configure_runtime_paths(workspace)
    catalog = ensure_catalog(
        data_root,
        (
            args.catalog_path
            or (
                data_root / "catalog.sqlite"
                if (data_root / "catalog.sqlite").is_file()
                else default_catalog_path(workspace)
            )
        ).resolve(),
    )
    turn_manifest = _read_json(args.turn_manifest.resolve())
    scenario_row_ids = validate_manifest(turn_manifest, catalog)

    from accelerate import Accelerator

    from .models import load_peft_bundle
    from .training_data import ChatExampleEncoder, DeltaAppendChatExampleEncoder

    accelerator = Accelerator(mixed_precision="bf16")
    bundle = load_peft_bundle(
        args.model,
        adapter_path=checkpoint / "adapter",
        gradient_checkpointing=False,
        cache_dir=workspace / "cache" / "huggingface" / "hub",
    )
    model = accelerator.prepare(bundle.model)
    model.eval()
    method = METHODS[args.method]()
    if method.name in {"mem0_one_pass", "mem0_two_stage"} and args.background_prefill:
        raise ValueError("Mem0 benchmarks do not support background prefill")
    if isinstance(method, Mem0TwoStageMethod) and args.warmup_turns:
        raise ValueError(
            "Mem0 two-stage benchmarking currently requires --warmup-turns 0"
        )
    encoder = (
        DeltaAppendChatExampleEncoder(bundle.tokenizer, max_length=args.max_length)
        if isinstance(method, DeltaV3AppendMethod)
        else ChatExampleEncoder(bundle.tokenizer, max_length=args.max_length)
    )
    source = IndexedMemoryDataset(
        catalog,
        method.source_view,
        row_ids=[row_id for values in scenario_row_ids.values() for row_id in values],
    )
    generator = HFPrefixCacheGenerator(
        model,
        bundle.tokenizer,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        cache_enabled=args.cache_mode == "on",
        device=accelerator.device,
    )
    _warm_up(
        generator,
        encoder,
        method,
        source,
        scenario_row_ids,
        turns=args.warmup_turns,
        append_max_turns=args.delta_v3_append_max_turns,
        append_reset_on_discontinuity=not args.append_allow_turn_gaps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for repetition in range(1, args.repetitions + 1):
        for scenario, row_ids in scenario_row_ids.items():
            generator.reset()
            predicted_state = method.initial_state()
            controlled_mem0_state = method.initial_state()
            append_session = _append_session(
                method,
                encoder,
                max_history_turns=args.delta_v3_append_max_turns,
                reset_on_discontinuity=not args.append_allow_turn_gaps,
            )
            background_cache_ready = False
            for sequence_index, row_id in enumerate(row_ids):
                canonical = source[source.position_for_row_id(row_id)]
                runtime_preparation_start = time.perf_counter()
                if args.replay_mode == "controlled":
                    if method.name in {"mem0_one_pass", "mem0_two_stage"}:
                        input_state = controlled_mem0_state
                        runtime_row = row_with_runtime_state(
                            canonical, method, input_state
                        )
                    else:
                        input_state = state_from_input(method, canonical["input"])
                        runtime_row = canonical
                else:
                    input_state = predicted_state
                    runtime_row = row_with_runtime_state(
                        canonical, method, predicted_state
                    )
                runtime_preparation_seconds = (
                    time.perf_counter() - runtime_preparation_start
                )
                pending_depth_before = _pending_depth(method, input_state)
                background_cache_available_before = background_cache_ready
                background_cache_ready = False
                append_prepared = None
                if append_session is not None:
                    append_prepared = append_session.prepare(
                        runtime_row,
                        materialized_memory=method.materialize_memory(input_state),
                    )
                    if (
                        append_prepared.reset_before
                        and not background_cache_available_before
                    ):
                        generator.reset()
                    prompt = append_prepared.prompt
                else:
                    prompt = encoder.generation_prompt(runtime_row, method)
                cache_anchor_prefixes = (
                    ()
                    if append_session is not None
                    else encoder.cache_anchor_prefixes(runtime_row, method)
                )
                extraction_generation = generator.generate(
                    prompt,
                    cache_anchor_prefixes=cache_anchor_prefixes,
                )
                manager_generation = None
                retrieval_seconds = (
                    runtime_preparation_seconds
                    if method.name == "mem0_one_pass"
                    else 0.0
                )
                generation = extraction_generation
                if isinstance(method, Mem0TwoStageMethod):
                    try:
                        extraction = method.parse_output(extraction_generation.text)
                        if extraction.payload.get("task") != "EXTRACT":
                            raise ValueError("First two-stage output is not EXTRACT")
                        facts = extraction.payload["facts"]
                        if facts:
                            manager_started = time.perf_counter()
                            manager_row = method.manager_row(runtime_row, facts)
                            retrieval_seconds = time.perf_counter() - manager_started
                            runtime_preparation_seconds += retrieval_seconds
                            manager_prompt = encoder.generation_prompt(
                                manager_row, method
                            )
                            # Both tasks share the same system prompt and have
                            # no other immutable input. A shared lane therefore
                            # reuses the safe prefix without retaining two full
                            # KV states on device.
                            manager_generation = generator.generate(
                                manager_prompt,
                                cache_anchor_prefixes=encoder.cache_anchor_prefixes(
                                    manager_row, method
                                ),
                            )
                            generation = _combine_two_stage_generations(
                                extraction_generation,
                                manager_generation,
                                retrieval_seconds=retrieval_seconds,
                            )
                        else:
                            generation = _two_stage_noop_generation(
                                extraction_generation
                            )
                    except Exception as exc:  # noqa: BLE001 - parsed below as INVALID.
                        generation = _two_stage_error_generation(
                            extraction_generation, exc
                        )
                predicted_decision = "INVALID"
                error = None
                applied_update = False
                state_compacted = False
                next_state = input_state
                try:
                    parsed = method.parse_output(generation.text)
                    predicted_decision = parsed.decision
                    next_state = method.apply_output(
                        input_state,
                        parsed,
                        turn_id=str(canonical.get("turn_id", "")),
                    )
                    applied_update = predicted_decision == "UPDATE"
                    state_compacted = (
                        isinstance(method, DeltaV2Method)
                        and applied_update
                        and pending_depth_before == method.compaction_interval - 1
                        and _pending_depth(method, next_state) == 0
                    )
                except Exception as exc:  # noqa: BLE001 - record invalid output.
                    error = f"{type(exc).__name__}: {exc}"
                background_state = next_state
                background_applied_update = applied_update
                if args.replay_mode == "controlled":
                    if method.name in {"mem0_one_pass", "mem0_two_stage"}:
                        controlled_mem0_state = _mem0_gold_state_after(
                            controlled_mem0_state, canonical
                        )
                        background_state = controlled_mem0_state
                        background_applied_update = (
                            canonical.get("target", {}).get("decision") == "UPDATE"
                        )
                    else:
                        gold_output = method.parse_output(
                            method.format_target(canonical)
                        )
                        background_state = method.apply_output(
                            input_state,
                            gold_output,
                            turn_id=str(canonical.get("turn_id", "")),
                        )
                        background_applied_update = gold_output.decision == "UPDATE"
                append_commit = None
                if append_session is not None:
                    decision_for_epoch = (
                        str(canonical["target"]["decision"])
                        if args.replay_mode == "controlled"
                        else (predicted_decision if error is None else "INVALID")
                    )
                    append_commit = append_session.commit(
                        assistant_output=generation.text,
                        decision_for_epoch=decision_for_epoch,
                    )
                if args.replay_mode == "predicted":
                    predicted_state = next_state
                compacted = (
                    append_commit.epoch_end_reason == "update_compaction"
                    if append_commit is not None
                    else state_compacted
                )
                background_prefill = None
                if args.background_prefill and _should_background_prefill(
                    append_commit=append_commit,
                    applied_update=background_applied_update,
                ):
                    background_row = row_with_runtime_state(
                        canonical, method, background_state
                    )
                    if append_session is not None:
                        background_prefix = encoder.stable_epoch_prefix(
                            background_row,
                            method,
                            base_summary=method.materialize_memory(background_state),
                        )
                    else:
                        background_prefix = encoder.stable_generation_prefix(
                            background_row, method
                        )
                    background_reference_prompt = (
                        None
                        if append_session is not None
                        else encoder.generation_prompt(background_row, method)
                    )
                    if generator.cache_strategy != "immutable_prefix_snapshots":
                        generator.reset()
                    background_prefill = generator.prefill(
                        background_prefix,
                        cache_anchor_prefixes=(
                            ()
                            if append_session is not None
                            else encoder.cache_anchor_prefixes(background_row, method)
                        ),
                        reference_prompt=background_reference_prompt,
                    )
                    background_cache_ready = True
                state_sha256 = hashlib.sha256(
                    method.materialize_memory(next_state).encode("utf-8")
                ).hexdigest()
                records.append(
                    {
                        "repetition": repetition,
                        "scenario_index": scenario,
                        "sequence_index": sequence_index,
                        "row_id": row_id,
                        "global_turn_index": int(canonical["global_turn_index"]),
                        "turn_id": str(canonical["turn_id"]),
                        "gold_decision": str(canonical["target"]["decision"]),
                        "predicted_decision": predicted_decision,
                        "applied_update": applied_update,
                        "pending_depth_before": pending_depth_before,
                        "compacted": compacted,
                        "state_compacted": state_compacted,
                        **_append_record_fields(append_prepared, append_commit),
                        "background_cache_available_before": (
                            background_cache_available_before
                        ),
                        "runtime_preparation_seconds": runtime_preparation_seconds,
                        "retrieval_seconds": retrieval_seconds,
                        **_two_stage_record_fields(
                            (
                                extraction_generation
                                if isinstance(method, Mem0TwoStageMethod)
                                else None
                            ),
                            manager_generation,
                        ),
                        **_background_record_fields(background_prefill),
                        "total_model_tokens": (
                            generation.evaluated_prefill_tokens
                            + generation.decode_tokens
                            + (
                                background_prefill.evaluated_prefill_tokens
                                if background_prefill is not None
                                else 0
                            )
                        ),
                        "total_model_seconds": (
                            generation.model_seconds
                            + (
                                background_prefill.prefill_seconds
                                if background_prefill is not None
                                else 0.0
                            )
                        ),
                        "state_sha256": state_sha256,
                        "error": error,
                        "output": generation.text,
                        **asdict(generation),
                    }
                )

    _write_jsonl(turns_path, records)
    equivalence = _compare_reference(records, args.reference_turns)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "signature": _signature(args, turn_manifest),
        "model": args.model,
        "method": args.method,
        "checkpoint": str(checkpoint),
        "turn_manifest": str(args.turn_manifest.resolve()),
        "turn_manifest_signature": turn_manifest["signature"],
        "cache_mode": args.cache_mode,
        "cache_strategy": generator.cache_strategy,
        "replay_mode": args.replay_mode,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "warmup_turns": args.warmup_turns,
        "repetitions": args.repetitions,
        "delta_v3_append_max_turns": args.delta_v3_append_max_turns,
        "append_reset_on_discontinuity": not args.append_allow_turn_gaps,
        "background_prefill": args.background_prefill,
        "scenario_batch_size": 1,
        "equivalence": equivalence,
        "metrics": aggregate_records(records),
    }
    _write_json(summary_path, result)
    return result


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "all_turns": _aggregate_slice(records),
        "by_gold_decision": {
            decision: _aggregate_slice(
                [row for row in records if row["gold_decision"] == decision]
            )
            for decision in ("NO_OP", "UPDATE")
        },
        "by_predicted_decision": {
            decision: _aggregate_slice(
                [row for row in records if row["predicted_decision"] == decision]
            )
            for decision in ("NO_OP", "UPDATE", "INVALID")
        },
        "delta_compaction": _aggregate_slice(
            [row for row in records if bool(row.get("compacted"))]
        ),
        "cache_epoch_rebuild": _aggregate_slice(
            [row for row in records if bool(row.get("cache_epoch_reset_before"))]
        ),
        "cache_epoch_steady_state": _aggregate_slice(
            [row for row in records if row.get("cache_epoch_reset_before") is False]
        ),
        "cache_epoch_rebuild_by_reason": {
            reason: _aggregate_slice(
                [
                    row
                    for row in records
                    if row.get("cache_epoch_reset_reason") == reason
                ]
            )
            for reason in (
                "scenario_start",
                "update_compaction",
                "turn_budget",
                "context_limit",
                "turn_discontinuity",
            )
        },
    }


def _aggregate_slice(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "logical_prompt_tokens",
        "candidate_prefix_tokens",
        "reused_prefix_tokens",
        "evaluated_prefill_tokens",
        "decode_tokens",
        "tokenization_seconds",
        "cache_management_seconds",
        "prefill_seconds",
        "decode_seconds",
        "model_seconds",
        "ttft_seconds",
        "end_to_end_seconds",
        "kv_cache_bytes",
        "peak_cuda_allocated_bytes",
        "background_prefill_tokens",
        "background_prefill_seconds",
        "background_prefill_end_to_end_seconds",
        "total_model_tokens",
        "total_model_seconds",
        "runtime_preparation_seconds",
        "retrieval_seconds",
        "extract_logical_prompt_tokens",
        "extract_evaluated_prefill_tokens",
        "extract_decode_tokens",
        "extract_model_seconds",
        "manager_logical_prompt_tokens",
        "manager_evaluated_prefill_tokens",
        "manager_decode_tokens",
        "manager_model_seconds",
    )
    result: dict[str, Any] = {"turns": len(records)}
    for metric in metrics:
        values = [_metric_value(row, metric) for row in records]
        result[metric] = {
            "total": sum(values),
            "mean": statistics.fmean(values) if values else 0.0,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
        }
    logical = sum(float(row["logical_prompt_tokens"]) for row in records)
    reused = sum(float(row["reused_prefix_tokens"]) for row in records)
    result["cache_reuse_ratio"] = reused / logical if logical else 0.0
    result["errors"] = sum(row.get("error") is not None for row in records)
    result["manager_invocations"] = sum(
        bool(row.get("manager_invoked")) for row in records
    )
    return result


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "total_model_tokens":
        return float(
            row.get(
                metric,
                float(row["evaluated_prefill_tokens"]) + float(row["decode_tokens"]),
            )
        )
    if metric == "total_model_seconds":
        return float(row.get(metric, row["model_seconds"]))
    if metric.startswith("background_"):
        return float(row.get(metric, 0.0))
    if metric in {
        "runtime_preparation_seconds",
        "retrieval_seconds",
        "extract_logical_prompt_tokens",
        "extract_evaluated_prefill_tokens",
        "extract_decode_tokens",
        "extract_model_seconds",
        "manager_logical_prompt_tokens",
        "manager_evaluated_prefill_tokens",
        "manager_decode_tokens",
        "manager_model_seconds",
    }:
        return float(row.get(metric, 0.0))
    return float(row[metric])


def _warm_up(
    generator: HFPrefixCacheGenerator,
    encoder: Any,
    method: Any,
    source: IndexedMemoryDataset,
    scenario_row_ids: Mapping[int, Sequence[int]],
    *,
    turns: int,
    append_max_turns: int,
    append_reset_on_discontinuity: bool,
) -> None:
    if turns == 0:
        return
    generator.reset()
    completed = 0
    for row_ids in scenario_row_ids.values():
        append_session = _append_session(
            method,
            encoder,
            max_history_turns=append_max_turns,
            reset_on_discontinuity=append_reset_on_discontinuity,
        )
        for row_id in row_ids:
            row = source[source.position_for_row_id(row_id)]
            if append_session is None:
                prompt = encoder.generation_prompt(row, method)
                cache_anchor_prefixes = encoder.cache_anchor_prefixes(row, method)
            else:
                input_state = state_from_input(method, row["input"])
                prepared = append_session.prepare(
                    row,
                    materialized_memory=method.materialize_memory(input_state),
                )
                if prepared.reset_before:
                    generator.reset()
                prompt = prepared.prompt
                cache_anchor_prefixes = ()
            generation = generator.generate(
                prompt,
                cache_anchor_prefixes=cache_anchor_prefixes,
            )
            if append_session is not None:
                append_session.commit(
                    assistant_output=generation.text,
                    decision_for_epoch=str(row["target"]["decision"]),
                )
            completed += 1
            if completed >= turns:
                generator.reset()
                return
    generator.reset()


def _append_session(
    method: Any,
    encoder: Any,
    *,
    max_history_turns: int,
    reset_on_discontinuity: bool,
) -> DeltaAppendPromptSession | None:
    if not isinstance(method, DeltaV3AppendMethod):
        return None
    if not hasattr(encoder, "generation_prompt_append"):
        raise TypeError("Delta-v3 append requires DeltaAppendChatExampleEncoder")
    return DeltaAppendPromptSession(
        method,
        encoder,
        max_history_turns=max_history_turns,
        reset_on_discontinuity=reset_on_discontinuity,
    )


def _append_record_fields(prepared: Any, committed: Any) -> dict[str, Any]:
    if prepared is None or committed is None:
        return {
            "cache_epoch_index": None,
            "cache_epoch_turn_index": None,
            "cache_epoch_updates_before": None,
            "cache_epoch_reset_before": None,
            "cache_epoch_reset_reason": None,
            "cache_epoch_end_after": None,
            "cache_epoch_end_reason": None,
        }
    return {
        "cache_epoch_index": prepared.epoch_index,
        "cache_epoch_turn_index": prepared.epoch_turn_index,
        "cache_epoch_updates_before": prepared.epoch_updates_before,
        "cache_epoch_reset_before": prepared.reset_before,
        "cache_epoch_reset_reason": prepared.reset_reason,
        "cache_epoch_end_after": committed.epoch_end_after,
        "cache_epoch_end_reason": committed.epoch_end_reason,
    }


def _should_background_prefill(*, append_commit: Any, applied_update: bool) -> bool:
    if append_commit is None:
        return applied_update
    return bool(append_commit.epoch_end_after)


def _background_record_fields(prefill: Any) -> dict[str, Any]:
    if prefill is None:
        return {
            "background_prefill_tokens": 0,
            "background_prefill_seconds": 0.0,
            "background_prefill_end_to_end_seconds": 0.0,
            "background_kv_cache_bytes": 0,
        }
    return {
        "background_prefill_tokens": prefill.evaluated_prefill_tokens,
        "background_prefill_seconds": prefill.prefill_seconds,
        "background_prefill_end_to_end_seconds": prefill.end_to_end_seconds,
        "background_kv_cache_bytes": prefill.kv_cache_bytes,
    }


def _combine_two_stage_generations(
    extraction: PrefixGenerationResult,
    manager: PrefixGenerationResult,
    *,
    retrieval_seconds: float,
) -> PrefixGenerationResult:
    """Combine EXTRACT and MANAGE into one turn-level model-cost record."""

    return PrefixGenerationResult(
        text=manager.text,
        logical_prompt_tokens=(
            extraction.logical_prompt_tokens + manager.logical_prompt_tokens
        ),
        candidate_prefix_tokens=(
            extraction.candidate_prefix_tokens + manager.candidate_prefix_tokens
        ),
        reused_prefix_tokens=(
            extraction.reused_prefix_tokens + manager.reused_prefix_tokens
        ),
        evaluated_prefill_tokens=(
            extraction.evaluated_prefill_tokens + manager.evaluated_prefill_tokens
        ),
        decode_tokens=extraction.decode_tokens + manager.decode_tokens,
        tokenization_seconds=(
            extraction.tokenization_seconds + manager.tokenization_seconds
        ),
        cache_management_seconds=(
            extraction.cache_management_seconds + manager.cache_management_seconds
        ),
        prefill_seconds=extraction.prefill_seconds + manager.prefill_seconds,
        decode_seconds=extraction.decode_seconds + manager.decode_seconds,
        model_seconds=extraction.model_seconds + manager.model_seconds,
        ttft_seconds=extraction.ttft_seconds + manager.ttft_seconds,
        end_to_end_seconds=(
            extraction.end_to_end_seconds
            + retrieval_seconds
            + manager.end_to_end_seconds
        ),
        kv_cache_bytes=extraction.kv_cache_bytes + manager.kv_cache_bytes,
        cache_length=extraction.cache_length + manager.cache_length,
        peak_cuda_allocated_bytes=max(
            extraction.peak_cuda_allocated_bytes,
            manager.peak_cuda_allocated_bytes,
        ),
    )


def _two_stage_noop_generation(
    extraction: PrefixGenerationResult,
) -> PrefixGenerationResult:
    """Represent an empty extraction as a valid manager-level NO_OP."""

    return PrefixGenerationResult(**{**asdict(extraction), "text": '{"memory":[]}'})


def _two_stage_error_generation(
    extraction: PrefixGenerationResult, error: Exception
) -> PrefixGenerationResult:
    """Preserve EXTRACT cost while making its parse failure explicit."""

    return PrefixGenerationResult(
        **{
            **asdict(extraction),
            "text": f"__TWO_STAGE_EXTRACT_ERROR__:{type(error).__name__}: {error}",
        }
    )


def _two_stage_record_fields(
    extraction: PrefixGenerationResult | None,
    manager: PrefixGenerationResult | None,
) -> dict[str, Any]:
    if extraction is None:
        return {
            "extract_invoked": False,
            "extract_logical_prompt_tokens": 0,
            "extract_evaluated_prefill_tokens": 0,
            "extract_decode_tokens": 0,
            "extract_model_seconds": 0.0,
            "extract_output": None,
            "manager_invoked": False,
            "manager_logical_prompt_tokens": 0,
            "manager_evaluated_prefill_tokens": 0,
            "manager_decode_tokens": 0,
            "manager_model_seconds": 0.0,
            "manager_output": None,
        }
    return {
        "extract_invoked": True,
        "extract_logical_prompt_tokens": extraction.logical_prompt_tokens,
        "extract_evaluated_prefill_tokens": extraction.evaluated_prefill_tokens,
        "extract_decode_tokens": extraction.decode_tokens,
        "extract_model_seconds": extraction.model_seconds,
        "extract_output": extraction.text,
        "manager_invoked": manager is not None,
        "manager_logical_prompt_tokens": (
            manager.logical_prompt_tokens if manager is not None else 0
        ),
        "manager_evaluated_prefill_tokens": (
            manager.evaluated_prefill_tokens if manager is not None else 0
        ),
        "manager_decode_tokens": manager.decode_tokens if manager is not None else 0,
        "manager_model_seconds": (
            manager.model_seconds if manager is not None else 0.0
        ),
        "manager_output": manager.text if manager is not None else None,
    }


def _mem0_gold_state_after(
    state: Mapping[str, Mapping[str, str]], canonical: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Advance the prepared Mem0 fact store with gold record identities."""

    result = {record_id: dict(fact) for record_id, fact in state.items()}
    target = canonical.get("target")
    provenance = canonical.get("provenance")
    if not isinstance(target, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError("Controlled Mem0 rows require target and provenance objects")
    events = target.get("memory")
    metadata = provenance.get("event_metadata")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TypeError("Controlled Mem0 target memory must be an array")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise TypeError("Controlled Mem0 event metadata must be an array")
    if len(events) != len(metadata):
        raise ValueError("Controlled Mem0 events and metadata are not aligned")
    for event, meta in zip(events, metadata, strict=True):
        if not isinstance(event, Mapping) or not isinstance(meta, Mapping):
            raise TypeError("Controlled Mem0 events and metadata must be objects")
        kind = event.get("event")
        record_id = str(meta.get("gold_record_id", event.get("id", "")))
        if not record_id:
            raise ValueError(f"Controlled Mem0 {kind} has no gold record id")
        if kind == "ADD":
            if record_id in result:
                raise ValueError(f"Duplicate controlled Mem0 ADD id: {record_id}")
            result[record_id] = {
                "subject": str(event["subject"]),
                "text": str(event["text"]),
            }
        elif kind == "UPDATE":
            if record_id not in result:
                raise ValueError(f"Unknown controlled Mem0 UPDATE id: {record_id}")
            result[record_id] = {
                "subject": str(event["subject"]),
                "text": str(event["text"]),
            }
        elif kind == "DELETE":
            if record_id not in result:
                raise ValueError(f"Unknown controlled Mem0 DELETE id: {record_id}")
            del result[record_id]
        else:
            raise ValueError(f"Unknown controlled Mem0 event: {kind!r}")

    gold_state = canonical.get("gold_state")
    if isinstance(gold_state, Mapping):
        active_after = gold_state.get("active_after")
        if isinstance(active_after, Sequence) and not isinstance(
            active_after, (str, bytes)
        ):
            expected = {
                str(version_id).rsplit(":v", 1)[0] for version_id in active_after
            }
            if set(result) != expected:
                raise ValueError(
                    "Controlled Mem0 state does not match gold active_after ids"
                )
    return result


def _pending_depth(method: Any, state: Any) -> int | None:
    return len(state.pending_updates) if isinstance(method, DeltaV2Method) else None


def _compare_reference(
    records: Sequence[Mapping[str, Any]], reference_path: Path | None
) -> dict[str, Any]:
    if reference_path is None:
        return {"checked": False}
    reference = list(_read_jsonl(reference_path.resolve()))
    keys = ("repetition", "scenario_index", "row_id")
    expected = {tuple(row[key] for key in keys): row for row in reference}
    mismatches = []
    compared_fields = (
        "output",
        "predicted_decision",
        "applied_update",
        "compacted",
        "state_sha256",
        "error",
    )
    for row in records:
        identity = tuple(row[key] for key in keys)
        other = expected.get(identity)
        differences = {
            key: {"reference": other.get(key), "candidate": row.get(key)}
            for key in compared_fields
            if other is not None and row.get(key) != other.get(key)
        }
        if other is None or differences:
            mismatches.append(
                {
                    "identity": identity,
                    "missing_reference": other is None,
                    "differences": differences,
                }
            )
    if len(expected) != len(records) or mismatches:
        raise ValueError(
            "Cache benchmark differs from its reference: "
            f"rows={len(records)}/{len(expected)}, "
            f"mismatches={json.dumps(mismatches[:10], ensure_ascii=False)}"
        )
    return {"checked": True, "reference": str(reference_path.resolve())}


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _signature(args: argparse.Namespace, manifest: Mapping[str, Any]) -> str:
    value = {
        "model": args.model,
        "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()),
        "turn_manifest_signature": manifest["signature"],
        "cache_mode": args.cache_mode,
        "replay_mode": args.replay_mode,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "repetitions": args.repetitions,
        "delta_v3_append_max_turns": args.delta_v3_append_max_turns,
        "append_reset_on_discontinuity": not args.append_allow_turn_gaps,
        "background_prefill": args.background_prefill,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object in {path}:{line_number}")
            yield value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "signature": result["signature"],
                "turns": result["metrics"]["all_turns"]["turns"],
                "cache_reuse_ratio": result["metrics"]["all_turns"][
                    "cache_reuse_ratio"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
