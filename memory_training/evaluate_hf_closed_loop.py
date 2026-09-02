"""Resumable HF/PEFT closed-loop Memory + Quiz evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .dataset import default_catalog_path, ensure_catalog
from .methods import METHODS
from .quiz_sft import (
    IndexedQuizSFTDataset,
    VehicleToolSchemaStore,
    sft_split_for_scenario,
)
from .validation import (
    closed_loop_quiz_snapshot_requests,
    evaluate_closed_loop_batched,
    evaluate_quiz_tool_calling,
    merge_decision_usage,
    quiz_indices_for_scenarios,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--vehicle-tools-path", type=Path)
    parser.add_argument("--quiz-sft-path", type=Path)
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
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = tuple(dict.fromkeys(args.scenarios))
    if not scenarios:
        raise ValueError("At least one Test scenario is required")
    if any(scenario < 1 or scenario > 120 for scenario in scenarios):
        raise ValueError("Scenario indices must be S1-S100 or encoded T1-T20")
    if args.scenario_batch_size < 1 or args.quiz_batch_size < 1:
        raise ValueError("Evaluation batch sizes must be positive")
    checkpoint = args.checkpoint.resolve()
    adapter = checkpoint / "adapter"
    if not adapter.is_dir():
        raise FileNotFoundError(f"Checkpoint adapter not found: {adapter}")

    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    scenario_dir = output_dir / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    _configure_runtime_paths(workspace)

    from accelerate import Accelerator

    from .models import load_peft_bundle

    catalog = ensure_catalog(
        data_root,
        (args.catalog_path or default_catalog_path(workspace)).resolve(),
    )
    for scenario in scenarios:
        if not catalog.scenario_row_ids(scenario):
            raise ValueError(f"Memory data is missing for S{scenario}")
    quiz_splits = {sft_split_for_scenario(scenario) for scenario in scenarios}
    if len(quiz_splits) != 1:
        raise ValueError(
            "One evaluation job cannot mix Quiz SFT splits: "
            f"{sorted(quiz_splits)}"
        )
    evaluation_split = quiz_splits.pop()
    quiz_source = IndexedQuizSFTDataset(
        args.quiz_sft_path or data_root / "quiz_sft.jsonl", split=evaluation_split
    )
    quiz_scenarios = {
        int(quiz_source[index]["scenario_index"]) for index in range(len(quiz_source))
    }
    missing_quizzes = sorted(set(scenarios) - quiz_scenarios)
    if missing_quizzes:
        raise ValueError(f"Quiz data is missing for scenarios: {missing_quizzes}")
    tools = VehicleToolSchemaStore(
        args.vehicle_tools_path or data_root / "vehicle_tools.json"
    )
    signature = _signature(args, checkpoint, scenarios, tools.sha256)
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    existing_progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {}
    )
    stored_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored_manifest.get("signature") != signature:
            raise ValueError(f"Stale Test output directory: {output_dir}")
    started_at = (
        stored_manifest.get("created_at")
        or (
            datetime.fromtimestamp(
                manifest_path.stat().st_mtime, timezone.utc
            ).isoformat()
            if stored_manifest
            else None
        )
        or existing_progress.get("started_at")
        or _utc_now()
    )
    if not stored_manifest:
        _write_json(
            manifest_path,
            {
                "schema_version": "palmclaw-hf-closed-loop-test-v1",
                "signature": signature,
                "model": args.model,
                "method": args.method,
                "checkpoint": str(checkpoint),
                "scenarios": list(scenarios),
                "quiz_split": evaluation_split,
                "memory_mode": "closed_loop_predicted_only",
                "gold_memory_evaluation": False,
                "created_at": started_at,
            },
        )

    restored = sorted(
        int(path.stem.removeprefix("s")) for path in scenario_dir.glob("s*.json")
    )
    _write_progress(
        progress_path,
        started_at=started_at,
        status="INITIALIZING",
        phase="model_loading",
        scenarios=scenarios,
        completed_scenarios=restored,
        current_scenario=next(
            (scenario for scenario in scenarios if scenario not in restored), None
        ),
        progress=len(restored) / len(scenarios),
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

    results_by_scenario: dict[int, dict[str, Any]] = {}
    for position, scenario in enumerate(scenarios, start=1):
        path = scenario_dir / f"s{scenario:03d}.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("signature") != signature:
                raise ValueError(f"Stale scenario checkpoint: {path}")
            results_by_scenario[scenario] = result
            print(
                f"[{position}/{len(scenarios)}] S{scenario} restored",
                flush=True,
            )
    pending = [
        scenario for scenario in scenarios if scenario not in results_by_scenario
    ]
    pending_batches = _scenario_batches(catalog, pending, args.scenario_batch_size)
    for batch_index, chunk in enumerate(pending_batches):
        print(
            f"Batch {chunk}: closed-loop started (batch={len(chunk)})",
            flush=True,
        )
        completed_before = sorted(results_by_scenario)

        def report_progress(
            payload: dict[str, Any],
            *,
            job_scenarios: list[int] = chunk,
            job_completed: list[int] = completed_before,
        ) -> None:
            phase = str(payload.get("phase", "closed_loop_memory"))
            item = int(payload.get("item", 0))
            items = max(int(payload.get("items", 0)), 1)
            chunk_fraction = (
                0.9 * item / items
                if phase == "closed_loop_memory"
                else 0.9 + 0.1 * item / items
            )
            overall = (len(job_completed) + len(job_scenarios) * chunk_fraction) / len(
                scenarios
            )
            _write_progress(
                progress_path,
                started_at=started_at,
                status="RUNNING",
                phase=phase,
                scenarios=scenarios,
                completed_scenarios=job_completed,
                current_scenario=job_scenarios[0],
                current_scenarios=job_scenarios,
                progress=overall,
                item=item,
                items=items,
            )

        quiz_indices = quiz_indices_for_scenarios(quiz_source, chunk)
        snapshot_requests = closed_loop_quiz_snapshot_requests(
            quiz_source, quiz_indices
        )
        splits = {
            catalog.record(catalog.scenario_row_ids(scenario)[0]).split
            for scenario in chunk
        }
        if len(splits) != 1:
            raise ValueError(f"Scenario batch crosses dataset splits: {chunk}")
        split = splits.pop()
        memory_by_scenario = evaluate_closed_loop_batched(
            model,
            bundle.tokenizer,
            method,
            catalog,
            chunk,
            accelerator=accelerator,
            split=split,
            batch_size=args.scenario_batch_size,
            snapshot_requests=snapshot_requests,
            progress_callback=report_progress,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        snapshots = {
            sample_id: memory
            for scenario in chunk
            for sample_id, memory in memory_by_scenario[scenario]
            .pop("quiz_snapshots")
            .items()
        }
        quiz_batch = evaluate_quiz_tool_calling(
            model,
            bundle.tokenizer,
            quiz_source,
            tools,
            quiz_indices,
            dataset_root=args.vehiclemembench_root,
            accelerator=accelerator,
            max_length=args.max_length,
            max_new_tokens=args.quiz_max_new_tokens,
            memory_overrides=snapshots,
            batch_size=args.quiz_batch_size,
            progress_callback=report_progress,
        )
        for scenario in chunk:
            memory = memory_by_scenario[scenario]
            quiz = _quiz_result_for_scenario(quiz_batch, scenario)
            result = {
                "signature": signature,
                "scenario_index": scenario,
                "memory_split_label": split,
                "execution": {
                    "scenario_batch_size": len(chunk),
                    "quiz_batch_size": args.quiz_batch_size,
                },
                "memory": memory,
                "closed_loop_quiz": quiz,
            }
            _write_json(scenario_dir / f"s{scenario:03d}.json", result)
            results_by_scenario[scenario] = result
            position = scenarios.index(scenario) + 1
            print(
                f"[{position}/{len(scenarios)}] S{scenario} complete: "
                f"ESM={quiz['esm']:.4f} UpdateF1={memory['update_f1']:.4f}",
                flush=True,
            )
        results = [
            results_by_scenario[value]
            for value in scenarios
            if value in results_by_scenario
        ]
        partial_summary = aggregate_results(results, scenarios)
        _write_json(output_dir / "summary.json", partial_summary)
        next_scenarios = (
            pending_batches[batch_index + 1]
            if batch_index + 1 < len(pending_batches)
            else []
        )
        _write_progress(
            progress_path,
            started_at=started_at,
            status="RUNNING" if next_scenarios else "COMPLETED",
            phase="scenario_complete" if next_scenarios else "complete",
            scenarios=scenarios,
            completed_scenarios=sorted(results_by_scenario),
            current_scenario=next_scenarios[0] if next_scenarios else None,
            current_scenarios=next_scenarios,
            progress=len(results_by_scenario) / len(scenarios),
            esm=float(partial_summary["closed_loop_quiz"]["esm"]),
            update_f1=float(partial_summary["memory"]["update_f1"]),
        )

    results = [results_by_scenario[value] for value in scenarios]
    summary = aggregate_results(results, scenarios)
    _write_json(output_dir / "summary.json", summary)
    return summary


def aggregate_results(
    results: Sequence[Mapping[str, Any]], expected_scenarios: Sequence[int]
) -> dict[str, Any]:
    counts: defaultdict[str, int] = defaultdict(int)
    reports = []
    quiz_records = []
    memory_prefill = memory_decode = quiz_prefill = quiz_decode = 0
    memory_latency = quiz_latency = 0.0
    turns = 0
    weighted_state_f1 = 0.0
    for result in results:
        memory = result["memory"]
        quiz = result["closed_loop_quiz"]
        for key, value in memory.get("decision_counts", {}).items():
            counts[key] += int(value)
        reports.extend(memory.get("scenario_reports", []))
        quiz_records.extend(quiz.get("records", []))
        scenario_turns = int(memory.get("turns", 0))
        turns += scenario_turns
        weighted_state_f1 += float(memory.get("state_f1", 0.0)) * scenario_turns
        memory_prefill += int(memory.get("prefill_tokens", 0))
        memory_decode += int(memory.get("decode_tokens", 0))
        memory_latency += float(memory.get("latency_seconds", 0.0))
        quiz_prefill += int(quiz.get("prefill_tokens", 0))
        quiz_decode += int(quiz.get("decode_tokens", 0))
        quiz_latency += float(quiz.get("latency_seconds", 0.0))
    decision = _decision_metrics(counts)
    quiz = _quiz_metrics(quiz_records)
    completed = sorted(int(result["scenario_index"]) for result in results)
    return {
        "schema_version": "palmclaw-hf-closed-loop-test-summary-v1",
        "memory_mode": "closed_loop_predicted_only",
        "gold_memory_evaluation": False,
        "expected_scenarios": list(expected_scenarios),
        "completed_scenarios": completed,
        "complete": completed == sorted(expected_scenarios),
        "memory": {
            **decision,
            **merge_decision_usage(
                result["memory"] for result in results
            ),
            "turns": turns,
            "state_f1": weighted_state_f1 / turns if turns else 0.0,
            "final_state_f1": _mean(
                float(report.get("final_state_f1", 0.0)) for report in reports
            ),
            "prefill_tokens": memory_prefill,
            "decode_tokens": memory_decode,
            "total_tokens": memory_prefill + memory_decode,
            "latency_seconds": memory_latency,
            "scenario_reports": reports,
        },
        "closed_loop_quiz": {
            **quiz,
            "prefill_tokens": quiz_prefill,
            "decode_tokens": quiz_decode,
            "total_tokens": quiz_prefill + quiz_decode,
            "latency_seconds": quiz_latency,
            "records": quiz_records,
        },
    }


def _scenario_batches(
    catalog: Any, scenarios: Sequence[int], batch_size: int
) -> list[list[int]]:
    """Batch adjacent scenarios without crossing canonical dataset splits."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_split = None
    for scenario in scenarios:
        split = catalog.record(catalog.scenario_row_ids(scenario)[0]).split
        if current and (split != current_split or len(current) >= batch_size):
            batches.append(current)
            current = []
        current.append(scenario)
        current_split = split
    if current:
        batches.append(current)
    return batches


def _decision_metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    tp = counts.get("true_update", 0)
    fp = counts.get("false_update", 0)
    fn = counts.get("missed_update", 0)
    tn = counts.get("true_noop", 0)
    invalid_noop = counts.get("invalid_noop", 0)
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
        "invalid_count": counts.get("invalid", 0),
        "decision_counts": dict(counts),
    }


def _quiz_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(records),
        "esm": _mean(float(record["exact_state_match"]) for record in records),
        "tool_f1": _mean(float(record["tool_f1"]) for record in records),
        "arg_exact": _mean(float(record["arg_exact"]) for record in records),
        "parse_success_rate": _mean(
            float(record["parse_success"]) for record in records
        ),
        "execution_success_rate": _mean(
            float(record["execution_success"]) for record in records
        ),
    }


def _quiz_result_for_scenario(
    combined: Mapping[str, Any], scenario: int
) -> dict[str, Any]:
    records = [
        dict(record)
        for record in combined.get("records", [])
        if int(record["scenario_index"]) == scenario
    ]
    prefill = sum(int(record.get("prefill_tokens", 0)) for record in records)
    decode = sum(int(record.get("decode_tokens", 0)) for record in records)
    latency = sum(float(record.get("latency_seconds", 0.0)) for record in records)
    return {
        **_quiz_metrics(records),
        "prefill_tokens": prefill,
        "decode_tokens": decode,
        "total_tokens": prefill + decode,
        "latency_seconds": latency,
        "records": records,
    }


def _mean(values: Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _signature(
    args: argparse.Namespace,
    checkpoint: Path,
    scenarios: Sequence[int],
    tools_sha256: str,
) -> str:
    weights = next((checkpoint / "adapter").glob("adapter_model.*"), None)
    payload = {
        "model": args.model,
        "method": args.method,
        "checkpoint": str(checkpoint),
        "adapter_size": weights.stat().st_size if weights else None,
        "adapter_mtime_ns": weights.stat().st_mtime_ns if weights else None,
        "scenarios": list(scenarios),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "quiz_max_new_tokens": args.quiz_max_new_tokens,
        "tools_sha256": tools_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_progress(
    path: Path,
    *,
    started_at: str,
    status: str,
    phase: str,
    scenarios: Sequence[int],
    completed_scenarios: Sequence[int],
    current_scenario: int | None,
    current_scenarios: Sequence[int] | None = None,
    progress: float,
    item: int | None = None,
    items: int | None = None,
    esm: float | None = None,
    update_f1: float | None = None,
) -> None:
    elapsed = max(
        0.0,
        (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ).total_seconds(),
    )
    progress = min(max(progress, 0.0), 1.0)
    eta = elapsed / progress * (1.0 - progress) if progress > 0 else None
    _write_json(
        path,
        {
            "schema_version": "palmclaw-evaluation-progress-v1",
            "updated_at": _utc_now(),
            "started_at": started_at,
            "status": status,
            "phase": phase,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "expected_scenarios": list(scenarios),
            "completed_scenarios": list(completed_scenarios),
            "current_scenario": current_scenario,
            "current_scenarios": list(current_scenarios or ()),
            "item": item,
            "items": items,
            "progress": progress,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "esm": esm,
            "update_f1": update_f1,
        },
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
