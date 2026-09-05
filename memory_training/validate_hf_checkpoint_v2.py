"""Run the fixed V2 Validation protocol against one saved HF/PEFT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT
from .dataset import IndexedMemoryDataset, default_catalog_path, ensure_catalog
from .methods import METHODS
from .quiz_sft import IndexedQuizSFTDataset, VehicleToolSchemaStore
from .training_data import ChatExampleEncoder, MethodSFTDataset, SFTCollator
from .validation import (
    closed_loop_quiz_snapshot_requests,
    evaluate_closed_loop,
    evaluate_closed_loop_batched,
    evaluate_one_step,
    evaluate_quiz_tool_calling,
    evaluate_teacher_forced_loss,
    merge_decision_usage,
    quiz_indices_for_scenarios,
    random_quiz_validation_indices,
    stratified_one_step_row_ids,
    stratified_teacher_forced_row_ids,
)

FULL_SCENARIOS = (83, 84)
SPARSE_SCENARIOS = (82, 85)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        help="Override the run seed used for deterministic evaluation sampling.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--scenario-batch-size", type=int, default=2)
    parser.add_argument(
        "--closed-loop-noop-per-update",
        type=float,
        help=(
            "Keep every UPDATE and this many deterministic NO_OP turns per "
            "UPDATE while preserving trajectory order."
        ),
    )
    parser.add_argument("--quiz-batch-size", type=int, default=16)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--one-step-batch-size", type=int, default=8)
    parser.add_argument("--validation-scenarios", nargs="+", type=int)
    parser.add_argument("--full-scenarios", nargs="+", type=int)
    parser.add_argument("--sparse-scenarios", nargs="+", type=int)
    parser.add_argument("--teacher-max-rows", type=int)
    parser.add_argument("--one-step-max-rows", type=int)
    parser.add_argument("--gold-quiz-rows", type=int)
    parser.add_argument(
        "--closed-loop-only",
        action="store_true",
        help=(
            "Evaluate every selected Validation scenario as one batched full "
            "trajectory with predicted-memory Quiz; skip teacher-forced, "
            "one-step, sparse, and Gold-memory diagnostics."
        ),
    )
    parser.add_argument(
        "--finalize-run",
        action="store_true",
        help="Mark an interrupted training run complete after successful validation.",
    )
    parser.add_argument(
        "--no-status-update",
        action="store_true",
        help="Do not overwrite the training run status during an analysis probe.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    run_dir = (args.workspace.resolve() / "runs" / args.run_id).resolve()
    if run_dir != output.parent or not run_dir.is_dir():
        raise ValueError("Output must be directly inside the selected run directory")
    if not (checkpoint / "adapter").is_dir():
        raise FileNotFoundError(f"Checkpoint adapter not found: {checkpoint}")
    if (
        min(
            args.scenario_batch_size,
            args.quiz_batch_size,
            args.teacher_batch_size,
            args.one_step_batch_size,
        )
        < 1
    ):
        raise ValueError("Validation batch sizes must be positive")
    if (
        args.closed_loop_noop_per_update is not None
        and args.closed_loop_noop_per_update < 0
    ):
        raise ValueError("Closed-loop NO_OP per UPDATE must be non-negative")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("Sampling requires temperature > 0 and 0 < top_p <= 1")
    config = _read_json(run_dir / "config.json")
    arguments = config["arguments"]
    model_key = str(config["model"]["key"])
    method_name = str(config["method"])
    seed = (
        args.evaluation_seed
        if args.evaluation_seed is not None
        else int(arguments.get("seed", 42))
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    max_length = int(arguments.get("max_length", 2048))
    max_new_tokens = int(arguments.get("max_new_tokens", 768))
    quiz_max_new_tokens = int(arguments.get("quiz_validation_max_new_tokens", 256))
    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    _configure_runtime_paths(workspace)

    from accelerate import Accelerator

    from .models import load_peft_bundle

    accelerator = Accelerator(mixed_precision="bf16")
    bundle = load_peft_bundle(
        model_key,
        adapter_path=checkpoint / "adapter",
        gradient_checkpointing=False,
        cache_dir=workspace / "cache" / "huggingface" / "hub",
    )
    model = accelerator.prepare(bundle.model)
    model.eval()
    method = METHODS[method_name]()
    catalog_path = (args.catalog_path or default_catalog_path(workspace)).resolve()
    catalog = ensure_catalog(data_root, catalog_path)

    validation_scenarios = tuple(
        args.validation_scenarios
        or arguments.get("validation_scenarios")
        or range(81, 86)
    )
    full_scenarios = tuple(
        args.full_scenarios
        or arguments.get("closed_loop_full_scenarios")
        or FULL_SCENARIOS
    )
    sparse_scenarios = tuple(
        args.sparse_scenarios
        or arguments.get("closed_loop_sparse_scenarios")
        or SPARSE_SCENARIOS
    )
    if args.closed_loop_only:
        full_scenarios = validation_scenarios
        sparse_scenarios = ()
    configured_one_step_rows = arguments.get("one_step_max_rows")
    one_step_max_rows = (
        args.one_step_max_rows
        if args.one_step_max_rows is not None
        else (
            int(configured_one_step_rows)
            if configured_one_step_rows is not None
            else None
        )
    )
    validation_row_ids = [
        row_id
        for scenario in validation_scenarios
        for row_id in catalog.scenario_row_ids(scenario)
    ]
    validation_source = IndexedMemoryDataset(
        catalog, method.source_view, row_ids=validation_row_ids
    )
    validation_summary = IndexedMemoryDataset(
        catalog, "summary", row_ids=validation_row_ids
    )
    teacher_row_ids = []
    teacher_loader = None
    if not args.closed_loop_only:
        teacher_row_ids = stratified_teacher_forced_row_ids(
            catalog,
            max_rows=(
                args.teacher_max_rows
                if args.teacher_max_rows is not None
                else int(arguments.get("eval_max_rows", 512))
            ),
            adjacent_noop_fraction=float(
                arguments.get("eval_adjacent_noop_fraction", 0.5)
            ),
            seed=seed,
            scenarios=validation_scenarios,
        )
        teacher_source = IndexedMemoryDataset(
            catalog, method.source_view, row_ids=teacher_row_ids
        )
        encoder = ChatExampleEncoder(bundle.tokenizer, max_length=max_length)
        teacher_loader = DataLoader(
            MethodSFTDataset(teacher_source, method, encoder),
            batch_size=args.teacher_batch_size,
            shuffle=False,
            collate_fn=SFTCollator(bundle.tokenizer.pad_token_id),
            num_workers=0,
        )
        teacher_loader = accelerator.prepare(teacher_loader)
    quiz_source = IndexedQuizSFTDataset(
        data_root / "quiz_sft.jsonl", split="validation"
    )
    quiz_tools = VehicleToolSchemaStore(data_root / "vehicle_tools.json")
    closed_quiz_indices = quiz_indices_for_scenarios(quiz_source, full_scenarios)
    gold_quiz_indices = []
    if not args.closed_loop_only:
        gold_quiz_indices = random_quiz_validation_indices(
            quiz_source,
            total_rows=(
                args.gold_quiz_rows
                if args.gold_quiz_rows is not None
                else int(arguments.get("gold_quiz_validation_rows", 50))
            ),
            excluded_scenarios=full_scenarios,
            seed=int(arguments.get("quiz_validation_seed", 42)),
        )
    snapshot_requests = closed_loop_quiz_snapshot_requests(
        quiz_source, closed_quiz_indices
    )
    epoch = _checkpoint_epoch(checkpoint)

    def report(payload: dict[str, Any]) -> None:
        if args.no_status_update:
            return
        _write_status(
            run_dir,
            state="VALIDATING",
            epoch=epoch,
            checkpoint=checkpoint,
            evaluation={
                "type": "VALIDATION_V2",
                "protocol": "run_config_scenarios_closed_loop_gold50",
                "epoch": epoch,
                **payload,
            },
        )

    if args.closed_loop_only:
        teacher = {"skipped": True, "reason": "closed_loop_only"}
        one_step_row_ids = []
        one_step = {"skipped": True, "reason": "closed_loop_only"}
    else:
        report({"phase": "teacher_forced", "item": 0, "items": len(teacher_row_ids)})
        teacher = evaluate_teacher_forced_loss(
            model, teacher_loader, accelerator=accelerator
        )
        one_step_row_ids = stratified_one_step_row_ids(
            catalog,
            max_rows=one_step_max_rows,
            seed=seed,
            scenarios=validation_scenarios,
        )
        report({"phase": "one_step", "item": 0, "items": len(one_step_row_ids)})
        one_step = evaluate_one_step(
            model,
            bundle.tokenizer,
            method,
            validation_source,
            validation_summary,
            one_step_row_ids,
            accelerator=accelerator,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            batch_size=args.one_step_batch_size,
        )
    full = evaluate_closed_loop_batched(
        model,
        bundle.tokenizer,
        method,
        catalog,
        full_scenarios,
        accelerator=accelerator,
        split="validation",
        batch_size=args.scenario_batch_size,
        noop_per_update=args.closed_loop_noop_per_update,
        sampling_seed=seed,
        snapshot_requests=snapshot_requests,
        progress_callback=report,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    snapshots = {
        sample_id: memory
        for scenario in full_scenarios
        for sample_id, memory in full[scenario].pop("quiz_snapshots").items()
    }
    memory_results = list(full.values())
    if sparse_scenarios:
        memory_results.append(
            evaluate_closed_loop(
                model,
                bundle.tokenizer,
                method,
                catalog,
                accelerator=accelerator,
                split="validation",
                full_scenarios=(),
                sparse_scenarios=sparse_scenarios,
                sparse_noop_keep_fraction=0.1,
                sparse_seed=seed,
                progress_callback=report,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
        )
    closed_loop = _aggregate_memory_results(memory_results)
    closed_quiz = evaluate_quiz_tool_calling(
        model,
        bundle.tokenizer,
        quiz_source,
        quiz_tools,
        closed_quiz_indices,
        dataset_root=Path(arguments["vehiclemembench_root"]),
        accelerator=accelerator,
        max_length=max_length,
        max_new_tokens=quiz_max_new_tokens,
        memory_overrides=snapshots,
        batch_size=args.quiz_batch_size,
        progress_callback=report,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.closed_loop_only:
        gold_quiz = {"skipped": True, "reason": "closed_loop_only"}
    else:
        gold_quiz = evaluate_quiz_tool_calling(
            model,
            bundle.tokenizer,
            quiz_source,
            quiz_tools,
            gold_quiz_indices,
            dataset_root=Path(arguments["vehiclemembench_root"]),
            accelerator=accelerator,
            max_length=max_length,
            max_new_tokens=quiz_max_new_tokens,
            batch_size=args.quiz_batch_size,
            progress_callback=report,
        )
    result = {
        "schema_version": "palmclaw-checkpoint-validation-v2",
        "created_at": _utc_now(),
        "run_id": args.run_id,
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "protocol": {
            "teacher_forced_rows": len(teacher_row_ids),
            "one_step_rows": len(one_step_row_ids),
            "validation_scenarios": list(validation_scenarios),
            "full_scenarios": list(full_scenarios),
            "sparse_scenarios": list(sparse_scenarios),
            "closed_loop_quiz_rows": len(closed_quiz_indices),
            "gold_quiz_rows": len(gold_quiz_indices),
            "scenario_batch_size": args.scenario_batch_size,
            "one_step_batch_size": args.one_step_batch_size,
            "quiz_batch_size": args.quiz_batch_size,
            "closed_loop_only": args.closed_loop_only,
            "closed_loop_noop_per_update": args.closed_loop_noop_per_update,
            "evaluation_seed": seed,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "teacher_forced": teacher,
        "one_step": one_step,
        "closed_loop": closed_loop,
        "closed_loop_quiz": closed_quiz,
        "quiz": gold_quiz,
    }
    _write_json(output, result)
    if args.finalize_run:
        _finalize_run(run_dir, checkpoint=checkpoint, result=result)
    return result


def _aggregate_memory_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: defaultdict[str, int] = defaultdict(int)
    reports = []
    turns = prefill = decode = 0
    state_exact_sum = state_f1_sum = latency = 0.0
    full_scenarios = []
    sparse_scenarios = []
    original_turns = evaluated_turns = 0
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
        original_turns += int(sampling.get("original_turns", 0))
        evaluated_turns += int(sampling.get("evaluated_turns", 0))
    tp = counts["true_update"]
    fp = counts["false_update"]
    fn = counts["missed_update"]
    tn = counts["true_noop"]
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
        "split": "validation",
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
            "sparse_noop_keep_fraction": 0.1,
            "original_turns": original_turns,
            "evaluated_turns": evaluated_turns,
        },
    }


def _write_status(
    run_dir: Path,
    *,
    state: str,
    epoch: int,
    checkpoint: Path,
    evaluation: Mapping[str, Any],
) -> None:
    current = _read_json(run_dir / "status.json")
    if current.get("global_step") is None:
        final_progress = _read_json(
            run_dir / "checkpoints" / "epoch-04" / "progress.json"
        )
        current["global_step"] = final_progress.get("global_step")
    current.pop("traceback", None)
    current.update(
        {
            "updated_at": _utc_now(),
            "state": state,
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "message": "Checkpoint V2 Validation in progress",
            "evaluation": dict(evaluation),
        }
    )
    _write_json(run_dir / "status.json", current)


def _finalize_run(
    run_dir: Path, *, checkpoint: Path, result: Mapping[str, Any]
) -> None:
    closed_quiz = result.get("closed_loop_quiz", {})
    closed = result.get("closed_loop", {})
    score = (
        0.60 * float(closed_quiz.get("esm", 0.0))
        + 0.25 * float(closed.get("final_state_f1", 0.0))
        + 0.15 * float(closed.get("update_f1", 0.0))
    )
    false_update_rate = float(result.get("one_step", {}).get("false_update_rate", 1.0))
    config = _read_json(run_dir / "config.json")
    threshold = float(config["arguments"].get("false_update_threshold", 0.15))
    best_path = run_dir / "best-checkpoint.json"
    best = _read_json(best_path) if best_path.exists() else {}
    if false_update_rate <= threshold and score > float(
        best.get("score", float("-inf"))
    ):
        _write_json(
            best_path,
            {
                "checkpoint": str(checkpoint),
                "score": score,
                "metric": "composite.quiz_esm_60.final_state_f1_25.update_f1_15",
                "false_update_rate": false_update_rate,
            },
        )

    status = _read_json(run_dir / "status.json")
    status.pop("traceback", None)
    status.pop("evaluation", None)
    status.update(
        {
            "updated_at": _utc_now(),
            "state": "COMPLETED",
            "epoch": _checkpoint_epoch(checkpoint),
            "checkpoint": str(checkpoint),
            "message": "Training complete; interrupted final Validation recovered",
        }
    )
    _write_json(run_dir / "status.json", status)


def _checkpoint_epoch(checkpoint: Path) -> int:
    try:
        return int(checkpoint.name.removeprefix("epoch-"))
    except ValueError as exc:
        raise ValueError(f"Expected epoch checkpoint, got: {checkpoint}") from exc


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
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple checkpoint validators can report progress to the same run at
    # once. Give each writer its own temporary file before the atomic replace.
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _mean(values: Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "epoch": result["epoch"],
                "closed_loop_quiz_esm": result["closed_loop_quiz"]["esm"],
                "closed_loop_update_f1": result["closed_loop"]["update_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
