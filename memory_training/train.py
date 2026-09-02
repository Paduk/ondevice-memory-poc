"""HF/PEFT SFT runner for the four on-device memory methods."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoints import (
    TrainingProgress,
    load_progress,
    restore_training_state,
    save_checkpoint,
)
from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORKSPACE_ROOT,
    MODEL_BY_KEY,
    split_for_scenario,
)
from .dataset import IndexedMemoryDataset, default_catalog_path, ensure_catalog
from .methods import METHODS
from .models import load_peft_bundle
from .quiz_sft import (
    IndexedQuizSFTDataset,
    VehicleToolSchemaStore,
    sft_split_for_scenario,
)
from .sampling import EpochSampler, SamplingConfig
from .tracking import MLflowMirror, RunTracker
from .training_data import (
    BalancedMultitaskBatchSampler,
    ChatExampleEncoder,
    MethodSFTDataset,
    MultitaskSFTDataset,
    PlannedBatchSampler,
    QuizChatExampleEncoder,
    QuizSFTDataset,
    SFTCollator,
    quiz_epoch_indices,
)
from .validation import (
    aggregate_closed_loop_results,
    closed_loop_quiz_snapshot_requests,
    evaluate_closed_loop,
    evaluate_closed_loop_batched,
    evaluate_one_step,
    evaluate_quiz_tool_calling,
    evaluate_teacher_forced_loss,
    quiz_indices_for_scenarios,
    random_quiz_validation_indices,
    run_quiz_hook,
    stratified_one_step_row_ids,
    stratified_quiz_validation_indices,
    stratified_teacher_forced_row_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--catalog-path",
        type=Path,
        help="Optional isolated dataset catalog (useful for alternate data views).",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Pause after this epoch's checkpoint and Validation; LR still uses --epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--multitask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alternate Memory and Tool-Calling Quiz batches during SFT.",
    )
    parser.add_argument(
        "--quiz-total-passes",
        type=int,
        default=2,
        help="Exact number of full Quiz-dataset exposures across all epochs.",
    )
    parser.add_argument("--quiz-batch-size", type=int)
    parser.add_argument("--quiz-sft-path", type=Path)
    parser.add_argument("--vehicle-tools-path", type=Path)
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--quiz-validation-rows-per-scenario", type=int, default=10)
    parser.add_argument(
        "--gold-quiz-validation-rows",
        type=int,
        default=50,
        help="Gold-memory Quiz rows when closed-loop Quiz Validation is enabled.",
    )
    parser.add_argument(
        "--quiz-validation-seed",
        type=int,
        default=42,
        help="Fixed Quiz Validation seed, independent of the training seed.",
    )
    parser.add_argument("--quiz-validation-max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-quiz-validation", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-seed",
        type=int,
        help=(
            "Optional model/dropout RNG seed. When set, --seed still controls "
            "training-data sampling and Validation sampling."
        ),
    )
    parser.add_argument(
        "--train-scenarios",
        nargs="+",
        type=int,
        help="Exact encoded scenario IDs used for Memory training.",
    )
    parser.add_argument(
        "--validation-scenarios",
        nargs="+",
        type=int,
        help="Exact encoded scenario IDs used for epoch Validation.",
    )
    parser.add_argument("--noop-per-update", type=int, default=10)
    parser.add_argument(
        "--delta-v3-noop-pending-weights",
        nargs=5,
        type=int,
        default=(40, 30, 15, 10, 5),
        metavar=("P0", "P1", "P2", "P3", "P4"),
        help="Delta-v3 independent NO_OP sampling weights by pending depth 0..4.",
    )
    parser.add_argument("--adjacent-noop-fraction", type=float, default=0.30)
    parser.add_argument("--trajectory-fraction", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-max-rows", type=int, default=512)
    parser.add_argument("--eval-adjacent-noop-fraction", type=float, default=0.50)
    parser.add_argument("--eval-max-batches", type=int)
    parser.add_argument("--one-step-max-rows", type=int)
    parser.add_argument("--one-step-validation-batch-size", type=int, default=1)
    parser.add_argument("--closed-loop-scenario-limit", type=int)
    parser.add_argument("--closed-loop-turn-limit", type=int)
    parser.add_argument("--closed-loop-full-scenarios", nargs="+", type=int)
    parser.add_argument("--closed-loop-sparse-scenarios", nargs="+", type=int)
    parser.add_argument("--closed-loop-scenario-batch-size", type=int, default=1)
    parser.add_argument(
        "--closed-loop-quiz",
        action="store_true",
        help="Run every Quiz for full closed-loop scenarios with predicted memory.",
    )
    parser.add_argument(
        "--closed-loop-sparse-noop-keep-fraction", type=float, default=0.10
    )
    parser.add_argument("--quiz-validation-batch-size", type=int, default=1)
    parser.add_argument("--false-update-threshold", type=float, default=0.20)
    parser.add_argument("--skip-generation-validation", action="store_true")
    parser.add_argument("--quiz-hook")
    parser.add_argument("--mlflow-uri")
    parser.add_argument("--mlflow-experiment", default="palmclaw-memory-sft")
    parser.add_argument("--max-train-steps", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    _configure_runtime_paths(workspace)

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import get_cosine_schedule_with_warmup

    set_seed(args.training_seed if args.training_seed is not None else args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision="bf16",
    )
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "This runner intentionally uses one process per model run; launch separate "
            "runs on separate GPUs for the 4x4 matrix."
        )
    run_id = args.run_id or _default_run_id(args.model, args.method)
    run_dir = workspace / "runs" / run_id
    tracker = RunTracker(run_dir)
    mlflow = MLflowMirror(
        args.mlflow_uri,
        experiment_name=args.mlflow_experiment,
        run_name=run_id,
    )
    catalog = ensure_catalog(
        data_root,
        (args.catalog_path or default_catalog_path(workspace)).resolve(),
    )
    method = METHODS[args.method]()
    train_source = IndexedMemoryDataset(catalog, method.source_view, split="train")
    noop_strata = (
        _delta_noop_pending_depths(catalog)
        if args.method == "delta_v3"
        else None
    )
    sampling_config = SamplingConfig(
        noop_per_update=args.noop_per_update,
        adjacent_noop_fraction=args.adjacent_noop_fraction,
        trajectory_fraction=args.trajectory_fraction,
        train_scenario_min=15 if args.multitask else 1,
        train_scenario_max=80,
        train_scenarios=tuple(args.train_scenarios or ()),
        noop_stratum_weights=(
            tuple(args.delta_v3_noop_pending_weights)
            if args.method == "delta_v3"
            else ()
        ),
        seed=args.seed,
    )
    epoch_sampler = EpochSampler(catalog, sampling_config, noop_strata=noop_strata)
    plans = [epoch_sampler.build(epoch) for epoch in range(args.epochs)]
    quiz_source = None
    quiz_batch_size = args.quiz_batch_size or args.batch_size
    if args.multitask:
        quiz_source = IndexedQuizSFTDataset(
            args.quiz_sft_path or data_root / "quiz_sft.jsonl", split="train"
        )
    quiz_schedules = (
        [
            quiz_epoch_indices(
                len(quiz_source),
                epoch=epoch,
                epochs=args.epochs,
                total_passes=args.quiz_total_passes,
                seed=args.seed,
            )
            for epoch in range(args.epochs)
        ]
        if quiz_source is not None
        else [None] * args.epochs
    )
    total_batches = sum(
        _epoch_batch_count(
            train_source,
            plan,
            batch_size=args.batch_size,
            quiz_size=len(quiz_source) if quiz_source is not None else None,
            quiz_indices=quiz_schedules[epoch],
            quiz_batch_size=quiz_batch_size,
        )
        for epoch, plan in enumerate(plans)
    )
    total_optimizer_steps = math.ceil(total_batches / args.gradient_accumulation)
    if args.max_train_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.max_train_steps)

    resume_progress = load_progress(args.resume) if args.resume else TrainingProgress()
    adapter_path = args.resume / "adapter" if args.resume else None
    bundle = load_peft_bundle(
        args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        adapter_path=adapter_path,
        gradient_checkpointing=args.gradient_checkpointing,
        cache_dir=workspace / "cache" / "huggingface" / "hub",
    )
    encoder = ChatExampleEncoder(bundle.tokenizer, max_length=args.max_length)
    collator = SFTCollator(bundle.tokenizer.pad_token_id)
    memory_train_dataset = MethodSFTDataset(train_source, method, encoder)
    quiz_tools = None
    if quiz_source is not None:
        quiz_tools = VehicleToolSchemaStore(
            args.vehicle_tools_path or data_root / "vehicle_tools.json"
        )
        quiz_encoder = QuizChatExampleEncoder(
            bundle.tokenizer, max_length=args.max_length
        )
        quiz_dataset = QuizSFTDataset(quiz_source, quiz_tools, quiz_encoder)
        train_dataset = MultitaskSFTDataset(memory_train_dataset, quiz_dataset)
    else:
        train_dataset = memory_train_dataset
    quiz_validation_source = None
    quiz_validation_indices = None
    closed_loop_quiz_indices = None
    if args.multitask and not args.skip_quiz_validation:
        quiz_validation_source = IndexedQuizSFTDataset(
            args.quiz_sft_path or data_root / "quiz_sft.jsonl", split="validation"
        )
        if args.closed_loop_quiz:
            closed_loop_quiz_indices = quiz_indices_for_scenarios(
                quiz_validation_source,
                args.closed_loop_full_scenarios or (),
            )
            quiz_validation_indices = (
                random_quiz_validation_indices(
                    quiz_validation_source,
                    total_rows=args.gold_quiz_validation_rows,
                    excluded_scenarios=args.closed_loop_full_scenarios or (),
                    seed=args.quiz_validation_seed,
                )
                if args.gold_quiz_validation_rows > 0
                else None
            )
        else:
            quiz_validation_indices = stratified_quiz_validation_indices(
                quiz_validation_source,
                rows_per_scenario=args.quiz_validation_rows_per_scenario,
                seed=args.quiz_validation_seed,
            )
    validation_scenario_max = 85 if args.multitask else 90
    validation_scenarios = tuple(
        args.validation_scenarios
        or range(81, validation_scenario_max + 1)
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
    teacher_forced_row_ids = stratified_teacher_forced_row_ids(
        catalog,
        max_rows=args.eval_max_rows,
        adjacent_noop_fraction=args.eval_adjacent_noop_fraction,
        seed=args.seed,
        scenarios=validation_scenarios,
    )
    teacher_forced_source = IndexedMemoryDataset(
        catalog, method.source_view, row_ids=teacher_forced_row_ids
    )
    validation_dataset = MethodSFTDataset(teacher_forced_source, method, encoder)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **_loader_worker_kwargs(args),
    )
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in bundle.model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_optimizer_steps * args.warmup_ratio),
        num_training_steps=max(1, total_optimizer_steps),
    )
    model, optimizer, scheduler, validation_loader = accelerator.prepare(
        bundle.model, optimizer, scheduler, validation_loader
    )
    if args.resume:
        restore_training_state(args.resume, optimizer=optimizer, scheduler=scheduler)
    progress = resume_progress
    configuration = {
        "schema_version": (
            "palmclaw-memory-tool-multitask-run-v1"
            if args.multitask
            else "palmclaw-memory-training-run-v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model": asdict(bundle.spec),
        "method": args.method,
        "arguments": _jsonable_args(args),
        "sampling": asdict(sampling_config),
        "validation_sampling": {
            "teacher_forced_rows": len(teacher_forced_row_ids),
            "teacher_forced_update_rows": sum(
                catalog.record(row_id).decision == "UPDATE"
                for row_id in teacher_forced_row_ids
            ),
            "teacher_forced_noop_rows": sum(
                catalog.record(row_id).decision == "NO_OP"
                for row_id in teacher_forced_row_ids
            ),
            "adjacent_noop_fraction": args.eval_adjacent_noop_fraction,
            "seed": args.seed,
            "quiz_seed": args.quiz_validation_seed,
            "memory_scenarios": list(validation_scenarios),
            "quiz_rows": len(quiz_validation_indices or ()),
            "quiz_scenarios": sorted(
                {
                    int(quiz_validation_source[index]["scenario_index"])
                    for index in (quiz_validation_indices or ())
                }
            )
            if quiz_validation_source is not None
            else [],
            "closed_loop_quiz_rows": len(closed_loop_quiz_indices or ()),
            "closed_loop_quiz_scenarios": list(
                args.closed_loop_full_scenarios or ()
            )
            if args.closed_loop_quiz
            else [],
        },
        "epoch_plans": [plan.summary() for plan in plans],
        "multitask": (
            {
                "enabled": True,
                "quiz_rows": len(quiz_source),
                "quiz_sft_path": str(quiz_source.path),
                "vehicle_tools_path": str(quiz_tools.path),
                "vehicle_tools_sha256": quiz_tools.sha256,
                "quiz_total_passes": args.quiz_total_passes,
                "quiz_batch_size": quiz_batch_size,
                "quiz_rows_per_epoch": [
                    len(schedule) for schedule in quiz_schedules if schedule is not None
                ],
            }
            if quiz_source is not None and quiz_tools is not None
            else {"enabled": False}
        ),
        "catalog": catalog.metadata(),
        "lora_targets": bundle.lora_targets,
        "total_optimizer_steps": total_optimizer_steps,
    }
    if accelerator.is_main_process and not args.resume:
        tracker.initialize(configuration)
        mlflow.parameters(
            {
                "model": args.model,
                "method": args.method,
                "seed": args.seed,
                "training_seed": (
                    args.training_seed
                    if args.training_seed is not None
                    else args.seed
                ),
                "learning_rate": args.learning_rate,
                "lora_rank": args.lora_rank,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "multitask": args.multitask,
                "quiz_total_passes": args.quiz_total_passes if args.multitask else 0,
                "dataset_fingerprint": catalog.metadata().get("source_fingerprint"),
            }
        )
    elif accelerator.is_main_process:
        tracker.resume(
            configuration,
            checkpoint=args.resume,
            global_step=progress.global_step,
        )
    if mlflow.error:
        tracker.metric("mlflow_unavailable", error=mlflow.error)

    optimizer.zero_grad()
    recent_loss = 0.0
    recent_batches = 0
    recent_target_tokens = 0
    recent_task_loss = {"memory": 0.0, "quiz": 0.0}
    recent_task_batches = {"memory": 0, "quiz": 0}
    log_started_at = time.perf_counter()
    stopped_early = False
    paused_after_epoch = bool(
        args.stop_after_epoch is not None and progress.epoch >= args.stop_after_epoch
    )
    for epoch in range(progress.epoch, args.epochs):
        if paused_after_epoch:
            break
        if (
            args.max_train_steps is not None
            and progress.global_step >= args.max_train_steps
        ):
            stopped_early = True
            break
        epoch_eta_started_at = time.perf_counter()
        epoch_eta_start_step = progress.global_step
        start_batch = progress.next_batch if epoch == progress.epoch else 0
        memory_batch_sampler = PlannedBatchSampler(
            train_source, plans[epoch], batch_size=args.batch_size
        )
        if quiz_source is not None:
            batch_sampler = BalancedMultitaskBatchSampler(
                memory_batch_sampler,
                memory_size=len(memory_train_dataset),
                quiz_size=len(quiz_source),
                quiz_indices=quiz_schedules[epoch] or (),
                quiz_batch_size=quiz_batch_size,
                start_batch=start_batch,
            )
        else:
            batch_sampler = PlannedBatchSampler(
                train_source,
                plans[epoch],
                batch_size=args.batch_size,
                start_batch=start_batch,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            **_loader_worker_kwargs(args),
        )
        train_loader = accelerator.prepare(train_loader)
        model.train()
        for relative_batch, batch in enumerate(train_loader):
            absolute_batch = start_batch + relative_batch
            metadata = batch.pop("metadata", [])
            task_types = {str(item.get("task_type", "memory")) for item in metadata}
            if len(task_types) != 1:
                raise ValueError(f"Expected a homogeneous task batch, got {task_types}")
            task_type = next(iter(task_types))
            recent_target_tokens += int((batch["labels"] != -100).sum().cpu())
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            recent_loss += float(loss.detach().cpu())
            recent_batches += 1
            if task_type in recent_task_loss:
                recent_task_loss[task_type] += float(loss.detach().cpu())
                recent_task_batches[task_type] += 1
            progress.epoch = epoch
            progress.next_batch = absolute_batch + 1
            if accelerator.sync_gradients:
                progress.global_step += 1
                if progress.global_step % args.log_steps == 0:
                    train_metrics = {
                        "loss": recent_loss / recent_batches,
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "target_tokens_per_second": recent_target_tokens
                        / max(time.perf_counter() - log_started_at, 1e-9),
                        "peak_cuda_memory_mib": (
                            torch.cuda.max_memory_allocated(accelerator.device)
                            / 1024**2
                            if accelerator.device.type == "cuda"
                            else 0.0
                        ),
                        "eta_seconds": (
                            (time.perf_counter() - epoch_eta_started_at)
                            / max(progress.global_step - epoch_eta_start_step, 1)
                            * max(total_optimizer_steps - progress.global_step, 0)
                        ),
                    }
                    for name in ("memory", "quiz"):
                        if recent_task_batches[name]:
                            train_metrics[f"{name}_loss"] = (
                                recent_task_loss[name] / recent_task_batches[name]
                            )
                            train_metrics[f"{name}_batches"] = recent_task_batches[name]
                    tracker.metric(
                        "train",
                        global_step=progress.global_step,
                        epoch=epoch,
                        batch=progress.next_batch,
                        **train_metrics,
                    )
                    mlflow.metrics(
                        {f"train_{key}": value for key, value in train_metrics.items()},
                        step=progress.global_step,
                    )
                    tracker.status(
                        "RUNNING",
                        global_step=progress.global_step,
                        epoch=epoch,
                        batch=progress.next_batch,
                    )
                    recent_loss = 0.0
                    recent_batches = 0
                    recent_target_tokens = 0
                    recent_task_loss = {"memory": 0.0, "quiz": 0.0}
                    recent_task_batches = {"memory": 0, "quiz": 0}
                    log_started_at = time.perf_counter()
                if args.eval_steps and progress.global_step % args.eval_steps == 0:
                    result = evaluate_teacher_forced_loss(
                        model,
                        validation_loader,
                        accelerator=accelerator,
                        max_batches=args.eval_max_batches,
                    )
                    tracker.metric(
                        "validation_teacher_forced",
                        global_step=progress.global_step,
                        epoch=epoch,
                        **result,
                    )
                    mlflow.metrics(
                        {
                            "validation_teacher_forced_loss": result["loss"],
                            "validation_teacher_forced_perplexity": result[
                                "perplexity"
                            ],
                        },
                        step=progress.global_step,
                    )
                if args.save_steps and progress.global_step % args.save_steps == 0:
                    save_checkpoint(
                        run_dir / "checkpoints" / f"step-{progress.global_step:07d}",
                        accelerator=accelerator,
                        model=model,
                        tokenizer=bundle.tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        progress=progress,
                    )
                if (
                    args.max_train_steps is not None
                    and progress.global_step >= args.max_train_steps
                ):
                    stopped_early = True
                    break
        checkpoint_dir = run_dir / "checkpoints" / f"epoch-{epoch + 1:02d}"
        progress.epoch = epoch + 1
        progress.next_batch = 0
        save_checkpoint(
            checkpoint_dir,
            accelerator=accelerator,
            model=model,
            tokenizer=bundle.tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
        )
        tracker.status(
            "VALIDATING",
            global_step=progress.global_step,
            epoch=progress.epoch,
            checkpoint=str(checkpoint_dir),
        )

        validation_checkpoint = str(checkpoint_dir)
        validation_step = progress.global_step
        validation_epoch = progress.epoch

        def report_validation_progress(
            payload: dict[str, Any],
            *,
            step: int = validation_step,
            validation_number: int = validation_epoch,
            checkpoint: str = validation_checkpoint,
        ) -> None:
            tracker.status(
                "VALIDATING",
                global_step=step,
                epoch=validation_number,
                checkpoint=checkpoint,
                evaluation={
                    "type": "VALIDATION",
                    "epoch": validation_number,
                    **payload,
                },
            )

        validation = _epoch_validation(
            args,
            model=model,
            tokenizer=bundle.tokenizer,
            method=method,
            catalog=catalog,
            validation_source=validation_source,
            validation_summary=validation_summary,
            validation_scenarios=validation_scenarios,
            validation_loader=validation_loader,
            quiz_validation_source=quiz_validation_source,
            quiz_validation_indices=quiz_validation_indices,
            closed_loop_quiz_indices=closed_loop_quiz_indices,
            quiz_tools=quiz_tools,
            accelerator=accelerator,
            checkpoint_dir=checkpoint_dir,
            progress_callback=report_validation_progress,
        )
        validation_path = tracker.write_json(
            f"validation-epoch-{epoch + 1:02d}.json", validation
        )
        tracker.metric(
            "validation_epoch",
            global_step=progress.global_step,
            epoch=epoch,
            **_flatten_primary_metrics(validation),
        )
        mlflow.metrics(_flatten_primary_metrics(validation), step=progress.global_step)
        mlflow.artifact(validation_path)
        score, metric = _selection_score(validation)
        false_update = validation.get("one_step", {}).get("false_update_rate", 0.0)
        eligible = false_update <= args.false_update_threshold
        if eligible and (progress.best_score is None or score > progress.best_score):
            progress.best_score = score
            progress.best_metric = metric
            progress.best_checkpoint = str(checkpoint_dir)
            tracker.write_json(
                "best-checkpoint.json",
                {
                    "checkpoint": str(checkpoint_dir),
                    "score": score,
                    "metric": metric,
                    "false_update_rate": false_update,
                },
            )
        save_checkpoint(
            checkpoint_dir,
            accelerator=accelerator,
            model=model,
            tokenizer=bundle.tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
        )
        if (
            args.stop_after_epoch is not None
            and progress.epoch >= args.stop_after_epoch
        ):
            paused_after_epoch = True
            break
        if stopped_early:
            break

    if paused_after_epoch:
        final_state = "PAUSED"
    elif stopped_early:
        final_state = "CANARY_COMPLETE"
    else:
        final_state = "COMPLETED"
    result = {
        "run_id": run_id,
        "final_state": final_state,
        "global_step": progress.global_step,
        "best_checkpoint": progress.best_checkpoint,
        "best_score": progress.best_score,
        "best_metric": progress.best_metric,
    }
    tracker.status(final_state, **result)
    mlflow.close()
    return result


def _epoch_validation(
    args: argparse.Namespace,
    *,
    model: Any,
    tokenizer: Any,
    method: Any,
    catalog: Any,
    validation_source: Any,
    validation_summary: Any,
    validation_scenarios: Any,
    validation_loader: Any,
    quiz_validation_source: Any,
    quiz_validation_indices: Any,
    closed_loop_quiz_indices: Any,
    quiz_tools: Any,
    accelerator: Any,
    checkpoint_dir: Path,
    progress_callback: Any = None,
) -> dict[str, Any]:
    if progress_callback is not None:
        progress_callback({"phase": "teacher_forced", "item": 0, "items": 1})
    result = {
        "teacher_forced": evaluate_teacher_forced_loss(
            model,
            validation_loader,
            accelerator=accelerator,
            max_batches=args.eval_max_batches,
        )
    }
    if progress_callback is not None:
        progress_callback({"phase": "teacher_forced", "item": 1, "items": 1})
    if not args.skip_generation_validation:
        row_ids = stratified_one_step_row_ids(
            catalog,
            max_rows=args.one_step_max_rows,
            seed=args.seed,
            scenarios=validation_scenarios,
        )
        if progress_callback is not None:
            progress_callback(
                {"phase": "one_step", "item": 0, "items": len(row_ids)}
            )
        result["one_step"] = evaluate_one_step(
            model,
            tokenizer,
            method,
            validation_source,
            validation_summary,
            row_ids,
            accelerator=accelerator,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.one_step_validation_batch_size,
        )
        if progress_callback is not None:
            progress_callback(
                {"phase": "one_step", "item": len(row_ids), "items": len(row_ids)}
            )
        snapshot_requests = None
        if (
            args.closed_loop_quiz
            and quiz_validation_source is not None
            and closed_loop_quiz_indices is not None
        ):
            snapshot_requests = closed_loop_quiz_snapshot_requests(
                quiz_validation_source,
                closed_loop_quiz_indices,
            )
        use_scenario_batching = (
            args.closed_loop_scenario_batch_size > 1
            and bool(args.closed_loop_full_scenarios)
            and args.closed_loop_scenario_limit is None
            and args.closed_loop_turn_limit is None
        )
        if use_scenario_batching:
            full_results = evaluate_closed_loop_batched(
                model,
                tokenizer,
                method,
                catalog,
                args.closed_loop_full_scenarios,
                accelerator=accelerator,
                split="validation",
                batch_size=args.closed_loop_scenario_batch_size,
                snapshot_requests=snapshot_requests,
                progress_callback=progress_callback,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            snapshots = {
                sample_id: memory
                for result_part in full_results.values()
                for sample_id, memory in result_part.pop(
                    "quiz_snapshots", {}
                ).items()
            }
            parts = list(full_results.values())
            if args.closed_loop_sparse_scenarios:
                parts.append(
                    evaluate_closed_loop(
                        model,
                        tokenizer,
                        method,
                        catalog,
                        accelerator=accelerator,
                        full_scenarios=(),
                        sparse_scenarios=args.closed_loop_sparse_scenarios,
                        sparse_noop_keep_fraction=(
                            args.closed_loop_sparse_noop_keep_fraction
                        ),
                        sparse_seed=args.seed,
                        progress_callback=progress_callback,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
            closed_loop = aggregate_closed_loop_results(parts)
            if snapshot_requests is not None:
                closed_loop["quiz_snapshots"] = snapshots
        else:
            closed_loop = evaluate_closed_loop(
                model,
                tokenizer,
                method,
                catalog,
                accelerator=accelerator,
                scenario_min=None,
                scenario_max=None,
                scenario_limit=args.closed_loop_scenario_limit,
                turn_limit_per_scenario=args.closed_loop_turn_limit,
                full_scenarios=args.closed_loop_full_scenarios,
                sparse_scenarios=args.closed_loop_sparse_scenarios,
                sparse_noop_keep_fraction=args.closed_loop_sparse_noop_keep_fraction,
                sparse_seed=args.seed,
                snapshot_requests=snapshot_requests,
                progress_callback=progress_callback,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
        snapshots = closed_loop.pop("quiz_snapshots", None)
        result["closed_loop"] = closed_loop
        if snapshots is not None:
            result["closed_loop_quiz"] = evaluate_quiz_tool_calling(
                model,
                tokenizer,
                quiz_validation_source,
                quiz_tools,
                closed_loop_quiz_indices,
                dataset_root=args.vehiclemembench_root,
                accelerator=accelerator,
                max_length=args.max_length,
                max_new_tokens=args.quiz_validation_max_new_tokens,
                memory_overrides=snapshots,
                batch_size=args.quiz_validation_batch_size,
                progress_callback=progress_callback,
            )
    if (
        quiz_validation_source is not None
        and quiz_validation_indices is not None
        and quiz_tools is not None
    ):
        result["quiz"] = evaluate_quiz_tool_calling(
            model,
            tokenizer,
            quiz_validation_source,
            quiz_tools,
            quiz_validation_indices,
            dataset_root=args.vehiclemembench_root,
            accelerator=accelerator,
            max_length=args.max_length,
            max_new_tokens=args.quiz_validation_max_new_tokens,
            batch_size=args.quiz_validation_batch_size,
            progress_callback=progress_callback,
        )
    if args.quiz_hook:
        result["quiz_hook"] = run_quiz_hook(
            args.quiz_hook,
            {
                "checkpoint": str(checkpoint_dir / "adapter"),
                "model": args.model,
                "method": args.method,
                "split": "validation",
            },
            checkpoint_dir / "quiz-validation.json",
        )
    return result


def _selection_score(validation: dict[str, Any]) -> tuple[float, str]:
    closed_quiz = validation.get("closed_loop_quiz", {})
    if "esm" in closed_quiz:
        tie_break = float(validation.get("closed_loop", {}).get("final_state_f1", 0.0))
        return (
            float(closed_quiz["esm"]) + tie_break * 1e-6,
            "closed_loop_quiz.esm_then_final_state_f1",
        )
    quiz = validation.get("quiz", {})
    if "esm" in quiz:
        tie_break = float(validation.get("closed_loop", {}).get("final_state_f1", 0.0))
        return (
            float(quiz["esm"]) + tie_break * 1e-6,
            "quiz.esm_then_final_state_f1",
        )
    closed = validation.get("closed_loop", {})
    if "final_state_f1" in closed:
        return float(closed["final_state_f1"]), "closed_loop.final_state_f1"
    return -float(validation["teacher_forced"]["loss"]), "-teacher_forced.loss"


def _flatten_primary_metrics(validation: dict[str, Any]) -> dict[str, Any]:
    result = {
        "teacher_forced_loss": validation["teacher_forced"]["loss"],
    }
    for group, names in {
        "one_step": ("update_f1", "false_update_rate", "state_f1"),
        "closed_loop": ("update_f1", "state_f1", "final_state_f1"),
        "quiz": ("esm", "tool_f1", "arg_exact"),
        "closed_loop_quiz": ("esm", "tool_f1", "arg_exact"),
    }.items():
        for name in names:
            if name in validation.get(group, {}):
                result[f"{group}_{name}"] = validation[group][name]
    return result


def _epoch_batch_count(
    train_source: IndexedMemoryDataset,
    plan: Any,
    *,
    batch_size: int,
    quiz_size: int | None,
    quiz_indices: tuple[int, ...] | None,
    quiz_batch_size: int,
) -> int:
    memory_batches = PlannedBatchSampler(train_source, plan, batch_size=batch_size)
    if quiz_size is None:
        return len(memory_batches)
    return len(
        BalancedMultitaskBatchSampler(
            memory_batches,
            memory_size=len(train_source),
            quiz_size=quiz_size,
            quiz_indices=quiz_indices or (),
            quiz_batch_size=quiz_batch_size,
        )
    )


def _delta_noop_pending_depths(catalog: Any) -> dict[int, int]:
    row_ids = catalog.eligible_row_ids(split="train", decision="NO_OP")
    source = IndexedMemoryDataset(catalog, "delta", row_ids=row_ids)
    depths = {}
    for position, row_id in enumerate(row_ids):
        row = source[position]
        memory_input = row.get("input")
        if not isinstance(memory_input, dict):
            raise TypeError(f"Delta-v3 row {row_id} has invalid input")
        pending = memory_input.get("pending_updates")
        if not isinstance(pending, list) or len(pending) > 4:
            raise ValueError(f"Delta-v3 row {row_id} has invalid pending_updates")
        depths[row_id] = len(pending)
    return depths


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.gradient_accumulation < 1
        or args.num_workers < 0
        or args.prefetch_factor < 1
    ):
        raise ValueError(
            "epochs, batch size, gradient accumulation, and prefetch factor must be "
            "positive; num workers must be non-negative"
        )
    if args.stop_after_epoch is not None and not (
        1 <= args.stop_after_epoch <= args.epochs
    ):
        raise ValueError("stop-after-epoch must be between 1 and epochs")
    if args.quiz_batch_size is not None and args.quiz_batch_size < 1:
        raise ValueError("quiz-batch-size must be positive")
    if args.quiz_total_passes < 1:
        raise ValueError("quiz-total-passes must be positive")
    if args.quiz_validation_rows_per_scenario < 2:
        raise ValueError("quiz-validation-rows-per-scenario must be at least two")
    if args.gold_quiz_validation_rows < 0:
        raise ValueError("gold-quiz-validation-rows must be non-negative")
    if args.quiz_validation_max_new_tokens < 1:
        raise ValueError("quiz-validation-max-new-tokens must be positive")
    if min(
        args.one_step_validation_batch_size,
        args.closed_loop_scenario_batch_size,
        args.quiz_validation_batch_size,
    ) < 1:
        raise ValueError("Validation batch sizes must be positive")
    if args.closed_loop_quiz and not args.multitask:
        raise ValueError("closed-loop Quiz Validation requires --multitask")
    if args.closed_loop_quiz and not args.closed_loop_full_scenarios:
        raise ValueError("closed-loop Quiz Validation requires full scenarios")
    if args.closed_loop_quiz and args.skip_quiz_validation:
        raise ValueError("closed-loop Quiz cannot be combined with skipped Quiz Validation")
    if args.closed_loop_quiz and args.skip_generation_validation:
        raise ValueError("closed-loop Quiz requires generation Validation")
    if args.resume and not (args.resume / "adapter").is_dir():
        raise FileNotFoundError(f"Resume adapter not found: {args.resume}")
    if not 0 <= args.closed_loop_sparse_noop_keep_fraction <= 1:
        raise ValueError("closed-loop sparse NO_OP keep fraction must be in [0, 1]")
    selected = [
        *(args.closed_loop_full_scenarios or []),
        *(args.closed_loop_sparse_scenarios or []),
    ]
    if len(selected) != len(set(selected)):
        raise ValueError("closed-loop full and sparse scenarios must be disjoint")
    validation_scenarios = tuple(
        args.validation_scenarios
        or range(81, (85 if args.multitask else 90) + 1)
    )
    invalid_validation = [
        scenario
        for scenario in validation_scenarios
        if split_for_scenario(scenario) != "validation"
        or (args.multitask and sft_split_for_scenario(scenario) != "validation")
    ]
    if invalid_validation:
        raise ValueError(
            f"Validation scenarios are outside the selected split: "
            f"{invalid_validation}"
        )
    if any(scenario not in set(validation_scenarios) for scenario in selected):
        raise ValueError("closed-loop scenarios must be listed in --validation-scenarios")
    if args.train_scenarios:
        invalid_train = [
            scenario
            for scenario in args.train_scenarios
            if split_for_scenario(scenario) != "train"
        ]
        if invalid_train:
            raise ValueError(f"Train scenarios are outside the train split: {invalid_train}")


def _configure_runtime_paths(workspace: Path) -> None:
    """Pin model and framework caches to the run workspace before HF imports."""
    huggingface = workspace / "cache" / "huggingface"
    paths = {
        "XDG_CACHE_HOME": workspace / "cache",
        "HF_HOME": huggingface,
        "HF_HUB_CACHE": huggingface / "hub",
        "HF_XET_CACHE": huggingface / "xet",
        "HF_DATASETS_CACHE": huggingface / "datasets",
        "TORCH_HOME": workspace / "cache" / "torch",
        "TMPDIR": workspace / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def _loader_worker_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    return kwargs


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _default_run_id(model: str, method: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model}-{method}-{stamp}"


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except KeyboardInterrupt:
        if args.run_id:
            tracker = RunTracker(args.workspace.resolve() / "runs" / args.run_id)
            tracker.status(
                "FAILED",
                interrupted=True,
                message="Training was interrupted before completion.",
            )
        raise
    except Exception:
        if args.run_id:
            tracker = RunTracker(args.workspace.resolve() / "runs" / args.run_id)
            tracker.status("FAILED", traceback=traceback.format_exc())
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
