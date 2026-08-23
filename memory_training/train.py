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
from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .dataset import IndexedMemoryDataset, default_catalog_path, ensure_catalog
from .methods import METHODS
from .models import load_peft_bundle
from .sampling import EpochSampler, SamplingConfig
from .tracking import MLflowMirror, RunTracker
from .training_data import (
    ChatExampleEncoder,
    MethodSFTDataset,
    PlannedBatchSampler,
    SFTCollator,
)
from .validation import (
    evaluate_closed_loop,
    evaluate_one_step,
    evaluate_teacher_forced_loss,
    run_quiz_hook,
    stratified_one_step_row_ids,
    stratified_teacher_forced_row_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noop-per-update", type=int, default=10)
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
    parser.add_argument("--closed-loop-scenario-limit", type=int)
    parser.add_argument("--closed-loop-turn-limit", type=int)
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

    set_seed(args.seed)
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
    catalog = ensure_catalog(data_root, default_catalog_path(workspace))
    method = METHODS[args.method]()
    sampling_config = SamplingConfig(
        noop_per_update=args.noop_per_update,
        adjacent_noop_fraction=args.adjacent_noop_fraction,
        trajectory_fraction=args.trajectory_fraction,
        seed=args.seed,
    )
    epoch_sampler = EpochSampler(catalog, sampling_config)
    plans = [epoch_sampler.build(epoch) for epoch in range(args.epochs)]
    train_source = IndexedMemoryDataset(catalog, method.source_view, split="train")
    total_batches = sum(
        len(PlannedBatchSampler(train_source, plan, batch_size=args.batch_size))
        for plan in plans
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
    train_dataset = MethodSFTDataset(train_source, method, encoder)
    validation_source = IndexedMemoryDataset(
        catalog, method.source_view, split="validation"
    )
    validation_summary = IndexedMemoryDataset(catalog, "summary", split="validation")
    teacher_forced_row_ids = stratified_teacher_forced_row_ids(
        catalog,
        max_rows=args.eval_max_rows,
        adjacent_noop_fraction=args.eval_adjacent_noop_fraction,
        seed=args.seed,
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
        "schema_version": "palmclaw-memory-training-run-v1",
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
        },
        "epoch_plans": [plan.summary() for plan in plans],
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
                "learning_rate": args.learning_rate,
                "lora_rank": args.lora_rank,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "dataset_fingerprint": catalog.metadata().get("source_fingerprint"),
            }
        )
    elif accelerator.is_main_process:
        tracker.status(
            "RUNNING",
            global_step=progress.global_step,
            message=f"resumed from {args.resume}",
        )
    if mlflow.error:
        tracker.metric("mlflow_unavailable", error=mlflow.error)

    optimizer.zero_grad()
    recent_loss = 0.0
    recent_batches = 0
    recent_target_tokens = 0
    log_started_at = time.perf_counter()
    run_started_at = log_started_at
    stopped_early = False
    for epoch in range(progress.epoch, args.epochs):
        if (
            args.max_train_steps is not None
            and progress.global_step >= args.max_train_steps
        ):
            stopped_early = True
            break
        start_batch = progress.next_batch if epoch == progress.epoch else 0
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
            batch.pop("metadata", None)
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
                            (time.perf_counter() - run_started_at)
                            / progress.global_step
                            * max(total_optimizer_steps - progress.global_step, 0)
                        ),
                    }
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
        validation = _epoch_validation(
            args,
            model=model,
            tokenizer=bundle.tokenizer,
            method=method,
            catalog=catalog,
            validation_source=validation_source,
            validation_summary=validation_summary,
            validation_loader=validation_loader,
            accelerator=accelerator,
            checkpoint_dir=checkpoint_dir,
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
        if stopped_early:
            break

    final_state = "CANARY_COMPLETE" if stopped_early else "COMPLETED"
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
    validation_loader: Any,
    accelerator: Any,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    result = {
        "teacher_forced": evaluate_teacher_forced_loss(
            model,
            validation_loader,
            accelerator=accelerator,
            max_batches=args.eval_max_batches,
        )
    }
    if not args.skip_generation_validation:
        row_ids = stratified_one_step_row_ids(
            catalog,
            max_rows=args.one_step_max_rows,
            seed=args.seed,
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
        )
        result["closed_loop"] = evaluate_closed_loop(
            model,
            tokenizer,
            method,
            catalog,
            accelerator=accelerator,
            scenario_limit=args.closed_loop_scenario_limit,
            turn_limit_per_scenario=args.closed_loop_turn_limit,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
    if args.quiz_hook:
        result["quiz"] = run_quiz_hook(
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
    }.items():
        for name in names:
            if name in validation.get(group, {}):
                result[f"{group}_{name}"] = validation[group][name]
    return result


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
    if args.resume and not (args.resume / "adapter").is_dir():
        raise FileNotFoundError(f"Resume adapter not found: {args.resume}")


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
