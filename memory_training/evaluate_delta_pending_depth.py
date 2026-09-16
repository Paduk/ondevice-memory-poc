"""Evaluate Delta-v3 decisions on fixed gold states grouped by pending depth."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import DEFAULT_WORKSPACE_ROOT
from .dataset import IndexedMemoryDataset, ensure_catalog
from .methods import METHODS, DeltaV3Method
from .pending_depth_ablation import depth_balanced_diagnostic_rows
from .validation import evaluate_one_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--scenarios", nargs="+", type=int, required=True)
    parser.add_argument("--rows-per-depth-decision", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    workspace = args.workspace.resolve()
    run_dir = workspace / "runs" / args.run_id
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "adapter").is_dir():
        raise FileNotFoundError(f"Checkpoint adapter not found: {checkpoint}")
    config = _read_json(run_dir / "config.json")
    method_name = str(config["method"])
    method = METHODS[method_name]()
    if not isinstance(method, DeltaV3Method):
        raise TypeError("Pending-depth diagnostic requires a Delta-v3 method")
    arguments = config["arguments"]
    model_key = str(config["model"]["key"])
    max_length = args.max_length or int(arguments.get("max_length", 2048))
    max_new_tokens = args.max_new_tokens or int(
        arguments.get("max_new_tokens", 768)
    )
    _configure_runtime_paths(workspace)
    torch.manual_seed(args.evaluation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.evaluation_seed)

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

    data_root = args.data_root.resolve()
    catalog_path = (args.catalog_path or data_root / "catalog.sqlite").resolve()
    catalog = ensure_catalog(data_root, catalog_path)
    row_ids = [
        row_id
        for scenario in args.scenarios
        for row_id in catalog.scenario_row_ids(scenario)
    ]
    source = IndexedMemoryDataset(catalog, method.source_view, row_ids=row_ids)
    gold_summary = IndexedMemoryDataset(catalog, "summary", row_ids=row_ids)
    selected = depth_balanced_diagnostic_rows(
        catalog,
        source,
        row_ids,
        compaction_interval=method.compaction_interval,
        max_rows_per_depth_decision=args.rows_per_depth_decision,
        seed=args.evaluation_seed,
    )
    depth_results = {}
    for depth, selected_row_ids in selected.items():
        depth_results[str(depth)] = evaluate_one_step(
            model,
            bundle.tokenizer,
            method,
            source,
            gold_summary,
            selected_row_ids,
            accelerator=accelerator,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            batch_size=args.batch_size,
        )
    aggregate = aggregate_depth_metrics(depth_results)
    result = {
        "schema_version": "palmclaw-delta-pending-depth-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "checkpoint": str(checkpoint),
        "model": model_key,
        "method": method_name,
        "sampling_condition": arguments.get("delta_v3_noop_sampling"),
        "protocol": {
            "gold_state": True,
            "scenarios": list(args.scenarios),
            "rows_per_depth_decision": len(next(iter(selected.values()))) // 2,
            "evaluation_seed": args.evaluation_seed,
            "selected_row_ids_by_depth": {
                str(depth): list(values) for depth, values in selected.items()
            },
        },
        "by_pending_depth": depth_results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return result


def aggregate_depth_metrics(depth_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not depth_results:
        raise ValueError("At least one pending-depth result is required")
    counts: Counter[str] = Counter()
    total_rows = 0
    state_f1_sum = 0.0
    for result in depth_results.values():
        rows = int(result["rows"])
        total_rows += rows
        state_f1_sum += float(result["state_f1"]) * rows
        counts.update(
            {key: int(value) for key, value in result["decision_counts"].items()}
        )
    tp, fp = counts["true_update"], counts["false_update"]
    fn, tn = counts["missed_update"], counts["true_noop"]
    invalid_noop = counts["invalid_noop"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "rows": total_rows,
        "decision_counts": dict(counts),
        "update_precision": precision,
        "update_recall": recall,
        "update_f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "false_update_rate": (
            fp / (tn + fp + invalid_noop)
            if tn + fp + invalid_noop
            else 0.0
        ),
        "state_f1": state_f1_sum / total_rows if total_rows else 0.0,
        "macro_depth_update_recall": _mean(
            float(result["update_recall"]) for result in depth_results.values()
        ),
        "macro_depth_false_update_rate": _mean(
            float(result["false_update_rate"]) for result in depth_results.values()
        ),
    }


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
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _mean(values: Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
