"""Replay memory only and persist selected quiz-time states plus decision traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .dataset import default_catalog_path, ensure_catalog
from .evaluate_hf_closed_loop import _configure_runtime_paths
from .methods import METHODS
from .methods.delta_v2 import DeltaV2State
from .validation import evaluate_closed_loop_batched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--scenario-batch-size", type=int, default=4)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _requests(
    path: Path, model_label: str
) -> tuple[dict[int, dict[int, list[str]]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["by_model"].get(model_label)
    if rows is None:
        raise ValueError(f"Model label is absent from request manifest: {model_label}")
    result: dict[int, dict[int, list[str]]] = {}
    for row in rows:
        scenario = int(row["scenario_index"])
        turn = int(row["global_turn_index"])
        result.setdefault(scenario, {}).setdefault(turn, []).append(
            str(row["sample_id"])
        )
    return result, len(rows)


def _serialize_state(method: Any, state: Any) -> dict[str, Any]:
    materialized = method.materialize_memory(state)
    if isinstance(state, DeltaV2State):
        return {
            "materialized_memory": materialized,
            "base_memory": state.base_summary,
            "pending_updates": state.pending_updates,
            "pending_depth": len(state.pending_updates),
        }
    serialized = asdict(state) if is_dataclass(state) else state
    return {
        "materialized_memory": materialized,
        "base_memory": materialized,
        "pending_updates": [],
        "pending_depth": 0,
        "raw_state": serialized,
    }


def _scenario_batches(
    catalog: Any, scenarios: Sequence[int], batch_size: int
) -> list[tuple[str, list[int]]]:
    batches: list[tuple[str, list[int]]] = []
    current: list[int] = []
    current_split = ""
    for scenario in scenarios:
        split = str(catalog.record(catalog.scenario_row_ids(scenario)[0]).split)
        if current and (split != current_split or len(current) >= batch_size):
            batches.append((current_split, current))
            current = []
        current.append(scenario)
        current_split = split
    if current:
        batches.append((current_split, current))
    return batches


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario_batch_size < 1:
        raise ValueError("scenario-batch-size must be positive")
    checkpoint = args.checkpoint.resolve()
    adapter = checkpoint / "adapter"
    if not adapter.is_dir():
        raise FileNotFoundError(f"Checkpoint adapter not found: {adapter}")
    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    scenario_dir = output / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.request_manifest.resolve()
    snapshot_requests, expected_snapshots = _requests(request_path, args.model_label)
    scenarios = sorted(snapshot_requests)
    request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    signature_payload = {
        "model": args.model,
        "model_label": args.model_label,
        "method": args.method,
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "request_sha256": request_sha,
        "scenarios": scenarios,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("signature") != signature:
            raise ValueError(f"Stale snapshot output directory: {output}")
    else:
        _write_json(
            manifest_path,
            {
                "schema_version": "palmclaw-hf-memory-snapshot-capture-v1",
                "signature": signature,
                **signature_payload,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    pending = [
        scenario
        for scenario in scenarios
        if not (scenario_dir / f"s{scenario:03d}.json").is_file()
    ]
    if pending:
        _configure_runtime_paths(workspace)
        from accelerate import Accelerator

        from .models import load_peft_bundle

        catalog = ensure_catalog(
            data_root,
            (args.catalog_path or default_catalog_path(workspace)).resolve(),
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
        batches = _scenario_batches(catalog, pending, args.scenario_batch_size)
        for batch_index, (split, chunk) in enumerate(batches, start=1):
            print(
                f"[{batch_index}/{len(batches)}] {args.model_label} {args.method} "
                f"scenarios={chunk}",
                flush=True,
            )
            results = evaluate_closed_loop_batched(
                model,
                bundle.tokenizer,
                method,
                catalog,
                chunk,
                accelerator=accelerator,
                split=split,
                batch_size=min(args.scenario_batch_size, len(chunk)),
                snapshot_requests={
                    scenario: snapshot_requests[scenario] for scenario in chunk
                },
                snapshot_serializer=lambda state: _serialize_state(method, state),
                capture_trace=True,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            for scenario, result in results.items():
                _write_json(scenario_dir / f"s{scenario:03d}.json", result)

    captured = 0
    completed = []
    trace_turns = 0
    for scenario in scenarios:
        path = scenario_dir / f"s{scenario:03d}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        captured += len(result.get("quiz_snapshots", {}))
        trace_turns += len(result.get("decision_trace", []))
        completed.append(scenario)
    summary = {
        "schema_version": "palmclaw-hf-memory-snapshot-capture-summary-v1",
        "model": args.model,
        "model_label": args.model_label,
        "method": args.method,
        "expected_scenarios": scenarios,
        "completed_scenarios": completed,
        "expected_snapshots": expected_snapshots,
        "captured_snapshots": captured,
        "trace_turns": trace_turns,
        "complete": completed == scenarios and captured == expected_snapshots,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
