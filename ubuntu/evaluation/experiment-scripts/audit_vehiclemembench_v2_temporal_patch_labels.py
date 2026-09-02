#!/usr/bin/env python3
"""Audit T1-T20 Temporal-Patch labels and materialize a safe training root."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from memory_training.dataset import build_catalog

from palmclaw_ubuntu.vehicle_bench.v2_temporal_label_audit import (
    OpenAITemporalLabelJudge,
    build_plan_first_audit,
    materialize_audited_dataset,
)

DEFAULT_SOURCE_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-plus-temporal-t1-t20-temporal-patch-t1t10-v1"
)
DEFAULT_TEMPORAL_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-temporal"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-plus-temporal-t1-t20-temporal-patch-t1t10-terra-audited-v2"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--temporal-source", type=Path, default=DEFAULT_TEMPORAL_SOURCE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--terra-escalation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send only unresolved plan mappings to Terra (default: enabled).",
    )
    parser.add_argument(
        "--review-all-with-terra",
        action="store_true",
        help="Costly diagnostic mode: re-review every operation, not only DEFER.",
    )
    parser.add_argument(
        "--allow-defer",
        action="store_true",
        help="Materialize even if unresolved labels remain (not recommended).",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("workers and max attempts must be positive")
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    records, summary = build_plan_first_audit(
        source_patch=source_root / "patch.jsonl",
        temporal_source_root=args.temporal_source,
    )

    selected = [
        record
        for record in records
        if record["verdict"] == "DEFER" or args.review_all_with_terra
    ]
    if selected and args.terra_escalation:
        records, live_summary = _run_terra_reviews(
            records=records,
            selected=selected,
            checkpoint_root=output_root.parent
            / f".{output_root.name}-terra-checkpoints",
            model=args.model,
            workers=args.workers,
            max_attempts=args.max_attempts,
            timeout_seconds=args.timeout_seconds,
        )
        summary["live_terra"] = live_summary
        summary["verdicts"] = _counts(records, "verdict")
        summary["after_actions"] = _label_counts(records, "effective_label")

    deferred = sum(record["verdict"] == "DEFER" for record in records)
    summary["deferred_operation_count"] = deferred
    if deferred and not args.allow_defer:
        raise RuntimeError(
            f"{deferred} temporal labels remain DEFER; refusing training output"
        )
    result = materialize_audited_dataset(
        source_root=source_root,
        output_root=output_root,
        audit_records=records,
        audit_summary=summary,
        force=args.force,
    )
    catalog_path = output_root / "catalog.sqlite"
    catalog_summary = build_catalog(output_root, catalog_path)
    result["catalog_path"] = str(catalog_path)
    result["catalog"] = catalog_summary
    summary_path = output_root / "temporal-label-audit" / "summary.json"
    _write_json(summary_path, result)
    return result


def _run_terra_reviews(
    *,
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    checkpoint_root: Path,
    model: str,
    workers: int,
    max_attempts: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    judge = OpenAITemporalLabelJudge(model, timeout_seconds=timeout_seconds)

    def review(record: dict[str, Any]) -> dict[str, Any]:
        key = f"{record['sample_id']}-op{record['operation_index']}"
        safe_key = key.replace(":", "-")
        checkpoint = checkpoint_root / f"{safe_key}.json"
        signature = _review_signature(record, model)
        if checkpoint.is_file():
            stored = json.loads(checkpoint.read_text(encoding="utf-8"))
            if stored.get("review_signature") != signature:
                raise ValueError(f"Stale Terra audit checkpoint: {checkpoint}")
            return stored
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = judge.audit(record)
                stored = {
                    **result,
                    "review_signature": signature,
                    "attempt": attempt,
                }
                _write_json(checkpoint, stored)
                return stored
            except Exception as exc:  # provider/schema retry boundary
                last_error = exc
        assert last_error is not None
        raise last_error

    by_key = {
        (str(record["sample_id"]), int(record["operation_index"])): record
        for record in records
    }
    failures: list[str] = []
    usage = {key: 0 for key in (
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "total_tokens",
    )}
    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as pool:
        futures = {pool.submit(review, record): record for record in selected}
        for future in as_completed(futures):
            record = futures[future]
            key = (str(record["sample_id"]), int(record["operation_index"]))
            try:
                reviewed = future.result()
                updated = {
                    **by_key[key],
                    "verdict": reviewed["verdict"],
                    "source": "TERRA_LIVE",
                    "effective_label": reviewed["effective_label"],
                    "reason": reviewed["reason"],
                    "live_terra": {
                        field: reviewed.get(field)
                        for field in (
                            "model_id",
                            "response_id",
                            "prompt_version",
                            "attempt",
                        )
                    },
                }
                by_key[key] = updated
                for usage_key in usage:
                    usage[usage_key] += int(reviewed.get("usage", {}).get(usage_key, 0))
            except Exception as exc:
                failures.append(
                    f"{record['sample_id']}/op{record['operation_index']}: "
                    f"{type(exc).__name__}: {exc}"
                )
    resolved = [
        by_key[(str(record["sample_id"]), int(record["operation_index"]))]
        for record in records
    ]
    return resolved, {
        "model": model,
        "selected": len(selected),
        "failures": failures,
        "usage": usage,
    }


def _review_signature(record: dict[str, Any], model: str) -> str:
    payload = {
        "model": model,
        "sample_id": record["sample_id"],
        "operation_index": record["operation_index"],
        "operation": record["operation"],
        "current_turn": record["current_turn"],
        "previous_memory": record["previous_memory"],
        "plan_transition": record.get("plan_transition"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for record in records:
        key = str(record[field])
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))


def _label_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for record in records:
        key = str(record[field]["temporal_action"])
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
