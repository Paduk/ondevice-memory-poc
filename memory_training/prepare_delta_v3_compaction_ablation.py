"""Prepare aligned compact Delta-v3 datasets for k=2/5/10 ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import build_catalog
from .methods.delta_v2 import compact_operations, expand_compact_operations
from .methods.operations import apply_operations, normalize_memory

DEFAULT_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2"
)
INVARIANT_FILES = (
    "summary.jsonl",
    "patch.jsonl",
    "quiz_sft.jsonl",
    "quiz_manifest.json",
    "vehicle_tools.json",
    "turn_quiz.jsonl",
    "final_quiz.jsonl",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--intervals", nargs="+", type=int, default=(2, 5, 10))
    parser.add_argument("--force", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve(strict=True)
    output_parent = (
        args.output_parent.expanduser().resolve()
        if args.output_parent
        else source.parent
    )
    intervals = tuple(dict.fromkeys(int(value) for value in args.intervals))
    if not intervals or any(value < 1 for value in intervals):
        raise ValueError("intervals must contain unique positive integers")
    outputs = {
        interval: output_parent / f"{source.name}-delta-v3-compact-k{interval}-v1"
        for interval in intervals
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(existing[0])
    for path in existing:
        shutil.rmtree(path)
    output_parent.mkdir(parents=True, exist_ok=True)

    temporary = {
        interval: Path(
            tempfile.mkdtemp(
                prefix=f".{path.name}.tmp-", dir=output_parent
            )
        )
        for interval, path in outputs.items()
    }
    try:
        for root in temporary.values():
            _link_invariant_files(source, root)
        export = _export_delta_views(source, temporary)
        created_at = _utc_now()
        for interval, root in temporary.items():
            manifest = _manifest(
                source,
                root,
                interval=interval,
                created_at=created_at,
                statistics=export[interval],
            )
            _write_json(root / "manifest.json", manifest)
            root.rename(outputs[interval])
    except BaseException:
        for root in temporary.values():
            if root.exists():
                shutil.rmtree(root)
        raise

    results = {}
    for interval, root in outputs.items():
        catalog = build_catalog(root, root / "catalog.sqlite")
        turn_manifest = _derive_turn_manifest(source, root, catalog)
        summary = {
            "schema_version": "palmclaw-delta-v3-compact-ablation-preparation-v1",
            "created_at": _utc_now(),
            "source": str(source),
            "data_root": str(root.resolve()),
            "compaction_interval": interval,
            "statistics": export[interval],
            "catalog": catalog,
            "turn_manifest": turn_manifest,
        }
        _write_json(root / "preparation-summary.json", summary)
        (root / "COMPLETED").write_text(_utc_now() + "\n", encoding="utf-8")
        results[str(interval)] = summary
    return {
        "schema_version": "palmclaw-delta-v3-compact-ablation-set-v1",
        "source": str(source),
        "intervals": list(intervals),
        "outputs": results,
    }


def _derive_turn_manifest(
    source: Path, output: Path, catalog: Mapping[str, Any]
) -> str | None:
    source_path = source / "turn_manifest.json"
    if not source_path.is_file():
        return None
    value = json.loads(source_path.read_text(encoding="utf-8"))
    value["dataset_source_fingerprint"] = str(catalog["source_fingerprint"])
    value.pop("signature", None)
    value.pop("created_at", None)
    signature = hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    path = output / "turn_manifest.json"
    _write_json(path, {**value, "signature": signature, "created_at": _utc_now()})
    return str(path)


def _link_invariant_files(source: Path, output: Path) -> None:
    for name in INVARIANT_FILES:
        source_path = source / name
        if not source_path.is_file():
            continue
        os.link(source_path, output / name)


def _export_delta_views(
    source: Path, outputs: Mapping[int, Path]
) -> dict[int, dict[str, Any]]:
    delta_handles = {
        interval: (root / "delta.jsonl").open("w", encoding="utf-8")
        for interval, root in outputs.items()
    }
    compaction_handles = {
        interval: (root / "compaction.jsonl").open("w", encoding="utf-8")
        for interval, root in outputs.items()
    }
    states = {
        interval: {"base": "", "pending": []} for interval in outputs
    }
    statistics = {
        interval: {
            "counts": Counter(),
            "pending_depth_all": Counter(),
            "pending_depth_noop": Counter(),
            "pending_depth_update": Counter(),
            "scenario_final_memory_sha256": {},
        }
        for interval in outputs
    }
    current_scenario: int | None = None
    current_memory = ""
    last_row: dict[str, Any] | None = None
    try:
        with (source / "patch.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                scenario = int(row["scenario_index"])
                if scenario != current_scenario:
                    if current_scenario is not None and last_row is not None:
                        _flush_scenario(
                            current_scenario,
                            current_memory,
                            last_row,
                            states,
                            compaction_handles,
                            statistics,
                        )
                    current_scenario = scenario
                    current_memory = ""
                    states = {
                        interval: {"base": "", "pending": []}
                        for interval in outputs
                    }

                previous = normalize_memory(row["input"]["previous_memory"])
                if previous != current_memory:
                    raise ValueError(
                        f"Patch trajectory mismatch at patch.jsonl:{line_number}"
                    )
                target = row["target"]
                decision = str(target["decision"])
                operations = list(target.get("operations", []))
                if decision == "UPDATE":
                    next_memory, _ = apply_operations(previous, operations)
                    compact = compact_operations(operations)
                elif decision == "NO_OP" and not operations:
                    next_memory = previous
                    compact = []
                else:
                    raise ValueError(
                        f"Invalid decision/operations at patch.jsonl:{line_number}"
                    )
                expected_hash = row.get("provenance", {}).get(
                    "after_memory_sha256"
                )
                if expected_hash and _sha256_text(next_memory) != expected_hash:
                    raise ValueError(
                        f"Patch after-memory hash mismatch at line {line_number}"
                    )

                for interval, state in states.items():
                    pending = state["pending"]
                    depth = len(pending)
                    statistics[interval]["pending_depth_all"][depth] += 1
                    statistics[interval][
                        "pending_depth_update"
                        if decision == "UPDATE"
                        else "pending_depth_noop"
                    ][depth] += 1
                    delta_row = _delta_row(
                        row,
                        base_summary=str(state["base"]),
                        pending_updates=pending,
                        compact_operations_value=compact,
                        interval=interval,
                    )
                    delta_handles[interval].write(_json_line(delta_row))
                    counts = statistics[interval]["counts"]
                    counts["rows"] += 1
                    counts[decision] += 1
                    counts["operations"] += len(operations)
                    if decision == "UPDATE":
                        pending.append(compact)
                        if len(pending) == interval:
                            _write_compaction(
                                compaction_handles[interval],
                                delta_row,
                                base_summary=str(state["base"]),
                                pending_updates=pending,
                                next_summary=next_memory,
                                trigger="UPDATE_INTERVAL",
                            )
                            state["base"] = next_memory
                            state["pending"] = []
                            counts["interval_compactions"] += 1
                current_memory = next_memory
                last_row = row
        if current_scenario is not None and last_row is not None:
            _flush_scenario(
                current_scenario,
                current_memory,
                last_row,
                states,
                compaction_handles,
                statistics,
            )
    finally:
        for handle in (*delta_handles.values(), *compaction_handles.values()):
            handle.close()

    return {
        interval: {
            "counts": dict(value["counts"]),
            "pending_depth_all": _counter_dict(value["pending_depth_all"]),
            "pending_depth_noop": _counter_dict(value["pending_depth_noop"]),
            "pending_depth_update": _counter_dict(value["pending_depth_update"]),
            "scenario_final_memory_sha256": value[
                "scenario_final_memory_sha256"
            ],
        }
        for interval, value in statistics.items()
    }


def _flush_scenario(
    scenario: int,
    current_memory: str,
    last_row: Mapping[str, Any],
    states: Mapping[int, dict[str, Any]],
    compaction_handles: Mapping[int, Any],
    statistics: Mapping[int, dict[str, Any]],
) -> None:
    final_hash = _sha256_text(current_memory)
    for interval, state in states.items():
        pending = state["pending"]
        materialized = _materialize(str(state["base"]), pending)
        if materialized != current_memory:
            raise ValueError(f"Delta replay mismatch at end of S{scenario}")
        statistics[interval]["scenario_final_memory_sha256"][str(scenario)] = (
            final_hash
        )
        if pending:
            _write_compaction(
                compaction_handles[interval],
                last_row,
                base_summary=str(state["base"]),
                pending_updates=pending,
                next_summary=current_memory,
                trigger="FINAL_FLUSH",
            )
            statistics[interval]["counts"]["final_flushes"] += 1


def _delta_row(
    row: Mapping[str, Any],
    *,
    base_summary: str,
    pending_updates: Sequence[Sequence[Sequence[str]]],
    compact_operations_value: Sequence[Sequence[str]],
    interval: int,
) -> dict[str, Any]:
    excluded = {"schema_version", "sample_id", "input", "target"}
    common = {key: value for key, value in row.items() if key not in excluded}
    target = row["target"]
    target_common = {
        key: target[key]
        for key in ("decision", "reason_code", "reason")
        if key in target
    }
    return {
        "schema_version": "palmclaw-delta-v3-compact-ablation-row-v1",
        "sample_id": str(row["sample_id"]).replace(":patch:", ":delta:"),
        **common,
        "input": {
            "base_summary": base_summary,
            "pending_updates": list(pending_updates),
        },
        "target": {
            **target_common,
            "operations": list(compact_operations_value),
        },
        "ablation": {
            "prompt_profile": "delta_v3_compact_v1",
            "compaction_interval": interval,
        },
    }


def _write_compaction(
    handle: Any,
    row: Mapping[str, Any],
    *,
    base_summary: str,
    pending_updates: Sequence[Sequence[Sequence[str]]],
    next_summary: str,
    trigger: str,
) -> None:
    materialized = _materialize(base_summary, pending_updates)
    if materialized != normalize_memory(next_summary):
        raise ValueError("Compaction target differs from pending replay")
    handle.write(
        _json_line(
            {
                "schema_version": "palmclaw-delta-v3-compact-compaction-v1",
                "scenario_index": row["scenario_index"],
                "split": row["split"],
                "global_turn_index": row["global_turn_index"],
                "turn_id": row["turn_id"],
                "trigger": trigger,
                "input": {
                    "base_summary": base_summary,
                    "pending_updates": list(pending_updates),
                },
                "target": {
                    "next_summary": materialized,
                    "next_summary_sha256": _sha256_text(materialized),
                },
            }
        )
    )


def _materialize(
    base_summary: str, pending_updates: Sequence[Sequence[Sequence[str]]]
) -> str:
    memory = normalize_memory(base_summary)
    for batch in pending_updates:
        memory, _ = apply_operations(memory, expand_compact_operations(batch))
    return memory


def _manifest(
    source: Path,
    output: Path,
    *,
    interval: int,
    created_at: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "palmclaw-delta-v3-compact-ablation-data-v1",
        "created_at": created_at,
        "source_data_root": str(source),
        "source_manifest_sha256": _sha256_file(source / "manifest.json"),
        "prompt_profile": {
            "method": f"delta_v3_compact_k{interval}",
            "format": "B/P/T plaintext base with compact pending arrays",
            "target_format": "unchanged Delta-v3 compact operation arrays",
            "information_policy": "all current-turn fields and pending batches retained",
        },
        "compaction_interval": interval,
        "statistics": dict(statistics),
        "storage": {
            "invariant_files": "hard-linked read-only from source data root",
            "delta_bytes": (output / "delta.jsonl").stat().st_size,
            "compaction_bytes": (output / "compaction.jsonl").stat().st_size,
        },
    }


def _counter_dict(value: Counter[int]) -> dict[str, int]:
    return {str(key): value[key] for key in sorted(value)}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
