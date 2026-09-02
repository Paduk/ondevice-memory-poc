"""Prepare deterministic V1 Cloud-Patch augmentation for Patch-only SFT.

The original V1 scenario IDs are encoded as 201-250 so they cannot collide
with Hybrid V2 S1-S100. All memory turns are retained for replay; the epoch
sampler excludes flagged compaction/unreliable transitions, keeps every other
UPDATE, and samples NO_OP at the requested ratio. Exactly
40% (4/10) of each selected scenario's final quizzes are retained for training.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import split_for_scenario
from .dataset import build_catalog
from .evaluate_hf_v1 import _build_v1_quiz_rows
from .methods.delta_v2 import compact_operations, expand_compact_operations
from .methods.operations import apply_operations, normalize_memory
from .quiz_sft import SCHEMA_VERSION, VehicleToolSchemaStore, sft_split_for_scenario
from .ubuntu_bridge import enable_ubuntu_runtime
from .v1_memory_replay import parse_history

DEFAULT_BASE_DATA = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2"
)
DEFAULT_TRACE_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-plus-temporal-t1-t20-plus-v1-10-patch-v1"
)
DEFAULT_SCENARIOS = (2, 5, 6, 14, 17, 23, 31, 32, 33, 36)
SCENARIO_SEED = 45
QUIZ_SEED = 450_040
TRACE_PATTERN = "turnwise-patch-soft30-fresh-r2-s{scenario}-20260816"
V1_OFFSET = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--scenarios", nargs="+", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--delta-v2",
        action="store_true",
        help="Write appended V1 rows using compact Delta-v2 state and targets.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def reconstruct_operations(previous: str, next_memory: str) -> tuple[list[dict[str, str]], bool]:
    """Return replay-exact line patches and whether full-memory fallback was used."""
    previous = normalize_memory(previous)
    next_memory = normalize_memory(next_memory)
    if previous == next_memory:
        return [], False
    if not previous:
        operations = [{"op": "add", "target": "", "content": next_memory}]
        replayed, _ = apply_operations(previous, operations)
        if replayed != next_memory:
            raise AssertionError("Empty-memory add did not replay")
        return operations, False
    if not next_memory:
        operations = [{"op": "delete", "target": previous, "content": ""}]
        replayed, _ = apply_operations(previous, operations)
        if replayed != next_memory:
            raise AssertionError("Full delete did not replay")
        return operations, False

    old_lines = previous.splitlines()
    new_lines = next_memory.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    operations: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
        if tag == "equal":
            continue
        old = "\n".join(old_lines[i1:i2])
        new = "\n".join(new_lines[j1:j2])
        if tag == "replace":
            target, content = _unique_replacement(
                old_lines, i1, i2, new, previous
            )
            operations.append({"op": "replace", "target": target, "content": content})
        elif tag == "delete":
            if previous.count(old) == 1:
                operations.append({"op": "delete", "target": old, "content": ""})
            else:
                target, content = _unique_replacement(
                    old_lines, i1, i2, "", previous
                )
                operations.append(
                    {"op": "replace", "target": target, "content": content}
                )
        elif i1 == len(old_lines):
            if new.lstrip().startswith(("### ", "**")):
                operations.append({"op": "add", "target": "", "content": new})
            else:
                anchor = _unique_ending_anchor(old_lines, i1, previous)
                operations.append({"op": "add", "target": anchor, "content": new})
        elif i1 > 0:
            anchor = _unique_ending_anchor(old_lines, i1, previous)
            if not new.lstrip().startswith(("### ", "**")):
                operations.append({"op": "add", "target": anchor, "content": new})
            else:
                operations.append(
                    {"op": "replace", "target": anchor, "content": f"{anchor}\n{new}"}
                )
        else:
            anchor = old_lines[0]
            operations.append(
                {"op": "replace", "target": anchor, "content": f"{new}\n{anchor}"}
            )
    try:
        if len(operations) > 32:
            raise ValueError("More than 32 reconstructed operations")
        replayed, _ = apply_operations(previous, operations)
        if replayed != next_memory:
            raise ValueError("Reconstructed patch does not match next memory")
        return operations, False
    except (TypeError, ValueError):
        fallback = [{"op": "replace", "target": previous, "content": next_memory}]
        replayed, _ = apply_operations(previous, fallback)
        if replayed != next_memory:
            raise AssertionError("Full-memory fallback did not replay")
        return fallback, True


def _unique_ending_anchor(lines: Sequence[str], end: int, memory: str) -> str:
    for start in range(end - 1, -1, -1):
        candidate = "\n".join(lines[start:end])
        if memory.count(candidate) == 1:
            return candidate
    raise ValueError("Could not find a unique insertion anchor")


def _unique_replacement(
    lines: Sequence[str], start: int, end: int, replacement: str, memory: str
) -> tuple[str, str]:
    """Widen a repeated changed block with unchanged context until unique."""
    maximum = max(start, len(lines) - end)
    for radius in range(maximum + 1):
        for left in range(min(radius, start) + 1):
            right = radius - left
            if right > len(lines) - end:
                continue
            target_lines = list(lines[start - left : end + right])
            target = "\n".join(target_lines)
            if memory.count(target) != 1:
                continue
            content_lines = list(lines[start - left : start])
            if replacement:
                content_lines.extend(replacement.splitlines())
            content_lines.extend(lines[end : end + right])
            content = "\n".join(content_lines)
            if content:
                return target, content
    raise ValueError("Could not make changed block unique")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark

    base_data = args.base_data.expanduser().resolve(strict=True)
    trace_root = args.trace_root.expanduser().resolve(strict=True)
    benchmark_root = args.vehiclemembench_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    scenarios = tuple(dict.fromkeys(args.scenarios))
    if len(scenarios) != 10 or any(value < 1 or value > 50 for value in scenarios):
        raise ValueError("Exactly 10 unique V1 scenarios in S1-S50 are required")
    if output.exists():
        if not args.force:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    benchmark = load_vehicle_benchmark(benchmark_root, strict=True)
    selected = [benchmark.scenario(value) for value in scenarios]
    view_paths = {name: output / f"{name}.jsonl" for name in ("summary", "patch", "delta")}
    handles = {name: path.open("wb") for name, path in view_paths.items()}
    counts: Counter[str] = Counter()
    scenario_records = []
    try:
        for name, handle in handles.items():
            with (base_data / f"{name}.jsonl").open("rb") as source:
                shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
        for scenario in selected:
            record, rows = _scenario_rows(
                scenario,
                trace_root,
                delta_v2=bool(args.delta_v2),
            )
            for summary_row, patch_row, delta_row in rows:
                for name, row in (
                    ("summary", summary_row),
                    ("patch", patch_row),
                    ("delta", delta_row),
                ):
                    handles[name].write(
                        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    )
            scenario_records.append(record)
            counts.update(record["counts"])
    finally:
        for handle in handles.values():
            handle.close()

    for name in ("compaction.jsonl", "turn_quiz.jsonl", "final_quiz.jsonl", "vehicle_tools.json"):
        source = base_data / name
        if source.exists():
            shutil.copy2(source, output / name)

    quiz_path = output / "quiz_sft.jsonl"
    shutil.copy2(base_data / "quiz_sft.jsonl", quiz_path)
    tools_path = output / "vehicle_tools.json"
    tools = VehicleToolSchemaStore(tools_path)
    all_quizzes = _build_v1_quiz_rows(selected, tools_path, tools.sha256)
    quiz_manifest = _append_sampled_quizzes(quiz_path, all_quizzes, scenarios)

    base_manifest = json.loads((base_data / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "palmclaw-v1-patch-augmentation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_data": str(base_data),
        "trace_root": str(trace_root),
        "teacher": "cloud Luna turn-wise Patch soft-30 fresh R2",
        "scenario_seed": SCENARIO_SEED,
        "quiz_seed": QUIZ_SEED,
        "scenario_encoding": "original V1 Sx -> 200+x; training-only",
        "selected_v1_scenarios": list(scenarios),
        "encoded_train_scenarios": [V1_OFFSET + value for value in scenarios],
        "split_policy": {
            "added_memory": "train only",
            "added_quiz": "train only; exactly 4 of 10 per V1 scenario",
            "validation_test": "unchanged from base data",
        },
        "sampling_policy": {
            "memory": "retain all turns; exclude train_eligible=false, keep other UPDATE, sample NO_OP at 5:1",
            "quiz": "fixed random 40% per selected scenario",
        },
        "delta_format": "compact_delta_v2" if args.delta_v2 else "legacy_delta_v1",
        "counts": dict(counts),
        "base_counts": base_manifest.get("counts", {}),
        "scenarios": scenario_records,
        "quiz": quiz_manifest,
        "files": {name: _file_record(path) for name, path in view_paths.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quiz_base = json.loads((base_data / "quiz_manifest.json").read_text(encoding="utf-8"))
    (output / "quiz_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "palmclaw-v1-patch-augmentation-quiz-v1",
                "base_manifest": quiz_base,
                "added_v1_quizzes": quiz_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = output / "catalog.sqlite"
    catalog_result = build_catalog(output, catalog)
    result = {
        "data_root": str(output),
        "catalog": str(catalog),
        "selected_v1_scenarios": list(scenarios),
        "encoded_train_scenarios": [V1_OFFSET + value for value in scenarios],
        "added_memory_counts": dict(counts),
        "added_quiz_count": quiz_manifest["count"],
        "catalog_result": catalog_result,
    }
    (output / "preparation-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _scenario_rows(
    scenario: Any,
    trace_root: Path,
    *,
    delta_v2: bool = False,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], ...]]]:
    source_id = int(scenario.index)
    encoded = V1_OFFSET + source_id
    trace = trace_root / TRACE_PATTERN.format(scenario=source_id)
    if not (trace / "COMPLETED").is_file():
        raise FileNotFoundError(f"Trusted trace is not marked complete: {trace}")
    databases = list((trace / "cache").rglob("memory.db"))
    if len(databases) != 1:
        raise ValueError(f"Expected one trace database, found {len(databases)}: {trace}")
    connection = sqlite3.connect(databases[0])
    try:
        run_rows = connection.execute(
            "SELECT id, end_message_id, status, output FROM consolidation_runs "
            "ORDER BY end_message_id"
        ).fetchall()
        metadata_by_run: dict[str, dict[str, Any]] = {}
        for run_id, metadata_json, error in connection.execute(
            "SELECT consolidation_run_id, metadata_json, error FROM model_calls "
            "WHERE consolidation_run_id IS NOT NULL ORDER BY created_at"
        ):
            if error is None:
                metadata_by_run[str(run_id)] = json.loads(metadata_json)
    finally:
        connection.close()
    turns = parse_history(scenario.history_path)
    if len(run_rows) != len(turns):
        raise ValueError(
            f"Trace/history length mismatch for V1 S{source_id}: {len(run_rows)} != {len(turns)}"
        )

    previous = ""
    delta_base = ""
    pending: list[Any] = []
    updates_since_compaction = 0
    counts: Counter[str] = Counter(turns=len(turns))
    rows = []
    for position, (run_row, turn) in enumerate(zip(run_rows, turns, strict=True)):
        run_id, _, status, raw_output = run_row
        if status != "completed":
            raise ValueError(f"Incomplete consolidation at V1 S{source_id} turn {position}: {status}")
        metadata = metadata_by_run.get(str(run_id))
        if metadata is None:
            raise ValueError(f"No successful model-call metadata for consolidation {run_id}")
        update_status = metadata.get("update_status")
        if update_status == "noop":
            next_memory = previous
        elif update_status == "updated":
            next_memory = normalize_memory(raw_output or "")
            if not next_memory:
                raise ValueError(f"UPDATE has empty materialized memory at V1 S{source_id} turn {position}")
        else:
            raise ValueError(
                f"Unexpected update_status at V1 S{source_id} turn {position}: {update_status}"
            )
        if metadata.get("memory_sha256_before") != _text_sha256(previous):
            raise ValueError(f"Before-memory hash mismatch at V1 S{source_id} turn {position}")
        # The legacy runtime records memory_sha256_after before its final
        # storage normalization on a small number of turns.  The following
        # turn's memory_sha256_before is the authoritative chain check above.
        if metadata.get("memory_sha256_after") != _text_sha256(next_memory):
            counts["legacy_after_hash_normalization_mismatches"] += 1
        operations, fallback = reconstruct_operations(previous, next_memory)
        decision = "UPDATE" if operations else "NO_OP"
        if decision != ("UPDATE" if update_status == "updated" else "NO_OP"):
            raise ValueError(f"Decision mismatch at V1 S{source_id} turn {position}")
        counts["updates" if operations else "no_ops"] += 1
        counts["operations"] += len(operations)
        counts["full_memory_fallbacks"] += int(fallback)
        counts.update(operation["op"] for operation in operations)
        source_operation_count = int(metadata.get("patch_operation_count") or 0)
        counts["source_patch_operations"] += source_operation_count
        counts["source_compactions"] += int(bool(metadata.get("compaction_triggered")))
        operation_shape_mismatch = False
        compaction_triggered = bool(metadata.get("compaction_triggered"))
        if operations and not metadata.get("compaction_triggered"):
            reconstructed_types = Counter(operation["op"] for operation in operations)
            expected_types = Counter(
                {
                    "add": int(metadata.get("patch_add_count") or 0),
                    "replace": int(metadata.get("patch_replace_count") or 0),
                    "delete": int(metadata.get("patch_delete_count") or 0),
                }
            )
            if len(operations) != source_operation_count or reconstructed_types != +expected_types:
                counts["non_compaction_operation_shape_mismatches"] += 1
                operation_shape_mismatch = True
        unreliable = fallback or operation_shape_mismatch
        exclude_from_training = unreliable or compaction_triggered
        counts["excluded_unreliable_update_rows"] += int(unreliable)
        counts["excluded_compaction_rows"] += int(compaction_triggered)
        counts["excluded_train_rows"] += int(exclude_from_training)
        turn_id = f"v1-s{source_id:03d}-turn-{position + 1:05d}"
        common = {
            "scenario_index": encoded,
            "split": split_for_scenario(encoded),
            "train_eligible": not exclude_from_training,
            "run_id": trace.name,
            "global_turn_index": position,
            "event_turn_index": position,
            "turn_id": turn_id,
            "event_id": f"v1-s{source_id:03d}",
            "timestamp": turn["timestamp"],
            "current_turn": {
                "speaker_id": turn["speaker"],
                "speaker_name": turn["speaker"],
                "text": turn["text"],
            },
            "provenance": {
                "source": "V1 Cloud Patch soft-30 fresh R2",
                "source_scenario_index": source_id,
                "source_consolidation_run_id": run_id,
                "before_memory_sha256": _text_sha256(previous),
                "after_memory_sha256": _text_sha256(next_memory),
                "reconstructed_patch": True,
                "full_memory_fallback": fallback,
            },
        }
        reason_code = "V1_CLOUD_TRACE_UPDATE" if operations else "V1_CLOUD_TRACE_NO_OP"
        reason = "Cloud R2 trace changed durable memory." if operations else "Cloud R2 trace kept memory unchanged."
        summary_row = {
            **common,
            "schema_version": "vehiclemembench-v2-summary-sft-v1",
            "sample_id": f"v1-s{source_id:03d}:summary:{position:05d}",
            "input": {"previous_memory": previous},
            "target": {
                "decision": decision,
                "reason_code": reason_code,
                "reason": reason,
                "next_memory": next_memory,
            },
        }
        patch_row = {
            **common,
            "schema_version": "vehiclemembench-v2-patch-sft-v1",
            "sample_id": f"v1-s{source_id:03d}:patch:{position:05d}",
            "input": {"previous_memory": previous},
            "target": {
                "decision": decision,
                "reason_code": reason_code,
                "reason": reason,
                "operations": operations,
            },
        }
        delta_row = {
            **common,
            "schema_version": (
                "vehiclemembench-v2-grouped-delta-v2-sft-v2"
                if delta_v2
                else "vehiclemembench-v2-appended-delta-sft-v1"
            ),
            "sample_id": f"v1-s{source_id:03d}:delta:{position:05d}",
            "input": (
                {"base_summary": delta_base, "pending_updates": list(pending)}
                if delta_v2
                else {
                    "base_summary": delta_base,
                    "pending_deltas": list(pending),
                    "updates_since_compaction": updates_since_compaction,
                }
            ),
            "target": {
                "decision": decision,
                "reason_code": reason_code,
                "reason": reason,
                "operations": (
                    compact_operations(operations)
                    if delta_v2 and operations
                    else operations
                ),
            },
        }
        # Keep every transition for exact replay. The catalog/sampler excludes
        # unreliable or compaction transitions from loss and splits trajectory
        # windows at those boundaries.
        rows.append((summary_row, patch_row, delta_row))
        if operations:
            pending.append(
                compact_operations(operations)
                if delta_v2
                else {"turn_id": turn_id, "operations": operations}
            )
            updates_since_compaction += 1
            if updates_since_compaction == 5:
                materialized = delta_base
                for batch in pending:
                    batch_operations = (
                        expand_compact_operations(batch)
                        if delta_v2
                        else batch["operations"]
                    )
                    materialized, _ = apply_operations(
                        materialized, batch_operations
                    )
                delta_base = materialized
                pending = []
                updates_since_compaction = 0
        materialized = delta_base
        for batch in pending:
            batch_operations = (
                expand_compact_operations(batch)
                if delta_v2
                else batch["operations"]
            )
            materialized, _ = apply_operations(materialized, batch_operations)
        if normalize_memory(materialized) != next_memory:
            raise AssertionError(f"Delta replay mismatch at V1 S{source_id} turn {position}")
        previous = next_memory
    record = {
        "source_scenario_index": source_id,
        "encoded_scenario_index": encoded,
        "trace": str(trace),
        "trace_database_sha256": _file_sha256(databases[0]),
        "history_sha256": _file_sha256(scenario.history_path),
        "counts": dict(counts),
        "final_memory_sha256": _text_sha256(previous),
    }
    return record, rows


def _append_sampled_quizzes(
    path: Path, rows: Sequence[dict[str, Any]], scenarios: Sequence[int]
) -> dict[str, Any]:
    by_scenario: dict[int, list[dict[str, Any]]] = {value: [] for value in scenarios}
    for row in rows:
        by_scenario[int(row["scenario_index"])].append(row)
    selected_records = []
    with path.open("a", encoding="utf-8") as handle:
        for source_id in scenarios:
            candidates = by_scenario[source_id]
            if len(candidates) != 10:
                raise ValueError(f"Expected 10 V1 quizzes for S{source_id}, found {len(candidates)}")
            generator = random.Random(QUIZ_SEED + source_id)
            selected = sorted(generator.sample(candidates, 4), key=lambda row: row["quiz_id"])
            encoded = V1_OFFSET + source_id
            quiz_ids = []
            for row in selected:
                row = dict(row)
                original_sample = str(row["sample_id"])
                row.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": f"v1-aug:{original_sample}",
                        "scenario_index": encoded,
                        "source_scenario_index": source_id,
                        "source_split": "V1_AUXILIARY_TRAIN",
                        "sft_split": sft_split_for_scenario(encoded),
                    }
                )
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                quiz_ids.append(row["quiz_id"])
            selected_records.append(
                {"source_scenario_index": source_id, "encoded_scenario_index": encoded, "quiz_ids": quiz_ids}
            )
    return {"count": len(selected_records) * 4, "fraction": 0.4, "scenarios": selected_records}


def _text_sha256(value: str) -> str:
    return hashlib.sha256(normalize_memory(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def main() -> None:
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
