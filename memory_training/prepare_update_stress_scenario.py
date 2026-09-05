"""Build one deterministic long-memory scenario with many UPDATE turns.

The source scenario's NO_OP and UPDATE turns are retained. Synthetic, unique,
unambiguous ADD turns are distributed through the trajectory until
``target_updates`` is reached. Summary, Patch, and Delta-v3 inputs are then
replayed from scratch so all three views describe exactly the same state at
every turn. Existing Quiz labels are retained and remapped to the rebased
trajectory checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .dataset import MEMORY_VIEWS, DatasetCatalog, build_catalog
from .methods.delta_v2 import compact_operations
from .methods.operations import apply_operations, normalize_memory
from .quiz_sft import sft_split_for_scenario

DEFAULT_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-v2-v1-10-eval-s81-update40-front-middle-v1"
)
TURN_MANIFEST_SCHEMA_VERSION = "palmclaw-hf-cache-benchmark-turns-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", type=int, default=81)
    parser.add_argument("--target-updates", type=int, default=40)
    parser.add_argument("--compaction-interval", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source.expanduser().resolve(strict=True)
    output_root = args.output.expanduser().resolve()
    if args.target_updates < 1:
        raise ValueError("target_updates must be positive")
    if args.compaction_interval < 1:
        raise ValueError("compaction_interval must be positive")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    source_rows = _load_aligned_source(source_root, args.scenario)
    source_updates = sum(
        row["patch"]["target"]["decision"] == "UPDATE" for row in source_rows
    )
    if source_updates > args.target_updates:
        raise ValueError(
            f"S{args.scenario} already has {source_updates} UPDATEs, more than "
            f"target {args.target_updates}"
        )
    synthetic_updates = args.target_updates - source_updates
    schedule = _build_schedule(source_rows, synthetic_updates)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        result = _write_dataset(
            temporary,
            source_root=source_root,
            source_rows=source_rows,
            schedule=schedule,
            scenario=args.scenario,
            target_updates=args.target_updates,
            synthetic_updates=synthetic_updates,
            compaction_interval=args.compaction_interval,
        )
        temporary.rename(output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    catalog_result = build_catalog(output_root, output_root / "catalog.sqlite")
    turn_manifest = _build_turn_manifest(
        output_root,
        scenario=args.scenario,
        split=str(source_rows[0]["patch"]["split"]),
        source_rows=len(source_rows),
        source_updates=source_updates,
        synthetic_updates=synthetic_updates,
    )
    _write_json(output_root / "turn_manifest.json", turn_manifest)
    completed = {
        **result,
        "data_root": str(output_root),
        "catalog": catalog_result,
        "turn_manifest": str(output_root / "turn_manifest.json"),
    }
    _write_json(output_root / "preparation-summary.json", completed)
    (output_root / "COMPLETED").write_text(_utc_now() + "\n", encoding="utf-8")
    return completed


def _load_aligned_source(
    source_root: Path, scenario: int
) -> list[dict[str, dict[str, Any]]]:
    paths = [source_root / f"{view}.jsonl" for view in MEMORY_VIEWS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source views are missing: {missing}")
    handles = [path.open(encoding="utf-8") for path in paths]
    selected = []
    try:
        for line_number, lines in enumerate(zip(*handles, strict=True), start=1):
            rows = [json.loads(line) for line in lines]
            identities = [_identity(row) for row in rows]
            if any(identity != identities[0] for identity in identities[1:]):
                raise ValueError(f"Cross-view mismatch at source line {line_number}")
            if int(rows[0]["scenario_index"]) != scenario:
                continue
            selected.append(dict(zip(MEMORY_VIEWS, rows, strict=True)))
    finally:
        for handle in handles:
            handle.close()
    if not selected:
        raise ValueError(f"Source contains no rows for S{scenario}")
    return selected


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("scenario_index"),
        row.get("global_turn_index"),
        row.get("turn_id"),
        row.get("split"),
        row.get("target", {}).get("decision"),
    )


def _build_schedule(
    source_rows: Sequence[dict[str, dict[str, Any]]], synthetic_updates: int
) -> list[tuple[str, Any]]:
    """Evenly interleave synthetic UPDATEs while preserving all source rows."""
    if synthetic_updates < 0:
        raise ValueError("synthetic_updates must be non-negative")
    first_update_position = next(
        (
            position
            for position, row in enumerate(source_rows, start=1)
            if row["patch"]["target"]["decision"] == "UPDATE"
        ),
        None,
    )
    if first_update_position is None:
        raise ValueError("Stress source scenario has no memory-creating UPDATE")
    available_gaps = len(source_rows) - first_update_position
    if synthetic_updates > available_gaps:
        raise ValueError(
            "At most one synthetic UPDATE may be inserted per source gap after "
            f"the first memory update ({synthetic_updates}/{available_gaps})"
        )
    insert_after: dict[int, list[int]] = defaultdict(list)
    previous_slot = first_update_position - 1
    for ordinal in range(1, synthetic_updates + 1):
        slot = first_update_position + (
            ordinal * (len(source_rows) - first_update_position) // (synthetic_updates + 1)
        )
        slot = max(first_update_position, min(len(source_rows) - 1, slot))
        if slot <= previous_slot:
            slot = previous_slot + 1
        insert_after[slot].append(ordinal)
        previous_slot = slot

    schedule: list[tuple[str, Any]] = []
    for source_position, aligned in enumerate(source_rows, start=1):
        schedule.append(("source", aligned))
        for ordinal in insert_after.get(source_position, []):
            schedule.append(("synthetic_update", ordinal))
    return schedule


def _write_dataset(
    output: Path,
    *,
    source_root: Path,
    source_rows: Sequence[dict[str, dict[str, Any]]],
    schedule: Sequence[tuple[str, Any]],
    scenario: int,
    target_updates: int,
    synthetic_updates: int,
    compaction_interval: int,
) -> dict[str, Any]:
    paths = {view: output / f"{view}.jsonl" for view in MEMORY_VIEWS}
    handles = {view: path.open("w", encoding="utf-8") for view, path in paths.items()}
    compaction_path = output / "compaction.jsonl"
    compaction_handle = compaction_path.open("w", encoding="utf-8")
    memory = ""
    delta_base = ""
    pending_updates: list[list[list[str]]] = []
    counts: Counter[str] = Counter()
    update_ordinal = 0
    source_position = 0
    source_turn_remap: dict[int, dict[str, Any]] = {}
    memory_split = str(source_rows[0]["patch"]["split"])
    quiz_split = sft_split_for_scenario(scenario)
    try:
        for global_turn_index, (kind, payload) in enumerate(schedule):
            before = memory
            if kind == "source":
                source_position += 1
                patch_source = payload["patch"]
                decision = str(patch_source["target"]["decision"])
                operations = list(patch_source["target"].get("operations", []))
                common = _source_common(
                    patch_source,
                    scenario=scenario,
                    global_turn_index=global_turn_index,
                    target_updates=target_updates,
                )
                target_common = {
                    key: patch_source["target"][key]
                    for key in ("decision", "reason_code", "reason")
                    if key in patch_source["target"]
                }
                source_provenance = dict(patch_source.get("provenance") or {})
                stress_metadata = {
                    "source_kind": "source",
                    "source_global_turn_index": patch_source["global_turn_index"],
                    "source_position": source_position,
                }
            else:
                synthetic_ordinal = int(payload)
                common, target_common, operations, position_metadata = _synthetic_update(
                    scenario=scenario,
                    global_turn_index=global_turn_index,
                    ordinal=synthetic_ordinal,
                    total=synthetic_updates,
                    split=memory_split,
                    target_updates=target_updates,
                    preceding_source=source_rows[source_position - 1]["patch"],
                    previous_memory=before,
                )
                decision = "UPDATE"
                source_provenance = {}
                stress_metadata = {
                    "source_kind": "synthetic_update",
                    "synthetic_update_ordinal": synthetic_ordinal,
                    "inserted_after_source_position": source_position,
                    **position_metadata,
                }

            if decision == "UPDATE" and not operations:
                raise ValueError(
                    f"UPDATE has no operations at turn {global_turn_index}"
                )
            if decision == "NO_OP" and operations:
                raise ValueError(f"NO_OP has operations at turn {global_turn_index}")

            after = before
            if operations:
                after, _ = apply_operations(before, operations)
                update_ordinal += 1
            before_hash = _sha256_text(before)
            after_hash = _sha256_text(after)
            if kind == "source":
                source_turn = int(patch_source["global_turn_index"])
                source_turn_remap[source_turn] = {
                    "global_turn_index": global_turn_index,
                    "turn_id": str(patch_source["turn_id"]),
                    "memory": after,
                    "memory_sha256": after_hash,
                }
            provenance = {
                **source_provenance,
                "source_before_memory_sha256": source_provenance.get(
                    "before_memory_sha256", before_hash
                ),
                "source_after_memory_sha256": source_provenance.get(
                    "after_memory_sha256", after_hash
                ),
                "before_memory_sha256": before_hash,
                "after_memory_sha256": after_hash,
                "grouping": "speaker_name_first_appearance",
                "stress_scenario": "preserve-noop-front-middle-add-v2",
            }
            row_metadata = {
                **stress_metadata,
                "update_ordinal_after_turn": update_ordinal,
                "update_bucket": (
                    f"U{((update_ordinal - 1) // 10) * 10 + 1}-"
                    f"{((update_ordinal - 1) // 10 + 1) * 10}"
                    if update_ordinal
                    else "U0"
                ),
            }
            compact = compact_operations(operations) if operations else []
            rows = {
                "summary": {
                    "schema_version": "vehiclemembench-v2-grouped-summary-sft-v2",
                    "sample_id": f"s{scenario:03d}:summary:{global_turn_index:05d}",
                    **common,
                    "input": {"previous_memory": before},
                    "target": {**target_common, "next_memory": after},
                    "provenance": provenance,
                    "stress_metadata": row_metadata,
                },
                "patch": {
                    "schema_version": "vehiclemembench-v2-grouped-patch-sft-v2",
                    "sample_id": f"s{scenario:03d}:patch:{global_turn_index:05d}",
                    **common,
                    "input": {"previous_memory": before},
                    "target": {**target_common, "operations": operations},
                    "provenance": provenance,
                    "stress_metadata": row_metadata,
                },
                "delta": {
                    "schema_version": "vehiclemembench-v2-grouped-delta-v2-sft-v2",
                    "sample_id": f"s{scenario:03d}:delta:{global_turn_index:05d}",
                    **common,
                    "input": {
                        "base_summary": delta_base,
                        "pending_updates": list(pending_updates),
                    },
                    "target": {**target_common, "operations": compact},
                    "provenance": provenance,
                    "stress_metadata": row_metadata,
                },
            }
            for view, row in rows.items():
                handles[view].write(_json_line(row))

            counts[decision] += 1
            counts[kind] += 1
            if kind == "synthetic_update":
                counts[f"synthetic_{position_metadata['update_position']}"] += 1
            counts["operations"] += len(operations)
            if operations:
                pending_updates.append(compact)
                if len(pending_updates) == compaction_interval:
                    _write_compaction(
                        compaction_handle,
                        rows["delta"],
                        base_summary=delta_base,
                        pending_updates=pending_updates,
                        next_summary=after,
                    )
                    delta_base = after
                    pending_updates = []
                    counts["compactions"] += 1
            memory = after

        if update_ordinal != target_updates:
            raise ValueError(
                f"Generated {update_ordinal} UPDATEs, expected {target_updates}"
            )
        materialized = _materialize_delta(delta_base, pending_updates)
        if materialized != memory:
            raise ValueError("Final Delta-v3 state differs from Summary/Patch state")
        if pending_updates:
            raise ValueError(
                "Target UPDATE count must end on a compaction boundary for this profile"
            )
    finally:
        for handle in handles.values():
            handle.close()
        compaction_handle.close()

    shutil.copy2(source_root / "vehicle_tools.json", output / "vehicle_tools.json")
    quiz_result = _write_remapped_quizzes(
        source_root,
        output,
        scenario=scenario,
        quiz_split=quiz_split,
        source_turn_remap=source_turn_remap,
    )
    _write_json(
        output / "quiz_manifest.json",
        {
            "schema_version": "palmclaw-update-stress-quiz-v1",
            "source_quiz_sft": str(source_root / "quiz_sft.jsonl"),
            "scenario_index": scenario,
            "memory_split": memory_split,
            "quiz_split": quiz_split,
            **quiz_result,
            "note": (
                "Original Quiz labels retained; memory checkpoints and embedded "
                "Gold memory remapped to the stress trajectory."
            ),
        },
    )
    manifest = {
        "schema_version": "palmclaw-update-stress-scenario-v1",
        "created_at": _utc_now(),
        "source_data_root": str(source_root),
        "scenario_index": scenario,
        "construction": {
            "source_rows_preserved": len(source_rows),
            "source_noops_preserved": counts["NO_OP"],
            "source_updates_preserved": counts["source"] - counts["NO_OP"],
            "synthetic_unique_add_updates": counts["synthetic_update"],
            "synthetic_front_updates": counts["synthetic_front"],
            "synthetic_middle_updates": counts["synthetic_middle"],
            "target_updates": target_updates,
            "synthetic_turn_distribution": "evenly interleaved across source-row gaps",
            "synthetic_update_positions": (
                "odd synthetic ordinals inserted after the first fact; "
                "even ordinals inserted after the median fact"
            ),
            "global_turn_index": "rebased to contiguous 0..N-1",
            "compaction_interval_updates": compaction_interval,
            "quiz_labels": "original labels with remapped stress-memory snapshots",
        },
        "counts": dict(counts),
        "final_memory": {
            "characters": len(memory),
            "sha256": _sha256_text(memory),
        },
        "quiz": quiz_result,
        "files": {
            **{view: _file_record(path) for view, path in paths.items()},
            "compaction": _file_record(compaction_path),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _source_common(
    row: Mapping[str, Any],
    *,
    scenario: int,
    global_turn_index: int,
    target_updates: int,
) -> dict[str, Any]:
    common = {
        key: row[key]
        for key in (
            "event_turn_index",
            "turn_id",
            "event_id",
            "timestamp",
            "current_turn",
        )
        if key in row
    }
    return {
        "scenario_index": scenario,
        "split": str(row["split"]),
        "run_id": f"s{scenario:03d}-update{target_updates}-stress-v2",
        "global_turn_index": global_turn_index,
        **common,
        "train_eligible": bool(row.get("train_eligible", True)),
    }


def _synthetic_update(
    *,
    scenario: int,
    global_turn_index: int,
    ordinal: int,
    total: int,
    split: str,
    target_updates: int,
    preceding_source: Mapping[str, Any],
    previous_memory: str,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    list[dict[str, str]],
    dict[str, Any],
]:
    timestamp = _timestamp_after(str(preceding_source["timestamp"]), ordinal)
    volume = 20 + ordinal
    route = f"stress-route-{ordinal:02d}"
    # Alternation keeps the same synthetic fact's position class stable across
    # the 20/40/60/80 workloads while maintaining a balanced stress profile.
    position = "front" if ordinal % 2 else "middle"
    target_line, target_fact_index, fact_count, speaker = _insertion_target(
        previous_memory, position=position
    )
    text = (
        "Remember this as an additional preference without replacing any existing "
        f"preference: on {route}, set my music volume to {volume}."
    )
    content = (
        f"- [{timestamp.replace('T', ' ')}] carcontrol_music_set_volume.volume; "
        f"value={volume}; condition=on {route}"
    )
    turn_id = f"stress-s{scenario:03d}-update-{ordinal:03d}"
    common = {
        "scenario_index": scenario,
        "split": split,
        "run_id": f"s{scenario:03d}-update{target_updates}-stress-v2",
        "global_turn_index": global_turn_index,
        "event_turn_index": ordinal - 1,
        "turn_id": turn_id,
        "event_id": f"stress-s{scenario:03d}-update{target_updates}",
        "timestamp": timestamp,
        "current_turn": {
            "speaker_id": f"synthetic-stress-s{scenario:03d}-{ordinal:03d}",
            "speaker_name": speaker,
            "text": text,
        },
        "train_eligible": False,
    }
    target = {
        "decision": "UPDATE",
        "reason_code": "NEW_VEHICLE_MEMORY",
        "reason": f"The synthetic driver states a unique preference for {route}.",
    }
    operations = [{"op": "add", "target": target_line, "content": content}]
    position_metadata = {
        "update_position": position,
        "target_fact_index_before_update": target_fact_index,
        "fact_count_before_update": fact_count,
        "target_fact_fraction_before_update": target_fact_index / fact_count,
    }
    return common, target, operations, position_metadata


def _insertion_target(memory: str, *, position: str) -> tuple[str, int, int, str]:
    lines = normalize_memory(memory).splitlines()
    fact_line_indexes = [index for index, line in enumerate(lines) if line.startswith("- ")]
    if not fact_line_indexes:
        raise ValueError("Synthetic targeted ADD requires at least one memory fact")
    if position == "front":
        fact_index = 0
    elif position == "middle":
        fact_index = len(fact_line_indexes) // 2
    else:
        raise ValueError(f"Unsupported synthetic update position: {position}")
    line_index = fact_line_indexes[fact_index]
    speaker = "shared-vehicle"
    for candidate in reversed(lines[:line_index]):
        if candidate.startswith("### "):
            speaker = candidate.removeprefix("### ")
            break
    return lines[line_index], fact_index, len(fact_line_indexes), speaker


def _timestamp_after(value: str, ordinal: int) -> str:
    parsed = datetime.fromisoformat(value)
    # Match the minute-resolution timestamps used by the original memory facts.
    # There is at most one synthetic turn per source gap, so no tie-breaker is
    # required.
    _ = ordinal
    shifted = parsed + timedelta(minutes=1)
    return shifted.isoformat(timespec="minutes")


def _write_remapped_quizzes(
    source_root: Path,
    output: Path,
    *,
    scenario: int,
    quiz_split: str,
    source_turn_remap: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    source_path = source_root / "quiz_sft.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path = output / "quiz_sft.jsonl"
    rows = 0
    unique_anchors: set[int] = set()
    query_collisions = 0
    with source_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            if int(row.get("scenario_index", -1)) != scenario:
                continue
            if str(row.get("sft_split")) != quiz_split:
                raise ValueError(
                    f"Quiz split mismatch at quiz_sft.jsonl:{line_number}"
                )
            memory_ref = row.get("memory_ref")
            if not isinstance(memory_ref, Mapping):
                raise TypeError(
                    f"Quiz memory_ref is missing at quiz_sft.jsonl:{line_number}"
                )
            source_turn = int(memory_ref["global_turn_index"])
            remapped = source_turn_remap.get(source_turn)
            if remapped is None:
                raise ValueError(
                    f"Quiz anchor S{scenario} turn {source_turn} is not retained"
                )
            new_turn = int(remapped["global_turn_index"])
            memory = str(remapped["memory"])
            new_ref = {
                **dict(memory_ref),
                "checkpoint_id": f"s{scenario:03d}:memory:{new_turn:05d}",
                "delta_sample_id": f"s{scenario:03d}:delta:{new_turn:05d}",
                "global_turn_index": new_turn,
                "memory_snapshot_sha256": str(remapped["memory_sha256"]),
                "patch_sample_id": f"s{scenario:03d}:patch:{new_turn:05d}",
                "summary_sample_id": f"s{scenario:03d}:summary:{new_turn:05d}",
                "turn_id": str(remapped["turn_id"]),
            }
            messages = _quiz_messages_with_memory(row.get("messages"), memory)
            query = _quiz_query(messages)
            if "stress-route-" in query.lower():
                query_collisions += 1
            remapped_row = {
                **row,
                "memory_ref": new_ref,
                "messages": messages,
                "stress_metadata": {
                    "source_global_turn_index": source_turn,
                    "remapped_global_turn_index": new_turn,
                },
            }
            target.write(_json_line(remapped_row))
            rows += 1
            unique_anchors.add(new_turn)
    if rows == 0:
        raise ValueError(f"Source contains no Quiz rows for S{scenario}")
    if query_collisions:
        raise ValueError("Synthetic route text collides with an original Quiz query")
    return {
        "rows": rows,
        "unique_anchor_turns": len(unique_anchors),
        "query_collisions": query_collisions,
        "quiz_sft_sha256": _sha256_file(output_path),
    }


def _quiz_messages_with_memory(messages: Any, memory: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise TypeError("Quiz messages must be a list")
    result = [dict(message) for message in messages]
    user = next(
        (
            message
            for message in result
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        None,
    )
    if user is None:
        raise ValueError("Quiz has no textual user message")
    content = str(user["content"])
    marker = "\n\n[Current request]\n"
    if not content.startswith("[Memory]\n") or marker not in content:
        raise ValueError("Quiz user message does not use the expected memory template")
    query = content.split(marker, 1)[1]
    user["content"] = f"[Memory]\n{memory or '(empty)'}{marker}{query}"
    return result


def _quiz_query(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and "[Current request]\n" in content:
            return content.split("[Current request]\n", 1)[1]
    raise ValueError("Quiz query is missing")


def _write_compaction(
    handle: Any,
    row: Mapping[str, Any],
    *,
    base_summary: str,
    pending_updates: Sequence[Sequence[Sequence[str]]],
    next_summary: str,
) -> None:
    materialized = _materialize_delta(base_summary, pending_updates)
    if materialized != normalize_memory(next_summary):
        raise ValueError("Compaction replay differs from next summary")
    handle.write(
        _json_line(
            {
                "schema_version": "vehiclemembench-v2-grouped-delta-v2-compaction-v2",
                "scenario_index": row["scenario_index"],
                "split": row["split"],
                "global_turn_index": row["global_turn_index"],
                "turn_id": row["turn_id"],
                "trigger": "UPDATE_INTERVAL",
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


def _materialize_delta(
    base_summary: str, pending_updates: Sequence[Sequence[Sequence[str]]]
) -> str:
    from .methods.delta_v2 import expand_compact_operations

    memory = normalize_memory(base_summary)
    for batch in pending_updates:
        memory, _ = apply_operations(memory, expand_compact_operations(batch))
    return memory


def _build_turn_manifest(
    data_root: Path,
    *,
    scenario: int,
    split: str,
    source_rows: int,
    source_updates: int,
    synthetic_updates: int,
) -> dict[str, Any]:
    catalog = DatasetCatalog(data_root / "catalog.sqlite", data_root)
    records = [catalog.record(row_id) for row_id in catalog.scenario_row_ids(scenario)]
    decision_counts = Counter(record.decision for record in records)
    stable = {
        "schema_version": TURN_MANIFEST_SCHEMA_VERSION,
        "dataset_source_fingerprint": catalog.metadata()["source_fingerprint"],
        "split": split,
        "selection": {
            "mode": "all_stress_scenario_rows",
            "scenarios": [scenario],
            "source_rows_preserved": source_rows,
            "source_updates_preserved": source_updates,
            "synthetic_updates": synthetic_updates,
        },
        "totals": {
            "original_turns": len(records),
            "selected_turns": len(records),
            "gold_updates": decision_counts["UPDATE"],
            "gold_noops": decision_counts["NO_OP"],
        },
        "scenarios": [
            {
                "scenario_index": scenario,
                "original_turns": len(records),
                "selected_turns": len(records),
                "gold_updates": decision_counts["UPDATE"],
                "gold_noops": decision_counts["NO_OP"],
                "required_quiz_anchor_turns": _quiz_anchor_count(
                    data_root / "quiz_sft.jsonl", scenario
                ),
                "rows": [
                    {
                        "row_id": record.row_id,
                        "global_turn_index": record.global_turn_index,
                        "turn_id": record.turn_id,
                        "gold_decision": record.decision,
                    }
                    for record in records
                ],
            }
        ],
    }
    signature = hashlib.sha256(
        json.dumps(
            stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {**stable, "signature": signature, "created_at": _utc_now()}


def _quiz_anchor_count(path: Path, scenario: int) -> int:
    anchors = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["scenario_index"]) == scenario:
                anchors.add(int(row["memory_ref"]["global_turn_index"]))
    return len(anchors)


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
