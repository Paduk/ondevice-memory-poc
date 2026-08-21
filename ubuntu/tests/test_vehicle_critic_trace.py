from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from palmclaw_ubuntu.vehicle_bench.critic_trace import (
    COMBINED_PROMPT_VERSION,
    COMBINED_SCHEMA_VERSION,
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SCHEMA_VERSION,
    build_r2_source_inventory,
    discover_r2_trace_source,
    extract_source_turns,
    resolve_r2_experiment_root,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_source(
    root: Path,
    *,
    method: str,
    scenario_index: int = 1,
    corrupt_after_hash: bool = False,
    corrupt_before_hash: bool = False,
) -> Path:
    experiment_root = resolve_r2_experiment_root(
        root,
        method=method,  # type: ignore[arg-type]
        scenario_index=scenario_index,
    )
    cache_dir = experiment_root / "cache" / "dataset" / "scenario-01" / "key"
    cache_dir.mkdir(parents=True)
    database_path = cache_dir / "memory.db"
    prompt_version, schema_version = (
        (SUMMARY_PROMPT_VERSION, SUMMARY_SCHEMA_VERSION)
        if method == "summary"
        else (COMBINED_PROMPT_VERSION, COMBINED_SCHEMA_VERSION)
    )
    manifest = {
        "cache_key": "cache-key",
        "config": {
            "dataset_sha256": "dataset-sha256",
            "scenario_index": scenario_index,
            "history_sha256": "history-sha256",
            "recursive_summary": {
                "model_id": "gpt-5.6-luna",
                "prompt_version": prompt_version,
                "schema_version": schema_version,
                "max_memory_chars": 8192,
                "update_cadence": "history_entry",
            },
        },
        "history": {"line_count": 3},
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
        CREATE TABLE consolidation_runs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            start_message_id INTEGER NOT NULL,
            end_message_id INTEGER NOT NULL,
            backend TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            schema_version TEXT,
            status TEXT NOT NULL,
            output TEXT NOT NULL,
            usage_json TEXT NOT NULL,
            started_at TEXT NOT NULL
        );
        CREATE TABLE model_calls (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            consolidation_run_id TEXT,
            role TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            schema_version TEXT,
            latency_ms INTEGER NOT NULL,
            usage_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    memories = ("", "- Gary\n  - Ambient color: green.", "")
    current_memory = ""
    for turn_index, output in enumerate(memories):
        message_id = turn_index + 1
        run_id = f"run-{turn_index}"
        decision = "updated" if output else "noop"
        after_memory = output or current_memory
        metadata = {
            "date": "2025-03-01",
            "update_status": decision,
            "update_mode": (
                "full_rewrite"
                if method == "summary"
                else "deterministic_temporal_patch_with_periodic_compaction"
            ),
        }
        if method == "combined":
            metadata.update(
                {
                    "memory_sha256_before": (
                        "bad-before-hash"
                        if corrupt_before_hash and turn_index == 1
                        else _sha256(current_memory)
                    ),
                    "memory_sha256_after": (
                        "bad-hash"
                        if corrupt_after_hash and turn_index == 1
                        else _sha256(after_memory)
                    ),
                }
            )
        usage = {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "cached_tokens": 0,
        }
        connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (
                message_id,
                "session-1",
                "user",
                f"[VehicleMemBench history turn={turn_index}]\nTurn {turn_index}",
            ),
        )
        connection.execute(
            "INSERT INTO consolidation_runs "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                "session-1",
                message_id,
                message_id,
                "openai",
                "gpt-5.6-luna",
                prompt_version,
                schema_version,
                "completed",
                output,
                json.dumps(usage),
                f"2026-08-15T00:00:0{turn_index}+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO model_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"call-{turn_index}",
                "session-1",
                run_id,
                "memory",
                "gpt-5.6-luna",
                prompt_version,
                schema_version,
                123,
                json.dumps(usage),
                json.dumps(metadata),
                f"2026-08-15T00:00:0{turn_index}+00:00",
            ),
        )
        current_memory = after_memory
    connection.commit()
    connection.close()
    return database_path


def test_extract_source_turns_reconstructs_before_and_after_memory(
    tmp_path: Path,
) -> None:
    _write_source(tmp_path, method="summary")
    source = discover_r2_trace_source(
        tmp_path,
        method="summary",
        scenario_index=1,
    )

    turns = extract_source_turns(source)

    assert [turn.original_decision for turn in turns] == [
        "NO_OP",
        "UPDATE",
        "NO_OP",
    ]
    assert turns[0].original_before_memory == ""
    assert turns[0].original_after_memory == ""
    assert turns[1].original_before_memory == ""
    assert turns[1].original_after_memory.endswith("Ambient color: green.")
    assert turns[2].original_before_memory == turns[1].original_after_memory
    assert turns[2].original_after_memory == turns[1].original_after_memory
    assert turns[2].turn_index == 2
    assert turns[2].message_id == 3


def test_extract_combined_source_records_post_provider_storage_transform(
    tmp_path: Path,
) -> None:
    _write_source(tmp_path, method="combined", corrupt_after_hash=True)
    source = discover_r2_trace_source(
        tmp_path,
        method="combined",
        scenario_index=1,
    )

    turns = extract_source_turns(source)

    assert turns[1].storage_transformed is True
    assert turns[1].recorded_provider_after_sha256 == "bad-hash"
    assert turns[0].storage_transformed is False


def test_extract_combined_source_rejects_broken_before_memory_chain(
    tmp_path: Path,
) -> None:
    _write_source(tmp_path, method="combined", corrupt_before_hash=True)
    source = discover_r2_trace_source(
        tmp_path,
        method="combined",
        scenario_index=1,
    )

    with pytest.raises(ValueError, match="before-memory hash mismatch"):
        extract_source_turns(source)


def test_build_r2_source_inventory_covers_both_methods(tmp_path: Path) -> None:
    _write_source(tmp_path, method="summary")
    _write_source(tmp_path, method="combined")

    inventory = build_r2_source_inventory(tmp_path, scenario_indices=(1,))

    assert inventory["source_repetition"] == 2
    assert inventory["scenario_indices"] == [1]
    assert inventory["totals"] == {
        "summary": {"scenarios": 1, "turns": 3, "updates": 1, "noops": 2},
        "combined": {"scenarios": 1, "turns": 3, "updates": 1, "noops": 2},
    }
    assert len(inventory["sources"]) == 2
    assert all(len(item["database_sha256"]) == 64 for item in inventory["sources"])
