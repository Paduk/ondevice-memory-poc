from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from palmclaw_ubuntu.cli import build_parser
from palmclaw_ubuntu.vehicle_bench.diagnostics import (
    run_vehicle_r0_diagnostics,
)


def test_vehicle_r0_replays_proposals_and_routes_without_writes(
    tmp_path: Path,
) -> None:
    dataset_hash = "a" * 64
    cache_key = "b" * 64
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cache_root = tmp_path / "memory"
    database_path = (
        cache_root
        / dataset_hash[:16]
        / "scenario-01"
        / cache_key[:16]
        / "memory.db"
    )
    database_path.parent.mkdir(parents=True)
    _create_r0_database(database_path)
    _write_fixture_run(
        run_dir,
        dataset_hash=dataset_hash,
        cache_key=cache_key,
    )
    modified_before = database_path.stat().st_mtime_ns

    result = run_vehicle_r0_diagnostics(
        run_dir,
        memory_cache_dir=cache_root,
    )

    proposals = result.summary["proposal_replay"]
    assert proposals["raw_proposal_count"] == 3
    assert proposals["unique_proposal_count"] == 2
    assert proposals["raw_status_counts"] == {"applied": 1, "rejected": 2}
    assert proposals["unique_status_counts"] == {"applied": 1, "rejected": 1}
    assert proposals["unique_relaxed_preview_counts"] == {
        "accepted_existing": 1,
        "would_accept": 1,
    }
    routing = result.summary["routing_replay"]
    assert routing["zero_selected_tasks"] == 1
    assert routing["tool_misrouting_tasks"] == 1
    assert routing["answer_memory_route_misses"] == 1
    assert routing["classifier_used_tasks"] == 0
    assert database_path.stat().st_mtime_ns == modified_before
    assert (result.output_dir / "r0-summary.json").is_file()
    assert (result.output_dir / "proposal-replay.jsonl").is_file()
    assert (result.output_dir / "routing-replay.jsonl").is_file()
    assert (result.output_dir / "results.md").is_file()


def test_vehicle_r0_cli_requires_frozen_run_and_cache_paths() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "vehicle-r0",
            "--run-dir",
            "/tmp/run",
            "--memory-cache-dir",
            "/tmp/cache",
        ]
    )

    assert args.run_dir == Path("/tmp/run")
    assert args.memory_cache_dir == Path("/tmp/cache")
    assert args.output_dir is None


def _create_r0_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE memory_patch_runs (
            id TEXT PRIMARY KEY
        );
        CREATE TABLE memory_patch_proposals (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            target_record_id TEXT,
            patch_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            status TEXT NOT NULL,
            rejection_reason TEXT,
            validation_json TEXT,
            sequence_index INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE TABLE tool_memory_records (
            id TEXT PRIMARY KEY,
            tool_domain TEXT NOT NULL,
            topic TEXT NOT NULL,
            entity_id TEXT,
            conditions_json TEXT NOT NULL,
            value_json TEXT NOT NULL,
            status TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO memory_patch_runs(id) VALUES (?)",
        (("run-1",), ("run-2",), ("run-3",)),
    )
    applied_patch = {
        "operation": "ADD",
        "identity": {
            "user_id": "vehicle_scenario_1",
            "tool_domain": "navigation",
            "topic": "voice_mode",
            "scope": "conditional",
            "scope_key": "vehicle_scenario_1",
            "conditions": {"person": "Patricia"},
        },
        "value": {"mode": "detailed"},
    }
    rejected_patch = {
        "operation": "ADD",
        "identity": {
            "user_id": "vehicle_scenario_1",
            "tool_domain": "seat",
            "topic": "headrest_height",
            "scope": "conditional",
            "scope_key": "vehicle_scenario_1",
            "conditions": {"person": "Patricia"},
        },
        "value": {"seat": "Patricia", "value": 3},
    }
    evidence_one = [{"message_id": 1, "quote": "Voice mode is detailed."}]
    evidence_two = [{"message_id": 2, "quote": "Lower the headrest to three."}]
    connection.execute(
        """
        INSERT INTO memory_patch_proposals VALUES (
            'proposal-1', 'run-1', 'ADD', NULL, ?, ?, 'applied',
            NULL, '{"accepted":true}', 0, '2026-01-01'
        )
        """,
        (json.dumps(applied_patch), json.dumps(evidence_one)),
    )
    for index, run_id in enumerate(("run-2", "run-3"), start=2):
        connection.execute(
            """
            INSERT INTO memory_patch_proposals VALUES (
                ?, ?, 'ADD', NULL, ?, ?, 'rejected',
                'unsupported_value: literal mismatch',
                '{"accepted":false,"code":"unsupported_value"}',
                0, ?
            )
            """,
            (
                f"proposal-{index}",
                run_id,
                json.dumps(rejected_patch),
                json.dumps(evidence_two),
                f"2026-01-0{index}",
            ),
        )
    connection.execute(
        """
        INSERT INTO tool_memory_records VALUES (
            'record-1',
            'navigation',
            'voice_mode',
            'patricia',
            '{"person":"patricia"}',
            '{"mode":"detailed"}',
            'active'
        )
        """
    )
    connection.commit()
    connection.close()


def _write_fixture_run(
    run_dir: Path,
    *,
    dataset_hash: str,
    cache_key: str,
) -> None:
    manifest = {
        "status": "completed",
        "config": {
            "profiles": ["cloud_schema_patch"],
            "benchmark": {"dataset_sha256": dataset_hash},
        },
    }
    case = {
        "profile": "cloud_schema_patch",
        "task_id": "vehicle-01-00",
        "scenario_index": 1,
        "reasoning_type": "preference_conflict",
        "context": {
            "retrieval_metadata": {
                "cache_key": cache_key,
                "tool_memory_route": {
                    "ambiguous": False,
                    "classifier_used": False,
                },
                "tool_memory_retrieval_candidate_count": 0,
                "tool_memory_retrieval_selected_count": 0,
            },
            "selector_tool_names": ["carcontrol_navigation_set_map_view"],
        },
        "reference_calls": [
            {
                "name": "carcontrol_navigation_set_voice_mode",
                "args": {"mode": "detailed"},
            }
        ],
        "score": {"exact_state_match": False},
        "retrieval_quality": {"recall_at_k": 0.0},
        "memory_trace": {"retrieval": {"candidates": []}},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (run_dir / "cases.jsonl").write_text(
        json.dumps(case) + "\n",
        encoding="utf-8",
    )
