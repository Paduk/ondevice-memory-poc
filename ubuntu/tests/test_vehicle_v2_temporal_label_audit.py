from __future__ import annotations

import json
from pathlib import Path

import pytest
from memory_training.methods.operations import apply_operations

from palmclaw_ubuntu.vehicle_bench.v2_hybrid import text_sha256
from palmclaw_ubuntu.vehicle_bench.v2_temporal_label_audit import (
    TemporalLabelAuditResponse,
    build_plan_first_audit,
    ground_temporal_audit,
    materialize_audited_dataset,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _update_row(*, action: str = "durable_upsert") -> dict:
    content = (
        "- [2025-01-01 08:00] driver: seat.position; value=48"
    )
    operation = {
        "op": "add",
        "target": "",
        "content": content,
        "identity_key": "driver.seat_position",
        "temporal_action": action,
        "temporal_cue": "",
    }
    after, _ = apply_operations(
        "", [{key: operation[key] for key in ("op", "target", "content")}]
    )
    return {
        "schema_version": "test",
        "sample_id": "s101:temporal_patch:00001",
        "scenario_index": 101,
        "split": "train",
        "event_id": "event-1",
        "current_turn": {
            "speaker_id": "person-1",
            "speaker_name": "Maya",
            "text": "My driver seat position is currently 48.",
        },
        "input": {"previous_memory": ""},
        "target": {
            "decision": "UPDATE",
            "reason_code": "NEW_VEHICLE_MEMORY",
            "reason": "Supported update.",
            "operations": [operation],
        },
        "provenance": {
            "source_update_indexes": [0],
            "before_memory_sha256": text_sha256(""),
            "after_memory_sha256": text_sha256(after),
        },
    }


def test_plan_first_audit_recovers_current_upsert(tmp_path: Path) -> None:
    source_patch = tmp_path / "source" / "patch.jsonl"
    source_patch.parent.mkdir()
    source_patch.write_text(
        json.dumps(_update_row(), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    plan = {
        "model_id": "gpt-5.6-terra",
        "artifact_sha256": "plan-hash",
        "temporal_transitions": [
            {
                "source_event_id": "event-1",
                "source_update_index": 0,
                "temporal_action": "current_upsert",
                "temporal_cue": "",
                "baseline_event_id": None,
                "baseline_source_update_index": None,
            }
        ],
    }
    _write_json(
        tmp_path
        / "temporal"
        / "hybrid-temporal-anchor-terra-t01-r2"
        / "temporal-plan.json",
        plan,
    )

    records, summary = build_plan_first_audit(
        source_patch=source_patch,
        temporal_source_root=tmp_path / "temporal",
        temporal_indices=(1,),
    )

    assert summary["verdicts"] == {"CORRECT": 1}
    assert summary["after_actions"] == {"current_upsert": 1}
    assert records[0]["effective_label"]["temporal_action"] == "current_upsert"
    assert records[0]["operation"]["content"] == _update_row()["target"][
        "operations"
    ][0]["content"]


def test_live_terra_contract_cannot_mutate_on_accept() -> None:
    row = _update_row()
    case = {
        "scenario_index": 101,
        "sample_id": row["sample_id"],
        "operation_index": 0,
        "current_turn": row["current_turn"],
        "previous_memory": "",
        "operation": row["target"]["operations"][0],
        "plan_transition": None,
    }
    response = TemporalLabelAuditResponse(
        verdict="ACCEPT",
        temporal_action="current_upsert",
        identity_key="driver.seat_position",
        temporal_cue="",
        reason="The turn explicitly describes a current observation.",
    )

    with pytest.raises(ValueError, match="ACCEPT cannot change"):
        ground_temporal_audit(case, response)


def test_materialize_keeps_patch_replay_identical(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    row = _update_row()
    (source_root / "patch.jsonl").write_text(
        json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    _write_json(source_root / "manifest.json", {"schema_version": "source"})
    record = {
        "sample_id": row["sample_id"],
        "operation_index": 0,
        "verdict": "CORRECT",
        "effective_label": {
            "identity_key": "driver.seat_position",
            "temporal_action": "current_upsert",
            "temporal_cue": "",
        },
    }

    result = materialize_audited_dataset(
        source_root=source_root,
        output_root=tmp_path / "audited",
        audit_records=[record],
        audit_summary={"verdicts": {"CORRECT": 1}},
    )

    stored = json.loads(
        (tmp_path / "audited" / "patch.jsonl").read_text(encoding="utf-8")
    )
    operation = stored["target"]["operations"][0]
    assert operation["temporal_action"] == "current_upsert"
    assert operation["op"] == "add"
    assert operation["content"] == row["target"]["operations"][0]["content"]
    assert result["replay_passed"] is True
