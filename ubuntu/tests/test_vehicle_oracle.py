from __future__ import annotations

import json
from pathlib import Path

import pytest

from palmclaw_ubuntu.fact_memory import FactMemoryLinker
from palmclaw_ubuntu.models import FactMemoryCandidate, MemoryEvidence
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.vehicle_bench import (
    OracleGateAnnotations,
    OracleGateLabel,
    apply_oracle_full_overlay,
    apply_oracle_gate_overlay,
    apply_oracle_stage_fact_overlay,
    cumulative_oracle_stages,
    load_oracle_gate_annotations,
    load_oracle_retrieval_annotations,
    load_oracle_stage_fact_annotations,
    load_vehicle_benchmark,
    oracle_fact_candidate_sha256,
    oracle_mode_spec,
    run_oracle_contract_audit,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _benchmark(root: Path) -> Path:
    _write(
        root / "evaluation/functions_schema.json",
        json.dumps(
            [
                {
                    "name": "carcontrol_HUD_set_brightness_level",
                    "description": "Set HUD brightness.",
                    "parameters": {
                        "type": "object",
                        "properties": {"level": {"type": "integer"}},
                        "required": ["level"],
                    },
                }
            ]
        ),
    )
    _write(
        root / "benchmark/qa_data/qa_1.json",
        json.dumps(
            {
                "related_to_vehicle_preference": [
                    {
                        "gold_memory": (
                            "[January 1, 2025] Gary said: "
                            "'Set HUD brightness to 8.'"
                        ),
                        "reasoning_type": "preference_conflict",
                        "query": "Apply Gary's HUD preference.",
                        "new_answer": [
                            "carcontrol_HUD_set_brightness_level(level=8)"
                        ],
                    }
                ]
            }
        ),
    )
    _write(
        root / "benchmark/history/history_1.txt",
        "[2025-01-01 08:00] Gary: Set HUD brightness to 8.\n",
    )
    _write(root / "environment/__init__.py", "")
    _write(
        root / "environment/vehicleworld.py",
        "class VehicleWorld:\n    pass\n",
    )
    _write(root / "evaluation/eval_utils.py", "")
    return root


def test_oracle_modes_keep_gold_arguments_outside_binding():
    for name in (
        "baseline_fact",
        "oracle_extraction",
        "oracle_structure",
        "oracle_gate",
        "oracle_route",
        "oracle_retrieval",
    ):
        assert oracle_mode_spec(name).exposes_gold_arguments is False

    assert oracle_mode_spec("oracle_route").exposes_gold_tool_names is True
    assert oracle_mode_spec("oracle_binding").exposes_gold_arguments is True
    assert cumulative_oracle_stages("retrieval") == (
        "extraction",
        "structure",
        "gate",
        "route",
        "retrieval",
    )
    with pytest.raises(ValueError, match="Unknown"):
        oracle_mode_spec("oracle_everything")


def test_oracle_contract_audit_links_gold_evidence_and_writes_artifacts(
    tmp_path: Path,
):
    dataset = load_vehicle_benchmark(_benchmark(tmp_path / "bench"), strict=False)

    result = run_oracle_contract_audit(
        dataset,
        scenario_indices=(1,),
        output_root=tmp_path / "artifacts",
    )

    assert result.status == "passed"
    assert result.audit["summary"] == {
        "tasks": 1,
        "missing_gold_memory": 0,
        "missing_gold_calls": 0,
        "unknown_gold_tool_calls": 0,
        "duplicate_gold_calls": 0,
        "gold_memory_lines": 1,
        "gold_tool_calls": 1,
        "gold_argument_leaves": 1,
        "tasks_with_quoted_evidence": 1,
        "tasks_with_all_quotes_linked": 1,
        "quoted_evidence": 1,
        "history_linked_quotes": 1,
        "fact_annotation_tasks": 0,
        "fact_annotation_status": "required",
    }
    assert result.audit["baseline"]["compatible"] is True
    assert (result.artifact_dir / "manifest.json").is_file()
    assert (result.artifact_dir / "oracle-contract-audit.json").is_file()
    assert (result.artifact_dir / "results.md").is_file()


def test_oracle_retrieval_annotations_require_complete_dataset_coverage(
    tmp_path: Path,
):
    dataset = load_vehicle_benchmark(_benchmark(tmp_path / "bench"), strict=False)
    annotation = tmp_path / "oracle.json"
    annotation.write_text(
        json.dumps(
            {
                "version": "vehiclemembench-oracle-retrieval-v1",
                "benchmark": {
                    "dataset_sha256": dataset.manifest.dataset_sha256,
                    "tool_schema_sha256": dataset.manifest.tool_schema_sha256,
                },
                "scenario_indices": [1],
                "record_present": [
                    {
                        "task_id": "vehicle-01-00",
                        "record_keys": [
                            {
                                "entity_id": "gary",
                                "predicate": "hud_brightness",
                                "value": 8,
                            }
                        ],
                    }
                ],
                "record_absent": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_oracle_retrieval_annotations(annotation, dataset=dataset)

    assert loaded.require("vehicle-01-00").record_status == "record_present"
    assert loaded.manifest()["record_present_tasks"] == 1

    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["record_present"] = []
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_oracle_retrieval_annotations(annotation, dataset=dataset)


def test_oracle_gate_annotations_require_complete_dataset_coverage(
    tmp_path: Path,
):
    dataset = load_vehicle_benchmark(_benchmark(tmp_path / "bench"), strict=False)
    annotation = tmp_path / "gate.json"
    payload = {
        "version": "vehiclemembench-oracle-gate-v1",
        "benchmark": {
            "dataset_sha256": dataset.manifest.dataset_sha256,
            "tool_schema_sha256": dataset.manifest.tool_schema_sha256,
        },
        "scenario_indices": [1],
        "gate_recoverable": [
            {
                "task_id": "vehicle-01-00",
                "candidate_sha256": "a" * 64,
                "original_status": "review",
                "validation_code": "semantic_review",
            }
        ],
        "not_gate_recoverable": [],
    }
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_oracle_gate_annotations(annotation, dataset=dataset)

    assert (
        loaded.require("vehicle-01-00").candidate_status
        == "gate_recoverable"
    )
    assert loaded.manifest()["gate_recoverable_tasks"] == 1

    payload["gate_recoverable"] = []
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_oracle_gate_annotations(annotation, dataset=dataset)


def test_oracle_gate_applies_reviewed_candidate_only_to_isolated_clone(
    tmp_path: Path,
):
    source = tmp_path / "gate.json"
    source.write_text("{}", encoding="utf-8")
    with SQLiteRepository(tmp_path / "memory.db") as repository:
        session = repository.create_session("Oracle Gate")
        turn_id = repository.create_turn(session.id)
        quote = "Brian prefers voice guidance muted in the morning."
        message_id = repository.append_message(
            session.id,
            "user",
            quote,
            turn_id=turn_id,
        )
        repository.finish_turn(turn_id, "completed", 1, "test")
        run_id = repository.begin_memory_patch_run(
            session_id=session.id,
            source_turn_id=None,
            backend="fake",
            model_id="fact-model",
            prompt_version="fact-v1",
            schema_version="fact-v1",
            status="running",
        )
        candidate = FactMemoryCandidate(
            entity_id="Brian",
            predicate="voice guidance preference",
            value={"enabled": False},
            identity_conditions={},
            applicability={"time_of_day": "morning"},
            capability_hints=("voice guidance",),
            memory_type="preference",
            confidence=0.4,
            evidence=(MemoryEvidence(message_id=message_id, quote=quote),),
        )
        outcome = FactMemoryLinker(
            repository,
            user_id="vehicle_scenario_1",
        ).apply_candidates(
            run_id=run_id,
            session_id=session.id,
            candidates=(candidate,),
        )
        assert outcome.outcomes[0].status == "review"
        event = repository.fact_memory_candidate_events(run_id)[0]
        digest = oracle_fact_candidate_sha256(
            event["candidate"],
            event["evidence"],
        )
        annotations = OracleGateAnnotations(
            version="vehiclemembench-oracle-gate-v1",
            dataset_sha256="dataset",
            tool_schema_sha256="tools",
            scenario_indices=(1,),
            labels={
                "vehicle-01-00": OracleGateLabel(
                    task_id="vehicle-01-00",
                    scenario_index=1,
                    candidate_status="gate_recoverable",
                    candidate_sha256=digest,
                    original_status="review",
                    validation_code="confidence_below_minimum",
                )
            },
            source_path=source,
        )

        isolated = repository.clone_in_memory()
        try:
            overlay = apply_oracle_gate_overlay(
                isolated,
                session_id=session.id,
                user_id="vehicle_scenario_1",
                annotations=annotations,
                scenario_index=1,
            )
            records = isolated.list_fact_memory_records(session.id)
            assert overlay.applied_candidates == 1
            assert len(records) == 1
            assert records[0].predicate == "voice_guidance_preference"
            assert records[0].value == {"enabled": False}
            assert repository.list_fact_memory_records(session.id) == []
        finally:
            isolated.close()


def test_oracle_extraction_fact_annotation_applies_only_to_isolated_clone(
    tmp_path: Path,
):
    dataset = load_vehicle_benchmark(_benchmark(tmp_path / "bench"), strict=False)
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "version": "vehiclemembench-oracle-retrieval-v1",
                "benchmark": {
                    "dataset_sha256": dataset.manifest.dataset_sha256,
                    "tool_schema_sha256": dataset.manifest.tool_schema_sha256,
                },
                "scenario_indices": [1],
                "record_present": [],
                "record_absent": ["vehicle-01-00"],
            }
        ),
        encoding="utf-8",
    )
    retrieval = load_oracle_retrieval_annotations(
        retrieval_path,
        dataset=dataset,
    )
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(
        json.dumps(
            {
                "version": "vehiclemembench-oracle-stage-facts-v1",
                "benchmark": {
                    "dataset_sha256": dataset.manifest.dataset_sha256,
                    "tool_schema_sha256": dataset.manifest.tool_schema_sha256,
                },
                "scenario_indices": [1],
                "gate_only_tasks": [],
                "structure_facts": [],
                "extraction_facts": [
                    {
                        "fact_id": "gary-hud-8",
                        "task_ids": ["vehicle-01-00"],
                        "entity_id": "gary",
                        "predicate": "hud_brightness",
                        "value": 8,
                        "identity_conditions": {},
                        "applicability": {},
                        "memory_type": "preference",
                        "capability_hints": ["hud", "brightness"],
                        "evidence_quotes": ["Set HUD brightness to 8."],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    facts = load_oracle_stage_fact_annotations(
        facts_path,
        dataset=dataset,
        retrieval_annotations=retrieval,
    )

    with SQLiteRepository(tmp_path / "memory.db") as repository:
        session = repository.create_session("Oracle Extraction")
        turn_id = repository.create_turn(session.id)
        repository.append_message(
            session.id,
            "user",
            "[2025-01-01 08:00] Gary: Set HUD brightness to 8.",
            turn_id=turn_id,
        )
        repository.finish_turn(turn_id, "completed", 1, "test")
        isolated = repository.clone_in_memory()
        try:
            overlay = apply_oracle_stage_fact_overlay(
                isolated,
                session_id=session.id,
                user_id="vehicle_scenario_1",
                annotations=facts,
                scenario_index=1,
                stage="extraction",
            )

            record_id = overlay.task_record_ids["vehicle-01-00"][0]
            record = isolated.fact_memory_record(record_id)
            assert overlay.outcome_counts == {"applied": 1}
            assert record is not None
            assert record.entity_id == "gary"
            assert record.predicate == "hud_brightness"
            assert record.value == 8
            assert repository.list_fact_memory_records(session.id) == []
        finally:
            isolated.close()

        full_isolated = repository.clone_in_memory()
        try:
            gate_annotations = OracleGateAnnotations(
                version="vehiclemembench-oracle-gate-v1",
                dataset_sha256=dataset.manifest.dataset_sha256,
                tool_schema_sha256=dataset.manifest.tool_schema_sha256,
                scenario_indices=(1,),
                labels={
                    "vehicle-01-00": OracleGateLabel(
                        task_id="vehicle-01-00",
                        scenario_index=1,
                        candidate_status="not_gate_recoverable",
                    )
                },
                source_path=facts_path,
            )
            full = apply_oracle_full_overlay(
                full_isolated,
                session_id=session.id,
                user_id="vehicle_scenario_1",
                stage_annotations=facts,
                gate_annotations=gate_annotations,
                scenario_index=1,
            )

            assert full.as_dict()["covered_tasks"] == 1
            assert full.as_dict()["unique_records"] == 1
            assert full.task_record_ids["vehicle-01-00"]
            assert repository.list_fact_memory_records(session.id) == []
        finally:
            full_isolated.close()
