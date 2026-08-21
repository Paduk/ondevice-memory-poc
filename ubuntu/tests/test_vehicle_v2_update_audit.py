from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventRecord,
    V1NamedValue,
    V1PersonaRecord,
    V1PreferenceUpdate,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    V1DialogueEventPayload,
    V1DialogueLine,
    V1GeneratedEventDialogue,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridEvidence,
    V2HybridPatchOperation,
    V2HybridTurnLabel,
    dialogue_sha256,
    event_sha256,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_audit import (
    UpdateAuditArtifactPaths,
    UpdateAuditBundle,
    _hybrid_records,
    _native_records,
    audit_turn_record,
    select_event_audit_cases,
)


def _persona() -> V1PersonaRecord:
    values = (V1NamedValue(key="key", value="value"),)
    return V1PersonaRecord(
        persona_id="p0",
        name="Person 0",
        basic_profile=values,
        cultural_interests=values,
        lifestyle_habits=values,
        vehicle_preferences=values,
    )


def _event() -> V1EventRecord:
    return V1EventRecord(
        event_id="vehicle-e1",
        timestamp="2025-02-01T10:00",
        description="Person 0 states a HUD preference.",
        participant_ids=("p0",),
        preference_updates=(
            V1PreferenceUpdate(
                subject_id="p0",
                attribute_path="carcontrol_HUD_set_brightness_level.level",
                new_value=7,
            ),
        ),
    )


def _turn() -> V1DialogueTurnRecord:
    return V1DialogueTurnRecord(
        turn_id="vehicle-e1-turn-001",
        source_event_id="vehicle-e1",
        timestamp="2025-02-01T10:00",
        speaker_id="p0",
        speaker_name="Person 0",
        text="I prefer the HUD brightness at level 7.",
    )


def _label(memory: str) -> V2HybridTurnLabel:
    turn = _turn()
    return V2HybridTurnLabel(
        global_turn_index=0,
        event_turn_index=0,
        turn_id=turn.turn_id,
        source_event_id=turn.source_event_id,
        decision="UPDATE",
        reason_code="NEW_VEHICLE_MEMORY",
        reason="The driver states a new HUD preference.",
        source_update_indexes=(0,),
        evidence=(
            V2HybridEvidence(
                source_event_id=turn.source_event_id,
                event_turn_index=0,
                turn_id=turn.turn_id,
                quote="HUD brightness at level 7",
            ),
        ),
        operations=(
            V2HybridPatchOperation(op="add", target="", content=memory),
        ),
        before_memory_sha256=text_sha256(""),
        after_memory_sha256=text_sha256(memory),
        after_memory=memory,
    )


def _stage2() -> SimpleNamespace:
    event = _event()
    source = SimpleNamespace(
        timeline_index=0,
        chain_kind="vehicle",
        reasoning_type="preference_conflict",
        event=event,
    )
    return SimpleNamespace(
        artifact_sha256="a" * 64,
        interleaved_timeline=(source,),
        persona_group=SimpleNamespace(
            payload=SimpleNamespace(personas=(_persona(),))
        ),
    )


@pytest.mark.parametrize("method", ["post_hoc", "hybrid"])
def test_hybrid_and_posthoc_adapters_build_grounded_records(
    tmp_path,
    method,
) -> None:
    stage2 = _stage2()
    turn = _turn()
    generated = V1GeneratedEventDialogue(
        event_id=_event().event_id,
        source_event_sha256=event_sha256(_event()),
        target_turn_count=1,
        payload=V1DialogueEventPayload(
            turns=(V1DialogueLine(speaker_id="p0", text=turn.text),)
        ),
        model_id="fake",
        prompt_version="test",
        input_sha256="b" * 64,
        usage={},
    )
    dialogue_root = tmp_path / "dialogues"
    dialogue_root.mkdir()
    (dialogue_root / "vehicle-e1.json").write_text(
        generated.model_dump_json(),
        encoding="utf-8",
    )
    memory = "- Person 0: HUD brightness=7"
    checkpoint = SimpleNamespace(
        event_id="vehicle-e1",
        source_event_sha256=event_sha256(_event()),
        source_dialogue_sha256=dialogue_sha256((turn,)),
        before_memory_sha256=text_sha256(""),
        turn_labels=(_label(memory),),
        after_memory=memory,
    )
    artifact = SimpleNamespace(
        artifact_sha256="c" * 64,
        event_checkpoints=(checkpoint,),
        final_memory=memory,
    )
    paths = UpdateAuditArtifactPaths(
        method=method,
        scenario_index=1,
        root=tmp_path,
        memory_artifact=tmp_path / "hybrid.json",
        stage2_artifact=tmp_path / "stage2.json",
        dialogue_root=dialogue_root,
    )

    records, errors = _hybrid_records(paths, stage2, artifact)

    assert errors == ()
    assert len(records) == 1
    assert records[0].method == method
    assert records[0].dialogue_prefix[0].text == turn.text
    assert audit_turn_record(records[0]).status == "PASS"


def test_native_adapter_uses_checkpoint_memory_and_commit_state(tmp_path) -> None:
    stage2 = _stage2()
    memory = "- Person 0: HUD brightness=7"
    checkpoint = SimpleNamespace(
        dialogue_turn=_turn(),
        label=_label(memory),
        after_memory=memory,
        committed_source_update_indexes=(0,),
    )
    event_artifact = SimpleNamespace(
        event_id="vehicle-e1",
        turn_checkpoints=(checkpoint,),
        after_memory=memory,
    )
    artifact = SimpleNamespace(
        artifact_sha256="d" * 64,
        event_artifacts=(event_artifact,),
        final_memory=memory,
    )
    paths = UpdateAuditArtifactPaths(
        method="native_turnwise",
        scenario_index=1,
        root=tmp_path,
        memory_artifact=tmp_path / "native.json",
        stage2_artifact=tmp_path / "stage2.json",
        dialogue_root=None,
    )

    records, errors = _native_records(paths, stage2, artifact)

    assert errors == ()
    assert records[0].committed_source_update_indexes_before == ()
    assert records[0].committed_source_update_indexes_after == (0,)
    assert audit_turn_record(records[0]).status == "PASS"

    bundle = UpdateAuditBundle(
        paths=paths,
        records=records,
        audits=(audit_turn_record(records[0]),),
        artifact_errors=(),
    )
    cases = select_event_audit_cases(bundle, no_op_event_sample_rate=0)
    assert len(cases) == 1
    assert cases[0].selection_reason == "EXPECTED_OR_GENERATED_UPDATE"
    assert cases[0].candidate_updates[0]["record_id"] == records[0].record_id


def test_deterministic_audit_rejects_patch_and_evidence_corruption(tmp_path) -> None:
    stage2 = _stage2()
    memory = "- Person 0: HUD brightness=7"
    checkpoint = SimpleNamespace(
        dialogue_turn=_turn(),
        label=_label(memory),
        after_memory=memory,
        committed_source_update_indexes=(0,),
    )
    artifact = SimpleNamespace(
        artifact_sha256="d" * 64,
        event_artifacts=(
            SimpleNamespace(
                event_id="vehicle-e1",
                turn_checkpoints=(checkpoint,),
                after_memory=memory,
            ),
        ),
        final_memory=memory,
    )
    paths = UpdateAuditArtifactPaths(
        method="native_turnwise",
        scenario_index=1,
        root=tmp_path,
        memory_artifact=tmp_path / "native.json",
        stage2_artifact=tmp_path / "stage2.json",
        dialogue_root=None,
    )
    record = _native_records(paths, stage2, artifact)[0][0]
    corrupted = replace(
        record,
        operations=(
            {"op": "replace", "target": "- missing", "content": memory},
        ),
        evidence=({**record.evidence[0], "quote": "not in dialogue"},),
    )

    result = audit_turn_record(corrupted)

    assert result.status == "FAIL"
    assert "PATCH_APPLY_FAILED" in result.error_codes
    assert "EVIDENCE_QUOTE_MISSING" in result.error_codes
    assert "RECORD_INPUT_HASH_MISMATCH" in result.error_codes
