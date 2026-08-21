from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import palmclaw_ubuntu.vehicle_bench.v2_native_turnwise as native_turnwise
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventRecord,
    V1NamedValue,
    V1PersonaRecord,
    V1PreferenceUpdate,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1DialogueGenerationContext
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridUpdateAlignment,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_native_turnwise import (
    V2_NATIVE_ALIGNMENT_PROMPT_VERSION,
    V2_NATIVE_TURN_PROMPT_VERSION,
    OpenAIV2NativeAlignmentModel,
    OpenAIV2NativeTurnGenerationModel,
    V2GeneratedNativeAlignment,
    V2GeneratedNativeTurn,
    V2NativeAlignmentPayload,
    V2NativeTurnPayload,
    assemble_native_event_dialogue,
    build_native_turn_checkpoint,
    deterministic_native_no_op_alignment,
    materialize_native_turn,
    native_causal_prefix_sha256,
    render_native_alignment_input,
    render_native_turn_generation_input,
    run_native_event,
    run_native_scenario,
    validate_native_alignment_payload,
    validate_native_checkpoint_chain,
    validate_native_turn_payload,
    write_native_turn_checkpoint,
)


def _personas() -> tuple[V1PersonaRecord, ...]:
    return tuple(
        V1PersonaRecord(
            persona_id=f"p{index}",
            name=f"Person {index}",
            basic_profile=(V1NamedValue(key="age", value=str(30 + index)),),
            cultural_interests=(V1NamedValue(key="music", value="jazz"),),
            lifestyle_habits=(V1NamedValue(key="travel", value="weekly"),),
            vehicle_preferences=(V1NamedValue(key="HUD", value="adaptive"),),
        )
        for index in range(3)
    )


def _context() -> V1DialogueGenerationContext:
    event = V1EventRecord(
        event_id="vehicle-e1",
        timestamp="2025-02-01T10:00",
        description="Person 0 confirms a HUD brightness preference.",
        participant_ids=("p0", "p1"),
        preference_updates=(
            V1PreferenceUpdate(
                subject_id="p0",
                attribute_path="carcontrol_HUD_set_brightness_level.level",
                new_value=7,
            ),
        ),
    )
    return V1DialogueGenerationContext(
        timeline_index=3,
        chain_id="vehicle-chain-1",
        chain_kind="vehicle",
        reasoning_type="state_shift",
        event=event,
        same_chain_prior_events=(),
        recent_global_events=(),
        previous_preference_state=(),
    )


def _materialized_turn(index: int, speaker_id: str, text: str) -> V1DialogueTurnRecord:
    return V1DialogueTurnRecord(
        turn_id=f"vehicle-e1-turn-{index + 1:03d}",
        source_event_id="vehicle-e1",
        timestamp="2025-02-01T10:00",
        speaker_id=speaker_id,
        speaker_name="Person 0" if speaker_id == "p0" else "Person 1",
        text=text,
    )


def _generated(
    index: int,
    prior: tuple[V1DialogueTurnRecord, ...],
    speaker_id: str,
    text: str,
    *,
    target: int = 2,
) -> V2GeneratedNativeTurn:
    context = _context()
    return V2GeneratedNativeTurn(
        event_id=context.event.event_id,
        source_event_sha256=canonical_json_sha256(
            context.event.model_dump(mode="json")
        ),
        timeline_index=context.timeline_index,
        event_turn_index=index,
        target_turn_count=target,
        causal_prefix_sha256=native_causal_prefix_sha256(prior),
        recent_window_start_index=0,
        payload=V2NativeTurnPayload(speaker_id=speaker_id, text=text),
        model_id="gpt-5.6-terra",
        prompt_version=V2_NATIVE_TURN_PROMPT_VERSION,
        input_sha256=f"{index + 1:064x}",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


class _FakeResponses:
    def __init__(self, payload: V2NativeTurnPayload) -> None:
        self.payload = payload
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="response-native-1",
            status="completed",
            output_parsed=self.payload,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                input_tokens_details=SimpleNamespace(cached_tokens=30),
            ),
        )


def _generated_alignment(
    turns: tuple[V1DialogueTurnRecord, ...],
    *,
    previous_memory: str,
    committed: tuple[int, ...],
    payload: V2NativeAlignmentPayload,
) -> V2GeneratedNativeAlignment:
    context = _context()
    provider_input = render_native_alignment_input(
        context.event,
        turns,
        previous_memory=previous_memory,
        committed_source_update_indexes=committed,
    )
    return V2GeneratedNativeAlignment(
        event_id=context.event.event_id,
        source_event_sha256=canonical_json_sha256(
            context.event.model_dump(mode="json")
        ),
        source_dialogue_prefix_sha256=native_causal_prefix_sha256(turns),
        before_memory_sha256=text_sha256(previous_memory),
        event_turn_index=len(turns) - 1,
        committed_source_update_indexes=committed,
        payload=payload,
        model_id="fake-alignment",
        prompt_version=V2_NATIVE_ALIGNMENT_PROMPT_VERSION,
        input_sha256=canonical_json_sha256(json.loads(provider_input)),
        usage={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    )


def test_native_input_is_causal_and_bounded() -> None:
    context = _context()
    prior = tuple(
        _materialized_turn(
            index,
            "p0" if index % 2 == 0 else "p1",
            f"Prior dialogue {index}",
        )
        for index in range(10)
    )

    payload = json.loads(
        render_native_turn_generation_input(
            _personas(),
            context,
            prior,
            target_turn_count=26,
            recent_turn_window=4,
        )
    )

    assert payload["turn_progress"]["event_turn_index"] == 10
    assert payload["turn_progress"]["recent_window_start_index"] == 6
    assert len(payload["recent_causal_event_prefix"]) == 4
    encoded = json.dumps(payload).casefold()
    assert "final quiz" not in encoded
    assert "target_state" not in encoded
    assert "approved_memory" not in encoded


def test_native_provider_returns_one_valid_turn() -> None:
    responses = _FakeResponses(
        V2NativeTurnPayload(
            speaker_id="p0",
            text="The display feels a little dim today.",
        )
    )
    model = OpenAIV2NativeTurnGenerationModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(
        _personas(),
        _context(),
        (),
        target_turn_count=26,
    )

    assert generated.event_turn_index == 0
    assert generated.usage["cached_tokens"] == 30
    assert responses.kwargs["store"] is False
    assert responses.kwargs["reasoning"] == {"effort": "medium"}


def test_native_validation_rejects_nonparticipant_and_agent_text() -> None:
    context = _context()
    with pytest.raises(ValueError, match="non-participant"):
        validate_native_turn_payload(
            context,
            (),
            V2NativeTurnPayload(speaker_id="p2", text="A normal utterance."),
        )
    with pytest.raises(ValueError, match="agent/tool"):
        validate_native_turn_payload(
            context,
            (),
            V2NativeTurnPayload(speaker_id="p0", text="Assistant: done"),
        )


def test_native_turns_materialize_and_assemble_v1_dialogue() -> None:
    context = _context()
    first = _generated(0, (), "p0", "How does the display look?", target=2)
    first_record = materialize_native_turn(_personas(), context, first, ())
    second = _generated(
        1,
        (first_record,),
        "p1",
        "I prefer the HUD brightness at level 7.",
        target=2,
    )

    dialogue = assemble_native_event_dialogue(
        _personas(),
        context,
        (first, second),
        target_turn_count=2,
    )

    assert dialogue.target_turn_count == 2
    assert [turn.speaker_id for turn in dialogue.payload.turns] == ["p0", "p1"]
    assert dialogue.usage["total_tokens"] == 30


def test_native_materialization_rejects_broken_prefix_hash() -> None:
    generated = _generated(0, (), "p0", "A normal utterance.", target=2)
    tampered = generated.model_copy(update={"causal_prefix_sha256": "f" * 64})

    with pytest.raises(ValueError, match="causal-prefix hash"):
        materialize_native_turn(_personas(), _context(), tampered, ())


def test_native_checkpoints_commit_no_op_then_deterministic_update(tmp_path) -> None:
    context = _context()
    first_generated = _generated(
        0,
        (),
        "p1",
        "How does the display look?",
        target=2,
    )
    first = build_native_turn_checkpoint(
        _personas(),
        context,
        first_generated,
        (),
        previous_entries=(),
        committed_source_update_indexes=(),
        alignments=(),
        global_turn_index=0,
        previous_checkpoint_sha256=None,
    )
    second_generated = _generated(
        1,
        (first.dialogue_turn,),
        "p0",
        "I prefer the HUD brightness at level 7.",
        target=2,
    )
    second = build_native_turn_checkpoint(
        _personas(),
        context,
        second_generated,
        (first.dialogue_turn,),
        previous_entries=first.memory_entries_after,
        committed_source_update_indexes=first.committed_source_update_indexes,
        alignments=(
            V2HybridUpdateAlignment(
                source_update_index=0,
                evidence_turn_index=1,
                evidence_quote="HUD brightness at level 7",
                reason="Person 0 explicitly confirms the HUD level.",
            ),
        ),
        global_turn_index=1,
        previous_checkpoint_sha256=first.checkpoint_sha256,
    )

    assert first.label.decision == "NO_OP"
    assert second.label.decision == "UPDATE"
    assert second.label.operations[0].op == "add"
    assert second.committed_source_update_indexes == (0,)
    assert "value=7" in second.after_memory
    validate_native_checkpoint_chain((first, second))

    path = write_native_turn_checkpoint(tmp_path, second)
    loaded = type(second).model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded == second


def test_native_checkpoint_rejects_future_or_duplicate_alignment() -> None:
    context = _context()
    generated = _generated(
        0,
        (),
        "p0",
        "I prefer the HUD brightness at level 7.",
        target=2,
    )
    future = V2HybridUpdateAlignment(
        source_update_index=0,
        evidence_turn_index=1,
        evidence_quote="HUD brightness at level 7",
        reason="The setting is explicit.",
    )
    with pytest.raises(ValueError, match="current turn"):
        build_native_turn_checkpoint(
            _personas(),
            context,
            generated,
            (),
            previous_entries=(),
            committed_source_update_indexes=(),
            alignments=(future,),
            global_turn_index=0,
            previous_checkpoint_sha256=None,
        )

    current = future.model_copy(update={"evidence_turn_index": 0})
    with pytest.raises(ValueError, match="already committed"):
        build_native_turn_checkpoint(
            _personas(),
            context,
            generated,
            (),
            previous_entries=(),
            committed_source_update_indexes=(0,),
            alignments=(current,),
            global_turn_index=0,
            previous_checkpoint_sha256=None,
        )


def test_native_checkpoint_chain_detects_broken_previous_hash() -> None:
    context = _context()
    generated = _generated(
        0,
        (),
        "p1",
        "How does the display look?",
        target=2,
    )
    checkpoint = build_native_turn_checkpoint(
        _personas(),
        context,
        generated,
        (),
        previous_entries=(),
        committed_source_update_indexes=(),
        alignments=(),
        global_turn_index=0,
        previous_checkpoint_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="hash chain"):
        validate_native_checkpoint_chain((checkpoint,))


def test_native_alignment_provider_uses_only_unresolved_causal_prefix() -> None:
    context = _context()
    turns = (
        _materialized_turn(
            0,
            "p0",
            "I prefer the HUD brightness at level 7.",
        ),
    )
    payload = V2NativeAlignmentPayload(
        decision="UPDATE",
        reason_code="NEW_VEHICLE_MEMORY",
        reason="The current turn explicitly confirms the preference.",
        alignments=(
            V2HybridUpdateAlignment(
                source_update_index=0,
                evidence_turn_index=0,
                evidence_quote="HUD brightness at level 7",
                reason="Person 0 explicitly confirms the HUD level.",
            ),
        ),
    )
    responses = _FakeResponses(payload)
    model = OpenAIV2NativeAlignmentModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(
        context.event,
        turns,
        previous_memory="",
        committed_source_update_indexes=(),
    )

    assert generated.payload.decision == "UPDATE"
    provider_input = json.loads(responses.kwargs["input"])
    assert provider_input["current_evidence_turn_index"] == 0
    assert len(provider_input["unresolved_structured_updates"]) == 1
    encoded = json.dumps(provider_input).casefold()
    assert "final quiz" not in encoded
    assert "target_state" not in encoded


def test_native_alignment_rejects_retrospective_evidence() -> None:
    context = _context()
    turns = (
        _materialized_turn(0, "p0", "HUD brightness at level 7."),
        _materialized_turn(1, "p1", "That sounds good."),
    )
    payload = V2NativeAlignmentPayload(
        decision="UPDATE",
        reason_code="NEW_VEHICLE_MEMORY",
        reason="The earlier turn contains the setting.",
        alignments=(
            V2HybridUpdateAlignment(
                source_update_index=0,
                evidence_turn_index=0,
                evidence_quote="HUD brightness at level 7",
                reason="The earlier turn states the setting.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="current causal turn"):
        validate_native_alignment_payload(
            context.event,
            turns,
            payload,
            committed_source_update_indexes=(),
        )


def test_deterministic_alignment_skips_provider_after_all_updates() -> None:
    context = _context()
    turns = (_materialized_turn(0, "p1", "Let us discuss something else."),)

    generated = deterministic_native_no_op_alignment(
        context.event,
        turns,
        previous_memory="stored",
        committed_source_update_indexes=(0,),
    )

    assert generated.payload.decision == "NO_OP"
    assert generated.usage["total_tokens"] == 0


class _SequenceTurnModel:
    def generate(
        self,
        personas,
        context,
        prior_event_turns,
        *,
        target_turn_count,
    ):
        index = len(prior_event_turns)
        if index == 0:
            return _generated(
                index,
                tuple(prior_event_turns),
                "p1",
                "How does the display look?",
                target=target_turn_count,
            )
        return _generated(
            index,
            tuple(prior_event_turns),
            "p0",
            "I prefer the HUD brightness at level 7.",
            target=target_turn_count,
        )


class _SequenceAlignmentModel:
    def generate(
        self,
        event,
        dialogue_turns,
        *,
        previous_memory,
        committed_source_update_indexes,
    ):
        turns = tuple(dialogue_turns)
        committed = tuple(committed_source_update_indexes)
        if len(turns) == 1:
            payload = V2NativeAlignmentPayload(
                decision="NO_OP",
                reason_code="NO_NEW_VEHICLE_FACT",
                reason="The question does not establish a vehicle preference.",
                alignments=(),
            )
        else:
            payload = V2NativeAlignmentPayload(
                decision="UPDATE",
                reason_code="NEW_VEHICLE_MEMORY",
                reason="The preference is explicit at the current turn.",
                alignments=(
                    V2HybridUpdateAlignment(
                        source_update_index=0,
                        evidence_turn_index=1,
                        evidence_quote="HUD brightness at level 7",
                        reason="Person 0 explicitly confirms the HUD level.",
                    ),
                ),
            )
        return _generated_alignment(
            turns,
            previous_memory=previous_memory,
            committed=committed,
            payload=payload,
        )


class _UnexpectedModel:
    def generate(self, *args, **kwargs):
        raise AssertionError("resume should not call a provider")


def test_native_event_runner_resumes_without_regeneration(tmp_path) -> None:
    kwargs = {
        "output_dir": tmp_path,
        "target_turn_count": 2,
        "previous_entries": (),
        "global_turn_offset": 0,
        "previous_checkpoint_sha256": None,
    }
    first = run_native_event(
        _personas(),
        _context(),
        _SequenceTurnModel(),
        _SequenceAlignmentModel(),
        **kwargs,
    )

    assert first.audit.turn_count == 2
    assert first.audit.no_op_count == 1
    assert first.audit.update_count == 1
    assert first.audit.committed_update_count == 1
    assert first.audit.alignment_usage["total_tokens"] == 30
    assert "value=7" in first.after_memory
    assert (tmp_path / "event.json").exists()
    assert first.checkpoint_sha256 == first.turn_checkpoints[-1].checkpoint_sha256
    assert first.turn_labels == tuple(
        checkpoint.label for checkpoint in first.turn_checkpoints
    )

    resumed = run_native_event(
        _personas(),
        _context(),
        _UnexpectedModel(),
        _UnexpectedModel(),
        **kwargs,
    )
    assert resumed == first


def test_native_scenario_runner_persists_and_resumes_full_prefix(
    tmp_path, monkeypatch
) -> None:
    context = _context()
    stage2 = SimpleNamespace(
        artifact_sha256="b" * 64,
        persona_group=SimpleNamespace(
            payload=SimpleNamespace(personas=_personas())
        ),
        interleaved_timeline=(
            SimpleNamespace(
                timeline_index=context.timeline_index,
                chain_kind=context.chain_kind,
                event=context.event,
            ),
        ),
    )
    monkeypatch.setattr(
        native_turnwise,
        "build_dialogue_generation_contexts",
        lambda _stage2: (context,),
    )
    monkeypatch.setattr(native_turnwise, "dialogue_turn_target", lambda _: 2)

    first = run_native_scenario(
        stage2,
        _SequenceTurnModel(),
        _SequenceAlignmentModel(),
        output_dir=tmp_path,
    )

    assert first.audit.completed
    assert first.audit.completed_event_count == 1
    assert first.audit.turn_count == 2
    assert first.audit.memory_hash_chain_passed
    assert first.audit.checkpoint_hash_chain_passed
    assert (tmp_path / "native.json").exists()
    assert (tmp_path / "progress.json").exists()
    assert (tmp_path / "dialogues" / "vehicle-e1.json").exists()

    resumed = run_native_scenario(
        stage2,
        _UnexpectedModel(),
        _UnexpectedModel(),
        output_dir=tmp_path,
    )
    assert resumed == first
