from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventRecord,
    V1NamedValue,
    V1PersonaRecord,
    V1PreferenceUpdate,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    OpenAIV2HybridEventAlignmentModel,
    V2GeneratedHybridEventAlignment,
    V2HybridEventAlignmentPayload,
    V2HybridUpdateAlignment,
    build_hybrid_event_checkpoint,
    dialogue_sha256,
    event_sha256,
    normalize_hybrid_alignment,
    render_hybrid_alignment_input,
    text_sha256,
    validate_hybrid_alignment,
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


def _event(
    event_id: str,
    value: int,
    *,
    previous_value: int | None = None,
) -> V1EventRecord:
    return V1EventRecord(
        event_id=event_id,
        timestamp="2025-02-01T10:00",
        description="Person 0 states a HUD brightness preference.",
        participant_ids=("p0", "p1"),
        preference_updates=(
            V1PreferenceUpdate(
                subject_id="p0",
                attribute_path="carcontrol_HUD_set_brightness_level.level",
                previous_value=previous_value,
                new_value=value,
            ),
        ),
    )


def _turns(event: V1EventRecord, value: int) -> tuple[V1DialogueTurnRecord, ...]:
    return (
        V1DialogueTurnRecord(
            turn_id=f"{event.event_id}-turn-001",
            source_event_id=event.event_id,
            timestamp=event.timestamp,
            speaker_id="p1",
            speaker_name="Person 1",
            text="How is the display looking today?",
        ),
        V1DialogueTurnRecord(
            turn_id=f"{event.event_id}-turn-002",
            source_event_id=event.event_id,
            timestamp=event.timestamp,
            speaker_id="p0",
            speaker_name="Person 0",
            text=f"I prefer the HUD brightness at level {value}.",
        ),
    )


def _alignment(
    event: V1EventRecord,
    turns: tuple[V1DialogueTurnRecord, ...],
    *,
    previous_memory: str,
    value: int,
) -> V2GeneratedHybridEventAlignment:
    provider_input = render_hybrid_alignment_input(
        event,
        turns,
        previous_memory=previous_memory,
    )
    return V2GeneratedHybridEventAlignment(
        event_id=event.event_id,
        source_event_sha256=event_sha256(event),
        source_dialogue_sha256=dialogue_sha256(turns),
        before_memory_sha256=text_sha256(previous_memory),
        payload=V2HybridEventAlignmentPayload(
            alignments=(
                V2HybridUpdateAlignment(
                    source_update_index=0,
                    evidence_turn_index=1,
                    evidence_quote=f"HUD brightness at level {value}",
                    reason="The speaker explicitly states the HUD level.",
                ),
            )
        ),
        model_id="fake",
        prompt_version="test",
        input_sha256=text_sha256(provider_input),
        usage={},
    )


def test_hybrid_checkpoint_adds_then_replaces_at_evidence_turn() -> None:
    first_event = _event("vehicle-e1", 7)
    first_turns = _turns(first_event, 7)
    first = build_hybrid_event_checkpoint(
        event=first_event,
        timeline_index=0,
        dialogue_turns=first_turns,
        generated_alignment=_alignment(
            first_event,
            first_turns,
            previous_memory="",
            value=7,
        ),
        previous_entries=(),
        personas=_personas(),
        global_turn_offset=0,
    )

    assert [label.decision for label in first.turn_labels] == ["NO_OP", "UPDATE"]
    assert first.turn_labels[1].operations[0].op == "add"
    assert "value=7" in first.after_memory

    second_event = _event("vehicle-e2", 9, previous_value=7)
    second_turns = _turns(second_event, 9)
    second = build_hybrid_event_checkpoint(
        event=second_event,
        timeline_index=1,
        dialogue_turns=second_turns,
        generated_alignment=_alignment(
            second_event,
            second_turns,
            previous_memory=first.after_memory,
            value=9,
        ),
        previous_entries=first.memory_entries_after,
        personas=_personas(),
        global_turn_offset=2,
    )

    assert second.turn_labels[1].reason_code == "UPDATED_VEHICLE_MEMORY"
    assert second.turn_labels[1].operations[0].op == "replace"
    assert "value=9" in second.after_memory
    assert "value=7" not in second.after_memory
    assert second.before_memory_sha256 == first.after_memory_sha256


def test_alignment_rejects_missing_update_and_future_quote() -> None:
    event = _event("vehicle-e1", 7)
    turns = _turns(event, 7)
    with pytest.raises(ValueError, match="every source update"):
        validate_hybrid_alignment(
            event,
            turns,
            V2HybridEventAlignmentPayload(alignments=()),
        )


def test_alignment_moves_late_recap_to_earliest_subject_confirmation() -> None:
    event = _event("vehicle-e1", 7)
    turns = (
        V1DialogueTurnRecord(
            turn_id="vehicle-e1-turn-001",
            source_event_id=event.event_id,
            timestamp=event.timestamp,
            speaker_id="p1",
            speaker_name="Person 1",
            text="We are discussing the rear-left HUD brightness.",
        ),
        V1DialogueTurnRecord(
            turn_id="vehicle-e1-turn-002",
            source_event_id=event.event_id,
            timestamp=event.timestamp,
            speaker_id="p0",
            speaker_name="Person 0",
            text="I confirm that my HUD brightness should be level 7.",
        ),
        V1DialogueTurnRecord(
            turn_id="vehicle-e1-turn-003",
            source_event_id=event.event_id,
            timestamp=event.timestamp,
            speaker_id="p1",
            speaker_name="Person 1",
            text="That sounds clear.",
        ),
        V1DialogueTurnRecord(
            turn_id="vehicle-e1-turn-004",
            source_event_id=event.event_id,
            timestamp=event.timestamp,
            speaker_id="p1",
            speaker_name="Person 1",
            text="I have now recorded HUD level 7.",
        ),
    )
    raw = V2HybridEventAlignmentPayload(
        alignments=(
            V2HybridUpdateAlignment(
                source_update_index=0,
                evidence_turn_index=3,
                evidence_quote="recorded HUD level 7",
                reason="The later turn records the setting.",
            ),
        )
    )

    normalized, notes = normalize_hybrid_alignment(event, turns, raw)

    assert normalized.alignments[0].evidence_turn_index == 1
    assert normalized.alignments[0].evidence_quote == turns[1].text
    assert notes == ("source_update_index=0:model_turn=3->subject_turn=1",)
    with pytest.raises(ValueError, match="absent from its prefix turn"):
        validate_hybrid_alignment(
            event,
            turns,
            V2HybridEventAlignmentPayload(
                alignments=(
                    V2HybridUpdateAlignment(
                        source_update_index=0,
                        evidence_turn_index=0,
                        evidence_quote="HUD brightness at level 7",
                        reason="The preference is explicit.",
                    ),
                )
            ),
        )


class _FakeResponses:
    def __init__(self, payload: V2HybridEventAlignmentPayload) -> None:
        self.payload = payload
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="response-1",
            status="completed",
            output_parsed=self.payload,
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=40,
                total_tokens=160,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
        )


def test_openai_hybrid_adapter_uses_strict_causal_schema() -> None:
    event = _event("vehicle-e1", 7)
    turns = _turns(event, 7)
    payload = V2HybridEventAlignmentPayload(
        alignments=(
            V2HybridUpdateAlignment(
                source_update_index=0,
                evidence_turn_index=1,
                evidence_quote="HUD brightness at level 7",
                reason="The preference is explicit in this turn.",
            ),
        )
    )
    responses = _FakeResponses(payload)
    model = OpenAIV2HybridEventAlignmentModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(event, turns, previous_memory="")

    assert generated.payload == payload
    assert generated.usage["cached_tokens"] == 20
    assert generated.usage["latency_ms"] >= 0
    assert responses.request["text_format"] is V2HybridEventAlignmentPayload
    assert responses.request["store"] is False
    provider_payload = json.loads(responses.request["input"])
    assert provider_payload["structured_preference_updates"][0][
        "source_update_index"
    ] == 0
