from __future__ import annotations

from collections import Counter

import pytest

from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventChainRecord,
    V1EventRecord,
    V1NamedValue,
    V1PersonaRecord,
    V1PreferenceUpdate,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2GeneratedHybridEventAlignment,
    V2HybridEventAlignmentPayload,
    V2HybridUpdateAlignment,
    build_hybrid_end_temporary_operations,
    build_hybrid_update_operations,
    dialogue_sha256,
    event_sha256,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_hybrid_validation import (
    exact_transition_cue,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_scenario import (
    V2_TEMPORAL_EVENT_PROMPT_VERSION_V1,
    V2_TEMPORAL_EVENT_PROMPT_VERSION_V2,
    V2_TEMPORAL_EVENT_PROMPT_VERSIONS,
    V2_TEMPORAL_MAX_UPDATE_CHECKPOINTS,
    V2TemporalEventChainPayload,
    V2TemporalTransition,
    build_temporal_scenario_profiles,
    extract_dialogue_temporal_cue,
    normalize_temporal_event_payload,
    normalize_temporal_hybrid_alignment,
    temporal_scenario_profile,
    validate_temporal_event_payload,
)


def _update(
    value: int,
    *,
    condition: str | None = None,
    previous: int | None = None,
    supersedes: str | None = None,
    subject: str = "p1",
) -> V1PreferenceUpdate:
    return V1PreferenceUpdate(
        subject_id=subject,
        attribute_path="carcontrol_light_set_brightness.brightness",
        previous_value=previous,
        new_value=value,
        context_arguments=(),
        condition=condition,
        supersedes_event_id=supersedes,
    )


def _event(
    event_id: str,
    minute: int,
    description: str,
    *updates: V1PreferenceUpdate,
) -> V1EventRecord:
    return V1EventRecord(
        event_id=event_id,
        timestamp=f"2026-01-{minute // 60 + 1:02d}T{minute % 24:02d}:00",
        description=description,
        participant_ids=("p1", "p2"),
        preference_updates=updates,
    )


def _chain(index: int, events: tuple[V1EventRecord, ...]) -> V1EventChainRecord:
    return V1EventChainRecord(
        chain_id=f"veh-{index:02d}",
        kind="vehicle",
        reasoning_type="state_shift",
        delayed_query_seed="Use the latest valid setting.",
        events=events,
    )


def _valid_payload() -> V2TemporalEventChainPayload:
    transitions = []
    chains = []
    for index in range(10):
        event_id = f"v{index:02d}-base"
        events = [
            _event(
                event_id,
                index,
                "A durable setting is confirmed.",
                _update(1),
            )
        ]
        transitions.append(
            V2TemporalTransition(
                source_event_id=event_id,
                source_update_index=0,
                temporal_action="durable_upsert",
            )
        )
        if index == 0:
            for offset, value in enumerate((2, 3, 4), start=1):
                current_id = f"v00-current-{offset}"
                events.append(
                    _event(
                        current_id,
                        20 + offset,
                        "The currently observed setting changes.",
                        _update(value, condition="currently observed setting"),
                    )
                )
                transitions.append(
                    V2TemporalTransition(
                        source_event_id=current_id,
                        source_update_index=0,
                        temporal_action="current_upsert",
                    )
                )
        elif index == 1:
            temporary_id = "v01-temporary"
            events.extend(
                (
                    _event(
                        temporary_id,
                        30,
                        "Use level 2 temporarily until clean air returns.",
                        _update(2, condition="temporarily until clean air returns"),
                    ),
                    _event(
                        "v01-end",
                        31,
                        "Clean air returned; go back to normal.",
                        _update(1, previous=2, supersedes=temporary_id),
                    ),
                )
            )
            transitions.extend(
                (
                    V2TemporalTransition(
                        source_event_id=temporary_id,
                        source_update_index=0,
                        temporal_action="temporary_override",
                        temporal_cue="until clean air returns",
                        baseline_event_id=event_id,
                        baseline_source_update_index=0,
                    ),
                    V2TemporalTransition(
                        source_event_id="v01-end",
                        source_update_index=0,
                        temporal_action="end_temporary",
                        temporal_cue="back to normal",
                        baseline_event_id=event_id,
                        baseline_source_update_index=0,
                    ),
                )
            )
        elif index == 2:
            conditional_id = "v02-conditional"
            events.append(
                _event(
                    conditional_id,
                    40,
                    "A reusable rainy-driving preference is confirmed.",
                    _update(2, condition="when driving in steady rain"),
                )
            )
            transitions.append(
                V2TemporalTransition(
                    source_event_id=conditional_id,
                    source_update_index=0,
                    temporal_action="conditional_upsert",
                )
            )
        elif index == 3:
            correction_id = "v03-correction"
            events.append(
                _event(
                    correction_id,
                    50,
                    "The durable setting is corrected from 1 to 2.",
                    _update(2, previous=1, supersedes=event_id),
                )
            )
            transitions.append(
                V2TemporalTransition(
                    source_event_id=correction_id,
                    source_update_index=0,
                    temporal_action="durable_upsert",
                )
            )
        elif index in {4, 5}:
            events.append(
                _event(
                    f"v{index:02d}-negative",
                    60 + index,
                    "They discuss a temporary current setting without agreement.",
                )
            )
        chains.append(_chain(index, tuple(events)))

    backgrounds = tuple(
        V1EventChainRecord(
            chain_id=f"bg-{index:02d}",
            kind="background",
            events=(
                _event(
                    f"bg-{index:02d}-event",
                    80 + index,
                    "An unrelated background event.",
                ),
            ),
        )
        for index in range(20)
    )
    return V2TemporalEventChainPayload(
        background_chains=backgrounds,
        vehicle_chains=tuple(chains),
        temporal_transitions=tuple(transitions),
    )


def test_end_temporary_accepts_present_tense_ends_cue() -> None:
    payload = _valid_payload()
    vehicle_chains = []
    for chain in payload.vehicle_chains:
        events = tuple(
            event.model_copy(
                update={
                    "description": (
                        "When the temporary detour ends, return to the baseline."
                    )
                }
            )
            if event.event_id == "v01-end"
            else event
            for event in chain.events
        )
        vehicle_chains.append(chain.model_copy(update={"events": events}))

    updated = payload.model_copy(update={"vehicle_chains": tuple(vehicle_chains)})
    audit = validate_temporal_event_payload(
        updated,
        profile=temporal_scenario_profile(1),
    )

    assert audit["passed"] is True


def test_temporary_override_accepts_bounded_during_cue() -> None:
    payload = _valid_payload()
    vehicle_chains = []
    for chain in payload.vehicle_chains:
        events = []
        for event in chain.events:
            if event.event_id != "v01-temporary":
                events.append(event)
                continue
            update = event.preference_updates[0].model_copy(
                update={"condition": "During the cold snap"}
            )
            events.append(
                event.model_copy(
                    update={
                        "description": (
                            "During the cold snap, use level 2 as a bounded override."
                        ),
                        "preference_updates": (update,),
                    }
                )
            )
        vehicle_chains.append(chain.model_copy(update={"events": tuple(events)}))

    updated = payload.model_copy(update={"vehicle_chains": tuple(vehicle_chains)})
    audit = validate_temporal_event_payload(
        updated,
        profile=temporal_scenario_profile(1),
    )

    assert audit["passed"] is True


def test_end_temporary_extracts_plural_are_over_cue() -> None:
    text = "My circulation is outside; the next twenty minutes are over."

    assert extract_dialogue_temporal_cue("end_temporary", text) == "are over"


def test_temporal_prompt_versions_keep_v1_replay_compatible() -> None:
    assert V2_TEMPORAL_EVENT_PROMPT_VERSION_V1 in (
        V2_TEMPORAL_EVENT_PROMPT_VERSIONS
    )
    assert V2_TEMPORAL_EVENT_PROMPT_VERSION_V2 in (
        V2_TEMPORAL_EVENT_PROMPT_VERSIONS
    )


@pytest.mark.parametrize(
    ("cue", "text"),
    (
        (
            "until the library recording ends",
            "Until the library recording ends, use volume 55.",
        ),
        (
            "Marco's ankle wrap is removed",
            "Use level 1 when Marco's ankle wrap is removed.",
        ),
        ("After Crystal's wet coat is dry", "After Crystal's wet coat is dry, use 21."),
        (
            "After the pollen plume passed",
            "After the pollen plume passed, return outside.",
        ),
    ),
)
def test_exact_transition_cue_accepts_gold_end_phrases(cue: str, text: str) -> None:
    assert exact_transition_cue({"temporal_cue": cue}, text) == cue


def test_temporal_profiles_freeze_15_1_4_split_and_focus_balance() -> None:
    profiles = build_temporal_scenario_profiles()

    assert len(profiles) == 20
    assert Counter(profile.split for profile in profiles) == {
        "train": 15,
        "validation": 1,
        "test": 4,
    }
    assert set(profile.focus for profile in profiles[16:]) == {
        "current_durable",
        "temporary_restore",
        "conditional",
        "correction_noop",
    }
    assert set(Counter(profile.focus for profile in profiles).values()) == {5}


def test_temporal_payload_validates_all_shared_actions() -> None:
    audit = validate_temporal_event_payload(
        _valid_payload(),
        profile=temporal_scenario_profile(1),
    )

    assert audit["passed"] is True
    assert audit["focus_count"] == 3
    assert audit["explicit_hard_negative_event_count"] == 2
    assert audit["hard_negative_opportunity_count"] == 7
    assert audit["durable_correction_count"] == 1


def test_temporal_payload_rejects_missing_transition() -> None:
    payload = _valid_payload()
    invalid = payload.model_copy(
        update={"temporal_transitions": payload.temporal_transitions[:-1]}
    )

    with pytest.raises(ValueError, match="do not cover updates"):
        validate_temporal_event_payload(
            invalid,
            profile=temporal_scenario_profile(1),
        )


def test_temporal_payload_rejects_more_than_28_update_checkpoints() -> None:
    payload = _valid_payload()
    extra_count = V2_TEMPORAL_MAX_UPDATE_CHECKPOINTS - len(
        payload.temporal_transitions
    ) + 1
    extra_events = tuple(
        _event(
            f"v00-extra-{index:02d}",
            100 + index,
            "A further durable preference is confirmed.",
            _update(10 + index),
        )
        for index in range(extra_count)
    )
    vehicle_chains = (
        payload.vehicle_chains[0].model_copy(
            update={"events": (*payload.vehicle_chains[0].events, *extra_events)}
        ),
        *payload.vehicle_chains[1:],
    )
    transitions = (
        *payload.temporal_transitions,
        *(
            V2TemporalTransition(
                source_event_id=event.event_id,
                source_update_index=0,
                temporal_action="durable_upsert",
            )
            for event in extra_events
        ),
    )
    invalid = payload.model_copy(
        update={
            "vehicle_chains": vehicle_chains,
            "temporal_transitions": transitions,
        }
    )

    with pytest.raises(ValueError, match="too many update checkpoints"):
        validate_temporal_event_payload(
            invalid,
            profile=temporal_scenario_profile(1),
        )


def test_end_temporary_deletes_override_without_rewriting_baseline() -> None:
    baseline_event = _event(
        "base",
        1,
        "A durable setting is confirmed.",
        _update(1),
    )
    temporary_event = _event(
        "temporary",
        2,
        "Use level 2 temporarily until clean air returns.",
        _update(2, condition="temporarily until clean air returns"),
    )
    entries = []
    build_hybrid_update_operations(
        entries,
        baseline_event,
        0,
        baseline_event.preference_updates[0],
        persona_names={"p1": "Person One"},
    )
    baseline_line = entries[0].memory_line
    build_hybrid_update_operations(
        entries,
        temporary_event,
        0,
        temporary_event.preference_updates[0],
        persona_names={"p1": "Person One"},
    )

    operations, changed = build_hybrid_end_temporary_operations(
        entries,
        _update(1, previous=2, supersedes="temporary"),
    )

    assert changed is True
    assert [operation.op for operation in operations] == ["delete"]
    assert len(entries) == 1
    assert entries[0].memory_line == baseline_line


def test_temporal_alignment_moves_to_first_self_contained_turn() -> None:
    event = _event(
        "temporary",
        2,
        "Use level 15 temporarily for the duration of a call.",
        _update(15, condition="temporary for the duration of a call"),
    )
    texts = (
        "Could we keep the brightness at 15 while I am on the call?",
        "Yes, 15 is comfortable.",
        "Then 15 will be the courtesy level for this call.",
    )
    turns = tuple(
        V1DialogueTurnRecord(
            turn_id=f"temporary-turn-{index + 1:03d}",
            source_event_id="temporary",
            timestamp=event.timestamp,
            speaker_id="p1" if index else "p2",
            speaker_name="Person One" if index else "Person Two",
            text=text,
        )
        for index, text in enumerate(texts)
    )
    generated = V2GeneratedHybridEventAlignment(
        event_id=event.event_id,
        source_event_sha256=event_sha256(event),
        source_dialogue_sha256=dialogue_sha256(turns),
        before_memory_sha256=text_sha256(""),
        payload=V2HybridEventAlignmentPayload(
            alignments=(
                V2HybridUpdateAlignment(
                    source_update_index=0,
                    evidence_turn_index=1,
                    evidence_quote="Yes, 15 is comfortable.",
                    reason="The value is confirmed here.",
                ),
            )
        ),
        model_id="test",
        prompt_version="test",
        input_sha256=canonical_json_sha256({"test": True}),
        usage={},
    )

    normalized = normalize_temporal_hybrid_alignment(
        event,
        turns,
        generated,
        {0: {"temporal_action": "temporary_override"}},
    )

    alignment = normalized.payload.alignments[0]
    assert alignment.evidence_turn_index == 2
    assert alignment.evidence_quote == texts[2]
    assert normalized.normalization_notes[-1].endswith("temporal_turn=1->2")


def test_temporal_payload_repairs_one_unambiguous_participant_id() -> None:
    payload = _valid_payload()
    first_chain = payload.background_chains[0]
    first_event = first_chain.events[0].model_copy(
        update={
            "description": "Person One and Yuting discuss an unrelated topic.",
            "participant_ids": ("p1", "malformed-provider-id"),
        }
    )
    payload = payload.model_copy(
        update={
            "background_chains": (
                first_chain.model_copy(update={"events": (first_event,)}),
                *payload.background_chains[1:],
            )
        }
    )
    field = (V1NamedValue(key="k", value="v"),)
    personas = (
        V1PersonaRecord(
            persona_id="p1",
            name="Person One",
            basic_profile=field,
            cultural_interests=field,
            lifestyle_habits=field,
            vehicle_preferences=field,
        ),
        V1PersonaRecord(
            persona_id="p2",
            name="Lin Yu-ting",
            aliases=("Yuting",),
            basic_profile=field,
            cultural_interests=field,
            lifestyle_habits=field,
            vehicle_preferences=field,
        ),
    )

    normalized, notes = normalize_temporal_event_payload(
        payload,
        personas=personas,
    )

    assert normalized.background_chains[0].events[0].participant_ids == (
        "p1",
        "p2",
    )
    assert "repaired unambiguous participant_id" in notes[-1]


def test_collective_temporal_subject_accepts_participant_confirmation() -> None:
    event = _event(
        "collective-temporary",
        2,
        "Use inside circulation for this temporary smoke detour.",
        _update(
            15,
            condition="temporary smoke detour",
            subject="shared-vehicle",
        ),
    )
    turn = V1DialogueTurnRecord(
        turn_id="collective-temporary-turn-001",
        source_event_id=event.event_id,
        timestamp=event.timestamp,
        speaker_id="p2",
        speaker_name="Person Two",
        text="For this temporary smoke detour, we agree on level 15.",
    )
    generated = V2GeneratedHybridEventAlignment(
        event_id=event.event_id,
        source_event_sha256=event_sha256(event),
        source_dialogue_sha256=dialogue_sha256((turn,)),
        before_memory_sha256=text_sha256(""),
        payload=V2HybridEventAlignmentPayload(
            alignments=(
                V2HybridUpdateAlignment(
                    source_update_index=0,
                    evidence_turn_index=0,
                    evidence_quote=turn.text,
                    reason="A participant confirms the collective setting.",
                ),
            )
        ),
        model_id="test",
        prompt_version="test",
        input_sha256=canonical_json_sha256({"test": "collective"}),
        usage={},
    )

    normalized = normalize_temporal_hybrid_alignment(
        event,
        (turn,),
        generated,
        {0: {"temporal_action": "temporary_override"}},
    )

    assert normalized.payload.alignments[0].evidence_turn_index == 0
