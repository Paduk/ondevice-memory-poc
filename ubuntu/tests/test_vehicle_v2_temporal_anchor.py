from __future__ import annotations

import pytest

from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventRecord,
    V1PreferenceUpdate,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_anchor import (
    V2TemporalAnchor,
    build_deterministic_anchor_alignment,
    temporal_setting_hint,
    validate_dialogue_contains_anchors,
)


def _event() -> V1EventRecord:
    return V1EventRecord(
        event_id="vehicle-1",
        timestamp="2026-01-01T09:00",
        description="A temporary smoke setting is confirmed.",
        participant_ids=("p1", "p2"),
        preference_updates=(
            V1PreferenceUpdate(
                subject_id="p1",
                attribute_path=(
                    "carcontrol_airConditioner_set_circulation.circulation"
                ),
                new_value="inside",
                context_arguments=(),
                condition="temporary smoke detour",
            ),
        ),
    )


def _turn(text: str) -> V1DialogueTurnRecord:
    return V1DialogueTurnRecord(
        turn_id="vehicle-1-turn-001",
        source_event_id="vehicle-1",
        timestamp="2026-01-01T09:00",
        speaker_id="p1",
        speaker_name="Person One",
        text=text,
    )


def test_temporal_setting_hint_is_human_readable_and_deduplicated() -> None:
    assert temporal_setting_hint(
        "carcontrol_airConditioner_set_circulation.circulation"
    ) == "air conditioner circulation"
    assert temporal_setting_hint(
        "carcontrol_light_set_reading_light.enabled"
    ) == "light reading"


def test_frozen_anchor_deterministically_fixes_update_turn() -> None:
    anchor = V2TemporalAnchor(
        source_event_id="vehicle-1",
        source_update_index=0,
        speaker_id="p1",
        anchor_text=(
            "For the air conditioner circulation, use inside during the "
            "temporary smoke detour."
        ),
    )
    turn = _turn(anchor.anchor_text)

    generated = build_deterministic_anchor_alignment(
        _event(),
        (turn,),
        (anchor,),
        previous_memory="- durable outside-air baseline",
    )

    alignment = generated.payload.alignments[0]
    assert generated.model_id == "deterministic:gold-anchor"
    assert alignment.evidence_turn_index == 0
    assert alignment.evidence_quote == anchor.anchor_text
    assert generated.usage["total_tokens"] == 0


def test_dialogue_rejects_a_paraphrased_frozen_anchor() -> None:
    anchor = V2TemporalAnchor(
        source_event_id="vehicle-1",
        source_update_index=0,
        speaker_id="p1",
        anchor_text="Use inside during the temporary smoke detour.",
    )

    with pytest.raises(ValueError, match="exactly once"):
        validate_dialogue_contains_anchors(
            (anchor,),
            (_turn("Use inside while smoke remains."),),
        )


def test_dialogue_rejects_reversed_multi_update_anchors() -> None:
    first = V2TemporalAnchor(
        source_event_id="vehicle-1",
        source_update_index=0,
        speaker_id="p1",
        anchor_text="First frozen update.",
    )
    second = V2TemporalAnchor(
        source_event_id="vehicle-1",
        source_update_index=1,
        speaker_id="p1",
        anchor_text="Second frozen update.",
    )

    with pytest.raises(ValueError, match="source_update_index order"):
        validate_dialogue_contains_anchors(
            (first, second),
            (_turn(second.anchor_text), _turn(first.anchor_text)),
        )
