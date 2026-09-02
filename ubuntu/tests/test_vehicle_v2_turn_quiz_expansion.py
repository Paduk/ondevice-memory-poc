from __future__ import annotations

import pytest

from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1ArgumentValue,
    V1GoldToolCallRecord,
)
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz_expansion import (
    V2CompositeQuizPlanItem,
    V2CompositeQuizPlanPayload,
    V2TurnQuizFact,
    _memory_line_supports_call,
    select_delayed_context_indexes,
    validate_composite_quiz_plan,
)


def _fact(index: int, *, level: int) -> V2TurnQuizFact:
    call = V1GoldToolCallRecord(
        name="carcontrol_HUD_set_brightness_level",
        arguments={"level": level},
    )
    return V2TurnQuizFact(
        fact_id=f"fact-event-{index}",
        source_context_index=index,
        source_event_id=f"event-{index}",
        source_chain_id=f"chain-{index}",
        reasoning_type="preference_conflict",
        memory_line=f"HUD brightness; value={level}",
        gold_call=call,
        hidden_value_argument=V1ArgumentValue(name="level", value=level),
    )


def test_composite_plan_requires_known_distinct_executable_facts() -> None:
    facts = (_fact(1, level=4), _fact(2, level=7))
    valid = V2CompositeQuizPlanPayload(
        groups=(
            V2CompositeQuizPlanItem(
                fact_ids=(facts[0].fact_id, facts[1].fact_id),
                assigned_reasoning_type="preference_conflict",
                reason="The two settings form one request.",
            ),
        )
    )

    validate_composite_quiz_plan(facts, valid, group_count=1)

    unknown = V2CompositeQuizPlanPayload(
        groups=(
            V2CompositeQuizPlanItem(
                fact_ids=(facts[0].fact_id, "fact-unknown"),
                assigned_reasoning_type="preference_conflict",
                reason="Unknown fact should fail.",
            ),
        )
    )
    with pytest.raises(ValueError, match="unknown fact"):
        validate_composite_quiz_plan(facts, unknown, group_count=1)


def test_delayed_quiz_selection_spreads_larger_update_sets() -> None:
    assert select_delayed_context_indexes(13, 13) == tuple(range(13))
    selected = select_delayed_context_indexes(15, 11)

    assert len(selected) == 11
    assert len(set(selected)) == 11
    assert selected[0] == 0
    assert selected[-1] == 14


def test_memory_line_call_matching_ignores_numeric_dates() -> None:
    call = V1GoldToolCallRecord(
        name="carcontrol_HUD_set_brightness_level",
        arguments={"level": 2},
    )
    wrong = (
        "- [2025-01-21 06:50] Ethan Park: "
        "carcontrol_HUD_set_brightness_level.level; value=3"
    )
    correct = (
        "- [2025-03-25 19:10] Ethan Park: "
        "carcontrol_HUD_set_brightness_level.level; value=2; "
        "condition=when driving at night"
    )

    assert not _memory_line_supports_call(wrong, call)
    assert _memory_line_supports_call(correct, call)


def test_memory_line_call_matching_requires_exact_selector_context() -> None:
    call = V1GoldToolCallRecord(
        name="carcontrol_light_set_reading_light_brightness",
        arguments={"light": "rear_left", "brightness": 5},
    )
    correct = (
        "- [2025-05-22 19:05] Dr. Elias Mercer: "
        "carcontrol_light_set_reading_light_brightness.brightness; value=5; "
        'context=(light="rear_left")'
    )
    wrong_selector = correct.replace("rear_left", "rear_right")

    assert _memory_line_supports_call(correct, call)
    assert not _memory_line_supports_call(wrong_selector, call)
