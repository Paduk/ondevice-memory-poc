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
