from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1ArgumentValue,
    V1DialogueTurnRecord,
    V1EventChainRecord,
    V1EventRecord,
    V1GoldToolCallRecord,
    V1PreferenceUpdate,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import text_sha256
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz import (
    OpenAIV2TurnQuizGenerationModel,
    V2TurnQuizGenerationContext,
    V2TurnQuizQueryPayload,
    derive_turn_quiz_gold_calls,
    finalize_turn_quiz,
    validate_turn_quiz_query,
)


def _event() -> V1EventRecord:
    return V1EventRecord(
        event_id="vehicle-e1",
        timestamp="2025-02-01T10:00",
        description="Person 0 confirms a HUD brightness preference.",
        participant_ids=("p0", "p1"),
        preference_updates=(
            V1PreferenceUpdate(
                subject_id="p0",
                attribute_path="carcontrol_HUD_set_brightness_level.level",
                previous_value=5,
                new_value=7,
            ),
        ),
    )


def _chain() -> V1EventChainRecord:
    return V1EventChainRecord(
        chain_id="vehicle-chain-1",
        kind="vehicle",
        reasoning_type="error_correction",
        delayed_query_seed="Apply the corrected HUD preference.",
        events=(_event(),),
    )


def _context() -> V2TurnQuizGenerationContext:
    turn = V1DialogueTurnRecord(
        turn_id="vehicle-e1-turn-001",
        source_event_id="vehicle-e1",
        timestamp="2025-02-01T10:00",
        speaker_id="p0",
        speaker_name="Person 0",
        text="Please remember that my HUD brightness should be level 7.",
    )
    memory = (
        "- [2025-02-01 10:00] Person 0: "
        "carcontrol_HUD_set_brightness_level.level; value=7"
    )
    calls = derive_turn_quiz_gold_calls(_event().preference_updates, (0,))
    return V2TurnQuizGenerationContext(
        quiz_id="turn-quiz-vehicle-e1-001",
        source_checkpoint_sha256="a" * 64,
        source_event_id="vehicle-e1",
        source_chain_id="vehicle-chain-1",
        reasoning_type="error_correction",
        global_turn_index=0,
        event_turn_index=0,
        turn_id=turn.turn_id,
        current_memory=memory,
        current_memory_sha256=text_sha256(memory),
        causal_dialogue_prefix=(turn,),
        causal_prefix_sha256=canonical_json_sha256(
            [turn.model_dump(mode="json")]
        ),
        gold_calls=calls,
        gold_calls_sha256=canonical_json_sha256(
            [call.model_dump(mode="json") for call in calls]
        ),
        hidden_value_arguments=(V1ArgumentValue(name="level", value=7),),
    )


def _payload() -> V2TurnQuizQueryPayload:
    return V2TurnQuizQueryPayload(
        query="Please restore my corrected HUD brightness preference.",
        memory_evidence_lines=(_context().current_memory,),
        reason="The current memory contains the corrected HUD level.",
    )


def _tool_schemas() -> tuple[dict, ...]:
    return (
        {
            "name": "carcontrol_HUD_set_brightness_level",
            "description": "Set HUD brightness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["level"],
                "additionalProperties": False,
            },
        },
    )


class _FakeRuntime:
    def create_world(self):
        return {"level": 5}

    def state(self, world):
        return deepcopy(world)

    def execute(self, world, name, arguments):
        if name != "carcontrol_HUD_set_brightness_level":
            return {"success": False}
        world["level"] = arguments["level"]
        return {"success": True}


class _FakeResponses:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="response-1",
            status="completed",
            output_parsed=self.payload,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )


def test_gold_call_is_derived_from_structured_update() -> None:
    calls = derive_turn_quiz_gold_calls(_event().preference_updates, (0,))

    assert calls == (
        V1GoldToolCallRecord(
            name="carcontrol_HUD_set_brightness_level",
            arguments={"level": 7},
        ),
    )


def test_query_requires_exact_support_and_hides_tool_name() -> None:
    context = _context()
    validate_turn_quiz_query(context, _payload())

    with pytest.raises(ValueError, match="exact current-memory line"):
        validate_turn_quiz_query(
            context,
            _payload().model_copy(update={"memory_evidence_lines": ("missing",)}),
        )
    with pytest.raises(ValueError, match="leaks an executable Tool name"):
        validate_turn_quiz_query(
            context,
            _payload().model_copy(
                update={
                    "query": "Call carcontrol_HUD_set_brightness_level now."
                }
            ),
        )
    with pytest.raises(ValueError, match="reveals hidden answer value"):
        validate_turn_quiz_query(
            context,
            _payload().model_copy(
                update={"query": "Please set my HUD brightness to level 7."}
            ),
        )


def test_turn_quiz_provider_and_simulator_finalize() -> None:
    context = _context()
    responses = _FakeResponses(_payload())
    model = OpenAIV2TurnQuizGenerationModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(
        context,
        personas=(),
        source_event_description=_event().description,
    )
    quiz = finalize_turn_quiz(
        context,
        generated,
        chain=_chain(),
        quiz_index=1,
        tool_schemas=_tool_schemas(),
        runtime=_FakeRuntime(),
    )

    assert quiz.query == _payload().query
    assert quiz.gold_calls[0].arguments == {"level": 7}
    assert quiz.state_changed is True
    assert quiz.target_state == {"level": 7}
    assert generated.usage["cached_tokens"] == 10
    assert responses.request["text_format"] is V2TurnQuizQueryPayload
    assert responses.request["store"] is False


def test_turn_quiz_retry_rephrases_incidental_numeric_cardinality() -> None:
    responses = _FakeResponses(_payload())
    model = OpenAIV2TurnQuizGenerationModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    model.generate(
        _context(),
        personas=(),
        source_event_description=_event().description,
        retry_feedback="Turn Quiz query reveals hidden answer value: level",
    )

    instructions = responses.request["instructions"]
    assert "same number appears incidentally" in instructions
    assert "group return" in instructions


def test_turn_quiz_provider_neutralizes_any_boolean_value_leak() -> None:
    context = _context().model_copy(
        update={
            "hidden_value_arguments": (
                V1ArgumentValue(name="is_open", value=True),
            )
        }
    )
    leaking = _payload().model_copy(
        update={"query": "Please turn on the remembered setting now."}
    )
    responses = _FakeResponses(leaking)
    model = OpenAIV2TurnQuizGenerationModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(
        context,
        personas=(),
        source_event_description=_event().description,
    )

    assert generated.payload.query == (
        "Please apply the relevant remembered vehicle setting now."
    )
    validate_turn_quiz_query(context, generated.payload)


def test_turn_quiz_provider_neutralizes_non_boolean_value_leak() -> None:
    context = _context()
    leaking = _payload().model_copy(
        update={"query": "Please set my HUD brightness to level 7."}
    )
    responses = _FakeResponses(leaking)
    model = OpenAIV2TurnQuizGenerationModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(
        context,
        personas=(),
        source_event_description=_event().description,
    )

    assert generated.payload.query == (
        "Please apply the relevant remembered vehicle setting now."
    )
    validate_turn_quiz_query(context, generated.payload)
