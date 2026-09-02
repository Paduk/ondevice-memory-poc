from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_EVENT_CHAIN_PROMPT_VERSION,
    V1_PERSONA_PROMPT_VERSION,
    V1EventChainPayload,
    V1GeneratedEventChains,
    V1GeneratedPersonaGroup,
    V1PersonaGroupPayload,
    V1PersonaSeed,
    V1Stage2Artifact,
    V1Stage2Audit,
    interleave_event_chains,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1ArgumentValue,
    V1EventChainRecord,
    V1EventRecord,
    V1NamedValue,
    V1PersonaRecord,
    V1PreferenceUpdate,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    V1_DIALOGUE_PROMPT_VERSION,
    V1_FINAL_QUIZ_PROMPT_VERSION,
    OpenAIV1DialogueGenerationModel,
    V1DialogueEventPayload,
    V1DialogueLine,
    V1FinalQuizPayload,
    V1GeneratedEventDialogue,
    V1GeneratedFinalQuiz,
    V1ToolCallDraft,
    build_dialogue_generation_contexts,
    build_stage3_artifact,
    dialogue_turn_target,
    finalize_generated_quiz,
    normalize_dialogue_speaker_ids,
    serialize_v1_history,
    serialize_v1_qa,
    validate_dialogue_payload,
    validate_public_v1_payload,
    write_stage3_artifact,
)


def test_dialogue_speaker_names_are_normalized_to_allowed_ids() -> None:
    stage2 = _stage2()
    context = build_dialogue_generation_contexts(stage2)[0]
    payload = V1DialogueEventPayload(
        turns=(
            V1DialogueLine(speaker_id="Person 0", text="First turn."),
            V1DialogueLine(speaker_id="Person 1", text="Second turn."),
        )
    )

    normalized = normalize_dialogue_speaker_ids(
        stage2.persona_group.payload.personas,
        context,
        payload,
    )

    assert tuple(turn.speaker_id for turn in normalized.turns) == ("p0", "p1")


def _personas() -> tuple[V1PersonaRecord, ...]:
    return tuple(
        V1PersonaRecord(
            persona_id=f"p{index}",
            name=f"Person {index}",
            basic_profile=(
                V1NamedValue(key="age", value=str(30 + index)),
                V1NamedValue(key="education", value="graduate"),
                V1NamedValue(key="occupation", value="researcher"),
                V1NamedValue(key="mbti", value="INTJ"),
            ),
            cultural_interests=(V1NamedValue(key="music", value="jazz"),),
            lifestyle_habits=(V1NamedValue(key="travel", value="weekly"),),
            vehicle_preferences=(V1NamedValue(key="HUD", value="contextual"),),
        )
        for index in range(3)
    )


def _tool_schemas() -> tuple[dict, ...]:
    return (
        {
            "name": "carcontrol_HUD_set_brightness_level",
            "description": "Set HUD brightness level (1-10).",
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


def _stage2() -> V1Stage2Artifact:
    personas = _personas()
    seeds = tuple(
        V1PersonaSeed(seed_id=f"p{index}", source="test", profile={"seed": index})
        for index in range(3)
    )
    persona_group = V1GeneratedPersonaGroup(
        candidate_group_id="group-001",
        seed_persona_ids=("p0", "p1", "p2"),
        seed_personas=seeds,
        payload=V1PersonaGroupPayload(personas=personas),
        model_id="fake",
        prompt_version=V1_PERSONA_PROMPT_VERSION,
        input_sha256="a" * 64,
        usage={},
    )
    background = []
    for chain_index in range(20):
        background.append(
            V1EventChainRecord(
                chain_id=f"bg-{chain_index:02d}",
                kind="background",
                events=tuple(
                    V1EventRecord(
                        event_id=f"bg-{chain_index:02d}-e{event_index}",
                        timestamp=(
                            f"2025-01-{chain_index + 1:02d}T"
                            f"{8 + event_index:02d}:00"
                        ),
                        description="A realistic non-vehicle life event.",
                        participant_ids=("p0", "p1"),
                    )
                    for event_index in range(2)
                ),
            )
        )
    reasoning_types = (
        "preference_conflict",
        "conditional_constraint",
        "coreference_resolution",
        "state_shift",
        "error_correction",
    )
    vehicle = []
    for chain_index in range(10):
        vehicle.append(
            V1EventChainRecord(
                chain_id=f"veh-{chain_index:02d}",
                kind="vehicle",
                reasoning_type=reasoning_types[chain_index % 5],
                delayed_query_seed="Apply the established HUD preference.",
                events=tuple(
                    V1EventRecord(
                        event_id=f"veh-{chain_index:02d}-e{event_index}",
                        timestamp=(
                            f"2025-02-{chain_index + 1:02d}T"
                            f"{8 + event_index:02d}:00"
                        ),
                        description="Person 0 reveals and confirms a HUD preference.",
                        participant_ids=("p0", "p2"),
                        preference_updates=(
                            (
                                V1PreferenceUpdate(
                                    subject_id="p0",
                                    attribute_path=(
                                        "carcontrol_HUD_set_brightness_level.level"
                                    ),
                                    new_value=(chain_index % 9) + 1,
                                ),
                            )
                            if event_index == 1
                            else ()
                        ),
                    )
                    for event_index in range(4)
                ),
            )
        )
    event_payload = V1EventChainPayload(
        background_chains=tuple(background),
        vehicle_chains=tuple(vehicle),
    )
    generated_events = V1GeneratedEventChains(
        scenario_candidate_id="scenario-candidate-001",
        planned_reasoning_types=tuple(chain.reasoning_type for chain in vehicle),
        payload=event_payload,
        model_id="fake",
        prompt_version=V1_EVENT_CHAIN_PROMPT_VERSION,
        input_sha256="b" * 64,
        vehicle_catalog_sha256="c" * 64,
        usage={},
    )
    audit = V1Stage2Audit(
        persona_count=3,
        background_chain_count=20,
        vehicle_chain_count=10,
        event_count=80,
        vehicle_event_count=40,
        background_event_count=40,
        timestamp_span_days=40,
        reasoning_type_counts={name: 2 for name in reasoning_types},
        attribute_update_count=10,
        passed=True,
    )
    body = {
        "schema_version": "vehiclemembench-v1-stage2-schema-v1",
        "scenario_candidate_id": "scenario-candidate-001",
        "persona_group": persona_group.model_dump(mode="json"),
        "event_chains": generated_events.model_dump(mode="json"),
        "interleaved_timeline": [
            item.model_dump(mode="json")
            for item in interleave_event_chains(event_payload.all_chains)
        ],
        "audit": audit.model_dump(mode="json"),
    }
    return V1Stage2Artifact(
        **body,
        artifact_sha256=canonical_json_sha256(body),
    )


class _FakeRuntime:
    def create_world(self):
        return {"level": 5}

    def state(self, world):
        return deepcopy(world)

    def execute(self, world, name, arguments):
        if name != "carcontrol_HUD_set_brightness_level":
            return {"success": False, "message": "unknown"}
        world["level"] = arguments["level"]
        return {"success": True}


class _FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
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


def test_contexts_are_causal_and_preserve_prior_preference_state() -> None:
    contexts = build_dialogue_generation_contexts(_stage2())

    assert len(contexts) == 80
    assert contexts[0].previous_preference_state == ()
    later = next(
        context
        for context in contexts
        if context.previous_preference_state
    )
    assert all(
        entry.source_timestamp <= later.event.timestamp
        for entry in later.previous_preference_state
    )


def test_dialogue_adapter_enforces_exact_human_turns() -> None:
    context = build_dialogue_generation_contexts(_stage2())[0]
    requested = dialogue_turn_target(context)
    payload = V1DialogueEventPayload(
        turns=tuple(
            V1DialogueLine(
                speaker_id=context.event.participant_ids[index % 2],
                text=f"Natural conversation line {index}.",
            )
            for index in range(requested)
        )
    )
    responses = _FakeResponses(payload)
    model = OpenAIV1DialogueGenerationModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(
        _personas(),
        context,
        requested_turn_count=requested,
    )

    assert generated.payload == payload
    assert responses.kwargs["store"] is False
    assert responses.kwargs["reasoning"] == {"effort": "medium"}
    with pytest.raises(ValueError, match="agent/tool"):
        validate_dialogue_payload(
            context,
            V1DialogueEventPayload(
                turns=tuple(
                    V1DialogueLine(
                        speaker_id=context.event.participant_ids[index % 2],
                        text="Assistant: done" if index == 0 else "Natural reply.",
                    )
                    for index in range(requested)
                )
            ),
            requested_turn_count=requested,
        )


def test_full_stage3_artifact_matches_public_v1_contract(tmp_path: Path) -> None:
    stage2 = _stage2()
    contexts = build_dialogue_generation_contexts(stage2)
    dialogues = tuple(
        V1GeneratedEventDialogue(
            event_id=context.event.event_id,
            source_event_sha256=canonical_json_sha256(
                context.event.model_dump(mode="json")
            ),
            target_turn_count=dialogue_turn_target(context),
            payload=V1DialogueEventPayload(
                turns=tuple(
                    V1DialogueLine(
                        speaker_id=context.event.participant_ids[index % 2],
                        text=f"Conversation {context.event.event_id} line {index}.",
                    )
                    for index in range(dialogue_turn_target(context))
                )
            ),
            model_id="fake",
            prompt_version=V1_DIALOGUE_PROMPT_VERSION,
            input_sha256="d" * 64,
            usage={},
        )
        for context in contexts
    )
    quizzes = tuple(
        V1GeneratedFinalQuiz(
            source_chain_id=chain.chain_id,
            source_chain_sha256=canonical_json_sha256(
                chain.model_dump(mode="json")
            ),
            payload=V1FinalQuizPayload(
                source_chain_id=chain.chain_id,
                reasoning_type=chain.reasoning_type,
                gold_memory="[2025-02-01] Person 0 established HUD level 8.",
                query="Please restore my established HUD brightness.",
                gold_calls=(
                    V1ToolCallDraft(
                        name="carcontrol_HUD_set_brightness_level",
                        arguments=(V1ArgumentValue(name="level", value=8),),
                    ),
                ),
            ),
            model_id="fake",
            prompt_version=V1_FINAL_QUIZ_PROMPT_VERSION,
            input_sha256="e" * 64,
            usage={},
        )
        for chain in stage2.event_chains.payload.vehicle_chains
    )

    artifact = build_stage3_artifact(
        stage2,
        dialogues,
        quizzes,
        scenario_id="scenario-reproduction-001",
        tool_schemas=_tool_schemas(),
        runtime=_FakeRuntime(),
    )
    paths = write_stage3_artifact(
        tmp_path,
        artifact,
        public_scenario_index=1,
    )

    history = serialize_v1_history(artifact.scenario.dialogue_turns)
    qa = serialize_v1_qa(artifact.scenario.final_quizzes)
    validate_public_v1_payload(history, qa, _tool_schemas())
    assert artifact.audit.dialogue_turn_count == 2640
    assert artifact.audit.quiz_count == 10
    assert paths["history"].read_text(encoding="utf-8") == history
    assert json.loads(paths["qa"].read_text(encoding="utf-8")) == qa


def test_simulator_rejects_no_effect_quiz() -> None:
    stage2 = _stage2()
    chain = stage2.event_chains.payload.vehicle_chains[0]
    generated = V1GeneratedFinalQuiz(
        source_chain_id=chain.chain_id,
        source_chain_sha256=canonical_json_sha256(chain.model_dump(mode="json")),
        payload=V1FinalQuizPayload(
            source_chain_id=chain.chain_id,
            reasoning_type=chain.reasoning_type,
            gold_memory="A sufficient memory.",
            query="Apply it.",
            gold_calls=(
                V1ToolCallDraft(
                    name="carcontrol_HUD_set_brightness_level",
                    arguments=(V1ArgumentValue(name="level", value=5),),
                ),
            ),
        ),
        model_id="fake",
        prompt_version=V1_FINAL_QUIZ_PROMPT_VERSION,
        input_sha256="f" * 64,
        usage={},
    )

    with pytest.raises(ValueError, match="does not change simulator state"):
        finalize_generated_quiz(
            generated,
            chain,
            quiz_index=1,
            tool_schemas=_tool_schemas(),
            runtime=_FakeRuntime(),
        )
