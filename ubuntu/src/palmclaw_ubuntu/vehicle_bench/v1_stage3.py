"""Paper-first dialogue and final-QA generation for VehicleMemBench V1."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from palmclaw_ubuntu.vehicle_bench.dataset import REASONING_TYPES, GoldToolCall
from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1ArgumentValue,
    V1DialogueTurnRecord,
    V1EventChainRecord,
    V1EventRecord,
    V1FinalQuizRecord,
    V1GoldToolCallRecord,
    V1PersonaRecord,
    V1ScalarValue,
    V1ScenarioArtifact,
    canonical_json_sha256,
)

V1_STAGE3_SCHEMA_VERSION = "vehiclemembench-v1-stage3-schema-v1"
V1_DIALOGUE_PROMPT_VERSION = "vehiclemembench-v1-dialogue-paper-figure6-rebuilt-v2"
V1_FINAL_QUIZ_PROMPT_VERSION = "vehiclemembench-v1-final-qa-paper-stage5-rebuilt-v1"
V1_BACKGROUND_DIALOGUE_TURNS = 40
V1_VEHICLE_DIALOGUE_TURNS = 26

V1_DIALOGUE_INSTRUCTIONS = """
Convert one structured event into a natural multi-turn conversation among the
supplied vehicle occupants.

Paper-specified rules:
1. Generate ONLY human-to-human dialogue. Never include a vehicle agent, AI,
   assistant, system, tool call, or tool response as a speaker.
2. For a vehicle-related event, reveal the preference as a natural aside,
   complaint, recollection, or request to another occupant. Do not phrase it as
   an explicit command to the vehicle.
3. Speaker IDs must come from the event participant IDs and the text must
   faithfully express the event without changing its person, value, condition,
   reference, correction, or temporal meaning.
4. Preserve chronology and do not reveal later events or the delayed query.
5. An unrelated/background event must contain at least 40 dialogue turns.

Reconstructed output rules:
- Return exactly requested_turn_count turns in the requested structured form.
- Use speaker IDs, not names, in speaker_id. Do not add timestamps; they are
  deterministically inherited from the source event.
- Write substantive, naturally alternating utterances that develop the topic.
  Avoid repetitive paraphrases, sentence fragments, or artificial log-reading.
- Do not invent an executable setting or numerical value absent from the event.
""".strip()

V1_FINAL_QUIZ_INSTRUCTIONS = """
Generate one delayed executable VehicleMemBench query and its reference action
from one completed vehicle-preference event chain.

Paper-specified rules:
- The query must require multi-hop recovery of the chain's user preference and
  be a natural continuation after the historical dialogue.
- The answer must be one or more executable calls from the supplied official
  vehicle Tool schemas and must realize the intended outcome.
- Preserve the assigned reasoning type and source chain.

Reconstructed output rules:
- Follow delayed_query_seed but express it as a natural occupant request.
- gold_memory must contain only the dated source-chain facts needed to resolve
  the query; do not add facts unavailable in the chain/dialogue.
- Use exact Tool and argument names and exact simulator values from the schemas
  and descriptions. Do not use aliases such as cabin, fresh_air, or
  front_passenger when the executable value is all/outside/passenger.
- Do not emit target state; it is computed by deterministic simulator execution.
""".strip()

_HISTORY_LINE_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] [^:]+: .+$"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V1DialogueLine(_StrictModel):
    speaker_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class V1DialogueEventPayload(_StrictModel):
    turns: tuple[V1DialogueLine, ...] = Field(min_length=1)


class V1GeneratedEventDialogue(_StrictModel):
    event_id: str = Field(min_length=1)
    source_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_turn_count: int = Field(gt=0)
    payload: V1DialogueEventPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V1ToolCallDraft(_StrictModel):
    name: str = Field(pattern=r"^carcontrol_[A-Za-z0-9_]+$")
    arguments: tuple[V1ArgumentValue, ...]

    @model_validator(mode="after")
    def validate_unique_arguments(self) -> V1ToolCallDraft:
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ValueError("Tool-call argument names must be unique")
        return self

    def argument_map(self) -> dict[str, V1ScalarValue]:
        return {
            argument.name: argument.value for argument in self.arguments
        }


class V1FinalQuizPayload(_StrictModel):
    source_chain_id: str = Field(min_length=1)
    reasoning_type: str
    gold_memory: str = Field(min_length=1)
    query: str = Field(min_length=1)
    gold_calls: tuple[V1ToolCallDraft, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reasoning_type(self) -> V1FinalQuizPayload:
        if self.reasoning_type not in REASONING_TYPES:
            raise ValueError("quiz requires an official reasoning type")
        return self


class V1GeneratedFinalQuiz(_StrictModel):
    source_chain_id: str = Field(min_length=1)
    source_chain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: V1FinalQuizPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V1PreferenceStateEntry(_StrictModel):
    subject_id: str
    attribute_path: str
    condition: str | None = None
    context_arguments: tuple[V1ArgumentValue, ...] = ()
    value: V1ScalarValue
    source_event_id: str
    source_timestamp: str


class V1DialogueGenerationContext(_StrictModel):
    timeline_index: int = Field(ge=0)
    chain_id: str
    chain_kind: str
    reasoning_type: str | None = None
    event: V1EventRecord
    same_chain_prior_events: tuple[V1EventRecord, ...]
    recent_global_events: tuple[V1EventRecord, ...]
    previous_preference_state: tuple[V1PreferenceStateEntry, ...]


class V1Stage3Audit(_StrictModel):
    dialogue_event_count: int
    dialogue_turn_count: int
    background_dialogue_turn_count: int
    vehicle_dialogue_turn_count: int
    quiz_count: int
    simulator_executed_call_count: int
    simulator_passed_quiz_count: int
    history_line_contract_passed: bool
    qa_contract_passed: bool
    passed: bool


class V1Stage3Artifact(_StrictModel):
    schema_version: str = V1_STAGE3_SCHEMA_VERSION
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario: V1ScenarioArtifact
    generated_dialogues: tuple[V1GeneratedEventDialogue, ...]
    generated_quizzes: tuple[V1GeneratedFinalQuiz, ...]
    audit: V1Stage3Audit
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact_hash(self) -> V1Stage3Artifact:
        payload = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(payload) != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match the Stage 3 artifact")
        return self


class V1DialogueGenerationModel(Protocol):
    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        context: V1DialogueGenerationContext,
        *,
        requested_turn_count: int,
        frozen_anchors: Sequence[Mapping[str, Any]] = (),
    ) -> V1GeneratedEventDialogue: ...


class V1FinalQuizGenerationModel(Protocol):
    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        chain: V1EventChainRecord,
        dialogue_turns: Sequence[V1DialogueTurnRecord],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> V1GeneratedFinalQuiz: ...


class OpenAIV1DialogueGenerationModel:
    """Generate one resumable event-dialogue checkpoint with Terra."""

    def __init__(
        self,
        model_id: str = V1_DEFAULT_GENERATION_MODEL,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 8_192,
        additional_instructions: str | None = None,
        prompt_version: str = V1_DIALOGUE_PROMPT_VERSION,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("dialogue generation model ID is required")
        if not prompt_version.strip():
            raise ValueError("dialogue prompt version is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.additional_instructions = (additional_instructions or "").strip()
        self.prompt_version = prompt_version
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        context: V1DialogueGenerationContext,
        *,
        requested_turn_count: int,
        frozen_anchors: Sequence[Mapping[str, Any]] = (),
    ) -> V1GeneratedEventDialogue:
        provider_payload = json.loads(
            render_dialogue_generation_input(
                personas,
                context,
                requested_turn_count=requested_turn_count,
            )
        )
        if frozen_anchors:
            provider_payload["required_exact_anchor_turns"] = [
                dict(item) for item in frozen_anchors
            ]
        provider_input = _canonical_json(provider_payload)
        instructions = V1_DIALOGUE_INSTRUCTIONS
        if self.additional_instructions:
            instructions += "\n\n" + self.additional_instructions
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": (
                instructions
                + "\n\nFor this event, speaker_id must be copied exactly from "
                "this allowlist and no other value is valid: "
                + json.dumps(list(context.event.participant_ids))
            ),
            "input": provider_input,
            "text_format": V1DialogueEventPayload,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        started_at = time.monotonic()
        response = self._client.responses.parse(**request)
        latency_ms = int((time.monotonic() - started_at) * 1_000)
        _raise_for_bad_response(response, stage="dialogue generation")
        payload = V1DialogueEventPayload.model_validate(response.output_parsed)
        payload = normalize_dialogue_speaker_ids(personas, context, payload)
        validate_dialogue_payload(
            context,
            payload,
            requested_turn_count=requested_turn_count,
        )
        return V1GeneratedEventDialogue(
            event_id=context.event.event_id,
            source_event_sha256=canonical_json_sha256(
                context.event.model_dump(mode="json")
            ),
            target_turn_count=requested_turn_count,
            payload=payload,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
        )


def normalize_dialogue_speaker_ids(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    payload: V1DialogueEventPayload,
) -> V1DialogueEventPayload:
    """Map unambiguous participant names/aliases back to canonical IDs."""

    allowed = set(context.event.participant_ids)
    aliases: dict[str, set[str]] = {}
    for persona in personas:
        if persona.persona_id not in allowed:
            continue
        values = (persona.persona_id, persona.name, *persona.aliases)
        for value in values:
            key = value.strip().casefold()
            if key:
                aliases.setdefault(key, set()).add(persona.persona_id)

    turns = []
    for turn in payload.turns:
        if turn.speaker_id in allowed:
            turns.append(turn)
            continue
        matches = aliases.get(turn.speaker_id.strip().casefold(), set())
        if len(matches) == 1:
            turn = turn.model_copy(update={"speaker_id": next(iter(matches))})
        turns.append(turn)
    return payload.model_copy(update={"turns": tuple(turns)})


class OpenAIV1FinalQuizGenerationModel:
    """Generate a delayed executable query and calls for one vehicle chain."""

    def __init__(
        self,
        model_id: str = V1_DEFAULT_GENERATION_MODEL,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 8_192,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("quiz generation model ID is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        chain: V1EventChainRecord,
        dialogue_turns: Sequence[V1DialogueTurnRecord],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> V1GeneratedFinalQuiz:
        provider_input = render_final_quiz_generation_input(
            personas,
            chain,
            dialogue_turns,
            tool_schemas,
        )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V1_FINAL_QUIZ_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V1FinalQuizPayload,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        started_at = time.monotonic()
        response = self._client.responses.parse(**request)
        latency_ms = int((time.monotonic() - started_at) * 1_000)
        _raise_for_bad_response(response, stage="final Quiz generation")
        payload = V1FinalQuizPayload.model_validate(response.output_parsed)
        if payload.source_chain_id != chain.chain_id:
            raise ValueError("generated Quiz changed source_chain_id")
        if payload.reasoning_type != chain.reasoning_type:
            raise ValueError("generated Quiz changed reasoning_type")
        return V1GeneratedFinalQuiz(
            source_chain_id=chain.chain_id,
            source_chain_sha256=canonical_json_sha256(
                chain.model_dump(mode="json")
            ),
            payload=payload,
            model_id=self.model_id,
            prompt_version=V1_FINAL_QUIZ_PROMPT_VERSION,
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
        )


def build_dialogue_generation_contexts(
    stage2: V1Stage2Artifact,
    *,
    recent_global_event_count: int = 8,
) -> tuple[V1DialogueGenerationContext, ...]:
    if recent_global_event_count < 0:
        raise ValueError("recent_global_event_count cannot be negative")
    prior_by_chain: dict[str, list[V1EventRecord]] = {}
    global_prior: list[V1EventRecord] = []
    state: dict[tuple[str, str, str | None, str], V1PreferenceStateEntry] = {}
    contexts: list[V1DialogueGenerationContext] = []
    for item in stage2.interleaved_timeline:
        contexts.append(
            V1DialogueGenerationContext(
                timeline_index=item.timeline_index,
                chain_id=item.chain_id,
                chain_kind=item.chain_kind,
                reasoning_type=item.reasoning_type,
                event=item.event,
                same_chain_prior_events=tuple(prior_by_chain.get(item.chain_id, ())),
                recent_global_events=tuple(global_prior[-recent_global_event_count:])
                if recent_global_event_count
                else (),
                previous_preference_state=tuple(
                    sorted(
                        state.values(),
                        key=lambda entry: (
                            entry.subject_id,
                            entry.attribute_path,
                            entry.condition or "",
                        ),
                    )
                ),
            )
        )
        for update in item.event.preference_updates:
            arguments = tuple(update.context_arguments)
            argument_key = _canonical_json(
                [argument.model_dump(mode="json") for argument in arguments]
            )
            key = (
                update.subject_id,
                update.attribute_path,
                update.condition,
                argument_key,
            )
            state[key] = V1PreferenceStateEntry(
                subject_id=update.subject_id,
                attribute_path=update.attribute_path,
                condition=update.condition,
                context_arguments=arguments,
                value=update.new_value,
                source_event_id=item.event.event_id,
                source_timestamp=item.event.timestamp,
            )
        prior_by_chain.setdefault(item.chain_id, []).append(item.event)
        global_prior.append(item.event)
    return tuple(contexts)


def dialogue_turn_target(context: V1DialogueGenerationContext) -> int:
    return (
        V1_BACKGROUND_DIALOGUE_TURNS
        if context.chain_kind == "background"
        else V1_VEHICLE_DIALOGUE_TURNS
    )


def render_dialogue_generation_input(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    *,
    requested_turn_count: int,
) -> str:
    persona_by_id = {persona.persona_id: persona for persona in personas}
    if set(context.event.participant_ids) - set(persona_by_id):
        raise ValueError("dialogue event references an unknown persona")
    payload = {
        "requested_turn_count": requested_turn_count,
        "event_kind": context.chain_kind,
        "assigned_reasoning_type": context.reasoning_type,
        "current_event": context.event.model_dump(mode="json"),
        "event_participant_profiles": [
            persona_by_id[participant].model_dump(mode="json")
            for participant in context.event.participant_ids
        ],
        "previous_context_summary": {
            "same_chain_prior_events": [
                event.model_dump(mode="json")
                for event in context.same_chain_prior_events
            ],
            "recent_global_events": [
                event.model_dump(mode="json")
                for event in context.recent_global_events
            ],
        },
        "previous_user_preference_state": [
            entry.model_dump(mode="json")
            for entry in context.previous_preference_state
        ],
    }
    return _canonical_json(payload)


def validate_dialogue_payload(
    context: V1DialogueGenerationContext,
    payload: V1DialogueEventPayload,
    *,
    requested_turn_count: int,
) -> None:
    minimum_turns = (
        V1_BACKGROUND_DIALOGUE_TURNS
        if context.chain_kind == "background"
        else max(2, requested_turn_count - 2)
    )
    maximum_turns = requested_turn_count + 2
    if not minimum_turns <= len(payload.turns) <= maximum_turns:
        raise ValueError(
            f"event {context.event.event_id} requires {minimum_turns}.."
            f"{maximum_turns} dialogue turns around target "
            f"{requested_turn_count}, found {len(payload.turns)}"
        )
    if context.chain_kind == "background" and len(payload.turns) < 40:
        raise ValueError("paper contract requires >=40 turns per background event")
    allowed = set(context.event.participant_ids)
    used: set[str] = set()
    for turn in payload.turns:
        if turn.speaker_id not in allowed:
            raise ValueError("dialogue output references a non-participant speaker")
        used.add(turn.speaker_id)
        if _looks_like_non_human_speaker(turn.text):
            raise ValueError("dialogue output appears to contain an agent/tool turn")
    if len(allowed) < 2 or len(used) < 2:
        raise ValueError("event dialogue requires at least two human speakers")


def materialize_dialogue_turns(
    stage2: V1Stage2Artifact,
    generated: Sequence[V1GeneratedEventDialogue],
) -> tuple[V1DialogueTurnRecord, ...]:
    by_event = {item.event_id: item for item in generated}
    if len(by_event) != len(generated):
        raise ValueError("generated dialogue event IDs must be unique")
    persona_by_id = {
        persona.persona_id: persona
        for persona in stage2.persona_group.payload.personas
    }
    turns: list[V1DialogueTurnRecord] = []
    for timeline_item in stage2.interleaved_timeline:
        checkpoint = by_event.get(timeline_item.event.event_id)
        if checkpoint is None:
            raise ValueError(
                f"missing dialogue for event {timeline_item.event.event_id}"
            )
        expected_hash = canonical_json_sha256(
            timeline_item.event.model_dump(mode="json")
        )
        if checkpoint.source_event_sha256 != expected_hash:
            raise ValueError("dialogue checkpoint source event hash mismatch")
        for offset, line in enumerate(checkpoint.payload.turns):
            persona = persona_by_id.get(line.speaker_id)
            if persona is None:
                raise ValueError("dialogue speaker is absent from Persona group")
            turns.append(
                V1DialogueTurnRecord(
                    turn_id=f"{timeline_item.event.event_id}-turn-{offset + 1:03d}",
                    source_event_id=timeline_item.event.event_id,
                    timestamp=timeline_item.event.timestamp,
                    speaker_id=line.speaker_id,
                    speaker_name=persona.name,
                    text=line.text.strip(),
                )
            )
    return tuple(turns)


def relevant_tool_schemas_for_chain(
    chain: V1EventChainRecord,
    tool_schemas: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    source_tools = {
        update.attribute_path.rsplit(".", 1)[0]
        for event in chain.events
        for update in event.preference_updates
    }
    if not source_tools:
        raise ValueError("vehicle chain has no executable preference update")
    modules = {
        name.removeprefix("carcontrol_").split("_", 1)[0]
        for name in source_tools
    }
    selected = tuple(
        dict(schema)
        for schema in tool_schemas
        if isinstance(schema.get("name"), str)
        and schema["name"].removeprefix("carcontrol_").split("_", 1)[0]
        in modules
    )
    if not source_tools.issubset(
        {str(schema.get("name")) for schema in selected}
    ):
        raise ValueError("source chain Tool is absent from official Tool schemas")
    return selected


def render_final_quiz_generation_input(
    personas: Sequence[V1PersonaRecord],
    chain: V1EventChainRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    tool_schemas: Sequence[Mapping[str, Any]],
) -> str:
    if chain.kind != "vehicle" or chain.reasoning_type not in REASONING_TYPES:
        raise ValueError("final Quiz input requires one official vehicle chain")
    source_event_ids = {event.event_id for event in chain.events}
    selected_dialogue = [
        turn.model_dump(mode="json")
        for turn in dialogue_turns
        if turn.source_event_id in source_event_ids
    ]
    if not selected_dialogue:
        raise ValueError("final Quiz input has no source-chain dialogue")
    payload = {
        "personas": [persona.model_dump(mode="json") for persona in personas],
        "source_vehicle_chain": chain.model_dump(mode="json"),
        "source_chain_dialogue": selected_dialogue,
        "allowed_vehicle_tools": [dict(schema) for schema in tool_schemas],
    }
    return _canonical_json(payload)


def finalize_generated_quiz(
    generated: V1GeneratedFinalQuiz,
    chain: V1EventChainRecord,
    *,
    quiz_index: int,
    tool_schemas: Sequence[Mapping[str, Any]],
    runtime: Any,
    require_state_change: bool = True,
) -> tuple[V1FinalQuizRecord, int]:
    payload = generated.payload
    if payload.source_chain_id != chain.chain_id:
        raise ValueError("Quiz source chain mismatch")
    if payload.reasoning_type != chain.reasoning_type:
        raise ValueError("Quiz reasoning type mismatch")
    validators = _tool_validators(tool_schemas)
    world = runtime.create_world()
    initial_state = runtime.state(world)
    calls: list[V1GoldToolCallRecord] = []
    for draft in payload.gold_calls:
        arguments = draft.argument_map()
        validator = validators.get(draft.name)
        if validator is None:
            raise ValueError(f"Quiz references unknown Tool {draft.name}")
        try:
            validator.validate(arguments)
        except JsonSchemaValidationError as exc:
            raise ValueError(
                f"invalid Quiz arguments for {draft.name}: {exc.message}"
            ) from exc
        result = runtime.execute(world, draft.name, arguments)
        if not isinstance(result, Mapping) or result.get("success") is not True:
            message = result.get("message") if isinstance(result, Mapping) else result
            raise ValueError(f"simulator rejected {draft.name}: {message}")
        calls.append(V1GoldToolCallRecord(name=draft.name, arguments=arguments))
    target_state = runtime.state(world)
    if require_state_change and target_state == initial_state:
        raise ValueError("Quiz Tool sequence does not change simulator state")
    return (
        V1FinalQuizRecord(
            quiz_id=f"quiz-{quiz_index:02d}",
            source_chain_id=chain.chain_id,
            reasoning_type=payload.reasoning_type,
            gold_memory=payload.gold_memory.strip(),
            query=payload.query.strip(),
            gold_calls=tuple(calls),
            target_state=target_state,
            target_state_sha256=canonical_json_sha256(target_state),
        ),
        len(calls),
    )


def build_stage3_artifact(
    stage2: V1Stage2Artifact,
    generated_dialogues: Sequence[V1GeneratedEventDialogue],
    generated_quizzes: Sequence[V1GeneratedFinalQuiz],
    *,
    scenario_id: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    runtime: Any,
) -> V1Stage3Artifact:
    dialogue_turns = materialize_dialogue_turns(stage2, generated_dialogues)
    chain_by_id = {
        chain.chain_id: chain
        for chain in stage2.event_chains.payload.vehicle_chains
    }
    quiz_by_chain = {item.source_chain_id: item for item in generated_quizzes}
    if len(quiz_by_chain) != 10 or set(quiz_by_chain) != set(chain_by_id):
        raise ValueError("Stage 3 requires one generated Quiz per vehicle chain")
    final_quizzes: list[V1FinalQuizRecord] = []
    executed_call_count = 0
    for index, chain in enumerate(stage2.event_chains.payload.vehicle_chains, start=1):
        quiz, call_count = finalize_generated_quiz(
            quiz_by_chain[chain.chain_id],
            chain,
            quiz_index=index,
            tool_schemas=tool_schemas,
            runtime=runtime,
            require_state_change=False,
        )
        final_quizzes.append(quiz)
        executed_call_count += call_count
    scenario = V1ScenarioArtifact(
        scenario_id=scenario_id,
        personas=stage2.persona_group.payload.personas,
        event_chains=stage2.event_chains.payload.all_chains,
        dialogue_turns=dialogue_turns,
        final_quizzes=tuple(final_quizzes),
    )
    background_event_ids = {
        event.event_id
        for chain in stage2.event_chains.payload.background_chains
        for event in chain.events
    }
    background_turn_count = sum(
        1 for turn in dialogue_turns if turn.source_event_id in background_event_ids
    )
    history_text = serialize_v1_history(scenario.dialogue_turns)
    qa_payload = serialize_v1_qa(scenario.final_quizzes)
    validate_public_v1_payload(history_text, qa_payload, tool_schemas)
    audit = V1Stage3Audit(
        dialogue_event_count=len(generated_dialogues),
        dialogue_turn_count=len(dialogue_turns),
        background_dialogue_turn_count=background_turn_count,
        vehicle_dialogue_turn_count=len(dialogue_turns) - background_turn_count,
        quiz_count=len(final_quizzes),
        simulator_executed_call_count=executed_call_count,
        simulator_passed_quiz_count=len(final_quizzes),
        history_line_contract_passed=True,
        qa_contract_passed=True,
        passed=True,
    )
    body = {
        "schema_version": V1_STAGE3_SCHEMA_VERSION,
        "source_stage2_sha256": stage2.artifact_sha256,
        "scenario": scenario.model_dump(mode="json"),
        "generated_dialogues": [
            item.model_dump(mode="json") for item in generated_dialogues
        ],
        "generated_quizzes": [
            item.model_dump(mode="json") for item in generated_quizzes
        ],
        "audit": audit.model_dump(mode="json"),
    }
    return V1Stage3Artifact(
        **body,
        artifact_sha256=canonical_json_sha256(body),
    )


def serialize_v1_history(turns: Sequence[V1DialogueTurnRecord]) -> str:
    return "\n".join(
        f"[{turn.timestamp.replace('T', ' ')}] {turn.speaker_name}: {turn.text.strip()}"
        for turn in turns
    ) + "\n"


def serialize_v1_qa(
    quizzes: Sequence[V1FinalQuizRecord],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "related_to_vehicle_preference": [
            {
                "gold_memory": quiz.gold_memory,
                "reasoning_type": quiz.reasoning_type,
                "query": quiz.query,
                "new_answer": [
                    format_v1_tool_call(call) for call in quiz.gold_calls
                ],
            }
            for quiz in quizzes
        ]
    }


def format_v1_tool_call(call: V1GoldToolCallRecord) -> str:
    arguments = ", ".join(
        f"{name}={json.dumps(value, ensure_ascii=False, allow_nan=False)}"
        for name, value in call.arguments.items()
    )
    return f"{call.name}({arguments})"


def validate_public_v1_payload(
    history_text: str,
    qa_payload: Mapping[str, Any],
    tool_schemas: Sequence[Mapping[str, Any]],
) -> None:
    lines = history_text.splitlines()
    if not lines or any(_HISTORY_LINE_RE.fullmatch(line) is None for line in lines):
        raise ValueError("generated history violates the public V1 line format")
    records = qa_payload.get("related_to_vehicle_preference")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("generated QA requires exactly 10 public V1 records")
    validators = _tool_validators(tool_schemas)
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "gold_memory",
            "reasoning_type",
            "query",
            "new_answer",
        }:
            raise ValueError("generated QA record violates the public V1 field schema")
        if record["reasoning_type"] not in REASONING_TYPES:
            raise ValueError("generated QA has an unknown reasoning type")
        answers = record["new_answer"]
        if not isinstance(answers, list) or not answers:
            raise ValueError("generated QA answer list is empty")
        for source in answers:
            call = _parse_serialized_tool_call(source)
            validator = validators.get(call.name)
            if validator is None:
                raise ValueError(f"generated QA references unknown Tool {call.name}")
            validator.validate(call.arguments)


def write_stage3_artifact(
    output_dir: Path | str,
    artifact: V1Stage3Artifact,
    *,
    public_scenario_index: int,
) -> dict[str, Path]:
    if public_scenario_index < 1:
        raise ValueError("public_scenario_index must be positive")
    root = Path(output_dir).expanduser().resolve()
    history_path = root / "benchmark" / "history" / (
        f"history_{public_scenario_index}.txt"
    )
    qa_path = root / "benchmark" / "qa_data" / f"qa_{public_scenario_index}.json"
    stage3_path = root / "stage3.json"
    for path in (history_path, qa_path, stage3_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        history_path,
        serialize_v1_history(artifact.scenario.dialogue_turns),
    )
    _atomic_write_text(
        qa_path,
        json.dumps(
            serialize_v1_qa(artifact.scenario.final_quizzes),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _atomic_write_text(
        stage3_path,
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return {"stage3": stage3_path, "history": history_path, "qa": qa_path}


def _tool_validators(
    tool_schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Draft202012Validator]:
    validators: dict[str, Draft202012Validator] = {}
    for schema in tool_schemas:
        name = schema.get("name")
        parameters = schema.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            raise ValueError("invalid official Tool schema")
        normalized = dict(parameters)
        normalized.setdefault("additionalProperties", False)
        validators[name] = Draft202012Validator(normalized)
    return validators


def _parse_serialized_tool_call(source: Any) -> GoldToolCall:
    if not isinstance(source, str):
        raise ValueError("serialized Tool call must be text")
    match = re.fullmatch(r"(carcontrol_[A-Za-z0-9_]+)\((.*)\)", source.strip())
    if match is None:
        raise ValueError(f"invalid serialized Tool call: {source!r}")
    name, raw_arguments = match.groups()
    arguments: dict[str, Any] = {}
    if raw_arguments.strip():
        decoder = json.JSONDecoder()
        position = 0
        while position < len(raw_arguments):
            key_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", raw_arguments[position:])
            if key_match is None:
                raise ValueError(f"invalid Tool arguments: {source!r}")
            key = key_match.group(1)
            position += key_match.end()
            value, consumed = decoder.raw_decode(raw_arguments[position:])
            arguments[key] = value
            position += consumed
            if position == len(raw_arguments):
                break
            if not raw_arguments.startswith(", ", position):
                raise ValueError(f"invalid Tool argument separator: {source!r}")
            position += 2
    return GoldToolCall(name=name, arguments=arguments, source=source)


def _looks_like_non_human_speaker(text: str) -> bool:
    lowered = text.strip().casefold()
    return lowered.startswith(("assistant:", "system:", "vehicle:", "tool:"))


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _raise_for_bad_response(response: Any, *, stage: str) -> None:
    status = getattr(response, "status", None)
    if status in {"failed", "incomplete", "cancelled"}:
        raise RuntimeError(f"V1 {stage} provider returned status {status}")


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
