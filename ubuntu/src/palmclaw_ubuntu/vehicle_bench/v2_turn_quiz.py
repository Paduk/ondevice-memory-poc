"""Causally bounded V1-style Turn Quiz generation for VehicleMemBench V2."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1ArgumentValue,
    V1DialogueTurnRecord,
    V1EventChainRecord,
    V1GoldToolCallRecord,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    V1FinalQuizPayload,
    V1GeneratedEventDialogue,
    V1GeneratedFinalQuiz,
    V1ToolCallDraft,
    finalize_generated_quiz,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridTurnLabel,
    event_dialogue_turns,
    text_sha256,
)

V2_TURN_QUIZ_SCHEMA_VERSION = "vehiclemembench-v2-turn-quiz-schema-v1"
V2_TURN_QUIZ_PROMPT_VERSION = "vehiclemembench-v2-turn-quiz-terra-v4"

V2_TURN_QUIZ_INSTRUCTIONS = """
Generate one natural VehicleMemBench-style executable vehicle request for the
supplied causal checkpoint.

Rules:
- Ask the vehicle to apply the remembered preference; do not ask the user to
  repeat, summarize, or explain the conversation.
- Preserve the assigned reasoning type, target person, setting, context, and
  condition, but phrase the request naturally as an occupant would.
- Do not reveal the target preference value in the query, including its number,
  enum, color, level, or on/off state. The query must require memory retrieval.
  Selector/context arguments such as a seat or lamp location may be stated.
- For boolean settings, never say enable, disable, enabled, disabled, turn on,
  turn off, activate, or deactivate. Say only "apply the remembered setting."
- Do not mention Tool/function names, JSON, schemas, gold calls, or benchmark
  terminology in the query.
- Use only current_memory and causal_dialogue_prefix. Never infer or mention a
  future event or final V1 Quiz.
- Treat the query as a new request at this checkpoint. Do not invent an
  accident, reset, changed setting, current value, or prior user action unless
  it is explicit in causal_dialogue_prefix. When it is not explicit, simply
  ask to apply the remembered setting in its supported condition.
- If multiple target calls are supplied, express them as one coherent request
  and make every call necessary to satisfy it.
- Return one or more exact complete lines copied from current_memory that make
  the query answerable, plus a short one-line explanation.
""".strip()


def _neutral_boolean_query(memory_evidence_lines: Sequence[str]) -> str:
    conditions = []
    for line in memory_evidence_lines:
        marker = "; condition="
        if marker not in line:
            continue
        condition = line.rsplit(marker, 1)[1].strip().rstrip(".")
        if condition and condition not in conditions:
            conditions.append(condition)
    if not conditions:
        return "Please apply the relevant remembered vehicle setting now."
    joined = "; and ".join(conditions)
    return f"Please apply the remembered vehicle settings for: {joined}."


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2TurnQuizEventCheckpoint(Protocol):
    """Structural event checkpoint view shared by all V2 generation paths."""

    event_id: str
    checkpoint_sha256: str
    turn_labels: tuple[V2HybridTurnLabel, ...]
    after_memory: str
    after_memory_sha256: str


class V2TurnQuizQueryPayload(_StrictModel):
    query: str = Field(min_length=1, max_length=1_024)
    memory_evidence_lines: tuple[str, ...] = Field(min_length=1, max_length=8)
    reason: str = Field(min_length=1, max_length=320)

    @field_validator("query", "reason")
    @classmethod
    def normalize_single_line(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Turn Quiz query and reason must each be one line")
        return normalized

    @field_validator("memory_evidence_lines")
    @classmethod
    def normalize_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value or "\n" in value or "\r" in value for value in normalized):
            raise ValueError("Turn Quiz evidence must contain complete single lines")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Turn Quiz evidence lines must be unique")
        return normalized


class V2TurnQuizGenerationContext(_StrictModel):
    quiz_id: str = Field(pattern=r"^turn-quiz-[A-Za-z0-9._-]+$")
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_event_id: str = Field(min_length=1)
    source_chain_id: str = Field(min_length=1)
    reasoning_type: str = Field(min_length=1)
    global_turn_index: int = Field(ge=0)
    event_turn_index: int = Field(ge=0)
    turn_id: str = Field(min_length=1)
    current_memory: str = Field(min_length=1)
    current_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_dialogue_prefix: tuple[V1DialogueTurnRecord, ...] = Field(min_length=1)
    causal_prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_calls: tuple[V1GoldToolCallRecord, ...] = Field(min_length=1)
    gold_calls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hidden_value_arguments: tuple[V1ArgumentValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_hashes(self) -> V2TurnQuizGenerationContext:
        if canonical_json_sha256(
            [turn.model_dump(mode="json") for turn in self.causal_dialogue_prefix]
        ) != self.causal_prefix_sha256:
            raise ValueError("Turn Quiz causal-prefix hash is invalid")
        if canonical_json_sha256(
            [call.model_dump(mode="json") for call in self.gold_calls]
        ) != self.gold_calls_sha256:
            raise ValueError("Turn Quiz gold-call hash is invalid")
        if text_sha256(self.current_memory) != self.current_memory_sha256:
            raise ValueError("Turn Quiz current-memory hash is invalid")
        return self


class V2GeneratedTurnQuizQuery(_StrictModel):
    quiz_id: str = Field(pattern=r"^turn-quiz-[A-Za-z0-9._-]+$")
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_calls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: V2TurnQuizQueryPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V2TurnQuizRecord(_StrictModel):
    quiz_id: str = Field(pattern=r"^turn-quiz-[A-Za-z0-9._-]+$")
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_event_id: str = Field(min_length=1)
    source_chain_id: str = Field(min_length=1)
    reasoning_type: str = Field(min_length=1)
    global_turn_index: int = Field(ge=0)
    event_turn_index: int = Field(ge=0)
    turn_id: str = Field(min_length=1)
    memory_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query: str = Field(min_length=1)
    memory_evidence_lines: tuple[str, ...] = Field(min_length=1)
    gold_calls: tuple[V1GoldToolCallRecord, ...] = Field(min_length=1)
    target_state: dict[str, JsonValue] = Field(min_length=1)
    target_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_query: V2GeneratedTurnQuizQuery
    simulator_call_count: int = Field(ge=1)
    state_changed: bool
    quiz_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hashes(self) -> V2TurnQuizRecord:
        if canonical_json_sha256(self.target_state) != self.target_state_sha256:
            raise ValueError("Turn Quiz target-state hash is invalid")
        body = self.model_dump(mode="json", exclude={"quiz_sha256"})
        if canonical_json_sha256(body) != self.quiz_sha256:
            raise ValueError("Turn Quiz record hash is invalid")
        return self


class V2TurnQuizAudit(_StrictModel):
    update_checkpoint_count: int = Field(ge=0)
    quiz_count: int = Field(ge=0)
    simulator_call_count: int = Field(ge=0)
    simulator_passed_quiz_count: int = Field(ge=0)
    answerability_evidence_passed: bool
    causal_cutoff_passed: bool
    passed: bool


class V2TurnQuizArtifact(_StrictModel):
    schema_version: str = V2_TURN_QUIZ_SCHEMA_VERSION
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_checkpoint_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quizzes: tuple[V2TurnQuizRecord, ...]
    audit: V2TurnQuizAudit
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> V2TurnQuizArtifact:
        body = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(body) != self.artifact_sha256:
            raise ValueError("Turn Quiz artifact hash is invalid")
        return self


class V2TurnQuizGenerationModel(Protocol):
    def generate(
        self,
        context: V2TurnQuizGenerationContext,
        *,
        personas: Sequence[Mapping[str, Any]],
        source_event_description: str,
        retry_feedback: str | None = None,
    ) -> V2GeneratedTurnQuizQuery: ...


class OpenAIV2TurnQuizGenerationModel:
    """Generate only the natural query; the executable answer stays deterministic."""

    def __init__(
        self,
        model_id: str = V1_DEFAULT_GENERATION_MODEL,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 2_048,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Turn Quiz generation model ID is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        context: V2TurnQuizGenerationContext,
        *,
        personas: Sequence[Mapping[str, Any]],
        source_event_description: str,
        retry_feedback: str | None = None,
    ) -> V2GeneratedTurnQuizQuery:
        provider_input = render_turn_quiz_generation_input(
            context,
            personas=personas,
            source_event_description=source_event_description,
        )
        instructions = V2_TURN_QUIZ_INSTRUCTIONS
        if retry_feedback is not None:
            instructions += (
                "\n\nThe previous candidate failed validation: "
                f"{retry_feedback}. Produce a materially different query that "
                "satisfies the rules; do not reveal the hidden value."
            )
            if "hidden answer value:" in retry_feedback:
                instructions += (
                    " Do not write or spell out any hidden scalar even when the "
                    "same number appears incidentally in a context phrase. "
                    "Rephrase such cardinalities without a number (for example, "
                    "use 'group return' instead of 'three-person return')."
                )
            if "hidden answer value: enabled" in retry_feedback:
                instructions += (
                    " For this boolean case, use the neutral pattern 'Please "
                    "apply the remembered setting for this situation: "
                    "<condition>.' Do not describe the direction or resulting "
                    "vehicle action."
                )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": instructions,
            "input": provider_input,
            "text_format": V2TurnQuizQueryPayload,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        started_at = time.monotonic()
        response = self._client.responses.parse(**request)
        latency_ms = int((time.monotonic() - started_at) * 1_000)
        status = getattr(response, "status", None)
        if status in {"failed", "incomplete", "cancelled"}:
            raise RuntimeError(f"V2 Turn Quiz provider returned status {status}")
        payload = V2TurnQuizQueryPayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        try:
            validate_turn_quiz_query(context, payload)
        except ValueError as exc:
            if (
                retry_feedback is None
                or "hidden answer value: enabled" not in str(exc)
            ):
                raise
            payload = payload.model_copy(
                update={
                    "query": _neutral_boolean_query(payload.memory_evidence_lines)
                }
            )
            try:
                validate_turn_quiz_query(context, payload)
            except ValueError:
                payload = payload.model_copy(
                    update={
                        "query": (
                            "Please apply the relevant remembered vehicle settings "
                            "for the current situation."
                        )
                    }
                )
                validate_turn_quiz_query(context, payload)
        return V2GeneratedTurnQuizQuery(
            quiz_id=context.quiz_id,
            source_checkpoint_sha256=context.source_checkpoint_sha256,
            current_memory_sha256=context.current_memory_sha256,
            causal_prefix_sha256=context.causal_prefix_sha256,
            gold_calls_sha256=context.gold_calls_sha256,
            payload=payload,
            model_id=self.model_id,
            prompt_version=V2_TURN_QUIZ_PROMPT_VERSION,
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
        )


def build_turn_quiz_contexts(
    stage2: V1Stage2Artifact,
    checkpoints: Sequence[V2TurnQuizEventCheckpoint],
    dialogues: Sequence[V1GeneratedEventDialogue],
) -> tuple[V2TurnQuizGenerationContext, ...]:
    dialogue_by_event = {dialogue.event_id: dialogue for dialogue in dialogues}
    if len(dialogue_by_event) != len(dialogues):
        raise ValueError("Turn Quiz dialogue event IDs must be unique")
    timeline_by_event = {
        item.event.event_id: item for item in stage2.interleaved_timeline
    }
    chain_by_event = {
        event.event_id: chain
        for chain in stage2.event_chains.payload.all_chains
        for event in chain.events
    }
    contexts = []
    for checkpoint in checkpoints:
        timeline_item = timeline_by_event.get(checkpoint.event_id)
        chain = chain_by_event.get(checkpoint.event_id)
        dialogue = dialogue_by_event.get(checkpoint.event_id)
        if timeline_item is None or chain is None or dialogue is None:
            raise ValueError("Turn Quiz source checkpoint is missing Stage 2 dialogue")
        if chain.kind != "vehicle" or chain.reasoning_type is None:
            if any(label.decision == "UPDATE" for label in checkpoint.turn_labels):
                raise ValueError(
                    "Turn Quiz UPDATE cannot originate in background chain"
                )
            continue
        event_turns = event_dialogue_turns(stage2, dialogue)
        for label in checkpoint.turn_labels:
            if label.decision != "UPDATE":
                continue
            contexts.append(
                build_turn_quiz_context(
                    checkpoint=checkpoint,
                    label=label,
                    chain=chain,
                    event=timeline_item.event,
                    event_turns=event_turns,
                )
            )
    return tuple(contexts)


def build_turn_quiz_context(
    *,
    checkpoint: V2TurnQuizEventCheckpoint,
    label: V2HybridTurnLabel,
    chain: V1EventChainRecord,
    event,
    event_turns: Sequence[V1DialogueTurnRecord],
) -> V2TurnQuizGenerationContext:
    if label.decision != "UPDATE" or label.after_memory is None:
        raise ValueError("Turn Quiz context requires an UPDATE memory snapshot")
    if label.source_event_id != event.event_id or checkpoint.event_id != event.event_id:
        raise ValueError("Turn Quiz label/event source mismatch")
    if label.event_turn_index >= len(event_turns):
        raise ValueError("Turn Quiz cutoff exceeds its event dialogue")
    causal_prefix = tuple(event_turns[: label.event_turn_index + 1])
    gold_calls = derive_turn_quiz_gold_calls(
        event.preference_updates,
        label.source_update_indexes,
    )
    hidden_values = tuple(
        V1ArgumentValue(
            name=event.preference_updates[index].attribute_path.rsplit(".", 1)[1],
            value=event.preference_updates[index].new_value,
        )
        for index in label.source_update_indexes
    )
    return V2TurnQuizGenerationContext(
        quiz_id=f"turn-quiz-{event.event_id}-{label.event_turn_index + 1:03d}",
        source_checkpoint_sha256=checkpoint.checkpoint_sha256,
        source_event_id=event.event_id,
        source_chain_id=chain.chain_id,
        reasoning_type=chain.reasoning_type,
        global_turn_index=label.global_turn_index,
        event_turn_index=label.event_turn_index,
        turn_id=label.turn_id,
        current_memory=label.after_memory,
        current_memory_sha256=label.after_memory_sha256,
        causal_dialogue_prefix=causal_prefix,
        causal_prefix_sha256=canonical_json_sha256(
            [turn.model_dump(mode="json") for turn in causal_prefix]
        ),
        gold_calls=gold_calls,
        gold_calls_sha256=canonical_json_sha256(
            [call.model_dump(mode="json") for call in gold_calls]
        ),
        hidden_value_arguments=hidden_values,
    )


def derive_turn_quiz_gold_calls(
    preference_updates: Sequence,
    source_update_indexes: Sequence[int],
) -> tuple[V1GoldToolCallRecord, ...]:
    if not source_update_indexes:
        raise ValueError("Turn Quiz requires at least one source update")
    calls = []
    for index in source_update_indexes:
        if not 0 <= index < len(preference_updates):
            raise ValueError("Turn Quiz source update index is out of range")
        update = preference_updates[index]
        try:
            tool_name, value_argument = update.attribute_path.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(
                "Turn Quiz update attribute_path is not executable"
            ) from exc
        if not tool_name.startswith("carcontrol_") or not value_argument:
            raise ValueError("Turn Quiz update has an invalid Tool attribute path")
        arguments = update.context_argument_map()
        if (
            value_argument in arguments
            and arguments[value_argument] != update.new_value
        ):
            raise ValueError(
                "Turn Quiz value argument conflicts with a context argument"
            )
        arguments[value_argument] = update.new_value
        calls.append(V1GoldToolCallRecord(name=tool_name, arguments=arguments))
    return tuple(calls)


def render_turn_quiz_generation_input(
    context: V2TurnQuizGenerationContext,
    *,
    personas: Sequence[Mapping[str, Any]],
    source_event_description: str,
) -> str:
    payload = {
        "quiz_checkpoint": {
            "global_turn_index": context.global_turn_index,
            "event_turn_index": context.event_turn_index,
            "assigned_reasoning_type": context.reasoning_type,
        },
        "personas": [dict(persona) for persona in personas],
        "source_event_description": source_event_description,
        "current_memory": context.current_memory,
        "causal_dialogue_prefix": [
            turn.model_dump(mode="json") for turn in context.causal_dialogue_prefix
        ],
        "generation_only_target_calls": [
            call.model_dump(mode="json") for call in context.gold_calls
        ],
    }
    return _canonical_json(payload)


def validate_turn_quiz_query(
    context: V2TurnQuizGenerationContext,
    payload: V2TurnQuizQueryPayload,
) -> None:
    memory_lines = {
        line.strip()
        for line in context.current_memory.splitlines()
        if line.strip()
    }
    if any(line not in memory_lines for line in payload.memory_evidence_lines):
        raise ValueError("Turn Quiz evidence is not an exact current-memory line")
    for call in context.gold_calls:
        if call.name.casefold() in payload.query.casefold():
            raise ValueError("Turn Quiz query leaks an executable Tool name")
        if not any(
            _memory_line_supports_call(line, call)
            for line in payload.memory_evidence_lines
        ):
            raise ValueError("Turn Quiz evidence does not support every gold call")
    for argument in context.hidden_value_arguments:
        if _query_reveals_scalar(payload.query, argument.value):
            raise ValueError(
                f"Turn Quiz query reveals hidden answer value: {argument.name}"
            )


def finalize_turn_quiz(
    context: V2TurnQuizGenerationContext,
    generated: V2GeneratedTurnQuizQuery,
    *,
    chain: V1EventChainRecord,
    quiz_index: int,
    tool_schemas: Sequence[Mapping[str, Any]],
    runtime: Any,
) -> V2TurnQuizRecord:
    _validate_generated_query(context, generated)
    drafts = tuple(
        V1ToolCallDraft(
            name=call.name,
            arguments=tuple(
                V1ArgumentValue(name=name, value=value)
                for name, value in call.arguments.items()
            ),
        )
        for call in context.gold_calls
    )
    compatibility_quiz = V1GeneratedFinalQuiz(
        source_chain_id=chain.chain_id,
        source_chain_sha256=canonical_json_sha256(chain.model_dump(mode="json")),
        payload=V1FinalQuizPayload(
            source_chain_id=chain.chain_id,
            reasoning_type=chain.reasoning_type,
            gold_memory="\n".join(generated.payload.memory_evidence_lines),
            query=generated.payload.query,
            gold_calls=drafts,
        ),
        model_id=generated.model_id,
        prompt_version=generated.prompt_version,
        input_sha256=generated.input_sha256,
        response_id=generated.response_id,
        usage=generated.usage,
    )
    finalized, call_count = finalize_generated_quiz(
        compatibility_quiz,
        chain,
        quiz_index=quiz_index,
        tool_schemas=tool_schemas,
        runtime=runtime,
        require_state_change=False,
    )
    initial_state = runtime.state(runtime.create_world())
    body = {
        "quiz_id": context.quiz_id,
        "source_checkpoint_sha256": context.source_checkpoint_sha256,
        "source_event_id": context.source_event_id,
        "source_chain_id": context.source_chain_id,
        "reasoning_type": context.reasoning_type,
        "global_turn_index": context.global_turn_index,
        "event_turn_index": context.event_turn_index,
        "turn_id": context.turn_id,
        "memory_snapshot_sha256": context.current_memory_sha256,
        "query": finalized.query,
        "memory_evidence_lines": generated.payload.memory_evidence_lines,
        "gold_calls": [call.model_dump(mode="json") for call in finalized.gold_calls],
        "target_state": finalized.target_state,
        "target_state_sha256": finalized.target_state_sha256,
        "generated_query": generated.model_dump(mode="json"),
        "simulator_call_count": call_count,
        "state_changed": finalized.target_state != initial_state,
    }
    return V2TurnQuizRecord(**body, quiz_sha256=canonical_json_sha256(body))


def build_turn_quiz_artifact(
    stage2: V1Stage2Artifact,
    contexts: Sequence[V2TurnQuizGenerationContext],
    quizzes: Sequence[V2TurnQuizRecord],
    *,
    update_checkpoint_count: int | None = None,
) -> V2TurnQuizArtifact:
    by_id = {quiz.quiz_id: quiz for quiz in quizzes}
    if len(by_id) != len(quizzes) or set(by_id) != {item.quiz_id for item in contexts}:
        raise ValueError("Turn Quiz records must cover every UPDATE checkpoint")
    ordered = tuple(by_id[context.quiz_id] for context in contexts)
    checkpoint_hashes = tuple(
        dict.fromkeys(context.source_checkpoint_sha256 for context in contexts)
    )
    audit = V2TurnQuizAudit(
        update_checkpoint_count=(
            len(contexts)
            if update_checkpoint_count is None
            else update_checkpoint_count
        ),
        quiz_count=len(ordered),
        simulator_call_count=sum(item.simulator_call_count for item in ordered),
        simulator_passed_quiz_count=len(ordered),
        answerability_evidence_passed=True,
        causal_cutoff_passed=True,
        passed=len(ordered) == len(contexts),
    )
    body = {
        "schema_version": V2_TURN_QUIZ_SCHEMA_VERSION,
        "source_stage2_sha256": stage2.artifact_sha256,
        "source_checkpoint_set_sha256": canonical_json_sha256(checkpoint_hashes),
        "quizzes": [quiz.model_dump(mode="json") for quiz in ordered],
        "audit": audit.model_dump(mode="json"),
    }
    return V2TurnQuizArtifact(**body, artifact_sha256=canonical_json_sha256(body))


def write_turn_quiz_artifact(
    output_dir: Path | str,
    artifact: V2TurnQuizArtifact,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "turn-quizzes.json"
    _atomic_write_json(path, artifact.model_dump(mode="json"))
    return path


def _validate_generated_query(
    context: V2TurnQuizGenerationContext,
    generated: V2GeneratedTurnQuizQuery,
) -> None:
    expected = (
        (generated.quiz_id, context.quiz_id, "quiz ID"),
        (
            generated.source_checkpoint_sha256,
            context.source_checkpoint_sha256,
            "checkpoint",
        ),
        (generated.current_memory_sha256, context.current_memory_sha256, "memory"),
        (generated.causal_prefix_sha256, context.causal_prefix_sha256, "prefix"),
        (generated.gold_calls_sha256, context.gold_calls_sha256, "gold calls"),
    )
    for actual, wanted, field in expected:
        if actual != wanted:
            raise ValueError(f"Turn Quiz generated {field} hash mismatch")
    validate_turn_quiz_query(context, generated.payload)


def _memory_line_supports_call(line: str, call: V1GoldToolCallRecord) -> bool:
    normalized_line = _normalized_words(line)
    if _normalized_words(call.name) not in normalized_line:
        return False
    for value in call.arguments.values():
        candidate = _normalized_words(str(value))
        if candidate and candidate not in normalized_line:
            return False
    return True


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))


def _query_reveals_scalar(query: str, value: Any) -> bool:
    normalized = f" {_normalized_words(query)} "
    if isinstance(value, bool):
        candidates = ("true", "on", "enabled") if value else (
            "false",
            "off",
            "disabled",
        )
    elif isinstance(value, int) and not isinstance(value, bool):
        words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        candidates = (str(value), words.get(value, ""))
    else:
        candidates = (_normalized_words(str(value)),)
    return any(
        candidate
        and re.search(rf"\s{re.escape(candidate)}\s", normalized) is not None
        for candidate in candidates
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
