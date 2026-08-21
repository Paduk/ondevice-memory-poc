"""Native one-turn dialogue generation for VehicleMemBench V2."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.providers import apply_recursive_summary_patch
from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventRecord,
    V1PersonaRecord,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    V1DialogueEventPayload,
    V1DialogueGenerationContext,
    V1DialogueLine,
    V1GeneratedEventDialogue,
    build_dialogue_generation_contexts,
    dialogue_turn_target,
    validate_dialogue_payload,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    HybridReasonCode,
    V2HybridEvidence,
    V2HybridMemoryEntry,
    V2HybridPatchOperation,
    V2HybridTurnLabel,
    V2HybridUpdateAlignment,
    build_hybrid_update_operations,
    render_hybrid_memory,
    text_sha256,
)

V2_NATIVE_TURNWISE_SCHEMA_VERSION = "vehiclemembench-v2-native-turnwise-v1"
V2_NATIVE_TURN_PROMPT_VERSION = "vehiclemembench-v2-native-turn-terra-v1"
V2_NATIVE_ALIGNMENT_PROMPT_VERSION = (
    "vehiclemembench-v2-native-turn-alignment-terra-v1"
)
V2_NATIVE_RECENT_TURN_WINDOW = 8

V2_NATIVE_TURN_INSTRUCTIONS = """
Generate exactly the next single human-to-human dialogue turn for one
VehicleMemBench event.

Paper-based dialogue rules:
- Use one supplied event participant as speaker. Never use a vehicle agent,
  assistant, system, Tool, or Tool response as speaker.
- Continue the supplied causal event prefix naturally and faithfully preserve
  the event's person, value, condition, reference, correction, and temporal
  meaning.
- For a vehicle event, reveal the preference naturally as an aside, complaint,
  recollection, confirmation, or request to another occupant. Do not phrase it
  as a direct command to the vehicle.
- Do not reveal a later event, delayed Quiz, target state, memory label, or
  benchmark terminology.
- Avoid repetitive paraphrase. Move the conversation toward covering the
  current event and a natural close by the final requested turn.

Output exactly one speaker_id and one non-empty single-line text value.
""".strip()

V2_NATIVE_ALIGNMENT_INSTRUCTIONS = """
At the current causal dialogue turn, decide whether one or more unresolved
structured vehicle-preference updates have become explicitly and completely
supported.

Rules:
- Use only current_approved_memory, unresolved_structured_updates, and the
  supplied causal_event_prefix. Never infer a future turn or external fact.
- UPDATE only when the prefix supports the update's subject, setting/value, and
  any condition or context needed to distinguish it.
- Every UPDATE alignment must quote an exact non-empty span from the current
  turn and use the current evidence_turn_index. Do not align an earlier turn
  retrospectively.
- Do not align a source_update_index that is absent from unresolved updates.
- Return NO_OP when evidence is incomplete, merely proposed by another person,
  unrelated, or already represented by the committed update indexes.
- Use DUPLICATE_ALREADY_STORED only for a clear repetition; otherwise use
  NO_NEW_VEHICLE_FACT. Give one short single-line reason.
""".strip()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2NativeTurnPayload(_StrictModel):
    speaker_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4_096)

    @field_validator("speaker_id", "text")
    @classmethod
    def normalize_single_line(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Native dialogue fields must be one non-empty line")
        return normalized


class V2GeneratedNativeTurn(_StrictModel):
    schema_version: str = V2_NATIVE_TURNWISE_SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    source_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeline_index: int = Field(ge=0)
    event_turn_index: int = Field(ge=0)
    target_turn_count: int = Field(gt=0)
    causal_prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recent_window_start_index: int = Field(ge=0)
    payload: V2NativeTurnPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]

    @model_validator(mode="after")
    def validate_turn_position(self) -> V2GeneratedNativeTurn:
        if self.event_turn_index >= self.target_turn_count:
            raise ValueError("Native event turn index exceeds target turn count")
        if self.recent_window_start_index > self.event_turn_index:
            raise ValueError("Native recent-window start exceeds turn index")
        return self


class V2NativeAlignmentPayload(_StrictModel):
    decision: str = Field(pattern=r"^(NO_OP|UPDATE)$")
    reason_code: HybridReasonCode
    reason: str = Field(min_length=1, max_length=320)
    alignments: tuple[V2HybridUpdateAlignment, ...] = Field(max_length=32)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Native alignment reason must be one non-empty line")
        return normalized

    @model_validator(mode="after")
    def validate_decision_shape(self) -> V2NativeAlignmentPayload:
        if self.decision == "UPDATE":
            if not self.alignments:
                raise ValueError("Native UPDATE alignment requires evidence")
            if self.reason_code not in {
                "NEW_VEHICLE_MEMORY",
                "UPDATED_VEHICLE_MEMORY",
            }:
                raise ValueError("Native UPDATE alignment requires an UPDATE code")
        else:
            if self.alignments:
                raise ValueError("Native NO_OP alignment cannot contain evidence")
            if self.reason_code not in {
                "NO_NEW_VEHICLE_FACT",
                "DUPLICATE_ALREADY_STORED",
            }:
                raise ValueError("Native NO_OP alignment requires a NO_OP code")
        return self


class V2GeneratedNativeAlignment(_StrictModel):
    event_id: str = Field(min_length=1)
    source_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dialogue_prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    before_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_turn_index: int = Field(ge=0)
    committed_source_update_indexes: tuple[int, ...]
    payload: V2NativeAlignmentPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V2NativeTurnCheckpoint(_StrictModel):
    schema_version: str = V2_NATIVE_TURNWISE_SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    source_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeline_index: int = Field(ge=0)
    event_turn_index: int = Field(ge=0)
    global_turn_index: int = Field(ge=0)
    previous_checkpoint_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    generated_turn: V2GeneratedNativeTurn
    generated_alignment: V2GeneratedNativeAlignment | None = None
    dialogue_turn: V1DialogueTurnRecord
    label: V2HybridTurnLabel
    committed_source_update_indexes: tuple[int, ...]
    memory_entries_after: tuple[V2HybridMemoryEntry, ...]
    after_memory: str
    after_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_checkpoint(self) -> V2NativeTurnCheckpoint:
        if self.generated_turn.event_id != self.event_id:
            raise ValueError("Native checkpoint generated-turn event mismatch")
        if self.dialogue_turn.source_event_id != self.event_id:
            raise ValueError("Native checkpoint dialogue event mismatch")
        if self.generated_turn.event_turn_index != self.event_turn_index:
            raise ValueError("Native checkpoint generated-turn index mismatch")
        if self.generated_alignment is not None:
            if self.generated_alignment.event_id != self.event_id:
                raise ValueError("Native checkpoint alignment event mismatch")
            if self.generated_alignment.event_turn_index != self.event_turn_index:
                raise ValueError("Native checkpoint alignment turn mismatch")
            if self.generated_alignment.payload.decision != self.label.decision:
                raise ValueError("Native checkpoint alignment decision mismatch")
        if self.label.event_turn_index != self.event_turn_index:
            raise ValueError("Native checkpoint label event-turn index mismatch")
        if self.label.global_turn_index != self.global_turn_index:
            raise ValueError("Native checkpoint label global-turn index mismatch")
        if len(set(self.committed_source_update_indexes)) != len(
            self.committed_source_update_indexes
        ) or tuple(sorted(self.committed_source_update_indexes)) != (
            self.committed_source_update_indexes
        ):
            raise ValueError("Native committed update indexes must be sorted/unique")
        if any(
            index not in self.committed_source_update_indexes
            for index in self.label.source_update_indexes
        ):
            raise ValueError("Native label update is absent from committed indexes")
        if render_hybrid_memory(self.memory_entries_after) != self.after_memory:
            raise ValueError("Native checkpoint entries do not reproduce memory")
        if text_sha256(self.after_memory) != self.after_memory_sha256:
            raise ValueError("Native checkpoint after-memory hash mismatch")
        if self.label.after_memory_sha256 != self.after_memory_sha256:
            raise ValueError("Native checkpoint label/memory hash mismatch")
        if self.label.decision == "UPDATE":
            if self.label.after_memory != self.after_memory:
                raise ValueError("Native UPDATE label does not store resulting memory")
        elif self.label.after_memory is not None:
            raise ValueError("Native NO_OP label cannot store resulting memory")
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if canonical_json_sha256(body) != self.checkpoint_sha256:
            raise ValueError("Native checkpoint hash mismatch")
        return self


class V2NativeEventAudit(_StrictModel):
    turn_count: int = Field(ge=1)
    no_op_count: int = Field(ge=0)
    update_count: int = Field(ge=0)
    patch_operation_count: int = Field(ge=0)
    committed_update_count: int = Field(ge=0)
    dialogue_usage: dict[str, int]
    alignment_usage: dict[str, int]
    completed: bool


class V2NativeEventArtifact(_StrictModel):
    schema_version: str = V2_NATIVE_TURNWISE_SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    source_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeline_index: int = Field(ge=0)
    target_turn_count: int = Field(gt=0)
    before_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_checkpoint_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    turn_checkpoints: tuple[V2NativeTurnCheckpoint, ...] = Field(min_length=1)
    generated_dialogue: V1GeneratedEventDialogue
    memory_entries_after: tuple[V2HybridMemoryEntry, ...]
    after_memory: str
    after_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit: V2NativeEventAudit
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def checkpoint_sha256(self) -> str:
        """Expose the final causal checkpoint to the shared Quiz pipeline."""

        return self.turn_checkpoints[-1].checkpoint_sha256

    @property
    def turn_labels(self) -> tuple[V2HybridTurnLabel, ...]:
        """Expose Native labels through the common event-checkpoint contract."""

        return tuple(checkpoint.label for checkpoint in self.turn_checkpoints)

    @model_validator(mode="after")
    def validate_artifact(self) -> V2NativeEventArtifact:
        if len(self.turn_checkpoints) != self.target_turn_count:
            raise ValueError("Native event artifact turn count is incomplete")
        if self.generated_dialogue.event_id != self.event_id:
            raise ValueError("Native event artifact dialogue mismatch")
        if render_hybrid_memory(self.memory_entries_after) != self.after_memory:
            raise ValueError("Native event artifact entries do not reproduce memory")
        if text_sha256(self.after_memory) != self.after_memory_sha256:
            raise ValueError("Native event artifact after-memory hash mismatch")
        if self.turn_checkpoints[-1].after_memory_sha256 != (
            self.after_memory_sha256
        ):
            raise ValueError("Native event artifact final checkpoint mismatch")
        body = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(body) != self.artifact_sha256:
            raise ValueError("Native event artifact hash mismatch")
        return self


class V2NativeScenarioAudit(_StrictModel):
    completed_event_count: int = Field(ge=1)
    total_event_count: int = Field(ge=1)
    turn_count: int = Field(ge=1)
    no_op_count: int = Field(ge=0)
    update_count: int = Field(ge=0)
    patch_operation_count: int = Field(ge=0)
    dialogue_usage: dict[str, int]
    alignment_usage: dict[str, int]
    memory_hash_chain_passed: bool
    checkpoint_hash_chain_passed: bool
    completed: bool

    @model_validator(mode="after")
    def validate_completion(self) -> V2NativeScenarioAudit:
        if self.completed_event_count > self.total_event_count:
            raise ValueError("Native scenario audit event count exceeds total")
        if self.completed != (
            self.completed_event_count == self.total_event_count
        ):
            raise ValueError("Native scenario audit completion flag is invalid")
        return self


class V2NativeScenarioArtifact(_StrictModel):
    schema_version: str = V2_NATIVE_TURNWISE_SCHEMA_VERSION
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_artifacts: tuple[V2NativeEventArtifact, ...] = Field(min_length=1)
    final_memory: str
    final_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit: V2NativeScenarioAudit
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_scenario_artifact(self) -> V2NativeScenarioArtifact:
        if self.event_artifacts[-1].after_memory != self.final_memory:
            raise ValueError("Native scenario final memory is inconsistent")
        if text_sha256(self.final_memory) != self.final_memory_sha256:
            raise ValueError("Native scenario final-memory hash is invalid")
        if self.audit.completed_event_count != len(self.event_artifacts):
            raise ValueError("Native scenario completed-event count is invalid")
        if self.audit.turn_count != sum(
            item.audit.turn_count for item in self.event_artifacts
        ):
            raise ValueError("Native scenario turn audit is inconsistent")
        if self.audit.no_op_count + self.audit.update_count != (
            self.audit.turn_count
        ):
            raise ValueError("Native scenario label audit is inconsistent")
        body = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(body) != self.artifact_sha256:
            raise ValueError("Native scenario artifact hash is invalid")
        return self


class V2NativeTurnGenerationModel(Protocol):
    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        context: V1DialogueGenerationContext,
        prior_event_turns: Sequence[V1DialogueTurnRecord],
        *,
        target_turn_count: int,
    ) -> V2GeneratedNativeTurn: ...


class V2NativeAlignmentModel(Protocol):
    def generate(
        self,
        event: V1EventRecord,
        dialogue_turns: Sequence[V1DialogueTurnRecord],
        *,
        previous_memory: str,
        committed_source_update_indexes: Sequence[int],
    ) -> V2GeneratedNativeAlignment: ...


class OpenAIV2NativeTurnGenerationModel:
    """Generate one causal dialogue turn per Responses call."""

    def __init__(
        self,
        model_id: str = V1_DEFAULT_GENERATION_MODEL,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 1_024,
        recent_turn_window: int = V2_NATIVE_RECENT_TURN_WINDOW,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Native dialogue model ID is required")
        if recent_turn_window < 1:
            raise ValueError("Native recent-turn window must be positive")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.recent_turn_window = recent_turn_window
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        context: V1DialogueGenerationContext,
        prior_event_turns: Sequence[V1DialogueTurnRecord],
        *,
        target_turn_count: int,
    ) -> V2GeneratedNativeTurn:
        provider_input = render_native_turn_generation_input(
            personas,
            context,
            prior_event_turns,
            target_turn_count=target_turn_count,
            recent_turn_window=self.recent_turn_window,
        )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V2_NATIVE_TURN_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V2NativeTurnPayload,
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
            raise RuntimeError(
                f"V2 Native turn provider returned status {status}"
            )
        payload = V2NativeTurnPayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        validate_native_turn_payload(context, prior_event_turns, payload)
        parsed_input = json.loads(provider_input)
        return V2GeneratedNativeTurn(
            event_id=context.event.event_id,
            source_event_sha256=canonical_json_sha256(
                context.event.model_dump(mode="json")
            ),
            timeline_index=context.timeline_index,
            event_turn_index=len(prior_event_turns),
            target_turn_count=target_turn_count,
            causal_prefix_sha256=native_causal_prefix_sha256(prior_event_turns),
            recent_window_start_index=parsed_input["turn_progress"][
                "recent_window_start_index"
            ],
            payload=payload,
            model_id=self.model_id,
            prompt_version=V2_NATIVE_TURN_PROMPT_VERSION,
            input_sha256=canonical_json_sha256(parsed_input),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
        )


class OpenAIV2NativeAlignmentModel:
    """Align unresolved structured updates at the current causal turn."""

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
            raise ValueError("Native alignment model ID is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        event: V1EventRecord,
        dialogue_turns: Sequence[V1DialogueTurnRecord],
        *,
        previous_memory: str,
        committed_source_update_indexes: Sequence[int],
    ) -> V2GeneratedNativeAlignment:
        provider_input = render_native_alignment_input(
            event,
            dialogue_turns,
            previous_memory=previous_memory,
            committed_source_update_indexes=committed_source_update_indexes,
        )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V2_NATIVE_ALIGNMENT_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V2NativeAlignmentPayload,
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
            raise RuntimeError(
                f"V2 Native alignment provider returned status {status}"
            )
        payload = V2NativeAlignmentPayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        validate_native_alignment_payload(
            event,
            dialogue_turns,
            payload,
            committed_source_update_indexes=committed_source_update_indexes,
        )
        parsed_input = json.loads(provider_input)
        committed = tuple(sorted(committed_source_update_indexes))
        return V2GeneratedNativeAlignment(
            event_id=event.event_id,
            source_event_sha256=canonical_json_sha256(
                event.model_dump(mode="json")
            ),
            source_dialogue_prefix_sha256=native_causal_prefix_sha256(
                dialogue_turns
            ),
            before_memory_sha256=text_sha256(previous_memory),
            event_turn_index=len(dialogue_turns) - 1,
            committed_source_update_indexes=committed,
            payload=payload,
            model_id=self.model_id,
            prompt_version=V2_NATIVE_ALIGNMENT_PROMPT_VERSION,
            input_sha256=canonical_json_sha256(parsed_input),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
        )


def render_native_alignment_input(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    *,
    previous_memory: str,
    committed_source_update_indexes: Sequence[int],
) -> str:
    committed = _validate_native_alignment_request(
        event,
        dialogue_turns,
        committed_source_update_indexes=committed_source_update_indexes,
    )
    unresolved = [
        {
            "source_update_index": index,
            **update.model_dump(mode="json"),
        }
        for index, update in enumerate(event.preference_updates)
        if index not in set(committed)
    ]
    payload = {
        "current_approved_memory": previous_memory or "(empty)",
        "committed_source_update_indexes": list(committed),
        "unresolved_structured_updates": unresolved,
        "current_evidence_turn_index": len(dialogue_turns) - 1,
        "causal_event_prefix": [
            {
                "evidence_turn_index": index,
                "turn_id": turn.turn_id,
                "speaker_id": turn.speaker_id,
                "speaker_name": turn.speaker_name,
                "text": turn.text,
            }
            for index, turn in enumerate(dialogue_turns)
        ],
    }
    return _canonical_json(payload)


def validate_native_alignment_payload(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    payload: V2NativeAlignmentPayload,
    *,
    committed_source_update_indexes: Sequence[int],
) -> None:
    committed = _validate_native_alignment_request(
        event,
        dialogue_turns,
        committed_source_update_indexes=committed_source_update_indexes,
    )
    unresolved = set(range(len(event.preference_updates))) - set(committed)
    current_index = len(dialogue_turns) - 1
    indexes = [alignment.source_update_index for alignment in payload.alignments]
    if len(indexes) != len(set(indexes)):
        raise ValueError("Native alignment update indexes must be unique")
    current_turn = dialogue_turns[-1]
    for alignment in payload.alignments:
        if alignment.source_update_index not in unresolved:
            raise ValueError("Native alignment references a resolved/unknown update")
        if alignment.evidence_turn_index != current_index:
            raise ValueError("Native alignment must use the current causal turn")
        if alignment.evidence_quote not in current_turn.text:
            raise ValueError("Native alignment quote is absent from current turn")


def deterministic_native_no_op_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    *,
    previous_memory: str,
    committed_source_update_indexes: Sequence[int],
) -> V2GeneratedNativeAlignment:
    committed = _validate_native_alignment_request(
        event,
        dialogue_turns,
        committed_source_update_indexes=committed_source_update_indexes,
    )
    unresolved = set(range(len(event.preference_updates))) - set(committed)
    if unresolved:
        raise ValueError("Unresolved Native updates require alignment review")
    provider_input = render_native_alignment_input(
        event,
        dialogue_turns,
        previous_memory=previous_memory,
        committed_source_update_indexes=committed,
    )
    reason = (
        "This event has no structured vehicle-memory update."
        if not event.preference_updates
        else "All structured updates for this event are already committed."
    )
    payload = V2NativeAlignmentPayload(
        decision="NO_OP",
        reason_code="NO_NEW_VEHICLE_FACT",
        reason=reason,
        alignments=(),
    )
    return V2GeneratedNativeAlignment(
        event_id=event.event_id,
        source_event_sha256=canonical_json_sha256(event.model_dump(mode="json")),
        source_dialogue_prefix_sha256=native_causal_prefix_sha256(dialogue_turns),
        before_memory_sha256=text_sha256(previous_memory),
        event_turn_index=len(dialogue_turns) - 1,
        committed_source_update_indexes=committed,
        payload=payload,
        model_id="deterministic:no-unresolved-update",
        prompt_version=V2_NATIVE_ALIGNMENT_PROMPT_VERSION,
        input_sha256=canonical_json_sha256(json.loads(provider_input)),
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "latency_ms": 0,
        },
    )


def render_native_turn_generation_input(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    prior_event_turns: Sequence[V1DialogueTurnRecord],
    *,
    target_turn_count: int,
    recent_turn_window: int = V2_NATIVE_RECENT_TURN_WINDOW,
) -> str:
    _validate_native_turn_request(
        context,
        prior_event_turns,
        target_turn_count=target_turn_count,
        recent_turn_window=recent_turn_window,
    )
    persona_by_id = {persona.persona_id: persona for persona in personas}
    if set(context.event.participant_ids) - set(persona_by_id):
        raise ValueError("Native event references an unknown persona")
    start = max(0, len(prior_event_turns) - recent_turn_window)
    speaker_counts = Counter(turn.speaker_id for turn in prior_event_turns)
    payload = {
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
        "turn_progress": {
            "event_turn_index": len(prior_event_turns),
            "next_turn_number": len(prior_event_turns) + 1,
            "target_turn_count": target_turn_count,
            "remaining_turn_count_including_next": (
                target_turn_count - len(prior_event_turns)
            ),
            "recent_window_start_index": start,
            "speaker_turn_counts": dict(sorted(speaker_counts.items())),
        },
        "recent_causal_event_prefix": [
            {
                "event_turn_index": index,
                "speaker_id": turn.speaker_id,
                "text": turn.text,
            }
            for index, turn in enumerate(prior_event_turns[start:], start=start)
        ],
    }
    return _canonical_json(payload)


def validate_native_turn_payload(
    context: V1DialogueGenerationContext,
    prior_event_turns: Sequence[V1DialogueTurnRecord],
    payload: V2NativeTurnPayload,
) -> None:
    _validate_prior_event_turns(context, prior_event_turns)
    if payload.speaker_id not in set(context.event.participant_ids):
        raise ValueError("Native dialogue output uses a non-participant speaker")
    if _looks_like_non_human_speaker(payload.text):
        raise ValueError("Native dialogue output appears to contain an agent/tool turn")


def materialize_native_turn(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    generated: V2GeneratedNativeTurn,
    prior_event_turns: Sequence[V1DialogueTurnRecord],
) -> V1DialogueTurnRecord:
    _validate_generated_native_turn(context, generated, prior_event_turns)
    persona_by_id = {persona.persona_id: persona for persona in personas}
    persona = persona_by_id.get(generated.payload.speaker_id)
    if persona is None:
        raise ValueError("Native dialogue speaker is absent from Persona group")
    return V1DialogueTurnRecord(
        turn_id=(
            f"{context.event.event_id}-turn-{generated.event_turn_index + 1:03d}"
        ),
        source_event_id=context.event.event_id,
        timestamp=context.event.timestamp,
        speaker_id=generated.payload.speaker_id,
        speaker_name=persona.name,
        text=generated.payload.text,
    )


def assemble_native_event_dialogue(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    generated_turns: Sequence[V2GeneratedNativeTurn],
    *,
    target_turn_count: int,
) -> V1GeneratedEventDialogue:
    if len(generated_turns) != target_turn_count:
        raise ValueError("Native event must contain exactly its target turn count")
    materialized: list[V1DialogueTurnRecord] = []
    for generated in generated_turns:
        materialized.append(
            materialize_native_turn(
                personas,
                context,
                generated,
                materialized,
            )
        )
    payload = V1DialogueEventPayload(
        turns=tuple(
            V1DialogueLine(speaker_id=turn.speaker_id, text=turn.text)
            for turn in materialized
        )
    )
    validate_dialogue_payload(
        context,
        payload,
        requested_turn_count=target_turn_count,
    )
    model_ids = {turn.model_id for turn in generated_turns}
    prompt_versions = {turn.prompt_version for turn in generated_turns}
    if len(model_ids) != 1 or len(prompt_versions) != 1:
        raise ValueError("Native event turns must use one model and prompt version")
    return V1GeneratedEventDialogue(
        event_id=context.event.event_id,
        source_event_sha256=canonical_json_sha256(
            context.event.model_dump(mode="json")
        ),
        target_turn_count=target_turn_count,
        payload=payload,
        model_id=next(iter(model_ids)),
        prompt_version=next(iter(prompt_versions)),
        input_sha256=canonical_json_sha256(
            [turn.input_sha256 for turn in generated_turns]
        ),
        response_id=None,
        usage=_sum_usage(generated_turns),
    )


def build_native_turn_checkpoint(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    generated_turn: V2GeneratedNativeTurn,
    prior_event_turns: Sequence[V1DialogueTurnRecord],
    *,
    previous_entries: Sequence[V2HybridMemoryEntry],
    committed_source_update_indexes: Sequence[int],
    alignments: Sequence[V2HybridUpdateAlignment],
    global_turn_index: int,
    previous_checkpoint_sha256: str | None,
    generated_alignment: V2GeneratedNativeAlignment | None = None,
    no_op_reason_code: HybridReasonCode = "NO_NEW_VEHICLE_FACT",
    no_op_reason: str = "This turn adds no new structured vehicle-memory fact.",
) -> V2NativeTurnCheckpoint:
    """Commit one Native turn and its deterministic memory transition."""

    if global_turn_index < 0:
        raise ValueError("Native global turn index cannot be negative")
    prior_committed = tuple(sorted(committed_source_update_indexes))
    if len(prior_committed) != len(set(prior_committed)):
        raise ValueError("Native committed update indexes must be unique")
    if any(
        not 0 <= index < len(context.event.preference_updates)
        for index in prior_committed
    ):
        raise ValueError("Native committed update index is outside current event")
    if no_op_reason_code not in {
        "NO_NEW_VEHICLE_FACT",
        "DUPLICATE_ALREADY_STORED",
    }:
        raise ValueError("Native NO_OP requires a NO_OP reason code")
    normalized_no_op_reason = no_op_reason.strip()
    if not normalized_no_op_reason or "\n" in normalized_no_op_reason:
        raise ValueError("Native NO_OP reason must be one non-empty line")

    dialogue_turn = materialize_native_turn(
        personas,
        context,
        generated_turn,
        prior_event_turns,
    )
    normalized_alignments = tuple(alignments)
    if generated_alignment is not None:
        _validate_generated_native_alignment(
            context.event,
            (*prior_event_turns, dialogue_turn),
            generated_alignment,
            previous_memory=render_hybrid_memory(previous_entries),
            committed_source_update_indexes=prior_committed,
        )
        if generated_alignment.payload.alignments != normalized_alignments:
            raise ValueError("Native checkpoint alignments differ from provider output")
        if generated_alignment.payload.decision == "NO_OP":
            no_op_reason_code = generated_alignment.payload.reason_code
            no_op_reason = generated_alignment.payload.reason
    _validate_native_turn_alignments(
        context,
        dialogue_turn,
        normalized_alignments,
        current_event_turn_index=generated_turn.event_turn_index,
        committed_source_update_indexes=prior_committed,
    )

    entries = list(previous_entries)
    before_memory = render_hybrid_memory(entries)
    persona_names = {persona.persona_id: persona.name for persona in personas}
    operations: list[V2HybridPatchOperation] = []
    evidences: list[V2HybridEvidence] = []
    source_indexes: list[int] = []
    reasons: list[str] = []
    updated_existing = False
    for alignment in sorted(
        normalized_alignments,
        key=lambda item: item.source_update_index,
    ):
        update_index = alignment.source_update_index
        update = context.event.preference_updates[update_index]
        update_operations, changed_existing = build_hybrid_update_operations(
            entries,
            context.event,
            update_index,
            update,
            persona_names=persona_names,
        )
        operations.extend(update_operations)
        source_indexes.append(update_index)
        reasons.append(alignment.reason)
        updated_existing = updated_existing or changed_existing
        evidences.append(
            V2HybridEvidence(
                source_event_id=context.event.event_id,
                event_turn_index=generated_turn.event_turn_index,
                turn_id=dialogue_turn.turn_id,
                quote=alignment.evidence_quote,
            )
        )

    if normalized_alignments and not operations:
        raise ValueError("Native aligned update produced no memory change")
    if operations:
        raw_operations = [operation.model_dump(mode="json") for operation in operations]
        after_memory, _ = apply_recursive_summary_patch(
            before_memory,
            raw_operations,
        )
        if after_memory != render_hybrid_memory(entries):
            raise ValueError("Native Patch and structured memory state diverged")
        reason_code: HybridReasonCode = (
            "UPDATED_VEHICLE_MEMORY"
            if updated_existing
            else "NEW_VEHICLE_MEMORY"
        )
        label = V2HybridTurnLabel(
            global_turn_index=global_turn_index,
            event_turn_index=generated_turn.event_turn_index,
            turn_id=dialogue_turn.turn_id,
            source_event_id=context.event.event_id,
            decision="UPDATE",
            reason_code=reason_code,
            reason=" ".join(reasons)[:320],
            source_update_indexes=tuple(source_indexes),
            evidence=tuple(evidences),
            operations=tuple(operations),
            before_memory_sha256=text_sha256(before_memory),
            after_memory_sha256=text_sha256(after_memory),
            after_memory=after_memory,
        )
    else:
        after_memory = before_memory
        label = V2HybridTurnLabel(
            global_turn_index=global_turn_index,
            event_turn_index=generated_turn.event_turn_index,
            turn_id=dialogue_turn.turn_id,
            source_event_id=context.event.event_id,
            decision="NO_OP",
            reason_code=no_op_reason_code,
            reason=normalized_no_op_reason,
            source_update_indexes=(),
            evidence=(),
            operations=(),
            before_memory_sha256=text_sha256(before_memory),
            after_memory_sha256=text_sha256(after_memory),
        )

    committed_after = tuple(sorted((*prior_committed, *source_indexes)))
    body = {
        "schema_version": V2_NATIVE_TURNWISE_SCHEMA_VERSION,
        "event_id": context.event.event_id,
        "source_event_sha256": canonical_json_sha256(
            context.event.model_dump(mode="json")
        ),
        "timeline_index": context.timeline_index,
        "event_turn_index": generated_turn.event_turn_index,
        "global_turn_index": global_turn_index,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "generated_turn": generated_turn.model_dump(mode="json"),
        "generated_alignment": (
            generated_alignment.model_dump(mode="json")
            if generated_alignment is not None
            else None
        ),
        "dialogue_turn": dialogue_turn.model_dump(mode="json"),
        "label": label.model_dump(mode="json"),
        "committed_source_update_indexes": list(committed_after),
        "memory_entries_after": [
            entry.model_dump(mode="json") for entry in entries
        ],
        "after_memory": after_memory,
        "after_memory_sha256": text_sha256(after_memory),
    }
    return V2NativeTurnCheckpoint(
        **body,
        checkpoint_sha256=canonical_json_sha256(body),
    )


def write_native_turn_checkpoint(
    output_dir: Path | str,
    checkpoint: V2NativeTurnCheckpoint,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"turn-{checkpoint.event_turn_index + 1:03d}.json"
    _atomic_write_text(
        path,
        json.dumps(
            checkpoint.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return path


def build_native_event_artifact(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    checkpoints: Sequence[V2NativeTurnCheckpoint],
    *,
    target_turn_count: int,
    initial_memory: str,
    initial_previous_checkpoint_sha256: str | None,
    global_turn_offset: int,
) -> V2NativeEventArtifact:
    if len(checkpoints) != target_turn_count:
        raise ValueError("Native event requires exactly its target checkpoints")
    validate_native_checkpoint_chain(
        checkpoints,
        initial_memory=initial_memory,
        initial_previous_checkpoint_sha256=initial_previous_checkpoint_sha256,
        initial_global_turn_index=global_turn_offset,
    )
    source_hash = canonical_json_sha256(context.event.model_dump(mode="json"))
    if any(
        checkpoint.event_id != context.event.event_id
        or checkpoint.source_event_sha256 != source_hash
        or checkpoint.timeline_index != context.timeline_index
        for checkpoint in checkpoints
    ):
        raise ValueError("Native event checkpoints do not match their source event")
    committed = checkpoints[-1].committed_source_update_indexes
    expected = tuple(range(len(context.event.preference_updates)))
    if committed != expected:
        raise ValueError("Native event ended with unresolved structured updates")
    generated_dialogue = assemble_native_event_dialogue(
        personas,
        context,
        tuple(checkpoint.generated_turn for checkpoint in checkpoints),
        target_turn_count=target_turn_count,
    )
    final = checkpoints[-1]
    alignment_records = tuple(
        checkpoint.generated_alignment
        for checkpoint in checkpoints
        if checkpoint.generated_alignment is not None
    )
    audit = V2NativeEventAudit(
        turn_count=len(checkpoints),
        no_op_count=sum(
            checkpoint.label.decision == "NO_OP" for checkpoint in checkpoints
        ),
        update_count=sum(
            checkpoint.label.decision == "UPDATE" for checkpoint in checkpoints
        ),
        patch_operation_count=sum(
            len(checkpoint.label.operations) for checkpoint in checkpoints
        ),
        committed_update_count=len(committed),
        dialogue_usage=_sum_usage(
            tuple(checkpoint.generated_turn for checkpoint in checkpoints)
        ),
        alignment_usage=_sum_alignment_usage(alignment_records),
        completed=True,
    )
    body = {
        "schema_version": V2_NATIVE_TURNWISE_SCHEMA_VERSION,
        "event_id": context.event.event_id,
        "source_event_sha256": source_hash,
        "timeline_index": context.timeline_index,
        "target_turn_count": target_turn_count,
        "before_memory_sha256": text_sha256(initial_memory),
        "previous_checkpoint_sha256": initial_previous_checkpoint_sha256,
        "turn_checkpoints": [
            checkpoint.model_dump(mode="json") for checkpoint in checkpoints
        ],
        "generated_dialogue": generated_dialogue.model_dump(mode="json"),
        "memory_entries_after": [
            entry.model_dump(mode="json") for entry in final.memory_entries_after
        ],
        "after_memory": final.after_memory,
        "after_memory_sha256": final.after_memory_sha256,
        "audit": audit.model_dump(mode="json"),
    }
    return V2NativeEventArtifact(
        **body,
        artifact_sha256=canonical_json_sha256(body),
    )


def build_native_scenario_artifact(
    stage2: V1Stage2Artifact,
    event_artifacts: Sequence[V2NativeEventArtifact],
) -> V2NativeScenarioArtifact:
    """Validate a causal prefix of Native events and build its scenario artifact."""

    if not event_artifacts:
        raise ValueError("Native scenario requires at least one completed event")
    timeline = stage2.interleaved_timeline
    if len(event_artifacts) > len(timeline):
        raise ValueError("Native scenario contains more events than Stage 2")
    expected_memory_sha256 = text_sha256("")
    expected_previous_checkpoint_sha256: str | None = None
    expected_global_turn_index = 0
    for artifact, timeline_item in zip(event_artifacts, timeline, strict=False):
        if (
            artifact.event_id != timeline_item.event.event_id
            or artifact.timeline_index != timeline_item.timeline_index
            or artifact.source_event_sha256
            != canonical_json_sha256(timeline_item.event.model_dump(mode="json"))
        ):
            raise ValueError("Native scenario event order/source is inconsistent")
        if artifact.target_turn_count != dialogue_turn_target(timeline_item):
            raise ValueError("Native scenario event has the wrong turn target")
        if artifact.before_memory_sha256 != expected_memory_sha256:
            raise ValueError("Native scenario memory hash-chain is broken")
        if artifact.previous_checkpoint_sha256 != (
            expected_previous_checkpoint_sha256
        ):
            raise ValueError("Native scenario checkpoint hash-chain is broken")
        first = artifact.turn_checkpoints[0]
        if first.global_turn_index != expected_global_turn_index:
            raise ValueError("Native scenario global turn indexes are not contiguous")
        expected_memory_sha256 = artifact.after_memory_sha256
        expected_previous_checkpoint_sha256 = artifact.checkpoint_sha256
        expected_global_turn_index += artifact.audit.turn_count

    final = event_artifacts[-1]
    audit = V2NativeScenarioAudit(
        completed_event_count=len(event_artifacts),
        total_event_count=len(timeline),
        turn_count=sum(item.audit.turn_count for item in event_artifacts),
        no_op_count=sum(item.audit.no_op_count for item in event_artifacts),
        update_count=sum(item.audit.update_count for item in event_artifacts),
        patch_operation_count=sum(
            item.audit.patch_operation_count for item in event_artifacts
        ),
        dialogue_usage=_sum_usage_maps(
            tuple(item.audit.dialogue_usage for item in event_artifacts)
        ),
        alignment_usage=_sum_usage_maps(
            tuple(item.audit.alignment_usage for item in event_artifacts)
        ),
        memory_hash_chain_passed=True,
        checkpoint_hash_chain_passed=True,
        completed=len(event_artifacts) == len(timeline),
    )
    body = {
        "schema_version": V2_NATIVE_TURNWISE_SCHEMA_VERSION,
        "source_stage2_sha256": stage2.artifact_sha256,
        "event_artifacts": [
            item.model_dump(mode="json") for item in event_artifacts
        ],
        "final_memory": final.after_memory,
        "final_memory_sha256": final.after_memory_sha256,
        "audit": audit.model_dump(mode="json"),
    }
    return V2NativeScenarioArtifact(
        **body,
        artifact_sha256=canonical_json_sha256(body),
    )


def write_native_scenario_artifact(
    output_dir: Path | str,
    artifact: V2NativeScenarioArtifact,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "native.json"
    _atomic_write_text(
        path,
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return path


def run_native_event(
    personas: Sequence[V1PersonaRecord],
    context: V1DialogueGenerationContext,
    turn_model: V2NativeTurnGenerationModel,
    alignment_model: V2NativeAlignmentModel,
    *,
    output_dir: Path | str,
    target_turn_count: int,
    previous_entries: Sequence[V2HybridMemoryEntry],
    global_turn_offset: int,
    previous_checkpoint_sha256: str | None,
    max_attempts: int = 3,
) -> V2NativeEventArtifact:
    """Resume or generate one complete Native event."""

    if max_attempts < 1:
        raise ValueError("Native event max_attempts must be positive")
    root = Path(output_dir).expanduser().resolve()
    turn_root = root / "turns"
    turn_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(turn_root.glob("turn-*.json"))
    checkpoints = [
        V2NativeTurnCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    if len(checkpoints) > target_turn_count:
        raise ValueError("Native resume has more turns than the event target")
    initial_memory = render_hybrid_memory(previous_entries)
    validate_native_checkpoint_chain(
        checkpoints,
        initial_memory=initial_memory,
        initial_previous_checkpoint_sha256=previous_checkpoint_sha256,
        initial_global_turn_index=global_turn_offset,
    )
    source_hash = canonical_json_sha256(context.event.model_dump(mode="json"))
    if any(
        checkpoint.event_id != context.event.event_id
        or checkpoint.source_event_sha256 != source_hash
        or checkpoint.timeline_index != context.timeline_index
        for checkpoint in checkpoints
    ):
        raise ValueError("Native resume checkpoint source does not match event")

    prior_turns = [checkpoint.dialogue_turn for checkpoint in checkpoints]
    entries = (
        list(checkpoints[-1].memory_entries_after)
        if checkpoints
        else list(previous_entries)
    )
    committed = (
        checkpoints[-1].committed_source_update_indexes if checkpoints else ()
    )
    prior_checkpoint = (
        checkpoints[-1].checkpoint_sha256
        if checkpoints
        else previous_checkpoint_sha256
    )
    expected_updates = tuple(range(len(context.event.preference_updates)))
    for event_turn_index in range(len(checkpoints), target_turn_count):
        last_error: Exception | None = None
        for _attempt in range(1, max_attempts + 1):
            try:
                generated_turn = turn_model.generate(
                    personas,
                    context,
                    tuple(prior_turns),
                    target_turn_count=target_turn_count,
                )
                dialogue_turn = materialize_native_turn(
                    personas,
                    context,
                    generated_turn,
                    tuple(prior_turns),
                )
                causal_prefix = (*prior_turns, dialogue_turn)
                unresolved = set(expected_updates) - set(committed)
                if unresolved:
                    generated_alignment = alignment_model.generate(
                        context.event,
                        causal_prefix,
                        previous_memory=render_hybrid_memory(entries),
                        committed_source_update_indexes=committed,
                    )
                else:
                    generated_alignment = deterministic_native_no_op_alignment(
                        context.event,
                        causal_prefix,
                        previous_memory=render_hybrid_memory(entries),
                        committed_source_update_indexes=committed,
                    )
                checkpoint = build_native_turn_checkpoint(
                    personas,
                    context,
                    generated_turn,
                    tuple(prior_turns),
                    previous_entries=entries,
                    committed_source_update_indexes=committed,
                    alignments=generated_alignment.payload.alignments,
                    global_turn_index=global_turn_offset + event_turn_index,
                    previous_checkpoint_sha256=prior_checkpoint,
                    generated_alignment=generated_alignment,
                )
                if (
                    event_turn_index == target_turn_count - 1
                    and checkpoint.committed_source_update_indexes
                    != expected_updates
                ):
                    raise ValueError(
                        "Native final turn left structured updates unresolved"
                    )
                write_native_turn_checkpoint(turn_root, checkpoint)
                checkpoints.append(checkpoint)
                prior_turns.append(checkpoint.dialogue_turn)
                entries = list(checkpoint.memory_entries_after)
                committed = checkpoint.committed_source_update_indexes
                prior_checkpoint = checkpoint.checkpoint_sha256
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Native event turn {event_turn_index} failed after "
                f"{max_attempts} attempts"
            ) from last_error

    artifact = build_native_event_artifact(
        personas,
        context,
        checkpoints,
        target_turn_count=target_turn_count,
        initial_memory=initial_memory,
        initial_previous_checkpoint_sha256=previous_checkpoint_sha256,
        global_turn_offset=global_turn_offset,
    )
    _atomic_write_text(
        root / "event.json",
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return artifact


def run_native_scenario(
    stage2: V1Stage2Artifact,
    turn_model: V2NativeTurnGenerationModel,
    alignment_model: V2NativeAlignmentModel,
    *,
    output_dir: Path | str,
    event_limit: int | None = None,
    max_attempts: int = 3,
) -> V2NativeScenarioArtifact:
    """Resume or generate one causal Stage 2 scenario event by event."""

    contexts = build_dialogue_generation_contexts(stage2)
    if not contexts:
        raise ValueError("Native scenario Stage 2 timeline is empty")
    if event_limit is not None and not 1 <= event_limit <= len(contexts):
        raise ValueError("Native scenario event limit is outside the timeline")
    selected = contexts[:event_limit] if event_limit is not None else contexts
    root = Path(output_dir).expanduser().resolve()
    event_root = root / "events"
    dialogue_root = root / "dialogues"
    event_root.mkdir(parents=True, exist_ok=True)
    dialogue_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[V2NativeEventArtifact] = []
    previous_entries: tuple[V2HybridMemoryEntry, ...] = ()
    previous_checkpoint_sha256: str | None = None
    global_turn_offset = 0
    for context in selected:
        target_turn_count = dialogue_turn_target(context)
        artifact = run_native_event(
            stage2.persona_group.payload.personas,
            context,
            turn_model,
            alignment_model,
            output_dir=(
                event_root
                / f"{context.timeline_index:03d}-{context.event.event_id}"
            ),
            target_turn_count=target_turn_count,
            previous_entries=previous_entries,
            global_turn_offset=global_turn_offset,
            previous_checkpoint_sha256=previous_checkpoint_sha256,
            max_attempts=max_attempts,
        )
        artifacts.append(artifact)
        _atomic_write_text(
            dialogue_root / f"{artifact.event_id}.json",
            json.dumps(
                artifact.generated_dialogue.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        previous_entries = artifact.memory_entries_after
        previous_checkpoint_sha256 = artifact.checkpoint_sha256
        global_turn_offset += artifact.audit.turn_count
        _write_native_progress(root, stage2, artifacts)
    scenario = build_native_scenario_artifact(stage2, artifacts)
    write_native_scenario_artifact(root, scenario)
    return scenario


def native_causal_prefix_sha256(
    prior_event_turns: Sequence[V1DialogueTurnRecord],
) -> str:
    return canonical_json_sha256(
        [turn.model_dump(mode="json") for turn in prior_event_turns]
    )


def _validate_native_turn_alignments(
    context: V1DialogueGenerationContext,
    dialogue_turn: V1DialogueTurnRecord,
    alignments: Sequence[V2HybridUpdateAlignment],
    *,
    current_event_turn_index: int,
    committed_source_update_indexes: Sequence[int],
) -> None:
    if context.chain_kind == "background" and alignments:
        raise ValueError("Native background event cannot commit a vehicle update")
    indexes = [alignment.source_update_index for alignment in alignments]
    if len(indexes) != len(set(indexes)):
        raise ValueError("Native turn alignments must use unique update indexes")
    committed = set(committed_source_update_indexes)
    for alignment in alignments:
        if not 0 <= alignment.source_update_index < len(
            context.event.preference_updates
        ):
            raise ValueError("Native alignment update index is outside current event")
        if alignment.source_update_index in committed:
            raise ValueError("Native alignment repeats an already committed update")
        if alignment.evidence_turn_index != current_event_turn_index:
            raise ValueError("Native alignment must point to the current turn")
        if alignment.evidence_quote not in dialogue_turn.text:
            raise ValueError("Native evidence quote is absent from the current turn")


def _validate_native_alignment_request(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    *,
    committed_source_update_indexes: Sequence[int],
) -> tuple[int, ...]:
    if not dialogue_turns:
        raise ValueError("Native alignment requires a causal dialogue prefix")
    if any(turn.source_event_id != event.event_id for turn in dialogue_turns):
        raise ValueError("Native alignment prefix contains another event")
    committed = tuple(sorted(committed_source_update_indexes))
    if len(committed) != len(set(committed)):
        raise ValueError("Native committed update indexes must be unique")
    if any(not 0 <= index < len(event.preference_updates) for index in committed):
        raise ValueError("Native committed update index is outside current event")
    return committed


def _validate_generated_native_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    generated: V2GeneratedNativeAlignment,
    *,
    previous_memory: str,
    committed_source_update_indexes: Sequence[int],
) -> None:
    committed = _validate_native_alignment_request(
        event,
        dialogue_turns,
        committed_source_update_indexes=committed_source_update_indexes,
    )
    if generated.event_id != event.event_id:
        raise ValueError("Native generated alignment changed event_id")
    if generated.source_event_sha256 != canonical_json_sha256(
        event.model_dump(mode="json")
    ):
        raise ValueError("Native generated alignment source event hash changed")
    if generated.source_dialogue_prefix_sha256 != native_causal_prefix_sha256(
        dialogue_turns
    ):
        raise ValueError("Native generated alignment causal-prefix hash changed")
    if generated.before_memory_sha256 != text_sha256(previous_memory):
        raise ValueError("Native generated alignment before-memory hash changed")
    if generated.event_turn_index != len(dialogue_turns) - 1:
        raise ValueError("Native generated alignment turn index changed")
    if generated.committed_source_update_indexes != committed:
        raise ValueError("Native generated alignment committed indexes changed")
    validate_native_alignment_payload(
        event,
        dialogue_turns,
        generated.payload,
        committed_source_update_indexes=committed,
    )


def validate_native_checkpoint_chain(
    checkpoints: Sequence[V2NativeTurnCheckpoint],
    *,
    initial_memory: str = "",
    initial_previous_checkpoint_sha256: str | None = None,
    initial_global_turn_index: int = 0,
) -> None:
    previous_memory_sha256 = text_sha256(initial_memory)
    previous_checkpoint_sha256 = initial_previous_checkpoint_sha256
    expected_global_turn_index = initial_global_turn_index
    current_event_id: str | None = None
    current_timeline_index = -1
    event_turns: list[V1DialogueTurnRecord] = []
    committed_indexes: tuple[int, ...] = ()
    for checkpoint in checkpoints:
        if checkpoint.global_turn_index != expected_global_turn_index:
            raise ValueError("Native checkpoint global turn indexes are not contiguous")
        if checkpoint.previous_checkpoint_sha256 != previous_checkpoint_sha256:
            raise ValueError("Native checkpoint hash chain is broken")
        if checkpoint.label.before_memory_sha256 != previous_memory_sha256:
            raise ValueError("Native checkpoint memory hash chain is broken")
        if checkpoint.event_id != current_event_id:
            if checkpoint.timeline_index <= current_timeline_index:
                raise ValueError("Native checkpoint event timeline is not increasing")
            if checkpoint.event_turn_index != 0:
                raise ValueError("Native checkpoint event must start at turn zero")
            current_event_id = checkpoint.event_id
            current_timeline_index = checkpoint.timeline_index
            event_turns = []
            committed_indexes = ()
        elif checkpoint.timeline_index != current_timeline_index:
            raise ValueError("Native checkpoint event changed timeline index")
        if checkpoint.event_turn_index != len(event_turns):
            raise ValueError("Native checkpoint event turn indexes are not contiguous")
        if checkpoint.generated_turn.causal_prefix_sha256 != (
            native_causal_prefix_sha256(event_turns)
        ):
            raise ValueError("Native checkpoint causal dialogue hash chain is broken")
        if not set(committed_indexes).issubset(
            checkpoint.committed_source_update_indexes
        ):
            raise ValueError("Native checkpoint lost a committed source update")
        event_turns.append(checkpoint.dialogue_turn)
        committed_indexes = checkpoint.committed_source_update_indexes
        previous_memory_sha256 = checkpoint.after_memory_sha256
        previous_checkpoint_sha256 = checkpoint.checkpoint_sha256
        expected_global_turn_index += 1


def _validate_native_turn_request(
    context: V1DialogueGenerationContext,
    prior_event_turns: Sequence[V1DialogueTurnRecord],
    *,
    target_turn_count: int,
    recent_turn_window: int,
) -> None:
    if target_turn_count < 1:
        raise ValueError("Native target turn count must be positive")
    if recent_turn_window < 1:
        raise ValueError("Native recent-turn window must be positive")
    _validate_prior_event_turns(context, prior_event_turns)
    if len(prior_event_turns) >= target_turn_count:
        raise ValueError("Native event already reached its target turn count")


def _validate_prior_event_turns(
    context: V1DialogueGenerationContext,
    prior_event_turns: Sequence[V1DialogueTurnRecord],
) -> None:
    for index, turn in enumerate(prior_event_turns):
        if turn.source_event_id != context.event.event_id:
            raise ValueError("Native causal prefix contains another event")
        expected_id = f"{context.event.event_id}-turn-{index + 1:03d}"
        if turn.turn_id != expected_id:
            raise ValueError("Native causal prefix turn IDs are not contiguous")
        if turn.timestamp != context.event.timestamp:
            raise ValueError("Native causal prefix timestamp changed")
        if turn.speaker_id not in set(context.event.participant_ids):
            raise ValueError("Native causal prefix uses a non-participant speaker")


def _validate_generated_native_turn(
    context: V1DialogueGenerationContext,
    generated: V2GeneratedNativeTurn,
    prior_event_turns: Sequence[V1DialogueTurnRecord],
) -> None:
    validate_native_turn_payload(context, prior_event_turns, generated.payload)
    if generated.event_id != context.event.event_id:
        raise ValueError("Native generated turn changed event_id")
    if generated.source_event_sha256 != canonical_json_sha256(
        context.event.model_dump(mode="json")
    ):
        raise ValueError("Native generated turn source event hash changed")
    if generated.timeline_index != context.timeline_index:
        raise ValueError("Native generated turn timeline index changed")
    if generated.event_turn_index != len(prior_event_turns):
        raise ValueError("Native generated turn is not the next causal turn")
    if generated.causal_prefix_sha256 != native_causal_prefix_sha256(
        prior_event_turns
    ):
        raise ValueError("Native generated turn causal-prefix hash changed")


def _looks_like_non_human_speaker(text: str) -> bool:
    lowered = text.strip().casefold()
    return lowered.startswith(("assistant:", "system:", "vehicle:", "tool:"))


def _sum_usage(turns: Sequence[V2GeneratedNativeTurn]) -> dict[str, int]:
    keys = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "latency_ms",
    }
    return {key: sum(int(turn.usage.get(key, 0)) for turn in turns) for key in keys}


def _sum_alignment_usage(
    alignments: Sequence[V2GeneratedNativeAlignment],
) -> dict[str, int]:
    keys = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "latency_ms",
    }
    return {
        key: sum(int(alignment.usage.get(key, 0)) for alignment in alignments)
        for key in keys
    }


def _sum_usage_maps(usages: Sequence[dict[str, int]]) -> dict[str, int]:
    keys = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "latency_ms",
    }
    return {key: sum(int(usage.get(key, 0)) for usage in usages) for key in keys}


def _write_native_progress(
    root: Path,
    stage2: V1Stage2Artifact,
    artifacts: Sequence[V2NativeEventArtifact],
) -> None:
    final = artifacts[-1]
    body = {
        "schema_version": V2_NATIVE_TURNWISE_SCHEMA_VERSION,
        "source_stage2_sha256": stage2.artifact_sha256,
        "completed_event_count": len(artifacts),
        "total_event_count": len(stage2.interleaved_timeline),
        "completed_event_artifact_sha256": [
            item.artifact_sha256 for item in artifacts
        ],
        "last_event_id": final.event_id,
        "last_checkpoint_sha256": final.checkpoint_sha256,
        "final_memory_sha256": final.after_memory_sha256,
    }
    payload = {**body, "progress_sha256": canonical_json_sha256(body)}
    _atomic_write_text(
        root / "progress.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
