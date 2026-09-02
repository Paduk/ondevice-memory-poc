"""Event-batched dialogue with causally committed turn-wise V2 memory labels."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventRecord,
    V1PersonaRecord,
    V1PreferenceUpdate,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    V1GeneratedEventDialogue,
)

V2_HYBRID_SCHEMA_VERSION = "vehiclemembench-v2-hybrid-schema-v1"
V2_HYBRID_ALIGNMENT_PROMPT_VERSION = (
    "vehiclemembench-v2-hybrid-event-alignment-terra-v4"
)

HybridDecision = Literal["NO_OP", "UPDATE"]
HybridReasonProfile = Literal["legacy", "state_evolution"]
HybridReasonCode = Literal[
    "NO_NEW_VEHICLE_FACT",
    "BACKGROUND",
    "NOT_CONFIRMED_YET",
    "NO_NEW_FACT",
    "NEW_VEHICLE_MEMORY",
    "UPDATED_VEHICLE_MEMORY",
    "DUPLICATE_ALREADY_STORED",
]
HybridPatchOp = Literal["add", "replace", "delete"]

V2_HYBRID_ALIGNMENT_INSTRUCTIONS = """
Align each supplied structured preference update to the earliest dialogue turn
where that update is explicitly and completely supported.

Rules:
- Return exactly one alignment for every source_update_index and no others.
- evidence_turn_index is zero-based within current_event_dialogue. Select the
  earliest turn after whose text, together with its preceding event prefix,
  completely supports the update.
- evidence_quote must be an exact, minimal, non-empty substring of that turn.
- The quote must support the update's subject, setting/value, and any condition
  needed to distinguish it. Do not use later turns to justify an earlier index.
- If another speaker proposes or recalls a value and the source subject then
  confirms or corrects it, use that first confirmation/correction turn. Do not
  delay the boundary to a later recap, acknowledgement, or statement that the
  preference was recorded.
- A turn that only says a setting was tried, tested, selected, or enabled is
  not yet a preference confirmation. When the dialogue evaluates the trial,
  wait for the first turn that accepts the result, reports that it works, or
  otherwise commits to it. Do not wait beyond that first acceptance.
- Do not create memory text or infer facts absent from the structured update.
- Give one short single-line reason for each alignment.
""".strip()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2HybridUpdateAlignment(_StrictModel):
    source_update_index: int = Field(ge=0)
    evidence_turn_index: int = Field(ge=0)
    evidence_quote: str = Field(min_length=1, max_length=1_024)
    reason: str = Field(min_length=1, max_length=320)

    @field_validator("evidence_quote", "reason")
    @classmethod
    def normalize_single_line(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Hybrid alignment text must be one non-empty line")
        return normalized


class V2HybridEventAlignmentPayload(_StrictModel):
    alignments: tuple[V2HybridUpdateAlignment, ...] = Field(max_length=32)


class V2GeneratedHybridEventAlignment(_StrictModel):
    event_id: str = Field(min_length=1)
    source_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dialogue_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    before_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: V2HybridEventAlignmentPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]
    normalization_notes: tuple[str, ...] = ()


class V2HybridMemoryEntry(_StrictModel):
    identity_key: str = Field(min_length=1)
    memory_line: str = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    source_update_index: int = Field(ge=0)


class V2HybridEvidence(_StrictModel):
    source_event_id: str = Field(min_length=1)
    event_turn_index: int = Field(ge=0)
    turn_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=1_024)


class V2HybridPatchOperation(_StrictModel):
    op: HybridPatchOp
    target: str
    content: str


class V2HybridTurnLabel(_StrictModel):
    global_turn_index: int = Field(ge=0)
    event_turn_index: int = Field(ge=0)
    turn_id: str = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    decision: HybridDecision
    reason_code: HybridReasonCode
    reason: str = Field(min_length=1, max_length=320)
    source_update_indexes: tuple[int, ...]
    evidence: tuple[V2HybridEvidence, ...]
    operations: tuple[V2HybridPatchOperation, ...]
    before_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_memory: str | None = None

    @model_validator(mode="after")
    def validate_decision_shape(self) -> V2HybridTurnLabel:
        if self.decision == "NO_OP":
            if self.reason_code in {
                "NEW_VEHICLE_MEMORY",
                "UPDATED_VEHICLE_MEMORY",
            }:
                raise ValueError("Hybrid NO_OP requires a NO_OP reason code")
            if self.source_update_indexes or self.evidence or self.operations:
                raise ValueError("Hybrid NO_OP cannot contain an update payload")
            if self.after_memory is not None:
                raise ValueError("Hybrid NO_OP stores only the unchanged memory hash")
            if self.before_memory_sha256 != self.after_memory_sha256:
                raise ValueError("Hybrid NO_OP must preserve the memory hash")
            return self
        if self.reason_code not in {
            "NEW_VEHICLE_MEMORY",
            "UPDATED_VEHICLE_MEMORY",
        }:
            raise ValueError("Hybrid UPDATE requires an UPDATE reason code")
        if not self.source_update_indexes or not self.evidence or not self.operations:
            raise ValueError("Hybrid UPDATE requires source, evidence, and Patch")
        if self.after_memory is None:
            raise ValueError("Hybrid UPDATE requires the resulting memory")
        if text_sha256(self.after_memory) != self.after_memory_sha256:
            raise ValueError("Hybrid UPDATE resulting memory hash is invalid")
        return self


class V2HybridEventCheckpoint(_StrictModel):
    event_id: str = Field(min_length=1)
    timeline_index: int = Field(ge=0)
    source_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dialogue_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    before_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_alignment: V2GeneratedHybridEventAlignment
    turn_labels: tuple[V2HybridTurnLabel, ...] = Field(min_length=1)
    memory_entries_after: tuple[V2HybridMemoryEntry, ...]
    after_memory: str
    after_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hashes(self) -> V2HybridEventCheckpoint:
        if text_sha256(self.after_memory) != self.after_memory_sha256:
            raise ValueError("Hybrid event after-memory hash is invalid")
        rendered = render_hybrid_memory(self.memory_entries_after)
        if rendered != self.after_memory:
            raise ValueError("Hybrid event entries do not reproduce after-memory")
        body = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if canonical_json_sha256(body) != self.checkpoint_sha256:
            raise ValueError("Hybrid event checkpoint hash is invalid")
        return self


class V2HybridAudit(_StrictModel):
    completed_event_count: int = Field(ge=0)
    total_event_count: int = Field(ge=1)
    turn_label_count: int = Field(ge=0)
    no_op_count: int = Field(ge=0)
    update_count: int = Field(ge=0)
    patch_operation_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    prefix_evidence_passed: bool
    memory_hash_chain_passed: bool
    completed: bool


class V2HybridArtifact(_StrictModel):
    schema_version: str = V2_HYBRID_SCHEMA_VERSION
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_checkpoints: tuple[V2HybridEventCheckpoint, ...]
    final_memory: str
    final_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit: V2HybridAudit
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hashes(self) -> V2HybridArtifact:
        if text_sha256(self.final_memory) != self.final_memory_sha256:
            raise ValueError("Hybrid artifact final-memory hash is invalid")
        body = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(body) != self.artifact_sha256:
            raise ValueError("Hybrid artifact hash is invalid")
        return self


class V2HybridEventAlignmentModel(Protocol):
    def generate(
        self,
        event: V1EventRecord,
        dialogue_turns: Sequence[V1DialogueTurnRecord],
        *,
        previous_memory: str,
    ) -> V2GeneratedHybridEventAlignment: ...


class OpenAIV2HybridEventAlignmentModel:
    """Locate the earliest evidence turn for known structured updates."""

    def __init__(
        self,
        model_id: str = V1_DEFAULT_GENERATION_MODEL,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 4_096,
        additional_instructions: str | None = None,
        prompt_version: str = V2_HYBRID_ALIGNMENT_PROMPT_VERSION,
        normalize_earlier_confirmation: bool = True,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Hybrid alignment model ID is required")
        if not prompt_version.strip():
            raise ValueError("Hybrid alignment prompt version is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.additional_instructions = (additional_instructions or "").strip()
        self.prompt_version = prompt_version
        self.normalize_earlier_confirmation = normalize_earlier_confirmation
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
    ) -> V2GeneratedHybridEventAlignment:
        provider_input = render_hybrid_alignment_input(
            event,
            dialogue_turns,
            previous_memory=previous_memory,
        )
        instructions = V2_HYBRID_ALIGNMENT_INSTRUCTIONS
        if self.additional_instructions:
            instructions += "\n\n" + self.additional_instructions
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": instructions,
            "input": provider_input,
            "text_format": V2HybridEventAlignmentPayload,
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
            raise RuntimeError(f"V2 Hybrid alignment provider returned {status}")
        raw_payload = V2HybridEventAlignmentPayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        if self.normalize_earlier_confirmation:
            payload, normalization_notes = normalize_hybrid_alignment(
                event,
                dialogue_turns,
                raw_payload,
            )
        else:
            payload, normalization_notes = raw_payload, ()
        validate_hybrid_alignment(event, dialogue_turns, payload)
        return V2GeneratedHybridEventAlignment(
            event_id=event.event_id,
            source_event_sha256=event_sha256(event),
            source_dialogue_sha256=dialogue_sha256(dialogue_turns),
            before_memory_sha256=text_sha256(previous_memory),
            payload=payload,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
            normalization_notes=normalization_notes,
        )


def render_hybrid_alignment_input(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    *,
    previous_memory: str,
) -> str:
    _validate_dialogue_belongs_to_event(event, dialogue_turns)
    payload = {
        "current_approved_memory": previous_memory or "(empty)",
        "structured_preference_updates": [
            {"source_update_index": index, **update.model_dump(mode="json")}
            for index, update in enumerate(event.preference_updates)
        ],
        "current_event_dialogue": [
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


def validate_hybrid_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    payload: V2HybridEventAlignmentPayload,
) -> None:
    _validate_dialogue_belongs_to_event(event, dialogue_turns)
    indexes = [item.source_update_index for item in payload.alignments]
    expected = list(range(len(event.preference_updates)))
    if sorted(indexes) != expected or len(indexes) != len(set(indexes)):
        raise ValueError("Hybrid alignment must cover every source update exactly once")
    for alignment in payload.alignments:
        if alignment.evidence_turn_index >= len(dialogue_turns):
            raise ValueError("Hybrid evidence turn is outside the current event")
        evidence_turn = dialogue_turns[alignment.evidence_turn_index]
        if alignment.evidence_quote not in evidence_turn.text:
            raise ValueError("Hybrid evidence quote is absent from its prefix turn")


def normalize_hybrid_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    payload: V2HybridEventAlignmentPayload,
) -> tuple[V2HybridEventAlignmentPayload, tuple[str, ...]]:
    """Move an alignment to an earlier unambiguous subject confirmation."""

    validate_hybrid_alignment(event, dialogue_turns, payload)
    normalized = []
    notes = []
    for alignment in payload.alignments:
        update = event.preference_updates[alignment.source_update_index]
        earlier = _earliest_direct_subject_confirmation(
            update,
            dialogue_turns,
            before_turn=alignment.evidence_turn_index,
        )
        if earlier is None:
            normalized.append(alignment)
            continue
        turn = dialogue_turns[earlier]
        normalized.append(
            alignment.model_copy(
                update={
                    "evidence_turn_index": earlier,
                    "evidence_quote": turn.text,
                    "reason": (
                        "The source subject first directly confirms the new "
                        "vehicle setting at this turn."
                    ),
                }
            )
        )
        notes.append(
            f"source_update_index={alignment.source_update_index}:"
            f"model_turn={alignment.evidence_turn_index}->subject_turn={earlier}"
        )
    result = V2HybridEventAlignmentPayload(alignments=tuple(normalized))
    validate_hybrid_alignment(event, dialogue_turns, result)
    return result, tuple(notes)


def deterministic_empty_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    *,
    previous_memory: str,
) -> V2GeneratedHybridEventAlignment:
    if event.preference_updates:
        raise ValueError("Only events without preference updates can skip alignment")
    _validate_dialogue_belongs_to_event(event, dialogue_turns)
    provider_input = render_hybrid_alignment_input(
        event,
        dialogue_turns,
        previous_memory=previous_memory,
    )
    return V2GeneratedHybridEventAlignment(
        event_id=event.event_id,
        source_event_sha256=event_sha256(event),
        source_dialogue_sha256=dialogue_sha256(dialogue_turns),
        before_memory_sha256=text_sha256(previous_memory),
        payload=V2HybridEventAlignmentPayload(alignments=()),
        model_id="deterministic:no-preference-update",
        prompt_version=V2_HYBRID_ALIGNMENT_PROMPT_VERSION,
        input_sha256=canonical_json_sha256(json.loads(provider_input)),
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "latency_ms": 0,
        },
    )


def build_hybrid_event_checkpoint(
    *,
    event: V1EventRecord,
    timeline_index: int,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    generated_alignment: V2GeneratedHybridEventAlignment,
    previous_entries: Sequence[V2HybridMemoryEntry],
    personas: Sequence[V1PersonaRecord],
    global_turn_offset: int,
    chain_kind: Literal["background", "vehicle"] | None = None,
    reason_profile: HybridReasonProfile = "legacy",
    temporal_transitions_by_update: Mapping[int, Mapping[str, Any]] | None = None,
) -> V2HybridEventCheckpoint:
    # Import lazily because providers imports the vehicle_bench package while it
    # registers its ontology-backed implementations.
    from palmclaw_ubuntu.providers import apply_recursive_summary_patch

    previous_memory = render_hybrid_memory(previous_entries)
    _validate_generated_alignment(
        event,
        dialogue_turns,
        generated_alignment,
        previous_memory=previous_memory,
    )
    persona_names = {persona.persona_id: persona.name for persona in personas}
    alignments_by_turn: dict[int, list[V2HybridUpdateAlignment]] = {}
    for alignment in generated_alignment.payload.alignments:
        alignments_by_turn.setdefault(alignment.evidence_turn_index, []).append(
            alignment
        )

    entries = list(previous_entries)
    memory = previous_memory
    labels: list[V2HybridTurnLabel] = []
    for event_turn_index, turn in enumerate(dialogue_turns):
        before = memory
        operations: list[V2HybridPatchOperation] = []
        evidences: list[V2HybridEvidence] = []
        source_indexes: list[int] = []
        reasons: list[str] = []
        updated_existing = False
        for alignment in sorted(
            alignments_by_turn.get(event_turn_index, ()),
            key=lambda item: item.source_update_index,
        ):
            update = event.preference_updates[alignment.source_update_index]
            transition = (temporal_transitions_by_update or {}).get(
                alignment.source_update_index
            )
            if transition is not None and transition.get("temporal_action") == (
                "end_temporary"
            ):
                update_ops, changed_existing = (
                    build_hybrid_end_temporary_operations(entries, update)
                )
            else:
                update_ops, changed_existing = build_hybrid_update_operations(
                    entries,
                    event,
                    alignment.source_update_index,
                    update,
                    persona_names=persona_names,
                )
            operations.extend(update_ops)
            source_indexes.append(alignment.source_update_index)
            reasons.append(alignment.reason)
            updated_existing = updated_existing or changed_existing
            evidences.append(
                V2HybridEvidence(
                    source_event_id=event.event_id,
                    event_turn_index=event_turn_index,
                    turn_id=turn.turn_id,
                    quote=alignment.evidence_quote,
                )
            )

        if operations:
            raw_operations = [item.model_dump(mode="json") for item in operations]
            memory, _ = apply_recursive_summary_patch(memory, raw_operations)
            rendered = render_hybrid_memory(entries)
            if memory != rendered:
                raise ValueError("Hybrid Patch result and structured state diverged")
            reason_code: HybridReasonCode = (
                "UPDATED_VEHICLE_MEMORY"
                if updated_existing
                else "NEW_VEHICLE_MEMORY"
            )
            label = V2HybridTurnLabel(
                global_turn_index=global_turn_offset + event_turn_index,
                event_turn_index=event_turn_index,
                turn_id=turn.turn_id,
                source_event_id=event.event_id,
                decision="UPDATE",
                reason_code=reason_code,
                reason=" ".join(reasons)[:320],
                source_update_indexes=tuple(source_indexes),
                evidence=tuple(evidences),
                operations=tuple(operations),
                before_memory_sha256=text_sha256(before),
                after_memory_sha256=text_sha256(memory),
                after_memory=memory,
            )
        else:
            if reason_profile == "state_evolution":
                reason_code, reason = _state_evolution_no_op_reason(
                    source_indexes=source_indexes,
                    chain_kind=chain_kind,
                    has_pending_alignment=any(
                        index > event_turn_index for index in alignments_by_turn
                    ),
                )
            else:
                reason_code = (
                    "DUPLICATE_ALREADY_STORED"
                    if source_indexes
                    else "NO_NEW_VEHICLE_FACT"
                )
                reason = (
                    "The aligned preference is already present in memory."
                    if source_indexes
                    else "This turn adds no new structured vehicle-memory fact."
                )
            label = V2HybridTurnLabel(
                global_turn_index=global_turn_offset + event_turn_index,
                event_turn_index=event_turn_index,
                turn_id=turn.turn_id,
                source_event_id=event.event_id,
                decision="NO_OP",
                reason_code=reason_code,
                reason=reason,
                source_update_indexes=(),
                evidence=(),
                operations=(),
                before_memory_sha256=text_sha256(before),
                after_memory_sha256=text_sha256(memory),
            )
        labels.append(label)

    body = {
        "event_id": event.event_id,
        "timeline_index": timeline_index,
        "source_event_sha256": event_sha256(event),
        "source_dialogue_sha256": dialogue_sha256(dialogue_turns),
        "before_memory_sha256": text_sha256(previous_memory),
        "generated_alignment": generated_alignment.model_dump(mode="json"),
        "turn_labels": [label.model_dump(mode="json") for label in labels],
        "memory_entries_after": [entry.model_dump(mode="json") for entry in entries],
        "after_memory": memory,
        "after_memory_sha256": text_sha256(memory),
    }
    return V2HybridEventCheckpoint(
        **body,
        checkpoint_sha256=canonical_json_sha256(body),
    )


def _state_evolution_no_op_reason(
    *,
    source_indexes: Sequence[int],
    chain_kind: Literal["background", "vehicle"] | None,
    has_pending_alignment: bool,
) -> tuple[HybridReasonCode, str]:
    """Produce prefix-safe deterministic S21+ NO_OP supervision."""

    if source_indexes:
        return (
            "DUPLICATE_ALREADY_STORED",
            "The supported vehicle fact is already represented in memory.",
        )
    if chain_kind == "background":
        return (
            "BACKGROUND",
            "This background turn contains no vehicle-memory fact.",
        )
    if has_pending_alignment:
        return (
            "NOT_CONFIRMED_YET",
            "The current prefix does not yet confirm a durable vehicle-memory fact.",
        )
    return (
        "NO_NEW_FACT",
        "This turn adds no new durable vehicle-memory fact.",
    )


def build_hybrid_artifact(
    stage2: V1Stage2Artifact,
    checkpoints: Sequence[V2HybridEventCheckpoint],
) -> V2HybridArtifact:
    if len(checkpoints) > len(stage2.interleaved_timeline):
        raise ValueError("Hybrid checkpoints exceed the Stage 2 timeline")
    previous_hash = text_sha256("")
    previous_global_turn = -1
    for index, checkpoint in enumerate(checkpoints):
        source = stage2.interleaved_timeline[index]
        if checkpoint.timeline_index != source.timeline_index:
            raise ValueError("Hybrid checkpoints must form a timeline prefix")
        if checkpoint.event_id != source.event.event_id:
            raise ValueError("Hybrid checkpoint event order differs from Stage 2")
        if checkpoint.before_memory_sha256 != previous_hash:
            raise ValueError("Hybrid checkpoint memory hash chain is broken")
        for label in checkpoint.turn_labels:
            if label.global_turn_index != previous_global_turn + 1:
                raise ValueError("Hybrid global turn indexes must be contiguous")
            previous_global_turn = label.global_turn_index
        previous_hash = checkpoint.after_memory_sha256
    final_memory = checkpoints[-1].after_memory if checkpoints else ""
    labels = [label for checkpoint in checkpoints for label in checkpoint.turn_labels]
    audit = V2HybridAudit(
        completed_event_count=len(checkpoints),
        total_event_count=len(stage2.interleaved_timeline),
        turn_label_count=len(labels),
        no_op_count=sum(label.decision == "NO_OP" for label in labels),
        update_count=sum(label.decision == "UPDATE" for label in labels),
        patch_operation_count=sum(len(label.operations) for label in labels),
        evidence_count=sum(len(label.evidence) for label in labels),
        prefix_evidence_passed=True,
        memory_hash_chain_passed=True,
        completed=len(checkpoints) == len(stage2.interleaved_timeline),
    )
    body = {
        "schema_version": V2_HYBRID_SCHEMA_VERSION,
        "source_stage2_sha256": stage2.artifact_sha256,
        "event_checkpoints": [
            checkpoint.model_dump(mode="json") for checkpoint in checkpoints
        ],
        "final_memory": final_memory,
        "final_memory_sha256": text_sha256(final_memory),
        "audit": audit.model_dump(mode="json"),
    }
    return V2HybridArtifact(**body, artifact_sha256=canonical_json_sha256(body))


def write_hybrid_artifact(
    output_dir: Path | str,
    artifact: V2HybridArtifact,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "hybrid.json"
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


def event_dialogue_turns(
    stage2: V1Stage2Artifact,
    generated_dialogue: V1GeneratedEventDialogue,
) -> tuple[V1DialogueTurnRecord, ...]:
    timeline_item = next(
        (
            item
            for item in stage2.interleaved_timeline
            if item.event.event_id == generated_dialogue.event_id
        ),
        None,
    )
    if timeline_item is None:
        raise ValueError("Hybrid dialogue references an unknown Stage 2 event")
    if generated_dialogue.source_event_sha256 != event_sha256(timeline_item.event):
        raise ValueError("Hybrid dialogue source event hash mismatch")
    persona_by_id = {
        persona.persona_id: persona
        for persona in stage2.persona_group.payload.personas
    }
    turns = []
    for index, line in enumerate(generated_dialogue.payload.turns):
        persona = persona_by_id.get(line.speaker_id)
        if persona is None:
            raise ValueError("Hybrid dialogue speaker is absent from Persona group")
        turns.append(
            V1DialogueTurnRecord(
                turn_id=f"{timeline_item.event.event_id}-turn-{index + 1:03d}",
                source_event_id=timeline_item.event.event_id,
                timestamp=timeline_item.event.timestamp,
                speaker_id=line.speaker_id,
                speaker_name=persona.name,
                text=line.text.strip(),
            )
        )
    return tuple(turns)


def render_hybrid_memory(entries: Sequence[V2HybridMemoryEntry]) -> str:
    return "\n\n".join(entry.memory_line.strip() for entry in entries).strip()


def format_hybrid_memory_line(
    event: V1EventRecord,
    update: V1PreferenceUpdate,
    *,
    subject_name: str,
) -> str:
    value = json.dumps(update.new_value, ensure_ascii=False, allow_nan=False)
    details = [f"value={value}"]
    context_arguments = _selector_context_arguments(update)
    if context_arguments:
        context = ", ".join(
            f"{argument.name}="
            f"{json.dumps(argument.value, ensure_ascii=False, allow_nan=False)}"
            for argument in context_arguments
        )
        details.append(f"context=({context})")
    if update.condition:
        details.append(f"condition={update.condition}")
    return (
        f"- [{event.timestamp.replace('T', ' ')}] {subject_name}: "
        f"{update.attribute_path}; " + "; ".join(details)
    )


def hybrid_memory_identity(update: V1PreferenceUpdate) -> str:
    payload = {
        "subject_id": update.subject_id,
        "attribute_path": update.attribute_path,
        "condition": update.condition,
        "context_arguments": [
            argument.model_dump(mode="json")
            for argument in _selector_context_arguments(update)
        ],
    }
    return canonical_json_sha256(payload)


def _selector_context_arguments(
    update: V1PreferenceUpdate,
):
    try:
        _, value_argument = update.attribute_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError("Hybrid memory update has an invalid attribute path") from exc
    retained = []
    for argument in update.context_arguments:
        if argument.name != value_argument:
            retained.append(argument)
            continue
        if argument.value != update.new_value:
            raise ValueError(
                "Hybrid memory update has conflicting context and new values"
            )
    return tuple(retained)


def text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def event_sha256(event: V1EventRecord) -> str:
    return canonical_json_sha256(event.model_dump(mode="json"))


def dialogue_sha256(turns: Sequence[V1DialogueTurnRecord]) -> str:
    return canonical_json_sha256([turn.model_dump(mode="json") for turn in turns])


def _operations_for_update(
    entries: list[V2HybridMemoryEntry],
    event: V1EventRecord,
    update_index: int,
    update: V1PreferenceUpdate,
    *,
    persona_names: Mapping[str, str],
) -> tuple[list[V2HybridPatchOperation], bool]:
    subject_name = persona_names.get(update.subject_id, update.subject_id)
    identity = hybrid_memory_identity(update)
    new_line = format_hybrid_memory_line(
        event,
        update,
        subject_name=subject_name,
    )
    operations: list[V2HybridPatchOperation] = []
    changed_existing = False

    if update.supersedes_event_id:
        obsolete = [
            entry
            for entry in entries
            if entry.source_event_id == update.supersedes_event_id
            and entry.identity_key != identity
        ]
        for entry in obsolete:
            operations.append(
                V2HybridPatchOperation(
                    op="delete",
                    target=entry.memory_line,
                    content="",
                )
            )
            entries.remove(entry)
            changed_existing = True

    existing = next(
        (entry for entry in entries if entry.identity_key == identity),
        None,
    )
    replacement = V2HybridMemoryEntry(
        identity_key=identity,
        memory_line=new_line,
        source_event_id=event.event_id,
        source_update_index=update_index,
    )
    if existing is None:
        entries.append(replacement)
        operations.append(V2HybridPatchOperation(op="add", target="", content=new_line))
    elif existing.memory_line != new_line:
        position = entries.index(existing)
        entries[position] = replacement
        operations.append(
            V2HybridPatchOperation(
                op="replace",
                target=existing.memory_line,
                content=new_line,
            )
        )
        changed_existing = True
    return operations, changed_existing


def build_hybrid_update_operations(
    entries: list[V2HybridMemoryEntry],
    event: V1EventRecord,
    update_index: int,
    update: V1PreferenceUpdate,
    *,
    persona_names: Mapping[str, str],
) -> tuple[list[V2HybridPatchOperation], bool]:
    """Apply one structured update to an entry working copy and return its Patch.

    ``entries`` is intentionally mutated. Callers should pass a private working
    copy and commit it only after the returned Patch applies successfully.
    """

    return _operations_for_update(
        entries,
        event,
        update_index,
        update,
        persona_names=persona_names,
    )


def build_hybrid_end_temporary_operations(
    entries: list[V2HybridMemoryEntry],
    update: V1PreferenceUpdate,
) -> tuple[list[V2HybridPatchOperation], bool]:
    """Remove an explicit temporary override without rewriting its baseline."""

    if not update.supersedes_event_id:
        raise ValueError("end_temporary requires a superseded temporary event")
    baseline_identity = hybrid_memory_identity(update)
    baseline = next(
        (entry for entry in entries if entry.identity_key == baseline_identity),
        None,
    )
    if baseline is None:
        raise ValueError("end_temporary cannot restore a missing durable baseline")
    temporary_entries = [
        entry
        for entry in entries
        if entry.source_event_id == update.supersedes_event_id
        and entry.identity_key != baseline_identity
    ]
    if not temporary_entries:
        raise ValueError("end_temporary cannot find its active temporary override")
    operations = []
    for entry in temporary_entries:
        entries.remove(entry)
        operations.append(
            V2HybridPatchOperation(
                op="delete",
                target=entry.memory_line,
                content="",
            )
        )
    return operations, True


def _validate_generated_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    generated: V2GeneratedHybridEventAlignment,
    *,
    previous_memory: str,
) -> None:
    if generated.event_id != event.event_id:
        raise ValueError("Hybrid alignment changed event_id")
    if generated.source_event_sha256 != event_sha256(event):
        raise ValueError("Hybrid alignment source event hash mismatch")
    if generated.source_dialogue_sha256 != dialogue_sha256(dialogue_turns):
        raise ValueError("Hybrid alignment source dialogue hash mismatch")
    if generated.before_memory_sha256 != text_sha256(previous_memory):
        raise ValueError("Hybrid alignment before-memory hash mismatch")
    validate_hybrid_alignment(event, dialogue_turns, generated.payload)


def _validate_dialogue_belongs_to_event(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
) -> None:
    if not dialogue_turns:
        raise ValueError("Hybrid event dialogue cannot be empty")
    if any(turn.source_event_id != event.event_id for turn in dialogue_turns):
        raise ValueError("Hybrid alignment received dialogue from another event")


def _earliest_direct_subject_confirmation(
    update: V1PreferenceUpdate,
    turns: Sequence[V1DialogueTurnRecord],
    *,
    before_turn: int,
) -> int | None:
    # Conditional facts can become valid only after a later condition clause;
    # leave those boundaries to the model rather than moving them mechanically.
    if update.condition or before_turn <= 0:
        return None
    for index, turn in enumerate(turns[:before_turn]):
        if turn.speaker_id != update.subject_id:
            continue
        if not _text_supports_scalar(turn.text, update.new_value):
            continue
        prefix = " ".join(item.text for item in turns[: index + 1])
        if not all(
            _text_supports_scalar(prefix, argument.value)
            for argument in update.context_arguments
        ):
            continue
        if not _prefix_mentions_setting(prefix, update.attribute_path):
            continue
        return index
    return None


def _text_supports_scalar(text: str, value: Any) -> bool:
    normalized = _normalized_words(text)
    if isinstance(value, bool):
        candidates = ("true", "on", "enabled") if value else (
            "false",
            "off",
            "disabled",
        )
    elif isinstance(value, int) and not isinstance(value, bool):
        number_words = {
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
        candidates = (str(value), number_words.get(value, ""))
    else:
        candidates = (_normalized_words(str(value)),)
    for candidate in candidates:
        if not candidate:
            continue
        pattern = rf"(?:^|\s){re.escape(candidate)}(?:$|\s)"
        if re.search(pattern, f" {normalized} ") is None:
            continue
        if re.search(rf"\b(?:not|isn t|wasn t)\s+{re.escape(candidate)}\b", normalized):
            continue
        return True
    return False


def _prefix_mentions_setting(text: str, attribute_path: str) -> bool:
    normalized = _normalized_words(text)
    tokens = {
        token
        for token in _normalized_words(attribute_path).split()
        if len(token) >= 4 and token not in {"carcontrol", "set"}
    }
    return any(
        re.search(rf"(?:^|\s){re.escape(token)}(?:$|\s)", f" {normalized} ")
        for token in tokens
    )


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))


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


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
