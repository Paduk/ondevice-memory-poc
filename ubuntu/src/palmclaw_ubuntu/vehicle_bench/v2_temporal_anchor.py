"""Gold-anchor-first generation for current-turn Temporal memory labels."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.v1_generation import V1Stage2Artifact
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventRecord,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2GeneratedHybridEventAlignment,
    V2HybridEventAlignmentPayload,
    V2HybridUpdateAlignment,
    dialogue_sha256,
    event_sha256,
    text_sha256,
)

V2_TEMPORAL_ANCHOR_SCHEMA_VERSION = "vehiclemembench-v2-temporal-anchor-v1"
V2_TEMPORAL_ANCHOR_PROMPT_VERSION = (
    "vehiclemembench-v2-temporal-anchor-terra-v2"
)
V2_TEMPORAL_ANCHOR_ALIGNMENT_VERSION = (
    "vehiclemembench-v2-temporal-anchor-deterministic-v1"
)

V2_TEMPORAL_ANCHOR_INSTRUCTIONS = """
Generate one frozen, self-contained vehicle-memory anchor utterance for every
supplied structured preference update and no others.

Each anchor is the exact current turn at which a stateless updater receives only
previous approved memory and this turn. Therefore anchor_text itself must state
the supplied setting_hint and new_value. For conditional_upsert and
current_upsert, copy the exact condition. Copy every context argument as an
explicit `name value` phrase; never
rely on a collective subject such as shared-vehicle to imply zone=all or another
selector. temporary_override and end_temporary must also copy temporal_cue
verbatim. Keep each anchor to one natural sentence without new facts.

speaker_id must equal required_speaker_id when it is non-null. For a collective
subject, required_speaker_id is null and speaker_id may be any allowed speaker.
Copy source_event_id and source_update_index exactly. Do not generate dialogue
around the anchor and do not emit explanations.
""".strip()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2TemporalAnchor(_StrictModel):
    source_event_id: str = Field(min_length=1)
    source_update_index: int = Field(ge=0)
    speaker_id: str = Field(min_length=1)
    anchor_text: str = Field(min_length=1, max_length=700)

    @field_validator("anchor_text")
    @classmethod
    def require_one_line(cls, value: str) -> str:
        value = value.strip()
        if not value or "\n" in value or "\r" in value:
            raise ValueError("Temporal anchor must be one non-empty line")
        return value


class V2TemporalAnchorPayload(_StrictModel):
    anchors: tuple[V2TemporalAnchor, ...] = Field(min_length=1, max_length=64)


class V2GeneratedTemporalAnchors(_StrictModel):
    schema_version: str = V2_TEMPORAL_ANCHOR_SCHEMA_VERSION
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_temporal_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: V2TemporalAnchorPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = V2_TEMPORAL_ANCHOR_PROMPT_VERSION
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact_hash(self) -> V2GeneratedTemporalAnchors:
        body = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(body) != self.artifact_sha256:
            raise ValueError("Temporal anchor artifact hash is invalid")
        return self


class OpenAIV2TemporalAnchorGenerationModel:
    """Generate isolated anchors before any surrounding dialogue exists."""

    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 8_192,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Temporal anchor model ID is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        *,
        stage2: V1Stage2Artifact,
        temporal_plan: Mapping[str, Any],
    ) -> V2GeneratedTemporalAnchors:
        provider_payload = build_temporal_anchor_input(stage2, temporal_plan)
        provider_input = _canonical_json(provider_payload)
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V2_TEMPORAL_ANCHOR_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V2TemporalAnchorPayload,
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
            raise RuntimeError(f"Temporal anchor provider returned {status}")
        payload = V2TemporalAnchorPayload.model_validate(response.output_parsed)
        body = {
            "schema_version": V2_TEMPORAL_ANCHOR_SCHEMA_VERSION,
            "source_stage2_sha256": stage2.artifact_sha256,
            "source_temporal_plan_sha256": str(temporal_plan["artifact_sha256"]),
            "payload": payload.model_dump(mode="json"),
            "model_id": self.model_id,
            "prompt_version": V2_TEMPORAL_ANCHOR_PROMPT_VERSION,
            "input_sha256": canonical_json_sha256(provider_payload),
            "response_id": getattr(response, "id", None),
            "usage": {**_response_usage(response), "latency_ms": latency_ms},
        }
        return V2GeneratedTemporalAnchors(
            **body,
            artifact_sha256=canonical_json_sha256(body),
        )


def build_temporal_anchor_input(
    stage2: V1Stage2Artifact,
    temporal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    transitions = _transition_map(temporal_plan)
    persona_by_id = {
        item.persona_id: item for item in stage2.persona_group.payload.personas
    }
    updates = []
    for item in stage2.interleaved_timeline:
        event = item.event
        for update_index, update in enumerate(event.preference_updates):
            transition = transitions[(event.event_id, update_index)]
            required_speaker = (
                update.subject_id
                if update.subject_id in event.participant_ids
                else None
            )
            updates.append(
                {
                    "source_event_id": event.event_id,
                    "source_update_index": update_index,
                    "event_description": event.description,
                    "allowed_speakers": [
                        {
                            "speaker_id": persona_id,
                            "name": persona_by_id[persona_id].name,
                        }
                        for persona_id in event.participant_ids
                    ],
                    "required_speaker_id": required_speaker,
                    "subject_id": update.subject_id,
                    "setting_hint": temporal_setting_hint(update.attribute_path),
                    "new_value": update.new_value,
                    "context_arguments": [
                        value.model_dump(mode="json")
                        for value in update.context_arguments
                    ],
                    "condition": update.condition,
                    "temporal_action": transition["temporal_action"],
                    "temporal_cue": transition["temporal_cue"],
                }
            )
    return {
        "scenario_candidate_id": stage2.scenario_candidate_id,
        "anchor_contract": "previous_approved_memory_plus_current_turn_only",
        "updates": updates,
    }


def validate_temporal_anchors(
    stage2: V1Stage2Artifact,
    temporal_plan: Mapping[str, Any],
    payload: V2TemporalAnchorPayload,
) -> None:
    transitions = _transition_map(temporal_plan)
    events = {item.event.event_id: item.event for item in stage2.interleaved_timeline}
    expected = {
        (event.event_id, index)
        for event in events.values()
        for index, _ in enumerate(event.preference_updates)
    }
    actual = [
        (anchor.source_event_id, anchor.source_update_index)
        for anchor in payload.anchors
    ]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("Temporal anchors must cover every update exactly once")
    for anchor in payload.anchors:
        key = (anchor.source_event_id, anchor.source_update_index)
        event = events[anchor.source_event_id]
        update = event.preference_updates[anchor.source_update_index]
        transition = transitions[key]
        if anchor.speaker_id not in event.participant_ids:
            raise ValueError(f"Temporal anchor {key} uses an unknown speaker")
        if update.subject_id in event.participant_ids and (
            anchor.speaker_id != update.subject_id
        ):
            raise ValueError(f"Temporal anchor {key} changed its source subject")
        normalized = _normalized_phrase(anchor.anchor_text)
        hint = temporal_setting_hint(update.attribute_path)
        if not all(
            _phrase_occurs(normalized, token) for token in hint.split()
        ):
            raise ValueError(f"Temporal anchor {key} omits setting_hint={hint!r}")
        action = str(transition["temporal_action"])
        if action != "end_temporary" and not _text_mentions_value(
            anchor.anchor_text, update.new_value
        ):
            raise ValueError(f"Temporal anchor {key} omits new_value")
        if action in {"conditional_upsert", "current_upsert"} and (
            not update.condition
            or not _phrase_occurs(normalized, update.condition)
        ):
            raise ValueError(f"Temporal anchor {key} omits its exact condition")
        for argument in update.context_arguments:
            if not _text_mentions_value(anchor.anchor_text, argument.value):
                raise ValueError(
                    f"Temporal anchor {key} omits context {argument.name}"
                )
        cue = str(transition.get("temporal_cue", "")).strip()
        if action in {"temporary_override", "end_temporary"} and not (
            cue and _phrase_occurs(normalized, cue)
        ):
            raise ValueError(f"Temporal anchor {key} omits exact temporal cue")


def validate_dialogue_contains_anchors(
    anchors: Sequence[V2TemporalAnchor],
    dialogue_turns: Sequence[Any],
) -> None:
    positions = []
    for anchor in anchors:
        matches = [
            index
            for index, turn in enumerate(dialogue_turns)
            if turn.speaker_id == anchor.speaker_id
            and turn.text.strip() == anchor.anchor_text
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Dialogue must contain anchor {anchor.source_event_id}/"
                f"{anchor.source_update_index} exactly once"
            )
        positions.append((anchor.source_update_index, matches[0]))
    ordered = [position for _, position in sorted(positions)]
    if ordered != sorted(ordered):
        raise ValueError("Dialogue anchors must follow source_update_index order")


def build_deterministic_anchor_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    anchors: Sequence[V2TemporalAnchor],
    *,
    previous_memory: str,
) -> V2GeneratedHybridEventAlignment:
    if len(anchors) != len(event.preference_updates):
        raise ValueError("Anchor count differs from event update count")
    validate_dialogue_contains_anchors(anchors, dialogue_turns)
    alignments = []
    for anchor in sorted(anchors, key=lambda item: item.source_update_index):
        indexes = [
            index
            for index, turn in enumerate(dialogue_turns)
            if turn.speaker_id == anchor.speaker_id
            and turn.text == anchor.anchor_text
        ]
        if len(indexes) != 1:
            raise ValueError("Frozen anchor position is not unique")
        alignments.append(
            V2HybridUpdateAlignment(
                source_update_index=anchor.source_update_index,
                evidence_turn_index=indexes[0],
                evidence_quote=anchor.anchor_text,
                reason="Frozen Gold anchor fully supports this current-turn update.",
            )
        )
    payload = V2HybridEventAlignmentPayload(alignments=tuple(alignments))
    input_payload = {
        "event_id": event.event_id,
        "previous_memory_sha256": text_sha256(previous_memory),
        "anchors": [item.model_dump(mode="json") for item in anchors],
    }
    return V2GeneratedHybridEventAlignment(
        event_id=event.event_id,
        source_event_sha256=event_sha256(event),
        source_dialogue_sha256=dialogue_sha256(dialogue_turns),
        before_memory_sha256=text_sha256(previous_memory),
        payload=payload,
        model_id="deterministic:gold-anchor",
        prompt_version=V2_TEMPORAL_ANCHOR_ALIGNMENT_VERSION,
        input_sha256=canonical_json_sha256(input_payload),
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "latency_ms": 0,
        },
        normalization_notes=("alignment_fixed_by_gold_anchor",),
    )


def temporal_setting_hint(attribute_path: str) -> str:
    value = attribute_path.replace(".", "_")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).casefold()
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", value)
        if token not in {"carcontrol", "set", "enabled"}
    ]
    deduplicated = []
    for token in tokens:
        if token not in deduplicated:
            deduplicated.append(token)
    if not deduplicated:
        raise ValueError(f"Cannot derive setting hint from {attribute_path!r}")
    return " ".join(deduplicated)


def write_temporal_anchor_plan(
    path: Path, artifact: V2GeneratedTemporalAnchors
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_temporal_anchor_plan(
    path: Path,
    *,
    source_stage2_sha256: str,
    source_temporal_plan_sha256: str,
    required: bool = True,
) -> tuple[V2GeneratedTemporalAnchors | None, dict[str, tuple[V2TemporalAnchor, ...]]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None, {}
    artifact = V2GeneratedTemporalAnchors.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if artifact.source_stage2_sha256 != source_stage2_sha256:
        raise ValueError("Temporal anchor plan belongs to another Stage 2")
    if artifact.source_temporal_plan_sha256 != source_temporal_plan_sha256:
        raise ValueError("Temporal anchor plan belongs to another temporal plan")
    by_event: dict[str, list[V2TemporalAnchor]] = {}
    for anchor in artifact.payload.anchors:
        by_event.setdefault(anchor.source_event_id, []).append(anchor)
    return artifact, {
        event_id: tuple(sorted(items, key=lambda item: item.source_update_index))
        for event_id, items in by_event.items()
    }


def _transition_map(
    temporal_plan: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(item["source_event_id"]), int(item["source_update_index"])): item
        for item in temporal_plan.get("temporal_transitions", [])
    }


def _normalized_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _phrase_occurs(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = _normalized_phrase(phrase)
    return bool(
        normalized_phrase
        and re.search(
            rf"(?:^|\s){re.escape(normalized_phrase)}(?:$|\s)",
            normalized_text,
        )
    )


def _text_mentions_value(text: str, value: Any) -> bool:
    normalized = _normalized_phrase(text)
    if isinstance(value, bool):
        candidates = ("true", "on", "enabled") if value else (
            "false",
            "off",
            "disabled",
        )
    elif isinstance(value, int) and not isinstance(value, bool):
        candidates = (str(value),)
    elif isinstance(value, float):
        candidates = (str(value), f"{value:g}")
    else:
        candidates = (_normalized_phrase(str(value)),)
    return any(
        candidate and _phrase_occurs(normalized, candidate)
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
