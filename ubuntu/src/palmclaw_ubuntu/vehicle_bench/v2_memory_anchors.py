"""Causal event anchors for projecting V1 event plans into V2 memory labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1EventChainPayload,
    V1Stage2Artifact,
    interleave_event_chains,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import canonical_json_sha256

V2_MEMORY_ANCHOR_SCHEMA_VERSION = "vehiclemembench-v2-memory-anchor-schema-v1"
V2_MEMORY_ANCHOR_PROMPT_VERSION = "vehiclemembench-v2-memory-anchor-terra-v1"

V2_MEMORY_ANCHOR_INSTRUCTIONS = """
Choose the earliest causally valid memory anchor for every supplied final
vehicle-preference update.

An anchor is the first event in the same chain after which a real-time memory
system can safely store the complete target preference: subject, exact value,
selector/context, and condition are explicit or directly confirmed. Prefer a
subject's first clear confirmation over a later recap or administrative
"recorded/saved" event. Do not anchor a tentative proposal, another person's
preference, an accidental setting, or an unresolved reference.

Return exactly one anchor per source_event_id/source_update_index. anchor_event_id
must not occur after source_event_id. evidence_excerpt must be an exact non-empty
substring of the chosen event description. Give one short single-line reason.
""".strip()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2MemoryAnchor(_StrictModel):
    chain_id: str = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    source_update_index: int = Field(ge=0)
    anchor_event_id: str = Field(min_length=1)
    evidence_excerpt: str = Field(min_length=1, max_length=2_048)
    reason: str = Field(min_length=1, max_length=320)

    @field_validator("evidence_excerpt", "reason")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("V2 memory anchor text must be one non-empty line")
        return normalized


class V2MemoryAnchorPayload(_StrictModel):
    anchors: tuple[V2MemoryAnchor, ...] = Field(min_length=1, max_length=64)


class V2GeneratedMemoryAnchors(_StrictModel):
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: V2MemoryAnchorPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V2MemoryAnchorAudit(_StrictModel):
    source_update_count: int = Field(ge=1)
    anchored_update_count: int = Field(ge=1)
    moved_update_count: int = Field(ge=0)
    cleared_future_supersedes_count: int = Field(ge=0)
    exact_evidence_count: int = Field(ge=1)
    causal_order_passed: bool
    source_coverage_passed: bool
    passed: bool


class V2MemoryAnchorArtifact(_StrictModel):
    schema_version: str = V2_MEMORY_ANCHOR_SCHEMA_VERSION
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anchored_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated: V2GeneratedMemoryAnchors
    audit: V2MemoryAnchorAudit
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> V2MemoryAnchorArtifact:
        body = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(body) != self.artifact_sha256:
            raise ValueError("V2 memory anchor artifact hash is invalid")
        return self


class OpenAIV2MemoryAnchorModel:
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
            raise ValueError("V2 memory anchor model ID is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(self, stage2: V1Stage2Artifact) -> V2GeneratedMemoryAnchors:
        provider_input = render_memory_anchor_input(stage2)
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V2_MEMORY_ANCHOR_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V2MemoryAnchorPayload,
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
            raise RuntimeError(f"V2 memory anchor provider returned {status}")
        payload = V2MemoryAnchorPayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        payload = normalize_memory_anchor_evidence(stage2, payload)
        validate_memory_anchors(stage2, payload)
        return V2GeneratedMemoryAnchors(
            source_stage2_sha256=stage2.artifact_sha256,
            payload=payload,
            model_id=self.model_id,
            prompt_version=V2_MEMORY_ANCHOR_PROMPT_VERSION,
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
        )


def normalize_memory_anchor_evidence(
    stage2: V1Stage2Artifact,
    payload: V2MemoryAnchorPayload,
) -> V2MemoryAnchorPayload:
    """Replace paraphrased evidence with the exact selected event description."""

    events = {
        (chain.chain_id, event.event_id): event
        for chain in stage2.event_chains.payload.vehicle_chains
        for event in chain.events
    }
    anchors = []
    for anchor in payload.anchors:
        target = events.get((anchor.chain_id, anchor.anchor_event_id))
        if target is not None and anchor.evidence_excerpt not in target.description:
            anchor = anchor.model_copy(
                update={"evidence_excerpt": target.description}
            )
        anchors.append(anchor)
    return payload.model_copy(update={"anchors": tuple(anchors)})


def render_memory_anchor_input(stage2: V1Stage2Artifact) -> str:
    chains = []
    for chain in stage2.event_chains.payload.vehicle_chains:
        sources = []
        for event in chain.events:
            for update_index, update in enumerate(event.preference_updates):
                sources.append(
                    {
                        "source_event_id": event.event_id,
                        "source_update_index": update_index,
                        "target_update": update.model_dump(mode="json"),
                    }
                )
        chains.append(
            {
                "chain_id": chain.chain_id,
                "reasoning_type": chain.reasoning_type,
                "events": [
                    {
                        "event_id": event.event_id,
                        "description": event.description,
                        "participant_ids": list(event.participant_ids),
                    }
                    for event in chain.events
                ],
                "source_updates": sources,
            }
        )
    return _canonical_json({"vehicle_chains": chains})


def validate_memory_anchors(
    stage2: V1Stage2Artifact,
    payload: V2MemoryAnchorPayload,
) -> None:
    chain_by_id = {
        chain.chain_id: chain
        for chain in stage2.event_chains.payload.vehicle_chains
    }
    expected = {
        (chain.chain_id, event.event_id, update_index)
        for chain in chain_by_id.values()
        for event in chain.events
        for update_index, _ in enumerate(event.preference_updates)
    }
    actual = {
        (anchor.chain_id, anchor.source_event_id, anchor.source_update_index)
        for anchor in payload.anchors
    }
    if actual != expected or len(actual) != len(payload.anchors):
        raise ValueError(
            "V2 memory anchors must cover every source update exactly once"
        )
    for anchor in payload.anchors:
        chain = chain_by_id.get(anchor.chain_id)
        if chain is None:
            raise ValueError("V2 memory anchor references an unknown chain")
        by_id = {event.event_id: event for event in chain.events}
        source = by_id.get(anchor.source_event_id)
        target = by_id.get(anchor.anchor_event_id)
        if source is None or target is None:
            raise ValueError("V2 memory anchor crosses its source chain")
        if anchor.source_update_index >= len(source.preference_updates):
            raise ValueError("V2 memory anchor source update index is invalid")
        if chain.events.index(target) > chain.events.index(source):
            raise ValueError("V2 memory anchor cannot use a future event")
        update = source.preference_updates[anchor.source_update_index]
        if update.subject_id not in target.participant_ids and (
            update.subject_id != "shared-vehicle"
        ):
            raise ValueError("V2 memory anchor excludes its preference subject")
        if anchor.evidence_excerpt not in target.description:
            raise ValueError("V2 memory anchor evidence is not exact event text")


def project_anchored_stage2(
    stage2: V1Stage2Artifact,
    generated: V2GeneratedMemoryAnchors,
) -> tuple[V1Stage2Artifact, V2MemoryAnchorAudit]:
    if generated.source_stage2_sha256 != stage2.artifact_sha256:
        raise ValueError("V2 memory anchors were generated from another Stage 2")
    validate_memory_anchors(stage2, generated.payload)
    anchors = {
        (item.chain_id, item.source_event_id, item.source_update_index): item
        for item in generated.payload.anchors
    }
    moved = 0
    cleared = 0
    vehicle_chains = []
    for chain in stage2.event_chains.payload.vehicle_chains:
        event_index = {
            event.event_id: index for index, event in enumerate(chain.events)
        }
        anchored_updates: dict[str, list] = {
            event.event_id: [] for event in chain.events
        }
        for source in chain.events:
            for update_index, update in enumerate(source.preference_updates):
                anchor = anchors[(chain.chain_id, source.event_id, update_index)]
                projected = _without_redundant_value_context(update)
                if anchor.anchor_event_id != source.event_id:
                    moved += 1
                if (
                    update.supersedes_event_id is not None
                    and event_index.get(update.supersedes_event_id, -1)
                    >= event_index[anchor.anchor_event_id]
                ):
                    projected = projected.model_copy(
                        update={"previous_value": None, "supersedes_event_id": None}
                    )
                    cleared += 1
                anchored_updates[anchor.anchor_event_id].append(projected)
        events = tuple(
            event.model_copy(
                update={"preference_updates": tuple(anchored_updates[event.event_id])}
            )
            for event in chain.events
        )
        vehicle_chains.append(chain.model_copy(update={"events": events}))
    payload = V1EventChainPayload(
        background_chains=stage2.event_chains.payload.background_chains,
        vehicle_chains=tuple(vehicle_chains),
    )
    event_chains = stage2.event_chains.model_copy(
        update={
            "payload": payload,
            "prompt_version": (
                f"{stage2.event_chains.prompt_version}+"
                f"{V2_MEMORY_ANCHOR_PROMPT_VERSION}"
            ),
        }
    )
    timeline = interleave_event_chains(payload.all_chains)
    body = {
        "schema_version": stage2.schema_version,
        "scenario_candidate_id": stage2.scenario_candidate_id,
        "persona_group": stage2.persona_group.model_dump(mode="json"),
        "event_chains": event_chains.model_dump(mode="json"),
        "interleaved_timeline": [item.model_dump(mode="json") for item in timeline],
        "audit": stage2.audit.model_dump(mode="json"),
    }
    anchored = V1Stage2Artifact(
        **body,
        artifact_sha256=canonical_json_sha256(body),
    )
    audit = V2MemoryAnchorAudit(
        source_update_count=len(anchors),
        anchored_update_count=len(anchors),
        moved_update_count=moved,
        cleared_future_supersedes_count=cleared,
        exact_evidence_count=len(anchors),
        causal_order_passed=True,
        source_coverage_passed=True,
        passed=True,
    )
    return anchored, audit


def _without_redundant_value_context(update):
    try:
        _, value_argument = update.attribute_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError("V2 memory update has an invalid attribute path") from exc
    retained = []
    for argument in update.context_arguments:
        if argument.name != value_argument:
            retained.append(argument)
            continue
        if argument.value != update.new_value:
            raise ValueError(
                "V2 memory update has conflicting context and new values"
            )
    if len(retained) == len(update.context_arguments):
        return update
    return update.model_copy(update={"context_arguments": tuple(retained)})


def build_memory_anchor_artifact(
    stage2: V1Stage2Artifact,
    anchored_stage2: V1Stage2Artifact,
    generated: V2GeneratedMemoryAnchors,
    audit: V2MemoryAnchorAudit,
) -> V2MemoryAnchorArtifact:
    body = {
        "schema_version": V2_MEMORY_ANCHOR_SCHEMA_VERSION,
        "source_stage2_sha256": stage2.artifact_sha256,
        "anchored_stage2_sha256": anchored_stage2.artifact_sha256,
        "generated": generated.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
    }
    return V2MemoryAnchorArtifact(**body, artifact_sha256=canonical_json_sha256(body))


def write_memory_anchor_artifacts(
    output_dir: Path | str,
    artifact: V2MemoryAnchorArtifact,
    anchored_stage2: V1Stage2Artifact,
) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    anchor_path = root / "memory-anchors.json"
    stage2_path = root / "stage2-v2-anchored.json"
    _write_json(anchor_path, artifact.model_dump(mode="json"))
    _write_json(stage2_path, anchored_stage2.model_dump(mode="json"))
    return {"anchors": anchor_path, "stage2": stage2_path}


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
