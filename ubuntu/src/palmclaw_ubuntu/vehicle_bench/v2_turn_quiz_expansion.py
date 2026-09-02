"""Delayed and compositional Turn Quiz planning for VehicleMemBench V2."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.dataset import REASONING_TYPES
from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1ArgumentValue,
    V1GoldToolCallRecord,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1GeneratedEventDialogue
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    event_dialogue_turns,
)
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz import (
    V2TurnQuizEventCheckpoint,
    V2TurnQuizGenerationContext,
)

V2_TURN_QUIZ_EXPANSION_SCHEMA_VERSION = (
    "vehiclemembench-v2-turn-quiz-expansion-schema-v1"
)
V2_COMPOSITE_QUIZ_PLAN_PROMPT_VERSION = (
    "vehiclemembench-v2-composite-quiz-plan-terra-v1"
)

V2_COMPOSITE_QUIZ_PLAN_INSTRUCTIONS = """
Select exactly the requested number of distinct two-fact groups for natural,
executable multi-memory vehicle requests.

Rules:
- Use exactly two different supplied fact_ids per group and no unknown fact.
- The two settings must be jointly plausible in one request. Prefer compatible
  person, trip, location, weather, or activity contexts.
- Do not pair contradictory conditions or two incompatible values for the same
  exact vehicle setting and selector.
- Prefer coverage diversity across people, modules, and reasoning types; facts
  may recur across different groups when needed.
- assigned_reasoning_type must equal one of the two facts' reasoning types.
- Return one short single-line reason. Do not write the eventual Quiz query.
""".strip()

QuizKind = Literal["delayed", "composite"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2TurnQuizFact(_StrictModel):
    fact_id: str = Field(pattern=r"^fact-[A-Za-z0-9._-]+$")
    source_context_index: int = Field(ge=0)
    source_event_id: str = Field(min_length=1)
    source_chain_id: str = Field(min_length=1)
    reasoning_type: str = Field(min_length=1)
    memory_line: str = Field(min_length=1)
    gold_call: V1GoldToolCallRecord
    hidden_value_argument: V1ArgumentValue


class V2CompositeQuizPlanItem(_StrictModel):
    fact_ids: tuple[str, ...] = Field(min_length=2, max_length=2)
    assigned_reasoning_type: str
    reason: str = Field(min_length=1, max_length=320)

    @field_validator("fact_ids")
    @classmethod
    def validate_distinct_facts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != 2:
            raise ValueError("Composite Quiz facts must be distinct")
        return values

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Composite Quiz plan reason must be one line")
        return normalized


class V2CompositeQuizPlanPayload(_StrictModel):
    groups: tuple[V2CompositeQuizPlanItem, ...] = Field(min_length=1, max_length=32)


class V2GeneratedCompositeQuizPlan(_StrictModel):
    payload: V2CompositeQuizPlanPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V2SupplementalQuizSpec(_StrictModel):
    quiz_id: str = Field(pattern=r"^turn-quiz-[A-Za-z0-9._-]+$")
    kind: QuizKind
    target_fact_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    cutoff_event_id: str = Field(min_length=1)
    cutoff_turn_id: str = Field(min_length=1)
    source_event_descriptions: tuple[str, ...] = Field(min_length=1, max_length=3)
    planning_reason: str = Field(min_length=1, max_length=320)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class V2TurnQuizExpansionPlan(_StrictModel):
    schema_version: str = V2_TURN_QUIZ_EXPANSION_SCHEMA_VERSION
    source_stage2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    immediate_quiz_count: int = Field(ge=1)
    delayed_quiz_count: int = Field(ge=1)
    composite_quiz_count: int = Field(ge=1)
    target_quiz_count: int = Field(ge=1)
    facts: tuple[V2TurnQuizFact, ...]
    generated_composite_plan: V2GeneratedCompositeQuizPlan
    supplemental_specs: tuple[V2SupplementalQuizSpec, ...]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_counts_and_hash(self) -> V2TurnQuizExpansionPlan:
        if (
            self.immediate_quiz_count
            + self.delayed_quiz_count
            + self.composite_quiz_count
            != self.target_quiz_count
        ):
            raise ValueError("Turn Quiz expansion counts do not reach target")
        if len(self.supplemental_specs) != (
            self.delayed_quiz_count + self.composite_quiz_count
        ):
            raise ValueError("Turn Quiz expansion spec count is invalid")
        body = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(body) != self.artifact_sha256:
            raise ValueError("Turn Quiz expansion plan hash is invalid")
        return self


class OpenAIV2CompositeQuizPlanModel:
    def __init__(
        self,
        model_id: str = V1_DEFAULT_GENERATION_MODEL,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 4_096,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Composite Quiz planning model ID is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        facts: Sequence[V2TurnQuizFact],
        *,
        group_count: int,
    ) -> V2GeneratedCompositeQuizPlan:
        provider_input = _canonical_json(
            {
                "requested_group_count": group_count,
                "facts": [fact.model_dump(mode="json") for fact in facts],
            }
        )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V2_COMPOSITE_QUIZ_PLAN_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V2CompositeQuizPlanPayload,
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
            raise RuntimeError(f"Composite Quiz planner returned status {status}")
        payload = V2CompositeQuizPlanPayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        validate_composite_quiz_plan(facts, payload, group_count=group_count)
        return V2GeneratedCompositeQuizPlan(
            payload=payload,
            model_id=self.model_id,
            prompt_version=V2_COMPOSITE_QUIZ_PLAN_PROMPT_VERSION,
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            response_id=getattr(response, "id", None),
            usage={**_response_usage(response), "latency_ms": latency_ms},
        )


def build_turn_quiz_facts(
    contexts: Sequence[V2TurnQuizGenerationContext],
    *,
    evidence_by_quiz_id: Mapping[str, Sequence[str]] | None = None,
) -> tuple[V2TurnQuizFact, ...]:
    facts = []
    for context_index, context in enumerate(contexts):
        if len(context.gold_calls) != len(context.hidden_value_arguments):
            raise ValueError("Turn Quiz calls and hidden values are not aligned")
        for call_index, (call, hidden) in enumerate(
            zip(context.gold_calls, context.hidden_value_arguments, strict=True)
        ):
            supporting = _supporting_memory_lines(context.current_memory, call)
            if evidence_by_quiz_id is not None:
                evidence = set(evidence_by_quiz_id.get(context.quiz_id, ()))
                supporting = tuple(line for line in supporting if line in evidence)
            if len(supporting) != 1:
                raise ValueError("Turn Quiz fact must have one exact memory line")
            facts.append(
                V2TurnQuizFact(
                    fact_id=(
                        f"fact-{context.source_event_id}-{context_index + 1:02d}-"
                        f"{call_index + 1:02d}"
                    ),
                    source_context_index=context_index,
                    source_event_id=context.source_event_id,
                    source_chain_id=context.source_chain_id,
                    reasoning_type=context.reasoning_type,
                    memory_line=supporting[0],
                    gold_call=call,
                    hidden_value_argument=hidden,
                )
            )
    return tuple(facts)


def validate_composite_quiz_plan(
    facts: Sequence[V2TurnQuizFact],
    payload: V2CompositeQuizPlanPayload,
    *,
    group_count: int,
) -> None:
    if len(payload.groups) != group_count:
        raise ValueError("Composite Quiz plan has the wrong group count")
    by_id = {fact.fact_id: fact for fact in facts}
    if len(by_id) != len(facts):
        raise ValueError("Composite Quiz facts must have unique IDs")
    pairs = []
    for group in payload.groups:
        if any(fact_id not in by_id for fact_id in group.fact_ids):
            raise ValueError("Composite Quiz plan references an unknown fact")
        selected = tuple(by_id[fact_id] for fact_id in group.fact_ids)
        if group.assigned_reasoning_type not in {
            fact.reasoning_type for fact in selected
        }:
            raise ValueError("Composite reasoning type must come from its facts")
        if group.assigned_reasoning_type not in REASONING_TYPES:
            raise ValueError("Composite reasoning type is not official")
        signatures = {
            canonical_json_sha256(fact.gold_call.model_dump(mode="json"))
            for fact in selected
        }
        if len(signatures) != 2:
            raise ValueError("Composite Quiz cannot duplicate one executable call")
        pairs.append(tuple(sorted(group.fact_ids)))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Composite Quiz plan contains a duplicate fact pair")


def repair_composite_quiz_plan_for_checkpoints(
    facts: Sequence[V2TurnQuizFact],
    generated_plan: V2GeneratedCompositeQuizPlan,
    immediate_contexts: Sequence[V2TurnQuizGenerationContext],
    checkpoints: Sequence[V2TurnQuizEventCheckpoint],
) -> V2GeneratedCompositeQuizPlan:
    """Deterministically replace pairs that never coexist in causal memory."""

    fact_by_id = {fact.fact_id: fact for fact in facts}
    context_by_index = {
        index: context for index, context in enumerate(immediate_contexts)
    }
    timeline_index = {item.event_id: index for index, item in enumerate(checkpoints)}
    checkpoint_index_by_sha = {
        item.checkpoint_sha256: timeline_index[item.event_id] for item in checkpoints
    }

    def pair_is_supported(pair: Sequence[V2TurnQuizFact]) -> bool:
        earliest_index = max(
            checkpoint_index_by_sha[
                context_by_index[fact.source_context_index].source_checkpoint_sha256
            ]
            for fact in pair
        )
        return any(
            index >= earliest_index
            and all(
                _supporting_memory_lines(checkpoint.after_memory, fact.gold_call)
                for fact in pair
            )
            for index, checkpoint in enumerate(checkpoints)
        )

    groups = list(generated_plan.payload.groups)
    reserved_ids = {
        fact_id for group in groups for fact_id in group.fact_ids
    }
    repaired_groups = []
    used_pairs: set[tuple[str, str]] = set()
    for group in groups:
        selected = tuple(fact_by_id[fact_id] for fact_id in group.fact_ids)
        pair_key = tuple(sorted(group.fact_ids))
        if pair_is_supported(selected) and pair_key not in used_pairs:
            repaired_groups.append(group)
            used_pairs.add(pair_key)
            continue

        candidates = []
        original_ids = set(group.fact_ids)
        for pair in combinations(facts, 2):
            candidate_ids = tuple(sorted(fact.fact_id for fact in pair))
            if candidate_ids in used_pairs:
                continue
            signatures = {
                canonical_json_sha256(fact.gold_call.model_dump(mode="json"))
                for fact in pair
            }
            if len(signatures) != 2 or not pair_is_supported(pair):
                continue
            retains_reasoning = group.assigned_reasoning_type in {
                fact.reasoning_type for fact in pair
            }
            overlap = len(original_ids.intersection(candidate_ids))
            reserved_collision = len(
                (set(candidate_ids) - original_ids).intersection(reserved_ids)
            )
            candidates.append(
                (
                    -overlap,
                    0 if retains_reasoning else 1,
                    reserved_collision,
                    candidate_ids,
                    pair,
                )
            )
        if not candidates:
            raise ValueError("Composite Quiz plan has no causally supported repair")
        *_, replacement = min(candidates, key=lambda item: item[:-1])
        reasoning_type = (
            group.assigned_reasoning_type
            if group.assigned_reasoning_type
            in {fact.reasoning_type for fact in replacement}
            else replacement[0].reasoning_type
        )
        repaired = V2CompositeQuizPlanItem(
            fact_ids=tuple(fact.fact_id for fact in replacement),
            assigned_reasoning_type=reasoning_type,
            reason=(
                "Deterministic causal repair: both facts coexist in the selected "
                "memory checkpoint."
            ),
        )
        repaired_groups.append(repaired)
        used_pairs.add(tuple(sorted(repaired.fact_ids)))

    repaired_payload = V2CompositeQuizPlanPayload(groups=tuple(repaired_groups))
    validate_composite_quiz_plan(
        facts,
        repaired_payload,
        group_count=len(groups),
    )
    if repaired_payload == generated_plan.payload:
        return generated_plan
    return generated_plan.model_copy(update={"payload": repaired_payload})


def select_delayed_context_indexes(
    context_count: int,
    selected_count: int,
) -> tuple[int, ...]:
    """Select a deterministic, timeline-spread subset for delayed quizzes."""

    if context_count < 1 or not 1 <= selected_count <= context_count:
        raise ValueError("Delayed Quiz selection counts are invalid")
    if selected_count == context_count:
        return tuple(range(context_count))
    if selected_count == 1:
        return (context_count - 1,)
    indexes = tuple(
        round(index * (context_count - 1) / (selected_count - 1))
        for index in range(selected_count)
    )
    if len(set(indexes)) != selected_count:
        raise ValueError("Delayed Quiz selection produced duplicate indexes")
    return indexes


def build_supplemental_turn_quiz_contexts(
    stage2: V1Stage2Artifact,
    checkpoints: Sequence[V2TurnQuizEventCheckpoint],
    dialogues: Sequence[V1GeneratedEventDialogue],
    immediate_contexts: Sequence[V2TurnQuizGenerationContext],
    generated_plan: V2GeneratedCompositeQuizPlan,
    *,
    facts: Sequence[V2TurnQuizFact] | None = None,
    delayed_context_indexes: Sequence[int] | None = None,
) -> tuple[
    tuple[V2TurnQuizGenerationContext, ...],
    tuple[V2SupplementalQuizSpec, ...],
    tuple[V2TurnQuizFact, ...],
]:
    facts = tuple(facts) if facts is not None else build_turn_quiz_facts(
        immediate_contexts
    )
    validate_composite_quiz_plan(
        facts,
        generated_plan.payload,
        group_count=len(generated_plan.payload.groups),
    )
    checkpoint_by_event = {item.event_id: item for item in checkpoints}
    dialogue_by_event = {item.event_id: item for item in dialogues}
    timeline_index = {item.event_id: index for index, item in enumerate(checkpoints)}
    event_by_id = {
        event.event_id: event
        for chain in stage2.event_chains.payload.all_chains
        for event in chain.events
    }
    chain_by_event = {
        event.event_id: chain
        for chain in stage2.event_chains.payload.vehicle_chains
        for event in chain.events
    }
    context_by_index = {index: item for index, item in enumerate(immediate_contexts)}
    facts_by_id = {fact.fact_id: fact for fact in facts}
    contexts = []
    specs = []

    selected_delayed_indexes = (
        set(range(len(immediate_contexts)))
        if delayed_context_indexes is None
        else set(delayed_context_indexes)
    )
    if any(
        index < 0 or index >= len(immediate_contexts)
        for index in selected_delayed_indexes
    ):
        raise ValueError("Delayed Quiz context index is outside the source set")

    for source_index, source in enumerate(immediate_contexts):
        if source_index not in selected_delayed_indexes:
            continue
        index = source_index + 1
        chain = chain_by_event[source.source_event_id]
        source_cutoff = checkpoint_by_sha(
            checkpoints,
            source.source_checkpoint_sha256,
        )
        intended_cutoff = checkpoint_by_event[chain.events[-1].event_id]
        source_timeline_index = timeline_index[source_cutoff.event_id]
        intended_timeline_index = timeline_index[intended_cutoff.event_id]
        supported_cutoffs = [
            item
            for item in checkpoints
            if source_timeline_index <= timeline_index[item.event_id]
            <= intended_timeline_index
            and all(
                _supporting_memory_lines(item.after_memory, call)
                for call in source.gold_calls
            )
        ]
        if not supported_cutoffs:
            raise ValueError("Delayed Quiz has no supported causal cutoff")
        cutoff = max(
            supported_cutoffs,
            key=lambda item: timeline_index[item.event_id],
        )
        cutoff_event_id = cutoff.event_id
        cutoff_turns = event_dialogue_turns(
            stage2, dialogue_by_event[cutoff_event_id]
        )
        quiz_id = (
            f"turn-quiz-delayed-{source.source_event_id}-{index:02d}-"
            f"at-{cutoff_event_id}"
        )
        context = _context_at_cutoff(
            quiz_id=quiz_id,
            source_event_id=source.source_event_id,
            source_chain_id=source.source_chain_id,
            reasoning_type=source.reasoning_type,
            cutoff=cutoff,
            cutoff_turns=cutoff_turns,
            gold_calls=source.gold_calls,
            hidden_values=source.hidden_value_arguments,
        )
        target_ids = tuple(
            fact.fact_id
            for fact in facts
            if fact.source_context_index == index - 1
        )
        spec = _build_spec(
            context,
            kind="delayed",
            target_fact_ids=target_ids,
            cutoff_event_id=cutoff_event_id,
            source_event_descriptions=(
                event_by_id[source.source_event_id].description,
                event_by_id[cutoff_event_id].description,
            ),
            planning_reason="Re-query the same memory after its source chain matures.",
        )
        contexts.append(context)
        specs.append(spec)

    for index, group in enumerate(generated_plan.payload.groups, start=1):
        selected_facts = tuple(facts_by_id[fact_id] for fact_id in group.fact_ids)
        selected_contexts = tuple(
            context_by_index[fact.source_context_index] for fact in selected_facts
        )
        earliest_timeline_index = max(
            timeline_index[
                checkpoint_by_sha(checkpoints, item.source_checkpoint_sha256).event_id
            ]
            for item in selected_contexts
        )
        supported_cutoffs = tuple(
            item
            for item in checkpoints
            if timeline_index[item.event_id] >= earliest_timeline_index
            and all(
                _supporting_memory_lines(item.after_memory, fact.gold_call)
                for fact in selected_facts
            )
        )
        if not supported_cutoffs:
            raise ValueError("Composite Quiz facts have no shared causal cutoff")
        cutoff = min(
            supported_cutoffs,
            key=lambda item: timeline_index[item.event_id],
        )
        cutoff_event_id = cutoff.event_id
        cutoff_turns = event_dialogue_turns(
            stage2, dialogue_by_event[cutoff_event_id]
        )
        primary = next(
            item
            for item in selected_contexts
            if item.reasoning_type == group.assigned_reasoning_type
        )
        quiz_id = f"turn-quiz-composite-{index:02d}-at-{cutoff_event_id}"
        context = _context_at_cutoff(
            quiz_id=quiz_id,
            source_event_id=cutoff_event_id,
            source_chain_id=primary.source_chain_id,
            reasoning_type=group.assigned_reasoning_type,
            cutoff=cutoff,
            cutoff_turns=cutoff_turns,
            gold_calls=tuple(fact.gold_call for fact in selected_facts),
            hidden_values=tuple(
                fact.hidden_value_argument for fact in selected_facts
            ),
        )
        descriptions = tuple(
            dict.fromkeys(
                event_by_id[fact.source_event_id].description
                for fact in selected_facts
            )
        )
        spec = _build_spec(
            context,
            kind="composite",
            target_fact_ids=group.fact_ids,
            cutoff_event_id=cutoff_event_id,
            source_event_descriptions=descriptions,
            planning_reason=group.reason,
        )
        contexts.append(context)
        specs.append(spec)

    return tuple(contexts), tuple(specs), facts


def build_turn_quiz_expansion_plan(
    stage2: V1Stage2Artifact,
    facts: Sequence[V2TurnQuizFact],
    generated_plan: V2GeneratedCompositeQuizPlan,
    specs: Sequence[V2SupplementalQuizSpec],
    *,
    immediate_quiz_count: int,
    target_quiz_count: int,
) -> V2TurnQuizExpansionPlan:
    delayed_count = sum(item.kind == "delayed" for item in specs)
    composite_count = sum(item.kind == "composite" for item in specs)
    body = {
        "schema_version": V2_TURN_QUIZ_EXPANSION_SCHEMA_VERSION,
        "source_stage2_sha256": stage2.artifact_sha256,
        "immediate_quiz_count": immediate_quiz_count,
        "delayed_quiz_count": delayed_count,
        "composite_quiz_count": composite_count,
        "target_quiz_count": target_quiz_count,
        "facts": [item.model_dump(mode="json") for item in facts],
        "generated_composite_plan": generated_plan.model_dump(mode="json"),
        "supplemental_specs": [item.model_dump(mode="json") for item in specs],
    }
    return V2TurnQuizExpansionPlan(
        **body,
        artifact_sha256=canonical_json_sha256(body),
    )


def checkpoint_by_sha(
    checkpoints: Sequence[V2TurnQuizEventCheckpoint], checkpoint_sha256: str
) -> V2TurnQuizEventCheckpoint:
    matches = [
        item for item in checkpoints if item.checkpoint_sha256 == checkpoint_sha256
    ]
    if len(matches) != 1:
        raise ValueError("Turn Quiz checkpoint hash is not unique")
    return matches[0]


def _context_at_cutoff(
    *,
    quiz_id: str,
    source_event_id: str,
    source_chain_id: str,
    reasoning_type: str,
    cutoff: V2TurnQuizEventCheckpoint,
    cutoff_turns,
    gold_calls: tuple[V1GoldToolCallRecord, ...],
    hidden_values: tuple[V1ArgumentValue, ...],
) -> V2TurnQuizGenerationContext:
    if not cutoff.turn_labels or not cutoff.after_memory:
        raise ValueError("Supplemental Quiz cutoff has no memory snapshot")
    last = cutoff.turn_labels[-1]
    if last.turn_id != cutoff_turns[-1].turn_id:
        raise ValueError("Supplemental Quiz cutoff turn is inconsistent")
    for call in gold_calls:
        if not _supporting_memory_lines(cutoff.after_memory, call):
            raise ValueError("Supplemental Quiz gold call is absent from memory")
    return V2TurnQuizGenerationContext(
        quiz_id=quiz_id,
        source_checkpoint_sha256=cutoff.checkpoint_sha256,
        source_event_id=source_event_id,
        source_chain_id=source_chain_id,
        reasoning_type=reasoning_type,
        global_turn_index=last.global_turn_index,
        event_turn_index=last.event_turn_index,
        turn_id=last.turn_id,
        current_memory=cutoff.after_memory,
        current_memory_sha256=cutoff.after_memory_sha256,
        causal_dialogue_prefix=tuple(cutoff_turns),
        causal_prefix_sha256=canonical_json_sha256(
            [turn.model_dump(mode="json") for turn in cutoff_turns]
        ),
        gold_calls=gold_calls,
        gold_calls_sha256=canonical_json_sha256(
            [call.model_dump(mode="json") for call in gold_calls]
        ),
        hidden_value_arguments=hidden_values,
    )


def _build_spec(
    context: V2TurnQuizGenerationContext,
    *,
    kind: QuizKind,
    target_fact_ids: tuple[str, ...],
    cutoff_event_id: str,
    source_event_descriptions: tuple[str, ...],
    planning_reason: str,
) -> V2SupplementalQuizSpec:
    body = context.model_dump(mode="json")
    return V2SupplementalQuizSpec(
        quiz_id=context.quiz_id,
        kind=kind,
        target_fact_ids=target_fact_ids,
        cutoff_event_id=cutoff_event_id,
        cutoff_turn_id=context.turn_id,
        source_event_descriptions=source_event_descriptions,
        planning_reason=planning_reason,
        context_sha256=canonical_json_sha256(body),
    )


def _supporting_memory_lines(
    memory: str, call: V1GoldToolCallRecord
) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in memory.splitlines()
        if line.strip() and _memory_line_supports_call(line, call)
    )


def _memory_line_supports_call(line: str, call: V1GoldToolCallRecord) -> bool:
    parsed = _parse_memory_line_call_fields(line)
    if parsed is None:
        return False
    tool_name, value_argument, value, context_arguments = parsed
    if tool_name != call.name:
        return False
    expected = dict(call.arguments)
    if expected.get(value_argument) != value:
        return False
    return all(
        name == value_argument or context_arguments.get(name) == argument_value
        for name, argument_value in expected.items()
    )


def _parse_memory_line_call_fields(
    line: str,
) -> tuple[str, str, Any, dict[str, Any]] | None:
    path_match = re.search(
        r": (?P<tool>carcontrol_[A-Za-z0-9_]+)\."
        r"(?P<argument>[A-Za-z0-9_]+); value=",
        line,
    )
    if path_match is None:
        return None
    decoder = json.JSONDecoder()
    try:
        value, value_end = decoder.raw_decode(line, path_match.end())
    except json.JSONDecodeError:
        return None

    context_arguments: dict[str, Any] = {}
    context_marker = "; context=("
    context_start = line.find(context_marker, value_end)
    if context_start >= 0:
        position = context_start + len(context_marker)
        while position < len(line):
            if line[position] == ")":
                break
            name_match = re.match(r"\s*([A-Za-z0-9_]+)=", line[position:])
            if name_match is None:
                return None
            name = name_match.group(1)
            position += name_match.end()
            try:
                argument_value, position = decoder.raw_decode(line, position)
            except json.JSONDecodeError:
                return None
            context_arguments[name] = argument_value
            if line[position : position + 2] == ", ":
                position += 2
            elif position < len(line) and line[position] != ")":
                return None

    return (
        path_match.group("tool"),
        path_match.group("argument"),
        value,
        context_arguments,
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
