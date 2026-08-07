from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from palmclaw_ubuntu.models import (
    FactMemoryRecord,
    FactQueryContext,
    ToolDefinition,
)
from palmclaw_ubuntu.tool_memory_execution import (
    ToolMemoryExecutionHint,
    ToolMemoryHintRejection,
    ToolMemoryHintResult,
    fact_record_arguments,
    validate_runtime_argument_semantics,
)
from palmclaw_ubuntu.tool_memory_schema import (
    ToolMemoryTool,
    build_tool_memory_ontology,
    normalize_identifier,
)

PlanStatus = Literal["ready", "ambiguous", "blocked"]
_WORD = re.compile(r"[\w가-힣]+", re.UNICODE)
_LOW_SIGNAL_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "car",
        "control",
        "for",
        "is",
        "of",
        "on",
        "set",
        "setting",
        "settings",
        "the",
        "to",
        "tool",
        "vehicle",
        "with",
    }
)


@dataclass(frozen=True)
class MemoryBundle:
    bundle_id: str
    entity_id: str
    applicability: Mapping[str, Any]
    records: tuple[FactMemoryRecord, ...]

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(record.id for record in self.records)


@dataclass(frozen=True)
class JointToolCandidate:
    tool_name: str
    domain: str
    score: float
    source_record_ids: tuple[str, ...]
    sources: tuple[str, ...]
    query_terms: tuple[str, ...]
    fact_predicates: tuple[str, ...]
    required_slots: tuple[str, ...]
    schema_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "domain": self.domain,
            "score": self.score,
            "source_record_ids": list(self.source_record_ids),
            "sources": list(self.sources),
            "query_terms": list(self.query_terms),
            "fact_predicates": list(self.fact_predicates),
            "required_slots": list(self.required_slots),
            "schema_tokens": self.schema_tokens,
        }


@dataclass(frozen=True)
class DesiredStateItem:
    id: str
    bundle_id: str
    capability: str
    operation: str
    target: str | None
    desired_value: Any
    selectors: Mapping[str, Any]
    evidence_record_ids: tuple[str, ...]
    confidence: float
    status: PlanStatus
    unresolved_slots: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bundle_id": self.bundle_id,
            "capability": self.capability,
            "operation": self.operation,
            "target": self.target,
            "desired_value": self.desired_value,
            "selectors": dict(self.selectors),
            "evidence_record_ids": list(self.evidence_record_ids),
            "confidence": self.confidence,
            "status": self.status,
            "unresolved_slots": list(self.unresolved_slots),
        }


@dataclass(frozen=True)
class PlannedToolCall:
    tool_name: str
    arguments: Mapping[str, Any]
    desired_state_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "desired_state_ids": list(self.desired_state_ids),
            "record_ids": list(self.record_ids),
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class MemoryToolPlan:
    status: PlanStatus
    bundles: tuple[MemoryBundle, ...]
    candidates: tuple[JointToolCandidate, ...]
    desired_states: tuple[DesiredStateItem, ...]
    calls: tuple[PlannedToolCall, ...]
    rejections: tuple[ToolMemoryHintRejection, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "memory_tool_plan_status": self.status,
            "memory_bundle_count": len(self.bundles),
            "memory_bundle_record_ids": [
                list(bundle.record_ids) for bundle in self.bundles
            ],
            "joint_tool_candidate_count": len(self.candidates),
            "joint_tool_candidates": [item.as_dict() for item in self.candidates],
            "desired_state_count": len(self.desired_states),
            "desired_states": [item.as_dict() for item in self.desired_states],
            "planned_tool_call_count": len(self.calls),
            "planned_tool_calls": [item.as_dict() for item in self.calls],
            "memory_tool_plan_rejections": [
                item.as_dict() for item in self.rejections
            ],
            **dict(self.metadata),
        }

    def as_hint_result(self) -> ToolMemoryHintResult:
        states = {item.id: item for item in self.desired_states}
        hints = []
        for call in self.calls:
            confidence = min(
                (
                    states[item_id].confidence
                    for item_id in call.desired_state_ids
                    if item_id in states
                ),
                default=0.5,
            )
            hints.append(
                ToolMemoryExecutionHint(
                    record_id=call.record_ids[0] if call.record_ids else "planner",
                    tool_name=call.tool_name,
                    arguments=call.arguments,
                    confidence=confidence,
                    tool_domain="joint_plan",
                    topic="desired_state",
                    record_ids=call.record_ids,
                    desired_state_ids=call.desired_state_ids,
                    depends_on=call.depends_on,
                )
            )
        return ToolMemoryHintResult(
            hints=tuple(hints),
            rejections=self.rejections,
        )


def choose_memory_tool_hint_result(
    planned: ToolMemoryHintResult | None,
    fallback: ToolMemoryHintResult,
) -> ToolMemoryHintResult:
    """Apply a plan only when it adds safe value over the legacy binder."""

    if planned is None or not planned.hints:
        return fallback
    if not fallback.hints:
        return planned
    has_multi_fact_call = any(len(hint.record_ids) > 1 for hint in planned.hints)
    planned_record_ids = {
        record_id for hint in planned.hints for record_id in hint.record_ids
    }
    fallback_record_ids = {hint.record_id for hint in fallback.hints}
    if (
        has_multi_fact_call
        and fallback_record_ids <= planned_record_ids
        and len(planned.hints) <= len(fallback.hints)
    ):
        return planned
    planned_signatures = {
        (hint.tool_name, _stable_arguments(hint.arguments)) for hint in planned.hints
    }
    fallback_signatures = {
        (hint.tool_name, _stable_arguments(hint.arguments)) for hint in fallback.hints
    }
    return planned if fallback_signatures <= planned_signatures else fallback


def build_memory_bundles(
    records: Sequence[FactMemoryRecord],
) -> tuple[MemoryBundle, ...]:
    """Group only records sharing source event, entity, and applicability."""

    groups: dict[tuple[str, str, str], list[FactMemoryRecord]] = {}
    for record in records:
        applicability_key = json.dumps(
            dict(record.applicability),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        source_bundle = record.bundle_id or f"fact:{record.id}"
        key = (source_bundle, record.entity_id, applicability_key)
        groups.setdefault(key, []).append(record)
    return tuple(
        MemoryBundle(
            bundle_id=(
                source_bundle
                if len(
                    {
                        json.dumps(
                            dict(item.applicability),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        for item in records
                        if (item.bundle_id or f"fact:{item.id}") == source_bundle
                    }
                )
                == 1
                else (
                    f"{source_bundle}:"
                    + hashlib.sha256(applicability_key.encode("utf-8"))
                    .hexdigest()[:12]
                )
            ),
            entity_id=entity_id,
            applicability=dict(items[0].applicability),
            records=tuple(sorted(items, key=lambda item: item.id)),
        )
        for (source_bundle, entity_id, applicability_key), items in sorted(
            groups.items()
        )
    )


def select_joint_tool_candidates(
    *,
    query: str,
    bundles: Sequence[MemoryBundle],
    definitions: Sequence[ToolDefinition],
    routed_tools: Sequence[str] = (),
    max_tools: int = 12,
    schema_token_budget: int = 1_800,
    max_tools_per_domain: int = 3,
    max_fact_recovery_tools: int = 1,
    domain_score_margin: float = 24.0,
) -> tuple[JointToolCandidate, ...]:
    """Rerank a bounded Tool set from query, Facts, routes, and schemas."""

    if (
        max_tools < 1
        or schema_token_budget < 1
        or max_tools_per_domain < 1
        or max_fact_recovery_tools < 0
        or domain_score_margin < 0
    ):
        raise ValueError("Joint selector limits must be positive")
    routed = set(routed_tools)
    query_tokens = _tokens(query)
    ontology = build_tool_memory_ontology(definitions)
    ranked: list[JointToolCandidate] = []
    for tool in ontology.tools:
        query_score = _query_tool_score(query_tokens, tool)
        record_scores = [
            (_record_tool_score(record, tool), record)
            for bundle in bundles
            for record in bundle.records
        ]
        supporting = tuple(record.id for score, record in record_scores if score > 0)
        supporting_records = tuple(
            record for score, record in record_scores if score > 0
        )
        record_score = sum(
            sorted((score for score, _ in record_scores if score > 0), reverse=True)[
                :3
            ]
        )
        # Route membership is preserved separately; this is only a light prior.
        route_score = 8.0 if tool.tool_name in routed else 0.0
        if route_score == 0 and not (query_score > 0 and supporting):
            continue
        score = query_score + record_score + route_score
        if score <= 0:
            continue
        sources = tuple(
            source
            for source, present in (
                ("query", query_score > 0),
                ("fact", bool(supporting)),
                ("route", route_score > 0),
            )
            if present
        )
        schema_tokens = _schema_tokens(tool)
        ranked.append(
            JointToolCandidate(
                tool_name=tool.tool_name,
                domain=tool.domain,
                score=round(score, 6),
                source_record_ids=tuple(dict.fromkeys(supporting)),
                sources=sources,
                query_terms=tuple(sorted(query_tokens & _tool_tokens(tool))),
                fact_predicates=tuple(
                    dict.fromkeys(
                        normalize_identifier(record.predicate)
                        for record in supporting_records
                    )
                ),
                required_slots=tuple(
                    slot.name for slot in tool.slots if slot.required
                ),
                schema_tokens=schema_tokens,
            )
        )
    ordered = sorted(ranked, key=lambda item: (-item.score, item.tool_name))
    selected: list[JointToolCandidate] = []
    used_tokens = 0
    domain_counts: dict[str, int] = {}
    domain_best_scores: dict[str, float] = {}

    def add(candidate: JointToolCandidate) -> bool:
        nonlocal used_tokens
        if len(selected) >= max_tools:
            return False
        if used_tokens + candidate.schema_tokens > schema_token_budget:
            return False
        selected.append(candidate)
        used_tokens += candidate.schema_tokens
        domain_counts[candidate.domain] = domain_counts.get(candidate.domain, 0) + 1
        domain_best_scores[candidate.domain] = max(
            candidate.score,
            domain_best_scores.get(candidate.domain, candidate.score),
        )
        return True

    # Preserve the established query route as the non-regression floor.
    for candidate in ordered:
        if candidate.tool_name in routed:
            add(candidate)

    # Add only the strongest Fact-supported recovery candidates per domain.
    added_recovery_tools = 0
    for candidate in ordered:
        if added_recovery_tools >= max_fact_recovery_tools:
            break
        if candidate.tool_name in routed:
            continue
        domain_count = domain_counts.get(candidate.domain, 0)
        if domain_count >= max_tools_per_domain:
            continue
        domain_best = domain_best_scores.get(candidate.domain, candidate.score)
        if candidate.score < domain_best - domain_score_margin:
            continue
        if add(candidate):
            added_recovery_tools += 1
    return tuple(sorted(selected, key=lambda item: (-item.score, item.tool_name)))


def render_joint_tool_candidate_rationale(
    candidates: Sequence[JointToolCandidate],
) -> str:
    """Render compact evidence so the Agent can prioritize candidate schemas."""

    lines = []
    for candidate in candidates:
        reasons = []
        if candidate.query_terms:
            reasons.append("query=" + ",".join(candidate.query_terms))
        if candidate.fact_predicates:
            reasons.append("facts=" + ",".join(candidate.fact_predicates))
        if "route" in candidate.sources:
            reasons.append("route")
        lines.append(
            f"- {candidate.tool_name}: score={candidate.score:g}; "
            + ("; ".join(reasons) if reasons else "bounded candidate")
        )
    return "\n".join(lines)


def build_memory_tool_plan(
    *,
    query: str,
    records: Sequence[FactMemoryRecord],
    definitions: Sequence[ToolDefinition],
    query_context: FactQueryContext,
    routed_tools: Sequence[str] = (),
    current_state: Mapping[str, Any] | None = None,
    max_tools: int = 12,
    schema_token_budget: int = 1_800,
) -> MemoryToolPlan:
    bundles = build_memory_bundles(records)
    candidates = select_joint_tool_candidates(
        query=query,
        bundles=bundles,
        definitions=definitions,
        routed_tools=routed_tools,
        max_tools=max_tools,
        schema_token_budget=schema_token_budget,
    )
    definitions_by_name = {item.name: item for item in definitions}
    desired_states: list[DesiredStateItem] = []
    calls: list[PlannedToolCall] = []
    rejections: list[ToolMemoryHintRejection] = []
    relative_operation = _relative_operation(query)
    claimed_by_bundle: dict[str, set[str]] = {}

    for candidate in candidates:
        definition = definitions_by_name[candidate.tool_name]
        for bundle in bundles:
            claimed = claimed_by_bundle.setdefault(bundle.bundle_id, set())
            supporting = tuple(
                record
                for record in bundle.records
                if record.id in candidate.source_record_ids
                and record.id not in claimed
            )
            if not supporting:
                continue
            arguments: dict[str, Any] = {}
            conflicts: set[str] = set()
            for record in supporting:
                values = fact_record_arguments(
                    record,
                    definition,
                    query_context,
                    query=query,
                )
                for key, value in values.items():
                    if key in arguments and arguments[key] != value:
                        conflicts.add(key)
                    else:
                        arguments[key] = value
            required = _required_arguments(definition)
            missing = tuple(name for name in required if name not in arguments)
            if relative_operation is not None:
                missing = tuple(
                    dict.fromkeys(
                        (
                            *missing,
                            (
                                "current_state"
                                if current_state is None
                                else "relative_value_resolution"
                            ),
                        )
                    )
                )
            state_status: PlanStatus = (
                "ambiguous" if conflicts else "blocked" if missing else "ready"
            )
            state_ids = []
            for record in supporting:
                state_id = f"state:{len(desired_states) + 1}"
                state_ids.append(state_id)
                desired_states.append(
                    DesiredStateItem(
                        id=state_id,
                        bundle_id=bundle.bundle_id,
                        capability=normalize_identifier(record.predicate),
                        operation=relative_operation or "set",
                        target=_record_target(record),
                        desired_value=record.value,
                        selectors={
                            **dict(record.identity_conditions),
                            **dict(record.applicability),
                        },
                        evidence_record_ids=(record.id,),
                        confidence=record.confidence,
                        status=state_status,
                        unresolved_slots=tuple(sorted((*conflicts, *missing))),
                    )
                )
            if state_status != "ready":
                rejections.append(
                    ToolMemoryHintRejection(
                        record_id=supporting[0].id,
                        code=(
                            "conflicting_bundle_arguments"
                            if conflicts
                            else "incomplete_or_invalid_arguments"
                        ),
                        candidate_tools=(candidate.tool_name,),
                    )
                )
                continue
            try:
                Draft202012Validator(definition.parameters).validate(arguments)
                validate_runtime_argument_semantics(arguments)
            except (ValidationError, ValueError):
                rejections.append(
                    ToolMemoryHintRejection(
                        record_id=supporting[0].id,
                        code="incomplete_or_invalid_arguments",
                        candidate_tools=(candidate.tool_name,),
                    )
                )
                continue
            calls.append(
                PlannedToolCall(
                    tool_name=candidate.tool_name,
                    arguments=arguments,
                    desired_state_ids=tuple(state_ids),
                    record_ids=tuple(record.id for record in supporting),
                )
            )
            claimed.update(record.id for record in supporting)

    ordered_calls = _order_calls(calls)
    status: PlanStatus = (
        "ready"
        if ordered_calls
        else "ambiguous"
        if any(item.status == "ambiguous" for item in desired_states)
        else "blocked"
    )
    return MemoryToolPlan(
        status=status,
        bundles=tuple(bundles),
        candidates=candidates,
        desired_states=tuple(desired_states),
        calls=ordered_calls,
        rejections=tuple(rejections),
        metadata={
            "joint_selection": True,
            "multi_fact_binding": True,
            "current_state_available": current_state is not None,
            "schema_token_budget": schema_token_budget,
            "schema_tokens_selected": sum(item.schema_tokens for item in candidates),
        },
    )


def _record_tool_score(record: FactMemoryRecord, tool: ToolMemoryTool) -> float:
    predicate = normalize_identifier(record.predicate)
    if predicate == tool.topic:
        return 80.0
    if predicate in {slot.name for slot in tool.slots}:
        return 70.0
    predicate_tokens = _tokens(predicate)
    topic_tokens = _tokens(tool.topic)
    slot_tokens = {token for slot in tool.slots for token in _tokens(slot.name)}
    direct_overlap = predicate_tokens & (topic_tokens | slot_tokens)
    if not direct_overlap:
        return 0.0
    hint_tokens = {
        token for hint in record.capability_hints for token in _tokens(hint)
    }
    hint_overlap = hint_tokens & (_tokens(tool.domain) | topic_tokens)
    return float(len(direct_overlap) * 20 + len(hint_overlap) * 4)


def _query_tool_score(query_tokens: set[str], tool: ToolMemoryTool) -> float:
    intent_overlap = query_tokens & _tool_intent_tokens(tool)
    domain_overlap = query_tokens & _tokens(tool.domain)
    description_only = query_tokens & (
        _tokens(tool.description) - _tool_intent_tokens(tool)
    )
    return float(
        len(intent_overlap) * 12
        + len(description_only) * 3
        + len(domain_overlap) * 2
    )


def _tool_intent_tokens(tool: ToolMemoryTool) -> set[str]:
    return {
        *_tokens(tool.topic),
        *_tokens(tool.action),
        *(token for slot in tool.slots for token in _tokens(slot.name)),
    }


def _tool_tokens(tool: ToolMemoryTool) -> set[str]:
    return {
        *_tokens(tool.tool_name),
        *_tokens(tool.domain),
        *_tokens(tool.topic),
        *_tokens(tool.action),
        *_tokens(tool.description),
        *(token for slot in tool.slots for token in _tokens(slot.name)),
    }


def _tokens(value: str) -> set[str]:
    tokens = set()
    for match in _WORD.finditer(value.casefold()):
        normalized = normalize_identifier(match.group(0))
        tokens.update(
            token
            for token in normalized.split("_")
            if token and token not in _LOW_SIGNAL_TOKENS
        )
    return tokens


def _schema_tokens(tool: ToolMemoryTool) -> int:
    payload = json.dumps(
        {
            "name": tool.tool_name,
            "description": tool.description,
            "slots": [
                {"name": slot.name, "required": slot.required, "schema": slot.schema}
                for slot in tool.slots
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return max(1, (len(payload) + 3) // 4)


def _stable_arguments(arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(arguments),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _required_arguments(definition: ToolDefinition) -> tuple[str, ...]:
    required = definition.parameters.get("required", ())
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return ()
    return tuple(str(item) for item in required)


def _record_target(record: FactMemoryRecord) -> str | None:
    target = record.identity_conditions.get("target")
    return str(target) if target is not None else None


def _relative_operation(query: str) -> str | None:
    folded = query.casefold()
    if any(term in folded for term in ("increase", "raise", "higher", "올려")):
        return "increase"
    if any(term in folded for term in ("decrease", "lower", "reduce", "내려")):
        return "decrease"
    return None


def _order_calls(calls: Sequence[PlannedToolCall]) -> tuple[PlannedToolCall, ...]:
    def priority(call: PlannedToolCall) -> tuple[int, str]:
        name = normalize_identifier(call.tool_name)
        if any(term in name for term in ("source", "switch", "power", "enable")):
            return (0, name)
        if any(term in name for term in ("play", "start", "resume")):
            return (2, name)
        return (1, name)

    ordered = sorted(calls, key=priority)
    completed: list[str] = []
    result = []
    for call in ordered:
        result.append(
            PlannedToolCall(
                tool_name=call.tool_name,
                arguments=call.arguments,
                desired_state_ids=call.desired_state_ids,
                record_ids=call.record_ids,
                depends_on=tuple(completed),
            )
        )
        completed.append(call.tool_name)
    return tuple(result)
