from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from palmclaw_ubuntu.models import (
    FactMemoryRecord,
    FactQueryContext,
    ToolDefinition,
    ToolMemoryRecord,
)
from palmclaw_ubuntu.tool_memory_schema import (
    ToolMemoryTool,
    build_tool_memory_ontology,
    normalize_identifier,
)


@dataclass(frozen=True)
class ToolMemoryExecutionHint:
    """A schema-valid candidate Tool call derived from one selected record."""

    record_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    confidence: float
    tool_domain: str
    topic: str
    record_ids: tuple[str, ...] = ()
    desired_state_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "record_id": self.record_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "confidence": self.confidence,
            "tool_domain": self.tool_domain,
            "topic": self.topic,
        }
        if self.record_ids:
            payload["record_ids"] = list(self.record_ids)
        if self.desired_state_ids:
            payload["desired_state_ids"] = list(self.desired_state_ids)
        if self.depends_on:
            payload["depends_on"] = list(self.depends_on)
        return payload


@dataclass(frozen=True)
class ToolMemoryHintRejection:
    record_id: str
    code: str
    candidate_tools: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "code": self.code,
            "candidate_tools": list(self.candidate_tools),
        }


@dataclass(frozen=True)
class ToolMemoryHintResult:
    hints: tuple[ToolMemoryExecutionHint, ...]
    rejections: tuple[ToolMemoryHintRejection, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "tool_memory_execution_hint_count": len(self.hints),
            "tool_memory_execution_hints": [
                hint.as_dict() for hint in self.hints
            ],
            "tool_memory_execution_hint_rejection_count": len(self.rejections),
            "tool_memory_execution_hint_rejections": [
                rejection.as_dict() for rejection in self.rejections
            ],
        }


def build_tool_memory_execution_hints(
    records: Sequence[ToolMemoryRecord],
    definitions: Sequence[ToolDefinition],
) -> ToolMemoryHintResult:
    """Convert selected memory records into unambiguous, schema-valid hints.

    The conversion is deliberately deterministic. A hint is emitted only when
    one best Tool in the record's domain can be identified and every required
    argument can be sourced from the record value or explicit conditions.
    """

    definitions_by_name = {definition.name: definition for definition in definitions}
    ontology = build_tool_memory_ontology(definitions)
    hints: list[ToolMemoryExecutionHint] = []
    rejections: list[ToolMemoryHintRejection] = []

    for record in records:
        ranked = sorted(
            (
                (score, tool)
                for tool in ontology.tools
                if (score := _tool_match_score(record, tool)) > 0
            ),
            key=lambda item: (-item[0], item[1].tool_name),
        )
        if not ranked:
            rejections.append(
                ToolMemoryHintRejection(
                    record_id=record.id,
                    code="no_compatible_tool",
                )
            )
            continue

        best_score = ranked[0][0]
        best_tools = [tool for score, tool in ranked if score == best_score]
        valid: list[tuple[ToolMemoryTool, dict[str, Any]]] = []
        invalid_names: list[str] = []
        for tool in best_tools:
            definition = definitions_by_name[tool.tool_name]
            arguments = _record_arguments(record, definition)
            try:
                Draft202012Validator(definition.parameters).validate(arguments)
            except ValidationError:
                invalid_names.append(tool.tool_name)
                continue
            valid.append((tool, arguments))

        if len(valid) != 1:
            candidate_names = tuple(
                sorted(
                    {
                        *(tool.tool_name for tool in best_tools),
                        *invalid_names,
                    }
                )
            )
            rejections.append(
                ToolMemoryHintRejection(
                    record_id=record.id,
                    code=(
                        "ambiguous_tool_mapping"
                        if len(valid) > 1
                        else "incomplete_or_invalid_arguments"
                    ),
                    candidate_tools=candidate_names,
                )
            )
            continue

        tool, arguments = valid[0]
        hints.append(
            ToolMemoryExecutionHint(
                record_id=record.id,
                tool_name=tool.tool_name,
                arguments=arguments,
                confidence=record.confidence,
                tool_domain=record.tool_domain,
                topic=record.topic,
            )
        )

    return ToolMemoryHintResult(
        hints=tuple(hints),
        rejections=tuple(rejections),
    )


def build_fact_memory_execution_hints(
    records: Sequence[FactMemoryRecord],
    definitions: Sequence[ToolDefinition],
    query_context: FactQueryContext,
    *,
    query: str = "",
    routed_tools: Sequence[str] = (),
) -> ToolMemoryHintResult:
    """Late-bind persistent facts to request-local, schema-valid Tool calls.

    Facts contain durable values and entities. Runtime arguments such as seat,
    light, side, and zone are supplied from the current request instead of
    being persisted as part of the Fact identity.
    """

    definitions_by_name = {definition.name: definition for definition in definitions}
    ontology = build_tool_memory_ontology(definitions)
    routed = set(routed_tools)
    hints: list[ToolMemoryExecutionHint] = []
    rejections: list[ToolMemoryHintRejection] = []

    for record in records:
        ranked = sorted(
            (
                (
                    _fact_tool_match_score(record, tool)
                    + (25 if tool.tool_name in routed else 0),
                    tool,
                )
                for tool in ontology.tools
                if _fact_tool_match_score(record, tool) > 0
            ),
            key=lambda item: (-item[0], item[1].tool_name),
        )
        if not ranked:
            rejections.append(
                ToolMemoryHintRejection(
                    record_id=record.id,
                    code="no_compatible_tool",
                )
            )
            continue

        best_score = ranked[0][0]
        best_tools = [tool for score, tool in ranked if score == best_score]
        valid: list[tuple[ToolMemoryTool, dict[str, Any]]] = []
        invalid: list[str] = []
        for tool in best_tools:
            definition = definitions_by_name[tool.tool_name]
            arguments = fact_record_arguments(
                record,
                definition,
                query_context,
                query=query,
            )
            try:
                Draft202012Validator(definition.parameters).validate(arguments)
                _validate_runtime_argument_semantics(arguments)
            except (ValidationError, ValueError):
                invalid.append(tool.tool_name)
                continue
            valid.append((tool, arguments))

        if len(valid) != 1:
            rejections.append(
                ToolMemoryHintRejection(
                    record_id=record.id,
                    code=(
                        "ambiguous_tool_mapping"
                        if len(valid) > 1
                        else "incomplete_or_invalid_arguments"
                    ),
                    candidate_tools=tuple(
                        sorted(
                            {
                                *(tool.tool_name for tool in best_tools),
                                *invalid,
                            }
                        )
                    ),
                )
            )
            continue

        tool, arguments = valid[0]
        hints.append(
            ToolMemoryExecutionHint(
                record_id=record.id,
                tool_name=tool.tool_name,
                arguments=arguments,
                confidence=record.confidence,
                tool_domain=tool.domain,
                topic=tool.topic,
            )
        )
    return ToolMemoryHintResult(
        hints=tuple(hints),
        rejections=tuple(rejections),
    )


def render_tool_memory_execution_hints(
    hints: Sequence[ToolMemoryExecutionHint],
) -> str:
    return json.dumps(
        [hint.as_dict() for hint in hints],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def routed_tool_names(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Read selector output from retrieval metadata without using gold labels."""

    route = metadata.get("tool_memory_route")
    if not isinstance(route, Mapping):
        return ()
    routes = route.get("routes")
    if not isinstance(routes, Sequence) or isinstance(routes, (str, bytes)):
        return ()
    names: list[str] = []
    for item in routes:
        if not isinstance(item, Mapping):
            continue
        tool_names = item.get("tool_names")
        if not isinstance(tool_names, Sequence) or isinstance(
            tool_names,
            (str, bytes),
        ):
            continue
        names.extend(str(name) for name in tool_names if str(name))
    return tuple(dict.fromkeys(names))


def _tool_match_score(
    record: ToolMemoryRecord,
    tool: ToolMemoryTool,
) -> int:
    if normalize_identifier(record.tool_domain) != tool.domain:
        return 0
    topic = normalize_identifier(record.topic)
    if topic == tool.topic:
        return 100
    slot_names = {slot.name for slot in tool.slots}
    if topic in slot_names:
        return 90
    topic_tokens = set(topic.split("_"))
    tool_tokens = set(tool.topic.split("_"))
    overlap = topic_tokens & tool_tokens
    if overlap and (
        topic_tokens <= tool_tokens
        or tool_tokens <= topic_tokens
        or len(overlap) >= 2
    ):
        return 60 + len(overlap)
    return 0


def _record_arguments(
    record: ToolMemoryRecord,
    definition: ToolDefinition,
) -> dict[str, Any]:
    properties = definition.parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        return {}
    names_by_normalized = {
        normalize_identifier(str(name)): str(name) for name in properties
    }
    arguments: dict[str, Any] = {}

    if isinstance(record.value, Mapping):
        _copy_matching_values(
            arguments,
            record.value,
            names_by_normalized,
        )
    else:
        topic = normalize_identifier(record.topic)
        target = names_by_normalized.get(topic)
        if target is None and len(names_by_normalized) == 1:
            target = next(iter(names_by_normalized.values()))
        if target is not None:
            arguments[target] = record.value

    for source in (
        record.identity_conditions,
        record.applicability,
        record.conditions,
    ):
        _copy_matching_values(
            arguments,
            source,
            names_by_normalized,
            missing_only=True,
        )
    return arguments


def _copy_matching_values(
    target: dict[str, Any],
    source: Mapping[str, Any],
    names_by_normalized: Mapping[str, str],
    *,
    missing_only: bool = False,
) -> None:
    for key, value in source.items():
        property_name = names_by_normalized.get(normalize_identifier(str(key)))
        if property_name is None:
            continue
        if missing_only and property_name in target:
            continue
        target[property_name] = value


_RUNTIME_ARGUMENTS = frozenset(
    {"driver", "light", "occupant", "passenger", "person", "seat", "side", "zone"}
)
_ROLE_ARGUMENTS = frozenset({"light", "occupant", "seat", "zone"})
_RUNTIME_ALLOWED = {
    "light": frozenset({"driver", "passenger", "rear_left", "rear_right"}),
    "occupant": frozenset({"driver", "passenger", "rear_left", "rear_right"}),
    "seat": frozenset({"driver", "passenger", "rear_left", "rear_right"}),
    "side": frozenset({"left", "right", "both"}),
    "zone": frozenset(
        {"driver", "passenger", "rear_left", "rear_right", "all"}
    ),
}
_VALUE_ALIASES = {
    "inside_air": "inside",
    "inside air": "inside",
    "recirculate": "inside",
    "recirculation": "inside",
    "outside_air": "outside",
    "outside air": "outside",
    "fresh_air": "outside",
    "fresh air": "outside",
}


def _fact_tool_match_score(
    record: FactMemoryRecord,
    tool: ToolMemoryTool,
) -> int:
    predicate = normalize_identifier(record.predicate)
    if predicate == tool.topic:
        return 120
    slot_names = {slot.name for slot in tool.slots}
    if predicate in slot_names:
        return 105
    predicate_tokens = set(predicate.split("_"))
    topic_tokens = set(tool.topic.split("_"))
    tool_tokens = {
        *topic_tokens,
        *tool.domain.split("_"),
        *(token for slot in tool.slots for token in slot.name.split("_")),
    }
    hint_tokens = {
        token
        for hint in record.capability_hints
        for token in normalize_identifier(hint).split("_")
    }
    overlap = predicate_tokens & tool_tokens
    hint_overlap = hint_tokens & tool_tokens
    if not overlap and not hint_overlap:
        return 0
    return len(overlap) * 15 + len(hint_overlap) * 5


def fact_record_arguments(
    record: FactMemoryRecord,
    definition: ToolDefinition,
    query_context: FactQueryContext,
    *,
    query: str,
) -> dict[str, Any]:
    properties = definition.parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        return {}
    names_by_normalized = {
        normalize_identifier(str(name)): str(name) for name in properties
    }
    arguments: dict[str, Any] = {}
    if isinstance(record.value, Mapping):
        _copy_matching_values(arguments, record.value, names_by_normalized)
    else:
        predicate = normalize_identifier(record.predicate)
        target = names_by_normalized.get(predicate)
        if target is None:
            predicate_tokens = set(predicate.split("_"))
            semantic_slots = [
                name
                for normalized, name in names_by_normalized.items()
                if normalized not in _RUNTIME_ARGUMENTS
                and (
                    normalized in predicate_tokens
                    or set(normalized.split("_")) <= predicate_tokens
                )
            ]
            if len(semantic_slots) == 1:
                target = semantic_slots[0]
        if target is None:
            value_slots = [
                name
                for normalized, name in names_by_normalized.items()
                if normalized not in _RUNTIME_ARGUMENTS
            ]
            if len(value_slots) == 1:
                target = value_slots[0]
        if target is not None:
            arguments[target] = _canonical_value(record.value)

    for source in (record.identity_conditions, record.applicability):
        _copy_matching_values(
            arguments,
            source,
            names_by_normalized,
            missing_only=True,
        )

    role = query_context.entity_roles.get(record.entity_id)
    folded = query.casefold()
    for normalized, property_name in names_by_normalized.items():
        if property_name in arguments:
            continue
        if normalized in _ROLE_ARGUMENTS:
            bound = _role_argument(normalized, role, folded)
            if bound is not None:
                arguments[property_name] = bound
        elif normalized == "side":
            if "both" in folded or "mirrors" in folded or "양쪽" in folded:
                arguments[property_name] = "both"
            elif "left" in folded or "왼쪽" in folded:
                arguments[property_name] = "left"
            elif "right" in folded or "오른쪽" in folded:
                arguments[property_name] = "right"

    for name, value in tuple(arguments.items()):
        arguments[name] = _canonical_value(value)
    return arguments


def validate_runtime_argument_semantics(arguments: Mapping[str, Any]) -> None:
    """Reject request-local selector values outside the supported vocabulary."""

    _validate_runtime_argument_semantics(arguments)


def _role_argument(
    argument_name: str,
    role: str | None,
    query: str,
) -> str | None:
    if argument_name == "zone" and (
        "all zone" in query
        or "all air" in query
        or "every zone" in query
        or "전체" in query
    ):
        return "all"
    if role in {"driver", "passenger", "rear_left", "rear_right"}:
        return role
    for candidate in ("rear_left", "rear_right", "passenger", "driver"):
        rendered = candidate.replace("_", " ")
        if rendered in query:
            return candidate
    return None


def _canonical_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    folded = value.casefold().strip()
    return _VALUE_ALIASES.get(folded, value)


def _validate_runtime_argument_semantics(arguments: Mapping[str, Any]) -> None:
    for name, allowed in _RUNTIME_ALLOWED.items():
        value = arguments.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or normalize_identifier(value) not in allowed:
            raise ValueError(f"Unsupported runtime argument {name}={value!r}")
