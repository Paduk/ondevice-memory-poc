from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from palmclaw_ubuntu.models import (
    ToolDefinition,
    ToolMemoryIdentity,
    ToolMemoryRecord,
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_IDENTIFIER = re.compile(r"[^\w]+", re.UNICODE)
_REPEATED_UNDERSCORE = re.compile(r"_+")
_WHITESPACE = re.compile(r"\s+")
_TOPIC_FILLERS = frozenset({"at", "by", "for", "from", "in", "on", "to", "with"})
_ACTION_TOKENS = frozenset(
    {
        "add",
        "adjust",
        "append",
        "apply",
        "clear",
        "close",
        "create",
        "delete",
        "disable",
        "enable",
        "fetch",
        "get",
        "list",
        "load",
        "lock",
        "merge",
        "move",
        "navigate",
        "open",
        "pause",
        "play",
        "read",
        "remove",
        "reset",
        "resume",
        "save",
        "search",
        "send",
        "set",
        "start",
        "stop",
        "switch",
        "unlock",
        "update",
        "write",
    }
)
_SCOPES = frozenset({"global", "vehicle", "session", "conditional"})
_ENTITY_CONDITION_KEYS = frozenset(
    {
        "driver",
        "occupant",
        "passenger",
        "person",
        "subject",
    }
)
_APPLICABILITY_CONDITION_KEYS = frozenset(
    {
        "climate",
        "context",
        "date",
        "day",
        "location",
        "place",
        "road",
        "season",
        "situation",
        "time",
        "time_of_day",
        "use_case",
        "weather",
    }
)


@dataclass(frozen=True)
class ToolMemorySlot:
    name: str
    path: str
    required: bool
    schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolMemoryTool:
    tool_name: str
    description: str
    namespace: str
    domain: str
    action: str
    topic: str
    slots: tuple[ToolMemorySlot, ...]


@dataclass(frozen=True)
class ToolMemoryOntology:
    tools: tuple[ToolMemoryTool, ...]

    def definition(self, tool_name: str) -> ToolMemoryTool | None:
        return next(
            (item for item in self.tools if item.tool_name == tool_name),
            None,
        )

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({item.domain for item in self.tools}))


def build_tool_memory_ontology(
    definitions: Sequence[ToolDefinition],
) -> ToolMemoryOntology:
    tools = tuple(
        sorted(
            (_tool_memory_definition(definition) for definition in definitions),
            key=lambda item: item.tool_name,
        )
    )
    return ToolMemoryOntology(tools=tools)


def normalize_tool_memory_identity(
    identity: ToolMemoryIdentity,
) -> ToolMemoryIdentity:
    scope = normalize_identifier(identity.scope)
    if scope not in _SCOPES:
        allowed = ", ".join(sorted(_SCOPES))
        raise ValueError(f"Unsupported memory scope: {identity.scope}; use {allowed}")
    user_id = normalize_identifier(identity.user_id)
    tool_domain = normalize_identifier(identity.tool_domain)
    topic = normalize_identifier(identity.topic)
    scope_key = normalize_identifier(identity.scope_key)
    for field_name, value in (
        ("user_id", user_id),
        ("tool_domain", tool_domain),
        ("topic", topic),
        ("scope_key", scope_key),
    ):
        if not value:
            raise ValueError(f"{field_name} cannot be empty")
    conditions = _normalize_conditions(identity.conditions)
    return ToolMemoryIdentity(
        user_id=user_id,
        tool_domain=tool_domain,
        topic=topic,
        scope=scope,
        scope_key=scope_key,
        conditions=conditions,
    )


def tool_memory_record_key(identity: ToolMemoryIdentity) -> str:
    normalized = normalize_tool_memory_identity(identity)
    canonical = json.dumps(
        {
            "conditions": normalized.conditions,
            "scope": normalized.scope,
            "scope_key": normalized.scope_key,
            "tool_domain": normalized.tool_domain,
            "topic": normalized.topic,
            "user_id": normalized.user_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_tool_memory_conditions(
    conditions: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = _normalize_conditions(conditions)
    identity_conditions = {
        key: value
        for key, value in normalized.items()
        if key not in _APPLICABILITY_CONDITION_KEYS
    }
    applicability = {
        key: value
        for key, value in normalized.items()
        if key in _APPLICABILITY_CONDITION_KEYS
    }
    return identity_conditions, applicability


def tool_memory_entity_id(identity: ToolMemoryIdentity) -> str | None:
    normalized = normalize_tool_memory_identity(identity)
    for key in ("person", "driver", "occupant", "passenger", "subject"):
        value = normalized.conditions.get(key)
        if isinstance(value, str):
            entity_id = normalize_identifier(value)
            if entity_id:
                return entity_id
    return None


def tool_memory_identity_family_key(identity: ToolMemoryIdentity) -> str:
    normalized = normalize_tool_memory_identity(identity)
    identity_conditions, _ = split_tool_memory_conditions(
        normalized.conditions
    )
    canonical = json.dumps(
        {
            "entity_id": tool_memory_entity_id(normalized),
            "identity_conditions": identity_conditions,
            "scope": normalized.scope,
            "scope_key": normalized.scope_key,
            "tool_domain": normalized.tool_domain,
            "topic": normalized.topic,
            "user_id": normalized.user_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonicalize_tool_memory_identity_aliases(
    identity: ToolMemoryIdentity,
    active_records: Sequence[ToolMemoryRecord],
) -> ToolMemoryIdentity:
    """Resolve learned entity aliases without a predefined name dictionary."""
    normalized = normalize_tool_memory_identity(identity)
    known_entities: set[str] = set()
    for record in active_records:
        for key, value in record.conditions.items():
            normalized_key = normalize_identifier(str(key))
            if normalized_key not in _ENTITY_CONDITION_KEYS:
                continue
            if isinstance(value, str) and normalize_identifier(value):
                known_entities.add(value)
    conditions = dict(normalized.conditions)
    for key, value in tuple(conditions.items()):
        if key not in _ENTITY_CONDITION_KEYS or not isinstance(value, str):
            continue
        compatible = sorted(
            candidate
            for candidate in known_entities
            if entity_alias_compatible(value, candidate)
        )
        if compatible and all(
            entity_alias_compatible(left, right)
            for index, left in enumerate(compatible)
            for right in compatible[index + 1 :]
        ):
            conditions[key] = max(
                compatible,
                key=lambda candidate: (
                    len(normalize_identifier(candidate).split("_")),
                    len(candidate),
                ),
            )
    return ToolMemoryIdentity(
        user_id=normalized.user_id,
        tool_domain=normalized.tool_domain,
        topic=normalized.topic,
        scope=normalized.scope,
        scope_key=normalized.scope_key,
        conditions=conditions,
    )


def entity_alias_compatible(first: str, second: str) -> bool:
    first_tokens = tuple(normalize_identifier(first).split("_"))
    second_tokens = tuple(normalize_identifier(second).split("_"))
    if not first_tokens or not second_tokens:
        return False
    first_set = set(first_tokens)
    second_set = set(second_tokens)
    return first_set <= second_set or second_set <= first_set


def normalize_identifier(value: str) -> str:
    expanded = _CAMEL_BOUNDARY.sub("_", unicodedata.normalize("NFKC", value.strip()))
    normalized = _NON_IDENTIFIER.sub("_", expanded.casefold())
    return _REPEATED_UNDERSCORE.sub("_", normalized).strip("_")


def _tool_memory_definition(definition: ToolDefinition) -> ToolMemoryTool:
    tokens = _identifier_tokens(definition.name)
    action_index = next(
        (index for index, token in enumerate(tokens) if token in _ACTION_TOKENS),
        None,
    )
    if action_index is None:
        namespace = ""
        domain_tokens = tokens[:1] or ("general",)
        action = "invoke"
        topic_tokens = tokens[1:]
    else:
        prefix = tokens[:action_index]
        action = tokens[action_index]
        topic_tokens = tokens[action_index + 1 :]
        namespace_tokens, domain_tokens = _namespace_and_domain(prefix, topic_tokens)
        namespace = "_".join(namespace_tokens)
    domain = "_".join(domain_tokens) or "general"
    topic = "_".join(topic_tokens) or action
    properties = definition.parameters.get("properties", {})
    required = {
        str(item)
        for item in definition.parameters.get("required", [])
        if isinstance(item, str)
    }
    if topic_tokens and set(topic_tokens) <= _TOPIC_FILLERS:
        candidates = sorted(required) or sorted(
            str(name)
            for name in (
                properties.keys() if isinstance(properties, Mapping) else ()
            )
        )
        topic = normalize_identifier(candidates[0]) if candidates else action
    slots = tuple(
        ToolMemorySlot(
            name=normalize_identifier(str(name)),
            path=f"{domain}.{topic}.{normalize_identifier(str(name))}",
            required=str(name) in required,
            schema=dict(schema) if isinstance(schema, Mapping) else {},
        )
        for name, schema in sorted(
            properties.items() if isinstance(properties, Mapping) else ()
        )
    )
    return ToolMemoryTool(
        tool_name=definition.name,
        description=definition.description,
        namespace=namespace,
        domain=domain,
        action=action,
        topic=topic,
        slots=slots,
    )


def _namespace_and_domain(
    prefix: tuple[str, ...],
    topic_tokens: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not prefix:
        fallback = topic_tokens[:1] or ("general",)
        return (), fallback
    first = prefix[0]
    if len(prefix) > 1 and first.endswith(("api", "control", "service", "tool")):
        return (first,), prefix[1:]
    return (), prefix


def _identifier_tokens(value: str) -> tuple[str, ...]:
    normalized = normalize_identifier(value)
    return tuple(token for token in normalized.split("_") if token)


def _normalize_conditions(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        normalized_key = normalize_identifier(str(key))
        if not normalized_key:
            raise ValueError("Condition keys cannot be empty")
        if normalized_key in normalized:
            raise ValueError(f"Duplicate normalized condition key: {normalized_key}")
        normalized[normalized_key] = _normalize_condition_value(item)
    return normalized


def _normalize_condition_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_conditions(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        normalized = [_normalize_condition_value(item) for item in value]
        unique = {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in normalized
        }
        return sorted(
            unique.values(),
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).strip()
        return _WHITESPACE.sub(" ", normalized).casefold()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"Unsupported condition value: {type(value).__name__}")
