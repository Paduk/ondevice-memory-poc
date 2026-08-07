from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from palmclaw_ubuntu.models import FactMemoryIdentity
from palmclaw_ubuntu.tool_memory_schema import normalize_identifier


def normalize_fact_memory_identity(
    identity: FactMemoryIdentity,
) -> FactMemoryIdentity:
    """Normalize only the stable identity fields used to version a fact."""
    user_id = normalize_identifier(identity.user_id)
    entity_id = normalize_identifier(identity.entity_id)
    predicate = normalize_identifier(identity.predicate)
    for field_name, value in (
        ("user_id", user_id),
        ("entity_id", entity_id),
        ("predicate", predicate),
    ):
        if not value:
            raise ValueError(f"{field_name} cannot be empty")
    return FactMemoryIdentity(
        user_id=user_id,
        entity_id=entity_id,
        predicate=predicate,
        identity_conditions=_normalize_mapping(identity.identity_conditions),
        applicability=_normalize_mapping(identity.applicability),
    )


def fact_memory_record_key(identity: FactMemoryIdentity) -> str:
    """Return a deterministic key that deliberately excludes the fact value."""
    normalized = normalize_fact_memory_identity(identity)
    canonical = json.dumps(
        {
            "applicability": normalized.applicability,
            "entity_id": normalized.entity_id,
            "identity_conditions": normalized.identity_conditions,
            "predicate": normalized.predicate,
            "user_id": normalized.user_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_capability_hints(hints: Sequence[str]) -> tuple[str, ...]:
    normalized = {
        normalize_identifier(str(hint))
        for hint in hints
        if normalize_identifier(str(hint))
    }
    return tuple(sorted(normalized))


def _normalize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        normalized_key: _normalize_condition_value(item)
        for key, item in value.items()
        if (normalized_key := normalize_identifier(str(key)))
    }


def _normalize_condition_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_condition_value(item) for item in value]
    if isinstance(value, str):
        normalized = normalize_identifier(value)
        return normalized or value.strip()
    return value
