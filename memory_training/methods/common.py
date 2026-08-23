"""Strict JSON and prompt helpers shared by memory methods."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

VALID_DECISIONS = frozenset({"NO_OP", "UPDATE"})
VALID_REASON_CODES = frozenset(
    {
        "NO_NEW_VEHICLE_FACT",
        "NEW_VEHICLE_MEMORY",
        "UPDATED_VEHICLE_MEMORY",
    }
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Model output must be one strict JSON object") from exc
    if not isinstance(value, dict):
        raise TypeError("Model output must be a JSON object")
    return value


def require_exact_fields(value: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"Output fields differ: expected={sorted(expected)}, actual={sorted(actual)}"
        )


def require_decision(value: Mapping[str, Any]) -> str:
    decision = value.get("decision")
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision: {decision}")
    return str(decision)


def format_turn_input(
    *, memory_input: Mapping[str, Any], row: Mapping[str, Any]
) -> str:
    turn = row.get("current_turn")
    if not isinstance(turn, Mapping):
        raise TypeError("Training row current_turn must be an object")
    payload = {
        "memory_state": dict(memory_input),
        "current_turn": {
            "turn_id": row.get("turn_id"),
            "timestamp": row.get("timestamp"),
            "speaker_id": turn.get("speaker_id"),
            "speaker_name": turn.get("speaker_name"),
            "text": turn.get("text"),
        },
    }
    return compact_json(payload)


def target_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise TypeError("Training row target must be an object")
    return target
