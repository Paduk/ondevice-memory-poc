"""Whole-memory Summary method with one decoded reason sentence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import ParsedMemoryOutput
from .common import (
    VALID_REASON_CODES,
    compact_json,
    parse_json_object,
    require_decision,
    require_exact_fields,
    target_mapping,
)
from .operations import normalize_memory
from .summary import SummaryMethod

SUMMARY_REASON_SYSTEM_PROMPT = """You maintain durable vehicle memory.
Return exactly one JSON object in reason_code, reason, decision, then memory order.
reason_code must be NO_NEW_VEHICLE_FACT, NEW_VEHICLE_MEMORY, or UPDATED_VEHICLE_MEMORY.
reason must be one short sentence explaining the current-turn decision.
For NO_OP omit memory. For UPDATE include the complete updated memory as next_memory.
Preserve every still-valid fact. Do not emit markdown fences or extra fields."""


class SummaryReasonMethod(SummaryMethod):
    name = "summary_reason"
    source_view = "summary"
    system_prompt = SUMMARY_REASON_SYSTEM_PROMPT

    def format_target(self, row: Mapping[str, Any]) -> str:
        target = target_mapping(row)
        decision = require_decision(target)
        reason_code = target.get("reason_code")
        reason = target.get("reason")
        _validate_reason(reason_code, reason)
        value: dict[str, Any] = {
            "reason_code": reason_code,
            "reason": reason,
            "decision": decision,
        }
        if decision == "UPDATE":
            memory = target.get("next_memory")
            if not isinstance(memory, str):
                raise TypeError("Summary+Reason UPDATE requires next_memory string")
            value["next_memory"] = memory
        return compact_json(value)

    def parse_output(self, text: str) -> ParsedMemoryOutput:
        value = parse_json_object(text)
        decision = require_decision(value)
        expected = {"reason_code", "reason", "decision"}
        if decision == "UPDATE":
            expected.add("next_memory")
        require_exact_fields(value, expected)
        reason_code = value["reason_code"]
        reason = value["reason"]
        _validate_reason(reason_code, reason)
        payload = {}
        if decision == "UPDATE":
            memory = value["next_memory"]
            if not isinstance(memory, str):
                raise TypeError("next_memory must be a string")
            payload["next_memory"] = normalize_memory(memory)
        return ParsedMemoryOutput(
            decision,
            payload,
            reason_code=str(reason_code),
            reason=str(reason),
        )


def _validate_reason(reason_code: Any, reason: Any) -> None:
    if reason_code not in VALID_REASON_CODES:
        raise ValueError(f"Invalid reason_code: {reason_code}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty sentence")
    if "\n" in reason.strip():
        raise ValueError("reason must be one line")
