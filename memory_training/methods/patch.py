"""Immediate deterministic exact-block Patch method."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import MemoryMethod, ParsedMemoryOutput
from .common import (
    compact_json,
    format_turn_input,
    parse_json_object,
    require_decision,
    require_exact_fields,
    target_mapping,
)
from .operations import apply_operations, normalize_memory, validate_operations

PATCH_SYSTEM_PROMPT = """You maintain durable vehicle memory using exact-block patches.
Given previous_memory and the current turn, return exactly one JSON object.
If nothing changes, return {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","operations":[...]}.
Each operation has exactly op, target, content. op is add, replace, or delete.
Targets must copy a unique complete block from previous_memory exactly.
Do not emit reasoning, markdown fences, an updated full memory, or extra fields."""


class PatchMethod(MemoryMethod[str]):
    name = "patch"
    source_view = "patch"
    system_prompt = PATCH_SYSTEM_PROMPT

    def initial_state(self) -> str:
        return ""

    def format_input(self, row: Mapping[str, Any]) -> str:
        memory_input = row.get("input")
        if not isinstance(memory_input, Mapping) or set(memory_input) != {
            "previous_memory"
        }:
            raise ValueError("Patch input requires only previous_memory")
        return format_turn_input(memory_input=memory_input, row=row)

    def format_target(self, row: Mapping[str, Any]) -> str:
        target = target_mapping(row)
        decision = require_decision(target)
        if decision == "NO_OP":
            return compact_json({"decision": "NO_OP"})
        operations = validate_operations(target.get("operations"))
        return compact_json({"decision": "UPDATE", "operations": operations})

    def parse_output(self, text: str) -> ParsedMemoryOutput:
        value = parse_json_object(text)
        decision = require_decision(value)
        if decision == "NO_OP":
            require_exact_fields(value, {"decision"})
            return ParsedMemoryOutput(decision, {})
        require_exact_fields(value, {"decision", "operations"})
        operations = validate_operations(value["operations"])
        return ParsedMemoryOutput(decision, {"operations": operations})

    def apply_output(
        self,
        state: str,
        output: ParsedMemoryOutput,
        *,
        turn_id: str | None = None,
    ) -> str:
        del turn_id
        state = normalize_memory(state)
        if output.decision == "NO_OP":
            return state
        operations = output.payload.get("operations")
        memory, _ = apply_operations(state, operations)
        return memory

    def materialize_memory(self, state: str) -> str:
        return normalize_memory(state)
