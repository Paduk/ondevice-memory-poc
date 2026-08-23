"""Whole-memory Summary method without decoded reasoning."""

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
from .operations import normalize_memory

SUMMARY_SYSTEM_PROMPT = """You maintain durable vehicle memory.
Given the previous full memory and current turn, return exactly one JSON object.
If no durable vehicle fact changes, return {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","next_memory":"<complete updated memory>"}.
Preserve every still-valid fact. Do not emit reasoning, markdown fences, or extra fields."""


class SummaryMethod(MemoryMethod[str]):
    name = "summary"
    source_view = "summary"
    system_prompt = SUMMARY_SYSTEM_PROMPT

    def initial_state(self) -> str:
        return ""

    def format_input(self, row: Mapping[str, Any]) -> str:
        memory_input = row.get("input")
        if not isinstance(memory_input, Mapping) or set(memory_input) != {
            "previous_memory"
        }:
            raise ValueError("Summary input requires only previous_memory")
        return format_turn_input(memory_input=memory_input, row=row)

    def format_target(self, row: Mapping[str, Any]) -> str:
        target = target_mapping(row)
        decision = require_decision(target)
        if decision == "NO_OP":
            return compact_json({"decision": "NO_OP"})
        memory = target.get("next_memory")
        if not isinstance(memory, str):
            raise TypeError("Summary UPDATE requires next_memory string")
        return compact_json({"decision": "UPDATE", "next_memory": memory})

    def parse_output(self, text: str) -> ParsedMemoryOutput:
        value = parse_json_object(text)
        decision = require_decision(value)
        if decision == "NO_OP":
            require_exact_fields(value, {"decision"})
            return ParsedMemoryOutput(decision, {})
        require_exact_fields(value, {"decision", "next_memory"})
        memory = value["next_memory"]
        if not isinstance(memory, str):
            raise TypeError("next_memory must be a string")
        return ParsedMemoryOutput(decision, {"next_memory": normalize_memory(memory)})

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
        memory = output.payload.get("next_memory")
        if not isinstance(memory, str):
            raise TypeError("Parsed Summary UPDATE has no next_memory")
        memory = normalize_memory(memory)
        if memory == state:
            raise ValueError("Summary UPDATE made no state change")
        return memory

    def materialize_memory(self, state: str) -> str:
        return normalize_memory(state)
