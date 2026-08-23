"""Appended Delta method with deterministic five-UPDATE compaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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

DELTA_SYSTEM_PROMPT = """You maintain vehicle memory as appended exact-block deltas.
The input contains a base_summary, pending_deltas, and the current turn.
If nothing changes, return {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","operations":[...]}.
Each operation has exactly op, target, content. op is add, replace, or delete.
Targets refer to the materialized base_summary plus pending_deltas.
The runtime deterministically compacts every five UPDATEs. Do not compact or emit reasoning."""


@dataclass(frozen=True)
class DeltaBatch:
    turn_id: str
    operations: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class DeltaState:
    base_summary: str = ""
    pending_deltas: tuple[DeltaBatch, ...] = ()
    updates_since_compaction: int = 0


class DeltaMethod(MemoryMethod[DeltaState]):
    name = "delta"
    source_view = "delta"
    system_prompt = DELTA_SYSTEM_PROMPT

    def __init__(self, compaction_interval: int = 5) -> None:
        if compaction_interval < 1:
            raise ValueError("compaction_interval must be positive")
        self.compaction_interval = compaction_interval

    def initial_state(self) -> DeltaState:
        return DeltaState()

    def format_input(self, row: Mapping[str, Any]) -> str:
        memory_input = row.get("input")
        expected = {"base_summary", "pending_deltas", "updates_since_compaction"}
        if not isinstance(memory_input, Mapping) or set(memory_input) != expected:
            raise ValueError("Delta input has invalid fields")
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
        state: DeltaState,
        output: ParsedMemoryOutput,
        *,
        turn_id: str | None = None,
    ) -> DeltaState:
        if output.decision == "NO_OP":
            return state
        if not turn_id:
            raise ValueError("Delta UPDATE requires current turn_id")
        operations = validate_operations(output.payload.get("operations"))
        pending = (*state.pending_deltas, DeltaBatch(turn_id, operations))
        count = state.updates_since_compaction + 1
        if count < self.compaction_interval:
            return DeltaState(normalize_memory(state.base_summary), pending, count)
        memory = normalize_memory(state.base_summary)
        for batch in pending:
            memory, _ = apply_operations(memory, batch.operations)
        return DeltaState(base_summary=memory)

    def materialize_memory(self, state: DeltaState) -> str:
        memory = normalize_memory(state.base_summary)
        for batch in state.pending_deltas:
            memory, _ = apply_operations(memory, batch.operations)
        return memory
