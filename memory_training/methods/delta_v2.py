"""Compact appended-Delta method with deterministic five-UPDATE compaction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

DELTA_V2_SYSTEM_PROMPT = """You maintain vehicle memory as compact appended deltas.
The input contains base_summary, zero to four pending_updates, and the current turn.
If nothing changes, return exactly {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","operations":[...]}.
Use compact operation arrays: ["add","content"] appends a block;
["add","target","content"] inserts after an exact target;
["replace","target","content"] replaces an exact target; and
["delete","target"] deletes an exact target.
Targets must be unique complete blocks in the materialized memory.
The runtime compacts every five UPDATEs. Do not emit reasoning or extra fields."""


@dataclass(frozen=True)
class DeltaV2State:
    base_summary: str = ""
    pending_updates: tuple[tuple[dict[str, str], ...], ...] = ()


class DeltaV2Method(MemoryMethod[DeltaV2State]):
    name = "delta_v2"
    # A data root selects either legacy Delta or Delta-v2; both occupy the
    # catalog's aligned third memory view.
    source_view = "delta"
    system_prompt = DELTA_V2_SYSTEM_PROMPT

    def __init__(self, compaction_interval: int = 5) -> None:
        if compaction_interval < 1:
            raise ValueError("compaction_interval must be positive")
        self.compaction_interval = compaction_interval

    def initial_state(self) -> DeltaV2State:
        return DeltaV2State()

    def format_input(self, row: Mapping[str, Any]) -> str:
        memory_input = row.get("input")
        state = delta_v2_state_from_input(
            memory_input, compaction_interval=self.compaction_interval
        )
        compact_input = delta_v2_input_from_state(
            state, compaction_interval=self.compaction_interval
        )
        return format_turn_input(memory_input=compact_input, row=row)

    def format_target(self, row: Mapping[str, Any]) -> str:
        target = target_mapping(row)
        decision = require_decision(target)
        if decision == "NO_OP":
            return compact_json({"decision": "NO_OP"})
        operations = expand_compact_operations(target.get("operations"))
        return compact_json(
            {"decision": "UPDATE", "operations": compact_operations(operations)}
        )

    def parse_output(self, text: str) -> ParsedMemoryOutput:
        value = parse_json_object(text)
        decision = require_decision(value)
        if decision == "NO_OP":
            require_exact_fields(value, {"decision"})
            return ParsedMemoryOutput(decision, {})
        require_exact_fields(value, {"decision", "operations"})
        operations = expand_compact_operations(value["operations"])
        return ParsedMemoryOutput(decision, {"operations": operations})

    def apply_output(
        self,
        state: DeltaV2State,
        output: ParsedMemoryOutput,
        *,
        turn_id: str | None = None,
    ) -> DeltaV2State:
        del turn_id
        if output.decision == "NO_OP":
            return state
        operations = validate_operations(output.payload.get("operations"))
        # Validate target applicability before persisting a deferred batch.
        # Otherwise an invalid replace/delete can remain hidden in pending_updates
        # until a later compaction or materialization crashes the whole scenario.
        current_memory = self.materialize_memory(state)
        next_memory, _ = apply_operations(current_memory, operations)
        pending = (*state.pending_updates, operations)
        if len(pending) < self.compaction_interval:
            return DeltaV2State(normalize_memory(state.base_summary), pending)
        return DeltaV2State(base_summary=next_memory)

    def materialize_memory(self, state: DeltaV2State) -> str:
        memory = normalize_memory(state.base_summary)
        for batch in state.pending_updates:
            memory, _ = apply_operations(memory, batch)
        return memory


def compact_operations(
    operations: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    """Encode validated exact-block operations without redundant empty fields."""
    compact = []
    for operation in validate_operations(operations):
        op = operation["op"]
        target = operation["target"]
        content = operation["content"]
        if op == "add":
            compact.append([op, target, content] if target else [op, content])
        elif op == "replace":
            compact.append([op, target, content])
        else:
            compact.append([op, target])
    return compact


def expand_compact_operations(value: Any) -> tuple[dict[str, str], ...]:
    """Decode compact arrays into the shared deterministic executor contract."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Delta-v2 operations must be an array")
    expanded = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError(f"Delta-v2 operation {index} must be an array")
        parts = list(raw)
        if not parts or not all(isinstance(part, str) for part in parts):
            raise TypeError(f"Delta-v2 operation {index} must contain strings")
        op = parts[0]
        if op == "add" and len(parts) == 2:
            operation = {"op": op, "target": "", "content": parts[1]}
        elif op == "add" and len(parts) == 3 or op == "replace" and len(parts) == 3:
            operation = {"op": op, "target": parts[1], "content": parts[2]}
        elif op == "delete" and len(parts) == 2:
            operation = {"op": op, "target": parts[1], "content": ""}
        else:
            raise ValueError(f"Delta-v2 operation {index} has invalid arity")
        expanded.append(operation)
    return validate_operations(expanded)


def delta_v2_state_from_input(
    value: Any, *, compaction_interval: int = 5
) -> DeltaV2State:
    """Restore the runtime state from the compact JSONL input contract."""
    if compaction_interval < 1:
        raise ValueError("compaction_interval must be positive")
    if not isinstance(value, Mapping) or set(value) != {
        "base_summary",
        "pending_updates",
    }:
        raise ValueError("Delta-v2 input has invalid fields")
    base = value.get("base_summary")
    if not isinstance(base, str):
        raise TypeError("Delta-v2 base_summary must be a string")
    batches = _validate_pending_updates(value.get("pending_updates"))
    if len(batches) >= compaction_interval:
        raise ValueError("Delta-v2 input must be compacted before the next turn")
    return DeltaV2State(base_summary=normalize_memory(base), pending_updates=batches)


def delta_v2_input_from_state(
    state: DeltaV2State, *, compaction_interval: int = 5
) -> dict[str, Any]:
    """Serialize runtime state for the next turn without materializing deltas."""
    if compaction_interval < 1:
        raise ValueError("compaction_interval must be positive")
    if not isinstance(state, DeltaV2State):
        raise TypeError("Delta-v2 runtime state must be DeltaV2State")
    if len(state.pending_updates) >= compaction_interval:
        raise ValueError("Delta-v2 state must be compacted before the next turn")
    return {
        "base_summary": normalize_memory(state.base_summary),
        "pending_updates": [
            compact_operations(batch) for batch in state.pending_updates
        ],
    }


def _validate_pending_updates(value: Any) -> tuple[tuple[dict[str, str], ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("pending_updates must be an array")
    return tuple(expand_compact_operations(batch) for batch in value)
