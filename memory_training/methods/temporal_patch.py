"""Deterministic temporal metadata layered over exact-block Patch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from .operations import apply_operations, normalize_memory

TEMPORAL_ACTIONS = frozenset(
    {
        "non_temporal",
        "durable_upsert",
        "current_upsert",
        "temporary_override",
        "end_temporary",
        "conditional_upsert",
    }
)
TEMPORAL_OPERATION_FIELDS = (
    "op",
    "target",
    "content",
    "identity_key",
    "temporal_action",
    "temporal_cue",
)

TEMPORAL_PATCH_SYSTEM_PROMPT = """You maintain durable vehicle memory using temporal-aware exact-block patches.
Given previous_memory and the current turn, return exactly one JSON object.
If nothing changes, return {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","operations":[...]}.
Each operation has exactly op, target, content, identity_key, temporal_action, and temporal_cue.
op is add, replace, or delete. temporal_action is non_temporal, durable_upsert,
current_upsert, temporary_override, end_temporary, or conditional_upsert.
Targets must copy a unique complete block from previous_memory exactly.
Keep a durable baseline when adding a temporary override. Use temporary_override
only with add and end_temporary only with delete or a shorter replacement.
temporary_override and end_temporary require the shortest exact temporal phrase
from the current turn; temporal_cue is empty for every other action.
Do not infer that a temporary state ended from elapsed time alone.
Do not emit reasoning, markdown fences, an updated full memory, or extra fields."""


def validate_temporal_operations(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("operations must be an array")
    if not 1 <= len(value) <= 32:
        raise ValueError("UPDATE must contain between 1 and 32 operations")
    operations = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != set(TEMPORAL_OPERATION_FIELDS):
            raise ValueError(f"temporal operation {index} has invalid fields")
        if any(not isinstance(raw[field], str) for field in TEMPORAL_OPERATION_FIELDS):
            raise TypeError(f"temporal operation {index} fields must be strings")
        operation = {field: str(raw[field]) for field in TEMPORAL_OPERATION_FIELDS}
        op = operation["op"]
        action = operation["temporal_action"]
        cue = operation["temporal_cue"].strip()
        identity_key = operation["identity_key"].strip()
        if op not in {"add", "replace", "delete"}:
            raise ValueError(f"temporal operation {index} has invalid op")
        if action not in TEMPORAL_ACTIONS:
            raise ValueError(f"temporal operation {index} has invalid action")
        if not identity_key or len(identity_key) > 160:
            raise ValueError(f"temporal operation {index} has invalid identity_key")
        if action == "temporary_override" and op != "add":
            raise ValueError("temporary_override must use add")
        if action == "end_temporary" and op not in {"delete", "replace"}:
            raise ValueError("end_temporary must use delete or replace")
        if (
            action in {"durable_upsert", "current_upsert", "conditional_upsert"}
            and op == "delete"
        ):
            raise ValueError(f"{action} cannot use delete")
        if action in {"temporary_override", "end_temporary"}:
            if not cue:
                raise ValueError(f"{action} requires temporal_cue")
        elif cue:
            raise ValueError(f"{action} requires an empty temporal_cue")
        operation["identity_key"] = identity_key
        operation["temporal_cue"] = cue
        operations.append(operation)
    # Reuse the exact-block validator for op/target/content constraints.
    apply_ready = [
        {key: operation[key] for key in ("op", "target", "content")}
        for operation in operations
    ]
    from .operations import validate_operations

    validate_operations(apply_ready)
    return tuple(operations)


class TemporalPatchMethod(MemoryMethod[str]):
    """Patch method whose extra temporal fields are validated, then stripped."""

    name = "temporal_patch"
    # The alternate Temporal dataset keeps its enriched rows at patch.jsonl so
    # it can reuse the existing three-view indexed catalog unchanged.
    source_view = "patch"
    system_prompt = TEMPORAL_PATCH_SYSTEM_PROMPT

    def initial_state(self) -> str:
        return ""

    def format_input(self, row: Mapping[str, Any]) -> str:
        memory_input = row.get("input")
        if not isinstance(memory_input, Mapping) or set(memory_input) != {
            "previous_memory"
        }:
            raise ValueError("Temporal Patch input requires only previous_memory")
        return format_turn_input(memory_input=memory_input, row=row)

    def format_target(self, row: Mapping[str, Any]) -> str:
        target = target_mapping(row)
        decision = require_decision(target)
        if decision == "NO_OP":
            return compact_json({"decision": "NO_OP"})
        operations = validate_temporal_operations(target.get("operations"))
        return compact_json({"decision": "UPDATE", "operations": operations})

    def parse_output(self, text: str) -> ParsedMemoryOutput:
        value = parse_json_object(text)
        decision = require_decision(value)
        if decision == "NO_OP":
            require_exact_fields(value, {"decision"})
            return ParsedMemoryOutput(decision, {})
        require_exact_fields(value, {"decision", "operations"})
        operations = validate_temporal_operations(value["operations"])
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
        operations = validate_temporal_operations(output.payload.get("operations"))
        stripped = [
            {key: operation[key] for key in ("op", "target", "content")}
            for operation in operations
        ]
        memory, _ = apply_operations(state, stripped)
        return memory

    def materialize_memory(self, state: str) -> str:
        return normalize_memory(state)
