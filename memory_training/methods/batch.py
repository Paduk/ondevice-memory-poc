"""Date-batched Summary and ordered-Patch adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import compact_json
from .patch import PatchMethod
from .summary import SummaryMethod

SUMMARY_BATCH_SYSTEM_PROMPT = """You maintain durable vehicle memory.
Given the previous full memory and all conversations in one chronological batch,
process every turn in order and return exactly one JSON object.
If no durable vehicle fact changes, return {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","next_memory":"<complete final memory>"}.
Preserve every still-valid fact. Do not emit reasoning, markdown fences, or extra fields."""


PATCH_BATCH_SYSTEM_PROMPT = """You maintain durable vehicle memory using exact-block patches.
Given previous_memory and all conversations in one chronological batch, process every
turn in order and return exactly one JSON object. If nothing changes, return
{"decision":"NO_OP"}. Otherwise return {"decision":"UPDATE","operations":[...]}.
Operations are ordered and each has exactly op, target, content; op is add, replace,
or delete. Targets must copy a unique complete block from previous_memory or from
content created by an earlier operation in this output. Do not emit reasoning,
markdown fences, an updated full memory, or extra fields."""


def _format_batch_input(row: Mapping[str, Any]) -> str:
    memory_input = row.get("input")
    if not isinstance(memory_input, Mapping) or set(memory_input) != {
        "previous_memory",
        "turns",
    }:
        raise ValueError("Batch input requires previous_memory and turns")
    turns = memory_input.get("turns")
    if not isinstance(turns, (list, tuple)) or not turns:
        raise TypeError("Batch input turns must be a non-empty sequence")
    return compact_json(
        {
            "memory_state": {"previous_memory": memory_input["previous_memory"]},
            "batch": {
                "date": row.get("batch_date"),
                "turns": list(turns),
            },
        }
    )


class SummaryBatchMethod(SummaryMethod):
    name = "summary_batch"
    source_view = "summary"
    system_prompt = SUMMARY_BATCH_SYSTEM_PROMPT

    def format_input(self, row: Mapping[str, Any]) -> str:
        return _format_batch_input(row)


class PatchBatchMethod(PatchMethod):
    name = "patch_batch"
    source_view = "patch"
    system_prompt = PATCH_BATCH_SYSTEM_PROMPT

    def format_input(self, row: Mapping[str, Any]) -> str:
        return _format_batch_input(row)
