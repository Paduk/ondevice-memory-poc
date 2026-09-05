"""Append-only Delta-v3 chat profile for persistent cross-turn KV epochs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import compact_json
from .delta_v2 import delta_v2_state_from_input
from .delta_v3 import DeltaV3Method
from .operations import normalize_memory

DELTA_V3_APPEND_SYSTEM_PROMPT = """You maintain vehicle memory in one append-only cache epoch.
The first user message contains base_summary and current_turn. Later user messages contain only current_turn.
Earlier assistant UPDATE operations in this epoch are pending deltas and must be applied in order to base_summary.
If nothing changes, return exactly {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","operations":[...]}.
Use compact operation arrays: ["add","content"] appends a block;
["add","target","content"] inserts after an exact target;
["replace","target","content"] replaces an exact target; and
["delete","target"] deletes an exact target.
Targets must be unique complete blocks in the materialized memory.
The runtime starts a new cache epoch after five UPDATEs or when the context budget is reached.
Do not emit reasoning or extra fields."""


class DeltaV3AppendMethod(DeltaV3Method):
    """Train Delta-v3 outputs inside an append-only multi-turn transcript."""

    name = "delta_v3_append"
    system_prompt = DELTA_V3_APPEND_SYSTEM_PROMPT

    def format_input(self, row: Mapping[str, Any]) -> str:
        """Provide a semantically complete fresh-epoch fallback for one-step callers."""
        state = delta_v2_state_from_input(
            row.get("input"), compaction_interval=self.compaction_interval
        )
        return self.format_epoch_turn(row, base_summary=self.materialize_memory(state))

    def format_epoch_turn(self, row: Mapping[str, Any], *, base_summary: str) -> str:
        return compact_json(
            {
                "base_summary": normalize_memory(base_summary),
                "current_turn": _current_turn(row),
            }
        )

    def format_followup_turn(self, row: Mapping[str, Any]) -> str:
        return compact_json({"current_turn": _current_turn(row)})


def _current_turn(row: Mapping[str, Any]) -> dict[str, Any]:
    turn = row.get("current_turn")
    if not isinstance(turn, Mapping):
        raise TypeError("Training row current_turn must be an object")
    return {
        "turn_id": row.get("turn_id"),
        "timestamp": row.get("timestamp"),
        "speaker_id": turn.get("speaker_id"),
        "speaker_name": turn.get("speaker_name"),
        "text": turn.get("text"),
    }
