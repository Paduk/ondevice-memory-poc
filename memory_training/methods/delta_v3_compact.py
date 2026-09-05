"""Token-efficient Delta-v3 prompt profiles for compaction ablations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import compact_json
from .delta_v2 import delta_v2_input_from_state, delta_v2_state_from_input
from .delta_v3 import DeltaV3Method

DELTA_V3_COMPACT_SYSTEM_PROMPT = """You maintain durable vehicle memory using exact-block patches.
Materialize previous_memory by applying pending patches P to base B in order.
Given previous_memory and current turn, return exactly {"decision":"NO_OP"} if nothing changes; otherwise return {"decision":"UPDATE","operations":[...]}.
Operations use compact arrays: ["add","content"], ["add","target","content"], ["replace","target","content"], or ["delete","target"].
Targets must copy unique complete blocks from previous_memory. Output JSON only."""


class DeltaV3CompactMethod(DeltaV3Method):
    """Render the existing Delta-v3 state without JSON-escaping the base memory."""

    name = "delta_v3_compact"
    system_prompt = DELTA_V3_COMPACT_SYSTEM_PROMPT

    def format_input(self, row: Mapping[str, Any]) -> str:
        state = delta_v2_state_from_input(
            row.get("input"), compaction_interval=self.compaction_interval
        )
        serialized = delta_v2_input_from_state(
            state, compaction_interval=self.compaction_interval
        )
        turn = row.get("current_turn")
        if not isinstance(turn, Mapping):
            raise TypeError("Training row current_turn must be an object")
        turn_metadata = "|".join(
            str(value)
            for value in (
                row.get("turn_id"),
                row.get("timestamp"),
                turn.get("speaker_id"),
                turn.get("speaker_name"),
            )
        )
        return "\n".join(
            (
                "B:",
                serialized["base_summary"],
                "P:",
                compact_json(serialized["pending_updates"]),
                "T:",
                turn_metadata,
                str(turn.get("text")),
            )
        )


class DeltaV3CompactK2Method(DeltaV3CompactMethod):
    name = "delta_v3_compact_k2"

    def __init__(self) -> None:
        super().__init__(compaction_interval=2)


class DeltaV3CompactK5Method(DeltaV3CompactMethod):
    name = "delta_v3_compact_k5"

    def __init__(self) -> None:
        super().__init__(compaction_interval=5)


class DeltaV3CompactK10Method(DeltaV3CompactMethod):
    name = "delta_v3_compact_k10"

    def __init__(self) -> None:
        super().__init__(compaction_interval=10)
