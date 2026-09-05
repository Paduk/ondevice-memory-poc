"""Append-only prompt/session state matching the Delta-v3 append SFT profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .methods import DeltaV3AppendMethod
from .training_data import DeltaAppendChatExampleEncoder


@dataclass(frozen=True)
class DeltaAppendPreparedTurn:
    """Prompt plus cache-epoch identity for one generation request."""

    prompt: str
    prompt_tokens: int
    epoch_index: int
    epoch_turn_index: int
    epoch_updates_before: int
    reset_before: bool
    reset_reason: str | None


@dataclass(frozen=True)
class DeltaAppendCommit:
    """Cache-epoch transition caused by one assistant decision."""

    epoch_turns_after: int
    epoch_updates_after: int
    epoch_end_after: bool
    epoch_end_reason: str | None


class DeltaAppendPromptSession:
    """Retain exact generated assistant messages inside one bounded KV epoch."""

    def __init__(
        self,
        method: DeltaV3AppendMethod,
        encoder: DeltaAppendChatExampleEncoder,
        *,
        max_history_turns: int = 32,
        reset_on_discontinuity: bool = True,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        self.method = method
        self.encoder = encoder
        self.max_history_turns = max_history_turns
        self.reset_on_discontinuity = reset_on_discontinuity
        self.reset()

    def reset(self) -> None:
        """Reset all scenario-local transcript state."""
        self._base_summary = ""
        self._history_rows: list[dict[str, Any]] = []
        self._history_outputs: list[str] = []
        self._updates = 0
        self._epoch_index = -1
        self._last_global_turn: int | None = None
        self._pending_reset_reason: str | None = None
        self._prepared_row: dict[str, Any] | None = None

    def prepare(
        self, row: dict[str, Any], *, materialized_memory: str
    ) -> DeltaAppendPreparedTurn:
        """Render the next prompt and decide whether its KV epoch must restart."""
        if self._prepared_row is not None:
            raise RuntimeError("Previous append-only turn has not been committed")
        global_turn = int(row["global_turn_index"])
        reset_reason = self._reset_reason(global_turn)
        if reset_reason is not None:
            self._start_epoch(materialized_memory)

        prompt = self.encoder.generation_prompt_append(
            row,
            self.method,
            base_summary=self._base_summary,
            history_rows=self._history_rows,
            history_outputs=self._history_outputs,
        )
        prompt_tokens = self.encoder.prompt_token_count(prompt)
        if prompt_tokens > self.encoder.max_length and self._history_rows:
            reset_reason = "context_limit"
            self._start_epoch(materialized_memory)
            prompt = self.encoder.generation_prompt_append(
                row,
                self.method,
                base_summary=self._base_summary,
                history_rows=self._history_rows,
                history_outputs=self._history_outputs,
            )
            prompt_tokens = self.encoder.prompt_token_count(prompt)
        if prompt_tokens > self.encoder.max_length:
            raise ValueError(
                "Fresh Delta-v3 append prompt exceeds max_length: "
                f"{prompt_tokens} > {self.encoder.max_length}"
            )

        self._prepared_row = row
        return DeltaAppendPreparedTurn(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            epoch_index=self._epoch_index,
            epoch_turn_index=len(self._history_rows),
            epoch_updates_before=self._updates,
            reset_before=reset_reason is not None,
            reset_reason=reset_reason,
        )

    def commit(
        self, *, assistant_output: str, decision_for_epoch: str
    ) -> DeltaAppendCommit:
        """Append one exact model output and schedule the next epoch boundary."""
        if self._prepared_row is None:
            raise RuntimeError("Append-only turn must be prepared before commit")
        if decision_for_epoch not in {"NO_OP", "UPDATE", "INVALID"}:
            raise ValueError(f"Unsupported append-only decision: {decision_for_epoch}")
        row = self._prepared_row
        self._history_rows.append(row)
        self._history_outputs.append(assistant_output)
        if decision_for_epoch == "UPDATE":
            self._updates += 1
        self._last_global_turn = int(row["global_turn_index"])
        self._prepared_row = None

        end_reason = None
        if self._updates >= self.method.compaction_interval:
            end_reason = "update_compaction"
        elif len(self._history_rows) >= self.max_history_turns:
            end_reason = "turn_budget"
        self._pending_reset_reason = end_reason
        return DeltaAppendCommit(
            epoch_turns_after=len(self._history_rows),
            epoch_updates_after=self._updates,
            epoch_end_after=end_reason is not None,
            epoch_end_reason=end_reason,
        )

    def _reset_reason(self, global_turn: int) -> str | None:
        if self._epoch_index < 0:
            return "scenario_start"
        if self._pending_reset_reason is not None:
            return self._pending_reset_reason
        if (
            self.reset_on_discontinuity
            and self._last_global_turn is not None
            and global_turn != self._last_global_turn + 1
        ):
            return "turn_discontinuity"
        return None

    def _start_epoch(self, materialized_memory: str) -> None:
        self._base_summary = materialized_memory
        self._history_rows = []
        self._history_outputs = []
        self._updates = 0
        self._epoch_index += 1
        self._pending_reset_reason = None

