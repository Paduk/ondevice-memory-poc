"""Common contract shared by Summary, Patch, Delta, and Summary+Reason."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ParsedMemoryOutput:
    decision: str
    payload: Mapping[str, Any]
    reason_code: str | None = None
    reason: str | None = None


class MemoryMethod(ABC, Generic[StateT]):
    """Method-specific serialization, parsing, and deterministic state transition."""

    name: str
    source_view: str

    @abstractmethod
    def initial_state(self) -> StateT:
        """Return the empty runtime state for a new scenario."""

    @abstractmethod
    def format_input(self, row: Mapping[str, Any]) -> str:
        """Serialize one canonical training row as model input."""

    @abstractmethod
    def format_target(self, row: Mapping[str, Any]) -> str:
        """Serialize only the fields decoded by this method."""

    @abstractmethod
    def parse_output(self, text: str) -> ParsedMemoryOutput:
        """Validate a model generation without mutating state."""

    @abstractmethod
    def apply_output(
        self,
        state: StateT,
        output: ParsedMemoryOutput,
        *,
        turn_id: str | None = None,
    ) -> StateT:
        """Apply a validated output deterministically."""

    @abstractmethod
    def materialize_memory(self, state: StateT) -> str:
        """Return the full memory visible to downstream Quiz evaluation."""
