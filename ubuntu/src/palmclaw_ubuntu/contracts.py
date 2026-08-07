from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from palmclaw_ubuntu.models import (
    AgentResponse,
    AMemConstructionResponse,
    AMemEvolutionDecision,
    AMemNeighbor,
    ChatMessage,
    CompactAMemEpisode,
    CompactAMemResponse,
    EmbeddingResponse,
    FactMemoryExtractionResponse,
    FactMemoryRecord,
    FactMemorySemanticReviewCase,
    FactMemorySemanticReviewResponse,
    MemoryMessage,
    MemoryResponse,
    PatchMemoryResponse,
    StructuredMemoryResponse,
    ToolDefinition,
    ToolMemoryRecord,
)
from palmclaw_ubuntu.tool_memory_schema import ToolMemoryOntology


class AgentModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> AgentResponse: ...


class MemoryModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str

    def consolidate(
        self,
        messages: Sequence[ChatMessage],
        previous_memory: str,
    ) -> MemoryResponse: ...


class RecursiveSummaryMemoryModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str
    max_memory_chars: int

    def update(
        self,
        *,
        previous_memory: str,
        date: str,
        daily_history: str,
    ) -> MemoryResponse: ...


class StructuredMemoryModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str

    def extract(
        self,
        messages: Sequence[MemoryMessage],
        existing_memory: str,
    ) -> StructuredMemoryResponse: ...


class PatchMemoryModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str

    def propose(
        self,
        messages: Sequence[MemoryMessage],
        ontology: ToolMemoryOntology,
        active_records: Sequence[ToolMemoryRecord],
        *,
        user_id: str,
        session_id: str,
    ) -> PatchMemoryResponse: ...


class FactMemoryModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str
    semantic_prompt_version: str
    semantic_schema_version: str

    def extract(
        self,
        messages: Sequence[MemoryMessage],
        active_records: Sequence[FactMemoryRecord],
        *,
        user_id: str,
        session_id: str,
    ) -> FactMemoryExtractionResponse: ...

    def review_uncertain(
        self,
        cases: Sequence[FactMemorySemanticReviewCase],
        messages: Sequence[MemoryMessage],
        active_records: Sequence[FactMemoryRecord],
        *,
        user_id: str,
        session_id: str,
    ) -> FactMemorySemanticReviewResponse: ...


class EmbeddingModel(Protocol):
    backend: str
    model_id: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse: ...


class AMemModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str
    construction_prompt_version: str
    construction_schema_version: str
    evolution_prompt_version: str
    evolution_schema_version: str

    def construct(
        self,
        entry: Mapping[str, Any],
    ) -> AMemConstructionResponse: ...

    def evolve(
        self,
        new_note: AMemNeighbor,
        neighbors: Sequence[AMemNeighbor],
    ) -> AMemEvolutionDecision: ...


class CompactAMemModel(Protocol):
    backend: str
    model_id: str
    prompt_version: str
    schema_version: str

    def compact_episode(
        self,
        episode: CompactAMemEpisode,
    ) -> CompactAMemResponse: ...
