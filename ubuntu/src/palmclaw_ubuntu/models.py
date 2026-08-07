from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    provider_items: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass(frozen=True)
class AgentResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    continuation_items: tuple[Mapping[str, Any], ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryResponse:
    content: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEvidence:
    message_id: int
    quote: str
    start_char: int | None = None
    end_char: int | None = None


@dataclass(frozen=True)
class MemoryMessage:
    id: int
    role: str
    content: str


@dataclass(frozen=True)
class MemoryCandidate:
    subject: str
    predicate: str
    value: str
    scope: str
    memory_type: str
    confidence: float
    sensitivity: str
    evidence: tuple[MemoryEvidence, ...]


@dataclass(frozen=True)
class MemoryGateDecision:
    decision: str
    evidence_relation: str
    conflict_relation: str
    reason: str
    policy_version: str = "memory-gate-v1"
    pii_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredMemoryResponse:
    candidates: tuple[MemoryCandidate, ...]
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: tuple[tuple[float, ...], ...]
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    session_id: str
    scope: str
    content: str
    status: str
    version: int
    subject: str
    predicate: str
    value: str
    fact_key: str
    memory_type: str
    confidence: float
    sensitivity: str
    supersedes_id: str | None
    created_at: str


@dataclass(frozen=True)
class MemoryRetrievalResult:
    content: str
    memories: tuple[MemoryRecord, ...] = ()
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolMemoryIdentity:
    user_id: str
    tool_domain: str
    topic: str
    scope: str
    scope_key: str
    conditions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolMemoryRecord:
    id: str
    session_id: str
    record_key: str
    user_id: str
    tool_domain: str
    topic: str
    scope: str
    scope_key: str
    conditions: Mapping[str, Any]
    value: Any
    memory_type: str
    status: str
    confidence: float
    version: int
    supersedes_id: str | None
    merged_into_id: str | None
    created_at: str
    updated_at: str
    entity_id: str | None = None
    identity_conditions: Mapping[str, Any] = field(default_factory=dict)
    applicability: Mapping[str, Any] = field(default_factory=dict)
    identity_family_key: str | None = None


@dataclass(frozen=True)
class ToolMemoryPatch:
    operation: str
    confidence: float
    evidence: tuple[MemoryEvidence, ...]
    identity: ToolMemoryIdentity | None = None
    value: Any = None
    memory_type: str | None = None
    target_record_id: str | None = None
    merge_record_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ToolMemoryPatchOutcome:
    operation: str
    proposal_id: str
    result_record_id: str
    record_status: str


@dataclass(frozen=True)
class ToolMemoryPatchRejection:
    sequence_index: int
    proposal_id: str
    code: str
    message: str
    evidence_message_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ToolMemoryPatchBatchResult:
    run_id: str
    outcomes: tuple[ToolMemoryPatchOutcome, ...]
    rejections: tuple[ToolMemoryPatchRejection, ...] = ()
    replayed: bool = False


@dataclass(frozen=True)
class FactMemoryIdentity:
    """Stable identity of a fact, independent of its value and tool arguments."""

    user_id: str
    entity_id: str
    predicate: str
    identity_conditions: Mapping[str, Any] = field(default_factory=dict)
    applicability: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactMemoryRecord:
    id: str
    session_id: str
    record_key: str
    user_id: str
    entity_id: str
    predicate: str
    identity_conditions: Mapping[str, Any]
    applicability: Mapping[str, Any]
    capability_hints: tuple[str, ...]
    value: Any
    memory_type: str
    status: str
    confidence: float
    version: int
    supersedes_id: str | None
    merged_into_id: str | None
    created_at: str
    updated_at: str
    bundle_id: str | None = None


@dataclass(frozen=True)
class FactMemoryPatch:
    operation: str
    confidence: float
    evidence: tuple[MemoryEvidence, ...]
    identity: FactMemoryIdentity | None = None
    value: Any = None
    memory_type: str | None = None
    capability_hints: tuple[str, ...] = ()
    target_record_id: str | None = None
    merge_record_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class FactMemoryPatchOutcome:
    operation: str
    event_id: str
    result_record_id: str
    record_status: str
    replayed: bool = False


@dataclass(frozen=True)
class FactMemoryCandidate:
    entity_id: str
    predicate: str
    value: Any
    identity_conditions: Mapping[str, Any]
    applicability: Mapping[str, Any]
    capability_hints: tuple[str, ...]
    memory_type: str
    confidence: float
    evidence: tuple[MemoryEvidence, ...]
    directive: str = "UPSERT"
    reason: str = ""


@dataclass(frozen=True)
class FactMemoryExtractionResponse:
    candidates: tuple[FactMemoryCandidate, ...]
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactMemorySemanticReviewCase:
    sequence_index: int
    candidate: FactMemoryCandidate
    review_codes: tuple[str, ...]
    proposed_operation: str | None = None
    related_record_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactMemorySemanticDecision:
    sequence_index: int
    decision: str
    confidence: float
    reason: str
    evidence_relation: str


@dataclass(frozen=True)
class FactMemorySemanticReviewResponse:
    decisions: tuple[FactMemorySemanticDecision, ...]
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactMemoryLinkOutcome:
    sequence_index: int
    decision: str
    status: str
    candidate_event_id: str
    result_record_id: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class FactMemoryLinkBatchResult:
    run_id: str
    outcomes: tuple[FactMemoryLinkOutcome, ...]


@dataclass(frozen=True)
class FactQueryContext:
    """Request-local identity and condition context used for late binding."""

    entities: tuple[str, ...] = ()
    entity_roles: Mapping[str, str] = field(default_factory=dict)
    requester_entity: str | None = None
    time_of_day: str | None = None
    weather: str | None = None
    situation: str | None = None


@dataclass(frozen=True)
class FactMemoryRetrievalResult:
    content: str
    records: tuple[FactMemoryRecord, ...] = ()
    related_records: tuple[FactMemoryRecord, ...] = ()
    query_context: FactQueryContext = field(default_factory=FactQueryContext)
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AMemNote:
    """Immutable source note; evolution must never rewrite these fields."""

    id: str
    session_id: str
    source_message_id: int
    timestamp: str
    speaker: str
    content: str
    status: str
    created_at: str


@dataclass(frozen=True)
class AMemNoteVersion:
    id: str
    note_id: str
    version: int
    context: str
    keywords: tuple[str, ...]
    tags: tuple[str, ...]
    status: str
    supersedes_id: str | None
    created_by_event_id: str | None
    created_at: str


@dataclass(frozen=True)
class AMemLink:
    id: str
    session_id: str
    left_note_id: str
    right_note_id: str
    created_by_note_id: str
    evolution_event_id: str | None
    similarity_score: float | None
    decision: str
    created_at: str


@dataclass(frozen=True)
class AMemNeighbor:
    note: AMemNote
    version: AMemNoteVersion
    similarity_score: float


@dataclass(frozen=True)
class AMemConstructionResponse:
    context: str
    keywords: tuple[str, ...]
    tags: tuple[str, ...]
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompactAMemEpisodeEntry:
    source_message_id: int
    timestamp: str
    speaker: str
    content: str


@dataclass(frozen=True)
class CompactAMemEpisode:
    index: int
    entries: tuple[CompactAMemEpisodeEntry, ...]
    start_timestamp: str
    end_timestamp: str
    rendered_content: str

    @property
    def source_message_ids(self) -> tuple[int, ...]:
        return tuple(entry.source_message_id for entry in self.entries)


@dataclass(frozen=True)
class CompactAMemNoteDraft:
    content: str
    context: str
    keywords: tuple[str, ...]
    tags: tuple[str, ...]
    memory_kind: str
    source_message_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompactAMemResponse:
    notes: tuple[CompactAMemNoteDraft, ...]
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AMemNeighborUpdate:
    note_id: str
    context: str
    keywords: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AMemEvolutionDecision:
    should_evolve: bool
    linked_note_ids: tuple[str, ...] = ()
    new_note_keywords: tuple[str, ...] = ()
    new_note_tags: tuple[str, ...] = ()
    neighbor_updates: tuple[AMemNeighborUpdate, ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AMemRetrievalResult:
    content: str
    notes: tuple[AMemNote, ...] = ()
    versions: tuple[AMemNoteVersion, ...] = ()
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchMemoryResponse:
    patches: tuple[ToolMemoryPatch, ...]
    usage: ModelUsage = field(default_factory=ModelUsage)
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolMemoryRetrievalResult:
    content: str
    records: tuple[ToolMemoryRecord, ...] = ()
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]
    timeout_seconds: float
    side_effect: str = "none"
    retry_safety: str = "safe"
    strict: bool = True

    def as_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "strict": self.strict,
        }


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
